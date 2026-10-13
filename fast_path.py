"""fast_path.py —— 自研大 7B 的**等价快路**。

7B 是怎么来的：把 19M 底座的每个线性层按 64 整块复制 20 份（列上再 /20），嵌入只扩维度不除，
头另存一份也 /20。推导一遍就知道：扩出来的 140 个头里，头 20b..20b+19 装的全是窄头 b 的同一份
64 维（宽向量是按 64 块整块复制的），线性层的 /20 正好抵掉 20 份重复的求和 —— 所以**整条前向
跟窄模型数学等价**，只有浮点舍入差。

于是服务端不需要按 6.79B 去算：跑这份 19M 窄路，出的字跟 4bit 7B 只差量化噪声（下面 verify_fast_path.py
会把这个噪声量出来）。速度从 7.67 秒/token 掉到几十毫秒/token。

严格路径（真按 6.79B 逐层算）仍在：7B推.py（自研 int4）/ 7B_push_MLX.py（MLX mxfp4）。
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F  # noqa: N812

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
根 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 根)
from tokenizer import 读分词器  # noqa: E402

_源名 = os.environ.get("PILC_BASE_PT", "base_55M.pt")
包 = torch.load(os.path.join(根, "models", _源名), map_location="cpu")
# 2026-10-11：原来读「自研大底座.pt」，全量改英文名后这个文件叫 base_55M.pt（真实 65.18M 参数，dim640×12层×10头）。
# 2026-10-12：改成可换源（PILC_BASE_PT），换底座不用改代码、也不动正在跑的服务。
配, 权重, 附 = 包["配"], 包["权重"], 包.get("附", {})
维, 层, 头, 词表, 上下文 = 配["维度"], 配["层数"], 配["头数"], 配["词表"], 配["上下文"]
头维 = 维 // 头
分 = 读分词器(os.path.join(根, "vocab.json"))
小 = torch.float32  # 这台机器 bf16 比 fp32 快 2.47 倍
W = {k: v.to(小) for k, v in 权重.items()}
W.update({k: v.to(小) for k, v in 权重.items() if k.endswith(".bias")})
头权 = W.get("头.weight", W["词嵌.weight"])   # 绑权重时就是词嵌
停号 = {分.字到号.get("<pad>"), 分.字到号.get("</s>")}


def 归一(x, 名):
    return F.layer_norm(x, [维], W[名 + ".weight"], W.get(名 + ".bias"), 1e-5)


def 线(x, 名):
    y = x @ W[名 + ".weight"].t()
    偏 = W.get(名 + ".bias")
    return y + 偏 if 偏 is not None else y


def 过块(i, x, 缓存=None):
    B, T, C = x.shape
    h = 归一(x, "块表.%d.归一1" % i)
    qkv = 线(h.reshape(-1, 维), "块表.%d.注意.qkv" % i).view(B, T, 3 * 维)
    q, k, v = qkv.split(维, dim=2)
    q = q.view(B, T, 头, 头维).transpose(1, 2)
    k = k.view(B, T, 头, 头维).transpose(1, 2)
    v = v.view(B, T, 头, 头维).transpose(1, 2)
    if 缓存 is not None:
        k = torch.cat([缓存[0], k], dim=2)
        v = torch.cat([缓存[1], v], dim=2)
    长 = k.shape[2]
    a = (q @ k.transpose(-2, -1)) / math.sqrt(头维)
    if T > 1:
        掩 = torch.ones(T, 长, dtype=torch.bool).tril(diagonal=长 - T)
        a = a.masked_fill(~掩, float("-inf"))
    a = torch.softmax(a, dim=-1)
    y = (a @ v).transpose(1, 2).contiguous().view(B, T, 维)
    x = x + 线(y.reshape(-1, 维), "块表.%d.注意.出" % i).view(B, T, 维)
    u = 线(归一(x, "块表.%d.归一2" % i).reshape(-1, 维), "块表.%d.馈.上" % i)
    d = F.gelu(u) @ W["块表.%d.馈.下.weight" % i].t() + W["块表.%d.馈.下.bias" % i]
    x = x + d.view(B, T, 维)
    return x, (k, v)


@torch.no_grad()
def 首步(ids):
    """只跑提示段，回第一步的完整逻辑值（对拍用，与 7B推 同源）。"""
    x = (W["词嵌.weight"][torch.tensor(ids, dtype=torch.long)] +
         W["位嵌.weight"][torch.arange(len(ids), dtype=torch.long)]).unsqueeze(0)
    缓存 = [None] * 层
    for i in range(层):
        x, 缓存[i] = 过块(i, x, 缓存[i])
    return 线(归一(x[:, -1, :], "末归一"), "头")[0], 缓存


默认停号 = tuple(x for x in (分.字到号.get("</s>"), 分.字到号.get("<问>"), 分.字到号.get("<答>"),
                          分.字到号.get("<s>")) if x is not None)


@torch.no_grad()
def 生成(提示, 最长=24, 温度=0.85, 顶k=40, 停号=None, 钩=None):
    ids = 分.编(提示)
    停 = 默认停号 if 停号 is None else tuple(停号)
    t0 = time.time()
    逻, 缓存 = 首步(ids)
    首段 = time.time() - t0
    文, 步时 = "", []
    for 步 in range(最长):
        概 = torch.softmax(逻.float() / max(0.05, 温度), dim=-1)
        if 顶k and 顶k > 0:
            阈 = torch.topk(概, min(顶k, 概.numel()))[0][-1]
            概 = 概.masked_fill(概 < 阈, 0.0)
        新 = int(torch.multinomial(概 / 概.sum(), 1))
        if 新 in 停:
            break
        字 = 分.解([新])
        文 += 字
        if 钩:
            钩(字)
        t1 = time.time()
        x = (W["词嵌.weight"][torch.tensor([新])] +
             W["位嵌.weight"][torch.tensor([len(ids) + 步])]).unsqueeze(0)
        for i in range(层):
            x, 缓存[i] = 过块(i, x, 缓存[i])
        逻 = 线(归一(x[:, -1, :], "末归一"), "头")[0]
        步时.append(time.time() - t1)
    return 文, 首段, 步时


if __name__ == "__main__":
    提示 = sys.argv[1] if len(sys.argv) > 1 else "你好，你是谁"
    最长 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    print("=== 自研大 %dM 等价快路（窄 %d 维 × %d 层，源自 7B 同一份权重）===" %
          (sum(v.numel() for v in W.values()) // 1000000, 维, 层), flush=True)
    print("训练信息：第 %s 步，验证损 %s" % (附.get("步"), 附.get("验证损")), flush=True)
    文, 首段, 步时 = 生成(提示, 最长)
    if 步时:
        print("提示：%s" % 提示, flush=True)
        print("出来：%s" % 文, flush=True)
        print("首段 %.3f 秒｜后续每 token 均值 %.3f 秒（最短 %.3f，最长 %.3f）" %
              (首段, sum(步时) / len(步时), min(步时), max(步时)), flush=True)
