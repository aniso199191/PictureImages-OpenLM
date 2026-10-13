"""从零训练自研大底座（**随机初始化开始**，不读任何现成权重）。

数据：ownlm/data/train.bin（uint16 token 流）
产物：ownlm/models/ownlmbase.pt（含权重 + 词表 + 配置，能独立加载）
日志：ownlm/log/train.jsonl（每 25 步一行，含真实 token/秒、预计剩时）
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", "4")))

大根 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 大根)

from model import 模型, 存模型, 配  # noqa: E402

分 = __import__("tokenizer").读分词器()
# 数据夹可用 PILC_DATA_DIR 指到别的流（例如带知识的 数据_知识），默认还是 data/
数夹 = os.environ.get("PILC_DATA_DIR") or os.path.join(大根, "data")
模夹 = os.path.join(大根, "models")
日夹 = os.path.join(大根, "log")
for d in (模夹, 日夹):
    os.makedirs(d, exist_ok=True)

上 = int(配["上下文"])
训数据 = np.memmap(os.path.join(数夹, "train.bin"), dtype=np.uint16, mode="r")
验数据 = np.memmap(os.path.join(数夹, "validate.bin"), dtype=np.uint16, mode="r")


def 取批(数据, 批, 上):
    n = len(数据) - 上 - 1
    起 = np.random.randint(0, n, size=批)
    x = np.stack([数据[i:i + 上] for i in 起]).astype(np.int64)
    y = np.stack([数据[i + 1:i + 1 + 上] for i in 起]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def 验证(模, 批=8, 轮=20):
    模.eval()
    总, 计 = 0.0, 0
    for _ in range(轮):
        x, y = 取批(验数据, 批, 上)
        损, _ = 模(x, y)
        总 += float(损)
        计 += 1
    return 总 / max(1, 计)


@torch.no_grad()
def 抽句(模, 提示=("你好", "你是谁", "3加5等于几", "计算 12 乘 7", "介绍一下 EPU")):
    模.eval()
    出表 = []
    for q in 提示:
        号 = [分.字到号["<s>"], 分.字到号["<问>"]] + 分.编(q) + [分.字到号["<思>"]]
        出 = 模.生成(号, 最长=90, 温度=0.7, 顶k=40,
                     停号=[分.字到号["</s>"], 分.字到号["<问>"], 分.字到号["<s>"]])
        出表.append("%s → %s" % (q, 分.解(出).replace("\n", " ")[:90]))
    return " ｜ ".join(出表)


def 训(总步=7000, 批=24, 峰=3e-4, 用bf16=True):
    # 重训大底座时用环境变量改：PILC产物 / PILC步数（产物写到新名字，不动线上那份）
    产物 = os.environ.get("PILC_OUT", "自研大底座")
    总步 = int(os.environ.get("PILC_STEPS", 总步))
    模 = 模型()
    初 = os.environ.get("PILC_INIT") or ""
    if 初 and os.path.exists(初):
        d = torch.load(初, map_location="cpu")
        w = d.get("权重", d) if isinstance(d, dict) else d
        缺 = 模.load_state_dict(w, strict=False)
        print("热启动：从 %s 接着训（缺键 %s）" % (初, list(缺.missing_keys)[:3] or "无"), flush=True)
    else:
        初 = ""
    print("%s：参数量 %.2f M｜每 token 激活 %.2f M｜训练 token %d｜验证 token %d｜批量 %d｜步数 %d｜峰值 lr %g" %
          ("热启动" if 初 else "从零开始",
           模.参数量 / 1e6, (模.激活参数量() if hasattr(模, "激活参数量") else 模.参数量) / 1e6,
           len(训数据), len(验数据), 批, 总步, 峰), flush=True)
    print("配置：%s" % json.dumps(模.配, ensure_ascii=False), flush=True)
    优 = torch.optim.AdamW(模.parameters(), lr=峰, betas=(0.9, 0.95), weight_decay=0.01)
    热 = max(50, 总步 // 40)
    最好 = 9e9
    开 = time.time()
    记 = open(os.path.join(日夹, "train.jsonl"), "a", encoding="utf-8")
    for 步 in range(1, 总步 + 1):
        模.train()
        lr = (峰 * 步 / 热) if 步 < 热 else (峰 / 10 + (峰 - 峰 / 10) * 0.5 *
                                        (1 + math.cos(math.pi * (步 - 热) / max(1, 总步 - 热))))
        for g in 优.param_groups:
            g["lr"] = lr
        x, y = 取批(训数据, 批, 上)
        if 用bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                主损, _ = 模(x, y)
        else:
            主损, _ = 模(x, y)
        # MoE 负载均衡：让专家别全挤在一个上（密集模型这项是 0）
        辅 = 模.取辅损() if hasattr(模, "取辅损") else None
        损 = 主损 + 0.01 * 辅 if isinstance(辅, torch.Tensor) else 主损
        优.zero_grad(set_to_none=True)
        损.backward()
        torch.nn.utils.clip_grad_norm_(模.parameters(), 1.0)
        优.step()
        if 步 % 25 == 0 or 步 == 1:
            用 = time.time() - 开
            行 = {"步": 步, "总步": 总步, "损": round(float(主损), 4),
                  "辅损": round(float(辅), 5) if isinstance(辅, torch.Tensor) else 0.0,
                  "lr": round(lr, 6),
                  "秒": round(用, 1), "token": 步 * 批 * 上,
                  "token每秒": round(步 * 批 * 上 / 用),
                  "预计剩秒": round((总步 - 步) * 用 / 步)}
            print(json.dumps(行, ensure_ascii=False), flush=True)
            记.write(json.dumps(行, ensure_ascii=False) + "\n")
            记.flush()
        if 步 % 500 == 0 or 步 == 总步:
            验损 = 验证(模)
            例 = 抽句(模)
            print("  【步 %d】验证损 %.4f｜抽句：%s" % (步, 验损, 例), flush=True)
            记.write(json.dumps({"步": 步, "验证损": round(验损, 4), "抽句": 例},
                                ensure_ascii=False) + "\n")
            记.flush()
            附 = {"步": 步, "验证损": 验损, "词表": 分.表, "配置": 模.配, "抽句": 例,
                  "从零": not 初, "训练token": int(len(训数据)), "时间": time.strftime("%F %T")}
            存模型(模, os.path.join(模夹, 产物 + "_ckpt.pt"), 附)
            if 验损 < 最好:
                最好 = 验损
                存模型(模, os.path.join(模夹, 产物 + ".pt"), 附)
                print("  ✔ 新最好，已存 %s.pt（验证损 %.4f）" % (产物, 验损), flush=True)
    记.close()
    print("训完：用时 %.1f 分钟｜最好验证损 %.4f" % ((time.time() - 开) / 60, 最好), flush=True)


if __name__ == "__main__":
    总 = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
    批 = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    峰 = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
    训(总, 批, 峰)
