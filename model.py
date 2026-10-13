"""PictureImages-Chater 的自研骨架：decoder-only Transformer（pre-LN + 权重绑定 + KV 缓存增量生成）。

这里只有我自己的实现：多头注意力、前馈、残差、层归一化、位置编码表、采样策略全是手写；
矩阵运算借 torch 当计算器（本机 CPU）。
"""
import json as _json
import math
import os as _os

import torch
import torch.nn as nn
import torch.nn.functional as F

_大根 = _os.path.dirname(_os.path.abspath(__file__))
# 配置可以换（重训大底座时用 PILC配置=config_55M.json 指到别的配置文件），默认还是 config.json
配 = _json.load(open(_os.environ.get("PILC_CONFIG", _os.path.join(_大根, "config.json")), encoding="utf-8"))



class 注意力(nn.Module):
    def __init__(self, 维度, 头数, 丢弃):
        super().__init__()
        assert 维度 % 头数 == 0
        self.头数 = 头数
        self.头维 = 维度 // 头数
        self.qkv = nn.Linear(维度, 3 * 维度, bias=False)
        self.出 = nn.Linear(维度, 维度, bias=False)
        self.丢弃 = nn.Dropout(丢弃)

    def forward(self, x, 掩=None, 缓存=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.头数, self.头维).transpose(1, 2)
        k = k.view(B, T, self.头数, self.头维).transpose(1, 2)
        v = v.view(B, T, self.头数, self.头维).transpose(1, 2)
        if 缓存 is not None:
            旧k, 旧v = 缓存
            k = torch.cat([旧k, k], dim=2)
            v = torch.cat([旧v, v], dim=2)
        新缓存 = (k, v)
        长 = k.shape[2]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.头维)
        if 掩 is None:
            掩 = torch.ones(T, 长, dtype=torch.bool).tril(diagonal=长 - T).to(x.device)
        a = a.masked_fill(~掩, float("-inf"))
        a = F.softmax(a, dim=-1)
        a = self.丢弃(a)
        y = (a @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.出(y), 新缓存


class 前馈(nn.Module):
    def __init__(self, 维度, 丢弃, 倍=4):
        super().__init__()
        self.上 = nn.Linear(维度, 倍 * 维度)
        self.下 = nn.Linear(倍 * 维度, 维度)
        self.丢弃 = nn.Dropout(丢弃)

    def forward(self, x):
        return self.丢弃(self.下(F.gelu(self.上(x))))


class 稀疏前馈(nn.Module):
    """MoE（专家混合）：每个 token 只走「门控挑出来的 激活 个专家」，总参可以堆很大、算力只按激活的算。

    这是本机这种「内存小、核少」的机器上唯一能把容量做大的路子：
    专家 8 个、每 token 只激活 2 个 → 容量 8 份，算力只用 2 份。
    """

    def __init__(self, 维度, 丢弃, 专家数=8, 激活=2, 专家倍=2):
        super().__init__()
        assert 激活 <= 专家数
        self.专家数 = 专家数
        self.激活 = 激活
        self.门 = nn.Linear(维度, 专家数, bias=False)
        self.专家表 = nn.ModuleList([前馈(维度, 丢弃, 倍=专家倍) for _ in range(专家数)])
        self.辅损 = None  # 训练时填负载均衡损失，防止全挤到一个专家上

    def forward(self, x):
        B, T, C = x.shape
        平 = x.reshape(-1, C)
        门分 = self.门(平)
        概 = F.softmax(门分, dim=-1)
        顶概, 顶号 = 概.topk(self.激活, dim=-1)
        顶概 = 顶概 / 顶概.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if self.training:
            # 负载均衡：希望每个专家被挑到的比例 ≈ 1/专家数
            用比 = (顶号.reshape(-1, self.专家数).float().sum(0)
                   if 顶号.numel() % self.专家数 == 0 else None)
            平概 = 概.mean(dim=0)
            该 = torch.full_like(平概, 1.0 / self.专家数)
            self.辅损 = ((平概 - 该) ** 2).mean() * self.专家数
            用比 = None
        出 = torch.zeros_like(平)
        for e, 专家 in enumerate(self.专家表):
            掩 = 顶号 == e
            有 = 掩.any(dim=-1)
            if not bool(有.any()):
                continue
            权 = (顶概 * 掩).sum(dim=-1)[有]
            出[有] = 出[有] + 专家(平[有]) * 权.unsqueeze(-1)
        return 出.view(B, T, C)


class 块(nn.Module):
    def __init__(self, 维度, 头数, 丢弃, 专家数=0, 激活=2, 专家倍=2):
        super().__init__()
        self.归一1 = nn.LayerNorm(维度)
        self.注意 = 注意力(维度, 头数, 丢弃)
        self.归一2 = nn.LayerNorm(维度)
        if 专家数:
            self.馈 = 稀疏前馈(维度, 丢弃, 专家数, 激活, 专家倍)
        else:
            self.馈 = 前馈(维度, 丢弃)

    def 取辅损(self):
        v = getattr(self.馈, "辅损", None)
        return v if isinstance(v, torch.Tensor) else None

    def forward(self, x, 掩=None, 缓存=None):
        h, 新缓存 = self.注意(self.归一1(x), 掩, 缓存)
        x = x + h
        x = x + self.馈(self.归一2(x))
        return x, 新缓存


class 模型(nn.Module):
    def __init__(self, 配置=None):
        super().__init__()
        c = dict(配)
        c.update(配置 or {})
        self.配 = c
        self.词嵌 = nn.Embedding(c["词表"], c["维度"])
        self.位嵌 = nn.Embedding(c["上下文"], c["维度"])
        self.丢弃 = nn.Dropout(c["丢弃"])
        self.块表 = nn.ModuleList([块(c["维度"], c["头数"], c["丢弃"],
                                       c.get("专家数", 0) or 0,
                                       c.get("激活", 2) or 2,
                                       c.get("专家倍", 2) or 2) for _ in range(c["层数"])])
        self.末归一 = nn.LayerNorm(c["维度"])
        self.头 = nn.Linear(c["维度"], c["词表"], bias=False)
        if c["绑定权重"]:
            self.头.weight = self.词嵌.weight
        self.apply(self._初始化)
        标 = sum(p.numel() for p in self.parameters())
        self.参数量 = 标

    @staticmethod
    def _初始化(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, 号, 目标=None, 缓存表=None):
        B, T = 号.shape
        if 缓存表 is None:
            起 = 0
        else:
            起 = 缓存表[0][0].shape[2]
        x = self.词嵌(号) + self.位嵌(torch.arange(起, 起 + T, device=号.device))
        x = self.丢弃(x)
        if 目标 is not None:
            掩 = torch.ones(T, T, dtype=torch.bool).tril().to(号.device)
        else:
            掩 = None
        新表 = []
        for i, b in enumerate(self.块表):
            x, 新 = b(x, 掩, None if 缓存表 is None else 缓存表[i])
            新表.append(新)
        x = self.末归一(x)
        出 = self.头(x)
        if 目标 is None:
            return 出, 新表
        损 = F.cross_entropy(出.view(-1, 出.size(-1)), 目标.reshape(-1), ignore_index=-100)
        return 损, 新表

    def 激活参数量(self):
        """MoE 里每个 token 实际要算多少参数（密集模型 = 总参）。"""
        总 = self.参数量
        for b in self.块表:
            馈 = getattr(b, "馈", None)
            if hasattr(馈, "专家表"):
                每 = sum(p.numel() for p in 馈.专家表[0].parameters())
                总 -= 每 * (馈.专家数 - 馈.激活)
        return 总

    def 取辅损(self):
        """MoE 的负载均衡损失（密集模型没有，返回 0）。"""
        总 = None
        for b in self.块表:
            v = b.取辅损()
            if v is not None:
                总 = v if 总 is None else 总 + v
        if 总 is None:
            return torch.zeros((), dtype=torch.float32)
        return 总 / max(1, len(self.块表))

    @torch.no_grad()
    def 生成(self, 提示号, 最长=200, 温度=0.9, 顶k=40, 顶p=0.95, 罚重=1.15, 停号=None, 钩=None):
        """增量生成：每次只喂一个新 token，KV 缓存复用。钩(新字) 用来做流式。"""
        self.eval()
        号 = torch.tensor([提示号], dtype=torch.long)
        出, 缓存 = self(号)
        已出 = []
        停号 = 停号 or []
        for _ in range(最长):
            逻 = 出[:, -1, :].clone()
            if 罚重 and 已出:
                for t in set(已出[-64:]):
                    逻[0, t] = 逻[0, t] / 罚重 if 逻[0, t] > 0 else 逻[0, t] * 罚重
            逻 = 逻 / max(1e-5, 温度)
            if 顶k:
                阈 = torch.topk(逻, min(顶k, 逻.size(-1)))[0][..., -1, None]
                逻 = 逻.masked_fill(逻 < 阈, float("-inf"))
            if 顶p and 顶p < 1.0:
                排序, 序 = torch.sort(逻, descending=True)
                累 = torch.cumsum(F.softmax(排序, dim=-1), dim=-1)
                去 = 累 - F.softmax(排序, dim=-1) > 顶p
                排序[去] = float("-inf")
                逻 = 排序.scatter(1, 序, 排序)
            概 = F.softmax(逻, dim=-1)
            下 = torch.multinomial(概, 1)
            t = int(下.item())
            if t in 停号:
                break
            已出.append(t)
            if 钩:
                钩(t)
            出, 缓存 = self(下, 缓存表=缓存)
        return 已出


def 存模型(模, 路径, 附=None):
    torch.save({"配": 模.配, "权重": 模.state_dict(), "附": 附 or {}}, 路径)


def 读模型(路径, 设备="cpu"):
    d = torch.load(路径, map_location=设备, weights_only=False)
    模 = 模型(d["配"])
    模.load_state_dict(d["权重"])
    模.eval()
    return 模, d.get("附", {})


if __name__ == "__main__":
    模 = 模型()
    print("参数量:", 模.参数量, "= %.2f M" % (模.参数量 / 1e6))
    print("配置:", 模.配)
