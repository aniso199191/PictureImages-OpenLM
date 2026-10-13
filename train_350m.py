# -*- coding: utf-8 -*-
"""训练 350M 档自研底座（约 360M 参数，18 层 / 1280 维 / 16 头）。

为什么单独一个脚本：纯 fp32 AdamW 训练 360M 参数要 5.7G 内存，爆本机
3.83G 单进程上限。这里改成「fp32 权重 + bf16 前向 + 8bit blockwise Adam」，
把 Adam 的 m/v 两个状态压到 8bit（带逐块 fp32 scale），内存压到 ~3.6G。

数据：ownlm/data/train.bin（uint16 token 流）
产物：ownlm/models/base_350M.pt（每 500 步存 ckpt + 最好那份）
日志：ownlm/log/train_350M.jsonl
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

from model import 模型, 存模型  # noqa: E402

配置 = {
    "词表": 4658, "维度": 1280, "层数": 18, "头数": 16,
    "上下文": 256, "丢弃": 0.1, "绑定权重": True,
}

分 = __import__("tokenizer").读分词器()
数夹 = os.environ.get("PILC_DATA_DIR") or os.path.join(大根, "data")
模夹 = os.path.join(大根, "models")
日夹 = os.path.join(大根, "log")
os.makedirs(模夹, exist_ok=True)
os.makedirs(日夹, exist_ok=True)

上 = 配置["上下文"]
训数据 = np.memmap(os.path.join(数夹, "train.bin"), dtype=np.uint16, mode="r")
验数据 = np.memmap(os.path.join(数夹, "validate.bin"), dtype=np.uint16, mode="r")


def 取批(数据, 批, 上):
    n = len(数据) - 上 - 1
    起 = np.random.randint(0, n, size=批)
    x = np.stack([数据[i:i + 上] for i in 起]).astype(np.int64)
    y = np.stack([数据[i + 1:i + 1 + 上] for i in 起]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def 验证(模, 批=6, 轮=16):
    模.eval()
    总, 计 = 0.0, 0
    for _ in range(轮):
        x, y = 取批(验数据, 批, 上)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            损, _ = 模(x, y)
        总 += float(损)
        计 += 1
    return 总 / max(1, 计)


@torch.no_grad()
def 抽句(模, 提示=("你好", "你是谁", "3加5等于几", "计算 12 乘 7", "介绍一下 PictureImages")):
    模.eval()
    出表 = []
    for q in 提示:
        号 = [分.字到号["<s>"], 分.字到号["<问>"]] + 分.编(q) + [分.字到号["<思>"]]
        出 = 模.生成(号, 最长=90, 温度=0.7, 顶k=40,
                     停号=[分.字到号["</s>"], 分.字到号["<问>"], 分.字到号["<s>"]])
        出表.append("%s → %s" % (q, 分.解(出).replace("\n", " ")[:90]))
    return " ｜ ".join(出表)


class 八位Adam:
    """blockwise 8bit Adam：m/v 状态存 int8 + 逐块 fp32 scale，节省 ~6 字节/参数。"""

    def __init__(self, 参数表, lr, b1=0.9, b2=0.95, 衰=0.01, 块=8192):
        self.表 = [p for p in 参数表 if p.requires_grad]
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.衰 = 衰
        self.块 = 块
        self.步 = 0
        self.m8, self.m标, self.v8, self.v标 = [], [], [], []
        for p in self.表:
            n = p.numel()
            n块 = (n + 块 - 1) // 块
            self.m8.append(torch.zeros(n, dtype=torch.int8))
            self.m标.append(torch.ones(n块, dtype=torch.float32))
            self.v8.append(torch.zeros(n, dtype=torch.int8))
            self.v标.append(torch.ones(n块, dtype=torch.float32))

    def zero_grad(self):
        for p in self.表:
            p.grad = None

    def step(self):
        self.步 += 1
        块 = self.块
        校1 = 1 - self.b1 ** self.步
        校2 = 1 - self.b2 ** self.步
        for i, p in enumerate(self.表):
            g = p.grad
            if g is None:
                continue
            平 = p.data.view(-1)
            平g = g.view(-1)
            n = p.numel()
            m8, m标 = self.m8[i], self.m标[i]
            v8, v标 = self.v8[i], self.v标[i]
            n块 = len(m标)
            for j in range(n块):
                s = j * 块
                e = min(n, s + 块)
                m = m8[s:e].float() * m标[j]
                v = v8[s:e].float() * v标[j]
                gs = 平g[s:e]
                m.mul_(self.b1).add_(gs, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(gs, gs, value=1 - self.b2)
                m帽 = m / 校1
                v帽 = v / 校2
                if self.衰:
                    平[s:e].mul_(1 - self.lr * self.衰)
                平[s:e].addcdiv_(m帽, v帽.sqrt().add_(1e-8), value=-self.lr)
                幅m = m.abs().max()
                标m = (幅m / 127.0).clamp(min=1e-6)
                m8[s:e] = (m / 标m).round().clamp(-127, 127).to(torch.int8)
                m标[j] = 标m
                幅v = v.abs().max()
                标v = (幅v / 127.0).clamp(min=1e-6)
                v8[s:e] = (v / 标v).round().clamp(-127, 127).to(torch.int8)
                v标[j] = 标v


def 训(总步=5000, 批=4, 峰=3e-4):
    产物 = os.environ.get("PILC_OUT", "base_350M")
    总步 = int(os.environ.get("PILC_STEPS", 总步))
    模 = 模型(配置)
    初 = os.environ.get("PILC_INIT") or ""
    if 初 and os.path.exists(初):
        d = torch.load(初, map_location="cpu")
        w = d.get("权重", d) if isinstance(d, dict) else d
        缺 = 模.load_state_dict(w, strict=False)
        print("热启动：从 %s 接着训（缺键 %s）" % (初, list(缺.missing_keys)[:3] or "无"), flush=True)
    else:
        初 = ""
    print("%s：参数量 %.2f M｜训练 token %d｜验证 token %d｜批量 %d｜步数 %d｜峰值 lr %g" %
          ("热启动" if 初 else "从零开始", 模.参数量 / 1e6,
           len(训数据), len(验数据), 批, 总步, 峰), flush=True)
    print("配置：%s" % json.dumps(模.配, ensure_ascii=False), flush=True)
    优 = 八位Adam(模.parameters(), lr=峰, 块=8192)
    热 = max(50, 总步 // 40)
    最好 = 9e9
    开 = time.time()
    记 = open(os.path.join(日夹, "train_350M.jsonl"), "a", encoding="utf-8")
    for 步 in range(1, 总步 + 1):
        模.train()
        lr = (峰 * 步 / 热) if 步 < 热 else (峰 / 10 + (峰 - 峰 / 10) * 0.5 *
                                        (1 + math.cos(math.pi * (步 - 热) / max(1, 总步 - 热))))
        优.lr = lr
        x, y = 取批(训数据, 批, 上)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            主损, _ = 模(x, y)
        优.zero_grad()
        主损.backward()
        torch.nn.utils.clip_grad_norm_(模.parameters(), 1.0)
        优.step()
        if 步 % 25 == 0 or 步 == 1:
            用 = time.time() - 开
            行 = {"步": 步, "总步": 总步, "损": round(float(主损), 4),
                  "lr": round(lr, 6), "秒": round(用, 1), "token": 步 * 批 * 上,
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
    总 = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    批 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    峰 = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
    训(总, 批, 峰)
