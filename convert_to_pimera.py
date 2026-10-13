# -*- coding: utf-8 -*-
""".pimer 超压缩格式：把 PyTorch .pt 压成 .pimera（8bit 量化）→ 再 zstd/xz 超压。

目标压缩比（设计指标，非本机实测）：
  5GB → 2GB (40%)  |  4GB → 1GB (25%)  |  2GB → 900MB (45%)

实际能压多少取决于权重分布——量化后 int8 数据 zstd 通常只能再压 5-15%，
但 optimizer 状态、梯度缓存、fp32 转 int8 本身就能砍掉 75% 体积。
"""
import os
import struct
import subprocess
import sys

import numpy as np
import torch

魔数 = b"PIMERA01"
版本 = 1
块大小 = 8192


def 转(源pt, 目pimera):
    """PyTorch .pt → .pimera（8bit blockwise 量化）"""
    d = torch.load(源pt, map_location="cpu", weights_only=False)
    配 = d.get("配", {})
    w = d.get("权重", d) if isinstance(d, dict) else d

    词表 = 配.get("词表", 4658)
    维度 = 配.get("维度", 640)
    层数 = 配.get("层数", 12)
    头数 = 配.get("头数", 10)
    上下文 = 配.get("上下文", 256)

    # 收集所有张量：词嵌(不量化) + 层权重(量化)
    词嵌键 = "词嵌.weight"
    词嵌 = w.pop(词嵌键, None)
    if 词嵌 is None:
        raise ValueError("缺词嵌权重")

    层键 = sorted([k for k in w if k.startswith("块表.") and ".weight" in k],
                  key=lambda x: (int(x.split(".")[1]), x))

    # 写头（64 字节）
    头 = struct.pack("<8s8I4I2I",
                     魔数, 版本, 词表, 维度, 层数, 头数, 上下文, 8, 块大小,
                     0, 0, 0, 0, 0, 0)  # 偏移后面回填

    权重区 = bytearray()
    标区 = bytearray()

    for k in 层键:
        t = w[k].float().numpy().flatten()
        n = len(t)
        n块 = (n + 块大小 - 1) // 块大小
        for b in range(n块):
            s, e = b * 块大小, min(n, (b + 1) * 块大小)
            幅 = np.abs(t[s:e]).max()
            标 = max(幅 / 127.0, 1e-6)
            量 = np.round(t[s:e] / 标).clip(-127, 127).astype(np.int8)
            权重区.extend(量.tobytes())
            标区.extend(struct.pack("<f", 标))

    嵌区 = 词嵌.float().numpy().astype(np.float32).tobytes()

    权重偏移 = 64
    标偏移 = 权重偏移 + len(权重区)
    嵌偏移 = 标偏移 + len(标区)
    总大小 = 嵌偏移 + len(嵌区)

    头 = struct.pack("<8s8I4I2I",
                     魔数, 版本, 词表, 维度, 层数, 头数, 上下文, 8, 块大小,
                     权重偏移, 标偏移, 嵌偏移, 总大小, 0, 0)

    with open(目pimera, "wb") as f:
        f.write(头)
        f.write(权重区)
        f.write(标区)
        f.write(嵌区)

    源大 = os.path.getsize(源pt)
    目大 = os.path.getsize(目pimera)
    print("转换: %s → %s" % (源pt, 目pimera))
    print("  源: %.1f MB → 目: %.1f MB (%.1f%%)" % (源大/1e6, 目大/1e6, 目大/源大*100))
    return 目pimera


def 压(源, 目):
    """zstd 最高档 + xz 兜底，取最小"""
    候选 = []
    # zstd --ultra -22
    z目 = 目 + ".zst"
    try:
        subprocess.run(["zstd", "--ultra", "-22", "-f", "-o", z目, 源],
                       check=True, capture_output=True, timeout=300)
        候选.append(z目)
    except Exception:
        pass
    # xz -9e
    x目 = 目 + ".xz"
    try:
        subprocess.run(["xz", "-9e", "-f", "-k", "-c", 源],
                       stdout=open(x目, "wb"), check=True, timeout=300)
        候选.append(x目)
    except Exception:
        pass
    if not 候选:
        print("zstd/xz 都没装，直接复制")
        import shutil
        shutil.copy(源, 目)
        return 目
    最小 = min(候选, key=os.path.getsize)
    import shutil
    shutil.move(最小, 目)
    for f in 候选:
        if os.path.exists(f):
            os.remove(f)
    源大 = os.path.getsize(源)
    目大 = os.path.getsize(目)
    print("压缩: %s → %s" % (源, 目))
    print("  源: %.1f MB → 目: %.1f MB (%.1f%%)" % (源大/1e6, 目大/1e6, 目大/源大*100))
    return 目


def 清垃圾(模夹):
    """训练后自动删 optimizer 状态、梯度缓存、临时文件"""
    删了 = 0
    for f in os.listdir(模夹):
        if f.endswith(("_ckpt.pt", ".tmp", ".grad", ".optim")):
            os.remove(os.path.join(模夹, f))
            删了 += 1
            print("  删: %s" % f)
    print("清垃圾: 删了 %d 个残留" % 删了)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法:")
        print("  convert_to_pimera.py 转 <源.pt> <目.pimera>")
        print("  convert_to_pimera.py 压 <源> <目.pimer>")
        print("  convert_to_pimera.py 清 <模型目录>")
        sys.exit(1)
    令 = sys.argv[1]
    if 令 == "转":
        转(sys.argv[2], sys.argv[3])
    elif 令 == "压":
        压(sys.argv[2], sys.argv[3])
    elif 令 == "清":
        清垃圾(sys.argv[2])
    else:
        print("不认的令: %s" % 令)
