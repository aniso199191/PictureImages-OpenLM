"""把自研大的语料打成 token 流（uint16）。

产物：ownlm/data/train.bin、validate.bin、info.json
留出：问答对尾部 2% + 生语料尾部 3% + 知识料尾部 2% 做验证（验证集不进训练）。

2026-10-09 改：
  · 加进「知识料」——语料/知识料/维基_*.jsonl（知识库 138 万条维基切出来的）。
    以前这里只吃 raw_corpus.txt + chat.jsonl，知识库一条都没进过训练，「知识少」的真因就在这。
  · 改成**分片落盘**：以前把全部 token 堆成一个 Python list 再 np.array，
    几十万 token 还行，上亿 token 会把内存直接顶穿（单进程上限 3.83G）。
    现在每 400 万 token 落一次盘，内存恒定。
  · 数据目录可用环境变量 PILC_DATA_DIR 指到别处（试跑用，不碰线上训练正在读的 bin）。
"""
import json
import os
import random
import re
import sys

import numpy as np

大根 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 大根)
from tokenizer import 读分词器  # noqa: E402

分 = 读分词器()
数夹 = os.environ.get("PILC_DATA_DIR") or os.path.join(大根, "data")
语料夹 = os.path.join(大根, "corpus")
知夹 = os.path.join(语料夹, "知识料")
os.makedirs(数夹, exist_ok=True)
随机 = random.Random(20260928)

行表 = [x.rstrip("\n") for x in open(os.path.join(语料夹, "raw_corpus.txt"), encoding="utf-8") if x.strip()]
配对 = [json.loads(x) for x in open(os.path.join(语料夹, "chat.jsonl"), encoding="utf-8") if x.strip()]

# 自述纠正：老语料里模型自报的是 19M 档的老身份（5 层 / 320 维 / 5 头 / 6.99M 参数），
# 现在的底座是 12 层 640 维 10 头 / 65.18M 真参数。不纠正就是拿旧规格教它说错自己的参数 —— 名字和内容必须一致。
自述对 = [
    (re.compile(r"19\.09\s*M"), "65.18M"),
    (re.compile(r"6\.99\s*M"), "65.18M"),
    (re.compile(r"5\s*层[、·]\s*320\s*维[、·]\s*5\s*个?\s*注意力头"), "12 层、640 维、10 个注意力头"),
    (re.compile(r"5\s*层[、·]\s*320\s*维[、·]\s*5\s*头"), "12 层、640 维、10 头"),
    (re.compile(r"5\s*层、320\s*维、5\s*个注意力头"), "12 层、640 维、10 个注意力头"),
    (re.compile(r"448\s*[×x]\s*7\s*[×x]\s*7"), "640×12×10"),
]


def 自述纠正(文):
    if not isinstance(文, str):
        return 文
    for 模, 换 in 自述对:
        文 = 模.sub(换, 文)
    return 文


行表 = [自述纠正(x) for x in 行表]
配对 = [{k: (自述纠正(v) if isinstance(v, str) else v) for k, v in d.items()} for d in 配对]
print("自述纠正已过：命中的行/对里旧规格（6.99M / 5 层 320 维）一律改写成现底座规格", flush=True)

知行 = []
if os.path.isdir(知夹):
    for 名 in sorted(os.listdir(知夹)):
        if not 名.endswith(".jsonl"):
            continue
        for 行 in open(os.path.join(知夹, 名), encoding="utf-8", errors="ignore"):
            行 = 行.strip()
            if not 行:
                continue
            try:
                对 = json.loads(行)
            except ValueError:
                continue
            if isinstance(对, dict) and isinstance(对.get("文"), str):
                知行.append(自述纠正(对["文"]))
print("生语料行:%d ｜问答对:%d ｜知识料行:%d" % (len(行表), len(配对), len(知行)), flush=True)

B = {k: 分.字到号[k] for k in ("<s>", "</s>", "<问>", "<答>", "<思>")}


def 生段(行):
    return [B["<s>"]] + 分.编(行) + [B["</s>"]]


def 对段(d):
    out = [B["<s>"], B["<问>"]] + 分.编(d["问"]) + [B["<思>"]]
    if d.get("思"):
        out += 分.编(d["思"])
    out += [B["<答>"]] + 分.编(d["答"]) + [B["</s>"]]
    return out


切行 = int(len(行表) * 0.97)
训行, 验行 = 行表[:切行], 行表[切行:]
切对 = int(len(配对) * 0.98)
训对, 验对 = 配对[:切对], 配对[切对:]
切知 = int(len(知行) * 0.98)
训知, 验知 = 知行[:切知], 知行[切知:]
print("训练对:%d 验证对:%d ｜训练知识行:%d 验证知识行:%d ｜训练生语料行:%d 验证生语料行:%d"
      % (len(训对), len(验对), len(训知), len(验知), len(训行), len(验行)), flush=True)

块token = 4_000_000


class 器:
    """每 400 万 token 落一次盘，内存恒定；收尾把总 token 数交出来。"""

    def __init__(self, 路):
        self.f = open(路, "wb")
        self.缓 = []
        self.缓长 = 0
        self.总 = 0

    def 加(self, 号表):
        self.缓.extend(号表)
        self.缓长 += len(号表)
        if self.缓长 >= 块token:
            self._落()

    def _落(self):
        if not self.缓:
            return
        np.array(self.缓, dtype=np.uint16).tofile(self.f)
        self.总 += self.缓长
        self.缓 = []
        self.缓长 = 0

    def 收(self):
        self._落()
        self.f.close()
        return self.总


def 打(路, 对表, 行表, 知表):
    它 = 器(路)
    for d in 对表:
        它.加(对段(d))
    for 行 in 行表:
        它.加(生段(行))
    for 文 in 知表:
        它.加(生段(文))
    return 它.收()


训t = 打(os.path.join(数夹, "train.bin"), 训对, 训行, 训知)
验t = 打(os.path.join(数夹, "validate.bin"), 验对, 验行, 验知)

信 = {"词表": 分.词表大小, "训练token": 训t, "验证token": 验t,
      "训练对": len(训对), "验证对": len(验对), "训练行": len(训行), "验证行": len(验行),
      "训练知识行": len(训知), "验证知识行": len(验知),
      "训练百万token": round(训t / 1e6, 2), "数据夹": 数夹}
json.dump(信, open(os.path.join(数夹, "info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(信, ensure_ascii=False))
