"""自研大底座的分词器：字级 + 高频词组（自己统计，不借任何外部词表）。

比现役的多一层「词组」：把语料里出现 ≥阈值 的二字/三字组合也收进词表，
一句话的 token 数能少掉三成左右 —— 同样的算力能多喂三成字。
"""
import json
import os
import re

大根 = os.path.dirname(os.path.abspath(__file__))
语料夹 = os.path.join(大根, "corpus")
存 = os.path.join(大根, "vocab.json")

特殊 = ["<pad>", "<s>", "</s>", "<问>", "<答>", "<unk>", "<思>"]
必留 = "".join(chr(c) for c in range(33, 127)) + " ¥　" + "，。、！？：；“”‘’（）《》—…·【】〔〕÷×＝＋－"
词组最小次数 = 150
最多词组 = 3000


def 收(计, 词计, 文):
    for 字 in 文:
        计[字] = 计.get(字, 0) + 1
    for 长 in (2, 3):
        for i in range(len(文) - 长 + 1):
            s = 文[i:i + 长]
            if "\n" in s or " " in s:
                continue
            词计[s] = 词计.get(s, 0) + 1


def 统计():
    计, 词计 = {}, {}
    with open(os.path.join(语料夹, "raw_corpus.txt"), encoding="utf-8") as f:
        for 行 in f:
            收(计, 词计, 行.rstrip("\n"))
    with open(os.path.join(语料夹, "chat.jsonl"), encoding="utf-8") as f:
        for 行 in f:
            行 = 行.strip()
            if not 行:
                continue
            try:
                d = json.loads(行)
            except Exception:
                continue
            for k in ("问", "答", "思"):
                收(计, 词计, str(d.get(k, "")))
    必留集 = set(必留)
    单字 = [c for c, n in sorted(计.items(), key=lambda x: -x[1]) if n >= 2 and c not in 必留集]
    单字 = [c for c in 必留 if c in 计] + 单字
    词组 = [w for w, n in sorted(词计.items(), key=lambda x: -x[1])[:4000] if n >= 词组最小次数]
    词组 = [w for w in 词组 if all(ch in set(单字) for ch in w)][:最多词组]
    return 特殊 + 单字 + 词组, 计, 词计


class 分词器:
    def __init__(self, 表):
        self.表 = list(表)
        self.字到号 = {c: i for i, c in enumerate(self.表)}
        self.号到字 = {i: c for i, c in enumerate(self.表)}
        self.长词 = sorted([c for c in self.字到号 if len(c) > 1], key=len, reverse=True)
        self.长序 = sorted({len(c) for c in self.字到号 if len(c) > 1}, reverse=True)

    @property
    def 词表大小(self):
        return len(self.表)

    def 编(self, 文):
        out = []
        i, n = 0, len(文 or "")
        到号 = self.字到号
        没了 = 到号["<unk>"]
        while i < n:
            命中 = None
            for 长 in self.长序:          # 从最长往下试，哈希查表，不做逐条比对
                if i + 长 <= n:
                    段 = 文[i:i + 长]
                    if 段 in 到号:
                        命中 = 段
                        break
            if 命中:
                out.append(到号[命中])
                i += len(命中)
            else:
                out.append(到号.get(文[i], 没了))
                i += 1
        return out

    def 解(self, 号表):
        out = []
        for i in 号表:
            c = self.号到字.get(int(i), "")
            if c in ("<pad>", "<s>", "</s>", "<思>"):
                continue
            out.append("" if c == "<unk>" else c)
        return "".join(out)


def 存词表(表):
    json.dump({"表": 表, "说明": "自研大底座的词表：字 + 高频词组，从我们自己的语料统计出来的"},
              open(存, "w", encoding="utf-8"), ensure_ascii=False)
    return 表


def 读分词器(路径=None):
    return 分词器(json.load(open(路径 or 存, encoding="utf-8"))["表"])


if __name__ == "__main__":
    表, 计, 词计 = 统计()
    存词表(表)
    单字数 = len([c for c in 表 if len(c) == 1])
    print(json.dumps({"词表大小": len(表), "单字": 单字数, "词组": len(表) - 单字数 - len(特殊),
                      "语料不同字": len(计), "不同词组": len(词计)}, ensure_ascii=False))
