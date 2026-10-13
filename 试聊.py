"""试聊.py —— 同一批问题，分别拿「线上底座」和「新底座」各答一遍，人工对看。

用法：python 试聊.py [新底座文件名，默认 base_6b2.pt]
"""
import os
import subprocess
import sys

根 = os.path.dirname(os.path.abspath(__file__))
妈 = os.path.join(根, "fast_path.py")
老 = os.environ.get("PILC_OLD_PT", "base_55M.pt")
新 = sys.argv[1] if len(sys.argv) > 1 else "base_6b2.pt"
问 = ["你好", "你是谁", "3加5等于几", "计算 12 乘 7", "介绍一下 PictureImages", "你会干什么", "今天天气怎么样"]


def 跑(源, 提示):
    环 = dict(os.environ, PILC_BASE_PT=源, THREADS="4")
    出 = subprocess.run([sys.executable, 妈, 提示, "24"], capture_output=True, text=True, env=环, timeout=900)
    行 = [x for x in 出.stdout.splitlines() if x.startswith("出来：")]
    return 行[0][3:] if 行 else ("(空/错) " + (出.stderr.strip().splitlines() or [""])[-1])


for q in 问:
    print("【问】%s" % q, flush=True)
    print("  旧 %s → %s" % (老, 跑(老, q)), flush=True)
    print("  新 %s → %s" % (新, 跑(新, q)), flush=True)
print("对看完毕", flush=True)
