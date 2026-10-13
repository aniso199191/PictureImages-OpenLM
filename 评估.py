"""给 checkpoint 算**真**验证损：不用随机抽样（20 批 ×256 token 的随机取样噪声有 ±0.13，
之前拿它比大小根本不可靠），改成把整条验证流按固定步长铺满、逐批前向。

用法：python 评估.py models/base_55M.pt [更多 ckpt...]
"""
import os
import sys
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
大根 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 大根)
from model import 模型  # noqa: E402

验夹 = os.environ.get("PILC_DATA_DIR") or os.path.join(大根, "data")
上 = 256
步长 = 128                      # 窗口重叠一半，铺满整条流
验 = np.memmap(os.path.join(验夹, "validate.bin"), dtype=np.uint16, mode="r")
起表 = list(range(0, len(验) - 上 - 1, 步长))
print("验证流 %d token ｜铺 %d 个窗口（每窗口 256）｜合计 %d 个 token 位置"
      % (len(验), len(起表), len(起表) * 上), flush=True)


@torch.no_grad()
def 铺满验损(模, 批=8):
    模.eval()
    总损 = 0.0
    计 = 0
    for i in range(0, len(起表), 批):
        起 = 起表[i:i + 批]
        x = torch.from_numpy(np.stack([验[j:j + 上] for j in 起]).astype(np.int64))
        y = torch.from_numpy(np.stack([验[j + 1:j + 1 + 上] for j in 起]).astype(np.int64))
        损, _ = 模(x, y)
        总损 += float(损) * len(起)
        计 += len(起)
    return 总损 / 计


for p in sys.argv[1:]:
    if not os.path.exists(p):
        print("%-34s 文件不在" % p)
        continue
    d = torch.load(p, map_location="cpu")
    模 = 模型()
    w = d.get("权重", d) if isinstance(d, dict) else d
    缺 = 模.load_state_dict(w, strict=False)
    附 = d.get("附", {}) if isinstance(d, dict) else {}
    配 = d.get("配", {}) or 模.配
    参 = sum(t.numel() for t in w.values() if hasattr(t, "numel"))
    v = 铺满验损(模)
    print("%-30s 层%2d 维%4d 头%3d ｜参 %s ｜存盘时(20批随机) %s ｜真验损(铺满) %.5f ｜缺键 %d"
          % (os.path.basename(p), 配.get("层数"), 配.get("维度"), 配.get("头数"), format(参, ","),
             round(附.get("验证损", 0), 4) if 附.get("验证损") else "—", v, len(缺.missing_keys)), flush=True)
