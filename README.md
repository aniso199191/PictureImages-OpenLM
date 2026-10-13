# PictureImages-OpenLM

从零手写的 decoder-only Transformer 语言模型，全自研、可离线、纯 CPU 可训可推理。目标是练出一个诚实的 350M 档中文对话底座，并持续走向 6B。

A from-scratch, self-authored decoder-only Transformer language model. Fully self-contained, offline-capable, trainable and runnable on pure CPU. Honest goal: a 350M-parameter Chinese chat base, on a roadmap toward 6B.

---

## ⚠️ 关于规模，先说清楚（Be honest about size）

本仓库里的模型 **不是 6B，也不是 MoE**。当前训练的 checkpoint 参数量如实标在下方：

- `base_350M` — **360.39 M 参数**（18 层 / 1280 维 / 16 头 / 词表 4658 / 上下文 256），dense 结构，无 router、无专家。
- 更早的 `base_6b2` / `base_6b` 文件名里的「6b」是**训练批次号**，其真实参数量是 **62.2 M**（12 层 / 640 维 / 10 头）。

之所以敢开源：小模型的每一步都是真的——参数对得上、权重能加载、推理能出字、验证损失能复现。没有把 62M 说成 6B，也没有把 dense 说成 MoE。

---

## 目录结构

```
.
├── model.py           # 模型骨架：多头注意力 + 前馈 + pre-LN + 绑定权重 + KV 缓存增量生成
├── tokenizer.py       # 字符级分词器（词表 4658，可自举，不依赖外部 tokenizer）
├── data.py            # 语料 → train.bin / validate.bin（uint16 token 流）
├── train.py           # 小档训练（fp32 AdamW，从零 / 热启动）
├── train_350m.py      # 350M 档训练（bf16 前向 + 8bit blockwise Adam，塞进 4G 内存）
├── fast_path.py       # 纯 CPU 推理入口（读 .pt 权重 + 增量生成）
├── 评估.py            # 验证集损失复现（3827 窗 / 979,712 位置铺满）
├── 试聊.py            # 新旧底座同题对看
└── config_*.json      # 各档模型配置
```

## 快速开始

### 训练 350M 档（纯 CPU，内存上限 ~3.8G 可跑）

```bash
# 依赖：torch + numpy（CPU 版即可）
pip install torch numpy

# 先准备数据（data.py 产出 train.bin / validate.bin）
python data.py

# 从零训练 350M（5000 步，约 26 小时 @ 4 核）
THREADS=4 python train_350m.py 5000 4 3e-4
```

### 推理（加载练好的权重对话）

```bash
python fast_path.py "你好" 24
```

### 复现验证损失

```bash
python 评估.py base_350M.pt
```

## 模型档位

| 档位 | 参数 | 配置 | 说明 |
|---|---|---|---|
| 19M | 19 M | `config_19M.json` | 448 维 / 7 层 / 7 头 |
| 55M | 55 M | — | 640 维 / 12 层 / 10 头（旧线上档） |
| 62M | 62.2 M | — | 批次名「6b2」实际参数 |
| **350M** | **360.39 M** | `config_350M.json` | 1280 维 / 18 层 / 16 头，当前主推 |

## Pimera 框架（C++ 推理 + 超压缩）

| 项 | 值 |
|---|---|
| 语言 | C++17，零依赖 |
| 模型格式 | `.pimera`（8bit blockwise 量化）→ `.pimer`（zstd/xz 超压缩） |
| 加载 | mmap 按需读盘，不整模型进内存 |
| 清理 | 训练后自动删 optimizer/梯度残留 |

### 量化档

| 档 | 定位 | 1B 体积 |
|---|---|---|
| 1MiraK | 思考最强 | 1.2 GB |
| M4QK | 回答最强 | 1.5 GB |
| Tyek_ML1 | 推理最强 | 1.7 GB |
| Worky1 | 工作 AI | 1.4 GB |
| BrigX | 最大最全 | 4.7 GB |
| FastX | 最快流式 | 1.2 GB |
| Agsi | AGI 匹配 | 2.1 GB |

> 体积为设计规格（跑在足够硬件上），非本机实测。

### 编译

```bash
g++ -O3 -std=c++17 -o pimera pimera.cpp
```

### 用

```bash
./pimera info model.pimera
python convert_to_pimera.py 转 model.pt model.pimera
python convert_to_pimera.py 压 model.pimera model.pimer
```

### 实测压缩率

| 阶段 | 体积 | 比例 |
|---|---|---|
| PyTorch fp32 | 248.9 MB | 100% |
| .pimera 8bit | 71.0 MB | 28.5% |
| .pimer 超压缩 | 63.4 MB | 25.5% |

## 诚实的技术债（Known limitations）

- **纯 CPU、无 GPU**：训练慢（350M 约 55 tok/s），6B 级在此硬件上训不动（Adam 状态内存需求是单进程上限的 10 倍以上）。
- **词表小**（4658）：字符级，长文本编码效率低，后续计划换更大词表 + BPE。
- **数据量少**：当前训练集 ~2200 万 token，350M 模型按 Chinchilla 规律远未喂饱——能力上限受数据而非架构约束。
- **无 MoE**：dense 结构。MoE 需先有「能分工的专家」，在算力到位前不硬上。

## 路线图（Roadmap to 6B）

1. ✅ 350M dense 训练 + 开源
2. ⬜ 大词表（32K+ BPE）+ RoPE 位置编码
3. ⬜ 高质量中文对话数据扩量（10 亿 token 级）
4. ⬜ 1B dense
5. ⬜ 6B dense（需 GPU 算力赞助 / 云 GPU）

## License

MIT License（见 LICENSE 文件）
