# Pimera 框架

自研 C++ 推理框架：mmap 按需加载、逐层推理、8bit 量化权重、训练后自动清垃圾 + 超强压缩。

Self-authored C++ inference framework: mmap on-demand loading, layer-by-layer execution, 8bit quantized weights, auto-cleanup + extreme compression after training.

---

## 规格

| 项 | 值 |
|---|---|
| 语言 | C++17（无依赖，单文件头 + 单文件实现） |
| 权重格式 | `.pimera`（8bit blockwise 量化 + fp32 scale） |
| 加载方式 | mmap（不整模型进内存，逐层读盘） |
| 推理 | KV 缓存增量生成，上下文 256 |
| 训练后 | 自动删 optimizer 状态 + 梯度缓存，权重压到 1/4 |

## 7 档量化标准

| 档名 | 定位 | 每 1B 参数体积 |
|---|---|---|
| **1MiraK** | 思考最强 | 1.2 GB |
| **M4QK** | 回答最强 | 1.5 GB |
| **Tyek_ML1** | 推理最强 | 1.7 GB |
| **Worky1** | 工作类 AI | 1.4 GB |
| **BrigX** | 最大最完整 | 4.7 GB |
| **FastX** | 最快流式 | 1.2 GB |
| **Agsi** | AGI/ASI 匹配 | 2.1 GB |

> 体积 = 量化后权重文件大小（含 scale/索引）。以上均为「跑在足够硬件上」的设计规格，非本机实测。

## 编译

```bash
g++ -O3 -std=c++17 -o pimera pimera.cpp
```

## 用

```bash
# 推理（加载 .pimera 权重）
./pimera chat model.pimera "你好"

# 训练后清理 + 压缩（自动删垃圾，压到最小）
./pimera clean model.pt model.pimera
```

## License

MIT
