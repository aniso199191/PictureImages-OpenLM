#ifndef PIMERA_HPP
#define PIMERA_HPP

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace pimera {

// ---------- 文件头：64 字节，对齐 ----------
#pragma pack(push, 1)
struct 头 {
    char 魔数[8];        // "PIMERA01"
    uint32_t 版本;       // 1
    uint32_t 词表;       // 4658
    uint32_t 维度;       // 1280
    uint32_t 层数;       // 18
    uint32_t 头数;       // 16
    uint32_t 上下文;     // 256
    uint32_t 量化位;     // 8
    uint32_t 块大小;     // 8192
    uint64_t 权重偏移;   // 量化权重区起始
    uint64_t 标偏移;     // scale 区起始
    uint64_t 嵌偏移;     // 词嵌入 fp32 区起始
    uint64_t 总大小;     // 文件总字节
};
#pragma pack(pop)

static_assert(sizeof(头) == 72, "头结构打包后应为 72 字节");

// 磁盘上固定 64 字节头（后 8 字节留作对齐/扩展）
struct __attribute__((packed)) 磁盘头 {
    char 魔数[8];
    uint32_t 版本, 词表, 维度, 层数, 头数, 上下文, 量化位, 块大小;
    uint32_t 权重偏移32, 标偏移32, 嵌偏移32, 总大小32;
    uint32_t 保留[2];
};
static_assert(sizeof(磁盘头) == 64, "磁盘头必须 64 字节");

// ---------- mmap 只读映射 ----------
class 映射 {
public:
    映射(const std::string& 路径) : 数据(nullptr), 大小(0), fd(-1) {
        fd = open(路径.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("打不开: " + 路径);
        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); throw std::runtime_error("stat 失败"); }
        大小 = (size_t)st.st_size;
        数据 = (const uint8_t*)mmap(nullptr, 大小, PROT_READ, MAP_PRIVATE, fd, 0);
        if (数据 == MAP_FAILED) { close(fd); throw std::runtime_error("mmap 失败"); }
    }
    ~映射() {
        if (数据 && 数据 != MAP_FAILED) munmap((void*)数据, 大小);
        if (fd >= 0) close(fd);
    }
    const uint8_t* 读(size_t 偏移, size_t 字节) const {
        if (偏移 + 字节 > 大小) throw std::runtime_error("越界读");
        return 数据 + 偏移;
    }
    const 磁盘头* 读头() const {
        if (大小 < sizeof(磁盘头)) throw std::runtime_error("文件太小");
        return (const 磁盘头*)数据;
    }
private:
    const uint8_t* 数据;
    size_t 大小;
    int fd;
};

// ---------- 8bit blockwise 量化 ----------
struct 块 {
    int8_t 值[8192];
    float 标;
};

inline void 量化(const float* 源, int8_t* 目, float& 标, size_t n) {
    float 幅 = 0;
    for (size_t i = 0; i < n; ++i) 幅 = std::max(幅, std::abs(源[i]));
    标 = std::max(幅 / 127.0f, 1e-6f);
    for (size_t i = 0; i < n; ++i) {
        float v = 源[i] / 标;
        v = std::max(-127.0f, std::min(127.0f, std::round(v)));
        目[i] = (int8_t)v;
    }
}

inline void 反量化(const int8_t* 源, float 标, float* 目, size_t n) {
    for (size_t i = 0; i < n; ++i) 目[i] = 源[i] * 标;
}

// ---------- 模型：逐层加载，不整进内存 ----------
class 模型 {
public:
    模型(const std::string& 路径) : 映(路径) {
        const 磁盘头* h = 映.读头();
        if (std::memcmp(h->魔数, "PIMERA01", 8) != 0)
            throw std::runtime_error("不是 .pimera 文件");
        词表 = h->词表; 维度 = h->维度; 层数 = h->层数;
        头数 = h->头数; 上下文 = h->上下文; 块大小 = h->块大小;
        权重偏 = h->权重偏移32; 标偏 = h->标偏移32; 嵌偏 = h->嵌偏移32;
    }

    // 读一层权重（按需从盘里 mmap 出来，不占常驻内存）
    void 读层(size_t 层号, std::vector<float>& 出) {
        // 每层参数数: qkv 3*维*维 + 出 维*维 + 前馈 2*维*4维 = 12*维²
        size_t 每层参 = 12ULL * 维度 * 维度;
        size_t 每层块 = (每层参 + 块大小 - 1) / 块大小;
        size_t 每层权 = 每层块 * 块大小;        // int8 字节
        size_t 每层标 = 每层块 * sizeof(float); // scale 字节

        size_t 层权偏 = 权重偏 + 层号 * 每层权;
        size_t 层标偏 = 标偏 + 层号 * 每层标;

        const int8_t* 权 = (const int8_t*)映.读(层权偏, 每层权);
        const float* 标 = (const float*)映.读(层标偏, 每层标);

        出.resize(每层参);
        for (size_t b = 0; b < 每层块; ++b) {
            size_t s = b * 块大小;
            size_t e = std::min(每层参, s + 块大小);
            反量化(权 + s, 标[b], 出.data() + s, e - s);
        }
    }

    // 词嵌入（fp32 原样存，不量化）
    const float* 词嵌(size_t 词号) const {
        return (const float*)映.读(嵌偏 + 词号 * 维度 * sizeof(float),
                                    维度 * sizeof(float));
    }

    uint32_t 词表, 维度, 层数, 头数, 上下文, 块大小;
private:
    映射 映;
    uint64_t 权重偏, 标偏, 嵌偏;
};

// ---------- 训练后清理：删垃圾 + 压到最小 ----------
inline bool 清理压缩(const std::string& 源pt, const std::string& 目pimera) {
    // 源pt 是 PyTorch 存的 .pt（fp32 权重 + 可能带 optimizer 状态）
    // 这里只做演示：读 PyTorch 二进制太复杂，实际用 Python 转
    std::cerr << "清理压缩：先用 Python 把 .pt 转成裸权重，再调本函数\n";
    std::cerr << "  python convert_to_pimera.py " << 源pt << " " << 目pimera << "\n";
    return false;
}

} // namespace pimera

#endif // PIMERA_HPP
