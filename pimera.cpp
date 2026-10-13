#include "pimera.hpp"
#include <iostream>
#include <string>

using namespace pimera;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Pimera 推理框架\n"
                  << "用法:\n"
                  << "  pimera chat <模型.pimera> <提示>\n"
                  << "  pimera clean <源.pt> <目.pimera>\n"
                  << "  pimera info <模型.pimera>\n";
        return 1;
    }

    std::string 令 = argv[1];
    std::string 路径 = argv[2];

    try {
        if (令 == "info") {
            模型 m(路径);
            std::cout << "词表: " << m.词表 << "\n"
                      << "维度: " << m.维度 << "\n"
                      << "层数: " << m.层数 << "\n"
                      << "头数: " << m.头数 << "\n"
                      << "上下文: " << m.上下文 << "\n";
            return 0;
        }

        if (令 == "chat") {
            if (argc < 4) { std::cerr << "缺提示\n"; return 1; }
            模型 m(路径);
            std::string 提示 = argv[3];
            std::cout << "[Pimera] 加载模型 " << m.维度 << "x" << m.层数
                      << "，提示: " << 提示 << "\n";
            std::cout << "[Pimera] 逐层推理中（mmap 按需加载）...\n";
            // 演示：读第 0 层验证加载
            std::vector<float> 层;
            m.读层(0, 层);
            std::cout << "[Pimera] 第 0 层 " << 层.size() << " 参数已加载\n";
            std::cout << "[Pimera] 推理完成（框架骨架已通，全图生成待接 tokenizer）\n";
            return 0;
        }

        if (令 == "clean") {
            if (argc < 4) { std::cerr << "缺目标文件\n"; return 1; }
            std::string 目 = argv[3];
            return 清理压缩(路径, 目) ? 0 : 1;
        }

        std::cerr << "不认的令: " << 令 << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "错: " << e.what() << "\n";
        return 1;
    }
}
