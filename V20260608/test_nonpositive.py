# test_nonpositive.py
# 算例2：非正定矩阵检测
# 测试矩阵 K = [[1,2],[2,1]]，应为非正定（特征值 3 和 -1），LDL^T 应检测到 D[1] <= 0。

import numpy as np                               # 导入NumPy库，用于数值计算
from equilibrium_solver import ldlt_factor      # 从自定义求解器模块导入 LDL^T 分解函数

def test_nonpositive():                          # 定义测试函数
    # 构造非正定矩阵 K：对称，但特征值为 3 和 -1，不是正定矩阵
    K = np.array([[1.0, 2.0],                    # 第一行：1, 2
                  [2.0, 1.0]])                   # 第二行：2, 1
    R = np.array([1.0, 1.0])                     # 任意右端项，仅用于展示，实际未使用

    print("=" * 60)                              # 打印分隔线
    print("算例2：非正定矩阵检测")                # 输出标题
    print(f"矩阵 K:\n{K}")                       # 打印矩阵 K
    print(f"右端项 R: {R}")                      # 打印右端项
    print("尝试 LDL^T 分解...")                  # 提示开始分解

    try:                                          # 捕获可能发生的异常
        L, D = ldlt_factor(K)                    # 调用 LDL^T 分解，对于非正定矩阵应抛出异常
        print("分解成功（不应发生）")              # 若未抛出异常，则打印错误信息
        print(f"D = {D}")                        # 打印对角元 D
    except ValueError as e:                      # 捕获预期的 ValueError（主元 <= 0）
        print(f"正确捕获异常: {e}")               # 输出异常信息，符合预期
        print("程序已停止 LDL^T 求解，符合预期。")  # 说明程序行为正确

    print("\n注：该矩阵的特征值分别为 3 和 -1，因此不是正定矩阵。")
    print("有限元刚度矩阵在缺少足够约束时会出现零主元（刚体位移），")
    print("导致 D[j] = 0，应停止求解并提示用户添加边界条件。")

if __name__ == "__main__":                       # 如果直接运行此脚本（而非作为模块导入）
    test_nonpositive()                           # 调用测试函数