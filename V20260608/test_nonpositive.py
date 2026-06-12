"""
test_nonpositive.py
算例2：非正定矩阵检测
测试矩阵 K = [[1,2],[2,1]]，应为非正定（特征值 3 和 -1），LDL^T 应检测到 D[1] <= 0。
"""

import numpy as np
from equilibrium_solver import ldlt_factor

def test_nonpositive():
    K = np.array([[1.0, 2.0],
                  [2.0, 1.0]])
    R = np.array([1.0, 1.0])

    print("=" * 60)
    print("算例2：非正定矩阵检测")
    print(f"矩阵 K:\n{K}")
    print(f"右端项 R: {R}")
    print("尝试 LDL^T 分解...")

    try:
        L, D = ldlt_factor(K)
        print("分解成功（不应发生）")
        print(f"D = {D}")
    except ValueError as e:
        print(f"正确捕获异常: {e}")
        print("程序已停止 LDL^T 求解，符合预期。")

    print("\n注：该矩阵的特征值分别为 3 和 -1，因此不是正定矩阵。")
    print("有限元刚度矩阵在缺少足够约束时会出现零主元（刚体位移），")
    print("导致 D[j] = 0，应停止求解并提示用户添加边界条件。")

if __name__ == "__main__":
    test_nonpositive()