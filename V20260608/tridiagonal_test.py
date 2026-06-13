# tridiagonal_test.py
# 算例1：三对角对称正定矩阵测试
# 构造 n×n 矩阵：主对角元=2，次对角元=-1
# 精确解 a_exact = [1,1,...,1]^T，右端项 R = K * a_exact
# 记录不同 n 下的求解时间，分析计算时间增长趋势。

import numpy as np                     # 导入NumPy库，用于数值计算（数组、线性代数等）
import time                            # 导入time模块，用于计时
from equilibrium_solver import solve_equilibrium   # 从自定义模块导入统一求解接口

def create_tridiagonal(n):
    """生成 n×n 三对角矩阵，主对角=2，次对角=-1"""
    K = np.zeros((n, n))               # 创建一个 n×n 的全零矩阵
    for i in range(n):                 # 遍历每一行
        K[i, i] = 2.0                  # 设置主对角线元素为 2.0
        if i > 0:                      # 如果行索引大于0（有左侧元素）
            K[i, i-1] = -1.0           # 设置次对角线（左下方）为 -1.0
        if i < n-1:                    # 如果行索引小于 n-1（有右侧元素）
            K[i, i+1] = -1.0           # 设置次对角线（右上方）为 -1.0
    return K                           # 返回生成的三对角矩阵

def run_tridiagonal_test(n, method="ldlt"):
    """测试给定规模 n，使用 method 求解"""
    K = create_tridiagonal(n)          # 生成三对角矩阵 K
    a_exact = np.ones(n)               # 精确解向量，所有分量均为 1
    R = K @ a_exact                    # 右端项 = K * 精确解

    t0 = time.perf_counter()           # 记录开始时刻（高精度计时）
    a, info = solve_equilibrium(K, R, method=method, sparse=False)   # 调用统一求解接口（稠密模式）
    t_elapsed = time.perf_counter() - t0   # 计算求解耗时

    # 计算相对误差：||a - a_exact|| / ||a_exact||
    rel_err = np.linalg.norm(a - a_exact) / np.linalg.norm(a_exact)

    # 打印结果：矩阵规模、耗时、相对误差、使用的求解器名称
    print(f"n={n:4d} | 时间: {t_elapsed:.4f}s | 相对误差: {rel_err:.2e} | {info['method']}")
    return t_elapsed, rel_err          # 返回耗时和相对误差

if __name__ == "__main__":             # 如果该脚本作为主程序运行（而非作为模块被导入）
    print("="*60)                      # 打印分隔线
    print("算例1：三对角对称正定矩阵 LDL^T 求解")
    print("="*60)
    # 依次测试 n = 10, 100, 500, 1000 四种规模
    for n in [10, 100, 500, 1000]:
        run_tridiagonal_test(n, method="ldlt")   # 调用测试函数，使用 LDL^T 求解方法