"""
tridiagonal_test.py
算例1：三对角对称正定矩阵测试
构造 n×n 矩阵：主对角元=2，次对角元=-1
精确解 a_exact = [1,1,...,1]^T，右端项 R = K * a_exact
记录不同 n 下的求解时间，分析计算时间增长趋势。
"""

import numpy as np
import time
from equilibrium_solver import solve_equilibrium

def create_tridiagonal(n):
    """生成 n×n 三对角矩阵，主对角=2，次对角=-1"""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
        if i < n-1:
            K[i, i+1] = -1.0
    return K

def run_tridiagonal_test(n, method="ldlt"):
    """测试给定规模 n，使用 method 求解"""
    K = create_tridiagonal(n)
    a_exact = np.ones(n)
    R = K @ a_exact

    t0 = time.perf_counter()
    a, info = solve_equilibrium(K, R, method=method, sparse=False)
    t_elapsed = time.perf_counter() - t0

    # 计算相对误差
    rel_err = np.linalg.norm(a - a_exact) / np.linalg.norm(a_exact)

    print(f"n={n:4d} | 时间: {t_elapsed:.4f}s | 相对误差: {rel_err:.2e} | {info['method']}")
    return t_elapsed, rel_err

if __name__ == "__main__":
    print("="*60)
    print("算例1：三对角对称正定矩阵 LDL^T 求解")
    print("="*60)
    for n in [10, 100, 500, 1000]:
        run_tridiagonal_test(n, method="ldlt")
