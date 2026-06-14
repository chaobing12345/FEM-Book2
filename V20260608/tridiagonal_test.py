import numpy as np
import time
from equilibrium_solver import solve_equilibrium, residual_norm
from scipy.sparse import csr_matrix

def create_tridiagonal(n):
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
        if i < n-1:
            K[i, i+1] = -1.0
    return K

def run_tridiagonal_test(n, method="ldlt"):
    K = create_tridiagonal(n)
    a_exact = np.ones(n)
    R = K @ a_exact

    # 求解
    t0 = time.perf_counter()
    a, info = solve_equilibrium(K, R, method=method, sparse=False)
    t_elapsed = time.perf_counter() - t0

    # 计算相对残差（使用提供的 residual_norm 函数）
    r, norm_r, rel_res = residual_norm(K, a, R)

    # 相对误差
    rel_err = np.linalg.norm(a - a_exact) / np.linalg.norm(a_exact)

    # 统计非零元个数（稠密矩阵中非零元素）
    nnz = np.count_nonzero(K)
    # 内存估计
    dense_mb = K.nbytes / 1024**2
    # 模拟稀疏存储内存（假设使用 CSR，每个非零元存值和列索引，每行一个行偏移）
    # 简单估算：CSR 需要 nnz 个 float64 + nnz 个 int32 + (n+1) 个 int32
    sparse_bytes = nnz * 8 + nnz * 4 + (n+1) * 4
    sparse_mb = sparse_bytes / 1024**2
    ratio = sparse_mb / dense_mb if dense_mb > 0 else 0

    print(f"阶数 n = {n:4d} | 耗时 = {t_elapsed:.6f}s | 相对残差 = {rel_res:.2e} | 相对误差 = {rel_err:.2e}")
    print(f"非零元个数: {nnz:5d} | 稠密内存: {dense_mb:.2f} MB | 稀疏内存: {sparse_mb:.2f} MB | 稀疏/稠密: {ratio:.2f}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("【算例1】三对角对称正定矩阵")
    print("="*60)
    for n in [10, 100, 500, 1000]:
        run_tridiagonal_test(n, method="ldlt")