"""
equilibrium_solver.py
任务1 + 任务3 的统一求解接口：
- 稠密对称正定矩阵的 LDL^T 分解与求解
- 稀疏矩阵 MKL PARDISO 求解器调用
- 残差计算函数
"""

import numpy as np      # 数值计算库
import time              # 计时模块

# ----------------------------------------------------------------------
# 1. LDL^T 分解与求解（稠密矩阵）
# ----------------------------------------------------------------------

def ldlt_factor(K):
    """
    对对称正定矩阵 K 进行 LDL^T 分解，返回 L, D。
    如果遇到非正主元（D[j] <= 0），抛出异常。
    """
    # 将输入矩阵转为双精度浮点数，避免单精度误差
    K = np.array(K, dtype=np.float64)
    n = K.shape[0]                     # 矩阵阶数

    L = np.eye(n)                      # 初始化 L 为单位下三角矩阵
    D = np.zeros(n)                    # D 为一维数组存储对角元

    # 按列进行分解，j 为当前列索引（0-based）
    for j in range(n):
        # 计算 D[j] = K[j][j] - sum_{k=0}^{j-1} L[j][k]^2 * D[k]
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - sum_d

        # 检查主元：若 D[j] <= 0 则矩阵非正定或奇异（符合作业要求）
        if D[j] <= 1e-12:
            raise ValueError(f"矩阵非正定或存在零主元：D[{j}] = {D[j]:.4e}")

        # 计算第 j 列下三角部分 L[i][j] (i > j)
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_l) / D[j]

    return L, D

def ldlt_solve(L, D, RHS):
    """
    求解 LDL^T a = RHS
    步骤：前代 L y = RHS -> 对角 D z = y -> 回代 L^T a = z
    """
    n = len(RHS)
    y = np.zeros(n)

    # 前代：解 L y = RHS (L 是单位下三角)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = RHS[i] - s

    # 对角求解：z = y ./ D
    z = y / D

    # 回代：解 L^T a = z (L^T 是上三角)
    a = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += L[j, i] * a[j]
        a[i] = z[i] - s

    return a

# ----------------------------------------------------------------------
# 2. 稀疏求解器接口（MKL PARDISO）
# ----------------------------------------------------------------------

def solve_mkl_pardiso(K_sparse, rhs):
    """
    调用基于 MKL 的 PARDISO 求解器。
    优先使用 pypardiso，其次 pydiso，最后回退到 scipy（Anaconda 下通常也链接 MKL）。
    返回 (解向量, 求解器名称, 求解耗时)
    """
    # 尝试 pypardiso（最推荐的专用接口）
    try:
        from pypardiso import spsolve
        solver_name = "PyPardiso (Intel MKL PARDISO)"
        t0 = time.perf_counter()
        x = spsolve(K_sparse, rhs)
        solve_time = time.perf_counter() - t0
        return x, solver_name, solve_time
    except ImportError:
        pass

    # 尝试 pydiso（另一绑定）
    try:
        from pydiso import spsolve
        solver_name = "PyDISO (Intel MKL PARDISO)"
        t0 = time.perf_counter()
        x = spsolve(K_sparse, rhs)
        solve_time = time.perf_counter() - t0
        return x, solver_name, solve_time
    except ImportError:
        pass

    # 最终回退到 scipy（Anaconda 环境一般也使用 MKL）
    try:
        from scipy.sparse.linalg import spsolve
        solver_name = "scipy.sparse.linalg.spsolve (likely MKL via Anaconda)"
        t0 = time.perf_counter()
        x = spsolve(K_sparse, rhs)
        solve_time = time.perf_counter() - t0
        return x, solver_name, solve_time
    except ImportError:
        raise ImportError("未找到任何稀疏求解器，请安装 scipy, pypardiso 或 pydiso")

# ----------------------------------------------------------------------
# 3. 统一求解接口
# ----------------------------------------------------------------------

def solve_equilibrium(K, rhs, method="ldlt", sparse=False):
    """
    统一接口：求解 K * a = rhs
    参数：
        K      : 系数矩阵（稠密 numpy.ndarray 或 scipy.sparse 矩阵）
        rhs    : 右端项 (numpy.ndarray)
        method : "ldlt" 或 "pardiso"
        sparse : 当 method="pardiso" 且 K 为稀疏矩阵时设为 True
    返回：
        a      : 解向量
        info   : 字典，包含求解器名称和求解时间
    """
    if method.lower() == "ldlt":
        # 稠密 LDL^T 求解
        t0 = time.perf_counter()
        L, D = ldlt_factor(K)        # 分解
        a = ldlt_solve(L, D, rhs)    # 求解
        solve_time = time.perf_counter() - t0
        info = {"method": "LDL^T (dense)", "time": solve_time}
        return a, info

    elif method.lower() == "pardiso":
        # 稀疏 PARDISO 求解
        if not sparse:
            # 若用户误传稠密矩阵，转换为 CSR 格式
            from scipy.sparse import csr_matrix
            K_sp = csr_matrix(K)
        else:
            K_sp = K
        a, solver_name, solve_time = solve_mkl_pardiso(K_sp, rhs)
        info = {"method": solver_name, "time": solve_time}
        return a, info

    else:
        raise ValueError(f"未知求解方法: {method}，可选 'ldlt' 或 'pardiso'")

# ----------------------------------------------------------------------
# 4. 残差计算（符合作业任务1要求）
# ----------------------------------------------------------------------

def residual_norm(K, a, R):
    """
    计算残差向量 r = R - K a 及其范数。
    返回 (残差向量, 残差范数, 相对残差范数)
    """
    r = R - K @ a                    # 残差向量
    norm_r = np.linalg.norm(r)       # L2 范数
    norm_R = np.linalg.norm(R)       # 右端项范数
    rel_residual = norm_r / norm_R if norm_R > 0 else 0.0
    return r, norm_r, rel_residual