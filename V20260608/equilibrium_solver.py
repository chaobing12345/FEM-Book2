"""
equilibrium_solver.py
任务1 + 任务3 的统一求解接口：
- 稠密对称正定矩阵的 LDL^T 分解与求解
- 稀疏矩阵 MKL PARDISO 求解器调用
- 残差计算函数
"""

import numpy as np      # 导入NumPy库，用于数值计算（数组、线性代数等）
import time              # 导入time模块，用于测量求解耗时

# ----------------------------------------------------------------------
# 1. LDL^T 分解与求解（稠密矩阵）
# ----------------------------------------------------------------------

def ldlt_factor(K):
    """
    对对称正定矩阵 K 进行 LDL^T 分解，返回 L, D。
    如果遇到非正主元（D[j] <= 0），抛出异常。
    """
    # 将输入矩阵转换为双精度浮点数，避免单精度误差导致的不稳定
    K = np.array(K, dtype=np.float64)
    n = K.shape[0]                     # 获取矩阵的阶数（行数，也是列数）

    L = np.eye(n)                      # 初始化 L 为单位下三角矩阵（对角线为1）
    D = np.zeros(n)                    # D 为一维数组，存储对角元

    # 按列进行 LDL^T 分解，j 为当前列索引（0‑based）
    for j in range(n):
        # 计算 D[j] = K[j][j] - sum_{k=0}^{j-1} L[j][k]^2 * D[k]
        sum_d = 0.0                    # 累加器清零
        for k in range(j):             # 遍历 j 列左侧的列
            sum_d += L[j, k] ** 2 * D[k]   # 累加 L_jk^2 * D_k
        D[j] = K[j, j] - sum_d         # 更新 D 的第 j 个对角元

        # 检查主元：若 D[j] <= 0 则矩阵非正定或奇异（符合作业要求）
        if D[j] <= 1e-12:              # 允许微小负值，但小于等于零时认为奇异
            raise ValueError(f"矩阵非正定或存在零主元：D[{j}] = {D[j]:.4e}")

        # 计算第 j 列下三角部分 L[i][j] (i > j)
        for i in range(j + 1, n):      # 遍历行 i，从 j+1 到 n-1
            sum_l = 0.0                # 累加器清零
            for k in range(j):         # 遍历 k = 0 … j-1
                # 累加 L_ik * L_jk * D_k
                sum_l += L[i, k] * L[j, k] * D[k]
            # 计算 L_ij = (K_ij - sum_l) / D_j
            L[i, j] = (K[i, j] - sum_l) / D[j]

    return L, D                        # 返回下三角矩阵 L 和向量 D

def ldlt_solve(L, D, RHS):
    """
    求解 LDL^T a = RHS
    步骤：前代 L y = RHS -> 对角 D z = y -> 回代 L^T a = z
    """
    n = len(RHS)                       # 方程组阶数
    y = np.zeros(n)                    # 存储中间变量 y

    # 前代：解 L y = RHS (L 是单位下三角，对角线为1)
    for i in range(n):                 # 按行正向推进
        s = 0.0
        for j in range(i):             # j 从 0 到 i-1
            s += L[i, j] * y[j]        # 累加 L_ij * y_j
        y[i] = RHS[i] - s              # y_i = RHS_i - Σ L_ij y_j

    # 对角求解：z = y ./ D
    z = y / D                          # 逐元素除法（广播）

    # 回代：解 L^T a = z (L^T 是上三角，对角线为1)
    a = np.zeros(n)                    # 存储最终解
    for i in range(n - 1, -1, -1):     # 逆序遍历 i = n-1 … 0
        s = 0.0
        for j in range(i + 1, n):      # j 从 i+1 到 n-1
            s += L[j, i] * a[j]        # 累加 L_ji * a_j（注意 L 的转置）
        a[i] = z[i] - s                # a_i = z_i - Σ L_ji a_j
    return a

# ----------------------------------------------------------------------
# 2. 稀疏求解器接口（MKL PARDISO）
# ----------------------------------------------------------------------

def solve_mkl_pardiso(K_sparse, rhs):
    """
    调用基于 Intel MKL 的 PARDISO 求解器。
    优先使用 pypardiso，其次 pydiso，最后回退到 scipy（Anaconda 下通常也链接 MKL）。
    返回 (解向量, 求解器名称, 求解耗时)
    """
    # 尝试导入 pypardiso（最推荐的专用接口，直接绑定 MKL PARDISO）
    try:
        from pypardiso import spsolve
        solver_name = "PyPardiso (Intel MKL PARDISO)"
        t0 = time.perf_counter()          # 开始计时（高精度）
        x = spsolve(K_sparse, rhs)        # 调用稀疏求解器
        solve_time = time.perf_counter() - t0  # 计算耗时
        return x, solver_name, solve_time
    except ImportError:
        pass                              # 未安装 pypardiso，继续尝试下一个

    # 尝试导入 pydiso（另一个 PARDISO 绑定）
    try:
        from pydiso import spsolve
        solver_name = "PyDISO (Intel MKL PARDISO)"
        t0 = time.perf_counter()
        x = spsolve(K_sparse, rhs)
        solve_time = time.perf_counter() - t0
        return x, solver_name, solve_time
    except ImportError:
        pass

    # 最终回退到 scipy（Anaconda 环境一般也使用 MKL 作为后端）
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
        t0 = time.perf_counter()          # 开始计时
        L, D = ldlt_factor(K)             # 分解
        a = ldlt_solve(L, D, rhs)         # 求解
        solve_time = time.perf_counter() - t0
        info = {"method": "LDL^T (dense)", "time": solve_time}
        return a, info

    elif method.lower() == "pardiso":
        # 稀疏 PARDISO 求解
        if not sparse:
            # 若用户误传稠密矩阵，转换为 CSR 格式（稀疏列压缩格式）
            from scipy.sparse import csr_matrix
            K_sp = csr_matrix(K)          # 转换为稀疏矩阵
        else:
            K_sp = K                      # 已经为稀疏格式，直接使用
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
    r = R - K @ a                        # 矩阵‑向量乘法，得到残差向量
    norm_r = np.linalg.norm(r)           # 计算 L2 范数（欧几里得范数）
    norm_R = np.linalg.norm(R)           # 右端项的范数
    # 相对残差 = ||r|| / ||R||，若 R 为零向量则避免除零
    rel_residual = norm_r / norm_R if norm_R > 0 else 0.0
    return r, norm_r, rel_residual