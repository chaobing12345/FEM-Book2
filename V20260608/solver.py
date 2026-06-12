
# solver.py
# 功能：缩减法求解位移与约束反力
# 修改：使用 equilibrium_solver 中的统一求解接口，支持 LDL^T 和 MKL PARDISO

import numpy as np
from equilibrium_solver import solve_equilibrium   # 导入新增模块

def solve_reduced_system(K, model, method="ldlt", use_sparse=False):
    """
    缩减法求解位移与反力
    参数：
        K         : 施加边界条件前的原始总刚（稠密矩阵）
        model     : 结构模型字典（包含约束、荷载等）
        method    : 求解方法，可选 "ldlt" 或 "pardiso"
        use_sparse: 当 method="pardiso" 时是否将 K_FF 转为稀疏矩阵（推荐 True）
    返回：
        d         : 全场位移向量
        R         : 约束反力向量
    """
    # 读取模型数据
    fixed_dof = model["fixed_dof"]      # 已知位移自由度编号（0-based）
    force_dof = model["force_dof"]      # 荷载作用自由度编号
    fixed_val = model["fixed_val"]      # 已知位移值
    neq = model["neq"]                  # 总自由度数

    # 所有自由度编号
    all_dof = np.arange(neq)
    # 未知自由度 = 全部自由度 - 已知自由度
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    # 提取分块矩阵
    K_FF = K[np.ix_(free_dof, free_dof)]          # 自由-自由子矩阵
    K_EF = K[np.ix_(fixed_dof, free_dof)]         # 约束-自由子矩阵

    # 输出缩减矩阵（与原来一致）
    print("\n 施加边界条件 后 的缩减刚度矩阵 K_FF ")
    print(np.round(K_FF, 4))
    det_KFF = np.linalg.det(K_FF)
    print(f"缩减矩阵行列式: {det_KFF:.4e}")
    print(f"缩减矩阵 非奇异 (可求解): {abs(det_KFF) > 1e-6}")
    print("=" * 60)

    # 构建自由节点上的荷载向量 f_F
    f_F = np.zeros(len(free_dof))
    for i, d in enumerate(force_dof):
        if d in free_dof:
            pos = np.where(free_dof == d)[0][0]
            f_F[pos] = model["force_val"][i]

    # 已知位移 d_E
    d_E = fixed_val

    # 缩减方程右端项：RHS = f_F - K_EF^T * d_E
    rhs = f_F - K_EF.T @ d_E

    # 调用统一求解接口
    if use_sparse and method.lower() == "pardiso":
        # 将 K_FF 转换为 scipy.sparse.csr_matrix 格式以调用稀疏求解器
        from scipy.sparse import csr_matrix
        K_FF_sp = csr_matrix(K_FF)
        d_F, info = solve_equilibrium(K_FF_sp, rhs, method=method, sparse=True)
    else:
        # 使用稠密矩阵 LDL^T 求解（或 PARDISO 处理稠密矩阵，但一般用于稀疏）
        d_F, info = solve_equilibrium(K_FF, rhs, method=method, sparse=False)

    # 打印求解器信息
    print(f"求解器: {info['method']}, 求解时间: {info['time']:.6f}s")

    # 组装全场位移 d
    d = np.zeros(neq)
    d[fixed_dof] = d_E
    d[free_dof] = d_F

    # 计算约束反力 R = K[约束自由度, :] @ d
    R = K[np.ix_(fixed_dof, all_dof)] @ d

    return d, R