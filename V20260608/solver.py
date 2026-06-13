
# solver.py
# 功能：缩减法求解位移与约束反力
# 修改：使用 equilibrium_solver 中的统一求解接口，支持 LDL^T 和 MKL PARDISO

import numpy as np                              # 导入 NumPy 库，用于数值计算（数组、线性代数等）
from equilibrium_solver import solve_equilibrium   # 从自定义模块导入统一求解接口（支持 LDL^T 和 PARDISO）

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
    fixed_dof = model["fixed_dof"]      # 已知位移自由度编号（0-based，列表或数组）
    force_dof = model["force_dof"]      # 荷载作用自由度编号（列表或数组）
    fixed_val = model["fixed_val"]      # 已知位移值（列表或数组，与 fixed_dof 一一对应）
    neq = model["neq"]                  # 总自由度数（标量整数）

    # 所有自由度编号（从 0 到 neq-1）
    all_dof = np.arange(neq)
    # 未知自由度 = 全部自由度 - 已知自由度（使用集合差集）
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    # 提取分块矩阵：K_FF（自由-自由），K_EF（约束-自由）
    # np.ix_ 用于从多维数组中提取子矩阵，方便保持矩阵形状
    K_FF = K[np.ix_(free_dof, free_dof)]          # 自由-自由子矩阵，大小为 n_free × n_free
    K_EF = K[np.ix_(fixed_dof, free_dof)]         # 约束-自由子矩阵，大小为 n_fixed × n_free

    # 输出缩减矩阵（与原来一致）
    print("\n 施加边界条件 后 的缩减刚度矩阵 K_FF ")
    print(np.round(K_FF, 4))                      # 打印 K_FF 矩阵，保留4位小数
    det_KFF = np.linalg.det(K_FF)                 # 计算缩减矩阵的行列式
    print(f"缩减矩阵行列式: {det_KFF:.4e}")        # 以科学计数法输出行列式
    print(f"缩减矩阵 非奇异 (可求解): {abs(det_KFF) > 1e-6}")   # 判断是否非奇异（行列式绝对值大于1e-6）
    print("=" * 60)                               # 打印分隔线

    # 构建自由节点上的荷载向量 f_F
    f_F = np.zeros(len(free_dof))                 # 初始化为零向量，长度等于自由度数
    for i, d in enumerate(force_dof):             # 遍历每个荷载作用自由度
        if d in free_dof:                         # 仅当该自由度是自由自由度时，才计入
            pos = np.where(free_dof == d)[0][0]   # 找到该自由度在 free_dof 中的索引位置
            f_F[pos] = model["force_val"][i]      # 将荷载值赋给对应位置

    # 已知位移 d_E（边界位移值）
    d_E = fixed_val                               # 直接使用模型中的固定位移值

    # 缩减方程右端项：RHS = f_F - K_EF^T * d_E
    # 因为原方程 K_FF * d_F = f_F - K_EF^T * d_E
    rhs = f_F - K_EF.T @ d_E                      # @ 表示矩阵乘法（numpy 3.5+ 语法）

    # 调用统一求解接口
    if use_sparse and method.lower() == "pardiso":
        # 当用户要求使用 PARDISO 且需要稀疏格式时，将 K_FF 转换为 scipy.sparse.csr_matrix 格式
        from scipy.sparse import csr_matrix       # 导入稀疏矩阵的 CSR 格式
        K_FF_sp = csr_matrix(K_FF)                # 将稠密矩阵 K_FF 转换为 CSR 稀疏矩阵
        d_F, info = solve_equilibrium(K_FF_sp, rhs, method=method, sparse=True)
    else:
        # 否则使用稠密矩阵求解（LDL^T 或稠密版本的 PARDISO）
        d_F, info = solve_equilibrium(K_FF, rhs, method=method, sparse=False)

    # 打印求解器信息（求解器名称和求解耗时）
    print(f"求解器: {info['method']}, 求解时间: {info['time']:.6f}s")

    # 组装全场位移 d（长度为 neq）
    d = np.zeros(neq)                             # 初始化为零向量
    d[fixed_dof] = d_E                            # 将已知位移放入对应自由度
    d[free_dof] = d_F                             # 将求解出的自由位移放入对应自由度

    # 计算约束反力 R = K[约束自由度, :] @ d
    # K 是原始总刚（未施加边界条件），R 作用于约束自由度上
    R = K[np.ix_(fixed_dof, all_dof)] @ d         # 提取约束自由度所在行，与全场位移相乘，得到反力

    return d, R                                   # 返回全场位移向量和约束反力向量