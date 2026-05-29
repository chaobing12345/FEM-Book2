
# solver.py
# 功能：缩减法求解位移、约束反力
# 新增：输出缩减矩阵 K_FF + 判断是否非奇异

import numpy as np

def solve_reduced_system(K, model):
    """
    缩减法求解位移与反力
    输入：K - 施加边界条件前的原始总刚
          model - 结构模型
    输出：d - 全场位移
          R - 约束反力
    """
    # 读取约束自由度（已知位移）
    fixed_dof = model["fixed_dof"]
    
    # 读取荷载自由度
    force_dof = model["force_dof"]
    
    # 读取约束位移值
    fixed_val = model["fixed_val"]
    
    # 总自由度数
    neq = model["neq"]

    # 1. 划分自由度
    # 生成所有自由度编号 0 ~ neq-1
    all_dof = np.arange(neq)
    
    # 自由自由度 = 全部自由度 - 约束自由度（未知位移）
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    #  2. 提取分块矩阵 
    # 缩减刚度矩阵：自由自由度之间的刚度（核心求解矩阵）
    K_FF = K[np.ix_(free_dof, free_dof)]
    
    # 约束自由度与自由自由度之间的刚度
    K_EF = K[np.ix_(fixed_dof, free_dof)]

    #输出缩减矩阵 K_FF 
    print("\n 施加边界条件 后 的缩减刚度矩阵 K_FF ")
    print(np.round(K_FF, 4))

    # 判断缩减矩阵是否非奇异 
    # 计算行列式
    det_KFF = np.linalg.det(K_FF)
    # 非奇异判定：行列式绝对值 > 1e-6
    is_non_singular = abs(det_KFF) > 1e-6

    print(f"缩减矩阵行列式: {det_KFF:.4e}")
    print(f"缩减矩阵 非奇异 (可求解): {is_non_singular}")
    print("="*60)

    # 3. 构建荷载向量 
    # 自由节点的荷载向量
    f_F = np.zeros(len(free_dof))
    
    # 遍历所有荷载，分配到对应自由自由度
    for i, d in enumerate(force_dof):
        if d in free_dof:
            pos = np.where(free_dof == d)[0][0]
            f_F[pos] = model["force_val"][i]

    #  4. 求解未知位移 
    # 已知位移
    d_E = fixed_val
    
    # 缩减方程求解
    d_F = np.linalg.solve(K_FF, f_F - K_EF.T @ d_E)

    #  5. 组装全场位移 
    d = np.zeros(neq)
    d[fixed_dof] = d_E   # 填入已知位移
    d[free_dof] = d_F    # 填入求解位移

    # 6. 计算约束反力 
    R = K[np.ix_(fixed_dof, all_dof)] @ d

    # 返回位移与反力
    return d, R