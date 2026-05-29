
# 组装文件：直接刚度法组装总体刚度矩阵K


import numpy as np
# 从element.py导入单元刚度计算函数
from element import element_stiffness

# 函数：组装总体刚度矩阵K
# 输入：模型model、对号矩阵LM
# 输出：总体刚度矩阵K
def assemble_global_K(model, LM):
    # 获取结构总自由度数
    neq = model["neq"]
    # 获取总单元数
    nel = model["nel"]

    # 初始化总体刚度矩阵K为全零矩阵（neq×neq）
    K = np.zeros((neq, neq))

    # 遍历所有单元，逐个将单元刚度累加到总刚
    for e in range(nel):
        # 调用element_stiffness，获取第e个单元的刚度矩阵ke
        ke, _, _, _ = element_stiffness(e, model)
        # 从LM中获取该单元对应的全局自由度编号
        dofs = LM[:, e]
        # 核心语句：按自由度对应关系，将单元刚度累加到总刚
        K[np.ix_(dofs, dofs)] += ke

    # 明确输出 施加边界条件前 的K
    print("\n 施加边界条件 前 的总体刚度矩阵 K ")
    
    # 输出总体刚度矩阵（保留4位小数）
    print("整体刚度矩阵 K：")
    print(np.round(K, 4))
    # 检查总刚的对称性与奇异性
    check_K_properties(K)

    # 返回组装完成的总体刚度矩阵
    return K

# 函数：检查刚度矩阵的数学性质
def check_K_properties(K):
    # 判断是否对称：K与K的转置是否相等
    is_symmetric = np.allclose(K, K.T)
    # 计算矩阵行列式
    det = np.linalg.det(K)
    # 判断是否奇异：行列式是否接近0
    is_singular = abs(det) < 1e-6
    # 输出判断结果
    print(f"对称: {is_symmetric} | 奇异: {is_singular}\n")