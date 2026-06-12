# 后处理文件：计算应力、轴力、输出结果
import numpy as np
from element import element_stiffness

# 函数：计算单元应力和轴力
def compute_stress_force(model, LM, d):                
    ndof = model["ndof"]
    nel = model["nel"]
    stress = []
    force = []
    lengths = []
    cx_list = []
    cy_list = []

    # 遍历所有单元
    for e in range(nel):
        # 获取单元刚度、长度、方向余弦
        ke, L, cx, cy = element_stiffness(e, model)
        # 提取单元节点位移
        de = d[LM[:, e]]
        # 获取材料参数
        E = model["E"][e]
        A = model["A"][e]

        # 一维单元应变计算
        if ndof == 1:
            eps = (de[1] - de[0]) / L
        # 二维单元应变计算
        else:
            eps = (cx*(de[2]-de[0]) + cy*(de[3]-de[1])) / L

        # 应力计算：σ = E·ε
        sig = E * eps
        # 轴力计算：N = σ·A
        N = sig * A

        # 保存结果
        stress.append(sig)
        force.append(N)
        lengths.append(L)
        cx_list.append(cx)
        cy_list.append(cy)

    return stress, force, lengths, cx_list, cy_list

# 函数：输出所有计算结果
def print_results(model, K, d, R, stress, force, lengths, cx, cy):
    print("节点位移：", np.round(d, 6))
    print("约束反力：", np.round(R, 4))
    print("\n单元结果：")
    for i in range(model["nel"]):
        print(f"单元{i+1}: L={lengths[i]:.3f}, cx={cx[i]:.3f}, σ={stress[i]:.2f}, N={force[i]:.2f}")

    # 刚体位移验证
    print("\n刚体位移检查（整体平移 → 内力≈0）")
    u_rigid = np.ones(model["neq"]) * 0.001
    F_rigid = K @ u_rigid
    ok = np.allclose(F_rigid, 0, atol=1e-6)
    print(f"刚体平移无内力: {ok}\n")