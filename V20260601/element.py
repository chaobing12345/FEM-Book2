
# element.py
# 功能：自动生成LM矩阵 + 计算单元刚度矩阵


# 导入numpy用于矩阵运算
import numpy as np

# 函数：自动生成对号矩阵 LM
# 功能：局部自由度 → 全局自由度映射
def build_LM(model):
    # 每个节点自由度
    ndof = model["ndof"]

    # 单元总数
    nel = model["nel"]

    # 单元-节点连接表
    IEN = model["IEN"]

    # 每个单元的局部自由度
    n_local_dof = ndof * 2

    # 初始化LM矩阵
    LM = np.zeros((n_local_dof, nel), dtype=int)

    # 遍历所有单元
    for e in range(nel):
        n1 = IEN[e, 0] - 1  # 节点1（转为0开始）
        n2 = IEN[e, 1] - 1  # 节点2（转为0开始）

        # 一维单元
        if ndof == 1:
            LM[0, e] = n1
            LM[1, e] = n2

        # 二维单元
        else:
            LM[0, e] = 2 * n1 + 0
            LM[1, e] = 2 * n1 + 1
            LM[2, e] = 2 * n2 + 0
            LM[3, e] = 2 * n2 + 1

    # 输出自动生成的LM矩阵（新增要求）
    print("\n【自动生成 LM 矩阵】(行=局部自由度，列=单元)：")
    print(LM)

    return LM

# 函数：计算单元刚度矩阵 ke
def element_stiffness(e, model):
    # 获取自由度类型（一维/二维）
    ndof = model["ndof"]

    # 单元弹性模量
    E = model["E"][e]

    # 单元截面面积
    A = model["A"][e]

    # 获取单元的两个节点
    IENe = model["IEN"][e] - 1
    n1 = IENe[0]
    n2 = IENe[1]

    # 一维杆单元 
    if ndof == 1:
        L = model["L"][e]               # 读取已计算的单元长度
        k = E * A / L                   # 刚度系数
        ke = np.array([[k, -k],         # 一维单元刚度矩阵
                       [-k, k]])
        return ke, L, 1.0, 0.0

    # 二维桁架单元 
    else:
        x1 = model["x"][n1]
        y1 = model["y"][n1]
        x2 = model["x"][n2]
        y2 = model["y"][n2]

        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)      # 单元长度
        cx = dx / L                     # 方向余弦
        cy = dy / L
        k = E * A / L                   # 轴向刚度

        # 二维单元刚度矩阵
        ke = k * np.array([
            [cx*cx, cx*cy, -cx*cx, -cx*cy],
            [cx*cy, cy*cy, -cx*cy, -cy*cy],
            [-cx*cx, -cx*cy, cx*cx, cx*cy],
            [-cx*cy, -cy*cy, cx*cy, cy*cy]
        ])

        return ke, L, cx, cy