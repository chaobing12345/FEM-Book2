
# model.py
# 功能：从JSON文件读取结构参数，自动计算一维单元长度


# 导入json库，用于读取JSON格式的输入文件
import json

# 导入numpy库，用于数值计算和数组处理
import numpy as np

# 函数：加载结构模型（从JSON文件读取所有参数）
def load_model(json_file):
    # 打开并读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建空字典，用于存储所有结构信息
    model = {}

    # 每个节点的自由度数目（一维=1，二维=2）
    model['ndof'] = data['ndof']

    # 节点总数
    model['nnp'] = data['nnp']

    # 单元总数
    model['nel'] = data['nel']

    # 每个单元的节点数（杆单元=2）
    model['nen'] = data['nen']

    # 总自由度数 = 节点数 × 每个节点自由度
    model['neq'] = model['ndof'] * model['nnp']

    # 弹性模量
    model['E'] = np.array(data['E'])

    # 截面面积
    model['A'] = np.array(data['CArea'])

    # 节点x坐标
    model['x'] = np.array(data['x'])

    # 如果是二维模型，读取y坐标
    if 'y' in data:
        model['y'] = np.array(data['y'])

    # 单元-节点连接矩阵
    model['IEN'] = np.array(data['IEN'])

    # 一维模型自动计算单元长度 L 
    # 判断是否为一维结构
    if model['ndof'] == 1:
        L = []                     # 新建空列表存储长度
        for e in range(model['nel']):          # 遍历所有单元
            n1 = model['IEN'][e, 0] - 1         # 单元第一个节点（转0编号）
            n2 = model['IEN'][e, 1] - 1         # 单元第二个节点（转0编号）
            length = model['x'][n2] - model['x'][n1]  # 计算长度
            L.append(length)                    # 存入列表
        model['L'] = np.array(L)                # 存入model字典
    

    # 约束自由度（从1号转0号）
    model['fixed_dof'] = np.array(data['fixed_dof']) - 1

    # 约束位移值
    model['fixed_val'] = np.array(data['fixed_value'])

    # 荷载自由度（从1号转0号）
    model['force_dof'] = np.array(data['force_dof']) - 1

    # 荷载大小
    model['force_val'] = np.array(data['force_value'])

    # 返回完整模型
    return model

# 函数：加载一维杆模型
def create_1d_truss():
    return load_model("truss_1d_input.json")

# 函数：加载二维桁架模型
def create_2d_truss():
    return load_model("truss_input.json")