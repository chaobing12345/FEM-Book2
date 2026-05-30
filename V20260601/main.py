# 导入数值计算库numpy
import numpy as np

# 从model模块导入加载一维、二维模型的函数
from model import create_1d_truss, create_2d_truss

# 从element模块导入生成LM矩阵和单元刚度的函数
from element import build_LM, element_stiffness

# 从assembly模块导入组装总刚的函数
from assembly import assemble_global_K

# 从solver模块导入求解函数
from solver import solve_reduced_system

# 从postprocess模块导入后处理函数
from postprocess import compute_stress_force, print_results

# 定义运行算例的函数
def run_example(example=1):
    # 打印分隔线
    print("=" * 70)

    # 判断是一维还是二维算例
    if example == 1:
        # 打印算例名称
        print(" 算例 1：一维两单元杆")
        # 加载一维模型
        model = create_1d_truss()
    else:
        # 打印算例名称
        print(" 算例 2：二维两杆桁架")
        # 加载二维模型
        model = create_2d_truss()

    # 打印分隔线
    print("=" * 70)

    # 生成对号矩阵LM
    LM = build_LM(model)

    # 组装整体刚度矩阵K
    K = assemble_global_K(model, LM)

    # 求解位移d和支座反力R
    d, R = solve_reduced_system(K, model)

    # 计算单元应力、轴力、长度、方向余弦
    stress, force, lengths, cx, cy = compute_stress_force(model, LM, d)

    # 输出所有结果
    print_results(model, K, d, R, stress, force, lengths, cx, cy)

# 主程序入口
if __name__ == "__main__":
    # 运行一维算例
    run_example(1)
    # 空两行
    print("\n" * 2)
    # 运行二维算例
    run_example(2)