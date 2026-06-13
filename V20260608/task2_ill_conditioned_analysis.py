# task2_ill_conditioned_analysis.py
# 任务2：病态线性方程组的误差、残差和条件数分析
# 构造矩阵 K = [[1,1],[1,1.0001]]，精确解 a = [1,1]^T。
# 分别用双精度、4位有效数字、半精度/单精度求解，比较误差。

import numpy as np                     # 导入NumPy库，用于数值计算（数组、线性代数、条件数等）

# ------------------------------------------------------------
# 1. 定义病态矩阵和精确解
# ------------------------------------------------------------
# 矩阵 K（对称且接近奇异，条件数很大）
K = np.array([[1.0000, 1.0000],        # 第一行
              [1.0000, 1.0001]], dtype=np.float64)   # 第二行，指定双精度浮点数类型

a_exact = np.array([1.0, 1.0])         # 精确解向量 a = [1, 1]^T
R = K @ a_exact                        # 精确右端项 R = K * a_exact（精确计算）

# 计算条件数（2-范数），用于评估矩阵的病态程度
cond_K = np.linalg.cond(K, p=2)        # p=2 表示谱条件数（最大奇异值/最小奇异值）

# 打印分析头信息
print("=" * 70)
print("任务2：病态线性方程组误差分析")
print("=" * 70)
print(f"系数矩阵 K:\n{K}")              # 显示矩阵 K
print(f"精确解 a_exact: {a_exact}")    # 显示精确解
print(f"右端项 R (精确计算): {R}")      # 显示右端项
print(f"矩阵条件数 cond(K) = {cond_K:.4f}")   # 打印条件数，保留4位小数
print("注：条件数远大于 1，为病态矩阵。")
print("=" * 70)

# ------------------------------------------------------------
# 2. 辅助函数：求解并输出各项指标
# ------------------------------------------------------------
def analyze_solution(K, R, a_exact, precision_name):
    """
    求解线性方程组 K a = R，并输出数值解、残差、相对残差、相对误差。
    参数：
        K              : 系数矩阵（目标精度的数组）
        R              : 右端项（目标精度的数组）
        a_exact        : 精确解向量
        precision_name : 字符串，描述当前精度（如"双精度"）
    """
    try:
        # 调用NumPy的线性求解器（内部使用LU分解，适用于一般矩阵）
        a = np.linalg.solve(K, R)
    except np.linalg.LinAlgError as e:   # 捕获奇异矩阵等求解错误
        print(f"{precision_name}: 求解失败 - {e}")
        return

    r = R - K @ a                        # 计算残差向量 r = R - K a
    norm_r = np.linalg.norm(r)           # 残差的L2范数
    norm_R = np.linalg.norm(R)           # 右端项的L2范数
    rel_residual = norm_r / norm_R       # 相对残差 = ||r|| / ||R||

    err_abs = a - a_exact                # 绝对误差向量
    rel_error = np.linalg.norm(err_abs) / np.linalg.norm(a_exact)   # 相对误差 = ||a - a_exact|| / ||a_exact||

    # 打印结果
    print(f"\n【{precision_name}】")
    print(f"  数值解 a = {a}")
    print(f"  残差 r = {r}")
    print(f"  残差范数 ||r|| = {norm_r:.4e}")
    print(f"  相对残差 ||r||/||R|| = {rel_residual:.4e}")
    print(f"  相对误差 ||a - a_exact||/||a_exact|| = {rel_error:.4e}")

# ------------------------------------------------------------
# 3. 双精度计算（基准）
# ------------------------------------------------------------
analyze_solution(K, R, a_exact, "双精度 (float64)")   # 使用原始双精度矩阵和右端项，作为基准

# ------------------------------------------------------------
# 4. 模拟四舍五入到 4 位有效数字的初始截断误差
# ------------------------------------------------------------
def round_to_sigfigs(x, sigfigs=4):
    """将浮点数 x 四舍五入到 sigfigs 位有效数字"""
    if x == 0:                         # 0 的特殊处理
        return 0.0
    magnitude = np.floor(np.log10(abs(x)))        # 确定数量级（例如 123 -> 2, 0.045 -> -2）
    factor = 10 ** (sigfigs - 1 - magnitude)      # 计算缩放因子，使得目标有效数字落在整数位上
    return np.round(x * factor) / factor         # 缩放 → 四舍五入 → 还原

# 对 K 和 R 中的每个元素应用四舍五入到4位有效数字
K_4sig = np.vectorize(round_to_sigfigs)(K)       # np.vectorize 将标量函数应用到数组每个元素
R_4sig = np.vectorize(round_to_sigfigs)(R)

print("\n" + "-" * 70)
print("四舍五入到 4 位有效数字后的矩阵和右端项：")
print(f"K_4sig =\n{K_4sig}")
print(f"R_4sig = {R_4sig}")

# 使用截断后的矩阵和右端项求解，并分析
analyze_solution(K_4sig, R_4sig, a_exact, "4位有效数字（模拟初始截断误差）")

# ------------------------------------------------------------
# 5. 半精度 (float16) 或单精度 (float32) 模拟
# ------------------------------------------------------------
try:
    # 尝试转换为 float16（半精度，约 3-4 位有效数字）
    K_f16 = K.astype(np.float16)          # 将矩阵 K 转为半精度
    R_f16 = R.astype(np.float16)          # 将右端项 R 转为半精度
    print("\n" + "-" * 70)
    print("转换为 float16 后的矩阵和右端项：")
    print(f"K_f16 =\n{K_f16}")
    print(f"R_f16 = {R_f16}")

    # 注意：np.linalg.solve 会将输入提升为 float64，但输入精度已经丢失
    # 所以先将半精度数据转回 float64，用于求解
    K_f16_for_solve = K_f16.astype(np.float64)
    R_f16_for_solve = R_f16.astype(np.float64)
    analyze_solution(K_f16_for_solve, R_f16_for_solve, a_exact, "半精度 (float16) 输入")

except Exception as e:                     # 若 float16 转换失败（某些环境不支持），则使用单精度
    print(f"\n半精度转换失败: {e}，改用单精度 (float32) 模拟")
    K_f32 = K.astype(np.float32)           # 转换为单精度
    R_f32 = R.astype(np.float32)
    print(f"K_f32 =\n{K_f32}")
    print(f"R_f32 = {R_f32}")
    K_f32_for_solve = K_f32.astype(np.float64)   # 转回 float64 用于求解
    R_f32_for_solve = R_f32.astype(np.float64)
    analyze_solution(K_f32_for_solve, R_f32_for_solve, a_exact, "单精度 (float32) 输入")

# ------------------------------------------------------------
# 6. 结论与解释
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("=" * 70)                         # 仅打印分隔线，实际结论需用户根据输出自行分析