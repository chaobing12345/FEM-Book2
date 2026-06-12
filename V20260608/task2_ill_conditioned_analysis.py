"""
task2_ill_conditioned_analysis.py
任务2：病态线性方程组的误差、残差和条件数分析
构造矩阵 K = [[1,1],[1,1.0001]]，精确解 a = [1,1]^T。
分别用双精度、4位有效数字、半精度/单精度求解，比较误差。
"""

import numpy as np

# ------------------------------------------------------------
# 1. 定义病态矩阵和精确解
# ------------------------------------------------------------
# 矩阵 K（对称且接近奇异）
K = np.array([[1.0000, 1.0000],
              [1.0000, 1.0001]], dtype=np.float64)

a_exact = np.array([1.0, 1.0])        # 精确解
R = K @ a_exact                       # 精确右端项

# 计算条件数（2-范数）
cond_K = np.linalg.cond(K, p=2)

print("=" * 70)
print("任务2：病态线性方程组误差分析")
print("=" * 70)
print(f"系数矩阵 K:\n{K}")
print(f"精确解 a_exact: {a_exact}")
print(f"右端项 R (精确计算): {R}")
print(f"矩阵条件数 cond(K) = {cond_K:.4f}")
print("注：条件数远大于 1，为病态矩阵。")
print("=" * 70)

# ------------------------------------------------------------
# 2. 辅助函数：求解并输出各项指标
# ------------------------------------------------------------
def analyze_solution(K, R, a_exact, precision_name):
    """
    求解 K a = R，输出解、残差、相对残差、相对误差。
    假设 K 和 R 已经是目标精度的数组。
    """
    try:
        a = np.linalg.solve(K, R)      # 调用求解器（此处使用 numpy 的求解）
    except np.linalg.LinAlgError as e:
        print(f"{precision_name}: 求解失败 - {e}")
        return

    r = R - K @ a                      # 残差向量
    norm_r = np.linalg.norm(r)
    norm_R = np.linalg.norm(R)
    rel_residual = norm_r / norm_R

    err_abs = a - a_exact
    rel_error = np.linalg.norm(err_abs) / np.linalg.norm(a_exact)

    print(f"\n【{precision_name}】")
    print(f"  数值解 a = {a}")
    print(f"  残差 r = {r}")
    print(f"  残差范数 ||r|| = {norm_r:.4e}")
    print(f"  相对残差 ||r||/||R|| = {rel_residual:.4e}")
    print(f"  相对误差 ||a - a_exact||/||a_exact|| = {rel_error:.4e}")

# ------------------------------------------------------------
# 3. 双精度计算（基准）
# ------------------------------------------------------------
analyze_solution(K, R, a_exact, "双精度 (float64)")

# ------------------------------------------------------------
# 4. 模拟四舍五入到 4 位有效数字的初始截断误差
# ------------------------------------------------------------
def round_to_sigfigs(x, sigfigs=4):
    """将浮点数 x 四舍五入到 sigfigs 位有效数字"""
    if x == 0:
        return 0.0
    magnitude = np.floor(np.log10(abs(x)))
    factor = 10 ** (sigfigs - 1 - magnitude)
    return np.round(x * factor) / factor

# 对 K 和 R 中的每个元素进行四舍五入
K_4sig = np.vectorize(round_to_sigfigs)(K)
R_4sig = np.vectorize(round_to_sigfigs)(R)

print("\n" + "-" * 70)
print("四舍五入到 4 位有效数字后的矩阵和右端项：")
print(f"K_4sig =\n{K_4sig}")
print(f"R_4sig = {R_4sig}")

analyze_solution(K_4sig, R_4sig, a_exact, "4位有效数字（模拟初始截断误差）")

# ------------------------------------------------------------
# 5. 半精度 (float16) 或单精度 (float32) 模拟
# ------------------------------------------------------------
try:
    # 转换为 float16（半精度，约 3-4 位有效数字）
    K_f16 = K.astype(np.float16)
    R_f16 = R.astype(np.float16)
    print("\n" + "-" * 70)
    print("转换为 float16 后的矩阵和右端项：")
    print(f"K_f16 =\n{K_f16}")
    print(f"R_f16 = {R_f16}")

    # 注意：np.linalg.solve 会将输入提升为 float64，但输入精度已经丢失
    K_f16_for_solve = K_f16.astype(np.float64)
    R_f16_for_solve = R_f16.astype(np.float64)
    analyze_solution(K_f16_for_solve, R_f16_for_solve, a_exact, "半精度 (float16) 输入")

except Exception as e:
    print(f"\n半精度转换失败: {e}，改用单精度 (float32) 模拟")
    K_f32 = K.astype(np.float32)
    R_f32 = R.astype(np.float32)
    print(f"K_f32 =\n{K_f32}")
    print(f"R_f32 = {R_f32}")
    K_f32_for_solve = K_f32.astype(np.float64)
    R_f32_for_solve = R_f32.astype(np.float64)
    analyze_solution(K_f32_for_solve, R_f32_for_solve, a_exact, "单精度 (float32) 输入")

# ------------------------------------------------------------
# 6. 结论与解释
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("=" * 70)
