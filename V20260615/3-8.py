"""
一维稳态对流扩散方程的有限元求解与稳定化
包含标准Galerkin、迎风格式和SUPG/Petrov-Galerkin方法
支持多个Peclet数对比和网格收敛性研究（附加题）
"""
# 模块文档字符串，说明本程序的功能

import numpy as np
import matplotlib.pyplot as plt
# 导入数值计算库 numpy 和绘图库 matplotlib

# ==================== 中文字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置 matplotlib 使用黑体，确保中文正常显示（Windows系统）
plt.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题

def element_matrix(kappa, v, le, alpha):
    """
    构造两节点线性单元的对流扩散单元矩阵
    参数：
        kappa: 扩散系数
        v: 对流速度
        le: 单元长度
        alpha: 稳定化参数（0=标准Galerkin，1=迎风，alpha_opt=SUPG）
    返回：
        2x2 单元刚度矩阵
    """
    # 人工扩散修正后的扩散系数：kappa_bar = kappa + alpha * v * le/2
    kappa_bar = kappa + alpha * v * le / 2.0
    # 扩散部分矩阵： (kappa_bar/le) * [[1, -1], [-1, 1]]
    K_diff = kappa_bar / le * np.array([[1, -1], [-1, 1]])
    # 对流部分矩阵： (v/2) * [[-1, 1], [-1, 1]]
    K_adv = v / 2.0 * np.array([[-1, 1], [-1, 1]])
    # 返回单元矩阵（扩散+对流）
    return K_diff + K_adv

def alpha_supg(Pe):
    """
    计算SUPG方法的最优alpha参数
    公式：alpha_opt = coth(Pe) - 1/Pe
    """
    # 若 Pe 极小（<1e-8），直接返回0，避免除零和数值不稳定
    if Pe < 1e-8:
        return 0.0
    # 若 Pe 极大（>50），coth(Pe) ≈ 1，直接返回1，防止溢出
    if Pe > 50:
        return 1.0
    # 计算双曲余切 coth(Pe) = 1/tanh(Pe)
    coth = 1.0 / np.tanh(Pe)
    # 返回最优 alpha
    return coth - 1.0 / Pe

def exact_solution(x, v, kappa, L):
    """
    计算精确解，使用expm1防止大指数溢出
    theta = (exp(v*x/kappa)-1) / (exp(v*L/kappa)-1)
    """
    # 使用 np.expm1(z) 计算 exp(z)-1，避免大指数时精度丢失
    numerator = np.expm1(v * x / kappa)
    denominator = np.expm1(v * L / kappa)
    # 若分母趋近于0（扩散主导，vL/kappa很小），解退化为线性分布
    if abs(denominator) < 1e-14:
        return x / L
    return numerator / denominator

def solve_advection_diffusion(nel, L, v, kappa, alpha, print_output=False):
    """
    求解一维对流扩散方程
    返回：x（节点坐标），theta（数值解），theta_exact（精确解）
    """
    le = L / nel                      # 单元长度
    n_nodes = nel + 1                 # 节点总数
    K = np.zeros((n_nodes, n_nodes))  # 初始化总体刚度矩阵（方阵）

    # 组装全局矩阵：遍历每个单元
    for i in range(nel):
        Ke = element_matrix(kappa, v, le, alpha)  # 计算单元矩阵
        nodes = [i, i+1]                          # 当前单元的两个全局节点编号
        # 将单元矩阵叠加到总体矩阵对应位置
        for a in range(2):
            for b in range(2):
                K[nodes[a], nodes[b]] += Ke[a, b]

    rhs = np.zeros(n_nodes)           # 右端项初始化为零向量

    # 施加 Dirichlet 边界条件：theta(0)=0, theta(L)=1
    # 左端点（节点0）：强制该行主对角元为1，其余为0，右端项为0
    K[0, :] = 0
    K[0, 0] = 1
    rhs[0] = 0
    # 右端点（最后一个节点）：强制该行主对角元为1，其余为0，右端项为1
    K[-1, :] = 0
    K[-1, -1] = 1
    rhs[-1] = 1

    # 求解线性方程组 K * theta = rhs
    theta = np.linalg.solve(K, rhs)

    # 生成节点坐标（等距）
    x = np.linspace(0, L, n_nodes)
    # 计算精确解
    theta_exact = exact_solution(x, v, kappa, L)

    # 若需要，打印节点坐标、数值解和精确解（满足任务1要求）
    if print_output:
        print("\n节点坐标 x：")
        print(x)
        print("\n数值解 theta_numerical：")
        print(theta)
        print("\n精确解 theta_exact：")
        print(theta_exact)

    return x, theta, theta_exact

def analyze_matrix(nel, L, v, kappa, alpha=0):
    """分析总体矩阵的对称性和正定性（任务4）"""
    le = L / nel
    n_nodes = nel + 1
    K = np.zeros((n_nodes, n_nodes))
    # 组装全局矩阵（与求解过程相同，但不施加边界条件）
    for i in range(nel):
        Ke = element_matrix(kappa, v, le, alpha)
        nodes = [i, i+1]
        for a in range(2):
            for b in range(2):
                K[nodes[a], nodes[b]] += Ke[a, b]

    print("\n=== 总体矩阵（未施加边界条件）===")
    # 如果节点数 ≤10，打印完整矩阵；否则只打印左上角5x5子块，避免输出过长
    if n_nodes <= 10:
        print(K)
    else:
        print(f"矩阵尺寸 {n_nodes} x {n_nodes}，仅显示左上角 5x5 子块：")
        print(K[:5, :5])

    # 检查对称性
    is_sym = np.allclose(K, K.T)
    print(f"\n矩阵是否对称？ {is_sym}")

    # 计算特征值，并取实部（因为可能是复数）
    eigvals = np.linalg.eigvals(K)
    min_eig = np.min(np.real(eigvals))
    # 若最小特征值实部 > 1e-12，认为正定
    is_pd = min_eig > 1e-12
    print(f"最小特征值实部： {min_eig:.3e}")
    print(f"矩阵是否正定？ {is_pd}")
    if not is_pd:
        print("注意：存在非正特征值，矩阵不定，这是高Peclet数下振荡的原因之一。")
    return K

def convergence_study(Pe, v=1.0, L=1.0, nel_list=[10, 20, 40, 80]):
    """附加题：网格收敛性分析"""
    errors = {'标准伽辽金': [], '迎风': [], 'SUPG': []}   # 存储各格式的误差
    for nel in nel_list:                     # 遍历不同的单元数（网格密度）
        le = L / nel                         # 单元长度
        kappa = v * le / (2 * Pe)            # 根据 Pe 计算扩散系数
        alpha_opt = alpha_supg(Pe)           # 计算 SUPG 最优参数

        # 分别求解三种格式
        _, theta_std, theta_ex = solve_advection_diffusion(nel, L, v, kappa, 0)
        _, theta_up, _ = solve_advection_diffusion(nel, L, v, kappa, 1)
        _, theta_supg, _ = solve_advection_diffusion(nel, L, v, kappa, alpha_opt)

        # 计算最大节点误差，存入列表
        errors['标准伽辽金'].append(np.max(np.abs(theta_std - theta_ex)))
        errors['迎风'].append(np.max(np.abs(theta_up - theta_ex)))
        errors['SUPG'].append(np.max(np.abs(theta_supg - theta_ex)))

    # 绘制误差-网格尺寸的双对数图
    plt.figure(figsize=(8,6))
    h = [L/nel for nel in nel_list]          # 各网格对应的单元长度
    for key, err in errors.items():
        plt.loglog(h, err, 'o-', label=key)   # 对数坐标绘图
    plt.xlabel('单元长度 h')
    plt.ylabel('最大节点误差')
    plt.title(f'网格收敛性分析 (Pe = {Pe})')
    plt.grid(True, which='both', ls='--')     # 显示网格
    plt.legend()
    plt.savefig(f'convergence_Pe{Pe}.png')    # 保存图片
    plt.show()

def main():
    """主函数：执行所有任务"""
    L = 1.0          # 区域长度
    nel = 20         # 单元数（固定）
    v = 1.0          # 对流速度
    Pe_list = [0.1, 3.0]   # 要计算的佩克莱数
    method_names = {0: '标准伽辽金', 1: '迎风', 'opt': 'SUPG'}  # 方法名称映射

    # 对每个 Pe 进行计算
    for Pe in Pe_list:
        le = L / nel
        kappa = v * le / (2 * Pe)                # 由 Pe 反推扩散系数
        print(f"\n========== Pe = {Pe}, kappa = {kappa:.6e} ==========")
        alpha_opt = alpha_supg(Pe)               # SUPG 最优参数

        # 在高密度点上计算精确解，用于绘制光滑曲线
        x_fine = np.linspace(0, L, 201)
        theta_exact_fine = exact_solution(x_fine, v, kappa, L)

        # 求解三种格式（标准伽辽金、迎风、SUPG）
        x, theta_std, theta_exact = solve_advection_diffusion(nel, L, v, kappa, 0,
                                                              print_output=(Pe==0.1))
        _, theta_up, _ = solve_advection_diffusion(nel, L, v, kappa, 1)
        _, theta_supg, _ = solve_advection_diffusion(nel, L, v, kappa, alpha_opt)

        # 计算各格式的最大节点误差
        errors = {
            '标准伽辽金': np.max(np.abs(theta_std - theta_exact)),
            '迎风': np.max(np.abs(theta_up - theta_exact)),
            'SUPG': np.max(np.abs(theta_supg - theta_exact))
        }
        for name, err in errors.items():
            print(f"{name:20s} : 最大节点误差 = {err:.6e}")

        # 绘制对比图（不同线型和标记）
        plt.figure(figsize=(10,6))
        plt.plot(x_fine, theta_exact_fine, 'k-', linewidth=2, label='精确解')
        plt.plot(x, theta_std, 'o-', label='标准伽辽金', linestyle='-', marker='o', markersize=4)
        plt.plot(x, theta_up, 's--', label='迎风', linestyle='--', marker='s', markersize=4)
        plt.plot(x, theta_supg, '^:', label='SUPG', linestyle=':', marker='^', markersize=4)

        plt.xlabel('x', fontsize=12)
        plt.ylabel('theta', fontsize=12)
        plt.title(f'对流扩散方程数值解对比 (Pe = {Pe}, nel = {nel})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(f'advection_diffusion_Pe{Pe}.png', dpi=150)   # 保存图片
        plt.show()

        # 输出误差表格
        print("\n误差汇总表：")
        print("-" * 40)
        print(f"{'方法':<20} {'最大节点误差':<15}")
        for name, err in errors.items():
            print(f"{name:<20} {err:.6e}")
        print("-" * 40)

    # 任务4：对 Pe=3.0 进行矩阵性质分析（标准伽辽金）
    print("\n" + "="*60)
    print("任务4：矩阵性质分析（Pe = 3.0，标准伽辽金）")
    Pe = 3.0
    le = L / nel
    kappa = v * le / (2 * Pe)
    analyze_matrix(nel, L, v, kappa, alpha=0)

    # 附加题：网格收敛性分析
    print("\n" + "="*60)
    print("附加题：网格收敛性分析 (Pe=3.0)")
    convergence_study(Pe=3.0, nel_list=[10,20,40,80])

if __name__ == "__main__":
    main()   # 程序入口，调用主函数