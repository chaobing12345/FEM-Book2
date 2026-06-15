"""
poisson_mkl_pardiso_complete.py
求解单位正方形上的 Poisson 方程 -Δu = f, u|∂Ω = 0
采用线性三角形单元（T3）有限元方法，网格为 nx × ny 矩形均匀剖分。
严格使用 MKL PARDISO 稀疏求解器，输出缩减系统相对残差及有限元离散误差。
"""

import numpy as np
import time
from scipy.sparse import coo_matrix, csr_matrix

# ----------------------------- 依赖检查 -----------------------------
# 尝试导入 pypardiso 的 spsolve 函数（底层调用 Intel MKL PARDISO）
try:
    from pypardiso import spsolve as pardiso_solve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False
    print("错误：未安装 pypardiso，请执行: pip install pypardiso")
    exit(1)

# 尝试导入 matplotlib 绘图库（若没有则跳过绘图，但程序仍可运行）
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("警告：未安装 matplotlib，无法绘图。请运行 pip install matplotlib")


# ---------------------------- 网格生成函数 -----------------------------
def mesh_rectangle(nx, ny, Lx=1.0, Ly=1.0):
    """
    生成矩形域 [0,Lx]×[0,Ly] 上的三角形网格。
    输入：nx, ny - x 和 y 方向的矩形单元个数
          Lx, Ly - 矩形域的宽度和高度（默认均为1.0）
    返回：
        nodes   : (N, 2) 节点坐标数组，N = (nx+1)*(ny+1)
        elements: (M, 3) 单元节点索引数组，M = 2 * nx * ny（每个矩形分成两个三角形）
    """
    npx = nx + 1                      # x方向节点数 = 单元数 + 1
    npy = ny + 1                      # y方向节点数
    x = np.linspace(0, Lx, npx)       # 在[0, Lx]上生成npx个等分点（包含端点）
    y = np.linspace(0, Ly, npy)       # 在[0, Ly]上生成npy个等分点
    # 生成所有节点坐标：按行优先（先y后x）遍历所有(x,y)对，并存入二维数组
    nodes = np.array([[xi, yj] for yj in y for xi in x])

    elements = []                     # 用于存储单元节点索引的列表
    for j in range(ny):               # 遍历每一行矩形单元（y方向）
        for i in range(nx):           # 遍历每一列矩形单元（x方向）
            # 当前矩形四个顶点的全局节点编号（左下，右下，左上，右上）
            n0 = j * npx + i          # 左下角节点编号
            n1 = j * npx + i + 1      # 右下角节点编号
            n2 = (j + 1) * npx + i    # 左上角节点编号
            n3 = (j + 1) * npx + i + 1# 右上角节点编号
            # 将一个矩形分割为两个三角形：左下-右下-左上 和 右下-右上-左上
            elements.append([n0, n1, n2])   # 第一个三角形（左下三角形）
            elements.append([n1, n3, n2])   # 第二个三角形（右上三角形）
    return nodes, np.array(elements)        # 返回节点数组和单元数组（M×3）


# ---------------------------- 单元刚度矩阵（线性三角形） -----------------
def tri_stiffness(nodes, elem):
    """
    计算单个三角形单元的刚度矩阵 (3x3)。
    公式：K_e = area * (B^T B)，其中 B 为应变-位移矩阵（常数）。
    输入：nodes - 全部节点坐标数组
          elem  - 长度为3的列表，包含该单元三个节点的全局编号
    输出：ke    - 3×3 的单元刚度矩阵
    """
    # 获取三个节点的坐标 (x0,y0), (x1,y1), (x2,y2)
    x0, y0 = nodes[elem[0]]
    x1, y1 = nodes[elem[1]]
    x2, y2 = nodes[elem[2]]

    # 计算三角形面积：0.5 * |(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)|
    area = 0.5 * abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0))

    # 构造 B 矩阵（2行3列），每个元素为形函数导数组合
    # B = [ [dN1/dx, dN2/dx, dN3/dx],
    #       [dN1/dy, dN2/dy, dN3/dy] ]
    # 对于线性三角形，导数由顶点坐标决定
    B = np.array([
        [y1 - y2, y2 - y0, y0 - y1],
        [x2 - x1, x0 - x2, x1 - x0]
    ]) / (2.0 * area)

    # 单元刚度矩阵 = area * (B^T B)
    ke = area * (B.T @ B)            # @ 表示矩阵乘法
    return ke


def tri_load(nodes, elem, f_func):
    """
    计算三角形单元的载荷向量 (3x1)。
    采用中心点积分近似：将单元中心点的 f 值乘以面积后均匀分配到三个节点。
    输入：nodes   - 全部节点坐标数组
          elem    - 单元三个节点的全局编号
          f_func  - 右端项函数 f(x, y)
    输出：fe      - 长度为3的单元载荷向量
    """
    # 计算单元中心坐标（三个顶点坐标的平均值）
    xc = np.mean(nodes[elem], axis=0)   # axis=0 表示对各列取平均，得到 [xc, yc]
    f_center = f_func(xc[0], xc[1])     # 计算中心点处的源项 f 值

    # 重新计算面积（也可以作为参数传入，但为函数独立而重复计算）
    x0, y0 = nodes[elem[0]]
    x1, y1 = nodes[elem[1]]
    x2, y2 = nodes[elem[2]]
    area = 0.5 * abs((x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0))

    # 每个节点分配到 f_center * area / 3
    fe = np.array([1.0, 1.0, 1.0]) * f_center * area / 3.0
    return fe


# ---------------------------- 总体装配（稀疏COO -> CSR） -----------------
def assemble_poisson(nodes, elements, f_func):
    """
    装配总体刚度矩阵（稀疏）和右端项。
    输入：nodes     - 节点坐标数组 (N,2)
          elements  - 单元节点索引数组 (M,3)
          f_func    - 源项函数 f(x,y)
    返回：
        K_sparse : csr_matrix 格式的总体刚度矩阵 (N×N)
        rhs      : 右端项向量 (N,)
    """
    n_nodes = len(nodes)               # 节点总数
    # 准备 COO 格式数据：行索引列表、列索引列表、值列表
    rows = []                          # 存储行号
    cols = []                          # 存储列号
    vals = []                          # 存储矩阵元素值
    rhs = np.zeros(n_nodes)            # 初始化右端项为全0

    # 遍历每一个单元
    for elem in elements:              # elem 是长度为3的列表 [n1, n2, n3]
        # 计算当前单元的单元刚度矩阵和单元载荷向量
        ke = tri_stiffness(nodes, elem)   # 3×3 矩阵
        fe = tri_load(nodes, elem, f_func)# 3×1 向量

        # 将单元信息装配到总体矩阵和右端项
        for i, gi in enumerate(elem):     # 局部节点 i 对应全局节点编号 gi
            rhs[gi] += fe[i]              # 载荷向量累加（集中到右端项）
            for j, gj in enumerate(elem): # 局部节点 j 对应全局节点编号 gj
                rows.append(gi)           # 记录行索引（全局节点号）
                cols.append(gj)           # 记录列索引（全局节点号）
                vals.append(ke[i, j])     # 记录刚度矩阵元素值

    # 使用 COO 格式创建稀疏矩阵
    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    # 将 COO 转换为 CSR 格式（压缩行），便于后续快速矩阵运算和求解
    K_sparse = K_coo.tocsr()
    return K_sparse, rhs


# ---------------------------- 边界条件施加（Dirichlet u=0）-----------------
def apply_dirichlet(K_sparse, rhs, nodes, tol=1e-10):
    """
    将边界上的自由度固定为 0（齐次Dirichlet条件）。
    边界条件：所有 x=0 或 x=1 或 y=0 或 y=1 的节点（考虑浮点容差 tol）。
    返回缩减后的刚度矩阵、右端项以及自由节点索引。
    输入：
        K_sparse : csr_matrix 总体刚度矩阵
        rhs      : 总体右端项向量
        nodes    : 节点坐标数组
        tol      : 判断边界坐标的容差（默认1e-10）
    输出：
        K_FF      : 自由-自由子矩阵 (csr_matrix)
        rhs_F     : 自由节点对应的右端项向量
        free_nodes: 自由节点的全局编号数组
    """
    # 找出边界节点：坐标在容差范围内接近0或1的节点
    fixed_nodes = []                     # 存储边界节点编号
    for i, (x, y) in enumerate(nodes):   # 遍历所有节点
        if x < tol or x > 1.0 - tol or y < tol or y > 1.0 - tol:
            fixed_nodes.append(i)        # 满足条件则加入边界列表
    fixed_nodes = np.array(fixed_nodes)  # 转换为NumPy数组
    fixed_vals = np.zeros(len(fixed_nodes))  # 边界位移值均为0（齐次边界）

    # 自由节点 = 全体节点 - 边界节点
    all_nodes = np.arange(len(nodes))    # 所有节点编号 [0,1,...,N-1]
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)  # 差集得到自由节点

    # 提取自由-自由子矩阵 K_FF
    K_FF = K_sparse[free_nodes, :][:, free_nodes]   # 先按行取自由节点，再按列取自由节点
    # 计算自由节点上的右端项：rhs_F = rhs[free] - K_FE * d_E，由于 d_E=0，故直接取 rhs[free]
    rhs_F = rhs[free_nodes] - K_sparse[free_nodes, :][:, fixed_nodes] @ fixed_vals

    return K_FF, rhs_F, free_nodes


# ---------------------------- 绘图函数（数值解、精确解、误差云图）-----------
def plot_solution(nodes, u_num, u_ex, nx, ny):
    """
    绘制三个子图：数值解云图、精确解云图、绝对误差云图。
    自动保存为 PNG 文件，并在屏幕上显示。
    """
    if not HAS_PLT:                     # 如果没有 matplotlib，直接返回
        return

    # 节点坐标按网格形状重塑（假设节点顺序是先 x 后 y，按行优先存储）
    npx = nx + 1                        # x方向节点数
    npy = ny + 1                        # y方向节点数
    X = nodes[:, 0].reshape(npy, npx)   # x坐标的网格矩阵（npy行，npx列）
    Y = nodes[:, 1].reshape(npy, npx)   # y坐标的网格矩阵
    U_num = u_num.reshape(npy, npx)     # 数值解网格
    U_ex = u_ex.reshape(npy, npx)       # 精确解网格
    Err = np.abs(U_num - U_ex)          # 绝对误差网格

    # 创建 1×3 的子图，整体画布大小为 15×5 英寸
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 第一个子图：数值解云图
    c1 = axes[0].contourf(X, Y, U_num, levels=50, cmap=cm.jet)  # 填充等高线，50个色阶
    axes[0].set_title(f"Numerical solution (nx={nx})")          # 设置标题
    axes[0].set_xlabel("x")                                     # x轴标签
    axes[0].set_ylabel("y")                                     # y轴标签
    plt.colorbar(c1, ax=axes[0])                                # 添加颜色条

    # 第二个子图：精确解云图
    c2 = axes[1].contourf(X, Y, U_ex, levels=50, cmap=cm.jet)
    axes[1].set_title("Exact solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(c2, ax=axes[1])

    # 第三个子图：绝对误差云图（使用 hot 色系强调误差）
    c3 = axes[2].contourf(X, Y, Err, levels=50, cmap=cm.hot)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(c3, ax=axes[2])

    plt.tight_layout()                              # 自动调整子图间距，避免重叠
    plt.savefig(f"poisson_solution_nx_{nx}.png", dpi=150)  # 保存图片，分辨率150dpi
    plt.show()                                      # 显示图片
    print(f"已保存图片 poisson_solution_nx_{nx}.png")


# ---------------------------- 有限元离散误差计算 ----------------------------
def compute_energy_error(u_num, u_ex, nx, ny):
    """
    计算能量范数误差 ∫_Ω |∇(u_num - u_exact)|² dΩ。
    输入：
        u_num : 数值解节点值（一维数组，按行优先排列）
        u_ex  : 精确解节点值（一维数组，按行优先排列）
        nx, ny: 网格单元数
    输出：
        energy_error : 能量范数误差
    """
    # 重塑为网格矩阵以便使用梯度函数
    u_num_mat = u_num.reshape(ny+1, nx+1)
    u_ex_mat = u_ex.reshape(ny+1, nx+1)
    # 计算梯度（注意 np.gradient 返回 (dy, dx) 顺序）
    grad_num_y, grad_num_x = np.gradient(u_num_mat)
    grad_ex_y, grad_ex_x = np.gradient(u_ex_mat)
    # 梯度差平方和
    grad_error_sq = (grad_num_x - grad_ex_x)**2 + (grad_num_y - grad_ex_y)**2
    # 单元面积（矩形单元的面积）
    hx = 1.0 / nx
    hy = 1.0 / ny
    area = hx * hy
    # 能量范数误差 = sqrt(∑(梯度差平方) * 单元面积)
    energy_error = np.sqrt(np.sum(grad_error_sq) * area)
    return energy_error


# ---------------------------- 主求解函数（严格使用 MKL PARDISO）-----------
def solve_poisson(nx, ny):
    """
    使用 MKL PARDISO 求解泊松方程，输出误差分析和性能数据，并绘制云图。
    参数：
        nx, ny : x 和 y 方向的单元数（矩形个数）
    """
    print(f"\n========== 泊松方程求解 (nx={nx}, ny={ny}) ==========")

    # 定义精确解 u(x,y) 和右端项 f(x,y) = -Δu
    def u_exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)   # 精确解

    def f_func(x, y):
        # 对精确解求拉普拉斯： -Δu = 2π² sin(πx) sin(πy)
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    # 1. 生成网格
    nodes, elements = mesh_rectangle(nx, ny)          # 返回节点坐标和单元节点索引

    # 2. 装配总体刚度矩阵和右端项
    t_assembly = time.perf_counter()                  # 开始计时（高精度时钟）
    K_sparse, rhs = assemble_poisson(nodes, elements, f_func)
    t_assembly = time.perf_counter() - t_assembly     # 计算装配耗时

    # 3. 施加齐次 Dirichlet 边界条件（u=0 在边界上）
    t_bc = time.perf_counter()                        # 边界处理开始计时
    K_FF, rhs_F, free_nodes = apply_dirichlet(K_sparse, rhs, nodes)
    t_bc = time.perf_counter() - t_bc                 # 边界处理耗时

    # ================== MKL PARDISO 求解器配置 ==================
    # 配置 iparm 参数数组（长度为 64，PARDISO 使用 1-based 索引）
    iparm = np.zeros(64, dtype=np.int32)

    # iparm[0] = 0: 使用默认值（其他参数默认使用内置值）
    # iparm[1] = 2: 使用 METIS 嵌套分割算法，减少填充元，提高求解效率
    iparm[1] = 2

    # iparm[7] = 1: 开启迭代精化 (iterative refinement)
    # 设置最大迭代步数为 1（启用迭代精化）
    iparm[7] = 1

    # iparm[9] = 8: 设置迭代精化的最大次数（默认 8，一般足够）
    iparm[9] = 8

    # iparm[10] = 1: 开启矩阵缩放（改善数值稳定性）
    iparm[10] = 1

    # 注意：iparm 数组其他元素保持 0（使用默认值）

    # 4. 调用 MKL PARDISO 求解线性方程组 K_FF * u_free = rhs_F
    t_solve = time.perf_counter()                     # 求解开始计时
    try:
        # 调用 pypardiso.spsolve，传递 iparm 参数
        u_free = pardiso_solve(K_FF, rhs_F, iparm=iparm)
        solver_name = "MKL PARDISO (with iparm[7]=1, iterative refinement)"
    except Exception as e:                            # 若求解失败（如矩阵奇异），捕获异常
        print(f"PARDISO 求解失败: {e}")
        return None, None, None, None
    t_solve = time.perf_counter() - t_solve           # 求解耗时

    # 5. 组装完整位移场：自由节点赋计算值，边界节点赋0（固定）
    u_num = np.zeros(len(nodes))                      # 初始化全零位移向量
    u_num[free_nodes] = u_free                        # 将自由节点解填入

    # 6. 计算有限元离散误差
    # 6.1 节点最大绝对误差 (L∞)
    u_ex = np.array([u_exact(x, y) for x, y in nodes])   # 列表推导式生成精确解数组
    err_max = np.max(np.abs(u_num - u_ex))                # 最大绝对误差

    # 6.2 离散 L2 相对误差
    # L2 相对误差 = sqrt(∑(u_num - u_ex)^2 / ∑(u_ex)^2)
    err_l2 = np.sqrt(np.sum((u_num - u_ex) ** 2) / np.sum(u_ex ** 2))

    # 6.3 能量范数误差
    energy_error = compute_energy_error(u_num, u_ex, nx, ny)

    # 7. 计算缩减系统的相对残差（这是代数求解器的关键指标）
    # 注意：这里的 K_FF, rhs_F, u_free 对应缩减后的系统（自由节点）
    residual = rhs_F - K_FF @ u_free
    rel_resid = np.linalg.norm(residual) / np.linalg.norm(rhs_F)

    # 8. 输出结果（明确标注“缩减系统相对残差”）
    print(f"单元类型: 线性三角形 (T3)")
    print(f"节点数: {len(nodes)}")
    print(f"单元数: {len(elements)}")
    print(f"未知自由度: {len(free_nodes)}")
    print(f"非零元个数: {K_sparse.nnz}")          # nnz 属性返回稀疏矩阵的非零元素个数
    print(f"装配时间: {t_assembly:.4f} s")
    print(f"边界条件时间: {t_bc:.4f} s")
    print(f"求解时间: {t_solve:.4f} s")
    print(f"求解器: {solver_name}")
    print(f"\n=== 有限元离散误差分析 ===")
    print(f"节点最大绝对误差 (L∞): {err_max:.4e}")
    print(f"离散 L2 相对误差: {err_l2:.4e}")
    print(f"能量范数误差: {energy_error:.4e}")
    print(f"缩减系统相对残差: {rel_resid:.2e}")   # 注意：明确输出缩减系统残差

    # 9. 绘制云图
    plot_solution(nodes, u_num, u_ex, nx, ny)

    return u_num, nodes, err_max, err_l2, energy_error, rel_resid


# ---------------------------- 主程序入口 ----------------------------------
if __name__ == "__main__":
    # 确保 pypardiso 已安装（前面已检查，若未安装则 exit(1) 已执行）
    if not HAS_PARDISO:
        print("请先安装 pypardiso: pip install pypardiso")
        exit(1)

    # 测试不同网格规模，可根据计算机性能调整列表中的数值
    for nx, ny in [(10, 10), (100, 100), (500, 500), (1000, 1000)]:
        solve_poisson(nx, ny)               # 依次求解并绘图
        print("-" * 60)                     # 打印分隔线，区分不同网格的结果