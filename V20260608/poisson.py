"""
poisson.py
算例4：求解单位正方形上的 Poisson 方程 -Δu = f, u|∂Ω = 0
采用线性三角形单元（T3）有限元方法，网格为 nx × ny 矩形均匀剖分（每个矩形分成两个三角形）。
调用 MKL PARDISO 稀疏求解器，输出误差分析并绘制云图。
"""

import numpy as np                     # 导入NumPy库，用于数值计算（数组、数学函数等）
import time                            # 导入time模块，用于测量各阶段运行时间
from scipy.sparse import coo_matrix, csr_matrix   # 导入稀疏矩阵格式：COO（三元组）和CSR（压缩行）
from equilibrium_solver import solve_equilibrium # 从自定义模块导入统一求解接口（支持LDLT和PARDISO）

# 尝试导入 matplotlib，若没有则提示但程序仍可运行
try:
    import matplotlib.pyplot as plt    # 导入pyplot，用于绘图
    from matplotlib import cm          # 导入colormap，用于云图配色
    HAS_PLT = True                     # 标记matplotlib可用
except ImportError:
    HAS_PLT = False                    # 标记matplotlib不可用
    print("警告：未安装 matplotlib，无法绘制云图。请运行 pip install matplotlib")

# ----------------------------------------------------------------------
# 网格生成
# ----------------------------------------------------------------------
def mesh_rectangle(nx, ny, Lx=1.0, Ly=1.0):
    """
    生成矩形域 [0,Lx]×[0,Ly] 上的三角形网格。
    输入：nx, ny - x 和 y 方向的单元数（每个方向矩形个数）
    返回：
        nodes : (N, 2) 节点坐标数组
        elements : (M, 3) 单元节点索引（0-based）
    """
    npx = nx + 1          # x 方向节点数 = 单元数 + 1
    npy = ny + 1          # y 方向节点数
    x = np.linspace(0, Lx, npx)   # 生成 x 坐标序列（等间距）
    y = np.linspace(0, Ly, npy)   # 生成 y 坐标序列

    # 生成节点坐标：先遍历 y 再遍历 x，形成 (x,y) 对，节点编号按行优先（先x后y）
    nodes = np.array([[xi, yj] for yj in y for xi in x])

    elements = []                     # 存放单元节点索引列表
    for j in range(ny):               # 遍历每个矩形单元（y方向）
        for i in range(nx):           # 遍历每个矩形单元（x方向）
            # 矩形四个顶点索引（左下，右下，左上，右上）
            n0 = j * npx + i          # 左下角节点编号
            n1 = j * npx + i + 1      # 右下角节点编号
            n2 = (j + 1) * npx + i    # 左上角节点编号
            n3 = (j + 1) * npx + i + 1# 右上角节点编号
            # 将一个矩形拆分成两个三角形：左下-右下-左上 和 右下-右上-左上
            elements.append([n0, n1, n2])   # 第一个三角形
            elements.append([n1, n3, n2])   # 第二个三角形
    return nodes, np.array(elements)        # 返回节点数组和单元数组（M×3）

# ----------------------------------------------------------------------
# 单元刚度矩阵（线性三角形，常数应变）
# ----------------------------------------------------------------------
def tri_stiffness(nodes, elem):
    """
    计算单个三角形单元的刚度矩阵 (3x3)
    公式：K_e = area * B^T * B, 其中 B 为应变-位移矩阵（常数）
    """
    # 三个节点坐标
    x0, y0 = nodes[elem[0]]
    x1, y1 = nodes[elem[1]]
    x2, y2 = nodes[elem[2]]

    # 单元面积（有向面积的一半，取绝对值）
    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    # 形函数导数（常数）矩阵 B，大小为 2x3
    # B = [ [dN1/dx, dN2/dx, dN3/dx],
    #       [dN1/dy, dN2/dy, dN3/dy] ]
    # 对于线性三角形，dNi/dx 和 dNi/dy 由边长决定
    B = np.array([
        [y1 - y2, y2 - y0, y0 - y1],
        [x2 - x1, x0 - x2, x1 - x0]
    ]) / (2.0 * area)

    # 单元刚度矩阵 = area * (B^T B)
    ke = area * (B.T @ B)
    return ke

# ----------------------------------------------------------------------
# 单元载荷向量（右端项），f 为已知函数
# ----------------------------------------------------------------------
def tri_load(nodes, elem, f_func):
    """
    计算三角形单元的载荷向量 (3x1)
    采用中心点积分近似： f_e ≈ f(中心点) * area / 3 分配到三个节点
    """
    # 单元中心坐标（三个顶点坐标的平均值）
    xc = np.mean(nodes[elem], axis=0)
    f_center = f_func(xc[0], xc[1])   # 计算中心点处的源项 f 值

    # 计算面积（再次计算，也可传入，但为独立函数而重复）
    x0, y0 = nodes[elem[0]]
    x1, y1 = nodes[elem[1]]
    x2, y2 = nodes[elem[2]]
    area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    # 集中到三个节点（每个节点得到 f_center * area / 3）
    fe = np.array([1.0, 1.0, 1.0]) * f_center * area / 3.0
    return fe

# ----------------------------------------------------------------------
# 总体装配（稀疏 COO 格式）
# ----------------------------------------------------------------------
def assemble_poisson(nodes, elements, f_func):
    """
    装配总体刚度矩阵（稀疏）和右端项。
    返回：
        K_sparse : csr_matrix 格式的总体刚度矩阵
        rhs      : 右端项向量（节点力）
    """
    n_nodes = len(nodes)              # 节点总数
    # 准备 COO 格式数据：行索引列表、列索引列表、值列表
    rows = []
    cols = []
    vals = []
    rhs = np.zeros(n_nodes)           # 初始化右端项为0

    for elem in elements:             # 遍历每个单元
        # 计算单元刚度矩阵和载荷向量
        ke = tri_stiffness(nodes, elem)
        fe = tri_load(nodes, elem, f_func)

        # 组装到总体矩阵和右端项
        for i, gi in enumerate(elem):           # 单元局部节点 i → 全局节点 gi
            rhs[gi] += fe[i]                    # 载荷向量累加
            for j, gj in enumerate(elem):       # 单元局部节点 j → 全局节点 gj
                rows.append(gi)                 # COO 行索引
                cols.append(gj)                 # COO 列索引
                vals.append(ke[i, j])           # 刚度矩阵元素值

    # 创建 COO 矩阵并转换为 CSR（提高求解效率）
    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    K_sparse = K_coo.tocsr()          # 转换为 CSR 格式，用于快速矩阵运算和求解
    return K_sparse, rhs

# ----------------------------------------------------------------------
# 施加 Dirichlet 边界条件（u=0 在边界上）
# ----------------------------------------------------------------------
def apply_dirichlet(K_sparse, rhs, nodes, tol=1e-10):
    """
    将边界上的自由度固定为 0。
    边界条件：所有 x=0 或 x=1 或 y=0 或 y=1 的节点。
    返回缩减后的刚度矩阵、右端项以及自由节点索引。
    """
    # 找出边界节点（坐标为 0 或 1 的节点，考虑浮点容差）
    fixed_nodes = []
    for i, (x, y) in enumerate(nodes):
        if x < tol or x > 1.0 - tol or y < tol or y > 1.0 - tol:
            fixed_nodes.append(i)
    fixed_nodes = np.array(fixed_nodes)          # 边界节点编号数组
    fixed_vals = np.zeros(len(fixed_nodes))      # 所有边界位移为 0

    # 自由节点集合：所有节点中除去边界节点
    all_nodes = np.arange(len(nodes))
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)

    # 从稀疏矩阵中提取自由-自由子矩阵 K_FF
    K_FF = K_sparse[free_nodes, :][:, free_nodes]
    # 右端项：rhs_F = rhs[free] - K_FE * d_E，由于 d_E = 0，所以 rhs_F = rhs[free]
    rhs_F = rhs[free_nodes] - K_sparse[free_nodes, :][:, fixed_nodes] @ fixed_vals

    return K_FF, rhs_F, free_nodes

# ----------------------------------------------------------------------
# 绘图函数（数值解、理论解、误差云图）
# ----------------------------------------------------------------------
def plot_solution(nodes, u_num, u_ex, nx, ny):
    """
    绘制数值解云图、理论解云图和绝对误差云图。
    自动保存为 PNG 文件并显示。
    """
    if not HAS_PLT:          # 如果没有matplotlib，直接返回，不绘图
        print("跳过绘图：未安装 matplotlib")
        return

    # 节点坐标 reshape 为网格形式（假设节点顺序是先 x 后 y）
    npx = nx + 1
    npy = ny + 1
    X = nodes[:, 0].reshape(npy, npx)   # x 坐标网格
    Y = nodes[:, 1].reshape(npy, npx)   # y 坐标网格
    U_num = u_num.reshape(npy, npx)     # 数值解网格
    U_ex = u_ex.reshape(npy, npx)       # 理论解网格
    Err = np.abs(U_num - U_ex)          # 绝对误差网格

    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 数值解云图
    c1 = axes[0].contourf(X, Y, U_num, levels=50, cmap=cm.jet)
    axes[0].set_title(f"Numerical solution (nx={nx})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(c1, ax=axes[0])

    # 理论解云图
    c2 = axes[1].contourf(X, Y, U_ex, levels=50, cmap=cm.jet)
    axes[1].set_title("Exact solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(c2, ax=axes[1])

    # 绝对误差云图
    c3 = axes[2].contourf(X, Y, Err, levels=50, cmap=cm.hot)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(c3, ax=axes[2])

    plt.tight_layout()                         # 自动调整子图间距
    plt.savefig(f"poisson_solution_nx_{nx}.png", dpi=150)   # 保存图片
    plt.show()                                 # 显示图片
    print(f"已保存图片 poisson_solution_nx_{nx}.png")

# ----------------------------------------------------------------------
# 主求解函数
# ----------------------------------------------------------------------
def solve_poisson(nx, ny, method="pardiso"):
    """
    求解泊松方程，输出误差和性能数据，并绘制云图。
    参数：
        nx, ny : x 和 y 方向单元数
        method : 求解方法（"ldlt" 或 "pardiso"），对于大规模问题建议 "pardiso"
    """
    print(f"\n========== 泊松方程求解 (nx={nx}, ny={ny}) ==========")

    # 1. 生成网格
    nodes, elements = mesh_rectangle(nx, ny)

    # 定义理论解和右端项函数（源项 f）
    def u_exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)   # 精确解

    def f_func(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)   # 拉普拉斯 -Δu 的结果

    # 2. 装配总体矩阵和右端项
    t_assembly_start = time.perf_counter()            # 开始计时
    K_sparse, rhs = assemble_poisson(nodes, elements, f_func)
    t_assembly = time.perf_counter() - t_assembly_start   # 装配耗时

    # 3. 施加边界条件
    t_bc_start = time.perf_counter()
    K_FF, rhs_F, free_nodes = apply_dirichlet(K_sparse, rhs, nodes)
    t_bc = time.perf_counter() - t_bc_start

    # 4. 求解缩减方程（使用统一接口 solve_equilibrium）
    t_solve_start = time.perf_counter()
    u_free, info = solve_equilibrium(K_FF, rhs_F, method=method, sparse=True)
    t_solve = time.perf_counter() - t_solve_start

    # 5. 组装完整位移场（包括边界节点）
    u_num = np.zeros(len(nodes))
    u_num[free_nodes] = u_free

    # 6. 误差分析
    u_ex = np.array([u_exact(x, y) for x, y in nodes])   # 所有节点处的精确解
    err_max = np.max(np.abs(u_num - u_ex))                # 最大绝对误差
    err_l2 = np.sqrt(np.sum((u_num - u_ex) ** 2) / np.sum(u_ex ** 2))   # 离散L2相对误差

    # 残差计算（在自由节点上）
    residual = rhs_F - K_FF @ u_free
    rel_resid = np.linalg.norm(residual) / np.linalg.norm(rhs_F)

    # 7. 输出结果
    print(f"单元类型: 线性三角形 (T3)")
    print(f"节点数: {len(nodes)}")
    print(f"单元数: {len(elements)}")
    print(f"未知自由度: {len(free_nodes)}")
    print(f"总体刚度矩阵非零元个数: {K_sparse.nnz}")
    print(f"装配时间: {t_assembly:.4f} s")
    print(f"边界条件处理时间: {t_bc:.4f} s")
    print(f"求解时间: {t_solve:.4f} s")
    print(f"求解器: {info['method']}")
    print(f"相对残差 ||R - Ka|| / ||R|| = {rel_resid:.2e}")
    print(f"节点最大误差: {err_max:.4e}")
    print(f"离散L2相对误差: {err_l2:.4e}")

    # 8. 绘图
    plot_solution(nodes, u_num, u_ex, nx, ny)

    return u_num, nodes, err_max, err_l2

# ----------------------------------------------------------------------
# 直接运行脚本时，测试多个网格规模
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 可根据计算机性能调整网格大小。推荐先测试 50 和 100。
    for nx, ny in [(10,10),(100,100),(500, 500), (1000, 1000)]:   # 依次测试两种网格密度
        solve_poisson(nx, ny, method="pardiso")   # 使用 PARDISO 求解
        print("-" * 60)                       # 打印分隔线