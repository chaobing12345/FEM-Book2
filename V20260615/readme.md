一维稳态对流扩散方程有限元求解程序

程序简介

本程序使用 Python 实现了对一维稳态对流扩散方程的有限元求解，包含三种数值格式：

标准伽辽金（Galerkin）：无稳定化，适用于扩散占优问题。
迎风格式（Upwind）：通过添加固定量的人工扩散消除振荡。
SUPG（Streamline Upwind Petrov-Galerkin）**：采用自适应最优稳定化参数，在稳定性和精度之间取得平衡。

程序支持多个佩克莱数（Pe）的对比计算，并提供网格收敛性分析作为附加题。

依赖库

- Python 3.6 及以上
- NumPy
- Matplotlib

安装命令（如使用 pip）：
```bash
pip install numpy matplotlib
文件结构：
.
main.py                    # 主程序源代码
EADME.md                  # 运行说明（本文件）
对流扩散方程程序设计作业.pdf  # 作业报告（PDF）
ction_diffusion_Pe0.1.png   # Pe=0.1 结果图
advection_diffusion_Pe3.0.png   # Pe=3.0 结果图
convergence_Pe3.0.png           # 附加题收敛曲线图
