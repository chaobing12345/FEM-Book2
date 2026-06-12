
有限元平衡方程组求解器
项目结构

equilibrium_solver.py # LDL^T + PARDISO 统一求解接口
solver.py # 缩减法求解（调用 equilibrium_solver）
model.py / element.py / assembly.py / postprocess.py # 2.3 作业模块（不变）
main.py # 桁架算例（一维/二维）
tridiagonal_test.py # 算例1：三对角矩阵
test_nonpositive.py # 算例2：非正定检测
task2_ill_conditioned_analysis.py # 任务2：病态分析
poisson.py # 算例4：Poisson 方程（含云图）
colsol.py # 活动列存储演示
truss_1d_input.json / truss_input.json # 输入数据
readme.md程序说明

环境配置
pip install numpy scipy matplotlib pypardiso
conda install mkl
运行所有算例
命令	对应内容
python  main.py	桁架算例（复用 2.3）
python tridiagonal_test.py	三对角矩阵性能测试
python test_nonpositive.py	非正定矩阵检测
python task2_ill_conditioned_analysis.py	病态方程组误差分析
python poisson.py	Poisson 方程求解 + 云图
python colsol.py	带状存储 LDL^T 演示
关键说明
所有求解器通过 solve_equilibrium(K, rhs, method="ldlt" or "pardiso") 统一调用。
稀疏求解器优先使用 pypardiso（MKL PARDISO），自动回退 scipy。
poisson.py 会生成 poisson_solution_nx_50.png 和 poisson_solution_nx_100.png。


