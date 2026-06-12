"""
colsol.py
活动列/轮廓存储求解器的简化实现：针对带状矩阵（半带宽较小）的 LDL^T 求解。
演示轮廓存储的基本思想：只存储下半三角的轮廓内元素，按列压缩为一维数组。
完整实现复杂，此处提供带状存储版本作为满足“相应循环范围实现”的示例。
"""

import numpy as np

class BandedLDLT:
    """
    带状对称正定矩阵的 LDL^T 分解与求解器。
    存储格式：压缩的带状矩阵，大小为 n × bw，其中 bw 为半带宽（包括对角线）。
    Aband[i, k] 存储 K[i, i - k]  (k = 0,1,...,bw-1)
    """
    def __init__(self, K, bw):
        """
        K: 稠密对称正定矩阵
        bw: 半带宽（包括对角线），即从对角线向左的非零元素个数
        """
        self.n = K.shape[0]
        self.bw = bw                     # 注意：这里是 self.bw，不是 self.bandwidth
        # 初始化带状存储数组，形状 (n, bw)，初始为 0
        self.Aband = np.zeros((self.n, self.bw))

        # 将 K 的下半三角部分存入 Aband
        for i in range(self.n):
            for j in range(max(0, i - self.bw + 1), i + 1):
                # 列偏移 = i - j
                self.Aband[i, i - j] = K[i, j]

    def factor(self):
        """
        带状矩阵的 LDL^T 分解，原地修改 Aband。
        Aband[i, 0] 存储 D[i] (对角元)
        Aband[i, k] (k>0) 存储 L[i, i-k] 乘以 D[i-k]？实际存储的是 L 元素。
        算法参考：LDL^T 分解的带状版本。
        """
        n = self.n
        bw = self.bw
        Ab = self.Aband

        for j in range(n):
            # 计算 D[j]：Ab[j, 0] 初始为 K[j,j]，减去之前列的贡献
            sum_d = 0.0
            # 只考虑 j 所在列的上方半带宽内的元素
            for k in range(max(0, j - bw + 1), j):
                # L[j, k] = Ab[j, j - k]  注意：存储的索引为 (j, j-k)
                # D[k] = Ab[k, 0]
                sum_d += Ab[j, j - k] ** 2 * Ab[k, 0]
            Ab[j, 0] -= sum_d          # 更新后的 D[j]

            if Ab[j, 0] <= 1e-12:
                raise ValueError(f"零主元 at j={j}")

            # 计算第 j 列下方元素 L[i, j] (i > j)
            for i in range(j + 1, min(n, j + bw)):
                sum_l = 0.0
                # 遍历 k 从 max(0, i-bw+1, j-bw+1) 到 j-1
                k_start = max(0, i - bw + 1, j - bw + 1)
                for k in range(k_start, j):
                    # L[i, k] 存储在 Ab[i, i - k]
                    # L[j, k] 存储在 Ab[j, j - k]
                    # D[k] = Ab[k, 0]
                    sum_l += Ab[i, i - k] * Ab[j, j - k] * Ab[k, 0]
                # L[i, j] = (K[i,j] - sum_l) / D[j]
                # 注意 K[i,j] 存储在 Ab[i, i - j] 中（因为 i > j）
                Ab[i, i - j] = (Ab[i, i - j] - sum_l) / Ab[j, 0]

    def solve(self, rhs):
        """
        求解 K a = rhs，使用已分解的 LDL^T 带状矩阵。
        前代：L y = rhs
        对角：D z = y
        回代：L^T a = z
        """
        n = self.n
        bw = self.bw
        Ab = self.Aband

        # 前代：解 L y = rhs
        y = rhs.copy()
        for i in range(n):
            # 第 i 行，j 从 max(0, i-bw+1) 到 i-1
            for j in range(max(0, i - bw + 1), i):
                y[i] -= Ab[i, i - j] * y[j]

        # 对角：z = y / D
        z = y / Ab[:, 0]

        # 回代：解 L^T a = z
        a = z.copy()
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, min(n, i + bw)):
                a[i] -= Ab[j, j - i] * a[j]
        return a


# ----------------------------------------------------------------------
# 示例运行
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 创建一个带状矩阵（例如三对角）
    n = 5
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i-1] = -1.0
            K[i-1, i] = -1.0
    # 半带宽 = 2（包括对角线）
    solver = BandedLDLT(K, bw=2)          # 注意：参数名改为 bw
    solver.factor()
    rhs = np.array([1, 0, 0, 0, 0])
    x = solver.solve(rhs)
    print("带状求解器结果:", x)
    print("对比直接求解:", np.linalg.solve(K, rhs))