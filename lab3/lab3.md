# 实验三

## 第三章上机题6

### 解题思路

实现改进的 Cholesky 分解算法（不需要开方），并根据 Cholesky 分解的性质实现上三角矩阵、下三角矩阵的线性方程组求解，最后求解 $\mathbf{H}_{n}\mathbf{x} = \mathbf{b}$ 。

### 实验结果

#### 关键代码

+ 基本的 Cholesky 分解算法

  + 由于矩阵规模较小，且原矩阵后续有用处，故没有实现原地计算
  + 由于需要开方运算，浮点运算的舍入误差可能会导致实际需要开方的数小于 0
  + 当 n=14 时，直接用该方法求解 Hilbert 矩阵的 Cholesky 分解就会出现开方数小于 0 的问题

  ```python
  def cholesky_decomposition(H):
      n = H.shape[0]
      L = np.copy(H)
      for i in range(n):
          for j in range(i):
              L[i, i] -= L[i, j] ** 2
          L[i, i] = np.sqrt(L[i, i])
          for j in range(i + 1, n):
              for k in range(i):
                  L[j, i] -= L[j, k] * L[i, k]
              L[j, i] /= L[i, i]
      for i in range(n):
          for j in range(i + 1, n):
              L[i, j] = 0
      return L
  ```

+ 改进的平方根法

  + 返回 $\mathbf{DL}$ ，其中 $\mathbf{H = LDL^T}$

  ```python
  def cholesky_decomposition_optimized(H):
      n = H.shape[0]
      DL = np.copy(H)
      for i in range(n):
          for j in range(i):
              DL[i, i] -= DL[i, j] ** 2 / DL[j, j]
          for j in range(i + 1, n):
              for k in range(i):
                  DL[j, i] -= DL[j, k] * DL[i, k] / DL[k, k]
      for i in range(n):
          for j in range(i + 1, n):
              DL[i, j] = 0
      return DL
  ```

+ 求解上三角矩阵、下三角矩阵线性方程组的解

  + 由于矩阵规模较小，且原矩阵后续有用处，故没有实现原地计算

  ```python
  def solve_lower_triangular(L, b):
      n = L.shape[0]
      x = np.zeros(n)
      for i in range(n):
          x[i] = b[i]
          for j in range(i):
              x[i] -= L[i, j] * x[j]
          x[i] /= L[i, i]
      return x
  
  def solve_upper_triangular(U, b):
      n = U.shape[0]
      x = np.zeros(n)
      for i in range(n - 1, -1, -1):
          x[i] = b[i]
          for j in range(i + 1, n):
              x[i] -= U[i, j] * x[j]
          x[i] /= U[i, i]
      return x
  ```

+ 求解 Cholesky 矩阵线性方程组的解

  ```python
  def solve_cholesky(H, b):
      """
      b = DL @ L^T @ x
      b = DL @ y
      y = L^T @ x
      """
      # L = cholesky_decomposition(H)
      # y = solve_lower_triangular(L, b)
      # x = solve_upper_triangular(L.T, y)
  
      DL = cholesky_decomposition_optimized(H)
      y = solve_lower_triangular(DL, b)
      x = solve_upper_triangular((DL / np.diag(DL)).T, y)
      return x
  ```

+ 求解扰动前、扰动后的解

  ```python
  def solve(n):
      H = hilbert_matrix(n)
      b = H @ np.ones(n)
      print("cond(H) = {}".format(np.linalg.cond(H, p=np.inf)))
      print("Accurate  b: ", end="")
      print_error(H, b, solve_cholesky(H, b))
  
      # Perturb b
      b = H @ np.ones(n)
      b += 1e-7 * np.random.randn(n) * np.linalg.norm(b, ord=np.inf)
      print("Perturbed b: ", end="")
      print_error(H, b, solve_cholesky(H, b))
      print("-" * 40)
  ```

#### 运行输出

|r|, |d| 分别表示残差、误差的无穷范数， cond(H) 为矩阵 $\mathbf{H}$ 的条件数。

```bash
cond(H) = 33872792385.924484
Accurate  b: n = 8, |r| = 2.220446049250313e-16, |d| = 3.0868604561362645e-08
Perturbed b: n = 8, |r| = 2.1094237467877974e-15, |d| = 91.35630567719005
----------------------------------------
cond(H) = 35356847610517.12
Accurate  b: n = 10, |r| = 2.220446049250313e-16, |d| = 1.4621392992353321e-05
Perturbed b: n = 10, |r| = 2.4354296357387284e-11, |d| = 826404.4584978868
----------------------------------------
cond(H) = 4.255399301891292e+16
Accurate  b: n = 12, |r| = 4.440892098500626e-16, |d| = 0.5487188691614726
Perturbed b: n = 12, |r| = 3.03663567624568e-08, |d| = 1004248548.9005389
----------------------------------------
cond(H) = 1.1489640282002911e+18
Accurate  b: n = 14, |r| = 4.440892098500626e-16, |d| = 7.333871933355823
Perturbed b: n = 14, |r| = 1.1692359636050753e-07, |d| = 6486920157.726316
----------------------------------------
```

### 实验结论

+ $n=10$ 时，$|r|_{\infty} = \epsilon_\text{mach},\ |d|_{\infty} = 3.08686 \times 10^{-8}$
+ 对 $\mathbf{b}$ 施加扰动后，残差有数量级上的变化，但依旧很小，而误差数量级上的提升远大于残差，这说明该问题很敏感，是病态的；通过计算 $\mathbf{H}$ 的条件数也能验证这一点
+ 通过观察 $n$ 取不同值时残差、误差的变化情况，发现 $n$ 越大， $\mathbf{H}_n\mathbf{x}=\mathbf{b}$ 问题的敏感性越高，$\mathbf{H}_n$ 的条件数也越大
+ $n$ 越大，不施加扰动时，解的残差和误差也越大，其中 $n=12$ 时，浮点运算引入的舍入误差就已经不可接受
+ 实验中还发现，改进的平方根法在求解问题时，比普通的 Cholesky 分解法更精确，推测是开方引入了更大的舍入误差
