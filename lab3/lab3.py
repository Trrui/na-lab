import numpy as np


def hilbert_matrix(n):
    return 1 / (np.arange(1, n + 1) + np.arange(0, n)[:, np.newaxis])


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


def print_error(H, b, x):
    r = b - H @ x
    d = x - np.ones(x.shape)
    print("n = {}, |r| = {}, |d| = {}".format(
        H.shape[0], np.linalg.norm(r, ord=np.inf), np.linalg.norm(d, ord=np.inf)))


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


def main():
    n_list = [8, 10, 12, 14]
    for n in n_list:
        solve(n)


if __name__ == "__main__":
    main()
