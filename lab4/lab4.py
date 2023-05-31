import numpy as np
from functools import reduce
import time
from matplotlib import pyplot as plt


class SparseMatrix:
    class Item:
        def __init__(self, col, value):
            self.col = col
            self.value = value

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.row_start = np.zeros(n + 1, dtype=int)
        self.items = []


def Jacobi(A, b, tol):
    '''
    Jacobi(A, b, tol)

    A: coefficient matrix
    b: right hand side vector
    tol: tolerance

    returns: x, number of iterations
    '''
    n = len(b)
    x = np.zeros_like(b)
    iter_count = 0
    while True:
        last_x = x.copy()
        iter_count += 1
        for i in range(n):
            A_ii = next(
                (item.value for item in A.items[A.row_start[i]:A.row_start[i + 1]] if item.col == i), None)
            if A_ii is None:
                raise ValueError('Diagonal element not found')
            x[i] = reduce(lambda acc, item: (acc - item.value * last_x[item.col])
                          if item.col != i else acc, A.items[A.row_start[i]:A.row_start[i + 1]], b[i]) / A_ii
        if np.linalg.norm(x - last_x, np.inf) <= tol:
            return x, iter_count


def GS(A, b, tol):
    '''
    GS(A, b, tol)

    A: coefficient matrix
    b: right hand side vector
    tol: tolerance

    returns: x, number of iterations
    '''
    n = len(b)
    x = np.zeros_like(b)
    iter_count = 0
    while True:
        last_x = x.copy()
        iter_count += 1
        for i in range(n):
            A_ii = next(
                (item.value for item in A.items[A.row_start[i]:A.row_start[i + 1]] if item.col == i), None)
            if A_ii is None:
                raise ValueError('Diagonal element not found')
            x[i] = reduce(lambda acc, item: (acc - item.value * x[item.col])
                          if item.col != i else acc, A.items[A.row_start[i]:A.row_start[i + 1]], b[i]) / A_ii
        if np.linalg.norm(x - last_x, np.inf) <= tol:
            return x, iter_count


def SOR(A, b, tol, omega=0.95):
    '''
    SOR(A, b, tol, maxiter, omega)

    A: coefficient matrix
    b: right hand side vector
    tol: tolerance
    omega: relaxation parameter

    returns: x, number of iterations
    '''
    n = len(b)
    x = np.zeros_like(b)
    iter_count = 0
    while True:
        last_x = x.copy()
        iter_count += 1
        for i in range(n):
            A_ii = next(
                (item.value for item in A.items[A.row_start[i]:A.row_start[i + 1]] if item.col == i), None)
            if A_ii is None:
                raise ValueError('Diagonal element not found')
            x[i] = (1 - omega) * x[i] + omega * \
                (reduce(lambda acc, item: (acc - item.value * x[item.col])
                        if item.col != i else acc, A.items[A.row_start[i]:A.row_start[i + 1]], b[i])) / A_ii
        if np.linalg.norm(x - last_x, np.inf) <= tol:
            return x, iter_count


def accurate_solution(eps, a, x):
    return (1 - a) / (1 - np.exp(-1 / eps)) * (1 - np.exp(-x / eps)) + a * x


def print_result(eps, a, n, method, iters, max_error, runtime):
    print('{} method: eps = {}, a = {}, n = {}, iters = {}, max error = {}, time = {:.3f} s'.format(
        method, eps, a, n, iters, max_error, runtime))
    with open('lab4.txt', 'a') as f:
        f.write('{} method: eps = {}, a = {}, n = {}, iters = {}, max error = {}, time = {:.3f} s\n'.format(
            method, eps, a, n, iters, max_error, runtime))


def solve(eps, a, n):
    print("n = {}".format(n))
    h = 1 / np.float64(n)
    A = SparseMatrix(n - 1, n - 1)

    # build matrix A
    for i in range(n - 1):
        A.row_start[i] = len(A.items)
        if i > 0:
            A.items.append(SparseMatrix.Item(i - 1, eps - h / 2))
        A.items.append(SparseMatrix.Item(i, -2 * eps))
        if i < n - 2:
            A.items.append(SparseMatrix.Item(i + 1, eps + h / 2))
    A.row_start[n - 1] = len(A.items)

    # build vector b
    b = np.full(n - 1, a * h * h)
    b[-1] -= eps + h / 2

    # solve Ay = b

    start_time = time.time()
    Jacobi_y, iters = Jacobi(A, b, 1e-5)
    end_time = time.time()
    max_error = np.max(
        np.abs(Jacobi_y - accurate_solution(eps, a, np.arange(h, 1, h))))
    print_result(eps, a, n, 'Ja ', iters, max_error, end_time - start_time)

    start_time = time.time()
    GS_y, iters = GS(A, b, 1e-5)
    end_time = time.time()
    max_error = np.max(
        np.abs(GS_y - accurate_solution(eps, a, np.arange(h, 1, h))))
    print_result(eps, a, n, 'GS ', iters, max_error, end_time - start_time)

    start_time = time.time()
    SOR_y, iters = SOR(A, b, 1e-5)
    end_time = time.time()
    max_error = np.max(
        np.abs(SOR_y - accurate_solution(eps, a, np.arange(h, 1, h))))
    print_result(eps, a, n, 'SOR', iters, max_error, end_time - start_time)

    print("-" * 40)

    return Jacobi_y, GS_y, SOR_y


def plot_error(eps, a, n, Jacobi_y, GS_y, SOR_y):
    _, ax = plt.subplots()
    ax.set_title('eps = {}, n = {}'.format(eps, n))
    ax.set_xlabel('x')
    ax.set_ylabel('error')
    h = 1 / np.float64(n)
    x = np.arange(h, 1, h)
    ax.plot(x, np.abs(accurate_solution(eps, a, x) - Jacobi_y),
            label='Jacobi')
    ax.plot(x, np.abs(accurate_solution(eps, a, x) - GS_y),
            label='GS')
    ax.plot(x, np.abs(accurate_solution(eps, a, x) - SOR_y),
            label='SOR')
    ax.legend(loc='upper right')
    plt.savefig('lab4-eps-{}-n-{}.png'.format(eps, n))


def plot_error_same_eps(eps, a, n_list, y_map):
    _, ax = plt.subplots()
    ax.set_title('eps = {}'.format(eps))
    ax.set_xlabel('x')
    ax.set_ylabel('error')
    for n in n_list:
        Jacobi_y, GS_y, SOR_y = y_map[eps][n]
        h = 1 / np.float64(n)
        x = np.arange(h, 1, h)
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - Jacobi_y),
                label='Jacobi(n={})'.format(n))
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - GS_y), linestyle='--',
                label='GS(n={})'.format(n))
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - SOR_y), linestyle='-.',
                label='SOR(n={})'.format(n))

    ax.legend(loc='upper right')
    plt.savefig('lab4-eps-{:.3f}.png'.format(eps))


def plot_error_same_n(n, a, eps_list, y_map):
    _, ax = plt.subplots()
    ax.set_title('n = {}'.format(n))
    ax.set_xlabel('x')
    ax.set_ylabel('error')
    for eps in eps_list:
        Jacobi_y, GS_y, SOR_y = y_map[eps][n]
        h = 1 / np.float64(n)
        x = np.arange(h, 1, h)
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - Jacobi_y),
                label='Jacobi(eps={:.3f})'.format(eps))
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - GS_y), linestyle='--',
                label='GS(eps={:.3f})'.format(eps))
        ax.plot(x, np.abs(accurate_solution(eps, a, x) - SOR_y), linestyle='-.',
                label='SOR(eps={:.3f})'.format(eps))

    ax.legend(loc='upper right')
    plt.savefig('lab4-n-{}.png'.format(n))


def main():
    # eps > 1/2n
    eps_list = [1, 0.1, 0.01, 0.001]
    n_lists = [[100, 500, 1000], [100, 500, 1000],
               [100, 500, 1000], [500, 1000, 1500]]
    y_map = {}
    for eps, n_list in zip(eps_list, n_lists):
        y_map[eps] = {}
        for n in n_list:
            y_map[eps][n] = solve(eps, 0.5, n)

    # plot n = 1000, eps = 1
    plot_error(1, 0.5, 1000, *y_map[1][1000])

    # plot n = 1000, eps = 1, 0.1, 0.01, 0.001
    plot_error_same_n(1000, 0.5, eps_list, y_map)

    # plot error with same eps
    for eps, n_list in zip(eps_list, n_lists):
        plot_error_same_eps(eps, 0.5, n_list, y_map)


if __name__ == '__main__':
    main()
