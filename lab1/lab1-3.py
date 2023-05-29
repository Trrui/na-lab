import numpy as np
from scipy.optimize import fsolve


def single_precision():
    sum = np.float32(0)
    n = 1
    while True:
        _sum = sum
        sum += np.float32(1) / np.float32(n)
        if sum == _sum:
            print('float32 n = {}, sum = {}'.format(n, sum))
            break
        n += 1
    return n, sum


def double_precision(n):
    sum = np.float64(0)
    for i in range(n):
        sum += np.float64(1) / np.float64(i + 1)
    print('float64 n = {}, sum = {}'.format(n, sum))
    return sum


def solve(eps_mach):
    n = fsolve(lambda n: 2 / n - eps_mach *
               (np.log(n) + np.euler_gamma + 0.5 / n), 1)
    return n


def main():
    n, sum_32 = single_precision()
    sum_64 = double_precision(n)
    absolute_error = np.abs(sum_64 - sum_32)
    relative_error = absolute_error / sum_64
    print('absolute error = {:.3f}, relative error = {:.3%}'.format(
        absolute_error, relative_error))
    print('float32 solved n = {}'.format(solve(np.finfo(np.float32).epsneg)))
    print('float64 solved n = {}'.format(solve(np.finfo(np.float64).epsneg)))


if __name__ == '__main__':
    main()
