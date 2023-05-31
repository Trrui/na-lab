import sys
import numpy as np
from scipy.optimize import newton as newton_scipy
from matplotlib import pyplot as plt


def newton(x0, f, f_der, ax, eps=1e-16, max_iter=100):
    x = x0
    converged = False
    x_list = [x]
    step = 0
    for _ in range(max_iter):
        new_x = x - f(x) / f_der(x)
        x_list.append(new_x)
        if np.abs(new_x - x) < eps:
            converged = True
            break
        x = new_x
        step += 1
        print("[newton] step = {}, x = {}".format(step, x), file=sys.stderr)
    x_list = np.array(x_list)
    ax.plot(x_list, f(x_list), 'ko', markersize=8)
    return x, step, converged


def newton_downhill(x0, f, f_der, ax, eps=1e-16, max_iter=100):
    x = x0
    converged = False
    x_list = [x]
    step = 0
    for _ in range(max_iter):
        s = f(x) / f_der(x)
        new_x = x - s
        if np.abs(new_x - x) < eps:
            converged = True
            x_list.append(new_x)
            break
        i = 0
        while np.abs(f(new_x)) >= np.abs(f(x)):
            i += 1
            new_x = x - s * np.power(0.9, i)
        x = new_x
        step += 1
        print("[newton_downhill] step = {}, x = {}, lambda = {}".format(
            step, x, s / np.power(4, i)), file=sys.stderr)
        x_list.append(x)
    x_list = np.array(x_list)
    ax.plot(x_list, f(x_list), 'o')
    return x, step, converged


def solve(f, f_der, x0, ax):
    print("solve f(x) = 0, x0 = {}".format(x0))
    x, step, converged = newton(x0, f, f_der, ax)
    print("newton method: root = {}, steps = {}, converged = {}".format(
        x, step, converged))
    x, step, converged = newton_downhill(x0, f, f_der, ax)
    print("newton downhill method: root = {}, steps = {}, converged = {}".format(
        x, step, converged))
    x, converged = newton_scipy(f, x0, full_output=True)[:2]
    print("scipy newton method: root = {}, converged = {}".format(
        x, converged.converged))
    print("-" * 40)


def main():
    fig, ax = plt.subplots()
    # f(x) = x^3 - 2x + 2
    def f(x): return x**3 - 2 * x + 2
    def f_der(x): return 3 * x**2 - 2
    x0 = np.float64(0)
    ax.plot(np.linspace(-2.5, 2.5, 1000), f(np.linspace(-2.5, 2.5, 1000)), '-')
    ax.axhline(0, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x) = x^3 - 2x + 2')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-4, 4)
    solve(f, f_der, x0, ax)
    fig.savefig('lab2-2-1.png')

    # g(x) = -x^3 + 5x
    fig, ax = plt.subplots()
    def g(x): return -x**3 + 5 * x
    def g_der(x): return -3 * x**2 + 5
    x0 = np.float64(1.35)
    x_range = np.linspace(0, 4, 1000)
    ax.plot(x_range, g(x_range), '-')
    ax.axhline(0, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('g(x) = -x^3 + 5x')
    ax.set_xlim(0, 4)
    ax.set_ylim(-5, 5)
    solve(g, g_der, x0, ax)

    fig.savefig('lab2-2-2.png')


if __name__ == '__main__':
    main()
