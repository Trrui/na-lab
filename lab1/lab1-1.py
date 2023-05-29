import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return np.sin(x, dtype=np.float64)


def f_der(x):
    return np.cos(x, dtype=np.float64)


def main():
    _, ax = plt.subplots()
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('step h')
    ax.set_ylabel('error')

    # plot real error
    step_list = np.logspace(-16, 0, 33, dtype=np.float64)
    eps_list = []
    for step in step_list:
        x = 1
        approx_der = (f(x + step) - f(x)) / step
        real_der = f_der(x)
        eps = np.abs(approx_der - real_der) / np.abs(real_der)
        eps_list.append(eps)
    ax.plot(step_list, eps_list, 'k-', label='real error')

    # plot rounding error, truncation error and total error
    x_range = np.logspace(-16, 0, 1000, dtype=np.float64)
    M = np.float64(1)
    eps_machine = np.power(10, -16, dtype=np.float64)
    ax.plot(x_range, 2 * eps_machine / x_range, 'r--', label='rounding error')
    ax.plot(x_range, M * x_range / 2, 'g--', label='truncation error')
    ax.plot(x_range, 2 * eps_machine / x_range + M *
            x_range / 2, 'b--', label='total error')

    ax.legend()
    ax.set_xticks(np.logspace(-16, 0, 9))
    ax.set_yticks(np.logspace(-17, 1, 10))
    ax.set_xlim(1e-16, 1)
    ax.set_ylim(1e-17, 10)

    # save figure
    plt.savefig('lab1-1.png', dpi=300)


if __name__ == '__main__':
    main()
