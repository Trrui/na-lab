import numpy as np
from scipy.special import j0
from matplotlib import pyplot as plt

# parameters: f, range, parameters of f


def fzerotx(f, range):
    a, b = range
    fa = f(a)
    fb = f(b)
    if np.sign(fa) == np.sign(fb):
        print("Error: f(a) and f(b) must have opposite signs")
        return None
    c = a
    fc = fa
    d = b - c
    e = d
    while fb != 0:
        if np.sign(fa) == np.sign(fb):
            a = c
            fa = fc
            d = b - c
            e = d
        if abs(fa) < abs(fb):
            c = b
            b = a
            a = c
            fc = fb
            fb = fa
            fa = fc
        m = 0.5 * (a - b)
        tol = 2.0 * np.finfo(float).eps * max(abs(b), 1.0)
        if abs(m) <= tol or fb == 0.0:
            break
        if abs(e) < tol or abs(fc) <= abs(fb):
            d = m
            e = m
        else:
            s = fb / fc
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * e * q):
                e = d
                d = p / q
            else:
                d = m
                e = m
        c = b
        fc = fb
        if abs(d) > tol:
            b += d
        else:
            b += np.sign(m) * tol
        fb = f(b)
    return b


def main():
    _, ax = plt.subplots()
    x = np.linspace(0, 35, 1000)
    y = j0(x)
    ax.plot(x, y, 'b-')
    ax.axhline(y=0, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('J0(x)')
    ax.set_xlim(0, 35)
    ax.grid(True)

    x_list = [0, 4, 7, 10, 13, 17, 20, 23, 26, 29, 32]
    for x in x_list:
        ax.plot(x, j0(x), 'mo')

    offset = 0.1
    for a, b in zip(x_list, x_list[1:]):
        x = fzerotx(j0, (a, b))
        assert(x is not None)
        ax.plot(x, j0(x), 'ro')
        ax.annotate('{:.2f}'.format(x), xy=(
            x, j0(x)), xytext=(x, j0(x) + offset))
        print('the {}th zero of J0 is {}'.format(x_list.index(b), x))
        offset *= -1

    plt.savefig('lab2-3.png')


if __name__ == '__main__':
    main()
