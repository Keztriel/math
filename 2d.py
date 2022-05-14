import numpy as np
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D


def f(x, y):
    return x ** 2 * y ** 2

def grad_x(x, y):
    return 2 * x * y ** 2

def grad_y(x, y):
    return 2 * (x ** 2) * y

def goldenratio(x, y, eps):  # Метод золотого сечения
    a = 0
    b = 1
    k1 = (3 - 5 ** 0.5) / 2
    k2 = (5 ** 0.5 - 1) / 2
    l1 = a + k1 * (b - a)
    l2 = a + k2 * (b - a)
    f1 = l1 * f(x, y)
    f2 = l2 * f(x, y)
    while (b - a) / 2 >= eps:
        if f1 < f2:
            b, l2, f2 = l2, l1, f1
            l1 = a + k1 * (b - a)
            f1 = l1 * f(x, y)
        else:
            a, l1, f1 = l1, l2, f2
            l2 = a + k2 * (b - a)
            f2 = l2 * f(x, y)

    return (a + b) / 2


def grad_descent_a(x, y, alpha):
    global f
    steps = [[20, 20, 20]]
    i = 0
    steps.append([x, y, f(x, y)])
    while abs(steps[-1][2] - steps[-2][2]) >= 10 ** (-6):
        i += 1
        lr = alpha
        x -= lr * grad_x(x, y)
        y -= lr * grad_y(x, y)
        steps.append([x, y, f(x, y)])
        # print(i, ") ", [x, y, f(x, y)])
    print(i)
    return np.array(steps)

def grad_descent_b(x, y, alpha):
    global f
    steps = [[20, 20, 20]]
    i = 0
    steps.append([x, y, f(x, y)])
    while abs(steps[-1][2] - steps[-2][2]) >= 10 ** (-6):
        i += 1
        lr = alpha
        x -= lr * grad_x(x, y)
        y -= lr * grad_y(x, y)
        steps.append([x, y, f(x, y)])
        # print(i, ") ", [x, y, f(x, y)])
    print(i)
    return np.array(steps)


def grad_descent_c(x, y, eps):
    global f
    steps = [[20, 20, 20]]
    i = 0
    steps.append([x, y, f(x, y)])
    while abs(steps[-1][2] - steps[-2][2]) >= 10 ** (-6):
        i += 1
        lr = goldenratio(x, y, eps)
        x -= lr * grad_x(x, y)
        y -= lr * grad_y(x, y)
        steps.append([x, y, f(x, y)])
        # print(i, ") ", [x, y, f(x, y)])
    print(i)
    return np.array(steps)

steps_a = grad_descent_a(10, 10, 10 ** -3)
# steps_b
steps_c = grad_descent_c(10, 10, 10 ** -5)
# steps_d
# steps_e
print(steps_a[len(steps_a) - 1])
print(steps_c[len(steps_c) - 1])