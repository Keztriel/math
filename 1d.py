import numpy as np
import math as m

def f(x):
    return x ** 2 + 3 * x -5

def df(x):
    return 2 * x + 3

def alfa(x, eps = 10 ** -7):
    return goldenSectionSearch(x, eps)

def goldenSectionSearch(x, eps):
    a = 0
    b = 1
    k1 = (3 - 5 ** 0.5) / 2
    k2 = (5 ** 0.5 - 1) / 2
    l1 = a + k1 * (b - a)
    l2 = a + k2 * (b - a)
    f1 = l1 * f(x)
    f2 = l2 * f(x)
    while(b - a) / 2 >= eps:
        if f1 < f2:
            b, l2, f2 = l2, l1, f1
            l1 = a + k1 * (b - a)
            f1 = l1 * f(x)
        else:
            a, l1, f1 = l1, l2, f2
            l2 = a + k2 * (b - a)
            f2 = l2 * f(x)
    return (a + b) / 2

def grad_decent(x0, eps):
    i = 0;
    x = x0;
    x_prev = x;
    while (i < 1) or (abs(x - x_prev) >= eps):
        x_prev = x;
        x = x - alfa(x, eps) * df(x);
        i += 1;
    print(i, " ", x)

for i in range(1, 10):
    grad_decent(10 , 10 ** (-i))