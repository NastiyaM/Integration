import math
import numpy as np
from sympy import *

def weight(degree, a, b, alpha=0., beta=0.):
    """
    integrate (x**degree / (x-a) ** alpha / (b-x) ** beta) from a to b
    """
    x = symbols('x')
    assert alpha * beta == 0,\
        f'at least one of alpha ({alpha}) or beta ({beta}) should be 0'

    if alpha == 0 and beta != 0:
        if degree == 0:
            return float(-((b-x)**(1-beta)/(beta-1)).subs(x,a))
        else:
            return float(-((x**degree*(b-x)**(1-beta))/(beta-1)).subs(x,a)) - float((degree/(beta-1))* weight(degree - 1, a, b,beta = beta-1))
    if alpha != 0 and beta == 0:
        if degree == 0:
            return float(((x-a)**(1-alpha)/(1-alpha)).subs(x,b))
        else:
            return float((x**degree/((1-alpha)*(x-a)**(alpha-1))).subs(x,b)) - float((degree/(1-alpha))* weight(degree - 1, a, b,alpha = alpha -1))
    k = degree + 1
    return b ** k / k - a ** k / k



def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """
    return -math.log(abs((s2 - s1) / (s1 - s0))) / math.log(L)


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to weight
    """
    d = {}
    for k, v in kwargs.items():
        d[k] = v
    U = []
    S = 0
    for i in range(0,len(xs)):
        U.append(weight(i, x0, x1,**d))
    X = [[0] * len(xs) for i in range(len(xs))]
    for i in range(0,len(xs)):
        for j in range(0,len(xs)):
            X[i][j] = xs[j]**i
    A = np.linalg.solve(X,U)
    for i in range(0,len(xs)):
        S += A[i] * func(xs[i])
    return S


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    d = {}
    U = []
    S = 0
    for k, v in kwargs.items():
        d[k] = v
    for i in range(0, 2 * n):
        U.append(weight(i, x0, x1, **d))
    A = [[0] * n for i in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = U[j + i]
    y = [0] * n
    for i in range(0, n):
        y[i] = -U[n + i]
    X = np.linalg.solve(A, y)
    l = [0] * (n + 1)
    l[0] = 1
    for i in range(1, n + 1):
        l[i] = X[n - i]
    X_ = np.roots(l)
    X__ = [[0] * n for i in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            X__[i][j] = X_[j] ** i
    U_ = [0] * n
    for i in range(0, n):
        U_[i] = U[i]
    A_ = np.linalg.solve(X__, U_)
    for i in range(n):
        S += A_[i] * func(X_[i])
    return S

def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    d = {}
    for k, v in kwargs.items():
        d[k] = v
    S = 0
    h = (x1-x0)/ n_intervals
    d = {}
    for i in range(0,n_intervals):
        S += quad(func, x0, x0 + h, [x0, (2 * x0 + h) / 2, x0 + h])
        x0 += h
    return(S)

def integrate(func, x0, x1, tol):
    L = 2
    m = 4
    x0_ = x0
    h = abs(x1 - x0)
    count = 1
    error = 1
    iteration = 0
    S1 = quad(func, x0, x1, [x0, (x0 + x1) / 2, x1])
    S2 = 0
    while abs(error) >= tol:
        S = 0
        h = h / 2
        count *= 2
        for i in range(0, count):
            S += quad(func, x0, x0 + h, [x0, (2 * x0 + h) / 2, x0 + h])
            x0 += h
        if iteration > 0:
            m = aitken(S2, S1, S, L)
            # print('Скорость сходимости ', m ,S)
        if iteration == 1:
            r1 = runge(S, S1, m, L)[0]
            m1 = aitken(S2, S1, S, L)
        error = runge(S, S1, m, L)[0]
        S2 = S1
        S1 = S
        iteration += 1
        x0 = x0_
    #print(S, error, "ИКФ")
    #h = abs(x1 - x0_)
    #S_ = 0
    #hopt = 0.95 * (h / 2) * ((tol / abs(r1)) ** (1 / m1))
    #count = round(h / hopt)
    #hopt = h / count
    #for i in range(0, int(count)):
        #S_ += quad(func, float(x0_), float(x0_ + hopt), [float(x0_), float((2 * x0_ + hopt) / 2), float(x0_ + hopt)])
        #x0_ += hopt
    #error_ = runge(S_, S1, m, L)[1]
    #print(S_, error_, 'ИКФ,hopt')
    return S, error
#pytest -v -s test_integration.py pytest -v -s chm/S3T1_integration/py/test_integration.py
