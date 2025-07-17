import numpy as np

def riemannSum(f, a, b, n, method):
    dx = (b - a) / n
    if method == 'left':
        x = np.linspace(a, b - dx, n)
    elif method == 'right':
        x = np.linspace(a + dx, b, n)
    elif method == 'midpoint':
        x = np.linspace(a + dx/2, b - dx/2, n)
    else:
        raise ValueError("method must be 'left', 'right' or 'midpoint'")
    area = np.sum(f(x) * dx)
    return area, x, dx
