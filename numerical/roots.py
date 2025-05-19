import numpy as np

def bisection(fn,a,b, xtol=1e-6):
    if fn(a)*fn(b) > 0:
        return []
    
    visited = []
    while abs(a-b) > xtol:
        mid = (a+b)/2
        visited.append(mid)

        if fn(a)*fn(mid) < 0:
            b = mid
            continue
        else:
            a = mid
            continue

    return visited


def newtonraphson(fn, dfn, x0, ytol=1e-6):
    y0 = fn(x0)
    m = dfn(x0)
    
    # y - y0 = m * (x - x0)
    # x = x0 + (y - y0)/m
    # setting y = 0, x = x0 - y0/m
    visited = [x0]
    while abs(y0) > ytol:
        #update
        x0 = x0 - y0/m
        
        visited.append(x0)
        
        # recalc intermediates
        y0 = fn(x0)
        m = dfn(x0)
    
    return visited


