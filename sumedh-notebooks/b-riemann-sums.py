import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Riemann Sums

    1. **Define the Riemann‐sum approximation**  
       $$R_n(f) = \sum_{i=1}^{n} f(x_i^*)\,\Delta x$$

    2. **Partition the interval and choose sample points**  
       $$\Delta x = \frac{b - a}{n},$$  
       $$x_i^* =
         \begin{cases}
           a + (i-1)\,\Delta x & \text{(left endpoint)}\\
           a + i\,\Delta x     & \text{(right endpoint)}\\
           a + \bigl(i - \tfrac12\bigr)\,\Delta x & \text{(midpoint)}
         \end{cases}$$

    3. **Compute the Riemann sum**  
       $$R_n(f) = \sum_{i=1}^{n} f(x_i^*)\,\Delta x$$

    4. **Result**  
       $$\lim_{n\to\infty} R_n(f) = \int_{a}^{b} f(x)\,\mathrm{d}x$$
    """
    )
    return


@app.cell
def _(mo):
    poly = mo.ui.radio(options=["cubic"], value="cubic")
    poly

    pickfn = mo.md(
        '''
        **Polynomial Type:**   {poly}
        '''
    ).batch(poly=poly)
    pickfn
    return (poly,)


@app.cell
def _(mo, poly, sliderelements):
    a,b = [sliderelements.elements[_].value for _ in range(2)]
    c = 0

    _fntext = "Polynomial: f(x) = (x-a)(x-b)"
    if poly.value == "cubic":
        _fntext += "(x-c)"
        c = sliderelements.elements[2].value

    _dfntext = "f'(x) = 2x - (a + b)"
    if poly.value == "cubic":
        _dfntext = "Derivative: f'(x) = 3x^2 - 2x(a+b+c) + (ab + bc + ca)"

    _d2fntext = "f''(x) = 2x - (a + b)"
    if poly.value == "cubic":
        _d2fntext = "2nd&nbsp;Derivative: f''(x) = 6x - 2(a+b+c)"


    mo.md(f"## ${_fntext}\\\\{_dfntext}\\\\{_d2fntext}$"
         )
    return a, b, c


@app.cell
def _(mo):
    sliderelements = mo.ui.array([
        mo.ui.slider(-10, -5, step=1, label='Root 1', value=-5, show_value=True),
        mo.ui.slider(0, 5, step=1, label='Root 2', value=4, show_value=True),
        mo.ui.slider(10, 15, step=1, label='Root 3', value=10, show_value=True)
    ])
    sliderelements
    return (sliderelements,)


@app.cell
def _(d2f, df, f, mo, np, plt, sns):
    x = np.arange(-10,15,0.01)
    y = f(x)
    dy = df(x)
    d2y = d2f(x)
    originalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    plt.title("Original Function (Cubic)")
    mo.md(f"Below is the cubic function based on the 3 roots you selected.{mo.as_html(originalplt)}")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###Computing the Left Riemann Sum 

    1. **Compute the partition width**  
       $$\Delta x = \frac{b - a}{n}$$

    2. **Choose the left‐endpoint sample points**  
       $$x_i = a + (i - 1)\,\Delta x,\quad i = 1,2,\dots,n$$

    3. **Evaluate the function at each left endpoint**  
       $$f_i = f(x_i)$$

    4. **Form the left Riemann sum**  
       $$R_n^\text{left} = \sum_{i=1}^{n} f_i\,\Delta x$$

    5. **In code**  
       ```python
       left_areas = [
           riemann_sum(f, a, b, n, method='left')[0]
           for n in range(1, max_n+1)
       ]
        ```
    """
    )
    return


@app.cell
def _(f, mo, np, plt, riemannSum):
    n = 25
    area, x_pts, dx = riemannSum(f, -10, 15, n, method='left')

    plt.figure(figsize=(8,5))
    plt.bar(
        x_pts,           # left edges
        f(x_pts),        # heights
        width=dx,
        align='edge',
        edgecolor='blue',
        alpha=0.4
    )
    xplot = np.linspace(-10, 15, 1000)
    leftPlot = plt.plot(xplot, f(xplot), '-', marker='.', markersize=4, color='black')
    plt.title(f"Left Riemann Sum (n={n})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    mo.as_html(leftPlot)
    return n, xplot


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Computing the Midpoint Riemann Sum

    1. **Compute the partition width**  
       $$\Delta x = \frac{b - a}{n}$$

    2. **Choose the midpoint sample points**  
       $$x_i = a + \Bigl(i - \tfrac12\Bigr)\,\Delta x,\quad i = 1,2,\dots,n$$

    3. **Evaluate the function at each midpoint**  
       $$f_i = f(x_i)$$

    4. **Form the midpoint Riemann sum**  
       $$R_n^\text{mid} = \sum_{i=1}^{n} f_i\,\Delta x$$

    5. **In code**  
       ```python
       midpoint_areas = [
           riemann_sum(f, a, b, n, method='midpoint')[0]
           for n in range(1, max_n+1)
       ]
        ```
    """
    )
    return


@app.cell
def _(f, mo, n, plt, riemannSum, xplot):
    midpointArea, midpointX, midpointdx = riemannSum(f, -10, 15, n, method='midpoint')

    plt.figure(figsize=(8,5))

    # draw bars centered on the midpoint samples
    plt.bar(
        midpointX,
        f(midpointX),
        width=midpointdx,
        align='center',        # ← center the bars on midpointX
        edgecolor='blue',
        alpha=0.4
    )

    # # plot the true curve
    # xplot = np.linspace(-10, 15, 1000)           # redefine xplot
    midpointPlot = plt.plot(xplot, f(xplot), '-', marker='.', markersize=4, color='black')
    plt.title(f"Midpoint Riemann Sum (n={n})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    mo.as_html(midpointPlot)

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Computing the Right Riemann Sum

    1. **Compute the partition width**  
       $$\Delta x = \frac{b - a}{n}$$

    2. **Choose the right‐endpoint sample points**  
       $$x_i = a + i\,\Delta x,\quad i = 1,2,\dots,n$$

    3. **Evaluate the function at each right endpoint**  
       $$f_i = f(x_i)$$

    4. **Form the right Riemann sum**  
       $$R_n^\text{right} = \sum_{i=1}^{n} f_i\,\Delta x$$

    5. **In code**  
       ```python
       right_areas = [
           riemann_sum(f, a, b, n, method='right')[0]
           for n in range(1, max_n+1)
       ]
    ```
    """
    )
    return


@app.cell
def _(f, mo, n, plt, riemannSum, xplot):
    rightArea, rightX, rightdx = riemannSum(f, -10, 15, n, method='right')

    plt.figure(figsize=(8,5))
    plt.bar(
        rightX - rightdx,
        f(rightX),
        width=rightdx,
        align='edge',
        edgecolor='green',
        alpha=0.4
    )
    rightPlot = plt.plot(xplot, f(xplot), '-', marker='.', markersize=4, color='black')
    plt.title(f"Right Riemann Sum (n={n})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    mo.as_html(rightPlot)

    return


@app.cell
def _(a, b, c, poly):
    def f(x):
        if poly.value == "cubic": return (x-a)*(x-b)*(x-c)

    def df(x):
        if poly.value == "cubic": return 3*x**2 - 2*x*(a+b+c) + (a*b + b*c + c*a)

    def d2f(x):
        if poly.value == "cubic": return 6*x - 2*(a+b+c)
    return d2f, df, f


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import sys,os
    sys.path.append(os.path.dirname(__name__))
    from numerical.areas import riemannSum

    import seaborn as sns
    sns.set_style("whitegrid")

    import matplotlib.pyplot as plt
    return mo, np, plt, riemannSum, sns


if __name__ == "__main__":
    app.run()
