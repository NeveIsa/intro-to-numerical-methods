import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Monte Carlo Integration

    1. **Define the Monte Carlo estimator**  
       $$M_n(f) = \frac{b - a}{n}\sum_{i=1}^{n} f(x_i)$$  
       where each \(x_i\) is sampled uniformly on \([a,b]\).

    2. **Draw random sample points**  
       $$x_i \sim \mathrm{Uniform}(a,b),\quad i=1,2,\dots,n$$

    3. **Compute the estimator**  
       $$\hat I_n = (b - a)\,\frac{1}{n}\sum_{i=1}^{n} f(x_i)$$

    4. **Result**  
       $$\lim_{n\to\infty}\hat I_n = \int_{a}^{b} f(x)\,\mathrm{d}x$$  
       with standard error decaying like \(\mathcal{O}(n^{-1/2})\).

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
def _(error, estimate, mo, n, trueVal):
    mo.md(f"""
    **Monte Carlo Estimate (n={n})** = {estimate:.6f}
    \n**True integral**             = {trueVal:.6f}
    \n**Error**                     = {error:.6f}
    """)

    return


@app.cell
def _(a, b, f, mo, np, plt, trapezoidalSum):
    n = 5000
    x_plot = np.linspace(a, b, 1000)
    y_max = f(x_plot).max()

    x_rand = np.random.uniform(a, b, n)
    y_rand = np.random.uniform(0, y_max, n)

    inside = y_rand <= f(x_rand)

    estimate = inside.mean() * (b - a) * y_max
    trueVal = trapezoidalSum(f, a, b)
    error = estimate - trueVal



    plt.figure(figsize=(8,5))
    plt.fill_between(x_plot, f(x_plot), alpha=0.2, label='Area under f(x)')
    monteCarloPlt = plt.scatter(x_rand[inside], y_rand[inside], s=10,  color='red', alpha=0.3, label='under curve')
    plt.title("Monte Carlo Area Approximation Method")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    mo.as_html(monteCarloPlt)

    return error, estimate, n, trueVal


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
    from numerical.areas import monteCarlo, trapezoidalSum

    import seaborn as sns
    sns.set_style("whitegrid")

    import matplotlib.pyplot as plt
    return mo, np, plt, sns, trapezoidalSum


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
