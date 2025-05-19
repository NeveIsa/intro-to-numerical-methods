import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    poly = mo.ui.dropdown(options=["quadratic", "cubic"], value="cubic", full_width=False)
    poly
    return (poly,)


@app.cell(hide_code=True)
def _(mo, poly, sliders):
    a,b = [sliders.elements[_].value for _ in range(2)]
    c = 0

    _fntext = "f(x) = (x-a)(x-b)"
    if poly.value == "cubic":
        _fntext += "(x-c)"
        c = sliders.elements[2].value


    _dfntext = "f'(x) = 2x - (a + b)"
    if poly.value == "cubic":
        _dfntext = "f'(x) = 3x^2 - 2x(a+b+c) + (ab + bc + ca)"



    mo.md(f"## ${_fntext}\\\\{_dfntext}$"
         )
    return a, b, c


@app.cell(hide_code=True)
def _(df, f, np, sns):
    x = np.arange(-10,15,0.01)
    y = f(x)
    dy = df(x)
    sns.lineplot(x=x,y=y, label="$f(x)$")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    return x, y


@app.cell(hide_code=True)
def _(mo, poly):
    sliderelements=[
        mo.ui.slider(-10,-5,step=1,label='a', value=-5, show_value=True),
        mo.ui.slider(0,5,step=1,label='b', value=4, show_value=True),
    ]

    if poly.value=="cubic": sliderelements.append(mo.ui.slider(10,15,step=1,label='c',value=10, show_value=True))

    sliders = mo.ui.array(sliderelements)
    sliders
    return (sliders,)


@app.cell(hide_code=True)
def _(bisection, f, np, plt, sns, x, y):
    initx = np.random.rand(10)*25 - 10
    initf=f(initx)
    x0,x1 = np.random.choice(initx[initf<0]),np.random.choice(initx[initf>0])

    bsearch = np.array(bisection(f,x0,x1))
    sns.lineplot(x=x,y=y)

    alphas=np.linspace(0.3, 1, len(bsearch) )
    colors = [(1, 0, 0, a) for a in alphas]

    sns.scatterplot(x=bsearch, y=f(bsearch), marker="o", color=colors, label="root approximations")
    sns.scatterplot( x=np.array([x0,x1]), y=np.array([f(x0),f(x1)]), color='black', label="x0,x1" )
    plt.title("Bisection Method")
    return


@app.cell(hide_code=True)
def _(df, f, newtonraphson, np, plt, sns, x, y):
    _x0 = np.random.rand()*25 - 10
    nrsearch = np.array(newtonraphson(f, df, _x0 ))
    sns.lineplot(x=x,y=y)

    alphas2 = np.linspace(0.1, 0.9, len(nrsearch) )
    colors2 = [(1, 0, 1, a) for a in alphas2]

    sns.scatterplot(x=nrsearch, y=f(nrsearch), marker="o", color=colors2, label="root approximations")
    sns.scatterplot(x=[_x0],y=[f(_x0)], color='black', label="x0")
    plt.title("Newtonraphson Method")
    return


@app.cell(hide_code=True)
def _(a, b, c, poly):
    def f(x):
        if poly.value == "cubic": return (x-a)*(x-b)*(x-c)
        elif poly.value == "quadratic": return (x-a)*(x-b)

    def df(x):
        if poly.value == "quadratic": return 2*x - (a+b)
        elif poly.value == "cubic": return 3*x**2 - 2*x*(a+b+c) + (a*b + b*c + c*a)
    return df, f


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    import sys,os
    sys.path.append(os.path.dirname(__name__))
    from numerical.roots import newtonraphson,bisection 

    import seaborn as sns
    sns.set_style("whitegrid")

    import matplotlib.pyplot as plt
    return bisection, mo, newtonraphson, np, plt, sns


if __name__ == "__main__":
    app.run()
