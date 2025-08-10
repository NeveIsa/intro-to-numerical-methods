import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    title = (
        """
    # 3a. Applications of Root Finding: **Function Values**
    """
    )

    description = ("Below is a visualization of applying root finding to help find the specific x-value of a cubic function value. We will first create a function g(x), which will be equivalent to f(x) - y, where y will be the y-value of the desired x-value. Then, we will utilize a root finding method in order to find the root of g(x) that is located at. That will be equivalent to finding the x-value of the initial y-value.")

    mo.md(f"{title}\n{description}")
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
        mo.ui.slider(10, 15, step=1, label='Root 3', value=10, show_value=True),
        mo.ui.slider(0, 15, step=1, label='Function Value', value=10, show_value=True)
    ])

    sliderelements
    return (sliderelements,)


@app.cell
def _(d2f, df, f, mo, np, plt, sliderelements, sns):
    x = np.arange(-10,15,0.01)
    y = f(x)
    dy = df(x)
    d2y = d2f(x)
    originalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=np.array(sliderelements.value[0:3]), y=0, color='black', label=f"Roots of f(x)")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Original Function (Cubic)")
    mo.md(f"Below is the cubic function based on the 3 roots you selected. The three roots are located at x = ${sliderelements[0].value}$, ${sliderelements[1].value}$, and ${sliderelements[2].value}$: {mo.as_html(originalplt)}")

    return x, y


@app.cell
def _(mo, np, plt, sliderelements, sns, x, y):
    horizontalY = np.full_like(x, sliderelements.value[3])
    zoomedPlt = sns.lineplot(x=x,y=y, label="f(x)")
    sns.scatterplot(x=np.array(sliderelements.value[0:3]), y=0, color='black', label=f"Roots of f(x)")
    plt.ylim(-50, 50)
    sns.lineplot(x=x,y=horizontalY, color='black', label=f"f(x)=${sliderelements.value[3]}$")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Original Function (Cubic)")
    mo.md(f"After zooming into the graph, we can see where the desired function value intersects f(x). The function x-values that we want to find have a y-value of y = ${sliderelements[3].value}$: {mo.as_html(zoomedPlt)}")
    return


@app.cell
def _(g, mo, plt, sliderelements, sns, x):
    shift = "down"
    sign = "-"
    if (sliderelements.value[3] < 0):
        shift = "up"
        sign = "+"

    shiftedplt = sns.lineplot(x=x,y=g(x), label="g(x) = f(x) " + str(sign) + str(sliderelements.value[3]))
    plt.ylim(-50, 50)
    sns.lineplot(x=x,y=0, color='black', label=f"f(x)=${sliderelements.value[3]}$")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Shifted Function (Cubic)")

    mo.md(f"Knowing that our desired points have a y-value of ${sliderelements.value[3]}$, we can create a function g(x), which will be the result of vertially shifting the f(x) ${shift}$ by ${sliderelements.value[3]}$. This will allow the function values to be on top of the x-axis: {mo.as_html(shiftedplt)}")
    return


@app.cell
def _(df, f, g, mo, newtonraphson, np, plt, sliderelements, sns, x, y):
    _x0 = np.random.rand()*25 - 10
    nrsearch = np.array(newtonraphson(g, df, _x0 ))
    nrplt= sns.lineplot(x=x,y=y)

    roundedNRRoot = np.round(nrsearch[-1], decimals=3)

    alphas0 = np.linspace(0.1, 0.9, len(nrsearch) )
    colors0 = [(1, 0, 1, a) for a in alphas0]

    sns.scatterplot(x=nrsearch, y=f(nrsearch), marker="o", color=colors0, label="root approximations")
    sns.scatterplot(x=[_x0],y=[f(_x0)], color='black', label="x0")
    plt.title("Newton Raphson Method")

    mo.md(f"We can now use root finding methods to determine the x-values where f(x) = ${sliderelements.value[3]}$. First, we will find the first root of g(x) (x=${roundedNRRoot}$) using the Newton Rhapson Method: {mo.as_html(nrplt)}")
    return nrsearch, roundedNRRoot


@app.cell
def _(g, nrsearch):
    #defining h(x)




    def h(x):
        return g(x)/(x-nrsearch[-1])
        # deflated_function = np.poly1d(h_func)
        # return deflated_function(x)
    return (h,)


@app.cell
def _(bisection, h, mo, np, plt, roundedNRRoot, sns, x):
    _x1 = np.random.rand(10)*25 - 10
    _y1= h(_x1)
    x0,x1 = np.random.choice(_x1[_y1<0]),np.random.choice(_x1[_y1>0])

    bsearch = np.array(bisection(h,x0,x1))
    bplt = sns.lineplot(x=x,y=h(x))
    roundedBRoot = np.round(bsearch[-1], decimals=3)

    alphas1 = np.linspace(0.3, 1, len(bsearch) )
    colors1 = [(1, 0, 0, a) for a in alphas1]

    sns.scatterplot(x=bsearch, y=h(bsearch), marker="o", color=colors1, label="root approximations")
    sns.scatterplot(x=np.array([x0,x1]), y=np.array([h(x0),h(x1)]), color='black', label="x0,x1" )
    plt.title("Bisection Method")


    mo.md(f"We will now utilize deflation to ensure we find all of the roots. After deflating the polynomial by dividing it by the first root (x-(${roundedNRRoot}$)), we can then find the second root (x=${roundedBRoot}$) using the bisection method: {mo.as_html(bplt)}")
    return bsearch, roundedBRoot


@app.cell
def _(bsearch, h):
    #define i(x)



    def i(x):
        return h(x)/(x-bsearch[-1])
        # deflated_function = np.poly1d(i_func)
        # return deflated_function(x)


    return (i,)


@app.cell
def _(bsearch, i, mo, np, plt, roundedBRoot, sns, x):
    from scipy.optimize import root_scalar
    rootproperties = root_scalar(i, bracket=[-100, 100])
    finalRoot = np.round(rootproperties.root, decimals=3)
    lplt = sns.lineplot(x=x,y=i(x))

    alphas2= np.linspace(0.3, 1, len(bsearch) )
    colors2= [(1, 0, 0, a) for a in alphas2]

    sns.scatterplot(x=[finalRoot], y=0, color='red', label=f'root' )
    plt.title("Root Finding Method for Linear Equations")

    mo.md(f"Finally, after deflating the polynomial once again by dividing it by (x-(${roundedBRoot}$)), we can then find the third root (x=${finalRoot}$) using the root finding method for linear equations (-b/a): {mo.as_html(lplt)}")

    return (finalRoot,)


@app.cell
def _(
    finalRoot,
    mo,
    np,
    plt,
    roundedBRoot,
    roundedNRRoot,
    sliderelements,
    sns,
    x,
    y,
):
    roots = ([
        roundedNRRoot,
        roundedBRoot,
        finalRoot
    ])

    finalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=np.array(roots[0:]), y=sliderelements.value[3], color='black', label=f"(x, ${sliderelements.value[3]}$)")
    plt.ylim(-50, 50)
    plt.title("All Min/Max Values")
    mo.md(f"To conclude, all of the points where f(x) = ${sliderelements.value[3]}$ are x = ${roundedNRRoot}$, ${roundedBRoot}$, and ${finalRoot}$.: {mo.as_html(finalplt)}")
    return


@app.cell
def _(a, b, c, poly, sliderelements):
    def f(x):
        if poly.value == "cubic": return (x-a)*(x-b)*(x-c)

    def g(x):
        return ((x-a)*(x-b)*(x-c))-sliderelements.value[3]

    def df(x):
        if poly.value == "cubic": return 3*x**2 - 2*x*(a+b+c) + (a*b + b*c + c*a)

    def d2f(x):
        if poly.value == "cubic": return 6*x - 2*(a+b+c)
    return d2f, df, f, g


@app.cell
def _(g):
    #deflation functions


    print(g(10.096))

    return


@app.cell
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
