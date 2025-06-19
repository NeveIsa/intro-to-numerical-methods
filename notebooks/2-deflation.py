import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# 2. Deflation: **Third Order Polynomials**""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


    mo.md(f"## ${_fntext}\\\\{_dfntext}$"
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


@app.cell(hide_code=True)
def _(df, f, mo, np, plt, sns):
    x = np.arange(-10,15,0.01)
    y = f(x)
    dy = df(x)
    originalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Cubic Function")
    mo.md(f"Below is the cubic function based on the 3 roots you selected: {mo.as_html(originalplt)}")
    return x, y


@app.cell(hide_code=True)
def _(a, b, c, poly):
    def f(x):
        if poly.value == "cubic": return (x-a)*(x-b)*(x-c)

    def df(x):
        if poly.value == "cubic": return 3*x**2 - 2*x*(a+b+c) + (a*b + b*c + c*a)
    return df, f


@app.cell(hide_code=True)
def _(df, f, mo, newtonraphson, np, plt, sns, x, y):
    _x0 = np.random.rand()*25 - 10
    nrsearch = np.array(newtonraphson(f, df, _x0 ))
    nrplt= sns.lineplot(x=x,y=y)

    roundedNRRoot = np.round(nrsearch[-1], decimals=2)

    alphas0 = np.linspace(0.1, 0.9, len(nrsearch) )
    colors0 = [(1, 0, 1, a) for a in alphas0]

    sns.scatterplot(x=nrsearch, y=f(nrsearch), marker="o", color=colors0, label="root approximations")
    sns.scatterplot(x=[_x0],y=[f(_x0)], color='black', label="x0")
    plt.title("Newton Raphson Method")

    mo.md(f"First, we will find the first root (x=${roundedNRRoot}$) using the Newton Rhapson Method: {mo.as_html(nrplt)}")
    return nrsearch, roundedNRRoot


@app.cell(hide_code=True)
def _(a, b, c, np):
    #deflation functions

    def deflateCubic(root):
        divisor = np.array([1, -root])
        coefficients = [1, -a - b - c, a*b + a*c + b*c, -a*b*c]
        deflated_poly, remainder = np.polydiv(coefficients, divisor)
        deflated_poly = np.round(deflated_poly, decimals=10)
        remainder = np.round(remainder, decimals=10)
        return deflated_poly, remainder

    def deflateQuad(root):
        divisor = np.array([1, -root])
        coefficients = [1, -(b+c), b*c]
        deflated_poly, remainder = np.polydiv(coefficients, divisor)
        deflated_poly = np.round(deflated_poly, decimals=10)
        remainder = np.round(remainder, decimals=10)
        return deflated_poly, remainder
    return deflateCubic, deflateQuad


@app.cell(hide_code=True)
def _(deflateCubic, np, nrsearch):
    #defining g(x)

    g_func= deflateCubic(nrsearch[-1])[0]

    def g(x):
        deflated_function = np.poly1d(g_func)
        return deflated_function(x)
    return (g,)


@app.cell
def _(bisection, g, mo, np, plt, roundedNRRoot, sns, x):
    _x1 = np.random.rand(10)*25 - 10
    _y1= g(_x1)
    x0,x1 = np.random.choice(_x1[_y1<0]),np.random.choice(_x1[_y1>0])

    bsearch = np.array(bisection(g,x0,x1))
    bplt = sns.lineplot(x=x,y=g(x))
    roundedBRoot = np.round(bsearch[-1], decimals=2)

    alphas1 = np.linspace(0.3, 1, len(bsearch) )
    colors1 = [(1, 0, 0, a) for a in alphas1]

    sns.scatterplot(x=bsearch, y=g(bsearch), marker="o", color=colors1, label="root approximations")
    sns.scatterplot(x=np.array([x0,x1]), y=np.array([g(x0),g(x1)]), color='black', label="x0,x1" )
    plt.title("Bisection Method")


    mo.md(f"Then, after deflating the polynomial by dividing it by the first root (x-(${roundedNRRoot}$)), we can then find the second root (x=${roundedBRoot}$) using the bisection method: {mo.as_html(bplt)}")
    return bsearch, roundedBRoot


@app.cell(hide_code=True)
def _(bsearch, deflateQuad, np):
    #define h(x)

    h_func = deflateQuad(bsearch[-1])[0]

    def h(x):
        deflated_function = np.poly1d(h_func)
        return deflated_function(x)
    return h, h_func


@app.cell
def _(bsearch, h, h_func, mo, np, plt, roundedBRoot, sns, x):
    finalRoot = (-(h_func[1]/(h_func[0])))
    lplt = sns.lineplot(x=x,y=h(x))

    alphas2= np.linspace(0.3, 1, len(bsearch) )
    colors2= [(1, 0, 0, a) for a in alphas2]

    sns.scatterplot(x=[finalRoot], y=[0], color='red', label=f'root' )
    plt.title("Root Finding Method for Linear Equations")

    mo.md(f"Finally, after deflating the polynomial once again by dividing it by (x-(${roundedBRoot}$)), we can then find the third root (x=${np.round(finalRoot)}$) using the root finding method for linear equations (-b/a): {mo.as_html(lplt)}")

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
