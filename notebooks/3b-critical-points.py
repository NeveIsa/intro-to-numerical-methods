import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    title = (
        """
    # 3b. Applications of Root Finding: **Critical Points**
    """
    )

    description = ("Below is a visualization of applying root finding to help find the critical points of a cubic function. We will first find the roots of the derivative of the cubic function, using deflation to ensure we find both roots. Then, after inputting these roots into the original cubic function, as well as the second derivative, we can determine the critical point, and whether it is a minima or maxima located at that point.")

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
        mo.ui.slider(10, 15, step=1, label='Root 3', value=10, show_value=True)
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
    sns.scatterplot(x=np.array(sliderelements.value), y=0, color='black', label=f"Roots")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Original Function (Cubic)")
    mo.md(f"Below is the cubic function based on the 3 roots you selected. The three roots are located at x = ${sliderelements[0].value}$, ${sliderelements[1].value}$, and ${sliderelements[2].value}$: {mo.as_html(originalplt)}")

    return dy, x, y


@app.cell
def _(dy, mo, plt, sns, x):
    derivplt = sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Derivative Function (Quadratic)")
    mo.md(f"Below is the derivative function. Finding the roots of the derivative function will allow us to determine the minima and maxima of the original function: {mo.as_html(derivplt)}")
    return


@app.cell
def _(d2f, df, dy, f, mo, newtonraphson, np, plt, sns, x):
    _x0 = np.random.rand()*25 - 10
    nrsearch = np.array(newtonraphson(df, d2f, _x0 ))
    nrplt= sns.lineplot(x=x,y=dy, label="$f'(x)$")

    roundedNRRoot = np.round(nrsearch[-1], decimals=2)

    alphas0 = np.linspace(0.1, 0.9, len(nrsearch) )
    colors0 = [(1, 0, 1, a) for a in alphas0]

    sns.scatterplot(x=nrsearch, y=df(nrsearch), marker="o", color=colors0, label="root approximations")
    sns.scatterplot(x=[_x0],y=[df(_x0)], color='black', label="x0")
    plt.title("Newton Raphson Method")

    val1 = np.round(f(roundedNRRoot), decimals=2)

    if (d2f(roundedNRRoot) > 0):
        minmax1 = "minima"
        sign1 = "positive"
    else:
        minmax1 = "maxima"
        sign1 = "negative"


    mo.md(f"First, we will find the first root of f'(x) (x=${roundedNRRoot}$) using the Newton Rhapson Method: {mo.as_html(nrplt)}")

    return minmax1, nrsearch, roundedNRRoot, sign1, val1


@app.cell
def _(minmax1, mo, plt, roundedNRRoot, sign1, sns, val1, x, y):
    firstrtplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=[roundedNRRoot], y=[val1], color='black', label=f"${minmax1}$")
    plt.title("First Min/Max Value")
    mo.md(f"Since the function value of x = ${roundedNRRoot}$ is located at f(x) = ${val1}$, and the second derivative at that point is ${sign1}$, there is an absolute ${minmax1}$ located at that point. {mo.as_html(firstrtplt)}")
    return


@app.cell
def _(d2f, f, g, g_func, mo, np, plt, roundedNRRoot, sns, x):
    finalRoot = np.round(-(g_func[1]/(g_func[0])), decimals=2)
    lplt = sns.lineplot(x=x,y=g(x), label="$g(x)$")
    val2 = np.round(f(finalRoot), decimals=2)

    if (d2f(finalRoot) > 0):
        minmax2 = "minima"
        sign2 = "positive"
    else:
        minmax2 = "maxima"
        sign2 = "negative"

    sns.scatterplot(x=[finalRoot], y=[0], color='red', label=f'root')
    plt.title("Root Finding Method for Linear Equations")

    mo.md(f"Then, after deflating the polynomial by dividing it by the first root (x-(${roundedNRRoot}$)) and turning it into g(x), we can then find the second root of f'(x) (x=${(finalRoot)}$) using the root finding method for linear equations (-b/a). This will alow us to determine the next critical point: {mo.as_html(lplt)}")

    return finalRoot, minmax2, sign2, val2


@app.cell
def _(finalRoot, minmax2, mo, plt, sign2, sns, val2, x, y):
    secondrtplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=[finalRoot], y=[val2], color='black', label=f"${minmax2}$")
    plt.title("Second Min/Max Value")
    mo.md(f"Since the function value of x = ${finalRoot}$ is located at f(x) = ${val2}$, and the second derivative at that point is ${sign2}$, there is an absolute ${minmax2}$ located at that point. {mo.as_html(secondrtplt)}")
    return


@app.cell
def _(
    finalRoot,
    minmax1,
    minmax2,
    mo,
    plt,
    roundedNRRoot,
    sns,
    val1,
    val2,
    x,
    y,
):
    finalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=[roundedNRRoot, finalRoot], y=[val1, val2], color='black', label=f"${minmax2}$")
    plt.title("All Min/Max Values")
    mo.md(f"To conclude, the ${minmax1}$ of the function is located at ***(${roundedNRRoot}$, ${val1}$)***, and the ${minmax2}$ of the function is located at ***(${finalRoot}$, ${val2}$)***: {mo.as_html(finalplt)}")
    return


@app.cell
def _(a, b, c, np, nrsearch):
    def deflateQuad(root):
        divisor = np.array([1, -root])
        coefficients = [3, -2*(a+b+c), a*b+a*c+b*c]
        deflated_poly, remainder = np.polydiv(coefficients, divisor)
        deflated_poly = np.round(deflated_poly, decimals=10)
        remainder = np.round(remainder, decimals=10)
        return deflated_poly, remainder

    g_func= deflateQuad(nrsearch[-1])[0]

    def g(x):
        deflated_function = np.poly1d(g_func)
        return deflated_function(x)
    return g, g_func


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

    import sys,os
    sys.path.append(os.path.dirname(__name__))
    from numerical.roots import newtonraphson,bisection 

    import seaborn as sns
    sns.set_style("whitegrid")

    import matplotlib.pyplot as plt
    return mo, newtonraphson, np, plt, sns


if __name__ == "__main__":
    app.run()
