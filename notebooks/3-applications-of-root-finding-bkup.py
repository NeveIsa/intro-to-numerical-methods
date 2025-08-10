import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# 3. Applications of Root Finding""")
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
def _(mo, pickRoot, poly):
    a,b = [pickRoot.elements["sliders"][_].value for _ in range(2)]
    c = 0

    _fntext = "Polynomial: f(x) = (x-a)(x-b)"
    if poly.value == "cubic":
        _fntext += "(x-c)"
        c = pickRoot.elements["sliders"][2].value


    _dfntext = "f'(x) = 2x - (a + b)"
    if poly.value == "cubic":
        _dfntext = "Derivative: f'(x) = 3x^2 - 2x(a+b+c) + (ab + bc + ca)"


    mo.md(f"## ${_fntext}\\\\{_dfntext}$"
         )
    return a, b, c


@app.cell(hide_code=True)
def _(mo):
    abc=[
        mo.ui.slider(-10,-5,step=1,label='a', value=-5, show_value=True),
        mo.ui.slider(0,5,step=1,label='b', value=4, show_value=True),
        mo.ui.slider(10,15,step=1,label='c',value=10, show_value=True)
    ]

    sliders = mo.ui.array(abc)
    sliders



    pickRoot = mo.md(
        '''
        **Begin by selecting the roots for the cubic function:**   {sliders}
        '''
    ).batch(sliders=sliders)
    pickRoot
    return (pickRoot,)


@app.cell(hide_code=True)
def _(df, f, mo, np, pickRoot, plt, sns):
    x = np.arange(-10,15,0.01)
    y = f(x)
    dy = df(x)
    originalplt = sns.lineplot(x=x,y=y, label="$f(x)$")
    sns.scatterplot(x=np.array(pickRoot.elements["sliders"].value), y=0, color='black', label=f"Roots")
    # sns.lineplot(x=x,y=dy,label="$f'(x)$")
    plt.title("Cubic Function")
    mo.md(f"Below is the cubic function based on the 3 roots you selected. The three roots are located at x = ${pickRoot.elements["sliders"][0].value}$, ${pickRoot.elements["sliders"][1].value}$, and ${pickRoot.elements["sliders"][2].value}$: {mo.as_html(originalplt)}")
    return


@app.cell(hide_code=True)
def _(a, b, c, poly):
    def f(x):
        if poly.value == "cubic": return (x-a)*(x-b)*(x-c)

    def df(x):
        if poly.value == "cubic": return 3*x**2 - 2*x*(a+b+c) + (a*b + b*c + c*a)
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
    return mo, np, plt, sns


if __name__ == "__main__":
    app.run()
