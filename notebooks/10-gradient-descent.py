import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1D Real valued Functions

    $f : \mathbb{R} \rightarrow \mathbb{R}$

    $f(x) := x^4 - 53x^2 + 196$

    ---

    $f' : \mathbb{R} \rightarrow \mathbb{R}$

    $f'(x) := 4x^3 - 106x$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Gradient Descent


    - Start at a random position $x_0 \in \mathbb{R}$.
    - Select the number of iterations $N$.
    - Select a learning rate $\gamma$.
    - For $t = 0....N-1\\$
    1. Calculate the gradient/derivative at $x_{t}$, i.e.  $f'(x_{t}) \in \mathbb{R}$.  
    2. Update $x$ according to the rule
          $x_{t+1} = x_{t} - \gamma \cdot f'(x_{t})$
    """
    )
    return


@app.cell
def _(np):
    def gradientdescent(df, x0, gamma=1e-3, N=20):
        xs = [x0]

        x = x0
        for i in range(N):
            x = x - gamma * df(x)
            xs.append(x)

        return np.array(xs)
    return (gradientdescent,)


@app.cell(hide_code=True)
def _(mo):
    x0slider = mo.ui.slider(-10,10,0.1,show_value=True, value=0, label='x')
    descend = mo.ui.run_button(label="descend")
    x0slider,descend
    return descend, x0slider


@app.cell(hide_code=True)
def _(descend, dfn, fn, gradientdescent, plt, sns, x, x0slider, y):
    x0 = x0slider.value
    y0 = fn(x0)
    m0 = dfn(x0)
    sns.lineplot(x=x,y=y, color="blue")
    sns.scatterplot(x=[x0],y=[y0], color="green", marker='o')
    sns.scatterplot(x=[x0],y=[0], color="red", marker='o')
    plt.quiver(x0,0, m0*0.01, 0, angles='xy', scale_units='xy', scale=1, color='red')
    plt.plot([x.min(),x.max()],[0,0], color="gray" )

    # plt.grid()
    if descend.value == False:
        plt.plot([x0-1,x0+1],[y0-1*m0,y0+1*m0], color="green")
        plt.title("Gradient and tangent")
    else:
        xs = gradientdescent(dfn,x0)
        sns.scatterplot(x=xs,y=fn(xs), hue=reversed([_ for _ in range(len(xs))]), legend=None)
        sns.scatterplot(x=[x0],y=[y0], color="green", marker='o')

    plt.xlim(x.min()-1,x.max()+1)
    plt.show()

    return


@app.cell(hide_code=True)
def _():
    def fn(x):
        return x**4 - 53*x**2 + 196

    def dfn(x):
        return 4*x**3 - 106*x
    return dfn, fn


@app.cell(hide_code=True)
def _(fn, np):
    x = np.arange(-8,8,0.05);
    y = fn(x)
    return x, y


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt, sns


if __name__ == "__main__":
    app.run()
