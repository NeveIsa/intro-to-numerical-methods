import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
        f :  \mathbb{R}^2 \rightarrow \mathbb{R} \\
        \nabla f : \mathbb{R}^2 \rightarrow \mathbb{R}^2
    $$

    $$
        f:= \frac{1}{100}\big[ (x - {20})^2 + 10(y-{20})^2 \big] \cdot \big[ 5(x - {20})^2 + (y-{20})^2 \big]
    $$

    $$
        \nabla f =\frac{1}{100} \begin{pmatrix}  2(x - {20}) \cdot \big[ 5(x - {20})^2 + (y-{20})^2 \big] + \big[ (x - {20})^2 + 10(y-{20})^2 \big] \cdot  10(x-20) \\ 
        20(y - {20}) \cdot \big[ 5(x - {20})^2 + (y-{20})^2 \big] + \big[ (x - {20})^2 + 10(y-{20})^2 \big] \cdot 2(y-20) 
            \end{pmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    x0slider = mo.ui.slider(-20,10,0.1,show_value=True, value=0, label='x')
    y0slider = mo.ui.slider(-10,10,0.1,show_value=True, value=0, label='y')

    descend = mo.ui.run_button(label="descend")
    x0slider,y0slider,descend
    return descend, x0slider, y0slider


@app.cell(hide_code=True)
def _(
    descend,
    dfn,
    fn,
    gradientdescent,
    np,
    plt,
    sns,
    x0slider,
    y0slider,
    zero,
):
    x = np.linspace(-30, 30, 400)
    y = np.linspace(-30, 30, 400)
    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y)

    plt.contour(X, Y, Z, levels=1000, cmap='viridis')
    plt.title('Contour Plot of the Function fn')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.colorbar(label='Function Value')
    # plt.gca()

    sns.scatterplot(x=[zero,-zero],y=[zero,-zero], color='red', marker='.')
    plt.scatter(x=[x0slider.value],y=[y0slider.value], color="green", marker='o', zorder=2)
    if descend.value:
        _xx = np.array([x0slider.value, y0slider.value])
        xs = gradientdescent(dfn, _xx )
        sns.scatterplot(x=xs[:,0],y=xs[:,1], color='green', marker='.', alpha=0.3 ,zorder=2)
    else:   
        pass

    plt.show()
    return


@app.cell(hide_code=True)
def _(np):
    def gradientdescent(df, x0, gamma=1e-3, N=300):
        xs = [x0]

        x = x0
        for i in range(N):
            x = x - gamma * np.array(df(*x))
            xs.append(x)
    
        return np.array(xs)
    return (gradientdescent,)


@app.cell(hide_code=True)
def _(grad, jit):
    zero = 20
    def fn(x,y):
        v =  ((x-zero)**2 + 10*(y-zero)**2) * (5*(x+zero)**2 + (y+zero)**2)
        return v/100

    # def dfn(x,y):
    #     dfx = 2*(x-zero)*( 5*(x-zero)**2 + (y-zero)**2 ) + ( (x-zero)**2 + 10*(y-zero)**2 )*10*(x-zero)
    #     dfy = 20*(y-20)*( 5*(x-20)**2 + (y-20)**2 ) + ( (x-20)**2 + 10*(y-20)**2 )*2*(y-20)

    #     dfx /= 100
    #     dfy /= 100

    #     return np.array([dfx,dfy])

    dfn = jit(grad(fn, argnums=(0,1)))
    return dfn, fn, zero


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    from jax import grad, jit
    return grad, jit, mo, np, plt, sns


if __name__ == "__main__":
    app.run()
