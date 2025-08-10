import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    from time import sleep

    import jax
    import jax.numpy as jnp
    return jax, jnp, mo, np, plt, sns, tqdm


@app.cell
def _(np, sns):
    A,B,C = 3, 5, 7
    x = np.random.randn(1000)*10
    y = A*x**2 + B*x + C + np.random.randn(1000)*100
    sns.scatterplot(x=x,y=y, marker='x')
    return x, y


@app.cell
def _(np):
    def model(params,x):
        a,b,c = params
        return a*x**2 + b*x + c

    def error(yhat, y):
        return np.mean((y - yhat)**2) 


    def loss(params, x, y):
        yhat = model(params,x)
        e = error(yhat,y)
        return e
    return loss, model


@app.cell
def _(a, b, c, loss, np, x, y):
    errmin = np.inf
    sol = None
    for _a in range(1,10):
        for _b in range(5,15):
            for _c in range(15,25):
                e = loss([_a,_b,_c],x,y)
                if e < errmin:
                    errmin = e
                    sol = [a,b,c]

    print(sol)
            
    return


@app.cell
def _(jax, jnp):
    @jax.jit
    def gloss(params, x, y):
        a,b,c = params
        common = 2*(a*x**2 + b*x + c - y) 

        ga = jnp.sum(common * x**2)
        gb = jnp.sum(common * x)
        gc = jnp.sum(common)

        return [ga,gb,gc]
    return (gloss,)


@app.cell
def _(gloss, np, tqdm, x, y):
    a,b,c = 0.0,0.0,0.0

    pbar = tqdm(range(10000), colour='green')
    lr = 2e-9

    sols = [(0,0,0)]
    for _i in pbar:
        ga,gb,gc = gloss([a,b,c], x,y)
        a = a - lr*ga
        b = b - lr*gb
        c = c - 100*lr*gc

        pbar.set_postfix_str(f"{a:.2f} {b:.2f} {c:.2f}")

        if _i % 1 == 0:
            sols.append(np.array([a,b,c]))

        # sleep(0.0001)
    return a, b, c, sols


@app.cell
def _(mo, sols):
    select = mo.ui.slider(start=0, stop=len(sols), step=1, debounce=False)
    select
    return (select,)


@app.cell
def _(model, plt, select, sns, sols, x, y):
    plt.xlim(-40,40)
    plt.ylim(-500,4000)
    sns.scatterplot(x=x,y=y, marker='x')
    sns.scatterplot(x=x, y = model(sols[select.value],x), marker='.',color='red')

    return


@app.cell
def _(sols):
    len(sols)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
