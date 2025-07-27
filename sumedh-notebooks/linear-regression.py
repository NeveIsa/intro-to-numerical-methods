import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
    ### Linear Regression


    1. Define the loss function:

    \[
    loss(m) = \sum_{i=1}^{n} (y_i - m x_i)^2
    \]

    2. Find the derivative of the loss function with respect to \( m \):

    \[
    dloss(m) = \\frac{d}{dm} \sum_{i=1}^{n} (y_i - m x_i)^2 = -2 \sum_{i=1}^{n} (y_i - m x_i) x_i
    \]

    3. Find the root of the derivative function:

    Use a numerical method like the bisection method or the Newton-Raphson method to solve:

    \[
    dloss(m) = 0
    \]

    4. Result:

    The value of \( m \) that satisfies this condition is the most optimal slope of the best-fit line.
    """
    )
    return


@app.cell
def _(plt, x, y):
    plt.scatter(x,y)
    plt.title("Dataset")
    return


@app.cell
def _(loss, np, plt, sns):
    _x0 = np.arange(0, 11, 0.01)
    lossVals = [loss(t) for t in _x0]
    sns.lineplot(x=_x0, y=lossVals)
    plt.title(label=f"Loss Function: loss(m)")
    return


@app.cell
def _(dloss, np, plt, sns):
    _x1 = np.arange(0, 11, 0.01)
    dlossVals = [dloss(m) for m in _x1]
    sns.lineplot(x=_x1, y=dlossVals)
    plt.title(label=f"Derivative of Loss Function: dloss(m)")
    return (dlossVals,)


@app.cell
def _(bisection, dloss, dlossVals, np, plt, sns):
    _x1 = np.arange(0, 11, 0.01)
    _x2 = np.random.rand(10)*25 - 10
    _y2 = np.array([dloss(m) for m in _x2])
    x0,x1 = np.random.choice(_x2[_y2<0]),np.random.choice(_x2[_y2>0])

    bsearch = np.array(bisection(dloss,x0,x1))
    sns.lineplot(x=_x1,y=dlossVals)

    alphas=np.linspace(0.3, 1, len(bsearch) )
    colors = [(1, 0, 0, a) for a in alphas]

    sns.scatterplot(x=bsearch, y=[dloss(m) for m in bsearch], marker="o", color=colors, label="root approximations")
    sns.scatterplot( x=np.array([x0,x1]), y=np.array([dloss(x0), dloss(x1)]), color='black', label="x0,x1" )
    plt.title("Bisection Method")

    return


@app.cell
def _(np, x, y):
    def loss(m):
        return np.sum((y-m*x)**2)

    def dloss(m):
        return -2*np.sum((y-m*x)*x)


    return dloss, loss


@app.cell
def _(np):
    X = [7.7396, 4.3888, 8.586, 6.9737, 0.9418, 9.7562, 7.6114, 7.8606, 1.2811, 4.5039, 3.708, 9.2676, 6.4387, 8.2276, 4.4341, 2.2724, 5.5458, 0.6382, 8.2763, 6.3166, 7.5809, 3.5453, 9.707, 8.9312, 7.7838, 1.9464, 4.6672, 0.438, 1.5429, 6.8305, 7.4476, 9.6751, 3.2583, 3.7046, 4.6956, 1.8947, 1.2992, 4.757, 2.2691, 6.6981, 4.3715, 8.3268, 7.0027, 3.1237, 8.3226, 8.0476, 3.8748, 2.8833, 6.825, 1.3975, 1.9991, 0.0736, 7.8692, 6.6485, 7.0517, 7.8073, 4.5892, 5.6874, 1.398, 1.1453, 6.684, 4.711, 5.6524, 7.65, 6.3472, 5.5358, 5.5921, 3.0395, 0.3082, 4.3672, 2.1458, 4.0853, 8.534, 2.3394, 0.583, 2.8138, 2.9359, 6.6192, 5.5703, 7.839, 6.6431, 4.0639, 8.1402, 1.6697, 0.2271, 0.9005, 7.2236, 4.6188, 1.6127, 5.0104, 1.5231, 6.9632, 4.4616, 3.8102, 3.0151, 6.3028, 3.6181, 0.8765, 1.1801, 9.619]

    x = np.array(X)

    Y = [19.7487, 10.0665, 21.0868, 18.7334, 1.9982, 25.1281, 18.0949, 19.4462, 2.2528, 10.9206, 10.1103, 21.4418, 16.5311, 20.8068, 10.4912, 4.2349, 13.9367, 1.0659, 20.9235, 15.8135, 20.554, 8.6238, 23.244, 22.5073, 19.6796, 6.2252, 12.5031, 1.452, 5.3205, 15.8875, 17.9793, 23.2612, 7.7558, 7.8848, 12.374, 4.5146, 1.7772, 10.877, 5.9862, 17.5835, 12.9255, 23.7308, 17.921, 6.8196, 18.6744, 20.3868, 8.874, 6.7928, 16.4503, 3.353, 6.0637, 0.3411, 19.5145, 15.5856, 15.9545, 19.0319, 11.4191, 15.9865, 3.6252, 3.846, 16.2108, 10.5925, 13.1658, 18.3997, 17.9964, 13.0181, 14.8187, 6.6958, 1.702, 11.3029, 5.208, 10.1725, 20.6803, 6.2946, 1.0026, 5.809, 6.0619, 16.7205, 15.5049, 19.7574, 16.4892, 10.4455, 21.6565, 4.3937, 0.1569, 3.3575, 18.4877, 13.0827, 4.215, 11.3017, 2.4396, 19.0589, 12.8776, 9.346, 7.1546, 17.2185, 7.9383, 1.2965, 3.5935, 23.6528]

    y = np.array(Y)


    return x, y


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import sys,os
    sys.path.append(os.path.dirname(__name__))
    from numerical.roots import newtonraphson,bisection 

    import seaborn as sns
    sns.set_theme(style='darkgrid', palette='deep')

    import matplotlib.pyplot as plt
    return bisection, mo, np, plt, sns


if __name__ == "__main__":
    app.run()
