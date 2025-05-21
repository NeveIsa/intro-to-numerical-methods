import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(e1, e2, plotvecs, plt, rotate, theta):
    plotvecs([e1,e2,rotate(e1),rotate(e2)], colors=['r','g','r','g'],alphas=[0.4,0.4,1,1])
    theta,plt.title(f"Rotation by {theta.value} deg")
    return


@app.cell(hide_code=True)
def _(e1, e2, np, plotvecs, plt, reflect, theta):
    plotvecs(
        [e1, e2, reflect(e1), reflect(e2)],
        colors=["r", "g", "r", "g"],
        alphas=[0.4, 0.4, 1, 1],
    )
    plt.plot(
        [-10, 10],
        [
            -np.tan(np.deg2rad(theta.value)) * 10,
            np.tan(np.deg2rad(theta.value)) * 10,
        ],
        alpha=0.5,
        linestyle=":",
        color="black",
    )
    theta, plt.title(f"Reflection about {theta.value} deg")
    return


@app.cell(hide_code=True)
def _(e1, e2, mo, plotvecs, plt, stretch, xstretch, ystretch):
    plotted = plotvecs(
        [e1, e2, stretch(e1), stretch(e2)],
        colors=["r", "g", "r", "g"],
        alphas=[0.4, 0.4, 1, 1],
    )
    plt.title(f"Stretch x by {xstretch.value} and y by {ystretch.value}")
    (
        mo.md(f"{xstretch}\n{ystretch} \n {plt}").batch(
            xstretch=xstretch, ystretch=ystretch
        ),
        plotted,
    )
    return


@app.cell
def _(np, plotvecs, plt, projectx, projecty, theta2):
    _v = np.array([np.cos(np.deg2rad(theta2.value)),np.sin(np.deg2rad(theta2.value))])

    plot = plotvecs(
        [_v, projectx(_v), projecty(_v)],
        colors=["black", "r", "g"],
        alphas=[0.5, 0.8,0.8],
    )
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    theta2,plot
    return


@app.cell(hide_code=True)
def _(np, plt):
    def plotvecs(vectors, colors=['r'], alphas=[1.0]):
        plt.figure()
        ax = plt.gca()

        if len(colors)==1:
            colors *= len(vectors)
        if len(alphas)==1:
            alphas *= len(vectors)
        
        for v,c,a in zip(vectors,colors,alphas):
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, alpha=a)
    
        # Auto-adjust limits from data
        all_points = np.vstack((np.zeros(2), vectors))
        min_xy = all_points.min(axis=0) - 1
        max_xy = all_points.max(axis=0) + 1
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
    
        ax.set_aspect('equal')
        ax.grid(True)
        return ax

    def plotvec(vector,color='r', alpha=1):
        return plotvecs([vector],[color],[alpha])
    return (plotvecs,)


@app.cell(hide_code=True)
def _(
    Projection,
    Reflection,
    Rotation,
    Stretching,
    np,
    theta,
    xstretch,
    ystretch,
):
    rotate = Rotation(theta.value)
    reflect = Reflection(theta.value)
    stretch = Stretching(xstretch=xstretch.value, ystretch=ystretch.value)
    projectx = Projection(projx=1, projy=0)
    projecty = Projection(projx=0, projy=1)

    e1, e2 = np.array([1, 0]), np.array([0, 1])

    return e1, e2, projectx, projecty, reflect, rotate, stretch


@app.cell(hide_code=True)
def _(mo):
    _theta = mo.ui.slider(0,180,step=1,show_value=True, label="theta")
    _xstretch = mo.ui.slider(-5,5,step=0.1, show_value=True, label="xstretch")
    _ystretch = mo.ui.slider(-5,5,step=0.1, show_value=True, label="ystretch")
    ctrl = mo.ui.array(elements=[_theta,_xstretch,_ystretch])
    theta,xstretch,ystretch = ctrl

    theta2 = mo.ui.slider(0,360,step=1,show_value=True,label="theta2")
    return theta, theta2, xstretch, ystretch


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import sys,os
    sys.path.append(os.path.dirname(__name__))
    from numerical.lineartransformations import Projection,Rotation,Reflection,Stretching

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    import numpy as np
    return Projection, Reflection, Rotation, Stretching, mo, np, plt


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
