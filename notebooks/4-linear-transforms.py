import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Rotation Matrix
    $$
        \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
    $$

    ### Reflection Matrix

    $$
    	 \begin{bmatrix} \cos(2w)  & \sin(2w) \\ \sin(2w) & -\cos(2w) \end{bmatrix} 
         = \frac{1}{1 + m^2} \begin{bmatrix} 1 - m^2 & 2m \\ 2m & m^2 -1 \end{bmatrix}
    $$

    ### [Reflection Matrix from slope](https://math.libretexts.org/Courses/Lake_Tahoe_Community_College/A_First_Course_in_Linear_Algebra_(Kuttler)/05%3A_Linear_Transformations/5.04%3A_Special_Linear_Transformations_in_R)

    $$
    	\cos(2w) = \cos^2(w) - \sin^2(w) = \frac{\cos^2(w) - \sin^2(w)}{\cos^2(w) + \sin^2(w)} 
    	= \frac{1 - \tan^2(w)}{1 + \tan^2(w)} = \frac{1 - m^2}{1 + m^2}
    $$


    $$
    	\sin(2w) = 2 \sin(w)\cos(w)= \frac{2 \sin(w)\cos(w)}{\cos^2(w) + \sin^2(w)} 
    	= \frac{2\tan(w)}{1 + \tan^2(w)} = \frac{2m}{1 + m^2}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(e1, e2, plotvecs, plt, rotate, theta):
    plotvecs([e1,e2,rotate(e1),rotate(e2)], colors=['r','g','r','g'],alphas=[1,1,0.4,0.4])

    print(f"Rotation by {theta.value} degrees matrix\n",rotate.matrix)

    theta,plt.title(f"Rotation by {theta.value} deg")
    return


@app.cell(hide_code=True)
def _(e1, e2, np, plotvecs, plt, reflect, theta):
    plotvecs(
        [e1, e2, reflect(e1), reflect(e2)],
        colors=["r", "g", "r", "g"],
        alphas=[1,1,0.4,0.4]
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
    print(f"Reflection matrix about a line angled at {theta.value} degrees\n",reflect.matrix)

    theta, plt.title(f"Reflection about {theta.value} deg")
    return


@app.cell(hide_code=True)
def _(e1, e2, mo, plotvecs, plt, stretch, xstretch, ystretch):
    plotted = plotvecs(
        [e1, e2, stretch(e1), stretch(e2)],
        colors=["r", "g", "r", "g"],
        alphas=[1,1,0.4,0.4],
    )
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.title(f"Stretch x by {xstretch.value} and y by {ystretch.value}")
    (
        mo.md(f"{xstretch}\n{ystretch} \n {plt}").batch(
            xstretch=xstretch, ystretch=ystretch
        ),
        plotted,
    )
    return


@app.cell(hide_code=True)
def _(np, plotvecs, plt, project, theta2):
    # _v = np.array([np.cos(np.deg2rad(theta2.value)),np.sin(np.deg2rad(theta2.value))])

    plot = plotvecs(
        [[1,0],[0,1] , project([1,0]), project([0,1])],
        colors=["r","g", "r", "g"],
        alphas=[1, 1, 0.4,0.4],
    )

    plt.plot(
        [-10, 10],
        [
            -np.tan(np.deg2rad(theta2.value)) * 10,
            np.tan(np.deg2rad(theta2.value)) * 10,
        ],
        alpha=0.5,
        linestyle=":",
        color="black",
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
    theta2,
    xstretch,
    ystretch,
):
    rotate = Rotation(theta.value)
    reflect = Reflection(theta.value)
    stretch = Stretching(xstretch=xstretch.value, ystretch=ystretch.value)
    projectx = Projection(direction = [1,0])
    projecty = Projection(direction = [0,1])
    project = Projection(direction = [np.cos(np.deg2rad(theta2.value)),np.sin(np.deg2rad(theta2.value))])

    e1, e2 = np.array([1, 0]), np.array([0, 1])

    return e1, e2, project, reflect, rotate, stretch


@app.cell(hide_code=True)
def _(mo):
    _theta = mo.ui.slider(0,180,step=1,show_value=True, label="theta")
    _xstretch = mo.ui.slider(-5,5,step=0.1, show_value=True, label="xstretch")
    _ystretch = mo.ui.slider(-5,5,step=0.1, show_value=True, label="ystretch")
    ctrl = mo.ui.array(elements=[_theta,_xstretch,_ystretch])
    theta,xstretch,ystretch = ctrl

    theta2 = mo.ui.slider(0,180,step=1,show_value=True,label="theta2")
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


if __name__ == "__main__":
    app.run()
