import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import jax
    import jax.numpy as jnp

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_sample_image

    from tqdm import tqdm

    import optax
    return jax, jnp, load_sample_image, np, optax, plt, tqdm


@app.cell
def _(load_sample_image, plt):
    img = load_sample_image("flower.jpg")[:,:,0]
    img = img/255 
    plt.imshow(img)
    return (img,)


@app.cell
def _(img, np, plt):
    mask = np.random.rand(*img.shape)
    mask = (mask < 0.1)*1

    maskedimg = mask*img
    plt.imshow(maskedimg)
    return mask, maskedimg


@app.cell
def _(jax, jnp):
    jax.jit
    def lossfn(params, img, mask):
        V,W = params
        return jnp.linalg.norm(mask*(img - V@W), ord='nuc')
    return (lossfn,)


@app.cell
def _(jax, lossfn):
    glossfn = jax.jit(jax.grad(lossfn))
    return (glossfn,)


@app.cell
def _(glossfn, lossfn, mask, maskedimg, np, optax, tqdm):
    EPOCSHS = 500
    RANK = 10
    pbar = tqdm(range(EPOCSHS), colour='green')
    lr = 1e-2

    A, B = np.random.rand(maskedimg.shape[0],RANK), np.random.rand(RANK, maskedimg.shape[1])
    params = [A*0.1,B*0.1]


    optimizer = optax.adam(learning_rate=lr)
    state = optimizer.init(params)

    for _i in pbar:
        g = glossfn(params, maskedimg, mask)

        updates, state = optimizer.update(g, state, params)
        params = optax.apply_updates(params, updates)

        if _i % 10 == 0:
            loss = lossfn(params, maskedimg, mask) 

        pbar.set_postfix_str(f"loss: {loss:.2f}")
    
    
    
    return (params,)


@app.cell
def _(params, plt):
    imghat = params[0] @ params[1]
    plt.imshow(imghat)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
