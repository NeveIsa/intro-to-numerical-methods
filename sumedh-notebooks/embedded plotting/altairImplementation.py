import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    refresh = mo.ui.refresh(
      label="Refresh",
      options=["0.1s", "1s", "5s", "10s", "30s"]
    )
    return (refresh,)


@app.cell
def _(mo, refresh):
    mo.hstack([refresh, refresh.value])
    return


@app.cell
def _(refresh):
    import altair as alt
    import numpy as np
    import pandas as pd

    refresh
    # Generate random data
    # np.random.seed(0)
    x = np.arange(100)
    y = np.random.normal(loc=0, scale=1, size=100)

    # Create DataFrame
    df = pd.DataFrame({
        "x": x,
        "y": y
    })

    # Plot with Altair
    line = alt.Chart(df).mark_line().encode(
        x="x:Q",
        y="y:Q",
        tooltip=["x", "y"]
    ).properties(
        title="Line Plot of Random Data"
    )

    line

    return


@app.cell(hide_code=True)
def _():
    import anywidget
    import traitlets

    class UPlotWidget(anywidget.AnyWidget):
        _esm = """
        import chartJs from 'https://cdn.jsdelivr.net/npm/chart.js@4.5.0/+esm'


        function render({ model, el }) {
         el = document.getElementById("abc100");
          new Chart(el, {
        type: 'bar',
        data: {
          labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
          datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
        }

        export default { render };
        """

        data = traitlets.List().tag(sync=True)

    return anywidget, traitlets


@app.cell(hide_code=True)
def _(anywidget, traitlets):


    class PlotWidget(anywidget.AnyWidget):
        _esm = """
        import chartJs from 'https://cdn.jsdelivr.net/npm/chart.js@4.5.0/+esm'
        new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
          datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
        }

        export default { render };
        """

        data = traitlets.List().tag(sync=True)
    return


if __name__ == "__main__":
    app.run()
