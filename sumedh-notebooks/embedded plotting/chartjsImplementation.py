import marimo

__generated_with = "0.14.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import anywidget
    import traitlets
    return anywidget, mo, np, traitlets


@app.cell
def _(mo):
    mo.Html("<div id=abc100>abc</div>")
    return


@app.cell
def _(anywidget, traitlets):
    class ChartWidget(anywidget.AnyWidget):
        _esm = """
         import { Chart, registerables } from 'https://cdn.jsdelivr.net/npm/chart.js@4.5.0/+esm'
             Chart.register(...registerables);

        function render({ model, el }) {
          // Create canvas
          const canvas = document.createElement("canvas");
          canvas.width = 600;
          canvas.height = 400;
          el.appendChild(canvas);

          // Get data from Python
          const labels = model.get("labels");
          const datasets = model.get("datasets");

          const config = {
            type: 'line',
            data: {
              labels: labels,
              datasets: datasets,
            },
            options: {
              responsive: true,
              animations: false,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: model.get("title"),
                },
              },
            },
          };

          new Chart(canvas, config);
        }

        export default { render };
        """

        labels = traitlets.List().tag(sync=True)
        datasets = traitlets.List().tag(sync=True)
        title = traitlets.Unicode().tag(sync=True)

    return (ChartWidget,)


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


@app.cell
def _(mo):
    refresh = mo.ui.refresh(
      label="Refresh",
      options=["0.1s", "1s", "5s", "10s", "30s"]
    )
    return (refresh,)


@app.cell
def _(mo, refresh):
    import time
    mo.hstack([refresh, refresh.value])

    return (time,)


@app.cell
def _(np, time):
    time.time()
    data = np.random.rand(8)
    return (data,)


@app.cell
def _(ChartWidget, data):
    labels = [f"Day {i}" for i in range(1, 8)]


    widget = ChartWidget(
        labels=labels,
        datasets=[
            {
                "label": "Temperature",
                "data": data,
                "borderColor": "rgb(75, 192, 192)",
                "tension": 0.4,
                "fill": False,
            }
        ],
        title="Weekly Temperature Trend"
    )

    widget

    return


if __name__ == "__main__":
    app.run()
