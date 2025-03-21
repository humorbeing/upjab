import gradio as gr
import pandas as pd
import numpy as np
import random
from gradio_datetimerange import DateTimeRange


from datetime import datetime, timedelta
now = datetime.now()

df = pd.DataFrame({
    'time': [now - timedelta(minutes=5*i) for i in range(25)],
    'price': np.random.randint(100, 1000, 25),
    'origin': [random.choice(["DFW", "DAL", "HOU"]) for _ in range(25)],
    'destination': [random.choice(["JFK", "LGA", "EWR"]) for _ in range(25)],
})

with gr.Blocks() as demo:
    # 1
    gr.LinePlot(df, x="time", y="price")
    gr.ScatterPlot(df, x="time", y="price", color="origin")


    # 2
    plot = gr.BarPlot(df, x="time", y="price", x_bin="10m")

    bins = gr.Radio(["10m", "30m", "1h"], label="Bin Size")
    bins.change(lambda bins: gr.BarPlot(x_bin=bins), bins, plot)
    with gr.Row():
        start = gr.DateTime("now - 24h")
        end = gr.DateTime("now")
        apply_btn = gr.Button("Apply")
    plot = gr.LinePlot(df, x="time", y="price")

    apply_btn.click(lambda start, end: gr.BarPlot(x_lim=[start, end]), [start, end], plot)


    # 3
    daterange = DateTimeRange(["now - 24h", "now"])
    plot1 = gr.LinePlot(df, x="time", y="price")
    plot2 = gr.LinePlot(df, x="time", y="price", color="origin")
    daterange.bind([plot1, plot2])


    # # 4
    # timer = gr.Timer(5)
    # plot1 = gr.BarPlot(x="time", y="price")
    # plot2 = gr.BarPlot(x="time", y="price", color="origin")

    # timer.tick(lambda: [get_data(), get_data()], outputs=[plot1, plot2])

demo.launch()