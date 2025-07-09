import gradio as gr
import pandas as pd
import numpy as np
import random
from gradio_datetimerange import DateTimeRange
from datetime import datetime, timedelta

now = datetime.now()

df = pd.DataFrame(
    {
        "time": [now - timedelta(minutes=5 * i) for i in range(25)],
        "price": np.random.randint(100, 1000, 25),
        "origin": [random.choice(["DFW", "DAL", "HOU"]) for _ in range(25)],
        "destination": [random.choice(["JFK", "LGA", "EWR"]) for _ in range(25)],
    }
)

with gr.Blocks() as demo:
    with gr.Row():
        origin = gr.Dropdown(["All", "DFW", "DAL", "HOU"], value="All", label="Origin")
        destination = gr.Dropdown(
            ["All", "JFK", "LGA", "EWR"], value="All", label="Destination"
        )
        max_price = gr.Slider(0, 1000, value=1000, label="Max Price")

    plt = gr.ScatterPlot(
        df, x="time", y="price", inputs=[origin, destination, max_price]
    )

    @gr.on(inputs=[origin, destination, max_price], outputs=plt)
    def filtered_data(origin, destination, max_price):
        _df = df[df["price"] <= max_price]
        if origin != "All":
            _df = _df[_df["origin"] == origin]
        if destination != "All":
            _df = _df[_df["destination"] == destination]
        return _df

    with gr.Row():
        gr.Label(len(df), label="Flight Count")
        gr.Label(f"${df['price'].min()}", label="Cheapest Flight")
    gr.DataFrame(df)


if __name__ == "__main__":
    demo.launch()
