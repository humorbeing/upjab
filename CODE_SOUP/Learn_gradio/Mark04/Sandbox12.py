import gradio as gr
import pandas as pd
import numpy as np
import random

df = pd.DataFrame({
    'height': np.random.randint(50, 70, 25),
    'weight': np.random.randint(120, 320, 25),
    'age': np.random.randint(18, 65, 25),
    'ethnicity': [random.choice(["white", "black", "asian"]) for _ in range(25)]
})


with gr.Blocks() as demo:
    with gr.Row():
        ethnicity = gr.Dropdown(["all", "white", "black", "asian"], value="all")
        max_age = gr.Slider(18, 65, value=65)

    def filtered_df(ethnic, age):
        _df = df if ethnic == "all" else df[df["ethnicity"] == ethnic]
        _df = _df[_df["age"] < age]
        return _df

    gr.ScatterPlot(filtered_df, inputs=[ethnicity, max_age], x="weight", y="height", title="Weight x Height")
    gr.LinePlot(filtered_df, inputs=[ethnicity, max_age], x="age", y="height", title="Age x Height")

demo.launch()