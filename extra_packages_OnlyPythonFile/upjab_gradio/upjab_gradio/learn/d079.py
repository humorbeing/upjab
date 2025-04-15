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
    plt = gr.LinePlot(df, x="weight", y="height")

    def select_region(selection: gr.SelectData):
        min_w, max_w = selection.index
        return gr.LinePlot(x_lim=(min_w, max_w)) 

    plt.select(select_region, None, plt)
    plt.double_click(lambda: gr.LinePlot(x_lim=None), None, plt)


if __name__ == "__main__":
    demo.launch()