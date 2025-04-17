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
    selection_total = gr.Number(label="Total Weight of Selection")

    def select_region(selection: gr.SelectData):
        min_w, max_w = selection.index
        return df[(df["weight"] >= min_w) & (df["weight"] <= max_w)]["weight"].sum()

    plt.select(select_region, None, selection_total)

if __name__ == "__main__":
    demo.launch()