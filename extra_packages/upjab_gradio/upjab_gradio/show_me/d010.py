import gradio as gr
import numpy as np


def flip(im):
    return np.flipud(im)


# demo = gr.Interface(
#         flip,
#         gr.Image(sources=["webcam"], streaming=True),
#         "image",
#         live=True
#     )
try:
    demo = gr.Interface(
        flip, gr.Image(sources=["webcam"], streaming=True), "image", live=True
    )
except:

    def greet():
        return "No webcam"

    demo = gr.Interface(
        fn=greet,
        inputs=None,
        outputs=["text"],
    )

if __name__ == "__main__":
    demo.launch()
