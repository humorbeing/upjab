import gradio as gr

import main_page, second_page
from upjab_gradio.show01 import d000

with gr.Blocks() as demo:
    main_page.demo.render()

with demo.route("Second Page"):
    second_page.demo.render()

with demo.route("1 Page"):
    d000.demo.render()

if __name__ == "__main__":
    demo.launch()
