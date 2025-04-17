import gradio as gr

with gr.Blocks() as demo:
    name = gr.Textbox(label="Error", value="Error Page")
    # output = gr.Textbox(label="Output Box")


if __name__ == "__main__":
    demo.launch()
