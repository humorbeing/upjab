import gradio as gr

with gr.Blocks() as demo:
    name = gr.Textbox(
        label="Welcome",
        value='Hello World',
        )

if __name__ == "__main__":
    demo.launch()