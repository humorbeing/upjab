import gradio as gr


input_textbox = gr.Textbox()

with gr.Blocks() as demo:
    gr.Examples(["hello", "bonjour", "merhaba"], input_textbox)
    input_textbox.render()


if __name__ == "__main__":
    demo.launch()
