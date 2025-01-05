import gradio as gr

# with gr.Blocks(fill_height=True) as demo:
#     gr.Chatbot(scale=1)
#     gr.Textbox(scale=0)

# import gradio as gr

with gr.Blocks() as demo:
    im = gr.ImageEditor(width="50vw")

demo.launch()