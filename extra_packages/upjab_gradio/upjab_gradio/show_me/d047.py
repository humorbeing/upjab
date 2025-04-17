import gradio as gr

def image_gen(prompt, image):
    return image

with gr.Blocks() as demo:
    prompt = gr.Textbox()
    image = gr.Image()
    generate_btn = gr.Button("Generate Image")
    generate_btn.click(image_gen, prompt, image, concurrency_limit=5)


if __name__ == "__main__":
    demo.launch()