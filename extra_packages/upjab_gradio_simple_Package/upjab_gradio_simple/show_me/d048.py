import gradio as gr


def image_gen(prompt, image):
    return image


image_gen_1 = image_gen
image_gen_2 = image_gen
image_gen_3 = image_gen

with gr.Blocks() as demo:
    prompt = gr.Textbox()
    image = gr.Image()
    generate_btn_1 = gr.Button("Generate Image via model 1")
    generate_btn_2 = gr.Button("Generate Image via model 2")
    generate_btn_3 = gr.Button("Generate Image via model 3")
    generate_btn_1.click(
        image_gen_1, prompt, image, concurrency_limit=2, concurrency_id="gpu_queue"
    )
    generate_btn_2.click(image_gen_2, prompt, image, concurrency_id="gpu_queue")
    generate_btn_3.click(image_gen_3, prompt, image, concurrency_id="gpu_queue")


if __name__ == "__main__":

    demo.launch()
