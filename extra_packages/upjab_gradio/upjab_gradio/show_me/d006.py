import gradio as gr

def generate_fake_image(prompt, seed, initial_image=None):
    if initial_image is None:
        return f"Used seed: {seed}", prompt, "https://dummyimage.com/300/09f.png"
    return f"Used seed: {seed}", prompt, initial_image

demo = gr.Interface(
    generate_fake_image,
    inputs=["textbox"],
    outputs=["textbox", "textbox", "image"],
    additional_inputs=[
        gr.Slider(0, 1000),
        "image"
    ]
)

if __name__ == "__main__":
    demo.launch()

