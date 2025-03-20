import gradio as gr

def greet(names, kaka, intensity):
    return "Hello, " + names + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text", "slider"],
    outputs=["text"],
)


if __name__ == "__main__":
    demo.launch()
