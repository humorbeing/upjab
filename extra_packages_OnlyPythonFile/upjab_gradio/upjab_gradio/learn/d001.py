import gradio as gr

def greet(names, intensity):
    return "Hello, " + names + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)


if __name__ == "__main__":
    demo.launch()
