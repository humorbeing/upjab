import gradio as gr

def greet(names, kaka, intensity):
    return "Hello, " + names + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text", "slider"],
    outputs=["text"],
)

demo.launch()
