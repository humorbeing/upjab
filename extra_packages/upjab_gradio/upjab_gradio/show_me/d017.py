import gradio as gr


def increase(num):
    return num + 1


with gr.Blocks() as demo:
    a = gr.Number(label="a")
    b = gr.Number(label="b")
    atob = gr.Button("a > b")
    btoa = gr.Button("b > a")
    atob.click(increase, a, b)
    btoa.click(increase, b, a)


if __name__ == "__main__":
    demo.launch()
