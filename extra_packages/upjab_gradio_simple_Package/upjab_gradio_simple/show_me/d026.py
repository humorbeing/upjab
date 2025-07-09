import time
import gradio as gr


def test(message):
    for i in range(len(message)):
        time.sleep(0.3)
        yield message[: i + 1]


demo = gr.Interface(test, "textbox", "textbox").queue()

if __name__ == "__main__":
    # demo.queue()
    demo.launch()
