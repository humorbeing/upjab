import gradio as gr
from time import sleep


def keep_repeating(audio_file):
    for _ in range(10):
        sleep(0.5)
        yield audio_file


demo = gr.Interface(
    keep_repeating,
    gr.Audio(sources=["microphone"], type="filepath"),
    gr.Audio(streaming=True, autoplay=True),
)

if __name__ == "__main__":
    demo.launch()
