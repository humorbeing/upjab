import random


def random_response(message, history):
    return random.choice(["Yes", "No"])


import gradio as gr

demo = gr.ChatInterface(fn=random_response, type="messages")

if __name__ == "__main__":
    demo.launch()
