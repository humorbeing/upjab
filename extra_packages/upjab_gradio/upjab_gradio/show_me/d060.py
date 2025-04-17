import gradio as gr


def alternatingly_agree(message, history):
    if len([h for h in history if h["role"] == "assistant"]) % 2 == 0:
        return f"Yes, I do think that: {message}"
    else:
        return "I don't think so"


demo = gr.ChatInterface(fn=alternatingly_agree, type="messages")


if __name__ == "__main__":
    demo.launch()
