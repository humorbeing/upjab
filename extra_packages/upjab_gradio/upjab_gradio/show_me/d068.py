import gradio as gr


def fake(message, history):
    if message.strip():
        return {
            "role": "assistant",
            "content": {
                "path": "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
            },
        }
    else:
        return "Please provide the name of an artist"


demo = gr.ChatInterface(
    fake,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Which artist's music do you want to listen to?", scale=7
    ),
    chatbot=gr.Chatbot(placeholder="Play music by any artist!"),
)

if __name__ == "__main__":
    demo.launch()
