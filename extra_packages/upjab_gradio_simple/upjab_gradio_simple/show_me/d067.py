import gradio as gr


def music(message, history):
    if message.strip():
        return gr.Audio(
            "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
        )
    else:
        return "Please provide the name of an artist"


demo = gr.ChatInterface(
    music,
    type="messages",
    textbox=gr.Textbox(
        placeholder="Which artist's music do you want to listen to?", scale=7
    ),
)

if __name__ == "__main__":
    demo.launch()
