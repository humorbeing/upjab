import gradio as gr

from .ReinforcementLearning_PNU.homeworks.homework03.value_iteration_and_policy_iteration.policy_iteration import main as fn001

def greet(name):
    return "Hello " + name + "!"

def me():
    return("Hello")

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output001 = gr.Textbox(label="Output Box")
    greet_btn001 = gr.Button("Greet")
    greet_btn001.click(fn=greet, inputs=name, outputs=output001, api_name="greet")

    output002 = gr.Textbox(label="Output Box", lines=5)
    # output002 = gr.Chatbot()
    greet_btn002 = gr.Button("Me")
    greet_btn002.click(fn=me, outputs=output002, api_name="me")

    # output002 = gr.Textbox(label="Output Box", lines=5)
    # output002 = gr.Chatbot()
    greet_btn003 = gr.Button("RL")
    greet_btn003.click(fn=fn001)


if __name__ == "__main__":
    demo.launch()
