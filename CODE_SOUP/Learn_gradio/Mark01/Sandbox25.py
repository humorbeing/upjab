import gradio as gr

# with gr.Blocks() as demo:
#     with gr.Row():
#         num1 = gr.Slider(1, 10)
#         num2 = gr.Slider(1, 10)
#         num3 = gr.Slider(1, 10)
#     output = gr.Number(label="Sum")

#     @gr.on(inputs=[num1, num2, num3], outputs=output)
#     def sum(a, b, c):
#         return a + b + c
    

# with gr.Blocks() as demo:
#   num1 = gr.Number()
#   num2 = gr.Number()
#   product = gr.Number(lambda a, b: a * b, inputs=[num1, num2])


with gr.Blocks() as demo:
  num1 = gr.Number()
  num2 = gr.Number()
  product = gr.Number()

  gr.on(
    [num1.change, num2.change, demo.load], 
    lambda a, b: a * b, 
    inputs=[num1, num2], 
    outputs=product
  )

demo.launch()
