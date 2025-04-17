__version__ = "0.0.1"

# # this will cause a heavy module loading.
# from upjab_gradio.show_me.gradio_multipage_demo import demo
# def show():    
#     demo.launch()

# # this will cause a heavy module loading.
# from upjab_gradio.show_me import gradio_multipage_demo
# def show():    
#     gradio_multipage_demo.demo.launch()


def show():
    from upjab_gradio.show_me.gradio_multipage_demo import demo
    demo.launch()



