import gradio as gr
import importlib

debug = False

from upjab_gradio.show_me import main_page

with gr.Blocks() as demo:
    main_page.demo.render()

# with demo.route("Second Page"):
#     second_page.demo.render()
if debug:
    from upjab_gradio.show_me import d000

    with demo.route("1 Page"):
        d000.demo.render()





NUM_APPS = 90
Start_NUM = 1
# NUM_APPS = 3
app_list = []

# import importlib.util

SPEC_OS = importlib.util.find_spec('os')
os1 = importlib.util.module_from_spec(SPEC_OS)


for i in range(Start_NUM, NUM_APPS+1):
    if i >= Start_NUM:
        try:
            app_list.append(importlib.import_module(f"upjab_gradio.show_me.d{i:03d}").demo)
        except:
            print(f"Module upjab_gradio.show_me.d{i:03d} not found, using default demo.")
            SPEC_OS = importlib.util.find_spec("upjab_gradio.show_me.d000")
            demo_module = importlib.util.module_from_spec(SPEC_OS)
            SPEC_OS.loader.exec_module(demo_module)            
            app_list.append(demo_module.demo)
    


for i in list(reversed(range(Start_NUM, NUM_APPS+1))):
    with demo.route(f"Page {i:03d}"):
        app_list[i-Start_NUM].render()
    

if __name__ == "__main__":
    demo.launch()