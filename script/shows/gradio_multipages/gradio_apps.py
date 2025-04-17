
import importlib

NUM_APPS = 90
Start_NUM = 1
# NUM_APPS = 2
app_list = []

for i in range(Start_NUM, NUM_APPS+1):
    if i >= Start_NUM:
        try:
            app_list.append(importlib.import_module(f"upjab_gradio.show_me.d{i:03d}").demo)
        except:
            print(f"Module upjab_gradio.show_me.d{i:03d} not found, using default demo.")
            app_list.append(importlib.import_module(f"upjab_gradio.show_me.d000").demo)
    


gradio_apps = []


from upjab_gradio.show_me import main_page


gradio_apps.append({
    "title": f"Main Page",
    "app": main_page.demo,
    "path": f"page_main"
})


for i in list(reversed(range(Start_NUM, NUM_APPS+1))):
    gradio_apps.append({
        "title": f"Page {i:03d}",
        "app": app_list[i-Start_NUM],
        "path": f"page_{i:03d}"
    })



if __name__ == "__main__":
    print(gradio_apps)