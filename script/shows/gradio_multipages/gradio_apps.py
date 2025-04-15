
import importlib

NUM_APPS = 83
NUM_APPS = 3
app_list = []

for i in range(NUM_APPS+1):
    app_list.append(importlib.import_module(f"upjab_gradio.learn.d{i:03d}").demo)

gradio_apps = []

for i in list(reversed(range(NUM_APPS+1))):
    gradio_apps.append({
        "title": f"Page {i:03d}",
        "app": app_list[i],
        "path": f"page_{i:03d}"
    })



if __name__ == "__main__":
    print(gradio_apps)