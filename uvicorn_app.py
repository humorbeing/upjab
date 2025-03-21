# run command:
# uvicorn app:app
#   App
#
#

import gradio as gr

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from CODE_SOUP.gradio_apps import gradio_apps


app = FastAPI()

for gradio_app in gradio_apps:
    app = gr.mount_gradio_app(app, gradio_app["app"], path="/gradio/" + gradio_app["path"])

templates = Jinja2Templates(directory="CODE_SOUP/gradio-multipage/templates")

app.mount("/static", StaticFiles(directory="CODE_SOUP/gradio-multipage/static"), name="static")

@app.get("/")
@app.get("/app/{path_name:path}")
def index(request: Request, path_name: str = ""):
    if not path_name:
        return RedirectResponse(url="/app/" + gradio_apps[0]["path"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "gradio_apps": gradio_apps,
        "current_path": path_name,
    })
