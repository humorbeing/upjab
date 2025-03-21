# run command:
# uvicorn app:app
# 
# This run command  
# uvicorn uvicorn_app:app
#

import gradio as gr

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from script.shows.gradio_multipage.gradio_apps import gradio_apps


app = FastAPI()

for gradio_app in gradio_apps:
    app = gr.mount_gradio_app(app, gradio_app["app"], path="/gradio/" + gradio_app["path"])

templates = Jinja2Templates(directory="script/shows/gradio_multipage/templates")

app.mount("/static", StaticFiles(directory="script/shows/gradio_multipage/static"), name="static")

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
