import subprocess
import os

save_folder = os.path.dirname(__file__)
# command: cd shows/gradio_multipages && uvicorn uvicorn_app:app
# subprocess.run("cd shows/gradio_multipages && uvicorn uvicorn_app:app", shell=True, check=True)  # not working
# subprocess.run(["cd", f"{save_folder}/.."])  # not working

target_dir = os.path.join(save_folder, "shows", "gradio_multipages")
# subprocess.run(["ls", "-a"], cwd = target_dir, shell = True)
# subprocess.run(["uvicorn", "uvicorn_app:app"], cwd = target_dir, shell = True)
# subprocess.run(["uvicorn", "uvicorn_app:app"])


# cmd: uvicorn --app-dir  ./two/app app:app
subprocess.run(["uvicorn", "--app-dir", f"{target_dir}", "uvicorn_app:app"])

print("done")