import subprocess
import os

# save_folder = os.path.dirname(__file__)
# # command: cd /home/ray/workspace/codes/projects/upjab/script/.. && uvicorn uvicorn_app:app
# subprocess.run(["cd", f"{save_folder}/..", "&&", "uvicorn", "uvicorn_app:app"])  # not working
# subprocess.run(["cd", f"{save_folder}/.."])  # not working

subprocess.run(["uvicorn", "uvicorn_app:app"])

print("done")