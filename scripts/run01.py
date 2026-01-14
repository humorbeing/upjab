import subprocess
import os

subprocess.run("python demo/omegaconf_demo.py", shell=True)
subprocess.run("python demo/loguru_demo.py", shell=True)
subprocess.run("python demo/Gaussian_process_demo.py", shell=True)
# Run all .py files in scripts/reinforcement_learning_PNU_scripts folder
rl_scripts_dir = "scripts/reinforcement_learning_PNU_scripts"
if os.path.exists(rl_scripts_dir):
    for filename in sorted(os.listdir(rl_scripts_dir)):
        if filename.endswith(".py"):
            script_path = os.path.join(rl_scripts_dir, filename)
            subprocess.run(f"python {script_path}", shell=True)
