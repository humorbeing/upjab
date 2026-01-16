from upjab.tool.run_folder_module import run_folder
import subprocess

target_folder = "extra_packages/build_cu_cpp_package/scripts"
run_folder(target_folder)

target_folder = "demo/pytorch_model_visualize"
run_folder(target_folder)

# subprocess.run(
#     "tensorboard --logdir=./saves/torchlogs",
#     shell=True
# )

# subprocess.run(
#     "python scripts/run01.py",
#     shell=True
# )

subprocess.run(
    "python demo/build_cu_cpp_NoPackage/setup.py build_ext --build-lib demo/build_cu_cpp_NoPackage --build-temp demo/build_cu_cpp_NoPackage/build",
    shell=True
)


subprocess.run(
    "python demo/build_cu_cpp_NoPackage/add_test.py",
    shell=True
)