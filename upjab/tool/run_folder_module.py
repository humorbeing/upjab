import subprocess
import os




def run_folder(target_folder):
    print("\n" + "="*60)
    print("Running all Python scripts in folder:")
    print(f">>>:  {target_folder}")    
    key = input("Proceed? (y/n): ")
    if key.lower() != 'y':
        print("Operation cancelled.")
        return

    if os.path.exists(target_folder):
        for filename in sorted(os.listdir(target_folder)):
            if filename.endswith(".py"):
                script_path = os.path.join(target_folder, filename)
                subprocess.run(f"python {script_path}", shell=True)
    



if __name__ == "__main__":
    from upjab.tool.run_folder_module import run_folder
    target_folder = "demo"
    run_folder(target_folder)