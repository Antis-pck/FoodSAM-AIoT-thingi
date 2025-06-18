import torch
import os

def check_pth_file(path):
    print(f"Testing {path} ...")
    try:
        torch.load(path, map_location="cpu")
        print(f"  [OK] {path} loaded successfully.\n")
    except Exception as e:
        print(f"  [ERROR] {path} failed to load: {e}\n")

def walk_and_check(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".pth"):
                check_pth_file(os.path.join(root, fname))

if __name__ == "__main__":
    walk_and_check("ckpts")  # or replace "ckpts" with your checkpoints folder