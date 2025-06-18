import subprocess
import pkg_resources
import sys
import os
from packaging import version

def check_python_version():
    required_version = "3.7"
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"Checking Python version (required: {required_version})...")
    if version.parse(current_version) != version.parse(required_version):
        print(f"❌ Wrong Python version. Required: {required_version}, Current: {current_version}")
        return False
    print(f"✅ Python version {current_version} OK")
    return True

def check_pytorch():
    try:
        import torch
        import torchvision
        
        required_torch = "1.8.1"
        required_torchvision = "0.9.1"
        
        print(f"Checking PyTorch (required: {required_torch}) and torchvision (required: {required_torchvision})...")
        
        current_torch = torch.__version__.split('+')[0]
        current_vision = torchvision.__version__.split('+')[0]
        
        torch_ok = version.parse(current_torch).release[:2] == version.parse(required_torch).release[:2]
        vision_ok = version.parse(current_vision).release[:2] == version.parse(required_torchvision).release[:2]
        
        if not torch_ok:
            print(f"❌ Wrong PyTorch version. Required: {required_torch}, Current: {current_torch}")
        if not vision_ok:
            print(f"❌ Wrong torchvision version. Required: {required_torchvision}, Current: {current_vision}")
        
        if torch_ok and vision_ok:
            print("✅ PyTorch and torchvision versions OK")
            return True
        return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_mmcv():
    try:
        import mmcv
        required_version = "1.3.0"
        current_version = mmcv.__version__
        
        print(f"Checking MMCV version (required: {required_version})...")
        if version.parse(current_version) != version.parse(required_version):
            print(f"❌ Wrong MMCV version. Required: {required_version}, Current: {current_version}")
            return False
        print(f"✅ MMCV version {current_version} OK")
        return True
    except ImportError:
        print("❌ MMCV not installed")
        return False

def check_requirements_file():
    requirements_file = "requirement.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    print(f"Checking dependencies from {requirements_file}...")
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing = []
    wrong_version = []
    
    for requirement in requirements:
        try:
            req = pkg_resources.Requirement.parse(requirement)
            pkg = pkg_resources.working_set.find(req)
            
            if pkg is None:
                missing.append(requirement)
            else:
                current_version = pkg.version
                if not req.specifier.contains(current_version):
                    wrong_version.append((requirement, current_version))
        except:
            print(f"⚠️ Warning: Could not check {requirement}")
    
    if missing:
        print("\n❌ Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
    
    if wrong_version:
        print("\n❌ Wrong versions:")
        for pkg, current in wrong_version:
            print(f"  - {pkg} (current: {current})")
    
    if not (missing or wrong_version):
        print("✅ All required packages are installed with correct versions")
        return True
    return False

def check_model_files():
    required_files = [
        ("ckpts/sam_vit_h_4b8939.pth", "SAM checkpoint"),
        ("ckpts/SETR_MLA/iter_80000.pth", "SETR-MLA checkpoint"),
    ]
    
    print("\nChecking model files...")
    missing_files = []
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            missing_files.append((file_path, description))
            print(f"❌ Missing {description}: {file_path}")
        else:
            print(f"✅ Found {description}")
    
    return len(missing_files) == 0

def main():
    print("FoodSAM Dependency Checker\n")
    
    checks = [
        check_python_version(),
        check_pytorch(),
        check_mmcv(),
        check_requirements_file(),
        check_model_files()
    ]
    
    print("\nSummary:")
    if all(checks):
        print("✅ All dependencies are satisfied!")
    else:
        print("❌ Some dependencies are missing or incorrect. Please fix the issues above.")

if __name__ == "__main__":
    main()