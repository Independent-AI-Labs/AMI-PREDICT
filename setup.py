#!/usr/bin/env python
"""CryptoBot Pro setup script - handles environment setup and dependencies."""

import os
import sys
import subprocess
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def setup_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = PROJECT_ROOT / ".venv"
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        if not run_command("uv venv .venv --python 3.12"):
            # Fallback to standard venv
            run_command("python -m venv .venv")
    else:
        print("Virtual environment already exists.")
    
    # Return activation command based on OS
    if sys.platform == "win32":
        activate = str(venv_path / "Scripts" / "activate")
        python_exe = str(venv_path / "Scripts" / "python.exe")
    else:
        activate = f"source {venv_path / 'bin' / 'activate'}"
        python_exe = str(venv_path / "bin" / "python")
    
    return python_exe, activate


def install_dependencies(python_exe):
    """Install project dependencies."""
    print("\nInstalling dependencies...")
    
    # Install base requirements
    run_command(f"{python_exe} -m pip install --upgrade pip", "Upgrading pip")
    run_command(f"{python_exe} -m pip install -r requirements.txt", "Installing base requirements")
    
    # Check for GPU and install XPU requirements if Intel GPU detected
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if gpu.Name and 'Intel' in gpu.Name and 'Arc' in gpu.Name:
                print(f"\nIntel Arc GPU detected: {gpu.Name}")
                print("Installing Intel XPU support...")
                run_command(f"{python_exe} -m pip install -r requirements-xpu.txt", 
                          "Installing XPU requirements")
                break
    except:
        # Try alternative GPU detection
        result = subprocess.run("wmic path win32_VideoController get name", 
                              shell=True, capture_output=True, text=True)
        if "Intel" in result.stdout and "Arc" in result.stdout:
            print("\nIntel Arc GPU detected")
            run_command(f"{python_exe} -m pip install -r requirements-xpu.txt", 
                      "Installing XPU requirements")
    
    # Install test requirements
    if (PROJECT_ROOT / "requirements-test.txt").exists():
        run_command(f"{python_exe} -m pip install -r requirements-test.txt", 
                   "Installing test requirements")


def setup_pre_commit_hooks(python_exe):
    """Install and configure pre-commit hooks."""
    print("\nSetting up pre-commit hooks...")
    
    # Install pre-commit
    run_command(f"{python_exe} -m pip install pre-commit", "Installing pre-commit")
    
    # Install the git hooks
    run_command(f"{python_exe} -m pre_commit install", "Installing pre-commit hooks")
    run_command(f"{python_exe} -m pre_commit install --hook-type pre-push", "Installing pre-push hooks")
    
    # Run pre-commit on all files to check
    print("\nRunning initial pre-commit checks...")
    run_command(f"{python_exe} -m pre_commit run --all-files", "Running pre-commit checks")


def verify_xpu_setup(python_exe):
    """Verify Intel XPU setup if available."""
    print("\nVerifying XPU setup...")
    
    test_script = '''
import sys
try:
    import torch
    import intel_extension_for_pytorch as ipex
    print(f"PyTorch version: {torch.__version__}")
    print(f"IPEX version: {ipex.__version__}")
    
    # Check for XPU devices
    if hasattr(torch, 'xpu'):
        if torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            print(f"XPU devices available: {device_count}")
            for i in range(device_count):
                print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
        else:
            print("XPU support installed but no devices available")
    else:
        print("XPU support not available in this PyTorch build")
except ImportError as e:
    print(f"XPU support not installed: {e}")
except Exception as e:
    print(f"Error checking XPU: {e}")
'''
    
    # Write temporary test script
    test_file = PROJECT_ROOT / "_test_xpu.py"
    test_file.write_text(test_script)
    
    # Run test
    run_command(f"{python_exe} {test_file}")
    
    # Clean up
    test_file.unlink()


def main():
    """Main setup function."""
    print("="*60)
    print("CryptoBot Pro Setup")
    print("="*60)
    
    # Setup virtual environment
    python_exe, activate_cmd = setup_virtual_environment()
    print(f"\nTo activate the environment, run: {activate_cmd}")
    
    # Install dependencies
    install_dependencies(python_exe)
    
    # Setup pre-commit hooks
    setup_pre_commit_hooks(python_exe)
    
    # Verify XPU setup
    verify_xpu_setup(python_exe)
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print(f"\nTo get started:")
    print(f"1. Activate environment: {activate_cmd}")
    print(f"2. Download data: python download_historical_data.py")
    print(f"3. Run experiments: python run_deep_learning.py")
    print(f"4. Start benchmark: python run_benchmark.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
