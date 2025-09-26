#!/usr/bin/env python
"""
Universal runner script that sets up Intel oneAPI environment for XPU support.
Usage: python run.py <script.py> [args...]
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_intel_environment():
    """Setup Intel oneAPI environment variables for XPU support."""

    # Find Intel oneAPI installation
    oneapi_root = Path("C:/Program Files (x86)/Intel/oneAPI")
    if not oneapi_root.exists():
        oneapi_root = Path("C:/Program Files/Intel/oneAPI")

    if not oneapi_root.exists():
        print("Warning: Intel oneAPI not found. Running without XPU optimization.")
        return False

    print(f"Setting up Intel oneAPI environment from: {oneapi_root}")

    # Critical paths for DLL loading
    critical_paths = []

    # Compiler runtime libraries
    compiler_path = oneapi_root / "compiler" / "latest" / "windows"
    if compiler_path.exists():
        critical_paths.extend(
            [
                compiler_path / "redist" / "intel64" / "compiler",
                compiler_path / "bin",
                compiler_path / "lib",
            ]
        )

    # MKL libraries
    mkl_path = oneapi_root / "mkl" / "latest"
    if mkl_path.exists():
        os.environ["MKLROOT"] = str(mkl_path)
        critical_paths.extend(
            [
                mkl_path / "redist" / "intel64",
                mkl_path / "bin" / "intel64",
            ]
        )

    # TBB libraries
    tbb_path = oneapi_root / "tbb" / "latest"
    if tbb_path.exists():
        os.environ["TBBROOT"] = str(tbb_path)
        critical_paths.extend(
            [
                tbb_path / "redist" / "intel64" / "vc14",
                tbb_path / "bin" / "intel64" / "vc14",
            ]
        )

    # Add all critical paths to PATH
    new_paths = []
    for path in critical_paths:
        if path.exists():
            new_paths.append(str(path))

    if new_paths:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(new_paths) + ";" + current_path

    # Set Intel GPU environment variables
    os.environ["SYCL_CACHE_PERSISTENT"] = "1"
    os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "2"
    os.environ["SETVARS_COMPLETED"] = "1"

    return True


def run_script(script_path, args):
    """Run a Python script with Intel environment."""

    # Setup Intel environment
    setup_intel_environment()

    # Get Python from venv
    venv_path = Path(".venv")
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        print(f"Error: Virtual environment not found at {venv_path}")
        print("Please run: uv venv .venv --python 3.11")
        return 1

    # Build command
    cmd = [str(python_exe), script_path] + args

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    # Run the script with the configured environment
    try:
        # Use subprocess to run in the same console
        result = subprocess.run(cmd, env=os.environ.copy(), cwd=Path(__file__).parent, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error running script: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <script.py> [args...]")
        print("\nExamples:")
        print("  python run.py download_historical_data.py")
        print("  python run.py run_deep_learning.py")
        print("  python run.py run_benchmark.py")
        print("  python run.py src/main.py --mode simulation")
        return 1

    script = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    return run_script(script, args)


if __name__ == "__main__":
    sys.exit(main())
