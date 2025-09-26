#!/usr/bin/env python
"""
Force XPU execution wrapper.
"""
import os
import subprocess
import sys

# Force Intel XPU environment
os.environ["PYTORCH_EXTENSION_NAME"] = "intel_extension_for_pytorch"
os.environ["USE_XPU"] = "1"
os.environ["FORCE_XPU"] = "1"

# Set Intel oneAPI paths
oneapi_root = r"C:\Program Files (x86)\Intel\oneAPI"
if os.path.exists(oneapi_root):
    os.environ["ONEAPI_ROOT"] = oneapi_root
    os.environ["PATH"] = f"{oneapi_root}\\compiler\\latest\\windows\\bin;{os.environ['PATH']}"
    os.environ["PATH"] = f"{oneapi_root}\\mkl\\latest\\redist\\intel64;{os.environ['PATH']}"
    print(f"✅ Intel oneAPI environment set from: {oneapi_root}")

# Run the actual script
if len(sys.argv) > 1:
    script = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    cmd = [sys.executable, script] + args
    print(f"Running with XPU: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=os.environ, check=False)
    sys.exit(result.returncode)
else:
    print("Usage: python run_with_xpu.py <script.py> [args...]")
