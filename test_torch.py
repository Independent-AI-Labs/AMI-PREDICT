#!/usr/bin/env python
"""Test PyTorch installation."""

import sys
import os

# Change to temp directory to avoid import issues
os.chdir(os.environ.get('TEMP', '/tmp'))

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    if hasattr(torch, 'xpu'):
        print(f"XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"XPU devices: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"  Device {i}: {torch.xpu.get_device_name(i)}")
    else:
        print("XPU support not available")
        
    # Test tensor operation
    x = torch.randn(100, 100)
    if torch.xpu.is_available():
        x = x.to('xpu')
        print(f"Created tensor on XPU: {x.device}")
    else:
        print(f"Created tensor on CPU: {x.device}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()