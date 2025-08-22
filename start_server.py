"""
Start the CryptoBot Pro API server
"""

import sys
import subprocess
import time
from pathlib import Path

def start_server():
    """Start the API server"""
    print("Starting CryptoBot Pro API server...")
    
    # Start the server in a subprocess
    process = subprocess.Popen(
        [sys.executable, "src/main.py", "run", "--mode", "simulation"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for server to start
    time.sleep(3)
    
    if process.poll() is None:
        print(f"API server started with PID: {process.pid}")
        print("Server running at: http://localhost:8000")
        print("Dashboard at: http://localhost:3000")
        return process
    else:
        print("Failed to start server")
        stdout, stderr = process.communicate()
        print("Error:", stderr)
        return None

if __name__ == "__main__":
    process = start_server()
    if process:
        try:
            # Keep running
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping server...")
            process.terminate()
            process.wait()
            print("Server stopped")