"""
Aphasia Speech Therapy Application Launcher

Run this file to start the application and open the web interface.
Usage: python launcher.py
"""

import subprocess
import webbrowser
import time
import sys
import os
import socket

# Configuration
HOST = "localhost"
PORT = 8001
LANDING_URL = f"http://{HOST}:{PORT}/"


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(timeout: int = 30) -> bool:
    """Wait for the server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(PORT):
            return True
        time.sleep(0.5)
    return False


def kill_process_on_port(port: int) -> bool:
    """Kill any process using the specified port (Windows)."""
    try:
        # Find PID using netstat
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True
        )

        for line in result.stdout.split('\n'):
            if f":{port}" in line and "LISTENING" in line:
                # Extract PID (last column)
                parts = line.split()
                if parts:
                    pid = parts[-1]
                    try:
                        pid = int(pid)
                        print(f"[INFO] Killing existing server (PID: {pid})...")
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True
                        )
                        time.sleep(1)  # Wait for process to terminate
                        return True
                    except ValueError:
                        continue
        return False
    except Exception as e:
        print(f"[WARNING] Could not kill existing process: {e}")
        return False


def main():
    print("=" * 50)
    print("  Aphasia Speech Therapy Application Launcher")
    print("=" * 50)
    print()

    # Check if server is already running - kill and restart
    if is_port_in_use(PORT):
        print(f"[INFO] Server already running on port {PORT}")
        print(f"[INFO] Restarting server...")
        kill_process_on_port(PORT)
        time.sleep(1)  # Give it a moment to fully release the port

    # Get the backend directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(script_dir, "backend")

    if not os.path.exists(backend_dir):
        print(f"[ERROR] Backend directory not found: {backend_dir}")
        sys.exit(1)

    print(f"[INFO] Starting server on http://{HOST}:{PORT}")
    print(f"[INFO] Backend directory: {backend_dir}")
    print()

    # Start the FastAPI server
    try:
        # Use subprocess to run uvicorn
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", "0.0.0.0",
                "--port", str(PORT),
                "--reload"
            ],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        print("[INFO] Waiting for server to start...")
        print()

        # Wait for server to be ready
        if wait_for_server(timeout=60):
            print("[SUCCESS] Server is ready!")
            print()
            print(f"[INFO] Opening browser to {LANDING_URL}")
            webbrowser.open(LANDING_URL)
            print()
            print("-" * 50)
            print("Server is running. Press Ctrl+C to stop.")
            print("-" * 50)
            print()

            # Stream server output
            try:
                while True:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="")
                    elif process.poll() is not None:
                        break
            except KeyboardInterrupt:
                print()
                print("[INFO] Shutting down server...")
                process.terminate()
                process.wait()
                print("[INFO] Server stopped.")
        else:
            print("[ERROR] Server failed to start within timeout period")
            process.terminate()
            sys.exit(1)

    except FileNotFoundError:
        print("[ERROR] Python or uvicorn not found. Make sure uvicorn is installed:")
        print("        pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
