"""
MLReady All-In-One Terminal Runner
Launches Backend API, Background Worker, and Frontend UI in ONE single terminal window.
Press Ctrl+C to cleanly stop all services.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

# Virtualenv Python & executables
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
NPM_CMD = "npm.cmd" if sys.platform == "win32" else "npm"

# ANSI Colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def stream_output(pipe, prefix: str, color: str):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            text = line.rstrip()
            if text:
                print(f"{color}{prefix}{RESET} {text}")
    except Exception:
        pass


def kill_proc_tree(pid: int):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass


def main():
    # Enable ANSI color escape codes on Windows console
    if sys.platform == "win32":
        os.system("")

    print(f"\n{BOLD}{CYAN}===================================================={RESET}")
    print(f"{BOLD}{CYAN}         Starting MLReady SaaS Platform            {RESET}")
    print(f"{BOLD}{CYAN}===================================================={RESET}\n")

    # 1. MongoDB check
    print(f"[*] Checking MongoDB on port 27017... ", end="", flush=True)
    if is_port_open(27017):
        print(f"{GREEN}[OK]{RESET}")
    else:
        print(f"{YELLOW}[Attempting to start MongoDB service]{RESET}")
        try:
            subprocess.run(["net", "start", "MongoDB"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        except Exception:
            pass

        if is_port_open(27017):
            print(f"{GREEN}[OK] MongoDB started on port 27017{RESET}")
        else:
            print(f"{RED}[WARN] MongoDB not found on port 27017.{RESET}")
            print(f"       If you are using Docker, run: docker compose up -d mongo")
            print(f"       If MongoDB is installed locally, ensure the service is running.\n")

    processes = []

    def cleanup():
        print(f"\n{YELLOW}[*] Shutting down all MLReady services...{RESET}")
        for p in processes:
            try:
                kill_proc_tree(p.pid)
            except Exception:
                pass
        print(f"{GREEN}[OK] All services stopped.{RESET}\n")

    try:
        # 2. Start Backend API
        print(f"{CYAN}[API]{RESET} Starting FastAPI backend on port 8000...")
        api_env = os.environ.copy()
        api_env["PYTHONPATH"] = str(BACKEND_DIR)
        api_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=api_env,
        )
        processes.append(api_proc)
        threading.Thread(target=stream_output, args=(api_proc.stdout, "[API]", CYAN), daemon=True).start()

        # 3. Start Background Worker
        print(f"{MAGENTA}[WORKER]{RESET} Starting background worker...")
        worker_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "app.jobs.worker"],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=api_env,
        )
        processes.append(worker_proc)
        threading.Thread(target=stream_output, args=(worker_proc.stdout, "[WORKER]", MAGENTA), daemon=True).start()

        # 4. Start Frontend
        print(f"{GREEN}[WEB]{RESET} Starting Vite frontend on port 5173...")
        web_proc = subprocess.Popen(
            [NPM_CMD, "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(web_proc)
        threading.Thread(target=stream_output, args=(web_proc.stdout, "[WEB]", GREEN), daemon=True).start()

        # 5. Wait for readiness
        def wait_and_open():
            opened = False
            for _ in range(30):
                time.sleep(1)
                try:
                    with urllib.request.urlopen("http://localhost:5173", timeout=1) as resp:
                        if resp.status == 200 and not opened:
                            print(f"\n{BOLD}{GREEN}===================================================={RESET}")
                            print(f"{BOLD}{GREEN}   MLReady is LIVE at: http://localhost:5173        {RESET}")
                            print(f"{BOLD}{GREEN}   API Docs available at: http://localhost:8000/docs {RESET}")
                            print(f"{BOLD}{YELLOW}   Press Ctrl+C to stop all services                {RESET}")
                            print(f"{BOLD}{GREEN}===================================================={RESET}\n")
                            try:
                                webbrowser.open("http://localhost:5173")
                            except Exception:
                                pass
                            opened = True
                            break
                except Exception:
                    pass

        threading.Thread(target=wait_and_open, daemon=True).start()

        # Keep running until Ctrl+C
        while True:
            time.sleep(0.5)
            # Check if any process unexpectedly died
            for p in processes:
                if p.poll() is not None:
                    print(f"\n{RED}[!] A process stopped unexpectedly with exit code {p.returncode}{RESET}")
                    cleanup()
                    return

    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
