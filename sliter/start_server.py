# client.py
import os
import sys
import time
import argparse
import requests
import subprocess
from datetime import datetime

INPY_DIR = '/storage/vbutoi/projects/ionpy'
SCRATCH_DIR = '/storage/vbutoi/scratch/Sliter' 

SERVER_URL = 'http://localhost:5000'
SERVER_SCRIPT = 'manager.py'  # Ensure this path is correct

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f'{SCRATCH_DIR}/scheduler_{timestamp}.log'


def is_server_running():
    try:
        response = requests.get(f"{SERVER_URL}/jobs")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def start_server():
    print("Scheduler server not running. Starting server...")
    with open(LOG_FILE, 'a') as log:
        process = subprocess.Popen(
            [sys.executable,SERVER_SCRIPT],
            stdout=log,
            stderr=log,
            preexec_fn=os.setsid,  # For Unix
        )
    for _ in range(10):
        if is_server_running():
            print("Scheduler server started.")
            return
        time.sleep(1)
    print("Failed to start scheduler server. Check logs for details.")
    sys.exit(1)


def main():
    start_server()


if __name__ == '__main__':
    main()
