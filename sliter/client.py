# client.py
import os
import sys
import time
import argparse
import requests
import subprocess
from datetime import datetime
import json  # Added import for json

INPY_DIR = '/storage/vbutoi/projects/ionpy'
SCRATCH_DIR = '/storage/vbutoi/scratch/Sliter' 

SERVER_URL = 'http://localhost:5000'

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f'{SCRATCH_DIR}/scheduler_{timestamp}.log'


def start_server():
    # Run the scheduler_server.py file in the sliter directory.
    sliter_dir = os.path.join(INPY_DIR, "sliter")
    subprocess.run(["python", f"{sliter_dir}/start_server.py"], cwd=sliter_dir)


def kill_job(job_id):
    url = f"{SERVER_URL}/kill"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                print("No jobs found.")
                return
            # Group jobs by status
            grouped_jobs = {}
            for job_id, job in jobs.items():
                status = job.get('status', 'unknown')
                grouped_jobs.setdefault(status, []).append({
                    'job_id': job_id,
                    'job_gpu': job.get('job_gpu')
                })
            for status in ['queued', 'running', 'completed', 'failed', 'cancelled']:
                if status in grouped_jobs:
                    print(f"\n{status.capitalize()} Jobs:")
                    for job in grouped_jobs[status]:
                        print(f"  ID: {job['job_id']}, GPU: {job['job_gpu']}")
        else:
            print("Failed to retrieve jobs.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)


def inspect_job(job_id):
    url = f"{SERVER_URL}/get_job"
    try:
        response = requests.get(url, json={'job_id': job_id})
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                print("No jobs found.")
                return
            # Group jobs by status
            grouped_jobs = {}
            for job_id, job in jobs.items():
                status = job.get('status', 'unknown')
                grouped_jobs.setdefault(status, []).append({
                    'job_id': job_id,
                    'job_gpu': job.get('job_gpu')
                })
            for status in ['queued', 'running', 'completed', 'failed', 'cancelled']:
                if status in grouped_jobs:
                    print(f"\n{status.capitalize()} Jobs:")
                    for job in grouped_jobs[status]:
                        print(f"  ID: {job['job_id']}, GPU: {job['job_gpu']}")
        else:
            print("Failed to retrieve jobs.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)


def flush_jobs(status):
    url = f"{SERVER_URL}/flush"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                print("No jobs found.")
                return
            # Group jobs by status
            grouped_jobs = {}
            for job_id, job in jobs.items():
                status = job.get('status', 'unknown')
                grouped_jobs.setdefault(status, []).append({
                    'job_id': job_id,
                    'job_gpu': job.get('job_gpu')
                })
            for status in ['queued', 'running', 'completed', 'failed', 'cancelled']:
                if status in grouped_jobs:
                    print(f"\n{status.capitalize()} Jobs:")
                    for job in grouped_jobs[status]:
                        print(f"  ID: {job['job_id']}, GPU: {job['job_gpu']}")
        else:
            print("Failed to retrieve jobs.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)


def list_jobs():
    url = f"{SERVER_URL}/jobs"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                print("No jobs found.")
                return
            # Group jobs by status
            grouped_jobs = {}
            for job_id, job in jobs.items():
                status = job.get('status', 'unknown')
                grouped_jobs.setdefault(status, []).append({
                    'job_id': job_id,
                    'job_gpu': job.get('job_gpu')
                })
            for status in ['queued', 'running', 'completed', 'failed', 'cancelled']:
                print(f"\n{status.capitalize()} Jobs:")
                for job in grouped_jobs.get(status, []):
                    print(f"  ID: {job['job_id']}, GPU: {job['job_gpu']}")
            print()# Added print statement to separate the output from the prompt.
        else:
            print("Failed to retrieve jobs.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)

def shutdown_scheduler():
    url = f"{SERVER_URL}/shutdown"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            print("Scheduler is shutting down.")
        else:
            print("Failed to shutdown scheduler.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Local GPU Job Queue Client with Submitit")
    subparsers = parser.add_subparsers(dest='command')

    # Start server command
    subparsers.add_parser('startup', help='Launch the slite job manager')

    # List command
    subparsers.add_parser('list', help='List all jobs')

    # Shutdown command
    subparsers.add_parser('shutdown', help='Shutdown the scheduler server')

    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Kill a job with the given ID')
    kill_parser.add_argument('job_id', type=str, help='ID of the job to kill')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a job with the given ID')
    inspect_parser.add_argument('job_id', type=str, help='ID of the job to inspect')

    # Flush command
    flush_parser = subparsers.add_parser('flush', help='Flush jobs based on their status')
    flush_parser.add_argument('status', choices=['running', 'queued', 'all'], help='Status of jobs to flush')

    args = parser.parse_args()

    if args.command == 'startup':
        start_server()
    elif args.command == 'kill':
        job_id = args.job_id
        kill_job(job_id)
    elif args.command == 'inspect':
        job_id = args.job_id
        inspect_job(job_id)
    elif args.command == 'list':
        list_jobs()
    elif args.command == 'flush':
        flush_jobs(args.status)
    elif args.command == 'shutdown':
        shutdown_scheduler()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

