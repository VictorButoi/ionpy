# client.py
import os
import sys
import time
import argparse
import requests
import subprocess
from pprint import pprint   
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
    data = {'job_id': job_id}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'Status Unknown.')
            print(status)
        else:
            error_message = response.json().get('error', 'No error message provided.')
            print(f"Failed to kill job {job_id}. Server responded with status code {response.status_code}: {error_message}")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)


def inspect_job(job_id):
    url = f"{SERVER_URL}/get_job"
    try:
        response = requests.get(url, json={'job_id': job_id})
        if response.status_code == 200:
            job_info = response.json()
            if not job_info:
                print(f"Job with id {job_id} not found.")
                return
            # Print the job information
            pprint(job_info)
        else:
            print(f"Failed to retrieve job with id {job_id}.")
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
    # Create the parser
    parser = argparse.ArgumentParser(description="Local GPU Job Queue Client with Submitit")
    parser.add_argument('-startup', action='store_true', help='Launch the slite job manager')
    parser.add_argument('-list', action='store_true', help='List all jobs')
    parser.add_argument('-shutdown', action='store_true', help='Shutdown the scheduler server')
    parser.add_argument('-kill', metavar='JOB_ID', type=str, help='Kill a job with the given ID')
    parser.add_argument('-inspect', metavar='JOB_ID', type=str, help='Inspect a job with the given ID')
    parser.add_argument('-flush', metavar='STATUS', choices=['running', 'queued', 'all'], help='Flush jobs based on their status')

    # Parse arguments
    args = parser.parse_args()

    if args.startup:
        start_server()
    elif args.list:
        list_jobs()
    elif args.shutdown:
        shutdown_scheduler()
    elif args.kill:
        if not args.kill:
            parser.error("-kill requires a job ID")
        kill_job(args.kill)
    elif args.inspect:
        if not args.inspect:
            parser.error("-inspect requires a job ID")
        inspect_job(args.inspect)
    elif args.flush:
        if not args.flush:
            parser.error("-flush requires a status")
        flush_jobs(args.flush)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

