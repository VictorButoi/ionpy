# misc imports
import sys
import requests
from typing import Any, List, Optional
from pydantic import validate_arguments


SERVER_URL = 'http://localhost:5000'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_jobs(
    config_list: List[Any],
    exp_class: Optional[Any] = None,
    job_func: Optional[Any] = None,
    available_gpus: Optional[List[str]] = None,
    submission_delay: int = 4.0
):
    # Precisely one of exp_class or job_func must be defined.
    assert (exp_class is not None) ^ (job_func is not None), \
        "Exactly one of exp_class or job_func must be defined."
    # We might have to convert exp_class and job_func to strings.
    if exp_class is not None:
        exp_class = str(exp_class)
    if job_func is not None:
        job_func = str(job_func)
    
    url = f"{SERVER_URL}/submit"
    payload = {
        'config_list': config_list,
        'exp_class': exp_class,
        'job_func': job_func,
        'available_gpus': available_gpus,
        'submission_delay': submission_delay 
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            job_id = response.json().get('job_id')
            print(f"Job {job_id} submitted.")
        else:
            print(f"Failed to submit job: {response.json().get('error')}")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the scheduler server. Is it running?")
        sys.exit(1)