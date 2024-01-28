# misc imports
import os
import sys
import time
import pathlib
import submitit
from typing import List, Any, Optional
from pydantic import validate_arguments
# local imports
from ..util import Config
from .utils import get_most_free_gpu


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_exp(
    exp_class: Any,
    config: Config,
    available_gpus: Optional[int] = None,
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    if available_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    # Get the experiment class, either fresh or from a path.
    exp = exp_class.from_config(config)
    # Run the experiment.
    exp.run()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_job(
    job_func: Any,
    config: Config,
    available_gpus: Optional[int] = None 
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    if available_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    job_func(config)


class SliteRunner:

    def __init__(
        self, 
        exp_root: str,
        available_gpus: Optional[List[str]] = None,
        exp_class: Optional[Any] = None,
    ):
        # Configure Submitit object
        self.exp_root = exp_root 
        self.avail_gpus = available_gpus
        if self.avail_gpus is None:
            self.num_gpus = 0
        else:
            self.num_gpus = len(available_gpus)
        self.exp_class = exp_class 
        self.init_executor()
        # Keep cache of jobs
        self.jobs = []
    
    def init_executor(self):
        # Create submitit executor
        submitit_root = f"{self.exp_root}/submitit"
        # Setup the excutor parameters
        self.executor = submitit.LocalExecutor(folder=submitit_root)
        self.executor.parameters['visible_gpus'] = self.avail_gpus
        self.executor.parameters['timeout_min'] = int(24 * 60 * 7)

    def set_exp_name(self, exp_name):
        self.exp_name = exp_name
        self.init_executor()

    def submit_exps(
        self,
        exp_configs: List[Config],
        submission_delay: int = 2.0
    ):
        gpu_string = ','.join([str(g) for g in self.avail_gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_string)
        # Keep track of the local jobs
        local_job_list = []
        for config in exp_configs:
            if self.avail_gpus is not None:
                c_gpu = get_most_free_gpu(self.avail_gpus)
            else:
                c_gpu = None
            # Submit the job
            job = self.executor.submit(
                run_exp,
                exp_class=self.exp_class,
                config=config,
                available_gpus=c_gpu
            )
            print(f"Submitted job id: {job.job_id} on gpu: {c_gpu}.")
            self.jobs.append(job)
            local_job_list.append(job)
            # Delay the submission.
            time.sleep(submission_delay)

        return local_job_list
    
    def submit_jobs(
        self,
        job_func: Any,
        job_cfgs: List[Any],
        submission_delay: int = 2.0
    ):
        gpu_string = ','.join([str(g) for g in self.avail_gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_string)
        # Keep track of the local jobs
        local_job_list = []
        for config in job_cfgs:
            if self.avail_gpus is not None:
                c_gpu = get_most_free_gpu(self.avail_gpus)
            else:
                c_gpu = None
            # Submit the job
            job = self.executor.submit(
                run_job,
                job_func=job_func,
                config=config,
                available_gpus=c_gpu
            )
            print(f"Submitted job id: {job.job_id} on gpu: {c_gpu}.")
            self.jobs.append(job)
            local_job_list.append(job)
            # Delay the submission.
            time.sleep(submission_delay)

        return local_job_list

    def kill_jobs(
        self,
        job_list = None
    ):
        if job_list is None:
            job_list = self.jobs
            
        for job in job_list:
            job.cancel()
