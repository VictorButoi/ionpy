from .utils import task

# misc imports
import os
import submitit
import time
from typing import List, Any


class SliteRunner:

    def __init__(
            self, 
            project: str,
            task_type: Any,
            exp_name: str = None, 
            available_gpus: List[str] = ['0'], 
            log_root_dir: str='/storage/vbutoi/scratch'
            ):

        # Configure Submitit object
        self.project_name = project 
        self.exp_name = exp_name
        self.log_root_dir = log_root_dir
        self.avail_gpus = available_gpus
        self.task_type = task_type

        # Initalize executor if not none
        if exp_name is not None:
            self.init_executor()

        # Keep cache of jobs
        self.jobs = []
    
    def init_executor(self):
        submitit_root = f"{self.log_root_dir}/{self.project_name}/{self.exp_name}/submitit"
        self.executor = submitit.LocalExecutor(folder=submitit_root)
        self.executor.parameters['visible_gpus'] = self.avail_gpus
        self.executor.parameters['timeout_min'] = int(24 * 60 * 7)
        print("Initalized SliteRunner")

    def set_exp_name(self, exp_name):
        self.exp_name = exp_name
        self.init_executor()
    
    def submit_exps(self, cfg_list):
        assert self.exp_name is not None, "Must set exp_name before running experiment."
        for cfg in cfg_list:
            job = self.executor.submit(task, cfg, self.task_type, self.avail_gpus)
            print(f"Submitted job {job.job_id}.")
            self.jobs.append(job)

    def run_exp(self, cfg):
        assert self.exp_name is not None, "Must set exp_name before running experiment."
        # Make sure GPUs are visible
        gpu_list = ','.join(self.avail_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        # Run experiment
        exp = self.task_type.from_config(cfg)
        exp.run()

    def kill_jobs(self):
        return None