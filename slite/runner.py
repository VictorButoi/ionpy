# misc imports
import os
import sys
import pathlib
import submitit
from typing import List, Any, Optional
from pydantic import validate_arguments
# ionpy imports
from ionpy.util import Config


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_exp(
    exp_class: Any,
    config: Config,
    available_gpus: int
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    # Get the experiment class, either fresh or from a path.
    exp = exp_class.from_config(config)
    # Run the experiment.
    exp.run()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_job(
    job_func: Any,
    config: Config,
    available_gpus: int
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    job_func(config)


class SliteRunner:

    def __init__(
            self, 
            exp_root: str,
            available_gpus: List[str],
            exp_class: Optional[Any] = None,
            ):
        # Configure Submitit object
        self.exp_root = exp_root 
        self.avail_gpus = available_gpus
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
        print("Initalized SliteRunner")

    def set_exp_name(self, exp_name):
        self.exp_name = exp_name
        self.init_executor()

    def submit_exps(
        self,
        exp_configs: List[Config]
    ):
        num_gpus = len(self.avail_gpus)
        # Keep track of the local jobs
        local_job_list = []
        for c_idx, config in enumerate(exp_configs):
            c_gpu = self.avail_gpus[c_idx % num_gpus]
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
        return local_job_list
    
    def submit_jobs(
        self,
        job_func: Any,
        job_cfgs: List[Any] = None
    ):
        # Keep track of the local jobs
        local_job_list = []

        # Chunk the job_cfgs list in len(self.avail_gpus) many lists of job_cfgs
        # as uniformly distributed as possible.
        num_used_gpus = min(len(job_cfgs), len(self.avail_gpus))
        used_avail_gpus = self.avail_gpus[:num_used_gpus]

        job_chunks = {gpu: [] for gpu in used_avail_gpus}
        for j_idx, cfg in enumerate(job_cfgs):
            job_chunks[str(used_avail_gpus[j_idx % len(used_avail_gpus)])].append(cfg)

        for gpu in used_avail_gpus:
            # Submit the job
            job = self.executor.submit(
                run_jobs,
                job_func=job_func,
                cfg_list=job_chunks[gpu],
                available_gpus=gpu
            )
            print(f"Submitted job id: {job.job_id}.")
            self.jobs.append(job)
            local_job_list.append(job)
        
        return local_job_list

    def kill_jobs(
        self,
        job_list = None
    ):
        if job_list is None:
            job_list = self.jobs
            
        for job in job_list:
            job.cancel()
