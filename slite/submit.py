# misc imports
import multiprocessing
from pydantic import validate_arguments
from typing import Any, List, Optional, Literal
# local imports
from .runner import SliteRunner
# ionpy imports
from ionpy.util import Config


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_exps(
    exp_root: str,
    exp_class: Any,
    config_list: List[Config],
    available_gpus: List[str]
):
    def launch_training():
        # Create a runner
        runner = SliteRunner(
            exp_root=exp_root,
            exp_class=exp_class,
            available_gpus=available_gpus,
        )
        # Submit the experiments
        runner.submit_exps(config_list)
    # Launch the training
    process = multiprocessing.Process(target=launch_training)
    process.start()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_jobs(
    exp_root: str,
    job_func: Any,
    config_list: List[Any],
    available_gpus: List[str]
):
    def launch_training():
        # Create a runner
        runner = SliteRunner(
            exp_root=exp_root,
            available_gpus=available_gpus,
        )
        # Submit the experiments
        runner.submit_jobs(job_func, config_list)
    # Launch the training
    process = multiprocessing.Process(target=launch_training)
    process.start()