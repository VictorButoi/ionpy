# misc imports
import multiprocessing
from typing import Any, List, Optional
from pydantic import validate_arguments
# local imports
from .runner import SliteRunner
# ionpy imports
from ionpy.util import Config


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_exps(
    exp_class: Any,
    config_list: List[Config],
    available_gpus: Optional[List[str]] = None
):
    for config in config_list:
        config_dict = config.to_dict()
        def launch_training():
            # Create a runner
            runner = SliteRunner(
                exp_root=config_dict["log"]["root"],
                exp_class=exp_class,
                available_gpus=available_gpus,
            )
            # Submit the experiments
            runner.submit_exps([config])
        # Launch the training
        process = multiprocessing.Process(target=launch_training)
        process.start()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_jobs(
    job_func: Any,
    config_list: List[Any],
    available_gpus: Optional[List[str]] = None
):
    for config in config_list:
        config_dict = config.to_dict()
        def launch_training():
            # Create a runner
            runner = SliteRunner(
                exp_root=config_dict["log"]["root"],
                available_gpus=available_gpus,
            )
            # Submit the experiments
            runner.submit_jobs(job_func, [config])
        # Launch the training
        process = multiprocessing.Process(target=launch_training)
        process.start()