# misc imports
import multiprocessing
from pydantic import validate_arguments
from typing import Any, List, Optional, Literal

# local imports
from .runner import SliteRunner


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_exps(
    project: str,
    exp_name: str,
    exp_class: Any,
    available_gpus: List[str],
    config_list: Optional[List[Any]] = None,
    exp_path_list: Optional[List[Any]] = None,
):
    assert exp_name != "debug", "Cannot launch debug jobs large-scale."
    assert not (config_list is None and exp_path_list is None), "Must provide either a list of configs or a list of experiment paths."
    assert not (config_list is not None and exp_path_list is not None), "Cannot provide both a list of configs and a list of experiment paths."

    def launch_training():
        # Create a runner
        runner = SliteRunner(
            project=project,
            exp_class=exp_class,
            available_gpus=available_gpus,
            exp_name=exp_name
        )
        # Submit the experiments
        if config_list is not None: # For training from scratch.
            runner.submit_exps(config_list)
        elif exp_path_list is not None: # For fine-tuning previously trained models.
            runner.submit_exps(exp_path_list)

    # Launch the training
    process = multiprocessing.Process(target=launch_training)
    process.start()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_jobs(
    project: str,
    exp_name: str,
    job_func: Any,
    config_list: List[Any],
    available_gpus: List[str]
):
    assert exp_name != "debug", "Cannot launch debug jobs large-scale."

    def launch_training():
        # Create a runner
        runner = SliteRunner(
            project=project,
            available_gpus=available_gpus,
            exp_name=exp_name,
        )
        # Submit the experiments
        runner.submit_jobs(job_func, config_list)

    # Launch the training
    process = multiprocessing.Process(target=launch_training)
    process.start()