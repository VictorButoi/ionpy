from typing import Any, List
from .runner import SliteRunner
import multiprocessing


def submit_exps(
    project: str,
    exp_name: str,
    exp_class: Any,
    available_gpus: List[str],
    config_list: List[Any]
):
    assert exp_name != "debug", "Cannot launch debug jobs large-scale."

    def launch_training():
        # Create a runner
        runner = SliteRunner(
            project=project,
            exp_class=exp_class,
            available_gpus=available_gpus,
            exp_name=exp_name
        )

        # Submit the experiments
        runner.submit_exps(config_list)

    # Launch the training
    process = multiprocessing.Process(target=launch_training)
    process.start()