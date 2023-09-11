from typing import Any, List
from .runner import SliteRunner


def submit_exps(
    project: str,
    exp_name: str,
    exp_class: Any,
    available_gpus: List[str],
    config_list: List[Any]
):
    assert exp_name != "debug", "Cannot launch debug jobs large-scale."

    # Create a runner
    runner = SliteRunner(
        project=project,
        exp_class=exp_class,
        available_gpus=available_gpus,
        exp_name=exp_name
    )

    # Submit the experiments
    jobs = runner.submit_exps(config_list)

    return jobs