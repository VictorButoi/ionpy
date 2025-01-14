
# ionpy imports
from datetime import datetime
from .util.ioutil import autosave
# misc imports
from pathlib import Path
from datetime import datetime
from pydantic import validate_arguments
from typing import Any, Optional, Callable


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def submit_input_check(
    experiment_class: Optional[Any] = None,
    job_func: Optional[Callable] = None
):
    use_exp_class = (experiment_class is not None)
    use_job_func = (job_func is not None)
    # xor images_defined pixel_preds_defined
    assert use_exp_class ^ use_job_func,\
        "Exactly one of experiment_class or job_func must be defined,"\
             + " but got experiment_clss defined = {} and job_func defined = {}.".format(\
            use_exp_class, use_job_func)


def log_exp_config_objs(
    exp_cfg: dict, 
    base_cfg: dict,
    submit_cfg: dict,
):
    # Get the experiment name.
    exp_name = f"{exp_cfg['group']}/{exp_cfg.get('subgroup', '')}"

    # Optionally, add today's date to the run name.
    if submit_cfg.get('add_date', True):
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        mod_exp_name = f"{formatted_date}_{exp_name}"
    else:
        mod_exp_name = exp_name

    # Save the experiment config.
    exp_root = Path(f"{submit_cfg['scratch_root']}/{submit_cfg['group']}/{mod_exp_name}")

    # Save the base config and the experiment config.
    autosave(base_cfg, exp_root / "base.yml") # SAVE #1: Experiment config
    autosave(exp_cfg, exp_root / "experiment.yml") # SAVE #1: Experiment config


def pop_wandb_callback(cfg):
    for epoch_callback in cfg["callbacks"]["epoch"]:
        if isinstance(epoch_callback, str) and epoch_callback.split(".")[-1] == "WandbLogger":
            cfg["callbacks"]["epoch"].remove(epoch_callback)
        elif isinstance(epoch_callback, dict) and list(epoch_callback.keys())[0].split(".")[-1] == "WandbLogger":
            cfg["callbacks"]["epoch"].remove(epoch_callback)