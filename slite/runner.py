# Slite imports
import slite.runner as slunner
import slite.submit as slubmit 
# misc imports
from pydantic import validate_arguments
from typing import List, Optional, Any, Callable
# Local imports
from .utils import log_exp_config_objs, submit_input_check, pop_wandb_callback


@validate_arguments
def run_job(
    gpu: str,
    config: Any,
    job_func: Callable,
    track_wandb: bool = False,
    show_examples: bool = False,
    run_name: Optional[str] = None,
):
    # Get the config as a dictionary.
    cfg = config.to_dict()
    cfg["log"]["show_examples"] = show_examples
    # If run num undefined, make a substitute.
    if run_name is not None:
        log_dir_root = "/".join(cfg["log"]["root"].split("/")[:-1])
        cfg["log"]["root"] = "/".join([log_dir_root, run_name])
    # Modify a few things relating to callbacks.
    if "callbacks" in cfg:
        # If you don't want to show examples, then remove the step callback.
        if not show_examples and "step" in cfg["callbacks"]:
            cfg["callbacks"].pop("step")
        # If not tracking wandb, remove the callback if its in the config.
        if not track_wandb:
            pop_wandb_callback(cfg)
    # Either run the experiment or the job function.
    run_args = {
        "config": cfg,
        "available_gpus": gpu,
    }
    slunner.run_job(
        job_func=job_func,
        **run_args
    )


@validate_arguments
def run_exp(
    gpu: str,
    config: Any,
    experiment_class: Any,
    track_wandb: bool = False,
    show_examples: bool = False,
    run_name: Optional[str] = None,
):
    # Get the config as a dictionary.
    cfg = config.to_dict()
    cfg["log"]["show_examples"] = show_examples
    # If run num undefined, make a substitute.
    if run_name is not None:
        log_dir_root = "/".join(cfg["log"]["root"].split("/")[:-1])
        cfg["log"]["root"] = "/".join([log_dir_root, run_name])
    # Modify a few things relating to callbacks.
    if "callbacks" in cfg:
        # If you don't want to show examples, then remove the step callback.
        if not show_examples and "step" in cfg["callbacks"]:
            cfg["callbacks"].pop("step")
        # If not tracking wandb, remove the callback if its in the config.
        if not track_wandb:
            pop_wandb_callback(cfg)
    # Either run the experiment or the job function.
    run_args = {
        "config": cfg,
        "available_gpus": gpu,
    }
    slunner.run_exp(
        exp_class=experiment_class,
        **run_args
    )


@validate_arguments
def submit_exps(
    cfg_list: List,
    exp_cfg: dict,
    base_cfg: dict,
    submit_cfg: dict,
    experiment_class: Any
):
    # Save the experiment configs so we can know what we ran.
    log_exp_config_objs(
        exp_cfg=exp_cfg, 
        base_cfg=base_cfg, 
        submit_cfg=submit_cfg,
    )
    # Modify a few things relating to callbacks.
    modified_cfgs = [] 
    for config in cfg_list:
        cfg = config.to_dict()
        if "callbacks" in cfg:
            # Get the config as a dictionary.
            # Remove the step callbacks because it will slow down training.
            if "step" in cfg["callbacks"]:
                cfg["callbacks"].pop("step")
            # If you don't want to track wandb, then remove the wandb callback.
            # If not tracking wandb, remove the callback if its in the config.
            if not submit_cfg['track_wandb']:
                pop_wandb_callback(cfg)
        # Add the modified config to the list.
        modified_cfgs.append(cfg)
    # Run the set of configs.
    slubmit.submit_jobs(
        exp_class=experiment_class,
        submit_cfg=submit_cfg,
        config_list=modified_cfgs
    )


@validate_arguments
def submit_jobs(
    cfg_list: List,
    exp_cfg: dict,
    base_cfg: dict,
    submit_cfg: dict,
    job_func: Callable
):
    # Save the experiment configs so we can know what we ran.
    log_exp_config_objs(
        exp_cfg=exp_cfg, 
        base_cfg=base_cfg, 
        submit_cfg=submit_cfg,
    )
    # Modify a few things relating to callbacks.
    modified_cfgs = [] 
    for config in cfg_list:
        cfg = config.to_dict()
        if "callbacks" in cfg:
            # Get the config as a dictionary.
            # Remove the step callbacks because it will slow down training.
            if "step" in cfg["callbacks"]:
                cfg["callbacks"].pop("step")
            # If you don't want to track wandb, then remove the wandb callback.
            # If not tracking wandb, remove the callback if its in the config.
            if not submit_cfg['track_wandb']:
                pop_wandb_callback(cfg)
        # Add the modified config to the list.
        modified_cfgs.append(cfg)
    # Run the set of configs.
    slubmit.submit_jobs(
        job_func=job_func,
        submit_cfg=submit_cfg,
        config_list=modified_cfgs
    )