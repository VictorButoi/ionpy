import os
import sys
from ionpy.util.config import check_missing


def task(cfg, exp_type, available_gpus):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    exp = exp_type.from_config(cfg)
    exp.run()


def proc_exp_name(exp_name, cfg):
    params = []
    params.append("exp_name:" + exp_name)
    for key, value in cfg.items():
        if key != "log.root":
            key_name = key.split(".")[-1]
            short_value = str(value).replace(" ", "")
            params.append(f"{key_name}:{short_value}")
    wandb_string = "-".join(params)
    return {"log.wandb_string": wandb_string}


def validate_cfg(cfg):
    # It's usually a good idea to do a sanity check of
    # inter-related settings or force them manually
    check_missing(cfg)        
    return cfg