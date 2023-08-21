import os
import time
import sys


def task(c_idx, cfg_group, exp_type, available_gpus):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')

    gpu_idx = c_idx % len(available_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[gpu_idx])
    for cfg in cfg_group:
        try:
            exp = exp_type.from_config(cfg)
            exp.run()
        except Exception as e:
            print(e)
            time.sleep(10)
            continue


def chunk_cfs(cfg_list, num_gpus):
    chunk_size = len(cfg_list) // num_gpus
    overflow_jobs = len(cfg_list) % num_gpus
    job_counts = [cfg_list[chunk_size*i:(chunk_size*i+ + chunk_size)] for i in range(num_gpus)]
    if overflow_jobs > 0:
        for i in range(overflow_jobs):
            job_counts[i].append(cfg_list[-(i+1)])
    return job_counts


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
