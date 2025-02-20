# misc imports
import os
import ast
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from typing import Literal
from datetime import datetime
from pydantic import validate_arguments
from itertools import chain, combinations
# Ionpy imports
from ionpy.util.meter import MeterDict
from ionpy.util.ioutil import autosave
from ionpy.util import Config, dict_product
from ionpy.experiment.util import generate_tuid
from ionpy.util.config import check_missing, HDict, valmap, config_digest


def gather_exp_paths(root):
    # For ensembles, define the root dir.
    run_names = os.listdir(root)
    # NOTE: Not the best way to do this, but we need to skip over some files/directories.
    skip_items = [
        "submitit",
        "wandb",
        "base.yml",
        "experiment.yml"
    ]
    # Filter out the skip_items
    valid_exp_paths = []
    for run_name in run_names:
        run_dir = f"{root}/{run_name}"
        # Make sure we don't include the skip items and that we actually have valid checkpoints.
        if (run_name not in skip_items) and os.path.isdir(f"{run_dir}/checkpoints"):
            valid_exp_paths.append(run_dir)
    # Return the valid experiment paths.
    return valid_exp_paths


def proc_cfg_name(
    exp_name,
    varying_keys,
    cfg
):
    params = []
    params.append("exp_name:" + exp_name)
    for key, value in cfg.items():
        if key in varying_keys:
            if key not in ["log.root", "train.pretrained_dir"]:
                key_name = key.split(".")[-1]
                short_value = str(value).replace(" ", "")
                if key_name == "exp_name":
                    params.append(str(short_value))
                else:
                    params.append(f"{key_name}:{short_value}")
    wandb_string = "-".join(params)
    return {"log.wandb_string": wandb_string}


def get_option_product(
    exp_name,
    option_set,
    base_cfg
):
    # If option_set is not a list, make it a list
    cfgs = []
    # Get all of the keys that have length > 1 (will be turned into different options)
    varying_keys = [key for key, value in option_set.items() if len(value) > 1]
    # Iterate through all of the different options
    for cfg_update in dict_product(option_set):
        # If one of the keys in the update is a dictionary, then we need to wrap
        # it in a list, otherwise the update will collapse the dictionary.
        for key in cfg_update:
            if isinstance(cfg_update[key], dict):
                cfg_update[key] = [cfg_update[key]]
        # Get the name that will be used for WANDB tracking and update the base with
        # this version of the experiment.
        cfg_name_args = proc_cfg_name(exp_name, varying_keys, cfg_update)
        cfg = base_cfg.update([cfg_update, cfg_name_args])
        # Verify it's a valid config
        check_missing(cfg)
        cfgs.append(cfg)
    return cfgs


def listify_dict(d):
    listy_d = {}
    # We need all of our options to be in lists as convention for the product.
    for ico_key in d:
        # If this is a tuple, then convert it to a list.
        if isinstance(d[ico_key], tuple):
            listy_d[ico_key] = list(d[ico_key])
        # Otherwise, make sure it is a list.
        elif not isinstance(d[ico_key], list):
            listy_d[ico_key] = [d[ico_key]]
        else:
            listy_d[ico_key] = d[ico_key]
    # Return the listified dictionary.
    return listy_d


def flatten_cfg2dict(cfg: Config):
    cfg = HDict(cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    return flat_exp_cfg


def power_set(in_set):
    return list(chain.from_iterable(combinations(in_set, r) for r in range(len(in_set)+1)))


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def get_exp_root(exp_name, group, add_date, scratch_root):
    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        exp_name = f"{formatted_date}_{exp_name}"
    # Save the experiment config.
    return scratch_root / group / exp_name


def log_exp_config_objs(
    group,
    base_cfg,
    exp_cfg, 
    add_date, 
    scratch_root
):
    # Get the experiment name.
    exp_name = f"{exp_cfg['group']}/{exp_cfg.get('subgroup', '')}"

    # Optionally, add today's date to the run name.
    if add_date:
        today_date = datetime.now()
        formatted_date = today_date.strftime("%m_%d_%y")
        mod_exp_name = f"{formatted_date}_{exp_name}"
    else:
        mod_exp_name = exp_name

    # Save the experiment config.
    exp_root = scratch_root / group / mod_exp_name

    # Save the base config and the experiment config.
    autosave(base_cfg, exp_root / "base.yml") # SAVE #1: Experiment config
    autosave(exp_cfg, exp_root / "experiment.yml") # SAVE #1: Experiment config


def add_dset_presets(
    mode: Literal["training", "calibrate", "inference"],
    inf_dset_name, 
    base_cfg, 
    code_root
):
    # Add the dataset specific details.
    dataset_cfg_file = code_root / "ese" / "configs" / mode / f"{inf_dset_name}.yaml"
    if dataset_cfg_file.exists():
        with open(dataset_cfg_file, 'r') as d_file:
            dataset_cfg = yaml.safe_load(d_file)
        # Update the base config with the dataset specific config.
        base_cfg = base_cfg.update([dataset_cfg])
    else:
        raise ValueError(f"Dataset config file not found: {dataset_cfg_file}")
    return base_cfg


def get_range_from_str(val):
    trimmed_range = val[1:-1] # Remove the parantheses on the ends.
    range_args = trimmed_range.split(',')
    assert len(range_args) == 4, f"Range sweeping requires format like (start, ..., end, interval). Got {len(range_args)}."
    arg_vals = np.arange(float(range_args[0]), float(range_args[2]), float(range_args[3]))
    # Finally stick this back in as a string tuple version.
    return str(tuple(arg_vals))


def get_inference_dset_info(
    cfg,
    code_root
):
    # Total model config
    base_model_cfg = yaml.safe_load(open(f"{cfg['experiment.model_dir']}/config.yml", "r"))

    # Get the data config from the model config.
    base_data_cfg = base_model_cfg["data"]
    # We need to remove a few keys that are not needed for inference.
    drop_keys = [
        "iters_per_epoch",
        "train_kwargs",
        "val_kwargs",
    ]
    for d_key in drop_keys:
        if d_key in base_data_cfg:
            base_data_cfg.pop(d_key)

    # Get the dataset name, and load the base inference dataset config for that.
    inf_dset_cls = cfg['inference_data._class']

    inf_dset_name = inf_dset_cls.split('.')[-1]
    # Add the dataset specific details.
    inf_dset_cfg_file = code_root / "ese" / "configs" / "inference" / f"{inf_dset_name}.yaml"
    if inf_dset_cfg_file.exists():
        with open(inf_dset_cfg_file, 'r') as d_file:
            inf_cfg_presets = yaml.safe_load(d_file)
    else:
        inf_cfg_presets = {}
    # Assert that 'version' is not defined in the base_inf_dataset_cfg, this is not allowed behavior.
    assert 'version' not in inf_cfg_presets.get("inference_data", {}), "Version should not be defined in the base inference dataset config."

    # NOW WE MODIFY THE ORIGINAL BASE DATA CFG TO INCLUDE THE INFERENCE DATASET CONFIG.

    # We need to modify the inference dataset config to include the data_cfg.
    inf_dset_presets = inf_cfg_presets.get("inference_data", {})

    # Now we update the trained model config with the inference dataset config.
    new_inf_dset_cfg = base_data_cfg.copy()
    new_inf_dset_cfg.update(inf_dset_presets)
    # And we put the updated data_cfg back into the inf_cfg_dict.
    inf_cfg_presets["inference_data"] = new_inf_dset_cfg

    # Return the data_cfg and the base_inf_dataset_cfg
    return inf_cfg_presets


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def generate_config_uuids(config_list: List[Config]):
    processed_cfgs = []
    for config in config_list:
        if isinstance(config, HDict):
            config = config.to_dict()
        create_time, nonce = generate_tuid()
        digest = config_digest(config)
        config['log']['uuid'] = f"{create_time}-{nonce}-{digest}"
        # Append the updated config to the processed list.
        processed_cfgs.append(Config(config))
    return processed_cfgs


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def save_records(records, log_dir):
    # Save the items in a pickle file.  
    df = pd.DataFrame(records)
    # Save or overwrite the file.
    df.to_pickle(log_dir)


def save_dict(dict, log_dir):
    # Check if log+dir exists, if not create it.
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    # save the dictionary to a pickl file at logdir
    with open(log_dir, 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


def save_trackers(output_root, trackers):
    for key, tracker in trackers.items():
        out_dir = output_root / f"{key}.pkl"
        # In this case, we need to save the tracker as a dictionary.
        if isinstance(tracker, MeterDict):
            tracker = tracker.asdict()
            save_dict(tracker, out_dir) 
        # If our tracker is a dict, then it is a dictionary
        # of meter dicts, so we need to unpack and save them 
        # as a dataframe
        elif isinstance(tracker, dict):
            # There is a special case where our tracker is a dictionary of
            # meter dicts. In this case, we need to do a bit of processing.
            unpacked_meter_dicts = []
            for metadata_key, tracker_obj in tracker.items():
                assert isinstance(tracker_obj, MeterDict), "Expected a MeterDict object."
                metadata_dict = ast.literal_eval(metadata_key)
                for metric_name, met_dict in tracker_obj.asdict().items():
                    metadata_dict[metric_name] = met_dict['mean'] 
                    metadata_dict[f"{metric_name}_std"] = met_dict['std']
                # Append the metadata dict to the list of metadata dicts.
                unpacked_meter_dicts.append(metadata_dict)
            # Finally, we save these as records.
            save_records(unpacked_meter_dicts, out_dir)
        # Otherwise, it is a list or a dataframe and we can 
        # save it if it is not empty.
        else:
            if len(tracker) > 0:
                if isinstance(tracker, pd.DataFrame):
                    tracker.to_pickle(out_dir)
                else:
                    save_records(tracker, out_dir)

# This function will take in a dictionary of pixel meters and a metadata dataframe
# from which to select the log_set corresponding to particular attributes, then 
# we index into the dictionary to get the corresponding pixel meters.
def select_pixel_dict(pixel_meter_logdict, metadata, kwargs):
    # Select the metadata
    metadata = metadata.select(**kwargs)
    # Get the log set
    assert len(metadata) == 1, f"Need exactly 1 log set, found: {len(metadata)}."
    log_set = metadata['log_set'].iloc[0]
    # Return the pixel dict
    return pixel_meter_logdict[log_set]
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def save_inference_metadata(cfg_dict, save_root: Optional[Path] = None):
    if save_root is None:
        save_root = cfg_dict['log']['root']
        inference_cfg_uuid = cfg_dict['log']['uuid']
        # Prepare the output dir for saving the results
        path = Path(f'{save_root}/{inference_cfg_uuid}')
        create_time, nonce, digest = inference_cfg_uuid.split("-")
        # Save the metadata.
        metadata = {"create_time": create_time, "nonce": nonce, "digest": digest}
        autosave(metadata, path / "metadata.json")
    else:
        path = save_root
    # Save the config.
    autosave(cfg_dict, path / "config.yml")
    return path


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_kwarg_sweep(
    inf_cfg_dict: dict
):
    # If there are inference kwargs, then we need to do a forward pass with those kwargs.
    inf_kwarg_opts = inf_cfg_dict['experiment'].get('inf_kwargs', None)
    if inf_kwarg_opts is not None:
        # Go through each, and if they are strings representing tuples, then we need to convert them to tuples.
        for inf_key, inf_val in inf_kwarg_opts.items(): 
            # Ensure this is a list.
            if not isinstance(inf_val, list):
                inf_kwarg_opts[inf_key] = [inf_val]
        # Now we need to do a grid of the options, similar to how we build configs.
        return list(dict_product(inf_kwarg_opts))
    else:
        return [{}]