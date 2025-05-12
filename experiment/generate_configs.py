# ionpy imports
from ionpy.util.ioutil import autosave
from ionpy.util import Config, dict_product
from ionpy.experiment.util import generate_tuid
from ionpy.util.config import check_missing, HDict, valmap, config_digest
# misc imports
import os
import ast
import yaml
import itertools
import numpy as np
from pathlib import Path
from pprint import pprint
from datetime import datetime
from pydantic import validate_arguments
from typing import List, Any, Optional, Callable


def tuplize_str_dict(d):
    for key, val in d.items():
        # If the value is NOT a dict, then we need to check if 
        # it a tuple hiding as a string.
        if not isinstance(val, dict):
            if isinstance(val, list):
               for vi_idx, val_item in enumerate(val):
                    if isinstance(val_item, dict):
                        d[key][vi_idx] = tuplize_str_dict(val_item)
                    elif isinstance(val_item, str) and val_item[0] == '(' and val_item[-1] == ')':
                        d[key][vi_idx] = ast.literal_eval(val_item)
            elif isinstance(val, str) and val[0] == '(' and val[-1] == ')':
                d[key] = ast.literal_eval(val)
        else:
            d[key] = tuplize_str_dict(val)
    return d


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_training_configs(
    exp_cfg: dict,
    default_cfg: Config,
    base_cfg_list: List[str],
    config_root: Path,
    scratch_root: Path,
    add_date: bool = True
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    train_exp_root = get_exp_root(exp_name, group="training", add_date=add_date, scratch_root=scratch_root)

    # Flatten the experiment config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)

    # Load in the different base configs.
    base_config_list = []
    for base_cfg_file in base_cfg_list:
        with open(f"{config_root}/{base_cfg_file}", 'r') as base_file:
            base_cfg_dict = yaml.safe_load(base_file)
        base_config_list.append(default_cfg.update([base_cfg_dict]))

    # Add the dataset specific details.
    if 'data._class' in flat_exp_cfg_dict:
        train_dataset_name = flat_exp_cfg_dict['data._class'].split('.')[-1]
        dataset_cfg_file = config_root / "training" / f"{train_dataset_name}.yaml"
        if dataset_cfg_file.exists():
            for cfg_idx, base_cfg in enumerate(base_config_list):
                with open(dataset_cfg_file, 'r') as d_file:
                    dataset_train_cfg = yaml.safe_load(d_file)
                # Update the base config with the dataset specific config.
                base_config_list[cfg_idx] = base_cfg.update([dataset_train_cfg])
        else:
            print(f"Warning: No dataset specific train config found for {train_dataset_name}.")
    
    # Get the information about seeds.
    seed = flat_exp_cfg_dict.pop('experiment.seed', 40)
    seed_range = flat_exp_cfg_dict.pop('experiment.seed_range', 1)

    # This is a required key. We want to get all of the models and vary everything else.
    pretrained_dir_list = flat_exp_cfg_dict.pop('model.pretrained_dir', None) 
    if pretrained_dir_list is not None:
        if isinstance(pretrained_dir_list, tuple):
            pretrained_dir_list = list(pretrained_dir_list)
        elif not isinstance(pretrained_dir_list, list):
            pretrained_dir_list = [pretrained_dir_list]
        flat_exp_cfg_dict['model.pretrained_dir'] = gather_pretrained_models(pretrained_dir_list) 

    # Create the ablation options.
    option_set = {
        'log.root': [str(train_exp_root)],
        'experiment.seed': [seed + seed_idx for seed_idx in range(seed_range)],
        **listify_dict(flat_exp_cfg_dict)
    }

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg_list=base_config_list)
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_inference_configs(
    exp_cfg: dict,
    default_cfg: Config,
    config_root: Path,
    scratch_root: Path,
    add_date: bool = True,
    base_cfg_list: Optional[List[str]] = None,
):
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    # Save the experiment config.
    group_str = exp_cfg.pop('group')
    sub_group_str = exp_cfg.pop('subgroup', "")
    exp_name = f"{group_str}/{sub_group_str}"

    # Get the root for the inference experiments.
    inference_log_root = get_exp_root(exp_name, group="inference", add_date=add_date, scratch_root=scratch_root)

    # Flatten the config.
    flat_exp_cfg_dict = flatten_cfg2dict(exp_cfg)
    # For any key that is a tuple we need to convert it to a list, this is an artifact of the flattening..
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, tuple):
            flat_exp_cfg_dict[key] = list(val)

    # Sometimes we want to do a range of values to sweep over, we will know this by ... in it.
    for key, val in flat_exp_cfg_dict.items():
        if isinstance(val, list):
            for idx, val_list_item in enumerate(val):
                if isinstance(val_list_item, str) and '...' in val_list_item:
                    # Replace the string with a range.
                    flat_exp_cfg_dict[key][idx] = get_range_from_str(val_list_item)
        elif isinstance(val, str) and  '...' in val:
            # Finally stick this back in as a string tuple version.
            flat_exp_cfg_dict[key] = get_range_from_str(val)

    # Gather the different config options.
    cfg_opt_keys = list(flat_exp_cfg_dict.keys())
    #First going through and making sure each option is a list and then using itertools.product.
    for ico_key in flat_exp_cfg_dict:
        if not isinstance(flat_exp_cfg_dict[ico_key], list):
            flat_exp_cfg_dict[ico_key] = [flat_exp_cfg_dict[ico_key]]
    
    # Generate product tuples 
    product_tuples = list(itertools.product(*[flat_exp_cfg_dict[key] for key in cfg_opt_keys]))

    # Convert product tuples to dictionaries
    total_run_cfg_options = [{cfg_opt_keys[i]: [item[i]] for i in range(len(cfg_opt_keys))} for item in product_tuples]

    # Define the set of default config options.
    default_config_options = {
        'experiment.exp_name': [exp_name],
        'experiment.exp_root': [str(inference_log_root)],
    }
    # Accumulate a set of config options for each dataset
    dataset_cfgs = []
    # Iterate through all of our inference options.
    for run_opt_dict in total_run_cfg_options: 
        model_set = gather_pretrained_models(run_opt_dict.pop('base_model')) 
        # Append these to the list of configs and roots.
        dataset_cfgs.append({
            'log.root': [str(inference_log_root)],
            'experiment.model_dir': model_set,
            **run_opt_dict,
            **default_config_options
        })
    # Keep a list of all the run configuration options.
    cfgs = []
    # Iterate over the different config options for this dataset. 
    for option_dict in dataset_cfgs:
        for exp_cfg_update in dict_product(option_dict):
            # Add the inference dataset specific details.
            dataset_inf_cfg_dict = get_inference_dset_info(
                cfg=exp_cfg_update,
                config_root=config_root
            )
            # Update the base config with the new options. Note the order is important here, such that 
            # the exp_cfg_update is the last thing to update.
            if base_cfg_list is not None:
                cfg = Config(dataset_inf_cfg_dict).update([base_cfg_list, exp_cfg_update])
            else:
                cfg = Config(dataset_inf_cfg_dict).update([exp_cfg_update]) 
            # Update the base config with the dataset specific config.
            cfg = default_cfg.update([cfg])
            # Make sure that we don't have any tuples.
            cfg_dict = cfg.to_dict()
            tuplized_cfg = Config(tuplize_str_dict(cfg_dict))
            # Verify it's a valid config
            check_missing(tuplized_cfg)
            # Add it to the total list of inference options.
            cfgs.append(tuplized_cfg)
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return cfgs


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_restart_configs(
    exp_cfg: dict,
    default_cfg: Config,
    config_root: Path,
    scratch_root: Path,
    add_date: bool = True,
): 
    # We need to flatten the experiment config to get the different options.
    # Building new yamls under the exp_name name for model type.
    exp_name = exp_cfg.pop('group')
    restart_exp_root = get_exp_root(exp_name, group="training", add_date=add_date, scratch_root=scratch_root)

    # Get the flat version of the experiment config.
    restart_cfg_dict = flatten_cfg2dict(exp_cfg)

    # If we are changing aspects of the dataset, we need to update the base config.
    if 'data._class' in restart_cfg_dict:
        # Add the dataset specific details.
        dataset_cfg_file = config_root / 'training' / f"{restart_cfg_dict['data._class'].split('.')[-1]}.yaml"
        if dataset_cfg_file.exists():
            with open(dataset_cfg_file, 'r') as d_file:
                dataset_train_cfg = yaml.safe_load(d_file)
            # Update the base config with the dataset specific config.
            base_cfg = base_cfg.update([dataset_train_cfg])
        
    # This is a required key. We want to get all of the models and vary everything else.
    pretrained_dir_list = restart_cfg_dict.pop('train.pretrained_dir') 
    all_pre_models = gather_pretrained_models(pretrained_dir_list) 

    # Listify the dict for the product.
    option_set = {
        'log.root': [str(restart_exp_root)],
        **listify_dict(restart_cfg_dict)
    }
    
    # Go through all the pretrained models and add the new options for the restart.
    pt_base_cfgs = []
    for pt_dir in all_pre_models:
        # Load the pre-trained model config.
        with open(f"{pt_dir}/config.yml", 'r') as file:
            pt_exp_cfg = Config(yaml.safe_load(file))
        # Make a copy of the listy_pt_cfg_dict.
        pt_default_cfg = default_cfg.update([pt_exp_cfg])
        # Turn this into a dict.
        pt_default_cfg_dict = pt_default_cfg.to_dict()
        pt_default_cfg_dict['train']['pretrained_dir'] = pt_dir
        pt_base_cfgs.append(Config(pt_default_cfg_dict))

    # Get the configs
    cfgs = get_option_product(exp_name, option_set, base_cfg_list=pt_base_cfgs)
    # Finally, generate the uuid that identify each of the configs.
    cfgs = generate_config_uuids(cfgs)

    return cfgs


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


def flatten_cfg2dict(cfg: Config):
    cfg = HDict(cfg)
    flat_exp_cfg = valmap(list2tuple, cfg.flatten())
    return flat_exp_cfg


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
    base_cfg_list: List[Config],
    add_wandb_string: bool = True
):
    # If option_set is not a list, make it a list
    cfg_list = []
    # Get all of the keys that have length > 1 (will be turned into different options)
    varying_keys = [key for key, value in option_set.items() if len(value) > 1]
    # Iterate through all of the different options
    for base_cfg in base_cfg_list:
        for cfg_update in dict_product(option_set):
            # If one of the keys in the update is a dictionary, then we need to wrap
            # it in a list, otherwise the update will collapse the dictionary.
            for key in cfg_update:
                if isinstance(cfg_update[key], dict):
                    cfg_update[key] = [cfg_update[key]]
            # Get the name that will be used for WANDB tracking and update the base with
            # this version of the experiment.
            if add_wandb_string:
                cfg_name_args = proc_cfg_name(exp_name, varying_keys, cfg_update)
                cfg = base_cfg.update([cfg_update, cfg_name_args])
            else:
                cfg = base_cfg.update([cfg_update])
            # Verify it's a valid config
            check_missing(cfg)
            # Make sure that our string tuples are converted to actual tuples.
            cfg_dict = cfg.to_dict()
            tuplized_cfg = Config(tuplize_str_dict(cfg_dict))
            # Finally, we need to 'tuplize' it, which means that we conver the tuples 
            # hiding as string into actual tuples.
            cfg_list.append(tuplized_cfg)
    return cfg_list


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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def gather_pretrained_models(directories: List[str]) -> List[str]:
    """
    Given a list of directory paths, returns those directories that contain a `checkpoints`
    subdirectory somewhere within their directory tree.
    
    Args:
        directories (List[str]): A list of directory paths.
    
    Returns:
        List[str]: A list of directories that have a `checkpoints` subdirectory at some depth.
    """
    valid_dirs = []
    for d in directories:
        for root, sub_d, _ in os.walk(d):
            if "checkpoints" in sub_d:
                valid_dirs.append(root)
    assert len(valid_dirs) > 0, f"No valid pretrained models found in {directories}."
    return valid_dirs


def get_range_from_str(val):
    trimmed_range = val[1:-1] # Remove the parantheses on the ends.
    range_args = trimmed_range.split(',')
    assert len(range_args) == 4, f"Range sweeping requires format like (start, ..., end, interval). Got {len(range_args)}."
    arg_vals = np.arange(float(range_args[0]), float(range_args[2]), float(range_args[3]))
    # Finally stick this back in as a string tuple version.
    return str(tuple(arg_vals))


def get_inference_dset_info(
    cfg,
    config_root 
):
    # Total model config
    base_model_cfg = yaml.safe_load(open(f"{cfg['experiment.model_dir']}/config.yml", "r"))

    # Get the data config from the model config.
    base_data_cfg = base_model_cfg["data"]

    # We need to remove a few keys that are not needed for inference.
    drop_keys = [
        "iters_per_epoch",
    ]
    for d_key in drop_keys:
        if d_key in base_data_cfg:
            base_data_cfg.pop(d_key)

    # Get the dataset name, and load the base inference dataset config for that.
    inf_dset_name = cfg.get('inference_data._class', "").split('.')[-1]
    # Add the dataset specific details.
    inf_dset_cfg_file = config_root / "inference" / f"{inf_dset_name}.yaml"
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