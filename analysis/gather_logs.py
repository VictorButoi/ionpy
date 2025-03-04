# misc imports
import os
import yaml
import json
import pickle
import hashlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from pydantic import validate_arguments
# ionpy imports
from ionpy.util.config import HDict, valmap
# local imports
from .analysis_utils.inference_utils import verify_graceful_exit


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def hash_dictionary(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary, sort_keys=True)
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the JSON string encoded as bytes
    hash_object.update(json_str.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def hash_list(input_list):
    # Convert the list to a JSON string
    json_str = json.dumps(input_list, sort_keys=True)
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the JSON string encoded as bytes
    hash_object.update(json_str.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_flat_cfg(
    cfg_name: str, 
    cfg_dir: Path
):
    print("cfg_dir:", cfg_dir)
    with open(cfg_dir, 'r') as stream:
        logset_cfg_yaml = yaml.safe_load(stream)
    logset_cfg = HDict(logset_cfg_yaml)
    logset_flat_cfg = valmap(list2tuple, logset_cfg.flatten())
    # Add some keys which are useful for the analysis.
    logset_flat_cfg["log_set"] = cfg_name
    # For the rest of the keys, if the length of the value is more than 1, convert it to a string.
    for key in logset_flat_cfg:
        if isinstance(logset_flat_cfg[key], list) or isinstance(logset_flat_cfg[key], tuple):
            logset_flat_cfg[key] = str(logset_flat_cfg[key])
    # Return the flattened configuration.
    return logset_flat_cfg


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_dfs(
    results_cfg: dict,
    load_cached: bool,
    inference_dir: str
) -> dict:
    # Build a dictionary to store the inference info.
    log_cfg = results_cfg["log"] 

    # Get the hash of the results_cfg dictionary.
    results_cfg_hash = hash_dictionary(results_cfg)
    precomputed_results_path = inference_dir + "/results_cache/" + results_cfg_hash + ".pkl"

    # Skip over metadata folders
    skip_log_folders = [
        "debug",
        "wandb", 
        "submitit", 
        "base.yml"
    ]
    # We need to get the roots and inference groups from the log_cfg.
    log_roots = log_cfg["root"]
    log_inference_groups = log_cfg.get("inference_group", "")
    if isinstance(log_roots, str):
        log_roots = [log_roots]
    if isinstance(log_inference_groups, str):
        log_inference_groups = [log_inference_groups]

    # Check to see if we have already built the inference info before.
    if not load_cached or not os.path.exists(precomputed_results_path):
        # Gather inference log paths.
        all_inference_log_paths = []
        for root in log_roots:
            for inf_group in log_inference_groups:
                # If inf_group is None, then we are in the root directory.
                if inf_group is None:
                    inf_group_dir = root
                else:
                    inf_group_dir = root + inf_group
                print("inf_group_dir:", inf_group_dir)
                group_folders = os.listdir(inf_group_dir)
                # If 'submitit' is in the highest dir, then we don't have subdirs (new folder structure).
                if "submitit" in group_folders:
                    # Check to make sure this log wasn't the result of a crash.
                    if results_cfg["options"].get('verify_graceful_exit', True):
                        verify_graceful_exit(
                            log_root=root,
                            log_path=inf_group_dir,
                        )
                    # Check to make sure that this log wasn't the result of a crash.
                    all_inference_log_paths.append(Path(inf_group_dir))
                # Otherwise, we had separated our runs in 'log sets', which isn't a good level of abstraction.
                # but it's what we had done before.
                else:
                    for sub_exp in group_folders:
                        sub_exp_log_path = inf_group_dir + "/" + sub_exp
                        # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
                        # Verify that it is a folder and also that it is not in the skip_log_folders.
                        if os.path.isdir(sub_exp_log_path) and sub_exp not in skip_log_folders:
                            sub_exp_group_folders = os.listdir(sub_exp_log_path)
                            # If 'submitit' is in the highest dir, then we don't have subdirs (new folder structure).
                            if "submitit" in sub_exp_group_folders:
                                # Check to make sure this log wasn't the result of a crash.
                                if results_cfg["options"].get('verify_graceful_exit', True):
                                    verify_graceful_exit(
                                        log_root=root,
                                        log_path=sub_exp_log_path
                                    ) 
                                # Check to make sure that this log wasn't the result of a crash.
                                all_inference_log_paths.append(Path(sub_exp_log_path))
        # We want to make a combined list of all the subdirs from all the all_inference_log_paths.
        # by combining their iterdir() results.
        combined_log_paths = []
        for log_dir in all_inference_log_paths:
            # If a config file exists, then we add it to the list of combined log paths.
            if (log_dir / "config.yml").exists():
                combined_log_paths.append(log_dir)
            else:
                combined_log_paths.extend(list(log_dir.iterdir()))
        # Loop through every configuration in the log directory.
        metadata_pd_collection = []
        for log_set in tqdm(combined_log_paths, desc="Loading log configs"):
            # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
            # Verify that log_set is a directory and that it's not in the skip_log_folders.
            if log_set.is_dir() and log_set.name not in skip_log_folders:
                # Load the metadata file (json) and add it to the metadata dataframe.
                logset_flat_cfg = get_flat_cfg(cfg_name=log_set.name, cfg_dir=log_set / "config.yml")
                # If there was a pretraining class, then we additionally add its config.
                # TODO: When restarting models, we use pretrained_dir as the name and when finetuning we use
                # base_pretrained dir, this causes some issues that need to be resolved.
                if results_cfg["options"].get('load_pretrained_cfg', True):
                    # Find if there is a key in our config that ends in 'pretrained_dir'.
                    pt_load_key = None
                    for key in logset_flat_cfg:
                        if key.endswith('pretrained_dir') or key.endswith('base_model_dir'):
                            pt_load_key = key
                            break
                    # Check if either is in the logset_flat_cfg (if they aren't we can't load the pretrained config).
                    if pt_load_key:
                        pretrained_cfg_dir = Path(logset_flat_cfg[pt_load_key]) / "config.yml"
                        pt_flat_cfg = get_flat_cfg(cfg_name=log_set.name, cfg_dir=pretrained_cfg_dir)
                        # Add 'pretraining' to the keys of the pretrained config.
                        pt_flat_cfg = {f"pretraining_{key}": val for key, val in pt_flat_cfg.items()}
                        # Update the logset_flat_cfg with the pretrained config.
                        logset_flat_cfg.update(pt_flat_cfg)
                # Append the df of the dictionary.
                metadata_pd_collection.append(logset_flat_cfg)
        # Finally, concatenate all of the metadata dataframes.
        metadata_df = pd.DataFrame(metadata_pd_collection) 
        # Gather the columns that have unique values amongst the different configurations.
        if results_cfg["options"].get("remove_shared_columns", False):
            meta_cols = []
            for col in metadata_df.columns:
                if len(metadata_df[col].unique()) > 1:
                    meta_cols.append(col)
        else:
            meta_cols = metadata_df.columns
        #############################
        inference_pd_collection = []
        # Loop through every configuration in the log directory.
        for log_set_path in tqdm(combined_log_paths, desc="Loading image stats"):
            # TODO: FIX THIS, I HATE THE SKIP_LOG_FOLDER PARADIGM.
            if log_set_path.is_dir() and log_set_path.name not in skip_log_folders:
                # Get the metadata corresponding to this log set.
                log_metadata_df = metadata_df[metadata_df["log_set"] == log_set_path.name].copy()
                assert len(log_metadata_df) == 1, \
                    f"Metadata configuration must have one instance, found {len(log_metadata_df)}."
                # First we try to load the accumulated metrics.
                if (log_set_path / "accumulate_stats.pkl").exists():
                    accumulate_obj = pd.read_pickle(log_set_path / "accumulate_stats.pkl")
                    # Since log_metadata_df is one row, get its index.
                    idx = log_metadata_df.index[0]
                    # If there are accumulated metrics, then we need to add them to the metadata dataframe.
                    if len(accumulate_obj) > 0:
                        # If accumulate_dict is a dictionary, then we need to unpack it ourselves.
                        if isinstance(accumulate_obj, dict):
                            # Go through the metrics and add them to the metadata dataframe.
                            for metric_key, metric_stats in accumulate_obj.items():
                                log_metadata_df.loc[idx, metric_key] = metric_stats['mean']
                                log_metadata_df.loc[idx, f'{metric_key}_std'] = metric_stats['std']
                                log_metadata_df.loc[idx, f'{metric_key}_n'] = metric_stats['n']
                        # If it's already a dataframe, then we can just add it to the metadata dataframe.
                        elif isinstance(accumulate_obj, pd.DataFrame):
                            # Tile the metadata df the number of times to match the number of rows in the log_image_df.
                            tiled_acc_metadata_df = pd.concat([log_metadata_df] * len(accumulate_obj), ignore_index=True)
                            # Add the columns from the metadata dataframe that have unique values.
                            log_metadata_df = pd.concat([accumulate_obj, tiled_acc_metadata_df], axis=1)
                        else:
                            raise ValueError("accumulate_dict must be either a dictionary or a dataframe.")
                # Then we try to load the per prediction metrics.
                if (log_set_path / "image_stats.pkl").exists():
                    perpred_df = pd.read_pickle(log_set_path / "image_stats.pkl")
                    # If there are individual predictions, then we need to add them to the metadata dataframe.
                    if len(perpred_df) > 0:
                        # Tile the metadata df the number of times to match the number of rows in the log_image_df.
                        tiled_pp_metadata_df = pd.concat([log_metadata_df] * len(perpred_df), ignore_index=True)
                        # Add the columns from the metadata dataframe that have unique values.
                        log_metadata_df = pd.concat([perpred_df, tiled_pp_metadata_df], axis=1)
                # Add this log to the dataframe.
                inference_pd_collection.append(log_metadata_df)
        # Finally concatenate all of the inference dataframes.
        inference_df = pd.concat(inference_pd_collection, axis=0)

        #########################################
        # POST-PROCESSING STEPS
        #########################################
        # Get the number of rows in image_info_df for each log set.
        num_rows_per_log_set = inference_df.groupby(["log.root", "log_set"]).size()
        if results_cfg["options"]["equal_rows_per_cfg_assert"]:
            # Make sure there is only one unique value in the above.
            assert len(num_rows_per_log_set.unique()) == 1, \
                f"The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}."
        else:
            if len(num_rows_per_log_set.unique()) != 1:
                print(f"Warning: The number of rows in the image_info_df is not the same for all log sets. Got {num_rows_per_log_set}.")
                results_cfg["options"]["print_row_summary"] = False  

        # Go through several optional keys, and add them if they don't exist
        new_columns = {}
        old_raw_keys = []
        # Go through several optional keys, and add them if they don't exist
        for raw_key in inference_df.columns:
            key_parts = raw_key.split(".")
            last_part = key_parts[-1]
            if last_part in ['_class', '_name']:
                new_key = "".join(key_parts)
            else:
                new_key = "_".join(key_parts)
            # If the new key isn't the same as the old key, add the new key.
            if new_key != raw_key and new_key not in inference_df.columns:
                new_columns[new_key] = inference_df[raw_key].fillna("None") # Fill the key with "None" if it is NaN.
                old_raw_keys.append(raw_key)

        # Add new columns to the DataFrame all at once
        inference_df = pd.concat([inference_df, pd.DataFrame(new_columns)], axis=1)
        inference_df.drop(columns=[col for col in inference_df.columns if col in old_raw_keys], inplace=True)

        def dataset(inference_data_class):
            return inference_data_class.split('.')[-1]

        inference_df.augment(dataset)

        # If precomputed_results_path doesn't exist, create it.
        if not os.path.exists(os.path.dirname(precomputed_results_path)):
            os.makedirs(os.path.dirname(precomputed_results_path))
        
        # Save the inference info to a pickle file.
        with open(precomputed_results_path, 'wb') as f:
            pickle.dump(inference_df, f)
    else:
        # load the inference info from the pickle file.
        with open(precomputed_results_path, 'rb') as f:
            inference_df = pickle.load(f)

    # Get the number of rows in image_info_df for each log set.
    final_num_rows_per_log_set = inference_df.groupby(["log_root", "log_set"]).size()
    # Print information about each log set.
    print("Finished loading inference stats.")
    if results_cfg["options"].get("print_row_summary", True):
        print(f"Log amounts: {final_num_rows_per_log_set}")

    # Finally, return the dictionary of inference info.
    return inference_df 

