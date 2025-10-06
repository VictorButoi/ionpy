# misc imports
import os
import yaml
import pickle
import inspect
import itertools
from typing import Optional
from pydantic import validate_arguments 
from torch.utils.data import DataLoader
# ionpy imports
from ionpy.util import Config
from ionpy.experiment.util import absolute_import, eval_config
from ionpy.augmentation.gpu_transform_wrappers import build_gpu_aug_pipeline
# local imports
from .helpers import save_inference_metadata
from ...experiment.util import load_experiment, get_exp_load_info


def init_inf_object(inference_cfg):
    ###################
    # BUILD THE MODEL #
    ###################
    inference_exp = load_inference_exp(
        inference_cfg=inference_cfg,
        to_device=True
    )

    # Update important keys in the inference cfg.
    inference_exp_total_cfg_dict = inference_exp.config.to_dict()
    inference_cfg.update({
        'train': inference_exp_total_cfg_dict['train'],
        'loss_func': inference_exp_total_cfg_dict['loss_func'],
        'training_data': inference_exp_total_cfg_dict['data'],
        "model": inference_exp_total_cfg_dict['model'],
    })
    inference_cfg['experiment']['pretrained_seed'] = inference_exp_total_cfg_dict['experiment']['seed']

    #####################
    # BUILD THE DATASET #
    #####################
    # Build the dataloaders.
    dataobj_dict = dataobjs_from_exp(inference_exp=inference_exp)

    #################
    # MISC SETTINGS #
    #################
    # Trackers store statistics over time. Either per prediction (each row is a different prediction)
    # or accumulated statistics (stat meter for the entire dataset for each metric).
    trackers = {
        "image_stats": [],
        "accumulate_stats": {} 
    }

    ####################################################
    # SAVE THE METADATA AND PRINT THE INFERENCE CONFIG #
    ####################################################
    task_root = save_inference_metadata(inference_cfg)
    print(f"Running:\n\n{str(yaml.safe_dump(Config(inference_cfg)._data, indent=0))}")

    ############################################################################
    # INITIALIZE THE METRICS #
    # NOTE: We have to do this AFTER saving the metadata because we can't save 
    # initialized functions.
    ############################################################################
    for metric_name, met_cfg in inference_cfg['metrics'].items():
        # Determine if the metric is accumulate or per prediction.
        met_type = met_cfg.pop('type', 'individual')
        met_label_type = met_cfg.pop('label_type', None)
        # Add the quality metric to the dictionary.
        inference_cfg['metrics'][metric_name] = {
            "name": metric_name,
            "type": met_type,
            "label_type": met_label_type,
            "_fn": eval_config(met_cfg),
        }

    # Return a dictionary of the components needed for the calibration statistics.
    return {
        "data_counter": 0,
        "exp": inference_exp,
        "trackers": trackers,
        "dataobj": dataobj_dict,
        "output_root": task_root,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_exp(
    inference_cfg: dict,
    to_device: bool = False,
): 
    # Load the experiment directly if you give a sub-path.
    inference_exp = load_experiment(
        exp_kwargs={
            "set_seed": False,
            "load_data": False,
            "load_aug_pipeline": False # Unclear if this is correct
        },
        device='cuda',
        config_update=inference_cfg,
        **inference_cfg['experiment']['exp_kwargs'],
        **get_exp_load_info(inference_cfg['experiment']['model_dir']),
    )
    # Then we build the callbacks.
    inference_exp.build_callbacks()
    # Set the model to evaluation mode.
    inference_exp.model.train(False)
    # Optionally, move the model to the device.
    if to_device:
        inference_exp.to_device()

    return inference_exp


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataobjs_from_exp(inference_exp,):
    data_cfg = inference_exp.config['data'].to_dict()
    split = data_cfg.pop('split')
    # First we build the dataset.
    inference_exp.build_data(load_data=True, cfg_dict=data_cfg)
    split_dloader = inference_exp.build_dataloader(
        return_obj=True
    )[split]
    data_obj = {
        'dloader': split_dloader,
        'aug_pipeline': inference_exp.aug_pipeline
    }
    # Return the dataobjs.
    return data_obj


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def verify_graceful_exit(log_root: str, log_path: str):
    submitit_dir = os.path.join(log_root, log_path, "submitit")
    # Check that the submitit directory exists if it doesnt then return.
    try:
        result_pickl_files = [logfile for logfile in os.listdir(submitit_dir) if logfile.endswith("_result.pkl")]
        unique_logs = list(set([logfile.split("_")[0] for logfile in result_pickl_files]))
    except:
        print(f"Error loading submitit directory: {submitit_dir}")
        return
    # Check that all the logs have a success result.
    for log_name in unique_logs:
        result_log_file = os.path.join(submitit_dir, f"{log_name}_0_result.pkl")
        try:
            with open(result_log_file, 'rb') as f:
                result = pickle.load(f)[0]
            if result != 'success':
                raise ValueError(f"Found non-success result in file {result_log_file}: {result}.")
        except Exception as e:
            print(f"Error loading result log file: {e}")