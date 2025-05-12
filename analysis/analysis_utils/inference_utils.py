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
        config=inference_cfg['experiment'],
        checkpoint=inference_cfg['model']['checkpoint'],
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
    dataobj_dict = dataobjs_from_exp(
        inference_exp=inference_exp,
        inference_cfg=inference_cfg
    )

    #################
    # MISC SETTINGS #
    #################
    # Build a visualizer if we want to visualize the results.
    visualizer = eval_config(inference_cfg.get('visualize'))
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
        "visualizer": visualizer,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_inference_exp(
    config: dict,
    to_device: bool = False,
    inf_kwargs: Optional[dict] = {},
    checkpoint: Optional[str] = None,
): 
    # If we are passing to the device, we need to set the 'device' of
    # our init to 'gpu'.
    if to_device:
        inf_kwargs['device'] = 'cuda'
    # Load the experiment directly if you give a sub-path.
    inference_exp = load_experiment(
        exp_class=config['_class'],
        checkpoint=checkpoint,
        exp_kwargs={
            "set_seed": False,
            "load_data": False,
            "load_aug_pipeline": False
        },
        **inf_kwargs,
        **get_exp_load_info(config['model_dir']),
    )
    # Set the model to evaluation mode.
    inference_exp.model.train(False)
    # Optionally, move the model to the device.
    if to_device:
        inference_exp.to_device()

    return inference_exp


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dataobjs_from_exp(
    inference_exp,
    inference_cfg
):
    inf_data_cfg = inference_cfg['inference_data']
    # Make sure we aren't sampling for evaluation. 
    if "slicing" in inf_data_cfg.keys():
        assert inf_data_cfg['slicing'] not in ['central', 'dense', 'uniform'], "Sampling methods not allowed for evaluation."

    # Get the dataset class
    dataset_cls_str = inf_data_cfg.pop('_class')
    dset_cls = absolute_import(dataset_cls_str)

    # Ensure that we return the different data ids.
    inf_data_cfg['return_data_id'] = True

    # Often we will have trained with 'transforms', we need to pop them here.
    dset_transforms = {
        "train": inf_data_cfg.pop("train_transforms", None),
        "val": inf_data_cfg.pop("val_transforms", None)
    }
    dset_kwargs = {
        "train": inf_data_cfg.pop("train_kwargs", {}),
        "val": inf_data_cfg.pop("val_kwargs", {})
    }

    # Initialize the dataloader configuration.
    modified_dataloader_cfg = inference_exp.config['dataloader'].to_dict()
    if 'dataloader' in inference_cfg:
        new_dloader_cfg = inference_cfg['dataloader']
        modified_dataloader_cfg.update(new_dloader_cfg)

    ###########################################################
    # Build the augmentation pipeline if we want augs on GPU. #
    ###########################################################
    # Assemble the augmentation pipeline.
    gpu_aug_cfg = inference_exp.config.get('gpu_augmentations', None)
    if 'gpu_augmentations' in inference_cfg:
        inf_gpu_aug_cfg_opts = inference_cfg['gpu_augmentations']
        if gpu_aug_cfg is not None:
            gpu_aug_cfg.update(inf_gpu_aug_cfg_opts)
        else:
            gpu_aug_cfg = inf_gpu_aug_cfg_opts
    # Convert the Config object to a dictionary.
    if gpu_aug_cfg is not None and not isinstance(gpu_aug_cfg, dict):
        gpu_aug_cfg = gpu_aug_cfg.to_dict()

    # If we have a gpu augmentation pipeline, we need to build it.
    if gpu_aug_cfg is not None:
        # Apply any data preprocessing or augmentation
        gpu_aug_pipeline_dict = {
            "train": build_gpu_aug_pipeline(
                gpu_aug_cfg.get('train_transforms'),
            ),
            "val": build_gpu_aug_pipeline(
                gpu_aug_cfg.get('val_transforms'),
            )
        }


    # We need to prune the dataset config to only include valid
    # parameters for the dataset class.
    dset_cfg = {}
    # Get the signature of the __init__ method.
    sig = inspect.signature(dset_cls.__init__)
    # Get the accepted parameter names (excluding 'self').
    valid_params = set(sig.parameters.keys()) - {'self'}
    # Iterate over a static list of keys since we are modifying the dictionary.
    for key in list(valid_params):
        if key in valid_params and key in inf_data_cfg:
            dset_cfg[key] = inf_data_cfg[key]

    # Get the split, used to determine the dataset and the transforms.
    split = dset_cfg['split']
    # First, we need to add the splitwise kwargs to the data cfg.
    dset_cfg.update(dset_kwargs[split])
    dset_cfg['transforms'] = dset_transforms[split]

    # Build the dataloader for this opt cfg and label.
    data_obj = {
        'dloader': DataLoader(
            dset_cls(**dset_cfg), 
            **modified_dataloader_cfg
        )
    }
    if gpu_aug_cfg is not None:
        data_obj['aug_pipeline'] = gpu_aug_pipeline_dict.get(split, None)

    # Modify the inference data cfg to reflect the new data objects.
    modified_inf_data_cfg = dset_cfg.copy()

    # Place a few last things in the modified data cfg.
    modified_inf_data_cfg.update({
        '_class': dataset_cls_str,
    })

    # Update the inference_data_cfg to reflect the data we are running on.
    inference_cfg.update({
        'inference_data': modified_inf_data_cfg,
        'inference_dataloaders': modified_dataloader_cfg,
        'inference_gpu_augmentations': gpu_aug_cfg
    })

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