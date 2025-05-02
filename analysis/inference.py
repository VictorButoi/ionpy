# torch imports
import torch
# ionpy imports
from ionpy.util.meter import MeterDict
from ionpy.experiment.util import fix_seed
from ionpy.util.torchutils import to_device
import ionpy.analysis.analysis_utils.helpers as inf_helpers 
import ionpy.analysis.analysis_utils.inference_utils as inf_utils
# Misc imports
import numpy as np
from tqdm import tqdm
from pprint import pprint
from typing import Any, Optional
from pydantic import validate_arguments
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_inference(
    config: Any,
) -> None:
    # Get the config dictionary
    if not isinstance(config, dict):
        inference_cfg_dict = config.to_dict()
    else:
        inference_cfg_dict = config
    # Ensure that inference seed is the same.
    fix_seed(inference_cfg_dict['experiment'].get('inference_seed', 40))
    # Initialize all the objects needed for inference.
    inference_init_obj = inf_utils.init_inf_object(inference_cfg_dict)
    inf_data_opts = inference_init_obj['dataobjs'].keys()
    # Loop through the data, gather your stats!
    tracker_objs = {
        "inf_cfg_dict": inference_cfg_dict, "inf_init_obj": inference_init_obj
    }
    # A dataloader is something that iterates through a set of datapoints we want to
    # run inference on. The invariant here is that we should expect to do inference
    # on every data point in the dataloader.
    for data_cfg_str in inf_data_opts:
        # Make the data opt args for this particular data configuration.
        if len(data_cfg_str) > 0:
            data_props = dict(item.split(':') for item in data_cfg_str.split('^'))
            data_props['data_cfg_str'] = data_cfg_str 
        else:
            data_props = {'data_cfg_str': data_cfg_str}
        # Iterate through this configuration's dataloader.
        standard_dataloader_loop(
            inf_data_obj=inference_init_obj['dataobjs'][data_cfg_str],
            data_props=data_props,
            **tracker_objs
        )
    # Save the records at the end too
    inf_helpers.save_trackers(inference_init_obj["output_root"], trackers=inference_init_obj["trackers"])


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_dataloader_loop(
    inf_cfg_dict,
    inf_init_obj,
    inf_data_obj, 
    data_props: Optional[dict] = {}
):
    # Extract the properties of the data.
    dloader = inf_data_obj["dloader"]
    gpu_aug_pipeline = inf_data_obj.get('aug_pipeline', None)
    # Go through each batch in the dataloader. We use a batch_idx
    # and a iterator because we want to keep track of the batch index.
    iter_dloader = iter(dloader)
    for batch_idx in range(len(dloader)):
        batch = next(iter_dloader)
        print(f"Working on batch #{batch_idx} out of",\
            len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")
        # and put them on the device of our experiment.
        x, y = to_device((batch["img"], batch["label"]), inf_init_obj['exp'].device)
        # If we have augs to apply on the image (on the GPU), then we need to do that here.
        if gpu_aug_pipeline is not None:
            with torch.no_grad():
                y_in = y if not isinstance(y, dict) else y["image"]
                x, y_in = gpu_aug_pipeline(x, y_in)
                if isinstance(y, dict):
                    y["image"] = y_in
                else:
                    y = y_in
        # If we are dealing with RGB mode, then we need to duplicate the img in the channel dim.
        # If the mode is rgb, then we need to duplicate the image 3 times.
        if inf_cfg_dict['train'].get('color_mode','default') == "rgb" and x.shape[1] == 1:
            x = torch.cat([x] * 3, axis=1)
        # Pack the batch into a dictionary.
        forward_batch = {
            "img": x, 
            "label": y,
            "batch_idx": batch_idx,
            "data_id": np.array(batch["data_id"]),
        }
        # Run the forward loop
        standard_image_forward_loop(
            forward_batch,
            inf_cfg_dict=inf_cfg_dict,
            inf_init_obj=inf_init_obj,
            data_props=data_props
        )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def standard_image_forward_loop(
    batch,
    inf_cfg_dict,
    inf_init_obj,
    data_props: Optional[dict] = {}
):
    # Get the experiment
    exp = inf_init_obj["exp"]
    # Get the example data
    x, y = batch.pop("img"), batch.pop("label")
    # If there are inference kwargs, then we need to do a forward pass with those kwargs.
    inf_kwarg_grid = inf_helpers.get_kwarg_sweep(inf_cfg_dict)
    # Iterate through each of the inference kwargs.
    for predict_params in tqdm(inf_kwarg_grid, disable=(len(inf_kwarg_grid) == 1)):
        # If we are doing patch-based prediction then we need to do that here.
        y_hat = exp.predict(x, **predict_params)
        # Wrap the outputs into a dictionary.
        forward_batch = {
            "x": x, "y_true": y, "y_pred": y_hat, "data_ids": batch["data_id"]
        }
        ###########################
        # VISUALIZING IMAGE PREDS #
        ###########################
        if inf_cfg_dict["log"].get("show_examples", False):
            inf_init_obj['visualizer'](forward_batch)
        # Get the calibration item info.  
        calculate_batch_stats(
            forward_batch=forward_batch,
            metadata_dict={**predict_params, **data_props},
            inference_cfg=inf_cfg_dict,
            trackers=inf_init_obj['trackers'],
        )
    # Save the records every so often, to get intermediate results. Note, because of data_ids
    # this can contain fewer than 'log interval' many items.
    if inf_init_obj['data_counter'] % inf_cfg_dict['log']['log_interval'] == 0:
        inf_helpers.save_trackers(
            inf_init_obj["output_root"], trackers=inf_init_obj['trackers']
        )
    # Increment the data counter.
    inf_init_obj['data_counter'] += 1


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def calculate_batch_stats(
    forward_batch, 
    metadata_dict,
    inference_cfg,
    trackers 
):
    inf_metric_cfg = inference_cfg["metrics"]
    #############################################################
    # CALCULATE QUALITY METRICS
    #############################################################
    assert not len(inf_metric_cfg) == 0,\
        "No metrics were specified in the config file."
    # Calculate all of the metrics.
    total_metscore_dict = {}
    for metric_name, metric_dict in inf_metric_cfg.items():
        # Get the predictions and the true labels.
        y_pred = forward_batch["y_pred"]
        y_true = forward_batch["y_true"]
        # If y_pred and y_true are dictionaries, then we need to choose the correct
        # key for the metric function.
        if isinstance(y_pred, dict) and isinstance(y_true, dict):
            label_type = metric_dict['label_type']
            y_pred = y_pred[label_type]
            y_true = y_true[label_type]

        total_metscore_dict[metric_name] = metric_dict['_fn'](
            y_pred=y_pred,
            y_true=y_true
        )

    #############################################################
    # PRINT OUR METRICS
    #############################################################
    if inference_cfg["log"].get("verbose", False):
        # Give a new line for the metrics.
        print()
        # Print the quality metrics.
        for metric_name, metric_dict in inf_metric_cfg.items():
            met_score = total_metscore_dict[metric_name]
            print(f"{metric_name}: {total_metscore_dict[metric_name]}")
            # If met_score is a tensor, then we also want to print the mean.
            if len(met_score.shape) > 0:
                print(f"batch {metric_name}: {met_score.mean()}")
        # Print the metadata
        print("METADATA:\n", metadata_dict)

    # We need to separate the metrics into the accumulate and individual stats.
    accumulate_scores = {}
    individual_scores = {}
    for met_name, met_cfg in inf_metric_cfg.items():
        met_score = total_metscore_dict[met_name]
        if met_cfg['type'] == 'accumulate':
            accumulate_scores[met_name] = met_score.item()
        elif met_cfg['type'] == 'individual':
            # If the metric is a single value, then we need to unsqueeze a batch dimension
            # because individual perform works by enumerating over the batch dimension.
            if len(met_score.shape) == 0:
                met_score = met_score.unsqueeze(0)
            # An important assert we need to do is that the first dimension of the met_score
            # should be equivalent to the batch-size. Otherwise, we did an unintended reduction.
            assert met_score.shape[0] == len(forward_batch["data_ids"]),\
                f"The metric score tensor does not have the same batch size ({met_score.shape[0]}) as the data_ids ({len(forward_batch['data_ids'])}) for metric {met_name}."
            individual_scores[met_name] = met_score
        else:
            raise ValueError("Metric is not in either accumulate or per prediction stats.")

    # Update the accumulate metrics.
    if accumulate_scores != {}:
        acc_stat_tracker = trackers['accumulate_stats']
        if metadata_dict == {}:
            if not isinstance(acc_stat_tracker, MeterDict):
                trackers['accumulate_stats'] = MeterDict()
            trackers['accumulate_stats'].update(accumulate_scores)
            # Print the accumulate stats.
            if inference_cfg["log"].get("verbose", False):
                print("Accumulate stats:")
                pprint(trackers['accumulate_stats'].asdict())
                print()
        else:
            metadata_key = str(metadata_dict)
            if metadata_key not in acc_stat_tracker:
                acc_stat_tracker[metadata_key] = MeterDict()
            acc_stat_tracker[metadata_key].update(accumulate_scores)
            # Print the accumulate stats.
            if inference_cfg["log"].get("verbose", False):
                print("Accumulate stats:")
                pprint(acc_stat_tracker[metadata_key].asdict())
                print()

    # Iterate through all of the collected metrics and add them to the records.
    if individual_scores != {}:
        for met_name, met_score_tensor in individual_scores.items():
            for met_idx, met_score in enumerate(met_score_tensor):
                # Get the data id corresponding.
                data_id = forward_batch["data_ids"][met_idx]
                # Add the dataset info to the record
                record = {
                    "data_id": data_id,
                    "image_metric": met_name,
                    "metric_score": met_score.item(),
                    **metadata_dict,
                }
                # Add the record to the list.
                trackers['image_stats'].append(record)