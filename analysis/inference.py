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
    # Loop through the data, gather your stats!
    tracker_objs = {
        "inf_cfg_dict": inference_cfg_dict, "inf_init_obj": inference_init_obj
    }
    # Iterate through this configuration's dataloader.
    standard_dataloader_loop(
        inf_data_obj=inference_init_obj['dataobj'],
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
    # Get the experiment and the kwarg sweep for inference once.
    exp = inf_init_obj["exp"]
    inf_kwarg_grid = inf_helpers.get_kwarg_sweep(inf_cfg_dict)
    # Go through each batch in the dataloader. We use a batch_idx
    # and a iterator because we want to keep track of the batch index.
    iter_dloader = iter(dloader)
    for batch_idx in range(len(dloader)):
        batch = next(iter_dloader)
        print(f"Working on batch #{batch_idx} out of",\
            len(dloader), "({:.2f}%)".format(batch_idx / len(dloader) * 100), end="\r")
        # Iterate through each of the inference kwargs and run forward + stats.
        for predict_params in tqdm(inf_kwarg_grid, disable=(len(inf_kwarg_grid) == 1)):
            forward_batch = exp.run_step(
                batch, 
                phase="val",
                backward=False,
                **predict_params,
            )
            calculate_batch_stats(
                forward_batch=forward_batch,
                metadata_dict={**predict_params, **data_props},
                inference_cfg=inf_cfg_dict,
                trackers=inf_init_obj['trackers'],
            )
        # Save intermediate results periodically.
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
    assert len(inf_metric_cfg) != 0, "No metrics were specified in the config file."

    # Calculate all metrics for this batch
    total_metscore_dict = {}
    for metric_name, metric_dict in inf_metric_cfg.items():
        total_metscore_dict[metric_name] = metric_dict['_fn'](
            y_pred=forward_batch["y_pred"],
            y_true=forward_batch["y_true"]
        )

    # Optionally print metrics
    if inference_cfg["log"].get("verbose", False):
        print()
        for metric_name, met_score in total_metscore_dict.items():
            print(f"{metric_name}: {met_score}")
            if len(met_score.shape) > 0:
                print(f"batch {metric_name}: {met_score.mean()}")
        print("METADATA:\n", metadata_dict)

    # Split into accumulate and per-sample (individual) metrics
    accumulate_scores = {}
    individual_scores = {}
    batch_size = len(forward_batch["data_id"])
    for met_name, met_cfg in inf_metric_cfg.items():
        met_score = total_metscore_dict[met_name]
        met_type = met_cfg['type']
        if met_type == 'accumulate':
            accumulate_scores[met_name] = met_score.item()
        elif met_type == 'individual':
            if len(met_score.shape) == 0:
                met_score = met_score.unsqueeze(0)
            assert met_score.shape[0] == batch_size, \
                f"The metric score tensor does not have the same batch size ({met_score.shape[0]}) as the data_ids ({batch_size}) for metric {met_name}."
            individual_scores[met_name] = met_score
        else:
            raise ValueError("Metric is not in either accumulate or per prediction stats.")

    # Update accumulate metrics
    if accumulate_scores:
        acc_stat_tracker = trackers['accumulate_stats']
        if metadata_dict == {}:
            if not isinstance(acc_stat_tracker, MeterDict):
                trackers['accumulate_stats'] = MeterDict()
            trackers['accumulate_stats'].update(accumulate_scores)
            if inference_cfg["log"].get("verbose", False):
                print("Accumulate stats:")
                pprint(trackers['accumulate_stats'].asdict())
                print()
        else:
            metadata_key = str(metadata_dict)
            if metadata_key not in acc_stat_tracker:
                acc_stat_tracker[metadata_key] = MeterDict()
            acc_stat_tracker[metadata_key].update(accumulate_scores)
            if inference_cfg["log"].get("verbose", False):
                print("Accumulate stats:")
                pprint(acc_stat_tracker[metadata_key].asdict())
                print()

    # Record individual metrics per sample
    if individual_scores:
        for met_name, met_score_tensor in individual_scores.items():
            for met_idx, met_score in enumerate(met_score_tensor):
                data_id = forward_batch["data_id"][met_idx]
                record = {
                    "data_id": data_id,
                    "image_metric": met_name,
                    "metric_score": met_score.item(),
                    **metadata_dict,
                }
                trackers['image_stats'].append(record)