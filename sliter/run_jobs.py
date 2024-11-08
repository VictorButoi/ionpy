# misc imports
import os
import sys
import time
import torch
import pathlib
import submitit
from typing import List, Any, Optional
from pydantic import validate_arguments
# local imports
from ionpy.util import Config


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_exp(
    exp_class: Any,
    config: Any,
    available_gpus: Optional[int] = None,
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    if available_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    # If config is not a 'Config' object, convert it.
    if not isinstance(config, Config):
        config = Config(config)
    # Get the experiment class, either fresh or from a path.
    exp = exp_class.from_config(config)
    # Run the experiment.
    exp.run()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def run_job(
    job_func: Any,
    config: Any,
    available_gpus: Optional[int] = None 
):
    # Important imports, otherwise the processes will not be able to import the necessary modules
    sys.path.append('/storage/vbutoi/projects')
    sys.path.append('/storage/vbutoi/projects/ESE')
    # Set the visible gpu.
    if available_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus)
    # If config is not a 'Config' object, convert it.
    if not isinstance(config, Config):
        config = Config(config)
    job_func(config)
