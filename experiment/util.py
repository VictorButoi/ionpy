# Misc imports
import os
import ast
import time
import json
import string
import random
import pathlib
import datetime
import functools
import importlib
import numpy as np
from pprint import pprint
from pathlib import Path
from pydantic import validate_arguments
from typing import Tuple, Any, Optional
# Torch imports
import torch
# Local imports
from ..analysis import ResultsLoader
from ..util.ioutil import autoload
from ..util.hash import json_digest
from ..util.config import HDict, Config
from ..util.more_functools import partial


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Set seed: {}".format(seed))


def generate_tuid(nonce_length: int = 4) -> Tuple[str, int]:
    rng = np.random.default_rng(time.time_ns())
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    char_options = list(string.ascii_uppercase + string.digits)
    nonce = "".join(rng.choice(char_options, size=nonce_length))
    return now, nonce


def absolute_import(reference):
    module, _, attr = reference.rpartition(".")
    if importlib.util.find_spec(module) is not None:
        module = importlib.import_module(module)
        if hasattr(module, attr):
            return getattr(module, attr)

    raise ImportError(f"Could not import {reference}")


def eval_config(config, opt_kwargs=None):

    if config is None:
        return None

    if not isinstance(config, (dict, list, HDict)):
        return config

    if isinstance(config, HDict):
        return eval_config(config.to_dict(), opt_kwargs)

    if isinstance(config, list):
        return [eval_config(v, opt_kwargs) for v in config]

    for k, v in config.items():
        if isinstance(v, (dict, list)):
            config[k] = eval_config(v, opt_kwargs)

    state = config.pop("_state", None)

    if "_class" in config:
        if opt_kwargs is None:
            config = absolute_import(config.pop("_class"))(**config)
        else:
            imp_class = config.pop("_class")
            config = absolute_import(imp_class)(**config, **opt_kwargs)
    elif "_fn" in config:
        fn = absolute_import(config.pop("_fn"))
        config = partial(fn, **config)

    if state is not None:
        key = None
        if isinstance(state, (list, tuple)):
            state, key = state
        with pathlib.Path(state).open("rb") as f:
            state_dict = torch.load(f)
            if key is not None:
                state_dict = state_dict[key]
            config.load_state_dict(state_dict)

    return config


def config_from_path(path):
    return Config(autoload(path / "config.yml"))


def path_from_job(job):
    import parse

    stdout = job.stdout()
    return pathlib.Path(
        parse.search('Running {exp_type}Experiment("{path}")', stdout)["path"]
    )


def config_from_job(job):
    return config_from_path(path_from_job(job))

#########################################################################
# Inference functions


def get_exp_load_info(pretrained_exp_root):
    is_exp_group = not ("config.yml" in os.listdir(pretrained_exp_root)) 
    # Load the results loader
    rs = ResultsLoader()
    # If the experiment is a group, then load the configs and build the experiment.
    if is_exp_group: 
        dfc = rs.load_configs(
            pretrained_exp_root,
            properties=False,
        )
        return {
            "df": rs.load_metrics(dfc),
        }
    else:
        return {
            "path": pretrained_exp_root
        }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def load_experiment(
    exp_class: Any,
    checkpoint: str,
    device: str = "cpu",
    df: Optional[Any] = None, 
    weights_only: bool = False,
    path: Optional[str] = None,
    load_data: bool = False,
    exp_kwargs: Optional[dict] = {},
    attr_dict: Optional[dict] = None,
    selection_metric: Optional[str] = None,
):
    if path is None:
        assert df is not None, "Must provide a dataframe if no path is provided."
        if attr_dict is not None:
            for attr_key in attr_dict:
                select_arg = {attr_key: attr_dict[attr_key]}
                if attr_key in ["mix_filters"]:
                    select_arg = {attr_key: ast.literal_eval(attr_dict[attr_key])}
                df = df.select(**select_arg)
        if selection_metric is not None:
            phase, score = selection_metric.split("-")
            df = df.select(phase=phase)
            df = df.sort_values(score, ascending=False)
        exp_path = df.iloc[0].path
    else:
        assert attr_dict is None, "Cannot provide both a path and an attribute dictionary."
        exp_path = path

    # Import the class if we are passing in a string.
    if isinstance(exp_class, str):
        exp_class = absolute_import(exp_class)
    # Actually load the class object.
    exp_obj = exp_class(
        exp_path, 
        init_metrics=False, 
        load_data=load_data,
        **exp_kwargs
    )

    # Load the experiment
    if checkpoint is not None:
        exp_path = Path(exp_path)
        weights_path = exp_path / "checkpoints" / f"{checkpoint}.pt"
        if not weights_path.exists():
            print("No checkpoint found at: {}, defaulting to last.pt".format(weights_path))
            weights_path = exp_path / "checkpoints" / "last.pt"
        with weights_path.open("rb") as f:
            state = torch.load(f, weights_only=weights_only)
        exp_obj.model.load_state_dict(state["model"])
        print(f"Loaded model from: {weights_path}")
    
    # Set the device
    exp_obj.device = torch.device(device)
    if device == "cuda":
        exp_obj.to_device()
    
    return exp_obj
