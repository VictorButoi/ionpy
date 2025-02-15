import os
import datetime
import functools
import string
import random
import time
import importlib
import pathlib
from typing import Tuple
from pprint import pprint

from ..util.config import HDict, Config
from ..util.ioutil import autoload
from ..util.more_functools import partial

import torch
import numpy as np


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
