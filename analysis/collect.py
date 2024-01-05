import collections
import getpass
import itertools
import torch
import pathlib
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from typing import Optional, Any

import more_itertools
import numpy as np
from tqdm.auto import tqdm

import pandas as pd

# unused import because we want the patching side-effects
# on pd.DataFrames
from ..pandas.api import augment_from_attrs
from ..pandas.convenience import ensure_hashable, to_categories, concat_with_attrs
from ..util import FileCache
from ..util.config import HDict, keymap, valmap


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def shorthand_names(names):
    cols = []
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            cols.append("_".join(parts[-i:]))
    counts = collections.Counter(cols)
    column_renames = {}
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            new_name = "_".join(parts[-i:])
            if counts[new_name] == 1:
                column_renames[c] = new_name
                break
    return column_renames


def shorthand_columns(df):
    column_renames = shorthand_names(df.columns)
    df.rename(columns=column_renames, inplace=True)
    return df


class ResultsLoader:

    def __init__(self, cache_file: Optional[str] = None, num_workers: int = 8):
        unixname = getpass.getuser()
        cache_file = cache_file or f"/tmp/{unixname}-results.diskcache"
        self._cache = FileCache(cache_file)
        self._num_workers = num_workers

    def load_configs(
        self,
        *paths,
        shorthand=True,
        properties=False,
        metadata=False,
        log=False,
        callbacks=False,
        categories=False,
    ):
        # ordered deduplication
        paths = list(dict.fromkeys(paths))
        assert all( isinstance(p, (str, pathlib.Path)) for p in paths)

        folders = list(
            itertools.chain.from_iterable(
                pathlib.Path(path).iterdir() for path in paths
            )
        )

        configs = self._cache.gets(
            (folder / "config.yml" for folder in folders), num_workers=self._num_workers
        )

        rows = []
        for folder, cfg in tqdm(zip(folders, configs), leave=False, total=len(folders)):
            if cfg is None:
                continue
            cfg = HDict(cfg)
            if not log:
                cfg.pop("log", None)
            if not callbacks:
                cfg.pop("callbacks", None)
            if metadata:
                cfg.update({"metadata": self._cache[folder / "metadata.json"]})
            if properties:
                cfg.update({"properties": self._cache[folder / "properties.json"]})

            flat_cfg = valmap(list2tuple, cfg.flatten())
            flat_cfg["path"] = folder

            flat_cfg = keymap(lambda x: x.replace("._class", ""), flat_cfg)
            flat_cfg = keymap(lambda x: x.replace("._fn", ""), flat_cfg)

            rows.append(flat_cfg)

        df = pd.DataFrame.from_records(rows)

        ensure_hashable(df, inplace=True)

        if shorthand:
            df = shorthand_columns(df)

        if categories:
            df = to_categories(df, inplace=True, threshold=0.5)
        return df

    def load_metrics(
        self,
        config_df,
        file="metrics.jsonl",
        prefix="log",
        shorthand=True,
        copy_cols=None,
        path_key=None,
        expand_attrs=False,
        categories=False,
    ):
        assert isinstance(config_df, pd.DataFrame)

        log_dfs = []
        if path_key is None:
            path_key = "path"
        if copy_cols is None:
            copy_cols = config_df.columns.to_list()

        folders = config_df[path_key].values

        files = [file] if isinstance(file, str) else file
        n_files = len(files)
        all_files = self._cache.gets(
            (folder / file for folder in folders for file in files),
            num_workers=self._num_workers,
        )
        config_iter = more_itertools.repeat_each(config_df.iterrows(), n_files)

        for (_, row), log_df in tqdm(
            zip(config_iter, all_files), total=len(config_df) * n_files, leave=False
        ):
            row = row.to_dict()
            path = pathlib.Path(row[path_key])
            if log_df is None:
                continue
            if prefix:
                log_df.rename(
                    columns={c: f"{prefix}.{c}" for c in log_df.columns}, inplace=True
                )
            if len(copy_cols) > 0:
                for col in copy_cols:
                    val = row[col]
                    if isinstance(val, tuple):
                        log_df[col] = np.array(itertools.repeat(val, len(log_df)))
                    else:
                        log_df[col] = val
            if expand_attrs:
                log_df = augment_from_attrs(log_df, prefix=f"{prefix}.")
            log_df["path"] = path
            log_dfs.append(log_df)

        full_df = concat_with_attrs(log_dfs, ignore_index=True)

        if shorthand:
            renames = {}
            for c in full_df.columns:
                if c.startswith("log."):
                    shortc = c[len("log.") :]
                    if shortc not in full_df.columns:
                        renames[c] = shortc
                    else:
                        renames[c] = c.replace(".", "__")
            full_df.rename(columns=renames, inplace=True)

        # ensure_hashable(full_df, inplace=True)

        if categories:
            to_categories(full_df, inplace=True, threshold=0.5)

        return full_df

    def load_aggregate(
        self, config_df, metric_df, agg=None, metrics_groupby=("phase",)
    ):
        assert isinstance(config_df, pd.DataFrame)
        assert isinstance(metric_df, pd.DataFrame)

        config_cols = list(config_df.columns)
        metric_cols = list(set(metric_df.columns) - set(config_df.columns))

        _agg_fns = collections.defaultdict(list)

        DEFAULT_AGGS = {
            "max": ["*acc*", "*score*", "*epoch*"],
            "min": ["*loss*", "*err*"],
        }

        for agg_fn, patterns in DEFAULT_AGGS.items():
            for column in metric_cols:
                if any(fnmatch(column, p) for p in patterns):
                    _agg_fns[column].append(agg_fn)
        if agg is not None:
            for column, agg_fns in agg.items():
                _agg_fns[column].extend(agg)

        g = config_cols + list(metrics_groupby)
        agg_df = metric_df.groupby(g, as_index=False, dropna=False, observed=True).agg(
            _agg_fns
        )
        agg_df.columns = [
            col if agg == "" else f"{agg}_{col}" for col, agg in agg_df.columns.values
        ]

        return agg_df

    def load_all(self, *paths, shorthand=True, **selector):

        dfc = self.load_configs(*paths, shorthand=shorthand,).select(**selector).copy()
        df = self.load_metrics(dfc)
        dfa = self.load_aggregate(dfc, df)
        return dfc, df, dfa

    def load_from_callable(self, config_df, load_fn, copy_cols=None, prefix="data"):

        assert isinstance(config_df, pd.DataFrame)

        if copy_cols is None:
            copy_cols = config_df.columns.to_list()

        def do_row(row):
            row = row.to_dict()
            data_df = load_fn(row["path"])
            if data_df is None:
                return pd.DataFrame()
            if prefix:
                data_df.rename(
                    columns={
                        c: f"{prefix}.{c}" for c in data_df.columns if c in copy_cols
                    },
                    inplace=True,
                )
            for col in copy_cols:
                val = row[col]
                if isinstance(val, tuple):
                    data_df[col] = np.array(itertools.repeat(val, len(data_df)))
                else:
                    data_df[col] = val
            return data_df

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            data_dfs = list(
                tqdm(
                    executor.map(do_row, (row for _, row in config_df.iterrows())),
                    total=len(config_df),
                    leave=False,
                )
            )

        return concat_with_attrs(data_dfs, ignore_index=True)

    @staticmethod
    def load_experiment(
        exp_class, 
        device="cuda",
        checkpoint="max-val-dice_score",
        build_data=True,
        df: Optional[Any] = None, 
        path: Optional[str] = None,
        selection_metric: Optional[str] = None,
        ):
        if path is None:
            assert selection_metric is not None, "Must provide a selection metric if no path is provided."
            assert df is not None, "Must provide a dataframe if no path is provided."
            phase, score = selection_metric.split("-")
            subdf = df.select(phase=phase)
            sorted_df = subdf.sort_values(score, ascending=False)
            exp_path = sorted_df.iloc[0].path
        else:
            exp_path = path
        # Load the experiment
        loaded_exp = exp_class(exp_path, build_data=build_data)
        if checkpoint is not None:
            loaded_exp.load(tag=checkpoint)
        # Set the device
        loaded_exp.device = torch.device(device)
        if device == "cuda":
            loaded_exp.to_device()
        # # Place the logs in the experiment, will be hand later
        # loaded_exp.logs = df.select(path=exp_path).reset_index(drop=True)
        # Return the modified loaded exp.
        return loaded_exp
