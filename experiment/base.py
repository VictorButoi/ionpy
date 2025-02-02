# Misc imports
import yaml
import pathlib
from typing import Optional
from abc import abstractmethod
# Local imports
from .util import fix_seed, absolute_import, generate_tuid
from ..util.metrics import MetricsDict
from ..util.config import HDict, FHDict, ImmutableConfig, config_digest
from ..util.ioutil import autosave
from ..util.libcheck import check_environment
from ..util.thunder import ThunderDict


def eval_callbacks(all_callbacks, experiment):
    evaluated_callbacks = {}
    for group, callbacks in all_callbacks.items():
        evaluated_callbacks[group] = []

        for callback in callbacks:
            if isinstance(callback, str):
                cb = absolute_import(callback)(experiment)
            elif isinstance(callback, dict):
                assert len(callback) == 1, "Callback must have length 1"
                callback, kwargs = next(iter(callback.items()))
                cb = absolute_import(callback)(experiment, **kwargs)
            else:
                raise TypeError("Callback must be either str or dict")
            evaluated_callbacks[group].append(cb)

    return evaluated_callbacks


class BaseExperiment:
    def __init__(self, path, set_seed=True):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path
        assert path.exists()
        self.name = self.path.stem

        self.config = ImmutableConfig.from_file(path / "config.yml")
        self.properties = FHDict(self.path / "properties.json")
        self.metadata = FHDict(self.path / "metadata.json")
        self.metricsd = MetricsDict(self.path)

        self.store = ThunderDict(self.path / "store")

        if "experiment.seed" in self.config and set_seed:
            fix_seed(self.config.get("experiment.seed"))
        check_environment()

        self.properties["experiment.class"] = self.__class__.__name__

        if "log.properties" in self.config:
            self.properties.update(self.config["log.properties"])

    @classmethod
    def from_config(cls, config, uuid: Optional[str] = None, **kwargs) -> "BaseExperiment":
        if isinstance(config, HDict):
            config = config.to_dict()
        root = pathlib.Path()
        if "log" in config:
            root = pathlib.Path(config["log"].get("root", "."))
        # UUID is how we separate similar runs that belong to one experiment.
        if uuid is None:
            create_time, nonce = generate_tuid()
            digest = config_digest(config)
            uuid = f"{create_time}-{nonce}-{digest}"
        else:
            create_time, nonce, digest = uuid.split("-")
        # Make the run path and save the metadata and config.
        path = root / uuid
        metadata = {"create_time": create_time, "nonce": nonce, "digest": digest}
        autosave(metadata, path / "metadata.json")
        autosave(config, path / "config.yml")
        return cls(str(path.absolute()), **kwargs)

    @property
    def metrics(self):
        return self.metricsd["metrics"]

    def __hash__(self):
        return hash(self.path)

    @abstractmethod
    def run(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}("{str(self.path)}")'

    def __str__(self):
        s = f"{repr(self)}\n---\n"
        s += yaml.safe_dump(self.config._data, indent=2)
        return s

    def build_callbacks(self):
        self.callbacks = {}
        if "callbacks" in self.config:
            self.callbacks = eval_callbacks(self.config["callbacks"], self)
