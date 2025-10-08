# Misc imports
from typing import List
# Local imports
import wandb


class WandbLogger:
    
    def __init__(
        self, 
        exp, 
        entity,
        project,
        watch_keys: List[str] = None,
    ):
        self.exp = exp
        exp_config = exp.config.to_dict() 
        wandb.init(
            entity=entity,
            project=project,
            config=exp_config,
            dir=exp_config["log"]["root"]
        )
        wandb.run.name = exp_config["log"]["wandb_string"]
        if watch_keys:
            for key in watch_keys:
                property = getattr(exp, key)
                wandb.watch(property, log_freq=1)
                print(f"Watching {property} with wandb.")

    def __call__(self, epoch):
        df = self.exp.metrics.df
        df = df[df.epoch == epoch]
        update = {}
        for _, row in df.iterrows():
            phase = row["phase"]
            for metric in df.columns.drop('phase'):
                if metric == "epoch":
                    update["epoch"] = row["epoch"]
                else:
                    update[f"{phase}_{metric}"] = row[metric]

        wandb.log(update)
