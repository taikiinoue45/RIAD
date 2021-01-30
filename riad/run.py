import logging
import os
import sys

import hydra
import mlflow
from omegaconf import DictConfig

from riad.runner import Runner


log = logging.getLogger(__name__)

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri("file:///app/RIAD/riad/mlruns")
    mlflow.set_experiment(cfg.params.experiment_name)
    mlflow.start_run(run_name=cfg.params.run_name)
    mlflow.log_params(cfg.params)
    mlflow.log_param("cwd", os.getcwd())

    runner = Runner(cfg)
    runner.run()

    mlflow.log_artifacts(".hydra", "hydra")
    mlflow.log_artifacts("epochs", "epochs")


if __name__ == "__main__":
    main()
