import logging
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

    # mlflow.set_tracking_uri("databricks")
    # mlflow.set_experiment("/Users/inoue@nablas.com/riad")
    mlflow.start_run(run_name=cfg.params.category)
    mlflow.log_artifacts(".hydra", "hydra")

    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
