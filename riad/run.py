import logging
import os
import sys

import hydra
from omegaconf import DictConfig

from riad.runner import Runner

log = logging.getLogger(__name__)

config_path = sys.argv[1]
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: DictConfig) -> None:

    os.rename(".hydra", "hydra")

    trainer = Runner(cfg)
    trainer.run_train()


if __name__ == "__main__":
    main()
