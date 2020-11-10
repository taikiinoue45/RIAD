import logging

from omegaconf import DictConfig
from riad.runner.initialize import RunnerInitialize
from riad.runner.train_val_test import RunnerTrainValTest


class Runner(RunnerInitialize, RunnerTrainValTest):
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "val", "test"]:
            self.augs_dict[data_type] = self.init_augs(self.cfg.augs[data_type])
            self.dataset_dict[data_type] = self.init_dataset(self.cfg.dataset[data_type])
            self.dataloader_dict = self.init_dataloader(self.cfg.dataloader[data_type])

        self.model = self.init_model(self.cfg.model)
        self.model = self.model.to(self.cfg.device)

        self.criterion_L2 = self.init_criterion(self.cfg.criterion.L2)
        self.criterion_SSIM = self.init_criterion(self.cfg.criterion.SSIM)
        self.criterion_MSGM = self.init_criterion(self.cfg.criterion.MSGM)
