from importlib import import_module
from typing import Dict

import riad.albu as albu
from omegaconf import DictConfig
from riad.albu import Compose
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


class RunnerInitialize:

    augs_dict: Dict[str, Compose]
    cfg: DictConfig
    dataset_dict: Dict[str, Dataset]
    model: Module

    def init_augs(self, augs_type: str) -> Compose:

        cfg = self.cfg.augs[augs_type]
        augs = albu.load(cfg.yaml, data_format="yaml")
        return augs

    def init_dataloader(self, data_type: str) -> DataLoader:

        cfg = self.cfg.dataloader[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, dataset=self.dataset_dict[data_type])

    def init_dataset(self, data_type: str) -> Dataset:

        cfg = self.cfg.dataset[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, augs_dict=self.augs_dict)

    def init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def init_criterion(self, criterion_type: str) -> _Loss:

        cfg = self.cfg.criterion[criterion_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, params=self.model.parameters())

    def _get_attr(self, name: str):

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
