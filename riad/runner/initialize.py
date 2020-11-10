from importlib import import_module

import riad.albu as albu
from omegaconf import DictConfig
from riad.albu import Compose
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Dataset


class RunnerInitialize:

    cfg: DictConfig
    dataset: Dataset
    augs: Compose

    def init_augs(self, cfg: DictConfig) -> Compose:

        augs = albu.load(cfg.yaml, data_format="yaml")
        albu.save(augs, "hydra/augs.yaml", data_format="yaml")
        return augs

    def init_dataloader(self, cfg: DictConfig) -> DataLoader:

        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, dataset=self.dataset)

    def init_dataset(self, cfg: DictConfig) -> Dataset:

        attr = self._get_attr(cfg.name)
        return attr(**cfg.args, augs=self.augs)

    def init_model(self, cfg: DictConfig) -> Module:

        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def init_criterion(self, cfg: DictConfig) -> _Loss:

        attr = self._get_attr(cfg.name)
        return attr(**cfg.args)

    def _get_attr(self, name: str):

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
