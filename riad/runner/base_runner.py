from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

from omegaconf.dictconfig import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from riad import transforms
from riad.transforms import Compose


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.preprocesses = {k: self._init_preprocess(k) for k in self.cfg.preprocess.keys()}
        self.datasets = {k: self._init_dataset(k) for k in self.cfg.dataset.keys()}
        self.dataloaders = {k: self._init_dataloader(k) for k in self.cfg.dataloader.keys()}
        self.model = self._init_model().to(self.cfg.params.device)
        self.optimizer = self._init_optimizer()
        self.criterions = {k: self._init_criterion(k) for k in self.cfg.criterion.keys()}

    def _init_preprocess(self, mode: str) -> Compose:

        cfg = self.cfg.preprocess[mode]
        return transforms.load(cfg.yaml, data_format="yaml")

    def _init_dataset(self, mode: str) -> Dataset:

        cfg = self.cfg.dataset[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), preprocess=self.preprocesses[mode])

    def _init_dataloader(self, mode: str) -> DataLoader:

        cfg = self.cfg.dataloader[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[mode])

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_criterion(self, mode: str) -> Module:

        cfg = self.cfg.criterion[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=self.model.parameters())

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    @abstractmethod
    def _train(self, epoch: int) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _validate(self, epoch: int) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _test(self) -> None:

        raise NotImplementedError()
