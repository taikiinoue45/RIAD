import os
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import torch
from albumentations import Compose
from omegaconf.dictconfig import DictConfig
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from riad.utils import EarlyStopping


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.transforms = {k: self._init_transforms(k) for k in self.cfg.transforms.keys()}
        self.datasets = {k: self._init_datasets(k) for k in self.cfg.datasets.keys()}
        self.dataloaders = {k: self._init_dataloaders(k) for k in self.cfg.dataloaders.keys()}
        self.model = self._init_model().to(self.cfg.params.device)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.criterions = {k: self._init_criterions(k) for k in self.cfg.criterions.keys()}
        self.early_stopping = self._init_early_stopping()

    def _init_transforms(self, key: str) -> Compose:

        transforms = []
        for cfg in self.cfg.transforms[key]:
            attr = self._get_attr(cfg.name)
            transforms.append(attr(**cfg.get("args", {})))
        return Compose(transforms)

    def _init_datasets(self, key: str) -> Dataset:

        cfg = self.cfg.datasets[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), transforms=self.transforms[key])

    def _init_dataloaders(self, key: str) -> DataLoader:

        cfg = self.cfg.dataloaders[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[key])

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_criterions(self, key: str) -> Module:

        cfg = self.cfg.criterions[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=self.model.parameters())

    def _init_scheduler(self) -> _LRScheduler:

        cfg = self.cfg.scheduler
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), optimizer=self.optimizer)

    def _init_early_stopping(self) -> EarlyStopping:

        cfg = self.cfg.early_stopping
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    def run(self) -> None:

        pbar = tqdm(range(1, self.cfg.params.epochs + 1), desc="epochs")
        for epoch in pbar:
            self._train(epoch)
            val_loss = self._validate(epoch)
            self.scheduler.step()

            if self.early_stopping(val_loss):
                torch.save(self.model.state_dict(), "model.pth")
                os.makedirs(f"epochs/{epoch}")
                self._test(epoch)
                print(f"Early stopped at {epoch} epoch")
                sys.exit(0)

            if epoch % 10 == 0:
                os.makedirs(f"epochs/{epoch}")
                self._test(epoch)

        torch.save(self.model.state_dict(), "model.pth")

    @abstractmethod
    def _train(self, epoch: int) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _validate(self, epoch: int) -> float:

        raise NotImplementedError()

    @abstractmethod
    def _test(self, epoch: int) -> None:

        raise NotImplementedError()
