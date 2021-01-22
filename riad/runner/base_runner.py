import os
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import torch
from albumentations import Compose
from omegaconf.dictconfig import DictConfig
from torch.nn import Module
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
        self.criterions = {k: self._init_criterions(k) for k in self.cfg.criterions.keys()}
        self.early_stopping = self._init_early_stopping()

    def _init_transforms(self, mode: str) -> Compose:

        transforms = []
        for cfg in self.cfg.transforms[mode]:
            attr = self._get_attr(cfg.name)
            transforms.append(attr(**cfg.get("args", {})))
        return Compose(transforms)

    def _init_datasets(self, mode: str) -> Dataset:

        cfg = self.cfg.datasets[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), transforms=self.transforms[mode])

    def _init_dataloaders(self, mode: str) -> DataLoader:

        cfg = self.cfg.dataloaders[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[mode])

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_criterions(self, mode: str) -> Module:

        cfg = self.cfg.criterions[mode]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=self.model.parameters())

    def _init_early_stopping(self) -> EarlyStopping:

        cfg = self.cfg.early_stopping
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    def run(self) -> None:

        interval = self.cfg.params.epochs // 10
        pbar = tqdm(range(1, self.cfg.params.epochs + 1), desc="epochs")
        for epoch in pbar:
            self._train(epoch)
            val_loss = self._validate(epoch)

            if self.early_stopping(val_loss):
                torch.save(self.model.state_dict(), "model.pth")
                self._test(epoch)
                print(f"Early stopped at {epoch} epoch")
                sys.exit(0)

            if epoch % interval == 0:
                os.makedirs(f"epochs/{epoch}")
                self._test(epoch)

    @abstractmethod
    def _train(self, epoch: int) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _validate(self, epoch: int) -> float:

        raise NotImplementedError()

    @abstractmethod
    def _test(self, epoch: int) -> None:

        raise NotImplementedError()
