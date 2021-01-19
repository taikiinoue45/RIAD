import math
import os
import random
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from kornia import gaussian_blur2d
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from torch import Tensor
from tqdm import tqdm

from riad.criterions import MSGMS_Score
from riad.metrics import compute_auroc
from riad.runner import BaseRunner


class Runner(BaseRunner):
    def run(self) -> None:

        interval = self.cfg.params.epochs // 10
        pbar = tqdm(range(1, self.cfg.params.epochs + 1), desc="epochs")
        for epoch in pbar:
            self._train(epoch)
            if epoch % interval == 0:
                os.makedirs(f"epochs/{epoch}")
                self._validate(epoch)
        self._test()

    def _train(self, epoch: int) -> None:

        metrics: Dict[str, List[float]] = {
            "MSE Loss": [],
            "MSGMS Loss": [],
            "SSIM Loss": [],
            "Total Loss": [],
        }
        self.model.train()
        for _, mb_img, _ in self.dataloaders["train"]:

            self.optimizer.zero_grad()

            mb_img = mb_img.to(self.cfg.params.device)
            cutout_size = random.choice(self.cfg.params.cutout_sizes)
            mb_reconst = self._reconstruct(mb_img, cutout_size)

            mb_mse = self.criterions["MSE"](mb_img, mb_reconst)
            mb_msgms = self.criterions["MSGMS"](mb_img, mb_reconst)
            mb_ssim = self.criterions["SSIM"](mb_img, mb_reconst)
            mb_total = mb_msgms + mb_ssim + mb_mse
            mb_total.backward()
            self.optimizer.step()

            metrics["MSE Loss"].append(mb_mse.detach().cpu().item())
            metrics["MSGMS Loss"].append(mb_msgms.detach().cpu().item())
            metrics["SSIM Loss"].append(mb_ssim.detach().cpu().item())
            metrics["Total Loss"].append(mb_total.detach().cpu().item())

        mlflow.log_metrics({k: mean(v) for k, v in metrics.items()}, step=epoch)

    def _validate(self, epoch: int) -> None:

        self.model.eval()
        artifacts: Dict[str, List[NDArray]] = {
            "img": [],
            "reconst": [],
            "gt": [],
            "amap": [],
        }
        msgms_score = MSGMS_Score()
        for mb_img_path, mb_img, mb_gt in self.dataloaders["val"]:

            mb_amap = 0
            with torch.no_grad():
                for cutout_size in self.cfg.params.cutout_sizes:
                    mb_img = mb_img.to(self.cfg.params.device)
                    mb_reconst = self._reconstruct(mb_img, cutout_size)
                    mb_amap += msgms_score(mb_img, mb_reconst) / (256 ** 2)

            mb_amap = gaussian_blur2d(mb_amap, kernel_size=(3, 3), sigma=(7.0, 7.0))
            artifacts["amap"].extend(mb_amap.squeeze().detach().cpu().numpy())
            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy())
            artifacts["reconst"].extend(mb_reconst.permute(0, 2, 3, 1).detach().cpu().numpy())
            artifacts["gt"].extend(mb_gt.detach().cpu().numpy())

        ep_amap = np.array(artifacts["amap"])
        ep_amap = (ep_amap - ep_amap.min()) / (ep_amap.max() - ep_amap.min())
        artifacts["amap"] = list(ep_amap)

        auroc = compute_auroc(epoch, np.array(artifacts["amap"]), np.array(artifacts["gt"]))
        mlflow.log_metric("AUROC", auroc, step=epoch)

        self._savefig(
            epoch,
            artifacts["img"],
            artifacts["reconst"],
            artifacts["amap"],
            artifacts["gt"],
        )

    def _test(self) -> None:

        pass

    def _reconstruct(self, mb_img: Tensor, cutout_size: int) -> Tensor:

        _, _, h, w = mb_img.shape
        num_disjoint_masks = self.cfg.params.num_disjoint_masks
        disjoint_masks = self._create_disjoint_masks((h, w), cutout_size, num_disjoint_masks)

        mb_reconst = 0
        for mask in disjoint_masks:
            mb_cutout = mb_img * mask
            mb_inpaint = self.model(mb_cutout)
            mb_reconst += mb_inpaint * (1 - mask)

        return mb_reconst

    def _create_disjoint_masks(
        self,
        img_size: Tuple[int, int],
        cutout_size: int = 8,
        num_disjoint_masks: int = 3,
    ) -> List[Tensor]:

        img_h, img_w = img_size
        grid_h = math.ceil(img_h / cutout_size)
        grid_w = math.ceil(img_w / cutout_size)
        num_grids = grid_h * grid_w
        disjoint_masks = []
        for grid_ids in np.array_split(np.random.permutation(num_grids), num_disjoint_masks):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_ids] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(cutout_size, axis=0).repeat(cutout_size, axis=1)
            mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            mask = mask.to(self.cfg.params.device)
            disjoint_masks.append(mask)

        return disjoint_masks

    def _savefig(
        self,
        epoch: int,
        ep_img: List[NDArray],
        ep_reconst: List[NDArray],
        ep_amap: List[NDArray],
        ep_gt: List[NDArray],
    ) -> None:

        for i, (img, reconst, score, gt) in enumerate(zip(ep_img, ep_reconst, ep_amap, ep_gt)):

            # How to get two subplots to share the same y-axis with a single colorbar
            # https://stackoverflow.com/a/38940369
            grid = ImageGrid(
                fig=plt.figure(figsize=(16, 4)),
                rect=111,
                nrows_ncols=(1, 4),
                axes_pad=0.15,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.15,
            )

            img = self.denormalize(img)
            reconst = self.denormalize(reconst)

            grid[0].imshow(img)
            grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

            grid[1].imshow(reconst)
            grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

            grid[2].imshow(img)
            grid[2].imshow(gt, alpha=0.3, cmap="Reds")
            grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

            grid[3].imshow(img)
            im = grid[3].imshow(score, alpha=0.3, cmap="jet")
            grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            grid[3].cax.colorbar(im)
            grid[3].cax.toggle_label(True)

            plt.savefig(f"epochs/{epoch}/{i}.png", bbox_inches="tight")
            plt.close()

    def denormalize(self, img: NDArray) -> NDArray:

        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = (img * std + mean) * 255.0
        return img.astype(np.uint8)
