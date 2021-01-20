from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class MSGMSLoss(Module):
    def __init__(self, num_scales: int = 3, in_channels: int = 3) -> None:

        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.prewitt_x, self.prewitt_y = self._create_prewitt_kernel()

    def forward(self, img1: Tensor, img2: Tensor, as_loss: bool = True) -> Tensor:

        if not self.prewitt_x.is_cuda or not self.prewitt_y.is_cuda:
            self.prewitt_x = self.prewitt_x.to(img1.device)
            self.prewitt_y = self.prewitt_y.to(img1.device)

        b, c, h, w = img1.shape
        msgms_map = 0
        for scale in range(self.num_scales):

            if scale > 0:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)

            gms_map = self._gms(img1, img2)
            msgms_map += F.interpolate(gms_map, size=(h, w), mode="bilinear", align_corners=False)

        if as_loss:
            return torch.mean(1 - msgms_map / self.num_scales)
        else:
            return torch.mean(1 - msgms_map / self.num_scales, axis=1).unsqueeze(1)

    def _gms(self, img1: Tensor, img2: Tensor) -> Tensor:

        gm1_x = F.conv2d(img1, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm1_y = F.conv2d(img1, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm1 = torch.sqrt(gm1_x ** 2 + gm1_y ** 2 + 1e-12)

        gm2_x = F.conv2d(img2, self.prewitt_x, stride=1, padding=1, groups=self.in_channels)
        gm2_y = F.conv2d(img2, self.prewitt_y, stride=1, padding=1, groups=self.in_channels)
        gm2 = torch.sqrt(gm2_x ** 2 + gm2_y ** 2 + 1e-12)

        # Constant c from the following paper. https://arxiv.org/pdf/1308.3052.pdf
        c = 0.0026
        numerator = 2 * gm1 * gm2 + c
        denominator = gm1 ** 2 + gm2 ** 2 + c
        return numerator / (denominator + 1e-12)

    def _create_prewitt_kernel(self) -> Tuple[Tensor, Tensor]:

        prewitt_x = torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_x = prewitt_x.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        prewitt_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0  # (1, 1, 3, 3)
        prewitt_y = prewitt_y.repeat(self.in_channels, 1, 1, 1)  # (self.in_channels, 1, 3, 3)
        return (prewitt_x, prewitt_y)
