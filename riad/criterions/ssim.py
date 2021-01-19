import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class SSIMLoss(Module):
    def __init__(self, window_size: int = 11, window_sigma: float = 1.5) -> None:

        self.window_size = window_size
        self.window_sigma = window_sigma
        self.window = self._create_gaussian_window(self.windown_size, self.windown_sigma)

    def forward(self, img1: Tensor, img2: Tensor, is_inference: bool = False) -> Tensor:

        if not self.window.is_cuda():
            self.window = self.window.to(img1.get_device())

        ssim_map = self._ssim(img1, img2)

        if is_inference:
            return ssim_map
        else:
            return ssim_map.mean()

    def _ssim(self, img1: Tensor, img2: Tensor) -> Tensor:


        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1 = F.conv2d(img1 - mu1, window, padding=window_size // 2, groups=channel)
        sigma2 = F.conv2d(img2 - mu2, window, padding=window_size // 2, groups=channel)
        sigma1_sq = sigma1.pow(2)
        sigma2_sq = sigma2.pow(2)
        sigma12 = sigma1 * sigma2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = numerator / denominator
        return ssim_map.mean()

    def _create_gaussian_window(window_size: int, sigma: float) -> Tensor:

        k = torch.arange(kernel_size).to(dtype=torch.float32)
        k -= (kernel_size - 1) / 2.0

        g = coords ** 2
        g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

        g /= g.sum()
        return g.unsqueeze(0)
