"""
The datasets' photometric augmentation applied to a whole training batch on its device (the
GPU) instead of to each image on the CPU: colour jitter, blur, simulated low light and random
erasing. On colour images these cost the CPU more than the networks cost the GPU (on EO-1's
384x384 colour crops, the hue jitter alone converts every crop to HSV and back), which left the
GPU waiting for data.

Every image gets its own random draws from the distributions of the per-image transforms
(dataloader._photometric_augmentation: torchvision's ColorJitter and GaussianBlur, RandomLowLight;
torchvision's RandomErasing), and each operation computes what torchvision's does. Gray images
(equal channels: the datasets expand one channel to three) are treated as one channel is on the
CPU: no saturation or hue change, contrast about the image mean, and gray erasing noise.
"""
import math

import torch
import torch.nn.functional as F


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    """[N, 3, H, W] -> [N, 1, H, W], as torchvision's rgb_to_grayscale."""
    r, g, b = x.unbind(dim=-3)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(-3)


def _blend(a: torch.Tensor, b: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """As torchvision's _blend, with one ratio per image ([N, 1, 1, 1])."""
    return (ratio * a + (1.0 - ratio) * b).clamp(0.0, 1.0)


def _rgb2hsv(img: torch.Tensor) -> torch.Tensor:
    """As torchvision's _rgb2hsv."""
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img: torch.Tensor) -> torch.Tensor:
    """As torchvision's _hsv2rgb."""
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class BatchPhotometric:
    """
    ``dataloader._photometric_augmentation(low_light, noise)`` and, with ``erasing`` > 0,
    ``T.RandomErasing(p=erasing, scale, ratio, value="random")``, for a batch [B, 3, H, W] of
    float images in [0, 1] (call it on the training batch on the GPU).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.02,
        blur_p: float = 0.3,
        blur_sigma: tuple[float, float] = (0.1, 1.5),
        blur_kernel: int = 5,
        low_light: float = 0.0,
        low_light_gain: tuple[float, float] = (0.03, 0.6),
        low_light_gamma: tuple[float, float] = (0.8, 1.5),
        noise: tuple[float, float] = (0.005, 0.02),
        erasing: float = 0.0,
        erasing_scale: tuple[float, float] = (0.02, 0.12),
        erasing_ratio: tuple[float, float] = (0.3, 3.3),
    ):
        # ColorJitter's ranges: factors in [max(0, 1 - x), 1 + x], hue in [-hue, hue].
        self.ranges = [(max(0.0, 1.0 - brightness), 1.0 + brightness), (max(0.0, 1.0 - contrast), 1.0 + contrast),
                       (max(0.0, 1.0 - saturation), 1.0 + saturation), (-hue, hue)]
        self.blur_p, self.blur_sigma, self.blur_kernel = blur_p, blur_sigma, blur_kernel
        self.low_light, self.low_light_gain, self.low_light_gamma, self.noise = low_light, low_light_gain, low_light_gamma, noise
        self.erasing, self.erasing_scale, self.erasing_ratio = erasing, erasing_scale, erasing_ratio

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = images.clone()
        # Gray images stay one channel on the CPU, unless simulated low light (which adds
        # noise per channel) expands every image to three first.
        gray = (x[:, 0] == x[:, 1]).flatten(1).all(1) & (x[:, 1] == x[:, 2]).flatten(1).all(1)
        if self.low_light > 0:
            gray = torch.zeros_like(gray)
        x = self.color_jitter(x, gray, *self.jitter_params(len(x), x.device))
        x = self.blur(x, *self.blur_params(len(x), x.device))
        if self.low_light > 0:
            x = self.darken(x)
        if self.erasing > 0:
            x = self.erase(x, gray)
        return x

    # Colour jitter -------------------------------------------------------------------------

    def jitter_params(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per image: the four factors [n, 4] (brightness, contrast, saturation, hue) and the
        order to apply them in [n, 4] (a random permutation each, as ColorJitter draws)."""
        low = torch.tensor([r[0] for r in self.ranges], device=device)
        high = torch.tensor([r[1] for r in self.ranges], device=device)
        factors = low + (high - low) * torch.rand(n, 4, device=device)
        order = torch.rand(n, 4, device=device).argsort(dim=1)
        return factors, order

    def color_jitter(self, x: torch.Tensor, gray: torch.Tensor, factors: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        for slot in range(4):
            for op in range(4):
                selected = order[:, slot] == op
                if op >= 2:
                    selected &= ~gray  # saturation and hue leave one channel unchanged
                index = torch.nonzero(selected).flatten()
                if index.numel() == 0:
                    continue
                sub, factor = x[index], factors[index, op].view(-1, 1, 1, 1)
                if op == 0:
                    sub = _blend(sub, torch.zeros_like(sub), factor)
                elif op == 1:
                    gray_sub = gray[index].view(-1, 1, 1, 1)
                    mean = torch.where(gray_sub, sub.mean(dim=(-3, -2, -1), keepdim=True),
                                       _grayscale(sub).mean(dim=(-3, -2, -1), keepdim=True))
                    sub = _blend(sub, mean, factor)
                elif op == 2:
                    sub = _blend(sub, _grayscale(sub), factor)
                else:
                    hsv = _rgb2hsv(sub)
                    h = (hsv[:, 0] + factor.view(-1, 1, 1)) % 1.0
                    sub = _hsv2rgb(torch.stack((h, hsv[:, 1], hsv[:, 2]), dim=-3))
                x[index] = sub
        return x

    # Blur ----------------------------------------------------------------------------------

    def blur_params(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per image: whether it is blurred (RandomApply's p) and its sigma."""
        applied = torch.rand(n, device=device) < self.blur_p
        sigma = self.blur_sigma[0] + (self.blur_sigma[1] - self.blur_sigma[0]) * torch.rand(n, device=device)
        return applied, sigma

    def blur(self, x: torch.Tensor, applied: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """As torchvision's gaussian_blur: a square kernel, reflect padding."""
        index = torch.nonzero(applied).flatten()
        if index.numel() == 0:
            return x
        k = self.blur_kernel
        half = (k - 1) * 0.5
        t = torch.linspace(-half, half, steps=k, device=x.device, dtype=x.dtype)
        pdf = torch.exp(-0.5 * (t[None, :] / sigma[index, None]).pow(2))
        kernel1d = pdf / pdf.sum(dim=1, keepdim=True)
        kernel2d = kernel1d[:, :, None] * kernel1d[:, None, :]          # [m, k (y), k (x)]
        sub = x[index]
        m, c, h, w = sub.shape
        padded = F.pad(sub, [k // 2] * 4, mode="reflect").reshape(1, m * c, h + 2 * (k // 2), w + 2 * (k // 2))
        weight = kernel2d.repeat_interleave(c, dim=0).unsqueeze(1)          # [m * c, 1, k, k]
        x[index] = F.conv2d(padded, weight, groups=m * c).reshape(m, c, h, w)
        return x

    # Simulated low light -------------------------------------------------------------------

    def darken(self, x: torch.Tensor) -> torch.Tensor:
        """As dataloader.RandomLowLight, per image."""
        index = torch.nonzero(torch.rand(len(x), device=x.device) < self.low_light).flatten()
        if index.numel() == 0:
            return x
        u = torch.rand(index.numel(), 3, device=x.device)
        g0, g1 = self.low_light_gain
        gain = torch.exp(math.log(g0) + u[:, 0] * math.log(g1 / g0)).view(-1, 1, 1, 1)
        gamma = (self.low_light_gamma[0] + u[:, 1] * (self.low_light_gamma[1] - self.low_light_gamma[0])).view(-1, 1, 1, 1)
        sigma = (self.noise[0] + u[:, 2] * (self.noise[1] - self.noise[0])).view(-1, 1, 1, 1)
        sub = x[index].clamp(0.0, 1.0)
        # Dark level: median of the border, per channel.
        border = torch.cat([sub[:, :, :8].flatten(2), sub[:, :, -8:].flatten(2),
                            sub[:, :, :, :8].flatten(2), sub[:, :, :, -8:].flatten(2)], dim=2)
        floor = border.median(dim=2).values[:, :, None, None]
        sub = torch.minimum(sub, floor) + (sub - floor).clamp_min(0.0).pow(gamma) * gain
        sub = sub + torch.randn_like(sub) * sigma
        x[index] = (sub.clamp(0.0, 1.0) * 255.0).round() / 255.0
        return x

    # Random erasing ------------------------------------------------------------------------

    def erase(self, x: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        """As torchvision's RandomErasing(value="random"): a rectangle of standard normal noise
        (gray noise on gray images), its area and aspect drawn up to 10 times until it fits."""
        n, c, height, width = x.shape
        index = torch.nonzero(torch.rand(n, device=x.device) < self.erasing).flatten()
        if index.numel() == 0:
            return x
        m, tries = index.numel(), 10
        area = height * width * (self.erasing_scale[0] + (self.erasing_scale[1] - self.erasing_scale[0]) * torch.rand(m, tries, device=x.device))
        log_low, log_high = math.log(self.erasing_ratio[0]), math.log(self.erasing_ratio[1])
        aspect = torch.exp(log_low + (log_high - log_low) * torch.rand(m, tries, device=x.device))
        h = torch.round(torch.sqrt(area * aspect)).long()
        w = torch.round(torch.sqrt(area / aspect)).long()
        fits = (h < height) & (w < width)
        first = fits.long().argmax(dim=1)                      # the first draw that fits
        erased = fits.any(dim=1)
        h = h.gather(1, first[:, None]).squeeze(1)
        w = w.gather(1, first[:, None]).squeeze(1)
        top = (torch.rand(m, device=x.device) * (height - h + 1)).long()
        left = (torch.rand(m, device=x.device) * (width - w + 1)).long()
        rows = torch.arange(height, device=x.device)[None, :, None]
        cols = torch.arange(width, device=x.device)[None, None, :]
        inside = ((rows >= top[:, None, None]) & (rows < (top + h)[:, None, None]) &
                  (cols >= left[:, None, None]) & (cols < (left + w)[:, None, None]) & erased[:, None, None])
        noise = torch.randn(m, c, height, width, device=x.device, dtype=x.dtype)
        noise = torch.where(gray[index].view(-1, 1, 1, 1), noise[:, :1].expand_as(noise), noise)
        x[index] = torch.where(inside[:, None], noise, x[index])
        return x
