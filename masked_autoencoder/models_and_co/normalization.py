"""
This is the μ-law based normalization from:

Puah et al., "EEGDM: EEG Representation Learning via Generative Diffusion Model", 2025.
arXiv: https://arxiv.org/abs/2508.14086

GitHub: https://github.com/jhpuah/EEGDM

The staged μ-law normalization (with pre-scaling) is taken from their official GitHub implementation.

@misc{puah2025eegdm,
  title={EEGDM: EEG Representation Learning via Generative Diffusion Model},
  author={Jia Hong Puah and Sim Kuan Goh and Ziwei Zhang and Zixuan Ye and Chow Khuen Chan and Kheng Seang Lim and Si Lei Fong and Kok Sin Woon},
  year={2025},
  eprint={2508.14086},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
"""

import numpy as np

def mu_law(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def staged_mu_law(x, mu=255, scale=1.0):
    x = scale * x
    _x = mu_law(x, mu=mu)
    x[x > 1] = _x[x > 1]
    x[x < -1] = _x[x < -1]
    return x / scale


# PyTorch tensor-native implementation
import torch

def mu_law_torch(x, mu=255.0):
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, device=x.device, dtype=x.dtype))

def staged_mu_law_torch(x, mu=255.0, scale=1e4):
    x_scaled = x * scale
    x_mu = mu_law_torch(x_scaled, mu=mu)
    out = torch.where(x_scaled > 1, x_mu, x_scaled)
    out = torch.where(x_scaled < -1, x_mu, out)
    return out / scale
