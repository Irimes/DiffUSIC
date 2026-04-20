
#谱条件扩散 V2

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float().to(t.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class ConditionalSpectrumDiffusionTrainerV2(nn.Module):
    #训练器

    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition, snr_norm=None, k_norm=None):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        noise_pred = self.model(x_t, t, condition, snr_norm=snr_norm, k_norm=k_norm)
        return F.mse_loss(noise_pred, noise, reduction='none')

    def training_terms(self, x_0, condition, snr_norm=None, k_norm=None):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)

        sqrt_ab = extract(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_1mab = extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        x_t = sqrt_ab * x_0 + sqrt_1mab * noise
        noise_pred = self.model(x_t, t, condition, snr_norm=snr_norm, k_norm=k_norm)

        noise_loss_map = F.mse_loss(noise_pred, noise, reduction='none')
        x0_pred = (x_t - sqrt_1mab * noise_pred) / torch.clamp(sqrt_ab, min=1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return {
            'noise_loss_map': noise_loss_map,
            'x0_pred': x0_pred,
            'x0_target': x_0,
            't': t
        }


class ConditionalSpectrumDiffusionSamplerV2(nn.Module):
    #采样器

    def __init__(self, model, beta_1, beta_T, T, cfg_scale=2.0):
        super().__init__()
        self.model = model
        self.T = T
        self.cfg_scale = cfg_scale

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def p_mean_variance(self, x_t, t, condition, snr_norm=None, k_norm=None):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        # 1. 有条件预测
        eps_cond = self.model(x_t, t, condition, snr_norm=snr_norm, k_norm=k_norm)

        if self.cfg_scale != 1.0:
            # 2. 无条件预测
            eps_uncond = self.model(x_t, t, condition, snr_norm=snr_norm, k_norm=k_norm, force_uncond=True)
            
            # 3. CFG 引导
            eps = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        mean = (extract(self.coeff1, t, x_t.shape) * x_t
                - extract(self.coeff2, t, x_t.shape) * eps)
        return mean, var

    @torch.no_grad()
    def forward(self, x_T, condition, snr_norm=None, k_norm=None):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_t.shape[0]], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t, t, condition, snr_norm=snr_norm, k_norm=k_norm)
            noise = torch.randn_like(x_t) if time_step > 0 else 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).sum() == 0, 'nan in spectrum diffusion'
        return torch.clamp(x_t, -1.0, 1.0)