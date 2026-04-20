
import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1).view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class AntiRectifier(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class CovConditionEncoder2D(nn.Module):
    def __init__(self, M, cond_dim, hidden_ch=64, in_channels=2,
                 use_anti_rectifier=True):
        super().__init__()
        self.use_anti_rectifier = use_anti_rectifier

        if use_anti_rectifier:
            self.block1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_ch, kernel_size=2, padding=1),
                nn.GroupNorm(8, hidden_ch),
            )
            self.ar1 = AntiRectifier()   
            self.block2 = nn.Sequential(
                nn.Conv2d(hidden_ch * 2, hidden_ch, kernel_size=2, padding=1),
                nn.GroupNorm(8, hidden_ch),
            )
            self.ar2 = AntiRectifier()  
            self.block3 = nn.Sequential(
                nn.Conv2d(hidden_ch * 2, hidden_ch * 2, kernel_size=2, padding=1),
                nn.GroupNorm(8, hidden_ch * 2),
            )
            self.ar3 = AntiRectifier()
            final_spatial_dim = M + 3
            self.flattened_dim = (hidden_ch * 4) * (final_spatial_dim ** 2)
            self.out_proj = nn.Linear(self.flattened_dim, cond_dim)
            self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, condition):
        if self.use_anti_rectifier:
            h = self.block1(condition)
            h = self.ar1(h)
            h = self.block2(h)
            h = self.ar2(h)
            h = self.block3(h)
            h = self.ar3(h)
        h = h.view(h.size(0), -1)     
        h = self.out_proj(h)           
        return h

class SNREmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, snr_norm):
        # snr_norm: (B, 1)
        return self.net(snr_norm)


class KEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, k_norm):
        # k_norm: (B, 1)
        return self.net(k_norm)


class FiLMResBlock1D(nn.Module):
    def __init__(self, channels, tdim, cond_dim, dropout):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, channels),
            Swish(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, channels),
        )
        self.film = nn.Linear(cond_dim, channels * 2)
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        init.zeros_(self.film.weight)
        init.zeros_(self.film.bias)
        self.film.bias.data[:self.film.bias.shape[0] // 2] = 1.0  

    def forward(self, x, temb, cond_vec):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None]
        film_params = self.film(cond_vec) 
        scale, shift = film_params.chunk(2, dim=1)
        h = h * scale[:, :, None] + shift[:, :, None]
        h = self.block2(h)
        return x + h


class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)



#  V2模型
class ConditionalSpectrumUNet1D_V2(nn.Module):
    def __init__(self, T, spec_len, M=8, base_ch=128,
                 num_res_blocks=2, dropout=0.1,
                 use_snr_cond=True, cfg_drop_prob=0.1,
                 tau=1, use_anti_rectifier=True,
                 use_k_cond=False):
        super().__init__()
        self.spec_len = spec_len
        self.use_snr_cond = use_snr_cond
        self.use_k_cond = use_k_cond
        self.cfg_drop_prob = cfg_drop_prob
        self.tau = max(1, int(tau))

        tdim = base_ch * 4
        cond_dim = base_ch * 2  

        self.time_embedding = TimeEmbedding(T, base_ch, tdim)
        cond_in_ch = self.tau * 2  
        self.cond_encoder = CovConditionEncoder2D(
            M=M, cond_dim=cond_dim, in_channels=cond_in_ch,
            use_anti_rectifier=use_anti_rectifier)

        if use_snr_cond:
            self.snr_embedding = SNREmbedding(cond_dim)
        else:
            self.snr_embedding = None

        if use_k_cond:
            self.k_embedding = KEmbedding(cond_dim)
        else:
            self.k_embedding = None

        n_cond_parts = 1 + int(use_snr_cond) + int(use_k_cond)
        merge_in = cond_dim * n_cond_parts
        self.cond_merge = nn.Sequential(
            nn.Linear(merge_in, cond_dim),
            Swish(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.head = nn.Conv1d(1, base_ch, kernel_size=3, padding=1)

        ch = base_ch
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.down_channels = [ch] 
        for level in range(2):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    FiLMResBlock1D(ch, tdim, cond_dim, dropout))
            self.down_blocks.append(level_blocks)
            self.downsamples.append(Downsample1D(ch))
            self.down_channels.append(ch)


        self.mid_blocks = nn.ModuleList([
            FiLMResBlock1D(ch, tdim, cond_dim, dropout)
            for _ in range(num_res_blocks)
        ])

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for level in range(2):
            self.skip_projs.append(
                nn.Conv1d(ch * 2, ch, kernel_size=1))
            self.upsamples.append(Upsample1D(ch))
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    FiLMResBlock1D(ch, tdim, cond_dim, dropout))
            self.up_blocks.append(level_blocks)

        groups = 8 if ch % 8 == 0 else 1
        self.tail = nn.Sequential(
            nn.GroupNorm(groups, ch),
            Swish(),
            nn.Conv1d(ch, 1, kernel_size=3, padding=1),
        )

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
        for m in self.cond_merge.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        for m in self.skip_projs:
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    # def _build_cond_vec(self, condition, snr_norm=None, k_norm=None):
    #     parts = [self.cond_encoder(condition)]       
    #     if self.use_snr_cond and snr_norm is not None:
    #         parts.append(self.snr_embedding(snr_norm))   
    #     if self.use_k_cond and k_norm is not None:
    #         parts.append(self.k_embedding(k_norm))       

    #     merged = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    #     cond_vec = self.cond_merge(merged)           

    #     if self.training and self.cfg_drop_prob > 0:
    #         mask = (torch.rand(cond_vec.shape[0], 1, device=cond_vec.device)
    #                 > self.cfg_drop_prob).float()
    #         cond_vec = cond_vec * mask

    #     return cond_vec

    # def forward(self, x_t, t, condition, snr_norm=None, k_norm=None):
    #     temb = self.time_embedding(t)                    
    #     cond_vec = self._build_cond_vec(condition, snr_norm, k_norm) 

    #     # ---- 编码 ----
    #     h = self.head(x_t)                              
    #     for level in range(2):
    #         for block in self.down_blocks[level]:
    #             h = block(h, temb, cond_vec)
    #         skips.append(h)
    #         h = self.downsamples[level](h)                

    #     # ---- 中间 ----
    #     for block in self.mid_blocks:
    #         h = block(h, temb, cond_vec)

    #     # ---- 解码 ----
    #     for level in range(2):
    #         h = self.upsamples[level](h)                  
    #         if h.shape[-1] != skip.shape[-1]:
    #             h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
    #         h = torch.cat([h, skip], dim=1)              
    #         h = self.skip_projs[level](h)                  
    #             h = block(h, temb, cond_vec)

    #     return self.tail(h)

    def _build_cond_vec(self, condition, snr_norm=None, k_norm=None, force_uncond=False):
        if force_uncond:
            B = condition.shape[0]
            cond_dim = self.cond_merge[-1].out_features
            return torch.zeros(B, cond_dim, device=condition.device)

        parts = [self.cond_encoder(condition)]       

        if self.use_snr_cond and snr_norm is not None:
            parts.append(self.snr_embedding(snr_norm))   

        if self.use_k_cond and k_norm is not None:
            parts.append(self.k_embedding(k_norm))       

        merged = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        cond_vec = self.cond_merge(merged)           

    
        if self.training and self.cfg_drop_prob > 0:
            mask = (torch.rand(cond_vec.shape[0], 1, device=cond_vec.device)
                    > self.cfg_drop_prob).float()
            cond_vec = cond_vec * mask
        return cond_vec

    def forward(self, x_t, t, condition, snr_norm=None, k_norm=None, force_uncond=False):
        temb = self.time_embedding(t)                    
        cond_vec = self._build_cond_vec(condition, snr_norm, k_norm, force_uncond)

        h = self.head(x_t)                               
        skips = []
        for level in range(2):
            for block in self.down_blocks[level]:
                h = block(h, temb, cond_vec)
            skips.append(h)
            h = self.downsamples[level](h)                

        for block in self.mid_blocks:
            h = block(h, temb, cond_vec)

        for level in range(2):
            h = self.upsamples[level](h)                  
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)              
            h = self.skip_projs[level](h)             
            for block in self.up_blocks[level]:
                h = block(h, temb, cond_vec)

        return self.tail(h)