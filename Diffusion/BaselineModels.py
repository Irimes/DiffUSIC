"""
DOA Baseline 模型

包含的模型:
  1. SubspaceNet     (需训练, 输入: Rx_tau)
  2. DeepSFNS        (需训练, 输入: A + Y)
  3. DeepSSE         (需训练, 输入: 3ch cov)
  4. IQResNet        (需训练, 输入: raw signal)
  5. DOALowSNRNet    (需训练, 输入: 3ch cov)
  6. daMUSIC         (需训练, 输入: raw signal)
"""

import copy
import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor



def build_steering_matrix(M, angle_min, angle_max, angle_step, d_lambda=0.5):
    """ULA 导向矩阵"""
    grids_deg = np.arange(angle_min, angle_max + angle_step * 0.5, angle_step)
    grids_rad = np.deg2rad(grids_deg)
    n = np.arange(M, dtype=np.float64)
    A = np.exp(-1j * 2 * np.pi * d_lambda
               * n[:, None] * np.sin(grids_rad[None, :]))
    return A.astype(np.complex64), grids_deg.astype(np.float32)


def signal_to_3ch_cov(signal_complex):
    """
    signal_complex: (M, N) complex → (3, M, M) float32 [real, imag, angle]
    归一化: Frobenius 范数归一 (与参考 DeepSFNS 项目一致, 使用 np.cov)
    """
    if signal_complex.shape[1] == 1:
        cov = signal_complex @ signal_complex.conj().T
    else:
        cov = np.cov(signal_complex)
    cov = cov / max(np.linalg.norm(cov), 1e-12)
    return np.stack([cov.real, cov.imag, np.angle(cov)], axis=0).astype(np.float32)


def autocorrelation_matrix_np(X, lag):
    M, N = X.shape
    Rx = np.zeros((M, M), dtype=np.complex128)
    for t in range(N - lag):
        x1 = X[:, t:t+1]
        x2 = X[:, t+lag:t+lag+1].conj().T
        Rx += x1 @ x2
    Rx /= max(N - lag, 1)
    return np.concatenate([Rx.real, Rx.imag], axis=0).astype(np.float32)


def build_Rx_tau(signal_complex, tau=8):
    layers = [autocorrelation_matrix_np(signal_complex, lag=i) for i in range(tau)]
    return np.stack(layers, axis=0)


def grid_peaks_to_angles(probs, k, angle_min, angle_max, angle_step, threshold=0.5):
    grids_deg = np.arange(angle_min, angle_max + angle_step * 0.5, angle_step)
    n = len(grids_deg)
    if len(probs) != n:
        grids_deg = np.linspace(angle_min, angle_max, len(probs))

    # 取 top-k
    idx = np.argsort(probs)[::-1][:k]
    est = sorted([float(grids_deg[i]) for i in idx])
    return est


#  1. SubspaceNet

def _find_roots_torch(coefficients):
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    return torch.linalg.eigvals(A)


def _sum_of_diags_torch(matrix):
    n = matrix.shape[0]
    diag_sum = []
    for idx in range(-n + 1, n):
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


def _gram_diagonal_overload(Kx, eps, batch_size):
    out = []
    for i in range(batch_size):
        K = Kx[i]
        Kg = torch.matmul(torch.t(torch.conj(K)), K)
        Kg = Kg + eps * torch.diag(torch.ones(Kg.shape[0], device=Kg.device))
        out.append(Kg)
    return torch.stack(out, dim=0)


def _root_music_batch(Rz, labels, batch_size):
    doa_batches = []
    for i in range(batch_size):
        M = int(labels[i].sum().item()) if labels is not None else 1
        if M <= 0:
            M = 1
        R = Rz[i]
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        diag_sum = _sum_of_diags_torch(F)
        roots = _find_roots_torch(diag_sum)
        roots_sorted = roots[sorted(range(roots.shape[0]),
                                    key=lambda k: abs(abs(roots[k]) - 1))]
        mask = (torch.abs(roots_sorted) - 1) < 0
        roots_inside = roots_sorted[mask][:M]
        angles = torch.angle(roots_inside)
        doa = torch.arcsin((1 / (2 * np.pi * 0.5 * 1)) * angles)
        doa_batches.append(doa)
    return doa_batches


def _esprit_batch(Rz, labels, batch_size):
    doa_batches = []
    for i in range(batch_size):
        M_src = int(labels[i].sum().item()) if labels is not None else 1
        M_src = max(M_src, 1)
        R = Rz[i]
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M_src]
        Us_upper, Us_lower = Us[:-1], Us[1:]
        if Us_upper.shape[0] < M_src:
            doa_batches.append(torch.zeros(M_src, device=Rz.device))
            continue
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        phi_eig, _ = torch.linalg.eig(phi)
        angles = torch.angle(phi_eig)
        doa = -1 * torch.arcsin(torch.clamp((1 / np.pi) * angles.real, -1, 1))
        doa_batches.append(doa)
    return doa_batches


class SubspaceNet(nn.Module):
    def __init__(self, tau=8, diff_method="esprit"):
        super().__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        if diff_method.startswith("root_music"):
            self.diff_method = _root_music_batch
        else:
            self.diff_method = _esprit_batch

    def anti_rectifier(self, X):
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau, labels=None):
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        Rx_real = Rx_View[:, :self.N, :]
        Rx_imag = Rx_View[:, self.N:, :]
        Kx_tag = torch.complex(Rx_real, Rx_imag)
        Rz = _gram_diagonal_overload(Kx_tag, eps=1, batch_size=self.batch_size)
        doa_prediction = self.diff_method(Rz, labels, self.batch_size)
        return doa_prediction, Rz


#  2. DeepSFNS (BeamformNet)

class _RNNBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, is_bidirectional=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.gru = nn.GRU(input_size=in_dim, hidden_size=out_dim,
                          num_layers=2, batch_first=True,
                          bidirectional=is_bidirectional)

    def forward(self, x):
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        return x


class DeepSFNS(nn.Module):
    def __init__(self, M, num_grids, snapshots, em_d=256, is_bidirectional=True):
        super().__init__()
        self.m = M
        self.n = num_grids
        self.snapshots = snapshots
        self.k = em_d
        self.is_bidirectional = is_bidirectional
        self.Ann1 = _RNNBlock(2 * M, num_grids, em_d, is_bidirectional)
        self.Ynn2 = _RNNBlock(2 * M, snapshots, em_d, is_bidirectional)
        k_eff = 2 * em_d if is_bidirectional else em_d
        self.Bnn3 = _RNNBlock(k_eff, num_grids, k_eff)
        self.W_v = nn.Linear(num_grids, num_grids)
        self.W_k = nn.Linear(num_grids, num_grids)
        self.W_q = nn.Linear(snapshots, num_grids)
        self.W_b = nn.Linear(k_eff, 2 * M)
        self.scale = num_grids ** 0.5

    def forward(self, A, Y):

        A = torch.cat((A.real, A.imag), dim=1).float()
        Y_real = Y.real
        Y_imag = Y.imag
        Y_cat = torch.cat((Y_real, Y_imag), dim=1).float()

        A_out = self.Ann1(A)
        Y_out = self.Ynn2(Y_cat)

        A_v = self.W_v(A_out)
        A_k = self.W_k(A_out)
        Y_q = self.W_q(Y_out)

        scores = torch.matmul(Y_q, A_k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        B = torch.matmul(attn, A_v)
        B = self.Bnn3(B)
        B = B.permute(0, 2, 1)
        B = self.W_b(B)

        B_real = B[:, :, :self.m]
        B_imag = B[:, :, self.m:]

        Y_origin = torch.complex(Y_real.float(), Y_imag.float())
        B_complex = torch.complex(B_real, B_imag)

        P = B_complex @ Y_origin
        P = torch.abs(P) ** 2
        P = torch.mean(P, dim=2)
        P = torch.tanh(P)
        return P


#  3. DeepSSE

def _get_activation(activation: str = "relu"):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "gelu":
        return nn.GELU()
    return nn.ReLU()


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class _ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = _get_activation(activation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class _SpatialFeatureExtractor(nn.Module):
    def __init__(self, img_channels=3, layers=(2, 2), in_ch=32, out_ch=32,
                 activation="gelu"):
        super().__init__()
        self.out_channels = out_ch
        self.input_layers = nn.Sequential(
            nn.Conv2d(img_channels, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            _get_activation(activation))
        cur = in_ch
        self.res_layers = nn.ModuleList()
        for n_blocks in layers:
            blks = []
            for _ in range(n_blocks):
                blks.append(_ResidualBlock2D(cur, out_ch, activation=activation))
                cur = out_ch
            self.res_layers.append(nn.Sequential(*blks))

    def forward(self, x):
        x = self.input_layers(x)
        for layer in self.res_layers:
            x = layer(x)
        return x


class _PositionEncoding2D(nn.Module):
    def __init__(self, num_pos_feats=64, maxh=16, maxw=16):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        eyes = torch.ones(1, maxh, maxw)
        y_embed = eyes.cumsum(1, dtype=torch.float32)
        x_embed = eyes.cumsum(2, dtype=torch.float32)
        scale = 2 * math.pi
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pe = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.register_buffer("pe", pe)

    def forward(self, inp):
        return self.pe.repeat((inp.size(0), 1, 1, 1))


class _CALayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.05,
                 activation="gelu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, query, feature, pos):
        tgt1, _ = self.multihead_attn(query=query, key=feature + pos,
                                       value=feature)
        tgt = query + self.dropout1(tgt1)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)


class _CABlock(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm

    def forward(self, query, feature, pos=None):
        out = query
        for layer in self.layers:
            out = layer(out, feature, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


class DeepSSE(nn.Module):
    def __init__(self, steering_vector_np, num_class, num_antenna,
                 d_model=128, nhead=8, num_ca_layers=2, dim_feedforward=512,
                 dropout=0.05, activation="gelu",
                 sfe_layers=(2, 2), sfe_ch=32):
        super().__init__()
        self.sfe = _SpatialFeatureExtractor(
            img_channels=3, layers=sfe_layers, in_ch=sfe_ch,
            out_ch=sfe_ch, activation=activation)

        ca_layer = _CALayer(d_model, nhead, dim_feedforward, dropout, activation)
        norm = nn.LayerNorm(d_model)
        self.ags = _CABlock(ca_layer, num_ca_layers, norm)
        self.d_model = d_model

        self.pos_embed = _PositionEncoding2D(
            num_pos_feats=d_model // 2, maxh=num_antenna, maxw=num_antenna)
        self.transform = nn.Sequential(
            nn.Conv2d(sfe_ch, d_model, kernel_size=1))
        self.angle_projector = nn.Sequential(
            nn.Linear(3 * num_antenna, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model))

        self.fc_W = nn.Parameter(torch.Tensor(1, num_class, d_model))
        self.fc_b = nn.Parameter(torch.Tensor(1, num_class))
        nn.init.xavier_uniform_(self.fc_W)
        nn.init.zeros_(self.fc_b)

        sv = torch.from_numpy(steering_vector_np.astype(np.complex64))
        sv_feat = torch.cat((sv.real, sv.imag, torch.angle(sv)), dim=0).float()
        self.register_buffer("angle_embed", sv_feat.transpose(0, 1))

    def forward(self, x):
        bs = x.shape[0]
        sf = self.sfe(x)
        angle_embed = self.angle_projector(self.angle_embed)
        angle_embed = angle_embed.unsqueeze(1).repeat(1, bs, 1)

        pos = self.pos_embed(sf).flatten(2).permute(2, 0, 1)
        feat = self.transform(sf).flatten(2).permute(2, 0, 1)

        out = self.ags(query=angle_embed, feature=feat, pos=pos)
        out = out.transpose(0, 1)
        out = (self.fc_W * out).sum(-1) + self.fc_b
        return torch.sigmoid(out)


#  4. IQResNet

class _ResLayer1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 3), stride=(1, 2), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 3), stride=(1, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, (1, 1), stride=(1, 2))

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class IQResNet(nn.Module):
    def __init__(self, num_classes, num_antennas):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(1, 64, (2 * num_antennas, 5),
                      stride=(2 * num_antennas, 1)),
            nn.BatchNorm2d(64), nn.ReLU())
        self.f2 = nn.MaxPool2d((1, 3), stride=(1, 2))
        self.f3 = _ResLayer1D(64, 64)
        self.f4 = _ResLayer1D(64, 128)
        self.f5 = _ResLayer1D(128, 256)
        self.f6 = _ResLayer1D(256, 512)
        self.fc = nn.Linear(512, num_classes)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = torch.cat((x.real, x.imag), dim=1).float().unsqueeze(1)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        x = torch.mean(x, dim=-1).squeeze(-1)
        return self.out(self.fc(x))

#  5. DOALowSNRNet

class DOALowSNRNet(nn.Module):
    def __init__(self, num_out_grids=121):
        Conv2d(stride=2) + Conv2d(k=2)×N + Flatten + FC(4096→2048→1024→out).
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 256, (3, 3), stride=2, padding=0),
            nn.BatchNorm2d(256), nn.ReLU())
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, (2, 2), stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU())
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 256, (2, 2), stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU())
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 256, (2, 2), stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.LazyLinear(4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(1024, num_out_grids), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        if x.shape[-1] >= 2 and x.shape[-2] >= 2:
            x = self.conv_layer4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


#  6. daMUSIC

class DeepAugmentMusic(nn.Module):
    def __init__(self, num_antennas, d_lambda, angle_min, angle_max, angle_step):
        super().__init__()
        self._num_antennas = num_antennas
        self._angle_grids = torch.arange(angle_min, angle_max + angle_step * 0.5, angle_step).float()

        self.norm = nn.BatchNorm1d(2 * num_antennas)
        self.gru = nn.GRU(input_size=2 * num_antennas,
                          hidden_size=2 * num_antennas)
        self.linear = nn.Linear(2 * num_antennas,
                                2 * num_antennas * num_antennas)
        self.eig_vec_pro = nn.Sequential(
            nn.Linear(2 * num_antennas, 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, num_antennas), nn.Sigmoid())
        self.peak_finder = nn.Sequential(
            nn.Linear(len(self._angle_grids), 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, 2 * num_antennas), nn.ReLU(),
            nn.Linear(2 * num_antennas, num_antennas - 1))

        # 预计算导向矢量
        pos = torch.arange(0, num_antennas).float() * d_lambda
        delay = pos.view(-1, 1) @ torch.sin(torch.deg2rad(self._angle_grids)).view(1, -1)
        sv = torch.exp(-2j * np.pi * delay.double()).cfloat()
        self.register_buffer("steering_vectors", sv)

    def forward(self, x):
        x = torch.cat((x.real, x.imag), dim=1).float()
        x = self.norm(x)
        x = x.permute(2, 0, 1)
        _, x = self.gru(x)
        x = self.linear(x)
        x = x.reshape(-1, 2 * self._num_antennas, self._num_antennas)
        x = torch.complex(x[:, :self._num_antennas, :],
                           x[:, self._num_antennas:, :])

        eig_val, eig_vec = torch.linalg.eig(x)
        prob = self.eig_vec_pro(torch.cat((eig_val.real, eig_val.imag), dim=1))
        prob = torch.diag_embed(prob)
        noise_space = torch.complex(
            torch.bmm(prob, eig_vec.real),
            torch.bmm(prob, eig_vec.imag))

        v = noise_space.transpose(1, 2).conj() @ self.steering_vectors
        spectrum = 1.0 / (torch.linalg.norm(v, axis=1) ** 2 + 1e-12)
        spectrum = spectrum.float()
        return self.peak_finder(spectrum)
