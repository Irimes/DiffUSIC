

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _signal_to_covariance(signals_complex):
    S, M, N = signals_complex.shape
    # 批量矩阵乘法
    Rxx = np.einsum('smn,skn->smk', signals_complex,
                    signals_complex.conj()) / N
    return Rxx  # (S, M, M) complex


def _complex_to_2ch(cov_complex, dtype=np.float32):
    real_part = cov_complex.real.astype(dtype)
    imag_part = cov_complex.imag.astype(dtype)
    return np.stack([real_part, imag_part], axis=1)


def _2ch_to_complex(cov_2ch):
    if isinstance(cov_2ch, torch.Tensor):
        cov_2ch = cov_2ch.cpu().numpy()
    return cov_2ch[:, 0, :, :] + 1j * cov_2ch[:, 1, :, :]


class DOACovDataset(Dataset):

    def __init__(self, npz_path, split='train', test_ratio=0.1, seed=42):
        super().__init__()
        assert split in ('train', 'test')

        self.split = split

        # ---------- 加载原始信号数据 ----------
        data = np.load(npz_path, allow_pickle=True)

        clean_complex = (data['clean_signals_real']
                         + 1j * data['clean_signals_imag'])   # (S, M, N)
        received_complex = (data['received_signals_real']
                            + 1j * data['received_signals_imag'])
        angles = data['angles'].astype(np.float32)
        snr = data['snr'].astype(np.float32)
        num_targets = data['num_targets'].astype(np.int32)
        self.params = data['params'][0] if 'params' in data else {}

        self.M = clean_complex.shape[1]  # 阵元数
        self.N = clean_complex.shape[2]  # 快拍数

        # ---------- 划分训练/测试 ----------
        num_total = len(clean_complex)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(num_total)
        num_test = max(1, int(num_total * test_ratio))
        if split == 'test':
            sel = indices[:num_test]
        else:
            sel = indices[num_test:]

        clean_complex = clean_complex[sel]
        received_complex = received_complex[sel]
        angles = angles[sel]
        snr = snr[sel]
        num_targets = num_targets[sel]

        # 计算协方差矩阵 
        clean_cov = _signal_to_covariance(clean_complex)      # (S, M, M)
        received_cov = _signal_to_covariance(received_complex) # (S, M, M)
        clean_cov = (clean_cov + clean_cov.conj().transpose(0, 2, 1)) / 2

        # 转为 2 通道实数
        clean_2ch = _complex_to_2ch(clean_cov)      # (S, 2, M, M)
        received_2ch = _complex_to_2ch(received_cov) # (S, 2, M, M)

        # 逐样本归一化
        self.global_mean = 0.0
        self.global_std = 1.0

        clean_max = np.max(np.abs(clean_2ch), axis=(1, 2, 3), keepdims=True)
        clean_max = np.clip(clean_max, 1e-8, None)
        clean_2ch = clean_2ch / clean_max

        received_max = np.max(np.abs(received_2ch), axis=(1, 2, 3), keepdims=True)
        received_max = np.clip(received_max, 1e-8, None)
        received_2ch = received_2ch / received_max

        # 转 tensor 
        self.clean_cov = torch.from_numpy(clean_2ch)       # (S, 2, M, M)
        self.received_cov = torch.from_numpy(received_2ch) # (S, 2, M, M)
        self.angles = angles
        self.snr = snr
        self.num_targets = num_targets

        self.norm_info = {
            'norm_mode': 'sample_maxabs',
        }

        #打印信息
        print(f"[DOACovDataset] {split.upper()} 集加载完毕:")
        print(f"  文件       : {npz_path}")
        print(f"  样本数     : {len(self.clean_cov)}")
        print(f"  协方差形状 : {tuple(self.clean_cov.shape)} (S, 2, M, M)")
        print(f"  阵元/快拍  : M={self.M}, N={self.N}")
        print(f"  归一化     : 逐样本 max-abs 归一化")
        cr = self.clean_cov
        print(f"  Clean 值域 : [{cr.min().item():.4f}, {cr.max().item():.4f}]")
        rv = self.received_cov
        print(f"  Recv  值域 : [{rv.min().item():.4f}, {rv.max().item():.4f}]")

    def __len__(self):
        return len(self.clean_cov)

    def __getitem__(self, idx):
        
    def denormalize(self, x):

        return x  # 直接返回，MUSIC 不需要反归一化

    def to_complex_cov(self, x):
        return _2ch_to_complex(x)

    def get_norm_stats(self):
        return {'norm_mode': 'frobenius'}


def find_latest_npz(directory='DOA_Dataset'):
    if not os.path.isdir(directory):
        return None
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    if not npz_files:
        return None
    npz_files.sort()
    return os.path.join(directory, npz_files[-1])
