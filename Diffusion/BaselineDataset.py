import os
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):

    def __init__(self, npz_path, split='train', test_ratio=0.1, seed=42,
                 angle_min=-60.0, angle_max=60.0, baseline_angle_step=1.0,
                 d_lambda=0.5, tau=8):
        super().__init__()
        assert split in ('train', 'test')

        data = np.load(npz_path, allow_pickle=True)
        angles = data['angles'].astype(np.float32)
        snr = data['snr'].astype(np.float32)
        num_targets = data['num_targets'].astype(np.int32)
        total = len(angles)
        del data
-
        rng = np.random.RandomState(seed)
        idx = rng.permutation(total)
        num_test = max(1, int(total * test_ratio))
        sel = idx[:num_test] if split == 'test' else idx[num_test:]

        self.angles = angles[sel]
        self.snr = snr[sel]
        self.num_targets = num_targets[sel]
        del angles, snr, num_targets


        cache_dir = os.path.join(os.path.dirname(npz_path), '_bl_cache')
        os.makedirs(cache_dir, exist_ok=True)

        real_npy = os.path.join(cache_dir, 'received_signals_real.npy')
        imag_npy = os.path.join(cache_dir, 'received_signals_imag.npy')

        if not os.path.isfile(real_npy) or not os.path.isfile(imag_npy):
            print("[BaselineDataset] 缓存npy到目录...")
            with zipfile.ZipFile(npz_path) as z:
                z.extract('received_signals_real.npy', cache_dir)
                z.extract('received_signals_imag.npy', cache_dir)

        self._real_path = real_npy
        self._imag_path = imag_npy

        _mm = np.load(self._real_path, mmap_mode='r')
        M, N = int(_mm.shape[1]), int(_mm.shape[2])
        del _mm

        self.M = M
        self.N = N
        self.d_lambda = d_lambda
        self.tau = tau

        self._mm_real = None
        self._mm_imag = None

        # 角度网格
        self.angle_grid = np.arange(
            angle_min, angle_max + baseline_angle_step * 0.5,
            baseline_angle_step, dtype=np.float32)
        self.num_grids = len(self.angle_grid)

        # 导向矩阵 (M, num_grids) complex
        grids_rad = np.deg2rad(self.angle_grid)
        n_idx = np.arange(M, dtype=np.float64)
        self.steering_matrix = np.exp(
            -1j * 2 * np.pi * d_lambda
            * n_idx[:, None] * np.sin(grids_rad[None, :])
        ).astype(np.complex64)

        print(f"[BaselineDataset] {split.upper()} 集: {len(self)} 条, "
              f"M={M}, N={N}, grid={self.num_grids} 点 "
              f"[{angle_min}°, {angle_max}°] step={baseline_angle_step}°")

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mm_real'] = None
        state['_mm_imag'] = None
        return state

    def _ensure_mmap(self):
        if self._mm_real is None:
            self._mm_real = np.load(self._real_path, mmap_mode='r')
            self._mm_imag = np.load(self._imag_path, mmap_mode='r')

    def __len__(self):
        return len(self._sel)

    def __getitem__(self, index):
        self._ensure_mmap()
        global_idx = self._sel[index]

        # 读取单个样本 
        signal = (self._mm_real[global_idx].astype(np.float32)
                  + 1j * self._mm_imag[global_idx].astype(np.float32))

        angles = self.angles[index]
        num_t = int(self.num_targets[index])
        snr = float(self.snr[index])

        cov = np.cov(signal)
        nrm = max(np.linalg.norm(cov), 1e-12)
        cov_n = cov / nrm
        cov3ch = np.stack(
            [cov_n.real, cov_n.imag, np.angle(cov_n)], axis=0
        ).astype(np.float32)

        # Autocorrelation
        Rx_tau = np.zeros((self.tau, 2 * self.M, self.M), dtype=np.float32)
        for lag in range(self.tau):
            valid = self.N - lag
            if valid <= 0:
                continue
            if lag == 0:
                Rx = signal @ signal.conj().T / self.N
            else:
                Rx = (signal[:, :self.N - lag]
                      @ signal[:, lag:self.N].conj().T / valid)
            Rx_tau[lag, :self.M, :] = Rx.real.astype(np.float32)
            Rx_tau[lag, self.M:, :] = Rx.imag.astype(np.float32)

        # 网格标签
        label = np.zeros(self.num_grids, dtype=np.float32)
        for i in range(num_t):
            idx = int(np.argmin(np.abs(self.angle_grid - angles[i])))
            label[idx] = 1.0

        # 角度 
        angles_rad = np.zeros(self.M - 1, dtype=np.float32)
        for i in range(min(num_t, self.M - 1)):
            angles_rad[i] = np.deg2rad(float(angles[i]))

        return {
            'signal': torch.from_numpy(signal.copy()),       # (M, N) complex64
            'cov3ch': torch.from_numpy(cov3ch),              # (3, M, M) float32
            'Rx_tau': torch.from_numpy(Rx_tau),              # (tau, 2M, M) float32
            'label_grid': torch.from_numpy(label),           # (num_grids,) float32
            'angles_rad': torch.from_numpy(angles_rad),      # (M-1,) float32
            'num_targets': num_t,
            'snr': snr,
        }
