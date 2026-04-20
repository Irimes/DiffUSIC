
import os
import hashlib
import shutil
import tempfile
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset


def _signal_to_covariance(signals_complex):
    s_num, _, snap = signals_complex.shape
    return np.einsum('smn,skn->smk', signals_complex,
                     signals_complex.conj()) / snap


def _autocorrelation_lag(signals_complex, lag):
    #计算单个样本的时滞自相关矩阵.
    M, N = signals_complex.shape
    if lag == 0:
        return signals_complex @ signals_complex.conj().T / N
    valid = N - abs(lag)
    if valid <= 0:
        return np.zeros((M, M), dtype=np.complex64)
    x1 = signals_complex[:, :valid]
    x2 = signals_complex[:, abs(lag):abs(lag) + valid]
    if lag > 0:
        return (x1 @ x2.conj().T) / valid
    else:
        return (x2 @ x1.conj().T) / valid


def _build_multi_lag_cov(signals_complex_batch, tau):
    #构建多时滞自相关张量.
    S, M, N = signals_complex_batch.shape
    out = np.empty((S, tau * 2, M, M), dtype=np.float32)
    for s in range(S):
        sig = signals_complex_batch[s]  # (M, N)
        for lag in range(tau):
            Rx = _autocorrelation_lag(sig, lag)
            out[s, lag * 2]     = Rx.real.astype(np.float32)
            out[s, lag * 2 + 1] = Rx.imag.astype(np.float32)
    return out


def _complex_to_2ch(cov_complex, dtype=np.float32):
    return np.stack([
        cov_complex.real.astype(dtype),
        cov_complex.imag.astype(dtype)
    ], axis=1)


def _build_angle_grid(angle_min=-90.0, angle_max=90.0, angle_step=1.0):
    grid = np.arange(angle_min, angle_max + angle_step, angle_step,
                     dtype=np.float32)
    return grid


def _steering_vector(theta_deg, m_num, d_lambda=0.5):
    theta_rad = np.deg2rad(theta_deg)
    idx = np.arange(m_num, dtype=np.float32)
    return np.exp(-1j * 2 * np.pi * d_lambda * idx * np.sin(theta_rad))


def _spatial_spectrum_from_cov(cov_complex, angle_grid, num_targets,
                               d_lambda=0.5):
    m_num = cov_complex.shape[0]
    k = int(num_targets)
    cov_h = 0.5 * (cov_complex + cov_complex.conj().T)

    eigvals, eigvecs = np.linalg.eigh(cov_h)
    noise_dim = m_num - k
    if noise_dim <= 0:
        noise_subspace = None
    else:
        noise_subspace = eigvecs[:, :noise_dim]

    spec = np.zeros(len(angle_grid), dtype=np.float32)
    for i, theta in enumerate(angle_grid):
        a = _steering_vector(theta, m_num, d_lambda).reshape(-1, 1)
        if noise_subspace is None:
            denom = np.real((a.conj().T @ cov_h @ a).item())
            spec[i] = max(float(denom), 1e-12)
            continue

        proj = noise_subspace @ noise_subspace.conj().T
        denom = np.real((a.conj().T @ proj @ a).item())
        spec[i] = 1.0 / max(float(denom), 1e-12)

    spec_db = 10.0 * np.log10(spec)
    spec_db = spec_db - np.max(spec_db)
    return spec_db


def _normalize_spec_db(spec_db, floor_db=-40.0):
    spec_clip = np.clip(spec_db, floor_db, 0.0)
    spec_01 = (spec_clip - floor_db) / (-floor_db)
    spec_m11 = spec_01 * 2.0 - 1.0
    return spec_m11.astype(np.float32)


def _gaussian_peak_spectrum(angles, num_targets, angle_grid, sigma=1.0):
    
    #根据真实 DOA 角度生成高斯峰标签谱。
    k = int(num_targets)
    spec = np.zeros(len(angle_grid), dtype=np.float64)
    for i in range(k):
        theta = float(angles[i])
        spec += np.exp(-0.5 * ((angle_grid - theta) / sigma) ** 2)
    # 裁剪到 [0, 1] 
    spec = np.clip(spec, 0.0, 1.0)
    # 映射到 [-1, 1]
    spec_m11 = spec * 2.0 - 1.0
    return spec_m11.astype(np.float32)


def _2ch_to_complex(cov_2ch):
    if isinstance(cov_2ch, torch.Tensor):
        cov_2ch = cov_2ch.cpu().numpy()
    return cov_2ch[:, 0, :, :] + 1j * cov_2ch[:, 1, :, :]


class DOASpectrumDataset(Dataset):
    def __init__(self, npz_path, split='train', test_ratio=0.1, seed=42,
                 angle_min=-90.0, angle_max=90.0, angle_step=1.0,
                 d_lambda=0.5, spec_floor_db=-40.0,
                 spec_label_type='music', gaussian_sigma=1.0,
                 tau=1):
        super().__init__()
        assert split in ('train', 'test')
        assert spec_label_type in ('music', 'gaussian'), \
            f"spec_label_type 必须是 'music' 或 'gaussian', 收到 '{spec_label_type}'"

        data = np.load(npz_path, allow_pickle=True)
        angles = data['angles'].astype(np.float32)
        snr = data['snr'].astype(np.float32)
        num_targets = data['num_targets'].astype(np.int32)

        # 读取 shape 信息，再 split 选子集后才构造复数数组，避免大数组 OOM
        total = len(angles)
        del data

        # 避免将完整信号数组加载到内存
        _npz_dir = os.path.dirname(npz_path)
        _tmpdir = tempfile.mkdtemp(prefix='doa_spec_', dir=_npz_dir)
        try:
            with zipfile.ZipFile(npz_path) as z:
                z.extract('received_signals_real.npy', _tmpdir)
            _mm = np.load(os.path.join(_tmpdir, 'received_signals_real.npy'),
                          mmap_mode='r')
            M = _mm.shape[1]
            N = _mm.shape[2]
            del _mm
        except Exception:
            data = np.load(npz_path, allow_pickle=True)
            M = data['received_signals_real'].shape[1]
            N = data['received_signals_real'].shape[2]
            del data
        finally:
            shutil.rmtree(_tmpdir, ignore_errors=True)
        self.M = M
        self.N = N
        self.d_lambda = d_lambda
        self.tau = max(1, int(tau))
        self.angle_grid = _build_angle_grid(angle_min, angle_max, angle_step)
        self.spec_floor_db = float(spec_floor_db)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(total)
        num_test = max(1, int(total * test_ratio))
        sel = idx[:num_test] if split == 'test' else idx[num_test:]

        self.angles = angles[sel]
        self.snr = snr[sel]
        self.num_targets = num_targets[sel]
        del angles, snr, num_targets

        _tmpdir2 = tempfile.mkdtemp(prefix='doa_sig_', dir=_npz_dir)
        n_sel = len(sel)
        CHUNK = 2000
        try:
            with zipfile.ZipFile(npz_path) as z:
                z.extract('received_signals_real.npy', _tmpdir2)
                z.extract('received_signals_imag.npy', _tmpdir2)
            _mm_r = np.load(os.path.join(_tmpdir2, 'received_signals_real.npy'),
                            mmap_mode='r')
            _mm_i = np.load(os.path.join(_tmpdir2, 'received_signals_imag.npy'),
                            mmap_mode='r')

            # 预分配协方差输出
            if self.tau > 1:
                # 多时滞自相关
                recv_multi = np.empty((n_sel, self.tau * 2, M, M), dtype=np.float32)
                for ci in range(0, n_sel, CHUNK):
                    cj = min(ci + CHUNK, n_sel)
                    chunk_idx = sel[ci:cj]
                    chunk_complex = (_mm_r[chunk_idx].astype(np.float32)
                                     + 1j * _mm_i[chunk_idx].astype(np.float32))
                    recv_multi[ci:cj] = _build_multi_lag_cov(
                        chunk_complex, self.tau)
                recv_cov_for_spec = None  
            else:
                recv_multi = None

            # 单协方差
            recv_cov = np.empty((n_sel, M, M), dtype=np.complex64)
            for ci in range(0, n_sel, CHUNK):
                cj = min(ci + CHUNK, n_sel)
                chunk_idx = sel[ci:cj]
                chunk_complex = (_mm_r[chunk_idx].astype(np.float32)
                                 + 1j * _mm_i[chunk_idx].astype(np.float32))
                chunk_cov = np.einsum('smn,skn->smk', chunk_complex,
                                      chunk_complex.conj()) / N
                recv_cov[ci:cj] = chunk_cov
            del _mm_r, _mm_i

            recv_cov = 0.5 * (recv_cov + recv_cov.conj().transpose(0, 2, 1))

            if spec_label_type == 'music':
                with zipfile.ZipFile(npz_path) as z:
                    z.extract('clean_signals_real.npy', _tmpdir2)
                    z.extract('clean_signals_imag.npy', _tmpdir2)
                _mm_cr = np.load(
                    os.path.join(_tmpdir2, 'clean_signals_real.npy'), mmap_mode='r')
                _mm_ci = np.load(
                    os.path.join(_tmpdir2, 'clean_signals_imag.npy'), mmap_mode='r')
                clean_cov = np.empty((n_sel, M, M), dtype=np.complex64)
                for ci in range(0, n_sel, CHUNK):
                    cj = min(ci + CHUNK, n_sel)
                    chunk_idx = sel[ci:cj]
                    cc = (_mm_cr[chunk_idx].astype(np.float32)
                          + 1j * _mm_ci[chunk_idx].astype(np.float32))
                    clean_cov[ci:cj] = np.einsum('smn,skn->smk', cc,
                                                  cc.conj()) / N
                del _mm_cr, _mm_ci
                clean_cov = 0.5 * (clean_cov + clean_cov.conj().transpose(0, 2, 1))
            else:
                clean_cov = None
        finally:
            shutil.rmtree(_tmpdir2, ignore_errors=True)

        del sel  

        if self.tau > 1 and recv_multi is not None:
            # 多时滞模式
            recv_max = np.max(np.abs(recv_multi), axis=(1, 2, 3), keepdims=True)
            recv_max = np.clip(recv_max, 1e-8, None)
            recv_2ch = recv_multi / recv_max
        else:
            recv_2ch = _complex_to_2ch(recv_cov)
            recv_max = np.max(np.abs(recv_2ch), axis=(1, 2, 3), keepdims=True)
            recv_max = np.clip(recv_max, 1e-8, None)
            recv_2ch = recv_2ch / recv_max

        self.spec_label_type = spec_label_type
        self.gaussian_sigma = float(gaussian_sigma)

        # 谱标签
        cache_key = hashlib.md5(
            f"{npz_path}|{split}|{test_ratio}|{seed}|"
            f"{angle_min}|{angle_max}|{angle_step}|"
            f"{d_lambda}|{spec_floor_db}|"
            f"{spec_label_type}|{gaussian_sigma}".encode()
        ).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(npz_path), '_spec_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'spec_{split}_{cache_key}.npy')

        if os.path.exists(cache_path):
            clean_spec = np.load(cache_path)
            print(f"[DOASpectrumDataset] 从缓存加载谱标签: {cache_path}")
        elif spec_label_type == 'gaussian':
            # 高斯峰标签
            num_samples = len(self.angles)
            spec_list = []
            print(f"[DOASpectrumDataset] 正在生成高斯峰标签 "
                  f"(σ={self.gaussian_sigma}°, {num_samples} 条)...")
            for i in range(num_samples):
                spec_norm = _gaussian_peak_spectrum(
                    self.angles[i],
                    self.num_targets[i],
                    self.angle_grid,
                    sigma=self.gaussian_sigma,
                )
                spec_list.append(spec_norm)
            clean_spec = np.stack(spec_list, axis=0)[:, None, :]
            np.save(cache_path, clean_spec)
            print(f"[DOASpectrumDataset] 高斯峰标签已缓存 -> {cache_path}")
        else:
            num_samples = clean_cov.shape[0]
            spec_list = []
            print_step = max(1, num_samples // 10)
            print(f"[DOASpectrumDataset] 正在计算 MUSIC 谱 ({num_samples} 条)...")
            for i in range(num_samples):
                spec_db = _spatial_spectrum_from_cov(
                    clean_cov[i],
                    self.angle_grid,
                    num_targets=self.num_targets[i],
                    d_lambda=self.d_lambda,
                )
                spec_norm = _normalize_spec_db(spec_db, floor_db=self.spec_floor_db)
                spec_list.append(spec_norm)
                if (i + 1) % print_step == 0:
                    print(f"  谱计算进度: {i + 1}/{num_samples}")
            clean_spec = np.stack(spec_list, axis=0)[:, None, :]
            np.save(cache_path, clean_spec)
            print(f"[DOASpectrumDataset] 谱标签已缓存 -> {cache_path}")

        self.received_cov = torch.from_numpy(recv_2ch)
        self.clean_spec = torch.from_numpy(clean_spec)

        self.norm_info = {
            'spec_floor_db': self.spec_floor_db,
            'angle_min': float(angle_min),
            'angle_max': float(angle_max),
            'angle_step': float(angle_step),
            'd_lambda': float(self.d_lambda),
            'spec_label_type': self.spec_label_type,
            'gaussian_sigma': self.gaussian_sigma,
        }

        label_desc = (f"高斯峰 σ={self.gaussian_sigma}°"
                      if self.spec_label_type == 'gaussian' else "MUSIC")
        cond_ch = self.tau * 2 if self.tau > 1 else 2
        print(f"[DOASpectrumDataset] {split.upper()} 集加载: "
              f"{len(self.clean_spec)} 条  标签类型: {label_desc}  tau={self.tau}")
        print(f"  条件输入: {tuple(self.received_cov.shape)} (S, {cond_ch}, M, M)")
        print(f"  谱目标  : {tuple(self.clean_spec.shape)} (S, 1, L)")

    def __len__(self):
        return len(self.clean_spec)

    def __getitem__(self, index):
        info = {
            'angles': self.angles[index],
            'snr': self.snr[index],
            'num_targets': self.num_targets[index],
        }
        return self.clean_spec[index], self.received_cov[index], info

    def denormalize_spec(self, spec_m11):
        if isinstance(spec_m11, torch.Tensor):
            spec_m11 = spec_m11.detach().cpu().numpy()
        spec_01 = (spec_m11 + 1.0) * 0.5
        spec_db = spec_01 * (-self.spec_floor_db) + self.spec_floor_db
        return spec_db

    def to_complex_cov(self, cov_2ch):
        return _2ch_to_complex(cov_2ch)


def find_latest_npz(directory='DOA_Dataset'):
    if not os.path.isdir(directory):
        return None
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    if not npz_files:
        return None
    npz_files.sort()
    return os.path.join(directory, npz_files[-1])