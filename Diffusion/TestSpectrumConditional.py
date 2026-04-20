
import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

from .DOASpectrumDataset import DOASpectrumDataset, find_latest_npz
from .DiffusionSpectrumConditional import ConditionalSpectrumDiffusionSampler
from .ModelSpectrumConditional import ConditionalSpectrumUNet1D

from .BaselineModels import (
    SubspaceNet, DeepSFNS, DeepSSE, IQResNet,
    DOALowSNRNet, DeepAugmentMusic,
    build_Rx_tau, signal_to_3ch_cov, grid_peaks_to_angles,
)
from .TrainBaselines import ALL_BASELINE_NAMES


def _music_doa_from_cov(cov_complex, num_targets, d_lambda=0.5,
                        angle_range=(-90, 90), angle_step=1.0):
    _, spec, angles_scan = _music_spectrum_from_cov(
        cov_complex,
        num_targets=num_targets,
        d_lambda=d_lambda,
        angle_range=angle_range,
        angle_step=angle_step,
    )
    return _find_k_peaks(spec, angles_scan, int(num_targets))


def _music_spectrum_from_cov(cov_complex, num_targets, d_lambda=0.5,
                             angle_range=(-90, 90), angle_step=1.0):
    k = int(num_targets)
    if k <= 0:
        angles_scan = np.arange(angle_range[0], angle_range[1] + angle_step,
                                angle_step)
        spec = np.zeros(len(angles_scan), dtype=np.float32)
        return [], spec, angles_scan

    m_num = cov_complex.shape[0]
    cov_h = 0.5 * (cov_complex + cov_complex.conj().T)
    _, eigvecs = np.linalg.eigh(cov_h)
    noise_subspace = eigvecs[:, :max(1, m_num - k)]

    angles_scan = np.arange(angle_range[0], angle_range[1] + angle_step,
                            angle_step)
    spec = np.zeros(len(angles_scan), dtype=np.float32)
    proj = noise_subspace @ noise_subspace.conj().T
    for i, theta in enumerate(angles_scan):
        theta_rad = np.deg2rad(theta)
        a = np.exp(-1j * 2 * np.pi * d_lambda * np.arange(m_num)
                   * np.sin(theta_rad)).reshape(-1, 1)
        denom = np.real((a.conj().T @ proj @ a).item())
        spec[i] = 1.0 / max(float(denom), 1e-12)
    est = _find_k_peaks(spec, angles_scan, k)
    return est, spec, angles_scan


def _esprit_doa_from_cov(cov_complex, num_targets, d_lambda=0.5,
                         angle_range=(-90, 90)):
    k = int(num_targets)
    if k <= 0:
        return []

    m_num = cov_complex.shape[0]
    if m_num < 2:
        return [0.0] * k

    cov_h = 0.5 * (cov_complex + cov_complex.conj().T)
    eigvals, eigvecs = np.linalg.eigh(cov_h)
    signal_subspace = eigvecs[:, -k:]

    us1 = signal_subspace[:-1, :]
    us2 = signal_subspace[1:, :]
    if us1.shape[0] < k:
        return [0.0] * k

    phi = np.linalg.pinv(us1) @ us2
    w, _ = np.linalg.eig(phi)

    phase = np.angle(w)
    sin_theta = -phase / (2.0 * np.pi * float(d_lambda))
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    est = np.rad2deg(np.arcsin(sin_theta))

    amin, amax = float(angle_range[0]), float(angle_range[1])
    est = np.clip(est, amin, amax)
    est_list = sorted([float(x) for x in est.tolist()])

    if len(est_list) < k:
        est_list += [0.0] * (k - len(est_list))
    return est_list[:k]


def _find_k_peaks(spec_vals, angle_grid, k):
    if k <= 0:
        return []
    n = len(spec_vals)
    if n == 0:
        return [0.0] * k

    peaks = []  # (value, angle)
   
    if n >= 2 and spec_vals[0] > spec_vals[1]:
        peaks.append((float(spec_vals[0]), float(angle_grid[0])))

    for i in range(1, n - 1):
        if spec_vals[i] > spec_vals[i - 1] and spec_vals[i] > spec_vals[i + 1]:
            peaks.append((float(spec_vals[i]), float(angle_grid[i])))

    if n >= 2 and spec_vals[n - 1] > spec_vals[n - 2]:
        peaks.append((float(spec_vals[n - 1]), float(angle_grid[n - 1])))

    if not peaks:
        idx = np.argsort(spec_vals)[::-1][:k]
        ret = [float(angle_grid[i]) for i in idx]
        ret.sort()
        return ret

    peaks.sort(key=lambda x: x[0], reverse=True)
    ret = [p[1] for p in peaks[:k]]

    if len(ret) < k:
        idx = np.argsort(spec_vals)[::-1]
        for i in idx:
            a = float(angle_grid[i])
            if a not in ret:
                ret.append(a)
            if len(ret) >= k:
                break
    ret = ret[:k]
    ret.sort()
    return ret


def _match_angles_min_mse(true_angles, est_angles, k):
    true_vals = [float(a) for a in true_angles[:k]]
    est_vals = [float(a) for a in est_angles]
    if len(true_vals) < k:
        true_vals += [0.0] * (k - len(true_vals))
    if len(est_vals) == 0:
        est_vals = [0.0]
    if len(est_vals) < k:
        est_vals += [0.0] * (k - len(est_vals))

    best_perm, best_mse = None, float('inf')
    if len(est_vals) == k:
        perms = itertools.permutations(est_vals, k)
        for p in perms:
            mse = float(np.mean([(t - e) ** 2 for t, e in zip(true_vals, p)]))
            if mse < best_mse:
                best_mse, best_perm = mse, p
    else:
        idx_range = range(len(est_vals))
        for comb in itertools.combinations(idx_range, k):
            selected = [est_vals[j] for j in comb]
            for p in itertools.permutations(selected, k):
                mse = float(np.mean([(t - e) ** 2 for t, e in zip(true_vals, p)]))
                if mse < best_mse:
                    best_mse, best_perm = mse, p

    if best_perm is None:
        best_perm = [0.0] * k
        best_mse = float(np.mean([(t - e) ** 2 for t, e in zip(true_vals, best_perm)]))
    return list(best_perm), best_mse


def _min_angle_gap(angles_list):
    if len(angles_list) < 2:
        return float('inf')
    s = sorted(angles_list)
    return min(s[i+1] - s[i] for i in range(len(s)-1))


def _print_summary(results, show_worst=5):
    all_mse = [r['mse'] for r in results]
    mse_arr = np.array(all_mse)
    rmse_deg = np.sqrt(np.mean(mse_arr))
    rmse_rad = np.deg2rad(rmse_deg)

    print(f'  总样本数: {len(results)}')
    print(f'  DOA RMSE: {rmse_deg:.4f} deg  ({rmse_rad:.4f} rad)')
    print(f'  MSE 分布: median={np.median(mse_arr):.4f}  '
          f'p90={np.percentile(mse_arr, 90):.2f}  '
          f'p99={np.percentile(mse_arr, 99):.2f}  '
          f'max={np.max(mse_arr):.2f}')

    k_set = sorted(set(int(r['K']) for r in results))
    if len(k_set) > 1:
        print('  按目标数 K:')
        for k in k_set:
            sub = [r for r in results if int(r['K']) == k]
            sub_mse = np.array([x['mse'] for x in sub])
            rmse_k = np.sqrt(np.mean(sub_mse))
            print(f'    K={k}  RMSE={rmse_k:.4f} deg ({np.deg2rad(rmse_k):.4f} rad)  '
                  f'median_MSE={np.median(sub_mse):.4f}  n={len(sub)}')

    snr_set = sorted(set(float(r['snr']) for r in results))
    print('  按 SNR:')
    for snr in snr_set:
        sub = [r for r in results if float(r['snr']) == snr]
        rmse = np.sqrt(np.mean([x['mse'] for x in sub]))
        print(f'    SNR={snr:+6.1f} dB  RMSE={rmse:.4f} deg ({np.deg2rad(rmse):.4f} rad)  n={len(sub)}')

    has_angles = any('true_angles' in r for r in results)
    if has_angles:
        close_sub = [r for r in results
                     if 'true_angles' in r and _min_angle_gap(r['true_angles']) < 5.0]
        far_sub = [r for r in results
                   if 'true_angles' in r and _min_angle_gap(r['true_angles']) >= 5.0]
        if close_sub and far_sub:
            print(f'  按最小角度间隔:')
            rmse_close = np.sqrt(np.mean([x['mse'] for x in close_sub]))
            rmse_far = np.sqrt(np.mean([x['mse'] for x in far_sub]))
            print(f'    间隔 < 5°  RMSE={rmse_close:.4f} deg  n={len(close_sub)}')
            print(f'    间隔 ≥ 5°  RMSE={rmse_far:.4f} deg  n={len(far_sub)}')

    if show_worst > 0:
        sorted_r = sorted(results, key=lambda x: x['mse'], reverse=True)
        print(f'  误差最大的 {min(show_worst, len(sorted_r))} 个样本:')
        for r in sorted_r[:show_worst]:
            ta = r.get('true_angles', [])
            ea = r.get('est_angles', ta)
            gap = _min_angle_gap(ta) if len(ta) >= 2 else float('inf')
            gap_str = f'{gap:.1f}°' if gap < 1e6 else 'N/A'
            print(f'    SNR={r["snr"]:+6.1f} K={r["K"]} '
                  f'真值={[round(a,1) for a in ta]} '
                  f'估计={[round(a,1) for a in ea]} '
                  f'MSE={r["mse"]:.2f} 最小间隔={gap_str}')


def _plot_worst_samples(worst_items, angle_grid, out_dir, tag='worst'):
    worst_items: list of dict, 每个包含:
        pred_spec, true_spec, true_angles, est_angles, mse, snr, K
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(worst_items)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             squeeze=False)

    for idx, item in enumerate(worst_items):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.plot(angle_grid, item['true_spec'], 'b-', linewidth=1.5,
                label='True (clean)', alpha=0.8)
        ax.plot(angle_grid, item['pred_spec'], 'r--', linewidth=1.5,
                label='Predicted', alpha=0.8)

        for a in item['true_angles']:
            ax.axvline(a, color='blue', linestyle=':', alpha=0.5, linewidth=0.8)

        for a in item['est_angles']:
            ax.axvline(a, color='red', linestyle=':', alpha=0.5, linewidth=0.8)

        ax.set_title(
            f"SNR={item['snr']:+.0f}dB  K={item['K']}  "
            f"RMSE={np.sqrt(item['mse']):.1f}°\n"
            f"真值={[round(a,1) for a in item['true_angles']]}\n"
            f"估计={[round(a,1) for a in item['est_angles']]}",
            fontsize=8)
        ax.set_xlabel('Angle (°)', fontsize=7)
        ax.set_ylabel('Spectrum', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f'Worst {n} Samples — Predicted vs True Spectrum', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(out_dir, f'{tag}_spectrum_compare.png')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'[Plot] 误差最大样本谱对比图已保存: {save_path}')


def _select_balanced_indices(test_ds, samples_per_snr, seed):
    all_snr_vals = np.array([float(test_ds[i][2]['snr']) for i in range(len(test_ds))])
    unique_snrs = np.unique(all_snr_vals)
    rng = np.random.RandomState(seed)
    selected_indices = []
    for snr in sorted(unique_snrs):
        idxs = np.where(all_snr_vals == snr)[0]
        if len(idxs) <= samples_per_snr:
            selected_indices.extend(idxs.tolist())
        else:
            selected_indices.extend(
                rng.choice(idxs, size=samples_per_snr, replace=False).tolist())
    selected_indices.sort()
    return selected_indices


def _method_rmse_by_snr(method_results):
    snr_vals = sorted(set(float(r['snr']) for r in method_results))
    out = []
    for snr in snr_vals:
        sub = [r for r in method_results if float(r['snr']) == snr]
        rmse = float(np.sqrt(np.mean([x['mse'] for x in sub])))
        out.append((snr, rmse))
    return out


def _plot_compare_figure(model_cfg, per_method):
    out_dir = model_cfg.get('compare_plot_dir', './ComparePlots')
    os.makedirs(out_dir, exist_ok=True)

    _COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
               'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
               'tab:olive', 'tab:cyan']
    _MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X', '*', 'h', '<']

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, method_name in enumerate(per_method):
        pairs = _method_rmse_by_snr(per_method[method_name])
        if pairs:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            ax.plot(xs, ys,
                    marker=_MARKERS[idx % len(_MARKERS)],
                    linewidth=2.0,
                    label=method_name,
                    color=_COLORS[idx % len(_COLORS)])
    ax.set_title('RMSE vs SNR')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE (deg)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    save_path = os.path.join(
        out_dir,
        f"compare_rmse_snr_{model_cfg.get('test_load_weight', 'model')}.png"
    )
    fig.savefig(save_path, dpi=180)
    if model_cfg.get('compare_show_plot', False):
        plt.show()
    else:
        plt.close(fig)
    print(f'[Compare] 图像已保存: {save_path}')


#  Baseline 辅助函数

def _load_test_raw_signals(npz_path, test_ratio, seed):
    import tempfile, shutil, zipfile

    data = np.load(npz_path, allow_pickle=True)
    total = len(data['angles'])
    del data

    rng = np.random.RandomState(seed)
    idx = rng.permutation(total)
    num_test = max(1, int(total * test_ratio))
    sel = idx[:num_test]

    tmpdir = tempfile.mkdtemp(prefix='cmp_sig_', dir=os.path.dirname(npz_path))
    try:
        with zipfile.ZipFile(npz_path) as z:
            z.extract('received_signals_real.npy', tmpdir)
        mm_r = np.load(os.path.join(tmpdir, 'received_signals_real.npy'), mmap_mode='r')
        sig_r = mm_r[sel].astype(np.float32)
        del mm_r

        with zipfile.ZipFile(npz_path) as z:
            z.extract('received_signals_imag.npy', tmpdir)
        mm_i = np.load(os.path.join(tmpdir, 'received_signals_imag.npy'), mmap_mode='r')
        sig_i = mm_i[sel].astype(np.float32)
        del mm_i
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return sig_r + 1j * sig_i


def _build_baseline_grid(cfg):
    #返回角度网格和导向矩阵。
    angle_min = cfg.get('angle_min', -60.0)
    angle_max = cfg.get('angle_max', 60.0)
    bl_step = cfg.get('baseline_angle_step', 1.0)
    d_lambda = cfg.get('d_lambda', 0.5)
    M = cfg.get('M', 8)

    grid = np.arange(angle_min, angle_max + bl_step * 0.5,
                     bl_step, dtype=np.float32)
    grids_rad = np.deg2rad(grid)
    n_idx = np.arange(M, dtype=np.float64)
    A = np.exp(-1j * 2 * np.pi * d_lambda
               * n_idx[:, None] * np.sin(grids_rad[None, :])).astype(np.complex64)
    return grid, A


def _load_baseline_models(cfg, device):
    base_dir = cfg.get('baseline_save_dir', './BaselineCheckpoints')
    bl_step = cfg.get('baseline_angle_step', 1.0)
    angle_min = cfg.get('angle_min', -60.0)
    angle_max = cfg.get('angle_max', 60.0)
    d_lambda = cfg.get('d_lambda', 0.5)
    M = cfg.get('M', 8)
    N = 100  # 快照数, 与数据集一致
    tau = cfg.get('baseline_tau', 8)

    bl_grid, A_np = _build_baseline_grid(cfg)
    num_grids = len(bl_grid)

    loaded = {}
    for name in ALL_BASELINE_NAMES:
        ckpt = os.path.join(base_dir, name, 'best.pt')
        if not os.path.exists(ckpt):
            print(f'[Compare] {name} checkpoint 未找到, 跳过: {ckpt}')
            continue

        try:
            if name == 'SubspaceNet':
                model = SubspaceNet(tau=tau, diff_method='esprit')
            elif name == 'DeepSFNS':
                model = DeepSFNS(M=M, num_grids=num_grids, snapshots=N)
            elif name == 'DeepSSE':
                model = DeepSSE(
                    steering_vector_np=A_np,
                    num_class=num_grids, num_antenna=M)
            elif name == 'IQResNet':
                model = IQResNet(num_classes=num_grids, num_antennas=M)
            elif name == 'DOALowSNRNet':
                model = DOALowSNRNet(num_out_grids=num_grids)
            elif name == 'daMUSIC':
                model = DeepAugmentMusic(
                    num_antennas=M, d_lambda=d_lambda,
                    angle_min=angle_min, angle_max=angle_max,
                    angle_step=bl_step)
            else:
                continue

            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device).eval()
            loaded[name] = model
            print(f'[Compare] {name} 加载成功: {ckpt}')
        except Exception as e:
            print(f'[Compare] {name} 加载失败: {e}')

    return loaded


def _baseline_inference_single(name, model, signal_np, k,
                               bl_grid, A_np, cfg, device):
    M = signal_np.shape[0]
    angle_min = cfg.get('angle_min', -60.0)
    angle_max = cfg.get('angle_max', 60.0)
    bl_step = cfg.get('baseline_angle_step', 1.0)
    num_grids = len(bl_grid)

    if name == 'SubspaceNet':
        Rx_tau = build_Rx_tau(signal_np, tau=cfg.get('baseline_tau', 8))
        Rx_t = torch.from_numpy(Rx_tau).unsqueeze(0).to(device)
        lbl = torch.zeros(1, num_grids, device=device)
        lbl[0, :k] = 1.0 
        doa_list, _ = model(Rx_t, lbl)
        est_rad = doa_list[0].detach().cpu().numpy()
        if np.iscomplexobj(est_rad):
            est_rad = est_rad.real
        est_deg = np.rad2deg(est_rad).tolist()
        est_deg = [float(np.clip(a, angle_min, angle_max)) for a in est_deg]
        return sorted(est_deg[:k])

    elif name == 'DeepSFNS':
        A_t = torch.from_numpy(A_np).unsqueeze(0).to(device)
        Y_t = torch.from_numpy(signal_np.copy()).unsqueeze(0).to(device)
        prob = model(A_t, Y_t).detach().cpu().numpy()[0]
        return grid_peaks_to_angles(prob, k, angle_min, angle_max, bl_step)

    elif name == 'DeepSSE':
        cov3ch = signal_to_3ch_cov(signal_np)
        inp = torch.from_numpy(cov3ch).unsqueeze(0).to(device)
        prob = model(inp).detach().cpu().numpy()[0]
        return grid_peaks_to_angles(prob, k, angle_min, angle_max, bl_step)

    elif name == 'IQResNet':
        inp = torch.from_numpy(signal_np.copy()).unsqueeze(0).to(device)
        prob = model(inp).detach().cpu().numpy()[0]
        return grid_peaks_to_angles(prob, k, angle_min, angle_max, bl_step)

    elif name == 'DOALowSNRNet':
        cov3ch = signal_to_3ch_cov(signal_np)
        inp = torch.from_numpy(cov3ch).unsqueeze(0).to(device)
        prob = model(inp).detach().cpu().numpy()[0]
        return grid_peaks_to_angles(prob, k, angle_min, angle_max, bl_step)

    elif name == 'daMUSIC':
        inp = torch.from_numpy(signal_np.copy()).unsqueeze(0).to(device)
        out = model(inp).detach().cpu().numpy()[0]  # (M-1,) radians
        est_deg = np.rad2deg(out).tolist()
        est_deg = [float(np.clip(a, angle_min, angle_max)) for a in est_deg]
        return sorted(est_deg[:k])

    return [0.0] * k


@torch.no_grad()
def test_compare_spectrum_music_esprit(model_config: dict):
    device = torch.device(model_config['device'])

    meta_path = os.path.join(model_config['save_weight_dir'], 'train_meta.npy')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'未找到训练元数据: {meta_path}')
    meta = np.load(meta_path, allow_pickle=True).item()
    spec_len = int(meta['spec_len'])

    npz_path = model_config.get('npz_path') or find_latest_npz()
    test_ds = DOASpectrumDataset(
        npz_path=npz_path,
        split='test',
        test_ratio=model_config.get('test_ratio', 0.1),
        seed=model_config.get('split_seed', 42),
        angle_min=model_config.get('angle_min', -90.0),
        angle_max=model_config.get('angle_max', 90.0),
        angle_step=model_config.get('angle_step', 1.0),
        d_lambda=model_config.get('d_lambda', 0.5),
        spec_floor_db=model_config.get('spec_floor_db', -40.0),
        spec_label_type=model_config.get('spec_label_type', 'music'),
        gaussian_sigma=model_config.get('gaussian_sigma', 1.0),
    )

    net = ConditionalSpectrumUNet1D(
        T=model_config['T'],
        spec_len=spec_len,
        M=model_config.get('M', test_ds.M),
        base_ch=model_config.get('spec_base_ch', 128),
        num_res_blocks=model_config.get('spec_res_blocks', 6),
        dropout=0.0,
    ).to(device)

    ckpt_path = os.path.join(model_config['save_weight_dir'], model_config['test_load_weight'])
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    print(f'[Compare] 扩散模型加载: {ckpt_path}')

    sampler = ConditionalSpectrumDiffusionSampler(
        net,
        model_config['beta_1'],
        model_config['beta_T'],
        model_config['T'],
    ).to(device)

    # 加载 baseline
    baseline_models = _load_baseline_models(model_config, device)
    bl_grid, A_np = _build_baseline_grid(model_config)

    # 加载测试集信号
    raw_signals = None
    if baseline_models:
        raw_signals = _load_test_raw_signals(
            npz_path,
            test_ratio=model_config.get('test_ratio', 0.1),
            seed=model_config.get('split_seed', 42))
        print(f'[Compare] 原始信号加载: {raw_signals.shape}')

    samples_per_snr = model_config.get('test_samples_per_snr', 50)
    shared_seed = model_config.get('compare_random_seed', model_config.get('split_seed', 42))
    selected_indices = _select_balanced_indices(
        test_ds,
        samples_per_snr=samples_per_snr,
        seed=shared_seed,
    )
    print(f'[Compare] 每个方法使用完全一致样本: {len(selected_indices)} 条')

    angle_grid = test_ds.angle_grid
    angle_range = (model_config.get('angle_min', -90.0),
                   model_config.get('angle_max', 90.0))
    angle_step = model_config.get('angle_step', 1.0)
    d_lambda = model_config.get('d_lambda', 0.5)

    per_method = {
        'Diffusion': [],
        'MUSIC': [],
        'ESPRIT': [],
    }
    for bname in baseline_models:
        per_method[bname] = []

    for count, i in enumerate(selected_indices):
        _, recv_cov, info = test_ds[i]
        k = int(info['num_targets'])
        snr_val = float(info['snr'])
        true_angles = info['angles']
        if isinstance(true_angles, np.ndarray):
            true_angles = true_angles.tolist()
        true_angles = sorted([float(a) for a in true_angles[:k]])

        recv_2ch = recv_cov.unsqueeze(0).cpu().numpy()
        recv_complex = recv_2ch[0, 0] + 1j * recv_2ch[0, 1]

        # Diffusion
        cond = recv_cov.unsqueeze(0).to(device)
        x_t = torch.randn(1, 1, spec_len, device=device)
        pred_spec = sampler(x_t, cond).cpu().numpy()[0, 0]
        diff_est = _find_k_peaks(pred_spec, angle_grid, k)
        diff_match, diff_mse = _match_angles_min_mse(true_angles, diff_est, k)

        # MUSIC
        music_est, _, _ = _music_spectrum_from_cov(
            recv_complex, num_targets=k, d_lambda=d_lambda,
            angle_range=angle_range, angle_step=angle_step)
        music_match, music_mse = _match_angles_min_mse(true_angles, music_est, k)

        # ESPRIT
        esprit_est = _esprit_doa_from_cov(
            recv_complex, num_targets=k, d_lambda=d_lambda,
            angle_range=angle_range)
        esprit_match, esprit_mse = _match_angles_min_mse(true_angles, esprit_est, k)

        rec = lambda match, mse: {
            'idx': int(i), 'snr': snr_val, 'K': k,
            'true_angles': true_angles, 'est_angles': list(match),
            'mse': float(mse),
        }
        per_method['Diffusion'].append(rec(diff_match, diff_mse))
        per_method['MUSIC'].append(rec(music_match, music_mse))
        per_method['ESPRIT'].append(rec(esprit_match, esprit_mse))

        # Baseline
        bl_info_parts = []
        for bname, bmodel in baseline_models.items():
            try:
                signal_np = raw_signals[i]
                bl_est = _baseline_inference_single(
                    bname, bmodel, signal_np, k,
                    bl_grid, A_np, model_config, device)
                bl_match, bl_mse = _match_angles_min_mse(true_angles, bl_est, k)
            except Exception as e:
                import traceback
                print(f"    [ERROR] {bname} 推理失败: {e}")
                traceback.print_exc()
                bl_match = [0.0] * k
                bl_mse = 9999.0
            per_method[bname].append(rec(bl_match, bl_mse))
            bl_info_parts.append(f"{bname}={bl_mse:.3f}")

        bl_str = ' '.join(bl_info_parts)
        print(f"  [{count + 1}/{len(selected_indices)}] SNR={snr_val:+6.1f}dB K={k} "
              f"Diff={diff_mse:.3f} MUSIC={music_mse:.3f} ESPRIT={esprit_mse:.3f} "
              f"{bl_str}")

    print('\n' + '=' * 64)
    print('同样本对比结果')
    print('=' * 64)
    for name in per_method:
        if per_method[name]:
            print(f'\n[{name}]')
            _print_summary(per_method[name], show_worst=3)

    _plot_compare_figure(model_config, per_method)


@torch.no_grad()
def test_spectrum_conditional(model_config: dict):
    device = torch.device(model_config['device'])

    meta_path = os.path.join(model_config['save_weight_dir'], 'train_meta.npy')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'未找到训练元数据: {meta_path}')
    meta = np.load(meta_path, allow_pickle=True).item()
    spec_len = int(meta['spec_len'])

    npz_path = model_config.get('npz_path') or find_latest_npz()
    test_ds = DOASpectrumDataset(
        npz_path=npz_path,
        split='test',
        test_ratio=model_config.get('test_ratio', 0.1),
        seed=model_config.get('split_seed', 42),
        angle_min=model_config.get('angle_min', -90.0),
        angle_max=model_config.get('angle_max', 90.0),
        angle_step=model_config.get('angle_step', 1.0),
        d_lambda=model_config.get('d_lambda', 0.5),
        spec_floor_db=model_config.get('spec_floor_db', -40.0),
        spec_label_type=model_config.get('spec_label_type', 'music'),
        gaussian_sigma=model_config.get('gaussian_sigma', 1.0),
    )

    net = ConditionalSpectrumUNet1D(
        T=model_config['T'],
        spec_len=spec_len,
        M=model_config.get('M', test_ds.M),
        base_ch=model_config.get('spec_base_ch', 128),
        num_res_blocks=model_config.get('spec_res_blocks', 6),
        dropout=0.0,
    ).to(device)

    ckpt_path = os.path.join(model_config['save_weight_dir'], model_config['test_load_weight'])
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    print(f'[Test-Spec] 模型加载: {ckpt_path}')

    sampler = ConditionalSpectrumDiffusionSampler(
        net,
        model_config['beta_1'],
        model_config['beta_T'],
        model_config['T'],
    ).to(device)

    samples_per_snr = model_config.get('test_samples_per_snr', 50)
    all_snr_vals = np.array([float(test_ds[i][2]['snr']) for i in range(len(test_ds))])
    unique_snrs = np.unique(all_snr_vals)
    rng = np.random.RandomState(model_config.get('split_seed', 42))
    selected_indices = []
    for snr in sorted(unique_snrs):
        idxs = np.where(all_snr_vals == snr)[0]
        if len(idxs) <= samples_per_snr:
            selected_indices.extend(idxs.tolist())
        else:
            selected_indices.extend(rng.choice(idxs, size=samples_per_snr, replace=False).tolist())
    selected_indices.sort()

    angle_grid = test_ds.angle_grid
    results = []
    baseline_results = []
    direct_music_results = []

    # 收集误差最大样本
    worst_heap = []  # (mse, dict) — 保留 top-N
    import heapq
    worst_n = model_config.get('plot_worst_n', 16)

    for count, i in enumerate(selected_indices):
        clean_spec, recv_cov, info = test_ds[i]
        k = int(info['num_targets'])
        snr_val = float(info['snr'])
        true_angles = info['angles']
        if isinstance(true_angles, np.ndarray):
            true_angles = true_angles.tolist()
        true_angles = sorted([float(a) for a in true_angles[:k]])

        cond = recv_cov.unsqueeze(0).to(device)
        x_t = torch.randn(1, 1, spec_len, device=device)
        pred_spec = sampler(x_t, cond).cpu().numpy()[0, 0]

        est_angles = _find_k_peaks(pred_spec, angle_grid, k)
        matched, mse = _match_angles_min_mse(true_angles, est_angles, k)
        results.append({
            'snr': snr_val,
            'K': k,
            'true_angles': true_angles,
            'est_angles': matched,
            'mse': mse,
        })

        # 跟踪误差最大的样本
        spec_item = {
            'pred_spec': pred_spec.copy(),
            'true_spec': clean_spec.numpy()[0].copy(),
            'true_angles': true_angles,
            'est_angles': list(matched),
            'mse': mse, 'snr': snr_val, 'K': k,
        }
        if len(worst_heap) < worst_n:
            heapq.heappush(worst_heap, (mse, count, spec_item))
        elif mse > worst_heap[0][0]:
            heapq.heapreplace(worst_heap, (mse, count, spec_item))

        base_spec = clean_spec.numpy()[0]
        base_est = _find_k_peaks(base_spec, angle_grid, k)
        base_matched, base_mse = _match_angles_min_mse(true_angles, base_est, k)
        baseline_results.append({
            'snr': snr_val,
            'K': k,
            'true_angles': true_angles,
            'est_angles': list(base_matched),
            'mse': base_mse,
        })

        recv_2ch = recv_cov.unsqueeze(0).cpu().numpy()
        recv_complex = recv_2ch[0, 0] + 1j * recv_2ch[0, 1]
        direct_music_est = _music_doa_from_cov(
            recv_complex,
            num_targets=k,
            d_lambda=model_config.get('d_lambda', 0.5),
            angle_range=(model_config.get('angle_min', -90.0),
                         model_config.get('angle_max', 90.0)),
            angle_step=model_config.get('angle_step', 1.0),
        )
        music_matched, direct_music_mse = _match_angles_min_mse(true_angles, direct_music_est, k)
        direct_music_results.append({
            'snr': snr_val,
            'K': k,
            'true_angles': true_angles,
            'est_angles': list(music_matched),
            'mse': direct_music_mse,
        })

        if True: 
            gap = _min_angle_gap(true_angles) if len(true_angles) >= 2 else float('inf')
            gap_str = f'{gap:.0f}°' if gap < 1e6 else '-'
            print(f"  [{count + 1}/{len(selected_indices)}] SNR={snr_val:+6.1f}dB K={k} "
                  f"MinGap={gap_str} "
                  f"真值={true_angles} 估计={[round(a, 1) for a in matched]} "
                  f"MSE={mse:.3f}  MUSIC_MSE={direct_music_mse:.3f}")

    print('\n' + '=' * 60)
    print('谱域条件扩散结果 (全部样本)')
    print('=' * 60)
    _print_summary(results)

    print('\n' + '=' * 60)
    print('参考上界: 干净谱直接峰值检测 (全部样本)')
    print('=' * 60)
    _print_summary(baseline_results)

    print('\n' + '=' * 60)
    print('基线对比: 不去噪直接对含噪协方差做 MUSIC (全部样本)')
    print('=' * 60)
    _print_summary(direct_music_results)

    outlier_mse_threshold = 100.0   # MSE > 此阈值视为异常
    n_total = len(results)
    filtered_results = []
    filtered_baseline = []
    filtered_music = []
    removed_count = 0

    for i in range(n_total):
        diffusion_mse = results[i]['mse']
        music_mse = direct_music_results[i]['mse']
        if diffusion_mse > outlier_mse_threshold or music_mse > outlier_mse_threshold:
            removed_count += 1
            continue
        filtered_results.append(results[i])
        filtered_baseline.append(baseline_results[i])
        filtered_music.append(direct_music_results[i])

    print('\n' + '=' * 60)
    print(f'过滤后结果 (剔除 MSE>{outlier_mse_threshold} 的异常样本: '
          f'{removed_count}/{n_total} 个, 剩余 {len(filtered_results)} 个)')
    print('=' * 60)

    print('\n谱域条件扩散 (过滤后)')
    _print_summary(filtered_results)

    print('\n参考上界: 干净谱直接峰值检测 (过滤后)')
    _print_summary(filtered_baseline)

    print('\n基线对比: 含噪 MUSIC (过滤后)')
    _print_summary(filtered_music)

    worst_sorted = sorted(worst_heap, key=lambda x: x[0], reverse=True)
    worst_items = [item for (_, _, item) in worst_sorted]
    if worst_items:
        plot_dir = os.path.join(
            model_config.get('compare_plot_dir', './ComparePlots'),
            'worst_samples')
        _plot_worst_samples(worst_items, angle_grid, plot_dir,
                            tag=f"worst_{model_config.get('test_load_weight', 'model')}")
