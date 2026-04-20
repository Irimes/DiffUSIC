"""
谱域条件扩散 V2 测试
======================
基于 V2 模型, 支持 SNR 条件和 Classifier-Free Guidance.
"""

import os
import itertools
import heapq
import numpy as np
import torch
import matplotlib.pyplot as plt

from .DOASpectrumDataset import DOASpectrumDataset, find_latest_npz
from .DiffusionSpectrumConditionalV2 import ConditionalSpectrumDiffusionSamplerV2
from .ModelSpectrumConditionalV2 import ConditionalSpectrumUNet1D_V2

# 复用 V1 里的辅助函数
from .TestSpectrumConditional import (
    _music_doa_from_cov, _music_spectrum_from_cov,
    _esprit_doa_from_cov, _find_k_peaks,
    _match_angles_min_mse, _min_angle_gap,
    _print_summary, _plot_worst_samples,
    _select_balanced_indices, _method_rmse_by_snr,
    _plot_compare_figure,
    _load_baseline_models, _build_baseline_grid,
    _load_test_raw_signals, _baseline_inference_single,
)


def _normalize_snr(snr_val, snr_min=-20.0, snr_max=20.0):
    return 2.0 * (snr_val - snr_min) / max(snr_max - snr_min, 1e-8) - 1.0


def _normalize_k(k_val, k_min=1, k_max=3):
    return 2.0 * (k_val - k_min) / max(k_max - k_min, 1e-8) - 1.0


def _build_v2_sampler(model_config, spec_len, test_ds, device):
    """构建 V2 模型 + 采样器."""
    tau = model_config.get('tau', 1)
    net = ConditionalSpectrumUNet1D_V2(
        T=model_config['T'],
        spec_len=spec_len,
        M=model_config.get('M', test_ds.M),
        base_ch=model_config.get('spec_base_ch', 128),
        num_res_blocks=model_config.get('spec_res_blocks', 2),
        dropout=0.0,
        use_snr_cond=model_config.get('use_snr_cond', True),
        use_k_cond=model_config.get('use_k_cond', False),
        cfg_drop_prob=0.0,  # 测试时不 drop
        tau=tau,
        use_anti_rectifier=model_config.get('use_anti_rectifier', True),
    ).to(device)

    ckpt_path = os.path.join(
        model_config['save_weight_dir'],
        model_config['test_load_weight'])
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    print(f'[Test-V2] 模型加载: {ckpt_path}')

    cfg_scale = model_config.get('cfg_scale', 2.0)
    sampler = ConditionalSpectrumDiffusionSamplerV2(
        net,
        model_config['beta_1'],
        model_config['beta_T'],
        model_config['T'],
        cfg_scale=cfg_scale,
    ).to(device)
    print(f'[Test-V2] CFG scale: {cfg_scale}')
    return sampler


@torch.no_grad()
def test_compare_v2(model_config: dict):
    """V2 对比测试: Diffusion-V2 vs MUSIC vs ESPRIT vs Baselines."""
    device = torch.device(model_config['device'])

    meta_path = os.path.join(model_config['save_weight_dir'], 'train_meta.npy')
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
        spec_label_type=model_config.get('spec_label_type', 'gaussian'),
        gaussian_sigma=model_config.get('gaussian_sigma', 1.0),
        tau=model_config.get('tau', 1),
    )

    sampler = _build_v2_sampler(model_config, spec_len, test_ds, device)

    # Baselines
    baseline_models = _load_baseline_models(model_config, device)
    bl_grid, A_np = _build_baseline_grid(model_config)
    raw_signals = None
    if baseline_models:
        raw_signals = _load_test_raw_signals(
            npz_path,
            test_ratio=model_config.get('test_ratio', 0.1),
            seed=model_config.get('split_seed', 42))

    samples_per_snr = model_config.get('test_samples_per_snr', 50)
    shared_seed = model_config.get('compare_random_seed',
                                   model_config.get('split_seed', 42))
    selected_indices = _select_balanced_indices(
        test_ds, samples_per_snr=samples_per_snr, seed=shared_seed)
    print(f'[Compare-V2] 样本数: {len(selected_indices)}')

    angle_grid = test_ds.angle_grid
    angle_range = (model_config.get('angle_min', -90.0),
                   model_config.get('angle_max', 90.0))
    angle_step = model_config.get('angle_step', 1.0)
    d_lambda = model_config.get('d_lambda', 0.5)
    use_snr_cond = model_config.get('use_snr_cond', True)
    use_k_cond = model_config.get('use_k_cond', False)
    snr_min = model_config.get('snr_range_min', -20.0)
    snr_max = model_config.get('snr_range_max', 20.0)
    k_min = float(model_config.get('k_range_min', 1))
    k_max = float(model_config.get('k_range_max', 3))

    per_method = {'Diffusion-V2': [], 'MUSIC': [], 'ESPRIT': []}
    for bname in baseline_models:
        per_method[bname] = []

    # 收集谱对比数据 (用于随机抽样画图)
    all_spec_items = []

    for count, i in enumerate(selected_indices):
        clean_spec, recv_cov, info = test_ds[i]
        k = int(info['num_targets'])
        snr_val = float(info['snr'])
        true_angles = info['angles']
        if isinstance(true_angles, np.ndarray):
            true_angles = true_angles.tolist()
        true_angles = sorted([float(a) for a in true_angles[:k]])

        recv_2ch = recv_cov.unsqueeze(0).cpu().numpy()
        recv_complex = recv_2ch[0, 0] + 1j * recv_2ch[0, 1]

        # -- Diffusion V2 --
        cond = recv_cov.unsqueeze(0).to(device)
        snr_norm = torch.tensor(
            [[_normalize_snr(snr_val, snr_min, snr_max)]],
            device=device, dtype=torch.float32)
        k_norm_t = None
        if use_k_cond:
            k_norm_t = torch.tensor(
                [[_normalize_k(float(k), k_min, k_max)]],
                device=device, dtype=torch.float32)
        x_t = torch.randn(1, 1, spec_len, device=device)
        pred_spec = sampler(
            x_t, cond,
            snr_norm=snr_norm if use_snr_cond else None,
            k_norm=k_norm_t,
        ).cpu().numpy()[0, 0]
        diff_est = _find_k_peaks(pred_spec, angle_grid, k)
        diff_match, diff_mse = _match_angles_min_mse(true_angles, diff_est, k)

        # 收集谱对比数据
        all_spec_items.append({
            'pred_spec': pred_spec.copy(),
            'true_spec': clean_spec.numpy()[0].copy(),
            'true_angles': true_angles,
            'est_angles': list(diff_match),
            'mse': float(diff_mse),
            'snr': snr_val,
            'K': k,
        })

        # -- MUSIC --
        music_est, _, _ = _music_spectrum_from_cov(
            recv_complex, num_targets=k, d_lambda=d_lambda,
            angle_range=angle_range, angle_step=angle_step)
        music_match, music_mse = _match_angles_min_mse(true_angles, music_est, k)

        # -- ESPRIT --
        esprit_est = _esprit_doa_from_cov(
            recv_complex, num_targets=k, d_lambda=d_lambda,
            angle_range=angle_range)
        esprit_match, esprit_mse = _match_angles_min_mse(true_angles, esprit_est, k)

        rec = lambda match, mse: {
            'idx': int(i), 'snr': snr_val, 'K': k,
            'true_angles': true_angles, 'est_angles': list(match),
            'mse': float(mse),
        }
        per_method['Diffusion-V2'].append(rec(diff_match, diff_mse))
        per_method['MUSIC'].append(rec(music_match, music_mse))
        per_method['ESPRIT'].append(rec(esprit_match, esprit_mse))

        # Baselines
        bl_parts = []
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
            bl_parts.append(f"{bname}={bl_mse:.3f}")

        bl_str = ' '.join(bl_parts)
        print(f"  [{count+1}/{len(selected_indices)}] SNR={snr_val:+6.1f}dB K={k} "
              f"Diff-V2={diff_mse:.3f} MUSIC={music_mse:.3f} ESPRIT={esprit_mse:.3f} "
              f"{bl_str}")

    print('\n' + '=' * 64)
    print('V2 同样本对比结果')
    print('=' * 64)
    for name in per_method:
        if per_method[name]:
            print(f'\n[{name}]')
            _print_summary(per_method[name], show_worst=3)

    _plot_compare_figure(model_config, per_method)

    # ---- 随机抽取 12 个样本绘制谱对比图 ----
    if all_spec_items:
        n_plot = min(12, len(all_spec_items))
        plot_rng = np.random.RandomState(model_config.get('compare_random_seed', 42))
        plot_indices = plot_rng.choice(len(all_spec_items), size=n_plot, replace=False)
        plot_items = [all_spec_items[j] for j in sorted(plot_indices)]
        plot_dir = model_config.get('compare_plot_dir', './ComparePlotsV2')
        _plot_worst_samples(
            plot_items, angle_grid, plot_dir,
            tag=f"compare_v2_random12_{model_config.get('test_load_weight', 'model')}")


@torch.no_grad()
def test_spectrum_conditional_v2(model_config: dict):
    """V2 单独测试 (含 worst sample 画图)."""
    device = torch.device(model_config['device'])

    meta_path = os.path.join(model_config['save_weight_dir'], 'train_meta.npy')
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
        spec_label_type=model_config.get('spec_label_type', 'gaussian'),
        gaussian_sigma=model_config.get('gaussian_sigma', 1.0),
        tau=model_config.get('tau', 1),
    )

    sampler = _build_v2_sampler(model_config, spec_len, test_ds, device)

    use_snr_cond = model_config.get('use_snr_cond', True)
    use_k_cond = model_config.get('use_k_cond', False)
    snr_min = model_config.get('snr_range_min', -20.0)
    snr_max = model_config.get('snr_range_max', 20.0)
    k_min = float(model_config.get('k_range_min', 1))
    k_max = float(model_config.get('k_range_max', 3))

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
            selected_indices.extend(
                rng.choice(idxs, size=samples_per_snr, replace=False).tolist())
    selected_indices.sort()

    angle_grid = test_ds.angle_grid
    results = []
    worst_heap = []
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
        snr_norm = torch.tensor(
            [[_normalize_snr(snr_val, snr_min, snr_max)]],
            device=device, dtype=torch.float32)
        k_norm_t = None
        if use_k_cond:
            k_norm_t = torch.tensor(
                [[_normalize_k(float(k), k_min, k_max)]],
                device=device, dtype=torch.float32)
        x_t = torch.randn(1, 1, spec_len, device=device)
        pred_spec = sampler(
            x_t, cond,
            snr_norm=snr_norm if use_snr_cond else None,
            k_norm=k_norm_t,
        ).cpu().numpy()[0, 0]

        est_angles = _find_k_peaks(pred_spec, angle_grid, k)
        matched, mse = _match_angles_min_mse(true_angles, est_angles, k)
        results.append({
            'snr': snr_val, 'K': k,
            'true_angles': true_angles,
            'est_angles': matched,
            'mse': mse,
        })

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

        gap = _min_angle_gap(true_angles) if len(true_angles) >= 2 else float('inf')
        gap_str = f'{gap:.0f}°' if gap < 1e6 else '-'
        print(f"  [{count+1}/{len(selected_indices)}] SNR={snr_val:+6.1f}dB K={k} "
              f"MinGap={gap_str} "
              f"est={[round(a,1) for a in matched]} MSE={mse:.3f}")

    print('\n' + '=' * 60)
    print('V2 谱域条件扩散结果')
    print('=' * 60)
    _print_summary(results)

    worst_sorted = sorted(worst_heap, key=lambda x: x[0], reverse=True)
    worst_items = [item for (_, _, item) in worst_sorted]
    if worst_items:
        plot_dir = os.path.join(
            model_config.get('compare_plot_dir', './ComparePlots'),
            'worst_samples')
        _plot_worst_samples(worst_items, angle_grid, plot_dir,
                            tag=f"worst_v2_{model_config.get('test_load_weight', 'model')}")
