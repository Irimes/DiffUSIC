
import os, sys, numpy as np, torch
from scipy.signal import find_peaks as sp_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Diffusion.DOASpectrumDataset import DOASpectrumDataset
from Diffusion.TestSpectrumConditionalV2 import (
    _normalize_snr, _normalize_k, _find_k_peaks,
    _match_angles_min_mse, _min_angle_gap,
)
from Diffusion.ModelSpectrumConditionalV2 import ConditionalSpectrumUNet1D_V2
from Diffusion.DiffusionSpectrumConditionalV2 import ConditionalSpectrumDiffusionSamplerV2

npz_path = r"D:\Files\Academic_Resourses\Codes\diffusion again - 副本 - 副本 - 副本\DOA_Dataset\LargeAngleMedium.npz"
device = torch.device('cuda:0')

meta = np.load('./CheckpointsSpectrumConditionalV2/train_meta.npy', allow_pickle=True).item()
spec_len = int(meta['spec_len'])

test_ds = DOASpectrumDataset(
    npz_path=npz_path, split='test', test_ratio=0.001, seed=42,
    angle_min=-60.0, angle_max=60.0, angle_step=1.0,
    d_lambda=0.5, spec_floor_db=-40.0,
    spec_label_type='gaussian', gaussian_sigma=1.0, tau=8,
)

ckpt_path = './CheckpointsSpectrumConditionalV2/ckpt_100_.pt'
state = torch.load(ckpt_path, map_location='cpu')
has_k = any('k_embedding' in k for k in state.keys())
print(f'Checkpoint has K embedding: {has_k}')

net = ConditionalSpectrumUNet1D_V2(
    T=400, spec_len=spec_len, M=8, base_ch=128,
    num_res_blocks=2, dropout=0.0,
    use_snr_cond=True, use_k_cond=has_k,
    cfg_drop_prob=0.0, tau=8, use_anti_rectifier=True
).to(device)
net.load_state_dict(state)
net.eval()

sampler = ConditionalSpectrumDiffusionSamplerV2(
    net, 1e-4, 0.02, 400, cfg_scale=2.0).to(device)

print(f'Test samples: {len(test_ds)}, spec_len={spec_len}')
angle_grid = test_ds.angle_grid

results = []
with torch.no_grad():
    for i in range(len(test_ds)):
        clean_spec, recv_cov, info = test_ds[i]
        k = int(info['num_targets'])
        snr_val = float(info['snr'])
        true_angles = info['angles']
        if isinstance(true_angles, np.ndarray):
            true_angles = true_angles.tolist()
        true_angles = sorted([float(a) for a in true_angles[:k]])

        cond = recv_cov.unsqueeze(0).to(device)
        snr_norm = torch.tensor([[_normalize_snr(snr_val, -20.0, 20.0)]],
                                device=device, dtype=torch.float32)
        k_norm_t = None
        if has_k:
            k_norm_t = torch.tensor([[_normalize_k(float(k), 1, 3)]],
                                    device=device, dtype=torch.float32)

        x_t = torch.randn(1, 1, spec_len, device=device)
        pred_spec = sampler(x_t, cond, snr_norm=snr_norm, k_norm=k_norm_t).cpu().numpy()[0, 0]

        est_angles = _find_k_peaks(pred_spec, angle_grid, k)
        matched, mse = _match_angles_min_mse(true_angles, est_angles, k)
        gap = _min_angle_gap(true_angles) if len(true_angles) >= 2 else float('inf')

        results.append({
            'idx': i, 'snr': snr_val, 'K': k,
            'true': true_angles, 'est': list(matched),
            'mse': mse, 'gap': gap,
            'pred_spec': pred_spec,
            'true_spec': clean_spec.numpy()[0],
        })

results.sort(key=lambda x: x['mse'], reverse=True)

print()
print('=' * 80)
print('WORST 16 SAMPLES')
print('=' * 80)
for r in results[:16]:
    gap_s = f"{r['gap']:.0f}deg" if r['gap'] < 1e6 else '-'
    true_s = [round(a, 1) for a in r['true']]
    est_s = [round(a, 1) for a in r['est']]
    print(f"  idx={r['idx']:3d} SNR={r['snr']:+6.1f}dB K={r['K']} Gap={gap_s:>6s} "
          f"True={true_s} Est={est_s} MSE={r['mse']:.2f}")

print()
print('=' * 80)
print('PER-SNR STATISTICS')
print('=' * 80)
snrs_arr = np.array([r['snr'] for r in results])
mses_arr = np.array([r['mse'] for r in results])
ks_arr = np.array([r['K'] for r in results])

for snr_val in sorted(np.unique(snrs_arr)):
    mask = snrs_arr == snr_val
    m = mses_arr[mask]
    print(f"SNR={snr_val:+6.1f}dB  n={mask.sum():3d}  "
          f"RMSE={np.sqrt(m.mean()):.3f}  median={np.median(m):.3f}  max={m.max():.2f}")

print()
print('PER-K STATISTICS')
print('-' * 40)
for k_val in sorted(np.unique(ks_arr)):
    mask = ks_arr == k_val
    m = mses_arr[mask]
    print(f"K={k_val}  n={mask.sum():3d}  "
          f"RMSE={np.sqrt(m.mean()):.3f}  median={np.median(m):.3f}  max={m.max():.2f}")

print()
print('=' * 80)
print('WORST 16 PATTERN ANALYSIS')
print('=' * 80)
worst16 = results[:16]
gaps_w = [r['gap'] for r in worst16 if r['gap'] < 1e6]
print(f"Avg angle gap: {np.mean(gaps_w):.1f} deg" if gaps_w else "No multi-target")
print(f"Avg SNR: {np.mean([r['snr'] for r in worst16]):.1f} dB")
k_dist = {}
for r in worst16:
    k_dist[r['K']] = k_dist.get(r['K'], 0) + 1
print(f"K distribution: {k_dist}")

print()
print('DETAILED WORST 4 PREDICTION SHAPE')
print('-' * 60)
for j, r in enumerate(results[:4]):
    ps = r['pred_spec']
    ts = r['true_spec']
    peaks_p, _ = sp_peaks(ps, height=0.3, distance=3)
    peaks_t, _ = sp_peaks(ts, height=0.3, distance=3)
    print(f"\n[Worst #{j+1}] idx={r['idx']} SNR={r['snr']:+.0f}dB K={r['K']}")
    print(f"  True angles: {r['true']}")
    print(f"  Est  angles: {[round(a,1) for a in r['est']]}")
    print(f"  Pred spec: min={ps.min():.3f} max={ps.max():.3f} mean={ps.mean():.3f}")
    print(f"  True spec: min={ts.min():.3f} max={ts.max():.3f} mean={ts.mean():.3f}")
    print(f"  Pred peaks (h>0.3): {len(peaks_p)} at {[angle_grid[p] for p in peaks_p]}")
    print(f"  True peaks (h>0.3): {len(peaks_t)} at {[angle_grid[p] for p in peaks_t]}")

    all_pred_peaks, _ = sp_peaks(ps, height=0.1, distance=2)
    print(f"  Pred peaks (h>0.1): {len(all_pred_peaks)} at {[angle_grid[p] for p in all_pred_peaks]}")

print()
print(f"\nOverall RMSE: {np.sqrt(mses_arr.mean()):.3f} deg")
print(f"Median MSE: {np.median(mses_arr):.4f}")
print(f"Worst MSE: {mses_arr.max():.2f}")
print(f"Fraction MSE>10: {(mses_arr > 10).sum()}/{len(mses_arr)}")
print(f"Fraction MSE>50: {(mses_arr > 50).sum()}/{len(mses_arr)}")
