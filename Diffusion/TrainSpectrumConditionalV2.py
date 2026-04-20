
#谱条件扩散 V2 训练

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

from .DOASpectrumDataset import DOASpectrumDataset, find_latest_npz
from .DiffusionSpectrumConditionalV2 import ConditionalSpectrumDiffusionTrainerV2
from .ModelSpectrumConditionalV2 import ConditionalSpectrumUNet1D_V2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scheduler import GradualWarmupScheduler


def _normalize_snr(snr_val, snr_min=-20.0, snr_max=20.0):
    #将 SNR 归一化到 [-1, 1].
    return 2.0 * (snr_val - snr_min) / max(snr_max - snr_min, 1e-8) - 1.0


def _normalize_k(k_val, k_min=1, k_max=3):
    #将目标数 K 归一化到 [-1, 1].
    return 2.0 * (k_val - k_min) / max(k_max - k_min, 1e-8) - 1.0


def _build_peak_weight(target_spec, angles, angle_min=-90.0, angle_step=1.0, peak_weight=6.0, neighborhood=2):
    bsz, _, length = target_spec.shape
    weights = torch.ones_like(target_spec) * 0.1  
    boost = float(peak_weight)
    
    for b in range(bsz):
        b_angles = angles[b] 
        for theta in b_angles:
            if torch.isnan(theta):
                continue
            center = int(round((theta.item() - angle_min) / angle_step))
            left = max(0, center - neighborhood)
            right = min(length, center + neighborhood + 1)
            weights[b, 0, left:right] += boost
            
    return weights


def _snr_loss_weight(snr_batch, snr_min=-20.0, snr_max=20.0, max_weight=3.0):
    t = (snr_batch - snr_min) / max(snr_max - snr_min, 1e-8)  # 0→1
    t = t.clamp(0.0, 1.0)
    return max_weight - (max_weight - 1.0) * t  # max_weight → 1.0


def train_v2(model_config: dict):
    device = torch.device(model_config['device'])
    npz_path = model_config.get('npz_path') or find_latest_npz()
    if npz_path is None:
        raise FileNotFoundError('找不到 DOA 数据集 (.npz)')

    train_ds = DOASpectrumDataset(
        npz_path=npz_path,
        split='train',
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

    save_dir = model_config['save_weight_dir']
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, 'train_meta.npy')
    np.save(meta_path, {
        'M': train_ds.M,
        'N': train_ds.N,
        'spec_len': int(train_ds.clean_spec.shape[-1]),
        'angle_grid': train_ds.angle_grid,
        'norm_info': train_ds.norm_info,
        'model_version': 'v2',
        'tau': train_ds.tau,
    })
    print(f'[Train-V2] 元数据已保存 -> {meta_path}')

    loader = DataLoader(
        train_ds,
        batch_size=model_config['batch_size'],
        shuffle=True,
        num_workers=model_config.get('num_workers', 4),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    use_snr_cond = model_config.get('use_snr_cond', True)
    use_k_cond = model_config.get('use_k_cond', False)
    cfg_drop_prob = model_config.get('cfg_drop_prob', 0.1)
    tau = model_config.get('tau', 1)
    use_anti_rectifier = model_config.get('use_anti_rectifier', True)

    net = ConditionalSpectrumUNet1D_V2(
        T=model_config['T'],
        spec_len=int(train_ds.clean_spec.shape[-1]),
        M=model_config.get('M', train_ds.M),
        base_ch=model_config.get('spec_base_ch', 128),
        num_res_blocks=model_config.get('spec_res_blocks', 2),
        dropout=model_config.get('dropout', 0.1),
        use_snr_cond=use_snr_cond,
        use_k_cond=use_k_cond,
        cfg_drop_prob=cfg_drop_prob,
        tau=tau,
        use_anti_rectifier=use_anti_rectifier,
    ).to(device)

    if model_config.get('training_load_weight') is not None:
        ckpt = os.path.join(save_dir, model_config['training_load_weight'])
        if os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt, map_location=device))
            print(f'[Train-V2] 加载权重: {ckpt}')

    optimizer = Adam(net.parameters(), lr=model_config['lr'])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_config['epoch'], eta_min=0, last_epoch=-1)
    warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=model_config.get('multiplier', 2.0),
        warm_epoch=max(1, model_config['epoch'] // 10),
        after_scheduler=cosine
    )

    trainer = ConditionalSpectrumDiffusionTrainerV2(
        net,
        model_config['beta_1'],
        model_config['beta_T'],
        model_config['T'],
    ).to(device)

    peak_lambda = float(model_config.get('peak_loss_lambda', 0.3))
    peak_weight = float(model_config.get('peak_weight', 6.0))
    peak_neighborhood = int(model_config.get('peak_neighborhood', 2))
    snr_weight_max = float(model_config.get('snr_loss_weight_max', 3.0))
    snr_min = float(model_config.get('snr_range_min', -20.0))
    snr_max = float(model_config.get('snr_range_max', 20.0))
    k_min = float(model_config.get('k_range_min', 1))
    k_max = float(model_config.get('k_range_max', 3))
    T_max = float(model_config['T'])

    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[Train-V2] 模型参数量: {param_count:,}')
    print(f'[Train-V2] CFG drop prob: {cfg_drop_prob}, SNR 条件: {use_snr_cond}, K 条件: {use_k_cond}')
    print(f'[Train-V2] SNR loss weight: min SNR→{snr_weight_max}x, max SNR→1.0x')
    print(f'[Train-V2] Time-Decaying Peak Loss: 开启 (基础 lambda={peak_lambda})')

    total_epochs = model_config['epoch']
    
    use_curriculum = model_config.get('use_curriculum', False)
    curriculum_epochs = int(model_config.get('curriculum_epochs', total_epochs // 2))
    curriculum_start_snr = float(model_config.get('curriculum_start_snr', 0.0))
    if use_curriculum:
        print(f'[Train-V2] Curriculum Learning: 开启 (前 {curriculum_epochs} 轮 SNR 阈值从 {curriculum_start_snr}dB 降至 {snr_min}dB)')

    history_noise = []
    history_peak = []
    plot_dir = save_dir  
    
    for epoch in range(1, total_epochs + 1):
        net.train()
        epoch_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_peak_loss = 0.0

        if use_curriculum:
            if epoch <= curriculum_epochs:
                # 随 Epoch 降低 SNR 门槛
                progress = (epoch - 1) / max(1, curriculum_epochs - 1)
                current_snr_threshold = curriculum_start_snr - progress * (curriculum_start_snr - snr_min)
            else:
                current_snr_threshold = snr_min  
        else:
            current_snr_threshold = snr_min     

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [SNR>={current_snr_threshold:.1f}dB]',
                    leave=False, ncols=110, position=0)
        
        valid_batches = 0

        for clean_spec, received_cov, info in pbar:
            snr_vals_all = info['snr'].float()
            valid_mask = snr_vals_all >= current_snr_threshold

            if not valid_mask.any():
                continue

            valid_batches += 1
            
            x_0 = clean_spec[valid_mask].to(device)
            condition = received_cov[valid_mask].to(device)
            snr_vals = snr_vals_all[valid_mask]
            num_targets = info['num_targets'][valid_mask]
            angles = info['angles'][valid_mask]

            # SNR 条件
            snr_norm = _normalize_snr(snr_vals, snr_min, snr_max)
            snr_norm = snr_norm.unsqueeze(1).to(device)  # (B', 1)

            # K 条件
            k_norm_t = None
            if use_k_cond:
                k_vals = num_targets.float()
                k_norm_t = _normalize_k(k_vals, k_min, k_max)
                k_norm_t = k_norm_t.unsqueeze(1).to(device)  # (B', 1)

            terms = trainer.training_terms(
                x_0, condition,
                snr_norm=snr_norm if use_snr_cond else None,
                k_norm=k_norm_t)

            noise_loss_per_sample = terms['noise_loss_map'].mean(dim=(1, 2))  # (B',)

            # SNR-weighted loss
            snr_w = _snr_loss_weight(snr_vals.to(device), snr_min, snr_max,
                                     snr_weight_max)  # (B',)
            noise_loss = (noise_loss_per_sample * snr_w).mean()

            # Peak loss
            peak_w = _build_peak_weight(
                terms['x0_target'],
                angles=angles,            
                angle_min=model_config.get('angle_min', -90.0),
                angle_step=model_config.get('angle_step', 1.0),
                peak_weight=peak_weight,
                neighborhood=peak_neighborhood,
            ).to(device)

            raw_peak_loss_per_sample = (
                F.smooth_l1_loss(terms['x0_pred'], terms['x0_target'], reduction='none') * peak_w
            ).mean(dim=(1, 2))  # (B',)

            # 时间衰退
            if 't' in terms:
                t_batch = terms['t'].float()  # (B',)
                dynamic_lambda = peak_lambda * (1.0 - t_batch / T_max)
            else:
                dynamic_lambda = peak_lambda
                
            weighted_peak_loss = (raw_peak_loss_per_sample * dynamic_lambda * snr_w).mean()

            # 联合 Loss
            loss = noise_loss + weighted_peak_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           model_config.get('grad_clip', 1.0))
            optimizer.step()

            pure_peak_loss_log = (raw_peak_loss_per_sample * snr_w).mean()

            epoch_loss += float(loss.item())
            epoch_noise_loss += float(noise_loss.item())
            epoch_peak_loss += float(pure_peak_loss_log.item())

            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             noise=f'{noise_loss.item():.4f}',
                             peak=f'{pure_peak_loss_log.item():.4f}')

        pbar.close()
        warmup.step()
        
        n_batches = max(valid_batches, 1)
        avg_loss = epoch_loss / n_batches
        avg_noise = epoch_noise_loss / n_batches
        avg_peak = epoch_peak_loss / n_batches
        
        tqdm.write(f"Epoch {epoch}/{model_config['epoch']} | Valid Batches: {valid_batches} | Loss: {avg_loss:.6f} "
                   f"(noise={avg_noise:.6f}, pure_peak={avg_peak:.6f}) | "
                   f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        history_noise.append(avg_noise)
        history_peak.append(avg_peak)

        # 每 10 轮画一次 Loss 曲线
        if epoch % 10 == 0 or epoch == total_epochs:
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs_x = list(range(1, epoch + 1))
            ax.plot(epochs_x, history_noise, color='#2196F3', label='Noise Loss', linewidth=1.5)
            ax.plot(epochs_x, history_peak, color='#FF5722', label='Pure Peak Loss', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss (Epoch {epoch})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = os.path.join(plot_dir, 'train_loss_curve.png')
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            tqdm.write(f'  [绘图] {fig_path}')

        if epoch % model_config.get('save_every', 20) == 0 or epoch == model_config['epoch']:
            ckpt_path = os.path.join(save_dir, f'ckpt_{epoch}_.pt')
            torch.save(net.state_dict(), ckpt_path)
            tqdm.write(f'  [保存] {ckpt_path}')


def eval_v2(model_config: dict):
    from .TestSpectrumConditionalV2 import test_spectrum_conditional_v2
    test_spectrum_conditional_v2(model_config)


def compare_v2(model_config: dict):
    from .TestSpectrumConditionalV2 import test_compare_v2
    test_compare_v2(model_config)