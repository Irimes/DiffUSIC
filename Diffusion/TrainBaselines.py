

import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .BaselineDataset import BaselineDataset
from .BaselineModels import (
    SubspaceNet, DeepSFNS, DeepSSE, IQResNet,
    DOALowSNRNet, DeepAugmentMusic,
)



#  损失函数

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = True
        self.eps = 1e-8

    def forward(self, x, y):
        xs_pos = x
        xs_neg = 1 - x
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum(dim=1).mean()


def rmspe_loss(predictions, true_angles_rad, num_targets_list):
    MOD = np.pi  # ULA 周期
    device = true_angles_rad.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(len(num_targets_list)):
        K = int(num_targets_list[b])
        if K <= 0:
            continue

        pred = predictions[b] if isinstance(predictions, list) else predictions[b]
        true = true_angles_rad[b, :K]

        if pred.is_complex():
            pred = pred.real
        pred = pred.float()
        true = true.float()

        pred_k = pred[:K]
        n_pred = pred_k.shape[0]
        n_true = true.shape[0]
        min_dim = min(n_pred, n_true)
        if min_dim <= 0:
            continue

        cost = (pred_k.unsqueeze(1) - true.unsqueeze(0)) ** 2
        cost_np = cost.detach().cpu().numpy().astype(np.float64)
        row_ind, col_ind = linear_sum_assignment(cost_np)

        loss = torch.tensor(0.0, device=device)
        for ri, ci in zip(row_ind, col_ind):
            diff = pred_k[ri] - true[ci]
            loss = loss + ((diff + MOD / 2) % MOD - MOD / 2) ** 2
        loss = torch.sqrt(loss / min_dim + 1e-12)

        total_loss = total_loss + loss
        count += 1

    return total_loss / max(count, 1)


#  模型结构

def _build_model(name, cfg, ds):
    M, N = ds.M, ds.N
    num_grids = ds.num_grids
    d_lambda = ds.d_lambda
    angle_min = cfg.get('angle_min', -60.0)
    angle_max = cfg.get('angle_max', 60.0)
    angle_step = cfg.get('baseline_angle_step', 1.0)

    if name == 'SubspaceNet':
        return SubspaceNet(tau=ds.tau, diff_method='root_music')
    elif name == 'DeepSFNS':
        return DeepSFNS(M=M, num_grids=num_grids, snapshots=N)
    elif name == 'DeepSSE':
        return DeepSSE(
            steering_vector_np=ds.steering_matrix,
            num_class=num_grids, num_antenna=M)
    elif name == 'IQResNet':
        return IQResNet(num_classes=num_grids, num_antennas=M)
    elif name == 'DOALowSNRNet':
        return DOALowSNRNet(num_out_grids=num_grids)
    elif name == 'daMUSIC':
        return DeepAugmentMusic(
            num_antennas=M, d_lambda=d_lambda,
            angle_min=angle_min, angle_max=angle_max, angle_step=angle_step)
    else:
        raise ValueError(f"未知模型: {name}")


#  单模型训练

_ASL = AsymmetricLoss()


def _compute_loss(name, model, batch, device, A_tensor=None):

    if name == 'SubspaceNet':
        Rx_tau = batch['Rx_tau'].to(device)
        labels = batch['label_grid'].to(device)
        angles_rad = batch['angles_rad'].to(device)
        num_t = batch['num_targets'].tolist()
        predictions, _ = model(Rx_tau, labels)
        return rmspe_loss(predictions, angles_rad, num_t)

    elif name == 'DeepSFNS':
        signal = batch['signal'].to(device)
        labels = batch['label_grid'].to(device)
        bs = signal.shape[0]
        A_batch = A_tensor.expand(bs, -1, -1)
        output = model(A_batch, signal)
        return _ASL(output, labels)

    elif name == 'DeepSSE':
        cov3ch = batch['cov3ch'].to(device)
        labels = batch['label_grid'].to(device)
        output = model(cov3ch)
        return _ASL(output, labels)

    elif name == 'IQResNet':
        signal = batch['signal'].to(device)
        labels = batch['label_grid'].to(device)
        output = model(signal)
        return nn.functional.binary_cross_entropy(output, labels)

    elif name == 'DOALowSNRNet':
        cov3ch = batch['cov3ch'].to(device)
        labels = batch['label_grid'].to(device)
        output = model(cov3ch)
        return nn.functional.binary_cross_entropy(output, labels)

    elif name == 'daMUSIC':
        signal = batch['signal'].to(device)
        angles_rad = batch['angles_rad'].to(device)
        num_t = batch['num_targets'].tolist()
        output = model(signal)
        return rmspe_loss(output, angles_rad, num_t)

    raise ValueError(f"未知模型: {name}")


def _train_single_baseline(name, cfg):
    device = torch.device(cfg['device'])
    print(f"\n{'=' * 60}\n  训练 {name}\n{'=' * 60}")

    common_ds_kwargs = dict(
        npz_path=cfg['npz_path'],
        test_ratio=cfg.get('baseline_test_ratio', cfg.get('test_ratio', 0.1)),
        seed=cfg.get('baseline_split_seed', 123),
        angle_min=cfg.get('angle_min', -60.0),
        angle_max=cfg.get('angle_max', 60.0),
        baseline_angle_step=cfg.get('baseline_angle_step', 1.0),
        d_lambda=cfg.get('d_lambda', 0.5),
        tau=cfg.get('baseline_tau', 8),
    )

    ds = BaselineDataset(split='train', **common_ds_kwargs)
    val_ds = BaselineDataset(split='test', **common_ds_kwargs)

    bs = cfg.get('baseline_batch_size', 64)
    loader = DataLoader(
        ds, batch_size=bs, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, drop_last=False, pin_memory=True,
    )
    num_batches = len(loader)

    model = _build_model(name, cfg, ds).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.get('baseline_lr', 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    epochs = cfg.get('baseline_epochs', 100)
    save_dir = os.path.join(
        cfg.get('baseline_save_dir', './BaselineCheckpoints'), name)
    os.makedirs(save_dir, exist_ok=True)

    A_tensor = None
    if name == 'DeepSFNS':
        A_tensor = torch.from_numpy(ds.steering_matrix).unsqueeze(0).to(device)

    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    import time as _time

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = _time.time()

        pbar = tqdm(loader, desc=f"  [{name}] Epoch {epoch}/{epochs}",
                    leave=True, dynamic_ncols=True)
        for bi, batch in enumerate(pbar):
            optimizer.zero_grad()
            try:
                loss = _compute_loss(name, model, batch, device, A_tensor)
            except Exception as e:
                print(f"    [WARN] batch 跳过: {e}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    vloss = _compute_loss(name, model, batch, device, A_tensor)
                    val_total += vloss.item()
                    val_count += 1
                except Exception:
                    continue
        avg_val_loss = val_total / max(val_count, 1)
        elapsed = _time.time() - t0

        print(f"  [{name}] Epoch {epoch}/{epochs}  "
              f"TrainLoss={avg_loss:.6f}  ValLoss={avg_val_loss:.6f}  "
              f"LR={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.0f}s)",
              flush=True)

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] 早停触发 (连续 {patience} 个 epoch 验证损失无改善)",
                      flush=True)
                break

        if epoch % 20 == 0 or epoch == epochs:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'ckpt_{epoch}.pt'))

        scheduler.step()

    print(f"  [{name}] 训练完成, best val loss = {best_val_loss:.6f}")


# 入口

ALL_BASELINE_NAMES = [
    'SubspaceNet', 'DeepSFNS', 'DeepSSE',
    'IQResNet', 'DOALowSNRNet', 'daMUSIC',
]


def train_baselines(model_config):
    models = model_config.get('baseline_models', ALL_BASELINE_NAMES)
    for name in models:
        _train_single_baseline(name, model_config)
    print('\n[TrainBaselines] 全部训练完成。')
