
#DOA数据集仿真


import numpy as np
import os
import argparse
from datetime import datetime


def steering_vector(angles_deg, M, d_lambda=0.5):
    angles_rad = np.deg2rad(angles_deg)  # 转为弧度
    m = np.arange(M).reshape(-1, 1)      # (M, 1)
    # 导向矢量: a(θ) = exp(-j * 2π * d/λ * m * sin(θ))
    A = np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(angles_rad).reshape(1, -1))
    return A


def generate_source_signals(K, N):
    S = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)
    return S


def generate_noise(M, N, snr_db, signal_power=1.0):
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    return noise


def generate_single_sample(M, N, K, snr_db, angle_range=(-90, 90), d_lambda=0.5,
                           min_angle_gap=5.0, max_angle_gap=None):
    # 随机不重叠DOA角度
    angles = _generate_separated_angles(K, angle_range, min_angle_gap, max_angle_gap)
    angles = np.sort(angles) 

    # 导向矢量矩阵
    A = steering_vector(angles, M, d_lambda)  # (M, K)

    # 信源信号
    S = generate_source_signals(K, N)  # (K, N)

    # 无噪声接收信号
    X_clean = A @ S  # (M, N)

    # 信号功率
    signal_power = np.mean(np.abs(X_clean) ** 2)

    # 噪声
    noise = generate_noise(M, N, snr_db, signal_power)  # (M, N)

    # 含噪声接收信号
    X = X_clean + noise  # (M, N)

    return S, X, X_clean, angles, snr_db


def _generate_separated_angles(K, angle_range, min_gap, max_gap=None):
    if K <= 0:
        raise ValueError('K 必须为正整数')
    if min_gap < 0:
        raise ValueError('min_gap 不能为负数')
    if max_gap is not None and max_gap <= 0:
        raise ValueError('max_gap 必须为正数或 None')
    if max_gap is not None and min_gap > max_gap:
        raise ValueError(f'min_gap({min_gap}) 不能大于 max_gap({max_gap})')

    int_min = int(np.ceil(angle_range[0]))
    int_max = int(np.floor(angle_range[1]))
    all_int_angles = np.arange(int_min, int_max + 1, dtype=float)
    if len(all_int_angles) < K:
        raise ValueError(
            f'角度范围内可选整数角度数量不足: 需要 K={K}, 实际只有 {len(all_int_angles)}'
        )

    # 检查
    if K > 1:
        span = float(int_max - int_min)
        if span < min_gap * (K - 1):
            raise ValueError(
                f'角度范围不足以满足最小间隔约束: span={span}°, K={K}, min_gap={min_gap}°'
            )
        if max_gap is not None and max_gap < 1.0:
            raise ValueError('在整数角度采样下，max_gap 不能小于 1°')

    max_attempts = 1000
    for _ in range(max_attempts):
        angles = np.sort(np.random.choice(all_int_angles, size=K, replace=False))
        if K == 1:
            return angles
        diffs = np.diff(angles)
        ok_min = np.all(diffs >= min_gap)
        ok_max = True if max_gap is None else np.all(diffs <= max_gap)
        if ok_min and ok_max:
            return angles

    raise RuntimeError(
        f'在 {max_attempts} 次尝试后仍未找到满足约束的角度组合: '
        f'K={K}, range={angle_range}, min_gap={min_gap}, max_gap={max_gap}'
    )


def generate_dataset(num_samples=10000,
                     M=8,
                     N=100,
                     K_range=(1, 3),
                     snr_range=(-20, 20),
                     snr_step=1,
                     snr_list=None,
                     angle_range=(-90, 90),
                     d_lambda=0.5,
                     min_angle_gap=5.0,
                     max_angle_gap=None,
                     seed=42):
  
    np.random.seed(seed)

    K_values = list(range(K_range[0], K_range[1] + 1))       
    use_fixed_snr = snr_list is not None
    if use_fixed_snr:
        snr_levels = np.array(sorted(snr_list), dtype=float)
    else:
        snr_levels = np.arange(snr_range[0], snr_range[1] + snr_step, snr_step, dtype=float)
    num_K = len(K_values)
    num_snr = len(snr_levels)
    num_cells = num_K * num_snr                               
    samples_per_cell = max(1, int(np.ceil(num_samples / num_cells)))
    actual_num_samples = samples_per_cell * num_cells          

    K_max = K_range[1]

    all_source_signals = np.zeros((actual_num_samples, K_max, N), dtype=np.complex128)
    all_received_signals = np.zeros((actual_num_samples, M, N), dtype=np.complex128)
    all_clean_signals = np.zeros((actual_num_samples, M, N), dtype=np.complex128)
    all_angles = np.full((actual_num_samples, K_max), np.nan, dtype=np.float64)
    all_snr = np.zeros(actual_num_samples, dtype=np.float64)
    all_num_targets = np.zeros(actual_num_samples, dtype=np.int32)

    print(f"开始生成DOA数据集（分层平衡采样）...")
    print(f"  阵元数 M = {M}")
    print(f"  快拍数 N = {N}")
    print(f"  目标数 K ∈ {K_values}")
    if use_fixed_snr:
        print(f"  SNR 固定值: {snr_levels.tolist()} dB → {num_snr} 档")
    else:
        print(f"  SNR ∈ [{snr_range[0]}, {snr_range[1]}] dB, 步长 {snr_step} dB → {num_snr} 档")
    print(f"  DOA角度 ∈ [{angle_range[0]}°, {angle_range[1]}°]")
    print(f"  最小角度间隔 = {min_angle_gap}°")
    if max_angle_gap is None:
        print("  最大角度间隔 = 不限制")
    else:
        print(f"  最大角度间隔 = {max_angle_gap}°")
    print(f"  分层网格: {num_K} (K) × {num_snr} (SNR) = {num_cells} 个组合")
    print(f"  每个组合 {samples_per_cell} 个样本 → 实际总样本数 = {actual_num_samples}")
    print()

    idx = 0
    progress_step = max(1, actual_num_samples // 10)

    for K in K_values:
        for snr_center in snr_levels:
            for _ in range(samples_per_cell):
                if use_fixed_snr:
                    # 固定 SNR 列表
                    snr_db = snr_center
                else:
                    # 连续 SNR 模式
                    snr_jitter = np.random.uniform(-snr_step / 2, snr_step / 2)
                    snr_db = np.clip(snr_center + snr_jitter, snr_range[0], snr_range[1])

                # 生成单个样本
                S, X, X_clean, angles, snr = generate_single_sample(
                    M, N, K, snr_db, angle_range, d_lambda, min_angle_gap, max_angle_gap
                )

                # 存储
                all_source_signals[idx, :K, :] = S
                all_received_signals[idx] = X
                all_clean_signals[idx] = X_clean
                all_angles[idx, :K] = angles
                all_snr[idx] = snr
                all_num_targets[idx] = K

                idx += 1
                if idx % progress_step == 0 or idx == 1:
                    print(f"  进度: {idx}/{actual_num_samples} ({100 * idx / actual_num_samples:.1f}%)")


    dataset = {
        'source_signals': all_source_signals,       
        'received_signals': all_received_signals,   
        'clean_signals': all_clean_signals,         
        'angles': all_angles,                        
        'snr': all_snr,                              
        'num_targets': all_num_targets,              
        'params': {
            'M': M,
            'N': N,
            'K_range': K_range,
            'snr_range': snr_range,
            'snr_step': snr_step,
            'snr_list': snr_levels.tolist() if use_fixed_snr else None,
            'angle_range': angle_range,
            'd_lambda': d_lambda,
            'min_angle_gap': min_angle_gap,
            'max_angle_gap': max_angle_gap,
            'seed': seed,
            'num_samples': actual_num_samples,
            'samples_per_cell': samples_per_cell,
            'num_K': num_K,
            'num_snr_levels': num_snr,
        }
    }

    print(f"\n数据集生成完毕！（实际样本数: {actual_num_samples}）")
    return dataset


def save_dataset(dataset, save_dir='DOA_Dataset', prefix='doa_dataset'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.npz"
    filepath = os.path.join(save_dir, filename)

    np.savez_compressed(
        filepath,
        source_signals_real=dataset['source_signals'].real,
        source_signals_imag=dataset['source_signals'].imag,
        received_signals_real=dataset['received_signals'].real,
        received_signals_imag=dataset['received_signals'].imag,
        clean_signals_real=dataset['clean_signals'].real,
        clean_signals_imag=dataset['clean_signals'].imag,
        angles=dataset['angles'],
        snr=dataset['snr'],
        num_targets=dataset['num_targets'],
        params=np.array([dataset['params']], dtype=object),
    )
    print(f"数据集已保存至: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    return filepath


def load_dataset(filepath):
    data = np.load(filepath, allow_pickle=True)
    dataset = {
        'source_signals': data['source_signals_real'] + 1j * data['source_signals_imag'],
        'received_signals': data['received_signals_real'] + 1j * data['received_signals_imag'],
        'clean_signals': data['clean_signals_real'] + 1j * data['clean_signals_imag'],
        'angles': data['angles'],
        'snr': data['snr'],
        'num_targets': data['num_targets'],
        'params': data['params'][0],
    }
    return dataset


def dataset_summary(dataset):
    params = dataset['params']
    n = params['num_samples']
    print("=" * 60)
    print("          DOA 估计仿真数据集摘要（分层平衡采样）")
    print("=" * 60)
    print(f"  样本总数          : {n}")
    print(f"  阵元数 M          : {params['M']}")
    print(f"  快拍数 N          : {params['N']}")
    print(f"  目标数 K          : {params['K_range'][0]} ~ {params['K_range'][1]} ({params['num_K']} 类)")
    print(f"  SNR 范围          : {params['snr_range'][0]} ~ {params['snr_range'][1]} dB, 步长 {params['snr_step']} dB ({params['num_snr_levels']} 档)")
    print(f"  DOA 角度范围      : {params['angle_range'][0]}° ~ {params['angle_range'][1]}°")
    print(f"  阵元间距 d/λ      : {params['d_lambda']}")
    print(f"  最小角度间隔      : {params['min_angle_gap']}°")
    max_gap = params.get('max_angle_gap', None)
    print(f"  最大角度间隔      : {'不限制' if max_gap is None else f'{max_gap}°'}")
    print(f"  随机种子          : {params['seed']}")
    print(f"  分层网格          : {params['num_K']} × {params['num_snr_levels']} = {params['num_K'] * params['num_snr_levels']} 个组合")
    print(f"  每组合样本数      : {params['samples_per_cell']}")
    print("-" * 60)

    # 各目标数样本
    print("  [按目标数 K 统计]")
    for k in range(params['K_range'][0], params['K_range'][1] + 1):
        count = np.sum(dataset['num_targets'] == k)
        print(f"    K={k} : {count} 个样本 ({100 * count / n:.1f}%)")

    # SNR分布
    snr = dataset['snr']
    print(f"  [SNR 分布统计]")
    print(f"    均值 = {np.mean(snr):.2f} dB, 标准差 = {np.std(snr):.2f} dB")
    snr_bins = np.arange(params['snr_range'][0], params['snr_range'][1] + 5, 5)
    for i in range(len(snr_bins) - 1):
        lo, hi = snr_bins[i], snr_bins[i + 1]
        count = np.sum((snr >= lo) & (snr < hi))
        print(f"    [{lo:+6.0f}, {hi:+6.0f}) dB : {count} 个样本 ({100 * count / n:.1f}%)")
    count = np.sum(snr >= snr_bins[-1])
    if count > 0:
        print(f"    [{snr_bins[-1]:+6.0f},   +∞) dB : {count} 个样本 ({100 * count / n:.1f}%)")

    print("=" * 60)


def show_sample(dataset, idx=0):
    #展示单个样本的详细信息
    K = dataset['num_targets'][idx]
    angles = dataset['angles'][idx, :K]
    snr = dataset['snr'][idx]
    X = dataset['received_signals'][idx]
    S = dataset['source_signals'][idx, :K]
    X_clean = dataset['clean_signals'][idx]

    print(f"\n--- 样本 #{idx} ---")
    print(f"  目标数 K     = {K}")
    print(f"  DOA 角度     = {np.array2string(angles, precision=2, separator=', ')}°")
    print(f"  SNR          = {snr:.2f} dB")
    print(f"  信源信号形状 = {S.shape}")
    print(f"  接收信号形状 = {X.shape}")
    print(f"  干净信号形状 = {X_clean.shape}")

    # 验证
    noise = X - X_clean
    actual_snr = 10 * np.log10(np.mean(np.abs(X_clean) ** 2) / np.mean(np.abs(noise) ** 2))
    print(f"  实际 SNR     = {actual_snr:.2f} dB")

# 入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DOA估计仿真数据集生成器')
    parser.add_argument('--num_samples', type=int, default=5000, help='样本总数 (默认: 500000)')
    parser.add_argument('--M', type=int, default=8, help='阵元数 (默认: 8)')
    parser.add_argument('--N', type=int, default=1000, help='快拍数 (默认: 100)')
    parser.add_argument('--K_min', type=int, default=1, help='最小目标数 (默认: 1)')
    parser.add_argument('--K_max', type=int, default=3, help='最大目标数 (默认: 3)')
    parser.add_argument('--snr_min', type=float, default=-20, help='最小SNR(dB) (默认: 0)')
    parser.add_argument('--snr_max', type=float, default=0, help='最大SNR(dB) (默认: 20)')
    parser.add_argument('--snr_step', type=float, default=5, help='SNR离散化步长(dB) (默认: 5)')
    parser.add_argument('--snr_list', type=float, nargs='+',
                        default=[-20, -15, -10, -5, 0],
                        help='固定SNR列表(dB), 设为空则用snr_range/snr_step')
    parser.add_argument('--angle_min', type=float, default=-60, help='最小DOA角度(度) (默认: -60)')
    parser.add_argument('--angle_max', type=float, default=60, help='最大DOA角度(度) (默认: 60)')
    parser.add_argument('--min_gap', type=float, default=1.0, help='目标间最小角度间隔(度) (默认: 5)')
    parser.add_argument('--max_gap', type=float, default=10,
                        help='目标间最大角度间隔(度), 默认不限制')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--save_dir', type=str, default='DOA_Dataset', help='保存目录 (默认: DOA_Dataset)')
    parser.add_argument('--no_save', action='store_true', help='不保存数据集到文件')

    args = parser.parse_args()

    # 生成数据集
    snr_list = args.snr_list if args.snr_list else None
    dataset = generate_dataset(
        num_samples=args.num_samples,
        M=args.M,
        N=args.N,
        K_range=(args.K_min, args.K_max),
        snr_range=(args.snr_min, args.snr_max),
        snr_step=args.snr_step,
        snr_list=snr_list,
        angle_range=(args.angle_min, args.angle_max),
        min_angle_gap=args.min_gap,
        max_angle_gap=args.max_gap,
        seed=args.seed,
    )

    # 打印
    dataset_summary(dataset)

    # 展示几个样本
    for idx in [0, 1, 2]:
        show_sample(dataset, idx)

    # 保存
    if not args.no_save:
        save_dataset(dataset, save_dir=args.save_dir)
