import os, sys, traceback
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import torch
    print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from Diffusion.BaselineDataset import BaselineDataset
    from Diffusion.BaselineModels import SubspaceNet
    from Diffusion.TrainBaselines import _compute_loss
    from torch.utils.data import DataLoader

    device = torch.device('cuda:0')

    ds = BaselineDataset(
        npz_path=r"D:\Files\Academic_Resourses\Codes\diffusion again - 副本 - 副本 - 副本\DOA_Dataset\SmallAngleLarge.npz",
        split='train', test_ratio=0.001, seed=42,
    )

    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                        drop_last=True, pin_memory=True)

    model = SubspaceNet(tau=8, diff_method='esprit').to(device)
    print(f"Model on device, params: {sum(p.numel() for p in model.parameters())}")

    print("Starting first batch...")
    for batch in loader:
        print(f"  Batch loaded: {list(batch.keys())}")
        print(f"  Rx_tau shape: {batch['Rx_tau'].shape}")
        print(f"  Computing loss...")
        loss = _compute_loss('SubspaceNet', model, batch, device)
        print(f"  Loss = {loss.item():.6f}")
        print(f"  Backward...")
        loss.backward()
        print(f"  SUCCESS - first step complete!")
        break

    print("ALL OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
