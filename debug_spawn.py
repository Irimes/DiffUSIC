"""Debug: 定位谁在 spawn 子进程"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Monkey-patch multiprocessing.Process.start to print caller traceback
import multiprocessing
_orig_start = multiprocessing.Process.start

def _debug_start(self):
    import traceback
    print(f"\n!!! multiprocessing.Process.start() 被调用 !!!")
    print(f"    target={self._target}, name={self.name}")
    traceback.print_stack()
    return _orig_start(self)

multiprocessing.Process.start = _debug_start

print(">>> 开始导入 Diffusion 模块...")
from Diffusion.TrainSpectrumConditional import train, eval, compare
from Diffusion.TrainBaselines import train_baselines
print(">>> 导入完成")

print(">>> 创建 BaselineDataset...")
from Diffusion.BaselineDataset import BaselineDataset
ds = BaselineDataset(
    npz_path=r"D:\Files\Academic_Resourses\Codes\diffusion again - 副本 - 副本 - 副本\DOA_Dataset\SmallAngleLarge.npz",
    split='train',
    test_ratio=0.001,
    seed=42,
)
print(f">>> Dataset 创建完成, len={len(ds)}")

print(">>> 创建 DataLoader (num_workers=0)...")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
print(">>> DataLoader 创建完成")

print(">>> 迭代第一个 batch...")
for batch in loader:
    print(f">>> 第一个 batch OK, keys={list(batch.keys())}")
    break

print(">>> 全部完成, 无子进程被 spawn")
