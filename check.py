import torch

try:
    import mmcv
    from mmcv.ops import roi_align
    import mmdet
    print("---------------------------------------")
    print("THANH CONG! Nhân _ext đã được nạp.")
    print(f"Torch version: {torch.__version__}")
    print(f"MMCV version: {mmcv.__version__}")
    print(f"MMDet version: {mmdet.__version__}")
except ImportError as e:
    print(f"Van loi DLL: {e}")