# MMDetection 3.3.0 Installation Guide (Windows + CUDA 12.1)

Tài liệu này mô tả **quy trình cài đặt MMDetection 3.3.0 trên Windows** với CUDA 12.1, PyTorch 2.3.x và MMCV 2.2.0.  
Quy trình đã được **kiểm chứng thành công**, bao gồm kiểm tra nhân `_ext` để tránh lỗi `DLL load failed`.

---

## 0. Xóa môi trường cũ (nếu có)

```bash
deactivate
conda remove -n mmdet --all -y
```

---

## 1. Tạo môi trường Conda (Python 3.10)

> ⚠️ **Chỉ dùng Python 3.10**  
> Python 3.11+ **không tương thích** với MMCV / MMDetection tại thời điểm hiện tại.

```bash
conda create -n mmdet python=3.10 -y
```

---

## 2. Kích hoạt môi trường

```bash
activate mmdet
```

---

## 3. Kiểm tra CUDA

```bash
nvidia-smi
```

Ví dụ:
```
CUDA Version: 12.6
```

> CUDA runtime cao hơn **không ảnh hưởng**, miễn là PyTorch dùng **CUDA 12.1**.

---

## 4. Cài đặt PyTorch + CUDA 12.1

MMCV 2.2.0 chỉ hỗ trợ:
- Windows
- CUDA 12.1
- PyTorch 2.3.x

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 5. Kiểm tra PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Kết quả mong đợi:
```
2.3.1
12.1
```

---

## 6. Cài OpenMIM

```bash
pip install -U openmim
```

---

## 7. Cài MMEngine

```bash
mim install mmengine
```

---

## 8. Kiểm tra MMEngine

File:
```
%CONDA_PREFIX%\Lib\site-packages\mmengine\__init__.py
```

```python
__version__ = '0.10.7'
```

✔️ Đạt yêu cầu (MMDet 3.x yêu cầu `<1.0.0`)

---

## 9. Cài MMCV 2.2.0

```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

---

## 10. Kiểm tra MMCV

```bash
python -c "import mmcv; print(mmcv.__version__)"
```

Kết quả:
```
2.2.0
```

---

## 11. Clone MMDetection 3.x

```bash
git clone https://github.com/open-mmlab/mmdetection.git
hoặc từ đây: https://github.com/thinhdoanvu/MMCV-MMDET-2025.git
```

---

## 12. Sửa giới hạn phiên bản MMCV

File:
```
mmdetection/mmdet/__init__.py
```

Sửa:
```python
mmcv_maximum_version = '2.3.0'
```

---

## 13. Cài MMDetection 3.3.0

```bash
cd mmdetection
pip install --no-build-isolation -v .
```

---

## 14. Kiểm tra nhân `_ext` (Bước quan trọng)

```bash
dir %CONDA_PREFIX%\Lib\site-packages\mmcv\_ext*
```

Kết quả mong đợi:
```
_ext.cp310-win_amd64.pyd
```

✔️ Nhân đã được build đúng

---

## 15. Kiểm tra import MMCV

```bash
python -c "import torch, mmcv; print(torch.__version__); print(mmcv.__version__)"
```
Kết quả mong đợi:
```
Torch: 2.3.1
MMCV: 2.2.0
```

---

## 16. Kiểm tra MMDetection

File `check.py`:

```python
import torch

try:
    import mmcv
    from mmcv.ops import roi_align
    import mmdet
    print("---------------------------------------")
    print("THÀNH CÔNG! Nhân _ext đã được nạp.")
    print(f"Torch version: {torch.__version__}")
    print(f"MMCV version: {mmcv.__version__}")
    print(f"MMDet version: {mmdet.__version__}")
except ImportError as e:
    print(f"Lỗi DLL: {e}")
```

Kết quả:
```
THÀNH CÔNG! Nhân _ext đã được nạp.
Torch version: 2.3.1
MMCV version: 2.2.0
MMDet version: 3.3.0
```

---

## 17. Demo inference

```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

### Ghi chú

- Cảnh báo `unexpected key in state_dict` là **bình thường**
- Cảnh báo registry / visualization **không ảnh hưởng inference**
- Kết quả được lưu trong thư mục `outputs/`

---

## ✅ Kết luận

Cấu hình này đã được xác nhận:
- Không lỗi DLL
- Load được MMCV CUDA ops
- Chạy inference MMDetection 3.3.0 thành công trên Windows

Phù hợp cho:
- Nghiên cứu
- Fine-tuning
- Demo & triển khai nội bộ
