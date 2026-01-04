# MMDetection 3.3.0 Installation Guide (LINUX + CUDA 12.1): H200 GPU SERVER

Tài liệu này mô tả **quy trình cài đặt MMDetection 3.3.0 trên Windows** với CUDA 12.1, PyTorch 2.3.x và MMCV 2.2.0.  
Quy trình đã được **kiểm chứng thành công**, bao gồm kiểm tra nhân `_ext` để tránh lỗi `DLL load failed`.

Nguyên nhân: 
https://mmcv.readthedocs.io/en/latest/get_started/installation.html 
![Yêu cầu CUDA và Pytorch](https://github.com/thinhdoanvu/MMCV-MMDET-2025/blob/main/imgs/requirement%20for%20CUDA%20and%20Pytorch.png)
---

## 0. Cài đặt Python 3.10 trong trường hợp GPU SERVER ĐÃ CÓ PYTHON 3.12

1️⃣ Cài công cụ cần thiết

```bash
sudo apt update
sudo apt install software-properties-common -y
```

2️⃣ Thêm PPA deadsnakes
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
```

3️⃣ Cài Python 3.10 đầy đủ
```bash
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

4️⃣ Kiểm tra
```bash
python3.10 --version
```
✔️ Kết quả
```bash
Python 3.10.x
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
source mmdet/bin/activate
```

---

## 3. Kiểm tra CUDA

```bash
nvidia-smi
```

✔️ Ví dụ:
```bash
NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0
```

⚠️ CUDA runtime cao hơn **không ảnh hưởng**, miễn là PyTorch dùng **CUDA 12.1**.

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
dir %CONDA_PREFIX%\Lib\site-packages\mmengine\__init__.py
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
mmcv_maximum_version = '2.1.0'
```

Thành:
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
## Kết quả Demo

![Kết quả chạy thử](https://github.com/thinhdoanvu/MMCV-MMDET-2025/blob/main/imgs/demo.jpg)

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
