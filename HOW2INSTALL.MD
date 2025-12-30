# 🚀 MMDetection 3.x Windows Installation Guide (Update 2025)

Hướng dẫn chi tiết cách cài đặt MMDetection trên **Windows** sử dụng **CUDA 12.1** và **PyTorch 2.3.1**, khắc phục triệt để lỗi `DLL load failed` và xung đột nhân `_ext`.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![MMCV](https://img.shields.io/badge/MMCV-2.2.0-blue)](https://mmcv.readthedocs.io/)
[![MMDet](https://img.shields.io/badge/MMDet-3.3.0-orange)](https://mmdetection.readthedocs.io/)

---

## 🛠️ Quy trình cài đặt 17 bước chuẩn

### 1. Khởi tạo môi trường sạch
Làm sạch môi trường cũ để tránh xung đột `mmcv-full`:
```bash
# Xóa môi trường cũ
conda deactivate
conda remove -n mmdet --all -y

# Tạo môi trường mới (Chỉ dùng Python 3.10 để đạt độ tương thích cao nhất)
conda create -n mmdet python=3.10 -y
conda activate mmdet

### 2. Cài đặt nhân PyTorch & CUDA Runtime
Sử dụng bộ đôi ổn định nhất cho Windows: Torch 2.3.1 + CUDA 12.1.
# Kiểm tra CUDA Driver trước
nvidia-smi

# Cài đặt PyTorch và CUDA toolkit tương ứng
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Kiểm tra kết quả cài đặt
python -c "import torch;print(torch.__version__);print(torch.version.cuda)"
# Kết quả mong muốn: 2.3.1 và 12.1
