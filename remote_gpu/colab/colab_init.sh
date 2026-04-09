#!/bin/bash
# Colab에 SSH 접속 후 실행하는 초기 환경 세팅 스크립트
# 사용법: ssh로 접속 후 이 스크립트를 실행하거나,
#         서버에서 직접: ssh ... 'bash -s' < remote_gpu/colab/colab_init.sh

echo "=== MarsRefSR Colab 환경 세팅 ==="

# GPU 확인
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Python 패키지 설치
echo ""
echo "--- 패키지 설치 ---"
pip install -q torch torchvision timm einops lpips kornia \
    numpy Pillow rasterio spiceypy scikit-image

# 작업 디렉토리 생성
mkdir -p ~/marsrefsr/{data,checkpoints,results}

echo ""
echo "--- 준비 완료 ---"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "데이터 전송: 서버에서 실행"
echo "  scp -o ProxyCommand=... remote_gpu/kaggle/train.py root@HOST:~/marsrefsr/"
echo ""
echo "학습 시작:"
echo "  cd ~/marsrefsr && python train.py"
