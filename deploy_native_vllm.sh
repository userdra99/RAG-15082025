#!/bin/bash

# Native vLLM Deployment Script for RTX 5090 Blackwell
# Uses latest vLLM source with full RTX 5090 support

set -e

echo "🚀 Native vLLM Installation for RTX 5090 Blackwell"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "/home/dra/vllm" ]; then
    print_error "vLLM source directory not found. Please run the clone first."
    exit 1
fi

cd /home/dra/vllm

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3.12 -m venv venv
fi

print_status "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Verify PyTorch is installed
if ! python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>/dev/null; then
    print_status "Installing PyTorch with CUDA 12.8..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

print_status "Running use_existing_torch.py..."
python use_existing_torch.py

print_status "Installing build requirements..."
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/common.txt

print_status "Setting CUDA environment for PyTorch compatibility..."
# PyTorch nightly with CUDA 12.9 is compatible with system CUDA 13.0
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

print_status "CUDA Environment:"
echo "PyTorch CUDA: $TORCH_CUDA_VERSION"
echo "System CUDA_HOME: $CUDA_HOME"
echo "nvcc version: $($CUDA_HOME/bin/nvcc --version | grep release)"

print_status "Installing vLLM from source (optimized for RTX 5090)..."
# Build with explicit CUDA architecture for RTX 5090 Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16
python -m pip install -e . --no-build-isolation

print_status "🔧 CRITICAL: Upgrading NCCL for multi-RTX 5090 support..."
print_status "Installing NCCL 2.26.5+ (required for TP>1 with RTX 5090)"
python -m pip install -U nvidia-nccl-cu12

print_status "Installing additional dependencies..."
python -m pip install ninja

print_status "Installing xformers for enhanced performance..."
python -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

print_success "Native vLLM installation complete!"

# Test the installation
print_status "Testing vLLM installation..."
if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
    print_success "vLLM successfully installed!"
else
    print_error "vLLM installation verification failed"
    exit 1
fi

print_status "Starting Llama-3.3-70B with native vLLM..."

# Set environment variables for optimal performance
export VLLM_FLASH_ATTN_VERSION=2
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export VLLM_TEST_FORCE_FP8_MARLIN=1

# Start the model server
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3.3-70b-instruct-awq \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8001 \
    --quantization awq \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager