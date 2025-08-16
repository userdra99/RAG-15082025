# Dual RTX 5090 Compatibility Guide for vLLM
*Complete reference for deploying large language models on dual RTX 5090 GPUs*

## 🚀 Hardware Overview

**RTX 5090 Specifications:**
- **Architecture**: Blackwell (CUDA Compute Capability 12.0)
- **Memory**: 32GB GDDR7 per GPU (64GB total)
- **Memory Bandwidth**: 1792 GB/s per GPU
- **CUDA Cores**: 21,760 per GPU
- **Tensor Cores**: 5th Gen with FP8 support

## 🔧 Critical Compatibility Requirements

### 1. CUDA Architecture Targeting
```bash
# Essential: RTX 5090 is Blackwell (12.0), NOT Ada Lovelace (8.9)
export TORCH_CUDA_ARCH_LIST="12.0"
```

### 2. NCCL Version Requirements
```bash
# CRITICAL: NCCL 2.26.5+ required for multi-RTX 5090 support
# Default PyTorch 2.7.0 ships with NCCL 2.26.2 which FAILS
pip install -U nvidia-nccl-cu12>=2.26.5

# Recommended: Use NCCL 2.27.7 for optimal stability
pip install nvidia-nccl-cu12==2.27.7
```

### 3. PyTorch Compatibility Matrix
```bash
# Option 1: Stable PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Option 2: Nightly PyTorch with CUDA 12.9 (better RTX 5090 support)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
```

## 🐳 Docker Configuration

### Dockerfile for RTX 5090 (Dockerfile.vllm_rtx5090)
```dockerfile
# vLLM container optimized for RTX 5090 with NCCL 2.27.7
FROM vllm/vllm-openai:v0.10.0

# Update NCCL to 2.27.7 for RTX 5090 multi-GPU support
RUN pip install --upgrade nvidia-nccl-cu12==2.27.7

# Create optimized startup script for RTX 5090
RUN cat <<'EOF' > /start.sh
#!/bin/bash
echo "Starting vLLM server optimized for RTX 5090..."
echo "Model: ${MODEL}"
echo "Port: ${PORT:-8000}"

# RTX 5090 NCCL optimizations
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=2
export NCCL_MAX_NRINGS=4
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=SYS

# Start vLLM server
exec /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT:-8000}" \
    "$@"
EOF
RUN chmod +x /start.sh

EXPOSE 8000
ENTRYPOINT ["bash", "/start.sh"]
```

### Docker Compose Configuration
```yaml
services:
  vllm-llm:
    build:
      context: .
      dockerfile: Dockerfile.vllm_rtx5090
    container_name: vllm-llama33-70b-awq
    ports:
      - "8001:8000"
    environment:
      # GPU Configuration
      - NVIDIA_VISIBLE_DEVICES=0,1
      - CUDA_VISIBLE_DEVICES=0,1
      - MODEL=casperhansen/llama-3.3-70b-instruct-awq
      - PORT=8000
      
      # vLLM Optimizations
      - VLLM_LOGGING_LEVEL=INFO
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - OMP_NUM_THREADS=4
      - VLLM_USE_V1=0
      - CUDA_LAUNCH_BLOCKING=1
      
      # NCCL Configuration
      - NCCL_DEBUG=INFO
      
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model casperhansen/llama-3.3-70b-instruct-awq
      --max-model-len 4096
      --gpu-memory-utilization 0.80
      --tensor-parallel-size 2
      --host 0.0.0.0
      --trust-remote-code
      --quantization awq
      --enforce-eager
      --disable-custom-all-reduce
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
    shm_size: 32gb
```

## 💻 Native Installation Guide

### Prerequisites
```bash
# Ensure CUDA 13.0+ is installed
nvcc --version

# Verify RTX 5090 detection
nvidia-smi

# Check GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv
# Should show: 12.0, 12.0
```

### Installation Script
```bash
#!/bin/bash
# Native vLLM Installation for RTX 5090 Blackwell

# 1. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with RTX 5090 support
python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129

# 3. Clone and prepare vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 4. Install build dependencies
python -m pip install -r requirements/build.txt
python -m pip install -r requirements/common.txt

# 5. Configure environment for RTX 5090
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 6. Build vLLM from source
python -m pip install -e . --no-build-isolation

# 7. CRITICAL: Upgrade NCCL for multi-RTX 5090 support
python -m pip install -U nvidia-nccl-cu12==2.27.7

# 8. Install performance optimizations
python -m pip install ninja
python -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

## ⚙️ Optimal Launch Parameters

### For Large Models (70B+)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3.3-70b-instruct-awq \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --host 0.0.0.0 \
    --port 8001 \
    --quantization awq \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager
```

### Environment Variables
```bash
# Performance optimizations
export VLLM_FLASH_ATTN_VERSION=2
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export VLLM_TEST_FORCE_FP8_MARLIN=1

# NCCL optimizations for RTX 5090
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=2
export NCCL_MAX_NRINGS=4
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=SYS
```

## 🐛 Common Issues & Solutions

### Issue 1: NCCL "unhandled cuda error"
**Symptoms**: Multi-GPU communication fails during tensor parallel operations
```
RuntimeError: NCCL error in: ../csrc/distributed/ProcessGroupNCCL.cpp:1970, unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

**Solution**: Upgrade NCCL to 2.26.5+
```bash
pip install -U nvidia-nccl-cu12>=2.26.5
```

### Issue 2: "namespace 'cub' has no member" during native build
**Symptoms**: CUDA 13.0 compatibility errors during compilation
```
error: namespace 'cub' has no member named 'Sum'
```

**Solution**: Use PyTorch nightly with CUDA 12.9
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
```

### Issue 3: Illegal memory access on single GPU
**Symptoms**: Model exceeds single GPU memory
```
RuntimeError: CUDA out of memory. Tried to allocate 30.63 GiB (GPU 0; 31.34 GiB total capacity)
```

**Solution**: Ensure tensor parallelism is enabled
```bash
--tensor-parallel-size 2
```

### Issue 4: Wrong GPU architecture detection
**Symptoms**: Model builds for wrong CUDA compute capability
```
WARNING: The CUDA version that was used to compile PyTorch (12.1) does not match the CUDA version used to compile vLLM (13.0)
```

**Solution**: Explicitly set CUDA architecture
```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

## 📊 Memory Planning

### RTX 5090 Memory Layout (per GPU)
- **Total VRAM**: 32GB
- **Recommended Usage**: 80% = 25.6GB per GPU
- **Dual GPU Total**: 64GB available
- **Reserved for System**: ~2GB per GPU

### Model Size Calculations
| Model | Parameters | AWQ Size | Dual RTX 5090 Fit |
|-------|------------|----------|-------------------|
| Llama-3.3-70B | 70B | ~37GB | ✅ Yes |
| Llama-3.1-405B | 405B | ~200GB | ❌ No |
| Mixtral-8x22B | 141B | ~75GB | ❌ Tight |

## 🔍 Verification Commands

### Check Installation
```bash
# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check NCCL version
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')"

# Verify GPU detection
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Check compute capability
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_capability(i)}') for i in range(torch.cuda.device_count())]"
```

### Monitor GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor NCCL communication
export NCCL_DEBUG=INFO
# Check logs for NCCL Ring/Tree topology
```

## 📚 References

- **vLLM Documentation**: https://github.com/vllm-project/vllm
- **NVIDIA RTX 5090 Specs**: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/
- **NCCL Compatibility**: https://github.com/NVIDIA/nccl
- **PyTorch CUDA Support**: https://pytorch.org/get-started/locally/

---

## ✅ Quick Deployment Checklist

- [ ] Verify RTX 5090s detected with 12.0 compute capability
- [ ] Install NCCL 2.26.5+ (recommend 2.27.7)
- [ ] Set `TORCH_CUDA_ARCH_LIST="12.0"`
- [ ] Use tensor parallelism for models >30GB
- [ ] Configure NCCL environment variables
- [ ] Test with AWQ quantized models first
- [ ] Monitor GPU memory usage during deployment

**Last Updated**: August 2025
**Tested Configuration**: Dual RTX 5090 + vLLM v0.10.0 + NCCL 2.27.7