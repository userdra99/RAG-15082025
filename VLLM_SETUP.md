# Custom vLLM Docker Setup Guide

This guide explains how to build and use custom vLLM Docker images to resolve compatibility issues with the specific models used in this RAG project.

## 🚨 Compatibility Issues Addressed

The original setup used `vllm/vllm-openai:latest` which had compatibility issues with:

1. **unsloth/Llama-3.2-3B-Instruct** - Requires specific unsloth dependencies
2. **BAAI/bge-m3** - Requires sentence-transformers and specific model handling
3. **Model-specific flags** - Different models need different vLLM arguments

## 🛠️ Solution Options

### Option 1: Full Custom Build (Recommended)

Uses a clean CUDA base image with all dependencies installed from scratch.

**File**: `Dockerfile.vllm`
**Compose**: `docker-compose.yml`

**Advantages**:
- Complete control over all dependencies
- Optimized for the specific models
- Better performance and stability

**Build Command**:
```bash
./build-vllm.sh
```

### Option 2: Hybrid Approach (Faster Build)

Uses the official vLLM image as base and adds custom dependencies.

**File**: `Dockerfile.vllm.alternative`
**Compose**: `docker-compose.alternative.yml`

**Advantages**:
- Faster build time
- Leverages official vLLM optimizations
- Easier to maintain

**Build Command**:
```bash
docker-compose -f docker-compose.alternative.yml build
```

## 🔧 Building the Custom Images

### Prerequisites

1. **Docker Desktop** with WSL integration enabled
2. **NVIDIA Docker** support (for GPU acceleration)
3. **Sufficient disk space** (10-15GB for build cache)

### Build Process

#### Option 1: Full Custom Build
```bash
# Make the build script executable
chmod +x build-vllm.sh

# Run the build script
./build-vllm.sh
```

#### Option 2: Alternative Build
```bash
# Build using the alternative compose file
docker-compose -f docker-compose.alternative.yml build

# Or build individual services
docker-compose -f docker-compose.alternative.yml build vllm-llm vllm-embedding
```

## 🚀 Running the System

### Using Full Custom Build
```bash
# Start all services
docker-compose up --build

# Or use the start script
./start.sh
```

### Using Alternative Build
```bash
# Start with alternative compose file
docker-compose -f docker-compose.alternative.yml up --build
```

## 📋 Model Configuration

### LLM Service (unsloth/Llama-3.2-3B-Instruct)

**Key Features**:
- Optimized for instruction following
- 3.2B parameters (efficient for most use cases)
- Unsloth optimizations for better performance

**Configuration**:
```yaml
environment:
  - MODEL=unsloth/Llama-3.2-3B-Instruct
command: >
  --model-name unsloth/Llama-3.2-3B-Instruct
  --max-model-len 8192
  --gpu-memory-utilization 0.5
  --enforce-eager
  --trust-remote-code
```

### Embedding Service (BAAI/bge-m3)

**Key Features**:
- Multilingual embedding model
- High-quality semantic representations
- Optimized for retrieval tasks

**Configuration**:
```yaml
environment:
  - MODEL=BAAI/bge-m3
command: >
  --max-model-len 8192
  --gpu-memory-utilization 0.25
  --task embed
  --trust-remote-code
  --dtype float16
```

## 🔍 Troubleshooting

### Build Issues

#### 1. CUDA Version Mismatch
```bash
# Check your CUDA version
nvidia-smi

# If you have CUDA 11.8, modify Dockerfile.vllm:
# Change FROM nvidia/cuda:12.1-devel-ubuntu22.04
# To: FROM nvidia/cuda:11.8-devel-ubuntu22.04
```

#### 2. Memory Issues During Build
```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB+ for building

# Or use build with reduced parallelism
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -f Dockerfile.vllm -t rag-vllm:latest .
```

#### 3. Network Issues During Build
```bash
# Use alternative approach which downloads less
docker-compose -f docker-compose.alternative.yml build
```

### Runtime Issues

#### 1. Model Loading Failures
```bash
# Check model compatibility
docker-compose logs vllm-llm
docker-compose logs vllm-embedding

# Common fixes:
# - Add --trust-remote-code flag
# - Reduce gpu-memory-utilization
# - Use smaller models for testing
```

#### 2. GPU Memory Issues
```bash
# Reduce memory utilization
# In docker-compose.yml, change:
# --gpu-memory-utilization 0.5 to 0.3
# --gpu-memory-utilization 0.25 to 0.15
```

#### 3. Model Download Issues
```bash
# Check Hugging Face access
curl -H "Authorization: Bearer YOUR_TOKEN" https://huggingface.co/api/models/unsloth/Llama-3.2-3B-Instruct

# Or use local model files
# Mount local model directory:
volumes:
  - ./models:/workspace/models
```

## 📊 Performance Optimization

### GPU Memory Allocation
```yaml
# For 8GB GPU
--gpu-memory-utilization 0.4  # LLM
--gpu-memory-utilization 0.2  # Embedding

# For 12GB GPU
--gpu-memory-utilization 0.5  # LLM
--gpu-memory-utilization 0.25 # Embedding

# For 16GB+ GPU
--gpu-memory-utilization 0.6  # LLM
--gpu-memory-utilization 0.3  # Embedding
```

### Model Loading Optimization
```yaml
# Use tensor parallelism for large models
--tensor-parallel-size 2

# Enable quantization for memory efficiency
--quantization awq

# Use CPU offloading for very large models
--cpu-offload
```

## 🔄 Updating Models

### Change LLM Model
```yaml
# In docker-compose.yml
environment:
  - MODEL=your-new-model-name
command: >
  --model-name your-new-model-name
  # ... other args
```

### Change Embedding Model
```yaml
# In docker-compose.yml
environment:
  - MODEL=your-new-embedding-model
command: >
  # ... other args
  --task embed
```

## 📝 Development Notes

### Custom Dependencies Added
- `unsloth==2024.1` - For Llama model optimizations
- `triton==2.1.0` - For GPU kernel optimizations
- `flash-attn==2.5.6` - For attention optimization
- `sentence-transformers==2.2.2` - For embedding models
- `transformers==4.36.2` - For model loading

### Version Pinning
All dependencies are pinned to specific versions to ensure compatibility:
- vLLM: 0.3.3
- PyTorch: 2.1.2
- Transformers: 4.36.2
- CUDA: 12.1

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: `docker-compose logs vllm-llm vllm-embedding`
2. **Verify GPU**: `nvidia-smi`
3. **Test Docker**: `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi`
4. **Check resources**: Ensure sufficient RAM and disk space
5. **Review compatibility**: Check model requirements on Hugging Face

## 🔗 Useful Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [BGE Model Documentation](https://huggingface.co/BAAI/bge-m3)
- [NVIDIA Docker Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 