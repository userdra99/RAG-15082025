# 🤖 Advanced RAG System with Llama-3.3-70B on Dual RTX 5090s

A cutting-edge Retrieval-Augmented Generation (RAG) system featuring **Llama-3.3-70B-Instruct-AWQ** running on dual RTX 5090 GPUs with vLLM, LlamaIndex, and Qdrant.

## 🚀 Features

- **🦙 Large Language Model**: Llama-3.3-70B-Instruct-AWQ (dual RTX 5090 deployment)
- **🧠 Embedding Service**: BGE-M3 (1024-dimensional vectors) for superior semantic understanding
- **📄 Document Processing**: PDF, DOCX, XLSX support with advanced Docling integration
- **🗄️ Vector Database**: Qdrant for lightning-fast similarity search
- **🌐 Web Interface**: Modern Flask-based UI for document management and querying
- **⚡ GPU Acceleration**: Optimized for RTX 5090 Blackwell architecture
- **🐳 Containerized**: Complete Docker deployment with multi-GPU support

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Flask RAG App     │    │  Llama-3.3-70B-AWQ │    │    BGE-M3 Embed    │
│    (Port 5000)      │    │  Dual RTX 5090     │    │    (Port 8002)      │
│                     │    │    (Port 8001)      │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                        ┌─────────────────────┐
                        │   Nginx Load Balancer│
                        │     (Port 8000)      │
                        └─────────────────────┘
                                     │
                        ┌─────────────────────┐
                        │    Qdrant Vector DB │
                        │     (Port 6333)     │
                        └─────────────────────┘
```

## 📋 System Requirements

### Hardware
- **GPUs**: 2x NVIDIA RTX 5090 (32GB VRAM each)
- **CPU**: High-core count CPU (16+ cores recommended)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ NVMe SSD

### Software
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **CUDA**: 13.0+
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **NCCL**: 2.26.5+ (2.27.7 recommended for RTX 5090)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/userdra99/RAG-15082025.git
cd RAG-15082025
```

### 2. Environment Setup
```bash
# Set your Hugging Face token
export HUGGING_FACE_HUB_TOKEN=your_token_here

# Verify RTX 5090 detection
nvidia-smi
```

### 3. Deploy Llama-3.3-70B System
```bash
# Deploy complete system with dual RTX 5090 support
docker compose -f docker-compose.llama33-70b.yml up -d

# Monitor deployment progress
docker logs vllm-llama33-70b-awq --follow
```

### 4. Access the Application
- **Web Interface**: http://localhost:5000
- **LLM API**: http://localhost:8001/v1/models
- **Embedding API**: http://localhost:8002/v1/models
- **Health Check**: http://localhost:5000/health

## 📁 Project Structure

```
rag-zero5/
├── app/                           # Flask RAG application
│   ├── main.py                    # Main application logic
│   ├── Dockerfile.bge-m3          # BGE-M3 optimized container
│   ├── requirements.bge-m3.txt    # Python dependencies
│   ├── templates/index.html       # Web interface
│   └── data/                      # Document storage
├── docker-compose.llama33-70b.yml # Llama-3.3-70B deployment
├── Dockerfile.vllm_rtx5090       # RTX 5090 optimized vLLM
├── Dockerfile.vllm_simple        # Standard vLLM container
├── deploy_llama33_70b.sh         # Deployment script
├── deploy_native_vllm.sh         # Native installation option
├── RTX_5090_Dual_GPU_Guide.md    # RTX 5090 compatibility guide
├── nginx/nginx.conf               # Load balancer config
└── README.md                      # This file
```

## ⚙️ Configuration

### Llama-3.3-70B Service
- **Model**: `casperhansen/llama-3.3-70b-instruct-awq`
- **Context Length**: 4096 tokens
- **Quantization**: AWQ (Activation-aware Weight Quantization)
- **Tensor Parallelism**: 2 (dual GPU)
- **GPU Memory Utilization**: 80% per GPU
- **Total Model Size**: ~37GB (distributed across GPUs)

### BGE-M3 Embedding Service
- **Model**: `BAAI/bge-m3`
- **Vector Dimensions**: 1024
- **Context Window**: 8192 tokens
- **GPU Assignment**: RTX 5090 #2
- **Memory Usage**: ~4GB VRAM

### RTX 5090 Optimizations
- **CUDA Architecture**: 12.0 (Blackwell)
- **NCCL Version**: 2.27.7 (multi-GPU support)
- **Flash Attention**: v2 backend
- **Memory Management**: Expandable segments

## 🔧 Advanced Deployment Options

### Option 1: Docker Deployment (Recommended)
```bash
# Full system deployment
docker compose -f docker-compose.llama33-70b.yml up -d

# Scale specific services
docker compose -f docker-compose.llama33-70b.yml up -d --scale vllm-llm=1
```

### Option 2: Native Installation
```bash
# Use the comprehensive installation script
./deploy_native_vllm.sh

# Or follow the RTX 5090 guide
cat RTX_5090_Dual_GPU_Guide.md
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGING_FACE_HUB_TOKEN` | HF token for model access | Required |
| `NVIDIA_VISIBLE_DEVICES` | GPU device IDs | `0,1` |
| `CUDA_VISIBLE_DEVICES` | CUDA device mapping | `0,1` |
| `NCCL_DEBUG` | NCCL logging level | `INFO` |
| `TORCH_CUDA_ARCH_LIST` | Target CUDA architecture | `12.0` |

## 📖 Usage Guide

### 1. System Initialization
```bash
# Initialize the RAG system
curl -X POST http://localhost:5000/initialize

# Verify services
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
```

### 2. Document Upload & Processing
1. Access web interface at http://localhost:5000
2. Upload PDF, DOCX, or XLSX documents
3. Click "Process Documents" to create embeddings
4. Wait for indexing completion (progress shown)

### 3. AI-Powered Querying
1. Enter questions in the query interface
2. Get contextual answers from your documents
3. Review source citations and confidence scores

### API Examples
```bash
# Test LLM endpoint
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "casperhansen/llama-3.3-70b-instruct-awq", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}'

# Test embedding endpoint
curl -X POST http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-m3", "input": "Test document", "encoding_format": "float"}'
```

## 🐛 Troubleshooting

### RTX 5090 Specific Issues

**NCCL Communication Errors**
```bash
# Verify NCCL version
python -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')"

# Should be 2.26.5 or higher
pip install -U nvidia-nccl-cu12>=2.26.5
```

**Memory Issues**
```bash
# Check GPU memory usage
nvidia-smi

# Reduce memory utilization if needed
--gpu-memory-utilization 0.75  # In docker-compose.yml
```

**Architecture Detection**
```bash
# Verify CUDA compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Should show: 12.0, 12.0 for RTX 5090s
```

### Debug Commands
```bash
# Service logs
docker logs vllm-llama33-70b-awq
docker logs vllm-embedding-bge-m3
docker logs rag-app-llama33-70b

# GPU monitoring
watch -n 1 nvidia-smi

# Container status
docker ps
docker compose -f docker-compose.llama33-70b.yml ps
```

## 📊 Performance Benchmarks

### Model Performance
- **Inference Speed**: ~28 seconds for complex responses (4096 tokens)
- **Throughput**: 8.00x concurrency capacity
- **Memory Efficiency**: 18.55 GiB per GPU (37GB total)
- **Context Processing**: Up to 4096 tokens

### System Metrics
- **Document Processing**: ~2-5 minutes per large PDF
- **Embedding Generation**: 1024-dimensional vectors
- **Query Response**: <30 seconds end-to-end
- **Concurrent Users**: 8+ simultaneous queries

## 🔄 Migration & Upgrades

### From Previous Versions
```bash
# Backup existing data
docker compose down
cp -r qdrant_storage qdrant_storage.backup

# Deploy new version
docker compose -f docker-compose.llama33-70b.yml up -d
```

### Model Upgrades
- Supports any AWQ-quantized model compatible with vLLM
- Memory requirements scale with model size
- RTX 5090 architecture supports models up to ~70B parameters

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Test on RTX 5090 hardware if possible
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for Llama-3.3-70B model architecture
- **vLLM Team** for efficient GPU inference
- **LlamaIndex** for RAG framework
- **BAAI** for BGE-M3 embedding model
- **Qdrant** for vector database technology
- **NVIDIA** for RTX 5090 Blackwell architecture

---

## 🎯 Current Status: ✅ FULLY OPERATIONAL

**Latest Achievement**: Successfully deployed Llama-3.3-70B-Instruct-AWQ on dual RTX 5090 GPUs with full RAG functionality.

### ✅ Verified Components
- ✅ Dual RTX 5090 GPU utilization
- ✅ NCCL 2.27.7 multi-GPU communication
- ✅ Llama-3.3-70B-AWQ inference
- ✅ BGE-M3 embedding generation
- ✅ Complete RAG pipeline
- ✅ Web interface and APIs
- ✅ Docker containerization

### 🚀 Ready for Production
This system represents the cutting edge of open-source RAG deployment, leveraging the latest RTX 5090 hardware for unprecedented performance in document understanding and generation.

**Deployment Date**: August 2025  
**Hardware**: Dual RTX 5090 (64GB VRAM total)  
**Model**: Llama-3.3-70B-Instruct-AWQ  
**Status**: Production Ready 🚀