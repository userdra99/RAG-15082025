# RAG Document Assistant - Llama3.2, Jina Embeddings V4, LlamaIndex

A production-ready Retrieval-Augmented Generation (RAG) system using Jina Embeddings V4 and LlamaIndex for document Q&A with multilingual support.

## 🚀 Features

- **Jina Embeddings V4**: 8192 token context, 2048D vectors, multilingual support
- **LlamaIndex Integration**: OpenAI-compatible API with vLLM backend
- **Document Processing**: PDF, DOCX, Excel support via Docling
- **Vector Database**: Qdrant with hybrid search (dense + sparse)
- **Web Interface**: Streamlit-based UI for document upload and Q&A
- **Docker Ready**: Complete containerized deployment
- **Memory Optimized**: Works with 24GB+ GPUs

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (24GB recommended)
- **Docker**: Latest Docker and Docker Compose
- **CUDA**: CUDA 12.1+ drivers
- **Storage**: 10GB+ free space for models

## 🛠️ Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd rag-zero2
```

### 2. Start Services
```bash
# Full deployment with jina-embeddings-v4
docker-compose up --build

# Or use memory-optimized version
docker-compose -f docker-compose.memory-optimized.yml up --build
```

### 3. Access Applications
- **Streamlit App**: http://localhost:8501
- **API Endpoints**: http://localhost:8000/v1/
- **Qdrant UI**: http://localhost:6333/dashboard

## 📁 Project Structure

```
├── app/
│   ├── main.py                 # Streamlit application
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile             # App container
├── data/                      # Document storage
├── nginx/
│   └── nginx.conf            # Load balancer config
├── docker-compose.yml        # Main deployment
├── docker-compose.memory-optimized.yml  # Memory efficient
├── docker-compose.fallback.yml          # BAAI/bge-m3 fallback
├── test_jina_embeddings.py   # Model testing
├── troubleshoot.sh          # Diagnostic script
└── README.md
```

## 🔧 Configuration

### Environment Variables
```bash
# GPU Configuration
NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0

# Model Settings
EMBEDDING_MODEL=jina-embeddings-v4
OPENAI_API_BASE=http://nginx:80/v1
```

### Memory Optimization
```bash
# For 24GB GPUs
--gpu-memory-utilization 0.25
--max-model-len 8192
```

## 📝 Usage

### 1. Upload Documents
1. Open http://localhost:8501
2. Use sidebar to upload PDF/DOCX/XLSX files
3. Click "Process Documents" to index

### 2. Ask Questions
1. Use the main interface to ask questions
2. View source documents and scores
3. Track conversation history

### 3. API Usage
```bash
# Test embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": ["Your text here"],
    "encoding_format": "float"
  }'
```

## 🔍 Troubleshooting

### Common Issues
- **Memory errors**: Use `docker-compose.fallback.yml`
- **Container conflicts**: Run `./troubleshoot.sh`
- **GPU not detected**: Check nvidia-smi

### Performance Tuning
- **Context length**: 4096-8192 tokens
- **Batch size**: 5-10 documents
- **GPU utilization**: 20-30%

## 📊 Performance Metrics

 < /dev/null |  Model | Context | Dimensions | Memory | Speed |
|-------|---------|------------|--------|-------|
| jina-embeddings-v4 | 8192 | 2048 | ~6GB | 50ms |
| BAAI/bge-m3 | 8192 | 1024 | ~2GB | 20ms |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Jina AI](https://jina.ai/) for embeddings model
- [vLLM](https://vllm.ai/) for efficient inference
- [LlamaIndex](https://llamaindex.ai/) for RAG framework
- [Qdrant](https://qdrant.tech/) for vector database
