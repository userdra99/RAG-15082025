# BGE-M3 Migration Guide

This guide helps you migrate from **Nomic Embed Text v1 (768-dim)** to **BGE-M3 (1024-dim)** for production testing.

## 🎯 Migration Overview

### Current Stack
```
Nomic Embed Text v1 (768-dim)
      ↓
OpenAI-like Embedding Client
      ↓
vLLM Serving (Port 8002)
      ↓
Nginx Proxy (Port 8000)
      ↓
LlamaIndex RAG Framework
      ↓
Qdrant Vector DB (768-dim vectors)
```

### Target Stack
```
BGE-M3 (1024-dim)
      ↓
OpenAI-like Embedding Client
      ↓
vLLM Serving (Port 8002)
      ↓
Nginx Proxy (Port 8000)
      ↓
LlamaIndex RAG Framework
      ↓
Qdrant Vector DB (1024-dim vectors)
```

## 📋 Prerequisites

1. **GPU Memory**: Ensure you have sufficient GPU memory (BGE-M3 requires ~2GB more than Nomic)
2. **Hugging Face Token**: Set `HUGGING_FACE_HUB_TOKEN` environment variable
3. **Docker & Docker Compose**: Ensure both are installed and working
4. **Backup**: Create backup of existing data (automated in deployment script)

## 🚀 Quick Migration (Automated)

### Option 1: One-Command Migration
```bash
# Set your Hugging Face token
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Run automated migration
./deploy_bge_m3.sh
```

### Option 2: Manual Step-by-Step Migration

#### Step 1: Backup Current System
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Backup configurations
cp docker-compose.yml "$BACKUP_DIR/"
cp app/requirements.txt "$BACKUP_DIR/"
```

#### Step 2: Prepare Collection Migration
```bash
# Run migration preparation
docker-compose exec app python migrate_collection.py
```

#### Step 3: Deploy BGE-M3 Configuration
```bash
# Stop current services
docker-compose down

# Build new images
docker-compose -f docker-compose.bge-m3.yml build --no-cache

# Start with BGE-M3
export EMBEDDING_MODEL="BAAI/bge-m3"
docker-compose -f docker-compose.bge-m3.yml up -d
```

#### Step 4: Validate Deployment
```bash
# Run comprehensive tests
python test_bge_m3.py
```

## 🔧 Configuration Files

### Key Changes Made

1. **docker-compose.bge-m3.yml**: Updated embedding service to use BGE-M3
2. **requirements.bge-m3.txt**: Pinned compatible LlamaIndex versions
3. **app/main.py**: Added dynamic vector dimension detection
4. **Dockerfile.bge-m3**: Updated app container for BGE-M3 dependencies

### Environment Variables
```bash
# Required for BGE-M3
export EMBEDDING_MODEL="BAAI/bge-m3"
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Optional tuning
export GPU_MEMORY_UTILIZATION="0.7"  # Increased from 0.5
```

## 📊 Testing & Validation

### Automated Testing
```bash
# Run full test suite
python test_bge_m3.py

# Check specific components
curl http://localhost:8002/v1/models  # Embedding service
curl http://localhost:5000/health     # Main application
curl http://localhost:6333/dashboard  # Qdrant dashboard
```

### Manual Validation
1. **Model Loading**: Verify BGE-M3 appears in `/v1/models` endpoint
2. **Embedding Dimensions**: Test embeddings return 1024-dimensional vectors
3. **Collection Config**: Confirm Qdrant collection uses 1024 dimensions
4. **End-to-End**: Upload documents and test query functionality

## 🔍 Performance Monitoring

### GPU Memory Usage
```bash
# Monitor GPU usage
docker exec vllm-embedding-cuda-dl nvidia-smi

# Expected memory usage: ~6-8GB for BGE-M3 (vs ~4-6GB for Nomic)
```

### Service Logs
```bash
# Check embedding service logs
docker-compose -f docker-compose.bge-m3.yml logs vllm-embedding

# Monitor for successful model loading
docker-compose -f docker-compose.bge-m3.yml logs -f vllm-embedding | grep "bge-m3"
```

## 🔄 Document Reprocessing

After migration, you'll need to reprocess documents:

1. **Access Application**: http://localhost:5000
2. **Initialize System**: Click "Initialize System" button
3. **Upload Documents**: Re-upload your document collection
4. **Process Documents**: Click "Process Documents" to create embeddings
5. **Test Queries**: Verify search functionality with new embeddings

## ⚠️ Troubleshooting

### Common Issues

#### BGE-M3 Model Not Loading
```bash
# Check if token is set correctly
echo $HUGGING_FACE_HUB_TOKEN

# Manual model download
docker exec vllm-embedding-cuda-dl python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3')"
```

#### GPU Memory Issues
```bash
# Reduce memory utilization
export GPU_MEMORY_UTILIZATION="0.6"  # Instead of 0.7

# Check GPU memory
nvidia-smi
```

#### Dimension Mismatch Errors
```bash
# Clear existing collection and restart
docker-compose exec qdrant curl -X DELETE http://localhost:6333/collections/documents
docker-compose -f docker-compose.bge-m3.yml restart app
```

### Rollback Procedure
```bash
# Stop BGE-M3 services
docker-compose -f docker-compose.bge-m3.yml down

# Restore original configuration
cp backups/BACKUP_DATE/docker-compose.yml .
cp backups/BACKUP_DATE/requirements.txt app/

# Restart original system
docker-compose up -d
```

## 📈 Performance Comparison

### Expected Improvements with BGE-M3
- **Retrieval Quality**: Better semantic understanding
- **Multi-language Support**: Enhanced cross-lingual capabilities  
- **Context Length**: Improved handling of longer documents
- **Embedding Quality**: Higher dimensional representations (1024 vs 768)

### Trade-offs
- **Memory Usage**: +30-40% GPU memory requirement
- **Latency**: Slightly higher embedding generation time
- **Storage**: +33% vector storage requirements (1024 vs 768 dims)

## 🔗 Service Endpoints

After successful deployment:

- **Main Application**: http://localhost:5000
- **LLM Service**: http://localhost:8001  
- **Embedding Service**: http://localhost:8002
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Nginx Proxy**: http://localhost:8000

## 📚 Additional Resources

- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LlamaIndex Embedding Guide](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html)
- [Qdrant Vector Database Docs](https://qdrant.tech/documentation/)

## 🆘 Support

If you encounter issues:

1. **Check Logs**: Review all service logs for errors
2. **Run Tests**: Execute `python test_bge_m3.py` for diagnostics
3. **Verify Resources**: Ensure sufficient GPU memory and disk space
4. **Backup Strategy**: Always maintain backups before migration

---

**✅ Migration Complete**: You now have BGE-M3 running for production testing!