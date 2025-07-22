#!/bin/bash

echo "=== RAG System Troubleshooting ==="
echo

# Check Docker and GPU
echo "1. Checking Docker..."
docker --version
docker-compose --version

echo
echo "2. Checking GPU availability..."
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found - GPU drivers may not be installed"

echo
echo "3. Checking Docker GPU support..."
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi 2>/dev/null || echo "❌ Docker GPU support not working"

echo
echo "4. Checking system resources..."
echo "Available memory:"
free -h
echo
echo "Disk space:"
df -h

echo
echo "5. Checking model sizes..."
echo "Llama-3.2-3B-Instruct: ~6GB"
echo "Jina embeddings v4: ~4GB"
echo "Recommended GPU memory: 12GB+"

echo
echo "6. Docker Compose validation..."
docker-compose config --quiet && echo "✅ Docker compose config valid" || echo "❌ Docker compose config issues"

echo
echo "=== Troubleshooting Tips ==="
echo "1. Install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
echo "2. Run: sudo systemctl restart docker"
echo "3. Check GPU memory: nvidia-smi"
echo "4. Reduce model sizes in docker-compose.yml if needed"
echo "5. Use CPU-only mode as fallback"