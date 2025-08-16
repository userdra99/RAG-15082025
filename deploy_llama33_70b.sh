#!/bin/bash

# Llama-3.3-70B-Instruct-AWQ Deployment Script
# Deploys the kosbu/Llama-3.3-70B-Instruct-AWQ model with dual-GPU support

set -e

echo "🚀 Llama-3.3-70B-Instruct-AWQ Deployment Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check GPU availability
print_status "Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    print_error "NVIDIA GPU not detected. This model requires 2 GPUs."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    print_error "This configuration requires 2 GPUs. Found: $GPU_COUNT"
    exit 1
fi
print_success "Found $GPU_COUNT GPUs"

# Check GPU memory
print_status "Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 30000 ]; then
    print_error "Insufficient GPU memory. Recommended: 32GB per GPU, Found: ${GPU_MEM}MB"
    print_error "Llama-3.3-70B-AWQ requires approximately 42-52GB total VRAM"
    exit 1
fi

# Check if we have RTX 5090s specifically
print_status "Checking GPU models..."
nvidia-smi --query-gpu=name --format=csv,noheader | while read gpu_name; do
    echo "  - $gpu_name"
done

# Calculate total VRAM
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
print_status "Total VRAM: ${TOTAL_VRAM}MB (~$((TOTAL_VRAM/1024))GB)"

if [ "$TOTAL_VRAM" -lt 60000 ]; then
    print_error "Insufficient total VRAM. Need ~60GB, found ${TOTAL_VRAM}MB"
    exit 1
fi
print_success "Sufficient VRAM for Llama-3.3-70B-AWQ (~42-52GB) + BGE-M3 (~1.3GB)"

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    print_error "docker is not installed or not in PATH"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    print_error "docker compose is not available"
    exit 1
fi

# Check if HUGGING_FACE_HUB_TOKEN is set
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    print_warning "HUGGING_FACE_HUB_TOKEN environment variable is not set"
    print_warning "Model download may require authentication"
    print_status "Proceeding anyway - kosbu/Llama-3.3-70B-Instruct-AWQ is a public model"
fi

# Step 1: Stop current services
print_status "Step 1: Stopping current services..."
docker compose down || true
docker compose -f docker-compose.qwen3.yml down || true
docker compose -f docker-compose.bge-m3.yml down || true

# Step 2: Build new images
print_status "Step 2: Building images for Llama-3.3-70B..."
docker compose -f docker-compose.llama33-70b.yml build --no-cache

# Step 3: Start services with Llama-3.3-70B configuration
print_status "Step 3: Starting services with Llama-3.3-70B-Instruct-AWQ..."
export LLM_MODEL="Llama-3.3-70B-Instruct-AWQ"
export EMBEDDING_MODEL="BAAI/bge-m3"
docker compose -f docker-compose.llama33-70b.yml up -d

# Step 4: Wait for services to be healthy
print_status "Step 4: Waiting for services to be healthy..."
echo "This may take 10-15 minutes for Llama-3.3-70B model download (~35GB AWQ)..."

# Wait for vLLM LLM service
print_status "Waiting for Llama-3.3-70B LLM service..."
for i in {1..180}; do
    if curl -f http://localhost:8001/v1/models >/dev/null 2>&1; then
        print_success "Llama-3.3-70B LLM service is ready"
        break
    fi
    echo -n "."
    sleep 10
done

# Wait for embedding service
print_status "Waiting for BGE-M3 embedding service..."
for i in {1..60}; do
    if curl -f http://localhost:8002/v1/models >/dev/null 2>&1; then
        print_success "BGE-M3 embedding service is ready"
        break
    fi
    echo -n "."
    sleep 5
done

# Wait for main app
print_status "Waiting for main application..."
for i in {1..30}; do
    if curl -f http://localhost:5000/health >/dev/null 2>&1; then
        print_success "Main application is ready"
        break
    fi
    echo -n "."
    sleep 5
done

# Step 5: Verify Llama-3.3-70B model is loaded
print_status "Step 5: Verifying Llama-3.3-70B model..."
MODELS_RESPONSE=$(curl -s http://localhost:8001/v1/models)
if echo "$MODELS_RESPONSE" | grep -q "Llama-3.3"; then
    print_success "Llama-3.3-70B-Instruct-AWQ model is loaded successfully"
else
    print_warning "Could not verify Llama-3.3-70B model loading"
    print_warning "Response: $MODELS_RESPONSE"
fi

# Step 6: Show GPU memory usage
print_status "Step 6: GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Step 7: Test the model with a simple query
print_status "Step 7: Testing model response..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Llama-3.3-70B-Instruct-AWQ",
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }')

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    print_success "Model is responding correctly"
else
    print_warning "Model test failed: $TEST_RESPONSE"
fi

# Step 8: Instructions
print_success "🎉 Llama-3.3-70B-Instruct-AWQ deployment completed!"
echo ""
echo "📋 System Information:"
echo "- Model: kosbu/Llama-3.3-70B-Instruct-AWQ"
echo "- Model Size: ~35GB (AWQ INT4 quantized)"
echo "- Context Length: 131,072 tokens"
echo "- Tensor Parallel: 2 GPUs"
echo "- Embedding: BGE-M3 on GPU 1"
echo "- Total VRAM Usage: ~43-53GB"
echo ""
echo "📊 Service Endpoints:"
echo "- LLM Service: http://localhost:8001"
echo "- Embedding Service: http://localhost:8002" 
echo "- Main Application: http://localhost:5000"
echo "- Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "🔧 Monitoring Commands:"
echo "- Check logs: docker compose -f docker-compose.llama33-70b.yml logs"
echo "- Monitor GPUs: watch -n 1 nvidia-smi"
echo "- Check model: curl http://localhost:8001/v1/models"
echo ""
echo "⚡ Performance Tips:"
echo "- Model uses AWQ 4-bit quantization for efficiency"
echo "- First query may be slower due to model loading"
echo "- Supports up to 131K context length"
echo "- GPU memory utilization set to 75% for stability"
echo ""
print_success "Your RAG system is now running with Llama-3.3-70B-Instruct-AWQ!"