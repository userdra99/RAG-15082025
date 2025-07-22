#!/bin/bash

echo "🧪 Testing Custom vLLM Setup"
echo "============================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

# Check if custom image exists
if ! docker images | grep -q "rag-vllm"; then
    echo "⚠️  Custom vLLM image not found. Building..."
    ./build-vllm.sh
fi

echo "🔍 Testing vLLM LLM service..."

# Test LLM service
docker run --rm --gpus all \
    -e MODEL=unsloth/Llama-3.2-3B-Instruct \
    -e PORT=8000 \
    -p 8001:8000 \
    rag-vllm:latest \
    --model-name unsloth/Llama-3.2-3B-Instruct \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.3 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code &

LLM_PID=$!

echo "⏳ Waiting for LLM service to start..."
sleep 30

# Test LLM endpoint
if curl -f http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "✅ LLM service is responding"
    
    # Test a simple completion
    echo "🧠 Testing completion..."
    curl -X POST http://localhost:8001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "unsloth/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50
        }' | jq '.choices[0].message.content' 2>/dev/null || echo "❌ Completion test failed"
else
    echo "❌ LLM service not responding"
fi

# Cleanup
kill $LLM_PID 2>/dev/null

echo ""
echo "🔍 Testing vLLM Embedding service..."

# Test Embedding service
docker run --rm --gpus all \
    -e MODEL=BAAI/bge-m3 \
    -e PORT=8000 \
    -p 8002:8000 \
    rag-vllm:latest \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.2 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --task embed \
    --trust-remote-code \
    --dtype float16 &

EMBED_PID=$!

echo "⏳ Waiting for Embedding service to start..."
sleep 30

# Test embedding endpoint
if curl -f http://localhost:8002/v1/models >/dev/null 2>&1; then
    echo "✅ Embedding service is responding"
    
    # Test embedding
    echo "🔢 Testing embedding..."
    curl -X POST http://localhost:8002/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{
            "model": "BAAI/bge-m3",
            "input": "Hello world"
        }' | jq '.data[0].embedding | length' 2>/dev/null || echo "❌ Embedding test failed"
else
    echo "❌ Embedding service not responding"
fi

# Cleanup
kill $EMBED_PID 2>/dev/null

echo ""
echo "🎉 Testing complete!"
echo ""
echo "📝 Next steps:"
echo "1. Run: docker-compose up --build"
echo "2. Or use: ./start.sh"
echo "3. Access the app at: http://localhost:8501" 