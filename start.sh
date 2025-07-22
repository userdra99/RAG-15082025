#!/bin/bash

echo "Starting RAG System with optimized settings..."

# Function to check if service is healthy
check_service() {
    local service=$1
    local port=$2
    
    echo "Waiting for $service to start..."
    for i in {1..30}; do
        if curl -f http://localhost:$port/v1/models >/dev/null 2>&1; then
            echo "✅ $service is ready"
            return 0
        fi
        echo "⏳ Waiting for $service... ($i/30)"
        sleep 10
    done
    
    echo "❌ $service failed to start"
    return 1
}

# Start services sequentially
echo "1. Starting Qdrant..."
docker-compose up -d qdrant

sleep 5

echo "2. Starting vLLM LLM service..."
docker-compose up -d vllm-llm

echo "3. Checking vLLM LLM service..."
check_service "vllm-llm" 8001

echo "4. Starting vLLM Embedding service..."
docker-compose up -d vllm-embedding

echo "5. Checking vLLM Embedding service..."
check_service "vllm-embedding" 8002

echo "6. Starting remaining services..."
docker-compose up -d nginx app

echo ""
echo "🚀 System starting..."
echo "Check status with: docker-compose ps"
echo "View logs with: docker-compose logs -f"
echo ""
echo "Access the app at: http://localhost:8501"