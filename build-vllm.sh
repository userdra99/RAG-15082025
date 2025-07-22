#!/bin/bash

echo "🔨 Building Custom vLLM Docker Image for RAG Project"
echo "=================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not available in PATH"
    echo "Please install Docker Desktop and enable WSL integration"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Dockerfile.vllm" ]; then
    echo "❌ Dockerfile.vllm not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "📋 Building vLLM image with custom dependencies..."
echo "This may take 10-15 minutes on first build..."

# Build the custom vLLM image
docker build -f Dockerfile.vllm -t rag-vllm:latest .

if [ $? -eq 0 ]; then
    echo "✅ Custom vLLM image built successfully!"
    echo ""
    echo "🚀 You can now start the RAG system with:"
    echo "   docker-compose up --build"
    echo ""
    echo "📝 Or use the start script:"
    echo "   ./start.sh"
else
    echo "❌ Failed to build vLLM image"
    echo "Check the error messages above for details"
    exit 1
fi 