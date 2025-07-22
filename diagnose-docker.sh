#!/bin/bash

echo "🔍 Docker Setup Diagnostic for WSL"
echo "=================================="

echo ""
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed: $(docker --version)"
else
    echo "❌ Docker is not installed"
    exit 1
fi

echo ""
echo "2. Checking Docker daemon access..."
if docker ps &> /dev/null; then
    echo "✅ Docker daemon is accessible"
else
    echo "❌ Cannot access Docker daemon"
    echo "   This usually means Docker Desktop isn't running or WSL integration isn't enabled"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Make sure Docker Desktop is running on Windows"
    echo "   2. Open Docker Desktop Settings > Resources > WSL Integration"
    echo "   3. Enable integration with your Ubuntu distro"
    echo "   4. Click 'Apply & Restart'"
    echo "   5. Restart WSL: wsl --shutdown (in Windows PowerShell)"
    exit 1
fi

echo ""
echo "3. Testing basic Docker functionality..."
if docker run --rm hello-world &> /dev/null; then
    echo "✅ Basic Docker functionality works"
else
    echo "❌ Basic Docker functionality failed"
    exit 1
fi

echo ""
echo "4. Checking GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected"
    if docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU support works"
    else
        echo "⚠️  Docker GPU support not working"
        echo "   This might be due to missing NVIDIA Docker runtime"
    fi
else
    echo "ℹ️  No NVIDIA GPU detected (this is fine for CPU-only mode)"
fi

echo ""
echo "5. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is installed: $(docker-compose --version)"
else
    echo "❌ Docker Compose is not installed"
    echo "   Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo ""
echo "6. Checking project files..."
if [ -f "Dockerfile.vllm" ]; then
    echo "✅ Custom vLLM Dockerfile found"
else
    echo "❌ Custom vLLM Dockerfile not found"
    exit 1
fi

if [ -f "docker-compose.yml" ]; then
    echo "✅ Docker Compose file found"
else
    echo "❌ Docker Compose file not found"
    exit 1
fi

echo ""
echo "🎉 Docker setup looks good!"
echo ""
echo "🚀 You can now proceed with:"
echo "   ./build-vllm.sh"
echo ""
echo "📖 For detailed setup instructions, see:"
echo "   cat DOCKER_SETUP.md" 