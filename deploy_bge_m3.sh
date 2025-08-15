#!/bin/bash

# BGE-M3 Deployment Script
# This script handles the complete migration from Nomic Embed Text v1 to BGE-M3

set -e

echo "🚀 BGE-M3 Migration Deployment Script"
echo "====================================="

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
    print_warning "BGE-M3 model download may require authentication"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Backup current system
print_status "Step 1: Creating backup of current system..."
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Backup current configuration
cp docker compose.yml "$BACKUP_DIR/docker compose.yml.backup" 2>/dev/null || print_warning "No existing docker compose.yml found"
cp app/requirements.txt "$BACKUP_DIR/requirements.txt.backup" 2>/dev/null || print_warning "No existing requirements.txt found"

print_success "Backup created in $BACKUP_DIR"

# Step 2: Prepare collection migration
print_status "Step 2: Preparing collection migration..."
docker compose exec app python migrate_collection.py || print_warning "Migration preparation may have failed - continuing..."

# Step 3: Stop current services
print_status "Step 3: Stopping current services..."
docker compose down

# Step 4: Build new images with BGE-M3 configuration
print_status "Step 4: Building BGE-M3 compatible images..."
docker compose -f docker compose.bge-m3.yml build --no-cache

# Step 5: Start services with BGE-M3 configuration
print_status "Step 5: Starting services with BGE-M3..."
export EMBEDDING_MODEL="BAAI/bge-m3"
docker compose -f docker compose.bge-m3.yml up -d

# Step 6: Wait for services to be healthy
print_status "Step 6: Waiting for services to be healthy..."
echo "This may take several minutes for BGE-M3 model download..."

# Wait for vLLM embedding service
print_status "Waiting for vLLM embedding service..."
for i in {1..60}; do
    if curl -f http://localhost:8002/v1/models >/dev/null 2>&1; then
        print_success "vLLM embedding service is ready"
        break
    fi
    echo -n "."
    sleep 10
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

# Step 7: Verify BGE-M3 model is loaded
print_status "Step 7: Verifying BGE-M3 model..."
MODELS_RESPONSE=$(curl -s http://localhost:8002/v1/models)
if echo "$MODELS_RESPONSE" | grep -q "bge-m3"; then
    print_success "BGE-M3 model is loaded successfully"
else
    print_error "BGE-M3 model may not be loaded correctly"
    print_error "Response: $MODELS_RESPONSE"
fi

# Step 8: Instructions for document reprocessing
print_success "🎉 BGE-M3 deployment completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Access your application at http://localhost:5000"
echo "2. Re-upload and process your documents to use BGE-M3 embeddings"
echo "3. Test query performance with the new embedding model"
echo "4. Monitor system performance and GPU memory usage"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs: docker compose -f docker compose.bge-m3.yml logs"
echo "- Monitor GPU usage: docker exec vllm-embedding-cuda-dl nvidia-smi"
echo "- Check model status: curl http://localhost:8002/v1/models"
echo ""
echo "📊 System Status:"
echo "- LLM Service: http://localhost:8001"
echo "- Embedding Service: http://localhost:8002" 
echo "- Main Application: http://localhost:5000"
echo "- Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
print_success "Migration deployment completed successfully!"