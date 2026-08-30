#!/bin/bash

# Production deployment script for FlowCraft-DiT

set -e

echo "🚀 Deploying FlowCraft-DiT..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints outputs/api outputs/ui logs config

# Check if checkpoint exists
if [ ! -f "checkpoints/flowcraft_step10000.pt" ]; then
    echo "⚠️  Warning: Checkpoint not found at checkpoints/flowcraft_step10000.pt"
    echo "   Please place your trained checkpoint in the checkpoints/ directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service status
echo "📊 Service status:"
docker-compose ps

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📡 Services available at:"
echo "   - API: http://localhost:8000"
echo "   - UI: http://localhost:8501"
echo "   - Health: http://localhost:8000/health"
echo ""
echo "📋 View logs with: docker-compose logs -f"
echo "🛑 Stop services with: docker-compose down"
