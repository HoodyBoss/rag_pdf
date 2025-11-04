#!/bin/bash

# Docker ChromaDB Setup Script for RAG PDF
echo "🐳 Setting up Docker ChromaDB for RAG PDF..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p chroma-config
mkdir -p ./data/docker_backups

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Pull and start Docker ChromaDB
echo "⬇️ Pulling Docker images..."
docker-compose pull

echo "🚀 Starting Docker ChromaDB..."
docker-compose up -d

# Wait for ChromaDB to be ready
echo "⏳ Waiting for ChromaDB to be ready..."
sleep 10

# Check if ChromaDB is accessible
echo "🔍 Checking ChromaDB status..."
if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null; then
    echo "✅ Docker ChromaDB is running and accessible!"
    echo ""
    echo "🎯 Setup Complete!"
    echo "📊 ChromaDB Web UI: http://localhost:8000"
    echo "🔌 Port: 8000"
    echo "💾 Data Volume: chromadb_data"
    echo ""
    echo "📋 Commands:"
    echo "  Start:    docker-compose up -d"
    echo "  Stop:     docker-compose down"
    echo "  Status:   docker-compose ps"
    echo "  Logs:     docker-compose logs chromadb"
    echo "  Backup:   docker run --rm -v chromadb_data:/data -v $(pwd)/data:/backup ubuntu tar czf /backup/backup.tar.gz -C /data ."
    echo "  Restore:  docker run --rm -v chromadb_data:/data -v $(pwd)/data:/backup ubuntu tar xzf /backup/backup.tar.gz -C /data && docker-compose restart chromadb"
else
    echo "❌ ChromaDB is not accessible. Please check the logs:"
    echo "docker-compose logs chromadb"
    exit 1
fi