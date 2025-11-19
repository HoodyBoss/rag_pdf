# Docker ChromaDB Setup for RAG PDF

## 🐳 Overview
This Docker-based ChromaDB setup eliminates file lock issues and improves performance for the RAG PDF application.

## 🚀 Quick Setup

### Windows
```bash
# Run the setup script
setup_docker.bat
```

### Linux/Mac
```bash
# Make script executable and run
chmod +x setup_docker.sh
./setup_docker.sh
```

### Manual Setup
```bash
# 1. Create directories
mkdir -p chroma-config ./data/docker_backups

# 2. Start Docker ChromaDB
docker-compose up -d

# 3. Verify it's running
curl http://localhost:8000/api/v1/heartbeat
```

## 📋 Docker Services

### ChromaDB
- **Container**: rag-chromadb
- **Image**: chromadb/chroma:latest
- **Port**: 8000
- **Data Volume**: chromadb_data
- **Web UI**: http://localhost:8000

### Redis (Optional)
- **Container**: rag-redis
- **Image**: redis:7-alpine
- **Port**: 6379
- **Data Volume**: redis_data

## 🔧 Configuration

### Environment Variables
- `CHROMA_SERVER_HOST`: 0.0.0.0
- `CHROMA_SERVER_HTTP_PORT`: 8000
- `CHROMA_LOG_LEVEL`: INFO
- `ANONYMIZED_TELEMETRY`: False

### Custom Configuration
Edit `chroma-config/chroma-server-config.yaml` to customize:
- Database settings
- Performance parameters
- Index configuration

## 💾 Backup & Restore

### Backup Data
```bash
# Backup to tar.gz file
docker run --rm \
  -v chromadb_data:/data \
  -v $(pwd)/data:/backup \
  ubuntu tar czf /backup/chromadb_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

### Restore Data
```bash
# Restore from backup
docker run --rm \
  -v chromadb_data:/data \
  -v $(pwd)/data:/backup \
  ubuntu tar xzf /backup/chromadb_backup_20251101_143000.tar.gz -C /data

# Restart ChromaDB
docker-compose restart chromadb
```

## 🔄 Management Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View status
docker-compose ps

# View logs
docker-compose logs chromadb
docker-compose logs redis

# Follow logs
docker-compose logs -f chromadb

# Update images
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate
```

## 🔍 Troubleshooting

### ChromaDB Not Accessible
```bash
# Check if container is running
docker-compose ps

# Check logs
docker-compose logs chromadb

# Restart container
docker-compose restart chromadb

# Check port accessibility
curl http://localhost:8000/api/v1/heartbeat
```

### Data Issues
```bash
# Backup current data first
docker run --rm -v chromadb_data:/data -v $(pwd)/data:/backup ubuntu tar czf /backup/emergency_backup.tar.gz -C /data .

# Reset database volume
docker-compose down
docker volume rm ragpdf_chromadb_data
docker-compose up -d
```

### Performance Issues
- Increase memory limit in docker-compose.yml
- Adjust batch size in configuration
- Monitor resource usage: `docker stats`

## 🌐 Web Interface
Access ChromaDB Web UI at: http://localhost:8000

## 📊 Monitoring
```bash
# Resource usage
docker stats rag-chromadb

# Container health
docker inspect rag-chromadb

# Disk usage
docker system df
```

## 🔗 Integration with RAG PDF
The Docker setup automatically integrates with the main application. The code will detect if Docker ChromaDB is running and use it instead of local ChromaDB.

## 🚨 Important Notes
- Docker must be running before starting the RAG PDF application
- Data persists in Docker volumes even after container restart
- Port 8000 must be available on the host system
- Redis is optional but recommended for caching