# Docker Setup for RAG PDF System

## Overview
This Docker Compose setup provides the complete infrastructure for the RAG PDF system:
- **MongoDB**: Document metadata storage and user management
- **ChromaDB**: Vector database for document embeddings
- **Redis**: Caching and session management

## Services Configuration

### MongoDB
- **Port**: `27017`
- **Database**: `rag_pdf`
- **Admin User**: `admin` / `password123`
- **Application User**: `rag_user` / `rag_password`
- **Data Volume**: `mongodb_data`

### ChromaDB
- **Port**: `8001` (moved from 8000 to avoid conflicts)
- **Data Volume**: `chromadb_data`
- **Configuration**: `./chroma-config`

### Redis
- **Port**: `6379`
- **Data Volume**: `redis_data`
- **Persistence**: Enabled with AOF

## Quick Start

### 1. Start all services
```bash
docker-compose up -d
```

### 2. Check service status
```bash
docker-compose ps
```

### 3. View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mongodb
docker-compose logs -f chromadb
docker-compose logs -f redis
```

### 4. Stop services
```bash
docker-compose down
```

### 5. Stop and remove volumes (⚠️ Deletes all data)
```bash
docker-compose down -v
```

## Environment Configuration

Update your backend environment variables to use these services:

```env
# MongoDB
MONGODB_URL=mongodb://rag_user:rag_password@localhost:27017/rag_pdf

# ChromaDB
CHROMADB_URL=http://localhost:8001

# Redis (optional)
REDIS_URL=redis://localhost:6379
```

## Database Initialization

The MongoDB container will automatically:
1. Create the `rag_pdf` database
2. Create application user `rag_user`
3. Create all necessary collections
4. Set up performance indexes
5. Insert default admin user

## Connection Testing

### MongoDB
```bash
# Connect to MongoDB
docker exec -it rag-pdf-mongodb mongosh -u rag_user -p rag_password --authenticationDatabase rag_pdf

# Show databases
show dbs

# Show collections
use rag_pdf
show collections
```

### ChromaDB
```bash
# Test ChromaDB connection
curl http://localhost:8001/api/v1/heartbeat
```

### Redis
```bash
# Connect to Redis
docker exec -it rag-redis redis-cli

# Test
ping
```

## Data Persistence

All data is persisted in Docker volumes:
- `mongodb_data`: MongoDB data files
- `chromadb_data`: ChromaDB vector embeddings
- `redis_data`: Redis cache data

Volumes are automatically created on first run and persist across container restarts.

## Troubleshooting

### Port Conflicts
If you encounter port conflicts, update the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "27018:27017"  # Change host port for MongoDB
  - "8002:8001"    # Change host port for ChromaDB
  - "6380:6379"    # Change host port for Redis
```

### Permission Issues
If you encounter permission issues, ensure the `mongodb-init` directory is accessible:
```bash
chmod -R 755 mongodb-init/
```

### Reset Everything
To completely reset the system:
```bash
docker-compose down -v
docker system prune -f
```

## Backup and Restore

### Backup MongoDB
```bash
docker exec rag-pdf-mongodb mongodump --out /backup
docker cp rag-pdf-mongodb:/backup ./mongodb-backup
```

### Restore MongoDB
```bash
docker cp ./mongodb-backup rag-pdf-mongodb:/restore
docker exec rag-pdf-mongodb mongorestore /restore
```

## Production Considerations

For production deployment:
1. Change default passwords
2. Enable authentication networks
3. Use SSL/TLS connections
4. Set up proper backup strategies
5. Monitor resource usage
6. Configure log rotation