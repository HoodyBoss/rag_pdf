@echo off
echo 🐳 Setting up Docker ChromaDB for RAG PDF...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop for Windows.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "chroma-config" mkdir chroma-config
if not exist "data\docker_backups" mkdir data\docker_backups

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down >nul 2>&1

REM Pull and start Docker ChromaDB
echo ⬇️ Pulling Docker images...
docker-compose pull

echo 🚀 Starting Docker ChromaDB...
docker-compose up -d

REM Wait for ChromaDB to be ready
echo ⏳ Waiting for ChromaDB to be ready...
timeout /t 10 /nobreak >nul

REM Check if ChromaDB is accessible
echo 🔍 Checking ChromaDB status...
curl -s http://localhost:8000/api/v1/heartbeat >nul 2>&1
if errorlevel 1 (
    echo ❌ ChromaDB is not accessible. Please check the logs:
    echo docker-compose logs chromadb
    pause
    exit /b 1
)

echo ✅ Docker ChromaDB is running and accessible!
echo.
echo 🎯 Setup Complete!
echo 📊 ChromaDB Web UI: http://localhost:8000
echo 🔌 Port: 8000
echo 💾 Data Volume: chromadb_data
echo.
echo 📋 Commands:
echo   Start:    docker-compose up -d
echo   Stop:     docker-compose down
echo   Status:   docker-compose ps
echo   Logs:     docker-compose logs chromadb
echo.
pause