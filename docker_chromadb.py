"""
Docker-based ChromaDB client for RAG PDF
Eliminates lock file issues and improves performance
"""
import chromadb
from chromadb.config import Settings
import logging
import time
import requests

def create_docker_chromadb_client():
    """
    Create ChromaDB client connected to Docker container
    """
    try:
        # Docker ChromaDB connection settings
        host = "localhost"  # Docker container host
        port = 8000         # Docker exposed port

        # Create client settings for Docker
        settings = Settings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=host,
            chroma_server_http_port=port,
            allow_reset=False,
            anonymized_telemetry=False
        )

        # Create client
        client = chromadb.HttpClient(
            host=f"{host}:{port}",
            settings=settings
        )

        # Test connection
        heartbeat = client.heartbeat()
        logging.info(f"✅ Docker ChromaDB connected: {heartbeat}")

        return client

    except Exception as e:
        logging.error(f"❌ Failed to connect to Docker ChromaDB: {e}")
        return None

def check_docker_chromadb_status():
    """
    Check if Docker ChromaDB is running and accessible
    """
    try:
        response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
        if response.status_code == 200:
            logging.info("✅ Docker ChromaDB is running and accessible")
            return True
        else:
            logging.warning(f"⚠️ Docker ChromaDB returned status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logging.error("❌ Docker ChromaDB is not running or not accessible")
        return False
    except Exception as e:
        logging.error(f"❌ Error checking Docker ChromaDB: {e}")
        return False

def wait_for_docker_chromadb(max_wait=30, interval=2):
    """
    Wait for Docker ChromaDB to be ready
    """
    logging.info("⏳ Waiting for Docker ChromaDB to be ready...")

    for i in range(max_wait // interval):
        if check_docker_chromadb_status():
            logging.info("✅ Docker ChromaDB is ready!")
            return True

        logging.info(f"Waiting... ({i * interval}/{max_wait}s)")
        time.sleep(interval)

    logging.error("❌ Docker ChromaDB did not become ready in time")
    return False

def setup_docker_chromadb():
    """
    Setup Docker ChromaDB connection and verify it works
    """
    logging.info("🐳 Setting up Docker ChromaDB...")

    # Check if Docker ChromaDB is running
    if not check_docker_chromadb_status():
        logging.error("❌ Docker ChromaDB is not running!")
        logging.info("Please start Docker ChromaDB with:")
        logging.info("docker-compose up -d")
        return None, None

    # Wait for Docker ChromaDB to be ready
    if not wait_for_docker_chromadb():
        return None, None

    # Create client
    client = create_docker_chromadb_client()
    if client is None:
        return None, None

    # Create or get collection
    try:
        collection = client.get_or_create_collection(name="pdf_data")
        count = collection.count()
        logging.info(f"✅ Docker ChromaDB collection ready: {count} records")
        return client, collection
    except Exception as e:
        logging.error(f"❌ Failed to create/get collection: {e}")
        return None, None

def backup_docker_chromadb(backup_name=None):
    """
    Backup Docker ChromaDB data (simplified version)
    """
    try:
        if backup_name is None:
            from datetime import datetime
            backup_name = f"docker_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # For Docker, backup is handled by Docker volumes
        # Users should backup the Docker volume directly
        logging.info(f"📦 Docker ChromaDB backup: {backup_name}")
        logging.info("💡 To backup Docker data, run:")
        logging.info(f"docker run --rm -v chromadb_data:/data -v {os.getcwd()}:/backup ubuntu tar czf /backup/{backup_name}.tar.gz -C /data .")

        return True
    except Exception as e:
        logging.error(f"❌ Docker backup failed: {e}")
        return False

def restore_docker_chromadb(backup_name):
    """
    Restore Docker ChromaDB data (simplified version)
    """
    try:
        logging.info(f"📦 Restoring Docker ChromaDB: {backup_name}")
        logging.info("💡 To restore Docker data, run:")
        logging.info(f"docker run --rm -v chromadb_data:/data -v {os.getcwd()}:/backup ubuntu tar xzf /backup/{backup_name}.tar.gz -C /data")
        logging.info("🔄 Then restart Docker ChromaDB: docker-compose restart chromadb")

        return True
    except Exception as e:
        logging.error(f"❌ Docker restore failed: {e}")
        return False

# Export functions for use in main app
__all__ = [
    'setup_docker_chromadb',
    'check_docker_chromadb_status',
    'create_docker_chromadb_client',
    'backup_docker_chromadb',
    'restore_docker_chromadb',
    'wait_for_docker_chromadb'
]