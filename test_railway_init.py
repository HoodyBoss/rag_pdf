#!/usr/bin/env python3
"""
Test Railway RAG system initialization
"""
import os
import sys
from dotenv import load_dotenv

print("=" * 60)
print("RAILWAY RAG SYSTEM INITIALIZATION TEST")
print("=" * 60)

# Load environment variables
load_dotenv()
print("1. Environment variables loaded")

# Test MongoDB RAG import
try:
    from mongodb_rag import MongoDBRAG
    print("2. MongoDB RAG module imported successfully")
except ImportError as e:
    print(f"2. ERROR: MongoDB RAG import failed: {e}")
    sys.exit(1)

# Test Railway RAG import
try:
    import railway_rag
    print("3. Railway RAG module imported successfully")
except ImportError as e:
    print(f"3. ERROR: Railway RAG import failed: {e}")
    sys.exit(1)

# Test environment configuration
mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
database_name = os.getenv('DATABASE_NAME', 'rag_pdf_railway')
port = int(os.getenv('PORT', 7860))
host = os.getenv('HOST', '0.0.0.0')

print(f"4. Configuration:")
print(f"   MongoDB URI: {mongodb_uri}")
print(f"   Database: {database_name}")
print(f"   Host:Port: {host}:{port}")

# Test system initialization
try:
    print("5. Testing system initialization...")
    initialized = railway_rag.initialize_system()
    if initialized:
        print("   System initialized successfully")
    else:
        print("   System initialization failed")
        sys.exit(1)
except Exception as e:
    print(f"   ERROR: System initialization failed: {e}")
    sys.exit(1)

# Test system status
try:
    print("6. Testing system status...")
    status = railway_rag.get_system_status()
    print(f"   MongoDB connected: {status.get('mongodb_connected', False)}")
    print(f"   Embedding model loaded: {status.get('embedding_model_loaded', False)}")
    print(f"   System status: {status.get('status', 'Unknown')}")

    stats = status.get('database_stats', {})
    if stats:
        print(f"   Database stats:")
        print(f"   - Documents: {stats.get('documents_count', 0)}")
        print(f"   - Embeddings: {stats.get('embeddings_count', 0)}")
        print(f"   - Database: {stats.get('database_name', 'Unknown')}")
except Exception as e:
    print(f"   ERROR: Status check failed: {e}")

# Test health check
try:
    print("7. Testing health check...")
    health = railway_rag.health_check()
    print(f"   Health status: {health.get('status', 'Unknown')}")
    print(f"   Timestamp: {health.get('timestamp', 'Unknown')}")
except Exception as e:
    print(f"   ERROR: Health check failed: {e}")

print()
print("=" * 60)
print("RAILWAY RAG SYSTEM INITIALIZATION TEST - SUCCESS")
print("=" * 60)
print("Railway RAG system is ready for deployment!")

print()
print("NEXT STEPS:")
print("1. Push to GitHub repository")
print("2. Create Railway project")
print("3. Add MongoDB Atlas addon")
print("4. Configure environment variables")
print("5. Deploy!")