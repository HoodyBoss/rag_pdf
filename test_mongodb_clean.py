#!/usr/bin/env python3
"""
Clean test for MongoDB RAG system without encoding issues
"""
import os
import sys
from datetime import datetime

print("=" * 60)
print("MONGODB RAG SYSTEM TEST")
print("=" * 60)

try:
    # Test MongoDB connection
    print("1. Testing MongoDB connection...")
    from mongodb_rag import MongoDBRAG

    # Use local MongoDB for testing
    mongodb_uri = "mongodb://localhost:27017/"
    db_name = "test_rag"

    print(f"   Connecting to: {mongodb_uri}")
    print(f"   Database: {db_name}")

    # Initialize RAG system
    rag = MongoDBRAG(mongodb_uri, db_name)

    # Test basic functionality
    print("2. Testing basic operations...")

    # Test embedding generation
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    test_text = "This is a test document for MongoDB RAG system."
    embedding = embed_model.encode(test_text)

    print(f"   Embedding generated: {len(embedding)} dimensions")

    # Test document storage
    print("3. Testing document storage...")
    chunks = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test.txt", "page": 1, "chunk_id": 0}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "test.txt", "page": 1, "chunk_id": 1}
        }
    ]

    doc_id = rag.store_document(chunks, "test_document.txt")
    print(f"   Document stored with ID: {doc_id}")

    # Test search functionality
    print("4. Testing semantic search...")
    query = "What is machine learning?"
    results = rag.search_similar(query, top_k=2, min_similarity=0.1)

    print(f"   Search completed: {len(results)} results found")
    for i, result in enumerate(results):
        similarity = result.get('similarity', 0)
        text = result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
        print(f"   Result {i+1}: {similarity:.3f} similarity - {text}")

    # Test database statistics
    print("5. Testing database statistics...")
    stats = rag.get_database_stats()
    print(f"   Database stats:")
    print(f"   - Documents: {stats.get('documents_count', 0)}")
    print(f"   - Embeddings: {stats.get('embeddings_count', 0)}")
    print(f"   - Metadata: {stats.get('metadata_count', 0)}")
    print(f"   - Connection: {stats.get('connection_status', 'unknown')}")

    print("6. Testing document listing...")
    documents = rag.list_documents()
    print(f"   Found {len(documents)} documents:")
    for doc in documents:
        name = doc.get('source_name', 'Unknown')
        chunks = doc.get('total_chunks', 0)
        created = doc.get('created_at', 'Unknown')
        print(f"   - {name}: {chunks} chunks, created {created}")

    print()
    print("=" * 60)
    print("MONGODB RAG SYSTEM TEST - SUCCESS")
    print("=" * 60)
    print("All tests completed successfully!")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Cleanup test data
    print("7. Cleaning up test data...")
    if doc_id:
        rag.delete_document(doc_id)
        print("   Test document deleted")

    print("Cleanup completed.")

except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    print("Please ensure MongoDB is running and dependencies are installed:")
    print("  pip install pymongo[srv] sentence-transformers")

except Exception as e:
    print(f"ERROR: Test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)