#!/usr/bin/env python3
"""
Debug script to investigate ChromaDB data persistence issues
"""
import os
import sqlite3
import chromadb
from chromadb.config import Settings

def debug_chromadb_data():
    """Debug ChromaDB data issues"""
    print("=== ChromaDB Debug Investigation ===\n")

    # 1. Check database file
    db_path = "./data/chromadb/chroma.sqlite3"
    print(f"1. Database file: {db_path}")
    print(f"   Exists: {os.path.exists(db_path)}")
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"   Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
    print()

    # 2. Check collection directories
    chroma_dir = "./data/chromadb"
    print("2. Collection directories:")
    if os.path.exists(chroma_dir):
        dirs = [d for d in os.listdir(chroma_dir) if os.path.isdir(os.path.join(chroma_dir, d))]
        for i, dir_name in enumerate(dirs):
            dir_path = os.path.join(chroma_dir, dir_name)
            print(f"   {i+1}. {dir_name}")
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                print(f"      Files: {files}")
    print()

    # 3. Try to connect with ChromaDB
    print("3. ChromaDB Connection Test:")
    try:
        settings = Settings(
            allow_reset=False,
            is_persistent=True,
            anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(path=chroma_dir, settings=settings)

        # List collections
        collections = client.list_collections()
        print(f"   Collections found: {len(collections)}")
        for coll in collections:
            print(f"   - {coll.name} (ID: {coll.id})")

            # Try to get collection and count
            try:
                collection = client.get_collection(coll.name)
                count = collection.count()
                print(f"     Records: {count}")

                # Try to get some data
                if count > 0:
                    sample = collection.get(limit=3, include=["documents", "metadatas"])
                    print(f"     Sample keys: {list(sample.keys())}")
                    if "documents" in sample:
                        print(f"     Sample documents count: {len(sample['documents'])}")
                        if sample["documents"]:
                            print(f"     First doc preview: {sample['documents'][0][:100]}...")

            except Exception as e:
                print(f"     Error accessing collection: {e}")

    except Exception as e:
        print(f"   ChromaDB connection failed: {e}")
    print()

    # 4. Direct SQLite inspection
    print("4. Direct SQLite Inspection:")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"   SQLite tables: {[t[0] for t in tables]}")

        # Check collections table
        if any('collection' in t[0] for t in tables):
            cursor.execute("SELECT * FROM collections;")
            collections_data = cursor.fetchall()
            print(f"   Collections in SQLite: {len(collections_data)}")
            for i, coll in enumerate(collections_data):
                print(f"     {i+1}. ID: {coll[0]}, Name: {coll[1]}")

        # Check embeddings table
        if any('embedding' in t[0] for t in tables):
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            embedding_count = cursor.fetchone()[0]
            print(f"   Embeddings in SQLite: {embedding_count}")

        conn.close()

    except Exception as e:
        print(f"   SQLite inspection failed: {e}")
    print()

    # 5. Check TEMP_VECTOR variable
    print("5. TEMP_VECTOR variable:")
    from rag_pdf import TEMP_VECTOR
    print(f"   TEMP_VECTOR path: {TEMP_VECTOR}")
    print(f"   Path exists: {os.path.exists(TEMP_VECTOR)}")
    print(f"   Same as chroma_dir: {os.path.abspath(TEMP_VECTOR) == os.path.abspath(chroma_dir)}")

if __name__ == "__main__":
    debug_chromadb_data()