#!/usr/bin/env python3
"""
Migration script to transfer data from ChromaDB to MongoDB
"""
import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

def migrate_chromadb_to_mongodb(chroma_path: str = "./data/chromadb",
                                 mongodb_uri: str = "mongodb://localhost:27017/",
                                 db_name: str = "rag_pdf_migrated"):
    """
    Migrate all data from ChromaDB to MongoDB

    Args:
        chroma_path: Path to ChromaDB data directory
        mongodb_uri: MongoDB connection string
        db_name: Target database name
    """
    try:
        # Import both systems
        import chromadb
        from chromadb.config import Settings
        from mongodb_rag import MongoDBRAG

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("🔄 Starting ChromaDB to MongoDB migration...")

        # Connect to ChromaDB
        logger.info(f"📊 Connecting to ChromaDB: {chroma_path}")
        chroma_settings = Settings(
            allow_reset=False,
            is_persistent=True,
            anonymized_telemetry=False
        )
        chroma_client = chromadb.PersistentClient(path=chroma_path, settings=chroma_settings)

        # Get all collections
        collections = chroma_client.list_collections()
        logger.info(f"📋 Found {len(collections)} collections in ChromaDB")

        # Initialize MongoDB
        logger.info(f"🗄️ Connecting to MongoDB: {mongodb_uri}")
        mongodb_rag = MongoDBRAG(mongodb_uri, db_name)

        total_documents_migrated = 0
        total_chunks_migrated = 0

        # Migrate each collection
        for collection in collections:
            collection_name = collection.name
            logger.info(f"📄 Migrating collection: {collection_name}")

            try:
                # Get all data from collection
                count = collection.count()
                logger.info(f"   📊 Found {count} items in collection")

                if count == 0:
                    continue

                # Get all documents and embeddings
                all_data = collection.get(
                    include=["documents", "metadatas", "embeddings"]
                )

                documents = all_data.get("documents", [])
                metadatas = all_data.get("metadatas", [])
                embeddings = all_data.get("embeddings", [])

                if not documents:
                    logger.warning(f"   ⚠️ No documents found in collection {collection_name}")
                    continue

                # Prepare chunks for MongoDB
                chunks = []
                for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                    chunk_data = {
                        "text": doc,
                        "metadata": meta or {},
                        "embedding": embeddings[i] if i < len(embeddings) else None,
                        "source": meta.get("source", collection_name) if meta else collection_name,
                        "page": meta.get("page", i) if meta else i,
                        "chunk_index": i
                    }
                    chunks.append(chunk_data)

                # Store in MongoDB
                doc_id = mongodb_rag.store_document(chunks, collection_name)
                total_documents_migrated += 1
                total_chunks_migrated += len(chunks)

                logger.info(f"   ✅ Migrated {len(chunks)} chunks to document {doc_id}")

            except Exception as e:
                logger.error(f"   ❌ Failed to migrate collection {collection_name}: {e}")
                continue

        # Migration summary
        logger.info("🎉 Migration completed!")
        logger.info(f"📊 Summary:")
        logger.info(f"   Collections migrated: {total_documents_migrated}")
        logger.info(f"   Total chunks migrated: {total_chunks_migrated}")
        logger.info(f"   Target database: {db_name}")

        # Get final MongoDB stats
        stats = mongodb_rag.get_database_stats()
        logger.info(f"📊 MongoDB stats: {stats}")

        # Create backup of migrated data
        backup_path = mongodb_rag.backup_data()
        logger.info(f"💾 Backup created: {backup_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def test_mongodb_data(mongodb_uri: str = "mongodb://localhost:27017/",
                     db_name: str = "rag_pdf_migrated"):
    """
    Test the migrated MongoDB data

    Args:
        mongodb_uri: MongoDB connection string
        db_name: Database name to test
    """
    try:
        from mongodb_rag import MongoDBRAG

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("🧪 Testing migrated MongoDB data...")

        # Connect to MongoDB
        mongodb_rag = MongoDBRAG(mongodb_uri, db_name)

        # Test search
        test_queries = [
            "ทดสอบ",
            "เอกสาร",
            "ข้อมูล"
        ]

        for query in test_queries:
            results = mongodb_rag.search_similar(query, top_k=3)
            logger.info(f"🔍 Query: '{query}' - Found {len(results)} results")

            for i, result in enumerate(results):
                logger.info(f"   Result {i+1}: similarity={result.get('similarity', 0):.3f}, text_preview={result.get('text', '')[:50]}...")

        # List documents
        documents = mongodb_rag.list_documents()
        logger.info(f"📋 Total documents: {len(documents)}")

        for doc in documents:
            logger.info(f"   - {doc.get('source_name', 'Unknown')} ({doc.get('total_chunks', 0)} chunks)")

        logger.info("✅ MongoDB data test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ MongoDB test failed: {e}")
        return False

def cleanup_chromadb_backup(chroma_path: str = "./data/chromadb",
                             backup_dir: str = "./data/chromadb_backups"):
    """
    Create backup of ChromaDB before cleanup

    Args:
        chroma_path: Path to ChromaDB data
        backup_dir: Directory to store backup
    """
    try:
        import shutil

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if not os.path.exists(chroma_path):
            logger.warning(f"⚠️ ChromaDB path not found: {chroma_path}")
            return False

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"chromadb_backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)

        logger.info(f"💾 Creating ChromaDB backup: {backup_path}")
        shutil.copytree(chroma_path, backup_path)

        logger.info("✅ ChromaDB backup created successfully!")
        return backup_path

    except Exception as e:
        logger.error(f"❌ ChromaDB backup failed: {e}")
        return False

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=== ChromaDB to MongoDB Migration Tool ===\n")

    # Configuration
    CHROMA_PATH = "./data/chromadb"
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DB_NAME = "rag_pdf_migrated"

    print(f"📊 ChromaDB Path: {CHROMA_PATH}")
    print(f"🗄️ MongoDB URI: {MONGODB_URI}")
    print(f"📊 Database Name: {DB_NAME}")
    print()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "migrate":
            print("🔄 Starting migration...")
            success = migrate_chromadb_to_mongodb(CHROMA_PATH, MONGODB_URI, DB_NAME)
            if success:
                print("✅ Migration completed successfully!")
            else:
                print("❌ Migration failed!")

        elif command == "test":
            print("🧪 Testing migrated data...")
            success = test_mongodb_data(MONGODB_URI, DB_NAME)
            if success:
                print("✅ Test completed successfully!")
            else:
                print("❌ Test failed!")

        elif command == "backup":
            print("💾 Creating ChromaDB backup...")
            backup_path = cleanup_chromadb_backup(CHROMA_PATH)
            if backup_path:
                print(f"✅ Backup created: {backup_path}")
            else:
                print("❌ Backup failed!")

        elif command == "full":
            print("🔄 Running full migration process...")

            # Step 1: Backup ChromaDB
            print("\n1️⃣ Creating ChromaDB backup...")
            backup_path = cleanup_chromadb_backup(CHROMA_PATH)
            if not backup_path:
                print("❌ Backup failed, stopping migration.")
                sys.exit(1)
            print(f"✅ Backup created: {backup_path}")

            # Step 2: Migrate data
            print("\n2️⃣ Migrating to MongoDB...")
            success = migrate_chromadb_to_mongodb(CHROMA_PATH, MONGODB_URI, DB_NAME)
            if not success:
                print("❌ Migration failed!")
                sys.exit(1)
            print("✅ Migration completed!")

            # Step 3: Test migrated data
            print("\n3️⃣ Testing migrated data...")
            success = test_mongodb_data(MONGODB_URI, DB_NAME)
            if not success:
                print("❌ Test failed!")
                sys.exit(1)
            print("✅ Test completed!")

            print("\n🎉 Full migration process completed successfully!")
            print(f"📁 ChromaDB backup: {backup_path}")
            print(f"🗄️ MongoDB database: {DB_NAME}")

        else:
            print("❌ Unknown command!")
            print("Available commands: migrate, test, backup, full")

    else:
        print("Usage: python chromadb_to_mongodb.py [command]")
        print("Commands:")
        print("  migrate  - Migrate ChromaDB to MongoDB")
        print("  test     - Test migrated MongoDB data")
        print("  backup   - Create ChromaDB backup")
        print("  full     - Run complete migration process")
        print()
        print("Environment variables:")
        print("  MONGODB_URI - MongoDB connection string")
        print()
        print("Example:")
        print("  python chromadb_to_mongodb.py full")
        print("  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/rag_pdf python chromadb_to_mongodb.py migrate")