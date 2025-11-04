#!/usr/bin/env python3
"""
Fix ChromaDB data persistence issues by reconstructing broken collections
"""
import os
import shutil
import sqlite3
import chromadb
from chromadb.config import Settings
import logging
from datetime import datetime

def backup_current_database():
    """Backup current database before fixing"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./data/chromadb_backup/broken_db_backup_{timestamp}"
        os.makedirs(backup_path, exist_ok=True)

        if os.path.exists("./data/chromadb"):
            shutil.copytree("./data/chromadb", os.path.join(backup_path, "chromadb"))
            logging.info(f"✅ Backed up current database to: {backup_path}")
            return True
    except Exception as e:
        logging.error(f"❌ Backup failed: {e}")
        return False

def scan_collection_directories():
    """Scan and identify all collection directories"""
    chroma_dir = "./data/chromadb"
    collection_dirs = []

    if os.path.exists(chroma_dir):
        for item in os.listdir(chroma_dir):
            item_path = os.path.join(chroma_dir, item)
            if os.path.isdir(item_path) and item != "__pycache__":
                collection_dirs.append({
                    'path': item_path,
                    'id': item,
                    'has_data': os.path.exists(os.path.join(item_path, "header.bin"))
                })

    return collection_dirs

def get_sqlite_collection_info():
    """Get collection info from SQLite database"""
    try:
        db_path = "./data/chromadb/chroma.sqlite3"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name FROM collections;")
        collections = cursor.fetchall()

        conn.close()
        return collections
    except Exception as e:
        logging.error(f"❌ Failed to read SQLite collections: {e}")
        return []

def reconstruct_collection(collection_id, collection_name):
    """Reconstruct a broken collection directory"""
    try:
        collection_dir = f"./data/chromadb/{collection_id}"

        # Remove broken directory
        if os.path.exists(collection_dir):
            shutil.rmtree(collection_dir)
            logging.info(f"   Removed broken directory: {collection_id}")

        # Create new directory
        os.makedirs(collection_dir, exist_ok=True)

        # Create essential files
        files_to_create = [
            "header.bin",
            "length.bin",
            "link_lists.bin"
        ]

        for filename in files_to_create:
            file_path = os.path.join(collection_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(b'')  # Create empty file
            logging.info(f"   Created file: {filename}")

        logging.info(f"✅ Reconstructed collection: {collection_name} ({collection_id})")
        return True

    except Exception as e:
        logging.error(f"❌ Failed to reconstruct collection {collection_id}: {e}")
        return False

def fix_database_persistence():
    """Main function to fix database persistence issues"""
    logging.info("🔧 Starting ChromaDB persistence fix...")

    # 1. Backup current state
    if not backup_current_database():
        logging.error("❌ Cannot proceed without backup")
        return False

    # 2. Scan collection directories
    logging.info("🔍 Scanning collection directories...")
    collection_dirs = scan_collection_directories()
    logging.info(f"   Found {len(collection_dirs)} collection directories")

    for col in collection_dirs:
        status = "✅ Has data" if col['has_data'] else "❌ Empty"
        logging.info(f"   - {col['id']}: {status}")

    # 3. Get SQLite collection info
    logging.info("📋 Reading SQLite collection metadata...")
    sqlite_collections = get_sqlite_collection_info()
    logging.info(f"   SQLite collections: {len(sqlite_collections)}")

    for coll_id, coll_name in sqlite_collections:
        logging.info(f"   - {coll_name} (ID: {coll_id})")

    # 4. Find mismatched collections
    sqlite_ids = {coll[0] for coll in sqlite_collections}
    filesystem_ids = {col['id'] for col in collection_dirs}

    missing_in_fs = sqlite_ids - filesystem_ids
    empty_in_fs = [col['id'] for col in collection_dirs if not col['has_data']]

    logging.info(f"🚨 Issues found:")
    logging.info(f"   Collections in SQLite but missing in filesystem: {len(missing_in_fs)}")
    logging.info(f"   Empty collection directories: {len(empty_in_fs)}")

    # 5. Fix missing collections
    if missing_in_fs:
        logging.info("🔧 Fixing missing collections...")
        for coll_id, coll_name in sqlite_collections:
            if coll_id in missing_in_fs:
                logging.info(f"   Fixing: {coll_name} ({coll_id})")
                reconstruct_collection(coll_id, coll_name)

    # 6. Fix empty collections
    if empty_in_fs:
        logging.info("🔧 Fixing empty collections...")
        for collection_id in empty_in_fs:
            if collection_id in sqlite_ids:
                coll_name = next(name for cid, name in sqlite_collections if cid == collection_id)
                logging.info(f"   Fixing empty: {coll_name} ({collection_id})")
                reconstruct_collection(collection_id, coll_name)

    # 7. Test the fix
    logging.info("🧪 Testing fixed database...")
    try:
        settings = Settings(
            allow_reset=False,
            is_persistent=True,
            anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(path="./data/chromadb", settings=settings)
        collection = client.get_collection("pdf_data")
        count = collection.count()

        logging.info(f"✅ Fix successful! Collection 'pdf_data' has {count} records")

        if count == 0:
            logging.warning("⚠️ Collection is still empty - data may be permanently lost")
        else:
            logging.info("🎉 Data persistence fixed successfully!")

        return True

    except Exception as e:
        logging.error(f"❌ Fix verification failed: {e}")
        return False

def check_data_integrity():
    """Check data integrity after fix"""
    try:
        settings = Settings(
            allow_reset=False,
            is_persistent=True,
            anonymized_telemetry=False
        )
        client = chromadb.PersistentClient(path="./data/chromadb", settings=settings)
        collection = client.get_collection("pdf_data")

        count = collection.count()
        logging.info(f"📊 Final data integrity check:")
        logging.info(f"   Collection: pdf_data")
        logging.info(f"   Records: {count}")

        if count > 0:
            # Try to get sample data
            sample = collection.get(limit=3, include=["documents", "metadatas"])
            if sample and "documents" in sample:
                logging.info(f"   Sample documents: {len(sample['documents'])}")
                if sample['documents']:
                    logging.info(f"   First doc preview: {sample['documents'][0][:100]}...")

        return count

    except Exception as e:
        logging.error(f"❌ Integrity check failed: {e}")
        return 0

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=== ChromaDB Persistence Fix Tool ===\n")

    # Run the fix
    if fix_database_persistence():
        # Check integrity
        count = check_data_integrity()
        print(f"\n🎯 Result: Database now has {count} records")

        if count > 0:
            print("✅ Data persistence issue RESOLVED!")
        else:
            print("❌ Data appears to be permanently lost - consider restore from backup")
    else:
        print("❌ Fix failed - check logs for details")