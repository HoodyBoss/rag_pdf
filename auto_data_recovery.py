#!/usr/bin/env python3
"""
Automatic data recovery system for ChromaDB
"""
import os
import logging
import glob
from datetime import datetime

def find_latest_backup():
    """Find the most recent backup"""
    try:
        backup_dir = "./data/chromadb_backup"
        if not os.path.exists(backup_dir):
            return None

        # Find all backup directories
        backup_pattern = os.path.join(backup_dir, "*_backup_*")
        backup_dirs = glob.glob(backup_pattern)

        if not backup_dirs:
            return None

        # Sort by modification time
        latest_backup = max(backup_dirs, key=os.path.getmtime)
        logging.info(f"📁 Found latest backup: {os.path.basename(latest_backup)}")
        return latest_backup

    except Exception as e:
        logging.error(f"❌ Error finding backup: {e}")
        return None

def restore_from_latest_backup():
    """Restore from the latest backup"""
    try:
        import shutil
        import time

        backup_path = find_latest_backup()
        if not backup_path:
            logging.warning("❌ No suitable backup found")
            return False

        logging.info(f"🔄 Restoring from backup: {os.path.basename(backup_path)}")

        # Import restore function
        from rag_pdf import restore_vector_db
        backup_name = os.path.basename(backup_path)

        # Try to restore
        success = restore_vector_db(backup_name)

        if success:
            logging.info("✅ Backup restored successfully")
            return True
        else:
            logging.error("❌ Backup restore failed")
            return False

    except Exception as e:
        logging.error(f"❌ Error during restore: {e}")
        return False

def check_available_backups():
    """Check what backups are available"""
    try:
        backup_dir = "./data/chromadb_backup"
        if not os.path.exists(backup_dir):
            print("No backup directory found")
            return []

        backups = []
        for item in os.listdir(backup_dir):
            item_path = os.path.join(backup_dir, item)
            if os.path.isdir(item_path) and "backup" in item:
                backup_path = os.path.join(item_path, "chromadb")
                if os.path.exists(backup_path):
                    # Try to count records in backup
                    try:
                        import chromadb
                        from chromadb.config import Settings
                        settings = Settings(
                            allow_reset=False,
                            is_persistent=True,
                            anonymized_telemetry=False
                        )
                        client = chromadb.PersistentClient(path=backup_path, settings=settings)
                        collection = client.get_collection("pdf_data")
                        count = collection.count()
                        backups.append({
                            'name': item,
                            'path': item_path,
                            'records': count,
                            'modified': datetime.fromtimestamp(os.path.getmtime(item_path))
                        })
                    except:
                        backups.append({
                            'name': item,
                            'path': item_path,
                            'records': 'Unknown',
                            'modified': datetime.fromtimestamp(os.path.getmtime(item_path))
                        })
                    finally:
                        client = None  # Cleanup

        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x['modified'], reverse=True)

        return backups

    except Exception as e:
        logging.error(f"❌ Error checking backups: {e}")
        return []

def auto_recovery_on_empty_database():
    """Automatically recover data when database is empty"""
    try:
        from rag_pdf import collection

        # Check if database is empty
        count = collection.count()
        if count > 0:
            logging.info(f"✅ Database has {count} records - no recovery needed")
            return True

        logging.warning("⚠️ Database is empty - attempting auto recovery...")

        # 1. Try to fix structure first
        logging.info("🔧 Step 1: Fixing database structure...")
        try:
            from fix_database_persistence import fix_database_persistence
            fix_database_persistence()
        except Exception as e:
            logging.error(f"❌ Structure fix failed: {e}")

        # 2. Check if data recovered
        try:
            count = collection.count()
            if count > 0:
                logging.info(f"✅ Data recovered! Now has {count} records")
                return True
        except:
            pass

        # 3. Try to restore from backup
        logging.info("🔄 Step 2: Trying backup restore...")
        if restore_from_latest_backup():
            return True

        logging.error("❌ Auto recovery failed - database remains empty")
        return False

    except Exception as e:
        logging.error(f"❌ Auto recovery failed: {e}")
        return False

def create_data_safety_backup():
    """Create a safety backup before critical operations"""
    try:
        import shutil
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"./data/chromadb_backup/safety_backup_{timestamp}"

        if os.path.exists("./data/chromadb"):
            shutil.copytree("./data/chromadb", os.path.join(backup_dir, "chromadb"))
            logging.info(f"✅ Safety backup created: {backup_dir}")
            return backup_dir
        else:
            logging.warning("⚠️ No chromadb directory to backup")
            return None

    except Exception as e:
        logging.error(f"❌ Safety backup failed: {e}")
        return None

def list_recovery_options():
    """List available recovery options"""
    print("=== Data Recovery Options ===\n")

    # Check available backups
    backups = check_available_backups()
    if backups:
        print("📦 Available Backups:")
        for i, backup in enumerate(backups):
            print(f"   {i+1}. {backup['name']} ({backup['records']} records, {backup['modified'].strftime('%Y-%m-%d %H:%M')})")
        print()
    else:
        print("❌ No backups available\n")

    # Check if database is empty
    try:
        from rag_pdf import collection
        count = collection.count()
        print(f"📊 Current Database Status:")
        print(f"   Records: {count}")
        print(f"   Status: {'🟢 EMPTY' if count == 0 else '🟢 Has Data'}")
        print()
    except:
        print("❌ Cannot check current database status\n")

    return backups

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=== Auto Data Recovery Tool ===\n")

    # List recovery options
    backups = list_recovery_options()

    if backups:
        print("🔧 Available Actions:")
        print("   1. Restore from latest backup")
        print("   2. List backup details")
        print("   3. Create safety backup")
        print("   4. Fix database structure")
        print()

        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            print("🔄 Restoring from latest backup...")
            success = restore_from_latest_backup()
            print(f"{'✅ Success' if success else '❌ Failed'}")

        elif choice == "2":
            print("\n📋 Backup Details:")
            for i, backup in enumerate(backups):
                print(f"   {i+1}. {backup['name']}")
                print(f"      Path: {backup['path']}")
                print(f"      Records: {backup['records']}")
                print(f"      Modified: {backup['modified']}")
                print()

        elif choice == "3":
            print("💾 Creating safety backup...")
            backup_dir = create_data_safety_backup()
            print(f"{'✅ Created' if backup_dir else '❌ Failed'}: {backup_dir}")

        elif choice == "4":
            print("🔧 Fixing database structure...")
            try:
                from fix_database_persistence import fix_database_persistence
                success = fix_database_persistence()
                print(f"{'✅ Success' if success else '❌ Failed'}")
            except Exception as e:
                print(f"❌ Error: {e}")

    else:
        print("⚠️ No backups available and database is empty")
        print("💡 Suggestions:")
        print("   - Upload your PDF files again to create new data")
        print("   - Consider enabling automatic backups")
        print("   - Use Docker ChromaDB for better reliability")