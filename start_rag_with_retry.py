#!/usr/bin/env python3
"""
Startup script with retry mechanism for MongoDB connection
"""
import time
import logging
from railway_rag import initialize_system, get_system_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_mongodb(max_retries=30, retry_delay=5):
    """Wait for MongoDB to be ready"""
    logger.info("Waiting for MongoDB to be ready...")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")

            if initialize_system():
                status = get_system_status()
                # Check if status contains "Connected" (string-based check)
                if "Connected" in status:
                    logger.info("MongoDB is ready!")
                    return True

            logger.info(f"MongoDB not ready, waiting {retry_delay} seconds...")
            time.sleep(retry_delay)

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)

    logger.error("Failed to connect to MongoDB after all retries")
    return False

if __name__ == "__main__":
    logger.info("Starting RAG system with MongoDB connection retry...")

    if wait_for_mongodb():
        logger.info("MongoDB connected successfully, starting web server...")
        from railway_rag import main
        main()
    else:
        logger.error("Could not start RAG system - MongoDB connection failed")
        exit(1)