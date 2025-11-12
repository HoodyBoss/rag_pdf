#!/usr/bin/env python3
"""
Authentication Database Initialization Script

This script initializes the MongoDB database for authentication
and creates a default admin account.
"""

import os
import sys
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth_models import auth_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the authentication database"""
    try:
        logger.info("🚀 Starting authentication database initialization...")

        # Connect to MongoDB
        logger.info("📡 Connecting to MongoDB...")
        if not auth_manager.connect():
            logger.error("❌ Failed to connect to MongoDB")
            return False

        # Create default admin account
        logger.info("👤 Creating default admin account...")
        if auth_manager.create_default_admin():
            logger.info("✅ Default admin account created successfully")
        else:
            logger.warning("⚠️  Failed to create admin account or already exists")

        # Test authentication with default credentials
        logger.info("🔐 Testing authentication...")
        test_username = "admin"
        test_password = os.getenv("ADMIN_PASSWORD", "admin123")

        user = auth_manager.authenticate_user(test_username, test_password)
        if user:
            logger.info(f"✅ Authentication test successful for user: {user['username']}")
            logger.info(f"   Role: {user['role']}")
            logger.info(f"   User ID: {user['user_id']}")
        else:
            logger.error("❌ Authentication test failed")

        # Generate test tokens
        logger.info("🎟️  Testing token generation...")
        tokens = auth_manager.generate_tokens(user['user_id'] if user else "test")
        if tokens:
            logger.info("✅ Token generation successful")
            logger.info(f"   Access token: {tokens['access_token'][:50]}...")
            logger.info(f"   Expires in: {tokens['expires_in']} seconds")
        else:
            logger.error("❌ Token generation failed")

        logger.info("🎉 Authentication database initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def show_admin_info():
    """Show admin account information"""
    try:
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database_name = os.getenv("DATABASE_NAME", "rag_pdf_auth")
        admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

        print("\n" + "="*60)
        print("🔐 AUTHENTICATION SETUP INFORMATION")
        print("="*60)
        print(f"MongoDB URI: {mongodb_uri}")
        print(f"Database: {database_name}")
        print(f"Admin Username: admin")
        print(f"Admin Password: {admin_password}")
        print("="*60)
        print("⚠️  IMPORTANT SECURITY NOTES:")
        print("1. Change the default admin password immediately!")
        print("2. Set ADMIN_PASSWORD environment variable for production")
        print("3. Configure MONGODB_URI for production database")
        print("4. Set JWT_SECRET environment variable for token security")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Error showing admin info: {e}")

def main():
    """Main initialization function"""
    print("🚀 Initializing Authentication Database...")

    # Show setup information
    show_admin_info()

    # Initialize database
    if init_database():
        print("\n✅ Authentication system is ready!")
        print("\nNext steps:")
        print("1. Run the authenticated application: python authenticated_app.py")
        print("2. Login with username 'admin' and your password")
        print("3. Change the default admin password")
        return 0
    else:
        print("\n❌ Authentication system initialization failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())