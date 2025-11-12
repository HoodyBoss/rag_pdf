#!/usr/bin/env python3
"""
Authentication System Test Script

This script tests the authentication system components.
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

def test_mongodb_connection():
    """Test MongoDB connection"""
    logger.info("📡 Testing MongoDB connection...")

    try:
        if auth_manager.connect():
            logger.info("✅ MongoDB connection successful")
            return True
        else:
            logger.error("❌ MongoDB connection failed")
            return False
    except Exception as e:
        logger.error(f"❌ MongoDB connection error: {e}")
        return False

def test_admin_creation():
    """Test default admin account creation"""
    logger.info("👤 Testing admin account creation...")

    try:
        if auth_manager.create_default_admin():
            logger.info("✅ Admin account creation successful")
            return True
        else:
            logger.error("❌ Admin account creation failed")
            return False
    except Exception as e:
        logger.error(f"❌ Admin account creation error: {e}")
        return False

def test_authentication():
    """Test user authentication"""
    logger.info("🔐 Testing user authentication...")

    try:
        # Test with default credentials
        username = "admin"
        password = os.getenv("ADMIN_PASSWORD", "admin123")

        user = auth_manager.authenticate_user(username, password)

        if user:
            logger.info(f"✅ Authentication successful for user: {user['username']}")
            logger.info(f"   Role: {user['role']}")
            logger.info(f"   User ID: {user['user_id']}")
            return user
        else:
            logger.error("❌ Authentication failed")
            return None
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return None

def test_token_generation(user):
    """Test JWT token generation"""
    logger.info("🎟️ Testing token generation...")

    try:
        if not user:
            logger.error("❌ No user provided for token generation")
            return False

        tokens = auth_manager.generate_tokens(user['user_id'])

        if tokens:
            logger.info("✅ Token generation successful")
            logger.info(f"   Access token length: {len(tokens.get('access_token', ''))}")
            logger.info(f"   Refresh token length: {len(tokens.get('refresh_token', ''))}")
            logger.info(f"   Expires in: {tokens.get('expires_in', 0)} seconds")
            return tokens
        else:
            logger.error("❌ Token generation failed")
            return None
    except Exception as e:
        logger.error(f"❌ Token generation error: {e}")
        return None

def test_token_validation(tokens):
    """Test JWT token validation"""
    logger.info("🔍 Testing token validation...")

    try:
        if not tokens or not tokens.get('access_token'):
            logger.error("❌ No tokens provided for validation")
            return False

        payload = auth_manager.validate_token(tokens['access_token'])

        if payload:
            logger.info("✅ Token validation successful")
            logger.info(f"   User ID: {payload.get('user_id')}")
            logger.info(f"   Token type: {payload.get('type')}")
            return True
        else:
            logger.error("❌ Token validation failed")
            return False
    except Exception as e:
        logger.error(f"❌ Token validation error: {e}")
        return False

def test_rate_limiting(user):
    """Test rate limiting functionality"""
    logger.info("🚦 Testing rate limiting...")

    try:
        if not user:
            logger.error("❌ No user provided for rate limiting test")
            return False

        # Test rate limit check (should allow admin)
        allowed = auth_manager.check_rate_limit(user['user_id'], "test_action")

        if allowed:
            logger.info("✅ Rate limiting check passed")
            return True
        else:
            logger.error("❌ Rate limiting check failed")
            return False
    except Exception as e:
        logger.error(f"❌ Rate limiting error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Authentication System Tests...")

    results = []

    # Test MongoDB connection
    results.append(test_mongodb_connection())

    # Test admin creation
    results.append(test_admin_creation())

    # Test authentication
    user = test_authentication()
    results.append(bool(user))

    # Test token generation
    tokens = test_token_generation(user)
    results.append(bool(tokens))

    # Test token validation
    results.append(test_token_validation(tokens))

    # Test rate limiting
    results.append(test_rate_limiting(user))

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if passed == total:
        print("ALL TESTS PASSED! Authentication system is ready.")
        return 0
    else:
        print("Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())