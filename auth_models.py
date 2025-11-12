import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta
import bcrypt
import jwt
import os
import secrets
import logging
from typing import Optional, Dict, Any

class AuthManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.sessions_collection = None
        self.secret_key = os.getenv("JWT_SECRET", "your-secret-key-change-this")
        self.algorithm = "HS256"
        self.token_expiry_hours = 24
        self.refresh_token_expiry_days = 7

    def connect(self):
        """Connect to MongoDB"""
        try:
            # Use existing MongoDB connection if available
            mongodb_uri = os.getenv("MONGODB_URI", os.getenv("MONGO_URL", "mongodb://localhost:27017"))
            database_name = os.getenv("DATABASE_NAME", "rag_pdf_auth")

            # Log all environment variables for debugging
            logging.info("🔍 Environment variables:")
            logging.info(f"   MONGODB_URI: {os.getenv('MONGODB_URI', 'NOT_SET')}")
            logging.info(f"   MONGO_URL: {os.getenv('MONGO_URL', 'NOT_SET')}")

            # Log the connection string (mask password)
            masked_uri = mongodb_uri.replace("rNcxrYpEyxpZajJUidlsZjrVgFqpEDmc", "***PASSWORD***")
            logging.info(f"📡 Attempting to connect to MongoDB with URI: {masked_uri}")
            logging.info(f"📊 Database name: {database_name}")

            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.users_collection = self.db["users"]
            self.sessions_collection = self.db["sessions"]

            # Create indexes
            self.users_collection.create_index("username", unique=True)
            self.users_collection.create_index("role")
            self.sessions_collection.create_index("token")
            self.sessions_collection.create_index("user_id")

            logging.info("✅ Connected to MongoDB for authentication")
            return True

        except Exception as e:
            logging.error(f"❌ MongoDB connection error: {e}")
            return False

    def create_default_admin(self) -> bool:
        """Create default admin account if not exists"""
        try:
            existing_admin = self.users_collection.find_one({"role": "admin"})
            if existing_admin:
                logging.info("ℹ️ Admin account already exists")
                return True

            # Default admin credentials
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt())

            admin_user = {
                "username": "admin",
                "password_hash": hashed_password.decode('utf-8'),
                "role": "admin",
                "profile": {
                    "full_name": "System Administrator"
                },
                "settings": {
                    "default_provider": "ollama",
                    "usage_count": 0,
                    "last_login": None,
                    "created_at": datetime.utcnow(),
                    "is_active": True
                }
            }

            self.users_collection.insert_one(admin_user)
            logging.info("✅ Default admin account created")
            logging.info(f"   Username: admin")
            logging.info(f"   Password: {admin_password}")
            logging.warning("⚠️  Please change the default admin password!")

            return True

        except Exception as e:
            logging.error(f"❌ Error creating admin account: {e}")
            return False

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except:
            return False

    def generate_tokens(self, user_id: str) -> Dict[str, Any]:
        """Generate JWT tokens"""
        try:
            # Access token
            access_payload = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                "iat": datetime.utcnow(),
                "type": "access"
            }
            access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)

            # Refresh token
            refresh_payload = {
                "user_id": user_id,
                "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expiry_days),
                "iat": datetime.utcnow(),
                "type": "refresh"
            }
            refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_in": self.token_expiry_hours * 3600
            }

        except Exception as e:
            logging.error(f"❌ Error generating tokens: {e}")
            return {}

    def create_session(self, user_id: str, tokens: Dict[str, Any], ip_address: str) -> Optional[str]:
        """Create session record"""
        try:
            session = {
                "user_id": user_id,
                "token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "ip_address": ip_address,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                "is_active": True
            }

            result = self.sessions_collection.insert_one(session)
            return str(result.inserted_id)

        except Exception as e:
            logging.error(f"❌ Error creating session: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                return None

            # Check if session exists and is active
            session = self.sessions_collection.find_one({
                "token": token,
                "is_active": True,
                "expires_at": {"$gt": datetime.utcnow()}
            })

            if not session:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logging.error(f"❌ Token validation error: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        try:
            user = self.users_collection.find_one({
                "username": username,
                "settings.is_active": True
            })

            if not user:
                return None

            if not self.verify_password(password, user['password_hash']):
                return None

            # Update last login
            self.users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"settings.last_login": datetime.utcnow()}}
            )

            return {
                "user_id": str(user["_id"]),
                "username": user["username"],
                "role": user["role"],
                "profile": user.get("profile", {}),
                "settings": user.get("settings", {})
            }

        except Exception as e:
            logging.error(f"❌ Authentication error: {e}")
            return None

    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating session"""
        try:
            result = self.sessions_collection.update_one(
                {"token": token},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0

        except Exception as e:
            logging.error(f"❌ Logout error: {e}")
            return False

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = self.users_collection.find_one({
                "_id": user_id,
                "settings.is_active": True
            })
            return user

        except Exception as e:
            logging.error(f"❌ Error getting user: {e}")
            return None

    def update_user_usage(self, user_id: str, action: str, metadata: Dict = None) -> bool:
        """Update user usage statistics"""
        try:
            self.users_collection.update_one(
                {"_id": user_id},
                {"$inc": {"settings.usage_count": 1}}
            )

            # Log usage
            if metadata:
                usage_log = {
                    "user_id": user_id,
                    "action": action,
                    "metadata": metadata or {},
                    "timestamp": datetime.utcnow()
                }
                # Store usage logs if needed for analytics
                # self.usage_logs_collection.insert_one(usage_log)

            return True

        except Exception as e:
            logging.error(f"❌ Error updating usage: {e}")
            return False

    def check_rate_limit(self, user_id: str, action: str, limit_per_hour: int = 100) -> bool:
        """Check if user has exceeded rate limit for an action"""
        try:
            from datetime import datetime, timedelta

            # Get usage count in the last hour
            hour_ago = datetime.utcnow() - timedelta(hours=1)

            # For simplicity, we'll use user's usage_count and assume it tracks recent usage
            # In production, you might want a more sophisticated rate limiting system
            user = self.users_collection.find_one({
                "_id": user_id,
                "settings.is_active": True
            })

            if not user:
                return False

            # Admin users bypass rate limits
            if user.get("role") == "admin":
                return True

            # Check usage (this is simplified - in production you'd track timestamped usage)
            current_usage = user.get("settings", {}).get("usage_count", 0)

            # Reset usage count if it's getting high (simple reset mechanism)
            if current_usage > limit_per_hour:
                self.users_collection.update_one(
                    {"_id": user_id},
                    {"$set": {"settings.usage_count": 0}}
                )
                return True

            return True

        except Exception as e:
            logging.error(f"❌ Rate limit check error: {e}")
            return True  # Allow on error (fail open)

    def log_activity(self, user_id: str, action: str, details: Dict = None):
        """Log user activity for security auditing"""
        try:
            activity_log = {
                "user_id": user_id,
                "action": action,
                "details": details or {},
                "timestamp": datetime.utcnow(),
                "ip_address": details.get("ip_address", "unknown") if details else "unknown",
                "user_agent": details.get("user_agent", "unknown") if details else "unknown"
            }

            # Store in activity_logs collection
            activity_logs_collection = self.db["activity_logs"]
            activity_logs_collection.insert_one(activity_log)

        except Exception as e:
            logging.error(f"❌ Activity logging error: {e}")

# Authentication decorator
def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        # This is a simple implementation
        # In production, you might want to check tokens, sessions, etc.
        return func(*args, **kwargs)
    return wrapper

# Global auth manager instance
auth_manager = AuthManager()