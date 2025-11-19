#!/usr/bin/env python3
"""
MongoDB Database Manager for RAG PDF Application
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

logger = logging.getLogger(__name__)

class DatabaseManager:
    """MongoDB Database Manager"""

    def __init__(self):
        self.client = None
        self.db = None
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = os.getenv("DATABASE_NAME", "rag_pdf")
        self.connect()

    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            return False
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            self.client = None
            self.db = None
            return False

    def is_connected(self) -> bool:
        """Check if database is connected"""
        if self.client is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False

    def get_collection(self, collection_name: str):
        """Get MongoDB collection"""
        if self.db is None:
            if not self.connect():
                raise Exception("Database not connected")
        return self.db[collection_name]

    # API Keys Management
    def save_api_key(self, provider: str, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """Save API key to MongoDB"""
        try:
            collection = self.get_collection("api_keys")

            # Check if provider already exists
            existing = collection.find_one({"provider": provider})

            api_key_data = {
                "provider": provider,
                "api_key": api_key.strip(),
                "base_url": base_url,
                "is_active": True,
                "updated_at": datetime.utcnow()
            }

            if existing:
                # Update existing
                api_key_data["created_at"] = existing.get("created_at", datetime.utcnow())
                collection.update_one(
                    {"provider": provider},
                    {"$set": api_key_data}
                )
                logger.info(f"✅ Updated API key for {provider}")
            else:
                # Insert new
                api_key_data["created_at"] = datetime.utcnow()
                collection.insert_one(api_key_data)
                logger.info(f"✅ Saved new API key for {provider}")

            return {
                "success": True,
                "message": f"✅ บันทึก API key สำหรับ {provider} สำเร็จ",
                "provider": provider
            }

        except Exception as e:
            logger.error(f"Error saving API key: {e}")
            return {
                "success": False,
                "message": f"❌ บันทึก API key ล้มเหลว: {str(e)}"
            }

    def get_all_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Get all API keys from MongoDB"""
        try:
            collection = self.get_collection("api_keys")
            cursor = collection.find({})

            result = {}
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])

                provider = doc["provider"]
                result[provider] = {
                    "provider": doc["provider"],
                    "api_key": doc["api_key"],
                    "base_url": doc.get("base_url"),
                    "is_active": doc.get("is_active", True),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None
                }

            return result

        except Exception as e:
            logger.error(f"Error retrieving API keys: {e}")
            return {}

    def get_api_key(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get specific API key from MongoDB"""
        try:
            collection = self.get_collection("api_keys")
            doc = collection.find_one({"provider": provider})

            if doc:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                return {
                    "provider": doc["provider"],
                    "api_key": doc["api_key"],
                    "base_url": doc.get("base_url"),
                    "is_active": doc.get("is_active", True),
                    "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                    "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving API key for {provider}: {e}")
            return None

    def delete_api_key(self, provider: str) -> Dict[str, Any]:
        """Delete API key from MongoDB"""
        try:
            collection = self.get_collection("api_keys")
            result = collection.delete_one({"provider": provider})

            if result.deleted_count > 0:
                logger.info(f"✅ Deleted API key for {provider}")
                return {
                    "success": True,
                    "message": f"✅ ลบ API key สำหรับ {provider} สำเร็จ"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ ไม่พบ API key สำหรับ {provider}"
                }

        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return {
                "success": False,
                "message": f"❌ ลบ API key ล้มเหลว: {str(e)}"
            }

    def test_api_key_connection(self, provider: str) -> Dict[str, Any]:
        """Test API key connectivity"""
        try:
            api_key_data = self.get_api_key(provider)
            if not api_key_data:
                return {
                    "success": False,
                    "message": "❌ ไม่พบ API key สำหรับ provider นี้"
                }

            # Test connectivity based on provider
            if provider == "ollama":
                return self._test_ollama_connection()
            elif provider == "openai":
                return self._test_openai_connection(api_key_data["api_key"])
            elif provider == "gemini":
                return self._test_gemini_connection(api_key_data["api_key"])
            else:
                return self._test_generic_connection(provider, api_key_data["api_key"], api_key_data.get("base_url"))

        except Exception as e:
            logger.error(f"Error testing API key connection: {e}")
            return {
                "success": False,
                "message": f"❌ ทดสอบ API key ล้มเหลว: {str(e)}"
            }

    def _test_ollama_connection(self) -> Dict[str, Any]:
        """Test Ollama connection"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "success": True,
                    "message": f"✅ Ollama connected - {len(models)} models available",
                    "details": {"models": [m["name"] for m in models]}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Ollama connection failed: {response.status_code}"
                }
        except:
            return {
                "success": False,
                "message": "❌ Ollama server not running or not accessible"
            }

    def _test_openai_connection(self, api_key: str) -> Dict[str, Any]:
        """Test OpenAI API connection"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            return {
                "success": True,
                "message": f"✅ OpenAI connected - {len(models.data)} models available"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ OpenAI connection failed: {str(e)}"
            }

    def _test_gemini_connection(self, api_key: str) -> Dict[str, Any]:
        """Test Gemini API connection"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            models = genai.list_models()
            return {
                "success": True,
                "message": f"✅ Gemini connected - {len(models)} models available"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Gemini connection failed: {str(e)}"
            }

    def _test_generic_connection(self, provider: str, api_key: str, base_url: Optional[str]) -> Dict[str, Any]:
        """Test generic OpenAI-compatible API connection"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            return {
                "success": True,
                "message": f"✅ {provider} connected - {len(models.data)} models available"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ {provider} connection failed: {str(e)}"
            }

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Global database manager instance
db_manager = DatabaseManager()