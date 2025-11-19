#!/usr/bin/env python3
"""
API Key Configuration Module
"""
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from database import db_manager

logger = logging.getLogger(__name__)

class APIKeyConfig(BaseModel):
    """API Key Configuration Model"""
    provider: str
    api_key: str
    base_url: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class APIKeyManager:
    """API Key Management System"""

    def __init__(self):
        self.configured_keys = {}
        self._load_existing_keys()

    def _load_existing_keys(self):
        """Load existing API keys from MongoDB and environment variables"""
        # Try to load from MongoDB first
        if db_manager.is_connected():
            try:
                mongodb_keys = db_manager.get_all_api_keys()
                for provider, key_data in mongodb_keys.items():
                    self.configured_keys[provider] = APIKeyConfig(
                        provider=key_data["provider"],
                        api_key=key_data["api_key"],
                        base_url=key_data.get("base_url"),
                        is_active=key_data.get("is_active", True),
                        created_at=key_data.get("created_at"),
                        updated_at=key_data.get("updated_at")
                    )
                logger.info(f"✅ Loaded {len(mongodb_keys)} API keys from MongoDB")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load API keys from MongoDB: {e}")

        # Fallback to environment variables
        provider_env_mapping = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "manus": "MANUS_API_KEY",
            "zhipu": "ZHIPU_API_KEY"
        }

        for provider, env_var in provider_env_mapping.items():
            if provider not in self.configured_keys:  # Only load if not already in MongoDB
                api_key = os.getenv(env_var)
                if api_key and api_key.strip():
                    self.configured_keys[provider] = APIKeyConfig(
                        provider=provider,
                        api_key=api_key,
                        is_active=True
                    )

    def get_supported_providers(self) -> List[Dict[str, Any]]:
        """Get list of supported AI providers"""
        providers = [
            {
                "id": "openai",
                "name": "OpenAI (ChatGPT)",
                "description": "GPT-4, GPT-4o, GPT-3.5 Turbo",
                "base_url": "https://api.openai.com/v1",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "required": True
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Gemini 2.5, 2.0, 1.5 models",
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                "required": True
            },
            {
                "id": "minimax",
                "name": "Minimax",
                "description": "Chinese AI models",
                "base_url": "https://api.minimax.chat/v1",
                "models": ["abab6.5", "abab6.5s", "abab5.5"],
                "required": True
            },
            {
                "id": "manus",
                "name": "Manus AI",
                "description": "Code, reasoning, vision models",
                "base_url": "https://api.manus.ai/v1",
                "models": ["manus-code", "manus-reasoning", "manus-vision"],
                "required": True
            },
            {
                "id": "zhipu",
                "name": "Zhipu AI (GLM)",
                "description": "GLM-4.6, GLM-4 models",
                "base_url": "https://api.z.ai/api/paas/v4",
                "models": ["GLM-4.6", "glm-4.6", "glm-4", "glm-4v", "glm-3-turbo"],
                "required": True
            },
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "description": "Local LLM server - no API key required",
                "base_url": "http://localhost:11434",
                "models": ["gemma3:latest", "llama3.1:latest", "qwen2.5:latest", "mistral:latest", "phi3:latest"],
                "required": False
            }
        ]

        # Add configuration status
        for provider in providers:
            provider_id = provider["id"]
            if provider_id in self.configured_keys:
                provider["configured"] = True
                provider["has_api_key"] = bool(self.configured_keys[provider_id].api_key)
                provider["is_active"] = self.configured_keys[provider_id].is_active
            else:
                provider["configured"] = False
                provider["has_api_key"] = False
                provider["is_active"] = False

        return providers

    def save_api_key(self, provider: str, api_key: str, base_url: Optional[str] = None) -> Dict[str, Any]:
        """Save or update API key for a provider"""
        try:
            # Validate provider
            supported_providers = [p["id"] for p in self.get_supported_providers()]
            if provider not in supported_providers:
                return {
                    "success": False,
                    "message": f"❌ Provider {provider} ไม่รองรับ"
                }

            # Validate API key
            if not api_key or not api_key.strip():
                return {
                    "success": False,
                    "message": "❌ API key ไม่สามารถว่างได้"
                }

            # Save to MongoDB first
            result = db_manager.save_api_key(provider, api_key.strip(), base_url)

            if result["success"]:
                # Update local cache
                config = APIKeyConfig(
                    provider=provider,
                    api_key=api_key.strip(),
                    base_url=base_url,
                    is_active=True,
                    updated_at=datetime.now().isoformat()
                )
                self.configured_keys[provider] = config

                # Update environment variable (for current session)
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "minimax": "MINIMAX_API_KEY",
                    "manus": "MANUS_API_KEY",
                    "zhipu": "ZHIPU_API_KEY"
                }

                if provider in env_var_map:
                    os.environ[env_var_map[provider]] = api_key.strip()

            return result

        except Exception as e:
            logger.error(f"Error saving API key: {e}")
            return {
                "success": False,
                "message": f"❌ บันทึก API key ล้มเหลว: {str(e)}"
            }

    def delete_api_key(self, provider: str) -> Dict[str, Any]:
        """Delete API key for a provider"""
        try:
            # Delete from MongoDB first
            result = db_manager.delete_api_key(provider)

            if result["success"]:
                # Remove from local cache
                if provider in self.configured_keys:
                    del self.configured_keys[provider]

                # Remove from environment
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "minimax": "MINIMAX_API_KEY",
                    "manus": "MANUS_API_KEY",
                    "zhipu": "ZHIPU_API_KEY"
                }

                if provider in env_var_map and env_var_map[provider] in os.environ:
                    del os.environ[env_var_map[provider]]

            return result

        except Exception as e:
            logger.error(f"Error deleting API key: {e}")
            return {
                "success": False,
                "message": f"❌ ลบ API key ล้มเหลว: {str(e)}"
            }

    def test_api_key(self, provider: str) -> Dict[str, Any]:
        """Test API key connectivity"""
        try:
            # Use database manager for testing if available
            if db_manager.is_connected():
                return db_manager.test_api_key_connection(provider)

            # Fallback to local testing
            if provider not in self.configured_keys:
                return {
                    "success": False,
                    "message": "❌ ไม่พบ API key สำหรับ provider นี้"
                }

            # Test connectivity based on provider
            if provider == "ollama":
                return self._test_ollama_connection()
            elif provider == "openai":
                return self._test_openai_connection()
            elif provider == "gemini":
                return self._test_gemini_connection()
            else:
                return self._test_generic_connection(provider)

        except Exception as e:
            logger.error(f"Error testing API key: {e}")
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

    def _test_openai_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection"""
        try:
            import openai
            config = self.configured_keys["openai"]
            client = openai.OpenAI(api_key=config.api_key)

            # Simple test - list models
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

    def _test_gemini_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection"""
        try:
            import google.generativeai as genai
            config = self.configured_keys["gemini"]
            genai.configure(api_key=config.api_key)

            # Simple test - list models
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

    def _test_generic_connection(self, provider: str) -> Dict[str, Any]:
        """Test generic OpenAI-compatible API connection"""
        try:
            import openai
            config = self.configured_keys[provider]

            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )

            # Simple test
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

    def get_all_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Get all configured API keys"""
        # Get from MongoDB for latest data
        if db_manager.is_connected():
            try:
                mongodb_keys = db_manager.get_all_api_keys()
                return mongodb_keys
            except Exception as e:
                logger.warning(f"⚠️ Failed to get API keys from MongoDB: {e}")

        # Fallback to local cache
        result = {}
        for provider, config in self.configured_keys.items():
            result[provider] = {
                "provider": config.provider,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "is_active": config.is_active,
                "created_at": config.created_at,
                "updated_at": config.updated_at
            }
        return result

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get overall configuration status"""
        providers = self.get_supported_providers()

        configured_count = sum(1 for p in providers if p["configured"])
        active_count = sum(1 for p in providers if p["is_active"])
        total_count = len(providers)

        return {
            "total_providers": total_count,
            "configured_providers": configured_count,
            "active_providers": active_count,
            "configuration_percentage": round((configured_count / total_count) * 100, 1) if total_count > 0 else 0,
            "providers": providers
        }

# Import datetime
from datetime import datetime

# Global instance
api_key_manager = APIKeyManager()