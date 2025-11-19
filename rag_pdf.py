import base64
import gradio as gr
import os
import shutil
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
  
import torch
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available - using alternative AI providers")
import shortuuid
import logging
import re
import time
import discord
import gc  # For garbage collection
import psutil  # For memory monitoring

# Flask removed - now using FastAPI for webhooks (no longer needed)

try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    LINE_AVAILABLE = True
except ImportError:
    LINE_AVAILABLE = False
    print("⚠️ LINE bot SDK not available - LINE integration disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ Pandas not available - data analysis features disabled")

def log_with_time(message):
    """Log message with timestamp and timing information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return logging.info(f"[{timestamp}] {message}")

def measure_time(start_time, label):
    """Measure and log elapsed time from start_time"""
    elapsed = time.time() - start_time
    log_with_time(f"⏱️ {label}: {elapsed:.2f}s")
    return elapsed
import asyncio
import requests
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Union
import threading
import docx
from pathlib import Path
import json
import hashlib
from collections import deque
from datetime import datetime

# Authentication imports
try:
    from auth_models import auth_manager, require_auth
    from login_page import get_current_user_info, logout_current_user
    AUTH_ENABLED = True
    logging.info("✅ Authentication system loaded successfully")
except ImportError as e:
    AUTH_ENABLED = False
    logging.warning(f"⚠️ Authentication system not available: {e}")
    # Fallback functions if auth models not available
    def auth_manager():
        return None
    def get_current_user_info():
        return {"authenticated": False, "user": None, "token": None}
    def require_auth(func):
        return func
    def logout_current_user():
        return "Authentication not available"

# ===== CONFIGURATION MANAGEMENT =====

# Environment detection
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') == 'production' or os.getenv('DYNO') == 'app'
IS_LOCAL = os.getenv('DEPLOYMENT_ENV') == 'local' or not IS_RAILWAY
IS_GPU_AVAILABLE = torch.cuda.is_available() and os.getenv('CUDA_VISIBLE_DEVICES') != ''

# Model configuration from environment variables
CONFIG = {
    'embedding_model': os.getenv('EMBEDDING_MODEL',
        'all-MiniLM-L6-v2' if IS_RAILWAY else
        'intfloat/multilingual-e5-base' if IS_GPU_AVAILABLE else
        'all-MiniLM-L6-v2'),

    'embedding_device': os.getenv('EMBEDDING_DEVICE',
        'cpu' if IS_RAILWAY else ('cuda' if IS_GPU_AVAILABLE else 'cpu')),

    'enable_summarization': os.getenv('ENABLE_SUMMARIZATION',
        'false' if IS_RAILWAY else 'true').lower() == 'true',

    'summarization_model': os.getenv('SUMMARIZATION_MODEL',
        'StelleX/mt5-base-thaisum-text-summarization'),

    'chunk_size': int(os.getenv('CHUNK_SIZE', '800' if IS_RAILWAY else '1000')),
    'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '150' if IS_RAILWAY else '200')),
    'max_chunks': int(os.getenv('MAX_CHUNKS', '100' if IS_RAILWAY else '500')),

    'use_gpu': os.getenv('USE_GPU', 'false' if IS_RAILWAY else str(IS_GPU_AVAILABLE).lower()).lower() == 'true',
    'auto_cleanup_interval': int(os.getenv('AUTO_CLEANUP_INTERVAL', '20' if IS_RAILWAY else '50')),
}

def apply_environment_config():
    """Apply environment-specific configurations"""
    logging.info("="*60)
    logging.info("🔧 Loading deployment configuration")
    logging.info(f"📍 Environment: {'Railway' if IS_RAILWAY else 'Local/Other'}")
    logging.info(f"🧠 Embedding Model: {CONFIG['embedding_model']}")
    logging.info(f"💾 Device: {CONFIG['embedding_device']}")
    logging.info(f"📝 Summarization: {'Enabled' if CONFIG['enable_summarization'] else 'Disabled'}")
    logging.info(f"📦 Chunk Size: {CONFIG['chunk_size']} (overlap: {CONFIG['chunk_overlap']})")
    logging.info(f"🔢 Max Chunks: {CONFIG['max_chunks']}")
    logging.info(f"⚡ GPU Usage: {'Enabled' if CONFIG['use_gpu'] else 'Disabled'}")
    logging.info("="*60)

    # Apply environment-specific optimizations
    if IS_RAILWAY:
        logging.info("🚂 Applying Railway optimizations...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:32'
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
    elif CONFIG['use_gpu'] and IS_GPU_AVAILABLE:
        logging.info("🚀 GPU optimizations enabled...")
        torch.cuda.empty_cache()
    else:
        logging.info("💻 CPU-only mode...")

# Apply configurations on startup
apply_environment_config()

# Additional AI Provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logging.info("✅ Google Generative AI library available")
except ImportError as e:
    GEMINI_AVAILABLE = False
    logging.warning(f"⚠️ Google Generative AI library not available: {e}")
    logging.info("Install with: pip install google-generativeai")

# LightRAG imports for graph reasoning
try:
    from lightrag_integration import initialize_lightrag_system, query_with_graph_reasoning, multi_hop_reasoning, get_lightrag_status
    LIGHT_RAG_AVAILABLE = True
    logging.info("✅ LightRAG integration loaded successfully")
except ImportError as e:
    logging.warning(f"⚠️ LightRAG integration not available: {e}")
    LIGHT_RAG_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Image folder
TEMP_IMG="./data/images"
TEMP_VECTOR="./data/chromadb"
TEMP_VECTOR_BACKUP="./data/chromadb_backup"
# รายชื่อ Model ที่คุณมีบน Ollama
AVAILABLE_MODELS = ["gemma3:latest", "qwen3:latest","llama3.2:latest"]

# AI Provider Configuration
AI_PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "models": AVAILABLE_MODELS,
        "api_key_required": False,
        "default_model": "gemma3:latest"
    },
    "minimax": {
        "name": "Minimax",
        "models": ["abab6.5", "abab6.5s", "abab5.5"],
        "api_key_required": True,
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "abab6.5"
    },
    "manus": {
        "name": "Manus",
        "models": ["manus-code", "manus-reasoning", "manus-vision"],
        "api_key_required": True,
        "api_key_env": "MANUS_API_KEY",
        "base_url": "https://api.manus.ai/v1",
        "default_model": "manus-code"
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "api_key_required": True,
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "default_model": "gemini-2.5-pro"
    },
    "zhipu": {
        "name": "Zhipu AI (GLM)",
        "models": ["GLM-4.6", "glm-4.6", "glm-4", "glm-4v", "glm-3-turbo"],
        "api_key_required": True,
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "default_model": "GLM-4.6"
    },
    "chatgpt": {
        "name": "ChatGPT (OpenAI)",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "api_key_required": True,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o"
    }
}

# Default AI Provider
DEFAULT_AI_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", "ollama")

def get_ai_provider_config(provider_name: str) -> dict:
    """Get AI provider configuration"""
    return AI_PROVIDERS.get(provider_name, AI_PROVIDERS["ollama"])

def get_available_providers() -> list:
    """Get list of available AI providers"""
    providers = []
    for key, config in AI_PROVIDERS.items():
        if key == "ollama":
            providers.append(key)
        elif config["api_key_required"]:
            api_key = os.getenv(config["api_key_env"])
            if api_key and api_key.strip():
                # Special check for Gemini - require library too
                if key == "gemini" and not GEMINI_AVAILABLE:
                    continue
                providers.append(key)
        else:
            providers.append(key)
    return providers

def get_provider_models(provider_name: str) -> list:
    """Get available models for a provider"""
    config = get_ai_provider_config(provider_name)
    return config.get("models", [])

def call_ai_provider(provider_name: str, model: str, messages: list, stream: bool = True, **kwargs):
    """
    Unified function to call different AI providers

    Args:
        provider_name: Name of the AI provider (ollama, minimax, manus, gemini, chatgpt)
        model: Model name to use
        messages: List of messages in format [{"role": "user", "content": "..."}]
        stream: Whether to stream the response
        **kwargs: Additional parameters for the specific provider

    Returns:
        Response stream or response object
    """
    config = get_ai_provider_config(provider_name)

    if provider_name == "ollama":
        # Check if Ollama is available
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama library not available. Install with: pip install ollama")

        # Use existing Ollama implementation
        try:
            return ollama.chat(
                model=model,
                messages=messages,
                stream=stream,
                options={
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "num_predict": kwargs.get("num_predict", 1500)
                }
            )
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            raise

    elif provider_name == "chatgpt":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")

        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            client = openai.OpenAI(api_key=api_key)

            if stream:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise

    elif provider_name == "gemini":
        if not GEMINI_AVAILABLE:
            error_msg = "Google Generative AI library not available. Please install with: pip install google-generativeai"
            logging.error(error_msg)
            return ({"message": {"content": error_msg}} for _ in range(1))

        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)

            # Convert messages to Gemini format
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"

            prompt += "Assistant: "

            response = model_obj.generate_content(
                prompt,
                stream=stream,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.3),
                    max_output_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            )

            # Convert Gemini stream to match Ollama/OpenAI format
            def gemini_stream_converter(gemini_response):
                for chunk in gemini_response:
                    if chunk.text:
                        yield {"message": {"content": chunk.text}}

            return gemini_stream_converter(response)
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            raise

    elif provider_name in ["minimax", "manus", "zhipu"]:
        # Generic OpenAI-compatible API implementation
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for {provider_name}. Set {config['api_key_env']} environment variable.")

        try:
            client = openai.OpenAI(
                api_key=api_key,
                base_url=config["base_url"]
            )

            if stream:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 0.9)
                )
        except Exception as e:
            logging.error(f"{provider_name} API error: {e}")
            raise

    else:
        raise ValueError(f"Unsupported AI provider: {provider_name}")

# Discord Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "YOUR_WEBHOOK_URL_HERE")  # ใส่ Webhook URL ที่นี่
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")  # ใส่ Bot Token ที่นี่
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID_HERE")  # ใส่ Channel ID ที่นี่
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"  # เปิด/ปิดการแจ้งเตือน Discord

# Discord Bot Configuration
DISCORD_BOT_ENABLED = os.getenv("DISCORD_BOT_ENABLED", "false").lower() == "true"  # เปิด/ปิด Bot สำหรับรับคำถาม
DISCORD_BOT_PREFIX = os.getenv("DISCORD_BOT_PREFIX", "!ask ")  # คำนำหน้าคำสั่ง
DISCORD_DEFAULT_MODEL = os.getenv("DISCORD_DEFAULT_MODEL", "gemma3:latest")  # โมเดลเริ่มต้นสำหรับ Discord
DISCORD_RESPOND_NO_PREFIX = os.getenv("DISCORD_RESPOND_NO_PREFIX", "true").lower() == "true"  # ตอบคำถามที่ไม่มี prefix หรือไม่
DISCORD_REPLY_MODE = os.getenv("DISCORD_REPLY_MODE", "channel")  # วิธีการตอบกลับ: channel/dm/both

# LINE OA Configuration
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_ENABLED = os.getenv("LINE_ENABLED", "false").lower() == "true"  # เปิด/ปิด LINE OA
LINE_DEFAULT_MODEL = os.getenv("LINE_DEFAULT_MODEL", "gemini-2.0-flash-exp")  # โมเดลเริ่มต้นสำหรับ LINE (ใช้ Gemini แทน Ollama บน Railway)
LINE_WEBHOOK_PORT = int(os.getenv("LINE_WEBHOOK_PORT", "5000"))  # Port สำหรับ LINE webhook

# Facebook Messenger Configuration
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "YOUR_FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "YOUR_FB_VERIFY_TOKEN")
FB_ENABLED = os.getenv("FB_ENABLED", "false").lower() == "true"  # เปิด/ปิด Facebook Messenger
FB_DEFAULT_MODEL = os.getenv("FB_DEFAULT_MODEL", "gemini-2.0-flash-exp")  # โมเดลเริ่มต้นสำหรับ FB (ใช้ Gemini แทน Ollama บน Railway)
FB_WEBHOOK_PORT = int(os.getenv("FB_WEBHOOK_PORT", "5001"))  # Port สำหรับ FB webhook
FB_WEBHOOK = int(os.getenv("FB_WEBHOOK", "5001"))  # Port สำหรับ FB webhook

# Enhanced RAG Configuration (MemoRAG-like features)
RAG_MODE = os.getenv("RAG_MODE", "enhanced")  # standard, enhanced
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))  # จำนวน conversations ที่จำไว้
ENABLE_CONTEXT_CHAINING = os.getenv("ENABLE_CONTEXT_CHAINING", "true").lower() == "true"
ENABLE_REASONING = os.getenv("ENABLE_REASONING", "true").lower() == "true"
ENABLE_SESSION_MEMORY = os.getenv("ENABLE_SESSION_MEMORY", "true").lower() == "true"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Chroma client Disable telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# สร้างโฟลเดอร์สำหรับ database ถ้ายังไม่มี
os.makedirs(TEMP_VECTOR, exist_ok=True)
os.makedirs(TEMP_VECTOR_BACKUP, exist_ok=True)

# ตั้งค่า ChromaDB ให้เก็บข้อมูลแบบถาวรพร้อมการจัดการดีขึ้น
settings = Settings(
    anonymized_telemetry=False,
    allow_reset=False,
    is_persistent=True
)

def cleanup_database_locks():
    """Clean up ChromaDB lock files that may cause access issues"""
    import glob
    import os
    import time
    import shutil

    try:
        db_path = TEMP_VECTOR
        if not os.path.exists(db_path):
            return True

        log_with_time("🔧 Cleaning up database lock files...")

        # More aggressive lock file removal
        lock_patterns = [
            os.path.join(db_path, "**", "*.lock"),
            os.path.join(db_path, "**", "*-wal"),
            os.path.join(db_path, "**", "*-shm"),
            os.path.join(db_path, "**", "data_level0.bin"),
            os.path.join(db_path, "**", "*.db"),
            os.path.join(db_path, "**", "*.sqlite")
        ]

        removed_count = 0
        for pattern in lock_patterns:
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    # Try normal removal first
                    os.remove(file_path)
                    log_with_time(f"   Removed: {os.path.basename(file_path)}")
                    removed_count += 1
                except PermissionError as pe:
                    # File is locked, try to force unlock by changing attributes
                    try:
                        import stat
                        os.chmod(file_path, stat.S_IWRITE)
                        os.remove(file_path)
                        log_with_time(f"   Force removed: {os.path.basename(file_path)}")
                        removed_count += 1
                    except:
                        log_with_time(f"   ⚠️ Cannot remove (locked): {os.path.basename(file_path)}")
                        # Try renaming as last resort
                        try:
                            backup_name = f"{file_path}.locked_{int(time.time())}"
                            os.rename(file_path, backup_name)
                            log_with_time(f"   Renamed locked file: {os.path.basename(file_path)}")
                        except:
                            log_with_time(f"   ❌ Cannot access: {os.path.basename(file_path)}")
                except Exception as e:
                    log_with_time(f"   Could not remove {file_path}: {e}")

        if removed_count > 0:
            log_with_time(f"✅ Removed {removed_count} lock files")
            time.sleep(2)  # Wait longer for file system to release
        else:
            log_with_time("ℹ️ No lock files found")

        return True

    except Exception as e:
        log_with_time(f"Error cleaning lock files: {e}")
        return False

def force_release_chromadb():
    """Force release ChromaDB by killing related connections and files"""
    try:
        log_with_time("🚨 Force releasing ChromaDB...")

        # Close any existing connections
        try:
            if 'collection' in globals():
                collection = None
                del collection
            if 'chroma_client' in globals():
                chroma_client = None
                del chroma_client
            log_with_time("Closed existing ChromaDB connections")
        except:
            pass

        # Cleanup lock files more aggressively
        cleanup_database_locks()

        # Kill any Python processes that might be holding the lock
        import subprocess
        import psutil
        import os

        try:
            current_pid = os.getpid()
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['pid'] != current_pid and proc.info['name'] == 'python.exe':
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'rag_pdf.py' in cmdline or 'chromadb' in cmdline.lower():
                            log_with_time(f"Killing process {proc.info['pid']} that might be holding DB lock")
                            proc.terminate()
                            proc.wait(timeout=3)
                except:
                    pass
        except:
            pass

        # Force garbage collection multiple times
        import gc
        gc.collect()
        gc.collect()

        log_with_time("✅ Force release completed")
        return True
    except Exception as e:
        log_with_time(f"Force release failed: {e}")
        return False

def move_database_to_temp():
    """Move locked database to temp location to avoid conflicts"""
    try:
        import shutil
        import tempfile

        if os.path.exists(TEMP_VECTOR):
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="chromadb_backup_")
            temp_db_path = os.path.join(temp_dir, "chromadb")

            log_with_time(f"Moving locked DB to temp location: {temp_db_path}")

            # Move the entire database directory
            shutil.move(TEMP_VECTOR, temp_db_path)

            # Create empty database directory
            os.makedirs(TEMP_VECTOR, exist_ok=True)

            log_with_time(f"✅ Database moved to: {temp_db_path}")
            return temp_db_path
        return None
    except Exception as e:
        log_with_time(f"Failed to move database: {e}")
        return None

def create_fresh_database():
    """Create a completely fresh database to avoid lock issues"""
    try:
        import shutil
        import tempfile

        log_with_time("Creating fresh database to avoid lock issues...")

        # If database exists, move it to temp first
        moved_path = move_database_to_temp()

        # Create new empty database directory
        os.makedirs(TEMP_VECTOR, exist_ok=True)

        # Wait a moment for file system to sync
        import time
        time.sleep(1)

        log_with_time("✅ Fresh database created successfully")
        return True
    except Exception as e:
        log_with_time(f"Failed to create fresh database: {e}")
        return False

# Clean up locks before initializing database
cleanup_database_locks()

chroma_client = chromadb.PersistentClient(
    path=TEMP_VECTOR,
    settings=settings
)

# สร้างหรือดึง collection ที่มีอยู่แล้ว
try:
    # พยายามโหลด collection ที่มีอยู่ก่อน
    collection = chroma_client.get_collection(name="pdf_data")
    count = collection.count()
    logging.info(f"✅ โหลด collection 'pdf_data' สำเร็จ - จำนวน {count} เรคคอร์ด")
    logging.info(f"📁 Database path: {TEMP_VECTOR}")
    logging.info(f"💾 Database exists: {os.path.exists(TEMP_VECTOR)}")

    # ตรวจสอบว่ามีข้อมูลจริงหรือไม่
    if count == 0:
        logging.warning("⚠️ Collection is empty! Checking for data persistence issues...")

        # เรียกใช้ฟังก์ชันตรวจสอบ
        try:
            from fix_database_persistence import scan_collection_directories, get_sqlite_collection_info, reconstruct_collection
        except ImportError:
            logging.warning("⚠️ fix_database_persistence module not found, skipping persistence check")
            count = 0  # Set to 0 to indicate no data available

        collection_dirs = scan_collection_directories()
        sqlite_collections = get_sqlite_collection_info()

        sqlite_ids = {coll[0] for coll in sqlite_collections}
        has_data_dirs = [col['id'] for col in collection_dirs if col['has_data']]

        if sqlite_collections and has_data_dirs:
            logging.info("🔧 Attempting to recover data from file system...")
            # พยายากใช้ข้อมูลจาก directories ที่มีข้อมูล
            for coll_id, coll_name in sqlite_collections:
                if coll_id in has_data_dirs:
                    logging.info(f"   Trying to restore: {coll_name}")
                    # TODO: Implement data recovery from directories
        else:
            logging.warning("❌ No recoverable data found - database appears to be empty")

except Exception as e:
    # ถ้าไม่มีให้สร้างใหม่
    logging.info(f"❌ ไม่พบ collection ที่มีอยู่ - กำลังสร้างใหม่: {str(e)}")
    collection = chroma_client.get_or_create_collection(name="pdf_data")
    logging.info(f"✅ สร้าง collection 'pdf_data' ใหม่สำเร็จ")

# Feedback Database Setup
FEEDBACK_DB_PATH = "./data/feedback.db"
os.makedirs(os.path.dirname(FEEDBACK_DB_PATH), exist_ok=True)

import sqlite3
from datetime import datetime

def init_feedback_db():
    """สร้างฐานข้อมูล feedback ถ้ายังไม่มี"""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback_type TEXT NOT NULL,  -- 'good' or 'bad'
            user_comment TEXT,
            corrected_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            sources TEXT  -- JSON array of source info
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_feedback_question ON feedback(question)
    ''')

    # ตรวจสอบและเพิ่มคอลัมน์ใหม่ถ้าจำเป็น
    try:
        cursor.execute('ALTER TABLE feedback ADD COLUMN applied BOOLEAN DEFAULT FALSE')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_applied ON feedback(applied)')
        logging.info("✅ Added 'applied' column to feedback table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logging.info("✅ 'applied' column already exists in feedback table")
        else:
            logging.warning(f"⚠️ Error adding 'applied' column: {str(e)}")

    # ตารางสำหรับเก็บคู่คำถาม-คำตอบที่ถูกแก้ไข
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corrected_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_question TEXT NOT NULL,
            original_answer TEXT NOT NULL,
            corrected_answer TEXT NOT NULL,
            feedback_id INTEGER,
            question_embedding TEXT,  -- embedding สำหรับการค้นหาคำถามที่คล้ายกัน
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            applied_count INTEGER DEFAULT 0,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id)
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_corrected_question ON corrected_answers(original_question)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_corrected_created ON corrected_answers(created_at)
    ''')

    # Tag System Tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            color TEXT DEFAULT '#007bff',
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,  -- ChromaDB ID
            tag_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tag_id) REFERENCES tags (id),
            UNIQUE(document_id, tag_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES feedback (id),
            FOREIGN KEY (tag_id) REFERENCES tags (id),
            UNIQUE(feedback_id, tag_id)
        )
    ''')

    # Indexes for tag performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_tags_doc_id ON document_tags(document_id)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_tags_tag_id ON document_tags(tag_id)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_feedback_tags_feedback_id ON feedback_tags(feedback_id)
    ''')

    # Insert default tags if empty
    cursor.execute("SELECT COUNT(*) FROM tags")
    if cursor.fetchone()[0] == 0:
        default_tags = [
            ('ทั่วไป', '#6c757d', 'คำถามทั่วไป'),
            ('เทคนิค', '#007bff', 'คำถามด้านเทคนิค'),
            ('การใช้งาน', '#28a745', 'คำถามเกี่ยวกับการใช้งาน'),
            ('ปัญหา', '#dc3545', 'คำถามเกี่ยวกับปัญหา'),
            ('ข้อมูล', '#17a2b8', 'คำถามเกี่ยวกับข้อมูล'),
            ('สอบถาม', '#ffc107', 'คำถามเพื่อสอบถามข้อมูล'),
            ('แก้ไข', '#fd7e14', 'คำถามที่ต้องการแก้ไข'),
            ('สำคัญ', '#e83e8c', 'คำถามสำคัญ')
        ]

        cursor.executemany('''
            INSERT INTO tags (name, color, description) VALUES (?, ?, ?)
        ''', default_tags)
        logging.info("✅ Created default tags")

    # Enhanced Memory System Tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT DEFAULT 'default',
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_embedding BLOB,  -- Store as bytes
            contexts TEXT,  -- JSON array
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            relevance_score REAL DEFAULT 0.0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rag_mode TEXT,  -- 'standard' or 'enhanced'
            question TEXT,
            response_time REAL,  -- milliseconds
            context_count INTEGER,
            memory_hit BOOLEAN,
            success BOOLEAN,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS context_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_hash TEXT UNIQUE NOT NULL,
            question TEXT NOT NULL,
            contexts TEXT,  -- JSON array of cached contexts
            embedding BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1
        )
    ''')

    # Indexes for performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session_memory_session ON session_memory(session_id)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session_memory_timestamp ON session_memory(timestamp)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_rag_performance_session ON rag_performance_log(session_id)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_context_cache_hash ON context_cache(question_hash)
    ''')

    conn.commit()
    conn.close()
    logging.info("✅ Feedback database initialized with learning, tag, and enhanced memory features")

# Initialize feedback database
init_feedback_db()

# ตั้งค่า device with memory optimization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Global variables for lazy loading
sentence_model = None
sum_model = None
sum_tokenizer = None

# Create directory for storing images
os.makedirs(TEMP_IMG, exist_ok=True)

def load_embedding_model():
    """Lazy load embedding model with configurable optimization"""
    global sentence_model
    if sentence_model is None:
        model_name = CONFIG['embedding_model']
        device = CONFIG['embedding_device']

        logging.info(f"🔄 Loading embedding model: {model_name} on {device}")

        # Configure cache based on deployment
        cache_folder = '/tmp/embeddings_cache' if IS_RAILWAY else None

        try:
            sentence_model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=cache_folder
            )

            # Apply optimizations based on deployment
            if IS_RAILWAY or not CONFIG['use_gpu']:
                sentence_model.eval()
                gc.collect()
                logging.info(f"✅ Embedding model loaded (CPU optimized): {model_name}")
            else:
                # GPU optimizations
                if device == 'cuda' and torch.cuda.is_available():
                    sentence_model.half()  # Use 16-bit precision
                    torch.cuda.empty_cache()
                logging.info(f"✅ Embedding model loaded (GPU optimized): {model_name}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to load {model_name}, falling back to MiniLM-L6-v2: {e}")
            # Fallback to smallest model
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logging.info("✅ Fallback model loaded successfully")

    return sentence_model

def load_summarization_model():
    """Configurable summarization model loading"""
    global sum_model, sum_tokenizer

    if not CONFIG['enable_summarization']:
        logging.info("⚠️ Summarization disabled by configuration")
        return None, None

    if sum_model is None:
        model_name = CONFIG['summarization_model']
        logging.info(f"🔄 Loading summarization model: {model_name}")

        try:
            sum_tokenizer = MT5Tokenizer.from_pretrained(model_name)
            sum_model = MT5ForConditionalGeneration.from_pretrained(model_name)

            # Configure device based on deployment
            if IS_RAILWAY or not CONFIG['use_gpu']:
                device = 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            sum_model = sum_model.to(device)
            sum_model.eval()

            # GPU optimizations if applicable
            if device == 'cuda':
                sum_model = sum_model.half()  # Use 16-bit precision
                torch.cuda.empty_cache()

            gc.collect()
            logging.info(f"✅ Summarization model loaded on {device}: {model_name}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to load {model_name}: {e}")
            logging.info("🔄 Falling back to simple summarization...")
            sum_model = None
            sum_tokenizer = None

    return sum_model, sum_tokenizer

def cleanup_models():
    """Unload models from memory to free up RAM"""
    global sentence_model, sum_model, sum_tokenizer
    if sentence_model is not None:
        del sentence_model
        sentence_model = None
    if sum_model is not None:
        del sum_model
        sum_model = None
    if sum_tokenizer is not None:
        del sum_tokenizer
        sum_tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()
    logging.info("🧹 Models unloaded from memory")

def get_memory_usage():
    """Get current memory usage information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

    return {
        'ram_mb': memory_mb,
        'gpu_mb': gpu_memory,
        'total_gpu_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 if torch.cuda.is_available() else None
    }

def optimize_memory():
    """Railway-optimized memory cleanup"""
    # Force aggressive garbage collection for Railway
    gc.collect()
    gc.collect()  # Run twice for thorough cleanup

    # Clear GPU cache if available (not needed on Railway)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log memory usage (Railway friendly)
    try:
        mem_info = get_memory_usage()
        logging.info(f"💾 Railway Memory - RAM: {mem_info['ram_mb']:.1f}MB" +
                    (f", GPU: {mem_info['gpu_mb']:.1f}MB" if mem_info['gpu_mb'] else ""))
    except:
        logging.info("💾 Memory optimized for Railway")

def railway_auto_cleanup():
    """Automatic cleanup to reduce Railway costs"""
    # Unload models after 5 minutes of inactivity
    cleanup_models()

    # Clear caches
    gc.collect()
    logging.info("🧹 Railway auto-cleanup completed to reduce costs")

def summarize_content(content: str) -> str:
    """
    Railway-optimized: Simple text summarization without heavy models
    """
    logging.info("%%%%%%%%%%%%%% SUMMARY (Railway Mode) %%%%%%%%%%%%%%%%%%%%%")

    # Railway: Use simple truncation instead of ML model
    if len(content) <= 300:
        return content

    # Simple extractive summarization for Railway (saves 1.2GB RAM)
    sentences = content.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 3:
        return content

    # Take first 2 sentences + last sentence (extractive summary)
    summary = '. '.join(sentences[:2] + sentences[-1:]) + '.'

    # Limit summary length
    if len(summary) > 500:
        summary = summary[:500] + '...'

    logging.info(f"Railway summary: {len(summary)} chars (saved 1.2GB RAM)")
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")

    return summary

# แยกเนื้อหา, รูป ออกจาก PDF
def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    แยกข้อความและรูปภาพจาก PDF โดยใช้ PyMuPDF
    """
    try:
        doc = fitz.open(pdf_path)
        content_chunks = []
        all_text=[]

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text("text").strip()
            all_text.append(f"{text} \n\n\n")
            if not text:
                text = f"ไม่มีข้อความในหน้า {page_num + 1}"
            
            logging.info("################# Text data ##################")
            chunk_data = {"text": f"ข้อมูลจากหน้า {page_num + 1} : {text}" , "images": []}
            
            # Extract images
            image_list = page.get_images(full=True)
            logging.info("################# images list ##################")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    img_id = f"pic_{str(page_num+1)}_{str(img_index+1)}"
                    img_path = f"{TEMP_IMG}/{img_id}.{image_ext}"
                    image.save(img_path, format=image_ext.upper())

                    img_desc = f"รูปภาพ จากหน้า {str(page_num+1)} ของ รูปที่ {str(img_index+1)}, บริบทข้อความ: {text[:80]}..."  
                    chunk_data["text"] += f"\n[ภาพ: {img_id}.{image_ext}]"                    
                    chunk_data["images"].append({
                        "data": image,
                        "path": img_path,
                        "description": img_desc
                    })
                except Exception as e:
                    logger.warning(f"ไม่สามารถประมวลผลรูปภาพที่หน้า {str(page_num+1)}, รูปที่ {str(img_index+1)}: {str(e)}")
            
            if chunk_data["text"]:
                content_chunks.append(chunk_data)
        
        if not any(chunk["images"] for chunk in content_chunks):
            logger.warning("ไม่พบรูปภาพใน PDF: %s", pdf_path)
        
        doc.close()
        content_text= "".join(all_text)
        # ตัดคำภาษาไทย
        thaitoken_text = preprocess_thai_text(content_text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
        print("################################")
        print(f"{ thaitoken_text }")
        print("################################")
        global summarize
        summarize = summarize_content(thaitoken_text)
        return content_chunks
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดในการแยก PDF: %s", str(e))
        raise

# ตัดคำภาษาไทย 
def preprocess_thai_text(text: str) -> str:
    """
    ตัดคำภาษาไทยด้วย pythainlp เพื่อเตรียมข้อความ

    Args:
        text (str): ข้อความภาษาไทย

    Returns:
        str: ข้อความที่ตัดคำแล้ว
    """
    return " ".join(word_tokenize(text, engine="newmm"))


def embed_text(text: str) -> np.ndarray:
    """
    Configurable text embedding using SentenceTransformer

    Args:
        text (str): ข้อความที่ต้องการสร้าง embedding

    Returns:
        np.ndarray: Embedding vector ที่รวมจากหลายโมเดล
    """
    logging.info("-------------- start embed text  -------------------")

    # Lazy load embedding model with configuration
    model = load_embedding_model()

    # ตัดคำภาษาไทย
    processed_text = preprocess_thai_text(text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text

    # สร้าง embedding ด้วย device จาก configuration
    device = CONFIG['embedding_device']

    sentence_embedding = model.encode(
        processed_text,
        normalize_embeddings=True,
        device=device,
        batch_size=1,
        show_progress_bar=False
    )

    # Memory optimization for Railway
    if IS_RAILWAY:
        gc.collect()

    return sentence_embedding

def store_in_chroma(content_chunks: List[Dict], source_name: str):
    """
    เก็บข้อมูลข้อความและรูปภาพใน Chroma พร้อม embedding
    รองรับรูปแบบใหม่ (chunks with metadata) และรูปแบบเก่า (backward compatibility)
    """
    logging.info(f"##### Start store {len(content_chunks)} chunks in chroma #########")

    if not content_chunks:
        logging.warning("No chunks to store")
        return

    # ตรวจสอบว่าเป็นรูปแบบใหม่หรือเก่า
    if isinstance(content_chunks[0], dict) and "text" in content_chunks[0] and "metadata" in content_chunks[0]:
        # รูปแบบใหม่: chunks พร้อม metadata แบบละเอียด
        store_chunks_modern(content_chunks)
    else:
        # รูปแบบเก่า: chunks พร้อม images (เพื่อความเข้ากันได้)
        store_chunks_legacy(content_chunks, source_name)


def store_chunks_modern(chunks: List[Dict]):
    """เก็บ chunks แบบใหม่ที่มี metadata แบบละเอียด"""
    logging.info("Storing chunks in modern format")

    for chunk in chunks:
        try:
            text = chunk["text"]
            chunk_metadata = chunk["metadata"]
            chunk_id = chunk["id"]

            logging.info(f"Processing chunk: {chunk_id} ({len(text)} chars)")

            # สร้าง embedding
            text_embedding = embed_text(text)

            # สร้าง metadata สำหรับ ChromaDB
            metadata = {
                "type": "text",
                "source": chunk_metadata.get("source", "unknown"),
                "file_type": chunk_metadata.get("file_type", "unknown"),
                "start": chunk_metadata.get("start", 0),
                "end": chunk_metadata.get("end", 0),
                "chunk_id": chunk_id
            }

            # เก็บใน ChromaDB
            collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[text_embedding.tolist()],
                ids=[chunk_id]
            )

            logging.info(f"✅ Stored chunk {chunk_id}")

        except Exception as e:
            logging.error(f"❌ Failed to store chunk {chunk.get('id', 'unknown')}: {str(e)}")


def store_chunks_legacy(chunks: List[Dict], source_name: str):
    """เก็บ chunks แบบเก่า (เพื่อความเข้ากันได้กับ PDF รูปแบบเดิม)"""
    logging.info("Storing chunks in legacy format")

    for chunk in chunks:
        try:
            text = chunk["text"]
            images = chunk.get("images", [])

            logging.info(f"Processing legacy chunk ({len(text)} chars)")

            # สร้าง embedding
            text_embedding = embed_text(text)
            text_id = shortuuid.uuid()[:8]

            # เก็บข้อความ
            collection.add(
                documents=[text],
                metadatas=[{
                    "type": "text",
                    "source": source_name,
                    "format": "legacy"
                }],
                embeddings=[text_embedding.tolist()],
                ids=[text_id]
            )

            # เก็บรูปภาพ (ถ้ามี)
            for img in images:
                try:
                    img_id = shortuuid.uuid()[:8]
                    img_path = img.get("path", "")

                    collection.add(
                        documents=[text],
                        metadatas=[{
                            "type": "image",
                            "source": source_name,
                            "image_path": img_path,
                            "description": img.get("description", ""),
                            "format": "legacy"
                        }],
                        embeddings=[text_embedding.tolist()],
                        ids=[img_id]
                    )
                    logging.info(f"✅ Stored image {img_id}")

                except Exception as e:
                    logging.error(f"❌ Failed to store image: {str(e)}")

        except Exception as e:
            logging.error(f"❌ Failed to store legacy chunk: {str(e)}")

def extract_text_from_file(file_path: str) -> str:
    """
    แยกข้อความจากไฟล์ต่างๆ (PDF, TXT, MD, DOCX)

    Args:
        file_path: พาธของไฟล์

    Returns:
        str: ข้อความที่แยกได้
    """
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return extract_pdf_text(file_path)
        elif file_ext in ['.txt', '.md']:
            return extract_text_file(file_path)
        elif file_ext == '.docx':
            return extract_docx_text(file_path)
        else:
            logging.warning(f"ไม่รองรับไฟล์ประเภท: {file_ext}")
            return ""

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความจาก {file_path}: {str(e)}")
        return ""


def extract_pdf_text(pdf_path: str) -> str:
    """แยกข้อความจาก PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความ PDF: {str(e)}")
        return ""


def extract_text_file(file_path: str) -> str:
    """แยกข้อความจากไฟล์ .txt หรือ .md"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                return f.read()
        except Exception as e:
            logging.error(f"ไม่สามารถอ่านไฟล์ {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_path}: {str(e)}")
        return ""


def extract_docx_text(docx_path: str) -> str:
    """แยกข้อความจากไฟล์ .docx"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความ DOCX: {str(e)}")
        return ""


def process_multiple_files(files, clear_before_upload: bool = False):
    """
    ประมวลผลไฟล์หลายไฟล์ (PDF, TXT, MD, DOCX)

    Args:
        files: list ของไฟล์จาก Gradio
        clear_before_upload: ล้างข้อมูลเก่าหรือไม่
    """
    try:
        if not files:
            return "❌ กรุณาเลือกไฟล์เพื่ออัปโหลด"

        current_count = collection.count()
        logging.info(f"เริ่มประมวลผล {len(files)} ไฟล์")

        # สำรองข้อมูลก่อนการเปลี่ยนแปลง (Enhanced Backup)
        auto_backup_result = auto_backup_before_operation()
        if not auto_backup_result["success"]:
            logging.warning(f"Auto backup failed: {auto_backup_result.get('error')}")
        else:
            logging.info(f"Auto backup created: {auto_backup_result['backup_name']}")

        # ถ้าเลือกล้างข้อมูลเก่า
        if clear_before_upload:
            logging.info("กำลังล้างข้อมูลเก่าตามคำร้อง...")
            clear_vector_db()

        total_chunks = 0
        successful_files = []
        failed_files = []

        # ประมวลผลทีละไฟล์
        for file_obj in files:
            try:
                file_path = file_obj.name
                file_name = os.path.basename(file_path)
                file_ext = Path(file_path).suffix.lower()

                logging.info(f"#### กำลังประมวลผลไฟล์: {file_name} ({file_ext}) ####")

                # แยกข้อความจากไฟล์
                text_content = extract_text_from_file(file_path)

                if not text_content.strip():
                    failed_files.append(f"{file_name}: ไม่สามารถแยกข้อความได้")
                    continue

                # แยกข้อความเป็น chunks
                content_chunks = chunk_text(text_content, file_name)

                if not content_chunks:
                    failed_files.append(f"{file_name}: ไม่มีเนื้อหาที่สามารถประมวลผลได้")
                    continue

                # เก็บใน ChromaDB
                store_in_chroma(content_chunks, file_name)

                total_chunks += len(content_chunks)
                successful_files.append(file_name)

                logging.info(f"✅ ประมวลผล {file_name} สำเร็จ - {len(content_chunks)} chunks")

            except Exception as e:
                error_msg = f"{file_name}: {str(e)}"
                failed_files.append(error_msg)
                logging.error(f"❌ ประมวลผล {file_name} ล้มเหลว: {str(e)}")

        # สำรองข้อมูลอัตโนมัติหลังประมวลผลสำเร็จ
        backup_vector_db()

        new_count = collection.count()
        added_records = new_count - current_count

        # สร้างรายงานผลลัพธ์
        result = f"🎉 ประมวลผลเสร็จสิ้น!\n\n"
        result += f"📊 สรุปผล:\n"
        result += f"• ไฟล์ทั้งหมด: {len(files)} ไฟล์\n"
        result += f"• สำเร็จ: {len(successful_files)} ไฟล์\n"
        result += f"• ล้มเหลว: {len(failed_files)} ไฟล์\n"
        result += f"• Chunks ทั้งหมด: {total_chunks} ชิ้น\n"
        result += f"• Records ที่เพิ่ม: {added_records} เรคคอร์ด\n"
        result += f"• Records รวม: {new_count} เรคคอร์ด\n\n"

        if successful_files:
            result += f"✅ ไฟล์ที่ประมวลผลสำเร็จ:\n"
            for file_name in successful_files:
                result += f"  • {file_name}\n"
            result += "\n"

        if failed_files:
            result += f"❌ ไฟล์ที่ประมวลผลล้มเหลว:\n"
            for error in failed_files:
                result += f"  • {error}\n"
            result += "\n"

        result += f"💾 ข้อมูลถูกสำรองอัตโนมัติ"

        return result

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการประมวลผลหลายไฟล์: {str(e)}")
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"


# Google Sheets Integration
def extract_google_sheets_data(sheets_url: str) -> str:
    """
    ดึงข้อมูลจาก Google Sheets

    Args:
        sheets_url: URL ของ Google Sheets

    Returns:
        ข้อความที่จัดรูปแบบแล้ว
    """
    try:
        # แปลง URL ให้เป็น export URL
        sheet_id = extract_sheet_id_from_url(sheets_url)
        if not sheet_id:
            return "❌ ไม่สามารถแยก Sheet ID จาก URL ได้"

        # Export หน้าแรกเป็น CSV
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"

        # อ่านข้อมูลเป็น DataFrame
        df = pd.read_csv(export_url)

        # แปลงข้อมูลเป็นข้อความ
        text_content = format_dataframe_to_text(df, sheets_url)

        return text_content

    except Exception as e:
        logging.error(f"Error extracting Google Sheets data: {str(e)}")
        return f"❌ ไม่สามารถดึงข้อมูลจาก Google Sheets ได้: {str(e)}"


def extract_sheet_id_from_url(url: str) -> str:
    """
    แยก Sheet ID จาก Google Sheets URL

    Args:
        url: Google Sheets URL

    Returns:
        Sheet ID
    """
    try:
        # Pattern สำหรับ Google Sheets URL
        patterns = [
            r"/d/([a-zA-Z0-9-_]+)",
            r"id=([a-zA-Z0-9-_]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return ""
    except:
        return ""


def format_dataframe_to_text(df, source_url: str) -> str:
    """
    แปลง DataFrame เป็นข้อความที่อ่านง่ายและเหมาะกับการค้นหา
    """
    try:
        text_content = f"# ข้อมูลจาก Google Sheets\n"
        text_content += f"ที่มา: {source_url}\n"
        text_content += f"แถว: {len(df)}, คอลัมน์: {len(df.columns)}\n\n"

        # สร้างคำอธิบายคอลัมน์
        col_descriptions = []
        for i, col in enumerate(df.columns):
            col_descriptions.append(f"{col}")

        text_content += f"## หัวข้อที่ครอบคลุม: {', '.join(col_descriptions)}\n\n"

        # แปลงข้อมูลเป็นรูปแบบ Q&A และคำอธิบายที่เข้าใจง่าย
        text_content += "## รายละเอียดข้อมูล:\n\n"

        for index, row in df.iterrows():
            # สร้างคำอธิบายรวมสำหรับแถวนี้
            row_content = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value) or value == "":
                    continue

                # จัดรูปแบบค่า
                if isinstance(value, str) and len(value) > 100:
                    # ถ้าข้อความยาว ให้ตัดและเพิ่ม "..."
                    value = value[:150] + "..." if len(value) > 150 else value

                row_content.append(f"{col}: {value}")

            if row_content:
                # สร้างข้อความที่เชื่อมโยงข้อมูลในแถวเดียวกัน
                text_content += f"### ข้อมูลที่ {index + 1}:\n"
                text_content += f"{' '.join(row_content)}.\n\n"

                # สร้างรูปแบบ Q&A สำหรับการค้นหาที่ง่ายขึ้น
                if len(df.columns) >= 2:
                    # สมมติว่าคอลัมน์แรกคือคำถาม และคอลัมน์อื่นเป็นคำตอบ
                    first_col = df.columns[0]
                    question_value = row[first_col]
                    if not pd.isna(question_value) and str(question_value).strip():
                        text_content += f"**คำถาม/หัวข้อ:** {question_value}\n"

                        # รวบรวมคำตอบจากคอลัมน์อื่นๆ
                        answers = []
                        for col in df.columns[1:]:
                            answer_value = row[col]
                            if not pd.isna(answer_value) and str(answer_value).strip():
                                answers.append(f"{answer_value}")

                        if answers:
                            text_content += f"**คำตอบ/รายละเอียด:** {' '.join(answers)}\n"

                        text_content += "\n"

        return text_content

    except Exception as e:
        logging.error(f"Error formatting DataFrame: {str(e)}")
        return f"❌ เกิดข้อผิดพลาดในการจัดรูปแบบข้อมูล: {str(e)}"


def process_google_sheets_url(sheets_url: str, clear_before_upload: bool = False):
    """
    ประมวลผล Google Sheets URL

    Args:
        sheets_url: URL ของ Google Sheets
        clear_before_upload: ล้างข้อมูลเก่าหรือไม่
    """
    try:
        logging.info(f"Starting to process Google Sheets: {sheets_url}")

        # ตรวจสอบ URL
        if not sheets_url.strip():
            return "❌ กรุณาใส่ Google Sheets URL"

        # ดึงข้อมูลจาก Google Sheets
        text_content = extract_google_sheets_data(sheets_url)

        if text_content.startswith("❌"):
            return text_content

        # สร้างชื่อสำหรับ source
        sheet_name = f"Google_Sheets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # แบ่งข้อความเป็น chunks
        chunks = chunk_text(text_content, sheet_name)

        # ล้างข้อมูลเก่าถ้าจำเป็น
        if clear_before_upload:
            clear_vector_db()
            logging.info("Cleared vector database before upload")

        # เก็บ chunks ลงใน ChromaDB
        # ตรวจสอบว่าเป็นรูปแบบใหม่หรือเก่า
        if chunks and isinstance(chunks[0], dict) and "text" in chunks[0] and "metadata" in chunks[0]:
            # รูปแบบใหม่: chunks พร้อม metadata แบบละเอียด
            store_chunks_modern(chunks)
        else:
            # รูปแบบเก่า: chunks พร้อม source name
            store_chunks_legacy(chunks, sheet_name)

        # อัปเดตข้อมูลสรุป
        update_summary_data(chunks)

        result_msg = f"""✅ นำเข้าข้อมูลจาก Google Sheets สำเร็จ!
📊 URL: {sheets_url}
📝 ชื่อที่เก็บ: {sheet_name}
📄 จำนวน chunks: {len(chunks)}
📏 ความยาวทั้งหมด: {len(text_content):,} ตัวอักษร

ข้อมูลพร้อมใช้งานในระบบ RAG แล้ว!"""

        logging.info(f"Successfully processed Google Sheets: {sheet_name}")
        return result_msg

    except Exception as e:
        logging.error(f"Error processing Google Sheets URL: {str(e)}")
        return f"❌ เกิดข้อผิดพลาดในการประมวลผล Google Sheets: {str(e)}"


def chunk_text(text: str, source_file: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
    """
    Configurable text chunking based on deployment settings

    Args:
        text: ข้อความทั้งหมด
        source_file: ชื่อไฟล์ต้นทาง
        chunk_size: ขนาดของ chunk (uses CONFIG default if None)
        overlap: ส่วนที่ทับซ้อนระหว่าง chunks (uses CONFIG default if None)

    Returns:
        List[Dict]: list ของ chunks พร้อม metadata
    """
    # Use configured values if not provided
    chunk_size = chunk_size or CONFIG['chunk_size']
    overlap = overlap or CONFIG['chunk_overlap']
    max_chunks = CONFIG['max_chunks']

    if not text or len(text.strip()) < 30:
        return []

    chunks = []
    start = 0
    total_chunks = 0
    cleanup_interval = CONFIG['auto_cleanup_interval']

    logging.info(f"📦 Chunking {source_file}: size={chunk_size}, overlap={overlap}, max_chunks={max_chunks}")

    while start < len(text) and total_chunks < max_chunks:
        end = start + chunk_size

        # หาตำแหน่งตัดคำที่เหมาะสม
        if end < len(text):
            search_window = min(50, chunk_size // 4)  # Dynamic search window
            for i in range(end, max(start, end - search_window), -1):
                if text[i] in [' ', '\n', '.', '!', '?']:
                    end = i + 1
                    break

        chunk_text = text[start:end].strip()

        if len(chunk_text) > 30:
            chunk_id = f"{source_file}_{start}_{end}"

            chunks.append({
                "text": chunk_text,
                "id": chunk_id,
                "metadata": {
                    "source": source_file,
                    "start": start,
                    "end": end,
                    "file_type": Path(source_file).suffix.lower(),
                    "chunk_index": total_chunks
                }
            })
            total_chunks += 1

            # Configurable memory cleanup
            if total_chunks % cleanup_interval == 0:
                optimize_memory()
                if IS_RAILWAY:  # Extra cleanup for Railway
                    gc.collect()

        start = end - overlap if end - overlap > start else end

    deployment_type = "Railway" if IS_RAILWAY else "Local"
    logging.info(f"✅ {deployment_type}: Created {total_chunks} chunks from {source_file}")

    if total_chunks >= max_chunks:
        logging.warning(f"⚠️ Reached max chunks limit ({max_chunks}), document truncated")

    return chunks


class EnhancedRAG:
    """
    Enhanced RAG System with MemoRAG-like features:
    - Memory management for conversations
    - Context chaining
    - Session-based reasoning
    - Cross-reference analysis
    """

    def __init__(self):
        self.session_memory = deque(maxlen=MEMORY_WINDOW_SIZE)
        self.session_context = []
        self.conversation_history = []
        self.current_session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        session_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"session_{session_hash}"

    def add_to_memory(self, question: str, answer: str, contexts: List[str]):
        """Add conversation to memory"""
        memory_entry = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "contexts": contexts[:3]  # Store top 3 contexts
        }
        self.session_memory.append(memory_entry)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": memory_entry["timestamp"]
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": memory_entry["timestamp"]
        })

        logging.info(f"Added to memory: Session {self.current_session_id}")

    def get_relevant_memory(self, current_question: str) -> List[Dict]:
        """Get relevant memory entries based on current question"""
        if not ENABLE_SESSION_MEMORY:
            return []

        relevant_memories = []
        current_embedding = embed_text(current_question)

        for memory_entry in self.session_memory:
            # Calculate similarity between current question and memory
            memory_question = memory_entry["question"]
            memory_embedding = embed_text(memory_question)

            # Simple similarity check (can be improved with proper similarity calculation)
            similarity = np.dot(current_embedding, memory_embedding)

            if similarity > 0.7:  # Threshold for relevance
                relevant_memories.append({
                    "question": memory_entry["question"],
                    "answer": memory_entry["answer"],
                    "similarity": float(similarity),
                    "contexts": memory_entry["contexts"]
                })

        # Sort by similarity and return top results
        relevant_memories.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_memories[:3]  # Return top 3 relevant memories

    def build_context_prompt(self, question: str, contexts: List[str], relevant_memories: List[Dict],
                           show_source: bool = False, formal_style: bool = False) -> str:
        """Build enhanced prompt with memory and context chaining"""

        if not ENABLE_CONTEXT_CHAINING and not ENABLE_REASONING:
            # สร้าง prompt ตามสไตล์ที่เลือก
            if formal_style:
                style_instruction = "ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพ และชัดเจน"
                source_phrase = ""
                response_prefix = "คำตอบ:"
            else:
                style_instruction = "ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
                source_phrase = ""
                response_prefix = "คำตอบ:"

            source_instruction = ""
            if show_source:
                source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else "" if source_phrase else ""

            return f"""คุณเป็นผู้ช่วยตอบคำถามที่มีความรู้ด้านเอกสารที่อัปโหลดไว้ กรุณาตอบคำถามโดยอาศัยข้อมูลจากบริบทที่ให้มาเป็นหลัก

**แนวทางการตอบ:**
- {style_instruction}
- ให้คำตอบที่สอดคล้องกับข้อมูลในเอกสารเป็นหลัก
- หากข้อมูลในเอกสารไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด
- ตอบอย่างสมบูรณ์และมีประโยชน์{source_instruction}
- อาจเพิ่มคำอธิบายเล็กน้อยเพื่อความชัดเจน แต่ไม่ตีความเกินไป

**คำถาม:** {question}

**บริบทจากเอกสาร:**
{summarize}

{context}

{response_prefix}"""

        # Enhanced prompt with memory and reasoning
        prompt_parts = []

        # Add context about memory if available
        if relevant_memories:
            prompt_parts.append("## บริบทที่เกี่ยวข้อง (จาก Memory):")
            for i, memory in enumerate(relevant_memories, 1):
                prompt_parts.append(f"ครั้งที่ {i}:")
                prompt_parts.append(f"คำถาม: {memory['question']}")
                prompt_parts.append(f"คำตอบ: {memory['answer'][:200]}...")
                prompt_parts.append("")

        # Add current contexts
        if contexts:
            prompt_parts.append("## บริบทปัจจุบันจากเอกสาร:")
            prompt_parts.extend(contexts)

        # Add reasoning prompt if enabled
        if ENABLE_REASONING:
            # สร้าง prompt ตามสไตล์ที่เลือก
            if formal_style:
                style_instruction = "- ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพและชัดเจน"
                source_phrase = ""
                response_prefix = "คำตอบ:"
            else:
                style_instruction = "- ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
                source_phrase = ""
                response_prefix = "คำตอบ:"

            source_instruction = ""
            if show_source:
                source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else "" if source_phrase else ""

            prompt_parts.extend([
                "",
                "## แนวทางการตอบคำถาม:",
                style_instruction,
                "- ให้ความสำคัญกับข้อมูลจากเอกสารและบริบทที่จดจำไว้เป็นหลัก",
                "- ตอบให้สมบูรณ์และเป็นประโยชน์ โดยยังคงอยู่ในขอบเขตของข้อมูลที่มี",
                "- หากข้อมูลไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด",
                "- อาจเชื่อมโยงข้อมูลจากความจำเพื่อให้คำตอบสอดคล้องมากขึ้น" + source_instruction,
                "",
                "## การวิเคราะห์:",
                "1. พิจารณาคำถามปัจจุบันร่วมกับบริบทที่เกี่ยวข้อง",
                "2. ตรวจสอบความสอดคล้องและเชื่อมโยงข้อมูลต่างๆ",
                "3. ให้คำตอบที่ครอบคลุมและมีประโยชน์ที่สุด",
                "4. ใช้ภาษาที่เข้าใจง่ายและเป็นธรรมชาติ"
            ])

        prompt_parts.extend([
            "",
            f"## คำถามปัจจุบัน: {question}",
            "",
            f"## {response_prefix}:",
            "กรุณาตอบคำถามโดยพิจารณาจากบริบททั้งหมดที่ให้มา ให้คำตอบที่สมบูรณ์และเป็นประโยชน์ที่สุด"
        ])

        return "\n".join(prompt_parts)

    def reset_session(self):
        """Reset current session"""
        self.current_session_id = self._generate_session_id()
        self.session_context = []
        self.conversation_history = []
        logging.info(f"Session reset: New session ID = {self.current_session_id}")

    def get_session_info(self) -> Dict:
        """Get current session information"""
        return {
            "session_id": self.current_session_id,
            "memory_size": len(self.session_memory),
            "conversation_length": len(self.conversation_history),
            "current_context_size": len(self.session_context)
        }


# Global Enhanced RAG instances
enhanced_rag = EnhancedRAG()  # Legacy for backward compatibility

# Initialize RAG Manager after class definition
rag_manager = None

def initialize_rag_manager():
    """Initialize RAG Manager after all classes are defined"""
    global rag_manager
    if rag_manager is None:
        rag_manager = RAGManager()
    return rag_manager


def process_pdf_upload(pdf_file):
    """
    ฟังก์ชันเก่าสำหรับความเข้ากันได้ (deprecated - ใช้ process_multiple_files แทน)
    """
    if pdf_file:
        return process_multiple_files([pdf_file], False)
    return "กรุณาเลือกไฟล์"

def clear_vector_db():
    try:
        
       # Clear existing collection to avoid duplicates
        try:
            chroma_client.delete_collection(name="pdf_data")
        except:
            pass  # Collection might not exist
        global collection
        collection = chroma_client.get_or_create_collection(name="pdf_data")

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"
    
def update_summary_data(chunks: List[Dict]):
    """
    อัปเดตข้อมูลสรุปจาก chunks ที่อัปโหลด
    """
    try:
        global summarize

        # สรุปข้อมูลจาก chunks
        total_chars = sum(len(chunk.get('text', '')) for chunk in chunks)
        source_files = set(chunk.get('metadata', {}).get('source', 'Unknown') for chunk in chunks)

        # สร้างข้อความสรุป
        summary_text = f"""📊 ข้อมูลที่อัปโหลด:
• แหล่งที่มา: {', '.join(source_files)}
• จำนวน chunks: {len(chunks)}
• ขนาดรวม: {total_chars:,} ตัวอักษร
• อัปเดตล่าสุด: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        summarize = summary_text
        logging.info(f"Updated summary data: {len(chunks)} chunks from {len(source_files)} sources")

    except Exception as e:
        logging.error(f"Error updating summary data: {str(e)}")
        # ถ้าเกิดข้อผิดพลาด ให้ใช้ค่าเริ่มต้น
        summarize = f"📊 ข้อมูล PDF: {len(chunks)} chunks, อัปเดต {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def clear_vector_db_and_images():
    """
    ล้างข้อมูลใน Chroma vector database และไฟล์ในโฟลเดอร์ images
    """
    
    try:
        clear_vector_db()
        
        pdf_input.clear()
        if os.path.exists(TEMP_IMG):
            shutil.rmtree(TEMP_IMG)
            os.makedirs(TEMP_IMG, exist_ok=True)
        
        return "ล้างข้อมูลใน vector database และโฟลเดอร์ images สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"


def extract_images_from_answer(answer: str):
    """
    ดึงพาธของรูปภาพที่เกี่ยวข้องจากคำตอบ

    Args:
        answer: ข้อความคำตอบ

    Returns:
        list: รายชื่อพาธของรูปภาพที่พบ
    """
    import re

    # ใช้ regex เพื่อดึงชื่อไฟล์ที่อยู่ใน [ภาพ: ...]
    pattern1 = r"\[(?:ภาพ:\s*)?(pic_\w+[-_]?\w*\.(?:jpe?g|png))\]"
    pattern2 = r"(pic_\w+[-_]?\w*\.(?:jpe?g|png))"

    # ค้นหารูปภาพในคำตอบ
    image_list = re.findall(pattern1, answer)
    if len(image_list) == 0:
        image_list = re.findall(pattern2, answer)

    # ดึงเฉพาะรูปที่ไม่ซ้ำกัน
    image_list_unique = list(dict.fromkeys(image_list))

    # สร้างเป็นพาธเต็มและตรวจสอบว่ามีไฟล์อยู่จริง
    valid_image_paths = []
    for img in image_list_unique:
        img_path = f"{TEMP_IMG}/{img}"
        if os.path.exists(img_path):
            valid_image_paths.append(img_path)
            logging.info(f"Found relevant image: {img_path}")

    return valid_image_paths


async def send_to_discord(question: str, answer: str):
    """
    ส่งคำถามและคำตอบไปยัง Discord channel พร้อมรูปภาพที่เกี่ยวข้อง (ถ้ามี)
    """
    if not DISCORD_ENABLED or DISCORD_WEBHOOK_URL == "YOUR_WEBHOOK_URL_HERE":
        logging.info("Discord integration is disabled or not configured")
        return

    try:
        # ดึงรูปภาพที่เกี่ยวข้องจากคำตอบ
        image_paths = extract_images_from_answer(answer)

        # ใช้ Webhook URL โดยตรง
        webhook_url = DISCORD_WEBHOOK_URL

        embed = discord.Embed(
            title="📚 RAG PDF Bot - คำถามใหม่",
            color=discord.Color.blue()
        )
        embed.add_field(name="❓ คำถาม", value=question, inline=False)
        embed.add_field(name="💬 คำตอบ", value=answer[:1024] + "..." if len(answer) > 1024 else answer, inline=False)

        # เพิ่มข้อมูลว่ามีรูปภาพประกอบหรือไม่
        if image_paths:
            embed.add_field(name="🖼️ รูปภาพที่เกี่ยวข้อง", value=f"พบ {len(image_paths)} รูปภาพที่เกี่ยวข้อง", inline=False)

        embed.set_footer(text="PDF RAG Assistant")

        # สร้าง payload สำหรับ Discord webhook
        payload_data = {
            "embeds": [embed.to_dict()]
        }

        # ถ้ามีรูปภาพ ให้แนบไปกับข้อความ
        if image_paths:
            # Discord webhook รองรับการแนบไฟล์ได้สูงสุด 10 ไฟล์
            files_to_send = image_paths[:10]  # จำกัดไว้ 10 รูป

            # สร้าง multipart/form-data payload
            files = {}
            for i, img_path in enumerate(files_to_send):
                try:
                    with open(img_path, 'rb') as f:
                        files[f'file{i}'] = (os.path.basename(img_path), f.read(), 'image/png')
                except Exception as e:
                    logging.error(f"Failed to read image {img_path}: {str(e)}")

            if files:
                # ส่งพร้อมไฟล์แนบ
                response = requests.post(
                    webhook_url,
                    files=files,
                    data={'payload_json': json.dumps(payload_data)},
                    timeout=30
                )
            else:
                # ถ้าไม่สามารถอ่านไฟล์ได้ ส่งเฉพาะ embed
                response = requests.post(webhook_url, json=payload_data, timeout=10)
        else:
            # ส่งเฉพาะ embed ถ้าไม่มีรูป
            response = requests.post(webhook_url, json=payload_data, timeout=10)

        if response.status_code == 204:
            logging.info("ส่งข้อความไปยัง Discord สำเร็จ")
            if image_paths:
                logging.info(f"ส่งรูปภาพ {len(image_paths)} รูปไปยัง Discord สำเร็จ")
        else:
            logging.error(f"ไม่สามารถส่งข้อความไปยัง Discord: {response.status_code} - {response.text}")

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่งข้อความไปยัง Discord: {str(e)}")


def send_to_discord_sync(question: str, answer: str):
    """
    ฟังก์ชันสำหรับเรียกใช้ discord แบบ synchronous
    """
    try:
        # สร้าง event loop ใหม่ถ้าจำเป็น
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # ถ้า loop กำลังทำงานอยู่ ใช้ create_task
            asyncio.create_task(send_to_discord(question, answer))
        else:
            # ถ้า loop ไม่ทำงาน ใช้ run_until_complete
            loop.run_until_complete(send_to_discord(question, answer))
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการเรียก Discord: {str(e)}")


def backup_vector_db():
    """
    สำรองข้อมูล vector database
    """
    try:
        if not os.path.exists(TEMP_VECTOR):
            log_with_time("ไม่พบโฟลเดอร์ database ที่จะสำรองข้อมูล")
            return False

        # Force release ChromaDB before backup
        force_release_chromadb()
        time.sleep(1)  # Wait for files to be released

        # สร้าง timestamp สำหรับ backup และตรวจสอบว่าซ้ำหรือไม่
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_backup_name = f"backup_{timestamp}"
        backup_folder = os.path.join(TEMP_VECTOR_BACKUP, base_backup_name)

        # หาชื่อที่ไม่ซ้ำกัน
        counter = 1
        while os.path.exists(backup_folder):
            backup_name = f"{base_backup_name}_{counter}"
            backup_folder = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            counter += 1

        # คัดลอกข้อมูลไปโฟลเดอร์ backup
        shutil.copytree(TEMP_VECTOR, backup_folder)
        log_with_time(f"สำรองข้อมูลสำเร็จ: {backup_folder}")

        # ลบ backup เก่าเกินกว่า 7 วัน
        cleanup_old_backups()

        return True
    except Exception as e:
        log_with_time(f"ไม่สามารถสำรองข้อมูลได้: {str(e)}")
        return False


def cleanup_old_backups(days_to_keep=7):
    """
    ลบข้อมูล backup ที่เก่าเกินกว่าที่กำหนด
    """
    try:
        now = datetime.now()

        for backup_name in os.listdir(TEMP_VECTOR_BACKUP):
            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            if os.path.isdir(backup_path) and backup_name.startswith("backup_"):
                # ดึง timestamp จากชื่อโฟลเดอร์
                try:
                    timestamp_str = backup_name.replace("backup_", "")
                    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    # ลบถ้าเกินกว่าจำนวนวันที่กำหนด
                    if (now - backup_time).days > days_to_keep:
                        shutil.rmtree(backup_path)
                        logging.info(f"ลบ backup เก่า: {backup_name}")
                except ValueError:
                    continue
    except Exception as e:
        logging.error(f"ไม่สามารถลบ backup เก่าได้: {str(e)}")


def backup_database_enhanced(backup_name=None, include_memory=False):
    """
    สำรองข้อมูล database แบบขั้นสูง
    """
    try:
        import json

        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"enhanced_backup_{timestamp}"

        # Ensure unique backup name
        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
        counter = 1
        original_backup_name = backup_name
        while os.path.exists(backup_path):
            backup_name = f"{original_backup_name}_{counter}"
            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            counter += 1

        os.makedirs(backup_path, exist_ok=False)  # Don't allow exist to prevent conflicts

        # Backup main database
        if os.path.exists(TEMP_VECTOR):
            backup_db_path = os.path.join(backup_path, "chromadb")
            shutil.copytree(TEMP_VECTOR, backup_db_path)
            logging.info(f"สำรอง ChromaDB: {backup_db_path}")

        # Backup Enhanced RAG memory if requested
        if include_memory and RAG_MODE == "enhanced":
            try:
                memory_info = enhanced_rag.get_memory_info()
                backup_memory_path = os.path.join(backup_path, "memory_info.json")

                with open(backup_memory_path, 'w', encoding='utf-8') as f:
                    json.dump(memory_info, f, ensure_ascii=False, indent=2)

                # Also backup memory collection if it exists
                if hasattr(enhanced_rag, 'memory_collection'):
                    memory_collection_backup = os.path.join(backup_path, "memory_collection")
                    os.makedirs(memory_collection_backup, exist_ok=True)

                    # Get all memories from collection
                    memories = enhanced_rag.memory_collection.get()
                    memory_json_path = os.path.join(memory_collection_backup, "memories.json")

                    with open(memory_json_path, 'w', encoding='utf-8') as f:
                        json.dump(memories, f, ensure_ascii=False, indent=2)

                logging.info("สำรอง Enhanced RAG Memory สำเร็จ")
            except Exception as e:
                logging.warning(f"ไม่สามารถสำรอง Memory: {str(e)}")

        # Create backup metadata
        metadata = {
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "type": "enhanced",
            "includes_memory": include_memory,
            "rag_mode": RAG_MODE,
            "database_info": get_database_info() if os.path.exists(TEMP_VECTOR) else None
        }

        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"สร้าง Enhanced Backup สำเร็จ: {backup_path}")

        # Clean old backups
        cleanup_old_backups()

        return {
            "success": True,
            "backup_name": backup_name,
            "backup_path": backup_path,
            "metadata": metadata
        }

    except Exception as e:
        logging.error(f"ไม่สามารถสร้าง Enhanced Backup: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def restore_database_enhanced(backup_name=None):
    """
    กู้คืน database แบบขั้นสูง
    """
    try:
        import json

        if backup_name is None:
            # หา backup ล่าสุด
            backups = [d for d in os.listdir(TEMP_VECTOR_BACKUP)
                      if os.path.isdir(os.path.join(TEMP_VECTOR_BACKUP, d))]
            if not backups:
                return {"success": False, "error": "ไม่พบข้อมูล backup"}
            backup_name = sorted(backups)[-1]

        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            return {"success": False, "error": f"ไม่พบ backup: {backup_name}"}

        # Validate backup integrity before restore
        is_valid, validation_message = validate_backup_integrity(backup_path)
        if not is_valid:
            logging.error(f"Backup validation failed: {validation_message}")
            return {"success": False, "error": f"Backup ไม่ถูกต้อง: {validation_message}"}

        logging.info(f"Backup validation passed: {backup_name}")

        # Load backup metadata
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded backup metadata from: {metadata_path}")
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON in backup metadata: {str(e)}")
                metadata = {
                    "backup_name": backup_name,
                    "type": "unknown",
                    "includes_memory": False,
                    "rag_mode": "unknown"
                }
            except Exception as e:
                logging.warning(f"Could not read backup metadata: {str(e)}")
                metadata = {
                    "backup_name": backup_name,
                    "type": "unknown",
                    "includes_memory": False,
                    "rag_mode": "unknown"
                }

        # Create emergency backup of current data
        try:
            emergency_backup = backup_database_enhanced(
                f"emergency_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if not emergency_backup["success"]:
                logging.warning(f"Emergency backup failed: {emergency_backup.get('error')}")
                emergency_backup = {"success": False, "error": "Emergency backup failed"}
        except Exception as e:
            logging.error(f"Emergency backup creation failed: {str(e)}")
            emergency_backup = {"success": False, "error": str(e)}

        # Restore main database
        backup_db_path = os.path.join(backup_path, "chromadb")
        if os.path.exists(backup_db_path):
            if os.path.exists(TEMP_VECTOR):
                shutil.rmtree(TEMP_VECTOR)
            shutil.copytree(backup_db_path, TEMP_VECTOR)
            logging.info(f"กู้คืน ChromaDB สำเร็จ")

        # Restore Enhanced RAG memory if available
        memory_collection_backup = os.path.join(backup_path, "memory_collection")
        if os.path.exists(memory_collection_backup) and RAG_MODE == "enhanced":
            try:
                memory_json_path = os.path.join(memory_collection_backup, "memories.json")
                if os.path.exists(memory_json_path):
                    try:
                        with open(memory_json_path, 'r', encoding='utf-8') as f:
                            memories = json.load(f)

                        # Validate memories structure
                        if all(key in memories for key in ['ids', 'documents', 'metadatas']):
                            # Clear current memory and restore from backup
                            if hasattr(enhanced_rag, 'memory_collection'):
                                enhanced_rag.memory_collection.delete()
                                enhanced_rag.memory_collection.add(
                                    ids=memories['ids'],
                                    documents=memories['documents'],
                                    metadatas=memories['metadatas']
                                )
                                logging.info("กู้คืน Enhanced RAG Memory สำเร็จ")
                        else:
                            logging.warning("Invalid memory backup structure - missing required keys")
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON in memory backup: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Could not restore memory backup: {str(e)}")

            except Exception as e:
                logging.warning(f"ไม่สามารถกู้คืน Memory: {str(e)}")

        # Reload collections
        global collection
        collection = chroma_client.get_or_create_collection(name="pdf_data")

        result = {
            "success": True,
            "backup_name": backup_name,
            "restored_at": datetime.now().isoformat(),
            "metadata": metadata,
            "emergency_backup": emergency_backup
        }

        logging.info(f"กู้คืน Database สำเร็จจาก: {backup_name}")
        return result

    except Exception as e:
        logging.error(f"ไม่สามารถกู้คืน Database: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def is_valid_backup_folder(backup_path):
    """
    ตรวจสอบว่าโฟลเดอร์เป็น backup ที่ถูกต้องหรือไม่
    """
    if not os.path.isdir(backup_path):
        return False

    # Check if it's a valid backup folder (has chromadb or backup_metadata.json)
    chromadb_path = os.path.join(backup_path, "chromadb")
    metadata_path = os.path.join(backup_path, "backup_metadata.json")

    return os.path.exists(chromadb_path) or os.path.exists(metadata_path)


def list_available_backups():
    """
    แสดงรายการ backup ที่มีอยู่ทั้งหมด
    """
    try:
        import json

        backups = []

        if not os.path.exists(TEMP_VECTOR_BACKUP):
            return backups

        for backup_name in sorted(os.listdir(TEMP_VECTOR_BACKUP), reverse=True):
            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

            # Only include valid backup folders
            if not is_valid_backup_folder(backup_path):
                continue

            # Try to load metadata
            metadata = {}
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass

            # Get backup size
            total_size = 0
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)

            backup_info = {
                "name": backup_name,
                "path": backup_path,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "created_at": metadata.get("created_at", "Unknown"),
                "type": metadata.get("type", "standard"),
                "includes_memory": metadata.get("includes_memory", False),
                "rag_mode": metadata.get("rag_mode", "unknown"),
                "database_info": metadata.get("database_info", {})
            }

            backups.append(backup_info)

        return backups

    except Exception as e:
        logging.error(f"ไม่สามารถแสดงรายการ backup: {str(e)}")
        return []


def delete_backup(backup_name):
    """
    ลบ backup ที่ระบุ
    """
    try:
        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            return {"success": False, "error": f"ไม่พบ backup: {backup_name}"}

        if not os.path.isdir(backup_path):
            return {"success": False, "error": f"ไม่ใช่โฟลเดอร์ backup: {backup_name}"}

        shutil.rmtree(backup_path)
        logging.info(f"ลบ backup สำเร็จ: {backup_name}")

        return {"success": True, "message": f"ลบ backup {backup_name} สำเร็จ"}

    except Exception as e:
        logging.error(f"ไม่สามารถลบ backup: {str(e)}")
        return {"success": False, "error": str(e)}


def validate_backup_integrity(backup_path):
    """
    ตรวจสอบความสมบูรณ์ของ backup
    """
    try:
        # Check main database
        chromadb_path = os.path.join(backup_path, "chromadb")
        if os.path.exists(chromadb_path):
            # Check if essential ChromaDB files exist
            sqlite_file = os.path.join(chromadb_path, "chroma.sqlite3")
            if not os.path.exists(sqlite_file):
                return False, "Missing chroma.sqlite3 file"

        # Check backup metadata JSON
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Try to parse JSON
            except json.JSONDecodeError:
                return False, "Invalid backup metadata JSON"

        # Check memory backup JSON if exists
        memory_collection_path = os.path.join(backup_path, "memory_collection")
        if os.path.exists(memory_collection_path):
            memory_json_path = os.path.join(memory_collection_path, "memories.json")
            if os.path.exists(memory_json_path):
                try:
                    with open(memory_json_path, 'r', encoding='utf-8') as f:
                        memories = json.load(f)
                        # Check if memory structure is valid
                        required_keys = ['ids', 'documents', 'metadatas']
                        if not all(key in memories for key in required_keys):
                            return False, "Invalid memory backup structure"
                except json.JSONDecodeError:
                    return False, "Invalid memory backup JSON"

        return True, "Backup is valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def cleanup_invalid_backups():
    """
    ลบโฟลเดอร์ที่ไม่ใช่ backup ที่ถูกต้อง
    """
    try:
        if not os.path.exists(TEMP_VECTOR_BACKUP):
            return

        cleaned_count = 0
        for folder_name in os.listdir(TEMP_VECTOR_BACKUP):
            folder_path = os.path.join(TEMP_VECTOR_BACKUP, folder_name)
            if os.path.isdir(folder_path):
                # Check if it's a valid backup folder
                if not is_valid_backup_folder(folder_path):
                    shutil.rmtree(folder_path)
                    cleaned_count += 1
                    logging.info(f"ลบโฟลเดอร์ที่ไม่ถูกต้อง: {folder_name}")
                else:
                    # Additional validation for backup integrity
                    is_valid, message = validate_backup_integrity(folder_path)
                    if not is_valid:
                        logging.warning(f"Backup {folder_name} failed validation: {message}")
                        # Don't delete automatically, just warn

        if cleaned_count > 0:
            logging.info(f"ลบโฟลเดอร์ที่ไม่ถูกต้องทั้งหมด {cleaned_count} โฟลเดอร์")

    except Exception as e:
        logging.error(f"ไม่สามารถ cleanup invalid backups: {str(e)}")


def auto_backup_before_operation():
    """
    สร้าง backup อัตโนมัติก่อนการดำเนินการที่สำคัญ
    """
    try:
        # Clean invalid backups first
        cleanup_invalid_backups()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"auto_backup_{timestamp}"

        result = backup_database_enhanced(
            backup_name=backup_name,
            include_memory=(RAG_MODE == "enhanced")
        )

        if result["success"]:
            logging.info(f"สร้าง Auto Backup สำเร็จ: {backup_name}")
        else:
            logging.warning(f"สร้าง Auto Backup ไม่สำเร็จ: {result.get('error')}")

        return result

    except Exception as e:
        logging.error(f"ไม่สามารถสร้าง Auto Backup: {str(e)}")
        return {"success": False, "error": str(e)}


def restore_vector_db(backup_name=None):
    """
    กู้คืนข้อมูลจาก backup

    Args:
        backup_name: ชื่อโฟลเดอร์ backup (ถ้าไม่ระบุจะใช้ล่าสุด)
    """
    global TEMP_VECTOR
    try:
        if backup_name is None:
            # หา backup ล่าสุด
            backups = [d for d in os.listdir(TEMP_VECTOR_BACKUP)
                      if d.startswith("backup_") and os.path.isdir(os.path.join(TEMP_VECTOR_BACKUP, d))]
            if not backups:
                logging.error("ไม่พบข้อมูล backup")
                return False
            backup_name = sorted(backups)[-1]

        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            logging.error(f"ไม่พบ backup: {backup_name}")
            return False

        # Try multiple strategies to restore database
        restore_success = False
        restore_method = ""

        # Strategy 1: Force release and restore
        try:
            log_with_time("Strategy 1: Force release and restore")
            force_release_chromadb()
            time.sleep(2)  # Wait longer for files to be released

            # สำรองข้อมูลปัจจุบันก่อน restore
            backup_vector_db()

            # ลบข้อมูลปัจจุบันและกู้คืนจาก backup
            if os.path.exists(TEMP_VECTOR):
                shutil.rmtree(TEMP_VECTOR)

            shutil.copytree(backup_path, TEMP_VECTOR)
            log_with_time(f"กู้คืนข้อมูลสำเร็จจาก: {backup_name}")
            restore_success = True
            restore_method = "force_release"
        except Exception as e1:
            log_with_time(f"Strategy 1 failed: {e1}")

        # Strategy 2: Move database and create fresh
        if not restore_success:
            try:
                log_with_time("Strategy 2: Move database and create fresh")

                # Move locked database to temp
                moved_path = move_database_to_temp()

                # Create fresh database
                create_fresh_database()

                # Copy backup to fresh database
                if os.path.exists(TEMP_VECTOR):
                    shutil.rmtree(TEMP_VECTOR)
                shutil.copytree(backup_path, TEMP_VECTOR)

                log_with_time(f"กู้คืนข้อมูลสำเร็จจาก: {backup_name} (fresh database)")
                restore_success = True
                restore_method = "fresh_database"
            except Exception as e2:
                log_with_time(f"Strategy 2 failed: {e2}")

        # Strategy 3: Use new timestamp path
        if not restore_success:
            try:
                log_with_time("Strategy 3: Use new timestamp path")

                # Create new database path with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_db_path = f"./data/chromadb_restored_{timestamp}"

                if os.path.exists(new_db_path):
                    shutil.rmtree(new_db_path)

                shutil.copytree(backup_path, new_db_path)

                # Update global TEMP_VECTOR to new path
                TEMP_VECTOR = new_db_path

                log_with_time(f"กู้คืนข้อมูลสำเร็จจาก: {backup_name} (new path: {new_db_path})")
                restore_success = True
                restore_method = "new_path"
            except Exception as e3:
                log_with_time(f"Strategy 3 failed: {e3}")

        if not restore_success:
            log_with_time("❌ All restore strategies failed")
            return False

        log_with_time(f"✅ Restore successful using method: {restore_method}")

        # รีโหลด collection
        global collection
        collection = chroma_client.get_or_create_collection(name="pdf_data")

        return True
    except Exception as e:
        logging.error(f"ไม่สามารถกู้คืนข้อมูลได้: {str(e)}")
        return False


def get_database_info():
    """
    แสดงข้อมูลสถานะ database อย่างละเอียด
    """
    try:
        # ข้อมูลพื้นฐาน
        count = collection.count()

        # ข้อมูลไฟล์
        db_exists = os.path.exists(TEMP_VECTOR)
        sqlite_exists = os.path.exists(os.path.join(TEMP_VECTOR, "chroma.sqlite3"))

        # ขนาด database
        db_size = 0
        if db_exists:
            for root, dirs, files in os.walk(TEMP_VECTOR):
                db_size += sum(os.path.getsize(os.path.join(root, name)) for name in files)

        # จำนวน backup
        backup_count = 0
        if os.path.exists(TEMP_VECTOR_BACKUP):
            backup_count = len([d for d in os.listdir(TEMP_VECTOR_BACKUP)
                               if d.startswith("backup_")])

        info = {
            "total_records": count,
            "database_path": TEMP_VECTOR,
            "database_exists": db_exists,
            "sqlite_exists": sqlite_exists,
            "database_size_mb": round(db_size / (1024 * 1024), 2),
            "backup_count": backup_count,
            "collections": [{"name": coll.name, "count": coll.count()} for coll in chroma_client.list_collections()]
        }

        logging.info(f"📊 Database Info: {count} records, {round(db_size/(1024*1024),2)}MB, {backup_count} backups")
        return info

    except Exception as e:
        logging.error(f"❌ ไม่สามารถดึงข้อมูล database ได้: {str(e)}")
        return {"error": str(e)}


def inspect_database():
    """
    ตรวจสอบข้อมูลภายใน database อย่างละเอียด
    """
    try:
        count = collection.count()
        if count == 0:
            return "Database ว่างเปล่า - ไม่มีข้อมูล"

        # ดึงข้อมูลตัวอย่าง 3 records
        sample_data = collection.get(limit=3, include=["documents", "metadatas"])

        result = f"📊 Database Inspection:\n"
        result += f"📝 Total Records: {count}\n"
        result += f"📁 Collections: {list(chroma_client.list_collections())}\n\n"

        result += "📋 Sample Data (first 3 records):\n"
        for i, (doc, meta) in enumerate(zip(sample_data["documents"][:3], sample_data["metadatas"][:3])):
            result += f"\n{i+1}. Document: {doc[:100]}...\n"
            result += f"   Metadata: {meta}\n"
            result += f"   ---"

        return result

    except Exception as e:
        return f"❌ ตรวจสอบ database ล้มเหลว: {str(e)}"


# Discord Bot for receiving questions
class RAGPDFBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # เปิดการอ่านข้อความ
        super().__init__(intents=intents)
        self.is_ready = False

    async def on_ready(self):
        """เมื่อ Bot เชื่อมต่อสำเร็จ"""
        logging.info(f'Bot เชื่อมต่อสำเร็จเป็น {self.user}')
        self.is_ready = True
        # ตั้งสถานะของ Bot
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{DISCORD_BOT_PREFIX}คำถาม"
            )
        )

    async def on_message(self, message):
        """รับข้อความจาก Discord"""
        # ไม่ตอบข้อความของตัวเอง
        if message.author == self.user:
            return

        # ตรวจสอบว่าควรตอบหรือไม่
        should_respond = False
        question = ""
        response_type = ""

        # กรณีที่ 1: มี prefix (เช่น !ask)
        if message.content.startswith(DISCORD_BOT_PREFIX):
            question = message.content[len(DISCORD_BOT_PREFIX):].strip()
            should_respond = True
            response_type = "prefix"

        # กรณีที่ 2: ถูก mention bot (เช่น @RAGPDFBot)
        elif self.user.mentioned_in(message):
            # ลบ mention ออกจากข้อความ
            question = message.content.replace(f'<@!{self.user.id}>', '').replace(f'<@{self.user.id}>', '').strip()
            should_respond = True
            response_type = "mention"

        # กรณีที่ 3: ไม่มี prefix แต่มีการตั้งค่าให้ตอบทุกข้อความ
        elif DISCORD_RESPOND_NO_PREFIX:
            question = message.content.strip()
            should_respond = True
            response_type = "auto"

        # ถ้าไม่ต้องการตอบ ให้จบการทำงาน
        if not should_respond or not question:
            return

        # ตรวจสอบว่ามีคำถามหรือไม่
        if not question:
            if response_type == "prefix":
                await message.reply(
                    "❌ กรุณาระบุคำถาม\n"
                    f"ตัวอย่าง: `{DISCORD_BOT_PREFIX}PDF นี้เกี่ยวกับอะไร`"
                )
            elif response_type == "mention":
                await message.reply(
                    "❌ กรุณาระบุคำถามหลังจาก mention\n"
                    f"ตัวอย่าง: `@{self.user.name} PDF นี้เกี่ยวกับอะไร`"
                )
            else:
                await message.reply(
                    "❌ กรุณาระบุคำถาม\n"
                    f"ตัวอย่าง: `PDF นี้เกี่ยวกับอะไร`"
                )
            return

        # แสดงสถานะกำลังประมวลผล
        logging.info(f"Discord Bot: รับคำถาม ({response_type}) - {question}")
        processing_msg = await message.reply("🔍 กำลังค้นหาคำตอบ...")

        try:
            # เรียกใช้ RAG system
            stream = query_rag(question, chat_llm=DISCORD_DEFAULT_MODEL)

            # รวบรวมคำตอบ
            full_answer = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                full_answer += content

            # จัดรูปแบบคำตอบ (จำกัดความยาวสำหรับ Discord)
            if len(full_answer) > 1990:  # Discord จำกัด 2000 ตัวอักษร
                full_answer = full_answer[:1980] + "...\n\n*คำตอบถูกตัดเนื่องจากความยาวเกินขีดจำกัด*"

            # ดึงรูปภาพที่เกี่ยวข้องจากคำตอบ
            image_paths = extract_images_from_answer(full_answer)

            # สร้าง embed สำหรับคำตอบ
            embed = discord.Embed(
                title="",
                description=full_answer,
                color=discord.Color.blue()
            )

            # เพิ่มข้อมูลว่ามีรูปภาพประกอบหรือไม่
            if image_paths:
                embed.add_field(name="🖼️ รูปภาพที่เกี่ยวข้อง", value=f"พบ {len(image_paths)} รูปภาพที่เกี่ยวข้อง", inline=False)

            # embed.add_field(name="❓ คำถาม", value=question, inline=False)
            # embed.set_footer(text="PDF RAG Assistant • ข้อมูลจาก PDF ที่อัปโหลด")
            # embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/2951/2951136.png")

            # ลบข้อความกำลังประมวลผล
            await processing_msg.delete()

            # ส่งคำตอบตามโหมดที่กำหนด พร้อมรูปภาพ
            await respond_to_discord_message_with_images(message, embed, image_paths, DISCORD_REPLY_MODE)

            logging.info(f"Discord Bot: ตอบคำถามเรียบร้อย (โหมด: {DISCORD_REPLY_MODE})")

        except Exception as e:
            # ลบข้อความกำลังประมวลผล
            await processing_msg.delete()

            error_embed = discord.Embed(
                title="❌ เกิดข้อผิดพลาด",
                description=f"ไม่สามารถตอบคำถามได้: {str(e)}",
                color=discord.Color.red()
            )
            await respond_to_discord_message(message, error_embed, DISCORD_REPLY_MODE)
            logging.error(f"Discord Bot error: {str(e)}")


async def send_discord_dm(user, embed):
    """ส่ง Direct Message ให้ผู้ใช้ Discord"""
    try:
        await user.send(embed=embed)
        return True
    except discord.Forbidden:
        logging.warning(f"ไม่สามารถส่ง DM ให้ผู้ใช้ {user.name} ได้ (อาจปิดการรับ DM)")
        return False
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่ง DM: {str(e)}")
        return False


async def respond_to_discord_message(message, embed, reply_type="channel"):
    """ตอบกลับข้อความใน Discord ตามโหมดที่กำหนด"""
    success_dm = False
    success_channel = False

    # ส่งใน channel (เฉพาะโหมด channel หรือ both)
    if reply_type in ["channel", "both"]:
        try:
            await message.reply(embed=embed)
            success_channel = True
        except Exception as e:
            logging.error(f"ไม่สามารถตอบใน channel: {str(e)}")

    # ส่ง DM (เฉพาะโหมด dm หรือ both)
    if reply_type in ["dm", "both"]:
        success_dm = await send_discord_dm(message.author, embed)

    # ถ้าส่ง DM ไม่ได้แต่เลือกโหมด dm ให้ส่งใน channel แทน
    if reply_type == "dm" and not success_dm:
        try:
            fallback_embed = discord.Embed(
                title="📬 คำตอบของคุณ",
                description=embed.description,
                color=embed.color
            )
            await message.reply(embed=fallback_embed)
            logging.info("ส่งคำตอบใน channel แทน เนื่องจากไม่สามารถส่ง DM ได้")
        except Exception as e:
            logging.error(f"ส่งข้อความ fallback ไม่ได้: {str(e)}")


async def respond_to_discord_message_with_images(message, embed, image_paths, reply_type="channel"):
    """ตอบกลับข้อความใน Discord ตามโหมดที่กำหนด พร้อมส่งรูปภาพ"""
    success_dm = False
    success_channel = False

    # ส่งใน channel (เฉพาะโหมด channel หรือ both)
    if reply_type in ["channel", "both"]:
        try:
            if image_paths:
                # ส่ง embed พร้อมรูปภาพใน channel
                await send_message_with_images(message.channel, embed, image_paths, reply_to=message)
            else:
                # ส่งเฉพาะ embed ถ้าไม่มีรูป
                await message.reply(embed=embed)
            success_channel = True
        except Exception as e:
            logging.error(f"ไม่สามารถตอบใน channel: {str(e)}")

    # ส่ง DM (เฉพาะโหมด dm หรือ both)
    if reply_type in ["dm", "both"]:
        success_dm = await send_discord_dm_with_images(message.author, embed, image_paths)

    # ถ้าส่ง DM ไม่ได้แต่เลือกโหมด dm ให้ส่งใน channel แทน
    if reply_type == "dm" and not success_dm:
        try:
            fallback_embed = discord.Embed(
                title="📬 คำตอบของคุณ",
                description=embed.description,
                color=embed.color
            )
            if image_paths:
                await send_message_with_images(message.channel, fallback_embed, image_paths, reply_to=message)
            else:
                await message.reply(embed=fallback_embed)
            logging.info("ส่งคำตอบใน channel แทน เนื่องจากไม่สามารถส่ง DM ได้")
        except Exception as e:
            logging.error(f"ส่งข้อความ fallback ไม่ได้: {str(e)}")


async def send_message_with_images(channel, embed, image_paths, reply_to=None):
    """ส่งข้อความพร้อมรูปภาพใน Discord channel"""
    try:
        # Discord จำกัดไฟล์แนบได้ 10 ไฟล์ต่อข้อความ
        files_to_send = image_paths[:10]

        # สร้าง discord.File objects
        files = []
        for img_path in files_to_send:
            try:
                # Discord รองรับไฟล์ที่มีขนาดไม่เกิน 8MB
                file_size = os.path.getsize(img_path)
                if file_size > 8 * 1024 * 1024:  # 8MB
                    logging.warning(f"รูปภาพ {img_path} มีขนาดใหญ่เกิน 8MB จะข้ามไป")
                    continue

                file = discord.File(img_path, filename=os.path.basename(img_path))
                files.append(file)
            except Exception as e:
                logging.error(f"ไม่สามารถอ่านไฟล์ {img_path}: {str(e)}")

        # ส่งข้อความพร้อมไฟล์
        if files:
            if reply_to:
                await reply_to.reply(embed=embed, files=files)
            else:
                await channel.send(embed=embed, files=files)
            logging.info(f"ส่งข้อความพร้อมรูปภาพ {len(files)} รูปไปยัง Discord สำเร็จ")
        else:
            # ถ้าไม่มีไฟล์ที่ส่งได้ ส่งเฉพาะ embed
            if reply_to:
                await reply_to.reply(embed=embed)
            else:
                await channel.send(embed=embed)

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่งข้อความพร้อมรูปภาพ: {str(e)}")
        # ส่งเฉพาะ embed ถ้าส่งรูปไม่ได้
        try:
            if reply_to:
                await reply_to.reply(embed=embed)
            else:
                await channel.send(embed=embed)
        except Exception as e2:
            logging.error(f"ส่งข้อความ fallback ก็ไม่ได้: {str(e2)}")


async def send_discord_dm_with_images(user, embed, image_paths):
    """ส่ง Direct Message พร้อมรูปภาพให้ผู้ใช้ Discord"""
    try:
        # Discord จำกัดไฟล์แนบได้ 10 ไฟล์ต่อข้อความ
        files_to_send = image_paths[:10]

        # สร้าง discord.File objects
        files = []
        for img_path in files_to_send:
            try:
                file_size = os.path.getsize(img_path)
                if file_size > 8 * 1024 * 1024:  # 8MB
                    logging.warning(f"รูปภาพ {img_path} มีขนาดใหญ่เกิน 8MB จะข้ามไป")
                    continue

                file = discord.File(img_path, filename=os.path.basename(img_path))
                files.append(file)
            except Exception as e:
                logging.error(f"ไม่สามารถอ่านไฟล์ {img_path}: {str(e)}")

        # ส่ง DM พร้อมไฟล์
        if files:
            await user.send(embed=embed, files=files)
        else:
            await user.send(embed=embed)

        logging.info(f"ส่ง DM พร้อมรูปภาพให้ผู้ใช้ {user.name} สำเร็จ")
        return True

    except discord.Forbidden:
        logging.warning(f"ไม่สามารถส่ง DM ให้ผู้ใช้ {user.name} ได้ (อาจปิดการรับ DM)")
        return False
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่ง DM: {str(e)}")
        return False


# Global variables for Discord Bot
discord_bot = None
discord_bot_thread = None


def start_discord_bot():
    """เริ่มทำงาน Discord Bot"""
    global discord_bot

    if not DISCORD_BOT_ENABLED or DISCORD_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        logging.info("Discord Bot ไม่ได้เปิดใช้งานหรือยังไม่ได้ตั้งค่า")
        return False

    try:
        discord_bot = RAGPDFBot()

        # สร้าง event loop ใหม่สำหรับ bot
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # เรียกใช้ bot
        loop.run_until_complete(discord_bot.start(DISCORD_BOT_TOKEN))

    except Exception as e:
        logging.error(f"ไม่สามารถเริ่ม Discord Bot ได้: {str(e)}")
        return False


def start_discord_bot_thread():
    """เริ่ม Discord Bot ใน thread แยก"""
    global discord_bot_thread

    if discord_bot_thread and discord_bot_thread.is_alive():
        logging.warning("Discord Bot กำลังทำงานอยู่แล้ว")
        return False

    discord_bot_thread = threading.Thread(target=start_discord_bot, daemon=True)
    discord_bot_thread.start()

    # รอสักครู่ให้ bot เริ่มทำงาน
    import time
    time.sleep(2)

    return True


def stop_discord_bot():
    """หยุด Discord Bot"""
    global discord_bot

    if discord_bot and discord_bot.is_ready:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(discord_bot.close())
            else:
                loop.run_until_complete(discord_bot.close())
            logging.info("Discord Bot หยุดทำงานแล้ว")
            return True
        except Exception as e:
            logging.error(f"ไม่สามารถหยุด Discord Bot ได้: {str(e)}")
            return False

    return False


# LINE OA and Facebook Messenger Setup (using FastAPI, not Flask)
# LINE OA Setup
line_bot_api = None
line_handler = None
line_thread = None

# Facebook Messenger Setup
fb_thread = None


def setup_line_bot():
    """ตั้งค่า LINE Bot"""
    global line_bot_api, line_handler
    if LINE_ENABLED and LINE_CHANNEL_ACCESS_TOKEN != "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        try:
            line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
            line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

            # Register handlers หลังจาก setup แล้วเท่านั้น
            register_line_handlers()

            logging.info("✅ LINE Bot setup completed")
            return True
        except Exception as e:
            logging.error(f"❌ LINE Bot setup failed: {str(e)}")
            return False
    else:
        logging.info("LINE Bot is disabled or not configured")
        return False


def register_line_handlers():
    """Register LINE message handlers"""
    if line_handler:
        @line_handler.add(MessageEvent, message=TextMessage)
        def handle_line_message(event):
            """จัดการข้อความจาก LINE"""
            try:
                user_message = event.message.text
                user_id = event.source.user_id

                logging.info(f"LINE Bot: รับคำถามจาก {user_id} - {user_message}")

                # ตอบกลับเพื่อแสดงว่ากำลังประมวลผล
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="🔍 กำลังค้นหาคำตอบ...")
                )

                # ประมวลผลคำถามใน background
                threading.Thread(
                    target=process_line_question,
                    args=(event, user_message, user_id)
                ).start()

            except Exception as e:
                logging.error(f"LINE Bot error: {str(e)}")
                try:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text="❌ เกิดข้อผิดพลาด กรุณาลองใหม่")
                    )
                except:
                    pass


def setup_facebook_bot():
    """ตั้งค่า Facebook Messenger Bot"""
    if FB_ENABLED and FB_PAGE_ACCESS_TOKEN != "YOUR_FB_PAGE_ACCESS_TOKEN":
        logging.info("✅ Facebook Messenger Bot setup completed")
        return True
    else:
        logging.info("Facebook Messenger Bot is disabled or not configured")
        return False


# Flask route removed - now using FastAPI endpoint at line ~5921
# The /callback endpoint is registered via gradio_app.add_api_route() in __main__

def process_line_question(event, user_id: str, question: str):
    """ประมวลผลคำถามจาก LINE"""
    try:
        # Detect AI provider from model name
        ai_provider = "gemini" if "gemini" in LINE_DEFAULT_MODEL.lower() else "ollama"

        # เรียกใช้ฟังก์ชัน RAG เพื่อตอบคำถาม (returns generator/stream)
        stream = query_rag(question, LINE_DEFAULT_MODEL, ai_provider=ai_provider, show_source=False)

        # Collect answer from stream
        answer = ""
        for chunk in stream:
            if isinstance(chunk, dict):
                # Handle streaming response format
                if "message" in chunk and "content" in chunk["message"]:
                    answer += chunk["message"]["content"]
                elif "content" in chunk:
                    answer += chunk["content"]

        # ถ้าไม่มีคำตอบ
        if not answer.strip():
            answer = "ไม่พบคำตอบ กรุณาลองถามใหม่อีกครั้ง"

        # จำกัดความยาวข้อความสำหรับ LINE (สูงสุด 5000 ตัวอักษร)
        if len(answer) > 4900:
            answer = answer[:4900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

        # ส่งคำตอบกลับไปยัง LINE
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=answer)
        )

        logging.info(f"LINE Bot: ตอบคำถามเรียบร้อย")

    except Exception as e:
        logging.error(f"LINE processing error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        try:
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text="ขออภัย เกิดข้อผิดพลาดในการค้นหาคำตอบ กรุณาลองใหม่อีกครั้ง")
            )
        except:
            pass


# Flask route removed - now using FastAPI endpoint at line ~5946
# The /webhook endpoint is registered via gradio_app.add_api_route() in __main__

def process_facebook_question(sender_id: str, question: str):
    """ประมวลผลคำถามจาก Facebook Messenger"""
    try:
        logging.info(f"Facebook Bot: รับคำถามจาก {sender_id} - {question}")

        # ส่งข้อความกำลังประมวลผล
        send_facebook_message(sender_id, "กำลังค้นหาคำตอบให้คุณ...")

        # Detect AI provider from model name
        ai_provider = "gemini" if "gemini" in FB_DEFAULT_MODEL.lower() else "ollama"

        # เรียกใช้ฟังก์ชัน RAG เพื่อตอบคำถาม (returns generator/stream)
        stream = query_rag(question, FB_DEFAULT_MODEL, ai_provider=ai_provider, show_source=False)

        # Collect answer from stream
        answer = ""
        for chunk in stream:
            if isinstance(chunk, dict):
                # Handle streaming response format
                if "message" in chunk and "content" in chunk["message"]:
                    answer += chunk["message"]["content"]
                elif "content" in chunk:
                    answer += chunk["content"]

        # ถ้าไม่มีคำตอบ
        if not answer.strip():
            answer = "ไม่พบคำตอบ กรุณาลองถามใหม่อีกครั้ง"

        # จำกัดความยาวข้อความสำหรับ Facebook (สูงสุด 2000 ตัวอักษร)
        if len(answer) > 1900:
            answer = answer[:1900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

        # ส่งคำตอบกลับไปยัง Facebook
        send_facebook_message(sender_id, answer)

        logging.info(f"Facebook Bot: ตอบคำถามเรียบร้อย")

    except Exception as e:
        logging.error(f"Facebook processing error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        try:
            send_facebook_message(sender_id, "ขออภัย เกิดข้อผิดพลาดในการค้นหาคำตอบ กรุณาลองใหม่อีกครั้ง")
        except:
            pass


def send_facebook_message(recipient_id: str, message_text: str):
    """ส่งข้อความไปยัง Facebook Messenger"""
    try:
        url = f"https://graph.facebook.com/v18.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"

        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text}
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

    except Exception as e:
        logging.error(f"Error sending Facebook message: {str(e)}")


# Flask server functions removed - webhooks now handled by FastAPI endpoints
# mounted to Gradio's FastAPI app at startup (see __main__ section)


def determine_optimal_results(question: str) -> int:
    """
    กำหนดจำนวนผลลัพธ์ที่เหมาะสมตามประเภทคำถาม
    """
    question_lower = question.lower()

    # คำถามที่ต้องการความครบถ้วน - ใช้จำนวนมากขึ้น
    comprehensive_keywords = ["ทั้งหมด", "ทั้งหมดทั้งสิ้น", "ทุก", "ทุกอย่าง", "สรุปทั้งหมด", "ทั้งหมดเลย"]
    if any(keyword in question_lower for keyword in comprehensive_keywords):
        return 8

    # คำถามแบบนับจำนวน - ใช้ปานกลาง
    counting_keywords = ["กี่", "จำนวน", "มีกี่", "กี่ตัว", "จำนวนเท่าไร", "มีกี่อย่าง"]
    if any(keyword in question_lower for keyword in counting_keywords):
        return 5

    # คำถามแบบเลือกบางส่วน - ใช้ปานกลาง
    selective_keywords = ["บ้าง", "บางส่วน", "บางอย่าง", "ตัวอย่าง", "กรณี", "เช่น", "ดังนี้"]
    if any(keyword in question_lower for keyword in selective_keywords):
        return 4

    # คำถามเฉพาะเจาะจง - ใช้จำนวนน้อย
    specific_keywords = ["คืออะไร", "อะไร", "ที่ไหน", "เมื่อไหร่", "ทำไม", "อย่างไร", "ใคร"]
    if any(keyword in question_lower for keyword in specific_keywords):
        return 3

    # คำถามทั่วไป - ใช้ค่าเริ่มต้น
    return 3


def calculate_relevance_score(question: str, context: str) -> float:
    """
    คำนวณคะแนนความเกี่ยวข้องระหว่างคำถามและ context (ปรับปรุงสำหรับภาษาไทย)
    """
    # แยกคำในคำถามและ context
    question_words = set(word_tokenize(question))
    context_words = set(word_tokenize(context))

    if len(question_words) == 0:
        return 0.0

    # คำนวณคำที่ซ้ำกัน
    common_words = question_words.intersection(context_words)

    # คำนวณ Jaccard similarity
    jaccard_similarity = len(common_words) / len(question_words.union(context_words))

    # คำนวณ keyword matching แบบ case-insensitive
    question_lower = question.lower()
    context_lower = context.lower()

    # ให้คะแนนพิเศษสำหรับคำสำคัญที่ตรงกัน
    exact_matches = 0
    partial_matches = 0

    for word in question_words:
        word_lower = word.lower()
        if word_lower in context_lower:
            exact_matches += 1

    # ตรวจสอบการ match แบบย่อย (เช่น "1-on-1" กับ "session", "นัด" กับ "คุย")
    for q_word in question_words:
        for c_word in context_words:
            # ถ้าคำใดคำหนึ่งเป็นส่วนหนึ่งของอีกคำ
            if q_word.lower() in c_word.lower() or c_word.lower() in q_word.lower():
                partial_matches += 0.5

    # ตรวจสอบคำที่เกี่ยวข้อง semantically สำหรับภาษาไทย
    semantic_matches = 0
    question_lower = question_lower.replace("1-on-1", "session สอนเสริม นัดคุย")
    question_lower = question_lower.replace("เฉพาะเรื่อง", "เฉพาะจุด")
    question_lower = question_lower.replace("อธิบาย", "สอนเสริม")
    question_lower = question_lower.replace("ไม่เข้าใจ", "ข้อสงสัย")

    for word in question_lower.split():
        if word in context_lower and len(word) > 2:  # คำที่มีความหมายมากกว่า 2 ตัวอักษร
            semantic_matches += 0.3

    # รวมคะแนนแบบถ่วงน้ำหนัก
    base_score = jaccard_similarity * 0.4
    exact_score = (exact_matches / len(question_words)) * 0.4
    partial_score = min(partial_matches / len(question_words), 0.3) * 0.2
    semantic_score = min(semantic_matches / len(question_words), 0.2) * 0.2

    final_score = base_score + exact_score + partial_score + semantic_score

    return min(final_score, 1.0)  # จำกัดคะแนนสูงสุดที่ 1.0


def filter_relevant_contexts(question: str, documents: list, metadatas: list, min_relevance: float = 0.05) -> list:
    """
    กรองเฉพาะ context ที่相关性สูง
    """
    if not documents:
        return []

    filtered_contexts = []

    for doc, metadata in zip(documents, metadatas):
        # คำนวณคะแนนความเกี่ยวข้อง
        relevance_score = calculate_relevance_score(question, doc)

        # เก็บเฉพาะ context ที่ผ่าน threshold
        if relevance_score >= min_relevance:
            filtered_contexts.append({
                'text': doc,
                'metadata': metadata,
                'relevance_score': relevance_score
            })

    # เรียงลำดับตามคะแนนความเกี่ยวข้อง (สูงสุดก่อน)
    filtered_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)

    # จำกัดจำนวน context สูงสุด
    max_contexts = 5
    return filtered_contexts[:max_contexts]


def query_rag(question: str, chat_llm: str = "gemma3:latest", ai_provider: str = "ollama", show_source: bool = False, formal_style: bool = False):
    """
    ค้นหาในระบบ Enhanced RAG และสร้างคำตอบแบบ streaming โดยใช้ Ollama
    """
    global summarize, enhanced_rag

    # ตรวจสอบว่ามีการสรุปเนื้อหาหรือไม่ ถ้าไม่มีให้ใช้ค่าว่าง
    if 'summarize' not in globals() or summarize is None:
        summarize = "ยังไม่มีการสรุปเนื้อหาจาก PDF"

    logging.info(f"#### Enhanced RAG Mode: {RAG_MODE} #### ")
    logging.info(f"#### Question: {question} #### ")

    # Get relevant memories if enhanced mode is enabled
    relevant_memories = []
    if RAG_MODE == "enhanced":
        relevant_memories = enhanced_rag.get_relevant_memory(question)
        logging.info(f"Found {len(relevant_memories)} relevant memories")

    question_embedding = embed_text(question)

    # Smart Retrieval: ปรับจำนวนผลลัพธ์ตามประเภทคำถาม
    max_result = determine_optimal_results(question)
    logging.info(f"Using max_result: {max_result}")

    # ค้นหาด้วย similarity threshold ที่สูงขึ้น
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=max_result
    )

    # Relevance Filtering: กรองเฉพาะ context ที่相关性สูง
    filtered_contexts = filter_relevant_contexts(question, results["documents"][0], results["metadatas"][0], min_relevance=0.05)
    logging.info(f"Filtered {len(results['documents'][0])} contexts to {len(filtered_contexts)} relevant contexts")

    # ถ้าไม่มี context ที่ผ่านการกรอง ให้ใช้ทั้งหมด
    if len(filtered_contexts) == 0:
        logging.warning("No contexts passed relevance filter, using all retrieved contexts")
        filtered_contexts = [{'text': doc, 'metadata': meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    context_texts = []
    image_paths = []

    # ใช้เฉพาะ context ที่ผ่านการกรอง
    for doc, metadata in zip([ctx['text'] for ctx in filtered_contexts], [ctx['metadata'] for ctx in filtered_contexts]):
        context_texts.append(doc)
        logging.info(f"Selected context: {doc}")
        logging.info(f"metadata: {metadata}")

        # Regex pattern สำหรับค้นหา [img: ชื่อไฟล์.jpeg]
        pattern = r"pic_(\d+)_(\d+)\.jpeg"
        imgs = re.findall(pattern, doc)

        if imgs:
            image_paths.append(imgs)
            logging.info(f"img: {imgs}")

        if metadata and metadata["type"] == "image":
            logging.info(f"image_path : { metadata['image_path']}")
            image_paths.append(metadata['image_path'])
    


    context = "\n".join(context_texts)

    # Build enhanced prompt using EnhancedRAG
    if RAG_MODE == "enhanced":
        prompt = enhanced_rag.build_context_prompt(question, context_texts, relevant_memories, show_source, formal_style)
        logging.info(f"Using Enhanced RAG prompt with {len(relevant_memories)} memories")
        logging.info("############## Begin Enhanced RAG Prompt #################")
        logging.info(f"prompt: {prompt}")
        logging.info("############## End Enhanced RAG Prompt #################")
    else:
        # Standard RAG prompt with stricter instructions
        if summarize is None:
            summarize = ""

        # สร้าง prompt ตามสไตล์ที่เลือก
    if formal_style:
        style_instruction = "ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพ และชัดเจน"
        source_phrase = ""
        response_prefix = "คำตอบ:"
    else:
        style_instruction = "ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
        source_phrase = ""
        response_prefix = "คำตอบ:"

    source_instruction = ""
    if show_source:
        source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else ""

    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามที่มีความรู้ด้านเอกสารที่อัปโหลดไว้ กรุณาตอบคำถามโดยอาศัยข้อมูลจากบริบทที่ให้มาเป็นหลัก

**แนวทางการตอบ:**
- {style_instruction}
- ให้คำตอบที่สอดคล้องกับข้อมูลในเอกสารเป็นหลัก
- หากข้อมูลในเอกสารไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด
- ตอบอย่างสมบูรณ์และมีประโยชน์{source_instruction}
- อาจเพิ่มคำอธิบายเล็กน้อยเพื่อความชัดเจน แต่ไม่ตีความเกินไป

**คำถาม:** {question}

**บริบทจากเอกสาร:**
{summarize}

{context}

{response_prefix}"""

    logging.info("############## Begin Standard Prompt #################")
    logging.info(f"prompt: {prompt}")
    logging.info("############## End Standard Prompt #################")
    logging.info(f"Prompt length: {len(prompt)} characters")

    log_with_time("+++++++++++++ Send prompt To LLM ++++++++++++++++++")
    overall_start = time.time()

    # Debug: Check server status before API call (only for Ollama)
    if ai_provider == "ollama":
        health_start = time.time()
        try:
            import requests
            logging.info("Checking Ollama health before API call...")
            ollama_api_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            logging.info(f"Ollama API URL: {ollama_api_url}")
            health_check = requests.get(f"{ollama_api_url}/api/tags", timeout=5)
            measure_time(health_start, "Ollama health check")
            log_with_time(f"Ollama health check: Status {health_check.status_code}")
            if health_check.status_code == 200:
                models = health_check.json().get('models', [])
                model_names = [m['name'] for m in models]
                log_with_time(f"Available models: {model_names}")
                log_with_time(f"Target model '{chat_llm}' available: {chat_llm in model_names}")
            else:
                log_with_time(f"Ollama server returned status: {health_check.status_code}")
        except Exception as e:
            log_with_time(f"Ollama health check failed: {e}")
            # Return empty stream instead of None
            error_msg = f"❌ Ollama server error: {str(e)}"
            return ({"message": {"content": error_msg}} for _ in range(1))
    else:
        logging.info(f"Skipping health check for {ai_provider} provider")

    api_call_start = time.time()
    log_with_time(f"Starting AI provider: {ai_provider} with model: {chat_llm}")
    log_with_time(f"Prompt preview: {prompt[:100]}...")

    ## Generation  เพื่อการตอบ chat
    try:
        stream = call_ai_provider(
            provider_name=ai_provider,
            model=chat_llm,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.3,
            top_p=0.9,
            max_tokens=2000,
            num_predict=1500
        )
        measure_time(api_call_start, f"{ai_provider}.chat() API call")
        log_with_time(f"{ai_provider} call successful, returning stream")
        measure_time(overall_start, "Total LLM response setup time")
        return stream
    except Exception as e:
        measure_time(api_call_start, f"{ai_provider} API call (failed)")
        log_with_time(f"{ai_provider} call failed: {e}")
        # Return empty stream instead of None
        error_msg = f"❌ LLM call failed: {str(e)}"
        return ({"message": {"content": error_msg}} for _ in range(1))

async def query_rag_with_lightrag(
    question: str,
    chat_llm: str = "gemma3:latest",
    ai_provider: str = "ollama",
    show_source: bool = False,
    formal_style: bool = False,
    use_graph_reasoning: bool = False,
    reasoning_mode: str = "hybrid"
):
    """
    Enhanced RAG query with LightRAG graph reasoning capabilities

    Args:
        question: User's question
        chat_llm: LLM model to use
        show_source: Whether to show source information
        formal_style: Whether to use formal response style
        use_graph_reasoning: Whether to use LightRAG graph reasoning
        reasoning_mode: LightRAG reasoning mode ("naive", "local", "global", "hybrid")
    """
    global summarize, enhanced_rag

    if not LIGHT_RAG_AVAILABLE or not use_graph_reasoning:
        # Fallback to standard query_rag
        logging.info("🔄 Using standard RAG (LightRAG not available or disabled)")
        return query_rag(question, chat_llm, show_source, formal_style)

    try:
        log_with_time("🧠 Starting LightRAG-enhanced query...")

        # Initialize LightRAG system if needed
        try:
            await initialize_lightrag_system()
        except Exception as e:
            logging.warning(f"⚠️ LightRAG initialization failed, using standard RAG: {e}")
            return query_rag(question, chat_llm, show_source, formal_style)

        # Perform graph reasoning query
        lightrag_result = await query_with_graph_reasoning(question, reasoning_mode)

        if "error" in lightrag_result:
            logging.warning(f"⚠️ LightRAG query failed, using standard RAG: {lightrag_result['error']}")
            return query_rag(question, chat_llm, show_source, formal_style)

        # Extract graph reasoning result
        graph_answer = lightrag_result.get("result", "")
        graph_insights = lightrag_result.get("graph_insights", {})
        processing_time = lightrag_result.get("processing_time", 0)

        log_with_time(f"✅ LightRAG query completed in {processing_time:.2f}s")
        log_with_time(f"🧠 Graph insights: {graph_insights}")

        # Build enhanced response combining standard RAG and graph reasoning
        standard_answer = query_rag(question, chat_llm, show_source, formal_style)

        # Extract text from standard answer stream
        standard_text = ""
        for chunk in standard_answer:
            if "message" in chunk and "content" in chunk["message"]:
                standard_text += chunk["message"]["content"]

        # Combine answers
        if graph_answer and graph_answer != "Error":
            combined_answer = f"""🧠 **Graph Reasoning Analysis:**
{graph_answer}

📚 **Traditional RAG Analysis:**
{standard_text}

---
*This response combines graph-based reasoning with traditional document retrieval for comprehensive analysis.*"""
        else:
            combined_answer = standard_text

        # Return as stream for compatibility
        return ({"message": {"content": combined_answer}} for _ in range(1))

    except Exception as e:
        logging.error(f"❌ LightRAG-enhanced query failed: {e}")
        # Fallback to standard RAG
        return query_rag(question, chat_llm, show_source, formal_style)

def query_rag_with_multi_hop(
    question: str,
    chat_llm: str = "gemma3:latest",
    ai_provider: str = "ollama",
    show_source: bool = False,
    formal_style: bool = False,
    hop_count: int = 2
):
    """
    Multi-hop reasoning query using LightRAG

    Args:
        question: Initial question
        chat_llm: LLM model to use
        show_source: Whether to show source information
        formal_style: Whether to use formal response style
        hop_count: Number of reasoning hops
    """
    if not LIGHT_RAG_AVAILABLE:
        # Fallback to standard query_rag
        logging.info("🔄 Using standard RAG (LightRAG not available)")
        return query_rag(question, chat_llm, show_source, formal_style)

    try:
        import asyncio

        log_with_time(f"🔄 Starting multi-hop reasoning query (hops: {hop_count})...")

        # Run async multi-hop query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(multi_hop_reasoning(question, hop_count))
        finally:
            loop.close()

        if "error" in result:
            logging.warning(f"⚠️ Multi-hop query failed, using standard RAG: {result['error']}")
            return query_rag(question, chat_llm, show_source, formal_style)

        # Format multi-hop result
        synthesis = result.get("final_synthesis", "No synthesis available")
        hop_results = result.get("hop_results", [])

        formatted_result = f"""🔄 **Multi-Hop Reasoning Analysis ({hop_count} hops):**

{synthesis}

**Reasoning Steps:**"""

        for i, hop in enumerate(hop_results):
            hop_text = hop.get("result", "")[:500]  # Limit length
            formatted_result += f"""
*Hop {i+1}:* {hop_text}..."""

        formatted_result += f"""

---
*This response uses multi-hop reasoning to explore relationships and implications beyond the initial query.*"""

        # Return as stream for compatibility
        return ({"message": {"content": formatted_result}} for _ in range(1))

    except Exception as e:
        logging.error(f"❌ Multi-hop query failed: {e}")
        # Fallback to standard RAG
        return query_rag(question, chat_llm, show_source, formal_style)

def get_lightrag_system_status():
    """Get comprehensive LightRAG system status"""
    if not LIGHT_RAG_AVAILABLE:
        return {
            "status": "❌ LightRAG Not Available",
            "version": "N/A",
            "graph_built": False,
            "chroma_records": "N/A"
        }

    try:
        # Get LightRAG status using synchronous function
        status = get_lightrag_status()

        # Get ChromaDB record count
        chroma_count = collection.count() if collection else 0

        return {
            "status": "✅ LightRAG Available",
            "lightrag_status": status,
            "chroma_records": chroma_count,
            "graph_available": os.path.exists("./data/lightrag")
        }

    except Exception as e:
        logging.error(f"❌ Failed to get LightRAG status: {e}")
        return {
            "status": "⚠️ LightRAG Error",
            "error": str(e),
            "chroma_records": collection.count() if collection else 0
        }

# UI Event Handlers
def handle_file_selection(files):
    """
    Handle file selection with improved file list display
    """
    if not files:
        return gr.update(value=""), gr.update(visible=False)

    # Prepare file information
    file_info_list = []
    for file in files:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        # Determine file type
        if file_ext == '.pdf':
            file_type = '📄 PDF'
        elif file_ext in ['.txt', '.md']:
            file_type = '📝 Text'
        elif file_ext in ['.docx', '.doc']:
            file_type = '📋 Word'
        else:
            file_type = '📁 File'

        # Get file size
        try:
            if hasattr(file, 'size'):
                file_size = file.size
            elif os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
            else:
                file_size = 0

            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
        except:
            size_str = "Unknown"

        file_info_list.append({
            "name": file_name,
            "type": file_type,
            "size": size_str,
            "path": file_path
        })

    # Format file list display
    file_list_text = "## 📋 Selected Files\n\n"
    for i, info in enumerate(file_info_list, 1):
        file_list_text += f"**{i}. {info['type']} {info['name']}**\n"
        file_list_text += f"   📏 Size: {info['size']}\n"
        file_list_text += f"   📍 Path: `{info['path']}`\n\n"

    return file_list_text, gr.update(visible=True)

def user(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    จัดการ input ของผู้ใช้และเพิ่มลงในประวัติการแชท
    """
    return "", history + [{"role": "user", "content": user_message}]


# ==================== FEEDBACK FUNCTIONS ====================

def save_feedback(question: str, answer: str, feedback_type: str, user_comment: str = "",
                  corrected_answer: str = "", model_used: str = "", sources: str = ""):
    """บันทึก feedback ลงฐานข้อมูล"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedback (question, answer, feedback_type, user_comment, corrected_answer, model_used, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (question, answer, feedback_type, user_comment, corrected_answer, model_used, sources))

        feedback_id = cursor.lastrowid

        # ถ้ามีคำตอบที่ถูกแก้ไข ให้บันทึกลงตาราง corrected_answers
        if feedback_type == "bad" and corrected_answer and corrected_answer.strip():
            try:
                # สร้าง embedding สำหรับคำถาม (เพื่อค้นหาคำถามที่คล้ายกัน)
                model = load_embedding_model()
                question_embedding = model.encode(question, convert_to_tensor=True).cpu().numpy()
                embedding_str = json.dumps(question_embedding.tolist())

                cursor.execute('''
                    INSERT INTO corrected_answers (original_question, original_answer, corrected_answer, feedback_id, question_embedding)
                    VALUES (?, ?, ?, ?, ?)
                ''', (question, answer, corrected_answer, feedback_id, embedding_str))

                logging.info(f"✅ Saved corrected answer for learning: {question[:50]}...")

            except Exception as e:
                logging.warning(f"⚠️ Failed to create embedding for corrected answer: {str(e)}")

        conn.commit()
        conn.close()

        logging.info(f"✅ Saved {feedback_type} feedback for question: {question[:50]}...")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to save feedback: {str(e)}")
        return False


def find_similar_corrected_answer(question: str, threshold: float = 0.8, include_weighted: bool = True) -> dict:
    """ค้นหาคำตอบที่ถูกแก้ไขสำหรับคำถามที่คล้ายกัน (Enhanced with weighted scoring)"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT ca.original_question, ca.original_answer, ca.corrected_answer,
                   ca.question_embedding, ca.applied_count, ca.feedback_id,
                   f.feedback_type, f.user_comment, f.timestamp
            FROM corrected_answers ca
            JOIN feedback f ON ca.feedback_id = f.id
            ORDER BY ca.created_at DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # สร้าง embedding สำหรับคำถามปัจจุบัน
        model = load_embedding_model()
        question_embedding = model.encode(question, convert_to_tensor=True).cpu().numpy()

        best_match = None
        best_score = 0

        for row in rows:
            try:
                stored_embedding = json.loads(row[3])
                stored_embedding = np.array(stored_embedding)

                # คำนวณ cosine similarity
                similarity = np.dot(question_embedding, stored_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(stored_embedding)
                )

                # คำนวณ weighted score (พิจารณาความเก่าและการใช้งาน)
                recency_factor = 1.0  # สามารถเพิ่ม logic สำหรับ recency ได้
                usage_factor = min(row[4] * 0.1, 1.0)  # จำกัด usage factor ที่ 1.0
                feedback_quality = 1.2 if row[6] == 'good' else 1.0  # feedback type weight

                weighted_score = similarity * recency_factor * usage_factor * feedback_quality

                if similarity > threshold and weighted_score > best_score:
                    best_match = {
                        'original_question': row[0],
                        'original_answer': row[1],
                        'corrected_answer': row[2],
                        'similarity': similarity,
                        'weighted_score': weighted_score,
                        'applied_count': row[4],
                        'feedback_type': row[6],
                        'user_comment': row[7],
                        'feedback_id': row[5]
                    }
                    best_score = weighted_score

            except Exception as e:
                logging.warning(f"⚠️ Error processing embedding: {str(e)}")
                continue

        return best_match

    except Exception as e:
        logging.error(f"❌ Failed to find similar corrected answer: {str(e)}")
        return None


def calculate_feedback_priority(question: str, corrected_answer: str, confidence: float) -> float:
    """คำนวณ priority score สำหรับ feedback (0.0 - 1.0)"""
    try:
        priority = confidence * 0.4  # 40% weight from confidence

        # ตรวจสอบความซับซ้อนของคำถาม (คำถามซับซ้อนได้คะแนนสูงกว่า)
        question_complexity = len(question.split()) * 0.01
        priority += min(question_complexity, 0.2)  # 20% weight from complexity

        # ตรวจสอบความยาวของคำตอบที่ถูกแก้ไข (คำตอบยาวๆ มักมีคุณค่าสูง)
        answer_value = min(len(corrected_answer.split()) * 0.005, 0.2)  # 20% weight from answer quality
        priority += answer_value

        # ตรวจสอบว่ามีคำถามที่คล้ายกันเคยมีปัญหาหรือไม่
        similar_issues = check_similar_issue_frequency(question)
        issue_frequency_bonus = min(similar_issues * 0.05, 0.2)  # 20% weight from frequency
        priority += issue_frequency_bonus

        return min(priority, 1.0)

    except Exception as e:
        logging.error(f"❌ Error calculating feedback priority: {str(e)}")
        return 0.5  # Default priority

def check_similar_issue_frequency(question: str) -> int:
    """ตรวจสอบว่ามีคำถามที่คล้ายกันเคยมีปัญหาบ่อยแค่ไหน"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # ค้นหาคำถามที่คล้ายกันที่เคยมีปัญหา
        question_start = question[:20] if len(question) > 20 else question
        question_end = question[-20:] if len(question) > 20 else ""
        cursor.execute('''
            SELECT COUNT(*) FROM feedback
            WHERE feedback_type = 'bad'
            AND (question LIKE ? OR question LIKE ?)
        ''', (f"%{question_start}%", f"%{question_end}%"))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    except Exception as e:
        logging.error(f"❌ Error checking similar issue frequency: {str(e)}")
        return 0

def apply_feedback_to_rag(question: str, corrected_answer: str, confidence: float = 0.9) -> bool:
    """นำ feedback ที่ถูกต้องไปปรับปรุง RAG ทันที (Real-time Learning Integration)"""
    try:
        # 1. สร้าง embedding สำหรับ corrected answer
        model = load_embedding_model()
        question_embedding = model.encode(question, convert_to_tensor=True).cpu().numpy()
        answer_embedding = model.encode(corrected_answer, convert_to_tensor=True).cpu().numpy()

        # 2. เพิ่ม corrected answer เข้า vector database พร้อม high weight
        global chroma_client
        collection = chroma_client.get_or_create_collection(name="pdf_data")

        # สร้าง unique ID สำหรับ corrected answer
        corrected_id = f"corrected_{abs(hash(question + corrected_answer))}_{int(time.time())}"

        # คำนวณ priority score สำหรับ corrected answer
        priority_score = calculate_feedback_priority(question, corrected_answer, confidence)

        # เพิ่มเข้า ChromaDB พร้อม metadata และ priority
        collection.add(
            embeddings=[question_embedding.tolist()],
            documents=[f"Q: {question}\nA: {corrected_answer}"],
            metadatas=[{
                "source": "feedback_corrected",
                "confidence": confidence,
                "priority_score": priority_score,
                "question": question,
                "answer": corrected_answer,
                "type": "corrected_answer",
                "created_at": datetime.now().isoformat()
            }],
            ids=[corrected_id]
        )

        logging.info(f"✅ Applied feedback to RAG system: {question[:50]}... -> {corrected_answer[:50]}...")
        return True

    except Exception as e:
        logging.error(f"❌ Failed to apply feedback to RAG: {str(e)}")
        return False


def increment_corrected_answer_usage(original_question: str) -> bool:
    """เพิ่มจำนวนครั้งที่คำตอบที่ถูกแก้ไขถูกนำไปใช้"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE corrected_answers
            SET applied_count = applied_count + 1
            WHERE original_question = ?
        ''', (original_question,))

        # อัปเดต feedback table ให้ applied = TRUE
        cursor.execute('''
            UPDATE feedback
            SET applied = TRUE
            WHERE question = ? AND corrected_answer != '' AND corrected_answer IS NOT NULL
        ''', (original_question,))

        conn.commit()
        conn.close()

        logging.info(f"✅ Incremented usage count for corrected answer: {original_question[:50]}...")
        return True

    except Exception as e:
        logging.error(f"❌ Failed to increment corrected answer usage: {str(e)}")
        return False


def get_learning_stats():
    """ดึงสถิติการเรียนรู้"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # จำนวนคำตอบที่ถูกแก้ไขทั้งหมด
        cursor.execute("SELECT COUNT(*) FROM corrected_answers")
        total_corrected = cursor.fetchone()[0]

        # จำนวนคำตอบที่ถูกนำไปใช้
        cursor.execute("SELECT COUNT(*) FROM corrected_answers WHERE applied_count > 0")
        used_corrected = cursor.fetchone()[0]

        # จำนวน feedback ทั้งหมด
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]

        # จำนวน feedback ที่ถูกแก้ไข
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE corrected_answer != '' AND corrected_answer IS NOT NULL")
        corrected_feedback = cursor.fetchone()[0]

        # คำตอบที่ถูกใช้บ่อยที่สุด
        cursor.execute('''
            SELECT original_question, applied_count
            FROM corrected_answers
            WHERE applied_count > 0
            ORDER BY applied_count DESC
            LIMIT 5
        ''')
        most_used = cursor.fetchall()

        conn.close()

        return {
            'total_corrected': total_corrected,
            'used_corrected': used_corrected,
            'total_feedback': total_feedback,
            'corrected_feedback': corrected_feedback,
            'learning_rate': (used_corrected / total_corrected * 100) if total_corrected > 0 else 0,
            'most_used': most_used
        }

    except Exception as e:
        logging.error(f"❌ Failed to get learning stats: {str(e)}")
        return {
            'total_corrected': 0, 'used_corrected': 0, 'total_feedback': 0,
            'corrected_feedback': 0, 'learning_rate': 0, 'most_used': []
        }

# Tag Management Functions
def create_tag(name: str, color: str = '#007bff', description: str = '') -> bool:
    """สร้าง tag ใหม่"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO tags (name, color, description) VALUES (?, ?, ?)
        ''', (name, color, description))

        conn.commit()
        conn.close()
        logging.info(f"✅ Created tag: {name}")
        return True

    except sqlite3.IntegrityError:
        logging.warning(f"⚠️ Tag '{name}' already exists")
        return False
    except Exception as e:
        logging.error(f"❌ Failed to create tag: {str(e)}")
        return False

def analyze_feedback_patterns() -> dict:
    """วิเคราะห์รูปแบบ feedback ด้วย AI เพื่อหา insights"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # ดึง feedback ล่าสุด 100 รายการ
        cursor.execute('''
            SELECT question, answer, feedback_type, user_comment, corrected_answer, timestamp
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT 100
        ''')

        feedback_data = cursor.fetchall()
        conn.close()

        if not feedback_data:
            return {"patterns": [], "recommendations": [], "quality_score": 0}

        # วิเคราะห์หมวดหมู่ปัญหา
        categories = {}
        quality_issues = []
        improvement_suggestions = []

        for fb in feedback_data:
            question, answer, feedback_type, comment, corrected, timestamp = fb

            # ตรวจจับรูปแบบจาก comments
            if comment:
                comment_lower = comment.lower()
                if any(word in comment_lower for word in ['ไม่เข้าใจ', 'สับสน', 'ยาก', 'ซับซ้อน']):
                    categories.setdefault('ความเข้าใจ', 0)
                    categories['ความเข้าใจ'] += 1
                elif any(word in comment_lower for word in ['ไม่ครบ', 'ขาด', 'เพิ่ม', 'ไม่พอ']):
                    categories.setdefault('ความครบถ้วน', 0)
                    categories['ความครบถ้วน'] += 1
                elif any(word in comment_lower for word in ['แหล่ง', 'อ้างอิง', 'source', 'reference']):
                    categories.setdefault('แหล่งข้อมูล', 0)
                    categories['แหล่งข้อมูล'] += 1

            # ตรวจจับคุณภาพต่ำ
            if feedback_type == 'bad' and corrected:
                quality_issues.append({
                    'question': question[:100],
                    'issue_type': 'incorrect_answer',
                    'has_correction': True
                })

        # สร้างคำแนะนำ
        if categories.get('ความเข้าใจ', 0) > 5:
            improvement_suggestions.append("🔍 ปรับปรุมคำอธิบายให้เข้าใจง่ายขึ้น")
        if categories.get('ความครบถ้วน', 0) > 5:
            improvement_suggestions.append("📝 เพิ่มรายละเอียดในคำตอบให้ครบถ้วนขึ้น")
        if categories.get('แหล่งข้อมูล', 0) > 3:
            improvement_suggestions.append("📎 ตรวจสอบความถูกต้องของแหล่งอ้างอิง")

        # คำนวณคะแนนคุณภาพ
        total_feedback = len(feedback_data)
        good_feedback = sum(1 for fb in feedback_data if fb[2] == 'good')
        quality_score = (good_feedback / total_feedback * 100) if total_feedback > 0 else 0

        return {
            "patterns": categories,
            "quality_issues": quality_issues[:10],  # จำกัด 10 รายการ
            "recommendations": improvement_suggestions,
            "quality_score": quality_score,
            "total_analyzed": total_feedback
        }

    except Exception as e:
        logging.error(f"❌ Failed to analyze feedback patterns: {str(e)}")
        return {"patterns": [], "recommendations": [], "quality_score": 0}

def get_comprehensive_analytics() -> dict:
    """ดึงข้อมูล analytics แบบครบถ้วน"""
    try:
        # ดึงสถิติพื้นฐาน
        basic_stats = get_feedback_stats()
        learning_stats = get_learning_stats()
        pattern_analysis = analyze_feedback_patterns()

        # ดึงสถิติตามช่วงเวลา
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # Feedback ในช่วง 7 วันล่าสุด
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count,
                   SUM(CASE WHEN feedback_type = 'good' THEN 1 ELSE 0 END) as good_count
            FROM feedback
            WHERE timestamp >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''')
        weekly_trend = cursor.fetchall()

        # Top problematic questions
        cursor.execute('''
            SELECT question, COUNT(*) as issue_count
            FROM feedback
            WHERE feedback_type = 'bad'
            GROUP BY question
            HAVING issue_count > 1
            ORDER BY issue_count DESC
            LIMIT 10
        ''')
        problematic_questions = cursor.fetchall()

        conn.close()

        return {
            "basic_stats": basic_stats,
            "learning_stats": learning_stats,
            "pattern_analysis": pattern_analysis,
            "weekly_trend": weekly_trend,
            "problematic_questions": problematic_questions,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"❌ Failed to get comprehensive analytics: {str(e)}")
        return {}

def get_all_tags() -> list:
    """ดึง tags ทั้งหมด"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, color, description, created_at
            FROM tags
            ORDER BY name
        ''')

        tags = cursor.fetchall()
        conn.close()
        return tags

    except Exception as e:
        logging.error(f"❌ Failed to get tags: {str(e)}")
        return []

def tag_document(document_id: str, tag_id: int) -> bool:
    """กำหนด tag ให้เอกสาร"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?, ?)
        ''', (document_id, tag_id))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logging.error(f"❌ Failed to tag document: {str(e)}")
        return False

def tag_feedback(feedback_id: int, tag_id: int) -> bool:
    """กำหนด tag ให้ feedback"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR IGNORE INTO feedback_tags (feedback_id, tag_id) VALUES (?, ?)
        ''', (feedback_id, tag_id))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logging.error(f"❌ Failed to tag feedback: {str(e)}")
        return False

def get_documents_by_tag(tag_id: int) -> list:
    """ดึงเอกสารตาม tag"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT dt.document_id
            FROM document_tags dt
            WHERE dt.tag_id = ?
        ''', (tag_id,))

        document_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return document_ids

    except Exception as e:
        logging.error(f"❌ Failed to get documents by tag: {str(e)}")
        return []

def get_feedback_by_tag(tag_id: int) -> list:
    """ดึง feedback ตาม tag"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT f.id, f.question, f.answer, f.feedback_type, f.timestamp, f.user_comment
            FROM feedback f
            JOIN feedback_tags ft ON f.id = ft.feedback_id
            WHERE ft.tag_id = ?
            ORDER BY f.timestamp DESC
        ''', (tag_id,))

        feedback = cursor.fetchall()
        conn.close()
        return feedback

    except Exception as e:
        logging.error(f"❌ Failed to get feedback by tag: {str(e)}")
        return []

def search_documents_by_tags(tag_ids: list) -> list:
    """ค้นหาเอกสารตามหลาย tags (AND logic)"""
    try:
        if not tag_ids:
            return []

        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # Build dynamic query for AND logic
        placeholders = ','.join(['?' for _ in tag_ids])
        query = f'''
            SELECT dt.document_id, COUNT(*) as match_count
            FROM document_tags dt
            WHERE dt.tag_id IN ({placeholders})
            GROUP BY dt.document_id
            HAVING match_count = ?
        '''

        cursor.execute(query, tag_ids + [len(tag_ids)])
        document_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return document_ids

    except Exception as e:
        logging.error(f"❌ Failed to search documents by tags: {str(e)}")
        return []

def delete_tag(tag_id: int) -> bool:
    """ลบ tag"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # Delete from junction tables first (foreign key constraints)
        cursor.execute('DELETE FROM document_tags WHERE tag_id = ?', (tag_id,))
        cursor.execute('DELETE FROM feedback_tags WHERE tag_id = ?', (tag_id,))

        # Delete the tag
        cursor.execute('DELETE FROM tags WHERE id = ?', (tag_id,))

        conn.commit()
        conn.close()
        logging.info(f"✅ Deleted tag: {tag_id}")
        return True

    except Exception as e:
        logging.error(f"❌ Failed to delete tag: {str(e)}")
        return False

def get_tag_stats() -> dict:
    """ดึงสถิติการใช้งาน tags"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # Most used tags
        cursor.execute('''
            SELECT t.name, COUNT(dt.id) as usage_count
            FROM tags t
            LEFT JOIN document_tags dt ON t.id = dt.tag_id
            GROUP BY t.id, t.name
            ORDER BY usage_count DESC
            LIMIT 10
        ''')

        most_used_tags = cursor.fetchall()

        # Tags with feedback
        cursor.execute('''
            SELECT t.name, COUNT(ft.id) as feedback_count
            FROM tags t
            LEFT JOIN feedback_tags ft ON t.id = ft.tag_id
            GROUP BY t.id, t.name
            ORDER BY feedback_count DESC
            LIMIT 10
        ''')

        feedback_tags = cursor.fetchall()

        conn.close()

        return {
            'most_used_tags': most_used_tags,
            'feedback_tags': feedback_tags
        }

    except Exception as e:
        logging.error(f"❌ Failed to get tag stats: {str(e)}")
        return {
            'most_used_tags': [],
            'feedback_tags': []
        }

# Tag UI Helper Functions
def refresh_tags_list():
    """รีเฟรชรายการ tags"""
    try:
        tags = get_all_tags()
        tag_choices = [(f"🏷️ {tag[1]}", tag[0]) for tag in tags]
        tag_data = [[tag[0], tag[1], tag[2], tag[3] or "", tag[4]] for tag in tags]
        return tag_data, tag_choices, gr.HTML(""), ""
    except Exception as e:
        logging.error(f"❌ Failed to refresh tags: {str(e)}")
        return [], [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>'), ""

def create_new_tag(name: str, color: str, description: str):
    """สร้าง tag ใหม่"""
    if not name.strip():
        return [], [], gr.HTML('<div style="color: orange;">⚠️ กรุณาใส่ชื่อ Tag</div>'), ""

    try:
        success = create_tag(name.strip(), color, description.strip())
        if success:
            return refresh_tags_list()
        else:
            return [], [], gr.HTML('<div style="color: orange;">⚠️ Tag นี้มีอยู่แล้ว</div>'), ""
    except Exception as e:
        logging.error(f"❌ Failed to create tag: {str(e)}")
        return [], [], gr.HTML(f'<div style="color: red;">❌ ไม่สามารถสร้าง Tag ได้: {str(e)}</div>'), ""

def delete_selected_tag(selected_row: dict):
    """ลบ tag ที่เลือก"""
    try:
        if not selected_row or not selected_row.get("ID"):
            return [], [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือก Tag ที่จะลบ</div>'), ""

        tag_id = selected_row["ID"]
        tag_name = selected_row.get("ชื่อ Tag", "")

        success = delete_tag(tag_id)
        if success:
            tag_data, tag_choices, _, _ = refresh_tags_list()
            return tag_data, tag_choices, gr.HTML(f'<div style="color: green;">✅ ลบ Tag "{tag_name}" สำเร็จ</div>'), ""
        else:
            return [], [], gr.HTML('<div style="color: red;">❌ ไม่สามารถลบ Tag ได้</div>'), ""
    except Exception as e:
        logging.error(f"❌ Failed to delete tag: {str(e)}")
        return [], [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>'), ""

def update_tag_statistics():
    """อัปเดตสถิติ tags"""
    try:
        stats = get_tag_stats()
        popular_data = [[tag[0], tag[1]] for tag in stats['most_used_tags']]
        feedback_data = [[tag[0], tag[1]] for tag in stats['feedback_tags']]
        return popular_data, feedback_data
    except Exception as e:
        logging.error(f"❌ Failed to update tag stats: {str(e)}")
        return [], []

def search_documents_by_selected_tags(selected_tags: list):
    """ค้นหาเอกสารตาม tags ที่เลือก"""
    try:
        if not selected_tags:
            return [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือกอย่างน้อย 1 Tag</div>')

        # Extract tag IDs from selected labels
        tags = get_all_tags()
        tag_id_map = {f"🏷️ {tag[1]}": tag[0] for tag in tags}
        selected_tag_ids = [tag_id_map[tag] for tag in selected_tags if tag in tag_id_map]

        if not selected_tag_ids:
            return [], gr.HTML('<div style="color: orange;">⚠️ ไม่พบ Tags ที่เลือก</div>')

        document_ids = search_documents_by_tags(selected_tag_ids)

        if not document_ids:
            return [], gr.HTML('<div style="color: blue;">ℹ️ ไม่พบเอกสารที่ตรงกับ Tags ที่เลือก</div>')

        # Get content preview from ChromaDB
        search_data = []
        for doc_id in document_ids[:20]:  # Limit to 20 results
            try:
                result = collection.get(ids=[doc_id])
                if result['documents']:
                    content = result['documents'][0][:100] + "..." if len(result['documents'][0]) > 100 else result['documents'][0]
                    search_data.append([doc_id, content])
            except:
                search_data.append([doc_id, "ไม่สามารถโหลดเนื้อหาได้"])

        status = gr.HTML(f'<div style="color: green;">✅ พบ {len(search_data)} เอกสาร</div>')
        return search_data, status
    except Exception as e:
        logging.error(f"❌ Failed to search by tags: {str(e)}")
        return [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>')

def load_feedback_by_selected_tag(tag_label: str):
    """โหลด feedback ตาม tag ที่เลือก"""
    try:
        if not tag_label:
            return [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือก Tag</div>')

        # Get tag ID from label
        tags = get_all_tags()
        tag_id_map = {f"🏷️ {tag[1]}": tag[0] for tag in tags}
        tag_id = tag_id_map.get(tag_label)

        if not tag_id:
            return [], gr.HTML('<div style="color: orange;">⚠️ ไม่พบ Tag ที่เลือก</div>')

        feedback_list = get_feedback_by_tag(tag_id)

        if not feedback_list:
            return [], gr.HTML('<div style="color: blue;">ℹ️ ไม่มี Feedback สำหรับ Tag นี้</div>')

        # Format feedback data
        feedback_data = []
        for fb in feedback_list:
            question = fb[1][:50] + "..." if len(fb[1]) > 50 else fb[1]
            answer = fb[2][:100] + "..." if len(fb[2]) > 100 else fb[2]
            feedback_data.append([fb[0], question, answer, fb[3], fb[4], fb[5] or ""])

        status = gr.HTML(f'<div style="color: green;">✅ พบ {len(feedback_data)} Feedback</div>')
        return feedback_data, status
    except Exception as e:
        logging.error(f"❌ Failed to load feedback by tag: {str(e)}")
        return [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>')

# Enhanced RAG System Classes
import time
import json
import pickle

class PerformanceMonitor:
    """Monitor RAG performance and log metrics"""

    @staticmethod
    def log_performance(session_id: str, rag_mode: str, question: str, response_time: float,
                       context_count: int, memory_hit: bool, success: bool, error_message: str = None):
        """Log performance metrics"""
        try:
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO rag_performance_log
                (session_id, rag_mode, question, response_time, context_count, memory_hit, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, rag_mode, question, response_time, context_count, memory_hit, success, error_message))

            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"❌ Failed to log performance: {str(e)}")

    @staticmethod
    def get_performance_stats(rag_mode: str = None, limit: int = 100):
        """Get performance statistics"""
        try:
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            query = "SELECT * FROM rag_performance_log"
            params = []

            if rag_mode:
                query += " WHERE rag_mode = ?"
                params.append(rag_mode)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()

            return results
        except Exception as e:
            logging.error(f"❌ Failed to get performance stats: {str(e)}")
            return []

class ContextCache:
    """Cache contexts for improved performance"""

    @staticmethod
    def _get_question_hash(question: str) -> str:
        """Generate hash for question"""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest()

    @staticmethod
    def get_cached_contexts(question: str) -> list:
        """Get cached contexts for question"""
        try:
            question_hash = ContextCache._get_question_hash(question)
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT contexts, access_count FROM context_cache
                WHERE question_hash = ?
            ''', (question_hash,))

            result = cursor.fetchone()

            if result:
                # Update access count
                cursor.execute('''
                    UPDATE context_cache
                    SET access_count = access_count + 1
                    WHERE question_hash = ?
                ''', (question_hash,))
                conn.commit()

                conn.close()
                return json.loads(result[0]) if result[0] else []

            conn.close()
            return None
        except Exception as e:
            logging.error(f"❌ Failed to get cached contexts: {str(e)}")
            return None

    @staticmethod
    def cache_contexts(question: str, contexts: list, question_embedding):
        """Cache contexts for question"""
        try:
            question_hash = ContextCache._get_question_hash(question)
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            embedding_bytes = pickle.dumps(question_embedding)
            contexts_json = json.dumps(contexts)

            cursor.execute('''
                INSERT OR REPLACE INTO context_cache
                (question_hash, question, contexts, embedding, access_count)
                VALUES (?, ?, ?, ?, 1)
            ''', (question_hash, question, contexts_json, embedding_bytes))

            conn.commit()
            conn.close()
            logging.info(f"✅ Cached contexts for question: {question[:50]}...")
        except Exception as e:
            logging.error(f"❌ Failed to cache contexts: {str(e)}")

class ImprovedStandardRAG:
    """Improved Standard RAG with memory and fallback"""

    def __init__(self, cache_size: int = 50):
        self.cache_size = cache_size
        self.question_cache = {}  # Simple in-memory cache
        self.fallback_responses = [
            "ขอโทษครับ ฉันไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัปโหลดไว้",
            "ตามเอกสารที่มีอยู่ ไม่พบข้อมูลเกี่ยวกับเรื่องนี้ครับ",
            "ฉันไม่สามารถตอบคำถามนี้จากเอกสารที่มีอยู่ได้ครับ"
        ]

    def get_similar_questions(self, question: str, limit: int = 3) -> list:
        """Get similar questions from database (improved memory)"""
        try:
            question_embedding = embed_text(question)
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            # Get recent questions with embeddings
            cursor.execute('''
                SELECT question, answer, question_embedding, relevance_score
                FROM session_memory
                WHERE question_embedding IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 50
            ''')

            results = cursor.fetchall()
            similar_questions = []

            for row in results:
                stored_embedding = pickle.loads(row[2])

                # Proper cosine similarity calculation
                similarity = self._cosine_similarity(question_embedding, stored_embedding)

                if similarity > 0.7:  # Threshold for similarity
                    similar_questions.append({
                        'question': row[0],
                        'answer': row[1],
                        'similarity': similarity,
                        'relevance_score': row[3]
                    })

            # Sort by similarity and return top results
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
            conn.close()

            return similar_questions[:limit]
        except Exception as e:
            logging.error(f"❌ Failed to get similar questions: {str(e)}")
            return []

    def _cosine_similarity(self, a, b):
        """Calculate proper cosine similarity"""
        import numpy as np
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    def get_fallback_answer(self, question: str) -> str:
        """Get fallback answer when no context found"""
        try:
            # Try to find similar questions first
            similar = self.get_similar_questions(question)
            if similar:
                return f"พบคำถามที่คล้ายกัน: {similar[0]['question']}\nคำตอบ: {similar[0]['answer']}"

            # Return generic fallback
            import random
            return random.choice(self.fallback_responses)
        except Exception as e:
            logging.error(f"❌ Failed to get fallback answer: {str(e)}")
            return "ขอโทษครับ เกิดข้อผิดพลาดในการประมวลผล"

    def save_to_memory(self, session_id: str, question: str, answer: str, contexts: list):
        """Save conversation to memory database"""
        try:
            question_embedding = embed_text(question)
            embedding_bytes = pickle.dumps(question_embedding)
            contexts_json = json.dumps(contexts[:3])  # Save top 3 contexts

            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO session_memory
                (session_id, question, answer, question_embedding, contexts, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, question, answer, embedding_bytes, contexts_json, 1.0))

            conn.commit()
            conn.close()
            logging.info(f"✅ Saved to memory: {question[:50]}...")
        except Exception as e:
            logging.error(f"❌ Failed to save to memory: {str(e)}")

class ImprovedEnhancedRAG:
    """Improved Enhanced RAG with proper similarity and persistence"""

    def __init__(self):
        self.session_memory = deque(maxlen=MEMORY_WINDOW_SIZE)
        self.conversation_history = []
        self.current_session_id = self._generate_session_id()
        self.performance_monitor = PerformanceMonitor()

    def _generate_session_id(self) -> str:
        """Generate unique session ID with better collision resistance"""
        timestamp = datetime.now().isoformat()
        random_str = str(time.time())[-6:]  # Add random component
        combined = f"{timestamp}_{random_str}"
        session_hash = hashlib.sha256(combined.encode()).hexdigest()[:12]
        return f"enhanced_session_{session_hash}"

    def get_relevant_memory(self, current_question: str, threshold: float = 0.75) -> list:
        """Get relevant memory with proper cosine similarity"""
        if not ENABLE_SESSION_MEMORY:
            return []

        try:
            start_time = time.time()
            current_embedding = embed_text(current_question)
            relevant_memories = []

            # Try database first
            db_memories = self._get_database_memories(current_embedding, threshold)
            if db_memories:
                relevant_memories.extend(db_memories)

            # Add in-memory results if needed
            if len(relevant_memories) < 3:
                memory_memories = self._get_in_memory_memories(current_embedding, threshold)
                relevant_memories.extend(memory_memories)

            # Sort by similarity and return top results
            relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
            result = relevant_memories[:5]  # Return top 5 instead of 3

            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.log_performance(
                self.current_session_id, "enhanced", current_question,
                response_time, 0, True, True
            )

            return result
        except Exception as e:
            logging.error(f"❌ Failed to get relevant memory: {str(e)}")
            return []

    def _get_database_memories(self, current_embedding, threshold: float) -> list:
        """Get memories from database with proper similarity"""
        try:
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            # Get recent memories from database
            cursor.execute('''
                SELECT question, answer, question_embedding, contexts
                FROM session_memory
                WHERE question_embedding IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            ''')

            results = cursor.fetchall()
            db_memories = []

            for row in results:
                stored_embedding = pickle.loads(row[2])

                # Proper cosine similarity
                similarity = self._cosine_similarity(current_embedding, stored_embedding)

                if similarity > threshold:
                    contexts = json.loads(row[3]) if row[3] else []
                    db_memories.append({
                        'question': row[0],
                        'answer': row[1],
                        'similarity': similarity,
                        'contexts': contexts,
                        'source': 'database'
                    })

            conn.close()
            return db_memories
        except Exception as e:
            logging.error(f"❌ Failed to get database memories: {str(e)}")
            return []

    def _get_in_memory_memories(self, current_embedding, threshold: float) -> list:
        """Get memories from in-memory deque"""
        memories = []

        for memory_entry in self.session_memory:
            memory_question = memory_entry["question"]
            memory_embedding = embed_text(memory_question)

            # Proper cosine similarity
            similarity = self._cosine_similarity(current_embedding, memory_embedding)

            if similarity > threshold:
                memories.append({
                    'question': memory_question,
                    'answer': memory_entry["answer"],
                    'similarity': similarity,
                    'contexts': memory_entry["contexts"],
                    'source': 'memory'
                })

        return memories

    def _cosine_similarity(self, a, b):
        """Calculate proper cosine similarity"""
        import numpy as np
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    def add_to_memory(self, session_id: str, question: str, answer: str, contexts: list):
        """Add conversation to both memory and database"""
        memory_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "contexts": contexts[:3]
        }

        self.session_memory.append(memory_entry)

        # Also save to database for persistence
        self._save_to_database(session_id, question, answer, contexts)

        # Add to conversation history
        self.conversation_history.extend([
            {"role": "user", "content": question, "timestamp": memory_entry["timestamp"]},
            {"role": "assistant", "content": answer, "timestamp": memory_entry["timestamp"]}
        ])

    def _save_to_database(self, session_id: str, question: str, answer: str, contexts: list):
        """Save to database for persistence"""
        try:
            question_embedding = embed_text(question)
            embedding_bytes = pickle.dumps(question_embedding)
            contexts_json = json.dumps(contexts[:3])

            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO session_memory
                (session_id, question, answer, question_embedding, contexts, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, question, answer, embedding_bytes, contexts_json, 1.0))

            conn.commit()
            conn.close()
            logging.info(f"✅ Enhanced RAG: Saved to database - {question[:50]}...")
        except Exception as e:
            logging.error(f"❌ Enhanced RAG: Failed to save to database - {str(e)}")

class RAGManager:
    """Main RAG Manager that handles both standard and enhanced modes"""

    def __init__(self):
        self.standard_rag = ImprovedStandardRAG()
        self.enhanced_rag = ImprovedEnhancedRAG()
        self.context_cache = ContextCache()
        self.performance_monitor = PerformanceMonitor()
        self.tag_retrieval = TagBasedRetrieval()
        self.current_session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate session ID"""
        timestamp = datetime.now().isoformat()
        session_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"rag_session_{session_hash}"

    def query(self, question: str, rag_mode: str = "enhanced", chat_llm: str = "gemma3:latest",
              show_source: bool = False, formal_style: bool = False):
        """Main query function that handles both modes"""
        start_time = time.time()

        try:
            # Check cache first
            cached_contexts = self.context_cache.get_cached_contexts(question)
            if cached_contexts:
                logging.info(f"✅ Cache hit for question: {question[:50]}...")
                # Use cached contexts but still process with LLM
                contexts = cached_contexts
                memory_hit = True
            else:
                memory_hit = False
                contexts = self._retrieve_contexts(question)

                # Cache the contexts for future use
                if contexts:
                    question_embedding = embed_text(question)
                    self.context_cache.cache_contexts(question, contexts, question_embedding)

            # Get relevant memories for enhanced mode
            relevant_memories = []
            if rag_mode == "enhanced":
                relevant_memories = self.enhanced_rag.get_relevant_memory(question)
                logging.info(f"Found {len(relevant_memories)} relevant memories")

            # Generate response
            response_generator = self._generate_response(
                question, contexts, relevant_memories, rag_mode,
                chat_llm, show_source, formal_style
            )

            # Save to memory
            if rag_mode == "enhanced":
                self.enhanced_rag.add_to_memory(
                    self.current_session_id, question, "", contexts
                )
            else:
                self.standard_rag.save_to_memory(
                    self.current_session_id, question, "", contexts
                )

            # Log performance
            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.log_performance(
                self.current_session_id, rag_mode, question,
                response_time, len(contexts), memory_hit, True
            )

            return response_generator

        except Exception as e:
            error_msg = f"❌ RAG Query failed: {str(e)}"
            logging.error(error_msg)

            # Log performance error
            response_time = (time.time() - start_time) * 1000
            self.performance_monitor.log_performance(
                self.current_session_id, rag_mode, question,
                response_time, 0, memory_hit, False, str(e)
            )

            # Return fallback response
            if rag_mode == "standard":
                fallback = self.standard_rag.get_fallback_answer(question)
            else:
                fallback = "ขอโทษครับ เกิดข้อผิดพลาดในระบบ Enhanced RAG กรุณาลองใหม่"

            def fallback_generator():
                yield fallback
                return

            return fallback_generator()

    def _retrieve_contexts(self, question: str) -> list:
        """Retrieve contexts from ChromaDB with tag-based enhancement"""
        try:
            # Step 1: Extract tags from question
            question_tags, tag_analysis = self.tag_retrieval.tag_question_and_enhance_search(question)

            # Step 2: Get base contexts from ChromaDB
            question_embedding = embed_text(question)
            max_result = determine_optimal_results(question)

            results = collection.query(
                query_embeddings=[question_embedding.tolist()],
                n_results=max_result
            )

            # Step 3: Filter relevant contexts
            base_contexts = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                base_contexts.append({'text': doc, 'metadata': meta})

            filtered_contexts = filter_relevant_contexts(
                question, results["documents"][0], results["metadatas"][0], min_relevance=0.05
            )

            if len(filtered_contexts) == 0:
                logging.warning("No contexts passed relevance filter, using all retrieved contexts")
                filtered_contexts = base_contexts

            # Step 4: Apply tag-based weighting if tags found
            if question_tags:
                final_contexts = self.tag_retrieval.get_tag_weighted_contexts(
                    question, filtered_contexts, question_tags
                )

                # Add tag info to context metadata
                for ctx in final_contexts:
                    if 'tag_relevance_score' in ctx:
                        ctx['metadata']['tag_relevance_score'] = ctx['tag_relevance_score']
                        ctx['metadata']['matching_tags'] = ctx.get('matching_tags', [])

                logging.info(f"🎯 Applied tag-based ranking: {len(final_contexts)} contexts with tags {question_tags}")
                return final_contexts
            else:
                logging.info("📝 No tags found, using standard retrieval")
                return filtered_contexts

        except Exception as e:
            logging.error(f"❌ Failed to retrieve contexts: {str(e)}")
            return []

    def _generate_response(self, question: str, contexts: list, relevant_memories: list,
                          rag_mode: str, chat_llm: str, show_source: bool, formal_style: bool):
        """Generate response using appropriate RAG mode"""
        try:
            if rag_mode == "enhanced":
                # Use Enhanced RAG logic
                return self._enhanced_response_generator(
                    question, contexts, relevant_memories, chat_llm, show_source, formal_style
                )
            else:
                # Use Standard RAG logic
                return self._standard_response_generator(
                    question, contexts, chat_llm, show_source, formal_style
                )
        except Exception as e:
            logging.error(f"❌ Failed to generate response: {str(e)}")
            def error_generator():
                yield "ขอโทษครับ เกิดข้อผิดพลาดในการสร้างคำตอบ"
                return
            return error_generator()

    def _standard_response_generator(self, question: str, contexts: list, chat_llm: str,
                                   show_source: bool, formal_style: bool):
        """Standard RAG response generator"""
        if not contexts:
            # Use fallback mechanism
            fallback = self.standard_rag.get_fallback_answer(question)
            def fallback_gen():
                yield fallback
                return
            return fallback_gen()

        # Build standard prompt
        prompt = self._build_standard_prompt(question, contexts, show_source, formal_style)

        # Generate streaming response
        return chat_with_model_streaming(chat_llm, prompt, [])

    def _enhanced_response_generator(self, question: str, contexts: list, relevant_memories: list,
                                   chat_llm: str, show_source: bool, formal_style: bool):
        """Enhanced RAG response generator with memory"""
        # Use existing enhanced RAG logic but with improvements
        return query_rag(question, chat_llm, show_source, formal_style)

    def _build_standard_prompt(self, question: str, contexts: list, show_source: bool, formal_style: bool) -> str:
        """Build standard RAG prompt"""
        style_instruction = "ตอบอย่างเป็นทางการ" if formal_style else "ตอบอย่างเป็นกันเอง"

        context_text = "\n\n".join([f"ข้อมูล {i+1}: {ctx['text']}" for i, ctx in enumerate(contexts[:5])])

        return f"""คุณเป็นผู้ช่วยตอบคำถามที่มีความรู้ด้านเอกสารที่อัปโหลดไว้

**แนวทางการตอบ:**
- {style_instruction}
- ให้คำตอบที่สอดคล้องกับข้อมูลในเอกสารเป็นหลัก
- ถ้าไม่พบข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่พบข้อมูล

**ข้อมูลที่เกี่ยวข้อง:**
{context_text}

**คำถาม:** {question}

**คำตอบ:**"""

# Advanced Tag System with LLM Integration
import re
from typing import List, Dict, Tuple

class LLMTagger:
    """LLM-powered tag suggestion and analysis"""

    def __init__(self):
        # Predefined tag patterns for automatic detection
        self.tag_patterns = {
            'ชำระ': [r'ชำระ', r'จ่ายเงิน', r'การจ่าย', r'เงิน', r'บิล', r'ค่าใช้จ่าย', r'ค่าบริการ'],
            'ปัญหา': [r'ปัญหา', r'ผิดพลาด', r'error', r'ไม่ได้', r'ล้มเหลว', r'บัก', r'ขัดข้อง'],
            'สอบถาม': [r'สอบถาม', r'ถาม', r'อยากรู้', r'ต้องการทราบ', r'ข้อมูล', r'รายละเอียด'],
            'เทคนิค': [r'วิธี', r'ขั้นตอน', r'การตั้งค่า', r'configure', r'setup', r'การใช้งาน'],
            'สำคัญ': [r'สำคัญ', r'เร่งด่วน', r'ฉุกเฉิน', r'ด่วน', r'จำเป็น', r'สำคัญมาก'],
            'เอกสาร': [r'เอกสาร', r'ไฟล์', r'PDF', r'doc', r'ข้อมูล', r'เนื้อหา'],
            'ระบบ': [r'ระบบ', r'system', r'โปรแกรม', r'application', r'แอป', r'ซอฟต์แวร์'],
            'บัญชี': [r'บัญชี', r'account', r'user', r'ผู้ใช้', r'login', r'รหัสผ่าน']
        }

    def extract_tags_from_text(self, text: str) -> List[str]:
        """Extract tags from text using pattern matching"""
        found_tags = []
        text_lower = text.lower()

        for tag_name, patterns in self.tag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_tags.append(tag_name)
                    break

        return list(set(found_tags))  # Remove duplicates

    def suggest_tags_with_llm(self, text: str, context: str = "") -> List[str]:
        """Use LLM to suggest relevant tags"""
        try:
            # Create a simple prompt for tag suggestion
            prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านการจัดหมวดหมู่เนื้อหา

จากข้อความต่อไปนี้:
"{text}"

กรุณาแนะนำ tags ที่เหมาะสม (ไม่เกิน 5 tags):
- ใช้ภาษาไทย
- ใช้คำสั้นๆ ที่เข้าใจง่าย
- พิจารณาความหมายโดยรวม

เลือกจาก tags เหล่านี้: {', '.join(self.tag_patterns.keys())}

ถ้าไม่มี tags ที่เหมาะสม ตอบว่า "ไม่มี"

Tags ที่แนะนำ:"""

            # Use existing model for tag suggestion
            response = list(chat_with_model_streaming("gemma3:latest", prompt, []))

            if response:
                suggested_text = "".join(response).strip()

                # Extract tags from response
                suggested_tags = []
                for tag_name in self.tag_patterns.keys():
                    if tag_name in suggested_text:
                        suggested_tags.append(tag_name)

                return suggested_tags

            return []
        except Exception as e:
            logging.error(f"❌ Failed to suggest tags with LLM: {str(e)}")
            return []

class TagBasedRetrieval:
    """Enhanced retrieval system using tags"""

    def __init__(self):
        self.llm_tagger = LLMTagger()

    def tag_question_and_enhance_search(self, question: str) -> Tuple[List[str], Dict]:
        """Tag question and enhance search with tag-based filtering"""
        try:
            # Extract tags using both methods
            pattern_tags = self.llm_tagger.extract_tags_from_text(question)
            llm_tags = self.llm_tagger.suggest_tags_with_llm(question)

            # Combine and prioritize
            all_tags = list(set(pattern_tags + llm_tags))

            tag_analysis = {
                'pattern_tags': pattern_tags,
                'llm_tags': llm_tags,
                'all_tags': all_tags,
                'tag_count': len(all_tags)
            }

            logging.info(f"🏷️ Question tags: {all_tags}")
            return all_tags, tag_analysis

        except Exception as e:
            logging.error(f"❌ Failed to tag question: {str(e)}")
            return [], {}

    def get_tag_weighted_contexts(self, question: str, base_contexts: List[Dict], question_tags: List[str]) -> List[Dict]:
        """Weight contexts based on tag relevance"""
        try:
            if not question_tags or not base_contexts:
                return base_contexts

            weighted_contexts = []

            for ctx in base_contexts:
                context_text = ctx.get('text', '')
                context_metadata = ctx.get('metadata', {})

                # Calculate tag relevance score
                tag_score = self._calculate_tag_relevance(context_text, question_tags)

                # Get document ID for additional tag lookup
                doc_id = context_metadata.get('id', '')
                document_tags = self._get_document_tags(doc_id)

                # Additional score if document has matching tags
                document_tag_score = len(set(question_tags) & set(document_tags))

                # Combined score
                total_score = tag_score + (document_tag_score * 0.5)

                # Create weighted context
                weighted_ctx = ctx.copy()
                weighted_ctx['tag_relevance_score'] = total_score
                weighted_ctx['matching_tags'] = list(set(question_tags) & set(document_tags))
                weighted_ctx['document_tags'] = document_tags

                weighted_contexts.append(weighted_ctx)

            # Sort by tag relevance
            weighted_contexts.sort(key=lambda x: x.get('tag_relevance_score', 0), reverse=True)

            logging.info(f"🎯 Tag-weighted contexts: {len(weighted_contexts)} with relevance scores")
            return weighted_contexts

        except Exception as e:
            logging.error(f"❌ Failed to weight contexts by tags: {str(e)}")
            return base_contexts

    def _calculate_tag_relevance(self, text: str, tags: List[str]) -> float:
        """Calculate how relevant text is to given tags"""
        score = 0.0
        text_lower = text.lower()

        for tag in tags:
            if tag in self.llm_tagger.tag_patterns:
                patterns = self.llm_tagger.tag_patterns[tag]
                tag_matches = sum(1 for pattern in patterns
                               if re.search(pattern, text_lower, re.IGNORECASE))
                score += tag_matches

        return score

    def _get_document_tags(self, document_id: str) -> List[str]:
        """Get tags associated with a document"""
        try:
            if not document_id:
                return []

            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT t.name FROM document_tags dt
                JOIN tags t ON dt.tag_id = t.id
                WHERE dt.document_id = ?
            ''', (document_id,))

            tags = [row[0] for row in cursor.fetchall()]
            conn.close()

            return tags
        except Exception as e:
            logging.error(f"❌ Failed to get document tags: {str(e)}")
            return []

    def auto_tag_document(self, document_id: str, content: str) -> List[str]:
        """Automatically tag a document based on its content"""
        try:
            suggested_tags = self.llm_tagger.suggest_tags_with_llm(content)
            pattern_tags = self.llm_tagger.extract_tags_from_text(content)

            all_tags = list(set(suggested_tags + pattern_tags))

            # Save tags to database
            for tag_name in all_tags:
                self._tag_document(document_id, tag_name)

            logging.info(f"🏷️ Auto-tagged document {document_id} with tags: {all_tags}")
            return all_tags

        except Exception as e:
            logging.error(f"❌ Failed to auto-tag document: {str(e)}")
            return []

    def _tag_document(self, document_id: str, tag_name: str):
        """Tag a document (create tag if needed)"""
        try:
            conn = sqlite3.connect(FEEDBACK_DB_PATH)
            cursor = conn.cursor()

            # Get or create tag
            cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
            result = cursor.fetchone()

            if result:
                tag_id = result[0]
            else:
                # Create new tag with default color
                cursor.execute('''
                    INSERT INTO tags (name, color, description) VALUES (?, ?, ?)
                ''', (tag_name, '#007bff', f'Auto-generated tag for {tag_name}'))
                tag_id = cursor.lastrowid

            # Tag the document
            cursor.execute('''
                INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?, ?)
            ''', (document_id, tag_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"❌ Failed to tag document: {str(e)}")

def get_feedback_stats():
    """ดึงสถิติ feedback"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        # สถิติทั่วไป
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'good'")
        good = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'bad'")
        bad = cursor.fetchone()[0]

        # Feedback ล่าสุด
        cursor.execute('''
            SELECT question, answer, feedback_type, timestamp, user_comment
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        recent_feedback = cursor.fetchall()

        conn.close()

        return {
            "total": total,
            "good": good,
            "bad": bad,
            "accuracy": (good / total * 100) if total > 0 else 0,
            "recent": recent_feedback
        }
    except Exception as e:
        logging.error(f"❌ Failed to get feedback stats: {str(e)}")
        return {"total": 0, "good": 0, "bad": 0, "accuracy": 0, "recent": []}


def delete_feedback(feedback_id: int):
    """ลบ feedback ตาม ID"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        affected = cursor.rowcount

        conn.commit()
        conn.close()

        if affected > 0:
            logging.info(f"✅ Deleted feedback ID: {feedback_id}")
            return True
        else:
            logging.warning(f"⚠️ Feedback ID {feedback_id} not found")
            return False
    except Exception as e:
        logging.error(f"❌ Failed to delete feedback: {str(e)}")
        return False


def export_feedback():
    """ส่งออกข้อมูล feedback เป็น CSV"""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, question, answer, feedback_type, user_comment, corrected_answer,
                   timestamp, model_used, sources
            FROM feedback
            ORDER BY timestamp DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        # สร้าง CSV string
        csv_data = []
        csv_data.append("ID,Question,Answer,Feedback Type,User Comment,Corrected Answer,Timestamp,Model,Sources")

        for row in rows:
            # Escape quotes in text fields
            q1 = str(row[1]).replace('"', '""') if row[1] else ""
            q2 = str(row[2]).replace('"', '""') if row[2] else ""
            q4 = str(row[4]).replace('"', '""') if row[4] else ""
            q5 = str(row[5]).replace('"', '""') if row[5] else ""
            q8 = str(row[8]).replace('"', '""') if row[8] else ""

            csv_row = [
                str(row[0]),
                f'"{q1}"',  # Question
                f'"{q2}"',  # Answer
                row[3],      # Feedback type
                f'"{q4}"',  # User comment
                f'"{q5}"',  # Corrected answer
                row[6],      # Timestamp
                row[7],      # Model
                f'"{q8}"'   # Sources
            ]
            csv_data.append(",".join(csv_row))

        return "\n".join(csv_data)
    except Exception as e:
        logging.error(f"❌ Failed to export feedback: {str(e)}")
        return None


# ==================== END FEEDBACK FUNCTIONS ====================


def chatbot_interface(history: List[Dict], llm_model: str, ai_provider: str = "ollama", show_source: bool = False, formal_style: bool = False,
                       send_to_discord: bool = False, send_to_line: bool = False, send_to_facebook: bool = False,
                       line_user_id: str = "", fb_user_id: str = "", use_graph_reasoning: bool = False,
                       reasoning_mode: str = "hybrid", multi_hop_enabled: bool = False, hop_count: int = 2):
    """
    อินเทอร์เฟซแชทบอทแบบ streaming with LightRAG support
    """
    print(f"DEBUG: chatbot_interface received - Provider: {ai_provider}, Model: {llm_model}")  # Debug
    user_message = history[-1]["content"]

    # ส่งคืนคำถามปัจจุบันสำหรับ feedback
    current_q = user_message

    # Choose query method based on LightRAG settings
    if use_graph_reasoning:
        if multi_hop_enabled:
            # Use multi-hop reasoning
            logging.info(f"🔄 Using Multi-Hop LightRAG (hops: {hop_count}) for query: {user_message}")
            stream = query_rag_with_multi_hop(
                user_message,
                chat_llm=llm_model,
                ai_provider=ai_provider,
                show_source=show_source,
                formal_style=formal_style,
                hop_count=hop_count
            )
        else:
            # Use standard graph reasoning
            logging.info(f"🧠 Using LightRAG Graph Reasoning (mode: {reasoning_mode}) for query: {user_message}")
            import asyncio

            # Run async query in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    query_rag_with_lightrag(
                        user_message,
                        chat_llm=llm_model,
                        ai_provider=ai_provider,
                        show_source=show_source,
                        formal_style=formal_style,
                        use_graph_reasoning=True,
                        reasoning_mode=reasoning_mode
                    )
                )
                # Convert result to stream format
                if isinstance(result, str):
                    # If result is already a string, create a simple stream
                    stream = ({"message": {"content": result}} for _ in range(1))
                else:
                    # If result is a stream generator, use it directly
                    stream = result
            finally:
                loop.close()
    else:
        # Use standard RAG
        stream = query_rag(user_message, chat_llm=llm_model, ai_provider=ai_provider, show_source=show_source, formal_style=formal_style)

    history.append({"role": "assistant", "content": ""})
    full_answer=""
    """
    ส่วนของการ ตอบคำถาม
    """
    for chunk in stream:
        content = chunk["message"]["content"]
        full_answer += content
        history[-1]["content"] += content
        #logging.info(f"content: {content}")
        yield history, current_q, full_answer, json.dumps([]) if show_source else ""

    """
    ส่วนของการดึงรูปภาพ ที่เกี่ยวข้องมาแสดง โดยดึงจาก คำตอบด้านบน
    """

    # ใช้ regex เพื่อดึงชื่อไฟล์ที่อยู่ใน [ภาพ: ...]
    print(full_answer)
    pattern1 = r"\[(?:ภาพ:\s*)?(pic_\w+[-_]?\w*\.(?:jpe?g|png))\]"
    pattern2 = r"(pic_\w+[-_]?\w*\.(?:jpe?g|png))"
    # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา

    print("----------PPPP------------")
    image_list = re.findall(pattern1, full_answer)
    print(image_list)
    if (len(image_list)==0):
        image_list = re.findall(pattern2, full_answer)
    print("----------xxxx------------")
    # ดึงเฉพาะรูปที่ไม่ซ้ำกัน
    image_list_uniq = list(dict.fromkeys(image_list))
    if image_list_uniq:
        history[-1]["content"] += "\n\nรูปภาพที่เกี่ยวข้อง:"
        yield history, current_q, full_answer, json.dumps([]) if show_source else ""

        # ดึงรูปมาแสดง
        for img in image_list_uniq:
            img_path = f"{TEMP_IMG}/{img}"
            logger.info(f"img_path: {img_path}")
            if os.path.exists(img_path):
                    image = Image.open(img_path)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_response = f"{img} ![{img}](data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()})"
                    #ส่งรูปไปที่ Chat
                    history.append({"role": "assistant", "content": image_response })
                    yield history, current_q, full_answer, json.dumps([]) if show_source else ""

    # Learning from corrected answers
    try:
        similar_corrected = find_similar_corrected_answer(user_message, threshold=0.85)
        if similar_corrected:
            # ใช้คำตอบที่ถูกแก้ไขแทน
            full_answer = similar_corrected['corrected_answer']
            logging.info(f"🎓 Applied learned correction (similarity: {similar_corrected['similarity']:.2f}): {user_message[:50]}...")

            # อัปเดตประวัติการใช้งาน
            increment_corrected_answer_usage(similar_corrected['original_question'])

            # เพิ่มข้อความแจ้งว่าใช้คำตอบที่ถูกเรียนรู้
            full_answer += f"\n\n💡 *คำตอบนี้ได้รับการปรับปรุงจากการเรียนรู้จากคำตอบที่ถูกแนะนำมาก่อน (ความคล้ายกัน: {similar_corrected['similarity']:.1%})*"
    except Exception as e:
        logging.warning(f"⚠️ Failed to apply learning from corrected answers: {str(e)}")

    # Store conversation in memory for Enhanced RAG
    if RAG_MODE == "enhanced":
        try:
            enhanced_rag.add_to_memory(user_message, full_answer, [])
            logging.info("Stored conversation in Enhanced RAG memory")
        except Exception as e:
            logging.error(f"Failed to store in Enhanced RAG memory: {str(e)}")

    # ส่งคำตอบไปยังแพลตฟอร์มที่เลือก
    try:
        # ส่งไปยัง Discord (ถ้าเลือก)
        if send_to_discord and DISCORD_ENABLED:
            send_to_discord_sync(user_message, full_answer)
            logging.info("✅ ส่งคำตอบไปยัง Discord สำเร็จ")

        # ส่งไปยัง LINE OA (ถ้าเลือกและมี user_id)
        if send_to_line and LINE_ENABLED and line_user_id and line_bot_api:
            try:
                # จำกัดความยาวข้อความสำหรับ LINE
                line_answer = full_answer
                if len(line_answer) > 4900:
                    line_answer = line_answer[:4900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

                line_bot_api.push_message(
                    line_user_id,
                    TextSendMessage(text=f"คำถาม: {user_message}\n\nคำตอบ:\n{line_answer}")
                )
                logging.info("✅ ส่งคำตอบไปยัง LINE OA สำเร็จ")
            except Exception as e:
                logging.error(f"❌ ไม่สามารถส่งไปยัง LINE OA: {str(e)}")

        # ส่งไปยัง Facebook Messenger (ถ้าเลือกและมี user_id)
        if send_to_facebook and FB_ENABLED and fb_user_id:
            try:
                # จำกัดความยาวข้อความสำหรับ Facebook
                fb_answer = full_answer
                if len(fb_answer) > 1900:
                    fb_answer = fb_answer[:1900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

                send_facebook_message(
                    fb_user_id,
                    f"คำถาม: {user_message}\n\nคำตอบ:\n{fb_answer}"
                )
                logging.info("✅ ส่งคำตอบไปยัง Facebook Messenger สำเร็จ")
            except Exception as e:
                logging.error(f"❌ ไม่สามารถส่งไปยัง Facebook Messenger: {str(e)}")

    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการส่งคำตอบไปยังแพลตฟอร์ม: {str(e)}")

    # Final yield with complete data
    yield history, current_q, full_answer, json.dumps([]) if show_source else ""

# Global LightRAG functions for UI access
def update_lightrag_status():
    """Get LightRAG system status for UI display"""
    try:
        if LIGHT_RAG_AVAILABLE:
            status = get_lightrag_system_status()
            if status.get("status") == "✅ LightRAG Available":
                return f"""✅ LightRAG พร้อมใช้งาน
• ChromaDB Records: {status.get('chroma_records', 'N/A')}
• Graph Available: {'✅' if status.get('graph_available') else '❌'}
• Mode: Mock Implementation (พร้อมอัปเกรดเมื่อ API สมบูรณ์)"""
            else:
                return f"⚠️ LightRAG มีปัญหา: {status.get('error', 'Unknown error')}"
        else:
            return "❌ LightRAG ไม่สามารถใช้งานได้ (Package ไม่พร้อม)"
    except Exception as e:
        return f"❌ ตรวจสอบสถานะไม่ได้: {str(e)}"

def test_graph_reasoning_interface():
    """Test LightRAG functionality for UI"""
    try:
        import asyncio

        async def run_test():
            test_query = "ทดสอบวิเคราะห์ความสัมพันธ์ในระบบ"
            result = await query_with_graph_reasoning(test_query, mode="hybrid")

            if result.get("error"):
                return f"❌ ทดสอบล้มเหลว: {result['error']}"

            return f"""✅ ทดสอบสำเร็จ!
• Query: {test_query}
• Processing Time: {result.get('processing_time', 0):.2f}s
• Response Length: {len(result.get('result', ''))} chars
• Mock Mode: {'ใช่' if result.get('mock') else 'ไม่ใช่'}
• Insights: {result.get('graph_insights', {})}"""

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            return result, gr.update(visible=True)
        finally:
            loop.close()

    except Exception as e:
        return f"❌ ทดสอบล้มเหลว: {str(e)}", gr.update(visible=True)

# Use main interface directly for now
# Authentication will be handled in a future update
def create_authenticated_interface():
    """Create interface with authentication check"""
    return demo

# Gradio interface

# Create webhook route handlers (will be added to Gradio's FastAPI app later)
try:
    from fastapi import Request, HTTPException
    from fastapi.responses import PlainTextResponse
    FASTAPI_AVAILABLE = True

    # Define webhook handlers as functions (will be registered to demo.app later)
    async def line_callback_api(request: Request):
        """LINE Webhook Callback via FastAPI"""
        if not LINE_ENABLED:
            raise HTTPException(status_code=403, detail="LINE Bot is disabled")

        try:
            # Get LINE signature
            signature = request.headers.get('X-Line-Signature', '')
            body = await request.body()

            # Setup LINE handler if not already initialized
            if not line_handler and LINE_ENABLED:
                setup_line_bot()

            if not line_handler:
                raise HTTPException(status_code=400, detail="LINE handler not initialized")

            # Handle webhook
            line_handler.handle(body.decode('utf-8'), signature)
            return PlainTextResponse('OK')

        except Exception as e:
            logging.error(f"LINE webhook error: {e}")
            return PlainTextResponse('OK', status_code=200)  # Return 200 to avoid webhook retries

    async def facebook_webhook_api(request: Request):
        """Facebook Webhook via FastAPI"""
        if not FB_ENABLED:
            raise HTTPException(status_code=403, detail="Facebook Bot is disabled")

        if request.method == 'GET':
            hub_mode = request.query_params.get('hub.mode')
            hub_challenge = request.query_params.get('hub.challenge')
            hub_verify_token = request.query_params.get('hub.verify_token')

            if hub_mode == 'subscribe' and hub_challenge:
                if hub_verify_token != FB_VERIFY_TOKEN:
                    raise HTTPException(status_code=403, detail="Verification token mismatch")
                return PlainTextResponse(hub_challenge)
            return PlainTextResponse('Hello')

        elif request.method == 'POST':
            try:
                data = await request.json()
                if data and "object" in data and data["object"] == "page":
                    for entry in data["entry"]:
                        for messaging_event in entry["messaging"]:
                            if messaging_event.get("message"):
                                sender_id = messaging_event["sender"]["id"]
                                message_text = messaging_event["message"].get("text")
                                if message_text:
                                    # Process in background
                                    threading.Thread(
                                        target=process_facebook_question,
                                        args=(sender_id, message_text)
                                    ).start()
                return PlainTextResponse('OK', status_code=200)

            except Exception as e:
                logging.error(f"Facebook webhook error: {e}")
                return PlainTextResponse('OK', status_code=200)

except ImportError as e:
    logging.warning(f"FastAPI not available: {e}")
    FASTAPI_AVAILABLE = False

# Gradio interface with FastAPI routes
demo = gr.Blocks(
    css="""
    .upload-container {
        border: 2px dashed #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        border-color: #4CAF50;
        background: #f5f5f5;
    }
    .drop-zone {
        min-height: 150px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        border: 2px dashed #ccc;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .drop-zone:hover {
        border-color: #4CAF50;
        background: #f9f9f9;
    }
    .options-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .checkbox-primary label {
        color: #495057;
        font-weight: 500;
    }
    .checkbox-secondary label {
        color: #6c757d;
        font-size: 0.9em;
    }
    .upload-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border: none;
        color: white;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .upload-button:hover {
        background: linear-gradient(45deg, #45a049, #3d8b40);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .status-output {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
    }
    """)

with demo:
    logo="https://camo.githubusercontent.com/9433204b08afdc976c2e4f5a4ba0d81f8877b585cc11206e2969326d25c41657/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f6e61726f6e67736b6d6c2f68746d6c352d6c6561726e406c61746573742f6173736574732f696d67732f546c697665636f64654c6f676f2d3435302e77656270"
    gr.Markdown(f"""<h3 style='display: flex; align-items: center; gap: 15px; padding: 10px; margin: 0;'>
        <img alt='T-LIVE-CODE' src='{logo}' style='height: 100px;' >
        <span style='font-size: 1.5em;'>แชทบอท PDF: RAG</span></h3>""")

    with gr.Tab("📚 อัปโหลดเอกสาร"):
        gr.Markdown("""
        ### 📁 อัปโหลดเอกสารและจัดการระบบข้อมูล
        รองรับไฟล์ **PDF, DOCX, TXT, MD** พร้อมระบบข้อมูลอัตโนมัติ
        """)

        # Create a more professional upload section with drag-and-drop
        with gr.Column():
            # Upload area with drag-and-drop support
            with gr.Group(elem_classes="upload-container"):
                files_input = gr.File(
                    label="ลากไฟล์มาวางที่นี่หรือคลิกเพื่อเลือกไฟล์",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                    height=150,
                    elem_classes="drop-zone",
                    show_label=True,
                    container=True,
                    scale=1
                )

            # File display area
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("""
                    **📋 รายการไฟล์ที่เลือก:**
                    - ยังไม่ได้เลือกไฟล์
                    - รองรับ PDF, TXT, MD, DOCX
                    - ขนาดสูงสุด: 100MB ต่อไฟล์
                    """)

                    selected_files_info = gr.Textbox(
                        label="ไฟล์ที่เลือก",
                        value="ยังไม่ได้เลือกไฟล์",
                        interactive=False,
                        lines=3,
                        max_lines=5
                    )

                with gr.Column(scale=2):
                    gr.Markdown("""
                    **💡 แนะนำ:**
                    • ลากไฟล์มาวางที่เลือก
                    • หรือคลิกปุ่ม "Browse Files" เพื่อเลือก
                    • รองรับไฟล์พร้อมกันได้หลายไฟล์
                    """)

        # Processing options with better styling
        with gr.Group(elem_classes="options-container"):
            with gr.Row():
                with gr.Column(scale=2):
                    clear_before_upload = gr.Checkbox(
                        label="🗑️ ล้างข้อมูลเก่าก่อนอัปโหลด",
                        value=False,
                        info="จะลบข้อมูลทั้งหมดก่อนเพิ่มเอกสารใหม่",
                        elem_classes="checkbox-primary"
                    )

                with gr.Column(scale=2):
                    include_memory_checkbox = gr.Checkbox(
                        label="🧠 รวม Enhanced RAG Memory",
                        value=(RAG_MODE == "enhanced"),
                        info="บันทึกความทรงจำการสนทนาพร้อม",
                        elem_classes="checkbox-secondary"
                    )

            # Action buttons with better styling
            with gr.Row():
                upload_button = gr.Button(
                    "📤 เริ่มต้นอัปโหลด",
                    variant="primary",
                    size="lg",
                    elem_classes="upload-button"
                )
                clear_button = gr.Button(
                    "🗑️ ล้างข้อมูลทั้งหมด",
                    variant="secondary",
                    size="lg"
                )

        # Status display
        with gr.Accordion("📊 สถานะการประมวลผล", open=True):
            upload_output = gr.Textbox(
                label="ผลลัพธ์",
                lines=5,
                interactive=False,
                elem_classes="status-output"
            )

        # Connect event handlers
        files_input.upload(
            fn=handle_file_selection,
            inputs=[files_input],
            outputs=[selected_files_info, upload_output]
        )

        files_input.clear(
            fn=lambda: ([], "ล้างรายการไฟล์แล้ว"),
            inputs=[],
            outputs=[selected_files_info]
        )

        upload_button.click(
            fn=process_multiple_files,
            inputs=[files_input, clear_before_upload],
            outputs=upload_output
        )

        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )
        upload_output = gr.Textbox(label="สถานะการประมวลผล", lines=3)
        upload_button.click(
            fn=process_multiple_files,
            inputs=[files_input, clear_before_upload],
            outputs=upload_output
        )
        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )

        # Google Sheets Import Section
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("### 📊 นำเข้าข้อมูลจาก Google Sheets")

        with gr.Group(elem_classes="upload-container"):
            gr.Markdown("""
            **🔗 วิธีการใช้งาน:**
            1. เปิด Google Sheets ที่ต้องการนำเข้า
            2. ตั้งค่าให้เป็น **"เผยแพร่ต่อทุกคนบนเว็บ"** (Share > General access > Anyone with the link)
            3. คัดลอก URL มาวางในช่องด้านล่าง
            """)

            with gr.Row():
                sheets_url_input = gr.Textbox(
                    label="🔗 Google Sheets URL",
                    placeholder="https://docs.google.com/spreadsheets/d/...",
                    info="วาง URL ของ Google Sheets ที่ต้องการนำเข้า",
                    scale=3
                )
                sheets_clear_checkbox = gr.Checkbox(
                    label="ล้างข้อมูลเก่าก่อนนำเข้า",
                    value=False,
                    info="ติ๊กถ้าต้องการล้างข้อมูลเก่าก่อนนำเข้าข้อมูลใหม่",
                    scale=1
                )

            with gr.Row():
                sheets_import_button = gr.Button(
                    "📊 นำเข้าข้อมูล Google Sheets",
                    variant="primary",
                    size="lg",
                    elem_classes="upload-button"
                )

            sheets_output = gr.Textbox(
                label="ผลลัพธ์การนำเข้าข้อมูล",
                lines=6,
                interactive=False,
                elem_classes="status-output"
            )

        # Connect Google Sheets event handlers
        sheets_import_button.click(
            fn=process_google_sheets_url,
            inputs=[sheets_url_input, sheets_clear_checkbox],
            outputs=sheets_output
        )

        # ส่วนจัดการ Database
        with gr.Row():
            gr.Markdown("### 🗄️ จัดการ Database")

        # Enhanced Backup & Restore Section
        with gr.Accordion("💾 Enhanced Backup & Restore", open=True):
            with gr.Row():
                # Backup Controls
                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 สำรองข้อมูล")

                    backup_name_input = gr.Textbox(
                        label="ชื่อ Backup (ถ้าว่างจะสร้างอัตโนมัติ)",
                        placeholder="เช่น: my_backup_20241030",
                        interactive=True
                    )

                    include_memory_checkbox = gr.Checkbox(
                        label="รวม Enhanced RAG Memory",
                        value=(RAG_MODE == "enhanced"),
                        info="สำรองข้อมูลความทรงจำการสนทนาด้วย"
                    )

                    with gr.Row():
                        enhanced_backup_button = gr.Button("สำรองข้อมูลขั้นสูง", variant="primary")
                        quick_backup_button = gr.Button("สำรองด่วน", variant="secondary")

                # Restore Controls
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔄 กู้คืนข้อมูล")

                    backup_selector = gr.Dropdown(
                        label="เลือก Backup ที่จะกู้คืน",
                        choices=[],
                        interactive=True,
                        info="รีเฟรชรายการเพื่อดู backup ทั้งหมด"
                    )

                    with gr.Row():
                        restore_button = gr.Button("กู้คืนข้อมูล", variant="primary")
                        refresh_backups_button = gr.Button("🔄 รีเฟรช", size="sm")

            # Backup Status and Results
            backup_status_output = gr.Textbox(
                label="สถานะการสำรอง/กู้คืน",
                lines=3,
                interactive=False
            )

            # Backup List Section
            with gr.Accordion("📋 รายการ Backup ทั้งหมด", open=False):
                backup_list_output = gr.Textbox(
                    label="รายการ Backup",
                    lines=8,
                    interactive=False
                )

                with gr.Row():
                    delete_backup_button = gr.Button("ลบ Backup ที่เลือก", variant="stop", size="sm")
                    validate_backup_button = gr.Button("✅ ตรวจสอบความสมบูรณ์", variant="secondary", size="sm")
                    clean_invalid_button = gr.Button("🧹 ลบที่ไม่ถูกต้อง", variant="secondary", size="sm")
                    refresh_list_button = gr.Button("รีเฟรชรายการ", size="sm")

        # Quick Database Operations
        with gr.Accordion("⚡ ดำเนินการด่วน", open=False):
            with gr.Row():
                db_info_button = gr.Button("แสดงข้อมูล Database", variant="secondary")
                inspect_button = gr.Button("ตรวจสอบข้อมูลละเอียด", variant="secondary")
                auto_backup_button = gr.Button("สร้าง Auto Backup", variant="secondary")

            db_info_output = gr.Textbox(label="ข้อมูล Database", lines=5, interactive=False)
            inspect_output = gr.Textbox(label="ข้อมูลภายใน Database", lines=10, interactive=False)

        # Event Handlers for Enhanced Backup & Restore
        def enhanced_backup_handler(backup_name, include_memory):
            if not backup_name.strip():
                backup_name = None
            result = backup_database_enhanced(backup_name, include_memory)
            if result["success"]:
                return f"""✅ สำรองข้อมูลสำเร็จ!
• ชื่อ Backup: {result['backup_name']}
• รวม Memory: {'✅' if include_memory else '❌'}
• ขนาด: {result['metadata']}

สามารถกู้คืนได้จากรายการ backup"""
            else:
                return f"❌ สำรองข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def quick_backup_handler():
            result = backup_database_enhanced()
            if result["success"]:
                return f"✅ สำรองข้อมูลด่วนสำเร็จ: {result['backup_name']}"
            else:
                return f"❌ สำรองข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def restore_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะกู้คืน"

            result = restore_database_enhanced(backup_name)
            if result["success"]:
                emergency_name = result.get('emergency_backup', {}).get('backup_name', 'N/A')
                return f"""✅ กู้คืนข้อมูลสำเร็จ!
• จาก Backup: {backup_name}
• เวลากู้คืน: {result['restored_at']}
• สร้าง Emergency Backup: {emergency_name}

โปรดตรวจสอบข้อมูลหลังกู้คืน"""
            else:
                return f"❌ กู้คืนข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def refresh_backups_handler():
            backups = list_available_backups()
            if backups:
                choices = [backup["name"] for backup in backups]
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            else:
                return gr.Dropdown(choices=[], value=None)

        def get_backup_list_handler():
            backups = list_available_backups()
            if backups:
                backup_info = []
                for backup in backups:
                    info = f"""📁 {backup['name']}
• สร้างเมื่อ: {backup['created_at']}
• ขนาด: {backup['size_mb']} MB
• ประเภท: {backup['type']}
• Memory: {'✅' if backup['includes_memory'] else '❌'}
• RAG Mode: {backup['rag_mode']}
• Records: {backup['database_info'].get('total_records', 'N/A') if backup['database_info'] else 'N/A'}
---"""
                    backup_info.append(info)
                return "\n".join(backup_info)
            else:
                return "ไม่พบข้อมูล backup ในระบบ"

        def delete_backup_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะลบ"

            result = delete_backup(backup_name)
            if result["success"]:
                return f"✅ {result['message']}"
            else:
                return f"❌ ไม่สามารถลบ backup: {result.get('error', 'Unknown error')}"

        def auto_backup_handler():
            result = auto_backup_before_operation()
            if result["success"]:
                return f"✅ สร้าง Auto Backup สำเร็จ: {result['backup_name']}"
            else:
                return f"❌ สร้าง Auto Backup ไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def clean_invalid_handler():
            try:
                cleanup_invalid_backups()
                return "✅ ลบโฟลเดอร์ที่ไม่ถูกต้องเรียบร้อยแล้ว"
            except Exception as e:
                return f"❌ ลบโฟลเดอร์ไม่ถูกต้องไม่สำเร็จ: {str(e)}"

        def validate_backup_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะตรวจสอบ"

            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            if not os.path.exists(backup_path):
                return f"❌ ไม่พบ backup: {backup_name}"

            is_valid, message = validate_backup_integrity(backup_path)
            if is_valid:
                return f"""✅ Backup ถูกต้อง!
• ชื่อ Backup: {backup_name}
• สถานะ: {message}
• พร้อมสำหรับการกู้คืน"""
            else:
                return f"""❌ Backup ไม่ถูกต้อง!
• ชื่อ Backup: {backup_name}
• ปัญหา: {message}
• แนะนำ: สร้าง backup ใหม่แทน"""

        # Connect event handlers
        enhanced_backup_button.click(
            fn=enhanced_backup_handler,
            inputs=[backup_name_input, include_memory_checkbox],
            outputs=backup_status_output,
            queue=False
        )

        quick_backup_button.click(
            fn=quick_backup_handler,
            inputs=None,
            outputs=backup_status_output,
            queue=False
        )

        restore_button.click(
            fn=restore_handler,
            inputs=backup_selector,
            outputs=backup_status_output,
            queue=False
        )

        refresh_backups_button.click(
            fn=refresh_backups_handler,
            inputs=None,
            outputs=backup_selector,
            queue=False
        )

        refresh_list_button.click(
            fn=get_backup_list_handler,
            inputs=None,
            outputs=backup_list_output,
            queue=False
        )

        delete_backup_button.click(
            fn=delete_backup_handler,
            inputs=backup_selector,
            outputs=backup_list_output,
            queue=False
        )

        auto_backup_button.click(
            fn=auto_backup_handler,
            inputs=None,
            outputs=db_info_output,
            queue=False
        )

        clean_invalid_button.click(
            fn=clean_invalid_handler,
            inputs=None,
            outputs=backup_status_output,
            queue=False
        )

        validate_backup_button.click(
            fn=validate_backup_handler,
            inputs=backup_selector,
            outputs=backup_status_output,
            queue=False
        )

        # Initialize backup list on page load
        demo.load(
            fn=refresh_backups_handler,
            inputs=None,
            outputs=backup_selector,
            queue=False
        )

        # Quick database operations handlers
        db_info_button.click(
            fn=lambda: str(get_database_info()),
            inputs=None,
            outputs=db_info_output,
            queue=False
        )

        inspect_button.click(
            fn=inspect_database,
            inputs=None,
            outputs=inspect_output,
            queue=False
        )

        # ส่วนจัดการ Discord Bot
        with gr.Row():
            gr.Markdown("### 🤖 จัดการ Discord Bot")

        with gr.Row():
            start_bot_button = gr.Button("เริ่ม Discord Bot", variant="primary")
            stop_bot_button = gr.Button("หยุด Discord Bot", variant="stop")

        bot_status_output = gr.Textbox(label="สถานะ Discord Bot", lines=3)

        with gr.Row():
            bot_model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=DISCORD_DEFAULT_MODEL,
                label="โมเดลสำหรับ Discord Bot"
            )

            bot_reply_mode = gr.Dropdown(
                choices=[
                    ("ตอบใน Channel", "channel"),
                    ("ตอบใน DM", "dm"),
                    ("ตอบทั้ง Channel และ DM", "both")
                ],
                value=DISCORD_REPLY_MODE,
                label="วิธีการตอบกลับ"
            )

        def update_discord_model(model):
            global DISCORD_DEFAULT_MODEL
            DISCORD_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล Discord Bot เป็น {model}"

        def update_discord_reply_mode(mode):
            global DISCORD_REPLY_MODE
            DISCORD_REPLY_MODE = mode
            mode_name = {"channel": "ใน Channel", "dm": "ใน DM", "both": "ทั้ง Channel และ DM"}
            return f"อัปเดตวิธีการตอบกลับเป็น: {mode_name.get(mode, mode)}"

        def start_bot_ui():
            if start_discord_bot_thread():
                return "Discord Bot เริ่มทำงานแล้ว! สามารถใช้คำสั่งได้ใน Discord"
            else:
                return "ไม่สามารถเริ่ม Discord Bot ได้ ตรวจสอบการตั้งค่าใน .env"

        def stop_bot_ui():
            if stop_discord_bot():
                return "Discord Bot หยุดทำงานแล้ว"
            else:
                return "ไม่สามารถหยุด Discord Bot ได้"

        start_bot_button.click(
            fn=start_bot_ui,
            inputs=None,
            outputs=bot_status_output,
            queue=False
        )

        stop_bot_button.click(
            fn=stop_bot_ui,
            inputs=None,
            outputs=bot_status_output,
            queue=False
        )

        bot_model_selector.change(
            fn=update_discord_model,
            inputs=bot_model_selector,
            outputs=bot_status_output,
            queue=False
        )

        bot_reply_mode.change(
            fn=update_discord_reply_mode,
            inputs=bot_reply_mode,
            outputs=bot_status_output,
            queue=False
        )

        # ส่วนจัดการ LINE OA Bot
        with gr.Row():
            gr.Markdown("### 📱 จัดการ LINE OA Bot")

        with gr.Row():
            start_line_button = gr.Button("เริ่ม LINE OA Bot", variant="primary")
            stop_line_button = gr.Button("หยุด LINE OA Bot", variant="stop")

        line_status_output = gr.Textbox(label="สถานะ LINE OA Bot", lines=3)
        line_model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=LINE_DEFAULT_MODEL,
            label="โมเดลสำหรับ LINE OA Bot"
        )

        def update_line_model(model):
            global LINE_DEFAULT_MODEL
            LINE_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล LINE OA Bot เป็น {model}"

        def start_line_ui():
            """Initialize LINE Bot - webhooks are already running via FastAPI"""
            if not LINE_ENABLED:
                return "❌ LINE Bot ถูกปิดใช้งาน กรุณาตั้งค่า LINE_ENABLED=true ใน environment variables"

            if LINE_CHANNEL_ACCESS_TOKEN == "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
                return "❌ กรุณาตั้งค่า LINE_CHANNEL_ACCESS_TOKEN และ LINE_CHANNEL_SECRET"

            # Setup LINE bot if not already initialized
            if not line_handler:
                if setup_line_bot():
                    # Get Railway URL or local URL
                    webhook_url = os.getenv('RAILWAY_PUBLIC_DOMAIN', f"http://localhost:7860")
                    return f"✅ LINE OA Bot พร้อมใช้งาน!\n\nWebhook URL: {webhook_url}/callback\n\nตั้งค่า webhook URL นี้ใน LINE Developers Console"
                else:
                    return "❌ ไม่สามารถตั้งค่า LINE Bot ได้ กรุณาตรวจสอบ credentials"
            else:
                webhook_url = os.getenv('RAILWAY_PUBLIC_DOMAIN', f"http://localhost:7860")
                return f"✅ LINE OA Bot ทำงานอยู่แล้ว\n\nWebhook URL: {webhook_url}/callback"

        def stop_line_ui():
            """Note: Webhooks cannot be stopped individually - they run with the main app"""
            return "ℹ️ Webhook endpoints ทำงานร่วมกับ Gradio app\nไม่สามารถหยุดแยกได้ - ต้อง restart ทั้งโปรแกรม"

        start_line_button.click(
            fn=start_line_ui,
            inputs=None,
            outputs=line_status_output,
            queue=False
        )

        stop_line_button.click(
            fn=stop_line_ui,
            inputs=None,
            outputs=line_status_output,
            queue=False
        )

        line_model_selector.change(
            fn=update_line_model,
            inputs=line_model_selector,
            outputs=line_status_output,
            queue=False
        )

        # ส่วนจัดการ Facebook Messenger Bot
        with gr.Row():
            gr.Markdown("### 💬 จัดการ Facebook Messenger Bot")

        with gr.Row():
            start_fb_button = gr.Button("เริ่ม Facebook Bot", variant="primary")
            stop_fb_button = gr.Button("หยุด Facebook Bot", variant="stop")

        fb_status_output = gr.Textbox(label="สถานะ Facebook Bot", lines=3)
        fb_model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=FB_DEFAULT_MODEL,
            label="โมเดลสำหรับ Facebook Bot"
        )

        def update_fb_model(model):
            global FB_DEFAULT_MODEL
            FB_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล Facebook Bot เป็น {model}"

        def start_fb_ui():
            """Initialize Facebook Bot - webhooks are already running via FastAPI"""
            if not FB_ENABLED:
                return "❌ Facebook Bot ถูกปิดใช้งาน กรุณาตั้งค่า FB_ENABLED=true ใน environment variables"

            if FB_PAGE_ACCESS_TOKEN == "YOUR_FB_PAGE_ACCESS_TOKEN":
                return "❌ กรุณาตั้งค่า FB_PAGE_ACCESS_TOKEN และ FB_VERIFY_TOKEN"

            # Setup Facebook bot
            if setup_facebook_bot():
                # Get Railway URL or local URL
                webhook_url = os.getenv('RAILWAY_PUBLIC_DOMAIN', f"http://localhost:7860")
                return f"""✅ Facebook Messenger Bot พร้อมใช้งาน!

Webhook URL: {webhook_url}/webhook
Verify Token: {FB_VERIFY_TOKEN}

ตั้งค่า webhook URL และ verify token นี้ใน Facebook App Dashboard"""
            else:
                return "❌ ไม่สามารถตั้งค่า Facebook Bot ได้ กรุณาตรวจสอบ credentials"

        def stop_fb_ui():
            """Note: Webhooks cannot be stopped individually - they run with the main app"""
            return "ℹ️ Webhook endpoints ทำงานร่วมกับ Gradio app\nไม่สามารถหยุดแยกได้ - ต้อง restart ทั้งโปรแกรม"

        start_fb_button.click(
            fn=start_fb_ui,
            inputs=None,
            outputs=fb_status_output,
            queue=False
        )

        stop_fb_button.click(
            fn=stop_fb_ui,
            inputs=None,
            outputs=fb_status_output,
            queue=False
        )

        fb_model_selector.change(
            fn=update_fb_model,
            inputs=fb_model_selector,
            outputs=fb_status_output,
            queue=False
        )

    with gr.Tab("📊 Feedback และสถิติ"):
        gr.Markdown("## 📊 ระบบ Feedback สำหรับปรับปรุงคำตอบ")

        # สถิติหลัก
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📈 สถิติการตอบกลับ")
                stats_display = gr.HTML()

                refresh_stats_btn = gr.Button("🔄 รีเฟรชสถิติ", variant="secondary")

                # ปุ่มส่งออกข้อมูล
                export_btn = gr.Button("📥 ส่งออกข้อมูล Feedback", variant="primary")
                download_file = gr.File(visible=False)

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Feedback ล่าสุด")
                feedback_display = gr.Dataframe(
                    headers=["คำถาม", "คำตอบ", "ประเภท", "เวลา", "ความคิดเห็น"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    value=[]  # เริ่มต้นด้วยค่าว่าง
                )

                # ส่วนจัดการ feedback
                with gr.Row():
                    feedback_id_input = gr.Number(
                        label="Feedback ID ที่ต้องการลบ",
                        minimum=1,
                        step=1,
                        info="ใส่ ID จากตารางด้านบน"
                    )
                    delete_feedback_btn = gr.Button("🗑️ ลบ Feedback", variant="stop")

                delete_status = gr.Textbox(label="สถานะการลบ", interactive=False)

        # สถิติการเรียนรู้
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎓 สถิติการเรียนรู้ (Learning Analytics)")
                learning_stats_display = gr.HTML()

                refresh_learning_btn = gr.Button("🔄 รีเฟรชสถิติการเรียนรู้", variant="secondary")

        # Enhanced Analytics Dashboard
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Analytics Dashboard (Advanced Insights)")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Quality Score Overview
                        quality_score_display = gr.HTML()

                        # Pattern Analysis Results
                        pattern_display = gr.HTML()

                    with gr.Column(scale=1):
                        # Weekly Trend Chart
                        weekly_trend_display = gr.HTML()

                        # Improvement Recommendations
                        recommendations_display = gr.HTML()

                with gr.Row():
                    refresh_analytics_btn = gr.Button("📈 รีเฟรช Analytics ขั้นสูง", variant="primary")
                    export_analytics_btn = gr.Button("📥 ส่งออกรายงาน", variant="secondary")

                analytics_export_file = gr.File(
                    label="📊 Analytics Report",
                    visible=False,
                    file_types=[".json"]
                )

                # แสดงคำตอบที่ถูกนำไปใช้บ่อยที่สุด
                most_used_display = gr.Dataframe(
                    headers=["คำถามที่ถูกแก้ไข", "จำนวนครั้งที่นำไปใช้"],
                    datatype=["str", "int"],
                    interactive=False,
                    wrap=True
                )

        # Enhanced Analytics Functions
        def update_analytics_dashboard():
            """อัปเดต analytics dashboard ขั้นสูง"""
            try:
                analytics = get_comprehensive_analytics()
                if not analytics:
                    return "<div>ไม่มีข้อมูล analytics</div>", "<div>ไม่มีข้อมูล</div>", "<div>ไม่มีข้อมูล</div>", "<div>ไม่มีข้อมูล</div>"

                # Quality Score Display
                pattern_analysis = analytics.get('pattern_analysis', {})
                quality_score = pattern_analysis.get('quality_score', 0)
                quality_color = '#4caf50' if quality_score >= 80 else '#ff9800' if quality_score >= 60 else '#f44336'

                quality_html = f"""
                <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;">
                    <h4 style="margin: 0 0 15px 0; color: #333;">🎯 คะแนนคุณภาพระบบ</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 3em; font-weight: bold; color: {quality_color}; margin: 10px 0;">
                            {quality_score:.1f}%
                        </div>
                        <div style="color: #666; font-size: 0.9em;">
                            จาก {pattern_analysis.get('total_analyzed', 0)} การตอบกลับล่าสุด
                        </div>
                    </div>
                </div>
                """

                # Pattern Analysis Display
                patterns = pattern_analysis.get('patterns', {})
                pattern_html = "<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'><h4 style='margin: 0 0 15px 0; color: #333;'>🔍 รูปแบบปัญหา</h4>"
                if patterns:
                    for category, count in patterns.items():
                        pattern_html += f"<div style='margin: 8px 0; padding: 8px; background: #f5f5f5; border-radius: 5px;'>{category}: {count} ครั้ง</div>"
                else:
                    pattern_html += "<div style='color: #666;'>ไม่พบรูปแบบที่น่าสนใจ</div>"
                pattern_html += "</div>"

                # Weekly Trend Display
                weekly_trend = analytics.get('weekly_trend', [])
                trend_html = "<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'><h4 style='margin: 0 0 15px 0; color: #333;'>📈 แนวโน้ม 7 วัน</h4>"
                if weekly_trend:
                    for date, count, good_count in weekly_trend:
                        accuracy = (good_count / count * 100) if count > 0 else 0
                        trend_html += f"<div style='margin: 8px 0; padding: 8px; background: #f5f5f5; border-radius: 5px; display: flex; justify-content: space-between;'><span>{date}</span><span>{count} ครั้ง ({accuracy:.0f}% ถูกต้อง)</span></div>"
                else:
                    trend_html += "<div style='color: #666;'>ไม่มีข้อมูลในช่วง 7 วัน</div>"
                trend_html += "</div>"

                # Recommendations Display
                recommendations = pattern_analysis.get('recommendations', [])
                rec_html = "<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'><h4 style='margin: 0 0 15px 0; color: #333;'>💡 คำแนะนำการปรับปรุง</h4>"
                if recommendations:
                    for rec in recommendations:
                        rec_html += f"<div style='margin: 8px 0; padding: 10px; background: #e3f2fd; border-radius: 5px; border-left: 4px solid #2196f3;'>{rec}</div>"
                else:
                    rec_html += "<div style='color: #666;'>ระบบทำงานได้ดีอยู่แล้ว</div>"
                rec_html += "</div>"

                return quality_html, pattern_html, trend_html, rec_html

            except Exception as e:
                error_html = f"<div style='color: red;'>เกิดข้อผิดพลาด: {str(e)}</div>"
                return error_html, error_html, error_html, error_html

        def export_analytics_report():
            """ส่งออกรายงาน analytics"""
            try:
                analytics = get_comprehensive_analytics()
                report = {
                    "report_type": "comprehensive_analytics",
                    "generated_at": datetime.now().isoformat(),
                    "data": analytics
                }

                # สร้าง JSON file สำหรับ download
                import json
                report_json = json.dumps(report, ensure_ascii=False, indent=2)

                return gr.File(value=report_json, visible=True, label="📊 Analytics Report.json")
            except Exception as e:
                return gr.HTML(f"<div style='color: red;'>❌ ส่งออกรายงานไม่สำเร็จ: {str(e)}</div>")

        # ฟังก์ชันสำหรับอัปเดตสถิติ
        def update_stats_display():
            try:
                stats = get_feedback_stats()

                stats_html = f"""
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                    <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #2e7d32;">{stats['total']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">ทั้งหมด</p>
                    </div>
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #1976d2;">{stats['good']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">ถูกต้อง 👍</p>
                    </div>
                    <div style="background: #ffebee; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #d32f2f;">{stats['bad']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">ผิดพลาด 👎</p>
                    </div>
                    <div style="background: #fff3e0; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #f57c00;">{stats['accuracy']:.1f}%</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">ความแม่นยำ</p>
                    </div>
                </div>
                """

                return stats_html, stats['recent']
            except Exception as e:
                error_html = f"""
                <div style="background: #ffebee; padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #d32f2f;">❌ เกิดข้อผิดพลาด</h3>
                    <p style="margin: 5px 0 0 0; color: #555;">ไม่สามารถโหลดข้อมูลได้: {str(e)}</p>
                </div>
                """
                return error_html, []

        # ฟังก์ชันสำหรับลบ feedback
        def delete_feedback_handler(feedback_id):
            if feedback_id is None or feedback_id <= 0:
                return "❌ กรุณาระบุ Feedback ID ที่ถูกต้อง"

            if delete_feedback(int(feedback_id)):
                # อัปเดตสถิติและตารางใหม่
                stats_html, recent_data = update_stats_display()
                return "✅ ลบ Feedback เรียบร้อยแล้ว", stats_html, recent_data
            else:
                return "❌ ไม่สามารถลบ Feedback ได้ (ID ไม่พบหรือเกิดข้อผิดพลาด)", None, None

        # ฟังก์ชันสำหรับส่งออกข้อมูล
        def export_feedback_handler():
            csv_data = export_feedback()
            if csv_data:
                import io
                from datetime import datetime

                filename = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                # สร้างไฟล์ชั่วคราว
                filepath = f"./data/{filename}"
                with open(filepath, 'w', encoding='utf-8-sig') as f:  # utf-8-sig สำหรับ Excel
                    f.write(csv_data)

                return filepath
            return None

        # ฟังก์ชันสำหรับอัปเดตสถิติการเรียนรู้
        def update_learning_display():
            try:
                learning_stats = get_learning_stats()

                learning_html = f"""
                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                    <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #2e7d32;">{learning_stats['total_corrected']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">คำตอบที่ถูกแก้ไข</p>
                    </div>
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #1976d2;">{learning_stats['used_corrected']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">ถูกนำไปใช้</p>
                    </div>
                    <div style="background: #fff3e0; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #f57c00;">{learning_stats['learning_rate']:.1f}%</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">อัตราการเรียนรู้</p>
                    </div>
                    <div style="background: #f3e5f5; padding: 15px; border-radius: 8px; text-align: center; flex: 1;">
                        <h3 style="margin: 0; color: #7b1fa2;">{learning_stats['corrected_feedback']}</h3>
                        <p style="margin: 5px 0 0 0; color: #555;">Feedback ที่มีการแก้ไข</p>
                    </div>
                </div>
                """

                # จัดรูปแบบข้อมูลสำหรับแสดงในตาราง
                most_used_data = []
                for item in learning_stats['most_used']:
                    # ตัดคำถามที่ยาวเกินไป
                    question = item[0]
                    if len(question) > 100:
                        question = question[:97] + "..."
                    most_used_data.append([question, item[1]])

                return learning_html, most_used_data
            except Exception as e:
                error_html = f"""
                <div style="background: #ffebee; padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #d32f2f;">❌ เกิดข้อผิดพลาด</h3>
                    <p style="margin: 5px 0 0 0; color: #555;">ไม่สามารถโหลดข้อมูลการเรียนรู้ได้: {str(e)}</p>
                </div>
                """
                return error_html, []

        # เชื่อมต่อ events
        refresh_stats_btn.click(
            fn=update_stats_display,
            inputs=[],
            outputs=[stats_display, feedback_display]
        )

        delete_feedback_btn.click(
            fn=delete_feedback_handler,
            inputs=[feedback_id_input],
            outputs=[delete_status, stats_display, feedback_display]
        )

        export_btn.click(
            fn=export_feedback_handler,
            inputs=[],
            outputs=[download_file]
        )

        refresh_learning_btn.click(
            fn=update_learning_display,
            inputs=[],
            outputs=[learning_stats_display, most_used_display]
        )

        # Analytics Dashboard Event Handlers
        refresh_analytics_btn.click(
            fn=update_analytics_dashboard,
            inputs=[],
            outputs=[quality_score_display, pattern_display, weekly_trend_display, recommendations_display]
        )

        export_analytics_btn.click(
            fn=export_analytics_report,
            inputs=[],
            outputs=[analytics_export_file]
        )

        # อัปเดตสถิติครั้งแรก (delayed load with error handling)
        demo.load(
            fn=lambda: [update_stats_display(), update_learning_display()],
            inputs=[],
            outputs=[stats_display, feedback_display, learning_stats_display, most_used_display],
            show_progress=True
        )

    # ==================== TAG MANAGEMENT TAB ====================
    with gr.Tab("🏷️ จัดการ Tag"):
        gr.Markdown("## 🏷️ ระบบจัดการ Tag")
        gr.Markdown("จัดการ tags เพื่อจัดกลุ่มเอกสารและ feedback ให้เข้าถึงได้รวดเร็วและตรงประเด็น")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 สร้าง Tag ใหม่")
                with gr.Row():
                    tag_name_input = gr.Textbox(
                        label="ชื่อ Tag",
                        placeholder="ตัวอย่าง: ปัญหาที่พบบ่อย, เอกสารสำคัญ, คำถามทั่วไป",
                        scale=3
                    )
                    tag_color_input = gr.ColorPicker(
                        label="สี Tag",
                        value="#007bff",
                        scale=1
                    )
                tag_desc_input = gr.Textbox(
                    label="รายละเอียด Tag",
                    placeholder="รายละเอียดเพิ่มเติมเกี่ยวกับ tag นี้"
                )
                create_tag_btn = gr.Button("🏷️ สร้าง Tag", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("### 📋 Tags ทั้งหมด")
                tags_list = gr.Dataframe(
                    headers=["ID", "ชื่อ Tag", "สี", "รายละเอียด", "วันที่สร้าง"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True
                )
                refresh_tags_btn = gr.Button("🔄 รีเฟรชรายการ Tag")
                delete_tag_btn = gr.Button("🗑️ ลบ Tag ที่เลือก", variant="stop")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🏆 Tags ที่ใช้บ่อย")
                popular_tags = gr.Dataframe(
                    headers=["ชื่อ Tag", "จำนวนการใช้"],
                    datatype=["str", "number"],
                    interactive=False,
                    wrap=True
                )

            with gr.Column():
                gr.Markdown("### 💬 Tags ใน Feedback")
                feedback_tags = gr.Dataframe(
                    headers=["ชื่อ Tag", "จำนวน Feedback"],
                    datatype=["str", "number"],
                    interactive=False,
                    wrap=True
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 ค้นหาเอกสารตาม Tag")
                with gr.Row():
                    selected_tags_search = gr.CheckboxGroup(
                        label="เลือก Tags (เลือกหลายอันได้)",
                        choices=[]
                    )
                    search_by_tags_btn = gr.Button("🔍 ค้นหา", variant="primary")

                search_results = gr.Dataframe(
                    headers=["Document ID", "Content Preview"],
                    datatype=["str", "str"],
                    interactive=False,
                    wrap=True
                )

            with gr.Column():
                gr.Markdown("### 💬 Feedback ตาม Tag")
                tag_feedback_selector = gr.Dropdown(
                    label="เลือก Tag",
                    choices=[]
                )
                load_feedback_by_tag_btn = gr.Button("📋 โหลด Feedback", variant="primary")

                tag_feedback_display = gr.Dataframe(
                    headers=["ID", "คำถาม", "คำตอบ", "ประเภท", "วันที่", "ความคิดเห็น"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True
                )

        # Status display
        tag_status = gr.HTML("")
        tag_status_display = gr.HTML("")  # สำหรับแสดงสถานะต่างๆ

    # ==================== END TAG MANAGEMENT TAB ====================

    with gr.Tab("🔑 API Key Configuration"):
        gr.Markdown("## ตั้งค่า API Key สำหรับ AI Providers")
        gr.Markdown("ใส่ API Key สำหรับแต่ละผู้ให้บริการ AI ที่ต้องการใช้งาน")

        with gr.Row():
            with gr.Column():
                # Minimax API Key
                minimax_api_key = gr.Textbox(
                    value=os.getenv("MINIMAX_API_KEY", ""),
                    label="Minimax API Key",
                    type="password",
                    placeholder="ใส่ API Key สำหรับ Minimax",
                    info="สำหรับใช้งานโมเดล Minimax (abab6.5, abab6.5s, abab5.5)"
                )
                minimax_status = gr.HTML("")

            with gr.Column():
                # Manus API Key
                manus_api_key = gr.Textbox(
                    value=os.getenv("MANUS_API_KEY", ""),
                    label="Manus API Key",
                    type="password",
                    placeholder="ใส่ API Key สำหรับ Manus",
                    info="สำหรับใช้งานโมเดล Manus (manus-code, manus-reasoning, manus-vision)"
                )
                manus_status = gr.HTML("")

        with gr.Row():
            with gr.Column():
                # Gemini API Key
                gemini_api_key = gr.Textbox(
                    value=os.getenv("GEMINI_API_KEY", ""),
                    label="Google Gemini API Key",
                    type="password",
                    placeholder="ใส่ API Key สำหรับ Google Gemini",
                    info="สำหรับใช้งานโมเดล Gemini (gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash)"
                )
                gemini_status = gr.HTML("")

            with gr.Column():
                # OpenAI API Key
                openai_api_key = gr.Textbox(
                    value=os.getenv("OPENAI_API_KEY", ""),
                    label="OpenAI API Key (ChatGPT)",
                    type="password",
                    placeholder="ใส่ API Key สำหรับ OpenAI",
                    info="สำหรับใช้งานโมเดล ChatGPT (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)"
                )
                openai_status = gr.HTML("")

        with gr.Row():
            with gr.Column():
                # Zhipu AI API Key
                zhipu_api_key = gr.Textbox(
                    value=os.getenv("ZHIPU_API_KEY", ""),
                    label="Zhipu AI API Key (GLM)",
                    type="password",
                    placeholder="ใส่ API Key สำหรับ Zhipu AI (z.ai)",
                    info="สำหรับใช้งานโมเดล GLM-4.6 (GLM-4.6, glm-4, glm-4v, glm-3-turbo) - กด 'ตรวจสอบโมเดล Zhipu' ก่อนใช้"
                )
                zhipu_status = gr.HTML("")

        # Test connection buttons
        with gr.Row():
            test_minimax_btn = gr.Button("ทดสอบ Minimax", variant="secondary")
            test_manus_btn = gr.Button("ทดสอบ Manus", variant="secondary")
            test_gemini_btn = gr.Button("ทดสอบ Gemini", variant="secondary")
            test_openai_btn = gr.Button("ทดสอบ OpenAI", variant="secondary")
            test_zhipu_btn = gr.Button("ทดสอบ Zhipu AI", variant="secondary")

        # Save configuration button
        save_config_btn = gr.Button("บันทึกการตั้งค่า", variant="primary")
        config_status = gr.HTML("")

        def test_api_connection(provider_name, api_key):
            """Test API connection for a provider"""
            if not api_key or not api_key.strip():
                return f"❌ กรุณาใส่ API Key สำหรับ {provider_name}"

            try:
                if provider_name == "minimax":
                    client = openai.OpenAI(api_key=api_key, base_url="https://api.minimax.chat/v1")
                    response = client.chat.completions.create(
                        model="abab6.5s",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return f"✅ {provider_name} API Key ถูกต้องและพร้อมใช้งาน"

                elif provider_name == "manus":
                    client = openai.OpenAI(api_key=api_key, base_url="https://api.manus.ai/v1")
                    response = client.chat.completions.create(
                        model="manus-code",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return f"✅ {provider_name} API Key ถูกต้องและพร้อมใช้งาน"

                elif provider_name == "gemini":
                    if not GEMINI_AVAILABLE:
                        return f"❌ {provider_name} library ไม่พร้อมใช้งาน กรุณาติดตั้ง: pip install google-generativeai"

                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content("Hello")
                        return f"✅ {provider_name} API Key ถูกต้องและพร้อมใช้งาน"
                    except Exception as api_error:
                        return f"❌ {provider_name} API Key ไม่ถูกต้องหรือ library มีปัญหา: {str(api_error)}"

                elif provider_name == "zhipu":
                    client = openai.OpenAI(api_key=api_key, base_url="https://api.z.ai/api/paas/v4")
                    response = client.chat.completions.create(
                        model="GLM-4.6",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return f"✅ {provider_name} API Key ถูกต้องและพร้อมใช้งาน"

                elif provider_name == "openai":
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return f"✅ {provider_name} API Key ถูกต้องและพร้อมใช้งาน"

            except Exception as e:
                return f"❌ {provider_name} API Key ไม่ถูกต้อง: {str(e)}"

        def save_api_config(minimax_key, manus_key, gemini_key, openai_key, zhipu_key):
            """Save API keys to environment variables"""
            env_file = ".env"
            lines = []

            # Read existing .env file
            if os.path.exists(env_file):
                with open(env_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

            # Update or add API keys
            updates = {
                "MINIMAX_API_KEY": minimax_key,
                "MANUS_API_KEY": manus_key,
                "GEMINI_API_KEY": gemini_key,
                "OPENAI_API_KEY": openai_key,
                "ZHIPU_API_KEY": zhipu_key
            }

            # Remove existing API key lines
            lines = [line for line in lines if not any(key in line for key in updates.keys())]

            # Add new API key lines
            for key, value in updates.items():
                if value and value.strip():
                    lines.append(f"{key}={value}\n")

            # Write back to .env file
            with open(env_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            # Update environment variables
            os.environ["MINIMAX_API_KEY"] = minimax_key
            os.environ["MANUS_API_KEY"] = manus_key
            os.environ["GEMINI_API_KEY"] = gemini_key
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["ZHIPU_API_KEY"] = zhipu_key

            return "✅ บันทึกการตั้งค่า API Key เรียบร้อยแล้ว กรุณารีสตาร์ทแอปพลิเคชันเพื่อให้การตั้งค่ามีผล"

        # Connect test buttons
        test_minimax_btn.click(
            fn=lambda k: test_api_connection("minimax", k),
            inputs=[minimax_api_key],
            outputs=[minimax_status]
        )

        test_manus_btn.click(
            fn=lambda k: test_api_connection("manus", k),
            inputs=[manus_api_key],
            outputs=[manus_status]
        )

        test_gemini_btn.click(
            fn=lambda k: test_api_connection("gemini", k),
            inputs=[gemini_api_key],
            outputs=[gemini_status]
        )

        test_openai_btn.click(
            fn=lambda k: test_api_connection("openai", k),
            inputs=[openai_api_key],
            outputs=[openai_status]
        )

        test_zhipu_btn.click(
            fn=lambda k: test_api_connection("zhipu", k),
            inputs=[zhipu_api_key],
            outputs=[zhipu_status]
        )

        # Add button to check available models
        with gr.Row():
            check_zhipu_models_btn = gr.Button("ตรวจสอบโมเดล Zhipu", variant="secondary", size="sm")

        def check_zhipu_models(api_key):
            if not api_key or not api_key.strip():
                return "❌ กรุณาใส่ API Key ก่อน"

            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }

                # Try to get available models
                response = requests.get("https://api.z.ai/api/paas/v4/models", headers=headers, timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    models = [model.get('id', 'unknown') for model in models_data.get('data', [])]
                    return f"✅ โมเดลที่มี: {', '.join(models[:10])}"
                else:
                    # If models endpoint doesn't work, try common model names
                    common_models = ["GLM-4.6", "glm-4.6", "glm-4", "glm-4v", "glm-3-turbo"]
                    results = []

                    for model in common_models:
                        try:
                            test_response = requests.post(
                                "https://api.z.ai/api/paas/v4/chat/completions",
                                headers=headers,
                                json={
                                    "model": model,
                                    "messages": [{"role": "user", "content": "test"}],
                                    "max_tokens": 1
                                },
                                timeout=5
                            )
                            if test_response.status_code == 200:
                                results.append(model)
                        except:
                            continue

                    if results:
                        return f"✅ โมเดลที่ใช้ได้: {', '.join(results)}"
                    else:
                        return f"❌ ไม่พบโมเดลที่ใช้ได้ ตรวจสอบ API Key หรือ endpoint"

            except Exception as e:
                return f"❌ ไม่สามารถตรวจสอบโมเดลได้: {str(e)}"

        check_zhipu_models_btn.click(
            fn=check_zhipu_models,
            inputs=[zhipu_api_key],
            outputs=[zhipu_status]
        )

        # Connect save button
        save_config_btn.click(
            fn=save_api_config,
            inputs=[minimax_api_key, manus_api_key, gemini_api_key, openai_api_key, zhipu_api_key],
            outputs=[config_status]
        )

    with gr.Tab("แชท"):
        # Simple provider initialization - start fast
        basic_providers = ["ollama"]

        # Check for external providers quickly (no blocking)
        external_providers = []
        if os.getenv("GEMINI_API_KEY") and GEMINI_AVAILABLE:
            external_providers.append("gemini")
        if os.getenv("OPENAI_API_KEY"):
            external_providers.append("chatgpt")
        if os.getenv("MINIMAX_API_KEY"):
            external_providers.append("minimax")
        if os.getenv("MANUS_API_KEY"):
            external_providers.append("manus")
        if os.getenv("ZHIPU_API_KEY"):
            external_providers.append("zhipu")

        all_providers = basic_providers + external_providers
        provider_choices = [(AI_PROVIDERS[p]["name"], p) for p in all_providers if p in AI_PROVIDERS]

        logging.info(f"Quick available providers: {all_providers}")

        # Start with Ollama - fastest loading
        provider_selector = gr.Dropdown(
            choices=provider_choices,
            value="ollama",
            label="เลือก AI Provider",
            info="เลือกผู้ให้บริการ AI ที่ต้องการใช้"
        )
        selected_provider = gr.State(value="ollama")

        # Combined update function
        def update_provider_and_models(provider):
            models = get_provider_models(provider)
            default_model = AI_PROVIDERS[provider]["default_model"] if models else None
            if models and default_model not in models:
                default_model = models[0] if models else None
            print(f"Provider: {provider}, Models: {models}, Default: {default_model}")  # Debug
            return gr.update(choices=models, value=default_model), provider, default_model

        # Start with Ollama models - no blocking operations
        initial_models = get_provider_models("ollama")
        initial_default_model = AI_PROVIDERS["ollama"]["default_model"] if initial_models else None

        model_selector = gr.Dropdown(
            choices=initial_models,
            value=initial_default_model,
            label="เลือก LLM Model",
            allow_custom_value=True
        )
        selected_model = gr.State(value=initial_default_model)

        # Update both provider state and models when provider changes
        provider_selector.change(
            fn=update_provider_and_models,
            inputs=provider_selector,
            outputs=[model_selector, selected_provider, selected_model]
        )

        model_selector.change(fn=lambda x: x, inputs=model_selector, outputs=selected_model)

        # Choice เลือก RAG Mode
        rag_mode_selector = gr.Radio(
            choices=[
                ("📖 Standard RAG - ค้นหาจากเอกสารเท่านั้น", "standard"),
                ("🧠 Enhanced RAG - จดจำบริบทการสนทนา", "enhanced")
            ],
            value="standard",
            label="เลือกโหมด RAG",
            info="Enhanced RAG จะจดจำประวัติการสนทนาและใช้บริบทในการตอบคำถาม"
        )
        selected_rag_mode = gr.State(value="standard")  # เก็บไว้ใน state
        rag_mode_status = gr.Textbox(label="สถานะ RAG Mode", value="📖 Standard RAG Mode", interactive=False)

        # เพิ่มตัวเลือกสำหรับการแสดงแหล่งที่มาของข้อมูล
        with gr.Row():
            show_source_checkbox = gr.Checkbox(
                label="🔍 แสดงแหล่งที่มาของข้อมูล",
                value=False,
                info="เพิ่มการระบุแหล่งที่มาของข้อมูลในคำตอบ"
            )

            formal_style_checkbox = gr.Checkbox(
                label="📝 สไตล์การตอบเป็นทางการ",
                value=False,
                info="ใช้ภาษาที่เป็นทางการและสุภาพมากขึ้น"
            )

        # LightRAG Graph Reasoning Options
        with gr.Accordion("🧠 LightRAG Graph Reasoning (Advanced)", open=False):
            gr.Markdown("""
            **🔬 เพิ่มความฉลาดด้วย Graph Reasoning:**
            • วิเคราะห์ความสัมพันธ์ระหว่าง concepts
            • Multi-hop reasoning สำหรับคำถามซับซ้อน
            • แสดงความเชื่อมโยงที่ซ่อนอยู่ในข้อมูล
            """)

            with gr.Row():
                use_graph_reasoning = gr.Checkbox(
                    label="🧠 เปิดใช้ Graph Reasoning",
                    value=False,
                    info="ใช้ LightRAG วิเคราะห์ความสัมพันธ์และเชื่อมโยง concepts"
                )

            with gr.Row():
                reasoning_mode = gr.Dropdown(
                    choices=[
                        ("🔄 Hybrid (แนะนำ)", "hybrid"),
                        ("🎯 Local (เฉพาะเจาะจง)", "local"),
                        ("🌐 Global (ครอบคลุม)", "global"),
                        ("⚡ Naive (เร็ว)", "naive")
                    ],
                    value="hybrid",
                    label="📊 โหมดการวิเคราะห์",
                    info="เลือกรูปแบบการวิเคราะห์ Graph Reasoning"
                )

            with gr.Row():
                multi_hop_enabled = gr.Checkbox(
                    label="🔄 Multi-Hop Reasoning",
                    value=False,
                    info="วิเคราะห์ความสัมพันธ์ขั้นสูงหลาย step"
                )

                hop_count = gr.Slider(
                    minimum=2,
                    maximum=5,
                    value=2,
                    step=1,
                    label="🔢 จำนวน Hop",
                    visible=False,
                    info="จำนวนขั้นตอนการวิเคราะห์ Multi-Hop"
                )

            # Toggle hop count visibility
            def toggle_hop_count(enabled):
                return gr.update(visible=enabled)

            multi_hop_enabled.change(
                fn=toggle_hop_count,
                inputs=multi_hop_enabled,
                outputs=hop_count
            )

            # LightRAG Status Display
            with gr.Row():
                lightrag_status_display = gr.Textbox(
                    label="📊 สถานะ LightRAG",
                    value="กำลังตรวจสอบ...",
                    interactive=False,
                    lines=2
                )

            # Update status on interface load
            demo.load(
                fn=update_lightrag_status,
                inputs=[],
                outputs=[lightrag_status_display]
            )

            # Test Graph Reasoning Button
            with gr.Row():
                test_lightrag_btn = gr.Button("🧪 ทดสอบ Graph Reasoning", variant="secondary", size="sm")
                lightrag_test_output = gr.Textbox(
                    label="ผลการทดสอบ",
                    interactive=False,
                    visible=False,
                    lines=3
                )

        
            # Update status on load and test
            test_lightrag_btn.click(
                fn=test_graph_reasoning_interface,
                inputs=[],
                outputs=[lightrag_test_output, lightrag_test_output]
            )

        # เพิ่มตัวเลือกสำหรับการส่งคำตอบไปยังแพลตฟอร์มอื่น
        with gr.Row():
            gr.Markdown("### 📤 ส่งคำตอบไปยังแพลตฟอร์ม:")

        with gr.Row():
            send_to_discord_checkbox = gr.Checkbox(
                label="🤖 Discord",
                value=False,
                info="ส่งคำตอบไปยัง Discord channel"
            )

            send_to_line_checkbox = gr.Checkbox(
                label="📱 LINE OA",
                value=False,
                info="ส่งคำตอบไปยัง LINE OA (ต้องมี LINE_USER_ID)"
            )

            send_to_facebook_checkbox = gr.Checkbox(
                label="💬 Facebook Messenger",
                value=False,
                info="ส่งคำตอบไปยัง Facebook Messenger (ต้องมี FB_USER_ID)"
            )

        # เพิ่ม text input สำหรับระบุ user ID
        with gr.Row():
            line_user_id_input = gr.Textbox(
                label="LINE User ID (สำหรับส่งข้อความ)",
                placeholder="ใส่ LINE User ID ที่ต้องการส่งคำตอบ",
                visible=False,
                info="รับ User ID จาก LINE Debug console หรือการทดสอบ"
            )

            fb_user_id_input = gr.Textbox(
                label="Facebook User ID (สำหรับส่งข้อความ)",
                placeholder="ใส่ Facebook User ID ที่ต้องการส่งคำตอบ",
                visible=False,
                info="รับ User ID จาก Facebook Graph API"
            )

        # ฟังก์ชันสำหรับแสดง/ซ่อน user ID input
        def toggle_line_input(checked):
            return gr.update(visible=checked)

        def toggle_fb_input(checked):
            return gr.update(visible=checked)

        send_to_line_checkbox.change(
            fn=toggle_line_input,
            inputs=send_to_line_checkbox,
            outputs=line_user_id_input
        )

        send_to_facebook_checkbox.change(
            fn=toggle_fb_input,
            inputs=send_to_facebook_checkbox,
            outputs=fb_user_id_input
        )

        def update_rag_mode(mode):
            global RAG_MODE
            RAG_MODE = mode
            if mode == "enhanced":
                return "🧠 Enhanced RAG Mode - จะจดจำบริบทการสนทนา"
            else:
                return "📖 Standard RAG Mode - ค้นหาจากเอกสารเท่านั้น"

        rag_mode_selector.change(
            fn=update_rag_mode,
            inputs=rag_mode_selector,
            outputs=rag_mode_status,
            queue=False
        )

        # Enhanced RAG Memory Status
        with gr.Accordion("🧠 Enhanced RAG Memory Status", open=False):
            memory_status_output = gr.Textbox(label="ข้อมูล Memory", lines=4, interactive=False)
            refresh_memory_button = gr.Button("รีเฟรชข้อมูล Memory", size="sm")

        def get_memory_status():
            if RAG_MODE == "enhanced":
                try:
                    memory_info = enhanced_rag.get_memory_info()
                    return f"""📊 Memory Status:
• Total memories: {memory_info['total_memories']}
• Session memories: {memory_info['session_memories']}
• Long-term memories: {memory_info['longterm_memories']}
• Memory window: {memory_info['memory_window']}
"""
                except Exception as e:
                    return f"Error getting memory status: {str(e)}"
            else:
                return "Enhanced RAG is not active. Switch to Enhanced RAG mode to use memory features."

        refresh_memory_button.click(
            fn=get_memory_status,
            inputs=None,
            outputs=memory_status_output,
            queue=False
        )

        # Chat Bot
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="ถามคำถามเกี่ยวกับ PDF")

        # Feedback Section
        with gr.Row():
            gr.Markdown("### 💡 คำตอบนี้ถูกต้องหรือไม่? ช่วยปรับปรุงให้ดีขึ้น")

        with gr.Row():
            with gr.Column(scale=3):
                # เก็บข้อมูลการสนทนาปัจจุบันสำหรับ feedback
                current_question = gr.State("")
                current_answer = gr.State("")
                current_sources = gr.State("")
                feedback_type_state = gr.State("")

                # Enhanced Feedback UI
                with gr.Row():
                    with gr.Column(scale=2):
                        # Rating scale (1-5 stars)
                        rating_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="⭐ ให้คะแนนความพึงพอใจ (1-5)",
                            info="1=แย่มาก, 5=ดีเยี่ยม"
                        )

                        # Quick feedback buttons
                        with gr.Row():
                            good_feedback_btn = gr.Button("👍 ถูกต้อง", variant="primary", size="sm")
                            bad_feedback_btn = gr.Button("👎 ผิดพลาด", variant="secondary", size="sm")

                    with gr.Column(scale=3):
                        # Feedback categories
                        feedback_category = gr.Radio(
                            choices=[
                                ("✅ คำตอบถูกต้อง", "correct"),
                                ("❌ คำตอบผิดพลาด", "incorrect"),
                                ("🤔 ไม่เข้าใจคำถาม", "misunderstood"),
                                ("📄 ข้อมูลไม่ครบ", "incomplete"),
                                ("🔗 แหล่งข้อมูลผิด", "wrong_source"),
                                ("🔄 ต้องการ context เพิ่ม", "need_context")
                            ],
                            value="correct",
                            label="📋 ประเภท Feedback",
                            info="เลือกประเภทที่ตรงที่สุด"
                        )

            with gr.Column(scale=4):
                # ช่องสำหรับใส่ความคิดเห็นและคำตอบที่ถูกต้อง
                with gr.Row():
                    user_comment = gr.Textbox(
                        label="💬 ความคิดเห็นเพิ่มเติม (ไม่บังคับ)",
                        placeholder="อธิบายเพิ่มเติมว่าทำไมคำตอบถึงถูกหรือผิด...",
                        lines=2
                    )

                with gr.Row():
                    corrected_answer = gr.Textbox(
                        label="✅ คำตอบที่ถูกต้อง (ถ้าผิด)",
                        placeholder="ใส่คำตอบที่ถูกต้องที่นี่...",
                        lines=3,
                        visible=False
                    )

                # Source relevance rating
                with gr.Row():
                    source_relevance = gr.Radio(
                        choices=[
                            ("🎯 แหล่งข้อมูลเกี่ยวข้องสูง", "high"),
                            ("📊 แหล่งข้อมูลเกี่ยวข้องปานกลาง", "medium"),
                            ("❌ แหล่งข้อมูลไม่เกี่ยวข้อง", "low")
                        ],
                        value="high",
                        label="📎 ความเกี่ยวข้องของแหล่งข้อมูล",
                        visible=True
                    )

                with gr.Row():
                    submit_feedback_btn = gr.Button("📝 ส่ง Feedback", variant="primary", visible=False)
                    feedback_status = gr.Textbox(label="สถานะ", interactive=False, visible=False)

        # Clear button
        clear_chat = gr.Button("ล้าง")
        # Submit function 
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=chatbot_interface,
            inputs=[chatbot, selected_model, selected_provider, show_source_checkbox, formal_style_checkbox,
                   send_to_discord_checkbox, send_to_line_checkbox, send_to_facebook_checkbox,
                   line_user_id_input, fb_user_id_input, use_graph_reasoning, reasoning_mode,
                   multi_hop_enabled, hop_count],
            outputs=[chatbot, current_question, current_answer, current_sources]
        )
        clear_chat.click(lambda: [], None, chatbot, queue=False)

        # ==================== FEEDBACK EVENT HANDLERS ====================

        def on_feedback_category_change(category):
            """เมื่อเปลี่ยนประเภท feedback"""
            if category in ["incorrect", "incomplete"]:
                return (
                    gr.update(visible=True),   # corrected_answer
                    gr.update(visible=True),   # submit_feedback_btn
                    gr.update(visible=True),   # feedback_status
                    "กรุณาระบุคำตอบที่ถูกต้อง..."
                )
            else:
                return (
                    gr.update(visible=False),  # corrected_answer
                    gr.update(visible=True),   # submit_feedback_btn
                    gr.update(visible=True),   # feedback_status
                    "กำลังส่ง feedback..."
                )

        def on_good_feedback():
            """เมื่อกดปุ่ม 👍"""
            return (
                gr.update(value="correct"),    # feedback_category
                gr.update(visible=False),      # corrected_answer
                gr.update(visible=True),       # submit_feedback_btn
                gr.update(visible=True),       # feedback_status
                "กำลังส่ง feedback ว่าคำตอบถูกต้อง..."
            )

        def on_bad_feedback():
            """เมื่อกดปุ่ม 👎"""
            return (
                gr.update(value="incorrect"),  # feedback_category
                gr.update(visible=True),       # corrected_answer
                gr.update(visible=True),       # submit_feedback_btn
                gr.update(visible=True),       # feedback_status
                "กรุณาระบุคำตอบที่ถูกต้อง..."
            )

        def submit_feedback_handler(category, rating, question, answer, user_comment, corrected_answer, model, source_relevance):
            """Enhanced feedback handler ส่ง feedback ไปยังฐานข้อมูล"""
            if not question or not answer:
                return "❌ ไม่พบข้อมูลการสนทนา กรุณาถามคำถามใหม่"

            # สร้าง detailed feedback comment
            detailed_comment = f"Category: {category}, Rating: {rating}/5, Source Relevance: {source_relevance}"
            if user_comment.strip():
                detailed_comment += f", Comment: {user_comment}"

            # กำหนดประเภท feedback ตาม category
            if category == "correct":
                f_type = "good"
                corrected = ""
            else:
                f_type = "bad"
                corrected = corrected_answer if corrected_answer else "ไม่ได้ระบุ"

            # บันทึกลงฐานข้อมูล
            if save_feedback(question, answer, f_type, detailed_comment, corrected, model, ""):

                # ถ้ามี corrected answer ให้นำไปปรับปรุง RAG ทันที
                if corrected and corrected != "ไม่ได้ระบุ":
                    apply_feedback_to_rag(question, corrected, confidence=rating/5.0)

                # ถ้า rating ต่ำมาก ให้ log เพื่อการวิเคราะห์
                if rating <= 2:
                    logging.warning(f"⚠️ Low quality response detected: Rating={rating}, Category={category}")

                return f"✅ ขอบคุณสำหรับ feedback ระดับ {rating}/5! คำตอบนี้ถูกบันทึกเพื่อปรับปรุงระบบแล้ว"
            else:
                return "❌ เกิดข้อผิดพลาดในการบันทึก feedback กรุณาลองใหม่"

        # Enhanced Feedback Event Handlers
        feedback_category.change(
            fn=on_feedback_category_change,
            inputs=[feedback_category],
            outputs=[corrected_answer, submit_feedback_btn, feedback_status, feedback_status]
        )

        good_feedback_btn.click(
            fn=on_good_feedback,
            inputs=[],
            outputs=[feedback_category, corrected_answer, submit_feedback_btn, feedback_status, feedback_status]
        )

        bad_feedback_btn.click(
            fn=on_bad_feedback,
            inputs=[],
            outputs=[feedback_category, corrected_answer, submit_feedback_btn, feedback_status, feedback_status]
        )

        submit_feedback_btn.click(
            fn=submit_feedback_handler,
            inputs=[feedback_category, rating_slider, current_question, current_answer,
                   user_comment, corrected_answer, selected_model, source_relevance],
            outputs=[feedback_status]
        ).then(
            fn=lambda: [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                       gr.update(value=3), gr.update(value="correct"), ""],
            outputs=[submit_feedback_btn, feedback_status, corrected_answer, rating_slider, feedback_category, user_comment]
        )

        # ==================== TAG MANAGEMENT EVENT HANDLERS ====================

        # Function to update all tag-related components
        def update_all_tag_components():
            """Update all tag components"""
            try:
                tag_data, tag_choices, status_html, _ = refresh_tags_list()
                popular_data, feedback_data = update_tag_statistics()

                # Create choices for dropdown (just the labels)
                dropdown_choices = [choice[0] for choice in tag_choices]

                return tag_data, dropdown_choices, status_html, "", popular_data, feedback_data
            except Exception as e:
                logging.error(f"❌ Failed to update tag components: {str(e)}")
                return [], [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>'), "", [], []

        # Initialize tag lists on load
        demo.load(
            fn=update_all_tag_components,
            inputs=[],
            outputs=[tags_list, selected_tags_search, tag_status, tag_status_display, popular_tags, feedback_tags]
        )

        # Create new tag
        def handle_create_tag(name, color, description):
            """Handle tag creation and update all components"""
            result = create_new_tag(name, color, description)
            if result[0]:  # If successful, update all components
                return update_all_tag_components()
            else:
                # Return the error message from create_new_tag
                tags = get_all_tags()
                tag_choices = [(f"🏷️ {tag[1]}", tag[0]) for tag in tags]
                dropdown_choices = [choice[0] for choice in tag_choices]
                tag_data = [[tag[0], tag[1], tag[2], tag[3] or "", tag[4]] for tag in tags]
                popular_data, feedback_data = update_tag_statistics()
                return tag_data, dropdown_choices, result[2], "", popular_data, feedback_data

        create_tag_btn.click(
            fn=handle_create_tag,
            inputs=[tag_name_input, tag_color_input, tag_desc_input],
            outputs=[tags_list, selected_tags_search, tag_status, tag_status_display, popular_tags, feedback_tags]
        )

        # Refresh tags list
        refresh_tags_btn.click(
            fn=update_all_tag_components,
            inputs=[],
            outputs=[tags_list, selected_tags_search, tag_status, tag_status_display, popular_tags, feedback_tags]
        )

        # Delete selected tag
        def delete_and_refresh(selected_row):
            """Delete tag and refresh all components"""
            if not selected_row or not selected_row.get("ID"):
                return [], [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือก Tag ที่จะลบ</div>'), "", [], []

            tag_id = selected_row["ID"]
            tag_name = selected_row.get("ชื่อ Tag", "")

            success = delete_tag(tag_id)
            if success:
                return update_all_tag_components()
            else:
                return [], [], gr.HTML('<div style="color: red;">❌ ไม่สามารถลบ Tag ได้</div>'), "", [], []

        delete_tag_btn.click(
            fn=delete_and_refresh,
            inputs=[tags_list],
            outputs=[tags_list, selected_tags_search, tag_status, tag_status_display, popular_tags, feedback_tags]
        )

        # Search documents by tags
        search_by_tags_btn.click(
            fn=search_documents_by_selected_tags,
            inputs=[selected_tags_search],
            outputs=[search_results, tag_status_display]
        )

        # Load feedback by tag
        load_feedback_by_tag_btn.click(
            fn=load_feedback_by_selected_tag,
            inputs=[tag_feedback_selector],
            outputs=[tag_feedback_display, tag_status_display]
        )

        # ==================== END TAG MANAGEMENT EVENT HANDLERS ====================

if __name__ == "__main__":
    # ล้างข้อมูล ออกจากระบบ ก่อน เริ่ม Start Web
    clear_vector_db_and_images()

    # Update LightRAG status on load
    try:
        initial_lightrag_status = update_lightrag_status()
        logging.info(f"Initial LightRAG Status: {initial_lightrag_status}")
    except Exception as e:
        logging.warning(f"Failed to update LightRAG status on load: {e}")

    # Note: Webhooks run on different ports (5000, 5001)
    # On Railway, only Gradio port (7860) is exposed
    # Webhooks need to be started manually from Gradio interface for testing

    # Create Gradio interface
    app_interface = create_authenticated_interface()

    # Create custom FastAPI app for full control
    if FASTAPI_AVAILABLE:
        try:
            from fastapi import FastAPI
            from gradio.routes import mount_gradio_app

            # Create FastAPI app
            custom_app = FastAPI()

            # Add webhook routes FIRST
            custom_app.add_api_route(
                "/callback",
                line_callback_api,
                methods=["POST"],
                name="line_webhook"
            )
            custom_app.add_api_route(
                "/webhook",
                facebook_webhook_api,
                methods=["GET", "POST"],
                name="facebook_webhook"
            )
            logging.info("✅ Webhook endpoints (/callback, /webhook) registered to FastAPI")
            logging.info(f"   - LINE callback: POST /callback (enabled: {LINE_ENABLED})")
            logging.info(f"   - Facebook webhook: GET/POST /webhook (enabled: {FB_ENABLED})")

            # Mount Gradio app to FastAPI
            custom_app = mount_gradio_app(custom_app, app_interface, path="/")

            # Launch using custom FastAPI app
            import uvicorn
            port = int(os.getenv('PORT', 7860))
            host = os.getenv('HOST', '0.0.0.0')
            logging.info(f"🚀 Starting server on {host}:{port}")
            uvicorn.run(custom_app, host=host, port=port)

        except Exception as e:
            logging.error(f"❌ Failed to create custom FastAPI app: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Fallback to regular Gradio launch
            logging.info("⚠️ Falling back to regular Gradio launch (webhooks disabled)")
            app_interface.launch()
    else:
        logging.warning("⚠️ FastAPI not available - using regular Gradio launch")
        app_interface.launch()
# Wrapper class for authenticated application
class RAGPDFApplication:
    """Wrapper class for RAG PDF application"""

    def __init__(self):
        self.interface = demo

    def create_interface(self):
        """Create the main RAG PDF interface"""
        try:
            return self.interface
        except Exception as e:
            logging.error(f"❌ Error creating interface: {e}")
            return None

    def get_interface(self):
        """Get the Gradio interface"""
        return self.interface

# ===== CONFIGURATION MANAGEMENT =====

# Environment detection
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT') == 'production' or os.getenv('DYNO') == 'app'
IS_LOCAL = os.getenv('DEPLOYMENT_ENV') == 'local' or not IS_RAILWAY
IS_GPU_AVAILABLE = torch.cuda.is_available() and os.getenv('CUDA_VISIBLE_DEVICES') != ''



