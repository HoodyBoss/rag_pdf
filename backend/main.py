#!/usr/bin/env python3
"""
FastAPI Backend for RAG PDF Application
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from pathlib import Path
import shortuuid
from datetime import datetime
import json

# Import AI providers and RAG core
from ai_providers import get_ai_provider_config, get_available_providers, get_provider_models, call_ai_provider, stream_response, get_non_stream_response
from rag_core import RAGCore
from google_sheets import google_sheets_processor
from feedback_system import feedback_system
from api_key_config import api_key_manager
from memory_manager import memory_manager
from social_bots import social_bot_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF API",
    description="RAG PDF Application Backend API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"],  # Next.js dev ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("data/chromadb", exist_ok=True)
os.makedirs("data/temp", exist_ok=True)

# Initialize RAG Core
rag_core = RAGCore()

# Pydantic Models
class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: str
    username: str
    full_name: str
    role: str

class Document(BaseModel):
    id: str
    filename: str
    file_type: str
    size: int
    upload_date: datetime
    processed: bool

class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    ai_provider: Optional[str] = "ollama"
    model: Optional[str] = "gemma3:latest"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 2000
    show_source: Optional[bool] = True

class GeneralChatRequest(BaseModel):
    message: str
    provider: Optional[str] = "ollama"
    model: Optional[str] = "gemma3:latest"

class GoogleSheetsRequest(BaseModel):
    sheets_url: str
    clear_before_upload: bool = False
    user_id: str = "1"

class FeedbackRequest(BaseModel):
    chat_id: Optional[str] = None
    question: str
    answer: str
    rating: int  # 1-5 stars
    feedback_type: str  # "good", "bad", "improvement"
    feedback_text: Optional[str] = None
    ai_provider: str
    model: str
    sources_quality: Optional[int] = None
    response_quality: Optional[int] = None
    user_id: str = "1"

class APIKeyRequest(BaseModel):
    provider: str
    api_key: str
    base_url: Optional[str] = None

class MemoryRequest(BaseModel):
    memory_type: str  # "session", "long_term", "working"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    user_id: str = "1"

class SocialBotRequest(BaseModel):
    platform: str  # "discord", "facebook", "line"
    bot_token: str
    webhook_url: Optional[str] = None
    channels: List[str] = []
    is_active: bool = True

class SocialMessageRequest(BaseModel):
    platform: str
    channel: str
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: datetime
    model: str
    ai_provider: str

# In-memory storage (replace with database later)
users_db = {
    "admin": {
        "id": "1",
        "username": "admin",
        "password": "admin123",  # In production, use hashed passwords
        "full_name": "Administrator",
        "role": "admin"
    }
}

documents_db = {}
chat_history = {}

# Authentication
def authenticate_user(username: str, password: str) -> Optional[User]:
    """Simple authentication function"""
    user_data = users_db.get(username)
    if user_data and user_data["password"] == password:
        return User(
            id=user_data["id"],
            username=user_data["username"],
            full_name=user_data["full_name"],
            role=user_data["role"]
        )
    return None

# Routes
@app.get("/")
async def root():
    return {"message": "RAG PDF API is running"}

@app.post("/api/auth/login")
async def login(user_login: UserLogin):
    """Login endpoint"""
    user = authenticate_user(user_login.username, user_login.password)
    if user:
        return {
            "success": True,
            "user": user.dict(),
            "message": "Login successful"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/me")
async def get_current_user():
    """Get current user info (placeholder for JWT implementation)"""
    return {"user": users_db.get("admin")}  # Return default user for now

# AI Provider endpoints
@app.get("/api/ai/providers")
async def get_ai_providers():
    """Get available AI providers"""
    try:
        providers_data = api_key_manager.get_supported_providers()
        available_providers = []
        provider_configs = {}

        for provider in providers_data:
            provider_id = provider["id"]
            if provider.get("configured", False) or not provider.get("required", True):
                available_providers.append(provider_id)
                provider_configs[provider_id] = {
                    "name": provider["name"],
                    "models": provider["models"],
                    "default_model": provider["models"][0] if provider["models"] else "gemma3:latest",
                    "api_key_required": provider.get("required", True),
                    "configured": provider.get("configured", False),
                    "has_api_key": provider.get("has_api_key", False),
                    "is_active": provider.get("is_active", False)
                }

        return {
            "providers": provider_configs,
            "available": available_providers
        }
    except Exception as e:
        logger.error(f"Error getting AI providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/providers/{provider_name}/models")
async def get_provider_models_endpoint(provider_name: str):
    """Get available models for a specific provider"""
    try:
        providers_data = api_key_manager.get_supported_providers()

        for provider in providers_data:
            if provider["id"] == provider_name:
                return {
                    "provider": provider_name,
                    "models": provider["models"]
                }

        raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Form("1")  # Get from auth token in real implementation
):
    """Upload multiple documents endpoint"""
    try:
        uploaded_files = []
        file_paths = []

        for file in files:
            # Validate file type - expanded to support more formats
            allowed_types = [
                "application/pdf",
                "text/plain",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
                "application/vnd.ms-excel",  # .xls
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/gif",
                "image/bmp"
            ]
            if file.content_type not in allowed_types:
                logger.warning(f"Skipping unsupported file type: {file.filename} ({file.content_type})")
                continue

            # Generate unique filename
            file_id = shortuuid.uuid()
            file_extension = Path(file.filename).suffix
            filename = f"{file_id}{file_extension}"
            file_path = Path("uploads") / filename

            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Store document info
            document = Document(
                id=file_id,
                filename=file.filename,
                file_type=file.content_type,
                size=len(content),
                upload_date=datetime.now(),
                processed=False
            )
            documents_db[file_id] = document
            uploaded_files.append({
                "document_id": file_id,
                "filename": file.filename,
                "size": len(content)
            })
            file_paths.append(str(file_path))

            logger.info(f"Document uploaded: {file.filename} (ID: {file_id})")

        return {
            "success": True,
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files),
            "file_paths": file_paths,
            "message": f"Successfully uploaded {len(uploaded_files)} files"
        }

    except Exception as e:
        logger.error(f"Error uploading multiple documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_uploaded_documents(
    file_ids: List[str],
    clear_before_upload: bool = False,
    user_id: str = "1"
):
    """Process uploaded documents into vector database"""
    try:
        # Get file paths from document IDs
        file_paths = []
        valid_documents = []

        for file_id in file_ids:
            if file_id in documents_db:
                doc = documents_db[file_id]
                file_extension = Path(doc.filename).suffix
                filename = f"{file_id}{file_extension}"
                file_path = Path("uploads") / filename
                file_paths.append(str(file_path))
                valid_documents.append(doc)
            else:
                logger.warning(f"Document ID {file_id} not found")

        if not file_paths:
            return {
                "success": False,
                "message": "❌ No valid documents found to process"
            }

        # Process files using RAG core
        result = rag_core.process_multiple_files(file_paths, clear_before_upload)

        # Mark documents as processed if successful
        if result.get("success", False):
            for doc in valid_documents:
                doc.processed = True
                documents_db[doc.id] = doc

        return result

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/process")
async def process_uploaded_documents_by_paths(
    file_paths: List[str],
    clear_before_upload: bool = False,
    user_id: str = "1"
):
    """Process uploaded documents into vector database (by file paths)"""
    try:
        # Process files using RAG core
        result = rag_core.process_multiple_files(file_paths, clear_before_upload)

        return result

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload-and-process")
async def upload_and_process_documents(
    files: List[UploadFile] = File(...),
    clear_before_upload: bool = Form(False),
    user_id: str = Form("1")  # Get from auth token in real implementation
):
    """Upload and process documents in one step"""
    try:
        # First upload files
        upload_result = await upload_multiple_documents(files, user_id)

        if not upload_result["success"]:
            return upload_result

        # Then process them
        process_result = rag_core.process_multiple_files(
            upload_result["file_paths"],
            clear_before_upload
        )

        return {
            "upload_result": upload_result,
            "process_result": process_result,
            "success": process_result.get("success", False),
            "message": f"Upload: {upload_result['message']} | Process: {process_result.get('message', 'Failed')}"
        }

    except Exception as e:
        logger.error(f"Error in upload and process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/google-sheets/process")
async def process_google_sheets(request: GoogleSheetsRequest):
    """Process Google Sheets data into vector database"""
    try:
        # Extract data from Google Sheets
        sheets_result = google_sheets_processor.extract_google_sheets_data(request.sheets_url)

        if not sheets_result["success"]:
            return sheets_result

        # Process the extracted content into RAG
        if request.clear_before_upload:
            rag_core.clear_database()

        # Add Google Sheets content to vector database
        content_chunks = rag_core.chunk_text(
            sheets_result["content"],
            "Google Sheets"
        )

        # Add to vector database
        if content_chunks:
            documents = [chunk["text"] for chunk in content_chunks]
            metadatas = [chunk["metadata"] for chunk in content_chunks]
            ids = [f"gs_{i}_{shortuuid.uuid()}" for i in range(len(documents))]

            rag_core.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return {
            "success": True,
            "message": "✅ ประมวลผล Google Sheets สำเร็จและเพิ่มลงในฐานข้อมูลแล้ว",
            "sheets_data": sheets_result,
            "chunks_processed": len(content_chunks) if content_chunks else 0,
            "source_url": request.sheets_url
        }

    except Exception as e:
        logger.error(f"Error processing Google Sheets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/google-sheets/preview")
async def preview_google_sheets(request: GoogleSheetsRequest):
    """Preview Google Sheets data before processing"""
    try:
        # Extract and preview data from Google Sheets
        sheets_result = google_sheets_processor.extract_google_sheets_data(request.sheets_url)

        return sheets_result

    except Exception as e:
        logger.error(f"Error previewing Google Sheets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback System Endpoints
@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest):
    """Save user feedback"""
    try:
        feedback_data = request.dict()
        result = feedback_system.save_feedback(feedback_data)
        return result
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    try:
        stats = feedback_system.get_feedback_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/recent")
async def get_recent_feedback(limit: int = 10):
    """Get recent feedback entries"""
    try:
        recent = feedback_system.get_recent_feedback(limit)
        return recent
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/analyze")
async def analyze_feedback():
    """Analyze feedback patterns"""
    try:
        analysis = feedback_system.analyze_feedback_patterns()
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API Key Configuration Endpoints
@app.get("/api/config/providers")
async def get_supported_providers():
    """Get supported AI providers"""
    try:
        providers = api_key_manager.get_supported_providers()
        return {
            "success": True,
            "providers": providers
        }
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/status")
async def get_configuration_status():
    """Get API key configuration status"""
    try:
        status = api_key_manager.get_configuration_status()
        return status
    except Exception as e:
        logger.error(f"Error getting config status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/api-keys")
async def get_api_keys():
    """Get all saved API keys"""
    try:
        result = api_key_manager.get_all_api_keys()
        return result
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/api-key")
async def save_api_key(request: APIKeyRequest):
    """Save API key for provider"""
    try:
        result = api_key_manager.save_api_key(
            provider=request.provider,
            api_key=request.api_key,
            base_url=request.base_url
        )
        return result
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/config/api-key/{provider}")
async def delete_api_key(provider: str):
    """Delete API key for provider"""
    try:
        result = api_key_manager.delete_api_key(provider)
        return result
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/test/{provider}")
async def test_api_key(provider: str):
    """Test API key connectivity"""
    try:
        result = api_key_manager.test_api_key(provider)
        return result
    except Exception as e:
        logger.error(f"Error testing API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Memory Management Endpoints
@app.post("/api/memory")
async def add_memory(request: MemoryRequest):
    """Add memory to specified type"""
    try:
        result = memory_manager.add_memory(
            memory_type=request.memory_type,
            content=request.content,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/relevant")
async def get_relevant_memories(query: str, memory_types: str = None, limit: int = 10):
    """Get relevant memories based on query"""
    try:
        types_list = memory_types.split(",") if memory_types else None
        result = memory_manager.get_relevant_memories(query, types_list, limit)
        return result
    except Exception as e:
        logger.error(f"Error getting relevant memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory statistics"""
    try:
        stats = memory_manager.get_memory_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/summary")
async def get_memory_summary():
    """Get memory summary"""
    try:
        summary = memory_manager.get_memory_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting memory summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/search")
async def search_memories(query: str, memory_types: str = None):
    """Search memories"""
    try:
        types_list = memory_types.split(",") if memory_types else None
        result = memory_manager.search_memories(query, types_list)
        return result
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/consolidate")
async def consolidate_memories():
    """Consolidate working memories to long-term"""
    try:
        result = memory_manager.consolidate_memories()
        return result
    except Exception as e:
        logger.error(f"Error consolidating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/memory/cleanup")
async def cleanup_memories(days: int = 30):
    """Clean up old memories"""
    try:
        result = memory_manager.cleanup_old_memories(days)
        return result
    except Exception as e:
        logger.error(f"Error cleaning up memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Social Media Bots Endpoints
@app.post("/api/social/bots")
async def add_bot_config(request: SocialBotRequest):
    """Add social media bot configuration"""
    try:
        result = social_bot_manager.add_bot_config(
            platform=request.platform,
            bot_token=request.bot_token,
            webhook_url=request.webhook_url,
            channels=request.channels
        )
        return result
    except Exception as e:
        logger.error(f"Error adding bot config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/bots")
async def get_bot_configs():
    """Get all bot configurations"""
    try:
        configs = social_bot_manager.get_bot_configs()
        return configs
    except Exception as e:
        logger.error(f"Error getting bot configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/social/message")
async def send_social_message(request: SocialMessageRequest):
    """Send message to social media platform"""
    try:
        result = social_bot_manager.send_message_to_platform(
            platform=request.platform,
            channel=request.channel,
            message=request.message,
            user_id=request.user_id
        )

        # Log the message
        social_bot_manager.log_message(
            platform=request.platform,
            channel=request.channel,
            message=request.message,
            direction="outgoing",
            user_id=request.user_id
        )

        return result
    except Exception as e:
        logger.error(f"Error sending social message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/messages")
async def get_message_history(platform: str = None, limit: int = 50):
    """Get message history"""
    try:
        history = social_bot_manager.get_message_history(platform, limit)
        return history
    except Exception as e:
        logger.error(f"Error getting message history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/stats")
async def get_platform_stats():
    """Get platform statistics"""
    try:
        stats = social_bot_manager.get_platform_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/social/test/{platform}")
async def test_bot_connection(platform: str):
    """Test bot connection"""
    try:
        result = social_bot_manager.test_bot_connection(platform)
        return result
    except Exception as e:
        logger.error(f"Error testing bot connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/social/bots/{platform}")
async def delete_bot_config(platform: str):
    """Delete bot configuration"""
    try:
        result = social_bot_manager.delete_bot_config(platform)
        return result
    except Exception as e:
        logger.error(f"Error deleting bot config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("1")  # Get from auth token in real implementation
):
    """Upload document endpoint"""
    try:
        # Validate file type - expanded to support more formats
        allowed_types = [
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
            "application/vnd.ms-excel",  # .xls
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/gif",
            "image/bmp"
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"File type not supported. Supported types: PDF, TXT, DOCX, Excel, JPG, PNG")

        # Generate unique filename
        file_id = shortuuid.uuid()
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = Path("uploads") / filename

        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Store document info
        document = Document(
            id=file_id,
            filename=file.filename,
            file_type=file.content_type,
            size=len(content),
            upload_date=datetime.now(),
            processed=False
        )
        documents_db[file_id] = document

        logger.info(f"Document uploaded: {file.filename} (ID: {file_id})")

        return {
            "success": True,
            "document_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "message": "Document uploaded successfully"
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def simple_upload_only(
    file: UploadFile = File(...),
    user_id: str = Form("1")
):
    """Simple upload endpoint - upload only, no processing"""
    try:
        # Validate file type
        allowed_types = [
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/gif",
            "image/bmp"
        ]
        if file.content_type not in allowed_types:
            logger.warning(f"Skipping unsupported file type: {file.filename} ({file.content_type})")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        # Generate unique filename
        file_id = shortuuid.uuid()
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = Path("uploads") / filename

        # Save file
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Store document info
        document = Document(
            id=file_id,
            filename=file.filename,
            file_type=file.content_type,
            size=len(content),
            upload_date=datetime.now(),
            processed=False  # Mark as not processed yet
        )
        documents_db[file_id] = document

        logger.info(f"Document uploaded (not processed): {file.filename} (ID: {file_id})")

        return {
            "success": True,
            "document_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "processed": False,
            "message": "✅ Document uploaded successfully. Click 'Process Data' to add to vector database."
        }

    except Exception as e:
        logger.error(f"Error in upload only: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents(user_id: str = "1"):
    """Get user documents"""
    return {
        "documents": [doc.dict() for doc in documents_db.values()],
        "total": len(documents_db)
    }

@app.post("/api/chat")
async def chat(request: ChatRequest, user_id: str = "1"):
    """Chat endpoint with RAG and AI providers"""
    try:
        # Search for relevant documents
        sources = []
        context = ""

        if rag_core.collection:
            try:
                search_results = rag_core.search_documents(request.question, n_results=5)
                sources = [
                    {
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                        "metadata": result["metadata"],
                        "similarity": result["similarity"]
                    }
                    for result in search_results
                ]

                # Build context from sources
                if sources:
                    context = "\n\n".join([source["text"] for source in sources[:3]])
            except Exception as e:
                logger.warning(f"Error searching documents: {e}")

        # Build prompt with context
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context from documents:\n{context}\n")

        prompt_parts.append(f"Question: {request.question}")

        if context:
            prompt_parts.append("\nPlease answer the question based on the provided context. If the context doesn't contain enough information, say so clearly.")
        else:
            prompt_parts.append("\nPlease answer this question to the best of your ability.")

        full_prompt = "".join(prompt_parts)

        # Get AI response
        try:
            messages = [{"role": "user", "content": full_prompt}]

            if request.stream:
                # For streaming, we'll return the first chunk for now
                # In a real implementation, you'd use Server-Sent Events
                response_stream = stream_response(
                    request.ai_provider,
                    request.model,
                    messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # Get first chunk from stream
                answer = ""
                for chunk in response_stream:
                    answer += chunk
                    if len(answer) > 100:  # Return first 100 chars for demo
                        break
            else:
                answer = get_non_stream_response(
                    request.ai_provider,
                    request.model,
                    messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            answer = f"Error getting AI response: {str(e)}"

        response = ChatResponse(
            answer=answer,
            sources=sources if request.show_source else [],
            timestamp=datetime.now(),
            model=request.model,
            ai_provider=request.ai_provider
        )

        # Store in chat history
        if user_id not in chat_history:
            chat_history[user_id] = []
        chat_history[user_id].append({
            "question": request.question,
            "response": response.dict(),
            "timestamp": datetime.now(),
            "ai_provider": request.ai_provider,
            "model": request.model
        })

        return response.dict()

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/general")
async def general_chat(request: GeneralChatRequest, user_id: str = "1"):
    """General chat endpoint with RAG across all documents"""
    try:
        # Search for relevant documents across all uploaded content
        sources = []
        context = ""

        if rag_core.collection:
            try:
                search_results = rag_core.search_documents(request.message, n_results=5)
                sources = [
                    {
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                        "metadata": result["metadata"],
                        "similarity": result["similarity"]
                    }
                    for result in search_results
                ]

                # Build context from sources
                if sources:
                    context = "\n\n".join([source["text"] for source in sources[:3]])
            except Exception as e:
                logger.warning(f"Error searching documents: {e}")

        # Build prompt with context
        prompt_parts = []
        if context:
            prompt_parts.append(f"Context from documents:\n{context}\n")

        prompt_parts.append(f"User: {request.message}")
        prompt = "\n".join(prompt_parts)

        # Get AI provider
        ai_provider = get_ai_provider_config(request.provider or "ollama")
        if not ai_provider:
            raise HTTPException(status_code=400, detail=f"AI provider '{request.provider}' not available")

        # Generate response
        messages = [{"role": "user", "content": prompt}]

        # Use non-streaming response for better stability
        full_response = get_non_stream_response(
            provider_name=request.provider or "ollama",
            model=request.model or "gemma3:latest",
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )

        # Extract just the answer part
        answer = full_response.strip()

        return {
            "response": answer,
            "sources": sources,
            "provider": request.provider or "ollama",
            "model": request.model or "gemma3:latest"
        }

    except Exception as e:
        logger.error(f"Error in general chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_chat_history(user_id: str = "1"):
    """Get chat history"""
    return {
        "history": chat_history.get(user_id, []),
        "total": len(chat_history.get(user_id, []))
    }

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user_id: str = "1"):
    """Delete document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file
    doc = documents_db[document_id]
    file_path = Path("uploads") / f"{document_id}{Path(doc.filename).suffix}"
    if file_path.exists():
        file_path.unlink()

    # Remove from database
    del documents_db[document_id]

    return {"success": True, "message": "Document deleted successfully"}

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)