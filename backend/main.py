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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # ElysiaJS dev ports
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

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: datetime

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

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("1")  # Get from auth token in real implementation
):
    """Upload document endpoint"""
    try:
        # Validate file type
        allowed_types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="File type not supported")

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

@app.get("/api/documents")
async def get_documents(user_id: str = "1"):
    """Get user documents"""
    return {
        "documents": [doc.dict() for doc in documents_db.values()],
        "total": len(documents_db)
    }

@app.post("/api/chat")
async def chat(request: ChatRequest, user_id: str = "1"):
    """Chat endpoint - simplified for now"""
    try:
        # Simple response for demo
        answer = f"Received question: {request.question}"
        if request.document_id:
            answer += f" (Document ID: {request.document_id})"

        response = ChatResponse(
            answer=answer,
            sources=[{"source": "demo", "text": "This is a demo response"}],
            timestamp=datetime.now()
        )

        # Store in chat history
        if user_id not in chat_history:
            chat_history[user_id] = []
        chat_history[user_id].append({
            "question": request.question,
            "response": response.dict(),
            "timestamp": datetime.now()
        })

        return response.dict()

    except Exception as e:
        logger.error(f"Error in chat: {e}")
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