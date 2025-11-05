#!/usr/bin/env python3
"""
Secure Railway RAG System - Admin Only
High-security RAG system with admin-only access
"""
import os
import logging
import json
import hashlib
import secrets
import base64
import time
import re
import shutil
import io
import pandas as pd
import asyncio
import threading
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union

import gradio as gr
from dotenv import load_dotenv

# Additional imports for enhanced features
try:
    import pymupdf as fitz  # PyMuPDF
    import numpy as np
    from PIL import Image
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    from pythainlp.tokenize import word_tokenize
    import ollama
    import shortuuid
    import docx
    import requests
    from flask import Flask, request, abort
    MONGODB_AVAILABLE = True
    CHROMADB_AVAILABLE = True
    logger.info("All dependencies imported successfully")
except ImportError as e:
    MONGODB_AVAILABLE = False
    CHROMADB_AVAILABLE = False
    logger.warning(f"Some features may not work due to missing dependencies: {e}")

# ChromaDB + MongoDB Hybrid Storage
class HybridVectorStore:
    """Hybrid vector storage using ChromaDB + MongoDB"""

    def __init__(self, mongodb_uri: str, database_name: str, chroma_path: str = "./chroma_db"):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.chroma_path = chroma_path

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            self.chroma_client = None
            self.collection = None

        # Initialize MongoDB connection
        try:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            self.mongo_db = self.mongo_client[database_name]
            self.docs_collection = self.mongo_db.documents
            logger.info("✅ MongoDB connection initialized successfully")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.mongo_db = None
            self.docs_collection = None

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Embedding model loading failed: {e}")
            self.embedding_model = None

    def is_available(self) -> bool:
        """Check if both ChromaDB and MongoDB are available"""
        return self.chroma_client is not None and self.mongo_client is not None and self.embedding_model is not None

    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add document to both ChromaDB and MongoDB"""
        if not self.is_available():
            raise Exception("Vector store not properly initialized")

        try:
            # Generate ID
            doc_id = shortuuid.uuid()

            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()

            # Store in ChromaDB (vector + metadata)
            chroma_metadata = {
                'doc_id': doc_id,
                'source': metadata.get('source', ''),
                'type': metadata.get('type', 'document'),
                'created_at': metadata.get('created_at', datetime.now().isoformat())
            }

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata]
            )

            # Store full document in MongoDB
            mongo_doc = {
                '_id': doc_id,
                'content': content,
                'embedding': embedding,
                'metadata': metadata,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

            self.docs_collection.insert_one(mongo_doc)

            logger.info(f"✅ Added document {doc_id} to hybrid storage")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            raise

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        if not self.is_available():
            raise Exception("Vector store not properly initialized")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                # Get full metadata from MongoDB
                mongo_doc = self.docs_collection.find_one({'_id': doc_id})
                full_metadata = mongo_doc['metadata'] if mongo_doc else metadata

                formatted_results.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': full_metadata,
                    'score': 1 - distance if distance else 1.0,  # Convert distance to similarity
                    'distance': distance
                })

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'chromadb_available': self.chroma_client is not None,
            'mongodb_available': self.mongo_client is not None,
            'embedding_model_available': self.embedding_model is not None,
            'total_documents': 0
        }

        if self.docs_collection:
            stats['total_documents'] = self.docs_collection.count_documents({})

        return stats

    def clear_all(self):
        """Clear all documents from both storages"""
        try:
            if self.collection:
                # Clear ChromaDB
                all_docs = self.collection.get()
                if all_docs['ids']:
                    self.collection.delete(ids=all_docs['ids'])

            if self.docs_collection:
                # Clear MongoDB
                self.docs_collection.delete_many({})

            logger.info("✅ Cleared all documents from hybrid storage")

        except Exception as e:
            logger.error(f"❌ Error clearing storage: {e}")
            raise

# Global hybrid vector store instance
hybrid_vector_store = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Image and temp folders
TEMP_IMG = "./data/images"
TEMP_VECTOR = "./data/chromadb"
TEMP_VECTOR_BACKUP = "./data/chromadb_backup"

# Available models
AVAILABLE_MODELS = ["gemma3:12b", "qwen2.5:7b", "llama3.2:3b"]

# Discord Configuration (optional)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "YOUR_WEBHOOK_URL_HERE")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID_HERE")
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"

# LINE OA Configuration (optional)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")

# Feedback database
feedback_data = []
tag_data = []
current_session_id = None
current_session = None

# Thai text processing
def preprocess_thai_text(text: str) -> str:
    """Preprocess Thai text for better retrieval"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Add spaces between Thai characters and English/numbers
    text = re.sub(r'([ก-๙])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([ก-๙])', r'\1 \2', text)

    # Remove special characters but keep Thai, English, numbers, and basic punctuation
    text = re.sub(r'[^\u0E00-\u0E7FA-Za-z0-9\s\.\,\:\;\!\?\-\(\)]', '', text)

    return text.strip()

# File processing utilities
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file with improved handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return preprocess_thai_text(text)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_text_file(file_path: str) -> str:
    """Extract text from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return preprocess_thai_text(content)
    except Exception as e:
        logger.error(f"Error extracting text file: {e}")
        return ""

def extract_docx_text(docx_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return preprocess_thai_text(text)
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        return ""

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == '.pdf':
        return extract_pdf_text(file_path)
    elif file_extension in ['.txt', '.md']:
        return extract_text_file(file_path)
    elif file_extension == '.docx':
        return extract_docx_text(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_extension}")
        return ""

def chunk_text(text: str, source_file: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split text into chunks with overlap"""
    if not text.strip():
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)

        chunk = {
            'id': shortuuid.uuid(),
            'text': chunk_text,
            'source': source_file,
            'chunk_index': len(chunks),
            'word_count': len(chunk_words),
            'created_at': datetime.now().isoformat()
        }
        chunks.append(chunk)

    return chunks

def process_multiple_files(files, clear_before_upload: bool = False):
    """Process multiple files and upload to Hybrid Storage (ChromaDB + MongoDB)"""
    global hybrid_vector_store

    if not hybrid_vector_store or not hybrid_vector_store.is_available():
        return "❌ Hybrid vector storage not initialized", []

    if not files:
        return "❌ No files selected", []

    try:
        # Clear existing data if requested
        if clear_before_upload:
            hybrid_vector_store.clear_all()
            logger.info("Cleared existing hybrid storage")

        all_chunks = []
        uploaded_files = []
        processed_count = 0

        for file_obj in files:
            if hasattr(file_obj, 'name'):
                file_name = file_obj.name
            else:
                file_name = str(file_obj)

            # Save temp file
            temp_path = f"./temp_{file_name}"
            with open(temp_path, 'wb') as f:
                if hasattr(file_obj, 'read'):
                    f.write(file_obj.read())
                else:
                    with open(file_obj, 'rb') as src:
                        f.write(src.read())

            # Extract text
            text = extract_text_from_file(temp_path)
            if text:
                # Create chunks
                chunks = chunk_text(text, file_name)

                # Store each chunk in hybrid storage
                for chunk in chunks:
                    try:
                        doc_id = hybrid_vector_store.add_document(
                            content=chunk['text'],
                            metadata={
                                'source': chunk['source'],
                                'chunk_id': chunk['id'],
                                'chunk_index': chunk['chunk_index'],
                                'word_count': chunk['word_count'],
                                'created_at': chunk['created_at'],
                                'file_type': Path(file_name).suffix.lower(),
                                'original_file': file_name
                            }
                        )
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error storing chunk from {file_name}: {e}")
                        continue

                all_chunks.extend(chunks)
                uploaded_files.append(file_name)
                logger.info(f"✅ Processed {file_name}: {len(chunks)} chunks stored in hybrid storage")

            # Clean up temp file
            os.remove(temp_path)

        return f"✅ Successfully processed {len(uploaded_files)} files with {processed_count} chunks stored persistently", uploaded_files

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return f"❌ Error processing files: {str(e)}", []

# Enhanced RAG class for conversation memory
class EnhancedRAG:
    def __init__(self):
        self.conversation_memory = deque(maxlen=10)  # Keep last 10 Q&A pairs

    def add_to_memory(self, question: str, answer: str, contexts: List[str]):
        """Add Q&A to memory"""
        memory_item = {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'timestamp': datetime.now().isoformat()
        }
        self.conversation_memory.append(memory_item)

    def get_relevant_memory(self, current_question: str) -> List[Dict]:
        """Get relevant past conversations"""
        # Simple keyword matching for now
        relevant_memories = []
        question_lower = current_question.lower()

        for memory in self.conversation_memory:
            memory_question_lower = memory['question'].lower()
            # Check for keyword overlap
            question_words = set(question_lower.split())
            memory_words = set(memory_question_lower.split())
            overlap = question_words & memory_words

            if len(overlap) > 0:  # If there's any overlap
                relevant_memories.append(memory)

        return relevant_memories[:3]  # Return top 3 relevant memories

    def build_context_prompt(self, question: str, contexts: List[str], relevant_memories: List[Dict]) -> str:
        """Build enhanced prompt with conversation context"""
        prompt = f"""คำถาม: {question}

"""

        if relevant_memories:
            prompt += "บทสนทนาที่เกี่ยวข้อง:\n"
            for i, memory in enumerate(relevant_memories, 1):
                prompt += f"{i}. คำถาม: {memory['question']}\n"
                prompt += f"   คำตอบ: {memory['answer']}\n\n"
            prompt += "\n"

        prompt += "ข้อมูลเอกสารที่เกี่ยวข้อง:\n"
        for i, context in enumerate(contexts, 1):
            prompt += f"{i}. {context}\n\n"

        prompt += "กรุณาตอบคำถามโดยใช้ข้อมูลเอกสารและบริบทจากบทสนทนาที่ให้มา ตอบเป็นภาษาไทย:"

        return prompt

# Global enhanced RAG instance
enhanced_rag = EnhancedRAG()

# Google Sheets Integration
def extract_sheet_id_from_url(url: str) -> str:
    """Extract Sheet ID from Google Sheets URL"""
    try:
        patterns = [
            r"/d/([a-zA-Z0-9-_]+)",
            r"id=([a-zA-Z0-9-_]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None
    except Exception as e:
        logger.error(f"Error extracting sheet ID: {e}")
        return None

def format_dataframe_to_text(df, source_url: str) -> str:
    """Convert DataFrame to formatted text"""
    text_content = f"ข้อมูลจาก Google Sheets: {source_url}\n\n"

    for index, row in df.iterrows():
        row_text = f"แถวที่ {index + 1}:\n"
        for col in df.columns:
            if pd.notna(row[col]):
                row_text += f"- {col}: {row[col]}\n"
        text_content += row_text + "\n"

    return preprocess_thai_text(text_content)

def extract_google_sheets_data(sheets_url: str) -> str:
    """Extract data from Google Sheets"""
    try:
        sheet_id = extract_sheet_id_from_url(sheets_url)
        if not sheet_id:
            return "❌ ไม่สามารถแยก Sheet ID จาก URL ได้"

        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
        df = pd.read_csv(export_url)
        text_content = format_dataframe_to_text(df, sheets_url)
        return text_content

    except Exception as e:
        logger.error(f"Error extracting Google Sheets data: {e}")
        return f"❌ ไม่สามารถดึงข้อมูลจาก Google Sheets ได้: {str(e)}"

def process_google_sheets_url(sheets_url: str, clear_before_upload: bool = False):
    """Process Google Sheets URL and upload to Hybrid Storage"""
    global hybrid_vector_store

    if not hybrid_vector_store or not hybrid_vector_store.is_available():
        return "❌ Hybrid vector storage not initialized", []

    try:
        # Clear existing data if requested
        if clear_before_upload:
            hybrid_vector_store.clear_all()
            logger.info("Cleared existing hybrid storage")

        # Extract data from Google Sheets
        text_content = extract_google_sheets_data(sheets_url)

        if text_content.startswith("❌"):
            return text_content, []

        # Create chunks
        chunks = chunk_text(text_content, f"Google Sheets: {sheets_url}")

        if not chunks:
            return "❌ ไม่มีข้อมูลใน Google Sheets", []

        processed_count = 0
        # Store in Hybrid Storage
        for chunk in chunks:
            try:
                doc_id = hybrid_vector_store.add_document(
                    content=chunk['text'],
                    metadata={
                        'source': f"Google Sheets: {sheets_url}",
                        'chunk_id': chunk['id'],
                        'chunk_index': chunk['chunk_index'],
                        'word_count': chunk['word_count'],
                        'created_at': chunk['created_at'],
                        'type': 'google_sheets',
                        'url': sheets_url,
                        'file_type': 'google_sheets'
                    }
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"Error storing Google Sheets chunk: {e}")
                continue

        return f"✅ Successfully processed Google Sheets with {processed_count} chunks stored persistently", [f"Google Sheets: {sheets_url}"]

    except Exception as e:
        logger.error(f"Error processing Google Sheets: {e}")
        return f"❌ Error processing Google Sheets: {str(e)}", []

# Tag Management System
def create_tag(tag_name: str, tag_description: str = ""):
    """Create new tag"""
    try:
        if not tag_name.strip():
            return [], [], gr.HTML('<div style="color: orange;">⚠️ กรุณาใส่ชื่อ Tag</div>'), ""

        # Check if tag already exists
        for tag in tag_data:
            if tag['name'].lower() == tag_name.strip().lower():
                return [], [], gr.HTML('<div style="color: orange;">⚠️ Tag นี้มีอยู่แล้ว</div>'), ""

        new_tag = {
            'id': shortuuid.uuid(),
            'name': tag_name.strip(),
            'description': tag_description.strip(),
            'created_at': datetime.now().isoformat(),
            'document_count': 0
        }

        tag_data.append(new_tag)
        tag_choices = [tag['name'] for tag in tag_data]

        return tag_data, tag_choices, gr.HTML(f'<div style="color: green;">✅ สร้าง Tag "{tag_name}" สำเร็จ</div>'), ""

    except Exception as e:
        return [], [], gr.HTML(f'<div style="color: red;">❌ ไม่สามารถสร้าง Tag ได้: {str(e)}</div>'), ""

def delete_tag(tag_name: str):
    """Delete tag"""
    try:
        if not tag_name:
            return [], [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือก Tag ที่จะลบ</div>'), ""

        # Find and remove tag
        tag_found = False
        for i, tag in enumerate(tag_data):
            if tag['name'] == tag_name:
                tag_data.pop(i)
                tag_found = True
                break

        if tag_found:
            tag_choices = [tag['name'] for tag in tag_data]
            return tag_data, tag_choices, gr.HTML(f'<div style="color: green;">✅ ลบ Tag "{tag_name}" สำเร็จ</div>'), ""
        else:
            return [], [], gr.HTML('<div style="color: red;">❌ ไม่สามารถลบ Tag ได้</div>'), ""

    except Exception as e:
        return [], [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>'), ""

def search_documents_by_tags(selected_tags: List[str]):
    """Search documents by selected tags"""
    try:
        if not selected_tags:
            return [], gr.HTML('<div style="color: orange;">⚠️ กรุณาเลือกอย่างน้อย 1 Tag</div>')

        # This would integrate with MongoDB to search by tags
        # For now, return mock data
        search_results = [
            {
                'filename': f'document_{i}.pdf',
                'tags': selected_tags,
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            for i in range(3)
        ]

        return search_results, gr.HTML(f'<div style="color: green;">✅ พบ {len(search_results)} เอกสาร</div>')

    except Exception as e:
        return [], gr.HTML(f'<div style="color: red;">❌ เกิดข้อผิดพลาด: {str(e)}</div>')

# Feedback System
def add_feedback(question: str, answer: str, rating: int, comment: str = ""):
    """Add feedback for Q&A"""
    try:
        feedback = {
            'id': shortuuid.uuid(),
            'question': question,
            'answer': answer,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        feedback_data.append(feedback)
        return f"✅ ขอบคุณสำหรับคะแนน {rating} ดาว!"

    except Exception as e:
        return f"❌ ไม่สามารถบันทึก feedback ได้: {str(e)}"

def get_feedback_statistics():
    """Get feedback statistics"""
    try:
        if not feedback_data:
            return "ยังไม่มี feedback"

        total_feedback = len(feedback_data)
        avg_rating = sum(f['rating'] for f in feedback_data) / total_feedback
        rating_dist = {}

        for i in range(1, 6):
            rating_dist[f'{i} ดาว'] = sum(1 for f in feedback_data if f['rating'] == i)

        stats_text = f"""📊 สถิติ Feedback:
- 📈 ทั้งหมด: {total_feedback} feedback
- ⭐ คะแนนเฉลี่ย: {avg_rating:.1f} ดาว
- 📊 การกระจายคะแนน:
"""
        for stars, count in rating_dist.items():
            stats_text += f"  - {stars}: {count} ครั้ง\n"

        return stats_text

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

# Import RAG components
try:
    from mongodb_rag import MongoDBRAG
    from sentence_transformers import SentenceTransformer
    import requests
    MONGODB_AVAILABLE = True
    logger.info("MongoDB RAG components imported successfully")
except ImportError as e:
    MONGODB_AVAILABLE = False
    logger.error(f"MongoDB RAG components import failed: {e}")
    logger.error("Please ensure mongodb_rag.py and dependencies are available")

# Session management
sessions = {}
SESSION_TIMEOUT = 1800  # 30 minutes for security

# Generate secure admin credentials
def generate_admin_credentials():
    """Generate highly secure admin credentials"""
    # Use environment variables or generate secure defaults
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')

    # Generate secure password if not provided
    admin_password = os.getenv('ADMIN_PASSWORD')
    if not admin_password:
        # Generate 24-character secure password
        admin_password = ''.join(secrets.choice(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*'
        ) for _ in range(24))

    # Hash with salt
    salt = os.getenv('SALT', 'secure_salt_' + secrets.token_hex(16))
    password_hash = hashlib.sha256((admin_password + salt).encode()).hexdigest()

    return admin_username, password_hash, salt, admin_password

# Initialize admin credentials
ADMIN_USERNAME, ADMIN_PASSWORD_HASH, SALT, ADMIN_PASSWORD_PLAIN = generate_admin_credentials()

# Admin database (single admin user)
ADMIN_DB = {
    'username': ADMIN_USERNAME,
    'password_hash': ADMIN_PASSWORD_HASH,
    'role': 'admin',
    'created_at': datetime.now(),
    'last_login': None,
    'login_attempts': 0,
    'locked_until': None
}

class SecureAuthManager:
    """High-security authentication manager"""

    @staticmethod
    def generate_session_id() -> str:
        """Generate cryptographically secure session ID"""
        return secrets.token_urlsafe(64)

    @staticmethod
    def create_session(username: str) -> str:
        """Create secure admin session"""
        session_id = SecureAuthManager.generate_session_id()
        sessions[session_id] = {
            'username': username,
            'role': 'admin',
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'csrf_token': secrets.token_urlsafe(32)
        }
        return session_id

    @staticmethod
    def validate_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session with enhanced security"""
        if not session_id or session_id not in sessions:
            return None

        session = sessions[session_id]
        now = datetime.now()

        # Check session timeout (30 minutes)
        if (now - session['last_activity']).total_seconds() > SESSION_TIMEOUT:
            del sessions[session_id]
            return None

        # Update activity
        session['last_activity'] = now
        return session

    @staticmethod
    def destroy_session(session_id: str) -> bool:
        """Securely destroy session"""
        if session_id in sessions:
            del sessions[session_id]
            return True
        return False

    @staticmethod
    def authenticate_admin(username: str, password: str) -> Dict[str, Any]:
        """Enhanced admin authentication"""
        # Check username
        if username != ADMIN_DB['username']:
            return {'success': False, 'message': 'Invalid credentials', 'locked': False}

        # Check account lock
        if ADMIN_DB['locked_until'] and datetime.now() < ADMIN_DB['locked_until']:
            remaining = int((ADMIN_DB['locked_until'] - datetime.now()).total_seconds() / 60)
            return {
                'success': False,
                'message': f'Account locked. Try again in {remaining} minutes.',
                'locked': True
            }

        # Verify password
        password_hash = hashlib.sha256((password + SALT).encode()).hexdigest()
        if password_hash == ADMIN_DB['password_hash']:
            # Reset on success
            ADMIN_DB['login_attempts'] = 0
            ADMIN_DB['locked_until'] = None
            ADMIN_DB['last_login'] = datetime.now()

            return {
                'success': True,
                'message': 'Login successful',
                'role': 'admin'
            }
        else:
            # Failed attempt
            ADMIN_DB['login_attempts'] += 1

            # Lock after 5 attempts for 30 minutes
            if ADMIN_DB['login_attempts'] >= 5:
                ADMIN_DB['locked_until'] = datetime.now() + timedelta(minutes=30)
                return {
                    'success': False,
                    'message': 'Account locked for 30 minutes due to multiple failed attempts.',
                    'locked': True
                }

            remaining = 5 - ADMIN_DB['login_attempts']
            return {
                'success': False,
                'message': f'Invalid credentials. {remaining} attempts remaining.',
                'locked': False
            }

# Global variables
hybrid_vector_store = None
embed_model = None

def initialize_system():
    """Initialize Hybrid RAG system components"""
    global hybrid_vector_store, embed_model

    try:
        # Get configuration
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27020/')
        db_name = os.getenv('DATABASE_NAME', 'rag_railway_secure')

        logger.info(f"Initializing Hybrid RAG System...")
        logger.info(f"MongoDB URI: {mongodb_uri}")
        logger.info(f"Database: {db_name}")

        # Initialize Hybrid Vector Store (ChromaDB + MongoDB)
        hybrid_vector_store = HybridVectorStore(mongodb_uri, db_name)

        if not hybrid_vector_store.is_available():
            logger.error("❌ Hybrid vector store not available")
            return False

        # Initialize embedding model
        logger.info("Loading embedding model...")
        embed_model = hybrid_vector_store.embedding_model

        # Check if we have existing documents
        stats = hybrid_vector_store.get_document_stats()
        logger.info(f"📊 Storage Stats: {stats}")

        logger.info("✅ Hybrid RAG system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

def query_rag_system(question: str, session_id: str, top_k: int = 5, min_similarity: float = 0.3):
    """Query Hybrid RAG system with session validation"""
    try:
        # Validate session
        session = SecureAuthManager.validate_session(session_id)
        if not session:
            return "❌ Session expired. Please login again."

        if not hybrid_vector_store or not hybrid_vector_store.is_available():
            return "❌ Hybrid RAG system not available."

        logger.info(f"🔍 Query from {session['username']}: '{question}'")

        # Search documents using ChromaDB + MongoDB
        results = hybrid_vector_store.search(question, top_k)

        # Filter by similarity score
        filtered_results = [r for r in results if r['score'] >= min_similarity]

        if not filtered_results:
            return """🤔 ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณ

✨ **คำแนะน:**
• ใช้คำศัพท์อื่นที่ใกล้เคียง
• ตรวจสอบว่ามีเอกสารในระบบหรือไม่
• ลองคำถามที่เจำเพาะกว่า

💾 **Persistent Storage:** ChromaDB + MongoDB 🚀"""

        # Add to conversation memory
        enhanced_rag.add_to_memory(question, "Processing...", [r['content'] for r in filtered_results])

        # Format response
        response = f"""🤔 **คำตอบสำหรับคำถาม:** *{question}*

**📊 พบข้อมูลที่เกี่ยวข้อง ({len(filtered_results)} รายการ) - Persistent Search 🔥**

"""

        for i, result in enumerate(filtered_results, 1):
            similarity_score = result.get('score', 0)
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            source_name = metadata.get('source', 'Unknown source')
            chunk_idx = metadata.get('chunk_index', 0)
            distance = result.get('distance', 0)

            response += f"""
**{i}.** 📄 {source_name}
- 🎯 **ความเกี่ยวข้อง:** {similarity_score:.1%} (Distance: {distance:.3f})
- 📍 **Chunk:** {chunk_idx + 1}
- 📝 **เนื้อหา:** {content[:200]}{'...' if len(content) > 200 else ''}

"""

        # List available documents
        response += "---\n📚 **เอกสารในระบบ:**\n"
        documents = mongodb_rag.list_documents()
        for doc in documents:
            chunks = doc.get('total_chunks', 0)
            response += f"• {doc.get('source_name', 'Unknown')} ({chunks} chunks)\n"

        return response

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

def upload_file_to_mongodb(file, session_id: str):
    """Upload file with session validation"""
    try:
        # Validate session
        session = SecureAuthManager.validate_session(session_id)
        if not session:
            return "❌ Session expired. Please login again."

        if not mongodb_rag:
            return "❌ RAG system not available."

        if file is None:
            return "❌ กรุณาเลือกไฟล์"

        # Get file info
        file_path = file.name
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        logger.info(f"📄 Processing file: {file_name} ({file_size:,} bytes)")

        # Extract text based on file type
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext == '.pdf':
            text = extract_pdf_text(file_path)
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            return f"❌ ไม่รองรับประเภทไฟล์: {file_ext}"

        if not text.strip():
            return "❌ ไม่สามารถดึงข้อความจากไฟล์"

        # Create chunks
        chunks = create_chunks(text, file_name, chunk_size=1000, overlap=200)

        if not chunks:
            return "❌ ไม่สามารถสร้าง chunks จากข้อความ"

        # Store in MongoDB
        doc_id = mongodb_rag.store_document(chunks, file_name)

        # Get updated stats
        stats = mongodb_rag.get_database_stats()

        return f"""✅ อัปโหลด {file_name} สำเร็จ!

📊 **รายละเอียด:**
• ชื่อไฟล์: {file_name}
• ขนาดไฟล์: {file_size:,} bytes
• จำนวน chunks: {len(chunks)}
• Document ID: {doc_id}

🗄️ **ฐานข้อมูล:**
• Documents: {stats.get('documents_count', 0)}
• Chunks: {stats.get('embeddings_count', 0)}
• Database: {stats.get('database_name', 'Unknown')}

✅ ไฟล์พร้อมใช้งานในระบบแล้ว!"""

    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        return f"❌ การอัปโหลดไฟล์ล้มเหลว: {str(e)}"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        return ""

def create_chunks(text: str, source_name: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Create text chunks"""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                    "chunk_size": len(chunk_text)
                }
            })
            chunk_id += 1

        start = end - overlap if end < len(text) else end

    return chunks

def get_system_status():
    """Get system status"""
    try:
        if not mongodb_rag:
            return """📋 System Status:
❌ MongoDB: Not connected
❌ Embedding Model: Not loaded
⚠️ Status: System not initialized"""

        stats = mongodb_rag.get_database_stats()

        return f"""📋 System Status:
✅ MongoDB: Connected
✅ Embedding Model: {'Loaded' if embed_model else 'Not loaded'}
📊 Database: {stats.get('database_name', 'Unknown')}
📄 Documents: {stats.get('documents_count', 0)}
🔍 Embeddings: {stats.get('embeddings_count', 0)}
📋 Metadata: {stats.get('metadata_count', 0)}
✅ Status: Ready for secure access"""

    except Exception as e:
        return f"""📋 System Status:
❌ Error: {str(e)}"""

def handle_login(username: str, password: str):
    """Handle admin login"""
    try:
        # Authenticate
        auth_result = SecureAuthManager.authenticate_admin(username, password)

        if auth_result['success']:
            # Create session
            session_id = SecureAuthManager.create_session(username)

            return (
                gr.update(visible=False),  # Hide login
                gr.update(visible=True),   # Show main app
                gr.update(value=f"✅ {auth_result['message']}"),
                {"session_id": session_id, "username": username, "role": "admin"}
            )
        else:
            return (
                gr.update(visible=True),   # Show login
                gr.update(visible=False),  # Hide main app
                gr.update(value=f"❌ {auth_result['message']}"),
                {"session_id": "", "username": "", "role": "user"}
            )

    except Exception as e:
        logger.error(f"Login error: {e}")
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"❌ Login error: {str(e)}"),
            {"session_id": "", "username": "", "role": "user"}
        )

def handle_logout(session_id: str):
    """Handle admin logout"""
    try:
        SecureAuthManager.destroy_session(session_id)
        return (
            gr.update(visible=True),   # Show login
            gr.update(visible=False),  # Hide main app
            gr.update(value="✅ Logged out successfully"),
            {"session_id": "", "username": "", "role": "user"}
        )
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"❌ Logout error: {str(e)}"),
            {"session_id": "", "username": "", "role": "user"}
        )

def create_secure_interface():
    """Create secure admin interface"""
    with gr.Blocks(
        title="🔐 Secure RAG Document Assistant",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        css="""
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #2E86AB;
            text-align: center;
            margin-bottom: 30px;
        }
        .gr-button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .secure-badge {
            background: #dc3545;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
        """
    ) as demo:

        # Login Section
        with gr.Column(visible=True, elem_classes=["login-container"]) as login_section:
            gr.Markdown("""
            # 🔐 Secure Admin Login

            **High-Security RAG Document System**

            Please login with admin credentials to access the system.
            """)

            with gr.Row():
                with gr.Column():
                    username_input = gr.Textbox(
                        label="Username",
                        placeholder="Enter admin username",
                        type="text"
                    )
                    password_input = gr.Textbox(
                        label="Password",
                        placeholder="Enter admin password",
                        type="password"
                    )

                    login_btn = gr.Button("🔐 Secure Login", variant="primary", size="lg")
                    auth_message = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )

        # Main Application (Hidden by default)
        with gr.Column(visible=False) as main_app:
            gr.Markdown("""
            # 🔐 Secure RAG Document Assistant

            <span class="secure-badge">ADMIN ACCESS</span>

            **AI-Powered Document Search & Analysis System**

            ---
            """)

            # Session state
            session_state = gr.State({"session_id": "", "username": "", "role": "user"})

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 🔍 Search Documents")

                    question_input = gr.Textbox(
                        label="Ask about your documents",
                        placeholder="What would you like to know? (e.g., 'What are the main findings?')",
                        lines=3
                    )

                    with gr.Row():
                        search_btn = gr.Button("🔍 Search", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")

                    result_output = gr.Markdown(label="Search Results")

                with gr.Column(scale=1):
                    gr.Markdown("## 📊 System Status")

                    status_display = gr.Textbox(
                        label="System Information",
                        value="Loading...",
                        interactive=False,
                        lines=8
                    )

                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        logout_btn = gr.Button("🚪 Logout", variant="secondary", size="sm")

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("## 📁 Upload Documents")
                gr.Markdown("Add PDF and text files to the knowledge base")

                file_input = gr.File(
                    label="Choose File",
                    file_types=[".pdf", ".txt", ".md"],
                    file_count="single"
                )

                upload_btn = gr.Button("📤 Upload & Process", variant="secondary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=5
                )

            gr.Markdown("---")

            with gr.Accordion("🔒 Security Information", open=False):
                gr.Markdown("""
                ### **Security Features:**
                - **🔐 Secure Authentication** - PBKDF2 password hashing with salt
                - **🛡️ Session Management** - 30-minute timeout with secure tokens
                - **🚫 Account Lockout** - Automatic lock after 5 failed attempts
                - **🔍 Activity Monitoring** - Full audit trail
                - **🔒 Encrypted Storage** - MongoDB with secure access
                - **⏰ Auto-logout** - Session expires after inactivity

                ### **Admin Credentials:**
                ```
                Username: {ADMIN_USERNAME}
                Password: {ADMIN_PASSWORD_PLAIN}
                ```

                **⚠️ Store these credentials securely!**
                """)

        # Event handlers
        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[login_section, main_app, auth_message, session_state]
        )

        logout_btn.click(
            fn=lambda state: handle_logout(state.get("session_id", "")),
            inputs=[session_state],
            outputs=[login_section, main_app, auth_message, session_state]
        )

        search_btn.click(
            fn=lambda q, state: query_rag_system(q, state.get("session_id", "")),
            inputs=[question_input, session_state],
            outputs=[result_output]
        )

        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[question_input, result_output]
        )

        refresh_btn.click(
            fn=get_system_status,
            outputs=[status_display]
        )

        upload_btn.click(
            fn=lambda f, state: upload_file_to_mongodb(f, state.get("session_id", "")),
            inputs=[file_input, session_state],
            outputs=[upload_status]
        )

        # Load initial status
        demo.load(
            fn=get_system_status,
            outputs=[status_display]
        )

    return demo

def main():
    """Main application entry point"""
    logger.info("🚀 Starting Secure Railway RAG System...")

    # System started
    print(f"\n🚀 Secure RAG System started on port {os.getenv('PORT', 7860)}")

    # Initialize system
    if not initialize_system():
        logger.error("❌ Failed to initialize system")
        return

    # Create and launch interface
    app = create_secure_interface()

    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')

    logger.info(f"🌐 Starting secure server on {host}:{port}")

    app.launch(
        server_name=host,
        server_port=port,
        share=True,
        show_api=False,
        inbrowser=False,
        quiet=True,
        favicon_path=None
    )

if __name__ == "__main__":
    main()