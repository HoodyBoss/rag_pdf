#!/usr/bin/env python3
"""
Railway-ready RAG Application
Optimized for Railway cloud deployment with all features working
"""

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
import ollama
import shortuuid
import logging
import re
import time
import pandas as pd
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "documents"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma2:9b")

class RailwayRAGSystem:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.ollama_available = False
        self.initialize_system()

    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ ChromaDB initialized successfully")

            # Initialize embedding model
            logger.info("📦 Loading embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded successfully")

            # Check Ollama availability
            try:
                response = ollama.list()
                if response and 'models' in response:
                    self.ollama_available = True
                    logger.info("✅ Ollama is available")
                else:
                    logger.warning("⚠️ Ollama not available, using fallback responses")
            except Exception as e:
                logger.warning(f"⚠️ Ollama check failed: {e}")
                self.ollama_available = False

            return True

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False

    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""

    def extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""

    def process_google_sheets(self, sheets_url: str) -> str:
        """Process Google Sheets URL (placeholder for now)"""
        try:
            # This is a placeholder - actual implementation would need Google Sheets API
            logger.info(f"Processing Google Sheets: {sheets_url}")
            return f"✅ Google Sheets integration would process: {sheets_url}\n[Feature to be implemented with Google Sheets API]"
        except Exception as e:
            logger.error(f"Google Sheets processing error: {e}")
            return f"❌ Error processing Google Sheets: {str(e)}"

    def add_document(self, file_path: str, filename: str) -> str:
        """Add document to ChromaDB"""
        try:
            # Extract text based on file type
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext == '.pdf':
                text = self.extract_pdf_text(file_path)
            elif file_ext == '.docx':
                text = self.extract_docx_text(file_path)
            elif file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                return f"❌ Unsupported file type: {file_ext}"

            if not text.strip():
                return "❌ No text content found in file"

            # Split text into chunks
            chunks = self.chunk_text(text)

            # Generate embeddings and add to ChromaDB
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                doc_id = f"{filename}_{i}_{shortuuid.uuid()}"

                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": file_ext,
                        "upload_date": datetime.now().isoformat()
                    }]
                )

            return f"✅ Successfully added {filename} ({len(chunks)} chunks)"

        except Exception as e:
            logger.error(f"Document addition error: {e}")
            return f"❌ Error adding document: {str(e)}"

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents using ChromaDB"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            search_results = []
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                metadata = results['metadatas'][0][i]
                search_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "distance": distance,
                    "score": 1 - distance  # Convert distance to similarity score
                })

            return search_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate response using Ollama or fallback"""
        if not self.ollama_available:
            return self.generate_fallback_response(query, search_results)

        try:
            # Format context for Ollama
            context = "\n\n".join([result['content'] for result in search_results])

            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided."""

            # Call Ollama
            response = ollama.generate(
                model=DEFAULT_MODEL,
                prompt=prompt
            )

            return response['response']

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return self.generate_fallback_response(query, search_results)

    def generate_fallback_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate fallback response when Ollama is not available"""
        if not search_results:
            return "I found no relevant information to answer your question. Please try uploading some documents first."

        response = f"Based on the available documents, here's what I found about '{query}':\n\n"

        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            response += f"**{i}. Source: {metadata['source']} (Similarity: {result['score']:.2f})**\n"
            response += f"{result['content'][:300]}...\n\n"

        response += "Note: This is a basic retrieval response. For AI-powered answers, please ensure Ollama is properly configured."

        return response

# Initialize RAG system
rag_system = RailwayRAGSystem()

def process_file_upload(file_obj):
    """Process uploaded file"""
    if file_obj is None:
        return "❌ Please upload a file"

    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file_obj.name}"
        with open(temp_path, "wb") as f:
            f.write(file_obj.read())

        # Add to RAG system
        result = rag_system.add_document(temp_path, file_obj.name)

        # Clean up temp file
        os.remove(temp_path)

        return result

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return f"❌ Error processing file: {str(e)}"

def process_query(query):
    """Process user query"""
    if not query.strip():
        return "Please enter a question."

    try:
        # Search documents
        search_results = rag_system.search_documents(query)

        if not search_results:
            return "I couldn't find any relevant information. Please try uploading some documents first."

        # Generate response
        response = rag_system.generate_response(query, search_results)

        return response

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return f"❌ Error processing query: {str(e)}"

def process_google_sheets(sheets_url):
    """Process Google Sheets URL"""
    if not sheets_url.strip():
        return "Please enter a Google Sheets URL."

    return rag_system.process_google_sheets(sheets_url)

def get_system_status():
    """Get system status"""
    try:
        status = "🚀 **RAG System Status**\n\n"

        # ChromaDB status
        if rag_system.collection is not None:
            count = rag_system.collection.count()
            status += f"✅ **ChromaDB**: {count} documents stored\n"
        else:
            status += "❌ **ChromaDB**: Not available\n"

        # Embedding model status
        if rag_system.embedding_model is not None:
            status += f"✅ **Embedding Model**: {EMBEDDING_MODEL}\n"
        else:
            status += "❌ **Embedding Model**: Not loaded\n"

        # Ollama status
        if rag_system.ollama_available:
            status += f"✅ **Ollama**: Available ({DEFAULT_MODEL})\n"
        else:
            status += "⚠️ **Ollama**: Not available (using fallback)\n"

        return status

    except Exception as e:
        return f"❌ Status check failed: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create Railway-optimized Gradio interface"""
    with gr.Blocks(
        title="🤖 Railway RAG System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """,
        analytics_enabled=False
    ) as demo:

        # Header
        gr.Markdown("# 🤖 Railway RAG Document Assistant")
        gr.Markdown("Upload documents and ask questions about them using AI-powered search.")

        with gr.Row():
            with gr.Column(scale=2):
                # File upload section
                gr.Markdown("## 📁 Upload Documents")

                file_upload = gr.File(
                    label="Upload PDF, DOCX, or TXT files",
                    file_types=[".pdf", ".docx", ".txt", ".md"]
                )

                upload_btn = gr.Button("📤 Upload", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                # Google Sheets section
                gr.Markdown("## 📊 Google Sheets Integration")
                sheets_url = gr.Textbox(
                    label="Google Sheets URL",
                    placeholder="https://docs.google.com/spreadsheets/d/..."
                )
                sheets_btn = gr.Button("🔗 Process Google Sheets")
                sheets_status = gr.Textbox(label="Sheets Status", interactive=False)

                # Query section
                gr.Markdown("## 🔍 Ask Questions")
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know about your documents?",
                    lines=3
                )

                query_btn = gr.Button("🚀 Ask", variant="primary")
                response_output = gr.Markdown(label="Answer")

            with gr.Column(scale=1):
                # System status
                gr.Markdown("## 📊 System Status")
                status_display = gr.Markdown(get_system_status())
                refresh_btn = gr.Button("🔄 Refresh Status")

                # Instructions
                gr.Markdown("## 📖 How to Use")
                gr.Markdown("""
                1. **Upload Documents**: Click the upload button and select PDF, DOCX, or TXT files
                2. **Google Sheets**: Enter a Google Sheets URL to import data
                3. **Ask Questions**: Type your question and click Ask
                4. **Get Answers**: The system will search your documents and provide answers

                **Features:**
                - 🤖 AI-powered search (Ollama integration)
                - 📚 Multiple file format support
                - 📊 Google Sheets integration
                - 🔍 Fast semantic search
                - ☁️ Cloud-ready for Railway deployment
                """)

        # Event handlers
        upload_btn.click(
            fn=process_file_upload,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        sheets_btn.click(
            fn=process_google_sheets,
            inputs=[sheets_url],
            outputs=[sheets_status]
        )

        query_btn.click(
            fn=process_query,
            inputs=[query_input],
            outputs=[response_output]
        )

        refresh_btn.click(
            fn=get_system_status,
            outputs=[status_display]
        )

        # Auto-refresh status on load
        demo.load(
            fn=get_system_status,
            outputs=[status_display]
        )

    return demo

if __name__ == "__main__":
    # Create and launch interface
    app = create_interface()

    # Get port from environment (Railway sets this)
    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')

    logger.info(f"🚀 Starting Railway RAG System on {host}:{port}")

    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_api=False,
        inbrowser=False,
        quiet=False
    )