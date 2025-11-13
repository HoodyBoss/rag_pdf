#!/usr/bin/env python3
"""
RAG Core Module for FastAPI Backend
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import shortuuid
from datetime import datetime

# Vector Database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

# Text Processing
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence Transformers not available")

# Thai Text Processing
try:
    from pythainlp.tokenize import word_tokenize
    THAI_NLP_AVAILABLE = True
except ImportError:
    THAI_NLP_AVAILABLE = False
    logging.warning("PyThaiNLP not available")

# PDF Processing
try:
    import pymupdf as fitz
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available")

# Document Processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available")

logger = logging.getLogger(__name__)

class RAGCore:
    """RAG (Retrieval-Augmented Generation) Core System"""

    def __init__(self, chroma_path: str = "./data/chromadb"):
        self.chroma_path = chroma_path
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize ChromaDB
            if CHROMADB_AVAILABLE:
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
                self.collection = self.chroma_client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB initialized successfully")

            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
                logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PYPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")

        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available")

        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError("Unable to decode file with any encoding")
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise

    def process_text(self, text: str) -> List[str]:
        """Process text into chunks"""
        if not text.strip():
            return []

        # Simple chunking strategy
        chunk_size = 1000
        chunk_overlap = 200

        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        if not self.embedding_model:
            # Fallback: create dummy embeddings
            import random
            return [[random.random() for _ in range(384)] for _ in texts]

        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            # Fallback to dummy embeddings
            import random
            return [[random.random() for _ in range(384)] for _ in texts]

    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add document to vector database"""
        if not self.collection:
            logger.warning("ChromaDB not available - skipping document addition")
            return

        try:
            # Process text into chunks
            chunks = self.process_text(text)
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return

            # Create embeddings
            embeddings = self.create_embeddings(chunks)

            # Add to ChromaDB
            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]

            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    **metadata
                }
                for i, chunk in enumerate(chunks)
            ]

            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Added {len(chunks)} chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Error adding document {document_id}: {e}")
            raise

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.collection:
            logger.warning("ChromaDB not available - returning empty results")
            return []

        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])[0]

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "rank": i + 1
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_document_by_id(self, document_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all chunks for a specific document"""
        if not self.collection:
            return []

        try:
            results = self.collection.get(
                where={"document_id": document_id}
            )

            if not results['documents']:
                return []

            formatted_results = []
            for i, (doc, metadata) in enumerate(zip(
                results['documents'],
                results['metadatas']
            )):
                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "chunk_index": i
                })

            # Sort by chunk index
            formatted_results.sort(key=lambda x: x['chunk_index'])

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return []

    def delete_document(self, document_id: str):
        """Delete document from vector database"""
        if not self.collection:
            logger.warning("ChromaDB not available - skipping document deletion")
            return

        try:
            # Get all chunk IDs for this document
            chunk_ids = [f"{document_id}_{i}" for i in range(1000)]  # Reasonable limit

            # Delete chunks
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted document {document_id}")

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            "chromadb_available": CHROMADB_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "thai_nlp_available": THAI_NLP_AVAILABLE,
            "pypdf_available": PYPDF_AVAILABLE,
            "docx_available": DOCX_AVAILABLE
        }

        if self.collection:
            try:
                stats["total_documents"] = self.collection.count()
            except:
                stats["total_documents"] = "Unknown"
        else:
            stats["total_documents"] = "Not available"

        return stats

# Global RAG instance
rag_core = None

def get_rag_core() -> RAGCore:
    """Get global RAG core instance"""
    global rag_core
    if rag_core is None:
        rag_core = RAGCore()
    return rag_core