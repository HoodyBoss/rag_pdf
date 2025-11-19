#!/usr/bin/env python3
"""
RAG Core Module for FastAPI Backend
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import hashlib
import shortuuid
from datetime import datetime
import numpy as np

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

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available")

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

    def extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file (.xlsx, .xls)"""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for Excel processing")
            return ""

        try:
            # Read Excel file
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_excel(file_path, engine='xlrd')

            # Convert DataFrame to text
            text_content = ""
            for _, row in df.iterrows():
                row_text = " ".join([str(val) for val in row if pd.notna(val)])
                if row_text.strip():
                    text_content += row_text + "\n"

            return text_content
        except Exception as e:
            logger.error(f"Error extracting text from Excel {file_path}: {e}")
            return ""

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from Image file using OCR if available"""
        if not PIL_AVAILABLE:
            logger.warning("PIL not available for image processing")
            return ""

        try:
            # For now, just return basic info about the image
            # In a real implementation, you might want to use OCR libraries like pytesseract
            with Image.open(file_path) as img:
                text_content = f"[Image: {img.size[0]}x{img.size[1]} pixels]\n"
                text_content += f"Format: {img.format}\n"
                text_content += "Note: OCR processing would be implemented here for text extraction"
                return text_content
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.extract_text_from_excel(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return self.extract_text_from_image(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""

    def chunk_text(self, text: str, source_file: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        แยกข้อความเป็น chunks พร้อมข้อมูล metadata

        Args:
            text: ข้อความทั้งหมด
            source_file: ชื่อไฟล์ต้นทาง
            chunk_size: ขนาดของ chunk
            overlap: ส่วนที่ทับซ้อนระหว่าง chunks

        Returns:
            List[Dict]: list ของ chunks พร้อม metadata
        """
        if not text or len(text.strip()) < 50:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # หาตำแหน่งตัดคำที่เหมาะสม
            if end < len(text):
                # พยายามตัดที่วรรค หรือ จบบรรทัด
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in [' ', '\n', '.', '!', '?']:
                        end = i + 1
                        break

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:  # ต้องมีความยาวอย่างน้อย 50 ตัวอักษร
                chunk_id = f"{source_file}_{start}_{end}"

                chunks.append({
                    "text": chunk_text,
                    "id": chunk_id,
                    "metadata": {
                        "source": source_file,
                        "start": start,
                        "end": end,
                        "file_type": Path(source_file).suffix.lower()
                    }
                })

            start = end - overlap if end - overlap > start else end

        return chunks

    def process_text(self, text: str) -> List[str]:
        """Process text into chunks (legacy method)"""
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

    def process_multiple_files(self, file_paths: List[str], clear_before_upload: bool = False) -> Dict[str, Any]:
        """
        ประมวลผลไฟล์หลายไฟล์ (PDF, TXT, MD, DOCX, Excel, Images)

        Args:
            file_paths: list ของ paths ของไฟล์
            clear_before_upload: ล้างข้อมูลเก่าหรือไม่

        Returns:
            Dict[str, Any]: ผลลัพธ์การประมวลผล
        """
        try:
            if not file_paths:
                return {"success": False, "message": "❌ กรุณาเลือกไฟล์เพื่ออัปโหลด"}

            if not self.collection:
                return {"success": False, "message": "❌ Vector database not available"}

            current_count = self.collection.count()
            logger.info(f"เริ่มประมวลผล {len(file_paths)} ไฟล์")

            # ถ้าเลือกล้างข้อมูลเก่า
            if clear_before_upload:
                logger.info("กำลังล้างข้อมูลเก่าตามคำร้อง...")
                # Clear existing data
                try:
                    # Get all existing IDs
                    existing_data = self.collection.get()
                    if existing_data['ids']:
                        self.collection.delete(ids=existing_data['ids'])
                        logger.info(f"ล้างข้อมูลเก่า {len(existing_data['ids'])} chunks แล้ว")
                except Exception as e:
                    logger.warning(f"ไม่สามารถล้างข้อมูลเก่าได้: {e}")

            total_chunks = 0
            successful_files = []
            failed_files = []
            all_chunks = []

            # ประมวลผลทีละไฟล์
            for file_path in file_paths:
                try:
                    file_name = os.path.basename(file_path)
                    file_ext = Path(file_path).suffix.lower()

                    logger.info(f"#### กำลังประมวลผลไฟล์: {file_name} ({file_ext}) ####")

                    # แยกข้อความจากไฟล์
                    text_content = self.extract_text_from_file(file_path)

                    if not text_content.strip():
                        failed_files.append(f"{file_name}: ไม่สามารถแยกข้อความได้")
                        continue

                    # แยกข้อความเป็น chunks
                    content_chunks = self.chunk_text(text_content, file_name)

                    if not content_chunks:
                        failed_files.append(f"{file_name}: ไม่สามารถแยกเป็น chunks ได้")
                        continue

                    all_chunks.extend(content_chunks)
                    total_chunks += len(content_chunks)
                    successful_files.append(file_name)

                    logger.info(f"✅ {file_name}: แยกได้ {len(content_chunks)} chunks")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    failed_files.append(f"{os.path.basename(file_path)}: {str(e)}")

            # สร้าง embeddings และเพิ่มลง vector database
            if all_chunks:
                try:
                    logger.info(f"กำลังสร้าง embeddings สำหรับ {len(all_chunks)} chunks...")
                    chunk_texts = [chunk["text"] for chunk in all_chunks]
                    chunk_ids = [chunk["id"] for chunk in all_chunks]
                    chunk_metadatas = [chunk["metadata"] for chunk in all_chunks]

                    # Create embeddings
                    embeddings = self.create_embeddings(chunk_texts)

                    # Add to ChromaDB
                    self.collection.add(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        embeddings=embeddings,
                        metadatas=chunk_metadatas
                    )

                    logger.info(f"✅ เพิ่ม {len(all_chunks)} chunks ลง vector database แล้ว")

                except Exception as e:
                    logger.error(f"Error adding chunks to vector database: {e}")
                    return {
                        "success": False,
                        "message": f"❌ ไม่สามารถเพิ่มข้อมูลลง vector database ได้: {str(e)}",
                        "successful_files": successful_files,
                        "failed_files": failed_files
                    }

            # สรุงผลลัพธ์
            message_parts = []
            if successful_files:
                message_parts.append(f"✅ ประมวลผลสำเร็จ {len(successful_files)} ไฟล์")
            if failed_files:
                message_parts.append(f"❌ ล้มเหลว {len(failed_files)} ไฟล์")
            if total_chunks > 0:
                message_parts.append(f"📝 สร้าง {total_chunks} chunks")

            result_message = " | ".join(message_parts)

            return {
                "success": True,
                "message": result_message,
                "total_files": len(file_paths),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_chunks": total_chunks,
                "previous_count": current_count,
                "new_count": self.collection.count() if self.collection else 0
            }

        except Exception as e:
            logger.error(f"Error in process_multiple_files: {e}")
            return {
                "success": False,
                "message": f"❌ เกิดข้อผิดพลาด: {str(e)}",
                "successful_files": [],
                "failed_files": [str(fp) for fp in file_paths]
            }

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
            "docx_available": DOCX_AVAILABLE,
            "pandas_available": PANDAS_AVAILABLE,
            "pil_available": PIL_AVAILABLE
        }

        if self.collection:
            try:
                stats["total_documents"] = self.collection.count()
            except:
                stats["total_documents"] = "Unknown"
        else:
            stats["total_documents"] = "Not available"

        return stats

    def clear_database(self):
        """Clear all documents from vector database"""
        if not self.collection:
            logger.warning("ChromaDB not available - skipping database clear")
            return

        try:
            # Get all documents and delete them
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info("Cleared all documents from vector database")
            else:
                logger.info("No documents to clear in vector database")

        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise

# Global RAG instance
rag_core = None

def get_rag_core() -> RAGCore:
    """Get global RAG core instance"""
    global rag_core
    if rag_core is None:
        rag_core = RAGCore()
    return rag_core