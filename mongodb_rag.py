#!/usr/bin/env python3
"""
MongoDB-based RAG System for Railway Deployment
Replaces ChromaDB with MongoDB for better stability and deployment
"""
import os
import logging
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import Binary, ObjectId
import hashlib

class MongoDBRAG:
    """MongoDB-based RAG System"""

    def __init__(self, connection_string: str = None, db_name: str = "rag_pdf_db"):
        """
        Initialize MongoDB RAG System

        Args:
            connection_string: MongoDB connection string
            db_name: Database name
        """
        self.db_name = db_name
        self.client = None
        self.db = None
        self.embed_model = None
        self.logger = logging.getLogger(__name__)

        # Initialize connection
        self._connect_mongodb(connection_string)
        self._init_embeddings()
        self._create_indexes()

    def _connect_mongodb(self, connection_string: str = None):
        """Connect to MongoDB"""
        try:
            if connection_string is None:
                # Try to get from environment
                connection_string = os.getenv('MONGODB_URI',
                    'mongodb://localhost:27017/')

            self.logger.info(f"🔌 Connecting to MongoDB...")
            self.client = MongoClient(connection_string,
                                     serverSelectionTimeoutMS=5000,
                                     connectTimeoutMS=5000)

            # Test connection
            self.client.admin.command('ping')

            self.db = self.client[self.db_name]
            self.logger.info(f"✅ Connected to MongoDB: {self.db_name}")

        except ConnectionFailure as e:
            self.logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ MongoDB error: {e}")
            raise

    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            self.logger.info("🧠 Loading embedding model...")
            self.embed_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.logger.info("✅ Embedding model loaded")
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def _create_indexes(self):
        """Create MongoDB indexes for better performance"""
        try:
            # Documents collection indexes
            self.db.documents.create_index([
                ("source", 1),
                ("chunk_id", 1)
            ])

            self.db.documents.create_index([
                ("embedding", "cosmosSearch")
            ])

            # Embeddings collection indexes
            self.db.embeddings.create_index([
                ("document_id", 1),
                ("chunk_index", 1)
            ])

            # Metadata collection indexes
            self.db.metadata.create_index([
                ("document_id", 1),
                ("key", 1)
            ])

            self.logger.info("✅ MongoDB indexes created")

        except Exception as e:
            self.logger.warning(f"⚠️ Index creation failed: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embed_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return []

    def store_document(self, chunks: List[Dict], source_name: str) -> str:
        """
        Store document chunks in MongoDB

        Args:
            chunks: List of text chunks with metadata
            source_name: Name of the source document

        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = str(ObjectId())

            self.logger.info(f"📄 Storing document: {source_name} ({len(chunks)} chunks)")

            # Store document metadata
            document = {
                "_id": doc_id,
                "source_name": source_name,
                "total_chunks": len(chunks),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "chunk_size": len(chunks[0].get("text", "")) if chunks else 0
            }

            self.db.documents.insert_one(document)

            # Store chunks with embeddings
            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})

                # Generate embedding
                embedding = self.generate_embedding(text)

                if embedding:
                    # Store embedding
                    embedding_doc = {
                        "document_id": doc_id,
                        "chunk_index": i,
                        "text": text,
                        "embedding": embedding,
                        "created_at": datetime.now()
                    }

                    self.db.embeddings.insert_one(embedding_doc)

                    # Store metadata separately
                    for key, value in metadata.items():
                        metadata_doc = {
                            "document_id": doc_id,
                            "chunk_index": i,
                            "key": str(key),
                            "value": str(value) if not isinstance(value, (list, dict)) else json.dumps(value),
                            "created_at": datetime.now()
                        }
                        self.db.metadata.insert_one(metadata_doc)

            self.logger.info(f"✅ Document stored successfully: {doc_id}")
            return doc_id

        except Exception as e:
            self.logger.error(f"❌ Document storage failed: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5, min_similarity: float = 0.5) -> List[Dict]:
        """
        Search for similar documents using vector similarity

        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            if not query_embedding:
                return []

            self.logger.info(f"🔍 Searching for: '{query}' (top_k: {top_k})")

            # Search using vector similarity (MongoDB Atlas Search)
            # Fallback to manual cosine similarity if Atlas Search not available
            results = self._vector_search(query_embedding, top_k, min_similarity)

            self.logger.info(f"✅ Found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []

    def _vector_search(self, query_embedding: List[float], top_k: int, min_similarity: float) -> List[Dict]:
        """
        Manual vector search using cosine similarity
        """
        try:
            # Get all embeddings (for smaller datasets)
            embeddings_cursor = self.db.embeddings.find({})

            results = []
            for doc in embeddings_cursor:
                stored_embedding = doc.get("embedding", [])

                if stored_embedding:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)

                    if similarity >= min_similarity:
                        results.append({
                            "document_id": doc["document_id"],
                            "chunk_index": doc["chunk_index"],
                            "text": doc["text"],
                            "similarity": similarity,
                            "created_at": doc["created_at"]
                        })

            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]

        except Exception as e:
            self.logger.error(f"❌ Vector search failed: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            self.logger.error(f"❌ Cosine similarity calculation failed: {e}")
            return 0.0

    def get_document_metadata(self, doc_id: str) -> Dict:
        """Get document metadata"""
        try:
            doc = self.db.documents.find_one({"_id": doc_id})
            if doc:
                # Convert ObjectId to string for JSON serialization
                doc["_id"] = str(doc["_id"])
                return doc
            return {}
        except Exception as e:
            self.logger.error(f"❌ Failed to get document metadata: {e}")
            return {}

    def get_chunk_metadata(self, doc_id: str, chunk_index: int) -> Dict:
        """Get metadata for a specific chunk"""
        try:
            metadata_cursor = self.db.metadata.find({
                "document_id": doc_id,
                "chunk_index": chunk_index
            })

            metadata = {}
            for doc in metadata_cursor:
                # Try to parse JSON values
                try:
                    value = json.loads(doc["value"])
                except:
                    value = doc["value"]
                metadata[doc["key"]] = value

            return metadata
        except Exception as e:
            self.logger.error(f"❌ Failed to get chunk metadata: {e}")
            return {}

    def list_documents(self) -> List[Dict]:
        """List all documents"""
        try:
            documents = list(self.db.documents.find({}))

            # Convert ObjectIds to strings
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return documents
        except Exception as e:
            self.logger.error(f"❌ Failed to list documents: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Delete embeddings
            self.db.embeddings.delete_many({"document_id": doc_id})

            # Delete metadata
            self.db.metadata.delete_many({"document_id": doc_id})

            # Delete document
            result = self.db.documents.delete_one({"_id": doc_id})

            if result.deleted_count > 0:
                self.logger.info(f"✅ Document deleted: {doc_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Document not found: {doc_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Document deletion failed: {e}")
            return False

    def clear_all_documents(self) -> bool:
        """Clear all documents"""
        try:
            self.db.embeddings.delete_many({})
            self.db.metadata.delete_many({})
            self.db.documents.delete_many({})

            self.logger.info("✅ All documents cleared")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to clear documents: {e}")
            return False

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {
                "documents_count": self.db.documents.count_documents({}),
                "embeddings_count": self.db.embeddings.count_documents({}),
                "metadata_count": self.db.metadata.count_documents({}),
                "database_name": self.db_name,
                "connection_status": "connected"
            }
            return stats
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

    def backup_data(self, backup_path: str = None) -> str:
        """Backup data to JSON files"""
        try:
            if backup_path is None:
                backup_path = f"./backups/mongodb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            os.makedirs(backup_path, exist_ok=True)

            # Backup documents
            documents = list(self.db.documents.find({}))
            with open(f"{backup_path}/documents.json", 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, default=str)

            # Backup embeddings (without embeddings for space)
            embeddings = list(self.db.embeddings.find({}, {"embedding": 0}))
            with open(f"{backup_path}/embeddings.json", 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2, default=str)

            # Backup metadata
            metadata = list(self.db.metadata.find({}))
            with open(f"{backup_path}/metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"✅ Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            raise

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.logger.info("🔌 MongoDB connection closed")

# Global MongoDB RAG instance
mongodb_rag = None

def init_mongodb_rag(connection_string: str = None, db_name: str = "rag_pdf_db") -> MongoDBRAG:
    """Initialize MongoDB RAG system"""
    global mongodb_rag
    try:
        mongodb_rag = MongoDBRAG(connection_string, db_name)
        logging.info("✅ MongoDB RAG system initialized")
        return mongodb_rag
    except Exception as e:
        logging.error(f"❌ Failed to initialize MongoDB RAG: {e}")
        raise

def get_mongodb_rag() -> MongoDBRAG:
    """Get MongoDB RAG instance"""
    global mongodb_rag
    if mongodb_rag is None:
        raise Exception("MongoDB RAG not initialized. Call init_mongodb_rag() first.")
    return mongodb_rag

if __name__ == "__main__":
    # Test MongoDB RAG system
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=== MongoDB RAG System Test ===")

    try:
        # Initialize
        rag = init_mongodb_rag()

        # Test embedding
        test_text = "นี่คือข้อความทดสอบสำหรับระบบ RAG"
        embedding = rag.generate_embedding(test_text)
        print(f"✅ Embedding generated: {len(embedding)} dimensions")

        # Test search
        results = rag.search_similar("ทดสอบ", top_k=3)
        print(f"✅ Search completed: {len(results)} results")

        # Get stats
        stats = rag.get_database_stats()
        print(f"✅ Database stats: {stats}")

        print("🎉 MongoDB RAG system test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")