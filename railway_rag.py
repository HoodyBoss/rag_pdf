#!/usr/bin/env python3
"""
Railway-deployable RAG PDF with MongoDB
Production-ready RAG system for Railway deployment
"""
import os
import logging
import json
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG components
MONGODB_AVAILABLE = False

try:
    from mongodb_rag import MongoDBRAG
    from sentence_transformers import SentenceTransformer
    import requests
    MONGODB_AVAILABLE = True
    logger.info("MongoDB RAG components imported successfully")
except ImportError as e:
    logger.error(f"MongoDB RAG components import failed: {e}")
    logger.error("Please ensure mongodb_rag.py is in the same directory")
    logger.error("Dependencies: pip install pymongo[srv] sentence-transformers")
    MONGODB_AVAILABLE = False

# Global variables
mongodb_rag = None
embed_model = None

def initialize_system():
    """Initialize RAG system components"""
    global mongodb_rag, embed_model

    try:
        # Check if MongoDB RAG is available
        if not MONGODB_AVAILABLE:
            logger.error("MongoDB RAG not available - cannot initialize system")
            return False

        # Initialize MongoDB RAG
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('DATABASE_NAME', 'rag_pdf_railway')

        logger.info(f"Initializing MongoDB RAG...")
        logger.info(f"MongoDB URI: {mongodb_uri}")
        logger.info(f"Database: {db_name}")

        mongodb_rag = MongoDBRAG(mongodb_uri, db_name)

        # Initialize embedding model
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        logger.info("RAG system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        logger.error("Please check:")
        logger.error("1. mongodb_rag.py file exists")
        logger.error("2. MongoDB connection string is correct")
        logger.error("3. Dependencies are installed")
        return False

def query_rag_system(question: str, top_k: int = 5, min_similarity: float = 0.3):
    """
    Query the RAG system

    Args:
        question: User question
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        Formatted response
    """
    try:
        if not mongodb_rag:
            return "RAG system not initialized. Please check system logs and try again."

        logger.info(f"Querying RAG system: '{question}'")

        # Search for similar documents
        results = mongodb_rag.search_similar(question, top_k, min_similarity)

        if not results:
            return """🤔 ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณ

กรุณาลองใหม่:
• ใช้คำศัพท์อื่น
• ตรวจสอบว่ามีเอกสารในระบบหรือไม่
• ลองคำถามที่เกี่ยวข้องโดยตรง

💡 **คำแนะนำ:** หากต้องการเพิ่มเอกสาร กรุณาอัปโหลด PDF หรือไฟล์อื่นๆ เพื่อเพิ่มข้อมูลในระบบ"""

        # Format response
        response = f"""🤔 **คำตอบสำหรับคำถาม: {question}**

**ข้อมูลที่เกี่ยวข้อง ({len(results)} รายการ):**

"""

        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            text = result.get('text', '')
            doc_id = result.get('document_id', '')
            chunk_idx = result.get('chunk_index', 0)

            # Get document metadata
            doc_metadata = mongodb_rag.get_document_metadata(doc_id)
            source_name = doc_metadata.get('source_name', 'Unknown source')
            chunk_metadata = mongodb_rag.get_chunk_metadata(doc_id, chunk_idx)
            page = chunk_metadata.get('page', chunk_idx)

            response += f"""
**{i}.** {source_name} (หน้า {page})
- **ความเกี่ยวข้อง:** {similarity:.2%}
- **เนื้อหา:** {text[:300]}{'...' if len(text) > 300 else ''}

"""

        response += """
---
💡 **หมายเหตุ:** คำตอบนี้สร้างจากการค้นหาข้อมูลที่มีอยู่ในระบบ โดยใช้เทคนิค Semantic Search

📚 **จำนวนเอกสารในระบบ:**
"""

        # List available documents
        documents = mongodb_rag.list_documents()
        for doc in documents:
            chunks = doc.get('total_chunks', 0)
            created = doc.get('created_at', 'Unknown')
            response += f"• {doc.get('source_name', 'Unknown')} ({chunks} chunks)\n"

        return response

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return f"❌ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"

def upload_file_to_mongodb(file):
    """
    Upload and process file for MongoDB RAG system

    Args:
        file: Uploaded file

    Returns:
        Status message
    """
    try:
        if not mongodb_rag:
            return "❌ RAG system not initialized"

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

        return f"""✅ อัปโหลดและประมวลผล {file_name} สำเร็จ!

📊 **รายละเอียด:**
• ชื่อไฟล์: {file_name}
• ขนาดไฟล์: {file_size:,} bytes
• จำนวน chunks: {len(chunks)}
• Document ID: {doc_id}

🗄️ **สถานะฐานข้อมูล MongoDB:**
• Documents: {stats.get('documents_count', 0)}
• Chunks: {stats.get('embeddings_count', 0)}
• Database: {stats.get('database_name', 'Unknown')}

✅ ไฟล์พร้อมใช้งานในระบบ RAG แล้ว!"""

    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        return f"❌ การอัปโหลดไฟล์ล้มเหลว: {str(e)}"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        import fitz  # PyMuPDF
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
    """Create chunks from text"""
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
            return {
                "mongodb_connected": False,
                "embedding_model_loaded": False,
                "database_stats": {},
                "status": "Not initialized"
            }

        stats = mongodb_rag.get_database_stats()

        return {
            "mongodb_connected": True,
            "embedding_model_loaded": embed_model is not None,
            "database_stats": stats,
            "status": "Ready"
        }

    except Exception as e:
        return {
            "mongodb_connected": False,
            "embedding_model_loaded": False,
            "database_stats": {},
            "status": f"Error: {str(e)}"
        }

def health_check():
    """Health check endpoint"""
    try:
        status = get_system_status()
        if status["mongodb_connected"] and status["embedding_model_loaded"]:
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        else:
            return {"status": "unhealthy", "timestamp": datetime.now().isoformat(), "issues": [k for k, v in status.items() if not v]}
    except:
        return {"status": "error", "timestamp": datetime.now().isoformat()}

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(
        title="RAG PDF with MongoDB - Railway Ready",
        theme=gr.themes.Soft(),
        css="""
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .status-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🚀 RAG PDF with MongoDB (Railway Ready)

        **Production-ready RAG system with MongoDB backend**
        - 🗄️ **MongoDB** - Stable vector database for Railway deployment
        - 🔍 **Semantic Search** - Find relevant information using AI
        - 📚 **Document Management** - Upload and search through PDF/text files
        - 🚀 **Railway Ready** - Optimized for cloud deployment
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🔍 Search Documents")

                question_input = gr.Textbox(
                    label="คำถามของคุณ",
                    placeholder="พิมพ์คำถามที่ต้องการค้นหาข้อมูล...",
                    lines=2
                )

                with gr.Row():
                    search_btn = gr.Button("🔍 ค้นหา", variant="primary")
                    clear_btn = gr.Button("🗑️ ล้าง")

                result_output = gr.Markdown(label="ผลการค้นหา")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 สถานะระบบ")

                status_display = gr.JSON(
                    label="System Status",
                    value={},
                    container=True
                )

                refresh_btn = gr.Button("🔄 รีเฟรชสถานะ", size="sm")

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("## 📤 อัปโหลดเอกสาร")
            gr.Markdown("รองรับไฟล์ PDF และข้อความ (.txt, .md) เพื่อเพิ่มลงในระบบค้นหา")

            file_input = gr.File(
                label="เลือกไฟล์",
                file_types=[".pdf", ".txt", ".md"],
                file_count="single"
            )

            upload_btn = gr.Button("📤 อัปโหลดและประมวลผล", variant="secondary")
            upload_status = gr.Textbox(
                label="สถานะการอัปโหลด",
                interactive=False,
                lines=3
            )

        gr.Markdown("---")

        with gr.Accordion("ℹ️ ข้อมูลระบบ", open=False):
            gr.Markdown("""
            ### 🛠️ เทคโนโลยีที่ใช้
            - **MongoDB** - ฐานข้อมูล vector สำหรับเก็บข้อมูล
            - **Sentence Transformers** - โมเดลสำหรับสร้าง text embeddings
            - **Cosine Similarity** - วิธีการคำนวณความใกล้าวข้อง

            ### 🚀 Railway Deployment
            - ✅ MongoDB Atlas addon
            - ✅ Auto-scaling
            - ✅ Health checks
            - ✅ Environment variables

            ### 📁 โครงสร้าง
            - `mongodb_rag.py` - MongoDB RAG system
            - `railway_rag.py` - Railway application
            - `railway.toml` - Railway configuration
            - `.env` - Environment variables
            """)

        # Event handlers
        search_btn.click(
            fn=query_rag_system,
            inputs=[question_input],
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
            fn=upload_file_to_mongodb,
            inputs=[file_input],
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
    logger.info("🚀 Starting Railway RAG PDF Application...")

    # Initialize system
    if not initialize_system():
        logger.error("❌ Failed to initialize system")
        return

    # Create and launch interface
    app = create_interface()

    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')

    logger.info(f"🌐 Starting web server on {host}:{port}")

    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_api=False
    )

if __name__ == "__main__":
    main()