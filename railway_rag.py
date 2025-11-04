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
    """Get system status as formatted string"""
    try:
        if not mongodb_rag:
            return """📋 System Status:
❌ MongoDB: Not connected
❌ Embedding Model: Not loaded
📊 Database: No stats available
⚠️ Status: System not initialized"""

        stats = mongodb_rag.get_database_stats()

        return f"""📋 System Status:
✅ MongoDB: Connected
✅ Embedding Model: {'Loaded' if embed_model else 'Not loaded'}
📊 Database: {stats.get('database_name', 'Unknown')}
📄 Documents: {stats.get('documents_count', 0)}
🔍 Embeddings: {stats.get('embeddings_count', 0)}
📋 Metadata: {stats.get('metadata_count', 0)}
✅ Status: Ready for use"""

    except Exception as e:
        return f"""📋 System Status:
❌ MongoDB: Connection failed
❌ Embedding Model: Not available
📊 Database: No access
⚠️ Status: Error - {str(e)}"""

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
        title="Smart RAG Document Assistant",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        css="""
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .gradio-container {
            background: linear-gradient(to bottom right, #f3f4f6, #ffffff);
        }

        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .gr-button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        .gr-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15) !important;
        }

        .gr-textbox {
            border-radius: 8px !important;
            border: 2px solid #e5e7eb !important;
            transition: border-color 0.3s ease !important;
        }

        .gr-textbox:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        """
    ) as demo:
        gr.Markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #2E86AB; font-size: 2.5em; margin-bottom: 10px;">
                📚 Smart RAG Document Assistant
            </h1>
            <p style="font-size: 1.2em; color: #555; margin-top: 0;">
                <strong>AI-Powered Document Search & Analysis System</strong>
            </p>
        </div>

        **🌟 Advanced Features:**
        - 🧠 **AI Search** - Find information with natural language queries
        - 📊 **Vector Database** - MongoDB-powered semantic search
        - 📄 **Document Processing** - Support for PDFs and text files
        - 🚀 **Cloud Ready** - Deploy on Railway with one click
        - 💾 **Persistent Storage** - Your documents are always safe
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🔍 Ask Your Documents")

                question_input = gr.Textbox(
                    label="💬 What would you like to know?",
                    placeholder="Ask me anything about your documents... (e.g., 'What are the main findings?' or 'Explain the key concepts')",
                    lines=3
                )

                with gr.Row():
                    search_btn = gr.Button("🔍 Search Documents", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")

                result_output = gr.Markdown(label="📋 Search Results")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 System Status")

                status_display = gr.Textbox(
                    label="📋 Database Information",
                    value="Loading system status...",
                    interactive=False,
                    lines=8
                )

                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("## 📁 Upload Documents")
            gr.Markdown("Add your PDF and text files to the knowledge base for AI-powered search")

            file_input = gr.File(
                label="📄 Choose File",
                file_types=[".pdf", ".txt", ".md"],
                file_count="single"
            )

            upload_btn = gr.Button("📤 Upload & Process", variant="secondary")
            upload_status = gr.Textbox(
                label="📋 Upload Status",
                interactive=False,
                lines=3
            )

        gr.Markdown("---")

        with gr.Accordion("ℹ️ System Information", open=False):
            gr.Markdown("""
            ### 🛠️ Technology Stack
            - **🗄️ MongoDB Atlas** - Vector database for persistent storage
            - **🧠 Sentence Transformers** - AI text embeddings (384 dimensions)
            - **📊 Cosine Similarity** - Advanced semantic search algorithm
            - **🌐 Gradio** - Modern web interface
            - **🚀 Railway** - Cloud deployment platform

            ### 🎯 Key Features
            - **💾 Persistent Storage** - Documents never disappear
            - **🔍 Semantic Search** - Understands context, not just keywords
            - **📄 Multi-format Support** - PDF, TXT, MD files
            - **⚡ Fast Processing** - Efficient document chunking and indexing
            - **📱 Responsive Design** - Works on all devices

            ### 🚀 Deployment Ready
            - **☁️ Cloud Optimized** - One-click Railway deployment
            - **🔒 Secure** - Environment variable configuration
            - **📈 Scalable** - Auto-scaling based on usage
            - **💚 Health Monitoring** - Built-in health checks
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

    # Configure for Docker/Railway deployment
    share_link = os.getenv('ENV', 'development') == 'production'

    app.launch(
        server_name=host,
        server_port=port,
        share=share_link,
        show_api=False,
        inbrowser=False,
        quiet=True,
        favicon_path=None
    )

if __name__ == "__main__":
    main()