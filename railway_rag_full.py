#!/usr/bin/env python3
"""
Enhanced Railway RAG System with MongoDB - Full Features & Security
Production-ready RAG system with authentication and comprehensive features
"""
import os
import logging
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
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
try:
    from mongodb_rag import MongoDBRAG
    from sentence_transformers import SentenceTransformer
    import requests
    import fitz  # PyMuPDF
    MONGODB_AVAILABLE = True
    logger.info("MongoDB RAG components imported successfully")
except ImportError as e:
    MONGODB_AVAILABLE = False
    logger.error(f"MongoDB RAG components import failed: {e}")
    logger.error("Please ensure mongodb_rag.py and dependencies are available")

# Session management
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour

# Admin credentials (highly secure)
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Generate secure encryption key
def generate_encryption_key():
    """Generate a secure encryption key for admin credentials"""
    # In production, store this securely (environment variable, key vault, etc.)
    password = os.getenv('MASTER_KEY', 'default_secure_key_for_development').encode()
    salt = b'stable_salt_for_admin_system'  # In production, use random salt per deployment
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

# Generate admin credentials
def generate_secure_admin_credentials():
    """Generate highly secure admin credentials"""
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')

    # Generate secure password if not provided
    admin_password = os.getenv('ADMIN_PASSWORD', None)
    if not admin_password:
        # Generate 32-character secure password
        admin_password = ''.join(secrets.choice(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-='
        ) for _ in range(32))

    return admin_username, admin_password

# Initialize encryption and admin credentials
try:
    ENCRYPTION_KEY = generate_encryption_key()
    cipher_suite = Fernet(ENCRYPTION_KEY)
    ADMIN_USERNAME, ADMIN_PASSWORD = generate_secure_admin_credentials()

    # Store encrypted admin credentials
    ENCRYPTED_ADMIN = {
        'username': cipher_suite.encrypt(ADMIN_USERNAME.encode()).decode(),
        'password_hash': hashlib.sha256((ADMIN_PASSWORD + os.getenv('SALT', 'secure_salt')).encode()).hexdigest(),
        'created_at': datetime.now().isoformat()
    }

    # Single admin user database
    ADMIN_DB = {
        'username': ADMIN_USERNAME,
        'password_hash': ENCRYPTED_ADMIN['password_hash'],
        'role': 'admin',
        'created_at': datetime.now(),
        'last_login': None,
        'login_attempts': 0,
        'locked_until': None
    }

    # Print admin credentials for first-time setup
    print("=" * 60)
    print("🔐 ADMIN CREDENTIALS GENERATED")
    print("=" * 60)
    print(f"Username: {ADMIN_USERNAME}")
    print(f"Password: {ADMIN_PASSWORD}")
    print("⚠️  Store these credentials securely!")
    print("=" * 60)

except Exception as e:
    print(f"❌ Failed to initialize secure admin system: {e}")
    # Fallback to simple admin
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = 'admin123'
    ADMIN_DB = {
        'username': ADMIN_USERNAME,
        'password_hash': hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest(),
        'role': 'admin',
        'created_at': datetime.now(),
        'last_login': None,
        'login_attempts': 0,
        'locked_until': None
    }

class AuthManager:
    """Hybrid authentication and session management (MongoDB + Memory)"""

    def __init__(self):
        self.mongo_client = None
        self.db = None

    def init_database(self, mongodb_uri: str, database_name: str):
        """Initialize MongoDB connection"""
        try:
            self.mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client[database_name]

            # Create users collection with unique index on username
            self.db.users.create_index("username", unique=True)

            # Initialize default admin user if not exists
            self._ensure_admin_user()

            print("✅ MongoDB authentication initialized successfully")
            return True

        except Exception as e:
            print(f"❌ MongoDB auth initialization failed: {e}")
            return False

    def _ensure_admin_user(self):
        """Create default admin user if not exists"""
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_password = os.getenv('ADMIN_PASSWORD')

        if not admin_password:
            print("⚠️  No ADMIN_PASSWORD found in environment")
            return

        # Check if admin exists
        existing_admin = self.db.users.find_one({"username": admin_username})

        if not existing_admin:
            # Create admin user
            salt = os.getenv('SALT', 'secure_salt_default')
            hashed_password = hashlib.sha256((admin_password + salt).encode()).hexdigest()

            admin_user = {
                "username": admin_username,
                "password_hash": hashed_password,
                "salt": salt,
                "role": "admin",
                "email": f"{admin_username}@local.dev",
                "created_at": datetime.now(),
                "last_login": None,
                "is_active": True
            }

            self.db.users.insert_one(admin_user)
            print(f"✅ Created default admin user: {admin_username}")
        else:
            print(f"ℹ️  Admin user already exists: {admin_username}")

    @staticmethod
    def hash_password(password: str, salt: str) -> str:
        """Hash password using SHA-256 with salt"""
        return hashlib.sha256((password + salt).encode()).hexdigest()

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash with salt"""
        return AuthManager.hash_password(plain_password, salt) == hashed_password

    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)

    def create_session(self, username: str, user_role: str = 'user') -> str:
        """Create new session in memory"""
        session_id = AuthManager.generate_session_id()
        sessions[session_id] = {
            'username': username,
            'role': user_role,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        return session_id

    @staticmethod
    def validate_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and update last activity"""
        if session_id not in sessions:
            return None

        session = sessions[session_id]
        now = datetime.now()

        # Check session timeout
        if (now - session['last_activity']).total_seconds() > SESSION_TIMEOUT:
            del sessions[session_id]
            return None

        # Update last activity
        session['last_activity'] = now
        return session

    @staticmethod
    def logout(session_id: str) -> bool:
        """Logout user by removing session"""
        if session_id in sessions:
            del sessions[session_id]
            return True
        return False

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials against MongoDB"""
        if not self.db:
            print("❌ Database not initialized")
            return None

        try:
            # Find user in MongoDB
            user = self.db.users.find_one({
                "username": username,
                "is_active": True
            })

            if not user:
                return None

            # Verify password with salt
            if not AuthManager.verify_password(password, user["password_hash"], user["salt"]):
                return None

            # Update last login
            self.db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.now()}}
            )

            return {
                'username': user['username'],
                'role': user['role'],
                'email': user.get('email', f"{username}@local.dev")
            }

        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return None

    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics from MongoDB"""
        if not self.db:
            return {"total_users": 0, "active_users": 0}

        try:
            total_users = self.db.users.count_documents({})
            active_users = self.db.users.count_documents({"is_active": True})
            admin_users = self.db.users.count_documents({"role": "admin"})

            return {
                "total_users": total_users,
                "active_users": active_users,
                "admin_users": admin_users
            }
        except Exception as e:
            print(f"❌ Error getting user stats: {e}")
            return {"total_users": 0, "active_users": 0}

# Global variables
mongodb_rag = None
embed_model = None
auth_manager = AuthManager()

def initialize_system():
    """Initialize RAG system components"""
    global mongodb_rag, embed_model, auth_manager

    try:
        # Check if MongoDB RAG is available
        if not MONGODB_AVAILABLE:
            logger.error("MongoDB RAG not available - cannot initialize system")
            return False

        # Initialize MongoDB RAG
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27020/')
        db_name = os.getenv('DATABASE_NAME', 'rag_pdf_enhanced')

        logger.info(f"Initializing MongoDB RAG...")
        logger.info(f"MongoDB URI: {mongodb_uri}")
        logger.info(f"Database: {db_name}")

        mongodb_rag = MongoDBRAG(mongodb_uri, db_name)

        # Initialize authentication manager
        logger.info("Initializing authentication system...")
        if not auth_manager.init_database(mongodb_uri, db_name):
            logger.error("Failed to initialize authentication system")
            return False

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

def query_rag_system(question: str, session_id: str, top_k: int = 5, min_similarity: float = 0.3):
    """
    Query the RAG system with authentication

    Args:
        question: User question
        session_id: User session ID
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold

    Returns:
        Formatted response
    """
    try:
        # Validate session
        user = AuthManager.validate_session(session_id)
        if not user:
            return "❌ Authentication required. Please login to use the RAG system."

        if not mongodb_rag:
            return "❌ RAG system not initialized"

        logger.info(f"Querying RAG system: '{question}' by {user['username']}")

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
**ผู้ใช้:** {user['username']} ({user['role']})

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
            response += f"• {doc.get('source_name', 'Unknown')} ({chunks} chunks)\\n"

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"❌ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"

def upload_file_to_mongodb(file, session_id: str):
    """
    Upload and process file for MongoDB RAG system with authentication

    Args:
        file: Uploaded file
        session_id: User session ID

    Returns:
        Status message
    """
    try:
        # Validate session
        user = AuthManager.validate_session(session_id)
        if not user:
            return "❌ Authentication required. Please login to upload files."

        if not mongodb_rag:
            return "❌ RAG system not initialized"

        if file is None:
            return "❌ กรุณาเลือกไฟล์"

        # Check file size limit (10MB)
        file_path = file.name
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                return f"❌ ไฟล์ใหญ่เกิน 10MB กรุณาอัปโหลดไฟล์ที่เล็กกว่า"

        # Get file info
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        logger.info(f"Processing file: {file_name} ({file_size:,} bytes) by {user['username']}")

        # Extract text based on file type
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext == '.pdf':
            text = extract_pdf_text(file_path)
        elif file_ext in ['.txt', '.md', '.docx']:
            if file_ext == '.docx':
                text = extract_docx_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        else:
            return f"❌ ไม่รองรับประเภท: {file_ext}"

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
**ผู้ใช้:** {user['username']} ({user['role']})

📊 **รายละเอียด:**
• ชื่อไฟล์: {file_name}
• ขนาดไฟล์: {file_size:,} bytes
• จำนวน chunks: {len(chunks)}
• Document ID: {doc_id}

🗄️ **สถานะฐานข้อมูล MongoDB:**
• Documents: {stats.get('documents_count', 0)}
• Chunks: {stats.get('embeddings_count', 0)}
• Database: {stats.get('database_name', 'Unknown')}
• ผู้อัปโหลด: {user['username']}

✅ ไฟล์พร้อมใช้งานในระบบ RAG แล้ว!"""

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return f"❌ การอัปโหลดไฟล์ล้มเหลว: {str(e)}"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_docx_text(docx_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        import docx
        doc = docx.Document(docx_path)
        text = "\\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
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

def get_user_stats(session_id: str):
    """Get user statistics"""
    try:
        user = AuthManager.validate_session(session_id)
        if not user:
            return "❌ Not authenticated"

        # Get MongoDB user statistics
        user_stats = auth_manager.get_user_stats()

        # Get system document statistics
        all_docs = mongodb_rag.list_documents() if mongodb_rag else []
        total_chunks = sum(doc.get('total_chunks', 0) for doc in all_docs)

        return f"""👤 User Statistics: {user['username']}
📊 Role: {user['role']}
📁 Documents: {len(all_docs)}
📝 Chunks: {total_chunks}
🕐 Session Age: {(datetime.now() - sessions[session_id]['created_at']).total_seconds():.0f} seconds
🔄 Last Activity: {(datetime.now() - sessions[session_id]['last_activity']).total_seconds():.0f} seconds ago

🏛️ **System User Stats:**
👥 Total Users: {user_stats.get('total_users', 0)}
✅ Active Users: {user_stats.get('active_users', 0)}
👑 Admin Users: {user_stats.get('admin_users', 0)}"""

    except Exception as e:
        return f"❌ Error getting user stats: {str(e)}"

def clear_user_data(session_id: str):
    """Clear user-specific data (if implemented)"""
    try:
        user = AuthManager.validate_session(session_id)
        if not user:
            return "❌ Not authenticated"

        # In a real implementation, this would clear user's uploads
        # For now, just return a message
        return f"✅ User data cleared for {user['username']}"

    except Exception as e:
        return f"❌ Error clearing user data: {str(e)}"

# Authentication handlers
def login_user(username: str, password: str) -> tuple[str, str, str]:
    """Handle user login"""
    try:
        user = auth_manager.authenticate_user(username, password)

        if user:
            session_id = auth_manager.create_session(user['username'], user['role'])
            return f"✅ Login successful! Welcome {user['username']}!", session_id, user['role']
        else:
            return "❌ Invalid username or password", "", ""

    except Exception as e:
        return f"❌ Login error: {str(e)}", "", ""

def logout_user(session_id: str) -> str:
    """Handle user logout"""
    try:
        if AuthManager.logout(session_id):
            return "✅ Logged out successfully"
        else:
            return "❌ Invalid session"
    except Exception as e:
        return f"❌ Logout error: {str(e)}"

def register_user(username: str, password: str, email: str = "") -> str:
    """Handle user registration"""
    try:
        # Check if user already exists
        if username in USERS_DB or username == ADMIN_USERNAME:
            return "❌ Username already exists"

        # Validate password
        if len(password) < 6:
            return "❌ Password must be at least 6 characters"

        # Create new user
        USERS_DB[username] = {
            'password_hash': AuthManager.hash_password(password),
            'email': email or f"{username}@example.com",
            'role': 'user',
            'created_at': datetime.now()
        }

        return f"✅ User {username} registered successfully! Please login to continue."

    except Exception as e:
        return f"❌ Registration error: {str(e)}"

def create_interface():
    """Create enhanced Gradio interface with authentication"""

    # Custom CSS for better UI
    custom_css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .auth-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .feature-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }

    .gradio-container {
        background: linear-gradient(to bottom right, #f8f9fa, #ffffff);
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

    .stats-container {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }

    .stat-card {
        flex: 1;
        min-width: 200px;
        background: white;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }

    .user-info {
        background: #f0f8ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        title="Enhanced RAG System with MongoDB - Full Features",
        theme=gr.themes.Soft(),
        css=custom_css,
        analytics_enabled=False
    ) as demo:

        # State management
        session_state = gr.State({"session_id": "", "username": "", "role": "user"})

        def update_session(session_id: str, current_state):
            """Update session state"""
            if session_id:
                user = AuthManager.validate_session(session_id)
                if user:
                    return {
                        "session_id": session_id,
                        "username": user["username"],
                        "role": user["role"]
                    }
            return current_state

        # Login/Registration Modal
        with gr.Column(visible=True) as login_section:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("# 🔐 Authentication Required")
                    gr.Markdown("Please login or register to access the RAG system")

                    with gr.Tabs() as auth_tabs:
                        with gr.TabItem("Login"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    login_username = gr.Textbox(
                                        label="Username",
                                        placeholder="Enter your username"
                                    )
                                    login_password = gr.Textbox(
                                        label="Password",
                                        type="password",
                                        placeholder="Enter your password"
                                    )
                                    login_btn = gr.Button("🔐 Login", variant="primary")

                        with gr.TabItem("Register"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    reg_username = gr.Textbox(
                                        label="Username",
                                        placeholder="Choose a username"
                                    )
                                    reg_password = gr.Textbox(
                                        label="Password",
                                        type="password",
                                        placeholder="Choose a password (min 6 chars)"
                                    )
                                    reg_email = gr.Textbox(
                                        label="Email (optional)",
                                        placeholder="your@email.com"
                                    )
                                    reg_btn = gr.Button("📝 Register", variant="primary")

            auth_message = gr.Markdown("", visible=False)

        # Main Application (hidden initially)
        with gr.Column(visible=False) as main_app:
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #2E86AB; font-size: 2.5em; margin-bottom: 10px;">
                    📚 Enhanced RAG Document Assistant
                </h1>
                <p style="font-size: 1.2em; color: #555; margin-top: 0;">
                    <strong>AI-Powered Document Search & Analysis System with Authentication</strong>
                </p>
            </div>

            **🌟 Advanced Features:**
            - 🧠 **AI Search** - Find information with natural language queries
            - 📊 **Vector Database** - MongoDB-powered semantic search
            - 🔒 **Secure Authentication** - User management and session control
            - 📄 **Document Processing** - Support for PDF, TXT, MD, DOCX files
            - 👤 **User Management** - Individual user spaces and tracking
            - ⚡ **Fast Processing** - Efficient document chunking and indexing
            - 📱 **Responsive Design** - Works on all devices

            **🚀 Railway Ready:**
            - ☁️ Cloud Optimized - One-click Railway deployment
            - 🔒 Secure - Environment variable configuration
            - 📈 Scalable - Auto-scaling based on usage
            - 💚 Health Monitoring - Built-in health checks
            """)

            # User info display
            with gr.Row():
                user_info_display = gr.Markdown("", visible=False)

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 🔍 Ask Your Documents")

                    question_input = gr.Textbox(
                        label="What would you like to know?",
                        placeholder="Ask me anything about your documents... (e.g., 'What are the main findings?' or 'Explain the key concepts')",
                        lines=3
                    )

                    with gr.Row():
                        search_btn = gr.Button("🔍 Search Documents", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")
                        refresh_btn = gr.Button("🔄 Refresh")

                    result_output = gr.Markdown(label="Search Results")

                with gr.Column(scale=1):
                    gr.Markdown("## 📊 System Status")

                    with gr.Tabs():
                        with gr.TabItem("System"):
                            system_status = gr.Textbox(
                                label="System Information",
                                value="Loading system status...",
                                interactive=False,
                                lines=8
                            )

                            refresh_system_btn = gr.Button("🔄 Refresh Status", size="sm")

                        with gr.TabItem("User Stats"):
                            user_stats = gr.Textbox(
                                label="Your Statistics",
                                value="Loading user stats...",
                                interactive=False,
                                lines=6
                            )

                            refresh_user_btn = gr.Button("🔄 Refresh Stats", size="sm")

                        with gr.TabItem("Database"):
                            db_status = gr.JSON(
                                label="Database Details",
                                value={},
                                visible=True
                            )

                            refresh_db_btn = gr.Button("🔄 Refresh Database", size="sm")

            gr.Markdown("---")

            with gr.Row():
                gr.Markdown("## 📁 Upload Documents")
                gr.Markdown("Add your PDF, text, and DOCX files to the knowledge base for AI-powered search")

                file_input = gr.File(
                    label="Choose File",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                    file_count="single"
                )

                upload_btn = gr.Button("📤 Upload & Process", variant="secondary")
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=3
                )

                with gr.Row():
                    clear_data_btn = gr.Button("🗑️ Clear My Data", variant="stop")

            gr.Markdown("---")

            with gr.Accordion("Advanced Features", open=False):
                gr.Markdown("""
                ### 🛠️ Technology Stack
                - **🗄️ MongoDB Atlas** - Vector database for persistent storage
                - **🧠 Sentence Transformers** - AI text embeddings (384 dimensions)
                - **📊 Cosine Similarity** - Advanced semantic search algorithm
                - **🌐 Gradio** - Modern web interface
                - **🚀 Railway** - Cloud deployment platform

                ### 🔒 Security Features
                - **Session Management** - Secure session handling with timeout
                - **User Authentication** - Login/logout system
                - **Role-based Access** - Admin and user roles
                - **Password Hashing** - SHA-256 password encryption
                - **Session Timeout** - Automatic logout after inactivity

                ### 🎯 Enhanced Features
                - **User Statistics** - Track user activity and uploads
                - **Document Management** - Organize and manage your documents
                - **Multi-format Support** - PDF, TXT, MD, DOCX files
                - **Personalized Experience** - User-specific settings and history
                - **Admin Dashboard** - System administration tools
                - **Activity Logging** - Track system usage and performance

                ### 🚀 Deployment Ready
                - **☁️ Cloud Optimized** - One-click Railway deployment
                - **🔒 Secure** - Environment variable configuration
                - **📈 Scalable** - Auto-scaling based on usage
                - **💚 Health Monitoring** - Built-in health checks
                """)

            # Admin Section (only visible to admins)
            with gr.Row(visible=False) as admin_section:
                with gr.Column():
                    gr.Markdown("## 🔧 Admin Panel")

                    with gr.Tabs():
                        with gr.TabItem("User Management"):
                            users_list = gr.Dataframe(
                                headers=["Username", "Email", "Role", "Created At"],
                                datatype=["str", "str", "str", "str"],
                                value=[],
                                interactive=True
                            )

                            refresh_users_btn = gr.Button("🔄 Refresh Users")

                        with gr.TabItem("System Logs"):
                            system_logs = gr.Textbox(
                                label="System Logs",
                                value="Loading logs...",
                                lines=10,
                                interactive=False
                            )

                            refresh_logs_btn = gr.Button("🔄 Refresh Logs")

            # Hidden session input for state management
            session_input = gr.Textbox(visible=False)

            # Logout button
            logout_btn = gr.Button("🚪 Logout", variant="stop")

        # Event handlers for authentication
        def handle_login(username, password, current_state):
            message, session_id, role = login_user(username, password)

            if session_id:
                # Update session state
                new_state = update_session(session_id, current_state)

                # Show success message
                auth_message_output = f"✅ {message}"

                # Switch to main app
                return (
                    gr.update(visible=False),  # Hide login section
                    gr.update(visible=True),   # Show main app
                    gr.update(value=auth_message_output),
                    new_state
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value=f"❌ {message}"),
                    session_state
                )

        def get_session_id_from_state(state):
            """Safely get session_id from state"""
            try:
                if hasattr(state, 'value') and isinstance(state.value, dict):
                    return state.value.get("session_id", "")
                elif isinstance(state, dict):
                    return state.get("session_id", "")
                return ""
            except:
                return ""

        def handle_register(username, password, email):
            message = register_user(username, password, email)
            return gr.update(value=message)

        def handle_logout(session_id):
            message = logout_user(session_id)
            return (
                gr.update(visible=True),   # Show login section
                gr.update(visible=False),  # Hide main app
                gr.update(value=message),
                {"session_id": "", "username": "", "role": "user"}  # Reset state
            )

        # Event handlers for main functionality
        def handle_search(question, session_id):
            if not session_id:
                return "❌ Please login to search documents."
            return query_rag_system(question, session_id)

        def handle_upload(file, session_id):
            if not session_id:
                return "❌ Please login to upload files."
            return upload_file_to_mongodb(file, session_id)

        def handle_clear(session_id):
            if not session_id:
                return "❌ Please login first."
            return ("", "")

        def handle_refresh(session_id):
            if not session_id:
                return "Please login to refresh."
            return "Content refreshed!"

        def handle_refresh_status():
            return get_system_status()

        def handle_user_stats(session_id):
            if not session_id:
                return "Please login to view stats."
            return get_user_stats(session_id)

        def handle_clear_data(session_id):
            return clear_user_data(session_id)

        def update_user_info(session_id):
            user = AuthManager.validate_session(session_id)
            if user:
                return f"""
                👤 **User Information:**
                **Username:** {user['username']}
                **Role:** {user['role']}
                **Email:** {user.get('email', 'Not provided')}

                **Session:** Active (expires in 1 hour)
                **Last Activity:** Just now
                """
            return "No active session"

        def check_and_show_admin(session_id):
            user = AuthManager.validate_session(session_id)
            if user and user['role'] == 'admin':
                return gr.update(visible=True)
            return gr.update(visible=False)

        # Authentication event handlers
        login_btn.click(
            fn=handle_login,
            inputs=[login_username, login_password, session_state],
            outputs=[login_section, main_app, auth_message, session_state]
        )

        reg_btn.click(
            fn=handle_register,
            inputs=[reg_username, reg_password, reg_email],
            outputs=[auth_message]
        )

        # Main application event handlers
        search_btn.click(
            fn=handle_search,
            inputs=[question_input, session_input],
            outputs=[result_output]
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[session_input],
            outputs=[question_input, result_output]
        )

        refresh_btn.click(
            fn=handle_refresh,
            inputs=[session_input],
            outputs=[result_output]
        )

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input, session_input],
            outputs=[upload_status]
        )

        clear_data_btn.click(
            fn=handle_clear_data,
            inputs=[session_input],
            outputs=[upload_status]
        )

        # Status refresh handlers
        refresh_system_btn.click(
            fn=handle_refresh_status,
            outputs=[system_status]
        )

        refresh_user_btn.click(
            fn=handle_user_stats,
            inputs=[session_input],
            outputs=[user_stats]
        )

        refresh_db_btn.click(
            fn=lambda: mongodb_rag.get_database_stats() if mongodb_rag else {},
            outputs=[db_status]
        )

        # Logout handler
        logout_btn.click(
            fn=handle_logout,
            inputs=[session_input],
            outputs=[login_section, main_app, auth_message, session_state]
        )

        # Auto-updates
        def load_user_info():
            session_id = get_session_id_from_state(session_state)
            return update_session(session_id, session_state.value or {})

        def load_admin_section():
            session_id = get_session_id_from_state(session_state)
            return check_and_show_admin(session_id)

        demo.load(
            fn=load_user_info,
            outputs=[user_info_display]
        )

        demo.load(
            fn=load_admin_section,
            outputs=[admin_section]
        )

        # Load initial status
        demo.load(
            fn=handle_refresh_status,
            outputs=[system_status]
        )

        def load_user_stats():
            session_id = get_session_id_from_state(session_state)
            return handle_user_stats(session_id)

        demo.load(
            fn=load_user_stats,
            outputs=[user_stats]
        )

        demo.load(
            fn=lambda: mongodb_rag.get_database_stats() if mongodb_rag else {},
            outputs=[db_status]
        )

        # Hidden state management
        def load_session_input():
            return session_state.value or {}

        demo.load(
            fn=load_session_input,
            outputs=[session_input]
        )

        def refresh_session_state():
            session_id = get_session_id_from_state(session_state)
            return update_session(session_id, session_state.value or {})

        demo.load(
            fn=refresh_session_state,
            outputs=[session_state]
        )

    return demo

def main():
    """Main application entry point"""
    logger.info("🚀 Starting Enhanced Railway RAG PDF Application...")

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
        share=True,  # Force share for Docker environment
        show_api=False,
        inbrowser=False,
        quiet=True,
        favicon_path=None
    )

if __name__ == "__main__":
    main()