# RAG PDF - FastAPI + ElysiaJS

RAG PDF Application ที่ถูกพัฒนาขึ้นมาใช้ FastAPI (Backend) และ ElysiaJS (Frontend)

## 🏗️ สถาปัตยกรรม

```
├── backend/                 # FastAPI Backend
│   ├── main.py             # FastAPI Application
│   ├── rag_core.py         # RAG Core Logic
│   ├── requirements.txt    # Python Dependencies
│   └── .env.example        # Environment Variables
├── frontend/               # ElysiaJS Frontend
│   ├── src/
│   │   └── index.js        # Main Frontend Application
│   ├── package.json       # Node.js Dependencies
│   └── .env.example        # Environment Variables
└── README_FASTAPI.md       # This file
```

## 🚀 Getting Started

### Backend (FastAPI)

1. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Backend Server**
   ```bash
   python main.py
   ```
   Backend will run on: http://localhost:8000

### Frontend (ElysiaJS)

1. **Install Node.js Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your JWT secret
   ```

3. **Start Frontend Development Server**
   ```bash
   npm run dev
   ```
   Frontend will run on: http://localhost:3000

## 📋 Features

### ✅ พัฒนาแล้ว
- **🔐 Authentication System** - Login/Logout ด้วย JWT
- **📁 Document Upload** - รองรับ PDF, TXT, DOCX
- **💬 Chat Interface** - ถามคำถามเกี่ยวกับเอกสาร
- **🎨 Modern UI** - Responsive design ด้วย Tailwind CSS
- **📚 Document Management** - ลิสต์และลบเอกสาร
- **🚀 Fast API** - REST API ด้วย FastAPI

### 🔄 Features ที่จะเพิ่ม
- **🧠 RAG Core** - ระบบ Retrieval-Augmented Generation
- **📊 Document Processing** - เปิดและทำ index เอกสาร
- **🔍 Vector Search** - ค้นหา semantic ในเอกสาร
- **📖 Chat History** - ประวัติการสนทนา
- **👥 User Management** - จัดการผู้ใช้และ permissions

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/login` - Login และรับ JWT token
- `GET /api/auth/me` - Get current user info

### Documents
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - Get user documents
- `DELETE /api/documents/{id}` - Delete document

### Chat
- `POST /api/chat` - Send question and get answer
- `GET /api/chat/history` - Get chat history

### System
- `GET /api/health` - Health check
- `GET /` - Root endpoint

## 🛠️ Development

### Running Both Services

1. **Start Backend**
   ```bash
   cd backend
   python main.py
   ```

2. **Start Frontend** (ใน terminal อื่น)
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Demo Credentials
- **Username:** admin
- **Password:** admin123

## 🚀 Deployment

### Railway Deployment

1. **Setup Railway Project**
   - Connect GitHub repository
   - Add environment variables

2. **Backend Service**
   ```bash
   # Railway.toml
   [build]
   builder = "dockerfile"

   [deploy]
   startCommand = "cd backend && python main.py"
   ```

3. **Frontend Service**
   ```bash
   # Railway.toml for frontend
   [build]
   builder = "nixpacks"

   [deploy]
   startCommand = "cd frontend && npm start"
   ```

## 📝 Notes

- **Backend** ใช้ FastAPI กับ Uvicorn
- **Frontend** ใช้ ElysiaJS กับ Bun/Node.js
- **Database** ใช้ MongoDB + ChromaDB (สำหรับ vectors)
- **Authentication** ใช้ JWT tokens
- **File Storage** ใช้ local storage (สามารถเปลี่ยนเป็น S3)

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.