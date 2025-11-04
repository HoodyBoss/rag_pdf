# 🚀 Railway RAG PDF Deployment Guide

## 📋 ภาพรวม

RAG PDF พร้อม Deploy บน Railway ด้วย MongoDB เป็นฐานข้อมูลหลัก แก้ไขปัญหา ChromaDB ที่มีปัญหาเรื่อง persistence

## 🏗️ สถาปัตยกรรม

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI      │    │   Railway App    │    │   MongoDB      │
│                 │    │                 │    │                 │
│ - Search        │◄──►│ - Railway Ready  │◄──►│ - Vector Store  │
│ - Upload        │    │ - Auto Scaling   │    │ - Persistent    │
│ - Management   │    │ - Health Checks  │    │ - Backups       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 โครงสร้างไฟล์

### **ไฟล์หลัก:**
- `railway_rag.py` - Railway deployment application
- `mongodb_rag.py` - MongoDB RAG system
- `railway.toml` - Railway configuration
- `requirements_railway.txt` - Python dependencies
- `.env.example` - Environment variables template

### **ไฟล์เสริม:**
- `chromadb_to_mongodb.py` - Migration tool
- `light_rag_integration.py` - LightRAG features
- `DOCKER_README.md` - Docker documentation

## 🚀 วิธี Deploy บน Railway

### **ขั้นที่ 1: เตรียม Repository**
```bash
git init
git add .
git commit -m "Initial RAG PDF Railway deployment"
git push origin main
```

### **ขั้นที่ 2: สร้าง Railway Project**
1. เข้าไปที่ [Railway](https://railway.app)
2. คลิก "New Project"
3. เลือก "Deploy from GitHub repo"
4. ใส่ repository URL ของคุณ
5. เลือก branch `main`

### **ขั้นที่ 3: เพิ่ม MongoDB Plugin**
1. ในหน้า Railway project
2. คลิก "Add New Service"
3. เลือก "MongoDB"
4. ตั้งค่า:
   - Plan: Starter (Free)
   - Region: เลือกใกล้้้ปิดใหมา้
   - Cluster Name: rag-pdf-cluster

### **ขั้นที่ 4: ตั้งค่า Environment Variables**
ในหน้า Railway project ไปที่ "Variables" และเพิ่ม:

```bash
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<database>
DATABASE_NAME=rag_pdf_railway
PORT=7860
HOST=0.0.0.0
LOG_LEVEL=INFO
ENV=production
```

**วิธีหา MongoDB URI:**
1. ใน MongoDB service คลิก "Connect"
2. เลือก "Connect your application"
3. เลือก "Python"
4. คัดลอก connection string ที่ได้

### **ขั้นที่ 5: ตั้งค่า Build**
ใน `railway.toml` ให้แน่ใจ:
- ✅ Python builder
- ✅ Start command: `python railway_rag.py`
- ✅ Port: 7860
- ✅ MongoDB addon

### **ขั้นที่ 6: Deploy**
คลิก "Deploy" เพื่อเริ่มการ deploy

## 🔧 Configuration

### **Environment Variables:**
```bash
# Required
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/db
DATABASE_NAME=rag_pdf_railway

# Optional
PORT=7860
HOST=0.0.0.0
LOG_LEVEL=INFO
ENV=production

# LLM (if using external LLM)
OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL=gemma3:12b
```

### **Railway.toml:**
```toml
[build]
builder = "python"

[deploy]
startCommand = "python railway_rag.py"
restartPolicyType = "ON_FAILURE"

[[services]]
name = "rag-pdf-app"

[services.variables]
PORT = "7860"
MONGODB_URI = "${{MONGODB_URI}}"
DATABASE_NAME = "rag_pdf_railway"
```

## 🗄️ MongoDB Schema

### **Collections:**
1. **`documents`** - Document metadata
   ```json
   {
     "_id": "doc_id",
     "source_name": "example.pdf",
     "total_chunks": 25,
     "created_at": "2024-01-01T00:00:00Z"
   }
   ```

2. **`embeddings`** - Text embeddings
   ```json
   {
     "document_id": "doc_id",
     "chunk_index": 0,
     "text": "Document text content",
     "embedding": [0.1, 0.2, ...],
     "created_at": "2024-01-01T00:00:00Z"
   }
   ```

3. **`metadata`** - Chunk metadata
   ```json
   {
     "document_id": "doc_id",
     "chunk_index": 0,
     "key": "page",
     "value": "1"
   }
   ```

## 📊 ฟีเจอร์

### **🔍 Semantic Search:**
- ค้นหาข้อมูลโดยใช้ vector similarity
- รองรับคำถามภาษาไทยและอังกฤษ
- ค้นหาข้อมูลที่เกี่ยวข้อง semantic

### **📚 Document Management:**
- อัปโหลดไฟล์ PDF และข้อความ
- แบ่งข้อความเป็น chunks อัตโนมัติ
- เก็บ metadata ของแต่ละ chunk

### **📈 Analytics:**
- จำนวน documents และ chunks
- การตรวจสอบสถานะระบบ
- Performance metrics

## 🔄 Migration จาก ChromaDB

ถ้าคุณมีข้อมูลใน ChromaDB อยู่แล้ว:

```bash
# 1. ติดตั้ง dependencies
pip install pymongo[srv]

# 2. Run migration
python chromadb_to_mongodb.py full

# 3. ตรวจสอบผล
python chromadb_to_mongodb.py test
```

## 🧪 การทดสอบ

### **ทดสอบใน Local:**
```bash
# 1. ติดตั้ง MongoDB local
docker run -d -p 27017:27017 mongo

# 2. รัน Railway app
python railway_rag.py

# 3. เปิด http://localhost:7860
```

### **ทดสอบใน Railway:**
1. เปิด Railway logs
2. ตรวจสอบ deployment status
3. ทดสอบ search functionality
4. ทดสอบ file upload

## 🔍 Troubleshooting

### **MongoDB Connection Issues:**
```bash
# ตรวจสอบ connection string
python -c "from pymongo import MongoClient; client = MongoClient('your_uri'); print('Connected!' if client.admin.command('ping') else 'Failed')"
```

### **Memory Issues:**
- จำกัด memory: 1024MB (default)
- ปรับ memory ใน `railway.toml`:
```toml
[resources]
memoryMb = 2048
```

### **Performance Issues:**
- ใช้ indexes ใน MongoDB
- จำกัด chunk size
- ใช้ vector search optimization

## 💰 Cost Optimization

### **Railway Free Plan:**
- ✅ 500 hours/month
- ✅ 1 service
- ✅ 100MB storage

### **MongoDB Free Plan:**
- ✅ 512MB storage
- ✅ Basic features
- ✅ 3 indexes

### **การปรับทุก:**
- Upgrade to Railway Pro สำหรับความต้องการสูง
- MongoDB Atlas scaling สำหรับข้อมูลมากขึ้น

## 🔒 Security

### **รักษาความปลอดภัย:**
- ใช้ environment variables สำหรับข้อมูล sensitive
- Enable authentication สำหรับ MongoDB
- ใช้ HTTPS สำหรับ communication
- Regular backups

### **MongoDB Security:**
```json
{
  "access": {
    "username": "user",
    "password": "strong_password",
    "database": "rag_pdf_railway"
  }
}
```

## 🚀 Production Best Practices

### **การตั้งค่า:**
- ใช้ MongoDB Atlas สำหรับ production
- เปิด auto-backup
- Monitor performance metrics
- เซ็ต health checks

### **Monitoring:**
- Railway logs
- MongoDB Atlas metrics
- Application performance monitoring
- Error tracking

### **Scaling:**
- Horizontal scaling ด้วย Railway
- MongoDB sharding สำหรับข้อมูลมาก
- CDN สำหรับ static assets

## 📞 Support

### **Railway Documentation:**
- [Railway Docs](https://docs.railway.app/)
- [MongoDB on Railway](https://docs.railway.app/marketplace/mongodb)

### **Troubleshooting:**
1. ตรวจสอบ Railway logs
2. ตรวจสอบ MongoDB connection
3. ทดสอบ environment variables
4. ทดสอบ health endpoint

## 🎉 สรุป

✅ **พร้อม Deploy บน Railway!**
- MongoDB integration
- Railway configuration
- Migration tools
- Production ready
- Auto-scaling
- Health checks

**เริ่ม deploy บน Railway วันนี้!** 🚀