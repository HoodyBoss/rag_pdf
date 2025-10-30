# 🧠 Enhanced RAG System Documentation

## 🎯 ภาพรวม

Enhanced RAG เป็นระบบที่พัฒนาขึ้นจาก MemoRAG concepts โดยใช้ libraries ที่มีอยู่แล้วในระบบเพื่อเพิ่มความสามารถในการจดจำบริบทการสนทนา (Memory-Augmented Generation)

## ✨ คุณสมบัติหลัก

### 🧠 **Memory Management**
- **Short-term Memory**: จดจำการสนทนาล่าสุด (memory window: 10 ข้อความ)
- **Long-term Memory**: จัดเก็บการสนทนาสำคัญไว้ในฐานข้อมูล ChromaDB
- **Session Tracking**: แยกข้อมูลตาม session ID ของการสนทนา
- **Context Chaining**: ใช้บริบทจากการสนทนาก่อนหน้ามาช่วยในการตอบคำถาม

### 🔍 **Intelligent Retrieval**
- **Similarity-based Memory Retrieval**: ดึงความทรงจำที่เกี่ยวข้องจาก vector database
- **Context Integration**: ผสานข้อมูลจากเอกสารและความทรงจำการสนทนา
- **Relevance Scoring**: ให้คะแนนความเกี่ยวข้องของความทรงจำที่ดึงมา

### 🎨 **Enhanced Prompting**
- **Structured Prompt Templates**: รูปแบบ prompt ที่ออกแบบมาเพื่อ Enhanced RAG
- **Context-aware Responses**: ตอบคำถามโดยคำนึงถึงบริบทการสนทนาก่อนหน้า
- **Reasoning Chain**: สร้างลำดับความคิดที่ต่อเนื่องกัน

## 🏗️ สถาปัตยกรรมระบบ

### **EnhancedRAG Class Components**

```python
class EnhancedRAG:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_WINDOW)      # Short-term memory
        self.long_term_memory = []                      # Long-term memories
        self.session_id = str(uuid.uuid4())             # Session identifier
        self.chroma_client = chromadb.PersistentClient() # Memory storage
        self.memory_collection = self._get_memory_collection()
```

### **Memory Storage Structure**

แต่ละความทรงจำจะถูกจัดเก็บในรูปแบบ:
```json
{
  "session_id": "uuid-string",
  "question": "คำถามของผู้ใช้",
  "answer": "คำตอบจากระบบ",
  "timestamp": "2024-10-30T15:30:00",
  "memory_id": "unique-identifier",
  "type": "qa_memory"
}
```

## 🚀 การใช้งานใน UI

### **เลือกโหมด RAG**
1. **Standard RAG** - ค้นหาจากเอกสารเท่านั้น
   - ไม่จดจำประวัติการสนทนา
   - เร็วและใช้ทรัพยากรน้อยกว่า
   - เหมาะสำหรับคำถามแบบสแตนด์อะโลน

2. **Enhanced RAG** - จดจำบริบทการสนทนา
   - จดจำการสนทนาก่อนหน้า
   - ตอบคำถามด้วยบริบทที่ต่อเนื่อง
   - เหมาะสำหรับการสนทนาที่ซับซ้อน

### **Memory Status Monitoring**
- ตรวจสอบจำนวนความทรงจำทั้งหมด
- ดูสถิติการใช้งาน memory
- รีเฟรชข้อมูล memory แบบ real-time

## 🔄 วงจรการทำงาน

### **1. การตอบคำถาม**
```
User Question → Enhanced RAG Process
                      ↓
            Retrieve Relevant Memories
                      ↓
            Build Context-enhanced Prompt
                      ↓
            Generate LLM Response
                      ↓
            Store Q&A in Memory
```

### **2. Memory Management**
```
New Conversation → Add to Short-term Memory
                      ↓
            Check Memory Importance
                      ↓
            Transfer to Long-term Memory
                      ↓
            Update Memory Database
```

## 🎯 ประโยชน์ของ Enhanced RAG

### **✅ ข้อดี**
- **Context Continuity**: จดจำบริบทการสนทนาต่อเนื่อง
- **Better Understanding**: เข้าใจคำถามที่อ้างอิงถึงการสนทนาก่อนหน้า
- **Personalized Responses**: ตอบสนองตามประวัติการสนทนา
- **Scalable Memory**: จัดเก็บความทรงจำได้ไม่จำกัด
- **Session Isolation**: แยกความทรงจำตาม session การใช้งาน

### **🔧 การใช้งานที่เหมาะสม**
- **Technical Support**: จดจำปัญหาและวิธีแก้ไขที่เคยถามไป
- **Learning Assistant**: ติดตามความคืบหน้าการเรียนรู้
- **Research Assistant**: จดจำข้อมูลที่ค้นหาไปแล้ว
- **Documentation Help**: จดจำส่วนของเอกสารที่เคยอ้างอิง

## 📊 Performance Considerations

### **Memory Window Size**
- **Current**: 10 ข้อความล่าสุด
- **Trade-off**: ยิ่งมากยิ่งจดจำได้นาน แต่ใช้ทรัพยากรมากขึ้น
- **Optimization**: สามารถปรับแต่งได้ตามความต้องการ

### **Retrieval Strategy**
- **Similarity Threshold**: กำหนดความเกี่ยวข้องขั้นต่ำในการดึงความทรงจำ
- **Max Retrieval**: จำกัดจำนวนความทรงจำที่ดึงมา (ปัจจุบัน: 5)
- **Hybrid Approach**: ผสมระหว่างความเก่าและความใหม่ของความทรงจำ

## 🛠️ การตั้งค่าและการปรับแต่ง

### **Memory Configuration**
```python
# ในไฟล์ .env
ENHANCED_RAG_ENABLED=true
MEMORY_WINDOW=10
MAX_RETRIEVED_MEMORIES=5
SIMILARITY_THRESHOLD=0.7
```

### **Database Setup**
- **ChromaDB Persistent Storage**: ข้อมูลความทรงจำอยู่ถาวร
- **Memory Collection**: แยก collection สำหรับเก็บความทรงจำ
- **Backup Strategy**: สำรองข้อมูลความทรงจำพร้อม database หลัก

## 🔮 การพัฒนาในอนาคต

### **Planned Features**
- **Memory Importance Scoring**: จัดลำดับความสำคัญของความทรงจำ
- **Automatic Memory Cleanup**: ลบความทรงจำที่ไม่สำคัญโดยอัตโนมัติ
- **Cross-session Memory**: แชร์ความทรงจำข้าม session การใช้งาน
- **Memory Analytics**: วิเคราะห์รูปแบบการสนทนาและการใช้ความทรงจำ

### **Potential Enhancements**
- **Multi-modal Memory**: จดจำรูปภาพและข้อมูลอื่นๆ นอกจากข้อความ
- **Memory Compression**: บีบอัดความทรงจำเพื่อประสิทธิภาพที่ดีขึ้น
- **Personalization Engine**: ปรับแต่งการตอบสนองตามสไตล์การสนทนา

---

**Enhanced RAG System** พร้อมใช้งานแล้ว! 🚀 ลองเปลี่ยนเป็น Enhanced RAG mode และสัมผัสประสบการณ์การสนทนาที่ฉลาดขึ้น