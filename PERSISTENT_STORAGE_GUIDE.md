# 💾 คู่มือการเก็บข้อมูลแบบถาวรสำหรับ RAG PDF

## 📋 สถานะปัจจุบัน

✅ **ChromaDB Persistent Storage ทำงานแล้ว!**

- **Database Type**: ChromaDB (SQLite backend)
- **Storage Location**: `./data/chromadb/`
- **Current Records**: 291 records
- **Storage Size**: ~4MB
- **Persistence**: ✅ ถาวร (ไม่หายเมื่อ restart)

## 🔧 การตั้งค่า Database ปัจจุบัน

### 1. **PersistentClient Configuration**
```python
# ใช้ PersistentClient สำหรับเก็บข้อมูลถาวร
chroma_client = chromadb.PersistentClient(
    path="./data/chromadb",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=False,
        is_persistent=True
    )
)
```

### 2. **Collection Management**
```python
# โหลด collection ที่มีอยู่แล้ว ไม่สร้างใหม่ทุกครั้ง
try:
    collection = chroma_client.get_collection(name="pdf_data")
    logging.info(f"Loaded existing collection: {collection.count()} records")
except:
    collection = chroma_client.create_collection(name="pdf_data")
```

### 3. **สำคัญ: ไม่ลบข้อมูลเก่า**
```python
# แก้ไขแล้ว - ไม่เรียก clear_vector_db() ทุกครั้งที่อัปโหลด PDF
# ตอนนี้เพิ่มข้อมูลเข้าไปใน database เดิมแทน
if current_count > 0:
    logging.info("Adding new PDF to existing database")
    # เพิ่มข้อมูล ไม่ลบข้อมูลเก่า
else:
    logging.info("Creating new database")
```

## 📁 โครงสร้างไฟล์

```
data/
├── chromadb/                    # Database หลัก (persistent)
│   ├── chroma.sqlite3          # SQLite database file
│   └── [collection-ids]/       # Collection data folders
├── chromadb_backup/            # Automatic backups
│   ├── backup_20241030_143000/
│   └── backup_20241030_150000/
└── images/                    # PDF images
```

## 🔄 การทำงานระบบ

### ✅ **อัปโหลด PDF ครั้งแรก**
1. สร้าง collection ใหม่ (ถ้ายังไม่มี)
2. แยกข้อความและรูปภาพจาก PDF
3. สร้าง embeddings และเก็บใน ChromaDB
4. สำรองข้อมูลอัตโนมัติ

### ✅ **อัปโหลด PDF ครั้งต่อไป**
1. ตรวจสอบข้อมูลที่มีอยู่ (ไม่ลบข้อมูลเก่า!)
2. สำรองข้อมูลปัจจุบัน
3. เพิ่ม PDF ใหม่เข้าไปใน database เดิม
4. สำรองข้อมูลอีกครั้ง

### ✅ **Restart โปรแกรม**
1. โหลด ChromaDB จาก disk
2. ดึง collection "pdf_data" ที่มีอยู่
3. ข้อมูลทั้งหมดยังคงอยู่ ✅

## 🛠️ เครื่องมือจัดการใน UI

### 1. **ส่วนจัดการ Database**
- **สำรองข้อมูล**: สำรองข้อมูลแบบ manual
- **แสดงข้อมูล Database**: ดูสถานะและข้อมูลทั่วไป
- **ตรวจสอบข้อมูลละเอียด**: ดูตัวอย่างข้อมูลภายใน database

### 2. **ข้อมูลที่แสดง**
```
{
  "total_records": 291,
  "database_path": "./data/chromadb",
  "database_exists": true,
  "sqlite_exists": true,
  "database_size_mb": 3.87,
  "backup_count": 2,
  "collections": ["pdf_data"]
}
```

## 🚨 ปัญหาที่แก้ไขแล้ว

### ❌ **ปัญหาเก่า: ข้อมูลหายเมื่อ restart**
- **สาเหตุ**: `clear_vector_db()` ถูกเรียกทุกครั้งที่อัปโหลด PDF
- **วิธีแก้**: แก้ไข `process_pdf_upload()` ไม่ให้ลบข้อมูลเก่า

### ❌ **ปัญหาเก่า: Database ถูก reset บ่อย**
- **สาเหตุ**: สร้าง collection ใหม่ทุกครั้ง
- **วิธีแก้**: โหลด collection ที่มีอยู่ก่อน

### ✅ **สถานะปัจจุบัน: ทำงานถูกต้อง**
- Database ถาวรด้วย ChromaDB
- ข้อมูลไม่หายเมื่อ restart
- เพิ่ม PDF ใหม่โดยไม่ลบข้อมูลเก่า
- มีระบบ backup อัตโนมัติ

## 🔍 การตรวจสอบว่าทำงานถูกต้อง

### 1. **ตรวจสอบใน terminal**
```bash
python -c "
import chromadb
client = chromadb.PersistentClient('./data/chromadb')
collection = client.get_collection('pdf_data')
print(f'Records: {collection.count()}')
"
```

### 2. **ตรวจสอบไฟล์**
```bash
ls -la data/chromadb/
# ควรเห็น chroma.sqlite3 และ folder collections
```

### 3. **ทดสอบ restart**
1. รันโปรแกรมและอัปโหลด PDF
2. ปิดโปรแกรม
3. รันโปรแกรมใหม่
4. ข้อมูลควรยังอยู่ ✅

## 💡 เคล็ดลับการใช้งาน

### ✅ **ปรับปรุงประสิทธิภาพ**
- Database ทำงานเร็วแม้มีข้อมูลจำนวนมาก
- ไม่จำเป็นต้อง cleanup ข้อมูลบ่อย
- Backup อัตโนมัติช่วยป้องกันการสูญหาย

### ✅ **การจัดการข้อมูล**
- ถ้าต้องการล้างข้อมูล: กดปุ่ม "ล้างข้อมูล" ใน UI
- ถ้าต้องการ backup: กดปุ่ม "สำรองข้อมูล"
- ถ้าต้องการ restore: ใช้ฟังก์ชัน `restore_vector_db()`

### ✅ **Monitoring**
- ตรวจสอบ log สำหรับ database operations
- ใช้ปุ่ม "ตรวจสอบข้อมูลละเอียด" เพื่อดูข้อมูลภายใน
- Monitor ขนาด database ผ่าน UI

## 🎯 สรุป

**ระบบเก็บข้อมูลแบบถาวรทำงานถูกต้องแล้ว!**

- ✅ ChromaDB Persistent Storage
- ✅ ไม่ลบข้อมูลเก่าเมื่ออัปโหลด PDF ใหม่
- ✅ ข้อมูลไม่หายเมื่อ restart โปรแกรม
- ✅ มีระบบ backup อัตโนมัติ
- ✅ UI สำหรับจัดการและตรวจสอบข้อมูล

**ปัญหาข้อมูลหายถูกแก้ไขเรียบร้อยแล้ว!** 🚀