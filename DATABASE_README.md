# 🗄️ Database แบบถาวรสำหรับ RAG PDF

## ภาพรวม

ระบบใช้ **ChromaDB** แบบถาวร (Persistent Storage) สำหรับเก็บข้อมูล embedding ของ PDF เพื่อให้ข้อมูลไม่สูญหายเมื่อปิดโปรแกรม

## โครงสร้างโฟลเดอร์

```
data/
├── chromadb/           # Database หลัก (persistent storage)
├── chromadb_backup/    # สำรองข้อมูลอัตโนมัติ
│   ├── backup_20241030_143000/
│   ├── backup_20241030_150000/
│   └── ...
└── images/            # รูปภาพจาก PDF
```

## คุณสมบัติ

### ✅ การเก็บข้อมูลแบบถาวร
- ข้อมูล embedding ถูกเก็บใน disk ไม่สูญหายเมื่อ restart โปรแกรม
- ใช้ `chromadb.PersistentClient` สำหรับการจัดเก็บแบบถาวร
- ปิดการใช้งาน telemetry เพื่อความเป็นส่วนตัว

### ✅ สำรองข้อมูลอัตโนมัติ
- สำรองข้อมูลทุกครั้งที่อัปโหลด PDF ใหม่
- สำรองข้อมูลไว้ในโฟลเดอร์ `data/chromadb_backup/`
- ลบ backup เก่าอัตโนมัติ (คงไว้ 7 วันล่าสุด)

### ✅ จัดการ Database ผ่าน UI
- ปุ่มสำรองข้อมูลแบบ manual
- แสดงสถานะ database (จำนวน records, พาธ, จำนวน backup)
- ล้างข้อมูลได้ตามต้องการ

## ฟังก์ชันหลัก

### 1. `backup_vector_db()`
```python
# สำรองข้อมูล database ทั้งหมด
success = backup_vector_db()
```

### 2. `restore_vector_db(backup_name=None)`
```python
# กู้คืนจาก backup ล่าสุด
success = restore_vector_db()

# กู้คืนจาก backup ที่ระบุ
success = restore_vector_db("backup_20241030_143000")
```

### 3. `get_database_info()`
```python
# แสดงข้อมูล database
info = get_database_info()
# ผลลัพธ์: {
#     "total_records": 150,
#     "database_path": "./data/chromadb",
#     "database_exists": True,
#     "backup_count": 3
# }
```

### 4. `cleanup_old_backups(days_to_keep=7)`
```python
# ลบ backup เก่าเกิน 7 วัน
cleanup_old_backups()
```

## การตั้งค่าใน Code

```python
# พาธของ database
TEMP_VECTOR = "./data/chromadb"
TEMP_VECTOR_BACKUP = "./data/chromadb_backup"

# ตั้งค่า ChromaDB
settings = Settings(
    anonymized_telemetry=False,    # ปิด telemetry
    allow_reset=False,             # ป้องกันการ reset
    is_persistent=True             # เปิดการเก็บแบบถาวร
)

# เชื่อมต่อ database
chroma_client = chromadb.PersistentClient(
    path=TEMP_VECTOR,
    settings=settings
)
```

## การใช้งานผ่าน Gradio UI

1. **อัปโหลด PDF** → สำรองข้อมูลอัตโนมัติ
2. **สำรองข้อมูล** → สำรองข้อมูลแบบ manual
3. **แสดงข้อมูล Database** → ดูสถานะปัจจุบัน
4. **ล้างข้อมูล** → ลบข้อมูลทั้งหมด

## การ Backup/Restore แบบ Manual

### Backup
```bash
# คัดลอกโฟลเดอร์ database
cp -r data/chromadb data/chromadb_backup/manual_$(date +%Y%m%d_%H%M%S)
```

### Restore
```bash
# คัดลอกข้อมูลกลับ
cp -r data/chromadb_backup/backup_20241030_143000/* data/chromadb/
```

## ข้อควรระวัง

- ⚠️ **อย่าลบโฟลเดอร์ `data/chromadb`** จะทำให้ข้อมูลสูญหายทั้งหมด
- ⚠️ การ **ล้างข้อมูล** จะลบ embedding ทั้งหมด ต้องอัปโหลด PDF ใหม่
- ⚠️ ให้มั่นใจว่ามี **disk space** เพียงพอสำหรับ backup
- ⚠️ backup จะถูกลบอัตโนมัติหลัง 7 วัน

## Performance

- **การค้นหา**: รวดเร็วแม้มีข้อมูลจำนวนมาก
- **การสำรองข้อมูล**: ใช้เวลาขึ้นอยู่กับขนาด database
- **Disk usage**: ขึ้นอยู่กับจำนวนและขนาด PDF ที่อัปโหลด

## Troubleshooting

### Database ไม่โหลด
- ตรวจสอบว่าโฟลเดอร์ `data/chromadb` มีอยู่จริง
- ตรวจสอบสิทธิ์การเข้าถึงโฟลเดอร์
- รีสตาร์ทโปรแกรม

### สำรองข้อมูลไม่สำเร็จ
- ตรวจสอบ disk space
- ตรวจสอบสิทธิ์การเขียนในโฟลเดอร์ backup
- ดู log สำหรับข้อผิดพลาด

### Performance ช้า
- ลบข้อมูลเก่าที่ไม่ได้ใช้
- ตรวจสอบ disk space ว่าเพียงพอ
- พิจารณาอัปเกรด hardware