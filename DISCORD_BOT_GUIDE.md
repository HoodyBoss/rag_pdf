# 🤖 Discord Bot สำหรับ RAG PDF Assistant

## ภาพรวม

Discord Bot ช่วยให้ผู้ใช้สามารถถามคำถามเกี่ยวกับ PDF ได้โดยตรงจาก Discord โดยใช้คำสั่ง `!ask`

## ฟีเจอร์หลัก

### ✅ รับคำถามจาก Discord
- ใช้คำสั่ง `!ask` ตามด้วยคำถาม
- Bot จะตอบกลับด้วยคำตอบจาก PDF
- แสดงผลในรูปแบบ Embed สวยงาม

### ✅ ควบคุมผ่าน UI
- เปิด/ปิด Bot ได้ผ่าน Gradio Interface
- เลือกโมเดล LLM ที่ต้องการ
- แสดงสถานะ Bot แบบ real-time

### ✅ ปลอดภัยและเสถียร
- ทำงานใน thread แยก ไม่กระทบ Gradio
- จัดการข้อผิดพลาดอย่างดี
- จำกัดความยาวข้อความตามข้อจำกัดของ Discord

## การติดตั้งและตั้งค่า

### 1. สร้าง Discord Bot

1. **ไปที่ Discord Developer Portal**
   - เข้าไปที่ https://discord.com/developers/applications

2. **สร้าง Application ใหม่**
   - คลิก "New Application"
   - ตั้งชื่อ (เช่น "RAG PDF Bot")
   - คลิก "Create"

3. **สร้าง Bot**
   - ไปที่แท็บ "Bot"
   - คลิก "Add Bot" -> "Yes, do it!"
   - ตั้งชื่อ Bot
   - เปิด "Public Bot" (ถ้าต้องการให้คนอื่นเชิญใช้ได้)

4. **ตั้งค่า Bot Permissions**
   - คลิก "Reset Token" เพื่อได้ Bot Token
   - กด "Copy" และเก็บ Token ไว้เป็นความลับ
   - เลือก Privileged Gateway Intents: `MESSAGE CONTENT INTENT`

### 2. สร้าง OAuth2 URL

1. **ไปที่แท็บ "OAuth2" -> "URL Generator"**
2. **เลือก Scopes**: `bot`
3. **เลือก Bot Permissions**:
   - `Read Messages/View Channels`
   - `Send Messages`
   - `Embed Links`
   - `Read Message History`
4. **คัดลอก Generated URL**
5. **วาง URL ใน browser** และเชิญ Bot เข้า Server ของคุณ

### 3. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` ในโปรเจค:

```env
# Discord Bot Configuration
DISCORD_BOT_ENABLED=true
DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
DISCORD_BOT_PREFIX=!ask
DISCORD_DEFAULT_MODEL=gemma3:latest

# Discord Webhook (สำหรับส่งคำตอบจาก Gradio)
DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=YOUR_WEBHOOK_URL_HERE
```

## การใช้งาน

### คำสั่งพื้นฐาน

```
!ask คำถามของคุณ
```

### ตัวอย่างการใช้งาน

```
!ask PDF นี้เกี่ยวกับอะไร

!ask บทที่ 3 พูดถึงเรื่องอะไร

!ask หลักการทำงานของระบบนี้คืออะไร

!ask สรุปเนื้อหาในหน้า 5
```

### การตอบกลับของ Bot

Bot จะตอบกลับในรูปแบบ Embed ที่มี:

- **📚 คำตอบจาก PDF** - คำตอบที่ได้จาก RAG system
- **❓ คำถาม** - คำถามที่ผู้ใช้ถาม
- **🖼️ Thumbnail** - รูป PDF icon
- **Footer** - ข้อมูลว่ามาจาก PDF ที่อัปโหลด

## การควบคุมผ่าน Gradio UI

### เมนู "แอดมิน - อัปโหลด PDF"

ในส่วน **"🤖 จัดการ Discord Bot"**:

1. **เริ่ม Discord Bot**
   - คลิกปุ่ม "เริ่ม Discord Bot"
   - Bot จะเริ่มทำงานและแสดงสถานะ

2. **หยุด Discord Bot**
   - คลิกปุ่ม "หยุด Discord Bot"
   - Bot จะหยุดทำงานทันที

3. **เลือกโมเดลสำหรับ Bot**
   - เลือกจาก dropdown: `gemma3:latest`, `qwen3:latest`, `llama3.2:latest`
   - Bot จะใช้โมเดลที่เลือกสำหรับการตอบคำถาม

## ข้อจำกัดและข้อควรระวัง

### ⚠️ ข้อจำกัด
- **ความยาวข้อความ**: จำกัดที่ ~2000 ตัวอักษร (ขีดจำกัดของ Discord)
- **ประมวลผล**: อาจใช้เวลาสักครู่ขึ้นอยู่กับความยาวคำถาม
- **PDF ต้องอัปโหลด**: Bot สามารถตอบได้เฉพาะ PDF ที่อัปโหลดในระบบแล้ว

### ⚠️ ข้อควรระวัง
- **Bot Token**: อย่าเปิดเผย Bot Token ต่อสาธารณะ
- **Permissions**: ให้ permissions เท่าที่จำเป็นเท่านั้น
- **Security**: Bot จะอ่านข้อความใน channel ที่มีสิทธิ์

## Troubleshooting

### Bot ไม่ตอบคำถาม
1. ตรวจสอบว่า `DISCORD_BOT_ENABLED=true` ใน .env
2. ตรวจสอบ Bot Token ถูกต้อง
3. ตรวจสอบว่า Bot อยู่ใน channel และมีสิทธิ์พอ
4. ตรวจสอบว่ามีการอัปโหลด PDF ในระบบแล้ว

### Bot ขึ้น Error
1. ตรวจสอบ log ใน console ของ Gradio
2. ตรวจสอบว่า LLM model ทำงานปกติ
3. ลอง restart Discord Bot ผ่าน UI

### หมดปุ่มใน Discord
1. ตรวจสอบว่าใส่คำสั่งถูกต้อง: `!ask คำถาม`
2. ตรวจสอบว่า prefix ตรงกับที่ตั้งค่า (`!ask`)
3. ลองพิมพ์ `!help` ดูว่า Bot ตอบหรือไม่

## Feature ที่จะมาในอนาคต

- [ ] สนับสนุน multiple choice questions
- [ ] ส่งรูปภาพที่เกี่ยวข้องจาก PDF
- [ ] ปรับแต่ง prefix และคำสั่ง
- [ ] ส่งคำตอบแบบ private message
- [ ] รองรับหลายภาษา
- [ ] บันทึกประวัติการสนทนา

---

🎉 **พร้อมใช้งานแล้ว!** หลังจากตั้งค่าตามขั้นตอนข้างต้น คุณสามารถใช้ Discord Bot ถามคำถามเกี่ยวกับ PDF ได้ทันที