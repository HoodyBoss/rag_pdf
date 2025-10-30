import base64
import gradio as gr
import os
import shutil
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

import torch
import ollama
import shortuuid
import logging
import re
import discord
import pandas as pd
import asyncio
import requests
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from typing import List, Dict, Tuple, Union
import threading
import docx
from pathlib import Path
import json
import hashlib
from collections import deque
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Image folder
TEMP_IMG="./data/images"
TEMP_VECTOR="./data/chromadb"
TEMP_VECTOR_BACKUP="./data/chromadb_backup"
# รายชื่อ Model ที่คุณมีบน Ollama
AVAILABLE_MODELS = ["gemma3:latest", "qwen3:latest","llama3.2:latest"]

# Discord Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "YOUR_WEBHOOK_URL_HERE")  # ใส่ Webhook URL ที่นี่
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")  # ใส่ Bot Token ที่นี่
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID_HERE")  # ใส่ Channel ID ที่นี่
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"  # เปิด/ปิดการแจ้งเตือน Discord

# Discord Bot Configuration
DISCORD_BOT_ENABLED = os.getenv("DISCORD_BOT_ENABLED", "false").lower() == "true"  # เปิด/ปิด Bot สำหรับรับคำถาม
DISCORD_BOT_PREFIX = os.getenv("DISCORD_BOT_PREFIX", "!ask ")  # คำนำหน้าคำสั่ง
DISCORD_DEFAULT_MODEL = os.getenv("DISCORD_DEFAULT_MODEL", "gemma3:latest")  # โมเดลเริ่มต้นสำหรับ Discord
DISCORD_RESPOND_NO_PREFIX = os.getenv("DISCORD_RESPOND_NO_PREFIX", "true").lower() == "true"  # ตอบคำถามที่ไม่มี prefix หรือไม่
DISCORD_REPLY_MODE = os.getenv("DISCORD_REPLY_MODE", "channel")  # วิธีการตอบกลับ: channel/dm/both

# LINE OA Configuration
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "YOUR_LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "YOUR_LINE_CHANNEL_SECRET")
LINE_ENABLED = os.getenv("LINE_ENABLED", "false").lower() == "true"  # เปิด/ปิด LINE OA
LINE_DEFAULT_MODEL = os.getenv("LINE_DEFAULT_MODEL", "gemma3:latest")  # โมเดลเริ่มต้นสำหรับ LINE
LINE_WEBHOOK_PORT = int(os.getenv("LINE_WEBHOOK_PORT", "5000"))  # Port สำหรับ LINE webhook

# Facebook Messenger Configuration
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "YOUR_FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "YOUR_FB_VERIFY_TOKEN")
FB_ENABLED = os.getenv("FB_ENABLED", "false").lower() == "true"  # เปิด/ปิด Facebook Messenger
FB_DEFAULT_MODEL = os.getenv("FB_DEFAULT_MODEL", "gemma3:latest")  # โมเดลเริ่มต้นสำหรับ FB
FB_WEBHOOK_PORT = int(os.getenv("FB_WEBHOOK_PORT", "5001"))  # Port สำหรับ FB webhook

# Enhanced RAG Configuration (MemoRAG-like features)
RAG_MODE = os.getenv("RAG_MODE", "enhanced")  # standard, enhanced
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))  # จำนวน conversations ที่จำไว้
ENABLE_CONTEXT_CHAINING = os.getenv("ENABLE_CONTEXT_CHAINING", "true").lower() == "true"
ENABLE_REASONING = os.getenv("ENABLE_REASONING", "true").lower() == "true"
ENABLE_SESSION_MEMORY = os.getenv("ENABLE_SESSION_MEMORY", "true").lower() == "true"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Chroma client Disable telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# สร้างโฟลเดอร์สำหรับ database ถ้ายังไม่มี
os.makedirs(TEMP_VECTOR, exist_ok=True)
os.makedirs(TEMP_VECTOR_BACKUP, exist_ok=True)

# ตั้งค่า ChromaDB ให้เก็บข้อมูลแบบถาวรพร้อมการจัดการดีขึ้น
settings = Settings(
    anonymized_telemetry=False,
    allow_reset=False,
    is_persistent=True
)

chroma_client = chromadb.PersistentClient(
    path=TEMP_VECTOR,
    settings=settings
)

# สร้างหรือดึง collection ที่มีอยู่แล้ว
try:
    # พยายามโหลด collection ที่มีอยู่ก่อน
    collection = chroma_client.get_collection(name="pdf_data")
    count = collection.count()
    logging.info(f"✅ โหลด collection 'pdf_data' สำเร็จ - จำนวน {count} เรคคอร์ด")
    logging.info(f"📁 Database path: {TEMP_VECTOR}")
    logging.info(f"💾 Database exists: {os.path.exists(TEMP_VECTOR)}")
except Exception as e:
    # ถ้าไม่มีให้สร้างใหม่
    logging.info(f"❌ ไม่พบ collection ที่มีอยู่ - กำลังสร้างใหม่: {str(e)}")
    collection = chroma_client.create_collection(name="pdf_data")
    logging.info(f"✅ สร้าง collection 'pdf_data' ใหม่สำเร็จ")

# ตั้งค่า device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# โหลดโมเดล embedding
# SentenceTransformer สำหรับข้อความหลายภาษา (เน้นภาษาไทย)
sentence_model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)

# Create directory for storing images
os.makedirs(TEMP_IMG, exist_ok=True)

sum_tokenizer = MT5Tokenizer.from_pretrained('StelleX/mt5-base-thaisum-text-summarization')
sum_model = MT5ForConditionalGeneration.from_pretrained('StelleX/mt5-base-thaisum-text-summarization')

def summarize_content(content: str) -> str:
    """
        สรุปเนื้อหา 
    """
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")    
       
    input_ = sum_tokenizer(content, truncation=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        preds = sum_model.generate(
            input_['input_ids'].to('cpu'),
            num_beams=15,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_length=250
        )

    summary = sum_tokenizer.decode(preds[0], skip_special_tokens=True)

    logging.info(f" summary: {summary}.")
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")
    return summary

# แยกเนื้อหา, รูป ออกจาก PDF
def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    แยกข้อความและรูปภาพจาก PDF โดยใช้ PyMuPDF
    """
    try:
        doc = fitz.open(pdf_path)
        content_chunks = []
        all_text=[]

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text("text").strip()
            all_text.append(f"{text} \n\n\n")
            if not text:
                text = f"ไม่มีข้อความในหน้า {page_num + 1}"
            
            logging.info("################# Text data ##################")
            chunk_data = {"text": f"ข้อมูลจากหน้า {page_num + 1} : {text}" , "images": []}
            
            # Extract images
            image_list = page.get_images(full=True)
            logging.info("################# images list ##################")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    img_id = f"pic_{str(page_num+1)}_{str(img_index+1)}"
                    img_path = f"{TEMP_IMG}/{img_id}.{image_ext}"
                    image.save(img_path, format=image_ext.upper())

                    img_desc = f"รูปภาพ จากหน้า {str(page_num+1)} ของ รูปที่ {str(img_index+1)}, บริบทข้อความ: {text[:80]}..."  
                    chunk_data["text"] += f"\n[ภาพ: {img_id}.{image_ext}]"                    
                    chunk_data["images"].append({
                        "data": image,
                        "path": img_path,
                        "description": img_desc
                    })
                except Exception as e:
                    logger.warning(f"ไม่สามารถประมวลผลรูปภาพที่หน้า {str(page_num+1)}, รูปที่ {str(img_index+1)}: {str(e)}")
            
            if chunk_data["text"]:
                content_chunks.append(chunk_data)
        
        if not any(chunk["images"] for chunk in content_chunks):
            logger.warning("ไม่พบรูปภาพใน PDF: %s", pdf_path)
        
        doc.close()
        content_text= "".join(all_text)
        # ตัดคำภาษาไทย
        thaitoken_text = preprocess_thai_text(content_text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
        print("################################")
        print(f"{ thaitoken_text }")
        print("################################")
        global summarize
        summarize = summarize_content(thaitoken_text)
        return content_chunks
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดในการแยก PDF: %s", str(e))
        raise

# ตัดคำภาษาไทย 
def preprocess_thai_text(text: str) -> str:
    """
    ตัดคำภาษาไทยด้วย pythainlp เพื่อเตรียมข้อความ

    Args:
        text (str): ข้อความภาษาไทย

    Returns:
        str: ข้อความที่ตัดคำแล้ว
    """
    return " ".join(word_tokenize(text, engine="newmm"))


def embed_text(text: str) -> np.ndarray:
    """
    สร้าง embedding สำหรับข้อความโดยใช้ SentenceTransformer 

    Args:
        text (str): ข้อความที่ต้องการสร้าง embedding        

    Returns:
        np.ndarray: Embedding vector ที่รวมจากหลายโมเดล
    """
    logging.info("-------------- start embed text  -------------------")
    
    # ตัดคำภาษาไทย
    processed_text = preprocess_thai_text(text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
    
    # สร้าง embedding ด้วย SentenceTransformer
    sentence_embedding = sentence_model.encode(processed_text, normalize_embeddings=True, device=device)    
        
    return sentence_embedding

def store_in_chroma(content_chunks: List[Dict], source_name: str):
    """
    เก็บข้อมูลข้อความและรูปภาพใน Chroma พร้อม embedding
    รองรับรูปแบบใหม่ (chunks with metadata) และรูปแบบเก่า (backward compatibility)
    """
    logging.info(f"##### Start store {len(content_chunks)} chunks in chroma #########")

    if not content_chunks:
        logging.warning("No chunks to store")
        return

    # ตรวจสอบว่าเป็นรูปแบบใหม่หรือเก่า
    if isinstance(content_chunks[0], dict) and "text" in content_chunks[0] and "metadata" in content_chunks[0]:
        # รูปแบบใหม่: chunks พร้อม metadata แบบละเอียด
        store_chunks_modern(content_chunks)
    else:
        # รูปแบบเก่า: chunks พร้อม images (เพื่อความเข้ากันได้)
        store_chunks_legacy(content_chunks, source_name)


def store_chunks_modern(chunks: List[Dict]):
    """เก็บ chunks แบบใหม่ที่มี metadata แบบละเอียด"""
    logging.info("Storing chunks in modern format")

    for chunk in chunks:
        try:
            text = chunk["text"]
            chunk_metadata = chunk["metadata"]
            chunk_id = chunk["id"]

            logging.info(f"Processing chunk: {chunk_id} ({len(text)} chars)")

            # สร้าง embedding
            text_embedding = embed_text(text)

            # สร้าง metadata สำหรับ ChromaDB
            metadata = {
                "type": "text",
                "source": chunk_metadata.get("source", "unknown"),
                "file_type": chunk_metadata.get("file_type", "unknown"),
                "start": chunk_metadata.get("start", 0),
                "end": chunk_metadata.get("end", 0),
                "chunk_id": chunk_id
            }

            # เก็บใน ChromaDB
            collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[text_embedding.tolist()],
                ids=[chunk_id]
            )

            logging.info(f"✅ Stored chunk {chunk_id}")

        except Exception as e:
            logging.error(f"❌ Failed to store chunk {chunk.get('id', 'unknown')}: {str(e)}")


def store_chunks_legacy(chunks: List[Dict], source_name: str):
    """เก็บ chunks แบบเก่า (เพื่อความเข้ากันได้กับ PDF รูปแบบเดิม)"""
    logging.info("Storing chunks in legacy format")

    for chunk in chunks:
        try:
            text = chunk["text"]
            images = chunk.get("images", [])

            logging.info(f"Processing legacy chunk ({len(text)} chars)")

            # สร้าง embedding
            text_embedding = embed_text(text)
            text_id = shortuuid.uuid()[:8]

            # เก็บข้อความ
            collection.add(
                documents=[text],
                metadatas=[{
                    "type": "text",
                    "source": source_name,
                    "format": "legacy"
                }],
                embeddings=[text_embedding.tolist()],
                ids=[text_id]
            )

            # เก็บรูปภาพ (ถ้ามี)
            for img in images:
                try:
                    img_id = shortuuid.uuid()[:8]
                    img_path = img.get("path", "")

                    collection.add(
                        documents=[text],
                        metadatas=[{
                            "type": "image",
                            "source": source_name,
                            "image_path": img_path,
                            "description": img.get("description", ""),
                            "format": "legacy"
                        }],
                        embeddings=[text_embedding.tolist()],
                        ids=[img_id]
                    )
                    logging.info(f"✅ Stored image {img_id}")

                except Exception as e:
                    logging.error(f"❌ Failed to store image: {str(e)}")

        except Exception as e:
            logging.error(f"❌ Failed to store legacy chunk: {str(e)}")

def extract_text_from_file(file_path: str) -> str:
    """
    แยกข้อความจากไฟล์ต่างๆ (PDF, TXT, MD, DOCX)

    Args:
        file_path: พาธของไฟล์

    Returns:
        str: ข้อความที่แยกได้
    """
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return extract_pdf_text(file_path)
        elif file_ext in ['.txt', '.md']:
            return extract_text_file(file_path)
        elif file_ext == '.docx':
            return extract_docx_text(file_path)
        else:
            logging.warning(f"ไม่รองรับไฟล์ประเภท: {file_ext}")
            return ""

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความจาก {file_path}: {str(e)}")
        return ""


def extract_pdf_text(pdf_path: str) -> str:
    """แยกข้อความจาก PDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความ PDF: {str(e)}")
        return ""


def extract_text_file(file_path: str) -> str:
    """แยกข้อความจากไฟล์ .txt หรือ .md"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                return f.read()
        except Exception as e:
            logging.error(f"ไม่สามารถอ่านไฟล์ {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_path}: {str(e)}")
        return ""


def extract_docx_text(docx_path: str) -> str:
    """แยกข้อความจากไฟล์ .docx"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการแยกข้อความ DOCX: {str(e)}")
        return ""


def process_multiple_files(files, clear_before_upload: bool = False):
    """
    ประมวลผลไฟล์หลายไฟล์ (PDF, TXT, MD, DOCX)

    Args:
        files: list ของไฟล์จาก Gradio
        clear_before_upload: ล้างข้อมูลเก่าหรือไม่
    """
    try:
        if not files:
            return "❌ กรุณาเลือกไฟล์เพื่ออัปโหลด"

        current_count = collection.count()
        logging.info(f"เริ่มประมวลผล {len(files)} ไฟล์")

        # สำรองข้อมูลก่อนการเปลี่ยนแปลง (Enhanced Backup)
        auto_backup_result = auto_backup_before_operation()
        if not auto_backup_result["success"]:
            logging.warning(f"Auto backup failed: {auto_backup_result.get('error')}")
        else:
            logging.info(f"Auto backup created: {auto_backup_result['backup_name']}")

        # ถ้าเลือกล้างข้อมูลเก่า
        if clear_before_upload:
            logging.info("กำลังล้างข้อมูลเก่าตามคำร้อง...")
            clear_vector_db()

        total_chunks = 0
        successful_files = []
        failed_files = []

        # ประมวลผลทีละไฟล์
        for file_obj in files:
            try:
                file_path = file_obj.name
                file_name = os.path.basename(file_path)
                file_ext = Path(file_path).suffix.lower()

                logging.info(f"#### กำลังประมวลผลไฟล์: {file_name} ({file_ext}) ####")

                # แยกข้อความจากไฟล์
                text_content = extract_text_from_file(file_path)

                if not text_content.strip():
                    failed_files.append(f"{file_name}: ไม่สามารถแยกข้อความได้")
                    continue

                # แยกข้อความเป็น chunks
                content_chunks = chunk_text(text_content, file_name)

                if not content_chunks:
                    failed_files.append(f"{file_name}: ไม่มีเนื้อหาที่สามารถประมวลผลได้")
                    continue

                # เก็บใน ChromaDB
                store_in_chroma(content_chunks, file_name)

                total_chunks += len(content_chunks)
                successful_files.append(file_name)

                logging.info(f"✅ ประมวลผล {file_name} สำเร็จ - {len(content_chunks)} chunks")

            except Exception as e:
                error_msg = f"{file_name}: {str(e)}"
                failed_files.append(error_msg)
                logging.error(f"❌ ประมวลผล {file_name} ล้มเหลว: {str(e)}")

        # สำรองข้อมูลอัตโนมัติหลังประมวลผลสำเร็จ
        backup_vector_db()

        new_count = collection.count()
        added_records = new_count - current_count

        # สร้างรายงานผลลัพธ์
        result = f"🎉 ประมวลผลเสร็จสิ้น!\n\n"
        result += f"📊 สรุปผล:\n"
        result += f"• ไฟล์ทั้งหมด: {len(files)} ไฟล์\n"
        result += f"• สำเร็จ: {len(successful_files)} ไฟล์\n"
        result += f"• ล้มเหลว: {len(failed_files)} ไฟล์\n"
        result += f"• Chunks ทั้งหมด: {total_chunks} ชิ้น\n"
        result += f"• Records ที่เพิ่ม: {added_records} เรคคอร์ด\n"
        result += f"• Records รวม: {new_count} เรคคอร์ด\n\n"

        if successful_files:
            result += f"✅ ไฟล์ที่ประมวลผลสำเร็จ:\n"
            for file_name in successful_files:
                result += f"  • {file_name}\n"
            result += "\n"

        if failed_files:
            result += f"❌ ไฟล์ที่ประมวลผลล้มเหลว:\n"
            for error in failed_files:
                result += f"  • {error}\n"
            result += "\n"

        result += f"💾 ข้อมูลถูกสำรองอัตโนมัติ"

        return result

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการประมวลผลหลายไฟล์: {str(e)}")
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"


# Google Sheets Integration
def extract_google_sheets_data(sheets_url: str) -> str:
    """
    ดึงข้อมูลจาก Google Sheets

    Args:
        sheets_url: URL ของ Google Sheets

    Returns:
        ข้อความที่จัดรูปแบบแล้ว
    """
    try:
        # แปลง URL ให้เป็น export URL
        sheet_id = extract_sheet_id_from_url(sheets_url)
        if not sheet_id:
            return "❌ ไม่สามารถแยก Sheet ID จาก URL ได้"

        # Export หน้าแรกเป็น CSV
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"

        # อ่านข้อมูลเป็น DataFrame
        df = pd.read_csv(export_url)

        # แปลงข้อมูลเป็นข้อความ
        text_content = format_dataframe_to_text(df, sheets_url)

        return text_content

    except Exception as e:
        logging.error(f"Error extracting Google Sheets data: {str(e)}")
        return f"❌ ไม่สามารถดึงข้อมูลจาก Google Sheets ได้: {str(e)}"


def extract_sheet_id_from_url(url: str) -> str:
    """
    แยก Sheet ID จาก Google Sheets URL

    Args:
        url: Google Sheets URL

    Returns:
        Sheet ID
    """
    try:
        # Pattern สำหรับ Google Sheets URL
        patterns = [
            r"/d/([a-zA-Z0-9-_]+)",
            r"id=([a-zA-Z0-9-_]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return ""
    except:
        return ""


def format_dataframe_to_text(df, source_url: str) -> str:
    """
    แปลง DataFrame เป็นข้อความที่อ่านง่ายและเหมาะกับการค้นหา
    """
    try:
        text_content = f"# ข้อมูลจาก Google Sheets\n"
        text_content += f"ที่มา: {source_url}\n"
        text_content += f"แถว: {len(df)}, คอลัมน์: {len(df.columns)}\n\n"

        # สร้างคำอธิบายคอลัมน์
        col_descriptions = []
        for i, col in enumerate(df.columns):
            col_descriptions.append(f"{col}")

        text_content += f"## หัวข้อที่ครอบคลุม: {', '.join(col_descriptions)}\n\n"

        # แปลงข้อมูลเป็นรูปแบบ Q&A และคำอธิบายที่เข้าใจง่าย
        text_content += "## รายละเอียดข้อมูล:\n\n"

        for index, row in df.iterrows():
            # สร้างคำอธิบายรวมสำหรับแถวนี้
            row_content = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value) or value == "":
                    continue

                # จัดรูปแบบค่า
                if isinstance(value, str) and len(value) > 100:
                    # ถ้าข้อความยาว ให้ตัดและเพิ่ม "..."
                    value = value[:150] + "..." if len(value) > 150 else value

                row_content.append(f"{col}: {value}")

            if row_content:
                # สร้างข้อความที่เชื่อมโยงข้อมูลในแถวเดียวกัน
                text_content += f"### ข้อมูลที่ {index + 1}:\n"
                text_content += f"{' '.join(row_content)}.\n\n"

                # สร้างรูปแบบ Q&A สำหรับการค้นหาที่ง่ายขึ้น
                if len(df.columns) >= 2:
                    # สมมติว่าคอลัมน์แรกคือคำถาม และคอลัมน์อื่นเป็นคำตอบ
                    first_col = df.columns[0]
                    question_value = row[first_col]
                    if not pd.isna(question_value) and str(question_value).strip():
                        text_content += f"**คำถาม/หัวข้อ:** {question_value}\n"

                        # รวบรวมคำตอบจากคอลัมน์อื่นๆ
                        answers = []
                        for col in df.columns[1:]:
                            answer_value = row[col]
                            if not pd.isna(answer_value) and str(answer_value).strip():
                                answers.append(f"{answer_value}")

                        if answers:
                            text_content += f"**คำตอบ/รายละเอียด:** {' '.join(answers)}\n"

                        text_content += "\n"

        return text_content

    except Exception as e:
        logging.error(f"Error formatting DataFrame: {str(e)}")
        return f"❌ เกิดข้อผิดพลาดในการจัดรูปแบบข้อมูล: {str(e)}"


def process_google_sheets_url(sheets_url: str, clear_before_upload: bool = False):
    """
    ประมวลผล Google Sheets URL

    Args:
        sheets_url: URL ของ Google Sheets
        clear_before_upload: ล้างข้อมูลเก่าหรือไม่
    """
    try:
        logging.info(f"Starting to process Google Sheets: {sheets_url}")

        # ตรวจสอบ URL
        if not sheets_url.strip():
            return "❌ กรุณาใส่ Google Sheets URL"

        # ดึงข้อมูลจาก Google Sheets
        text_content = extract_google_sheets_data(sheets_url)

        if text_content.startswith("❌"):
            return text_content

        # สร้างชื่อสำหรับ source
        sheet_name = f"Google_Sheets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # แบ่งข้อความเป็น chunks
        chunks = chunk_text(text_content, sheet_name)

        # ล้างข้อมูลเก่าถ้าจำเป็น
        if clear_before_upload:
            clear_vector_db()
            logging.info("Cleared vector database before upload")

        # เก็บ chunks ลงใน ChromaDB
        # ตรวจสอบว่าเป็นรูปแบบใหม่หรือเก่า
        if chunks and isinstance(chunks[0], dict) and "text" in chunks[0] and "metadata" in chunks[0]:
            # รูปแบบใหม่: chunks พร้อม metadata แบบละเอียด
            store_chunks_modern(chunks)
        else:
            # รูปแบบเก่า: chunks พร้อม source name
            store_chunks_legacy(chunks, sheet_name)

        # อัปเดตข้อมูลสรุป
        update_summary_data(chunks)

        result_msg = f"""✅ นำเข้าข้อมูลจาก Google Sheets สำเร็จ!
📊 URL: {sheets_url}
📝 ชื่อที่เก็บ: {sheet_name}
📄 จำนวน chunks: {len(chunks)}
📏 ความยาวทั้งหมด: {len(text_content):,} ตัวอักษร

ข้อมูลพร้อมใช้งานในระบบ RAG แล้ว!"""

        logging.info(f"Successfully processed Google Sheets: {sheet_name}")
        return result_msg

    except Exception as e:
        logging.error(f"Error processing Google Sheets URL: {str(e)}")
        return f"❌ เกิดข้อผิดพลาดในการประมวลผล Google Sheets: {str(e)}"


def chunk_text(text: str, source_file: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    แยกข้อความเป็น chunks พร้อมข้อมูล metadata

    Args:
        text: ข้อความทั้งหมด
        source_file: ชื่อไฟล์ต้นทาง
        chunk_size: ขนาดของ chunk
        overlap: ส่วนที่ทับซ้อนระหว่าง chunks

    Returns:
        List[Dict]: list ของ chunks พร้อม metadata
    """
    if not text or len(text.strip()) < 50:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # หาตำแหน่งตัดคำที่เหมาะสม
        if end < len(text):
            # พยายามตัดที่วรรค หรือ จบบรรทัด
            for i in range(end, max(start, end - 100), -1):
                if text[i] in [' ', '\n', '.', '!', '?']:
                    end = i + 1
                    break

        chunk_text = text[start:end].strip()

        if len(chunk_text) > 50:  # ต้องมีความยาวอย่างน้อย 50 ตัวอักษร
            chunk_id = f"{source_file}_{start}_{end}"

            chunks.append({
                "text": chunk_text,
                "id": chunk_id,
                "metadata": {
                    "source": source_file,
                    "start": start,
                    "end": end,
                    "file_type": Path(source_file).suffix.lower()
                }
            })

        start = end - overlap if end - overlap > start else end

    return chunks


class EnhancedRAG:
    """
    Enhanced RAG System with MemoRAG-like features:
    - Memory management for conversations
    - Context chaining
    - Session-based reasoning
    - Cross-reference analysis
    """

    def __init__(self):
        self.session_memory = deque(maxlen=MEMORY_WINDOW_SIZE)
        self.session_context = []
        self.conversation_history = []
        self.current_session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        session_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"session_{session_hash}"

    def add_to_memory(self, question: str, answer: str, contexts: List[str]):
        """Add conversation to memory"""
        memory_entry = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "contexts": contexts[:3]  # Store top 3 contexts
        }
        self.session_memory.append(memory_entry)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": memory_entry["timestamp"]
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": memory_entry["timestamp"]
        })

        logging.info(f"Added to memory: Session {self.current_session_id}")

    def get_relevant_memory(self, current_question: str) -> List[Dict]:
        """Get relevant memory entries based on current question"""
        if not ENABLE_SESSION_MEMORY:
            return []

        relevant_memories = []
        current_embedding = embed_text(current_question)

        for memory_entry in self.session_memory:
            # Calculate similarity between current question and memory
            memory_question = memory_entry["question"]
            memory_embedding = embed_text(memory_question)

            # Simple similarity check (can be improved with proper similarity calculation)
            similarity = np.dot(current_embedding, memory_embedding)

            if similarity > 0.7:  # Threshold for relevance
                relevant_memories.append({
                    "question": memory_entry["question"],
                    "answer": memory_entry["answer"],
                    "similarity": float(similarity),
                    "contexts": memory_entry["contexts"]
                })

        # Sort by similarity and return top results
        relevant_memories.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_memories[:3]  # Return top 3 relevant memories

    def build_context_prompt(self, question: str, contexts: List[str], relevant_memories: List[Dict],
                           show_source: bool = False, formal_style: bool = False) -> str:
        """Build enhanced prompt with memory and context chaining"""

        if not ENABLE_CONTEXT_CHAINING and not ENABLE_REASONING:
            # สร้าง prompt ตามสไตล์ที่เลือก
            if formal_style:
                style_instruction = "ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพ และชัดเจน"
                source_phrase = ""
                response_prefix = "คำตอบ:"
            else:
                style_instruction = "ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
                source_phrase = ""
                response_prefix = "คำตอบ:"

            source_instruction = ""
            if show_source:
                source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else "" if source_phrase else ""

            return f"""คุณเป็นผู้ช่วยตอบคำถามที่มีความรู้ด้านเอกสารที่อัปโหลดไว้ กรุณาตอบคำถามโดยอาศัยข้อมูลจากบริบทที่ให้มาเป็นหลัก

**แนวทางการตอบ:**
- {style_instruction}
- ให้คำตอบที่สอดคล้องกับข้อมูลในเอกสารเป็นหลัก
- หากข้อมูลในเอกสารไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด
- ตอบอย่างสมบูรณ์และมีประโยชน์{source_instruction}
- อาจเพิ่มคำอธิบายเล็กน้อยเพื่อความชัดเจน แต่ไม่ตีความเกินไป

**คำถาม:** {question}

**บริบทจากเอกสาร:**
{summarize}

{context}

{response_prefix}"""

        # Enhanced prompt with memory and reasoning
        prompt_parts = []

        # Add context about memory if available
        if relevant_memories:
            prompt_parts.append("## บริบทที่เกี่ยวข้อง (จาก Memory):")
            for i, memory in enumerate(relevant_memories, 1):
                prompt_parts.append(f"ครั้งที่ {i}:")
                prompt_parts.append(f"คำถาม: {memory['question']}")
                prompt_parts.append(f"คำตอบ: {memory['answer'][:200]}...")
                prompt_parts.append("")

        # Add current contexts
        if contexts:
            prompt_parts.append("## บริบทปัจจุบันจากเอกสาร:")
            prompt_parts.extend(contexts)

        # Add reasoning prompt if enabled
        if ENABLE_REASONING:
            # สร้าง prompt ตามสไตล์ที่เลือก
            if formal_style:
                style_instruction = "- ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพและชัดเจน"
                source_phrase = ""
                response_prefix = "คำตอบ:"
            else:
                style_instruction = "- ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
                source_phrase = ""
                response_prefix = "คำตอบ:"

            source_instruction = ""
            if show_source:
                source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else "" if source_phrase else ""

            prompt_parts.extend([
                "",
                "## แนวทางการตอบคำถาม:",
                style_instruction,
                "- ให้ความสำคัญกับข้อมูลจากเอกสารและบริบทที่จดจำไว้เป็นหลัก",
                "- ตอบให้สมบูรณ์และเป็นประโยชน์ โดยยังคงอยู่ในขอบเขตของข้อมูลที่มี",
                "- หากข้อมูลไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด",
                "- อาจเชื่อมโยงข้อมูลจากความจำเพื่อให้คำตอบสอดคล้องมากขึ้น" + source_instruction,
                "",
                "## การวิเคราะห์:",
                "1. พิจารณาคำถามปัจจุบันร่วมกับบริบทที่เกี่ยวข้อง",
                "2. ตรวจสอบความสอดคล้องและเชื่อมโยงข้อมูลต่างๆ",
                "3. ให้คำตอบที่ครอบคลุมและมีประโยชน์ที่สุด",
                "4. ใช้ภาษาที่เข้าใจง่ายและเป็นธรรมชาติ"
            ])

        prompt_parts.extend([
            "",
            f"## คำถามปัจจุบัน: {question}",
            "",
            f"## {response_prefix}:",
            "กรุณาตอบคำถามโดยพิจารณาจากบริบททั้งหมดที่ให้มา ให้คำตอบที่สมบูรณ์และเป็นประโยชน์ที่สุด"
        ])

        return "\n".join(prompt_parts)

    def reset_session(self):
        """Reset current session"""
        self.current_session_id = self._generate_session_id()
        self.session_context = []
        self.conversation_history = []
        logging.info(f"Session reset: New session ID = {self.current_session_id}")

    def get_session_info(self) -> Dict:
        """Get current session information"""
        return {
            "session_id": self.current_session_id,
            "memory_size": len(self.session_memory),
            "conversation_length": len(self.conversation_history),
            "current_context_size": len(self.session_context)
        }


# Global Enhanced RAG instance
enhanced_rag = EnhancedRAG()


def process_pdf_upload(pdf_file):
    """
    ฟังก์ชันเก่าสำหรับความเข้ากันได้ (deprecated - ใช้ process_multiple_files แทน)
    """
    if pdf_file:
        return process_multiple_files([pdf_file], False)
    return "กรุณาเลือกไฟล์"

def clear_vector_db():
    try:
        
       # Clear existing collection to avoid duplicates
        chroma_client.delete_collection(name="pdf_data")
        global collection
        collection = chroma_client.create_collection(name="pdf_data")

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"
    
def update_summary_data(chunks: List[Dict]):
    """
    อัปเดตข้อมูลสรุปจาก chunks ที่อัปโหลด
    """
    try:
        global summarize

        # สรุปข้อมูลจาก chunks
        total_chars = sum(len(chunk.get('text', '')) for chunk in chunks)
        source_files = set(chunk.get('metadata', {}).get('source', 'Unknown') for chunk in chunks)

        # สร้างข้อความสรุป
        summary_text = f"""📊 ข้อมูลที่อัปโหลด:
• แหล่งที่มา: {', '.join(source_files)}
• จำนวน chunks: {len(chunks)}
• ขนาดรวม: {total_chars:,} ตัวอักษร
• อัปเดตล่าสุด: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        summarize = summary_text
        logging.info(f"Updated summary data: {len(chunks)} chunks from {len(source_files)} sources")

    except Exception as e:
        logging.error(f"Error updating summary data: {str(e)}")
        # ถ้าเกิดข้อผิดพลาด ให้ใช้ค่าเริ่มต้น
        summarize = f"📊 ข้อมูล PDF: {len(chunks)} chunks, อัปเดต {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def clear_vector_db_and_images():
    """
    ล้างข้อมูลใน Chroma vector database และไฟล์ในโฟลเดอร์ images
    """
    
    try:
        clear_vector_db()
        
        pdf_input.clear()
        if os.path.exists(TEMP_IMG):
            shutil.rmtree(TEMP_IMG)
            os.makedirs(TEMP_IMG, exist_ok=True)
        
        return "ล้างข้อมูลใน vector database และโฟลเดอร์ images สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"


async def send_to_discord(question: str, answer: str):
    """
    ส่งคำถามและคำตอบไปยัง Discord channel
    """
    if not DISCORD_ENABLED or DISCORD_WEBHOOK_URL == "YOUR_WEBHOOK_URL_HERE":
        logging.info("Discord integration is disabled or not configured")
        return

    try:
        # ใช้ Webhook URL โดยตรง
        webhook_url = DISCORD_WEBHOOK_URL

        embed = discord.Embed(
            title="📚 RAG PDF Bot - คำถามใหม่",
            color=discord.Color.blue()
        )
        embed.add_field(name="❓ คำถาม", value=question, inline=False)
        embed.add_field(name="💬 คำตอบ", value=answer[:1024] + "..." if len(answer) > 1024 else answer, inline=False)
        embed.set_footer(text="PDF RAG Assistant")

        # ใช้ requests แทน discord client ที่ต้องทำงานใน event loop
        payload = {
            "embeds": [embed.to_dict()]
        }

        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            logging.info("ส่งข้อความไปยัง Discord สำเร็จ")
        else:
            logging.error(f"ไม่สามารถส่งข้อความไปยัง Discord: {response.status_code} - {response.text}")

    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่งข้อความไปยัง Discord: {str(e)}")


def send_to_discord_sync(question: str, answer: str):
    """
    ฟังก์ชันสำหรับเรียกใช้ discord แบบ synchronous
    """
    try:
        # สร้าง event loop ใหม่ถ้าจำเป็น
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # ถ้า loop กำลังทำงานอยู่ ใช้ create_task
            asyncio.create_task(send_to_discord(question, answer))
        else:
            # ถ้า loop ไม่ทำงาน ใช้ run_until_complete
            loop.run_until_complete(send_to_discord(question, answer))
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการเรียก Discord: {str(e)}")


def backup_vector_db():
    """
    สำรองข้อมูล vector database
    """
    try:
        if not os.path.exists(TEMP_VECTOR):
            logging.warning("ไม่พบโฟลเดอร์ database ที่จะสำรองข้อมูล")
            return False

        # สร้าง timestamp สำหรับ backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(TEMP_VECTOR_BACKUP, f"backup_{timestamp}")

        # คัดลอกข้อมูลไปโฟลเดอร์ backup
        shutil.copytree(TEMP_VECTOR, backup_folder)
        logging.info(f"สำรองข้อมูลสำเร็จ: {backup_folder}")

        # ลบ backup เก่าเกินกว่า 7 วัน
        cleanup_old_backups()

        return True
    except Exception as e:
        logging.error(f"ไม่สามารถสำรองข้อมูลได้: {str(e)}")
        return False


def cleanup_old_backups(days_to_keep=7):
    """
    ลบข้อมูล backup ที่เก่าเกินกว่าที่กำหนด
    """
    try:
        now = datetime.now()

        for backup_name in os.listdir(TEMP_VECTOR_BACKUP):
            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            if os.path.isdir(backup_path) and backup_name.startswith("backup_"):
                # ดึง timestamp จากชื่อโฟลเดอร์
                try:
                    timestamp_str = backup_name.replace("backup_", "")
                    backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                    # ลบถ้าเกินกว่าจำนวนวันที่กำหนด
                    if (now - backup_time).days > days_to_keep:
                        shutil.rmtree(backup_path)
                        logging.info(f"ลบ backup เก่า: {backup_name}")
                except ValueError:
                    continue
    except Exception as e:
        logging.error(f"ไม่สามารถลบ backup เก่าได้: {str(e)}")


def backup_database_enhanced(backup_name=None, include_memory=False):
    """
    สำรองข้อมูล database แบบขั้นสูง
    """
    try:
        import json

        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"enhanced_backup_{timestamp}"

        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
        os.makedirs(backup_path, exist_ok=True)

        # Backup main database
        if os.path.exists(TEMP_VECTOR):
            backup_db_path = os.path.join(backup_path, "chromadb")
            shutil.copytree(TEMP_VECTOR, backup_db_path)
            logging.info(f"สำรอง ChromaDB: {backup_db_path}")

        # Backup Enhanced RAG memory if requested
        if include_memory and RAG_MODE == "enhanced":
            try:
                memory_info = enhanced_rag.get_memory_info()
                backup_memory_path = os.path.join(backup_path, "memory_info.json")

                with open(backup_memory_path, 'w', encoding='utf-8') as f:
                    json.dump(memory_info, f, ensure_ascii=False, indent=2)

                # Also backup memory collection if it exists
                if hasattr(enhanced_rag, 'memory_collection'):
                    memory_collection_backup = os.path.join(backup_path, "memory_collection")
                    os.makedirs(memory_collection_backup, exist_ok=True)

                    # Get all memories from collection
                    memories = enhanced_rag.memory_collection.get()
                    memory_json_path = os.path.join(memory_collection_backup, "memories.json")

                    with open(memory_json_path, 'w', encoding='utf-8') as f:
                        json.dump(memories, f, ensure_ascii=False, indent=2)

                logging.info("สำรอง Enhanced RAG Memory สำเร็จ")
            except Exception as e:
                logging.warning(f"ไม่สามารถสำรอง Memory: {str(e)}")

        # Create backup metadata
        metadata = {
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "type": "enhanced",
            "includes_memory": include_memory,
            "rag_mode": RAG_MODE,
            "database_info": get_database_info() if os.path.exists(TEMP_VECTOR) else None
        }

        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"สร้าง Enhanced Backup สำเร็จ: {backup_path}")

        # Clean old backups
        cleanup_old_backups()

        return {
            "success": True,
            "backup_name": backup_name,
            "backup_path": backup_path,
            "metadata": metadata
        }

    except Exception as e:
        logging.error(f"ไม่สามารถสร้าง Enhanced Backup: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def restore_database_enhanced(backup_name=None):
    """
    กู้คืน database แบบขั้นสูง
    """
    try:
        import json

        if backup_name is None:
            # หา backup ล่าสุด
            backups = [d for d in os.listdir(TEMP_VECTOR_BACKUP)
                      if os.path.isdir(os.path.join(TEMP_VECTOR_BACKUP, d))]
            if not backups:
                return {"success": False, "error": "ไม่พบข้อมูล backup"}
            backup_name = sorted(backups)[-1]

        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            return {"success": False, "error": f"ไม่พบ backup: {backup_name}"}

        # Validate backup integrity before restore
        is_valid, validation_message = validate_backup_integrity(backup_path)
        if not is_valid:
            logging.error(f"Backup validation failed: {validation_message}")
            return {"success": False, "error": f"Backup ไม่ถูกต้อง: {validation_message}"}

        logging.info(f"Backup validation passed: {backup_name}")

        # Load backup metadata
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logging.info(f"Loaded backup metadata from: {metadata_path}")
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON in backup metadata: {str(e)}")
                metadata = {
                    "backup_name": backup_name,
                    "type": "unknown",
                    "includes_memory": False,
                    "rag_mode": "unknown"
                }
            except Exception as e:
                logging.warning(f"Could not read backup metadata: {str(e)}")
                metadata = {
                    "backup_name": backup_name,
                    "type": "unknown",
                    "includes_memory": False,
                    "rag_mode": "unknown"
                }

        # Create emergency backup of current data
        try:
            emergency_backup = backup_database_enhanced(
                f"emergency_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if not emergency_backup["success"]:
                logging.warning(f"Emergency backup failed: {emergency_backup.get('error')}")
                emergency_backup = {"success": False, "error": "Emergency backup failed"}
        except Exception as e:
            logging.error(f"Emergency backup creation failed: {str(e)}")
            emergency_backup = {"success": False, "error": str(e)}

        # Restore main database
        backup_db_path = os.path.join(backup_path, "chromadb")
        if os.path.exists(backup_db_path):
            if os.path.exists(TEMP_VECTOR):
                shutil.rmtree(TEMP_VECTOR)
            shutil.copytree(backup_db_path, TEMP_VECTOR)
            logging.info(f"กู้คืน ChromaDB สำเร็จ")

        # Restore Enhanced RAG memory if available
        memory_collection_backup = os.path.join(backup_path, "memory_collection")
        if os.path.exists(memory_collection_backup) and RAG_MODE == "enhanced":
            try:
                memory_json_path = os.path.join(memory_collection_backup, "memories.json")
                if os.path.exists(memory_json_path):
                    try:
                        with open(memory_json_path, 'r', encoding='utf-8') as f:
                            memories = json.load(f)

                        # Validate memories structure
                        if all(key in memories for key in ['ids', 'documents', 'metadatas']):
                            # Clear current memory and restore from backup
                            if hasattr(enhanced_rag, 'memory_collection'):
                                enhanced_rag.memory_collection.delete()
                                enhanced_rag.memory_collection.add(
                                    ids=memories['ids'],
                                    documents=memories['documents'],
                                    metadatas=memories['metadatas']
                                )
                                logging.info("กู้คืน Enhanced RAG Memory สำเร็จ")
                        else:
                            logging.warning("Invalid memory backup structure - missing required keys")
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON in memory backup: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Could not restore memory backup: {str(e)}")

            except Exception as e:
                logging.warning(f"ไม่สามารถกู้คืน Memory: {str(e)}")

        # Reload collections
        global collection
        collection = chroma_client.get_or_create_collection(name="pdf_data")

        result = {
            "success": True,
            "backup_name": backup_name,
            "restored_at": datetime.now().isoformat(),
            "metadata": metadata,
            "emergency_backup": emergency_backup
        }

        logging.info(f"กู้คืน Database สำเร็จจาก: {backup_name}")
        return result

    except Exception as e:
        logging.error(f"ไม่สามารถกู้คืน Database: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def is_valid_backup_folder(backup_path):
    """
    ตรวจสอบว่าโฟลเดอร์เป็น backup ที่ถูกต้องหรือไม่
    """
    if not os.path.isdir(backup_path):
        return False

    # Check if it's a valid backup folder (has chromadb or backup_metadata.json)
    chromadb_path = os.path.join(backup_path, "chromadb")
    metadata_path = os.path.join(backup_path, "backup_metadata.json")

    return os.path.exists(chromadb_path) or os.path.exists(metadata_path)


def list_available_backups():
    """
    แสดงรายการ backup ที่มีอยู่ทั้งหมด
    """
    try:
        import json

        backups = []

        if not os.path.exists(TEMP_VECTOR_BACKUP):
            return backups

        for backup_name in sorted(os.listdir(TEMP_VECTOR_BACKUP), reverse=True):
            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

            # Only include valid backup folders
            if not is_valid_backup_folder(backup_path):
                continue

            # Try to load metadata
            metadata = {}
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass

            # Get backup size
            total_size = 0
            for root, dirs, files in os.walk(backup_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)

            backup_info = {
                "name": backup_name,
                "path": backup_path,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "created_at": metadata.get("created_at", "Unknown"),
                "type": metadata.get("type", "standard"),
                "includes_memory": metadata.get("includes_memory", False),
                "rag_mode": metadata.get("rag_mode", "unknown"),
                "database_info": metadata.get("database_info", {})
            }

            backups.append(backup_info)

        return backups

    except Exception as e:
        logging.error(f"ไม่สามารถแสดงรายการ backup: {str(e)}")
        return []


def delete_backup(backup_name):
    """
    ลบ backup ที่ระบุ
    """
    try:
        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            return {"success": False, "error": f"ไม่พบ backup: {backup_name}"}

        if not os.path.isdir(backup_path):
            return {"success": False, "error": f"ไม่ใช่โฟลเดอร์ backup: {backup_name}"}

        shutil.rmtree(backup_path)
        logging.info(f"ลบ backup สำเร็จ: {backup_name}")

        return {"success": True, "message": f"ลบ backup {backup_name} สำเร็จ"}

    except Exception as e:
        logging.error(f"ไม่สามารถลบ backup: {str(e)}")
        return {"success": False, "error": str(e)}


def validate_backup_integrity(backup_path):
    """
    ตรวจสอบความสมบูรณ์ของ backup
    """
    try:
        # Check main database
        chromadb_path = os.path.join(backup_path, "chromadb")
        if os.path.exists(chromadb_path):
            # Check if essential ChromaDB files exist
            sqlite_file = os.path.join(chromadb_path, "chroma.sqlite3")
            if not os.path.exists(sqlite_file):
                return False, "Missing chroma.sqlite3 file"

        # Check backup metadata JSON
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Try to parse JSON
            except json.JSONDecodeError:
                return False, "Invalid backup metadata JSON"

        # Check memory backup JSON if exists
        memory_collection_path = os.path.join(backup_path, "memory_collection")
        if os.path.exists(memory_collection_path):
            memory_json_path = os.path.join(memory_collection_path, "memories.json")
            if os.path.exists(memory_json_path):
                try:
                    with open(memory_json_path, 'r', encoding='utf-8') as f:
                        memories = json.load(f)
                        # Check if memory structure is valid
                        required_keys = ['ids', 'documents', 'metadatas']
                        if not all(key in memories for key in required_keys):
                            return False, "Invalid memory backup structure"
                except json.JSONDecodeError:
                    return False, "Invalid memory backup JSON"

        return True, "Backup is valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def cleanup_invalid_backups():
    """
    ลบโฟลเดอร์ที่ไม่ใช่ backup ที่ถูกต้อง
    """
    try:
        if not os.path.exists(TEMP_VECTOR_BACKUP):
            return

        cleaned_count = 0
        for folder_name in os.listdir(TEMP_VECTOR_BACKUP):
            folder_path = os.path.join(TEMP_VECTOR_BACKUP, folder_name)
            if os.path.isdir(folder_path):
                # Check if it's a valid backup folder
                if not is_valid_backup_folder(folder_path):
                    shutil.rmtree(folder_path)
                    cleaned_count += 1
                    logging.info(f"ลบโฟลเดอร์ที่ไม่ถูกต้อง: {folder_name}")
                else:
                    # Additional validation for backup integrity
                    is_valid, message = validate_backup_integrity(folder_path)
                    if not is_valid:
                        logging.warning(f"Backup {folder_name} failed validation: {message}")
                        # Don't delete automatically, just warn

        if cleaned_count > 0:
            logging.info(f"ลบโฟลเดอร์ที่ไม่ถูกต้องทั้งหมด {cleaned_count} โฟลเดอร์")

    except Exception as e:
        logging.error(f"ไม่สามารถ cleanup invalid backups: {str(e)}")


def auto_backup_before_operation():
    """
    สร้าง backup อัตโนมัติก่อนการดำเนินการที่สำคัญ
    """
    try:
        # Clean invalid backups first
        cleanup_invalid_backups()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"auto_backup_{timestamp}"

        result = backup_database_enhanced(
            backup_name=backup_name,
            include_memory=(RAG_MODE == "enhanced")
        )

        if result["success"]:
            logging.info(f"สร้าง Auto Backup สำเร็จ: {backup_name}")
        else:
            logging.warning(f"สร้าง Auto Backup ไม่สำเร็จ: {result.get('error')}")

        return result

    except Exception as e:
        logging.error(f"ไม่สามารถสร้าง Auto Backup: {str(e)}")
        return {"success": False, "error": str(e)}


def restore_vector_db(backup_name=None):
    """
    กู้คืนข้อมูลจาก backup

    Args:
        backup_name: ชื่อโฟลเดอร์ backup (ถ้าไม่ระบุจะใช้ล่าสุด)
    """
    try:
        if backup_name is None:
            # หา backup ล่าสุด
            backups = [d for d in os.listdir(TEMP_VECTOR_BACKUP)
                      if d.startswith("backup_") and os.path.isdir(os.path.join(TEMP_VECTOR_BACKUP, d))]
            if not backups:
                logging.error("ไม่พบข้อมูล backup")
                return False
            backup_name = sorted(backups)[-1]

        backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)

        if not os.path.exists(backup_path):
            logging.error(f"ไม่พบ backup: {backup_name}")
            return False

        # สำรองข้อมูลปัจจุบันก่อน restore
        backup_vector_db()

        # ลบข้อมูลปัจจุบันและกู้คืนจาก backup
        if os.path.exists(TEMP_VECTOR):
            shutil.rmtree(TEMP_VECTOR)

        shutil.copytree(backup_path, TEMP_VECTOR)
        logging.info(f"กู้คืนข้อมูลสำเร็จจาก: {backup_name}")

        # รีโหลด collection
        global collection
        collection = chroma_client.get_or_create_collection(name="pdf_data")

        return True
    except Exception as e:
        logging.error(f"ไม่สามารถกู้คืนข้อมูลได้: {str(e)}")
        return False


def get_database_info():
    """
    แสดงข้อมูลสถานะ database อย่างละเอียด
    """
    try:
        # ข้อมูลพื้นฐาน
        count = collection.count()

        # ข้อมูลไฟล์
        db_exists = os.path.exists(TEMP_VECTOR)
        sqlite_exists = os.path.exists(os.path.join(TEMP_VECTOR, "chroma.sqlite3"))

        # ขนาด database
        db_size = 0
        if db_exists:
            for root, dirs, files in os.walk(TEMP_VECTOR):
                db_size += sum(os.path.getsize(os.path.join(root, name)) for name in files)

        # จำนวน backup
        backup_count = 0
        if os.path.exists(TEMP_VECTOR_BACKUP):
            backup_count = len([d for d in os.listdir(TEMP_VECTOR_BACKUP)
                               if d.startswith("backup_")])

        info = {
            "total_records": count,
            "database_path": TEMP_VECTOR,
            "database_exists": db_exists,
            "sqlite_exists": sqlite_exists,
            "database_size_mb": round(db_size / (1024 * 1024), 2),
            "backup_count": backup_count,
            "collections": list(chroma_client.list_collections())
        }

        logging.info(f"📊 Database Info: {count} records, {round(db_size/(1024*1024),2)}MB, {backup_count} backups")
        return info

    except Exception as e:
        logging.error(f"❌ ไม่สามารถดึงข้อมูล database ได้: {str(e)}")
        return {"error": str(e)}


def inspect_database():
    """
    ตรวจสอบข้อมูลภายใน database อย่างละเอียด
    """
    try:
        count = collection.count()
        if count == 0:
            return "Database ว่างเปล่า - ไม่มีข้อมูล"

        # ดึงข้อมูลตัวอย่าง 3 records
        sample_data = collection.get(limit=3, include=["documents", "metadatas"])

        result = f"📊 Database Inspection:\n"
        result += f"📝 Total Records: {count}\n"
        result += f"📁 Collections: {list(chroma_client.list_collections())}\n\n"

        result += "📋 Sample Data (first 3 records):\n"
        for i, (doc, meta) in enumerate(zip(sample_data["documents"][:3], sample_data["metadatas"][:3])):
            result += f"\n{i+1}. Document: {doc[:100]}...\n"
            result += f"   Metadata: {meta}\n"
            result += f"   ---"

        return result

    except Exception as e:
        return f"❌ ตรวจสอบ database ล้มเหลว: {str(e)}"


# Discord Bot for receiving questions
class RAGPDFBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # เปิดการอ่านข้อความ
        super().__init__(intents=intents)
        self.is_ready = False

    async def on_ready(self):
        """เมื่อ Bot เชื่อมต่อสำเร็จ"""
        logging.info(f'Bot เชื่อมต่อสำเร็จเป็น {self.user}')
        self.is_ready = True
        # ตั้งสถานะของ Bot
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{DISCORD_BOT_PREFIX}คำถาม"
            )
        )

    async def on_message(self, message):
        """รับข้อความจาก Discord"""
        # ไม่ตอบข้อความของตัวเอง
        if message.author == self.user:
            return

        # ตรวจสอบว่าควรตอบหรือไม่
        should_respond = False
        question = ""
        response_type = ""

        # กรณีที่ 1: มี prefix (เช่น !ask)
        if message.content.startswith(DISCORD_BOT_PREFIX):
            question = message.content[len(DISCORD_BOT_PREFIX):].strip()
            should_respond = True
            response_type = "prefix"

        # กรณีที่ 2: ถูก mention bot (เช่น @RAGPDFBot)
        elif self.user.mentioned_in(message):
            # ลบ mention ออกจากข้อความ
            question = message.content.replace(f'<@!{self.user.id}>', '').replace(f'<@{self.user.id}>', '').strip()
            should_respond = True
            response_type = "mention"

        # กรณีที่ 3: ไม่มี prefix แต่มีการตั้งค่าให้ตอบทุกข้อความ
        elif DISCORD_RESPOND_NO_PREFIX:
            question = message.content.strip()
            should_respond = True
            response_type = "auto"

        # ถ้าไม่ต้องการตอบ ให้จบการทำงาน
        if not should_respond or not question:
            return

        # ตรวจสอบว่ามีคำถามหรือไม่
        if not question:
            if response_type == "prefix":
                await message.reply(
                    "❌ กรุณาระบุคำถาม\n"
                    f"ตัวอย่าง: `{DISCORD_BOT_PREFIX}PDF นี้เกี่ยวกับอะไร`"
                )
            elif response_type == "mention":
                await message.reply(
                    "❌ กรุณาระบุคำถามหลังจาก mention\n"
                    f"ตัวอย่าง: `@{self.user.name} PDF นี้เกี่ยวกับอะไร`"
                )
            else:
                await message.reply(
                    "❌ กรุณาระบุคำถาม\n"
                    f"ตัวอย่าง: `PDF นี้เกี่ยวกับอะไร`"
                )
            return

        # แสดงสถานะกำลังประมวลผล
        logging.info(f"Discord Bot: รับคำถาม ({response_type}) - {question}")
        processing_msg = await message.reply("🔍 กำลังค้นหาคำตอบ...")

        try:
            # เรียกใช้ RAG system
            stream = query_rag(question, chat_llm=DISCORD_DEFAULT_MODEL)

            # รวบรวมคำตอบ
            full_answer = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                full_answer += content

            # จัดรูปแบบคำตอบ (จำกัดความยาวสำหรับ Discord)
            if len(full_answer) > 1990:  # Discord จำกัด 2000 ตัวอักษร
                full_answer = full_answer[:1980] + "...\n\n*คำตอบถูกตัดเนื่องจากความยาวเกินขีดจำกัด*"

            # สร้าง embed สำหรับคำตอบ
            embed = discord.Embed(
                title="",
                description=full_answer,
                color=discord.Color.blue()
            )

            # embed.add_field(name="❓ คำถาม", value=question, inline=False)
            # embed.set_footer(text="PDF RAG Assistant • ข้อมูลจาก PDF ที่อัปโหลด")
            # embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/2951/2951136.png")

            # ลบข้อความกำลังประมวลผล
            await processing_msg.delete()

            # ส่งคำตอบตามโหมดที่กำหนด
            await respond_to_discord_message(message, embed, DISCORD_REPLY_MODE)

            logging.info(f"Discord Bot: ตอบคำถามเรียบร้อย (โหมด: {DISCORD_REPLY_MODE})")

        except Exception as e:
            # ลบข้อความกำลังประมวลผล
            await processing_msg.delete()

            error_embed = discord.Embed(
                title="❌ เกิดข้อผิดพลาด",
                description=f"ไม่สามารถตอบคำถามได้: {str(e)}",
                color=discord.Color.red()
            )
            await respond_to_discord_message(message, error_embed, DISCORD_REPLY_MODE)
            logging.error(f"Discord Bot error: {str(e)}")


async def send_discord_dm(user, embed):
    """ส่ง Direct Message ให้ผู้ใช้ Discord"""
    try:
        await user.send(embed=embed)
        return True
    except discord.Forbidden:
        logging.warning(f"ไม่สามารถส่ง DM ให้ผู้ใช้ {user.name} ได้ (อาจปิดการรับ DM)")
        return False
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการส่ง DM: {str(e)}")
        return False


async def respond_to_discord_message(message, embed, reply_type="channel"):
    """ตอบกลับข้อความใน Discord ตามโหมดที่กำหนด"""
    success_dm = False
    success_channel = False

    # ส่งใน channel (เฉพาะโหมด channel หรือ both)
    if reply_type in ["channel", "both"]:
        try:
            await message.reply(embed=embed)
            success_channel = True
        except Exception as e:
            logging.error(f"ไม่สามารถตอบใน channel: {str(e)}")

    # ส่ง DM (เฉพาะโหมด dm หรือ both)
    if reply_type in ["dm", "both"]:
        success_dm = await send_discord_dm(message.author, embed)

    # ถ้าส่ง DM ไม่ได้แต่เลือกโหมด dm ให้ส่งใน channel แทน
    if reply_type == "dm" and not success_dm:
        try:
            fallback_embed = discord.Embed(
                title="📬 คำตอบของคุณ",
                description=embed.description,
                color=embed.color
            )
            await message.reply(embed=fallback_embed)
            logging.info("ส่งคำตอบใน channel แทน เนื่องจากไม่สามารถส่ง DM ได้")
        except Exception as e:
            logging.error(f"ส่งข้อความ fallback ไม่ได้: {str(e)}")


# Global variables for Discord Bot
discord_bot = None
discord_bot_thread = None


def start_discord_bot():
    """เริ่มทำงาน Discord Bot"""
    global discord_bot

    if not DISCORD_BOT_ENABLED or DISCORD_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        logging.info("Discord Bot ไม่ได้เปิดใช้งานหรือยังไม่ได้ตั้งค่า")
        return False

    try:
        discord_bot = RAGPDFBot()

        # สร้าง event loop ใหม่สำหรับ bot
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # เรียกใช้ bot
        loop.run_until_complete(discord_bot.start(DISCORD_BOT_TOKEN))

    except Exception as e:
        logging.error(f"ไม่สามารถเริ่ม Discord Bot ได้: {str(e)}")
        return False


def start_discord_bot_thread():
    """เริ่ม Discord Bot ใน thread แยก"""
    global discord_bot_thread

    if discord_bot_thread and discord_bot_thread.is_alive():
        logging.warning("Discord Bot กำลังทำงานอยู่แล้ว")
        return False

    discord_bot_thread = threading.Thread(target=start_discord_bot, daemon=True)
    discord_bot_thread.start()

    # รอสักครู่ให้ bot เริ่มทำงาน
    import time
    time.sleep(2)

    return True


def stop_discord_bot():
    """หยุด Discord Bot"""
    global discord_bot

    if discord_bot and discord_bot.is_ready:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(discord_bot.close())
            else:
                loop.run_until_complete(discord_bot.close())
            logging.info("Discord Bot หยุดทำงานแล้ว")
            return True
        except Exception as e:
            logging.error(f"ไม่สามารถหยุด Discord Bot ได้: {str(e)}")
            return False

    return False


# Flask App for LINE OA and Facebook Messenger
app = Flask(__name__)

# LINE OA Setup
line_bot_api = None
line_handler = None
line_thread = None

# Facebook Messenger Setup
fb_thread = None


def setup_line_bot():
    """ตั้งค่า LINE Bot"""
    global line_bot_api, line_handler
    if LINE_ENABLED and LINE_CHANNEL_ACCESS_TOKEN != "YOUR_LINE_CHANNEL_ACCESS_TOKEN":
        try:
            line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
            line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

            # Register handlers หลังจาก setup แล้วเท่านั้น
            register_line_handlers()

            logging.info("✅ LINE Bot setup completed")
            return True
        except Exception as e:
            logging.error(f"❌ LINE Bot setup failed: {str(e)}")
            return False
    else:
        logging.info("LINE Bot is disabled or not configured")
        return False


def register_line_handlers():
    """Register LINE message handlers"""
    if line_handler:
        @line_handler.add(MessageEvent, message=TextMessage)
        def handle_line_message(event):
            """จัดการข้อความจาก LINE"""
            try:
                user_message = event.message.text
                user_id = event.source.user_id

                logging.info(f"LINE Bot: รับคำถามจาก {user_id} - {user_message}")

                # ตอบกลับเพื่อแสดงว่ากำลังประมวลผล
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="🔍 กำลังค้นหาคำตอบ...")
                )

                # ประมวลผลคำถามใน background
                threading.Thread(
                    target=process_line_question,
                    args=(event, user_message, user_id)
                ).start()

            except Exception as e:
                logging.error(f"LINE Bot error: {str(e)}")
                try:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text="❌ เกิดข้อผิดพลาด กรุณาลองใหม่")
                    )
                except:
                    pass


def setup_facebook_bot():
    """ตั้งค่า Facebook Messenger Bot"""
    if FB_ENABLED and FB_PAGE_ACCESS_TOKEN != "YOUR_FB_PAGE_ACCESS_TOKEN":
        logging.info("✅ Facebook Messenger Bot setup completed")
        return True
    else:
        logging.info("Facebook Messenger Bot is disabled or not configured")
        return False


@app.route("/callback", methods=['POST'])
def line_callback():
    """LINE Webhook Callback"""
    if not LINE_ENABLED or not line_handler:
        abort(400)

    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'




def process_line_question(event, user_id: str, question: str):
    """ประมวลผลคำถามจาก LINE"""
    try:
        # เรียกใช้ฟังก์ชัน RAG เพื่อตอบคำถาม
        response = query_rag(question, LINE_DEFAULT_MODEL, show_source=False)
        answer = response.get('answer', 'ไม่พบคำตอบ')

        # จำกัดความยาวข้อความสำหรับ LINE (สูงสุด 5000 ตัวอักษร)
        if len(answer) > 4900:
            answer = answer[:4900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

        # ส่งคำตอบกลับไปยัง LINE
        line_bot_api.push_message(
            user_id,
            TextSendMessage(text=answer)
        )

        logging.info(f"LINE Bot: ตอบคำถามเรียบร้อย")

    except Exception as e:
        logging.error(f"LINE processing error: {str(e)}")
        try:
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text="ขออภัย เกิดข้อผิดพลาดในการค้นหาคำตอบ กรุณาลองใหม่อีกครั้ง")
            )
        except:
            pass


@app.route("/webhook", methods=['GET', 'POST'])
def facebook_webhook():
    """Facebook Messenger Webhook"""
    if not FB_ENABLED:
        abort(403)

    if request.method == 'GET':
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
            if not request.args.get("hub.verify_token") == FB_VERIFY_TOKEN:
                return "Verification token mismatch", 403
            return request.args.get("hub.challenge"), 200
        return "Hello", 200

    elif request.method == 'POST':
        data = request.get_json()

        if data and "object" in data and data["object"] == "page":
            for entry in data["entry"]:
                for messaging_event in entry["messaging"]:
                    if messaging_event.get("message"):
                        sender_id = messaging_event["sender"]["id"]
                        message_text = messaging_event["message"].get("text")

                        if message_text:
                            # ใช้ thread แยกเพื่อประมวลผลคำถาม
                            threading.Thread(
                                target=process_facebook_question,
                                args=(sender_id, message_text),
                                daemon=True
                            ).start()

        return "OK", 200


def process_facebook_question(sender_id: str, question: str):
    """ประมวลผลคำถามจาก Facebook Messenger"""
    try:
        logging.info(f"Facebook Bot: รับคำถามจาก {sender_id} - {question}")

        # ส่งข้อความกำลังประมวลผล
        send_facebook_message(sender_id, "กำลังค้นหาคำตอบให้คุณ...")

        # เรียกใช้ฟังก์ชัน RAG เพื่อตอบคำถาม
        response = query_rag(question, FB_DEFAULT_MODEL, show_source=False)
        answer = response.get('answer', 'ไม่พบคำตอบ')

        # จำกัดความยาวข้อความสำหรับ Facebook (สูงสุด 2000 ตัวอักษร)
        if len(answer) > 1900:
            answer = answer[:1900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

        # ส่งคำตอบกลับไปยัง Facebook
        send_facebook_message(sender_id, answer)

        logging.info(f"Facebook Bot: ตอบคำถามเรียบร้อย")

    except Exception as e:
        logging.error(f"Facebook processing error: {str(e)}")
        try:
            send_facebook_message(sender_id, "ขออภัย เกิดข้อผิดพลาดในการค้นหาคำตอบ กรุณาลองใหม่อีกครั้ง")
        except:
            pass


def send_facebook_message(recipient_id: str, message_text: str):
    """ส่งข้อความไปยัง Facebook Messenger"""
    try:
        url = f"https://graph.facebook.com/v18.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"

        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text}
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

    except Exception as e:
        logging.error(f"Error sending Facebook message: {str(e)}")


def start_line_server():
    """เริ่ม LINE Bot Server"""
    if setup_line_bot():
        try:
            app.run(host='0.0.0.0', port=LINE_WEBHOOK_PORT, debug=False)
        except Exception as e:
            logging.error(f"LINE server error: {str(e)}")


def start_facebook_server():
    """เริ่ม Facebook Bot Server"""
    if setup_facebook_bot():
        try:
            # Facebook ใช้ port เดียวกับ LINE ถ้า LINE ไม่ได้เปิดใช้งาน
            port = FB_WEBHOOK_PORT if LINE_ENABLED else LINE_WEBHOOK_PORT
            app.run(host='0.0.0.0', port=port, debug=False)
        except Exception as e:
            logging.error(f"Facebook server error: {str(e)}")


def start_line_bot_thread():
    """เริ่ม LINE Bot ใน thread แยก"""
    global line_thread
    if line_thread and line_thread.is_alive():
        logging.warning("LINE Bot กำลังทำงานอยู่แล้ว")
        return False

    if not LINE_ENABLED:
        logging.info("LINE Bot is disabled")
        return False

    try:
        line_thread = threading.Thread(target=start_line_server, daemon=True)
        line_thread.start()
        logging.info(f"LINE Bot server started on port {LINE_WEBHOOK_PORT}")
        return True
    except Exception as e:
        logging.error(f"Failed to start LINE Bot: {str(e)}")
        return False


def start_facebook_bot_thread():
    """เริ่ม Facebook Bot ใน thread แยก"""
    global fb_thread
    if fb_thread and fb_thread.is_alive():
        logging.warning("Facebook Bot กำลังทำงานอยู่แล้ว")
        return False

    if not FB_ENABLED:
        logging.info("Facebook Bot is disabled")
        return False

    try:
        fb_thread = threading.Thread(target=start_facebook_server, daemon=True)
        fb_thread.start()
        logging.info(f"Facebook Bot server started on port {FB_WEBHOOK_PORT}")
        return True
    except Exception as e:
        logging.error(f"Failed to start Facebook Bot: {str(e)}")
        return False


def determine_optimal_results(question: str) -> int:
    """
    กำหนดจำนวนผลลัพธ์ที่เหมาะสมตามประเภทคำถาม
    """
    question_lower = question.lower()

    # คำถามที่ต้องการความครบถ้วน - ใช้จำนวนมากขึ้น
    comprehensive_keywords = ["ทั้งหมด", "ทั้งหมดทั้งสิ้น", "ทุก", "ทุกอย่าง", "สรุปทั้งหมด", "ทั้งหมดเลย"]
    if any(keyword in question_lower for keyword in comprehensive_keywords):
        return 8

    # คำถามแบบนับจำนวน - ใช้ปานกลาง
    counting_keywords = ["กี่", "จำนวน", "มีกี่", "กี่ตัว", "จำนวนเท่าไร", "มีกี่อย่าง"]
    if any(keyword in question_lower for keyword in counting_keywords):
        return 5

    # คำถามแบบเลือกบางส่วน - ใช้ปานกลาง
    selective_keywords = ["บ้าง", "บางส่วน", "บางอย่าง", "ตัวอย่าง", "กรณี", "เช่น", "ดังนี้"]
    if any(keyword in question_lower for keyword in selective_keywords):
        return 4

    # คำถามเฉพาะเจาะจง - ใช้จำนวนน้อย
    specific_keywords = ["คืออะไร", "อะไร", "ที่ไหน", "เมื่อไหร่", "ทำไม", "อย่างไร", "ใคร"]
    if any(keyword in question_lower for keyword in specific_keywords):
        return 3

    # คำถามทั่วไป - ใช้ค่าเริ่มต้น
    return 3


def calculate_relevance_score(question: str, context: str) -> float:
    """
    คำนวณคะแนนความเกี่ยวข้องระหว่างคำถามและ context (ปรับปรุงสำหรับภาษาไทย)
    """
    # แยกคำในคำถามและ context
    question_words = set(word_tokenize(question))
    context_words = set(word_tokenize(context))

    if len(question_words) == 0:
        return 0.0

    # คำนวณคำที่ซ้ำกัน
    common_words = question_words.intersection(context_words)

    # คำนวณ Jaccard similarity
    jaccard_similarity = len(common_words) / len(question_words.union(context_words))

    # คำนวณ keyword matching แบบ case-insensitive
    question_lower = question.lower()
    context_lower = context.lower()

    # ให้คะแนนพิเศษสำหรับคำสำคัญที่ตรงกัน
    exact_matches = 0
    partial_matches = 0

    for word in question_words:
        word_lower = word.lower()
        if word_lower in context_lower:
            exact_matches += 1

    # ตรวจสอบการ match แบบย่อย (เช่น "1-on-1" กับ "session", "นัด" กับ "คุย")
    for q_word in question_words:
        for c_word in context_words:
            # ถ้าคำใดคำหนึ่งเป็นส่วนหนึ่งของอีกคำ
            if q_word.lower() in c_word.lower() or c_word.lower() in q_word.lower():
                partial_matches += 0.5

    # ตรวจสอบคำที่เกี่ยวข้อง semantically สำหรับภาษาไทย
    semantic_matches = 0
    question_lower = question_lower.replace("1-on-1", "session สอนเสริม นัดคุย")
    question_lower = question_lower.replace("เฉพาะเรื่อง", "เฉพาะจุด")
    question_lower = question_lower.replace("อธิบาย", "สอนเสริม")
    question_lower = question_lower.replace("ไม่เข้าใจ", "ข้อสงสัย")

    for word in question_lower.split():
        if word in context_lower and len(word) > 2:  # คำที่มีความหมายมากกว่า 2 ตัวอักษร
            semantic_matches += 0.3

    # รวมคะแนนแบบถ่วงน้ำหนัก
    base_score = jaccard_similarity * 0.4
    exact_score = (exact_matches / len(question_words)) * 0.4
    partial_score = min(partial_matches / len(question_words), 0.3) * 0.2
    semantic_score = min(semantic_matches / len(question_words), 0.2) * 0.2

    final_score = base_score + exact_score + partial_score + semantic_score

    return min(final_score, 1.0)  # จำกัดคะแนนสูงสุดที่ 1.0


def filter_relevant_contexts(question: str, documents: list, metadatas: list, min_relevance: float = 0.05) -> list:
    """
    กรองเฉพาะ context ที่相关性สูง
    """
    if not documents:
        return []

    filtered_contexts = []

    for doc, metadata in zip(documents, metadatas):
        # คำนวณคะแนนความเกี่ยวข้อง
        relevance_score = calculate_relevance_score(question, doc)

        # เก็บเฉพาะ context ที่ผ่าน threshold
        if relevance_score >= min_relevance:
            filtered_contexts.append({
                'text': doc,
                'metadata': metadata,
                'relevance_score': relevance_score
            })

    # เรียงลำดับตามคะแนนความเกี่ยวข้อง (สูงสุดก่อน)
    filtered_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)

    # จำกัดจำนวน context สูงสุด
    max_contexts = 5
    return filtered_contexts[:max_contexts]


def query_rag(question: str, chat_llm: str = "gemma3:latest", show_source: bool = False, formal_style: bool = False):
    """
    ค้นหาในระบบ Enhanced RAG และสร้างคำตอบแบบ streaming โดยใช้ Ollama
    """
    global summarize, enhanced_rag

    # ตรวจสอบว่ามีการสรุปเนื้อหาหรือไม่ ถ้าไม่มีให้ใช้ค่าว่าง
    if 'summarize' not in globals() or summarize is None:
        summarize = "ยังไม่มีการสรุปเนื้อหาจาก PDF"

    logging.info(f"#### Enhanced RAG Mode: {RAG_MODE} #### ")
    logging.info(f"#### Question: {question} #### ")

    # Get relevant memories if enhanced mode is enabled
    relevant_memories = []
    if RAG_MODE == "enhanced":
        relevant_memories = enhanced_rag.get_relevant_memory(question)
        logging.info(f"Found {len(relevant_memories)} relevant memories")

    question_embedding = embed_text(question)

    # Smart Retrieval: ปรับจำนวนผลลัพธ์ตามประเภทคำถาม
    max_result = determine_optimal_results(question)
    logging.info(f"Using max_result: {max_result}")

    # ค้นหาด้วย similarity threshold ที่สูงขึ้น
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=max_result
    )

    # Relevance Filtering: กรองเฉพาะ context ที่相关性สูง
    filtered_contexts = filter_relevant_contexts(question, results["documents"][0], results["metadatas"][0], min_relevance=0.05)
    logging.info(f"Filtered {len(results['documents'][0])} contexts to {len(filtered_contexts)} relevant contexts")

    # ถ้าไม่มี context ที่ผ่านการกรอง ให้ใช้ทั้งหมด
    if len(filtered_contexts) == 0:
        logging.warning("No contexts passed relevance filter, using all retrieved contexts")
        filtered_contexts = [{'text': doc, 'metadata': meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    context_texts = []
    image_paths = []

    # ใช้เฉพาะ context ที่ผ่านการกรอง
    for doc, metadata in zip([ctx['text'] for ctx in filtered_contexts], [ctx['metadata'] for ctx in filtered_contexts]):
        context_texts.append(doc)
        logging.info(f"Selected context: {doc}")
        logging.info(f"metadata: {metadata}")

        # Regex pattern สำหรับค้นหา [img: ชื่อไฟล์.jpeg]
        pattern = r"pic_(\d+)_(\d+)\.jpeg"
        imgs = re.findall(pattern, doc)

        if imgs:
            image_paths.append(imgs)
            logging.info(f"img: {imgs}")

        if metadata and metadata["type"] == "image":
            logging.info(f"image_path : { metadata['image_path']}")
            image_paths.append(metadata['image_path'])
    


    context = "\n".join(context_texts)

    # Build enhanced prompt using EnhancedRAG
    if RAG_MODE == "enhanced":
        prompt = enhanced_rag.build_context_prompt(question, context_texts, relevant_memories, show_source, formal_style)
        logging.info(f"Using Enhanced RAG prompt with {len(relevant_memories)} memories")
        logging.info("############## Begin Enhanced RAG Prompt #################")
        logging.info(f"prompt: {prompt}")
        logging.info("############## End Enhanced RAG Prompt #################")
    else:
        # Standard RAG prompt with stricter instructions
        if summarize is None:
            summarize = ""

        # สร้าง prompt ตามสไตล์ที่เลือก
    if formal_style:
        style_instruction = "ตอบอย่างเป็นทางการ ใช้ภาษาที่สุภาพ และชัดเจน"
        source_phrase = ""
        response_prefix = "คำตอบ:"
    else:
        style_instruction = "ตอบอย่างเป็นกันเอง ใช้ภาษาที่เข้าใจง่าย"
        source_phrase = ""
        response_prefix = "คำตอบ:"

    source_instruction = ""
    if show_source:
        source_instruction = f"\n- หากตอบโดยใช้ข้อมูลจากบริบท ให้ระบุว่า '{source_phrase}'" if source_phrase else ""

    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามที่มีความรู้ด้านเอกสารที่อัปโหลดไว้ กรุณาตอบคำถามโดยอาศัยข้อมูลจากบริบทที่ให้มาเป็นหลัก

**แนวทางการตอบ:**
- {style_instruction}
- ให้คำตอบที่สอดคล้องกับข้อมูลในเอกสารเป็นหลัก
- หากข้อมูลในเอกสารไม่เพียงพอ ให้ตอบตามที่มีและระบุข้อจำกัด
- ตอบอย่างสมบูรณ์และมีประโยชน์{source_instruction}
- อาจเพิ่มคำอธิบายเล็กน้อยเพื่อความชัดเจน แต่ไม่ตีความเกินไป

**คำถาม:** {question}

**บริบทจากเอกสาร:**
{summarize}

{context}

{response_prefix}"""

    logging.info("############## Begin Standard Prompt #################")
    logging.info(f"prompt: {prompt}")
    logging.info("############## End Standard Prompt #################")
    logging.info(f"Prompt length: {len(prompt)} characters")

    logging.info("+++++++++++++  Send prompt To LLM  ++++++++++++++++++")
    ## Generation  เพื่อการตอบ chat
    stream = ollama.chat(
        model=chat_llm,
        messages=[{"role": "user", "content": prompt}],      
        stream=True
    )
    
    return stream

# UI Event Handlers
def handle_file_selection(files):
    """
    Handle file selection with improved file list display
    """
    if not files:
        return gr.update(value=""), gr.update(visible=False)

    # Prepare file information
    file_info_list = []
    for file in files:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        # Determine file type
        if file_ext == '.pdf':
            file_type = '📄 PDF'
        elif file_ext in ['.txt', '.md']:
            file_type = '📝 Text'
        elif file_ext in ['.docx', '.doc']:
            file_type = '📋 Word'
        else:
            file_type = '📁 File'

        # Get file size
        try:
            if hasattr(file, 'size'):
                file_size = file.size
            elif os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
            else:
                file_size = 0

            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
        except:
            size_str = "Unknown"

        file_info_list.append({
            "name": file_name,
            "type": file_type,
            "size": size_str,
            "path": file_path
        })

    # Format file list display
    file_list_text = "## 📋 Selected Files\n\n"
    for i, info in enumerate(file_info_list, 1):
        file_list_text += f"**{i}. {info['type']} {info['name']}**\n"
        file_list_text += f"   📏 Size: {info['size']}\n"
        file_list_text += f"   📍 Path: `{info['path']}`\n\n"

    return file_list_text, gr.update(visible=True)

def user(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    จัดการ input ของผู้ใช้และเพิ่มลงในประวัติการแชท
    """
    return "", history + [{"role": "user", "content": user_message}]

def chatbot_interface(history: List[Dict], llm_model: str, show_source: bool = False, formal_style: bool = False,
                       send_to_discord: bool = False, send_to_line: bool = False, send_to_facebook: bool = False,
                       line_user_id: str = "", fb_user_id: str = ""):
    """
    อินเทอร์เฟซแชทบอทแบบ streaming
    """
    user_message = history[-1]["content"]

    stream= query_rag(user_message, chat_llm=llm_model, show_source=show_source, formal_style=formal_style)

    history.append({"role": "assistant", "content": ""})
    full_answer=""
    """
    ส่วนของการ ตอบคำถาม
    """
    for chunk in stream:
        content = chunk["message"]["content"]
        full_answer += content 
        history[-1]["content"] += content
        #logging.info(f"content: {content}")
        yield history
    

    """
    ส่วนของการดึงรูปภาพ ที่เกี่ยวข้องมาแสดง โดยดึงจาก คำตอบด้านบน 
    """

    # ใช้ regex เพื่อดึงชื่อไฟล์ที่อยู่ใน [ภาพ: ...] 
    print(full_answer)
    pattern1 = r"\[(?:ภาพ:\s*)?(pic_\w+[-_]?\w*\.(?:jpe?g|png))\]"
    pattern2 = r"(pic_\w+[-_]?\w*\.(?:jpe?g|png))"
    # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
    
    print("----------PPPP------------")       
    image_list = re.findall(pattern1, full_answer)
    print(image_list)
    if (len(image_list)==0):
        image_list = re.findall(pattern2, full_answer)
    print("----------xxxx------------")  
    # ดึงเฉพาะรูปที่ไม่ซ้ำกัน
    image_list_uniq = list(dict.fromkeys(image_list))  
    if image_list_uniq:
        history[-1]["content"] += "\n\nรูปภาพที่เกี่ยวข้อง:"
        yield history    
        # ดึงรูปมาแสดง 
        for img in image_list_uniq:
            img_path = f"{TEMP_IMG}/{img}"
            logger.info(f"img_path: {img_path}")      
            if os.path.exists(img_path):
                    image = Image.open(img_path)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_response = f"{img} ![{img}](data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()})"
                    #ส่งรูปไปที่ Chat
                    history.append({"role": "assistant", "content": image_response })
                    yield history

    # Store conversation in memory for Enhanced RAG
    if RAG_MODE == "enhanced":
        try:
            enhanced_rag.store_memory(user_message, full_answer)
            logging.info("Stored conversation in Enhanced RAG memory")
        except Exception as e:
            logging.error(f"Failed to store in Enhanced RAG memory: {str(e)}")

    # ส่งคำตอบไปยังแพลตฟอร์มที่เลือก
    try:
        # ส่งไปยัง Discord (ถ้าเลือก)
        if send_to_discord and DISCORD_ENABLED:
            send_to_discord_sync(user_message, full_answer)
            logging.info("✅ ส่งคำตอบไปยัง Discord สำเร็จ")

        # ส่งไปยัง LINE OA (ถ้าเลือกและมี user_id)
        if send_to_line and LINE_ENABLED and line_user_id and line_bot_api:
            try:
                # จำกัดความยาวข้อความสำหรับ LINE
                line_answer = full_answer
                if len(line_answer) > 4900:
                    line_answer = line_answer[:4900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

                line_bot_api.push_message(
                    line_user_id,
                    TextSendMessage(text=f"คำถาม: {user_message}\n\nคำตอบ:\n{line_answer}")
                )
                logging.info("✅ ส่งคำตอบไปยัง LINE OA สำเร็จ")
            except Exception as e:
                logging.error(f"❌ ไม่สามารถส่งไปยัง LINE OA: {str(e)}")

        # ส่งไปยัง Facebook Messenger (ถ้าเลือกและมี user_id)
        if send_to_facebook and FB_ENABLED and fb_user_id:
            try:
                # จำกัดความยาวข้อความสำหรับ Facebook
                fb_answer = full_answer
                if len(fb_answer) > 1900:
                    fb_answer = fb_answer[:1900] + "\n\n... (คำตอบถูกตัดเนื่องจากความยาว)"

                send_facebook_message(
                    fb_user_id,
                    f"คำถาม: {user_message}\n\nคำตอบ:\n{fb_answer}"
                )
                logging.info("✅ ส่งคำตอบไปยัง Facebook Messenger สำเร็จ")
            except Exception as e:
                logging.error(f"❌ ไม่สามารถส่งไปยัง Facebook Messenger: {str(e)}")

    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาดในการส่งคำตอบไปยังแพลตฟอร์ม: {str(e)}")



# Gradio interface

with gr.Blocks(
    css="""
    .upload-container {
        border: 2px dashed #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        border-color: #4CAF50;
        background: #f5f5f5;
    }
    .drop-zone {
        min-height: 150px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        border: 2px dashed #ccc;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .drop-zone:hover {
        border-color: #4CAF50;
        background: #f9f9f9;
    }
    .options-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .checkbox-primary label {
        color: #495057;
        font-weight: 500;
    }
    .checkbox-secondary label {
        color: #6c757d;
        font-size: 0.9em;
    }
    .upload-button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        border: none;
        color: white;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .upload-button:hover {
        background: linear-gradient(45deg, #45a049, #3d8b40);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .status-output {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
    }
    """
) as demo:
    logo="https://camo.githubusercontent.com/9433204b08afdc976c2e4f5a4ba0d81f8877b585cc11206e2969326d25c41657/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f6e61726f6e67736b6d6c2f68746d6c352d6c6561726e406c61746573742f6173736574732f696d67732f546c697665636f64654c6f676f2d3435302e77656270"
    gr.Markdown(f"""<h3 style='display: flex; align-items: center; gap: 15px; padding: 10px; margin: 0;'>
        <img alt='T-LIVE-CODE' src='{logo}' style='height: 100px;' >
        <span style='font-size: 1.5em;'>แชทบอท PDF: RAG</span></h3>""")

    with gr.Tab("📚 อัปโหลดเอกสาร"):
        gr.Markdown("""
        ### 📁 อัปโหลดเอกสารและจัดการระบบข้อมูล
        รองรับไฟล์ **PDF, DOCX, TXT, MD** พร้อมระบบข้อมูลอัตโนมัติ
        """)

        # Create a more professional upload section with drag-and-drop
        with gr.Column():
            # Upload area with drag-and-drop support
            with gr.Group(elem_classes="upload-container"):
                files_input = gr.File(
                    label="ลากไฟล์มาวางที่นี่หรือคลิกเพื่อเลือกไฟล์",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                    height=150,
                    elem_classes="drop-zone",
                    show_label=True,
                    container=True,
                    scale=0.98
                )

            # File display area
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("""
                    **📋 รายการไฟล์ที่เลือก:**
                    - ยังไม่ได้เลือกไฟล์
                    - รองรับ PDF, TXT, MD, DOCX
                    - ขนาดสูงสุด: 100MB ต่อไฟล์
                    """)

                    selected_files_info = gr.Textbox(
                        label="ไฟล์ที่เลือก",
                        value="ยังไม่ได้เลือกไฟล์",
                        interactive=False,
                        lines=3,
                        max_lines=5
                    )

                with gr.Column(scale=2):
                    gr.Markdown("""
                    **💡 แนะนำ:**
                    • ลากไฟล์มาวางที่เลือก
                    • หรือคลิกปุ่ม "Browse Files" เพื่อเลือก
                    • รองรับไฟล์พร้อมกันได้หลายไฟล์
                    """)

        # Processing options with better styling
        with gr.Group(elem_classes="options-container"):
            with gr.Row():
                with gr.Column(scale=2):
                    clear_before_upload = gr.Checkbox(
                        label="🗑️ ล้างข้อมูลเก่าก่อนอัปโหลด",
                        value=False,
                        info="จะลบข้อมูลทั้งหมดก่อนเพิ่มเอกสารใหม่",
                        elem_classes="checkbox-primary"
                    )

                with gr.Column(scale=2):
                    include_memory_checkbox = gr.Checkbox(
                        label="🧠 รวม Enhanced RAG Memory",
                        value=(RAG_MODE == "enhanced"),
                        info="บันทึกความทรงจำการสนทนาพร้อม",
                        elem_classes="checkbox-secondary"
                    )

            # Action buttons with better styling
            with gr.Row():
                upload_button = gr.Button(
                    "📤 เริ่มต้นอัปโหลด",
                    variant="primary",
                    size="lg",
                    elem_classes="upload-button"
                )
                clear_button = gr.Button(
                    "🗑️ ล้างข้อมูลทั้งหมด",
                    variant="secondary",
                    size="lg"
                )

        # Status display
        with gr.Accordion("📊 สถานะการประมวลผล", open=True):
            upload_output = gr.Textbox(
                label="ผลลัพธ์",
                lines=5,
                interactive=False,
                elem_classes="status-output"
            )

        # Connect event handlers
        files_input.upload(
            fn=handle_file_selection,
            inputs=[files_input],
            outputs=[selected_files_info, upload_output]
        )

        files_input.clear(
            fn=lambda: ([], "ล้างรายการไฟล์แล้ว"),
            inputs=[],
            outputs=[selected_files_info]
        )

        upload_button.click(
            fn=process_multiple_files,
            inputs=[files_input, clear_before_upload],
            outputs=upload_output
        )

        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )
        upload_output = gr.Textbox(label="สถานะการประมวลผล", lines=3)
        upload_button.click(
            fn=process_multiple_files,
            inputs=[files_input, clear_before_upload],
            outputs=upload_output
        )
        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )

        # Google Sheets Import Section
        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("### 📊 นำเข้าข้อมูลจาก Google Sheets")

        with gr.Group(elem_classes="upload-container"):
            gr.Markdown("""
            **🔗 วิธีการใช้งาน:**
            1. เปิด Google Sheets ที่ต้องการนำเข้า
            2. ตั้งค่าให้เป็น **"เผยแพร่ต่อทุกคนบนเว็บ"** (Share > General access > Anyone with the link)
            3. คัดลอก URL มาวางในช่องด้านล่าง
            """)

            with gr.Row():
                sheets_url_input = gr.Textbox(
                    label="🔗 Google Sheets URL",
                    placeholder="https://docs.google.com/spreadsheets/d/...",
                    info="วาง URL ของ Google Sheets ที่ต้องการนำเข้า",
                    scale=3
                )
                sheets_clear_checkbox = gr.Checkbox(
                    label="ล้างข้อมูลเก่าก่อนนำเข้า",
                    value=False,
                    info="ติ๊กถ้าต้องการล้างข้อมูลเก่าก่อนนำเข้าข้อมูลใหม่",
                    scale=1
                )

            with gr.Row():
                sheets_import_button = gr.Button(
                    "📊 นำเข้าข้อมูล Google Sheets",
                    variant="primary",
                    size="lg",
                    elem_classes="upload-button"
                )

            sheets_output = gr.Textbox(
                label="ผลลัพธ์การนำเข้าข้อมูล",
                lines=6,
                interactive=False,
                elem_classes="status-output"
            )

        # Connect Google Sheets event handlers
        sheets_import_button.click(
            fn=process_google_sheets_url,
            inputs=[sheets_url_input, sheets_clear_checkbox],
            outputs=sheets_output
        )

        # ส่วนจัดการ Database
        with gr.Row():
            gr.Markdown("### 🗄️ จัดการ Database")

        # Enhanced Backup & Restore Section
        with gr.Accordion("💾 Enhanced Backup & Restore", open=True):
            with gr.Row():
                # Backup Controls
                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 สำรองข้อมูล")

                    backup_name_input = gr.Textbox(
                        label="ชื่อ Backup (ถ้าว่างจะสร้างอัตโนมัติ)",
                        placeholder="เช่น: my_backup_20241030",
                        interactive=True
                    )

                    include_memory_checkbox = gr.Checkbox(
                        label="รวม Enhanced RAG Memory",
                        value=(RAG_MODE == "enhanced"),
                        info="สำรองข้อมูลความทรงจำการสนทนาด้วย"
                    )

                    with gr.Row():
                        enhanced_backup_button = gr.Button("สำรองข้อมูลขั้นสูง", variant="primary")
                        quick_backup_button = gr.Button("สำรองด่วน", variant="secondary")

                # Restore Controls
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔄 กู้คืนข้อมูล")

                    backup_selector = gr.Dropdown(
                        label="เลือก Backup ที่จะกู้คืน",
                        choices=[],
                        interactive=True,
                        info="รีเฟรชรายการเพื่อดู backup ทั้งหมด"
                    )

                    with gr.Row():
                        restore_button = gr.Button("กู้คืนข้อมูล", variant="primary")
                        refresh_backups_button = gr.Button("🔄 รีเฟรช", size="sm")

            # Backup Status and Results
            backup_status_output = gr.Textbox(
                label="สถานะการสำรอง/กู้คืน",
                lines=3,
                interactive=False
            )

            # Backup List Section
            with gr.Accordion("📋 รายการ Backup ทั้งหมด", open=False):
                backup_list_output = gr.Textbox(
                    label="รายการ Backup",
                    lines=8,
                    interactive=False
                )

                with gr.Row():
                    delete_backup_button = gr.Button("ลบ Backup ที่เลือก", variant="stop", size="sm")
                    validate_backup_button = gr.Button("✅ ตรวจสอบความสมบูรณ์", variant="secondary", size="sm")
                    clean_invalid_button = gr.Button("🧹 ลบที่ไม่ถูกต้อง", variant="secondary", size="sm")
                    refresh_list_button = gr.Button("รีเฟรชรายการ", size="sm")

        # Quick Database Operations
        with gr.Accordion("⚡ ดำเนินการด่วน", open=False):
            with gr.Row():
                db_info_button = gr.Button("แสดงข้อมูล Database", variant="secondary")
                inspect_button = gr.Button("ตรวจสอบข้อมูลละเอียด", variant="secondary")
                auto_backup_button = gr.Button("สร้าง Auto Backup", variant="secondary")

            db_info_output = gr.Textbox(label="ข้อมูล Database", lines=5, interactive=False)
            inspect_output = gr.Textbox(label="ข้อมูลภายใน Database", lines=10, interactive=False)

        # Event Handlers for Enhanced Backup & Restore
        def enhanced_backup_handler(backup_name, include_memory):
            if not backup_name.strip():
                backup_name = None
            result = backup_database_enhanced(backup_name, include_memory)
            if result["success"]:
                return f"""✅ สำรองข้อมูลสำเร็จ!
• ชื่อ Backup: {result['backup_name']}
• รวม Memory: {'✅' if include_memory else '❌'}
• ขนาด: {result['metadata']}

สามารถกู้คืนได้จากรายการ backup"""
            else:
                return f"❌ สำรองข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def quick_backup_handler():
            result = backup_database_enhanced()
            if result["success"]:
                return f"✅ สำรองข้อมูลด่วนสำเร็จ: {result['backup_name']}"
            else:
                return f"❌ สำรองข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def restore_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะกู้คืน"

            result = restore_database_enhanced(backup_name)
            if result["success"]:
                emergency_name = result.get('emergency_backup', {}).get('backup_name', 'N/A')
                return f"""✅ กู้คืนข้อมูลสำเร็จ!
• จาก Backup: {backup_name}
• เวลากู้คืน: {result['restored_at']}
• สร้าง Emergency Backup: {emergency_name}

โปรดตรวจสอบข้อมูลหลังกู้คืน"""
            else:
                return f"❌ กู้คืนข้อมูลไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def refresh_backups_handler():
            backups = list_available_backups()
            if backups:
                choices = [backup["name"] for backup in backups]
                return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            else:
                return gr.Dropdown(choices=[], value=None)

        def get_backup_list_handler():
            backups = list_available_backups()
            if backups:
                backup_info = []
                for backup in backups:
                    info = f"""📁 {backup['name']}
• สร้างเมื่อ: {backup['created_at']}
• ขนาด: {backup['size_mb']} MB
• ประเภท: {backup['type']}
• Memory: {'✅' if backup['includes_memory'] else '❌'}
• RAG Mode: {backup['rag_mode']}
• Records: {backup['database_info'].get('total_records', 'N/A') if backup['database_info'] else 'N/A'}
---"""
                    backup_info.append(info)
                return "\n".join(backup_info)
            else:
                return "ไม่พบข้อมูล backup ในระบบ"

        def delete_backup_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะลบ"

            result = delete_backup(backup_name)
            if result["success"]:
                return f"✅ {result['message']}"
            else:
                return f"❌ ไม่สามารถลบ backup: {result.get('error', 'Unknown error')}"

        def auto_backup_handler():
            result = auto_backup_before_operation()
            if result["success"]:
                return f"✅ สร้าง Auto Backup สำเร็จ: {result['backup_name']}"
            else:
                return f"❌ สร้าง Auto Backup ไม่สำเร็จ: {result.get('error', 'Unknown error')}"

        def clean_invalid_handler():
            try:
                cleanup_invalid_backups()
                return "✅ ลบโฟลเดอร์ที่ไม่ถูกต้องเรียบร้อยแล้ว"
            except Exception as e:
                return f"❌ ลบโฟลเดอร์ไม่ถูกต้องไม่สำเร็จ: {str(e)}"

        def validate_backup_handler(backup_name):
            if not backup_name:
                return "❌ กรุณาเลือก backup ที่จะตรวจสอบ"

            backup_path = os.path.join(TEMP_VECTOR_BACKUP, backup_name)
            if not os.path.exists(backup_path):
                return f"❌ ไม่พบ backup: {backup_name}"

            is_valid, message = validate_backup_integrity(backup_path)
            if is_valid:
                return f"""✅ Backup ถูกต้อง!
• ชื่อ Backup: {backup_name}
• สถานะ: {message}
• พร้อมสำหรับการกู้คืน"""
            else:
                return f"""❌ Backup ไม่ถูกต้อง!
• ชื่อ Backup: {backup_name}
• ปัญหา: {message}
• แนะนำ: สร้าง backup ใหม่แทน"""

        # Connect event handlers
        enhanced_backup_button.click(
            fn=enhanced_backup_handler,
            inputs=[backup_name_input, include_memory_checkbox],
            outputs=backup_status_output,
            queue=False
        )

        quick_backup_button.click(
            fn=quick_backup_handler,
            inputs=None,
            outputs=backup_status_output,
            queue=False
        )

        restore_button.click(
            fn=restore_handler,
            inputs=backup_selector,
            outputs=backup_status_output,
            queue=False
        )

        refresh_backups_button.click(
            fn=refresh_backups_handler,
            inputs=None,
            outputs=backup_selector,
            queue=False
        )

        refresh_list_button.click(
            fn=get_backup_list_handler,
            inputs=None,
            outputs=backup_list_output,
            queue=False
        )

        delete_backup_button.click(
            fn=delete_backup_handler,
            inputs=backup_selector,
            outputs=backup_list_output,
            queue=False
        )

        auto_backup_button.click(
            fn=auto_backup_handler,
            inputs=None,
            outputs=db_info_output,
            queue=False
        )

        clean_invalid_button.click(
            fn=clean_invalid_handler,
            inputs=None,
            outputs=backup_status_output,
            queue=False
        )

        validate_backup_button.click(
            fn=validate_backup_handler,
            inputs=backup_selector,
            outputs=backup_status_output,
            queue=False
        )

        # Initialize backup list on page load
        demo.load(
            fn=refresh_backups_handler,
            inputs=None,
            outputs=backup_selector,
            queue=False
        )

        # Quick database operations handlers
        db_info_button.click(
            fn=lambda: str(get_database_info()),
            inputs=None,
            outputs=db_info_output,
            queue=False
        )

        inspect_button.click(
            fn=inspect_database,
            inputs=None,
            outputs=inspect_output,
            queue=False
        )

        # ส่วนจัดการ Discord Bot
        with gr.Row():
            gr.Markdown("### 🤖 จัดการ Discord Bot")

        with gr.Row():
            start_bot_button = gr.Button("เริ่ม Discord Bot", variant="primary")
            stop_bot_button = gr.Button("หยุด Discord Bot", variant="stop")

        bot_status_output = gr.Textbox(label="สถานะ Discord Bot", lines=3)

        with gr.Row():
            bot_model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=DISCORD_DEFAULT_MODEL,
                label="โมเดลสำหรับ Discord Bot"
            )

            bot_reply_mode = gr.Dropdown(
                choices=[
                    ("ตอบใน Channel", "channel"),
                    ("ตอบใน DM", "dm"),
                    ("ตอบทั้ง Channel และ DM", "both")
                ],
                value=DISCORD_REPLY_MODE,
                label="วิธีการตอบกลับ"
            )

        def update_discord_model(model):
            global DISCORD_DEFAULT_MODEL
            DISCORD_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล Discord Bot เป็น {model}"

        def update_discord_reply_mode(mode):
            global DISCORD_REPLY_MODE
            DISCORD_REPLY_MODE = mode
            mode_name = {"channel": "ใน Channel", "dm": "ใน DM", "both": "ทั้ง Channel และ DM"}
            return f"อัปเดตวิธีการตอบกลับเป็น: {mode_name.get(mode, mode)}"

        def start_bot_ui():
            if start_discord_bot_thread():
                return "Discord Bot เริ่มทำงานแล้ว! สามารถใช้คำสั่งได้ใน Discord"
            else:
                return "ไม่สามารถเริ่ม Discord Bot ได้ ตรวจสอบการตั้งค่าใน .env"

        def stop_bot_ui():
            if stop_discord_bot():
                return "Discord Bot หยุดทำงานแล้ว"
            else:
                return "ไม่สามารถหยุด Discord Bot ได้"

        start_bot_button.click(
            fn=start_bot_ui,
            inputs=None,
            outputs=bot_status_output,
            queue=False
        )

        stop_bot_button.click(
            fn=stop_bot_ui,
            inputs=None,
            outputs=bot_status_output,
            queue=False
        )

        bot_model_selector.change(
            fn=update_discord_model,
            inputs=bot_model_selector,
            outputs=bot_status_output,
            queue=False
        )

        bot_reply_mode.change(
            fn=update_discord_reply_mode,
            inputs=bot_reply_mode,
            outputs=bot_status_output,
            queue=False
        )

        # ส่วนจัดการ LINE OA Bot
        with gr.Row():
            gr.Markdown("### 📱 จัดการ LINE OA Bot")

        with gr.Row():
            start_line_button = gr.Button("เริ่ม LINE OA Bot", variant="primary")
            stop_line_button = gr.Button("หยุด LINE OA Bot", variant="stop")

        line_status_output = gr.Textbox(label="สถานะ LINE OA Bot", lines=3)
        line_model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=LINE_DEFAULT_MODEL,
            label="โมเดลสำหรับ LINE OA Bot"
        )

        def update_line_model(model):
            global LINE_DEFAULT_MODEL
            LINE_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล LINE OA Bot เป็น {model}"

        def start_line_ui():
            if start_line_bot_thread():
                return f"LINE OA Bot เริ่มทำงานแล้ว! Webhook URL: http://localhost:{LINE_WEBHOOK_PORT}/callback"
            else:
                return "ไม่สามารถเริ่ม LINE OA Bot ได้ ตรวจสอบการตั้งค่าใน .env"

        def stop_line_ui():
            if line_thread and line_thread.is_alive():
                return "LINE OA Bot หยุดทำงานแล้ว (รีสตาร์ทต้องการ restart โปรแกรม)"
            else:
                return "LINE OA Bot ไม่ได้ทำงานอยู่แล้ว"

        start_line_button.click(
            fn=start_line_ui,
            inputs=None,
            outputs=line_status_output,
            queue=False
        )

        stop_line_button.click(
            fn=stop_line_ui,
            inputs=None,
            outputs=line_status_output,
            queue=False
        )

        line_model_selector.change(
            fn=update_line_model,
            inputs=line_model_selector,
            outputs=line_status_output,
            queue=False
        )

        # ส่วนจัดการ Facebook Messenger Bot
        with gr.Row():
            gr.Markdown("### 💬 จัดการ Facebook Messenger Bot")

        with gr.Row():
            start_fb_button = gr.Button("เริ่ม Facebook Bot", variant="primary")
            stop_fb_button = gr.Button("หยุด Facebook Bot", variant="stop")

        fb_status_output = gr.Textbox(label="สถานะ Facebook Bot", lines=3)
        fb_model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=FB_DEFAULT_MODEL,
            label="โมเดลสำหรับ Facebook Bot"
        )

        def update_fb_model(model):
            global FB_DEFAULT_MODEL
            FB_DEFAULT_MODEL = model
            return f"อัปเดตโมเดล Facebook Bot เป็น {model}"

        def start_fb_ui():
            if start_facebook_bot_thread():
                return f"Facebook Bot เริ่มทำงานแล้ว! Webhook URL: http://localhost:{FB_WEBHOOK_PORT}/webhook"
            else:
                return "ไม่สามารถเริ่ม Facebook Bot ได้ ตรวจสอบการตั้งค่าใน .env"

        def stop_fb_ui():
            if fb_thread and fb_thread.is_alive():
                return "Facebook Bot หยุดทำงานแล้ว (รีสตาร์ทต้องการ restart โปรแกรม)"
            else:
                return "Facebook Bot ไม่ได้ทำงานอยู่แล้ว"

        start_fb_button.click(
            fn=start_fb_ui,
            inputs=None,
            outputs=fb_status_output,
            queue=False
        )

        stop_fb_button.click(
            fn=stop_fb_ui,
            inputs=None,
            outputs=fb_status_output,
            queue=False
        )

        fb_model_selector.change(
            fn=update_fb_model,
            inputs=fb_model_selector,
            outputs=fb_status_output,
            queue=False
        )

    with gr.Tab("แชท"):
        # Choice เลือก Model
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="gemma3:latest",
            label="เลือก LLM Model"
        )
        selected_model = gr.State(value="gemma3:latest")  # เก็บไว้ใน state
        model_selector.change(fn=lambda x: x, inputs=model_selector, outputs=selected_model)

        # Choice เลือก RAG Mode
        rag_mode_selector = gr.Radio(
            choices=[
                ("📖 Standard RAG - ค้นหาจากเอกสารเท่านั้น", "standard"),
                ("🧠 Enhanced RAG - จดจำบริบทการสนทนา", "enhanced")
            ],
            value="standard",
            label="เลือกโหมด RAG",
            info="Enhanced RAG จะจดจำประวัติการสนทนาและใช้บริบทในการตอบคำถาม"
        )
        selected_rag_mode = gr.State(value="standard")  # เก็บไว้ใน state
        rag_mode_status = gr.Textbox(label="สถานะ RAG Mode", value="📖 Standard RAG Mode", interactive=False)

        # เพิ่มตัวเลือกสำหรับการแสดงแหล่งที่มาของข้อมูล
        with gr.Row():
            show_source_checkbox = gr.Checkbox(
                label="🔍 แสดงแหล่งที่มาของข้อมูล",
                value=False,
                info="เพิ่มการระบุแหล่งที่มาของข้อมูลในคำตอบ"
            )

            formal_style_checkbox = gr.Checkbox(
                label="📝 สไตล์การตอบเป็นทางการ",
                value=False,
                info="ใช้ภาษาที่เป็นทางการและสุภาพมากขึ้น"
            )

        # เพิ่มตัวเลือกสำหรับการส่งคำตอบไปยังแพลตฟอร์มอื่น
        with gr.Row():
            gr.Markdown("### 📤 ส่งคำตอบไปยังแพลตฟอร์ม:")

        with gr.Row():
            send_to_discord_checkbox = gr.Checkbox(
                label="🤖 Discord",
                value=False,
                info="ส่งคำตอบไปยัง Discord channel"
            )

            send_to_line_checkbox = gr.Checkbox(
                label="📱 LINE OA",
                value=False,
                info="ส่งคำตอบไปยัง LINE OA (ต้องมี LINE_USER_ID)"
            )

            send_to_facebook_checkbox = gr.Checkbox(
                label="💬 Facebook Messenger",
                value=False,
                info="ส่งคำตอบไปยัง Facebook Messenger (ต้องมี FB_USER_ID)"
            )

        # เพิ่ม text input สำหรับระบุ user ID
        with gr.Row():
            line_user_id_input = gr.Textbox(
                label="LINE User ID (สำหรับส่งข้อความ)",
                placeholder="ใส่ LINE User ID ที่ต้องการส่งคำตอบ",
                visible=False,
                info="รับ User ID จาก LINE Debug console หรือการทดสอบ"
            )

            fb_user_id_input = gr.Textbox(
                label="Facebook User ID (สำหรับส่งข้อความ)",
                placeholder="ใส่ Facebook User ID ที่ต้องการส่งคำตอบ",
                visible=False,
                info="รับ User ID จาก Facebook Graph API"
            )

        # ฟังก์ชันสำหรับแสดง/ซ่อน user ID input
        def toggle_line_input(checked):
            return gr.update(visible=checked)

        def toggle_fb_input(checked):
            return gr.update(visible=checked)

        send_to_line_checkbox.change(
            fn=toggle_line_input,
            inputs=send_to_line_checkbox,
            outputs=line_user_id_input
        )

        send_to_facebook_checkbox.change(
            fn=toggle_fb_input,
            inputs=send_to_facebook_checkbox,
            outputs=fb_user_id_input
        )

        def update_rag_mode(mode):
            global RAG_MODE
            RAG_MODE = mode
            if mode == "enhanced":
                return "🧠 Enhanced RAG Mode - จะจดจำบริบทการสนทนา"
            else:
                return "📖 Standard RAG Mode - ค้นหาจากเอกสารเท่านั้น"

        rag_mode_selector.change(
            fn=update_rag_mode,
            inputs=rag_mode_selector,
            outputs=rag_mode_status,
            queue=False
        )

        # Enhanced RAG Memory Status
        with gr.Accordion("🧠 Enhanced RAG Memory Status", open=False):
            memory_status_output = gr.Textbox(label="ข้อมูล Memory", lines=4, interactive=False)
            refresh_memory_button = gr.Button("รีเฟรชข้อมูล Memory", size="sm")

        def get_memory_status():
            if RAG_MODE == "enhanced":
                try:
                    memory_info = enhanced_rag.get_memory_info()
                    return f"""📊 Memory Status:
• Total memories: {memory_info['total_memories']}
• Session memories: {memory_info['session_memories']}
• Long-term memories: {memory_info['longterm_memories']}
• Memory window: {memory_info['memory_window']}
"""
                except Exception as e:
                    return f"Error getting memory status: {str(e)}"
            else:
                return "Enhanced RAG is not active. Switch to Enhanced RAG mode to use memory features."

        refresh_memory_button.click(
            fn=get_memory_status,
            inputs=None,
            outputs=memory_status_output,
            queue=False
        )

        # Chat Bot
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="ถามคำถามเกี่ยวกับ PDF")
        # Clear button 
        clear_chat = gr.Button("ล้าง")
        # Submit function 
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=chatbot_interface,
            inputs=[chatbot, selected_model, show_source_checkbox, formal_style_checkbox,
                   send_to_discord_checkbox, send_to_line_checkbox, send_to_facebook_checkbox,
                   line_user_id_input, fb_user_id_input],
            outputs=chatbot
        )
        clear_chat.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    # ล้างข้อมูล ออกจากระบบ ก่อน เริ่ม Start Web
    clear_vector_db_and_images()
    demo.launch()