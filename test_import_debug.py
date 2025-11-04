#!/usr/bin/env python3
"""
Debug import issues for Railway deployment
"""
import sys
import os

print("=== DEBUG IMPORTS ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

print("\n=== TESTING IMPORTS ===")

try:
    print("1. Testing gradio import...")
    import gradio as gr
    print("   ✅ gradio imported successfully")
except ImportError as e:
    print(f"   ❌ gradio import failed: {e}")

try:
    print("2. Testing dotenv import...")
    from dotenv import load_dotenv
    print("   ✅ dotenv imported successfully")
except ImportError as e:
    print(f"   ❌ dotenv import failed: {e}")

try:
    print("3. Testing pymongo import...")
    import pymongo
    print("   ✅ pymongo imported successfully")
except ImportError as e:
    print(f"   ❌ pymongo import failed: {e}")

try:
    print("4. Testing sentence_transformers import...")
    import sentence_transformers
    print("   ✅ sentence_transformers imported successfully")
except ImportError as e:
    print(f"   ❌ sentence_transformers import failed: {e}")

try:
    print("5. Testing mongodb_rag import...")
    from mongodb_rag import MongoDBRAG
    print("   ✅ mongodb_rag imported successfully")
except ImportError as e:
    print(f"   ❌ mongodb_rag import failed: {e}")

print("\n=== TESTING FILE ACCESS ===")
try:
    with open('mongodb_rag.py', 'r') as f:
        content = f.read()[:200]
        print(f"mongodb_rag.py content preview: {content[:100]}...")
except FileNotFoundError:
    print("❌ mongodb_rag.py not found")
except Exception as e:
    print(f"❌ Error reading mongodb_rag.py: {e}")

print("\n=== TESTING RAILWAY_RAG IMPORT ===")
try:
    print("6. Testing railway_rag import...")
    import railway_rag
    print("   ✅ railway_rag imported successfully")
except ImportError as e:
    print(f"   ❌ railway_rag import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")