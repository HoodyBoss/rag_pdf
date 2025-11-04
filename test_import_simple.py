#!/usr/bin/env python3
"""
Simple debug script without emojis for Railway deployment
"""
import sys
import os

print("=== RAILWAY DEPLOYMENT DEBUG ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

print("\n=== CHECKING REQUIRED FILES ===")
required_files = ['railway_rag.py', 'mongodb_rag.py', 'requirements_railway.txt', 'railway.toml']
for file in required_files:
    exists = os.path.exists(file)
    print(f"{file}: {'EXISTS' if exists else 'MISSING'}")

print("\n=== TESTING BASIC IMPORTS ===")
imports_to_test = [
    ('gradio', 'gradio'),
    ('dotenv', 'dotenv'),
    ('pymongo', 'pymongo[srv]'),
    ('sentence_transformers', 'sentence-transformers'),
    ('PyMuPDF', 'PyMuPDF'),
    ('requests', 'requests')
]

for module_name, package in imports_to_test:
    try:
        __import__(module_name)
        print(f"{module_name}: OK")
    except ImportError as e:
        print(f"{module_name}: FAILED - {e}")

print("\n=== TESTING CUSTOM IMPORTS ===")
try:
    from mongodb_rag import MongoDBRAG
    print("mongodb_rag.MongoDBRAG: OK")
except ImportError as e:
    print(f"mongodb_rag.MongoDBRAG: FAILED - {e}")

try:
    import railway_rag
    print("railway_rag: OK")
except ImportError as e:
    print(f"railway_rag: FAILED - {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()

print("\n=== ENVIRONMENT VARIABLES ===")
env_vars = ['MONGODB_URI', 'DATABASE_NAME', 'PORT', 'HOST', 'LOG_LEVEL', 'ENV']
for var in env_vars:
    value = os.getenv(var, 'NOT_SET')
    print(f"{var}: {'SET' if value != 'NOT_SET' else 'NOT_SET'}")

print("\n=== DEPLOYMENT READY CHECK ===")
try:
    from mongodb_rag import MongoDBRAG
    import railway_rag
    print("System appears ready for Railway deployment")
except Exception as e:
    print(f"Deployment issue found: {e}")
    print("Please check imports and dependencies")

print("\n=== DEBUG COMPLETE ===")