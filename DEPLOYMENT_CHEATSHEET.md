# 🚀 Deployment Cheatsheet - RAG PDF

## ⚡ Quick Setup

### 🚂 **Railway (Production - Cost Optimized)**
```bash
# Copy Railway config
cp .env.railway .env

# Deploy
git add .
git commit -m "Deploy to Railway - memory optimized"
git push origin main
```
- **Memory**: ~200MB startup
- **Cost**: $5-15/month
- **Models**: MiniLM (80MB), No summarization
- **GPU**: No

### 💻 **Local CPU (Development)**
```bash
# Copy CPU config
cp .env.cpu .env

# Run
python rag_pdf.py
```
- **Memory**: ~2GB
- **Models**: MiniLM (80MB), MT5 summarization
- **GPU**: No

### 🚀 **Local GPU (High Performance)**
```bash
# Copy GPU config
cp .env.gpu .env

# Run
python rag_pdf.py
```
- **Memory**: ~4GB
- **Models**: multilingual-e5 (1.1GB), MT5 summarization
- **GPU**: Yes (NVIDIA)

## ⚙️ **Environment Variables**

### Core Settings
```bash
DEPLOYMENT_ENV=railway|local_cpu|local_gpu
USE_GPU=true|false
EMBEDDING_DEVICE=cpu|cuda
```

### Model Selection
```bash
# Embedding Models
EMBEDDING_MODEL=all-MiniLM-L6-v2           # 80MB, CPU
EMBEDDING_MODEL=intfloat/multilingual-e5-base  # 1.1GB, GPU

# Summarization
ENABLE_SUMMARIZATION=true|false
SUMMARIZATION_MODEL=StelleX/mt5-base-thaisum-text-summarization
```

### Performance Tuning
```bash
CHUNK_SIZE=800-1000
CHUNK_OVERLAP=150-200
MAX_CHUNKS=100-500
AUTO_CLEANUP_INTERVAL=20-50
```

## 🎯 **Performance Matrix**

| Deployment | Startup RAM | Peak RAM | Models | GPU | Cost |
|------------|--------------|----------|---------|-----|------|
| **Railway** | 200MB | 800MB | MiniLM | ❌ | $5-15/mo |
| **Local CPU** | 500MB | 2GB | MiniLM + MT5 | ❌ | $0 |
| **Local GPU** | 1GB | 4GB | e5 + MT5 | ✅ | $0 |

## 📊 **Model Comparison**

| Embedding Model | Size | Accuracy | Language Support |
|-----------------|------|----------|------------------|
| `all-MiniLM-L6-v2` | 80MB | 85% | 100+ languages |
| `intfloat/multilingual-e5-base` | 1.1GB | 95% | Multilingual, Thai optimized |

## 🔧 **Troubleshooting**

### Memory Issues
```bash
# Reduce chunk size
export CHUNK_SIZE=600

# Reduce max chunks
export MAX_CHUNKS=50

# Disable summarization
export ENABLE_SUMMARIZATION=false
```

### GPU Issues
```bash
# Force CPU mode
export USE_GPU=false
export CUDA_VISIBLE_DEVICES=""

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Railway Issues
```bash
# Check Railway logs
railway logs

# Check environment
curl $RAILWAY_PUBLIC_URL | head -20
```

## 🎉 **Success Indicators**

✅ **Startup**: Configuration logged correctly
✅ **Memory**: Within expected range
✅ **Models**: Load without errors
✅ **Processing**: Documents upload successfully
✅ **Chat**: Q&A works with uploaded documents

---

**Choose your deployment based on budget and performance needs! 🎯**