# Railway-optimized Dockerfile for RAG PDF
# Optimized for 1GB RAM limit and CPU-only execution

FROM python:3.10-slim

WORKDIR /app

# Install only essential system dependencies for Railway
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements files for Railway
COPY requirements_railway.txt .

# Install Python dependencies with Railway optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_railway.txt && \
    pip cache purge

# Copy only necessary files for Railway
COPY rag_pdf.py .
COPY auth_models.py .
COPY login_page.py .
COPY fix_database_persistence.py .
COPY .env.example .env

# Create necessary directories
RUN mkdir -p /app/data/chromadb /app/data/images /tmp/transformers_cache /tmp/sentence_transformers

# Environment variables for Railway (1GB RAM optimization)
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=false
ENV GRADIO_IN_PROCESS=true
ENV LOG_LEVEL=INFO
ENV ENV=production

# Railway memory optimizations
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers
ENV HF_HOME=/tmp/huggingface_cache

# Railway deployment optimizations
ENV RAILWAY_MEMORY_LIMIT=1GB
ENV MAX_WORKERS=1

# Expose port for Railway
EXPOSE 7860

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start command for Railway (optimized startup)
CMD ["python", "-u", "rag_pdf.py"]