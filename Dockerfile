FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_railway.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_railway.txt

# Copy application files
COPY railway_rag_full.py .
COPY mongodb_rag.py .
COPY .env.example .env

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 7860

# Environment variables for Railway deployment
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860
ENV LOG_LEVEL=INFO
ENV ENV=production
ENV GRADIO_SERVER_NAME=0.0.0.0

# Railway-specific environment variables (will be set in Railway dashboard)
# ENV MONGODB_URI=${MONGODB_URI}
# ENV DATABASE_NAME=${DATABASE_NAME}
# ENV ADMIN_USERNAME=${ADMIN_USERNAME}
# ENV ADMIN_PASSWORD=${ADMIN_PASSWORD}

# Start command - use the enhanced system with authentication
CMD ["python", "railway_rag_full.py"]