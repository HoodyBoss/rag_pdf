FROM python:3.10.19-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements_minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_minimal.txt

# Copy the application files
COPY rag_pdf.py .
COPY auth_models.py .
COPY login_page.py .
COPY authenticated_app.py .

# Create necessary directories
RUN mkdir -p chroma_db temp_files

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Command to run the application
CMD ["python", "authenticated_app.py"]