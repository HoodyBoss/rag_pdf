# RAG PDF System Architecture Document

## Document Information
- **Version**: 1.0
- **Date**: October 29, 2025
- **Author**: Winston (Architect)
- **Status**: Draft
- **Architecture Type**: Full-Stack RAG System

## 1. Executive Summary

The RAG PDF system implements a Retrieval-Augmented Generation architecture specifically optimized for Thai language PDF processing and multi-modal document analysis. This document outlines the complete system architecture, including component design, data flow, technology choices, and implementation patterns.

### 1.1 Architecture Overview
- **Pattern**: RAG (Retrieval-Augmented Generation) with Multi-modal Processing
- **Style**: Event-driven, modular, and extensible
- **Deployment**: Local-first with privacy focus
- **Language**: Thai-optimized with multilingual support

### 1.2 Key Architectural Principles
1. **Multi-modal Integration**: Seamless processing of text and images
2. **Thai Language First**: Optimized tokenization and model selection
3. **Privacy by Design**: Local processing with no external data transmission
4. **Modular Architecture**: Loosely coupled components for maintainability
5. **Progressive Enhancement**: Simple deployment with advanced capabilities

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Gradio)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Admin Tab     │    │        Chat Tab                │ │
│  │                 │    │                                 │ │
│  │ • PDF Upload    │    │ • Question Input               │ │
│  │ • Processing    │    │ • Answer Display               │ │
│  │ • Model Select  │    │ • Image Integration            │ │
│  │ • Status Display│    │ • History Management           │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Engine                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ PDF Processing  │    │      Query Processing           │ │
│  │                 │    │                                 │ │
│  │ • Text Extract  │    │ • Question Embedding           │ │
│  │ • Image Extract │    │ • Semantic Search              │ │
│  │ • Thai Tokenize │    │ • Context Retrieval            │ │
│  │ • Summarization │    │ • Answer Generation            │ │
│  │ • Vector Store  │    │ • Response Formatting          │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage & Models Layer                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Vector DB     │    │      AI Models                  │ │
│  │   (ChromaDB)    │    │                                 │ │
│  │                 │    │ • Embedding: multilingual-e5    │ │
│  │ • Text Embeds   │    │ • Summarization: MT5-Thai       │ │
│  │ • Metadata      │    │ • Chat: Qwen/Gemma/Llama       │ │
│  │ • Image Paths   │    │ • Models via Ollama             │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│                      ┌─────────────────────────────────┐   │
│                      │     File System                 │   │
│                      │                                 │   │
│                      │ • Image Storage                 │   │
│                      │ • Temp Files                    │   │
│                      │ • Configuration                 │   │
│                      └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Web Interface Layer
- **Framework**: Gradio (Python-based web UI)
- **Pattern**: Two-tab interface (Admin + Chat)
- **Responsibility**: User interaction and data presentation

#### 2.2.2 Processing Engine Layer
- **Pattern**: Pipeline-based processing
- **Responsibility**: Core business logic and AI coordination
- **Components**: PDF processor, RAG query engine, model coordinator

#### 2.2.3 Storage Layer
- **Vector Database**: ChromaDB for semantic search
- **File System**: Local storage for images and temporary files
- **Configuration**: Model settings and application config

#### 2.2.4 Model Layer
- **Inference Engine**: Ollama for local LLM execution
- **Model Registry**: Multiple model support with switching capabilities
- **Optimization**: Model caching and resource management

## 3. Data Architecture

### 3.1 Data Flow Diagram

```
PDF Document → Content Extraction → Processing Pipeline → Vector Storage
    │                │                    │                   │
    │                ▼                    ▼                   ▼
    │         ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │         │ Text Extract│    │Image Extract│    │Thai Token   │
    │         │             │    │             │    │             │
    │         └─────────────┘    └─────────────┘    └─────────────┘
    │                │                    │                   │
    ▼                ▼                    ▼                   ▼
┌─────────────┐ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Summarization│ │ Embedding   │  │Image Desc   │  │Vector Store │
│             │ │ Generation  │  │Generation   │  │             │
└─────────────┘ └─────────────┘  └─────────────┘  └─────────────┘
```

### 3.2 Data Models

#### 3.2.1 Document Schema
```python
Document {
    id: str                    # Unique document identifier
    filename: str              # Original filename
    content: str               # Extracted text content
    summary: str               # Generated summary
    images: List[Image]        # Extracted images
    metadata: Dict            # Document metadata
    created_at: datetime      # Processing timestamp
}
```

#### 3.2.2 Image Schema
```python
Image {
    id: str                   # Unique image identifier
    document_id: str          # Parent document
    page_number: int          # Source page number
    image_path: str           # Local file path
    description: str          # Generated description
    base64_data: str          # Base64 encoded data
    bounding_box: Dict        # Position on page
}
```

#### 3.2.3 Vector Schema (ChromaDB)
```python
VectorEmbedding {
    id: str                   # Embedding identifier
    document_id: str          # Source document
    chunk_type: str           # "text" or "image"
    content: str              # Original content
    embedding: List[float]    # Vector representation
    metadata: Dict            # Additional context
}
```

### 3.3 Data Lifecycle

1. **Ingestion**: PDF upload → content extraction → processing
2. **Processing**: Text tokenization → embedding generation → storage
3. **Query**: Question embedding → similarity search → context retrieval
4. **Generation**: Context augmentation → LLM inference → response formatting
5. **Cleanup**: Temporary file removal → storage optimization

## 4. Component Design

### 4.1 PDF Processing Pipeline

#### 4.1.1 Text Extraction Component
```python
class TextExtractor:
    def __init__(self):
        self.pdf_processor = fitz  # PyMuPDF

    def extract_text(self, pdf_path: str) -> List[str]:
        """Extract text page by page"""
        # Implementation using PyMuPDF
        # Returns list of text content per page
```

#### 4.1.2 Image Extraction Component
```python
class ImageExtractor:
    def extract_images(self, pdf_path: str) -> List[Image]:
        """Extract and process images from PDF"""
        # Convert images to RGB format
        # Save to local storage
        # Generate metadata
```

#### 4.1.3 Thai Language Processor
```python
class ThaiProcessor:
    def __init__(self):
        self.tokenizer = pythainlp.word_tokenize
        self.summarizer = MT5ThaiSummarizer()

    def process_text(self, text: str) -> ProcessedText:
        """Thai-specific text processing"""
        # Tokenization using PyThaiNLP
        # Summarization using MT5 model
        # Text cleaning and normalization
```

### 4.2 Vector Storage System

#### 4.2.1 Embedding Generator
```python
class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for text chunks"""
        # Process texts in batches
        # Handle Thai language specifics
        # Return normalized embeddings
```

#### 4.2.2 Vector Database Manager
```python
class VectorDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./data/chromadb")
        self.collection = self.client.get_or_create_collection("pdf_embeddings")

    def store_embeddings(self, embeddings: List[VectorEmbedding]):
        """Store embeddings with metadata"""
        # Batch insertion for performance
        # Metadata indexing for search

    def search_similar(self, query_embedding: List[float], n_results: int = 5):
        """Semantic similarity search"""
        # ChromaDB similarity search
        # Result ranking and filtering
```

### 4.3 RAG Query Engine

#### 4.3.1 Query Processor
```python
class QueryProcessor:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDBManager()
        self.llm_manager = LLMManager()

    def process_query(self, question: str, model_name: str) -> QueryResponse:
        """Process user question with RAG"""
        # 1. Generate question embedding
        # 2. Retrieve relevant content
        # 3. Augment context
        # 4. Generate answer using LLM
        # 5. Format response with images
```

#### 4.3.2 LLM Manager
```python
class LLMManager:
    def __init__(self):
        self.ollama_client = ollama.Client()
        self.available_models = {
            "pdf-qwen": "qwen2.5:1.5b",
            "pdf-gemma": "gemma3:1b",
            "pdf-llama": "llama3.2:latest"
        }

    def generate_answer(self, prompt: str, model_name: str) -> str:
        """Generate answer using selected LLM"""
        # Model selection and configuration
        # Prompt engineering for Thai context
        # Streaming response generation
```

## 5. Technology Architecture

### 5.1 Technology Stack Rationale

#### 5.1.1 Core Technologies
- **Python 3.8+**: Rich AI/ML ecosystem and Thai language support
- **Gradio**: Rapid web interface development for ML applications
- **Ollama**: Local LLM inference with privacy benefits
- **ChromaDB**: Lightweight, efficient vector database

#### 5.1.2 AI/ML Technologies
- **PyMuPDF**: Robust PDF processing with Thai font support
- **PyThaiNLP**: Comprehensive Thai language processing toolkit
- **SentenceTransformers**: State-of-the-art multilingual embeddings
- **Transformers (Hugging Face)**: Model loading and optimization

#### 5.1.3 Model Selection Strategy
- **Embedding Model**: `intfloat/multilingual-e5-base`
  - Superior multilingual performance
  - Optimized for semantic search
  - Thai language support

- **Summarization Model**: `StelleX/mt5-base-thaisum-text-summarization`
  - Thai-specific summarization
  - MT5 architecture proven for Thai
  - Good balance of quality and speed

- **Chat Models**: Multiple options via Ollama
  - **Qwen2.5**: Strong multilingual capabilities
  - **Gemma3**: Efficient performance
  - **Llama3.2**: Advanced reasoning capabilities

### 5.2 Infrastructure Architecture

#### 5.2.1 Deployment Model
- **Local-First**: All processing occurs on user's machine
- **Privacy-Focused**: No data leaves the local environment
- **Resource-Aware**: Optimized for consumer hardware

#### 5.2.2 Storage Architecture
```
./data/
├── chromadb/              # Vector database storage
│   ├── pdf_embeddings/    # Document embeddings
│   └── metadata/          # Collection metadata
├── images/                # Extracted images
│   └── {document_id}/     # Per-document image folders
└── temp/                  # Temporary processing files
```

#### 5.2.3 Model Management
- **Ollama Integration**: Centralized model serving
- **Model Caching**: Persistent model storage
- **Resource Optimization**: Memory-efficient model loading
- **Version Management**: Model update and rollback capabilities

## 6. Security Architecture

### 6.1 Privacy-First Design

#### 6.1.1 Data Protection
- **Local Processing**: All data processing occurs locally
- **No External Transmission**: No data sent to external services
- **Temporary File Cleanup**: Automatic removal of processing artifacts
- **Secure Storage**: Local filesystem with proper permissions

#### 6.1.2 Access Control
- **Single User Model**: Simplified security for local deployment
- **Process Isolation**: Sandboxed processing environments
- **Input Validation**: Comprehensive sanitization of user inputs
- **Error Handling**: Secure error reporting without information leakage

### 6.2 Threat Mitigation

#### 6.2.1 Security Threats Addressed
- **Data Exfiltration**: Prevented by local-only architecture
- **Injection Attacks**: Mitigated through input validation
- **Resource Exhaustion**: Controlled through resource limits
- **Malicious Files**: Handled through secure file processing

#### 6.2.2 Security Best Practices
- **Principle of Least Privilege**: Minimal required permissions
- **Defense in Depth**: Multiple security layers
- **Secure Defaults**: Safe configuration out of the box
- **Regular Updates**: Dependency vulnerability management

## 7. Performance Architecture

### 7.1 Performance Optimization Strategies

#### 7.1.1 Processing Optimization
- **Batch Processing**: Vector embedding generation in batches
- **Parallel Processing**: Concurrent text and image processing
- **Caching Strategy**: Model and embedding caching
- **Memory Management**: Efficient resource utilization

#### 7.1.2 Search Optimization
- **Vector Indexing**: Optimized ChromaDB indexing
- **Query Caching**: Cached results for similar queries
- **Result Pagination**: Efficient large result set handling
- **Lazy Loading**: On-demand content loading

#### 7.1.3 Response Optimization
- **Streaming Responses**: Real-time answer generation
- **Progressive Enhancement**: Incremental content loading
- **Image Optimization**: Efficient image compression and encoding
- **UI Responsiveness**: Non-blocking user interface

### 7.2 Scalability Architecture

#### 7.2.1 Horizontal Scaling Considerations
- **Stateless Design**: Components designed for distribution
- **Load Balancing**: Potential for multiple processing nodes
- **Database Sharding**: Vector database distribution strategies
- **Caching Layers**: Multi-level caching architecture

#### 7.2.2 Vertical Scaling Optimization
- **Resource Monitoring**: CPU, memory, and storage tracking
- **Model Optimization**: Quantization and pruning techniques
- **Garbage Collection**: Memory leak prevention
- **Storage Management**: Automated cleanup and compression

## 8. Integration Architecture

### 8.1 External Dependencies

#### 8.1.1 Model Dependencies
- **Ollama Server**: Local LLM inference engine
- **Hugging Face Models**: Pre-trained model downloads
- **System Libraries**: PDF processing and Thai language support

#### 8.1.2 Integration Patterns
- **Service Integration**: RESTful communication with Ollama
- **Library Integration**: Direct Python library usage
- **File System Integration**: Local storage and caching
- **Process Integration**: External process execution and monitoring

### 8.2 Future Integration Points

#### 8.2.1 API Integration
- **REST API**: Potential for external service integration
- **Webhook Support**: Event-driven external notifications
- **Authentication**: Multi-user access control
- **Rate Limiting**: Usage management and control

#### 8.2.2 Database Integration
- **External Databases**: PostgreSQL, MongoDB integration
- **Cloud Storage**: AWS S3, Google Cloud Storage
- **Search Engines**: Elasticsearch integration
- **Analytics**: Usage tracking and analysis

## 9. Monitoring & Observability

### 9.1 Logging Architecture

#### 9.1.1 Application Logging
- **Structured Logging**: JSON-based log format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Component Logging**: Per-component log streams
- **Performance Logging**: Request timing and resource usage

#### 9.1.2 System Monitoring
- **Health Checks**: Component health monitoring
- **Resource Metrics**: CPU, memory, disk usage
- **Performance Metrics**: Query response times, processing speeds
- **Error Tracking**: Error rate and pattern analysis

### 9.2 Debugging Architecture

#### 9.2.1 Debug Interfaces
- **Debug Mode**: Enhanced logging and tracing
- **Performance Profiling**: Component performance analysis
- **Query Debugging**: Step-by-step query processing
- **Model Inspection**: Model input/output analysis

#### 9.2.2 Troubleshooting Tools
- **Diagnostic Scripts**: Automated system health checks
- **Performance Benchmarks**: Standardized performance testing
- **Error Simulation**: Controlled error scenario testing
- **Recovery Procedures**: Automated recovery mechanisms

## 10. Development Architecture

### 10.1 Code Architecture

#### 10.1.1 Module Structure
```
rag_pdf/
├── rag_pdf.py              # Main application file
├── components/             # Reusable components
│   ├── pdf_processor.py    # PDF processing logic
│   ├── query_engine.py     # RAG query processing
│   └── ui_components.py    # Gradio interface
├── models/                 # Model management
│   ├── embedding.py        # Embedding generation
│   ├── summarization.py    # Text summarization
│   └── chat_models.py      # LLM integration
├── utils/                  # Utility functions
│   ├── text_processing.py  # Text utilities
│   ├── image_processing.py # Image utilities
│   └── file_management.py  # File operations
└── config/                 # Configuration files
    ├── app_config.py       # Application settings
    └── model_config.py     # Model configurations
```

#### 10.1.2 Design Patterns
- **Strategy Pattern**: Multiple model support and switching
- **Factory Pattern**: Model and component instantiation
- **Observer Pattern**: UI updates and progress tracking
- **Singleton Pattern**: Resource managers and configuration

### 10.2 Testing Architecture

#### 10.2.1 Test Strategy
- **Unit Tests**: Component-level testing with mocked dependencies
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Full workflow testing with real PDFs
- **Performance Tests**: Load and stress testing

#### 10.2.2 Test Framework
- **Testing Framework**: pytest with fixtures and parameterization
- **Mocking**: unittest.mock for external dependencies
- **Test Data**: Standardized PDF test suite
- **Coverage**: pytest-cov for code coverage tracking

## 11. Deployment Architecture

### 11.1 Local Deployment

#### 11.1.1 Deployment Model
- **Single Machine**: All components on one machine
- **User Installation**: Python environment setup
- **Dependency Management**: pip and requirements.txt
- **Configuration**: File-based configuration management

#### 11.1.2 Installation Process
```bash
# Environment setup
python -m venv rag_pdf_env
source rag_pdf_env/bin/activate  # Windows: rag_pdf_env\Scripts\activate

# Dependency installation
pip install -r requirements.txt

# Model installation
ollama pull qwen2.5:1.5b
ollama pull gemma3:1b
ollama pull llama3.2:latest

# Application launch
python rag_pdf.py
```

### 11.2 Container Deployment (Future)

#### 11.2.1 Docker Architecture
- **Multi-stage Builds**: Optimized image sizes
- **Health Checks**: Container health monitoring
- **Volume Mounting**: Persistent data storage
- **Resource Limits**: CPU and memory constraints

#### 11.2.2 Docker Compose
- **Service Orchestration**: Multi-container coordination
- **Network Configuration**: Inter-service communication
- **Environment Variables**: Configuration management
- **Dependency Services**: Database and model services

## 12. Evolution Architecture

### 12.1 Roadmap Architecture

#### 12.1.1 Phase 1: Foundation (Current)
- Core RAG functionality
- Single document processing
- Basic Thai language support
- Local deployment

#### 12.1.2 Phase 2: Enhancement (3-6 months)
- Multi-document support
- Advanced UI features
- Performance optimization
- Extended model support

#### 12.1.3 Phase 3: Scale (6-12 months)
- Multi-user capabilities
- Cloud deployment options
- API development
- Enterprise features

### 12.2 Architectural Evolution

#### 12.2.1 Microservices Migration
- **Service Decomposition**: Component isolation
- **API Gateway**: Centralized request routing
- **Service Discovery**: Dynamic service location
- **Load Balancing**: Request distribution

#### 12.2.2 Cloud-Native Architecture
- **Kubernetes Deployment**: Container orchestration
- **Service Mesh**: Inter-service communication
- **Observability**: Comprehensive monitoring
- **Auto-scaling**: Dynamic resource management

## 13. Decision Records

### 13.1 Architectural Decisions

#### ADR-001: Gradio as Web Framework
- **Decision**: Use Gradio for web interface
- **Rationale**: Rapid development, ML-focused features, Python integration
- **Alternatives Considered**: Flask, FastAPI, Streamlit
- **Consequences**: Easy deployment, limited customization

#### ADR-002: Local-First Architecture
- **Decision**: Local processing only
- **Rationale**: Privacy, offline capability, user control
- **Alternatives Considered**: Cloud-based, hybrid approach
- **Consequences**: Resource requirements, scaling limitations

#### ADR-003: ChromaDB for Vector Storage
- **Decision**: Use ChromaDB as vector database
- **Rationale**: Lightweight, Python-native, easy setup
- **Alternatives Considered**: FAISS, Pinecone, Weaviate
- **Consequences**: Performance limitations, feature constraints

#### ADR-004: Thai Language Optimization
- **Decision**: Prioritize Thai language processing
- **Rationale**: Market differentiation, user need, competitive advantage
- **Alternatives Considered**: English-first, multilingual equal
- **Consequences**: Increased complexity, specialized knowledge required

### 13.2 Technology Decisions

#### 13.2.1 Model Selection Criteria
- **Performance**: Accuracy and speed benchmarks
- **Resource Requirements**: Hardware constraints
- **Language Support**: Thai language capabilities
- **License**: Open source and commercial usage
- **Community Support**: Active development and documentation

#### 13.2.2 Infrastructure Decisions
- **Python 3.8+**: Minimum version requirement
- **Local Deployment**: Privacy and offline requirements
- **Single-threaded UI**: Simplified user interface
- **File-based Storage**: Simplicity and reliability

## 14. Conclusion

The RAG PDF system architecture demonstrates a thoughtful balance between functionality, performance, and usability. The privacy-first local deployment model addresses growing concerns about data security while the Thai language optimization provides significant value to the target market.

### 14.1 Architectural Strengths
- **Multi-modal Integration**: Seamless text and image processing
- **Thai Language Focus**: Specialized processing for Thai content
- **Privacy by Design**: Local-only data processing
- **Modular Design**: Maintainable and extensible architecture
- **Technology Optimization**: Appropriate technology choices

### 14.2 Areas for Future Enhancement
- **Scalability**: Multi-user and multi-document support
- **Performance**: Query speed and processing optimization
- **Features**: Advanced analytics and collaboration tools
- **Deployment**: Cloud and container deployment options
- **Integration**: External system and API capabilities

The architecture provides a solid foundation for the current educational release while allowing for future growth and enhancement as the product matures and user requirements evolve.

---

*This architecture document serves as the technical foundation for RAG PDF system development and should be updated as the system evolves and new requirements emerge.*