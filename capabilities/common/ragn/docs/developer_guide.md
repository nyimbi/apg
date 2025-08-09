# APG RAG Developer Guide

> **Complete developer onboarding and contribution guide**

## 🚀 Getting Started

### Prerequisites

Before you begin developing with APG RAG, ensure you have:

- **Python 3.11+** with pip and virtual environment support
- **Docker 20.10+** with Docker Compose
- **PostgreSQL 15+** with pgvector and pgai extensions
- **Git** for version control
- **IDE/Editor** with Python support (VS Code, PyCharm, etc.)

### Development Environment Setup

#### 1. Clone and Setup Repository

```bash
# Clone the APG repository
git clone <your-repo-url>

# Navigate to RAG capability
cd apg/capabilities/common/rag

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit development configuration
nano .env
```

**Key Development Settings:**
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_RELOAD=true
ENABLE_DEBUG=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/apg_rag_dev
OLLAMA_BASE_URL=http://localhost:11434
```

#### 3. Start Development Services

```bash
# Start PostgreSQL, Redis, and Ollama
docker-compose -f docker-compose.dev.yml up -d postgres redis ollama

# Wait for services to be ready
./scripts/wait-for-services.sh

# Initialize database
python -c "
from capabilities.common.rag.database import init_database
import asyncio
asyncio.run(init_database())
"

# Load Ollama models
docker-compose exec ollama ollama pull bge-m3
docker-compose exec ollama ollama pull qwen3
docker-compose exec ollama ollama pull deepseek-r1
```

#### 4. Run Development Server

```bash
# Start the development server with hot reload
python -m uvicorn capabilities.common.rag.service:app \
    --host 0.0.0.0 \
    --port 5000 \
    --reload \
    --log-level debug
```

## 🏗️ Project Structure

### Directory Layout

```
capabilities/common/rag/
├── docs/                    # Detailed documentation
│   ├── architecture.md      # System architecture
│   ├── developer_guide.md   # This file
│   └── operations_manual.md # Operations guide
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/               # End-to-end tests
├── scripts/                # Utility scripts
│   ├── setup.sh           # Environment setup
│   ├── test.sh            # Test runner
│   └── deploy.sh          # Deployment script
├── config/                 # Configuration files
│   ├── development.py      # Dev configuration
│   ├── production.py       # Prod configuration
│   └── testing.py         # Test configuration
├── migrations/             # Database migrations
│   └── versions/          # Migration versions
└── capabilities/common/rag/
    ├── __init__.py         # APG capability metadata
    ├── models.py           # Pydantic data models
    ├── service.py          # Main service orchestration
    ├── views.py            # Flask-AppBuilder views
    ├── document_processor.py  # Document processing
    ├── vector_service.py   # Vector operations
    ├── retrieval_engine.py # Document retrieval
    ├── generation_engine.py # RAG generation
    ├── conversation_manager.py # Conversation state
    ├── security.py         # Security and compliance
    ├── monitoring.py       # Performance monitoring
    ├── ollama_integration.py # Ollama AI integration
    └── tests.py           # Test suite
```

### Core Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `service.py` | Main orchestration | `RAGService` |
| `models.py` | Data models | `Document`, `Query`, `Conversation` |
| `document_processor.py` | Document ingestion | `DocumentProcessor` |
| `vector_service.py` | Vector operations | `VectorService` |
| `retrieval_engine.py` | Document retrieval | `RetrievalEngine` |
| `generation_engine.py` | Response generation | `GenerationEngine` |
| `conversation_manager.py` | Context management | `ConversationManager` |
| `security.py` | Security framework | `SecurityManager` |
| `monitoring.py` | Performance tracking | `PerformanceMonitor` |

## 🧪 Development Workflow

### Code Standards and Conventions

#### Python Style Guide

```python
# Use async throughout
async def process_document(document: Document) -> ProcessedDocument:
    """Process a document asynchronously."""
    pass

# Modern type hints
def search_documents(
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 10
) -> list[Document]:
    """Search documents with optional filters."""
    pass

# Use tabs for indentation (not spaces)
class DocumentProcessor:
	def __init__(self, config: ProcessorConfig):
		self.config = config
		self._initialize_processors()
	
	async def process(self, document: Document) -> ProcessedDocument:
		"""Process document with quality validation."""
		# Implementation here
		pass
```

#### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict
from typing import Annotated
from uuid_extensions import uuid7str

class Document(BaseModel):
    """Document model with validation."""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    id: str = Field(default_factory=uuid7str)
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
```

#### Logging Standards

```python
# Use _log_ prefixed methods
def _log_document_processed(document_id: str, processing_time: float) -> str:
    """Log document processing completion."""
    return f"Document {document_id} processed in {processing_time:.2f}s"

# Usage in methods
class DocumentProcessor:
    async def process_document(self, document: Document) -> ProcessedDocument:
        start_time = time.time()
        
        # Processing logic here
        
        processing_time = time.time() - start_time
        logger.info(self._log_document_processed(document.id, processing_time))
        
        return processed_document
```

### Testing Framework

#### Test Structure

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from capabilities.common.rag.service import RAGService
from capabilities.common.rag.models import Document, Query

class TestRAGService:
    """Test suite for RAG service."""
    
    @pytest.fixture
    async def rag_service(self):
        """Create RAG service instance for testing."""
        config = TestConfig()
        service = RAGService(config)
        await service.initialize()
        yield service
        await service.cleanup()
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return Document(
            title="Test Document",
            content="This is test content for validation.",
            source="test_source"
        )
    
    async def test_document_processing(self, rag_service, sample_document):
        """Test document processing pipeline."""
        # Given
        knowledge_base_id = "test_kb"
        
        # When
        result = await rag_service.process_document(
            document=sample_document,
            knowledge_base_id=knowledge_base_id
        )
        
        # Then
        assert result.success is True
        assert result.document_id is not None
        assert len(result.chunks) > 0
        
        # Verify document was stored
        stored_doc = await rag_service.get_document(result.document_id)
        assert stored_doc.title == sample_document.title
    
    async def test_query_processing(self, rag_service):
        """Test query processing and response generation."""
        # Given
        query = Query(
            text="What is the main topic of the test document?",
            knowledge_base_id="test_kb"
        )
        
        # When
        response = await rag_service.process_query(query)
        
        # Then
        assert response.success is True
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.confidence_score > 0.5
```

#### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/e2e/ -v                     # End-to-end tests

# Run tests with coverage
pytest --cov=capabilities.common.rag --cov-report=html

# Run performance tests
pytest tests/performance/ -v --benchmark-only

# Run tests in parallel
pytest -n auto
```

### Development Scripts

#### Database Management

```bash
# Create new migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Reset database (development only)
python scripts/reset_database.py
```

#### Code Quality

```bash
# Format code with black
black capabilities/common/rag/

# Sort imports with isort
isort capabilities/common/rag/

# Lint code with flake8
flake8 capabilities/common/rag/

# Type checking with mypy
mypy capabilities/common/rag/

# All quality checks
./scripts/quality_check.sh
```

## 🔧 API Development

### Creating New Endpoints

#### 1. Define Data Models

```python
# In models.py
class NewFeatureRequest(BaseModel):
    """Request model for new feature."""
    
    model_config = ConfigDict(extra='forbid')
    
    feature_name: str = Field(..., min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)

class NewFeatureResponse(BaseModel):
    """Response model for new feature."""
    
    success: bool
    result: dict[str, Any]
    processing_time_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
```

#### 2. Implement Service Logic

```python
# In service.py
class RAGService:
    async def process_new_feature(
        self,
        request: NewFeatureRequest,
        tenant_id: str
    ) -> NewFeatureResponse:
        """Process new feature request."""
        start_time = time.time()
        
        try:
            # Validate request
            await self._validate_feature_request(request, tenant_id)
            
            # Process feature
            result = await self._execute_feature_logic(request)
            
            # Log success
            processing_time = (time.time() - start_time) * 1000
            logger.info(self._log_feature_processed(
                request.feature_name,
                processing_time
            ))
            
            return NewFeatureResponse(
                success=True,
                result=result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Feature processing failed: {e}")
            return NewFeatureResponse(
                success=False,
                result={},
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
    
    def _log_feature_processed(
        self,
        feature_name: str,
        processing_time: float
    ) -> str:
        """Log feature processing completion."""
        return f"Feature '{feature_name}' processed in {processing_time:.2f}ms"
```

#### 3. Add API Endpoints

```python
# In views.py
@rag_bp.route('/new-feature', methods=['POST'])
@require_auth
async def process_new_feature():
    """Process new feature request."""
    try:
        # Parse request
        data = await request.get_json()
        feature_request = NewFeatureRequest(**data)
        
        # Get tenant context
        tenant_id = get_current_tenant()
        
        # Process request
        response = await rag_service.process_new_feature(
            feature_request,
            tenant_id
        )
        
        # Return response
        return jsonify({
            "success": response.success,
            "data": response.result,
            "metadata": {
                "processing_time_ms": response.processing_time_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": tenant_id
            }
        }), 200 if response.success else 500
        
    except ValidationError as e:
        return jsonify({
            "success": False,
            "errors": e.errors(),
            "message": "Invalid request data"
        }), 400
    
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return jsonify({
            "success": False,
            "message": "Internal server error"
        }), 500
```

#### 4. Add Tests

```python
# In tests.py
class TestNewFeatureAPI:
    """Test new feature API endpoints."""
    
    async def test_new_feature_success(self, client, auth_headers):
        """Test successful new feature processing."""
        # Given
        request_data = {
            "feature_name": "test_feature",
            "parameters": {"param1": "value1"},
            "options": ["option1", "option2"]
        }
        
        # When
        response = await client.post(
            '/api/v1/rag/new-feature',
            json=request_data,
            headers=auth_headers
        )
        
        # Then
        assert response.status_code == 200
        data = await response.get_json()
        assert data["success"] is True
        assert "processing_time_ms" in data["metadata"]
    
    async def test_new_feature_validation_error(self, client, auth_headers):
        """Test validation error handling."""
        # Given
        invalid_request = {
            "feature_name": "",  # Invalid: empty string
            "parameters": "not_a_dict"  # Invalid: wrong type
        }
        
        # When
        response = await client.post(
            '/api/v1/rag/new-feature',
            json=invalid_request,
            headers=auth_headers
        )
        
        # Then
        assert response.status_code == 400
        data = await response.get_json()
        assert data["success"] is False
        assert "errors" in data
```

## 🔍 Debugging and Troubleshooting

### Logging Configuration

```python
# Development logging setup
import logging
import sys

def setup_development_logging():
    """Configure logging for development."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/rag_development.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('capabilities.common.rag').setLevel(logging.DEBUG)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
```

### Common Development Issues

#### Database Connection Issues

```python
# Check database connectivity
async def check_database_connection():
    """Verify database connection and extensions."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Check PostgreSQL version
        version = await conn.fetchval('SELECT version()')
        print(f"PostgreSQL version: {version}")
        
        # Check pgvector extension
        pgvector_version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        print(f"pgvector version: {pgvector_version}")
        
        # Check pgai extension
        pgai_version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'ai'"
        )
        print(f"pgai version: {pgai_version}")
        
        await conn.close()
        print("✅ Database connection successful")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
```

#### Ollama Integration Issues

```python
# Check Ollama service health
async def check_ollama_health():
    """Verify Ollama service and models."""
    try:
        async with httpx.AsyncClient() as client:
            # Check service health
            response = await client.get(f'{OLLAMA_BASE_URL}/api/tags')
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Ollama service healthy")
                print(f"Available models: {[m['name'] for m in models['models']]}")
                
                # Check required models
                required_models = ['bge-m3', 'qwen3', 'deepseek-r1']
                available_models = [m['name'] for m in models['models']]
                
                for model in required_models:
                    if model in available_models:
                        print(f"✅ Model {model} available")
                    else:
                        print(f"❌ Model {model} missing")
            else:
                print(f"❌ Ollama service unhealthy: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
```

### Performance Profiling

```python
# Profile API endpoint performance
import cProfile
import pstats
from functools import wraps

def profile_endpoint(func):
    """Decorator to profile endpoint performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
    
    return wrapper

# Usage
@profile_endpoint
async def process_complex_query(query: Query) -> QueryResponse:
    """Process query with performance profiling."""
    # Implementation here
    pass
```

## 🚀 Deployment Development

### Local Development Deployment

```bash
# Build development image
docker build -t apg-rag:dev --target development .

# Run with development settings
docker run -it --rm \
    -p 5000:5000 \
    -p 9090:9090 \
    -v $(pwd):/app \
    -e ENVIRONMENT=development \
    -e ENABLE_RELOAD=true \
    apg-rag:dev
```

### Integration Testing Environment

```yaml
# docker-compose.integration.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: apg_rag_integration
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"
    volumes:
      - ./test_models:/root/.ollama
  
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
  
  rag-service:
    build:
      context: .
      target: development
    environment:
      DATABASE_URL: postgresql://test_user:test_password@postgres:5432/apg_rag_integration
      OLLAMA_BASE_URL: http://ollama:11434
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - ollama
      - redis
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## 📖 Additional Resources

### Learning Resources

- **APG Architecture Guide**: Understanding the APG ecosystem
- **PostgreSQL + pgvector**: Vector database operations
- **Ollama Documentation**: AI model integration
- **FastAPI/Flask**: Web framework patterns
- **Pydantic**: Data validation and serialization
- **AsyncIO**: Asynchronous programming in Python

### Community and Support

- **Internal Wiki**: APG development standards and patterns
- **Code Reviews**: Best practices and feedback
- **Technical Meetings**: Weekly architecture discussions
- **Slack Channels**: Real-time development support

### Contributing Guidelines

1. **Fork and Branch**: Create feature branches from `main`
2. **Code Quality**: Follow all style and testing requirements
3. **Documentation**: Update docs for any public API changes
4. **Testing**: Maintain 95%+ test coverage
5. **Security**: Follow security best practices
6. **Performance**: Profile and benchmark new features
7. **Reviews**: Get approval from at least two reviewers

### Development Checklist

Before submitting a pull request:

- [ ] Code follows APG style guidelines
- [ ] All tests pass with 95%+ coverage
- [ ] Documentation is updated
- [ ] Security review completed
- [ ] Performance impact assessed
- [ ] Integration tests pass
- [ ] Pre-commit hooks configured
- [ ] Migration scripts (if needed) are included

This comprehensive developer guide should help you get started with APG RAG development and contribute effectively to the project!