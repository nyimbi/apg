# APG Document Management - Developer Guide

**Complete Developer Guide for Revolutionary Document Management System**

Copyright © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>  
Website: www.datacraft.co.ke

## Table of Contents

1. [Development Environment](#development-environment)
2. [Architecture & Design](#architecture--design)
3. [API Development](#api-development)
4. [Service Integration](#service-integration)
5. [Database Schema](#database-schema)
6. [Testing Framework](#testing-framework)
7. [Performance Optimization](#performance-optimization)
8. [Deployment Patterns](#deployment-patterns)
9. [Extension Points](#extension-points)
10. [Contributing Guidelines](#contributing-guidelines)

## Development Environment

### Setup

#### Prerequisites
```bash
# Python 3.12+ with development tools
sudo apt install python3.12 python3.12-dev python3.12-venv
sudo apt install build-essential libpq-dev redis-server postgresql-client

# Node.js for frontend tooling
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs

# Docker for containerized development
sudo apt install docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

#### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd apg/capabilities/general_cross_functional/document_content_management

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Setup pre-commit hooks
pre-commit install

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
python scripts/init_dev_db.py

# Run development server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

#### IDE Configuration

**VS Code Settings (`.vscode/settings.json`)**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

**PyCharm Configuration**
- Python Interpreter: `./venv/bin/python`
- Code Style: Black formatter
- Inspections: Enable all Python inspections
- Testing: pytest as default test runner

### Development Tools

#### Code Quality Tools
```bash
# Linting
pylint src/
flake8 src/
mypy src/

# Formatting
black src/
isort src/

# Security scanning
bandit -r src/
safety check

# Dependency analysis
pip-audit
```

#### Testing Tools
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Coverage reporting
pytest --cov=src/ --cov-report=html

# Performance testing
pytest tests/performance/ -v
```

## Architecture & Design

### System Architecture

```python
"""
APG Document Management Architecture

┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Router  │  Flask-AppBuilder  │  WebSocket Handler     │
├─────────────────────────────────────────────────────────────────┤
│                     Service Layer                              │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│ Core Service│Business Logic │   Orchestration │   Event Handling │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│                    Engine Layer                                │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│ IDP Engine  │ Search Engine │ GenAI Engine  │ Analytics Engine │
├─────────────┼───────────────┼───────────────┼──────────────────┤
│Classification│ Retention    │ Content Fabric│  DLP Engine      │
│   Engine    │   Engine     │    Engine     │                  │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│                 Integration Layer                              │
├─────────────────────────────────────────────────────────────────┤
│ APG Clients │ External APIs │ Message Queues│ Event Bus        │
├─────────────────────────────────────────────────────────────────┤
│                    Data Layer                                  │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│ PostgreSQL  │ Redis Cache   │ File Storage  │ Vector Database  │
└─────────────┴───────────────┴───────────────┴──────────────────┘
"""
```

### Design Patterns

#### Service Layer Pattern
```python
"""Service layer implementation pattern"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

class BaseService(ABC):
    """Base service class with common functionality"""
    
    def __init__(self, session=None, cache=None, logger=None):
        self.session = session
        self.cache = cache
        self.logger = logger or self._get_logger()
    
    def _get_logger(self):
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    async def execute_with_transaction(self, operation):
        """Execute operation within database transaction"""
        try:
            result = await operation()
            await self.session.commit()
            return result
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise

class DocumentService(BaseService):
    """Document management service implementation"""
    
    async def create_document(
        self, 
        document_data: Dict[str, Any],
        file_path: str,
        user_id: str,
        tenant_id: str
    ) -> Document:
        """Create new document with validation and processing"""
        
        # Validation
        await self._validate_document_data(document_data)
        await self._validate_file(file_path)
        
        # Create document record
        document = Document(
            id=uuid7str(),
            tenant_id=tenant_id,
            created_by=user_id,
            **document_data
        )
        
        # Execute within transaction
        return await self.execute_with_transaction(
            lambda: self._create_document_transaction(document, file_path)
        )
```

#### Repository Pattern
```python
"""Repository pattern for data access"""

from typing import Generic, TypeVar, List, Optional

T = TypeVar('T')

class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations"""
    
    def __init__(self, session, model_class):
        self.session = session
        self.model_class = model_class
    
    async def create(self, entity: T) -> T:
        """Create new entity"""
        self.session.add(entity)
        await self.session.flush()
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID"""
        return await self.session.get(self.model_class, entity_id)
    
    async def get_by_tenant(
        self, 
        tenant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[T]:
        """Get entities by tenant with pagination"""
        query = select(self.model_class).where(
            self.model_class.tenant_id == tenant_id
        ).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        await self.session.merge(entity)
        await self.session.flush()
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Soft delete entity"""
        entity = await self.get_by_id(entity_id)
        if entity:
            entity.is_deleted = True
            entity.deleted_at = datetime.utcnow()
            await self.session.flush()
            return True
        return False

class DocumentRepository(BaseRepository[Document]):
    """Document-specific repository operations"""
    
    async def search_by_content(
        self,
        tenant_id: str,
        query: str,
        filters: Dict[str, Any] = None
    ) -> List[Document]:
        """Search documents by content using full-text search"""
        
        base_query = select(Document).where(
            Document.tenant_id == tenant_id,
            Document.is_active == True,
            Document.is_deleted == False
        )
        
        # Add full-text search
        if query:
            base_query = base_query.where(
                func.to_tsvector('english', 
                    Document.title + ' ' + 
                    Document.description + ' ' + 
                    Document.keywords
                ).match(query)
            )
        
        # Apply filters
        if filters:
            if 'document_type' in filters:
                base_query = base_query.where(
                    Document.document_type == filters['document_type']
                )
            if 'date_from' in filters:
                base_query = base_query.where(
                    Document.created_at >= filters['date_from']
                )
        
        result = await self.session.execute(base_query)
        return result.scalars().all()
```

#### Event-Driven Architecture
```python
"""Event-driven architecture implementation"""

from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class DomainEvent:
    """Base domain event"""
    event_type: str
    tenant_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime
    event_id: str = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid7str()

class EventBus:
    """Event bus for handling domain events"""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe handler to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def publish(self, event: DomainEvent):
        """Publish event to all subscribers"""
        handlers = self._handlers.get(event.event_type, [])
        
        # Execute handlers concurrently
        if handlers:
            await asyncio.gather(*[
                handler(event) for handler in handlers
            ], return_exceptions=True)

# Event definitions
@dataclass
class DocumentCreatedEvent(DomainEvent):
    def __init__(self, tenant_id: str, user_id: str, document_id: str, **kwargs):
        super().__init__(
            event_type='document.created',
            tenant_id=tenant_id,
            user_id=user_id,
            data={'document_id': document_id, **kwargs},
            timestamp=datetime.utcnow()
        )

# Event handlers
class DocumentEventHandlers:
    """Event handlers for document events"""
    
    def __init__(self, ai_service, audit_service, notification_service):
        self.ai_service = ai_service
        self.audit_service = audit_service
        self.notification_service = notification_service
    
    async def handle_document_created(self, event: DocumentCreatedEvent):
        """Handle document creation event"""
        document_id = event.data['document_id']
        
        # Trigger AI processing
        await self.ai_service.process_document_async(document_id)
        
        # Log audit event
        await self.audit_service.log_event(
            'document_created',
            event.tenant_id,
            event.user_id,
            {'document_id': document_id}
        )
        
        # Send notifications
        await self.notification_service.notify_stakeholders(
            event.tenant_id,
            f"New document created: {document_id}"
        )
```

### Configuration Management

#### Environment-Based Configuration
```python
"""Configuration management using Pydantic settings"""

from pydantic import BaseSettings, validator
from typing import List, Optional

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False
    
    class Config:
        env_prefix = "DATABASE_"

class RedisSettings(BaseSettings):
    """Redis configuration"""
    url: str = "redis://localhost:6379/0"
    pool_size: int = 20
    
    class Config:
        env_prefix = "REDIS_"

class APGIntegrationSettings(BaseSettings):
    """APG service integration settings"""
    ai_endpoint: str
    rag_endpoint: str
    genai_endpoint: str
    ml_endpoint: str
    blockchain_endpoint: str
    
    timeout: int = 30
    retries: int = 3
    
    class Config:
        env_prefix = "APG_"

class ApplicationSettings(BaseSettings):
    """Application settings"""
    debug: bool = False
    secret_key: str
    jwt_secret_key: str
    
    # Feature flags
    enable_ai_processing: bool = True
    enable_semantic_search: bool = True
    enable_blockchain: bool = True
    
    # Performance settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_concurrent_uploads: int = 10
    
    # Security settings
    session_timeout: int = 3600
    max_login_attempts: int = 5
    
    @validator('secret_key', 'jwt_secret_key')
    def validate_secrets(cls, v):
        if len(v) < 32:
            raise ValueError('Secret keys must be at least 32 characters')
        return v

class Settings:
    """Main settings class"""
    
    def __init__(self):
        self.database = DatabaseSettings()
        self.redis = RedisSettings()
        self.apg = APGIntegrationSettings()
        self.app = ApplicationSettings()
    
    @property
    def is_development(self) -> bool:
        return self.app.debug
    
    @property
    def is_production(self) -> bool:
        return not self.app.debug

# Global settings instance
settings = Settings()
```

## API Development

### FastAPI Implementation

#### Router Structure
```python
"""API router implementation"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any

from ..models import DocumentCreate, DocumentResponse, SearchRequest, SearchResponse
from ..service import DocumentManagementService
from ..auth import get_current_user, get_current_tenant

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
security = HTTPBearer()

@router.post("/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    file: UploadFile = File(None),
    service: DocumentManagementService = Depends(),
    current_user = Depends(get_current_user),
    current_tenant = Depends(get_current_tenant)
):
    """Create new document with optional file upload"""
    
    try:
        # Validate file if provided
        if file:
            await validate_uploaded_file(file)
        
        # Create document
        result = await service.create_document(
            document_data=document.dict(),
            file_path=file.filename if file else None,
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            process_ai=document.process_ai
        )
        
        return DocumentResponse.from_orm(result)
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: DocumentManagementService = Depends(),
    current_user = Depends(get_current_user),
    current_tenant = Depends(get_current_tenant)
):
    """Get document by ID"""
    
    document = await service.get_document(
        document_id, 
        current_user.id, 
        current_tenant.id
    )
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse.from_orm(document)

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest,
    service: DocumentManagementService = Depends(),
    current_user = Depends(get_current_user),
    current_tenant = Depends(get_current_tenant)
):
    """Search documents using semantic search"""
    
    try:
        result = await service.search_documents(
            query=search_request.query,
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            search_options=search_request.options.dict() if search_request.options else {}
        )
        
        return SearchResponse(
            query=search_request.query,
            total_results=len(result.matching_documents),
            search_result=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/interact")
async def interact_with_document(
    document_id: str,
    interaction_request: InteractionRequest,
    service: DocumentManagementService = Depends(),
    current_user = Depends(get_current_user)
):
    """Interact with document using generative AI"""
    
    try:
        result = await service.interact_with_content(
            document_id=document_id,
            user_prompt=interaction_request.user_prompt,
            interaction_type=interaction_request.interaction_type,
            user_id=current_user.id,
            context_documents=interaction_request.context_documents or [],
            options=interaction_request.options or {}
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### Request/Response Models
```python
"""Pydantic models for API requests and responses"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Document type enumeration"""
    TEXT_DOCUMENT = "text_document"
    CONTRACT = "contract"
    INVOICE = "invoice"
    POLICY = "policy"
    MANUAL = "manual"
    EMAIL = "email"
    TEMPORARY = "temporary"

class ContentFormat(str, Enum):
    """Content format enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"
    HTML = "html"
    JSON = "json"
    XML = "xml"

class DocumentCreate(BaseModel):
    """Document creation request"""
    name: str = Field(..., min_length=1, max_length=255)
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    document_type: DocumentType
    content_format: ContentFormat
    keywords: Optional[List[str]] = Field(default_factory=list)
    categories: Optional[List[str]] = Field(default_factory=list)
    process_ai: bool = Field(default=True)
    
    @validator('keywords', 'categories')
    def validate_string_lists(cls, v):
        if v and len(v) > 20:
            raise ValueError('Maximum 20 items allowed')
        return v

class DocumentResponse(BaseModel):
    """Document response model"""
    id: str
    name: str
    title: str
    description: Optional[str]
    document_type: DocumentType
    content_format: ContentFormat
    file_name: Optional[str]
    file_size: Optional[int]
    keywords: List[str]
    categories: List[str]
    ai_tags: Optional[List[str]]
    content_summary: Optional[str]
    sentiment_score: Optional[float]
    view_count: int
    download_count: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    tenant_id: str
    
    class Config:
        from_attributes = True

class SearchOptions(BaseModel):
    """Search options"""
    semantic_search: bool = Field(default=True)
    include_content: bool = Field(default=False)
    limit: int = Field(default=50, ge=1, le=1000)
    filters: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=500)
    options: Optional[SearchOptions] = None

class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    total_results: int
    search_result: Dict[str, Any]
    
class InteractionType(str, Enum):
    """AI interaction types"""
    SUMMARIZE = "summarize"
    QA = "qa"
    TRANSLATE = "translate"
    ENHANCE = "enhance"
    GENERATE = "generate"
    EXTRACT = "extract"
    COMPARE = "compare"
    ANALYZE = "analyze"

class InteractionRequest(BaseModel):
    """AI interaction request"""
    user_prompt: str = Field(..., min_length=1, max_length=1000)
    interaction_type: InteractionType
    context_documents: Optional[List[str]] = Field(default_factory=list)
    options: Optional[Dict[str, Any]] = None
    
    @validator('context_documents')
    def validate_context_documents(cls, v):
        if v and len(v) > 10:
            raise ValueError('Maximum 10 context documents allowed')
        return v
```

#### Error Handling
```python
"""Centralized error handling"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

class DocumentManagementException(Exception):
    """Base exception for document management"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class DocumentNotFoundError(DocumentManagementException):
    """Document not found error"""
    
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            error_code="DOCUMENT_NOT_FOUND",
            details={"document_id": document_id}
        )

class InsufficientPermissionsError(DocumentManagementException):
    """Insufficient permissions error"""
    
    def __init__(self, user_id: str, resource: str, action: str):
        super().__init__(
            message=f"Insufficient permissions for {action} on {resource}",
            error_code="INSUFFICIENT_PERMISSIONS",
            details={
                "user_id": user_id,
                "resource": resource,
                "action": action
            }
        )

async def document_management_exception_handler(
    request: Request, 
    exc: DocumentManagementException
):
    """Handle document management exceptions"""
    
    logger.error(f"DocumentManagementException: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "path": request.url.path
    })
    
    status_code = 400
    if isinstance(exc, DocumentNotFoundError):
        status_code = 404
    elif isinstance(exc, InsufficientPermissionsError):
        status_code = 403
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    
    logger.warning(f"Validation error: {exc.errors()}", extra={
        "path": request.url.path,
        "body": exc.body
    })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors()
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "path": request.url.path,
        "exception_type": type(exc).__name__
    }, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": str(uuid7str())
        }
    )
```

### Authentication & Authorization

#### JWT Authentication
```python
"""JWT-based authentication implementation"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from ..models import User, Tenant
from ..settings import settings

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Authentication service"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        
        return jwt.encode(
            to_encode, 
            settings.app.jwt_secret_key, 
            algorithm="HS256"
        )
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                settings.app.jwt_secret_key, 
                algorithms=["HS256"]
            )
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user"""
    
    payload = AuthService.verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    # Get user from database
    user = await User.get_by_id(user_id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user

async def get_current_tenant(current_user: User = Depends(get_current_user)) -> Tenant:
    """Get current user's tenant"""
    
    tenant = await Tenant.get_by_id(current_user.tenant_id)
    
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant not found",
        )
    
    return tenant

def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    
    def permission_checker(current_user: User = Depends(get_current_user)):
        user_permissions = current_user.get_permissions()
        
        if not all(perm in user_permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        
        return current_user
    
    return permission_checker
```

## Service Integration

### APG Platform Integration

#### AI Service Client
```python
"""APG AI service integration"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from ..settings import settings

class APGAIClient:
    """APG AI service client"""
    
    def __init__(self):
        self.base_url = settings.apg.ai_endpoint
        self.timeout = aiohttp.ClientTimeout(total=settings.apg.timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_document(
        self, 
        document_content: bytes,
        document_type: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process document using AI"""
        
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        form_data = aiohttp.FormData()
        form_data.add_field('file', document_content, filename='document')
        form_data.add_field('document_type', document_type)
        
        if options:
            form_data.add_field('options', json.dumps(options))
        
        async with self.session.post(
            f"{self.base_url}/process",
            data=form_data
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"AI processing failed: {error_text}")
    
    async def classify_content(
        self,
        content: str,
        classification_types: List[str] = None
    ) -> Dict[str, Any]:
        """Classify content using AI"""
        
        payload = {
            'content': content,
            'classification_types': classification_types or ['document_type', 'category', 'sentiment']
        }
        
        async with self.session.post(
            f"{self.base_url}/classify",
            json=payload
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Classification failed: {error_text}")
    
    async def extract_entities(
        self,
        content: str,
        entity_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities from content"""
        
        payload = {
            'content': content,
            'entity_types': entity_types or ['person', 'organization', 'location', 'date', 'money']
        }
        
        async with self.session.post(
            f"{self.base_url}/extract-entities",
            json=payload
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                return result.get('entities', [])
            else:
                error_text = await response.text()
                raise Exception(f"Entity extraction failed: {error_text}")

class APGGenAIClient:
    """APG Generative AI service client"""
    
    def __init__(self):
        self.base_url = settings.apg.genai_endpoint
        self.timeout = aiohttp.ClientTimeout(total=settings.apg.timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_summary(
        self,
        content: str,
        summary_type: str = "executive",
        max_length: int = 300
    ) -> Dict[str, Any]:
        """Generate content summary"""
        
        payload = {
            'content': content,
            'summary_type': summary_type,
            'max_length': max_length
        }
        
        async with self.session.post(
            f"{self.base_url}/summarize",
            json=payload
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Summary generation failed: {error_text}")
    
    async def answer_question(
        self,
        content: str,
        question: str,
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Answer question about content"""
        
        payload = {
            'content': content,
            'question': question,
            'context': context or []
        }
        
        async with self.session.post(
            f"{self.base_url}/qa",
            json=payload
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Question answering failed: {error_text}")
    
    async def translate_content(
        self,
        content: str,
        target_language: str,
        source_language: str = "auto"
    ) -> Dict[str, Any]:
        """Translate content to target language"""
        
        payload = {
            'content': content,
            'target_language': target_language,
            'source_language': source_language
        }
        
        async with self.session.post(
            f"{self.base_url}/translate",
            json=payload
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Translation failed: {error_text}")
```

#### Service Integration Patterns
```python
"""Service integration patterns with retry and circuit breaker"""

import asyncio
from typing import Any, Callable, TypeVar, Optional
from functools import wraps
import time

T = TypeVar('T')

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker"""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = backoff_factor * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator

class APGServiceManager:
    """Manages APG service integrations with resilience patterns"""
    
    def __init__(self):
        self.ai_client = None
        self.genai_client = None
        self.rag_client = None
        
        # Circuit breakers for each service
        self.ai_circuit_breaker = CircuitBreaker()
        self.genai_circuit_breaker = CircuitBreaker()
        self.rag_circuit_breaker = CircuitBreaker()
    
    async def initialize(self):
        """Initialize service clients"""
        self.ai_client = APGAIClient()
        self.genai_client = APGGenAIClient()
        # Initialize other clients...
    
    @retry_with_backoff(max_retries=3, backoff_factor=1.0)
    async def process_document_with_ai(
        self,
        document_content: bytes,
        document_type: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process document with AI service using resilience patterns"""
        
        if not self.ai_client:
            await self.initialize()
        
        return await self.ai_circuit_breaker.call(
            self.ai_client.process_document,
            document_content,
            document_type,
            options
        )
    
    @retry_with_backoff(max_retries=2, backoff_factor=0.5)
    async def generate_summary_with_genai(
        self,
        content: str,
        summary_type: str = "executive",
        max_length: int = 300
    ) -> Dict[str, Any]:
        """Generate summary with GenAI service using resilience patterns"""
        
        if not self.genai_client:
            await self.initialize()
        
        return await self.genai_circuit_breaker.call(
            self.genai_client.generate_summary,
            content,
            summary_type,
            max_length
        )
```

## Database Schema

### SQLAlchemy Models

#### Core Models
```python
"""SQLAlchemy models for document management"""

from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from datetime import datetime
from uuid_extensions import uuid7str
import sqlalchemy as sa

Base = declarative_base()

class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class TenantMixin:
    """Mixin for multi-tenant support"""
    tenant_id = Column(String(255), nullable=False, index=True)

class AuditMixin:
    """Mixin for audit fields"""
    created_by = Column(String(255), nullable=False)
    updated_by = Column(String(255), nullable=False)

class DCMDocument(Base, TimestampMixin, TenantMixin, AuditMixin):
    """Core document model"""
    __tablename__ = 'dcm_documents'
    
    # Primary key
    id = Column(String(255), primary_key=True, default=uuid7str)
    
    # Basic document info
    name = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    document_type = Column(String(50), nullable=False)
    content_format = Column(String(50), nullable=False)
    
    # File information
    file_name = Column(String(255))
    file_path = Column(String(1000))
    file_size = Column(Integer)
    mime_type = Column(String(255))
    checksum = Column(String(255))
    
    # Document metadata
    keywords = Column(ARRAY(String), default=list)
    categories = Column(ARRAY(String), default=list)
    
    # AI-generated metadata
    ai_tags = Column(ARRAY(String), default=list)
    content_summary = Column(Text)
    sentiment_score = Column(Float)
    ai_confidence_score = Column(Float)
    
    # Status and lifecycle
    status = Column(String(50), default='active')
    version_number = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime)
    
    # Usage tracking
    view_count = Column(Integer, default=0)
    download_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime)
    
    # Relationships
    versions = relationship("DCMDocumentVersion", back_populates="document")
    permissions = relationship("DCMPermission", back_populates="document")
    comments = relationship("DCMComment", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_tenant_status', 'tenant_id', 'status'),
        Index('idx_documents_type_tenant', 'document_type', 'tenant_id'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_fts', sa.text("to_tsvector('english', title || ' ' || description || ' ' || array_to_string(keywords, ' '))")),
    )

class DCMDocumentVersion(Base, TimestampMixin, TenantMixin):
    """Document version model"""
    __tablename__ = 'dcm_document_versions'
    
    id = Column(String(255), primary_key=True, default=uuid7str)
    document_id = Column(String(255), ForeignKey('dcm_documents.id'), nullable=False)
    version_number = Column(Integer, nullable=False)
    version_label = Column(String(255))
    
    # Version content
    file_path = Column(String(1000))
    file_size = Column(Integer)
    checksum = Column(String(255))
    
    # Change tracking
    changed_by = Column(String(255), nullable=False)
    change_description = Column(Text)
    change_type = Column(String(50))  # created, updated, restored
    
    # Status
    is_current = Column(Boolean, default=False)
    status = Column(String(50), default='active')
    
    # Relationships
    document = relationship("DCMDocument", back_populates="versions")
    
    __table_args__ = (
        Index('idx_versions_document_number', 'document_id', 'version_number'),
    )

class DCMPermission(Base, TimestampMixin, TenantMixin):
    """Document permission model"""
    __tablename__ = 'dcm_permissions'
    
    id = Column(String(255), primary_key=True, default=uuid7str)
    document_id = Column(String(255), ForeignKey('dcm_documents.id'))
    folder_id = Column(String(255), ForeignKey('dcm_folders.id'))
    
    # Permission subject
    subject_type = Column(String(50), nullable=False)  # user, group, role
    subject_id = Column(String(255), nullable=False)
    
    # Permission level
    permission_level = Column(String(50), nullable=False)
    can_read = Column(Boolean, default=False)
    can_write = Column(Boolean, default=False)
    can_delete = Column(Boolean, default=False)
    can_share = Column(Boolean, default=False)
    can_approve = Column(Boolean, default=False)
    
    # Permission metadata
    granted_by = Column(String(255), nullable=False)
    effective_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime)
    applies_to_children = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    document = relationship("DCMDocument", back_populates="permissions")
    folder = relationship("DCMFolder", back_populates="permissions")

class DCMIntelligentProcessing(Base, TimestampMixin, TenantMixin):
    """AI processing results model"""
    __tablename__ = 'dcm_intelligent_processing'
    
    id = Column(String(255), primary_key=True, default=uuid7str)
    document_id = Column(String(255), ForeignKey('dcm_documents.id'), nullable=False)
    
    # Processing metadata
    processing_type = Column(String(100), nullable=False)
    ai_model_id = Column(String(255))
    rag_context_id = Column(String(255))
    
    # Processing results
    extracted_data = Column(JSONB)
    confidence_score = Column(Float)
    processing_time_ms = Column(Integer)
    
    # Status
    status = Column(String(50), default='completed')
    error_message = Column(Text)
    
    __table_args__ = (
        Index('idx_processing_document_type', 'document_id', 'processing_type'),
    )

class DCMRetentionPolicy(Base, TimestampMixin, TenantMixin):
    """Retention policy model"""
    __tablename__ = 'dcm_retention_policies'
    
    id = Column(String(255), primary_key=True, default=uuid7str)
    name = Column(String(255), nullable=False)
    policy_code = Column(String(100), unique=True)
    description = Column(Text)
    
    # Retention periods
    retention_period_years = Column(Integer)
    retention_period_months = Column(Integer)
    retention_period_days = Column(Integer)
    
    # Policy settings
    auto_delete_enabled = Column(Boolean, default=False)
    auto_archive_enabled = Column(Boolean, default=True)
    legal_hold_override = Column(Boolean, default=False)
    
    # Regulatory basis
    regulatory_basis = Column(String(255))
    
    # Policy lifecycle
    is_active = Column(Boolean, default=True)
    effective_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime)
    
    created_by = Column(String(255), nullable=False)
    updated_by = Column(String(255), nullable=False)
```

#### Migration Scripts
```python
"""Alembic migration example"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create initial document management schema"""
    
    # Create documents table
    op.create_table(
        'dcm_documents',
        sa.Column('id', sa.String(255), primary_key=True),
        sa.Column('tenant_id', sa.String(255), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('document_type', sa.String(50), nullable=False),
        sa.Column('content_format', sa.String(50), nullable=False),
        
        # File information
        sa.Column('file_name', sa.String(255)),
        sa.Column('file_path', sa.String(1000)),
        sa.Column('file_size', sa.Integer),
        sa.Column('mime_type', sa.String(255)),
        sa.Column('checksum', sa.String(255)),
        
        # Metadata
        sa.Column('keywords', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('categories', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('ai_tags', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('content_summary', sa.Text),
        sa.Column('sentiment_score', sa.Float),
        
        # Status
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('version_number', sa.Integer, default=1),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_deleted', sa.Boolean, default=False),
        
        # Audit fields
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('updated_by', sa.String(255), nullable=False),
    )
    
    # Create indexes
    op.create_index('idx_documents_tenant_status', 'dcm_documents', ['tenant_id', 'status'])
    op.create_index('idx_documents_type_tenant', 'dcm_documents', ['document_type', 'tenant_id'])
    op.create_index('idx_documents_created_at', 'dcm_documents', ['created_at'])
    
    # Create full-text search index
    op.execute("""
        CREATE INDEX idx_documents_fts ON dcm_documents 
        USING gin(to_tsvector('english', title || ' ' || description || ' ' || array_to_string(keywords, ' ')))
    """)
    
    # Create document versions table
    op.create_table(
        'dcm_document_versions',
        sa.Column('id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), sa.ForeignKey('dcm_documents.id'), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('version_number', sa.Integer, nullable=False),
        sa.Column('version_label', sa.String(255)),
        sa.Column('file_path', sa.String(1000)),
        sa.Column('file_size', sa.Integer),
        sa.Column('checksum', sa.String(255)),
        sa.Column('changed_by', sa.String(255), nullable=False),
        sa.Column('change_description', sa.Text),
        sa.Column('change_type', sa.String(50)),
        sa.Column('is_current', sa.Boolean, default=False),
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    
    op.create_index('idx_versions_document_number', 'dcm_document_versions', ['document_id', 'version_number'])

def downgrade():
    """Drop document management schema"""
    op.drop_table('dcm_document_versions')
    op.drop_table('dcm_documents')
```

## Testing Framework

### Unit Testing

#### Test Structure
```python
"""Unit test example"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ..service import DocumentManagementService
from ..models import DCMDocument, DCMDocumentType, DCMContentFormat

class TestDocumentManagementService:
    """Test suite for DocumentManagementService"""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing"""
        return DocumentManagementService()
    
    @pytest.fixture
    def sample_document_data(self):
        """Sample document data for testing"""
        return {
            'name': 'Test Document',
            'title': 'Test Document Title',
            'description': 'Test document description',
            'document_type': DCMDocumentType.TEXT_DOCUMENT,
            'content_format': DCMContentFormat.PDF,
            'keywords': ['test', 'document'],
            'categories': ['testing']
        }
    
    @pytest.mark.asyncio
    async def test_create_document_success(self, service, sample_document_data):
        """Test successful document creation"""
        
        # Mock dependencies
        service.idp_processor.process_document = AsyncMock(return_value=MagicMock(
            extracted_data={'text_content': 'Sample content'},
            confidence_score=0.95
        ))
        
        service.classification_engine.classify_document = AsyncMock(return_value=MagicMock(
            ai_classification={'document_type': {'primary_type': 'contract', 'confidence': 0.9}},
            content_summary='Test summary'
        ))
        
        # Execute test
        document = await service.create_document(
            document_data=sample_document_data,
            file_path='/tmp/test.pdf',
            user_id='test-user',
            tenant_id='test-tenant',
            process_ai=True
        )
        
        # Assertions
        assert document.name == 'Test Document'
        assert document.tenant_id == 'test-tenant'
        assert document.created_by == 'test-user'
        assert service.idp_processor.process_document.called
        assert service.classification_engine.classify_document.called
    
    @pytest.mark.asyncio
    async def test_create_document_validation_error(self, service):
        """Test document creation with validation error"""
        
        invalid_data = {
            'name': '',  # Invalid: empty name
            'title': 'Test',
            'document_type': 'invalid_type',  # Invalid type
            'content_format': DCMContentFormat.PDF
        }
        
        with pytest.raises(ValueError, match="Invalid document data"):
            await service.create_document(
                document_data=invalid_data,
                file_path='/tmp/test.pdf',
                user_id='test-user',
                tenant_id='test-tenant'
            )
    
    @pytest.mark.asyncio
    async def test_search_documents_semantic(self, service):
        """Test semantic document search"""
        
        # Mock search engine
        service.search_engine.search_documents = AsyncMock(return_value=MagicMock(
            matching_documents=['doc1', 'doc2'],
            semantic_similarity_scores=[0.95, 0.87],
            intent_classification={'intent': 'find_documents', 'confidence': 0.9},
            confidence_score=0.88
        ))
        
        # Execute search
        result = await service.search_documents(
            query='legal contracts',
            user_id='test-user',
            tenant_id='test-tenant'
        )
        
        # Assertions
        assert len(result.matching_documents) == 2
        assert result.confidence_score == 0.88
        assert service.search_engine.search_documents.called
    
    @pytest.mark.asyncio
    async def test_ai_interaction_summarization(self, service):
        """Test AI-powered document summarization"""
        
        # Mock GenAI engine
        service.genai_engine.process_interaction = AsyncMock(return_value=MagicMock(
            interaction_type='summarize',
            genai_response='This is a test document summary.',
            confidence_score=0.91
        ))
        
        # Execute interaction
        result = await service.interact_with_content(
            document_id='test-doc-123',
            user_prompt='Summarize this document',
            interaction_type='summarize',
            user_id='test-user'
        )
        
        # Assertions
        assert result.interaction_type == 'summarize'
        assert 'test document summary' in result.genai_response
        assert result.confidence_score == 0.91
```

#### Integration Testing
```python
"""Integration test example"""

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..main import app
from ..database import Base
from ..settings import settings

class TestDocumentAPI:
    """Integration tests for document API"""
    
    @pytest.fixture(scope="session")
    async def test_db(self):
        """Create test database"""
        engine = create_async_engine(
            "postgresql+asyncpg://test:test@localhost/test_docmgmt",
            echo=False
        )
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield engine
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        await engine.dispose()
    
    @pytest.fixture
    async def client(self, test_db):
        """Create test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    async def auth_headers(self, client):
        """Get authentication headers"""
        # Login and get token
        login_response = await client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "testpassword"
        })
        
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.mark.asyncio
    async def test_create_document_endpoint(self, client, auth_headers):
        """Test document creation endpoint"""
        
        document_data = {
            "name": "Test Document",
            "title": "Test Document Title",
            "description": "Test description",
            "document_type": "text_document",
            "content_format": "pdf",
            "keywords": ["test", "integration"],
            "categories": ["testing"],
            "process_ai": True
        }
        
        response = await client.post(
            "/api/v1/documents/",
            json=document_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["name"] == "Test Document"
        assert result["document_type"] == "text_document"
        assert "id" in result
    
    @pytest.mark.asyncio
    async def test_search_documents_endpoint(self, client, auth_headers):
        """Test document search endpoint"""
        
        search_request = {
            "query": "test document",
            "options": {
                "semantic_search": True,
                "limit": 10
            }
        }
        
        response = await client.post(
            "/api/v1/documents/search",
            json=search_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert "query" in result
        assert "total_results" in result
        assert "search_result" in result
    
    @pytest.mark.asyncio
    async def test_document_interaction_endpoint(self, client, auth_headers):
        """Test AI document interaction endpoint"""
        
        # First create a document
        document_data = {
            "name": "Contract Document",
            "title": "Service Agreement",
            "document_type": "contract",
            "content_format": "pdf"
        }
        
        create_response = await client.post(
            "/api/v1/documents/",
            json=document_data,
            headers=auth_headers
        )
        
        document_id = create_response.json()["id"]
        
        # Test interaction
        interaction_request = {
            "user_prompt": "Summarize this contract",
            "interaction_type": "summarize",
            "options": {"max_length": 300}
        }
        
        response = await client.post(
            f"/api/v1/documents/{document_id}/interact",
            json=interaction_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["interaction_type"] == "summarize"
        assert "genai_response" in result
```

#### Performance Testing
```python
"""Performance test example using pytest-benchmark"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from ..service import DocumentManagementService

class TestPerformance:
    """Performance tests"""
    
    @pytest.fixture
    def service(self):
        return DocumentManagementService()
    
    @pytest.mark.asyncio
    async def test_document_creation_performance(self, service, benchmark):
        """Test document creation performance"""
        
        document_data = {
            'name': 'Performance Test Document',
            'title': 'Performance Test',
            'document_type': 'text_document',
            'content_format': 'pdf'
        }
        
        async def create_document():
            return await service.create_document(
                document_data=document_data,
                file_path=None,
                user_id='perf-user',
                tenant_id='perf-tenant',
                process_ai=False
            )
        
        # Benchmark the operation
        result = benchmark.pedantic(
            lambda: asyncio.run(create_document()),
            rounds=10,
            iterations=1
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_document_operations(self, service):
        """Test concurrent document operations"""
        
        async def create_document(doc_id):
            document_data = {
                'name': f'Concurrent Document {doc_id}',
                'title': f'Concurrent Test {doc_id}',
                'document_type': 'text_document',
                'content_format': 'pdf'
            }
            
            return await service.create_document(
                document_data=document_data,
                file_path=None,
                user_id=f'user-{doc_id}',
                tenant_id='test-tenant',
                process_ai=False
            )
        
        # Test concurrent creation
        start_time = time.time()
        
        tasks = [create_document(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 50
        assert duration < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_search_performance_with_large_dataset(self, service):
        """Test search performance with large dataset"""
        
        # This would require a pre-populated test database
        search_queries = [
            'contract agreement',
            'financial report',
            'employee handbook',
            'technical documentation',
            'legal document'
        ]
        
        async def search_documents(query):
            return await service.search_documents(
                query=query,
                user_id='perf-user',
                tenant_id='perf-tenant'
            )
        
        start_time = time.time()
        
        # Execute searches concurrently
        tasks = [search_documents(query) for query in search_queries * 10]  # 50 searches
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify performance
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 50
        assert duration < 15.0  # All searches should complete within 15 seconds
        
        # Check average response time
        avg_response_time = duration / len(successful_results)
        assert avg_response_time < 0.3  # Average response time < 300ms
```

---

**Development Support**

- 📧 Developer Support: dev-support@datacraft.co.ke
- 🌐 Developer Portal: [dev.datacraft.co.ke](https://dev.datacraft.co.ke)
- 📖 API Docs: [docs.datacraft.co.ke/api](https://docs.datacraft.co.ke/api)
- 💬 Developer Chat: [chat.datacraft.co.ke](https://chat.datacraft.co.ke)

**Powered by APG Platform** | **Built with ❤️ by Datacraft**