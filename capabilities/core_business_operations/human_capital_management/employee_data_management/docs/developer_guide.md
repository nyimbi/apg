# APG Employee Data Management - Developer Guide

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Environment](#development-environment)
- [Core Components](#core-components)
- [API Development](#api-development)
- [Database Schema](#database-schema)
- [AI Integration](#ai-integration)
- [Testing Framework](#testing-framework)
- [Deployment Pipeline](#deployment-pipeline)
- [Performance Optimization](#performance-optimization)
- [Security Implementation](#security-implementation)

## 🏗️ Architecture Overview

### System Architecture

The Employee Data Management capability follows a microservices architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   API Gateway   │    │  AI Services    │
│                 │    │                 │    │                 │
│ - React/Vue     │◄──►│ - Rate Limiting │◄──►│ - ML Models     │
│ - Real-time UI  │    │ - Auth/AuthZ    │    │ - Predictions   │
│ - Analytics     │    │ - Caching       │    │ - NLP           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Core Service   │    │   Data Layer    │    │  Integration    │
    │                 │    │                 │    │                 │
    │ - Business      │◄──►│ - PostgreSQL    │◄──►│ - External APIs │
    │   Logic         │    │ - Redis Cache   │    │ - Webhooks      │
    │ - Validation    │    │ - Vector DB     │    │ - Event Stream  │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

#### Backend
- **Python 3.11+**: Core language with async/await support
- **FastAPI/Flask**: API framework with automatic documentation
- **SQLAlchemy 2.0**: ORM with async support
- **Pydantic v2**: Data validation and serialization
- **PostgreSQL 14+**: Primary database with advanced features
- **Redis 7+**: Caching and session storage

#### AI/ML
- **OpenAI GPT-4**: Natural language processing
- **Anthropic Claude**: Alternative LLM support
- **Ollama**: Local LLM deployment option
- **pgvector**: Vector similarity search
- **scikit-learn**: Traditional ML algorithms

#### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **nginx**: Load balancing and reverse proxy
- **Prometheus**: Monitoring and metrics
- **Grafana**: Visualization and alerting

### Design Patterns

#### Service Layer Pattern
```python
class RevolutionaryEmployeeDataManagementService:
    """Core service implementing business logic"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.db_session = get_async_session(tenant_id)
        self.ai_service = AIOrchestrationService(tenant_id)
        self.cache = RedisCache(tenant_id)
    
    async def create_employee_revolutionary(self, data: dict) -> OperationResult:
        # Implementation with validation, AI enhancement, caching
        pass
```

#### Repository Pattern
```python
class EmployeeRepository:
    """Data access layer abstraction"""
    
    async def create(self, employee: HREmployee) -> HREmployee:
        async with self.session.begin():
            self.session.add(employee)
            await self.session.flush()
            return employee
    
    async def get_by_id(self, employee_id: str) -> Optional[HREmployee]:
        return await self.session.get(HREmployee, employee_id)
```

#### Command Query Responsibility Segregation (CQRS)
- **Commands**: Write operations (create, update, delete)
- **Queries**: Read operations (search, analytics, reporting)
- **Separate optimization**: Different strategies for reads vs writes

## 🛠️ Development Environment

### Prerequisites

```bash
# Python 3.11+
python --version  # Should be 3.11 or higher

# PostgreSQL with extensions
psql --version    # PostgreSQL 14+
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

# Redis
redis-cli --version  # Redis 7+

# Docker (optional)
docker --version
docker-compose --version
```

### Local Setup

```bash
# Clone repository
git clone <apg-repository>
cd apg/capabilities/core_business_operations/human_capital_management/employee_data_management

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Environment configuration
cp .env.example .env
# Edit .env with your configuration

# Database setup
alembic upgrade head

# Run tests
pytest

# Start development server
python run.py
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/apg_hr
DATABASE_POOL_SIZE=20
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=apg_hr

# AI Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AI_MODEL_PROVIDER=openai
AI_MODEL_NAME=gpt-4

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600

# APG Platform
APG_TENANT_ID=development
APG_AI_ORCHESTRATION_URL=http://localhost:8001
APG_FEDERATED_LEARNING_URL=http://localhost:8002
```

### IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)
```json
{
    "python.defaultInterpreter": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm Configuration
- Set Python interpreter to virtual environment
- Enable pytest as test runner
- Configure black as code formatter
- Set up type checking with mypy

## 🔧 Core Components

### Models (`models.py`)

#### Employee Model
```python
from sqlalchemy import Column, String, Date, Decimal, JSON, Index
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from uuid_extensions import uuid7str

class HREmployee(Model, AuditMixin, BaseMixin):
    __tablename__ = 'hr_employees'
    
    # Core fields
    id: str = Field(default_factory=uuid7str, primary_key=True)
    employee_number: str = Field(unique=True, index=True)
    first_name: str = Field(max_length=100)
    last_name: str = Field(max_length=100)
    work_email: str = Field(unique=True, index=True)
    
    # AI Enhancement
    profile_embedding: Optional[List[float]] = Field(default=None)
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_hr_employee_search', 'first_name', 'last_name', 'work_email'),
        Index('ix_hr_employee_department', 'department_id', 'employment_status'),
        Index('ix_hr_employee_embedding', 'profile_embedding', postgresql_using='ivfflat'),
    )
```

#### AI Profile Model
```python
class HREmployeeAIProfile(Model, BaseMixin):
    __tablename__ = 'hr_edm_ai_profile'
    
    employee_id: str = Field(foreign_key='hr_employees.id')
    
    # AI Embeddings
    profile_embedding: Optional[List[float]] = Field(default=None)
    skills_embedding: Optional[List[float]] = Field(default=None)
    
    # Predictive Scores
    retention_risk_score: Optional[Decimal] = Field(decimal_places=4)
    performance_prediction: Optional[Decimal] = Field(decimal_places=4)
    promotion_readiness_score: Optional[Decimal] = Field(decimal_places=4)
    
    # Analysis Results
    skills_analysis: Dict[str, Any] = Field(default_factory=dict)
    career_recommendations: List[str] = Field(default_factory=list)
    last_analysis_date: Optional[datetime] = Field(default=None)
```

### Service Layer (`service.py`)

#### Core Service Implementation
```python
class RevolutionaryEmployeeDataManagementService:
    def __init__(self, tenant_id: str, session: Optional[AsyncSession] = None):
        self.tenant_id = tenant_id
        self.session = session or get_async_session(tenant_id)
        self.ai_orchestration = AIOrchestrationService(tenant_id)
        self.cache = RedisCache(f"employee_service:{tenant_id}")
        self.validator = IntelligentDataQualityEngine(tenant_id)
    
    async def create_employee_revolutionary(self, employee_data: Dict[str, Any]) -> OperationResult:
        """Create employee with AI enhancement and validation"""
        
        # Data validation and quality assessment
        validation_result = await self.validator.validate_employee_data(employee_data)
        if not validation_result.is_valid:
            return OperationResult(
                success=False,
                validation_errors=validation_result.errors
            )
        
        # AI enhancement
        enhanced_data = await self._enhance_employee_data_with_ai(employee_data)
        
        # Create employee record
        async with self.session.begin():
            employee = HREmployee(**enhanced_data)
            self.session.add(employee)
            await self.session.flush()
            
            # Generate AI profile
            ai_profile = await self._generate_ai_profile(employee)
            self.session.add(ai_profile)
            
            # Cache the result
            await self.cache.set(f"employee:{employee.id}", employee.dict())
            
            return OperationResult(
                success=True,
                employee_data=employee.dict()
            )
```

#### Advanced Search Implementation
```python
async def search_employees_intelligent(
    self, 
    search_criteria: EmployeeSearchCriteria
) -> EmployeeSearchResult:
    """Intelligent search with AI-powered ranking"""
    
    query = select(HREmployee)
    
    # Text search with full-text capabilities
    if search_criteria.search_text:
        # Vector similarity search for AI-enhanced matching
        if search_criteria.use_ai_search:
            search_embedding = await self.ai_orchestration.generate_embedding(
                search_criteria.search_text
            )
            query = query.where(
                HREmployee.profile_embedding.cosine_distance(search_embedding) < 0.3
            ).order_by(HREmployee.profile_embedding.cosine_distance(search_embedding))
        else:
            # Traditional text search
            query = query.where(
                or_(
                    HREmployee.first_name.ilike(f"%{search_criteria.search_text}%"),
                    HREmployee.last_name.ilike(f"%{search_criteria.search_text}%"),
                    HREmployee.work_email.ilike(f"%{search_criteria.search_text}%")
                )
            )
    
    # Apply filters
    if search_criteria.department_id:
        query = query.where(HREmployee.department_id == search_criteria.department_id)
    
    # Execute query with pagination
    result = await self.session.execute(
        query.offset(search_criteria.offset).limit(search_criteria.limit)
    )
    
    employees = result.scalars().all()
    return EmployeeSearchResult(
        employees=[emp.dict() for emp in employees],
        total_count=len(employees)
    )
```

### AI Intelligence Engine (`ai_intelligence_engine.py`)

#### Employee Analysis
```python
class EmployeeAIIntelligenceEngine:
    async def analyze_employee_comprehensive(self, employee_id: str) -> EmployeeAnalysisResult:
        """Comprehensive AI analysis of employee"""
        
        # Fetch employee data with related records
        employee_data = await self._get_comprehensive_employee_data(employee_id)
        
        # Parallel AI analysis
        tasks = [
            self._analyze_retention_risk(employee_data),
            self._predict_performance(employee_data),
            self._analyze_skills_gaps(employee_data),
            self._generate_career_recommendations(employee_data)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return EmployeeAnalysisResult(
            employee_id=employee_id,
            retention_risk_score=results[0],
            performance_prediction=results[1],
            skills_analysis=results[2],
            career_recommendations=results[3],
            analysis_timestamp=datetime.utcnow()
        )
    
    async def _analyze_retention_risk(self, employee_data: Dict) -> float:
        """AI-powered retention risk analysis"""
        
        prompt = f"""
        Analyze employee retention risk based on the following data:
        
        Employee Profile:
        - Tenure: {employee_data.get('tenure_months', 0)} months
        - Performance Rating: {employee_data.get('performance_rating', 'N/A')}
        - Salary: ${employee_data.get('base_salary', 0):,}
        - Department: {employee_data.get('department', 'Unknown')}
        - Last Promotion: {employee_data.get('last_promotion_date', 'Never')}
        
        Recent Activity:
        - Training Completions: {len(employee_data.get('recent_training', []))}
        - Goal Achievements: {employee_data.get('goal_completion_rate', 0)}%
        - Feedback Sentiment: {employee_data.get('feedback_sentiment', 'Neutral')}
        
        Provide a retention risk score between 0 (low risk) and 1 (high risk).
        """
        
        response = await self.ai_orchestration.analyze_text_with_ai(
            prompt=prompt,
            model_config={"temperature": 0.1}
        )
        
        # Extract numeric score from AI response
        risk_score = self._extract_numeric_score(response.get('analysis', ''))
        return min(max(risk_score, 0.0), 1.0)
```

## 🔌 API Development

### FastAPI Implementation

#### Main Application Setup
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn

app = FastAPI(
    title="APG Employee Data Management API",
    description="Revolutionary employee management with AI-powered insights",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.datacraft.co.ke"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Validate JWT token and return user info"""
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")
```

#### Employee Endpoints
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class EmployeeCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    work_email: str = Field(regex=r'^[^@]+@[^@]+\.[^@]+$')
    hire_date: date
    department_id: str
    position_id: str
    base_salary: Optional[Decimal] = None

class EmployeeResponse(BaseModel):
    id: str
    employee_number: str
    first_name: str
    last_name: str
    work_email: str
    employment_status: str
    created_at: datetime
    updated_at: datetime

@app.post("/api/v1/employees", response_model=EmployeeResponse)
async def create_employee(
    employee_data: EmployeeCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create new employee with AI validation"""
    
    tenant_id = current_user.get("tenant_id")
    service = RevolutionaryEmployeeDataManagementService(tenant_id)
    
    result = await service.create_employee_revolutionary(employee_data.dict())
    
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail={"errors": result.validation_errors}
        )
    
    return EmployeeResponse(**result.employee_data)

@app.get("/api/v1/employees", response_model=List[EmployeeResponse])
async def list_employees(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    department: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List employees with filtering and pagination"""
    
    tenant_id = current_user.get("tenant_id")
    service = RevolutionaryEmployeeDataManagementService(tenant_id)
    
    search_criteria = EmployeeSearchCriteria(
        search_text=search,
        department_id=department,
        offset=(page - 1) * limit,
        limit=limit
    )
    
    result = await service.search_employees_intelligent(search_criteria)
    return [EmployeeResponse(**emp) for emp in result.employees]
```

#### AI Analysis Endpoints
```python
class AIAnalysisResponse(BaseModel):
    employee_id: str
    retention_risk_score: float
    performance_prediction: float
    skills_analysis: Dict[str, Any]
    career_recommendations: List[str]
    analysis_timestamp: datetime

@app.post("/api/v1/employees/{employee_id}/analyze", response_model=AIAnalysisResponse)
async def analyze_employee(
    employee_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Perform comprehensive AI analysis on employee"""
    
    tenant_id = current_user.get("tenant_id")
    ai_engine = EmployeeAIIntelligenceEngine(tenant_id)
    
    analysis = await ai_engine.analyze_employee_comprehensive(employee_id)
    return AIAnalysisResponse(**analysis.dict())
```

### Error Handling

#### Custom Exception Classes
```python
class EmployeeManagementException(Exception):
    """Base exception for employee management operations"""
    pass

class EmployeeNotFoundError(EmployeeManagementException):
    """Employee not found in system"""
    pass

class ValidationError(EmployeeManagementException):
    """Data validation failed"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {', '.join(errors)}")

class AIServiceError(EmployeeManagementException):
    """AI service unavailable or failed"""
    pass
```

#### Exception Handlers
```python
@app.exception_handler(EmployeeNotFoundError)
async def employee_not_found_handler(request: Request, exc: EmployeeNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "Employee not found", "detail": str(exc)}
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation failed", "errors": exc.errors}
    )

@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    return JSONResponse(
        status_code=503,
        content={"error": "AI service unavailable", "detail": str(exc)}
    )
```

## 🗄️ Database Schema

### Migration Management

#### Alembic Configuration
```python
# alembic/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from models import Base

config = context.config
target_metadata = Base.metadata

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True
        )

        with context.begin_transaction():
            context.run_migrations()
```

#### Sample Migration
```python
# alembic/versions/001_initial_schema.py
"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create employees table
    op.create_table(
        'hr_employees',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('employee_number', sa.String(20), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('work_email', sa.String(255), nullable=False),
        sa.Column('hire_date', sa.Date(), nullable=False),
        sa.Column('profile_embedding', postgresql.VECTOR(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('employee_number'),
        sa.UniqueConstraint('work_email')
    )
    
    # Create indexes
    op.create_index('ix_hr_employee_search', 'hr_employees', ['first_name', 'last_name', 'work_email'])
    op.create_index('ix_hr_employee_embedding', 'hr_employees', ['profile_embedding'], postgresql_using='ivfflat')

def downgrade():
    op.drop_table('hr_employees')
```

### Performance Optimization

#### Database Connection Pool
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

def create_database_engine(database_url: str):
    """Create optimized database engine"""
    
    return create_async_engine(
        database_url,
        # Connection pool settings
        pool_size=20,
        max_overflow=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        
        # Performance settings
        echo=False,
        future=True,
        
        # Connection arguments
        connect_args={
            "server_settings": {
                "application_name": "apg_employee_management",
                "timezone": "UTC",
            }
        }
    )

async_session_factory = sessionmaker(
    class_=AsyncSession,
    expire_on_commit=False
)
```

#### Query Optimization
```python
async def get_employees_with_performance_data(
    department_id: str,
    limit: int = 100
) -> List[Dict]:
    """Optimized query with selective loading"""
    
    query = (
        select(HREmployee)
        .options(
            selectinload(HREmployee.performance_reviews),
            selectinload(HREmployee.ai_profile)
        )
        .where(HREmployee.department_id == department_id)
        .where(HREmployee.employment_status == "Active")
        .order_by(HREmployee.last_name, HREmployee.first_name)
        .limit(limit)
    )
    
    result = await session.execute(query)
    return [employee.dict() for employee in result.scalars().all()]
```

## 🤖 AI Integration

### AI Service Abstraction

#### AI Provider Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def analyze_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        pass

class OpenAIProvider(AIProvider):
    """OpenAI implementation"""
    
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def analyze_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "model": response.model,
            "tokens_used": response.usage.total_tokens
        }
    
    async def generate_embedding(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
```

#### AI Service Factory
```python
class AIServiceFactory:
    """Factory for creating AI service instances"""
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> AIProvider:
        provider_type = config.get("provider", "openai")
        
        if provider_type == "openai":
            return OpenAIProvider(config["api_key"])
        elif provider_type == "anthropic":
            return AnthropicProvider(config["api_key"])
        elif provider_type == "ollama":
            return OllamaProvider(config["base_url"])
        else:
            raise ValueError(f"Unsupported AI provider: {provider_type}")
```

### ML Model Training

#### Retention Risk Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class RetentionRiskModel:
    """ML model for predicting employee retention risk"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_columns = [
            'tenure_months', 'performance_rating', 'salary_percentile',
            'promotion_frequency', 'training_hours', 'manager_rating'
        ]
    
    def prepare_features(self, employee_data: List[Dict]) -> np.ndarray:
        """Extract features from employee data"""
        features = []
        for emp in employee_data:
            feature_vector = [
                emp.get('tenure_months', 0),
                emp.get('performance_rating', 3.0),
                emp.get('salary_percentile', 50),
                emp.get('promotions_count', 0) / max(emp.get('tenure_months', 1), 1),
                emp.get('training_hours', 0),
                emp.get('manager_rating', 3.0)
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def train(self, training_data: List[Dict]):
        """Train the retention risk model"""
        X = self.prepare_features(training_data)
        y = [emp['left_company'] for emp in training_data]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(self.model, 'retention_risk_model.pkl')
    
    def predict_risk(self, employee_data: Dict) -> float:
        """Predict retention risk for single employee"""
        features = self.prepare_features([employee_data])
        risk_probability = self.model.predict_proba(features)[0][1]
        return float(risk_probability)
```

## 🧪 Testing Framework

### Test Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_service.py
│   ├── test_ai_engine.py
│   └── test_validators.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_ai_integration.py
├── performance/
│   ├── test_load.py
│   └── test_concurrency.py
└── e2e/
    ├── test_employee_lifecycle.py
    └── test_workflow_automation.py
```

### Unit Tests

#### Service Layer Tests
```python
import pytest
from unittest.mock import AsyncMock, patch
from service import RevolutionaryEmployeeDataManagementService

@pytest.mark.asyncio
class TestEmployeeService:
    
    @pytest.fixture
    async def employee_service(self):
        service = RevolutionaryEmployeeDataManagementService("test_tenant")
        service.ai_orchestration = AsyncMock()
        service.validator = AsyncMock()
        return service
    
    @pytest.fixture
    def sample_employee_data(self):
        return {
            "first_name": "John",
            "last_name": "Doe",
            "work_email": "john.doe@company.com",
            "hire_date": "2024-01-15",
            "department_id": "dept_001"
        }
    
    async def test_create_employee_success(self, employee_service, sample_employee_data):
        """Test successful employee creation"""
        # Mock validation success
        employee_service.validator.validate_employee_data.return_value = AsyncMock(
            is_valid=True,
            errors=[]
        )
        
        # Mock AI enhancement
        employee_service.ai_orchestration.analyze_text_with_ai.return_value = {
            "enhanced_data": sample_employee_data,
            "confidence_score": 0.95
        }
        
        result = await employee_service.create_employee_revolutionary(sample_employee_data)
        
        assert result.success is True
        assert "employee_id" in result.employee_data
        assert result.employee_data["first_name"] == "John"
    
    async def test_create_employee_validation_failure(self, employee_service):
        """Test employee creation with validation errors"""
        invalid_data = {"first_name": "", "work_email": "invalid-email"}
        
        # Mock validation failure
        employee_service.validator.validate_employee_data.return_value = AsyncMock(
            is_valid=False,
            errors=["First name is required", "Invalid email format"]
        )
        
        result = await employee_service.create_employee_revolutionary(invalid_data)
        
        assert result.success is False
        assert len(result.validation_errors) == 2
```

#### AI Engine Tests
```python
@pytest.mark.asyncio
class TestAIIntelligenceEngine:
    
    @pytest.fixture
    async def ai_engine(self):
        engine = EmployeeAIIntelligenceEngine("test_tenant")
        engine.ai_orchestration = AsyncMock()
        return engine
    
    async def test_retention_risk_analysis(self, ai_engine):
        """Test retention risk calculation"""
        employee_data = {
            "tenure_months": 24,
            "performance_rating": 4.5,
            "base_salary": 75000,
            "last_promotion_date": "2023-06-01"
        }
        
        # Mock AI response
        ai_engine.ai_orchestration.analyze_text_with_ai.return_value = {
            "analysis": "Based on the data, retention risk score: 0.25"
        }
        
        risk_score = await ai_engine._analyze_retention_risk(employee_data)
        
        assert 0.0 <= risk_score <= 1.0
        assert isinstance(risk_score, float)
    
    async def test_comprehensive_analysis(self, ai_engine):
        """Test comprehensive employee analysis"""
        employee_id = "emp_001"
        
        # Mock data retrieval
        with patch.object(ai_engine, '_get_comprehensive_employee_data') as mock_data:
            mock_data.return_value = {
                "employee_id": employee_id,
                "tenure_months": 12,
                "performance_rating": 4.0
            }
            
            analysis = await ai_engine.analyze_employee_comprehensive(employee_id)
            
            assert analysis.employee_id == employee_id
            assert hasattr(analysis, 'retention_risk_score')
            assert hasattr(analysis, 'performance_prediction')
```

### Integration Tests

#### API Integration Tests
```python
import httpx
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.mark.asyncio
class TestAPIIntegration:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        # Create test JWT token
        token = create_test_jwt_token({"tenant_id": "test_tenant", "user_id": "test_user"})
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_employee_endpoint(self, client, auth_headers):
        """Test employee creation via API"""
        employee_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "work_email": "jane.smith@company.com",
            "hire_date": "2024-01-20",
            "department_id": "dept_002",
            "position_id": "pos_002"
        }
        
        response = client.post(
            "/api/v1/employees",
            json=employee_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["first_name"] == "Jane"
        assert "id" in data
    
    def test_list_employees_endpoint(self, client, auth_headers):
        """Test employee listing with pagination"""
        response = client.get(
            "/api/v1/employees?page=1&limit=10",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10
    
    def test_ai_analysis_endpoint(self, client, auth_headers):
        """Test AI analysis endpoint"""
        # First create an employee
        employee_data = {
            "first_name": "AI",
            "last_name": "Test",
            "work_email": "ai.test@company.com",
            "hire_date": "2024-01-15",
            "department_id": "dept_001"
        }
        
        create_response = client.post(
            "/api/v1/employees",
            json=employee_data,
            headers=auth_headers
        )
        employee_id = create_response.json()["id"]
        
        # Test AI analysis
        analysis_response = client.post(
            f"/api/v1/employees/{employee_id}/analyze",
            headers=auth_headers
        )
        
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        assert "retention_risk_score" in analysis_data
        assert "performance_prediction" in analysis_data
```

### Performance Tests

#### Load Testing
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import aiohttp

class LoadTestSuite:
    """Performance and load testing"""
    
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {auth_token}"}
    
    async def test_concurrent_employee_creation(self, concurrent_users: int = 50):
        """Test concurrent employee creation"""
        
        async def create_employee(session: aiohttp.ClientSession, user_id: int):
            employee_data = {
                "first_name": f"User{user_id}",
                "last_name": "LoadTest",
                "work_email": f"user{user_id}@loadtest.com",
                "hire_date": "2024-01-15",
                "department_id": "dept_loadtest"
            }
            
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/api/v1/employees",
                json=employee_data,
                headers=self.headers
            ) as response:
                response_time = time.time() - start_time
                return {
                    "status_code": response.status,
                    "response_time": response_time,
                    "user_id": user_id
                }
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                create_employee(session, i) 
                for i in range(concurrent_users)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Analyze results
            successful_requests = sum(1 for r in results if r["status_code"] == 201)
            avg_response_time = sum(r["response_time"] for r in results) / len(results)
            max_response_time = max(r["response_time"] for r in results)
            
            print(f"Successful requests: {successful_requests}/{concurrent_users}")
            print(f"Average response time: {avg_response_time:.3f}s")
            print(f"Max response time: {max_response_time:.3f}s")
            
            assert successful_requests >= concurrent_users * 0.95  # 95% success rate
            assert avg_response_time < 1.0  # Average under 1 second
```

## 🚀 Deployment Pipeline

### Docker Configuration

#### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "4"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/apg_hr
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: pgvector/pgvector:pg14
    environment:
      - POSTGRES_DB=apg_hr
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: employee-api
  namespace: apg-platform
  labels:
    app: employee-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: employee-api
  template:
    metadata:
      labels:
        app: employee-api
        version: v1.0.0
    spec:
      containers:
      - name: employee-api
        image: apg/employee-api:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-credentials
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: employee-api-service
  namespace: apg-platform
spec:
  selector:
    app: employee-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
name: Deploy Employee API

on:
  push:
    branches: [main]
    paths:
      - 'capabilities/employee_data_management/**'
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:pg14
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql+asyncpg://postgres:test@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      env:
        REGISTRY: ghcr.io
        IMAGE_NAME: apg/employee-api
      run: |
        echo ${{ secrets.GITHUB_TOKEN }} | docker login $REGISTRY -u ${{ github.actor }} --password-stdin
        docker build -t $REGISTRY/$IMAGE_NAME:${{ github.sha }} .
        docker push $REGISTRY/$IMAGE_NAME:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          ghcr.io/apg/employee-api:${{ github.sha }}
        kubectl-version: 'v1.24.0'
```

## ⚡ Performance Optimization

### Database Optimization

#### Index Strategy
```sql
-- Core search indexes
CREATE INDEX CONCURRENTLY idx_hr_employees_search 
ON hr_employees USING gin(to_tsvector('english', first_name || ' ' || last_name || ' ' || work_email));

-- Department and status filtering
CREATE INDEX CONCURRENTLY idx_hr_employees_dept_status 
ON hr_employees (department_id, employment_status) 
WHERE employment_status = 'Active';

-- Vector similarity search
CREATE INDEX CONCURRENTLY idx_hr_employees_embedding 
ON hr_employees USING ivfflat (profile_embedding vector_cosine_ops)
WITH (lists = 100);

-- Performance analytics
CREATE INDEX CONCURRENTLY idx_hr_performance_reviews_employee_date
ON hr_performance_reviews (employee_id, review_date DESC);
```

#### Query Optimization
```python
async def get_employees_optimized(
    department_id: Optional[str] = None,
    search_term: Optional[str] = None,
    limit: int = 50
) -> List[HREmployee]:
    """Optimized employee query with strategic loading"""
    
    query = select(HREmployee)
    
    # Use index-friendly filters
    if department_id:
        query = query.where(HREmployee.department_id == department_id)
    
    # Full-text search with index
    if search_term:
        query = query.where(
            func.to_tsvector('english', 
                HREmployee.first_name + ' ' + 
                HREmployee.last_name + ' ' + 
                HREmployee.work_email
            ).match(search_term)
        )
    
    # Optimize loading with selectinload for related data
    query = query.options(
        selectinload(HREmployee.department),
        selectinload(HREmployee.position),
        selectinload(HREmployee.ai_profile)
    ).limit(limit)
    
    result = await session.execute(query)
    return result.scalars().all()
```

### Caching Strategy

#### Redis Caching Implementation
```python
import redis.asyncio as redis
import json
from typing import Any, Optional
import pickle

class DistributedCache:
    """Redis-based distributed caching"""
    
    def __init__(self, redis_url: str, prefix: str = "apg_hr"):
        self.redis = redis.from_url(redis_url)
        self.prefix = prefix
        self.default_ttl = 3600  # 1 hour
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(self._make_key(key))
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            serialized = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return await self.redis.setex(self._make_key(key), ttl, serialized)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return await self.redis.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def get_or_set(self, key: str, factory, ttl: Optional[int] = None) -> Any:
        """Get from cache or set using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        await self.set(key, value, ttl)
        return value

# Cache decorator
def cached(key_pattern: str, ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from pattern
            cache_key = key_pattern.format(*args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Usage example
@cached("employee_analysis:{employee_id}", ttl=1800)
async def get_employee_analysis(employee_id: str) -> Dict[str, Any]:
    """Cached employee analysis"""
    return await ai_engine.analyze_employee_comprehensive(employee_id)
```

### Asynchronous Processing

#### Background Task Queue
```python
import asyncio
from celery import Celery
from kombu import Queue

# Celery configuration
celery_app = Celery(
    'employee_tasks',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'employee_tasks.ai_analysis': {'queue': 'ai_queue'},
        'employee_tasks.data_sync': {'queue': 'sync_queue'},
        'employee_tasks.notifications': {'queue': 'notification_queue'}
    }
)

@celery_app.task(bind=True, max_retries=3)
def analyze_employee_background(self, employee_id: str, tenant_id: str):
    """Background AI analysis task"""
    try:
        # Initialize services
        ai_engine = EmployeeAIIntelligenceEngine(tenant_id)
        
        # Perform analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        analysis = loop.run_until_complete(
            ai_engine.analyze_employee_comprehensive(employee_id)
        )
        
        # Store results
        cache_key = f"analysis:{employee_id}"
        loop.run_until_complete(
            cache.set(cache_key, analysis.dict(), ttl=3600)
        )
        
        return analysis.dict()
        
    except Exception as exc:
        logger.error(f"Analysis failed for {employee_id}: {exc}")
        raise self.retry(countdown=60, exc=exc)

@celery_app.task
def sync_employee_data(employee_id: str, external_system: str):
    """Sync employee data with external systems"""
    # Implementation for data synchronization
    pass

@celery_app.task
def send_notification(user_id: str, message: str, notification_type: str):
    """Send notification to user"""
    # Implementation for notifications
    pass
```

## 🔒 Security Implementation

### Authentication & Authorization

#### JWT Token Management
```python
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional

class JWTManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_data["user_id"],
            "tenant_id": user_data["tenant_id"],
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", []),
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "iss": "apg_employee_management"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Role-based access control
class RBACManager:
    """Role-based access control"""
    
    PERMISSIONS = {
        "employee.create": "Create new employees",
        "employee.read": "View employee data",
        "employee.update": "Update employee data",
        "employee.delete": "Delete employees",
        "employee.ai_analysis": "Access AI analysis",
        "analytics.view": "View analytics dashboard",
        "admin.settings": "Modify system settings"
    }
    
    ROLES = {
        "hr_admin": [
            "employee.create", "employee.read", "employee.update", 
            "employee.delete", "employee.ai_analysis", "analytics.view"
        ],
        "hr_manager": [
            "employee.read", "employee.update", "employee.ai_analysis", "analytics.view"
        ],
        "employee": ["employee.read"],
        "system_admin": list(PERMISSIONS.keys())
    }
    
    @classmethod
    def check_permission(cls, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = set()
        for role in user_roles:
            user_permissions.update(cls.ROLES.get(role, []))
        
        return required_permission in user_permissions

# Permission decorator
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from context (set by auth middleware)
            current_user = kwargs.get('current_user') or g.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_roles = current_user.get("roles", [])
            if not RBACManager.check_permission(user_roles, permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### Data Encryption

#### Field-Level Encryption
```python
from cryptography.fernet import Fernet
import base64
import os

class FieldEncryption:
    """Field-level encryption for sensitive data"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            self.key = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key())
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not data:
            return data
        
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception:
            # Return original if decryption fails (backward compatibility)
            return encrypted_data

# Custom SQLAlchemy type for encrypted fields
from sqlalchemy import TypeDecorator, String

class EncryptedString(TypeDecorator):
    """SQLAlchemy type for encrypted string fields"""
    
    impl = String
    cache_ok = True
    
    def __init__(self, *args, **kwargs):
        self.encryptor = FieldEncryption()
        super().__init__(*args, **kwargs)
    
    def process_bind_param(self, value, dialect):
        """Encrypt value before storing in database"""
        if value is not None:
            return self.encryptor.encrypt(value)
        return value
    
    def process_result_value(self, value, dialect):
        """Decrypt value when retrieving from database"""
        if value is not None:
            return self.encryptor.decrypt(value)
        return value

# Usage in models
class HREmployee(Model):
    # Regular fields
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    
    # Encrypted sensitive fields
    ssn = Column(EncryptedString(20), nullable=True)
    bank_account = Column(EncryptedString(50), nullable=True)
    emergency_contact = Column(EncryptedString(200), nullable=True)
```

### Audit Logging

#### Comprehensive Audit Trail
```python
from sqlalchemy import Column, String, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
import asyncio
from datetime import datetime

class HRAuditLog(Model):
    """Audit log for all employee management operations"""
    
    __tablename__ = 'hr_audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(50), nullable=True, index=True)
    
    # Change tracking
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    
    # Request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(50), nullable=True)
    
    # Metadata
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    session_id = Column(String(100), nullable=True)
    
    # Compliance tracking
    compliance_flags = Column(JSON, nullable=True)
    data_classification = Column(String(20), nullable=True)

class AuditLogger:
    """Service for logging audit events"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def log_action(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Log an audit event"""
        
        audit_entry = HRAuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=metadata.get("ip_address") if metadata else None,
            user_agent=metadata.get("user_agent") if metadata else None,
            request_id=metadata.get("request_id") if metadata else None
        )
        
        self.session.add(audit_entry)
        await self.session.commit()
    
    async def get_audit_trail(
        self,
        tenant_id: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HRAuditLog]:
        """Retrieve audit trail with filters"""
        
        query = select(HRAuditLog).where(HRAuditLog.tenant_id == tenant_id)
        
        if resource_id:
            query = query.where(HRAuditLog.resource_id == resource_id)
        if user_id:
            query = query.where(HRAuditLog.user_id == user_id)
        if start_date:
            query = query.where(HRAuditLog.timestamp >= start_date)
        if end_date:
            query = query.where(HRAuditLog.timestamp <= end_date)
        
        query = query.order_by(HRAuditLog.timestamp.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()

# Audit decorator
def audit_action(action: str, resource_type: str):
    """Decorator to automatically audit function calls"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Extract context
            tenant_id = getattr(self, 'tenant_id', 'unknown')
            user_id = kwargs.get('user_id') or getattr(g, 'user_id', 'system')
            
            # Get old values if this is an update
            old_values = None
            if action.startswith('update') and args:
                resource_id = args[0]
                old_values = await self._get_current_values(resource_id)
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Extract new values and resource ID from result
            new_values = result if isinstance(result, dict) else None
            resource_id = new_values.get('id') if new_values else args[0] if args else None
            
            # Log audit event
            audit_logger = AuditLogger(self.session)
            await audit_logger.log_action(
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=old_values,
                new_values=new_values
            )
            
            return result
        return wrapper
    return decorator
```

---

© 2025 Datacraft. All rights reserved.  
Developer Guide Version 1.0 | Last Updated: January 2025