# APG GraphRAG Installation Guide

Complete installation guide for APG GraphRAG capability including prerequisites, installation steps, configuration, and verification.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Docker Installation](#docker-installation)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB available space
- **Network**: Stable internet connection for model downloads

### Recommended Requirements

- **OS**: Ubuntu 22.04 LTS or macOS 13+
- **CPU**: 8 cores, 3.0GHz
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for acceleration)

### Software Dependencies

- **Python**: 3.9+ (3.11 recommended)
- **PostgreSQL**: 12+ with Apache AGE extension
- **Ollama**: Latest version with required models
- **Redis**: 6+ (optional, for caching)
- **Docker**: 20+ (optional, for containerized deployment)

## 🔧 Prerequisites

### 1. Python Environment

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Create virtual environment (recommended)
python3 -m venv apg-graphrag-env
source apg-graphrag-env/bin/activate  # Linux/macOS
# OR
apg-graphrag-env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. PostgreSQL with Apache AGE

#### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all

# Install Apache AGE
sudo apt install -y git build-essential
git clone https://github.com/apache/age.git
cd age
git checkout release/PG13/1.3.0  # Use appropriate version
make install

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### macOS
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql

# Install Apache AGE
git clone https://github.com/apache/age.git
cd age
make install PG_CONFIG=/opt/homebrew/bin/pg_config
```

#### Docker (Alternative)
```bash
# Use pre-built Docker image with PostgreSQL + Apache AGE
docker run -d \
  --name graphrag-postgres \
  -e POSTGRES_DB=graphrag_db \
  -e POSTGRES_USER=graphrag_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  apache/age:latest
```

### 3. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models (in separate terminal)
ollama pull bge-m3        # Embedding model (8k context)
ollama pull qwen3         # Generation model
ollama pull deepseek-r1   # Advanced reasoning model
```

Verify models are installed:
```bash
ollama list
# Should show:
# NAME           ID         SIZE      MODIFIED
# bge-m3:latest  abc123...  2.3GB     2 hours ago
# qwen3:latest   def456...  7.1GB     2 hours ago  
# deepseek-r1    ghi789...  4.8GB     2 hours ago
```

### 4. Database Setup

```bash
# Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database and user
CREATE DATABASE graphrag_db;
CREATE USER graphrag_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE graphrag_db TO graphrag_user;

-- Create Apache AGE extension
\c graphrag_db
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Exit PostgreSQL
\q
```

## 📦 Installation Steps

### 1. Clone APG Repository

```bash
# Clone the main APG repository
git clone https://github.com/datacraft/apg.git
cd apg

# Navigate to GraphRAG capability
cd capabilities/common/graphrag
```

### 2. Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install the capability in development mode
pip install -e .
```

### 3. Install Database Schema

```bash
# Initialize the database schema
psql -h localhost -U graphrag_user -d graphrag_db -f database_schema.sql

# Verify schema installation
psql -h localhost -U graphrag_user -d graphrag_db -c "\dt"
```

Expected output:
```
                   List of relations
 Schema |          Name           | Type  |     Owner
--------+-------------------------+-------+---------------
 public | gr_knowledge_graphs     | table | graphrag_user
 public | gr_graph_entities       | table | graphrag_user
 public | gr_graph_relationships  | table | graphrag_user
 public | gr_graph_communities    | table | graphrag_user
 public | gr_document_sources     | table | graphrag_user
 public | gr_query_history        | table | graphrag_user
 public | gr_curation_suggestions | table | graphrag_user
 public | gr_update_operations    | table | graphrag_user
 public | gr_performance_metrics  | table | graphrag_user
 public | gr_system_config        | table | graphrag_user
```

### 4. Install APG Framework Dependencies

```bash
# Navigate back to APG root
cd ../../..

# Install APG framework if not already installed
pip install -e .

# Install Flask-AppBuilder dependencies
pip install flask-appbuilder
pip install flask-restx
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the GraphRAG capability directory:

```bash
# capabilities/common/graphrag/.env

# Database Configuration
DATABASE_URL=postgresql://graphrag_user:secure_password@localhost:5432/graphrag_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=graphrag_db
DB_USER=graphrag_user
DB_PASSWORD=secure_password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_GENERATION_MODELS=qwen3,deepseek-r1,llama3.2
OLLAMA_MAX_CONTEXT_LENGTH=8000

# APG Configuration
APG_TENANT_ID=default_tenant
APG_DEBUG=true
APG_LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key

# Optional: Redis Configuration (for caching)
REDIS_URL=redis://localhost:6379/0

# Optional: Monitoring
ENABLE_METRICS=true
METRICS_PORT=8090
```

### 2. Configuration File

Create `config.json` in the capability directory:

```json
{
  "database": {
    "pool_size": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo_sql": false
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "embedding_model": "bge-m3",
    "generation_models": ["qwen3", "deepseek-r1"],
    "max_concurrent_requests": 10,
    "request_timeout_seconds": 60,
    "retry_attempts": 3,
    "embedding_dimensions": 1024,
    "max_context_length": 8000,
    "generation_temperature": 0.7,
    "generation_max_tokens": 4000,
    "enable_embedding_cache": true,
    "cache_ttl_hours": 24,
    "max_cache_size": 10000
  },
  "graph_processing": {
    "similarity_threshold": 0.85,
    "confidence_threshold": 0.7,
    "max_merge_candidates": 5,
    "enable_auto_merge": true,
    "max_entities_per_query": 100,
    "max_relationships_per_query": 200,
    "max_reasoning_hops": 5
  },
  "visualization": {
    "default_width": 1200,
    "default_height": 800,
    "max_nodes": 1000,
    "max_edges": 2000,
    "enable_3d": false,
    "physics_enabled": true,
    "animation_duration": 500.0
  },
  "api": {
    "enable_cors": true,
    "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
    "rate_limit_per_minute": 100,
    "max_request_size_mb": 50,
    "enable_api_docs": true
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/graphrag.log",
    "max_file_size_mb": 100,
    "backup_count": 5
  }
}
```

### 3. Logging Configuration

Create logging directory and configuration:

```bash
# Create logs directory
mkdir -p logs

# Set appropriate permissions
chmod 755 logs
```

## ✅ Verification

### 1. Basic Installation Check

```python
# test_installation.py
import asyncio
import sys

async def test_installation():
    """Test basic GraphRAG installation"""
    
    print("🧪 Testing APG GraphRAG Installation...")
    
    try:
        # Test imports
        print("1️⃣ Testing imports...")
        from capabilities.common.graphrag.service import GraphRAGService
        from capabilities.common.graphrag.database import GraphRAGDatabaseService
        from capabilities.common.graphrag.ollama_integration import OllamaClient, OllamaConfig
        from capabilities.common.graphrag.views import KnowledgeGraphRequest
        print("✅ All imports successful")
        
        # Test database connection
        print("2️⃣ Testing database connection...")
        db_service = GraphRAGDatabaseService()
        await db_service.initialize()
        print("✅ Database connection successful")
        await db_service.cleanup()
        
        # Test Ollama connection
        print("3️⃣ Testing Ollama connection...")
        ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            embedding_model="bge-m3",
            generation_models=["qwen3", "deepseek-r1"]
        )
        ollama_client = OllamaClient(ollama_config)
        await ollama_client.initialize()
        
        # Test embedding generation
        test_embedding = await ollama_client.generate_embedding("test text")
        print(f"✅ Ollama connection successful (embedding dims: {len(test_embedding.embeddings)})")
        await ollama_client.cleanup()
        
        print("\n🎉 Installation verification completed successfully!")
        print("APG GraphRAG is ready to use!")
        
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        print("\nPlease check:")
        print("- PostgreSQL is running and accessible")
        print("- Apache AGE extension is installed")
        print("- Ollama is running with required models")
        print("- All dependencies are installed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_installation())
```

Run the verification:
```bash
python test_installation.py
```

### 2. Create Test Knowledge Graph

```python
# test_graphrag.py
import asyncio
from capabilities.common.graphrag.service import GraphRAGService
from capabilities.common.graphrag.views import (
    KnowledgeGraphRequest, DocumentProcessingRequest, 
    GraphRAGQuery, QueryContext
)

async def test_graphrag_workflow():
    """Test complete GraphRAG workflow"""
    
    print("🧪 Testing Complete GraphRAG Workflow...")
    
    # Initialize service
    service = await GraphRAGService.create()
    
    try:
        # Create knowledge graph
        print("1️⃣ Creating knowledge graph...")
        graph_request = KnowledgeGraphRequest(
            tenant_id="test_tenant",
            name="Installation Test Graph",
            description="Test graph for installation verification",
            domain="testing"
        )
        graph = await service.create_knowledge_graph(graph_request)
        print(f"✅ Created graph: {graph.knowledge_graph_id}")
        
        # Process test document
        print("2️⃣ Processing test document...")
        doc_request = DocumentProcessingRequest(
            tenant_id="test_tenant",
            knowledge_graph_id=graph.knowledge_graph_id,
            title="Test Document",
            content="John Doe is the CEO of Acme Corporation located in San Francisco. The company was founded in 2020 and specializes in AI technology.",
            source_type="text"
        )
        doc_result = await service.process_document(doc_request)
        print(f"✅ Document processed: {doc_result.entities_extracted} entities, {doc_result.relationships_extracted} relationships")
        
        # Test query
        print("3️⃣ Testing GraphRAG query...")
        query = GraphRAGQuery(
            tenant_id="test_tenant",
            knowledge_graph_id=graph.knowledge_graph_id,
            query_text="Who is the CEO of Acme Corporation?",
            query_type="factual",
            context=QueryContext(
                user_id="test_user",
                session_id="test_session",
                conversation_history=[],
                domain_context={"domain": "testing"}
            )
        )
        response = await service.process_query(query)
        print(f"✅ Query processed successfully")
        print(f"   Answer: {response.answer}")
        print(f"   Confidence: {response.confidence_score:.2f}")
        
        print("\n🎉 Complete workflow test successful!")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        raise
    finally:
        await service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_graphrag_workflow())
```

Run the workflow test:
```bash
python test_graphrag.py
```

### 3. Web Interface Test

```bash
# Start the Flask-AppBuilder development server
cd capabilities/common/graphrag
python -c "
from flask import Flask
from flask_appbuilder import AppBuilder, SQLA
from blueprint import create_graphrag_blueprint

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://graphrag_user:secure_password@localhost:5432/graphrag_db'

db = SQLA(app)
appbuilder = AppBuilder(app, db.session)

create_graphrag_blueprint(appbuilder)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
"
```

Visit `http://localhost:5000` to test the web interface.

### 4. API Test

```bash
# Test REST API endpoints
curl -X GET "http://localhost:5000/api/v1/graphrag/admin/health" \
  -H "Content-Type: application/json"

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-21T15:40:00Z",
#   "services": {
#     "database": "healthy",
#     "ollama": "healthy",
#     "graph_engine": "healthy"
#   },
#   "version": "1.0.0"
# }
```

## 🐳 Docker Installation

### 1. Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: graphrag_db
      POSTGRES_USER: graphrag_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U graphrag_user -d graphrag_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    command: ollama serve

  graphrag:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://graphrag_user:secure_password@postgres:5432/graphrag_db
      - OLLAMA_BASE_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      ollama:
        condition: service_started
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config

volumes:
  postgres_data:
  redis_data:
  ollama_data:
```

### 2. Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Install the capability
RUN pip install -e .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/api/v1/graphrag/admin/health || exit 1

# Run the application
CMD ["python", "run.py"]
```

### 3. Production Requirements

Create `requirements-prod.txt`:

```txt
# Core dependencies (from requirements.txt)
asyncio>=3.4.3
aiohttp>=3.8.0
asyncpg>=0.28.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
alembic>=1.12.0
pydantic>=2.5.0
uuid-extensions>=0.1.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Web framework
flask>=3.0.0
flask-appbuilder>=4.3.0
flask-restx>=1.3.0
flask-cors>=4.0.0

# Production server
gunicorn>=21.2.0
uvicorn[standard]>=0.24.0

# Monitoring and logging
prometheus-client>=0.17.0
structlog>=23.2.0
sentry-sdk>=1.38.0

# Performance
redis>=5.0.0
celery>=5.3.0

# Security
cryptography>=41.0.0
bcrypt>=4.0.0
```

### 4. Run with Docker

```bash
# Build and start services
docker-compose up -d

# Initialize Ollama models
docker-compose exec ollama ollama pull bge-m3
docker-compose exec ollama ollama pull qwen3
docker-compose exec ollama ollama pull deepseek-r1

# Check service health
docker-compose ps
curl http://localhost:5000/api/v1/graphrag/admin/health
```

## 🚀 Production Deployment

### 1. Environment Setup

```bash
# Production environment variables
export APG_ENV=production
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export DATABASE_URL=postgresql://user:pass@prod-db:5432/graphrag_prod
export OLLAMA_BASE_URL=http://ollama-cluster:11434
export REDIS_URL=redis://redis-cluster:6379/0
```

### 2. Production Configuration

Create `config-prod.json`:

```json
{
  "database": {
    "pool_size": 50,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo_sql": false
  },
  "ollama": {
    "max_concurrent_requests": 50,
    "request_timeout_seconds": 120,
    "retry_attempts": 5,
    "enable_embedding_cache": true,
    "cache_ttl_hours": 168,
    "max_cache_size": 100000
  },
  "api": {
    "enable_cors": false,
    "rate_limit_per_minute": 1000,
    "max_request_size_mb": 100,
    "enable_api_docs": false
  },
  "logging": {
    "level": "WARNING",
    "file": "/var/log/graphrag/graphrag.log",
    "max_file_size_mb": 500,
    "backup_count": 10
  },
  "monitoring": {
    "enable_metrics": true,
    "metrics_port": 8090,
    "health_check_interval": 30,
    "enable_tracing": true
  }
}
```

### 3. Systemd Service

Create `/etc/systemd/system/apg-graphrag.service`:

```ini
[Unit]
Description=APG GraphRAG Service
After=network.target postgresql.service

[Service]
Type=forking
User=graphrag
Group=graphrag
WorkingDirectory=/opt/apg/capabilities/common/graphrag
Environment=APG_ENV=production
Environment=CONFIG_FILE=/opt/apg/config/graphrag-prod.json
ExecStart=/opt/apg/venv/bin/gunicorn --daemon --bind 0.0.0.0:5000 --workers 4 --worker-class uvicorn.workers.UvicornWorker app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable apg-graphrag
sudo systemctl start apg-graphrag
sudo systemctl status apg-graphrag
```

### 4. Nginx Configuration

Create `/etc/nginx/sites-available/apg-graphrag`:

```nginx
upstream graphrag_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;  # Additional workers
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    client_max_body_size 100M;

    location / {
        proxy_pass http://graphrag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /api/v1/graphrag/admin/health {
        proxy_pass http://graphrag_backend;
        access_log off;
    }

    location /metrics {
        proxy_pass http://127.0.0.1:8090;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }
}
```

Enable the configuration:
```bash
sudo ln -s /etc/nginx/sites-available/apg-graphrag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 🔧 Troubleshooting

### Common Issues

#### 1. PostgreSQL Connection Error

**Error:**
```
psycopg2.OperationalError: could not connect to server
```

**Solutions:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U graphrag_user -d graphrag_db -c "SELECT 1;"

# Check pg_hba.conf
sudo nano /etc/postgresql/14/main/pg_hba.conf
# Add line: host all all 127.0.0.1/32 md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### 2. Apache AGE Extension Not Found

**Error:**
```
ERROR: extension "age" is not available
```

**Solutions:**
```bash
# Reinstall Apache AGE
cd age
make clean
make install

# Check installation
psql -d graphrag_db -c "SELECT * FROM pg_available_extensions WHERE name = 'age';"

# Load extension
psql -d graphrag_db -c "CREATE EXTENSION IF NOT EXISTS age;"
```

#### 3. Ollama Models Not Found

**Error:**
```
ModelNotAvailableError: Model 'bge-m3' not found
```

**Solutions:**
```bash
# Check Ollama status
ollama list

# Pull missing models
ollama pull bge-m3
ollama pull qwen3
ollama pull deepseek-r1

# Restart Ollama
pkill ollama
ollama serve &
```

#### 4. Permission Denied Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'logs/graphrag.log'
```

**Solutions:**
```bash
# Create logs directory with proper permissions
sudo mkdir -p /var/log/graphrag
sudo chown graphrag:graphrag /var/log/graphrag
sudo chmod 755 /var/log/graphrag

# Update log file permissions
sudo touch /var/log/graphrag/graphrag.log
sudo chown graphrag:graphrag /var/log/graphrag/graphrag.log
sudo chmod 644 /var/log/graphrag/graphrag.log
```

#### 5. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'capabilities.common.graphrag'
```

**Solutions:**
```bash
# Ensure APG is in Python path
export PYTHONPATH=/path/to/apg:$PYTHONPATH

# Reinstall in development mode
cd /path/to/apg/capabilities/common/graphrag
pip install -e .

# Check installation
python -c "import capabilities.common.graphrag; print('OK')"
```

### Health Check Script

Create `health_check.py`:

```python
#!/usr/bin/env python3
"""
APG GraphRAG Health Check Script
"""
import asyncio
import sys
import time
from datetime import datetime

async def comprehensive_health_check():
    """Comprehensive health check for APG GraphRAG"""
    
    print("🏥 APG GraphRAG Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()
    
    checks = []
    
    # Database check
    print("1️⃣ Database Connection...")
    try:
        from capabilities.common.graphrag.database import GraphRAGDatabaseService
        db = GraphRAGDatabaseService()
        await db.initialize()
        
        # Test query
        graphs = await db.list_knowledge_graphs("health_check", limit=1)
        await db.cleanup()
        
        checks.append(("Database", "✅ Healthy"))
        print("   ✅ PostgreSQL connection successful")
        
    except Exception as e:
        checks.append(("Database", f"❌ {str(e)[:50]}..."))
        print(f"   ❌ Database error: {e}")
    
    # Ollama check
    print("2️⃣ Ollama Service...")
    try:
        from capabilities.common.graphrag.ollama_integration import OllamaClient, OllamaConfig
        
        config = OllamaConfig()
        client = OllamaClient(config)
        await client.initialize()
        
        # Test embedding
        result = await client.generate_embedding("health check")
        await client.cleanup()
        
        checks.append(("Ollama", "✅ Healthy"))
        print(f"   ✅ Ollama connection successful (dims: {len(result.embeddings)})")
        
    except Exception as e:
        checks.append(("Ollama", f"❌ {str(e)[:50]}..."))
        print(f"   ❌ Ollama error: {e}")
    
    # API check
    print("3️⃣ API Health...")
    try:
        import requests
        response = requests.get("http://localhost:5000/api/v1/graphrag/admin/health", timeout=10)
        
        if response.status_code == 200:
            checks.append(("API", "✅ Healthy"))
            print("   ✅ API responding correctly")
        else:
            checks.append(("API", f"❌ HTTP {response.status_code}"))
            print(f"   ❌ API returned status {response.status_code}")
            
    except Exception as e:
        checks.append(("API", f"❌ {str(e)[:50]}..."))
        print(f"   ❌ API error: {e}")
    
    # System resources check
    print("4️⃣ System Resources...")
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory Usage: {memory.percent:.1f}% ({memory.used // 1024**3}GB/{memory.total // 1024**3}GB)")
        print(f"   Disk Usage: {disk.percent:.1f}% ({disk.used // 1024**3}GB/{disk.total // 1024**3}GB)")
        
        if cpu_percent < 80 and memory.percent < 80 and disk.percent < 80:
            checks.append(("Resources", "✅ Healthy"))
        else:
            checks.append(("Resources", "⚠️ High Usage"))
            
    except ImportError:
        checks.append(("Resources", "❓ No psutil"))
        print("   ❓ psutil not available for resource monitoring")
    except Exception as e:
        checks.append(("Resources", f"❌ {str(e)[:50]}..."))
        print(f"   ❌ Resource check error: {e}")
    
    # Summary
    print("\n📊 Health Check Summary")
    print("-" * 30)
    
    healthy_count = 0
    for component, status in checks:
        print(f"{component:12} {status}")
        if "✅" in status:
            healthy_count += 1
    
    print(f"\nOverall: {healthy_count}/{len(checks)} components healthy")
    
    if healthy_count == len(checks):
        print("🎉 All systems operational!")
        return 0
    else:
        print("⚠️ Some issues detected. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(comprehensive_health_check())
    sys.exit(exit_code)
```

Run the health check:
```bash
python health_check.py
```

### Log Analysis

```bash
# View recent logs
tail -f logs/graphrag.log

# Search for errors
grep -i error logs/graphrag.log

# Check specific component logs
grep -i "ollama" logs/graphrag.log
grep -i "database" logs/graphrag.log

# Monitor API requests
grep -i "api" logs/graphrag.log | tail -20
```

---

## 🎯 Next Steps

After successful installation:

1. **[Quick Start Guide](./quickstart.md)** - Get up and running in 5 minutes
2. **[User Guide](./user_guide.md)** - Learn all features and capabilities
3. **[Configuration Guide](./configuration.md)** - Advanced configuration options
4. **[Performance Tuning](./performance.md)** - Optimize for your use case

For support, contact nyimbi@gmail.com or visit www.datacraft.co.ke.