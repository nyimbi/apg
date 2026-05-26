# APG Quick Start Guide

Get up and running with APG in 15 minutes. This guide will walk you through setting up APG and creating your first application.

## 🚀 Quick Installation

### Prerequisites Check
```bash
# Check Python version (3.9+ required)
python --version

# Check if PostgreSQL is installed
psql --version

# Check if Redis is installed  
redis-server --version
```

If any are missing, see the [Installation Guide](./installation.md) for detailed setup instructions.

### 1-Minute Setup

```bash
# Clone and setup
git clone <repository-url>
cd apg
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick database setup
createdb apg_quickstart
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/apg_quickstart"
export REDIS_URL="redis://localhost:6379/0"

# Initialize database
python -c "from capabilities.composition.database import init_db; init_db()"

# Start APG
python cli.py run --debug
```

🎉 **APG is now running at** `http://localhost:5000`

## 🏗️ Your First APG Application

### Step 1: Create a Simple Workflow

Let's create a data processing workflow using the APG CLI:

```bash
# Create a new workflow
python cli.py workflow create \
  --name "data-processor" \
  --description "Process CSV data files" \
  --engine "native"
```

### Step 2: Define Workflow Tasks

Create a workflow definition file `my_first_workflow.json`:

```json
{
  "name": "Data Processing Pipeline",
  "description": "Extract, transform, and load CSV data",
  "engine": "native",
  "tasks": [
    {
      "id": "extract_data",
      "type": "python",
      "config": {
        "function": "extract_csv_data",
        "parameters": {
          "file_path": "/tmp/sample_data.csv"
        }
      },
      "dependencies": []
    },
    {
      "id": "transform_data",
      "type": "python", 
      "config": {
        "function": "clean_and_transform",
        "parameters": {
          "remove_nulls": true,
          "normalize_names": true
        }
      },
      "dependencies": ["extract_data"]
    },
    {
      "id": "generate_report",
      "type": "document",
      "config": {
        "template": "data_report",
        "output_format": "pdf"
      },
      "dependencies": ["transform_data"]
    }
  ]
}
```

### Step 3: Create Sample Data

Create a sample CSV file for testing:

```bash
# Create sample data
cat > /tmp/sample_data.csv << EOF
name,age,city,salary
John Doe,30,New York,75000
Jane Smith,25,San Francisco,82000
Bob Johnson,35,Chicago,68000
Alice Brown,28,Boston,71000
EOF
```

### Step 4: Register Custom Functions

Create `workflow_functions.py`:

```python
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def extract_csv_data(file_path: str) -> dict:
    """Extract data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Extracted {len(df)} records from {file_path}")
        
        return {
            "status": "success",
            "data": df.to_dict('records'),
            "row_count": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        return {"status": "error", "message": str(e)}

def clean_and_transform(data: dict, remove_nulls: bool = True, normalize_names: bool = True) -> dict:
    """Clean and transform the data"""
    try:
        df = pd.DataFrame(data['data'])
        
        if remove_nulls:
            df = df.dropna()
            
        if normalize_names:
            df['name'] = df['name'].str.title()
            df['city'] = df['city'].str.title()
        
        # Add calculated fields
        df['salary_category'] = df['salary'].apply(
            lambda x: 'High' if x > 75000 else 'Medium' if x > 65000 else 'Low'
        )
        
        logger.info(f"Transformed data: {len(df)} records")
        
        return {
            "status": "success",
            "data": df.to_dict('records'),
            "row_count": len(df),
            "transformations_applied": ["remove_nulls", "normalize_names", "salary_category"]
        }
    except Exception as e:
        logger.error(f"Failed to transform data: {e}")
        return {"status": "error", "message": str(e)}
```

### Step 5: Create the Workflow

```bash
# Register the workflow
curl -X POST http://localhost:5000/api/workflows/ \
  -H "Content-Type: application/json" \
  -d @my_first_workflow.json
```

Or use the Python API:

```python
import requests
import json

# Load workflow definition
with open('my_first_workflow.json', 'r') as f:
    workflow_data = json.load(f)

# Create workflow
response = requests.post('http://localhost:5000/api/workflows/', json=workflow_data)
workflow = response.json()

print(f"Created workflow: {workflow['data']['id']}")
```

### Step 6: Execute the Workflow

```bash
# Execute the workflow
curl -X POST http://localhost:5000/api/workflows/{workflow_id}/execute \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "input_file": "/tmp/sample_data.csv",
      "output_format": "pdf"
    }
  }'
```

Or using Python:

```python
# Execute workflow
execution_response = requests.post(
    f'http://localhost:5000/api/workflows/{workflow["data"]["id"]}/execute',
    json={
        "parameters": {
            "input_file": "/tmp/sample_data.csv",
            "output_format": "pdf"
        }
    }
)

execution = execution_response.json()
execution_id = execution['data']['execution_id']
print(f"Started execution: {execution_id}")

# Monitor execution
import time
while True:
    status_response = requests.get(
        f'http://localhost:5000/api/workflows/executions/{execution_id}'
    )
    status = status_response.json()
    
    print(f"Status: {status['data']['status']} - Progress: {status['data']['progress']}%")
    
    if status['data']['status'] in ['completed', 'failed']:
        break
        
    time.sleep(2)

print("Workflow execution completed!")
```

## 🎯 Quick Examples

### Example 1: Simple API Integration

Create a workflow that fetches data from an API:

```json
{
  "name": "API Data Fetcher",
  "description": "Fetch data from external API",
  "engine": "native",
  "tasks": [
    {
      "id": "fetch_api_data",
      "type": "http",
      "config": {
        "url": "https://jsonplaceholder.typicode.com/users",
        "method": "GET",
        "headers": {
          "Accept": "application/json"
        }
      },
      "dependencies": []
    },
    {
      "id": "process_users",
      "type": "python",
      "config": {
        "function": "process_user_data",
        "parameters": {
          "format_names": true
        }
      },
      "dependencies": ["fetch_api_data"]
    }
  ]
}
```

### Example 2: Document Generation

Create a workflow that generates reports:

```json
{
  "name": "Monthly Report Generator",
  "description": "Generate monthly business reports",
  "engine": "native",
  "tasks": [
    {
      "id": "collect_metrics",
      "type": "python",
      "config": {
        "function": "collect_monthly_metrics"
      },
      "dependencies": []
    },
    {
      "id": "generate_pdf_report",
      "type": "document",
      "config": {
        "template": "monthly_report",
        "output_format": "pdf",
        "include_charts": true
      },
      "dependencies": ["collect_metrics"]
    },
    {
      "id": "send_email_notification",
      "type": "notification",
      "config": {
        "channels": ["email"],
        "recipients": ["manager@company.com"],
        "subject": "Monthly Report Generated",
        "attach_report": true
      },
      "dependencies": ["generate_pdf_report"]
    }
  ]
}
```

### Example 3: Real-time Data Processing

Create a workflow with real-time capabilities:

```json
{
  "name": "Real-time Data Processor",
  "description": "Process streaming data in real-time",
  "engine": "celery",
  "tasks": [
    {
      "id": "setup_stream_listener",
      "type": "streaming",
      "config": {
        "source": "bytewax",
        "topic": "user_events",
        "batch_size": 100
      },
      "dependencies": []
    },
    {
      "id": "process_events",
      "type": "python",
      "config": {
        "function": "process_user_events",
        "parameters": {
          "real_time": true
        }
      },
      "dependencies": ["setup_stream_listener"]
    },
    {
      "id": "update_dashboard",
      "type": "websocket",
      "config": {
        "channel": "dashboard_updates",
        "event_type": "metrics_update"
      },
      "dependencies": ["process_events"]
    }
  ]
}
```

## 🔧 Quick Configuration

### Environment Setup

Create a `.env` file for quick configuration:

```env
# APG Quick Start Configuration
DATABASE_URL=postgresql://$(whoami)@localhost:5432/apg_quickstart
REDIS_URL=redis://localhost:6379/0

# Application Settings
FLASK_ENV=development
DEBUG=True
SECRET_KEY=quickstart-secret-key

# APG Features
APG_DATA_DIR=./data
APG_LOGS_DIR=./logs
APG_ENABLE_WORKFLOWS=true
APG_ENABLE_AI_ML=false
APG_ENABLE_BLOCKCHAIN=false

# Quick Start Defaults
DEFAULT_WORKFLOW_ENGINE=native
ENABLE_REAL_TIME_FEATURES=true
AUTO_CREATE_SAMPLE_DATA=true
```

### Quick Feature Toggle

```python
# quick_config.py
QUICKSTART_FEATURES = {
    'workflows': True,
    'document_generation': True,
    'notifications': True,
    'real_time_collaboration': False,
    'ai_ml': False,
    'blockchain': False,
    'mobile_apps': False,
    'biometric_auth': False
}

def enable_feature(feature_name: str):
    """Enable a specific feature for quickstart"""
    if feature_name in QUICKSTART_FEATURES:
        QUICKSTART_FEATURES[feature_name] = True
        print(f"✅ Enabled {feature_name}")
    else:
        print(f"❌ Unknown feature: {feature_name}")

def disable_feature(feature_name: str):
    """Disable a specific feature for quickstart"""
    if feature_name in QUICKSTART_FEATURES:
        QUICKSTART_FEATURES[feature_name] = False
        print(f"🔴 Disabled {feature_name}")
    else:
        print(f"❌ Unknown feature: {feature_name}")
```

## 🎨 Web Interface Quick Tour

### Dashboard Access

1. **Open Dashboard**: Navigate to `http://localhost:5000`
2. **Login**: Use default credentials (user: admin, password: admin)
3. **Explore Features**:
   - **Workflows**: View and manage your workflows
   - **Executions**: Monitor running and completed workflows
   - **Analytics**: View system performance and usage
   - **Settings**: Configure APG features

### Creating Workflows via Web UI

1. **Navigate to Workflows** → Click "Create New Workflow"
2. **Choose Template** → Select from pre-built templates
3. **Configure Tasks** → Drag and drop tasks in the visual editor
4. **Set Parameters** → Configure task parameters and dependencies
5. **Test Workflow** → Run a test execution
6. **Deploy** → Enable the workflow for production use

## 📱 Mobile App Quick Setup

### BeeWare Mobile App

```bash
# Navigate to mobile app directory
cd mobile_apps/beeware

# Install BeeWare dependencies
pip install briefcase

# Create development build
briefcase dev

# Build for Android (requires Android SDK)
briefcase build android

# Build for iOS (macOS only, requires Xcode)
briefcase build iOS
```

### Mobile App Features

- **Offline Workflow Management**: Create and edit workflows offline
- **Real-time Synchronization**: Auto-sync with server when online
- **Push Notifications**: Receive workflow status updates
- **Biometric Authentication**: Secure access with fingerprint/face
- **Camera Integration**: Capture and process images in workflows

## 🤖 AI/ML Quick Start

### Enable AI Features

```bash
# Install AI/ML dependencies
pip install torch torchvision transformers scikit-learn

# Enable AI features
export APG_ENABLE_AI_ML=true
export PYTORCH_DEVICE=cpu  # or 'cuda' for GPU

# Restart APG
python cli.py run --debug
```

### Simple ML Workflow

```json
{
  "name": "Image Classification Pipeline",
  "description": "Classify uploaded images using AI",
  "engine": "native",
  "tasks": [
    {
      "id": "load_image",
      "type": "python",
      "config": {
        "function": "load_and_preprocess_image"
      },
      "dependencies": []
    },
    {
      "id": "classify_image",
      "type": "ai_ml",
      "config": {
        "model": "image_classifier",
        "model_type": "pytorch"
      },
      "dependencies": ["load_image"]
    },
    {
      "id": "save_results",
      "type": "python",
      "config": {
        "function": "save_classification_results"
      },
      "dependencies": ["classify_image"]
    }
  ]
}
```

## 🔗 Blockchain Quick Start

### Enable Blockchain Features

```bash
# Install blockchain dependencies
pip install web3 py-solc-x eth-account

# Enable blockchain features
export APG_ENABLE_BLOCKCHAIN=true
export WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# Install Solidity compiler
python -c "from solcx import install_solc; install_solc('0.8.19')"

# Restart APG
python cli.py run --debug
```

### Simple Blockchain Workflow

```json
{
  "name": "Smart Contract Deployment",
  "description": "Deploy and interact with smart contracts",
  "engine": "native",
  "tasks": [
    {
      "id": "compile_contract",
      "type": "blockchain",
      "config": {
        "action": "compile",
        "contract_source": "pragma solidity ^0.8.0; contract SimpleStorage { uint256 value; function set(uint256 _value) public { value = _value; } function get() public view returns (uint256) { return value; } }"
      },
      "dependencies": []
    },
    {
      "id": "deploy_contract",
      "type": "blockchain",
      "config": {
        "action": "deploy",
        "network": "ethereum",
        "gas_limit": 2000000
      },
      "dependencies": ["compile_contract"]
    }
  ]
}
```

## 🔍 Troubleshooting Quick Fixes

### Common Issues

**1. Database Connection Error**
```bash
# Check if PostgreSQL is running
pg_isready

# Create database if missing
createdb apg_quickstart
```

**2. Redis Connection Error**
```bash
# Start Redis
redis-server

# Test connection
redis-cli ping
```

**3. Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**4. Port Already in Use**
```bash
# Use different port
python cli.py run --debug --port 5001

# Or kill process using port 5000
lsof -ti:5000 | xargs kill -9
```

### Quick Health Check

```bash
# Run system health check
python -c "
from capabilities.composition.workflow_orchestration.service import WorkflowOrchestrationService
print('✅ Workflow service available')

import redis
r = redis.from_url('redis://localhost:6379/0')
r.ping()
print('✅ Redis connection working')

import psycopg2
conn = psycopg2.connect('postgresql://$(whoami)@localhost:5432/apg_quickstart')
conn.close()
print('✅ Database connection working')

print('🎉 All systems operational!')
"
```

## 📚 Next Steps

Now that you have APG running, explore these areas:

1. **[Architecture Guide](./architecture.md)** - Understand APG's design
2. **[Capabilities Overview](./capabilities/README.md)** - Explore all available features
3. **[API Reference](./api/README.md)** - Learn the complete API
4. **[Deployment Guide](./deployment.md)** - Deploy to production
5. **[Troubleshooting](./troubleshooting.md)** - Solve common issues

### Example Projects

Create these example projects to learn APG:

- **Data Pipeline**: ETL workflows with CSV/JSON processing
- **API Integration**: Connect to external services and APIs  
- **Document Generator**: Create PDF reports and invoices
- **Notification System**: Multi-channel alert system
- **Real-time Dashboard**: Live data visualization
- **Mobile App**: Cross-platform workflow management
- **AI Assistant**: Chatbot with NLP capabilities
- **Blockchain DApp**: Smart contract interaction

### Community & Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Email**: nyimbi@gmail.com for direct support
- **Documentation**: This comprehensive guide

---

*🚀 Happy building with APG!*

*Next: [Architecture Overview](./architecture.md) →*