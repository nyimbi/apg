#!/usr/bin/env python3
"""
APG Base Template System
========================

Defines the core application architectures that serve as foundations
for capability composition.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BaseTemplateType(Enum):
    """Available base template types"""
    FLASK_WEBAPP = "flask_webapp"
    MICROSERVICE = "microservice" 
    API_ONLY = "api_only"
    DASHBOARD = "dashboard"
    REAL_TIME = "real_time"
    CLI_TOOL = "cli_tool"


@dataclass
class BaseTemplate:
    """Base template definition"""
    name: str
    type: BaseTemplateType
    description: str
    framework: str
    capabilities_supported: List[str]
    default_capabilities: List[str]
    structure: Dict[str, Any]
    requirements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'framework': self.framework,
            'capabilities_supported': self.capabilities_supported,
            'default_capabilities': self.default_capabilities,
            'structure': self.structure,
            'requirements': self.requirements
        }


class BaseTemplateManager:
    """Manages base templates"""
    
    def __init__(self, bases_dir: Path):
        self.bases_dir = bases_dir
        self._templates_cache = {}
        
    def get_available_bases(self) -> List[BaseTemplateType]:
        """Get list of available base template types"""
        return list(BaseTemplateType)
    
    def get_base_template(self, template_type: BaseTemplateType) -> Optional[BaseTemplate]:
        """Get base template by type"""
        if template_type in self._templates_cache:
            return self._templates_cache[template_type]
            
        template_dir = self.bases_dir / template_type.value
        if not template_dir.exists():
            return None
            
        template_json = template_dir / "base.json"
        if not template_json.exists():
            return None
            
        try:
            with open(template_json, 'r') as f:
                data = json.load(f)
            
            template = BaseTemplate(
                name=data['name'],
                type=template_type,
                description=data['description'],
                framework=data['framework'],
                capabilities_supported=data.get('capabilities_supported', []),
                default_capabilities=data.get('default_capabilities', []),
                structure=data.get('structure', {}),
                requirements=data.get('requirements', [])
            )
            
            self._templates_cache[template_type] = template
            return template
            
        except Exception as e:
            print(f"Error loading base template {template_type.value}: {e}")
            return None
    
    def create_base_structure(self, template_dir: Path, template: BaseTemplate) -> bool:
        """Create base template directory structure"""
        try:
            template_dir.mkdir(parents=True, exist_ok=True)
            
            # Create base.json
            with open(template_dir / "base.json", 'w') as f:
                json.dump(template.to_dict(), f, indent=2)
            
            # Create directory structure based on template.structure
            self._create_directories(template_dir, template.structure)
            
            # Create template files
            self._create_template_files(template_dir, template)
            
            return True
            
        except Exception as e:
            print(f"Error creating base template structure: {e}")
            return False
    
    def _create_directories(self, base_dir: Path, structure: Dict[str, Any]):
        """Create directory structure recursively"""
        for name, content in structure.items():
            if isinstance(content, dict):
                dir_path = base_dir / name
                dir_path.mkdir(exist_ok=True)
                self._create_directories(dir_path, content)
            else:
                # It's a file
                file_path = base_dir / name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if not file_path.exists():
                    file_path.touch()
    
    def _create_template_files(self, template_dir: Path, template: BaseTemplate):
        """Create template files for the base"""
        files_to_create = [
            'app.py.template',
            'config.py.template',
            'requirements.txt.template',
            'README.md.template',
            '__init__.py.template'
        ]
        
        for filename in files_to_create:
            file_path = template_dir / filename
            if not file_path.exists():
                content = self._generate_template_content(filename, template)
                with open(file_path, 'w') as f:
                    f.write(content)
    
    def _generate_template_content(self, filename: str, template: BaseTemplate) -> str:
        """Generate template file content"""
        if filename == 'app.py.template':
            return self._generate_app_template(template)
        elif filename == 'config.py.template':
            return self._generate_config_template(template)
        elif filename == 'requirements.txt.template':
            return self._generate_requirements_template(template)
        elif filename == 'README.md.template':
            return self._generate_readme_template(template)
        else:
            return f"""# APG Base Template: {template.name}
# File: {filename}
# Framework: {template.framework}
#
# This file is part of the {template.name} base template.
# Capabilities will be integrated into this base structure.
"""
    
    def _generate_app_template(self, template: BaseTemplate) -> str:
        """Generate app.py template"""
        if template.type == BaseTemplateType.FLASK_WEBAPP:
            return '''"""
{{project_name}} - Flask-AppBuilder Application
================================================

Generated by APG (Application Programming Generation) using {{base_template}} base template.
This application combines multiple capabilities into a cohesive Flask-AppBuilder application.
"""

import logging
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.security.decorators import has_access

# Configure logging
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)
app.config.from_object('config')

# Initialize Flask-AppBuilder
db = SQLA(app)
appbuilder = AppBuilder(app, db.session)

# Import capability modules
{% for capability in capabilities %}
try:
    from capabilities.{{capability}}.integration import integrate_{{capability}}
    integrate_{{capability}}(app, appbuilder, db)
    log.info(f"Integrated capability: {{capability}}")
except ImportError as e:
    log.warning(f"Could not integrate capability {{capability}}: {e}")
{% endfor %}

# Application health check
@app.route('/health')
def health_check():
    """Application health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'application': '{{project_name}}',
        'version': '{{version}}',
        'capabilities': {{capabilities|tojson}}
    })

# Initialize database
with app.app_context():
    try:
        db.create_all()
        log.info("Database tables created successfully")
    except Exception as e:
        log.error(f"Error creating database tables: {e}")

if __name__ == '__main__':
    # Development server configuration
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    
    log.info(f"Starting {{project_name}} on {host}:{port}")
    log.info(f"Capabilities enabled: {', '.join({{capabilities|tojson}})}")
    
    app.run(host=host, port=port, debug=debug)
'''
        
        elif template.type == BaseTemplateType.MICROSERVICE:
            return '''"""
{{project_name}} - Microservice
===============================

Generated by APG using {{base_template}} base template.
This microservice combines multiple capabilities into a focused service.
"""

import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="{{project_name}}",
    description="{{project_description}}",
    version="{{version}}"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and integrate capabilities
{% for capability in capabilities %}
try:
    from capabilities.{{capability}}.integration import integrate_{{capability}}
    integrate_{{capability}}(app)
    logger.info(f"Integrated capability: {{capability}}")
except ImportError as e:
    logger.warning(f"Could not integrate capability {{capability}}: {e}")
{% endfor %}

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "{{project_name}}",
        "version": "{{version}}",
        "capabilities": {{capabilities|tojson}}
    }

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting {{project_name}} microservice on {host}:{port}")
    logger.info(f"Capabilities: {', '.join({{capabilities|tojson}})}")
    
    uvicorn.run(app, host=host, port=port)
'''
        
        else:
            return f'''"""
{{{{project_name}}}} - {template.name}
{'=' * (len(template.name) + 20)}

Generated by APG using {template.name} base template.
"""

# TODO: Implement {template.name} application structure
'''
    
    def _generate_config_template(self, template: BaseTemplate) -> str:
        """Generate config.py template"""
        return '''"""
Configuration for {{project_name}}
==================================

Generated by APG with composable capabilities.
"""

import os
from flask_appbuilder.security.manager import AUTH_DB

# Base directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Core configuration
SECRET_KEY = os.environ.get('SECRET_KEY') or '{{secret_key}}'
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or '{{database_url}}'

# Flask-AppBuilder configuration
APP_NAME = "{{project_name}}"
APP_THEME = ""
APP_ICON = ""

# Authentication
AUTH_TYPE = AUTH_DB
AUTH_ROLE_ADMIN = 'Admin'
AUTH_ROLE_PUBLIC = 'Public'

# Security
CSRF_ENABLED = True
WTF_CSRF_ENABLED = True

# Database
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_POOL_PRE_PING = True

# Capability-specific configuration
{% for capability in capabilities %}
# Configuration for {{capability}} capability
{{capability.upper()}}_ENABLED = True
{% endfor %}

# Development/Production settings
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
TESTING = False

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}
'''
    
    def _generate_requirements_template(self, template: BaseTemplate) -> str:
        """Generate requirements.txt template"""
        base_requirements = template.requirements.copy()
        
        requirements_text = f"""# APG Generated Requirements - {template.name}
# Base template requirements
"""
        
        for req in base_requirements:
            requirements_text += f"{req}\n"
        
        requirements_text += """
# Capability-specific requirements
{% for capability in capabilities %}
# Requirements for {{capability}} capability
{% include 'capabilities/' + capability + '/requirements.txt' ignore missing %}
{% endfor %}

# Development dependencies (optional)
pytest>=7.4.0
pytest-flask>=1.2.0
black>=23.7.0
flake8>=6.0.0
"""
        
        return requirements_text
    
    def _generate_readme_template(self, template: BaseTemplate) -> str:
        """Generate README.md template"""
        return '''# {{project_name}}

{{project_description}}

Generated by APG (Application Programming Generation) using the **{{base_template}}** base template with the following capabilities:

{% for capability in capabilities %}
- **{{capability}}**: {{capability_descriptions[capability]}}
{% endfor %}

## Features

This application provides:

{% for capability in capabilities %}
{% include 'capabilities/' + capability + '/FEATURES.md' ignore missing %}
{% endfor %}

## Quick Start

### Prerequisites

- Python {{python_version}} or higher
- pip package manager

### Installation

1. Clone or download this application
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   export SECRET_KEY="your-secret-key"
   export DATABASE_URL="{{database_url}}"
   ```

4. Initialize the database:
   ```bash
   flask fab create-admin
   ```

### Running the Application

```bash
python app.py
```

The application will be available at: http://localhost:8080

### Default Login

- Username: admin
- Password: (set during `flask fab create-admin`)

## Architecture

This application uses the APG composable capability architecture:

- **Base Template**: {{base_template}} - {{base_description}}
- **Capabilities**: Modular features that can be combined
- **Integration Layer**: Seamless integration between capabilities

### Capabilities Included

{% for capability in capabilities %}
#### {{capability}}

{{capability_descriptions[capability]}}

- Configuration: See `capabilities/{{capability}}/README.md`
- API Endpoints: See `capabilities/{{capability}}/API.md`

{% endfor %}

## Development

### Adding New Capabilities

This application supports adding new capabilities:

```bash
apg add capability new_capability_name
apg regenerate
```

### Testing

```bash
pytest tests/
```

### Code Quality

```bash
black .
flake8 .
```

## Deployment

### Docker

```bash
docker build -t {{project_name}} .
docker run -p 8080:8080 {{project_name}}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | Development key |
| `DATABASE_URL` | Database connection URL | SQLite database |
| `FLASK_ENV` | Flask environment | production |

## Support

This application was generated by APG. For support:

- APG Documentation: https://apg-lang.org/docs
- APG GitHub: https://github.com/apg-lang/apg

## License

{{license}}
'''


# Predefined base templates
BASE_TEMPLATES = {
    BaseTemplateType.FLASK_WEBAPP: BaseTemplate(
        name="Flask Web Application",
        type=BaseTemplateType.FLASK_WEBAPP,
        description="Full-featured web application with Flask-AppBuilder",
        framework="Flask-AppBuilder",
        capabilities_supported=["auth", "ai", "data", "payments", "business", "analytics"],
        default_capabilities=["auth/basic_authentication", "data/postgresql_database"],
        structure={
            "models": {},
            "views": {},
            "templates": {"html": {}},
            "static": {"css": {}, "js": {}, "img": {}},
            "capabilities": {},
            "tests": {}
        },
        requirements=[
            "Flask-AppBuilder>=4.3.0",
            "Flask>=2.3.0",
            "SQLAlchemy>=2.0.0",
            "WTForms>=3.0.0"
        ]
    ),
    
    BaseTemplateType.MICROSERVICE: BaseTemplate(
        name="Microservice",
        type=BaseTemplateType.MICROSERVICE,
        description="Lightweight microservice with FastAPI",
        framework="FastAPI",
        capabilities_supported=["auth", "ai", "data", "business"],
        default_capabilities=["auth/jwt_authentication"],
        structure={
            "api": {"v1": {}},
            "models": {},
            "capabilities": {},
            "tests": {}
        },
        requirements=[
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.0.0",
            "sqlalchemy>=2.0.0"
        ]
    ),
    
    BaseTemplateType.API_ONLY: BaseTemplate(
        name="API-Only Service",
        type=BaseTemplateType.API_ONLY,
        description="Pure API service without UI",
        framework="FastAPI",
        capabilities_supported=["auth", "ai", "data", "business"],
        default_capabilities=["auth/jwt_authentication"],
        structure={
            "api": {},
            "models": {},
            "capabilities": {},
            "tests": {}
        },
        requirements=[
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0"
        ]
    ),
    
    BaseTemplateType.DASHBOARD: BaseTemplate(
        name="Analytics Dashboard",
        type=BaseTemplateType.DASHBOARD,
        description="Real-time dashboard with analytics",
        framework="Flask-AppBuilder",
        capabilities_supported=["analytics", "data", "auth"],
        default_capabilities=["analytics/basic_analytics", "data/postgresql_database", "auth/basic_authentication"],
        structure={
            "dashboards": {},
            "charts": {},
            "data": {},
            "capabilities": {},
            "templates": {"dashboards": {}},
            "static": {"css": {}, "js": {}},
            "tests": {}
        },
        requirements=[
            "Flask-AppBuilder>=4.3.0",
            "plotly>=5.0.0",
            "pandas>=2.0.0"
        ]
    ),
    
    BaseTemplateType.REAL_TIME: BaseTemplate(
        name="Real-Time Application",
        type=BaseTemplateType.REAL_TIME,
        description="Real-time application with WebSocket support",
        framework="Flask-SocketIO",
        capabilities_supported=["communication", "auth", "data"],
        default_capabilities=["communication/websocket_communication", "auth/basic_authentication"],
        structure={
            "events": {},
            "rooms": {},
            "capabilities": {},
            "templates": {"real_time": {}},
            "static": {"js": {}},
            "tests": {}
        },
        requirements=[
            "Flask-SocketIO>=5.0.0",
            "eventlet>=0.33.0"
        ]
    )
}