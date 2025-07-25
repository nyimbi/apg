#!/usr/bin/env python3
"""
APG Capability System
====================

Defines focused capability modules that can be composed into applications.
Each capability is a self-contained feature module with its own models, views,
templates, and integration logic.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class CapabilityCategory(Enum):
    """Capability categories"""
    AUTH = "auth"
    AI = "ai"
    DATA = "data"
    PAYMENTS = "payments"
    IOT = "iot"
    BUSINESS = "business"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class CapabilityDependency:
    """Capability dependency definition"""
    name: str
    version: Optional[str] = None
    optional: bool = False
    reason: str = ""


@dataclass
class CapabilityIntegration:
    """Capability integration metadata"""
    models: List[str] = field(default_factory=list)
    views: List[str] = field(default_factory=list)
    apis: List[str] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)
    static_files: List[str] = field(default_factory=list)
    config_additions: Dict[str, Any] = field(default_factory=dict)
    database_migrations: List[str] = field(default_factory=list)


@dataclass
class Capability:
    """Capability definition"""
    name: str
    category: CapabilityCategory
    description: str
    version: str
    author: str = "APG Team"
    
    # Technical details
    python_requirements: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    dependencies: List[CapabilityDependency] = field(default_factory=list)
    
    # Integration metadata
    integration: CapabilityIntegration = field(default_factory=CapabilityIntegration)
    
    # Capability features and configuration
    features: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Compatibility
    compatible_bases: List[str] = field(default_factory=list)
    incompatible_capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'python_requirements': self.python_requirements,
            'system_requirements': self.system_requirements,
            'dependencies': [
                {
                    'name': dep.name,
                    'version': dep.version,
                    'optional': dep.optional,
                    'reason': dep.reason
                } for dep in self.dependencies
            ],
            'integration': {
                'models': self.integration.models,
                'views': self.integration.views,
                'apis': self.integration.apis,
                'templates': self.integration.templates,
                'static_files': self.integration.static_files,
                'config_additions': self.integration.config_additions,
                'database_migrations': self.integration.database_migrations
            },
            'features': self.features,
            'configuration': self.configuration,
            'compatible_bases': self.compatible_bases,
            'incompatible_capabilities': self.incompatible_capabilities
        }


class CapabilityManager:
    """Manages capability modules"""
    
    def __init__(self, capabilities_dir: Path):
        self.capabilities_dir = capabilities_dir
        self._capabilities_cache = {}
        self._category_cache = {}
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of available capability names"""
        capabilities = []
        for category_dir in self.capabilities_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                for cap_dir in category_dir.iterdir():
                    if cap_dir.is_dir() and (cap_dir / 'capability.json').exists():
                        capabilities.append(f"{category_dir.name}/{cap_dir.name}")
        return sorted(capabilities)
    
    def get_capabilities_by_category(self, category: CapabilityCategory) -> List[str]:
        """Get capabilities in a specific category"""
        if category in self._category_cache:
            return self._category_cache[category]
        
        category_dir = self.capabilities_dir / category.value
        capabilities = []
        
        if category_dir.exists():
            for cap_dir in category_dir.iterdir():
                if cap_dir.is_dir() and (cap_dir / 'capability.json').exists():
                    capabilities.append(f"{category.value}/{cap_dir.name}")
        
        self._category_cache[category] = capabilities
        return capabilities
    
    def get_capability(self, capability_name: str) -> Optional[Capability]:
        """Get capability by name (category/name format)"""
        if capability_name in self._capabilities_cache:
            return self._capabilities_cache[capability_name]
        
        if '/' not in capability_name:
            # Try to find in any category
            for category in CapabilityCategory:
                full_name = f"{category.value}/{capability_name}"
                capability = self.get_capability(full_name)
                if capability:
                    return capability
            return None
        
        capability_path = self.capabilities_dir / capability_name
        capability_json = capability_path / 'capability.json'
        
        if not capability_json.exists():
            return None
        
        try:
            with open(capability_json, 'r') as f:
                data = json.load(f)
            
            # Parse dependencies
            dependencies = []
            for dep_data in data.get('dependencies', []):
                dependencies.append(CapabilityDependency(
                    name=dep_data['name'],
                    version=dep_data.get('version'),
                    optional=dep_data.get('optional', False),
                    reason=dep_data.get('reason', '')
                ))
            
            # Parse integration
            integration_data = data.get('integration', {})
            integration = CapabilityIntegration(
                models=integration_data.get('models', []),
                views=integration_data.get('views', []),
                apis=integration_data.get('apis', []),
                templates=integration_data.get('templates', []),
                static_files=integration_data.get('static_files', []),
                config_additions=integration_data.get('config_additions', {}),
                database_migrations=integration_data.get('database_migrations', [])
            )
            
            capability = Capability(
                name=data['name'],
                category=CapabilityCategory(data['category']),
                description=data['description'],
                version=data['version'],
                author=data.get('author', 'APG Team'),
                python_requirements=data.get('python_requirements', []),
                system_requirements=data.get('system_requirements', []),
                dependencies=dependencies,
                integration=integration,
                features=data.get('features', []),
                configuration=data.get('configuration', {}),
                compatible_bases=data.get('compatible_bases', []),
                incompatible_capabilities=data.get('incompatible_capabilities', [])
            )
            
            self._capabilities_cache[capability_name] = capability
            return capability
            
        except Exception as e:
            print(f"Error loading capability {capability_name}: {e}")
            return None
    
    def resolve_dependencies(self, capability_names: List[str]) -> List[str]:
        """Resolve capability dependencies and return ordered list"""
        resolved = []
        visited = set()
        visiting = set()
        
        def visit(cap_name: str):
            if cap_name in visiting:
                raise ValueError(f"Circular dependency detected involving {cap_name}")
            if cap_name in visited:
                return
            
            visiting.add(cap_name)
            capability = self.get_capability(cap_name)
            
            if capability:
                for dep in capability.dependencies:
                    if not dep.optional:
                        visit(dep.name)
            
            visiting.remove(cap_name)
            visited.add(cap_name)
            resolved.append(cap_name)
        
        for cap_name in capability_names:
            visit(cap_name)
        
        return resolved
    
    def validate_capability_combination(self, capability_names: List[str]) -> Dict[str, List[str]]:
        """Validate that capabilities can work together"""
        issues = {
            'incompatible': [],
            'missing_dependencies': [],
            'conflicts': []
        }
        
        capabilities = {}
        for cap_name in capability_names:
            cap = self.get_capability(cap_name)
            if cap:
                capabilities[cap_name] = cap
        
        # Check incompatibilities
        for cap_name, capability in capabilities.items():
            for incompatible in capability.incompatible_capabilities:
                if incompatible in capability_names:
                    issues['incompatible'].append(f"{cap_name} is incompatible with {incompatible}")
        
        # Check missing dependencies
        for cap_name, capability in capabilities.items():
            for dep in capability.dependencies:
                if not dep.optional and dep.name not in capability_names:
                    issues['missing_dependencies'].append(f"{cap_name} requires {dep.name}: {dep.reason}")
        
        return issues
    
    def create_capability_structure(self, capability_dir: Path, capability: Capability) -> bool:
        """Create capability directory structure"""
        try:
            capability_dir.mkdir(parents=True, exist_ok=True)
            
            # Create capability.json
            with open(capability_dir / 'capability.json', 'w') as f:
                json.dump(capability.to_dict(), f, indent=2)
            
            # Create standard capability directories
            standard_dirs = [
                'models', 'views', 'templates', 'static', 'tests',
                'migrations', 'docs', 'config', 'scripts'
            ]
            
            for dir_name in standard_dirs:
                (capability_dir / dir_name).mkdir(exist_ok=True)
            
            # Create template files
            self._create_capability_files(capability_dir, capability)
            
            return True
            
        except Exception as e:
            print(f"Error creating capability structure: {e}")
            return False
    
    def _create_capability_files(self, capability_dir: Path, capability: Capability):
        """Create template files for the capability"""
        files_to_create = {
            '__init__.py.template': self._generate_init_template(capability),
            'integration.py.template': self._generate_integration_template(capability),
            'requirements.txt': '\n'.join(capability.python_requirements),
            'README.md': self._generate_readme_template(capability),
            'FEATURES.md': self._generate_features_template(capability),
            'API.md': self._generate_api_template(capability)
        }
        
        for filename, content in files_to_create.items():
            file_path = capability_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
    
    def _generate_init_template(self, capability: Capability) -> str:
        """Generate __init__.py template"""
        return f'''"""
{capability.name} Capability
{'=' * (len(capability.name) + 20)}

{capability.description}

Category: {capability.category.value}
Version: {capability.version}
Author: {capability.author}
"""

from .integration import integrate_{capability.name.lower().replace(' ', '_')}

__version__ = "{capability.version}"
__capability_name__ = "{capability.name}"
__category__ = "{capability.category.value}"

# Capability metadata
CAPABILITY_INFO = {{
    'name': '{capability.name}',
    'category': '{capability.category.value}',
    'version': '{capability.version}',
    'features': {capability.features},
    'author': '{capability.author}'
}}
'''
    
    def _generate_integration_template(self, capability: Capability) -> str:
        """Generate integration.py template"""
        return f'''"""
{capability.name} Integration
{'=' * (len(capability.name) + 20)}

Integration logic for the {capability.name} capability.
This module handles integrating the capability into the base application.
"""

import logging
from flask import Blueprint
from flask_appbuilder import BaseView

# Configure logging
log = logging.getLogger(__name__)

# Create capability blueprint
{capability.name.lower().replace(' ', '_')}_bp = Blueprint(
    '{capability.name.lower().replace(' ', '_')}',
    __name__,
    url_prefix='/{capability.category.value}/{capability.name.lower().replace(' ', '_')}',
    template_folder='templates',
    static_folder='static'
)


def integrate_{capability.name.lower().replace(' ', '_')}(app, appbuilder, db):
    """
    Integrate {capability.name} capability into the application.
    
    Args:
        app: Flask application instance
        appbuilder: Flask-AppBuilder instance
        db: SQLAlchemy database instance
    """
    try:
        # Register blueprint
        app.register_blueprint({capability.name.lower().replace(' ', '_')}_bp)
        
        # Import and register models
        from .models import *  # noqa
        
        # Import and register views
        from .views import *  # noqa
        
        # Register views with AppBuilder
        # appbuilder.add_view(YourView, "Your View", category="{capability.name}")
        
        # Apply configuration
        config_additions = {capability.integration.config_additions}
        for key, value in config_additions.items():
            app.config[key] = value
        
        log.info(f"Successfully integrated {capability.name} capability")
        
    except Exception as e:
        log.error(f"Failed to integrate {capability.name} capability: {{e}}")
        raise


class {capability.name.replace(' ', '')}Capability:
    """
    Main capability class for {capability.name}.
    
    This class provides the core functionality of the {capability.name} capability.
    """
    
    def __init__(self, app=None, appbuilder=None, db=None):
        self.app = app
        self.appbuilder = appbuilder
        self.db = db
        
        if app is not None:
            self.init_app(app, appbuilder, db)
    
    def init_app(self, app, appbuilder, db):
        """Initialize the capability with the application"""
        self.app = app
        self.appbuilder = appbuilder
        self.db = db
        
        # Capability-specific initialization
        self._initialize_capability()
    
    def _initialize_capability(self):
        """Initialize capability-specific components"""
        # TODO: Implement capability initialization
        pass
'''
    
    def _generate_readme_template(self, capability: Capability) -> str:
        """Generate README.md template"""
        return f'''# {capability.name} Capability

{capability.description}

## Overview

- **Category**: {capability.category.value}
- **Version**: {capability.version}
- **Author**: {capability.author}

## Features

{chr(10).join(f"- {feature}" for feature in capability.features)}

## Requirements

### Python Packages

{chr(10).join(f"- {req}" for req in capability.python_requirements)}

### System Requirements

{chr(10).join(f"- {req}" for req in capability.system_requirements)}

## Dependencies

{chr(10).join(f"- **{dep.name}**: {dep.reason}" for dep in capability.dependencies if not dep.optional)}

### Optional Dependencies

{chr(10).join(f"- **{dep.name}**: {dep.reason}" for dep in capability.dependencies if dep.optional)}

## Configuration

This capability can be configured through the following settings:

```python
# Configuration options
{chr(10).join(f"{key} = {repr(value)}" for key, value in capability.configuration.items())}
```

## Integration

This capability integrates with the following base templates:

{chr(10).join(f"- {base}" for base in capability.compatible_bases)}

## API Endpoints

See [API.md](API.md) for detailed API documentation.

## Usage Examples

```python
# TODO: Add usage examples
```

## Testing

```bash
pytest tests/test_{capability.name.lower().replace(' ', '_')}.py
```

## Development

### File Structure

```
{capability.category.value}/{capability.name.lower().replace(' ', '_')}/
├── capability.json         # Capability metadata
├── __init__.py.template    # Package initialization
├── integration.py.template # Integration logic
├── models/                 # Database models
├── views/                  # Web views and APIs
├── templates/              # HTML templates
├── static/                 # Static files (CSS, JS)
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── docs/                   # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This capability is part of the APG project and follows the same license.
'''
    
    def _generate_features_template(self, capability: Capability) -> str:
        """Generate FEATURES.md template"""
        return f'''## {capability.name} Features

{chr(10).join(f"### {feature}" + chr(10) + "Description of this feature." + chr(10) for feature in capability.features)}
'''
    
    def _generate_api_template(self, capability: Capability) -> str:
        """Generate API.md template"""
        return f'''# {capability.name} API Documentation

## Endpoints

### Health Check

```
GET /{capability.category.value}/{capability.name.lower().replace(' ', '_')}/health
```

Returns the health status of the {capability.name} capability.

**Response:**
```json
{{
  "status": "healthy",
  "capability": "{capability.name}",
  "version": "{capability.version}"
}}
```

## Authentication

This capability supports the following authentication methods:

- Basic Authentication
- JWT Tokens
- API Keys

## Error Handling

All API endpoints return standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

## Examples

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:8080/{capability.category.value}/{capability.name.lower().replace(' ', '_')}/health

# TODO: Add more examples
```

### Python Examples

```python
import requests

# Health check
response = requests.get('http://localhost:8080/{capability.category.value}/{capability.name.lower().replace(' ', '_')}/health')
print(response.json())

# TODO: Add more examples
```
'''


# Predefined core capabilities
CORE_CAPABILITIES = [
    # Authentication capabilities
    Capability(
        name="Basic Authentication",
        category=CapabilityCategory.AUTH,
        description="Username/password authentication with Flask-AppBuilder",
        version="1.0.0",
        python_requirements=["Flask-AppBuilder>=4.3.0", "WTForms>=3.0.0"],
        features=["User Registration", "Login/Logout", "Password Reset", "User Management"],
        compatible_bases=["flask_webapp", "dashboard", "real_time"],
        integration=CapabilityIntegration(
            models=["User", "Role", "Permission"],
            views=["UserView", "RoleView"],
            config_additions={"AUTH_TYPE": "AUTH_DB"}
        )
    ),
    
    Capability(
        name="JWT Authentication",
        category=CapabilityCategory.AUTH,
        description="JSON Web Token authentication for APIs",
        version="1.0.0",
        python_requirements=["PyJWT>=2.8.0", "cryptography>=3.0.0"],
        features=["Token Generation", "Token Validation", "Token Refresh", "API Authentication"],
        compatible_bases=["microservice", "api_only"],
        integration=CapabilityIntegration(
            models=["JWTToken"],
            apis=["auth/login", "auth/refresh", "auth/logout"]
        )
    ),
    
    # AI capabilities
    Capability(
        name="LLM Integration",
        category=CapabilityCategory.AI,
        description="Large Language Model integration with OpenAI, Anthropic, and local models",
        version="1.0.0",
        python_requirements=["openai>=1.0.0", "anthropic>=0.3.0", "transformers>=4.30.0"],
        features=["Text Generation", "Chat Completion", "Embeddings", "Model Management"],
        compatible_bases=["flask_webapp", "microservice", "api_only"],
        integration=CapabilityIntegration(
            models=["LLMModel", "Conversation", "Message"],
            views=["ChatView", "ModelView"],
            apis=["ai/chat", "ai/generate", "ai/embed"]
        )
    ),
    
    # Data capabilities
    Capability(
        name="PostgreSQL Database",
        category=CapabilityCategory.DATA,
        description="PostgreSQL database with SQLAlchemy integration",
        version="1.0.0",
        python_requirements=["psycopg2-binary>=2.9.0", "SQLAlchemy>=2.0.0"],
        system_requirements=["PostgreSQL 12+"],
        features=["Database Connectivity", "Migrations", "Connection Pooling"],
        compatible_bases=["flask_webapp", "microservice", "dashboard"],
        integration=CapabilityIntegration(
            config_additions={"SQLALCHEMY_DATABASE_URI": "postgresql://user:pass@localhost/db"}
        )
    ),
    
    Capability(
        name="Vector Database",
        category=CapabilityCategory.DATA,
        description="Vector database for AI/ML applications with similarity search",
        version="1.0.0",
        python_requirements=["pgvector>=0.2.0", "numpy>=1.24.0"],
        dependencies=[
            CapabilityDependency("data/postgresql", reason="Requires PostgreSQL with pgvector extension")
        ],
        features=["Vector Storage", "Similarity Search", "Embeddings Management"],
        compatible_bases=["flask_webapp", "microservice"],
        integration=CapabilityIntegration(
            models=["VectorDocument", "Embedding"],
            apis=["vector/search", "vector/store", "vector/similarity"]
        )
    ),
    
    # Payment capabilities
    Capability(
        name="Stripe Payments",
        category=CapabilityCategory.PAYMENTS,
        description="Stripe payment processing with webhooks",
        version="1.0.0",
        python_requirements=["stripe>=7.0.0"],
        features=["Payment Processing", "Subscription Management", "Webhook Handling", "Refunds"],
        compatible_bases=["flask_webapp", "microservice"],
        integration=CapabilityIntegration(
            models=["Payment", "Subscription", "Customer"],
            views=["PaymentView", "SubscriptionView"],
            apis=["payments/charge", "payments/subscribe", "payments/webhook"]
        )
    ),
    
    # Business capabilities
    Capability(
        name="Inventory Management",
        category=CapabilityCategory.BUSINESS,
        description="Inventory tracking and management system",
        version="1.0.0",
        features=["Stock Tracking", "Low Stock Alerts", "Inventory Reports", "Supplier Management"],
        compatible_bases=["flask_webapp", "dashboard"],
        integration=CapabilityIntegration(
            models=["Product", "Inventory", "Supplier", "StockMovement"],
            views=["InventoryView", "ProductView", "SupplierView"]
        )
    ),
    
    # Analytics capabilities
    Capability(
        name="Basic Analytics",
        category=CapabilityCategory.ANALYTICS,
        description="Basic analytics and reporting with charts",
        version="1.0.0",
        python_requirements=["plotly>=5.0.0", "pandas>=2.0.0"],
        features=["Dashboard Charts", "Data Visualization", "Report Generation", "KPI Tracking"],
        compatible_bases=["flask_webapp", "dashboard"],
        integration=CapabilityIntegration(
            views=["AnalyticsView", "ChartsView", "ReportsView"],
            templates=["analytics_dashboard.html", "charts.html"]
        )
    ),
    
    # Communication capabilities
    Capability(
        name="WebSocket Communication",
        category=CapabilityCategory.COMMUNICATION,
        description="Real-time WebSocket communication",
        version="1.0.0",
        python_requirements=["Flask-SocketIO>=5.3.0", "eventlet>=0.33.0"],
        features=["Real-time Messaging", "Room Management", "Event Broadcasting"],
        compatible_bases=["real_time", "flask_webapp"],
        integration=CapabilityIntegration(
            templates=["websocket_client.html"],
            static_files=["js/websocket.js"]
        )
    )
]