#!/usr/bin/env python3
"""
Setup Composable Template System
================================

Initializes the complete composable template system with base templates,
capabilities, and integration patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Import our composable system
import sys
sys.path.insert(0, str(Path(__file__).parent))

from templates.composable.base_template import (
    BaseTemplateManager, BaseTemplateType, BASE_TEMPLATES
)
from templates.composable.capability import (
    CapabilityManager, CapabilityCategory, CORE_CAPABILITIES
)


def setup_base_templates():
    """Set up all base templates"""
    print("🏗️  Setting up base templates...")
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    bases_dir = composable_root / 'bases'
    bases_dir.mkdir(parents=True, exist_ok=True)
    
    base_manager = BaseTemplateManager(bases_dir)
    
    for template_type, template in BASE_TEMPLATES.items():
        template_dir = bases_dir / template_type.value
        
        if base_manager.create_base_structure(template_dir, template):
            print(f"  ✅ Created base template: {template.name}")
        else:
            print(f"  ❌ Failed to create base template: {template.name}")
    
    print(f"📁 Base templates directory: {bases_dir}")


def setup_capabilities():
    """Set up all core capabilities"""
    print("\n🔧 Setting up core capabilities...")
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    capabilities_dir = composable_root / 'capabilities'
    capabilities_dir.mkdir(parents=True, exist_ok=True)
    
    capability_manager = CapabilityManager(capabilities_dir)
    
    # Create category directories
    for category in CapabilityCategory:
        category_dir = capabilities_dir / category.value
        category_dir.mkdir(exist_ok=True)
    
    # Create core capabilities
    for capability in CORE_CAPABILITIES:
        category_dir = capabilities_dir / capability.category.value
        capability_name = capability.name.lower().replace(' ', '_')
        capability_dir = category_dir / capability_name
        
        if capability_manager.create_capability_structure(capability_dir, capability):
            print(f"  ✅ Created capability: {capability.category.value}/{capability_name}")
        else:
            print(f"  ❌ Failed to create capability: {capability.category.value}/{capability_name}")
    
    print(f"📁 Capabilities directory: {capabilities_dir}")


def setup_integration_patterns():
    """Set up common integration patterns"""
    print("\n🔗 Setting up integration patterns...")
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    integrations_dir = composable_root / 'integrations'
    integrations_dir.mkdir(parents=True, exist_ok=True)
    
    # Define common integration patterns
    patterns = {
        'ai_platform': {
            'name': 'AI Platform',
            'description': 'Complete AI platform with LLM, vector database, and analytics',
            'base_template': 'flask_webapp',
            'capabilities': [
                'auth/basic_authentication',
                'ai/llm_integration',
                'data/postgresql_database',
                'data/vector_database',
                'analytics/basic_analytics'
            ],
            'use_cases': ['AI Applications', 'Chatbots', 'Knowledge Management']
        },
        
        'ecommerce_complete': {
            'name': 'Complete E-Commerce',
            'description': 'Full e-commerce platform with payments and inventory',
            'base_template': 'flask_webapp',
            'capabilities': [
                'auth/basic_authentication',
                'payments/stripe_payments',
                'business/inventory_management',
                'data/postgresql_database',
                'analytics/basic_analytics'
            ],
            'use_cases': ['Online Stores', 'Marketplaces', 'B2B Commerce']
        },
        
        'iot_monitoring': {
            'name': 'IoT Monitoring Platform',
            'description': 'IoT device monitoring with real-time data and analytics',
            'base_template': 'dashboard',
            'capabilities': [
                'auth/basic_authentication',
                'data/postgresql_database',
                'communication/websocket_communication',
                'analytics/basic_analytics'
            ],
            'use_cases': ['Smart Cities', 'Industrial IoT', 'Environmental Monitoring']
        },
        
        'microservice_api': {
            'name': 'Microservice API',
            'description': 'Lightweight microservice with JWT authentication',
            'base_template': 'microservice',
            'capabilities': [
                'auth/jwt_authentication',
                'data/postgresql_database'
            ],
            'use_cases': ['API Services', 'Microservices Architecture', 'Backend Services']
        },
        
        'analytics_dashboard': {
            'name': 'Analytics Dashboard',
            'description': 'Real-time analytics dashboard with data visualization',
            'base_template': 'dashboard',
            'capabilities': [
                'auth/basic_authentication',
                'data/postgresql_database',
                'analytics/basic_analytics',
                'communication/websocket_communication'
            ],
            'use_cases': ['Business Intelligence', 'KPI Dashboards', 'Data Visualization']
        }
    }
    
    for pattern_id, pattern_info in patterns.items():
        pattern_dir = integrations_dir / pattern_id
        pattern_dir.mkdir(exist_ok=True)
        
        # Create pattern.json
        with open(pattern_dir / 'pattern.json', 'w') as f:
            json.dump(pattern_info, f, indent=2)
        
        # Create README.md
        readme_content = f"""# {pattern_info['name']} Integration Pattern

{pattern_info['description']}

## Components

- **Base Template**: {pattern_info['base_template']}
- **Capabilities**: {len(pattern_info['capabilities'])} capabilities

### Capabilities Included

{chr(10).join(f"- {cap}" for cap in pattern_info['capabilities'])}

## Use Cases

{chr(10).join(f"- {use_case}" for use_case in pattern_info['use_cases'])}

## Usage

```bash
apg create project --pattern {pattern_id}
```

## Generated Features

This pattern generates a complete, production-ready application with:

- Authentication and user management
- Database integration with proper models
- Professional web interface
- API endpoints
- Real-time capabilities (if applicable)
- Analytics and reporting (if applicable)

## Customization

You can customize this pattern by:

1. Adding additional capabilities
2. Modifying capability configurations
3. Extending with custom code

```bash
apg add capability custom_capability
apg regenerate
```
"""
        
        with open(pattern_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"  ✅ Created integration pattern: {pattern_id}")
    
    print(f"📁 Integration patterns directory: {integrations_dir}")


def create_system_overview():
    """Create system overview documentation"""
    print("\n📚 Creating system overview...")
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    
    overview_content = """# APG Composable Template System

The APG Composable Template System enables generating world-class applications by intelligently combining base templates with focused capability modules.

## Architecture

```
APG Application = Base Template + Capability Modules + Integration Layer
```

### Components

1. **Base Templates** (4 core architectures)
   - `flask_webapp`: Full-featured web application
   - `microservice`: Lightweight microservice
   - `api_only`: Pure API service
   - `dashboard`: Analytics dashboard
   - `real_time`: Real-time application

2. **Capabilities** (20+ focused modules)
   - `auth/*`: Authentication and authorization
   - `ai/*`: Artificial intelligence and ML
   - `data/*`: Data storage and processing
   - `payments/*`: Payment processing
   - `business/*`: Business logic modules
   - `communication/*`: Communication and messaging
   - `analytics/*`: Analytics and reporting

3. **Integration Patterns** (Pre-defined combinations)
   - `ai_platform`: Complete AI platform
   - `ecommerce_complete`: Full e-commerce solution
   - `iot_monitoring`: IoT monitoring platform
   - `microservice_api`: Microservice with JWT auth
   - `analytics_dashboard`: Real-time analytics

## Usage

### Automatic Composition

```bash
# APG analyzes your source and selects optimal components
apg compile my_app.apg
```

### Manual Composition

```bash
# Create project with specific base and capabilities
apg create project --base flask_webapp --capabilities auth/basic ai/llm data/postgresql
```

### Integration Patterns

```bash
# Use pre-defined patterns for common use cases
apg create project --pattern ai_platform
```

## Benefits

1. **Modularity**: Each capability is focused and testable
2. **Reusability**: Capabilities work across multiple base templates
3. **Maintainability**: Update one capability → all apps benefit
4. **Extensibility**: Community can contribute capabilities
5. **Quality**: Each component embodies best practices

## Directory Structure

```
templates/composable/
├── bases/                  # Base template architectures
│   ├── flask_webapp/
│   ├── microservice/
│   ├── api_only/
│   ├── dashboard/
│   └── real_time/
├── capabilities/           # Capability modules
│   ├── auth/
│   ├── ai/
│   ├── data/
│   ├── payments/
│   ├── business/
│   ├── communication/
│   └── analytics/
└── integrations/          # Pre-defined patterns
    ├── ai_platform/
    ├── ecommerce_complete/
    ├── iot_monitoring/
    └── analytics_dashboard/
```

## Development

### Adding New Capabilities

1. Create capability directory structure
2. Define capability.json metadata
3. Implement integration logic
4. Add templates and static files
5. Test with multiple base templates

### Adding New Base Templates

1. Define base template metadata
2. Create template structure
3. Implement capability integration hooks
4. Test with various capability combinations

### Creating Integration Patterns

1. Define pattern metadata
2. Specify base template and capabilities
3. Document use cases and benefits
4. Test end-to-end application generation

## Quality Standards

Every component must:
- Be production-ready
- Include comprehensive tests
- Have clear documentation
- Follow security best practices
- Support multiple deployment targets
- Include error handling and logging

This ensures APG generates truly world-class applications.
"""
    
    with open(composable_root / 'README.md', 'w') as f:
        f.write(overview_content)
    
    print(f"  ✅ Created system overview: {composable_root / 'README.md'}")


def verify_system():
    """Verify the composable system is properly set up"""
    print("\n🔍 Verifying composable template system...")
    
    composable_root = Path(__file__).parent / 'templates' / 'composable'
    
    # Check base templates
    bases_dir = composable_root / 'bases'
    base_count = len([d for d in bases_dir.iterdir() if d.is_dir()])
    print(f"  📁 Base templates: {base_count}")
    
    # Check capabilities
    capabilities_dir = composable_root / 'capabilities'
    capability_count = 0
    for category_dir in capabilities_dir.iterdir():
        if category_dir.is_dir():
            capability_count += len([d for d in category_dir.iterdir() if d.is_dir()])
    print(f"  🔧 Capabilities: {capability_count}")
    
    # Check integration patterns
    integrations_dir = composable_root / 'integrations'
    integration_count = len([d for d in integrations_dir.iterdir() if d.is_dir()])
    print(f"  🔗 Integration patterns: {integration_count}")
    
    # Check system files
    system_files = [
        '__init__.py',
        'base_template.py',
        'capability.py',
        'composition_engine.py',
        'README.md'
    ]
    
    missing_files = []
    for filename in system_files:
        if not (composable_root / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"  ❌ Missing system files: {', '.join(missing_files)}")
    else:
        print(f"  ✅ All system files present")
    
    print(f"\n📊 System Summary:")
    print(f"   • Base Templates: {base_count}")
    print(f"   • Capabilities: {capability_count}")
    print(f"   • Integration Patterns: {integration_count}")
    print(f"   • Status: {'✅ Ready' if not missing_files else '⚠️  Incomplete'}")


def main():
    """Main setup function"""
    print("🚀 Setting up APG Composable Template System")
    print("=" * 60)
    
    try:
        # Set up all components
        setup_base_templates()
        setup_capabilities()
        setup_integration_patterns()
        create_system_overview()
        
        # Verify everything is working
        verify_system()
        
        print("\n🎉 APG Composable Template System Setup Complete!")
        print("=" * 60)
        print("\n📖 Next Steps:")
        print("1. Implement individual base templates")
        print("2. Implement core capabilities")
        print("3. Test composition engine")
        print("4. Create additional capabilities")
        print("5. Build community contribution system")
        
        print(f"\n📁 System Location: {Path(__file__).parent / 'templates' / 'composable'}")
        
    except Exception as e:
        print(f"\n💥 Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()