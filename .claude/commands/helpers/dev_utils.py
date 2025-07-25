"""
Development Utilities for /dev Command

Provides helper functions and utilities to support the automated
capability development process.
"""

import os
import re
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

def parse_capability_path(capability_path: str) -> Tuple[str, str]:
    """
    Parse capability path into capability and sub-capability names.
    
    Args:
        capability_path: Path in format "capability/sub_capability" or "capability"
        
    Returns:
        Tuple of (capability_name, sub_capability_name)
    """
    parts = capability_path.strip('/').split('/')
    
    if len(parts) == 1:
        return parts[0], parts[0]  # Use same name for both
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Invalid capability path: {capability_path}")

def normalize_name(name: str) -> str:
    """
    Normalize capability/sub-capability name for use in code.
    
    Args:
        name: Raw name string
        
    Returns:
        Normalized name suitable for Python identifiers
    """
    # Convert to snake_case
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = name.lower()
    
    # Ensure it starts with a letter
    if name and name[0].isdigit():
        name = f"cap_{name}"
    
    return name

def to_pascal_case(name: str) -> str:
    """
    Convert name to PascalCase for class names.
    
    Args:
        name: Name in snake_case or other format
        
    Returns:
        PascalCase version of the name
    """
    return ''.join(word.capitalize() for word in name.split('_'))

def to_title_case(name: str) -> str:
    """
    Convert name to Title Case for display purposes.
    
    Args:
        name: Name in snake_case or other format
        
    Returns:
        Title Case version of the name
    """
    return ' '.join(word.capitalize() for word in name.split('_'))

def generate_model_prefix(capability: str, sub_capability: str) -> str:
    """
    Generate model prefix for database tables and class names.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Prefix string (e.g., "GCCRMR" for General Cross-functional CRM)
    """
    # Map capability names to short codes
    capability_codes = {
        'core_financials': 'CF',
        'human_resources': 'HR',
        'procurement_purchasing': 'PP',
        'inventory_management': 'IM',
        'sales_order_management': 'SO',
        'manufacturing': 'MF',
        'supply_chain_management': 'SC',
        'service_specific': 'SS',
        'pharmaceutical_specific': 'PH',
        'mining_specific': 'MN',
        'platform_services': 'PS',
        'general_cross_functional': 'GC'
    }
    
    capability_code = capability_codes.get(capability, capability[:2].upper())
    
    # Generate sub-capability code
    if sub_capability == capability:
        sub_code = capability_code
    else:
        # Take first letters of each word
        words = sub_capability.replace('_', ' ').split()
        if len(words) == 1:
            sub_code = words[0][:3].upper()
        else:
            sub_code = ''.join(word[0].upper() for word in words)
    
    return f"{capability_code}{sub_code}"

def get_capability_directory(capability: str, sub_capability: str) -> Path:
    """
    Get the directory path for a capability/sub-capability.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Path object for the capability directory
    """
    base_dir = Path("capabilities")
    return base_dir / capability / sub_capability

def create_directory_structure(capability: str, sub_capability: str) -> Path:
    """
    Create the complete directory structure for a capability.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Path to the created capability directory
    """
    cap_dir = get_capability_directory(capability, sub_capability)
    
    # Create main directories
    directories = [
        cap_dir,
        cap_dir / "tests",
        cap_dir / "tests" / "fixtures",
        cap_dir / "docs",
        cap_dir / "docs" / "screenshots",
        cap_dir / "static",
        cap_dir / "static" / "css",
        cap_dir / "static" / "js",
        cap_dir / "static" / "images",
        cap_dir / "templates",
        cap_dir / "templates" / "base",
        cap_dir / "templates" / "forms",
        cap_dir / "templates" / "dashboards"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return cap_dir

def load_template(template_name: str, **kwargs) -> str:
    """
    Load and process a template file with variable substitution.
    
    Args:
        template_name: Name of the template file
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template content
    """
    template_dir = Path(__file__).parent.parent / "commands" / "templates"
    template_path = template_dir / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitute template variables
    for key, value in kwargs.items():
        placeholder = f"{{{key.upper()}}}"
        content = content.replace(placeholder, str(value))
    
    return content

def generate_capability_metadata(capability: str, sub_capability: str, **extra_config) -> Dict[str, Any]:
    """
    Generate capability metadata configuration.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        **extra_config: Additional configuration options
        
    Returns:
        Capability metadata dictionary
    """
    model_prefix = generate_model_prefix(capability, sub_capability)
    
    metadata = {
        'name': to_title_case(sub_capability),
        'code': model_prefix,
        'version': '1.0.0',
        'capability': capability,
        'description': f'Advanced {to_title_case(sub_capability)} capability with AI integration',
        'industry_focus': 'All',
        'dependencies': [],
        'optional_dependencies': ['document_management', 'business_intelligence', 'workflow_management'],
        'database_tables': [],
        'api_endpoints': [],
        'views': [],
        'permissions': [
            f'{sub_capability}.read',
            f'{sub_capability}.write',
            f'{sub_capability}.delete',
            f'{sub_capability}.admin'
        ],
        'menu_items': [],
        'configuration': {
            'max_records_per_page': 100,
            'search_results_limit': 1000,
            'dashboard_refresh_minutes': 15,
            'report_cache_hours': 4,
            'bulk_operation_limit': 500
        }
    }
    
    # Merge extra configuration
    metadata.update(extra_config)
    
    return metadata

def generate_todo_from_template(capability: str, sub_capability: str, priority: str = "High") -> str:
    """
    Generate a todo.md file from template.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        priority: Development priority level
        
    Returns:
        Generated todo content
    """
    return load_template(
        "todo_template.md",
        capability_name=to_title_case(capability),
        sub_capability_name=to_title_case(sub_capability),
        start_date=datetime.now().strftime('%Y-%m-%d'),
        target_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        priority_level=priority,
        capability_path=f"{capability}/{sub_capability}",
        entity_plural=f"{sub_capability}s",
        primary_entity=sub_capability,
        primary_model_class=f"{generate_model_prefix(capability, sub_capability)}{to_pascal_case(sub_capability)}",
        primary_table=f"{capability}_{sub_capability}",
        list_technical_risks="Technical complexity, integration challenges",
        list_business_risks="User adoption, business process changes",
        list_timeline_risks="Scope creep, resource availability",
        list_resource_risks="Developer availability, infrastructure capacity",
        list_internal_deps="Authentication system, database infrastructure",
        list_external_deps="Third-party APIs, cloud services",
        list_third_party_deps="AI/ML services, notification services"
    )

def generate_cap_spec_from_template(capability: str, sub_capability: str) -> str:
    """
    Generate a cap_spec.md file from template.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Generated capability specification content
    """
    return load_template(
        "cap_spec_template.md",
        capability_name=to_title_case(sub_capability)
    )

def get_industry_best_practices(capability: str, sub_capability: str) -> List[Dict[str, Any]]:
    """
    Get industry best practices for a specific capability.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        List of best practices and features
    """
    # This would ideally connect to a knowledge base or API
    # For now, return common enterprise features
    
    common_features = [
        {
            'category': 'AI/ML Integration',
            'features': [
                'Predictive analytics and forecasting',
                'Intelligent automation and recommendations',
                'Natural language processing for search',
                'Computer vision for document processing',
                'Anomaly detection and alerts'
            ]
        },
        {
            'category': 'User Experience',
            'features': [
                'Mobile-first responsive design',
                'Real-time collaboration features',
                'Contextual help and guidance',
                'Personalized dashboards',
                'Voice interface support'
            ]
        },
        {
            'category': 'Enterprise Features',
            'features': [
                'Advanced role-based access control',
                'Comprehensive audit trails',
                'Multi-tenant architecture',
                'SSO and identity federation',
                'Advanced reporting and analytics'
            ]
        },
        {
            'category': 'Integration & APIs',
            'features': [
                'RESTful and GraphQL APIs',
                'Real-time WebSocket connections',
                'Webhook support for integrations',
                'Event-driven architecture',
                'Microservices compatibility'
            ]
        },
        {
            'category': 'Performance & Scalability',
            'features': [
                'Horizontal scaling support',
                'Advanced caching strategies',
                'Database optimization',
                'CDN integration',
                'Performance monitoring'
            ]
        }
    ]
    
    return common_features

def validate_capability_structure(capability_dir: Path) -> List[str]:
    """
    Validate that a capability has the required file structure.
    
    Args:
        capability_dir: Path to capability directory
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_files = [
        'cap_spec.md',
        'todo.md',
        '__init__.py',
        'models.py',
        'service.py',
        'views.py',
        'api.py',
        'blueprint.py'
    ]
    
    required_dirs = [
        'tests',
        'docs',
        'static',
        'templates'
    ]
    
    # Check required files
    for file_name in required_files:
        file_path = capability_dir / file_name
        if not file_path.exists():
            errors.append(f"Missing required file: {file_name}")
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = capability_dir / dir_name
        if not dir_path.exists():
            errors.append(f"Missing required directory: {dir_name}")
    
    return errors

def estimate_development_time(capability: str, sub_capability: str) -> Dict[str, int]:
    """
    Estimate development time for different phases.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Dictionary with time estimates in hours
    """
    # Base estimates for different phases
    base_estimates = {
        'analysis_planning': 16,    # 2 days
        'models_service': 24,       # 3 days
        'api_development': 16,      # 2 days
        'ui_views': 32,            # 4 days
        'testing': 24,             # 3 days
        'documentation': 16,        # 2 days
        'integration': 8,          # 1 day
        'deployment': 8            # 1 day
    }
    
    # Complexity multipliers based on capability type
    complexity_multipliers = {
        'core_financials': 1.5,
        'human_resources': 1.3,
        'manufacturing': 1.4,
        'platform_services': 1.6,
        'general_cross_functional': 1.2
    }
    
    multiplier = complexity_multipliers.get(capability, 1.0)
    
    # Apply multiplier to all estimates
    estimates = {
        phase: int(hours * multiplier)
        for phase, hours in base_estimates.items()
    }
    
    estimates['total'] = sum(estimates.values())
    
    return estimates

def generate_project_plan(capability: str, sub_capability: str) -> Dict[str, Any]:
    """
    Generate a comprehensive project plan.
    
    Args:
        capability: Capability name
        sub_capability: Sub-capability name
        
    Returns:
        Project plan dictionary
    """
    estimates = estimate_development_time(capability, sub_capability)
    start_date = datetime.now()
    
    project_plan = {
        'capability': capability,
        'sub_capability': sub_capability,
        'start_date': start_date.isoformat(),
        'estimated_completion': (start_date + timedelta(days=estimates['total'] // 8)).isoformat(),
        'time_estimates': estimates,
        'phases': [
            {
                'name': 'Analysis & Planning',
                'duration_hours': estimates['analysis_planning'],
                'deliverables': ['cap_spec.md', 'todo.md', 'architecture_design'],
                'dependencies': []
            },
            {
                'name': 'Core Development',
                'duration_hours': estimates['models_service'] + estimates['api_development'],
                'deliverables': ['models.py', 'service.py', 'api.py'],
                'dependencies': ['Analysis & Planning']
            },
            {
                'name': 'User Interface',
                'duration_hours': estimates['ui_views'],
                'deliverables': ['views.py', 'templates', 'static_assets'],
                'dependencies': ['Core Development']
            },
            {
                'name': 'Testing & QA',
                'duration_hours': estimates['testing'],
                'deliverables': ['test_suite', 'qa_report'],
                'dependencies': ['User Interface']
            },
            {
                'name': 'Documentation',
                'duration_hours': estimates['documentation'],
                'deliverables': ['user_guide.md', 'developer_guide.md', 'integration_guide.md'],
                'dependencies': ['Testing & QA']
            },
            {
                'name': 'Integration & Deployment',
                'duration_hours': estimates['integration'] + estimates['deployment'],
                'deliverables': ['blueprint.py', 'deployment_config'],
                'dependencies': ['Documentation']
            }
        ],
        'milestones': [
            {
                'name': 'Requirements Complete',
                'date': (start_date + timedelta(days=2)).isoformat(),
                'criteria': ['cap_spec.md approved', 'todo.md finalized']
            },
            {
                'name': 'Core Features Complete',
                'date': (start_date + timedelta(days=8)).isoformat(),
                'criteria': ['models implemented', 'service layer complete', 'APIs functional']
            },
            {
                'name': 'UI Complete',
                'date': (start_date + timedelta(days=12)).isoformat(),
                'criteria': ['all views implemented', 'responsive design', 'accessibility compliance']
            },
            {
                'name': 'Testing Complete',
                'date': (start_date + timedelta(days=15)).isoformat(),
                'criteria': ['>95% test coverage', 'performance benchmarks met', 'security scan clean']
            },
            {
                'name': 'Production Ready',
                'date': (start_date + timedelta(days=18)).isoformat(),
                'criteria': ['documentation complete', 'deployment tested', 'monitoring configured']
            }
        ]
    }
    
    return project_plan