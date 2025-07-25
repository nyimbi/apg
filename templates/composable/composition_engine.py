#!/usr/bin/env python3
"""
APG Composition Engine
======================

Intelligently composes applications by combining base templates with capability modules
based on APG AST analysis and user requirements.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from jinja2 import Environment, FileSystemLoader, Template

from .base_template import BaseTemplateManager, BaseTemplateType, BaseTemplate
from .capability import CapabilityManager, Capability, CapabilityCategory


@dataclass
class CompositionContext:
    """Context for application composition"""
    project_name: str
    project_description: str
    author: str = "APG Developer"
    version: str = "1.0.0"
    
    # Base template
    base_template: Optional[BaseTemplate] = None
    
    # Selected capabilities
    capabilities: List[Capability] = field(default_factory=list)
    capability_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # APG-specific context
    apg_agents: List[Dict[str, Any]] = field(default_factory=list)
    apg_digital_twins: List[Dict[str, Any]] = field(default_factory=list)
    apg_workflows: List[Dict[str, Any]] = field(default_factory=list)
    apg_databases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Generated file paths
    output_directory: Optional[Path] = None
    
    def to_template_context(self) -> Dict[str, Any]:
        """Convert to Jinja2 template context"""
        return {
            'project_name': self.project_name,
            'project_description': self.project_description,
            'author': self.author,
            'version': self.version,
            'base_template': self.base_template.name if self.base_template else '',
            'base_description': self.base_template.description if self.base_template else '',
            'capabilities': [cap.name.lower().replace(' ', '_') for cap in self.capabilities],
            'capability_descriptions': {
                cap.name.lower().replace(' ', '_'): cap.description 
                for cap in self.capabilities
            },
            'python_version': '3.12',
            'database_url': 'sqlite:///app.db',
            'secret_key': 'dev-secret-key-change-in-production',
            'license': 'MIT',
            
            # APG entities
            'agents': self.apg_agents,
            'digital_twins': self.apg_digital_twins,
            'workflows': self.apg_workflows,
            'databases': self.apg_databases,
            
            # Capability-specific context
            **self.capability_configs
        }


class APGASTAnalyzer:
    """Analyzes APG AST to detect required capabilities"""
    
    def __init__(self):
        self.capability_keywords = {
            # Authentication indicators
            'auth': ['user', 'login', 'password', 'authenticate', 'session', 'token'],
            'auth_jwt': ['jwt', 'token', 'api_key', 'bearer'],
            'auth_oauth': ['oauth', 'google', 'github', 'facebook'],
            
            # AI indicators
            'ai_llm': ['llm', 'gpt', 'claude', 'openai', 'chat', 'conversation', 'generate'],
            'ai_ml': ['model', 'predict', 'train', 'inference', 'ml', 'ai'],
            'ai_vision': ['image', 'vision', 'ocr', 'detection', 'classification'],
            'ai_nlp': ['nlp', 'text', 'sentiment', 'entity', 'language', 'parse'],
            
            # Data indicators
            'data_postgresql': ['postgresql', 'postgres', 'pg'],
            'data_mysql': ['mysql'],
            'data_mongodb': ['mongodb', 'mongo', 'document'],
            'data_redis': ['redis', 'cache'],
            'data_vector': ['vector', 'embedding', 'similarity', 'search'],
            
            # Payment indicators
            'payments_stripe': ['stripe', 'payment', 'charge', 'subscription'],
            'payments_paypal': ['paypal'],
            'payments_crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain'],
            
            # Business indicators
            'business_inventory': ['inventory', 'stock', 'product', 'warehouse'],
            'business_crm': ['customer', 'contact', 'lead', 'sales'],
            'business_accounting': ['invoice', 'accounting', 'finance', 'ledger'],
            'business_hr': ['employee', 'hr', 'payroll', 'recruitment'],
            
            # Communication indicators
            'comm_email': ['email', 'smtp', 'mail'],
            'comm_sms': ['sms', 'text', 'twilio'],
            'comm_websocket': ['websocket', 'realtime', 'live', 'broadcast'],
            'comm_notification': ['notification', 'alert', 'push'],
            
            # Analytics indicators
            'analytics_basic': ['analytics', 'chart', 'dashboard', 'report', 'metrics'],
            'analytics_advanced': ['bi', 'olap', 'warehouse', 'etl'],
            
            # IoT indicators
            'iot_devices': ['device', 'sensor', 'iot', 'mqtt'],
            'iot_twins': ['twin', 'digital_twin', 'simulation'],
            
            # Security indicators
            'security_encryption': ['encrypt', 'decrypt', 'cipher', 'ssl', 'tls'],
            'security_audit': ['audit', 'log', 'compliance', 'gdpr'],
        }
    
    def analyze_ast(self, apg_ast) -> Dict[str, Any]:
        """Analyze APG AST to extract application characteristics"""
        characteristics = {
            'base_template_hints': [],
            'required_capabilities': [],
            'optional_capabilities': [],
            'agents': [],
            'digital_twins': [],
            'workflows': [],
            'databases': [],
            'detected_keywords': set()
        }
        
        # Extract entities from AST
        if hasattr(apg_ast, 'entities'):
            for entity in apg_ast.entities:
                self._analyze_entity(entity, characteristics)
        
        # Detect capabilities based on keywords
        self._detect_capabilities_from_keywords(characteristics)
        
        # Infer base template
        characteristics['base_template_hints'] = self._infer_base_template(characteristics)
        
        return characteristics
    
    def _analyze_entity(self, entity, characteristics: Dict[str, Any]):
        """Analyze individual APG entity"""
        entity_info = {
            'name': getattr(entity, 'name', 'Unknown'),
            'type': getattr(entity, 'entity_type', 'Unknown'),
            'properties': []
        }
        
        # Extract keywords from entity
        if hasattr(entity, 'name'):
            self._extract_keywords(entity.name, characteristics['detected_keywords'])
        
        # Analyze properties
        if hasattr(entity, 'properties'):
            for prop in entity.properties:
                prop_info = {
                    'name': getattr(prop, 'name', ''),
                    'type': str(getattr(prop, 'type_annotation', ''))
                }
                entity_info['properties'].append(prop_info)
                
                # Extract keywords from property names and types
                self._extract_keywords(prop_info['name'], characteristics['detected_keywords'])
                self._extract_keywords(prop_info['type'], characteristics['detected_keywords'])
        
        # Analyze methods
        if hasattr(entity, 'methods'):
            for method in entity.methods:
                if hasattr(method, 'name'):
                    self._extract_keywords(method.name, characteristics['detected_keywords'])
        
        # Categorize entity
        entity_type = getattr(entity, 'entity_type', None)
        if entity_type:
            if entity_type.name == 'AGENT':
                characteristics['agents'].append(entity_info)
            elif entity_type.name == 'DIGITAL_TWIN':
                characteristics['digital_twins'].append(entity_info)
            elif entity_type.name == 'WORKFLOW':
                characteristics['workflows'].append(entity_info)
            elif entity_type.name == 'DATABASE':
                characteristics['databases'].append(entity_info)
    
    def _extract_keywords(self, text: str, keywords_set: Set[str]):
        """Extract relevant keywords from text"""
        if not text:
            return
        
        # Clean and split text
        words = re.findall(r'\w+', text.lower())
        keywords_set.update(words)
    
    def _detect_capabilities_from_keywords(self, characteristics: Dict[str, Any]):
        """Detect required capabilities based on keywords"""
        detected_keywords = characteristics['detected_keywords']
        required_caps = set()
        optional_caps = set()
        
        for capability, keywords in self.capability_keywords.items():
            score = len(detected_keywords.intersection(keywords))
            
            if score >= 2:  # Strong indication
                required_caps.add(capability)
            elif score >= 1:  # Weak indication
                optional_caps.add(capability)
        
        # Always include basic auth for web apps
        if characteristics['agents'] or characteristics['workflows']:
            required_caps.add('auth_basic')
        
        # If any database indicators, add PostgreSQL
        if any('database' in str(entity).lower() for entity in characteristics['databases']):
            required_caps.add('data_postgresql')
        
        characteristics['required_capabilities'] = list(required_caps)
        characteristics['optional_capabilities'] = list(optional_caps)
    
    def _infer_base_template(self, characteristics: Dict[str, Any]) -> List[str]:
        """Infer best base template based on characteristics"""
        hints = []
        
        # Web application indicators
        if (characteristics['agents'] or 
            any('auth' in cap for cap in characteristics['required_capabilities'])):
            hints.append('flask_webapp')
        
        # API service indicators
        if any('api' in str(entity).lower() for entity in characteristics['agents']):
            hints.append('api_only')
            hints.append('microservice')
        
        # Dashboard indicators
        if any('analytics' in cap for cap in characteristics['required_capabilities']):
            hints.append('dashboard')
        
        # Real-time indicators
        if any('websocket' in cap or 'realtime' in cap for cap in characteristics['required_capabilities']):
            hints.append('real_time')
        
        # Default to web application
        if not hints:
            hints.append('flask_webapp')
        
        return hints


class CompositionEngine:
    """Main composition engine that combines base templates with capabilities"""
    
    def __init__(self, composable_root: Path):
        self.composable_root = composable_root
        self.base_manager = BaseTemplateManager(composable_root / 'bases')
        self.capability_manager = CapabilityManager(composable_root / 'capabilities')
        self.ast_analyzer = APGASTAnalyzer()
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(composable_root)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def compose_application(self, apg_ast, project_name: str, project_description: str, 
                          author: str = "APG Developer") -> CompositionContext:
        """Compose application from APG AST"""
        
        # Analyze AST
        analysis = self.ast_analyzer.analyze_ast(apg_ast)
        
        # Select base template
        base_template = self._select_base_template(analysis['base_template_hints'])
        
        # Select capabilities
        capabilities = self._select_capabilities(analysis['required_capabilities'], base_template)
        
        # Create composition context
        context = CompositionContext(
            project_name=project_name,
            project_description=project_description,
            author=author,
            base_template=base_template,
            capabilities=capabilities,
            apg_agents=analysis['agents'],
            apg_digital_twins=analysis['digital_twins'],
            apg_workflows=analysis['workflows'],
            apg_databases=analysis['databases']
        )
        
        return context
    
    def _select_base_template(self, hints: List[str]) -> BaseTemplate:
        """Select best base template from hints"""
        # Try hints in order
        for hint in hints:
            try:
                template_type = BaseTemplateType(hint)
                template = self.base_manager.get_base_template(template_type)
                if template:
                    return template
            except ValueError:
                continue
        
        # Default to flask_webapp
        return self.base_manager.get_base_template(BaseTemplateType.FLASK_WEBAPP)
    
    def _select_capabilities(self, required_caps: List[str], base_template: BaseTemplate) -> List[Capability]:
        """Select and validate capabilities"""
        capabilities = []
        
        # Add default capabilities from base template
        for default_cap in base_template.default_capabilities:
            capability = self.capability_manager.get_capability(default_cap)
            if capability:
                capabilities.append(capability)
        
        # Add required capabilities
        for cap_name in required_caps:
            # Try to find capability by name
            capability = self.capability_manager.get_capability(cap_name)
            if capability and capability not in capabilities:
                capabilities.append(capability)
        
        # Resolve dependencies
        cap_names = [f"{cap.category.value}/{cap.name.lower().replace(' ', '_')}" for cap in capabilities]
        resolved_names = self.capability_manager.resolve_dependencies(cap_names)
        
        # Build final capability list
        final_capabilities = []
        for cap_name in resolved_names:
            capability = self.capability_manager.get_capability(cap_name)
            if capability:
                final_capabilities.append(capability)
        
        return final_capabilities
    
    def generate_application_files(self, context: CompositionContext) -> Dict[str, str]:
        """Generate all application files from composition context"""
        generated_files = {}
        
        # Generate base template files
        base_files = self._generate_base_files(context)
        generated_files.update(base_files)
        
        # Generate capability files
        capability_files = self._generate_capability_files(context)
        generated_files.update(capability_files)
        
        # Generate integration files
        integration_files = self._generate_integration_files(context)
        generated_files.update(integration_files)
        
        return generated_files
    
    def _generate_base_files(self, context: CompositionContext) -> Dict[str, str]:
        """Generate base template files"""
        if not context.base_template:
            return {}
        
        base_dir = self.composable_root / 'bases' / context.base_template.type.value
        template_context = context.to_template_context()
        generated_files = {}
        
        # Process all .template files in base directory
        for template_file in base_dir.rglob('*.template'):
            relative_path = template_file.relative_to(base_dir)
            output_path = str(relative_path).replace('.template', '')
            
            try:
                template = self.jinja_env.get_template(f'bases/{context.base_template.type.value}/{relative_path}')
                content = template.render(**template_context)
                generated_files[output_path] = content
            except Exception as e:
                print(f"Error processing base template {template_file}: {e}")
        
        return generated_files
    
    def _generate_capability_files(self, context: CompositionContext) -> Dict[str, str]:
        """Generate capability-specific files"""
        generated_files = {}
        template_context = context.to_template_context()
        
        for capability in context.capabilities:
            cap_dir_name = f"{capability.category.value}/{capability.name.lower().replace(' ', '_')}"
            cap_dir = self.composable_root / 'capabilities' / cap_dir_name
            
            if not cap_dir.exists():
                continue
            
            # Process capability template files
            for template_file in cap_dir.rglob('*.template'):
                relative_path = template_file.relative_to(cap_dir)
                output_path = f"capabilities/{cap_dir_name}/{str(relative_path).replace('.template', '')}"
                
                try:
                    template = self.jinja_env.get_template(f'capabilities/{cap_dir_name}/{relative_path}')
                    content = template.render(**template_context)
                    generated_files[output_path] = content
                except Exception as e:
                    print(f"Error processing capability template {template_file}: {e}")
        
        return generated_files
    
    def _generate_integration_files(self, context: CompositionContext) -> Dict[str, str]:
        """Generate integration and glue code"""
        generated_files = {}
        
        # Generate master integration file
        integration_content = self._generate_master_integration(context)
        generated_files['integration.py'] = integration_content
        
        # Generate capability registry
        registry_content = self._generate_capability_registry(context)
        generated_files['capability_registry.py'] = registry_content
        
        return generated_files
    
    def _generate_master_integration(self, context: CompositionContext) -> str:
        """Generate master integration file"""
        template_str = '''"""
Master Integration Module
========================

Integrates all capabilities with the {{base_template}} base template.
Generated by APG Composition Engine.
"""

import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

# Capability integration functions
{% for capability in capabilities %}
try:
    from capabilities.{{capability}}.integration import integrate_{{capability}}
except ImportError as e:
    log.warning(f"Could not import {{capability}} integration: {e}")
    integrate_{{capability}} = None
{% endfor %}

def integrate_all_capabilities(app, appbuilder=None, db=None) -> Dict[str, Any]:
    """
    Integrate all capabilities into the application.
    
    Returns:
        Dict with integration status for each capability
    """
    integration_status = {}
    
    {% for capability in capabilities %}
    # Integrate {{capability}}
    try:
        if integrate_{{capability}}:
            integrate_{{capability}}(app, appbuilder, db)
            integration_status['{{capability}}'] = {'status': 'success', 'error': None}
            log.info("Successfully integrated {{capability}}")
        else:
            integration_status['{{capability}}'] = {'status': 'skipped', 'error': 'Integration function not available'}
    except Exception as e:
        integration_status['{{capability}}'] = {'status': 'error', 'error': str(e)}
        log.error(f"Failed to integrate {{capability}}: {e}")
    
    {% endfor %}
    
    return integration_status

def get_capability_info() -> Dict[str, Any]:
    """Get information about integrated capabilities"""
    return {
        'base_template': '{{base_template}}',
        'capabilities': {{capabilities|tojson}},
        'integration_version': '1.0.0',
        'generated_by': 'APG Composition Engine'
    }
'''
        
        template = Template(template_str)
        return template.render(**context.to_template_context())
    
    def _generate_capability_registry(self, context: CompositionContext) -> str:
        """Generate capability registry"""
        template_str = '''"""
Capability Registry
==================

Registry of all capabilities integrated in this application.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class CapabilityInfo:
    name: str
    category: str
    version: str
    description: str
    features: List[str]

# Registered capabilities
CAPABILITIES: Dict[str, CapabilityInfo] = {
    {% for capability in capabilities %}
    '{{capability}}': CapabilityInfo(
        name='{{capability_descriptions[capability]}}',
        category='{{capability}}',  # TODO: Get actual category
        version='1.0.0',  # TODO: Get actual version
        description='{{capability_descriptions[capability]}}',
        features=[]  # TODO: Get actual features
    ),
    {% endfor %}
}

def get_capability(name: str) -> CapabilityInfo:
    """Get capability information by name"""
    return CAPABILITIES.get(name)

def list_capabilities() -> List[str]:
    """List all registered capability names"""
    return list(CAPABILITIES.keys())

def get_capabilities_by_category(category: str) -> List[CapabilityInfo]:
    """Get capabilities by category"""
    return [cap for cap in CAPABILITIES.values() if cap.category == category]
'''
        
        template = Template(template_str)
        return template.render(**context.to_template_context())
    
    def validate_composition(self, context: CompositionContext) -> Dict[str, List[str]]:
        """Validate that the composition is valid"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        if not context.base_template:
            issues['errors'].append("No base template selected")
            return issues
        
        if not context.capabilities:
            issues['warnings'].append("No capabilities selected - application will be minimal")
        
        # Validate capability compatibility
        cap_names = [f"{cap.category.value}/{cap.name.lower().replace(' ', '_')}" for cap in context.capabilities]
        validation_result = self.capability_manager.validate_capability_combination(cap_names)
        
        issues['errors'].extend(validation_result['incompatible'])
        issues['errors'].extend(validation_result['missing_dependencies'])
        issues['warnings'].extend(validation_result['conflicts'])
        
        # Check base template compatibility
        for capability in context.capabilities:
            if (capability.compatible_bases and 
                context.base_template.type.value not in capability.compatible_bases):
                issues['warnings'].append(
                    f"Capability {capability.name} may not be fully compatible with {context.base_template.name}"
                )
        
        issues['info'].append(f"Selected base template: {context.base_template.name}")
        issues['info'].append(f"Selected {len(context.capabilities)} capabilities")
        
        return issues