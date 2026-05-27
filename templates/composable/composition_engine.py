#!/usr/bin/env python3
"""
APG Composition Engine
======================

Intelligently composes applications by combining base templates with capability modules
based on APG AST analysis and user requirements.
"""

import re
import json
from pprint import pformat
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
    apg_agent_teams: List[Dict[str, Any]] = field(default_factory=list)
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
            'agent_teams': self.apg_agent_teams,
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
            'agent_teams': [],
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
            'properties': [],
            'model': getattr(entity, 'model', None),
            'tools': getattr(entity, 'tools', []),
            'memory': str(getattr(entity, 'memory', '') or '')
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

        for attr in ('model', 'role', 'system_prompt', 'tools'):
            value = getattr(entity, attr, None)
            if value:
                self._extract_keywords(str(value), characteristics['detected_keywords'])
        
        # Categorize entity
        entity_type = getattr(entity, 'entity_type', None)
        if entity_type:
            if entity_type.name in {'AGENT', 'AI_AGENT'}:
                characteristics['agents'].append(entity_info)
                self._extract_keywords('ai llm agent', characteristics['detected_keywords'])
            elif entity_type.name == 'AGENT_TEAM':
                characteristics['agent_teams'].append(entity_info)
                self._extract_keywords('ai llm agent orchestration', characteristics['detected_keywords'])
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

        if characteristics['agents'] or characteristics['agent_teams']:
            required_caps.add('ai/llm_integration')
            if any('vector' in str(entity).lower() for entity in characteristics['agents']):
                required_caps.add('data/vector_database')
        
        # If any database indicators, add PostgreSQL
        if any('database' in str(entity).lower() for entity in characteristics['databases']):
            required_caps.add('data_postgresql')
        
        characteristics['required_capabilities'] = list(required_caps)
        characteristics['optional_capabilities'] = list(optional_caps)
    
    def _infer_base_template(self, characteristics: Dict[str, Any]) -> List[str]:
        """Infer best base template based on characteristics"""
        hints = []
        
        # Python web artifact indicators
        if (characteristics['agents'] or 
            any('auth' in cap for cap in characteristics['required_capabilities'])):
            hints.append('python_web')
        
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
        
        # Default to the Python web artifact base
        if not hints:
            hints.append('python_web')
        
        return hints


class CompositionEngine:
    """Main composition engine that combines base templates with capabilities"""
    
    def __init__(self, composable_root: Path):
        if not (composable_root / 'bases').exists():
            composable_root = Path(__file__).resolve().parent

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
            apg_agent_teams=analysis.get('agent_teams', []),
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
        
        # Default to python_web
        return self.base_manager.get_base_template(BaseTemplateType.PYTHON_WEB)
    
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

        # Generate executable capability contracts for the composed app
        contracts_content = self._generate_capability_contracts(context)
        generated_files['capability_contracts.py'] = contracts_content
        
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

def integrate_all_capabilities(application=None, registry=None, configuration=None) -> Dict[str, Any]:
    """
    Register all capability contracts with the composed APG application.
    
    Returns:
        Dict with integration status for each capability
    """
    integration_status = {}
    capability_config = configuration or {}
    
    {% for capability in capabilities %}
    # Integrate {{capability}}
    try:
        if integrate_{{capability}}:
            contract = integrate_{{capability}}(
                application,
                registry,
                capability_config.get('{{capability}}', {})
            )
            integration_status['{{capability}}'] = {'status': 'success', 'error': None}
            if isinstance(contract, dict):
                integration_status['{{capability}}']['contract'] = contract.get('name', '{{capability}}')
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
        registry = {
            record["id"]: {
                "name": record["name"],
                "category": record["category"],
                "version": record["version"],
                "description": record["description"],
                "features": record["features"],
            }
            for record in self._capability_records(context)
        }
        registry_literal = pformat(registry, width=100, sort_dicts=True)
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
_CAPABILITY_DATA: Dict[str, Dict[str, Any]] = {{registry_literal}}

CAPABILITIES: Dict[str, CapabilityInfo] = {
    capability_id: CapabilityInfo(**metadata)
    for capability_id, metadata in _CAPABILITY_DATA.items()
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
        return template.render(registry_literal=registry_literal)

    def _generate_capability_contracts(self, context: CompositionContext) -> str:
        """Generate executable capability contracts for the composed application."""
        contracts = {
            record["id"]: self._build_generated_contract(record, context)
            for record in self._capability_records(context)
        }
        contracts_literal = pformat(contracts, width=120, sort_dicts=True)
        template_str = '''"""
Generated Capability Contracts
==============================

Executable capability contracts for this composed APG application.
"""

from copy import deepcopy
from typing import Any, Dict, List


REQUIRED_CONTRACT_KEYS = {"configuration", "configuration_schema", "rule_engine", "ui", "theme"}
REQUIRED_SCHEMA_KEYS = {"tenant_id", "ui", "theme"}
REQUIRED_RULE_KEYS = {"name", "condition", "effect"}
REQUIRED_ROUTE_KEYS = {"name", "path", "component", "permission"}
REQUIRED_THEME_TOKENS = {"border.radius"}

CAPABILITY_CONTRACTS: Dict[str, Dict[str, Any]] = {{contracts_literal}}


def list_capability_contracts(tenant_id: str = "default") -> Dict[str, Dict[str, Any]]:
    """Return all capability contracts keyed by capability id."""
    return {
        capability_id: get_capability_contract(capability_id, tenant_id)
        for capability_id in CAPABILITY_CONTRACTS
    }


def get_capability_contract(capability_id: str, tenant_id: str = "default") -> Dict[str, Any]:
    """Return one generated capability contract."""
    if capability_id not in CAPABILITY_CONTRACTS:
        raise KeyError(f"Unknown capability contract: {capability_id}")
    contract = deepcopy(CAPABILITY_CONTRACTS[capability_id])
    contract["configuration"]["tenant_id"] = tenant_id
    return contract


def validate_capability_contracts() -> Dict[str, List[str]]:
    """Validate the generated contract shape without external dependencies."""
    errors: List[str] = []
    for capability_id, contract in CAPABILITY_CONTRACTS.items():
        errors.extend(_validate_contract(capability_id, contract))
    return {"errors": errors, "validated": [] if errors else sorted(CAPABILITY_CONTRACTS)}


def _validate_contract(capability_id: str, contract: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_CONTRACT_KEYS - set(contract))
    if missing:
        errors.append(f"{capability_id} missing keys: {', '.join(missing)}")
        return errors
    if contract.get("capability") != capability_id:
        errors.append(f"{capability_id} capability id mismatch")
    errors.extend(_validate_configuration(capability_id, contract.get("configuration"), contract.get("configuration_schema")))
    errors.extend(_validate_rule_engine(capability_id, contract.get("rule_engine")))
    errors.extend(_validate_ui(capability_id, contract.get("ui")))
    errors.extend(_validate_theme(capability_id, contract.get("theme")))
    return errors


def _validate_configuration(capability_id: str, configuration: Any, schema: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(configuration, dict):
        return [f"{capability_id} configuration must be a dict"]
    if not isinstance(configuration.get("tenant_id"), str) or not configuration["tenant_id"]:
        errors.append(f"{capability_id} configuration.tenant_id must be a non-empty string")
    if not isinstance(schema, dict):
        return [*errors, f"{capability_id} configuration_schema must be a dict"]
    missing_schema = sorted(REQUIRED_SCHEMA_KEYS - set(schema.get("required", [])))
    if missing_schema:
        errors.append(f"{capability_id} configuration_schema.required missing: {', '.join(missing_schema)}")
    return errors


def _validate_rule_engine(capability_id: str, rule_engine: Any) -> List[str]:
    if not isinstance(rule_engine, dict):
        return [f"{capability_id} rule_engine must be a dict"]
    errors: List[str] = []
    if rule_engine.get("type") != "deterministic":
        errors.append(f"{capability_id} rule_engine.type must be deterministic")
    rules = rule_engine.get("rules")
    if not isinstance(rules, list) or not rules:
        return [*errors, f"{capability_id} rule_engine.rules must be a non-empty list"]
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"{capability_id} rule_engine.rules[{index}] must be a dict")
            continue
        missing = sorted(REQUIRED_RULE_KEYS - set(rule))
        if missing:
            errors.append(f"{capability_id} rule_engine.rules[{index}] missing: {', '.join(missing)}")
        if not isinstance(rule.get("name"), str) or not rule["name"]:
            errors.append(f"{capability_id} rule_engine.rules[{index}].name must be a non-empty string")
        if not isinstance(rule.get("condition"), dict):
            errors.append(f"{capability_id} rule_engine.rules[{index}].condition must be a dict")
        if not isinstance(rule.get("effect"), dict):
            errors.append(f"{capability_id} rule_engine.rules[{index}].effect must be a dict")
        elif not rule["effect"].get("decision"):
            errors.append(f"{capability_id} rule_engine.rules[{index}].effect.decision is required")
    return errors


def _validate_ui(capability_id: str, ui: Any) -> List[str]:
    if not isinstance(ui, dict):
        return [f"{capability_id} ui must be a dict"]
    errors: List[str] = []
    if ui.get("requires_theme") is not True:
        errors.append(f"{capability_id} ui.requires_theme must be true")
    if not isinstance(ui.get("shell"), str) or not ui["shell"]:
        errors.append(f"{capability_id} ui.shell must be a non-empty string")
    if not isinstance(ui.get("template_roots"), list) or not ui["template_roots"]:
        errors.append(f"{capability_id} ui.template_roots must be a non-empty list")
    routes = ui.get("routes")
    if not isinstance(routes, list) or not routes:
        return [*errors, f"{capability_id} ui.routes must be a non-empty list"]
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            errors.append(f"{capability_id} ui.routes[{index}] must be a dict")
            continue
        missing = sorted(REQUIRED_ROUTE_KEYS - set(route))
        if missing:
            errors.append(f"{capability_id} ui.routes[{index}] missing: {', '.join(missing)}")
        for key in REQUIRED_ROUTE_KEYS:
            if key in route and (not isinstance(route[key], str) or not route[key]):
                errors.append(f"{capability_id} ui.routes[{index}].{key} must be a non-empty string")
        if isinstance(route.get("path"), str) and not route["path"].startswith("/"):
            errors.append(f"{capability_id} ui.routes[{index}].path must start with /")
    return errors


def _validate_theme(capability_id: str, theme: Any) -> List[str]:
    if not isinstance(theme, dict):
        return [f"{capability_id} theme must be a dict"]
    errors: List[str] = []
    if not isinstance(theme.get("name"), str) or not theme["name"]:
        errors.append(f"{capability_id} theme.name must be a non-empty string")
    tokens = theme.get("tokens")
    if not isinstance(tokens, dict) or not tokens:
        return [*errors, f"{capability_id} theme.tokens must be a non-empty dict"]
    missing_tokens = sorted(REQUIRED_THEME_TOKENS - set(tokens))
    if missing_tokens:
        errors.append(f"{capability_id} theme.tokens missing: {', '.join(missing_tokens)}")
    if not isinstance(theme.get("components"), dict) or not theme["components"]:
        errors.append(f"{capability_id} theme.components must be a non-empty dict")
    return errors


def evaluate_capability_rules(capability_id: str, context: Dict[str, Any], tenant_id: str = "default") -> Dict[str, Any]:
    """Evaluate deterministic rules for one generated capability contract."""
    contract = get_capability_contract(capability_id, tenant_id)
    matched: List[str] = []
    actions: List[Dict[str, Any]] = []
    decision = "allow"
    for rule in contract["rule_engine"]["rules"]:
        if _matches(rule["condition"], context):
            matched.append(rule["name"])
            effect = dict(rule["effect"])
            actions.append(effect)
            if effect.get("decision") == "deny":
                decision = "deny"
            elif effect.get("decision") == "require_review" and decision != "deny":
                decision = "require_review"
    return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
    for key, expected in condition.items():
        if key.endswith("_lt"):
            if not context.get(key[:-3], 0) < expected:
                return False
        elif key.endswith("_gt"):
            if not context.get(key[:-3], 0) > expected:
                return False
        elif context.get(key) != expected:
            return False
    return True
'''
        template = Template(template_str)
        return template.render(contracts_literal=contracts_literal)

    def _capability_records(self, context: CompositionContext) -> List[Dict[str, Any]]:
        """Return normalized metadata for selected capabilities."""
        records = []
        for capability in context.capabilities:
            slug = self._capability_slug(capability)
            capability_id = f"{capability.category.value}/{slug}"
            records.append({
                "id": capability_id,
                "slug": slug,
                "name": capability.name,
                "category": capability.category.value,
                "version": capability.version,
                "description": capability.description,
                "features": list(capability.features),
                "configuration": dict(capability.configuration),
                "dependencies": [
                    {
                        "name": dependency.name,
                        "version": dependency.version,
                        "optional": dependency.optional,
                        "reason": dependency.reason,
                    }
                    for dependency in capability.dependencies
                ],
            })
        return records

    def _build_generated_contract(self, record: Dict[str, Any], context: CompositionContext) -> Dict[str, Any]:
        """Build the executable contract shape for one generated app capability."""
        route_prefix = f"/{record['category']}/{record['slug'].replace('_', '-')}"
        default_theme = f"{record['slug']}_operations"
        return {
            "capability": record["id"],
            "display_name": record["name"],
            "configuration": {
                "tenant_id": "default",
                "capability": {
                    "id": record["id"],
                    "name": record["name"],
                    "category": record["category"],
                    "version": record["version"],
                    "description": record["description"],
                    "enabled": True,
                    "features": record["features"],
                    "dependencies": record["dependencies"],
                },
                "composition": {
                    "project_name": context.project_name,
                    "base_template": context.base_template.name if context.base_template else "",
                    "generated_by": "APG Composition Engine",
                },
                "execution": {
                    "require_tenant_context": True,
                    "audit_operations": True,
                    "policy_enforced": True,
                    "async_supported": True,
                },
                "capability_settings": record["configuration"],
                "ui": {
                    "enable_dashboard": True,
                    "enable_operations": True,
                    "enable_rules": True,
                    "enable_settings": True,
                },
                "theme": {
                    "default_theme": default_theme,
                    "allow_tenant_overrides": True,
                },
            },
            "configuration_schema": {
                "type": "object",
                "required": ["tenant_id", "capability", "composition", "execution", "ui", "theme"],
                "properties": {
                    "tenant_id": {"type": "string", "minLength": 1},
                    "capability": {"type": "object"},
                    "composition": {"type": "object"},
                    "execution": {"type": "object"},
                    "capability_settings": {"type": "object"},
                    "ui": {"type": "object"},
                    "theme": {"type": "object"},
                },
            },
            "rule_engine": {
                "type": "deterministic",
                "rules": [
                    {
                        "name": "tenant_context_required",
                        "description": f"{record['name']} operations require tenant context.",
                        "condition": {"tenant_context_present": False},
                        "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
                    },
                    {
                        "name": "operation_policy_required",
                        "description": f"{record['name']} write operations require policy enforcement.",
                        "condition": {"operation_type": "write", "policy_attached": False},
                        "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
                    },
                    {
                        "name": "high_risk_requires_review",
                        "description": f"High-risk {record['name']} operations require review.",
                        "condition": {"risk_level": "high", "review_recorded": False},
                        "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_review"},
                    },
                ],
            },
            "ui": {
                "shell": "apg_python",
                "api_prefix": f"{route_prefix}/api/v1",
                "routes": [
                    {"name": "dashboard", "path": f"{route_prefix}/dashboard", "component": "CapabilityDashboard", "permission": f"{record['id']}:view", "nav_group": "Overview"},
                    {"name": "operations", "path": f"{route_prefix}/operations", "component": "CapabilityOperations", "permission": f"{record['id']}:operate", "nav_group": "Operations"},
                    {"name": "rules", "path": f"{route_prefix}/rules", "component": "CapabilityRules", "permission": f"{record['id']}:govern", "nav_group": "Governance"},
                    {"name": "settings", "path": f"{route_prefix}/settings", "component": "CapabilitySettings", "permission": f"{record['id']}:admin", "nav_group": "Administration"},
                ],
                "template_roots": ["templates/", "static/"],
                "requires_theme": True,
            },
            "theme": {
                "name": default_theme,
                "tokens": {
                    "color.primary": "#28536B",
                    "color.accent": "#C44536",
                    "color.success": "#2F855A",
                    "color.warning": "#B7791F",
                    "color.danger": "#C53030",
                    "surface.canvas": "#F7F8FA",
                    "surface.panel": "#FFFFFF",
                    "text.primary": "#172033",
                    "text.secondary": "#52606D",
                    "border.radius": "8px",
                    "density": "compact",
                },
                "components": {
                    "dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "risk_style": "policy-band"},
                    "operations": {"visual": "work-queue", "status_style": "sla-chip"},
                    "rules": {"visual": "rule-list", "status_style": "decision-chip"},
                    "settings": {"visual": "settings-panel", "density": "compact"},
                },
            },
        }

    @staticmethod
    def _capability_slug(capability: Capability) -> str:
        """Return the generated-app-safe slug for a capability name."""
        return capability.name.lower().replace(" ", "_").replace("-", "_")
    
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
