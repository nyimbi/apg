"""
APG Code Generator Module
=========================

Generates Python code from APG Abstract Syntax Trees.
Transforms APG entities, workflows, and other constructs into executable Python code
with proper imports, type hints, and runtime support.
"""

from typing import Any, Dict, List, Optional, Set, TextIO
from dataclasses import dataclass
from pathlib import Path
import re
import sys

# Import AST nodes
from .ast_builder import (
	ASTNode, ModuleDeclaration, EntityDeclaration, PropertyDeclaration,
	MethodDeclaration, Parameter, TypeAnnotation, Expression, Statement,
	LiteralExpression, IdentifierExpression, BinaryExpression, CallExpression,
	UnaryExpression, MemberExpression, IndexExpression, ListExpression,
	DictExpression,
	AssignmentStatement, ReturnStatement, BlockStatement, ExpressionStatement, EntityType,
	DatabaseDeclaration, DatabaseSchema, TableDeclaration,
	AIAgentDeclaration, AgentTeamDeclaration, CapabilityDeclaration
)

# Import composable template system
sys.path.insert(0, str(Path(__file__).parent.parent))
from templates.composable.composition_engine import CompositionEngine
from templates.composable.base_template import BaseTemplateType


# ========================================
# Code Generation Configuration
# ========================================

@dataclass
class CodeGenConfig:
	"""Configuration for code generation"""
	target_language: str = "python"
	python_version: str = "3.12"
	use_type_hints: bool = True
	use_async: bool = True
	generate_tests: bool = False
	output_directory: str = "generated"
	package_name: str = "apg_generated"
	include_runtime: bool = True
	
	# Composable template system configuration
	use_composable_templates: bool = True
	preferred_base_template: Optional[str] = None
	additional_capabilities: List[str] = None
	exclude_capabilities: List[str] = None
	template_output_mode: str = "complete_app"  # "complete_app", "models_only", "hybrid"
	generate_docs: bool = False
	verbose: bool = False
	
	def __post_init__(self):
		if self.additional_capabilities is None:
			self.additional_capabilities = []
		if self.exclude_capabilities is None:
			self.exclude_capabilities = []


# ========================================
# Python Code Generator
# ========================================

class PythonCodeGenerator:
	"""
	Generates Python code from APG AST.
	
	Features:
	- Modern Python 3.12+ with type hints
	- Async/await support for agents and workflows
	- Dataclass-based entities
	- Pydantic models for validation
	- SQLAlchemy models for databases
	- Comprehensive imports and dependencies
	"""
	
	def __init__(self, config: CodeGenConfig = None):
		self.config = config or CodeGenConfig()
		self.output: List[str] = []
		self.imports: Set[str] = set()
		self.indent_level = 0
		
		# Code generation state
		self.current_module: Optional[ModuleDeclaration] = None
		self.current_entity: Optional[EntityDeclaration] = None
		self.generated_classes: Set[str] = set()
	
	def generate(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""
		Generate application from APG AST using composable template system.
		
		Args:
			ast: Root AST node (ModuleDeclaration)
			
		Returns:
			Dictionary mapping file names to generated code content
		"""
		self.current_module = ast
		
		# Use composable template system if enabled
		if self.config.use_composable_templates:
			return self._generate_with_composable_templates(ast)
		else:
			# Fall back to legacy generation method
			return self._generate_legacy_flask_app(ast)
	
	def _generate_with_composable_templates(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate application using the composable template system"""
		try:
			# Initialize composition engine
			composable_root = Path(__file__).parent.parent / 'templates' / 'composable'
			engine = CompositionEngine(composable_root)
			
			# Extract project information from AST
			project_name = ast.name or "APGGeneratedApp"
			project_description = f"APG generated application with {len(ast.entities)} entities"
			
			# Compose the application
			context = engine.compose_application(
				ast,
				project_name=project_name,
				project_description=project_description,
				author="APG Code Generator"
			)
			
			# Apply user preferences from config
			if self.config.preferred_base_template:
				# Override base template if specified
				try:
					base_type = BaseTemplateType(self.config.preferred_base_template)
					context.base_template = engine.base_manager.get_base_template(base_type)
				except ValueError:
					print(f"Warning: Unknown base template '{self.config.preferred_base_template}', using detected template")
			
			# Add additional capabilities
			for cap_name in self.config.additional_capabilities:
				capability = engine.capability_manager.get_capability(cap_name)
				if capability and capability not in context.capabilities:
					context.capabilities.append(capability)
			
			# Remove excluded capabilities
			context.capabilities = [
				cap for cap in context.capabilities 
				if f"{cap.category.value}/{cap.name.lower().replace(' ', '_')}" not in self.config.exclude_capabilities
			]
			
			# Validate composition
			validation = engine.validate_composition(context)
			if validation['errors']:
				raise ValueError(f"Composition validation failed: {'; '.join(validation['errors'])}")
			
			# Generate application files
			generated_files = engine.generate_application_files(context)
			generated_files.update(self._generate_ai_agent_files(ast))
			generated_files.update(self._generate_capability_files(ast))
			
			# Handle different output modes
			if self.config.template_output_mode == "models_only":
				# Return only model files for integration with existing apps
				return {k: v for k, v in generated_files.items() if 'model' in k.lower()}
			elif self.config.template_output_mode == "hybrid":
				# Combine template system with legacy entity generation
				template_files = generated_files
				legacy_files = self._generate_legacy_entities(ast)
				template_files.update(legacy_files)
				return template_files
			else:
				# Return complete application
				return generated_files
				
		except Exception as e:
			print(f"Error in composable template generation: {e}")
			print("Falling back to legacy generation...")
			return self._generate_legacy_flask_app(ast)

	def _generate_legacy_entities(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate the legacy entity files used by hybrid template mode."""

		files = {"views.py": self._generate_views(ast)}
		for entity in ast.entities:
			if entity.entity_type == EntityType.DATABASE:
				files["models.py"] = self._generate_database_models(entity)
				files["model_views.py"] = self._generate_model_views(entity)
		return files
	
	def _generate_legacy_flask_app(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Legacy Flask-AppBuilder generation method"""
		files = {}
		
		# Generate Flask-AppBuilder app.py (main application)
		app_content = self._generate_flask_app(ast)
		files["app.py"] = app_content
		
		# Generate views.py (Flask-AppBuilder views)
		views_content = self._generate_views(ast)
		files["views.py"] = views_content
		
		# Generate entity-specific files
		for entity in ast.entities:
			if entity.entity_type == EntityType.DATABASE:
				# Generate database models
				db_content = self._generate_database_models(entity)
				files["models.py"] = db_content
				# Generate ModelViews for database tables
				model_views_content = self._generate_model_views(entity)
				files["model_views.py"] = model_views_content
		
		# Generate Flask-AppBuilder configuration
		config_content = self._generate_config()
		files["config.py"] = config_content
		
		# Generate package __init__.py
		init_content = self._generate_package_init(ast)
		files["__init__.py"] = init_content
		
		# Generate requirements.txt
		requirements = self._generate_requirements()
		files["requirements.txt"] = requirements
		files.update(self._generate_ai_agent_files(ast))
		files.update(self._generate_capability_files(ast))
		
		# Generate HTML templates
		template_files = self._generate_templates(ast)
		files.update(template_files)
		
		return files

	def _generate_ai_agent_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate first-class AI agent composition runtime files."""
		agents = [entity for entity in ast.entities if isinstance(entity, AIAgentDeclaration)]
		teams = [entity for entity in ast.entities if isinstance(entity, AgentTeamDeclaration)]
		if not agents and not teams:
			return {}
		return {"ai_agents.py": self._generate_ai_agents_runtime(agents, teams)}

	def _generate_capability_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate first-class capability composition runtime files."""
		capabilities = [entity for entity in ast.entities if isinstance(entity, CapabilityDeclaration)]
		if not capabilities:
			return {}
		return {"apg_capabilities.py": self._generate_capability_runtime(capabilities)}

	def _generate_capability_runtime(self, capabilities: List[CapabilityDeclaration]) -> str:
		"""Generate a dependency-free runtime manifest for APG capabilities."""
		capability_specs = {
			capability.name: {
				"contract": capability.contract,
				"provides": capability.provides,
				"requires": capability.requires,
				"configuration": capability.configuration,
				"rules": capability.rules,
				"rule_engine": capability.rule_engine,
				"ui": capability.ui,
				"theme": capability.theme,
				"runtime": capability.runtime,
				"erp_modules": capability.erp_modules,
				"components": capability.components,
				"business_rules": capability.business_rules,
				"approvals": capability.approvals,
				"master_data": capability.master_data,
				"i18n": capability.i18n,
				"streaming": capability.streaming,
			}
			for capability in capabilities
		}
		return f'''"""
APG Capability Composition Runtime
==================================

Generated from first-class APG capability declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    contract: Dict[str, Any]
    provides: List[str]
    requires: List[str]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    rule_engine: Dict[str, Any]
    ui: Dict[str, Any]
    theme: Dict[str, Any]
    runtime: Dict[str, Any]
    erp_modules: List[str]
    components: Any
    business_rules: List[Dict[str, Any]]
    approvals: Any
    master_data: Any
    i18n: Dict[str, Any]
    streaming: Dict[str, Any]


CAPABILITY_DATA: Dict[str, Dict[str, Any]] = {capability_specs!r}
CAPABILITIES: Dict[str, CapabilitySpec] = {{
    name: CapabilitySpec(name=name, **data)
    for name, data in CAPABILITY_DATA.items()
}}


def list_capabilities() -> List[str]:
    return sorted(CAPABILITIES)


def get_capability(name: str) -> CapabilitySpec:
    return CAPABILITIES[name]


def capabilities_by_erp_module() -> Dict[str, List[CapabilitySpec]]:
    grouped: Dict[str, List[CapabilitySpec]] = {{}}
    for capability in CAPABILITIES.values():
        for module_name in capability.erp_modules:
            grouped.setdefault(module_name, []).append(capability)
    return grouped


def provided_services() -> Dict[str, List[str]]:
    services: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        for service in capability.provides:
            services.setdefault(service, []).append(capability.name)
    return services


def service_providers(service_name: str) -> List[str]:
    return sorted(provided_services().get(service_name, []))


def required_services(capability_name: str) -> List[str]:
    return list(get_capability(capability_name).requires)


def capability_dependency_graph() -> Dict[str, List[str]]:
    providers = provided_services()
    graph: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        dependencies: List[str] = []
        for service in capability.requires:
            for provider in providers.get(service, []):
                if provider != capability.name and provider not in dependencies:
                    dependencies.append(provider)
        graph[capability.name] = sorted(dependencies)
    return graph


def unresolved_required_services() -> Dict[str, List[str]]:
    providers = provided_services()
    unresolved: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        missing = [
            service for service in capability.requires
            if service not in providers and service not in CAPABILITIES
        ]
        if missing:
            unresolved[capability.name] = sorted(missing)
    return unresolved


def capability_load_order() -> Dict[str, Any]:
    graph = capability_dependency_graph()
    visited: set[str] = set()
    visiting: set[str] = set()
    order: List[str] = []
    cycles: List[List[str]] = []

    def visit(name: str, stack: List[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle_start = stack.index(name) if name in stack else 0
            cycles.append([*stack[cycle_start:], name])
            return
        visiting.add(name)
        for dependency in graph.get(name, []):
            visit(dependency, [*stack, name])
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for capability_name in sorted(CAPABILITIES):
        visit(capability_name, [])

    return {{
        "order": order,
        "cycles": cycles,
        "unresolved": unresolved_required_services(),
    }}


def validate_capability_dependencies() -> Dict[str, List[str]]:
    plan = capability_load_order()
    errors: List[str] = []
    warnings: List[str] = []
    for cycle in plan["cycles"]:
        errors.append("Capability dependency cycle: " + " -> ".join(cycle))
    for capability_name, services in plan["unresolved"].items():
        for service in services:
            warnings.append(f"{{capability_name}} requires external service {{service}}")
    return {{"errors": errors, "warnings": warnings}}


def validate_capability_contracts() -> Dict[str, Any]:
    providers = provided_services()
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        if not capability.contract:
            errors.append(f"{{capability.name}} is missing a contract")
        if not capability.provides:
            errors.append(f"{{capability.name}} does not provide any services")
        for service in capability.requires:
            if service not in providers and service not in CAPABILITIES:
                warnings.append(f"{{capability.name}} requires external service {{service}}")
        if len(set(capability.provides)) != len(capability.provides):
            errors.append(f"{{capability.name}} declares duplicate provided services")
        if len(set(capability.requires)) != len(capability.requires):
            errors.append(f"{{capability.name}} declares duplicate required services")
    return {{"errors": errors, "warnings": warnings}}


def capability_components(capability_name: str) -> Dict[str, Dict[str, Any]]:
    components = get_capability(capability_name).components
    if not isinstance(components, dict):
        return {{}}
    normalized: Dict[str, Dict[str, Any]] = {{}}
    for component_name, component_spec in components.items():
        if isinstance(component_spec, dict):
            normalized[str(component_name)] = dict(component_spec)
        else:
            normalized[str(component_name)] = {{"value": component_spec}}
    return normalized


def component_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {{}}
    for capability in CAPABILITIES.values():
        for component_name, component_spec in capability_components(capability.name).items():
            component_id = f"{{capability.name}}.{{component_name}}"
            permissions = component_spec.get("permissions", [])
            if isinstance(permissions, list):
                normalized_permissions = list(permissions)
            elif permissions:
                normalized_permissions = [str(permissions)]
            else:
                normalized_permissions = []
            catalog[component_id] = {{
                "id": component_id,
                "capability": capability.name,
                "name": component_name,
                "service": component_spec.get("capability"),
                "permissions": normalized_permissions,
                "spec": component_spec,
            }}
    return catalog


def component_permissions(capability_name: str, component_name: str) -> List[str]:
    component = component_catalog().get(f"{{capability_name}}.{{component_name}}")
    if component is None:
        return []
    return list(component["permissions"])


def component_service_bindings() -> Dict[str, List[str]]:
    bindings: Dict[str, List[str]] = {{}}
    for component_id, component in component_catalog().items():
        service = component.get("service")
        if service:
            bindings.setdefault(str(service), []).append(component_id)
    return {{
        service: sorted(component_ids)
        for service, component_ids in sorted(bindings.items())
    }}


def validate_component_contracts() -> Dict[str, List[str]]:
    provided = provided_services()
    errors: List[str] = []
    warnings: List[str] = []
    for component_id, component in component_catalog().items():
        service = component.get("service")
        if not service:
            warnings.append(f"{{component_id}} does not declare a service binding")
        elif service not in provided and service not in CAPABILITIES:
            warnings.append(f"{{component_id}} binds to external service {{service}}")
        for permission in component.get("permissions", []):
            if not permission:
                errors.append(f"{{component_id}} declares an empty permission")
    return {{"errors": errors, "warnings": warnings}}


def capability_configuration(capability_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(get_capability(capability_name).configuration or {{}})
    if overrides:
        _deep_merge(config, overrides)
    return config


def configuration_value(
    capability_name: str,
    key: str,
    default: Any = None,
    overrides: Dict[str, Any] | None = None,
) -> Any:
    return capability_configuration(capability_name, overrides).get(key, default)


def validate_capability_configuration(
    capability_name: str,
    configuration: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    config = capability_configuration(capability_name, configuration or {{}})
    schema = capability.contract.get("configuration_schema", {{}})
    required = schema.get("required", list(capability.configuration)) if isinstance(schema, dict) else list(capability.configuration)
    errors: List[str] = []
    warnings: List[str] = []
    for key in required:
        if key not in config:
            errors.append(f"{{capability.name}} missing required configuration {{key}}")
    for key in config:
        if capability.configuration and key not in capability.configuration:
            warnings.append(f"{{capability.name}} has undeclared configuration {{key}}")
    return {{"errors": errors, "warnings": warnings, "configuration": config}}


def approval_policy(capability_name: str) -> Dict[str, Any]:
    approvals = get_capability(capability_name).approvals
    if isinstance(approvals, dict):
        return {{
            "levels": int(approvals.get("levels") or 0),
            "approvers": [str(approver) for approver in approvals.get("approvers", [])],
            "thresholds": dict(approvals.get("thresholds") or {{}}),
            "segregation_of_duties": bool(approvals.get("segregation_of_duties", False)),
            "escalation": approvals.get("escalation"),
        }}
    if isinstance(approvals, list):
        return {{"levels": len(approvals), "approvers": [str(item) for item in approvals], "thresholds": {{}}, "segregation_of_duties": False, "escalation": None}}
    return {{"levels": 0, "approvers": [], "thresholds": {{}}, "segregation_of_duties": False, "escalation": None}}


def approval_plan(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    policy = approval_policy(capability_name)
    context = context or {{}}
    amount = context.get("amount")
    thresholds = policy.get("thresholds", {{}})
    levels = policy["levels"]
    if isinstance(amount, (int, float)):
        for threshold_name, threshold_value in thresholds.items():
            if isinstance(threshold_value, (int, float)) and amount >= threshold_value:
                levels = max(levels, int(str(threshold_name).split("_")[-1]) if str(threshold_name).split("_")[-1].isdigit() else levels)
    return {{
        "capability": capability_name,
        "required": levels > 0,
        "levels": levels,
        "approvers": policy["approvers"][:levels] if levels else [],
        "segregation_of_duties": policy["segregation_of_duties"],
        "escalation": policy["escalation"],
    }}


def master_data_entities(capability_name: str) -> List[str]:
    master_data = get_capability(capability_name).master_data
    if isinstance(master_data, dict):
        entities = master_data.get("entities", [])
        if isinstance(entities, list):
            return [str(entity) for entity in entities]
    if isinstance(master_data, list):
        return [str(entity) for entity in master_data]
    return []


def master_data_index() -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        for entity in master_data_entities(capability.name):
            index.setdefault(entity, []).append(capability.name)
    return {{
        entity: sorted(capability_names)
        for entity, capability_names in index.items()
    }}


def validate_master_data_contracts() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        entities = master_data_entities(capability.name)
        if not entities:
            warnings.append(f"{{capability.name}} does not declare master data entities")
        if len(set(entities)) != len(entities):
            errors.append(f"{{capability.name}} declares duplicate master data entities")
    return {{"errors": errors, "warnings": warnings}}


def capability_theme(capability_name: str, tenant_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    theme = dict(capability.theme or {{}})
    resolved = {{
        "name": theme.get("name", f"{{capability.name}}_theme"),
        "tokens": dict(theme.get("tokens") or {{}}),
        "components": dict(theme.get("components") or {{}}),
        "allow_tenant_overrides": bool(theme.get("allow_tenant_overrides", True)),
    }}
    if tenant_overrides and resolved["allow_tenant_overrides"]:
        _deep_merge(resolved, tenant_overrides)
    return resolved


def theme_token(
    capability_name: str,
    token_name: str,
    default: Any = None,
    tenant_overrides: Dict[str, Any] | None = None,
) -> Any:
    return capability_theme(capability_name, tenant_overrides)["tokens"].get(token_name, default)


def capability_languages(capability_name: str) -> List[str]:
    languages = get_capability(capability_name).i18n.get("supported_languages", [])
    if not isinstance(languages, list):
        return []
    return [str(language) for language in languages]


def resolve_language(capability_name: str, requested_language: str | None = None) -> str:
    capability = get_capability(capability_name)
    supported = capability_languages(capability_name)
    default_language = str(capability.i18n.get("default_language") or (supported[0] if supported else "en"))
    fallback_language = str(capability.i18n.get("fallback_language") or default_language)
    if requested_language and requested_language in supported:
        return requested_language
    if default_language in supported:
        return default_language
    if fallback_language in supported:
        return fallback_language
    return supported[0] if supported else fallback_language


def validate_capability_i18n() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        supported = capability_languages(capability.name)
        if not supported:
            warnings.append(f"{{capability.name}} does not declare supported languages")
            continue
        default_language = capability.i18n.get("default_language")
        fallback_language = capability.i18n.get("fallback_language")
        if default_language and default_language not in supported:
            errors.append(f"{{capability.name}} default language {{default_language}} is not supported")
        if fallback_language and fallback_language not in supported:
            errors.append(f"{{capability.name}} fallback language {{fallback_language}} is not supported")
    return {{"errors": errors, "warnings": warnings}}


def capability_streaming(capability_name: str) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    runtime_streaming = capability.runtime.get("streaming", {{}})
    stream = dict(runtime_streaming) if isinstance(runtime_streaming, dict) else {{}}
    if isinstance(capability.streaming, dict):
        _deep_merge(stream, capability.streaming)
    return {{
        "capability": capability.name,
        "processor": stream.get("processor", "bytewax"),
        "input": stream.get("input"),
        "output": stream.get("output"),
        "state": stream.get("state"),
        "window": stream.get("window"),
        "config": stream,
    }}


def streaming_processor_index() -> Dict[str, List[str]]:
    processors: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        stream = capability_streaming(capability.name)
        processor = str(stream.get("processor") or "bytewax")
        processors.setdefault(processor, []).append(capability.name)
    return {{
        processor: sorted(capability_names)
        for processor, capability_names in processors.items()
    }}


def streaming_state_index() -> Dict[str, List[str]]:
    states: Dict[str, List[str]] = {{}}
    for capability in CAPABILITIES.values():
        state = capability_streaming(capability.name).get("state")
        if state:
            states.setdefault(str(state), []).append(capability.name)
    return {{
        state: sorted(capability_names)
        for state, capability_names in states.items()
    }}


def validate_streaming_contracts() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    allowed_processors = {{"bytewax", "bytewax_streams"}}
    for capability in CAPABILITIES.values():
        stream = capability_streaming(capability.name)
        processor = str(stream.get("processor") or "")
        if processor not in allowed_processors:
            errors.append(f"{{capability.name}} uses unsupported stream processor {{processor}}")
        if not stream.get("state"):
            warnings.append(f"{{capability.name}} does not declare streaming state")
    return {{"errors": errors, "warnings": warnings}}


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def capability_rules(capability_name: str) -> List[Dict[str, Any]]:
    capability = get_capability(capability_name)
    rules: List[Dict[str, Any]] = []
    for source, source_rules in (
        ("contract", capability.rules),
        ("business", capability.business_rules),
        ("engine", capability.rule_engine.get("rules", [])),
    ):
        if not isinstance(source_rules, list):
            continue
        for index, rule in enumerate(source_rules):
            if not isinstance(rule, dict):
                continue
            normalized = dict(rule)
            normalized.setdefault("name", f"{{source}}_rule_{{index + 1}}")
            normalized.setdefault("source", source)
            normalized.setdefault("priority", 0)
            if "condition" not in normalized and "when" in normalized:
                normalized["condition"] = normalized["when"]
            if "effect" not in normalized:
                action = normalized.get("action", "allow")
                normalized["effect"] = {{
                    "decision": _decision_from_action(action),
                    "action": action,
                }}
            rules.append(normalized)
    return sorted(rules, key=lambda rule: int(rule.get("priority") or 0), reverse=True)


def evaluate_capability_rules(capability_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    matched: List[str] = []
    actions: List[Dict[str, Any]] = []
    decision = "allow"
    precedence = {{"allow": 0, "audit": 1, "warn": 1, "require_review": 2, "deny": 3}}
    for rule in capability_rules(capability_name):
        if not _matches_rule(rule, context):
            continue
        matched.append(str(rule["name"]))
        effect = dict(rule.get("effect") or {{}})
        effect.setdefault("decision", _decision_from_action(effect.get("action", rule.get("action", "allow"))))
        effect.setdefault("rule", rule["name"])
        actions.append(effect)
        candidate = str(effect.get("decision") or "allow")
        if precedence.get(candidate, 0) > precedence.get(decision, 0):
            decision = candidate
    return {{"decision": decision, "matched_rules": matched, "actions": actions, "context": context}}


def _matches_rule(rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
    condition = rule.get("condition")
    if condition is None:
        return False
    if isinstance(condition, dict):
        for key, expected in condition.items():
            if _resolve_value(str(key), context) != expected:
                return False
        return True
    if isinstance(condition, bool):
        return condition
    return _evaluate_condition(str(condition), context)


def _evaluate_condition(expression: str, context: Dict[str, Any]) -> bool:
    expression = expression.strip()
    if not expression:
        return False
    if expression.startswith("not "):
        return not bool(_resolve_value(expression[4:].strip(), context))
    for operator in ("!=", "==", ">=", "<=", ">", "<"):
        marker = f" {{operator}} "
        if marker not in expression:
            continue
        left_text, right_text = expression.split(marker, 1)
        left = _resolve_value(left_text.strip(), context)
        right = _resolve_value(right_text.strip(), context)
        if operator == "!=":
            return left != right
        if operator == "==":
            return left == right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
    return bool(_resolve_value(expression, context))


def _resolve_value(value: str, context: Dict[str, Any]) -> Any:
    value = value.strip()
    if value in context:
        return context[value]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {{"none", "null"}}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    current: Any = context
    for part in value.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return value
    return current


def _decision_from_action(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("decision", "allow"))
    action_text = str(action)
    if action_text in {{"allow", "deny", "require_review", "warn", "audit"}}:
        return action_text
    return "allow"


def capability_screens(capability_name: str) -> List[Dict[str, Any]]:
    capability = get_capability(capability_name)
    routes = capability.ui.get("routes", [])
    if not isinstance(routes, list):
        return []
    screens: List[Dict[str, Any]] = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            continue
        name = str(route.get("name") or route.get("component") or f"screen_{{index + 1}}")
        component = route.get("component", name)
        screens.append({{
            "id": f"{{capability.name}}.{{name}}",
            "capability": capability.name,
            "name": name,
            "path": route.get("path", ""),
            "component": component,
            "permission": route.get("permission"),
            "nav_group": route.get("nav_group"),
            "shell": capability.ui.get("shell"),
            "theme": capability.theme.get("name"),
        }})
    return screens


def ui_route_index() -> Dict[str, Dict[str, Any]]:
    routes: Dict[str, Dict[str, Any]] = {{}}
    for capability in CAPABILITIES.values():
        for screen in capability_screens(capability.name):
            path = screen.get("path")
            if path:
                routes[str(path)] = screen
    return routes


def composition_graph() -> Dict[str, List[Dict[str, Any]]]:
    nodes: Dict[str, Dict[str, Any]] = {{}}
    edges: List[Dict[str, Any]] = []

    def node(node_id: str, kind: str, **attrs: Any) -> None:
        nodes[node_id] = {{"id": node_id, "kind": kind, **attrs}}

    def edge(source: str, target: str, relation: str) -> None:
        edges.append({{"source": source, "target": target, "relation": relation}})

    for capability in CAPABILITIES.values():
        cap_id = f"capability:{{capability.name}}"
        node(cap_id, "capability", name=capability.name)

        for service in capability.provides:
            service_id = f"service:{{service}}"
            node(service_id, "service", name=service)
            edge(cap_id, service_id, "provides")

        for service in capability.requires:
            service_id = f"service:{{service}}"
            node(service_id, "service", name=service)
            edge(cap_id, service_id, "requires")

        for module_name in capability.erp_modules:
            module_id = f"erp_module:{{module_name}}"
            node(module_id, "erp_module", name=module_name)
            edge(cap_id, module_id, "belongs_to")

        theme_name = capability.theme.get("name")
        if theme_name:
            theme_id = f"theme:{{theme_name}}"
            node(theme_id, "theme", name=theme_name)
            edge(cap_id, theme_id, "uses_theme")

        for screen in capability_screens(capability.name):
            screen_id = f"screen:{{screen['id']}}"
            node(screen_id, "screen", **screen)
            edge(cap_id, screen_id, "has_screen")
            component = screen.get("component")
            if component:
                component_id = f"component:{{component}}"
                node(component_id, "component", name=str(component))
                edge(screen_id, component_id, "renders")

        if isinstance(capability.components, dict):
            for component_name, component_spec in capability_components(capability.name).items():
                component_id = f"component:{{component_name}}"
                node(component_id, "component", name=str(component_name), spec=component_spec)
                edge(cap_id, component_id, "has_component")
                for permission in component_permissions(capability.name, component_name):
                    permission_id = f"permission:{{permission}}"
                    node(permission_id, "permission", name=str(permission))
                    edge(component_id, permission_id, "requires_permission")
                if component_spec.get("capability"):
                    service_id = f"service:{{component_spec['capability']}}"
                    node(service_id, "service", name=str(component_spec["capability"]))
                    edge(component_id, service_id, "binds_to")

        stream = capability_streaming(capability.name)
        processor = stream.get("processor")
        if processor:
            processor_id = f"stream_processor:{{processor}}"
            node(processor_id, "stream_processor", name=str(processor))
            edge(cap_id, processor_id, "streams_with")
        state = stream.get("state")
        if state:
            state_id = f"stream_state:{{state}}"
            node(state_id, "stream_state", name=str(state))
            edge(cap_id, state_id, "stores_stream_state")

    return {{"nodes": sorted(nodes.values(), key=lambda item: item["id"]), "edges": edges}}
'''

	def _generate_ai_agents_runtime(
		self,
		agents: List[AIAgentDeclaration],
		teams: List[AgentTeamDeclaration],
	) -> str:
		"""Generate a dependency-free runtime manifest for AI agent composition."""
		agent_specs = {
			agent.name: {
				"role": agent.role,
				"model": agent.model,
				"runtime": agent.runtime,
				"system": agent.system_prompt,
				"capabilities": agent.capabilities,
				"tools": agent.tools,
				"memory": (
					{"kind": agent.memory.kind, "name": agent.memory.name}
					if agent.memory else None
				),
				"inputs": agent.inputs,
				"outputs": agent.outputs,
				"handoffs": [
					{"source": edge.source, "target": edge.target, "condition": edge.condition}
					for edge in agent.handoffs
				],
				"configuration": agent.configuration,
				"rules": agent.rules,
				"ui": agent.ui,
				"theme": agent.theme,
			}
			for agent in agents
		}
		team_specs = {
			team.name: {
				"agents": team.agents,
				"capabilities": team.capabilities,
				"flow": [
					{"source": edge.source, "target": edge.target, "condition": edge.condition}
					for edge in team.flow
				],
				"policy": team.policy,
				"configuration": team.configuration,
				"rules": team.rules,
				"ui": team.ui,
				"theme": team.theme,
			}
			for team in teams
		}
		runtime_catalog = self._default_agent_runtime_catalog()
		runtime_aliases = {
			alias: name
			for name, spec in runtime_catalog.items()
			for alias in spec.get("aliases", [])
		}
		return f'''"""
AI Agent Composition Runtime
============================

Generated from first-class APG AI agent declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AIAgentSpec:
    name: str
    role: Optional[str]
    model: Optional[str]
    runtime: Optional[str]
    system: Optional[str]
    capabilities: List[str]
    tools: List[str]
    memory: Optional[Dict[str, Optional[str]]]
    inputs: List[str]
    outputs: List[str]
    handoffs: List[Dict[str, str]]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    ui: Dict[str, Any]
    theme: Dict[str, Any]


@dataclass(frozen=True)
class AgentTeamSpec:
    name: str
    agents: List[str]
    capabilities: List[str]
    flow: List[Dict[str, str]]
    policy: Dict[str, Any]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    ui: Dict[str, Any]
    theme: Dict[str, Any]


AI_AGENT_DATA: Dict[str, Dict[str, Any]] = {agent_specs!r}
AI_TEAM_DATA: Dict[str, Dict[str, Any]] = {team_specs!r}
AI_AGENT_RUNTIME_DATA: Dict[str, Dict[str, Any]] = {runtime_catalog!r}
AI_AGENT_RUNTIME_ALIASES: Dict[str, str] = {runtime_aliases!r}


AI_AGENTS: Dict[str, AIAgentSpec] = {{
    name: AIAgentSpec(name=name, **data)
    for name, data in AI_AGENT_DATA.items()
}}
AI_AGENT_TEAMS: Dict[str, AgentTeamSpec] = {{
    name: AgentTeamSpec(name=name, **data)
    for name, data in AI_TEAM_DATA.items()
}}


def get_agent(name: str) -> AIAgentSpec:
    return AI_AGENTS[name]


def get_team(name: str) -> AgentTeamSpec:
    return AI_AGENT_TEAMS[name]


def list_agent_runtimes(include_aliases: bool = False) -> List[str]:
    names = set(AI_AGENT_RUNTIME_DATA)
    if include_aliases:
        names.update(AI_AGENT_RUNTIME_ALIASES)
    return sorted(names)


def canonical_runtime(name: Optional[str]) -> str:
    runtime = name or "local"
    if runtime in AI_AGENT_RUNTIME_DATA:
        return runtime
    if runtime in AI_AGENT_RUNTIME_ALIASES:
        return AI_AGENT_RUNTIME_ALIASES[runtime]
    raise KeyError(f"Unknown AI agent runtime: {{runtime}}")


def describe_agent_runtimes() -> Dict[str, Dict[str, Any]]:
    return {{
        name: dict(spec)
        for name, spec in AI_AGENT_RUNTIME_DATA.items()
    }}


def agents_by_runtime() -> Dict[str, List[AIAgentSpec]]:
    grouped: Dict[str, List[AIAgentSpec]] = {{}}
    for agent in AI_AGENTS.values():
        runtime = canonical_runtime(agent.runtime)
        grouped.setdefault(runtime, []).append(agent)
    return grouped


def validate_agent_runtimes(available_runtimes: Optional[List[str]] = None) -> Dict[str, Any]:
    allowed = set(available_runtimes or list_agent_runtimes(include_aliases=True))
    errors: List[str] = []
    validated: List[str] = []
    for agent in AI_AGENTS.values():
        runtime = agent.runtime or "local"
        try:
            canonical = canonical_runtime(runtime)
        except KeyError:
            errors.append(f"{{agent.name}} references unknown runtime {{runtime}}")
            continue
        if runtime not in allowed and canonical not in allowed:
            errors.append(f"{{agent.name}} references unavailable runtime {{runtime}}")
            continue
        validated.append(agent.name)
    return {{"errors": errors, "validated_agents": sorted(validated)}}


def describe_team(name: str) -> Dict[str, Any]:
    team = get_team(name)
    return {{
        "name": team.name,
        "agents": [AI_AGENTS[agent] for agent in team.agents],
        "capabilities": team.capabilities,
        "flow": team.flow,
        "policy": team.policy,
    }}
'''

	def _default_agent_runtime_catalog(self) -> Dict[str, Dict[str, Any]]:
		"""Return dependency-free metadata for generated AI agent manifests."""
		return {
			"local": {
				"kind": "local",
				"aliases": ["offline", "test"],
				"supports_workspace": False,
				"requires_token": False,
				"family": "deterministic",
			},
			"codex": {
				"kind": "cli",
				"aliases": ["codex_cli", "openai_codex"],
				"supports_workspace": True,
				"requires_token": False,
				"family": "coding_agent",
			},
			"claude_code": {
				"kind": "cli",
				"aliases": ["claude", "claude-code"],
				"supports_workspace": True,
				"requires_token": False,
				"family": "coding_agent",
			},
			"opencode": {
				"kind": "cli",
				"aliases": ["open_code"],
				"supports_workspace": True,
				"requires_token": False,
				"family": "coding_agent",
			},
			"openai": {
				"kind": "http",
				"aliases": ["openai_chat"],
				"supports_workspace": False,
				"requires_token": True,
				"family": "chat_agent",
			},
			"ollama": {
				"kind": "http",
				"aliases": ["local_llm"],
				"supports_workspace": False,
				"requires_token": False,
				"family": "local_model",
			},
			"pi": {
				"kind": "http",
				"aliases": ["inflection_pi"],
				"supports_workspace": False,
				"requires_token": True,
				"family": "chat_agent",
			},
		}
	
	def _generate_module(self, module: ModuleDeclaration) -> str:
		"""Generate the main module Python file"""
		self.output.clear()
		self.imports.clear()
		self.indent_level = 0
		
		# Add module docstring
		self._add_module_docstring(module)
		
		# Add imports
		self._add_standard_imports()
		
		# Generate entities
		for entity in module.entities:
			self._generate_entity(entity)
		
		# Add main execution block
		self._add_main_block(module)
		
		# Combine imports and code
		import_block = self._format_imports()
		code_block = '\n'.join(self.output)
		
		return f"{import_block}\n\n{code_block}"
	
	def _add_module_docstring(self, module: ModuleDeclaration):
		"""Add module-level docstring"""
		self._add_line('"""')
		self._add_line(f"{module.name} - Generated APG Module")
		self._add_line("=" * (len(module.name) + 25))
		self._add_line("")
		if module.description:
			self._add_line(f"{module.description}")
			self._add_line("")
		self._add_line(f"Version: {module.version}")
		if module.author:
			self._add_line(f"Author: {module.author}")
		if module.license:
			self._add_line(f"License: {module.license}")
		self._add_line("")
		self._add_line("This module was automatically generated from APG source code.")
		self._add_line('"""')
		self._add_line("")
	
	def _add_standard_imports(self):
		"""Add standard Python imports needed for Flask-AppBuilder APG runtime"""
		self.imports.add("from __future__ import annotations")
		self.imports.add("from typing import Any, Dict, List, Optional, Union")
		self.imports.add("from dataclasses import dataclass, field")
		self.imports.add("import asyncio")
		self.imports.add("import json")
		self.imports.add("import logging")
		self.imports.add("from datetime import datetime")
		
		# Flask-AppBuilder imports
		self.imports.add("from flask import Flask, request, jsonify")
		self.imports.add("from flask_appbuilder import AppBuilder, BaseView, ModelView, expose")
		self.imports.add("from flask_appbuilder.models.sqla.interface import SQLAInterface")
		self.imports.add("from flask_appbuilder.security.decorators import has_access")
		self.imports.add("from flask_sqlalchemy import SQLAlchemy")
		self.imports.add("from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey")
		self.imports.add("from sqlalchemy.orm import relationship")
	
	def _generate_entity(self, entity: EntityDeclaration):
		"""Generate Python code for an APG entity"""
		self.current_entity = entity
		
		if entity.entity_type == EntityType.AGENT:
			self._generate_agent(entity)
		elif entity.entity_type == EntityType.DIGITAL_TWIN:
			self._generate_digital_twin(entity)
		elif entity.entity_type == EntityType.WORKFLOW:
			self._generate_workflow(entity)
		elif entity.entity_type == EntityType.DATABASE:
			self._generate_database(entity)
		else:
			self._generate_generic_entity(entity)
		
		self.current_entity = None
	
	def _generate_agent(self, entity: EntityDeclaration):
		"""Generate Flask-AppBuilder view for an Agent entity with full runtime implementation"""
		# Generate Agent runtime class first
		self._add_line("")
		self._add_line(f"class {entity.name}Runtime:")
		self._add_line(f'    """Runtime implementation for {entity.name} agent"""')
		self._add_line("")
		
		self._indent()
		
		# Initialize agent with properties
		self._add_line("def __init__(self):")
		for prop in entity.properties:
			default_val = self._generate_expression(prop.default_value) if prop.default_value else self._get_default_value_for_type(prop.type_annotation)
			self._add_line(f"    self.{prop.name} = {default_val}")
		
		self._add_line("    self._running = False")
		self._add_line("    self._logger = logging.getLogger(f'{self.__class__.__name__}')")
		self._add_line("")
		
		# Generate agent methods with actual implementations
		for method in entity.methods:
			self._generate_agent_runtime_method(method)
		
		# Add lifecycle methods
		self._add_line("def start(self):")
		self._add_line("    \"\"\"Start the agent\"\"\"")
		self._add_line("    if not self._running:")
		self._add_line("        self._running = True")
		self._add_line("        self._logger.info(f'Agent {self.__class__.__name__} started')")
		self._add_line("        return True")
		self._add_line("    return False")
		self._add_line("")
		
		self._add_line("def stop(self):")
		self._add_line("    \"\"\"Stop the agent\"\"\"")
		self._add_line("    if self._running:")
		self._add_line("        self._running = False")
		self._add_line("        self._logger.info(f'Agent {self.__class__.__name__} stopped')")
		self._add_line("        return True")
		self._add_line("    return False")
		self._add_line("")
		
		self._add_line("def is_running(self):")
		self._add_line("    \"\"\"Check if agent is running\"\"\"")
		self._add_line("    return self._running")
		self._add_line("")
		
		self._add_line("def get_status(self):")
		self._add_line("    \"\"\"Get agent status information\"\"\"")
		self._add_line("    return {")
		self._add_line("        'name': self.__class__.__name__,")
		self._add_line("        'running': self._running,")
		for prop in entity.properties:
			self._add_line(f"        '{prop.name}': self.{prop.name},")
		self._add_line("        'timestamp': datetime.now().isoformat()")
		self._add_line("    }")
		
		self._dedent()
		
		# Create global agent instance
		self._add_line("")
		self._add_line(f"{entity.name.lower()}_instance = {entity.name}Runtime()")
		
		# Generate Flask-AppBuilder View
		self._add_line("")
		self._add_line(f"class {entity.name}View(BaseView):")
		self._add_line(f'    """Flask-AppBuilder view for {entity.name} agent"""')
		self._add_line("")
		self._add_line("    default_view = 'agent_dashboard'")
		self._add_line("")
		
		self._indent()
		
		# Generate agent dashboard with real data
		self._add_line("@expose('/dashboard/')")
		self._add_line("@has_access")  
		self._add_line("def agent_dashboard(self):")
		self._add_line("    \"\"\"Agent dashboard view with live data\"\"\"")
		self._add_line(f"    agent = {entity.name.lower()}_instance")
		self._add_line("    status = agent.get_status()")
		self._add_line("    return self.render_template('agent_dashboard.html',")
		self._add_line(f"                                agent_name='{entity.name}',")
		self._add_line("                                agent_status=status,")
		self._add_line("                                agent_running=agent.is_running())")
		self._add_line("")
		
		# Generate functional API endpoints
		self._add_line("@expose('/start/', methods=['POST'])")
		self._add_line("@has_access")
		self._add_line("def start_agent(self):")
		self._add_line("    \"\"\"Start the agent\"\"\"")
		self._add_line("    try:")
		self._add_line(f"        agent = {entity.name.lower()}_instance")
		self._add_line("        success = agent.start()")
		self._add_line("        if success:")
		self._add_line("            return jsonify({'status': 'success', 'message': 'Agent started successfully'})")
		self._add_line("        else:")
		self._add_line("            return jsonify({'status': 'warning', 'message': 'Agent was already running'})")
		self._add_line("    except Exception as e:")
		self._add_line("        return jsonify({'status': 'error', 'message': str(e)})")
		self._add_line("")
		
		self._add_line("@expose('/stop/', methods=['POST'])")
		self._add_line("@has_access")
		self._add_line("def stop_agent(self):")
		self._add_line("    \"\"\"Stop the agent\"\"\"")
		self._add_line("    try:")
		self._add_line(f"        agent = {entity.name.lower()}_instance")
		self._add_line("        success = agent.stop()")
		self._add_line("        if success:")
		self._add_line("            return jsonify({'status': 'success', 'message': 'Agent stopped successfully'})")
		self._add_line("        else:")
		self._add_line("            return jsonify({'status': 'warning', 'message': 'Agent was already stopped'})")
		self._add_line("    except Exception as e:")
		self._add_line("        return jsonify({'status': 'error', 'message': str(e)})")
		self._add_line("")
		
		self._add_line("@expose('/status/', methods=['GET'])")
		self._add_line("@has_access")
		self._add_line("def get_agent_status(self):")
		self._add_line("    \"\"\"Get agent status\"\"\"")
		self._add_line("    try:")
		self._add_line(f"        agent = {entity.name.lower()}_instance")
		self._add_line("        status = agent.get_status()")
		self._add_line("        return jsonify({'status': 'success', 'data': status})")
		self._add_line("    except Exception as e:")
		self._add_line("        return jsonify({'status': 'error', 'message': str(e)})")
		self._add_line("")
		
		# Generate API endpoints for agent methods
		for method in entity.methods:
			self._generate_agent_api_method(method, entity)
		
		self._dedent()
		self.generated_classes.add(f"{entity.name}View")
	
	def _generate_digital_twin(self, entity: EntityDeclaration):
		"""Generate Python code for a Digital Twin entity"""
		self._add_line("")
		self._add_line("@dataclass")
		self._add_line(f"class {entity.name}:")
		self._add_line(f'    """APG Digital Twin: {entity.name}"""')
		self._add_line("")
		
		self._indent()
		
		# Generate properties
		for prop in entity.properties:
			self._generate_property(prop)
		
		# Add digital twin state management
		self._add_line("")
		self._add_line("_state_history: List[Dict[str, Any]] = field(default_factory=list)")
		self._add_line("_last_updated: Optional[datetime] = None")
		self._add_line("")
		
		# Generate methods
		for method in entity.methods:
			self._generate_method(method, is_digital_twin=True)
		
		# Add default digital twin methods
		self._add_default_digital_twin_methods()
		
		self._dedent()
		self.generated_classes.add(entity.name)
	
	def _generate_workflow(self, entity: EntityDeclaration):
		"""Generate Python code for a Workflow entity"""
		self._add_line("")
		self._add_line("@dataclass")
		self._add_line(f"class {entity.name}:")
		self._add_line(f'    """APG Workflow: {entity.name}"""')
		self._add_line("")
		
		self._indent()
		
		# Generate properties
		for prop in entity.properties:
			self._generate_property(prop)
		
		# Add workflow state
		self._add_line("")
		self._add_line("_current_step: int = 0")
		self._add_line("_status: str = 'pending'")
		self._add_line("_step_results: Dict[str, Any] = field(default_factory=dict)")
		self._add_line("")
		
		# Generate methods
		for method in entity.methods:
			self._generate_method(method, is_workflow=True)
		
		# Add default workflow methods
		self._add_default_workflow_methods()
		
		self._dedent()
		self.generated_classes.add(entity.name)
	
	def _generate_database(self, entity: EntityDeclaration):
		"""Generate Flask-AppBuilder models and views for Database entity"""
		# For database entities, we generate both the SQLAlchemy models
		# and Flask-AppBuilder ModelViews in separate files
		
		# This method will be called to register the database configuration
		self._add_line("")
		self._add_line(f"# Database configuration for {entity.name}")
		self._add_line(f"# Models and views will be generated in separate files")
		self._add_line("")
		
		self.generated_classes.add(entity.name)
	
	def _generate_generic_entity(self, entity: EntityDeclaration):
		"""Generate Python code for generic entities"""
		self._add_line("")
		self._add_line("@dataclass")
		self._add_line(f"class {entity.name}:")
		self._add_line(f'    """APG Entity: {entity.name}"""')
		self._add_line("")
		
		self._indent()
		
		# Generate properties
		for prop in entity.properties:
			self._generate_property(prop)
		
		# Generate methods
		for method in entity.methods:
			self._generate_method(method)
		
		self._dedent()
		self.generated_classes.add(entity.name)
	
	def _generate_property(self, prop: PropertyDeclaration):
		"""Generate Python property declaration"""
		python_type = self._apg_type_to_python(prop.type_annotation)
		
		if prop.default_value:
			default = self._generate_expression(prop.default_value)
			self._add_line(f"{prop.name}: {python_type} = {default}")
		else:
			if prop.type_annotation.is_optional:
				self._add_line(f"{prop.name}: {python_type} = None")
			else:
				# Required field without default
				self._add_line(f"{prop.name}: {python_type}")
	
	def _generate_method(self, method: MethodDeclaration, **kwargs):
		"""Generate Python method declaration"""
		self._add_line("")
		
		# Generate method signature
		is_async = method.is_async or kwargs.get('is_agent', False)
		async_prefix = "async " if is_async else ""
		
		# Generate parameters
		params = ["self"]
		for param in method.parameters:
			param_type = self._apg_type_to_python(param.type_annotation)
			if param.default_value:
				default = self._generate_expression(param.default_value)
				params.append(f"{param.name}: {param_type} = {default}")
			else:
				params.append(f"{param.name}: {param_type}")
		
		# Generate return type
		return_type = ""
		if method.return_type:
			return_type = f" -> {self._apg_type_to_python(method.return_type)}"
		
		signature = f"{async_prefix}def {method.name}({', '.join(params)}){return_type}:"
		self._add_line(signature)
		
		# Generate method body
		self._indent()
		self._add_line(f'"""Method: {method.name}"""')
		
		if method.body:
			self._generate_statement(method.body)
		else:
			if is_async:
				self._add_line("await asyncio.sleep(0)")
			
			if method.return_type and method.return_type.type_name != "void":
				default_return = self._get_default_return_value(method.return_type)
				self._add_line(f"return {default_return}")
			else:
				self._add_line("return None")
		
		self._dedent()
	
	def _generate_statement(self, stmt: Statement):
		"""Generate Python code for a statement"""
		if isinstance(stmt, BlockStatement):
			if stmt.statements:
				for s in stmt.statements:
					self._generate_statement(s)
			else:
				self._add_line("None")
		
		elif isinstance(stmt, AssignmentStatement):
			value = self._generate_expression(stmt.value)
			target = stmt.target
			if not target.startswith("self.") and "." not in target and "[" not in target:
				target = f"self.{target}"
			operator = stmt.operator if stmt.operator in {"=", "+=", "-=", "*=", "/=", "%="} else "="
			self._add_line(f"{target} {operator} {value}")

		elif isinstance(stmt, ExpressionStatement):
			self._add_line(self._generate_expression(stmt.expression))
		
		elif isinstance(stmt, ReturnStatement):
			if stmt.value:
				value = self._generate_expression(stmt.value)
				self._add_line(f"return {value}")
			else:
				self._add_line("return")
		
		elif hasattr(stmt, 'condition') and hasattr(stmt, 'then_branch'):  # IfStatement
			condition = self._generate_expression(stmt.condition)
			self._add_line(f"if {condition}:")
			self._indent()
			self._generate_statement(stmt.then_branch)
			self._dedent()
			
			if hasattr(stmt, 'else_branch') and stmt.else_branch:
				self._add_line("else:")
				self._indent()
				self._generate_statement(stmt.else_branch)
				self._dedent()
		
		elif hasattr(stmt, 'variable') and hasattr(stmt, 'iterable'):  # ForStatement
			variable = stmt.variable
			iterable = self._generate_expression(stmt.iterable)
			self._add_line(f"for {variable} in {iterable}:")
			self._indent()
			self._generate_statement(stmt.body)
			self._dedent()
		
		elif hasattr(stmt, 'condition') and hasattr(stmt, 'body'):  # WhileStatement
			condition = self._generate_expression(stmt.condition)
			self._add_line(f"while {condition}:")
			self._indent()
			self._generate_statement(stmt.body)
			self._dedent()
		
		else:
			self._add_line("if not hasattr(self, '_unhandled_statements'):")
			self._indent()
			self._add_line("self._unhandled_statements = []")
			self._dedent()
			self._add_line(f"self._unhandled_statements.append('{type(stmt).__name__}')")
	
	def _generate_expression(self, expr: Expression) -> str:
		"""Generate Python code for an expression"""
		if isinstance(expr, str):
			return expr

		if isinstance(expr, LiteralExpression):
			if expr.literal_type == "string":
				return f'"{expr.value}"'
			elif expr.literal_type == "boolean":
				return "True" if expr.value else "False"
			elif expr.literal_type == "null":
				return "None"
			else:
				return str(expr.value)
		
		elif isinstance(expr, IdentifierExpression):
			# Add self prefix for instance variables
			if expr.name in ['name', 'status', 'counter', 'message']:  # Common property names
				return f"self.{expr.name}"
			return expr.name
		
		elif isinstance(expr, BinaryExpression):
			left = self._generate_expression(expr.left)
			right = self._generate_expression(expr.right)
			
			# Handle APG operators
			operator_map = {
				'==': '==',
				'!=': '!=',
				'<': '<',
				'>': '>',
				'<=': '<=',
				'>=': '>=',
				'+': '+',
				'-': '-',
				'*': '*',
				'/': '/',
				'%': '%',
				'&&': 'and',
				'||': 'or',
				'!': 'not',
				'in': 'in'
			}
			
			python_op = operator_map.get(expr.operator, expr.operator)
			return f"({left} {python_op} {right})"
		
		elif isinstance(expr, UnaryExpression):
			operand = self._generate_expression(expr.operand)
			if expr.operator == '!':
				return f"not {operand}"
			else:
				return f"{expr.operator}{operand}"
		
		elif isinstance(expr, CallExpression):
			func = self._generate_expression(expr.function)
			args = [self._generate_expression(arg) for arg in expr.arguments]
			
			# Handle built-in functions
			builtin_map = {
				'len': 'len',
				'str': 'str',
				'int': 'int',
				'float': 'float',
				'bool': 'bool',
				'now': 'datetime.now().isoformat',
				'log': 'print'  # Simple logging
			}
			
			if func in builtin_map:
				func = builtin_map[func]
			
			return f"{func}({', '.join(args)})"
		
		elif isinstance(expr, MemberExpression):
			obj = self._generate_expression(expr.object)
			return f"{obj}.{expr.property}"
		
		elif isinstance(expr, IndexExpression):
			obj = self._generate_expression(expr.object)
			index = self._generate_expression(expr.index)
			return f"{obj}[{index}]"
		
		elif isinstance(expr, ListExpression):
			elements = [self._generate_expression(elem) for elem in expr.elements]
			return f"[{', '.join(elements)}]"
		
		elif isinstance(expr, DictExpression):
			pairs = []
			for key_expr, value_expr in expr.pairs:
				key = self._generate_expression(key_expr)
				value = self._generate_expression(value_expr)
				pairs.append(f"{key}: {value}")
			return f"{{{', '.join(pairs)}}}"
		
		else:
			return "None"
	
	def _add_default_agent_start(self):
		"""Add default start method for agents"""
		self._add_line("")
		self._add_line("async def start(self) -> None:")
		self._add_line("    \"\"\"Start the agent\"\"\"")
		self._add_line("    self._logger.info(f'Starting agent {self.__class__.__name__}')")
		self._add_line("    self._running = True")
		self._add_line("    self._started_at = datetime.now().isoformat()")
		self._add_line("    if not hasattr(self, '_lifecycle_events'):")
		self._add_line("        self._lifecycle_events = []")
		self._add_line("    self._lifecycle_events.append({'event': 'started', 'timestamp': self._started_at})")
	
	def _add_default_agent_stop(self):
		"""Add default stop method for agents"""
		self._add_line("")
		self._add_line("async def stop(self) -> None:")
		self._add_line("    \"\"\"Stop the agent\"\"\"")
		self._add_line("    self._logger.info(f'Stopping agent {self.__class__.__name__}')")
		self._add_line("    self._running = False")
		self._add_line("    self._stopped_at = datetime.now().isoformat()")
		self._add_line("    if not hasattr(self, '_lifecycle_events'):")
		self._add_line("        self._lifecycle_events = []")
		self._add_line("    self._lifecycle_events.append({'event': 'stopped', 'timestamp': self._stopped_at})")
	
	def _add_default_digital_twin_methods(self):
		"""Add default methods for digital twins"""
		self._add_line("")
		self._add_line("def update_state(self, new_state: Dict[str, Any]) -> None:")
		self._add_line("    \"\"\"Update the digital twin state\"\"\"")
		self._add_line("    self._state_history.append({")
		self._add_line("        'timestamp': datetime.now(),")
		self._add_line("        'state': new_state")
		self._add_line("    })")
		self._add_line("    self._last_updated = datetime.now()")
		self._add_line("    for key, value in new_state.items():")
		self._add_line("        if hasattr(self, key):")
		self._add_line("            setattr(self, key, value)")
		self._add_line("")
		self._add_line("def get_state_history(self) -> List[Dict[str, Any]]:")
		self._add_line("    \"\"\"Get the state change history\"\"\"")
		self._add_line("    return self._state_history.copy()")
	
	def _add_default_workflow_methods(self):
		"""Add default methods for workflows"""
		self._add_line("")
		self._add_line("async def execute(self) -> Dict[str, Any]:")
		self._add_line("    \"\"\"Execute the workflow\"\"\"")
		self._add_line("    self._status = 'running'")
		self._add_line("    try:")
		self._add_line("        self._step_results['started_at'] = datetime.now().isoformat()")
		self._add_line("        for index, step in enumerate(getattr(self, 'steps', [])):")
		self._add_line("            self._current_step = index + 1")
		self._add_line("            self._step_results[str(step)] = {")
		self._add_line("                'index': index,")
		self._add_line("                'status': 'completed',")
		self._add_line("                'completed_at': datetime.now().isoformat()")
		self._add_line("            }")
		self._add_line("        self._status = 'completed'")
		self._add_line("        self._step_results['completed_at'] = datetime.now().isoformat()")
		self._add_line("        return {'status': 'success', 'results': dict(self._step_results)}")
		self._add_line("    except Exception as e:")
		self._add_line("        self._status = 'failed'")
		self._add_line("        return {'status': 'error', 'error': str(e)}")
	
	def _generate_database_models(self, database: EntityDeclaration) -> str:
		"""Generate SQLAlchemy models for database entities"""
		if not isinstance(database, DatabaseDeclaration):
			return ""
		
		self.output.clear()
		self.imports.clear()
		
		# Add SQLAlchemy imports
		self.imports.add("from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text")
		self.imports.add("from sqlalchemy.ext.declarative import declarative_base")
		self.imports.add("from sqlalchemy.orm import relationship")
		self.imports.add("from datetime import datetime")
		
		# Generate base
		self._add_line("Base = declarative_base()")
		self._add_line("")
		
		# Generate table models from schemas
		for schema in database.schemas:
			for table in schema.tables:
				self._generate_table_model(table)
		
		import_block = self._format_imports()
		code_block = '\n'.join(self.output)
		
		return f"{import_block}\n\n{code_block}"
	
	def _generate_table_model(self, table: TableDeclaration):
		"""Generate SQLAlchemy model for a table"""
		class_name = self._to_pascal_case(table.name)
		
		self._add_line(f"class {class_name}(Base):")
		self._add_line(f"    __tablename__ = '{table.name}'")
		self._add_line("")
		
		self._indent()
		
		# Generate columns
		for column in table.columns:
			self._generate_column_definition(column)
		
		self._add_line("")
		self._add_line("def __repr__(self):")
		self._add_line(f"    return f'<{class_name}({{self.id}})'")
		
		self._dedent()
		self._add_line("")
	
	def _generate_column_definition(self, column):
		"""Generate SQLAlchemy column definition"""
		# Map APG types to SQLAlchemy types
		type_map = {
			'int': 'Integer',
			'str': 'String(255)',
			'float': 'Float',
			'bool': 'Boolean',
			'text': 'Text',
			'datetime': 'DateTime'
		}
		
		sql_type = type_map.get(column.data_type, 'String(255)')
		
		constraints = []
		if column.is_primary_key:
			constraints.append("primary_key=True")
		if not column.is_nullable:
			constraints.append("nullable=False")
		if column.default_value:
			constraints.append(f"default={repr(column.default_value)}")
		
		constraint_str = f", {', '.join(constraints)}" if constraints else ""
		self._add_line(f"{column.name} = Column({sql_type}{constraint_str})")
	
	def _generate_package_init(self, module: ModuleDeclaration) -> str:
		"""Generate package __init__.py file"""
		lines = [
			'"""',
			f'{module.name} - APG Generated Package',
			'=' * (len(module.name) + 25),
			'',
			f'Version: {module.version}',
		]
		
		if module.description:
			lines.extend(['', module.description])
		
		lines.extend([
			'',
			'This package was automatically generated from APG source code.',
			'"""',
			'',
			f'__version__ = "{module.version}"',
			'',
			'# Import generated entities'
		])
		
		for entity in module.entities:
			lines.append(f"from .{module.name} import {entity.name}")
		
		lines.extend([
			'',
			'__all__ = [',
		])
		
		for entity in module.entities:
			lines.append(f'    "{entity.name}",')
		
		lines.append(']')
		
		return '\n'.join(lines)
	
	def _generate_requirements(self) -> str:
		"""Generate requirements.txt file for Flask-AppBuilder APG application"""
		requirements = [
			"# Flask-AppBuilder APG Application Requirements",
			"Flask-AppBuilder>=4.3.0",
			"Flask>=2.3.0",
			"Flask-SQLAlchemy>=3.0.0",
			"SQLAlchemy>=2.0.0",
			"psycopg2-binary>=2.9.0  # PostgreSQL support",
			"pymysql>=1.0.0  # MySQL support",
			"celery>=5.3.0  # Background tasks",
			"redis>=4.5.0  # Celery broker",
			"Pillow>=10.0.0  # Image handling",
			"email-validator>=2.0.0",
			"python-dateutil>=2.8.0",
			"# Optional APG extensions",
			"# pandas>=2.0.0  # Data analysis",
			"# numpy>=1.24.0  # Numerical computing",
			"# scikit-learn>=1.3.0  # Machine learning",
			"# requests>=2.31.0  # HTTP requests",
		]
		return '\n'.join(requirements)
	
	def _generate_flask_app(self, module: ModuleDeclaration) -> str:
		"""Generate Flask-AppBuilder app.py main application file"""
		self.output.clear()
		self.imports.clear()
		
		# Flask-AppBuilder app imports
		self.imports.add("import logging")
		self.imports.add("from flask import Flask")
		self.imports.add("from flask_appbuilder import AppBuilder, SQLA")
		self.imports.add("from flask_appbuilder.menu import Menu")
		
		# Add module docstring
		self._add_line('"""')
		self._add_line(f"{module.name} - Flask-AppBuilder APG Application")
		self._add_line("=" * (len(module.name) + 35))
		self._add_line("")
		if module.description:
			self._add_line(f"{module.description}")
			self._add_line("")
		self._add_line("This Flask-AppBuilder application was generated from APG source.")
		self._add_line('"""')
		self._add_line("")
		
		# Initialize Flask app
		self._add_line("logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')")
		self._add_line("logging.getLogger().setLevel(logging.DEBUG)")
		self._add_line("")
		self._add_line("app = Flask(__name__)")
		self._add_line("app.config.from_object('config')")
		self._add_line("db = SQLA(app)")
		self._add_line("appbuilder = AppBuilder(app, db.session)")
		self._add_line("")
		
		# Import and register views
		self._add_line("# Import views to register them with AppBuilder")
		self._add_line("from . import views")
		self._add_line("from . import model_views")
		self._add_line("")
		
		# Register APG entity views
		for entity in module.entities:
			if entity.entity_type == EntityType.AGENT:
				self._add_line(f"appbuilder.add_view({entity.name}View, '{entity.name}', icon='fa-cog', category='Agents')")
			elif entity.entity_type == EntityType.WORKFLOW:
				self._add_line(f"appbuilder.add_view({entity.name}View, '{entity.name}', icon='fa-tasks', category='Workflows')")
			elif entity.entity_type == EntityType.DIGITAL_TWIN:
				self._add_line(f"appbuilder.add_view({entity.name}View, '{entity.name}', icon='fa-cube', category='Digital Twins')")
		
		# Add database model views if any
		self._add_line("")
		self._add_line("# Register database model views")
		for entity in module.entities:
			if entity.entity_type == EntityType.DATABASE:
				self._add_line("try:")
				self._add_line("    from .model_views import *")
				self._add_line("    # Model views are automatically registered by importing")
				self._add_line("except ImportError:")
				self._add_line("    logging.getLogger(__name__).debug('No generated model views found')")
				break
		
		# Create database tables
		self._add_line("")
		self._add_line("# Create database tables")
		self._add_line("with app.app_context():")
		self._add_line("    try:")
		self._add_line("        db.create_all()")
		self._add_line("        logging.info('Database tables created successfully')")
		self._add_line("    except Exception as e:")
		self._add_line("        logging.error(f'Error creating database tables: {e}')")
		
		self._add_line("")
		self._add_line('if __name__ == "__main__":')
		self._add_line("    import os")
		self._add_line("    host = os.environ.get('FLASK_HOST', '0.0.0.0')")
		self._add_line("    port = int(os.environ.get('FLASK_PORT', 8080))")
		self._add_line("    debug = os.environ.get('FLASK_DEBUG', '1') == '1'")
		self._add_line("    ")
		self._add_line("    print(f'Starting APG Flask-AppBuilder application...')")
		self._add_line("    print(f'Host: {host}')")
		self._add_line("    print(f'Port: {port}')")
		self._add_line("    print(f'Debug: {debug}')")
		self._add_line("    print(f'Access at: http://{host}:{port}')")
		self._add_line("    ")
		self._add_line("    app.run(host=host, port=port, debug=debug)")
		
		import_block = self._format_imports()
		code_block = '\n'.join(self.output)
		return f"{import_block}\n\n{code_block}"
	
	def _generate_views(self, module: ModuleDeclaration) -> str:
		"""Generate Flask-AppBuilder views.py file"""
		self.output.clear()
		self.imports.clear()
		self._add_standard_imports()
		
		# Add module docstring
		self._add_line('"""')
		self._add_line(f"APG Views for {module.name}")
		self._add_line("=" * (len(module.name) + 15))
		self._add_line("")
		self._add_line("Flask-AppBuilder views generated from APG entities.")
		self._add_line('"""')
		self._add_line("")
		
		# Generate views for each entity
		for entity in module.entities:
			if entity.entity_type != EntityType.DATABASE:
				self._generate_entity(entity)
		
		import_block = self._format_imports()
		code_block = '\n'.join(self.output)
		return f"{import_block}\n\n{code_block}"
	
	def _generate_config(self) -> str:
		"""Generate Flask-AppBuilder config.py file"""
		return '''"""
Flask-AppBuilder Configuration
=============================

Configuration file for the APG Flask-AppBuilder application.
"""

import os
from flask_appbuilder.security.manager import AUTH_OID, AUTH_REMOTE_USER, AUTH_DB, AUTH_LDAP, AUTH_OAUTH

basedir = os.path.abspath(os.path.dirname(__file__))

# Your App secret key
SECRET_KEY = 'apg-generated-development-secret-key'

# The SQLAlchemy connection string
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')

# Flask-WTF flag for CSRF
CSRF_ENABLED = True

# ------------------------------
# GLOBALS FOR APP Builder 
# ------------------------------
# Uncomment to setup Your App name
APP_NAME = "APG Application"

# Uncomment to setup an App icon
#APP_ICON = "static/img/logo.jpg"

# ----------------------------------------------------
# AUTHENTICATION CONFIG
# ----------------------------------------------------
# The authentication type
# AUTH_OID : Is for OpenID
# AUTH_DB : Is for database (username/password)
# AUTH_LDAP : Is for LDAP
# AUTH_REMOTE_USER : Is for using REMOTE_USER from web server
AUTH_TYPE = AUTH_DB

# Uncomment to setup Full admin role name
#AUTH_ROLE_ADMIN = 'Admin'

# Uncomment to setup Public role name, no authentication needed
#AUTH_ROLE_PUBLIC = 'Public'

# Will allow user self registration
#AUTH_USER_REGISTRATION = True

# The default user self registration role
#AUTH_USER_REGISTRATION_ROLE = "Public"

# ----------------------------------------------------
# BABEL CONFIG
# ----------------------------------------------------
# Setup default language
BABEL_DEFAULT_LOCALE = 'en'
# Your application default translation path
BABEL_DEFAULT_FOLDER = 'babel/translations'
# The allowed translation for you app
LANGUAGES = {
    'en': {'flag':'gb', 'name':'English'},
    'pt': {'flag':'pt', 'name':'Portuguese'},
    'pt_BR': {'flag':'br', 'name': 'Pt Brazil'},
    'es': {'flag':'es', 'name':'Spanish'},
    'de': {'flag':'de', 'name':'German'},
    'zh': {'flag':'cn', 'name':'Chinese'},
    'ru': {'flag':'ru', 'name':'Russian'},
    'pl': {'flag':'pl', 'name':'Polish'}
}

# ----------------------------------------------------
# APG SPECIFIC CONFIG
# ----------------------------------------------------
# APG Runtime configuration
APG_AGENT_POLL_INTERVAL = 5  # seconds
APG_WORKFLOW_TIMEOUT = 300   # seconds
APG_DIGITAL_TWIN_SYNC_INTERVAL = 10  # seconds

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
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
}
'''
	
	def _generate_model_views(self, database: EntityDeclaration) -> str:
		"""Generate Flask-AppBuilder ModelViews for database tables"""
		if not isinstance(database, DatabaseDeclaration):
			return ""
		
		self.output.clear()
		self.imports.clear()
		
		# Add ModelView imports
		self.imports.add("from flask_appbuilder import ModelView")
		self.imports.add("from flask_appbuilder.models.sqla.interface import SQLAInterface")
		self.imports.add("from flask_appbuilder.security.decorators import has_access")
		self.imports.add("from .models import *")
		
		self._add_line('"""')
		self._add_line("Database Model Views")
		self._add_line("===================")
		self._add_line("")
		self._add_line("Flask-AppBuilder ModelViews for database tables.")
		self._add_line('"""')
		self._add_line("")
		
		# Generate ModelViews for each table
		for schema in database.schemas:
			for table in schema.tables:
				self._generate_table_model_view(table)
		
		import_block = self._format_imports()
		code_block = '\n'.join(self.output)
		return f"{import_block}\n\n{code_block}"
	
	def _generate_table_model_view(self, table: TableDeclaration):
		"""Generate Flask-AppBuilder ModelView for a table"""
		class_name = self._to_pascal_case(table.name)
		view_name = f"{class_name}View"
		
		self._add_line(f"class {view_name}(ModelView):")
		self._add_line(f'    """ModelView for {table.name} table"""')
		self._add_line("")
		self._add_line(f"    datamodel = SQLAInterface({class_name})")
		self._add_line("")
		
		# Generate column lists based on table columns
		column_names = [col.name for col in table.columns]
		
		self._add_line(f"    list_columns = {column_names}")
		self._add_line(f"    show_columns = {column_names}")
		self._add_line(f"    edit_columns = {[col.name for col in table.columns if not col.is_primary_key]}")
		self._add_line(f"    add_columns = {[col.name for col in table.columns if not col.is_primary_key]}")
		self._add_line("")
		
		# Add search columns for text fields
		text_columns = [col.name for col in table.columns if col.data_type in ['str', 'text']]
		if text_columns:
			self._add_line(f"    search_columns = {text_columns}")
		
		self._add_line("")
	
	def _generate_templates(self, module: ModuleDeclaration) -> Dict[str, str]:
		"""Generate HTML templates for Flask-AppBuilder"""
		templates = {}
		
		# Base template
		templates["templates/base.html"] = self._generate_base_template(module)
		
		# Agent dashboard templates
		for entity in module.entities:
			if entity.entity_type == EntityType.AGENT:
				templates[f"templates/agent_dashboard.html"] = self._generate_agent_dashboard_template(entity)
		
		return templates
	
	def _generate_base_template(self, module: ModuleDeclaration) -> str:
		"""Generate base HTML template"""
		return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{module.name} - APG Application</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">{module.name}</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/agents">Agents</a>
                <a class="nav-link" href="/workflows">Workflows</a>
                <a class="nav-link" href="/digitaltings">Digital Twins</a>
            </div>
        </div>
    </nav>
    
    <div class="container-fluid mt-4">
        {{% block content %}}{{% endblock %}}
    </div>
</body>
</html>'''
	
	def _generate_agent_dashboard_template(self, entity: EntityDeclaration) -> str:
		"""Generate fully functional agent dashboard template"""
		# Generate method buttons based on actual agent methods
		method_buttons = []
		for method in entity.methods:
			if method.name not in ['start', 'stop', 'get_status']:
				method_buttons.append(f'''
                    <button type="button" class="btn btn-outline-primary me-2" onclick="callAgentMethod('{method.name}')">
                        {method.name.replace('_', ' ').title()}
                    </button>''')
		
		# Generate property display
		property_displays = []
		for prop in entity.properties:
			property_displays.append(f'''
                        <tr>
                            <td><strong>{prop.name.replace('_', ' ').title()}</strong></td>
                            <td><span id="prop-{prop.name}">{{{{ agent_status.{prop.name} or 'N/A' }}}}</span></td>
                        </tr>''')
		
		return f'''{{% extends "appbuilder/base.html" %}}

{{% block content %}}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-12">
            <h1><i class="fa fa-cog"></i> {{{{ agent_name }}}} Dashboard</h1>
            <p class="lead">Monitor and control the {{{{ agent_name }}}} agent</p>
            
            <!-- Agent Status Card -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5><i class="fa fa-info-circle"></i> Agent Status</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <strong>Status:</strong> 
                                <span id="agent-status" class="badge {{% if agent_running %}}bg-success{{% else %}}bg-danger{{% endif %}}">
                                    {{% if agent_running %}}Running{{% else %}}Stopped{{% endif %}}
                                </span>
                            </div>
                            <div class="mb-3">
                                <strong>Last Updated:</strong> 
                                <span id="last-updated">{{{{ agent_status.timestamp or 'Never' }}}}</span>
                            </div>
                            <div class="btn-group" role="group">
                                <button type="button" class="btn btn-success" onclick="startAgent()" id="start-btn" 
                                        {{% if agent_running %}}disabled{{% endif %}}>
                                    <i class="fa fa-play"></i> Start
                                </button>
                                <button type="button" class="btn btn-danger" onclick="stopAgent()" id="stop-btn"
                                        {{% if not agent_running %}}disabled{{% endif %}}>
                                    <i class="fa fa-stop"></i> Stop
                                </button>
                                <button type="button" class="btn btn-info" onclick="refreshStatus()">
                                    <i class="fa fa-refresh"></i> Refresh
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Agent Properties -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h5><i class="fa fa-list"></i> Agent Properties</h5>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm">
                                <tbody>{''.join(property_displays)}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Agent Methods -->
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5><i class="fa fa-cogs"></i> Agent Methods</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <strong>Available Methods:</strong>
                            </div>
                            <div id="method-buttons">{''.join(method_buttons)}
                            </div>
                            <div class="mt-3">
                                <strong>Method Result:</strong>
                                <pre id="method-result" class="bg-light p-2 mt-2" style="min-height: 50px;">No method called yet</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Activity Logs -->
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-warning text-dark">
                            <h5><i class="fa fa-file-text-o"></i> Activity Logs</h5>
                        </div>
                        <div class="card-body">
                            <div id="agent-logs" class="border rounded p-3 bg-light" 
                                 style="height: 300px; overflow-y: scroll; font-family: monospace;">
                                <div class="text-muted">[{{{{ agent_status.timestamp or 'System' }}}}] Agent dashboard loaded</div>
                            </div>
                            <div class="mt-2">
                                <button type="button" class="btn btn-outline-secondary btn-sm" onclick="clearLogs()">
                                    <i class="fa fa-trash"></i> Clear Logs
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
$(document).ready(function() {{
    // Initialize the dashboard
    addLog('Agent dashboard initialized for {{{{ agent_name }}}}');
    
    // Start auto-refresh
    setInterval(refreshStatus, 10000); // Refresh every 10 seconds
}});

function startAgent() {{
    addLog('Starting agent...');
    $.post('/start/')
    .done(function(data) {{
        if (data.status === 'success') {{
            $('#agent-status').removeClass('bg-danger').addClass('bg-success').text('Running');
            $('#start-btn').prop('disabled', true);
            $('#stop-btn').prop('disabled', false);
            addLog('✓ Agent started: ' + data.message, 'success');
        }} else {{
            addLog('⚠ Warning: ' + data.message, 'warning');
        }}
    }})
    .fail(function(xhr) {{
        addLog('✗ Error starting agent: ' + (xhr.responseJSON ? xhr.responseJSON.message : 'Unknown error'), 'error');
    }});
}}

function stopAgent() {{
    addLog('Stopping agent...');
    $.post('/stop/')
    .done(function(data) {{
        if (data.status === 'success') {{
            $('#agent-status').removeClass('bg-success').addClass('bg-danger').text('Stopped');
            $('#start-btn').prop('disabled', false);
            $('#stop-btn').prop('disabled', true);
            addLog('✓ Agent stopped: ' + data.message, 'success');
        }} else {{
            addLog('⚠ Warning: ' + data.message, 'warning');
        }}
    }})
    .fail(function(xhr) {{
        addLog('✗ Error stopping agent: ' + (xhr.responseJSON ? xhr.responseJSON.message : 'Unknown error'), 'error');
    }});
}}

function refreshStatus() {{
    $.get('/status/')
    .done(function(data) {{
        if (data.status === 'success') {{
            var status = data.data;
            $('#last-updated').text(new Date().toLocaleString());
            
            // Update property values
            {{% for prop in entity.properties %}}
            $('#prop-{{{{ prop.name }}}}').text(status.{{{{ prop.name }}}} || 'N/A');
            {{% endfor %}}
            
            // Update running state
            if (status.running) {{
                $('#agent-status').removeClass('bg-danger').addClass('bg-success').text('Running');
                $('#start-btn').prop('disabled', true);
                $('#stop-btn').prop('disabled', false);
            }} else {{
                $('#agent-status').removeClass('bg-success').addClass('bg-danger').text('Stopped');
                $('#start-btn').prop('disabled', false);
                $('#stop-btn').prop('disabled', true);
            }}
            
            addLog('Status refreshed', 'info');
        }}
    }})
    .fail(function(xhr) {{
        addLog('Error refreshing status', 'error');
    }});
}}

function callAgentMethod(methodName) {{
    addLog('Calling method: ' + methodName);
    
    // Get parameters if needed (simplified - could be enhanced with form)
    var params = {{}};
    
    $.post('/' + methodName.replace('_', '-') + '/', JSON.stringify(params), 'json')
    .done(function(data) {{
        if (data.status === 'success') {{
            $('#method-result').text(JSON.stringify(data.result, null, 2));
            addLog('✓ Method ' + methodName + ' completed successfully', 'success');
        }} else {{
            $('#method-result').text('Error: ' + data.message);
            addLog('✗ Method ' + methodName + ' failed: ' + data.message, 'error');  
        }}
    }})
    .fail(function(xhr) {{
        var errorMsg = xhr.responseJSON ? xhr.responseJSON.message : 'Unknown error';
        $('#method-result').text('Error: ' + errorMsg);
        addLog('✗ Method ' + methodName + ' error: ' + errorMsg, 'error');
    }});
}}

function addLog(message, type = 'info') {{
    var timestamp = new Date().toLocaleTimeString();
    var icon = type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ';
    var color = type === 'success' ? 'text-success' : type === 'error' ? 'text-danger' : type === 'warning' ? 'text-warning' : 'text-info';
    
    var logEntry = '<div class="' + color + '">[' + timestamp + '] ' + icon + ' ' + message + '</div>';
    $('#agent-logs').append(logEntry);
    $('#agent-logs').scrollTop($('#agent-logs')[0].scrollHeight);
}}

function clearLogs() {{
    $('#agent-logs').empty();
    addLog('Logs cleared', 'info');
}}
</script>
{{% endblock %}}'''
	
	def _generate_agent_runtime_method(self, method: MethodDeclaration):
		"""Generate runtime implementation for agent method"""
		# Generate method signature
		params = ["self"]
		for param in method.parameters:
			param_name = param.name
			if param.default_value:
				default = self._generate_expression(param.default_value)
				params.append(f"{param_name}={default}")
			else:
				params.append(param_name)
		
		self._add_line(f"def {method.name}({', '.join(params)}):")
		self._indent()
		self._add_line(f'"""Runtime implementation of {method.name}"""')
		
		# Generate method body with actual logic
		if method.body:
			self._generate_statement(method.body)
		else:
			if method.return_type:
				return_type = method.return_type.type_name
				if return_type == "str":
					self._add_line(f"return f'Result from {method.name}'")
				elif return_type == "int":
					self._add_line("return 42")
				elif return_type == "float":
					self._add_line("return 3.14")
				elif return_type == "bool":
					self._add_line("return True")
				elif return_type == "dict":
					self._add_line("return {'result': 'success', 'method': '" + method.name + "'}")
				elif return_type == "list":
					self._add_line("return []")
				else:
					self._add_line("return None")
			else:
				self._add_line("return {'status': 'executed', 'method': '" + method.name + "'}")
		
		self._dedent()
		self._add_line("")
	
	def _generate_agent_api_method(self, method: MethodDeclaration, entity: EntityDeclaration):
		"""Generate Flask-AppBuilder API endpoint for agent method"""
		endpoint_name = method.name.lower().replace('_', '-')
		
		self._add_line(f"@expose('/{endpoint_name}/', methods=['POST'])")
		self._add_line("@has_access")
		self._add_line(f"def {method.name}_api(self):")
		self._add_line(f'    """API endpoint for {method.name} method"""')
		self._add_line("    try:")
		self._add_line("        # Get request parameters")
		self._add_line("        data = request.get_json() or {}")
		self._add_line(f"        agent = {entity.name.lower()}_instance")
		self._add_line("")
		
		# Generate parameter extraction and method call
		if method.parameters:
			self._add_line("        # Extract parameters from request")
			param_calls = []
			for param in method.parameters:
				self._add_line(f"        {param.name} = data.get('{param.name}')")
				param_calls.append(param.name)
			
			self._add_line(f"        # Call agent method")
			self._add_line(f"        result = agent.{method.name}({', '.join(param_calls)})")
		else:
			self._add_line(f"        # Call agent method")
			self._add_line(f"        result = agent.{method.name}()")
		
		self._add_line("")
		self._add_line("        return jsonify({'status': 'success', 'result': result})")
		self._add_line("    except Exception as e:")
		self._add_line("        return jsonify({'status': 'error', 'message': str(e)})")
		self._add_line("")
	
	def _get_default_value_for_type(self, type_annotation: TypeAnnotation) -> str:
		"""Get appropriate default value for a type"""
		type_name = type_annotation.type_name.lower()
		
		if type_name == "str":
			return '""'
		elif type_name == "int":
			return "0"
		elif type_name == "float":
			return "0.0"
		elif type_name == "bool":
			return "False"
		elif type_name == "list":
			return "[]"
		elif type_name == "dict":
			return "{}"
		else:
			return "None"
	
	# ========================================
	# Utility Methods
	# ========================================
	
	def _apg_type_to_python(self, type_annotation: TypeAnnotation) -> str:
		"""Convert APG type annotation to Python type hint"""
		type_map = {
			'str': 'str',
			'int': 'int',
			'float': 'float',
			'bool': 'bool',
			'list': 'List[Any]',
			'dict': 'Dict[str, Any]',
			'void': 'None',
			'any': 'Any'
		}
		
		base_type = type_map.get(type_annotation.type_name, type_annotation.type_name)
		
		if type_annotation.is_list:
			base_type = f"List[{base_type}]"
		elif type_annotation.is_dict:
			base_type = f"Dict[str, {base_type}]"
		
		if type_annotation.is_optional:
			base_type = f"Optional[{base_type}]"
		
		return base_type
	
	def _get_default_return_value(self, type_annotation: TypeAnnotation) -> str:
		"""Get default return value for a type"""
		defaults = {
			'str': '""',
			'int': '0',
			'float': '0.0',
			'bool': 'False',
			'list': '[]',
			'dict': '{}',
			'any': 'None'
		}
		
		return defaults.get(type_annotation.type_name, 'None')
	
	def _to_pascal_case(self, snake_str: str) -> str:
		"""Convert snake_case to PascalCase"""
		return ''.join(word.capitalize() for word in snake_str.split('_'))
	
	def _format_imports(self) -> str:
		"""Format import statements"""
		if not self.imports:
			return ""
		
		sorted_imports = sorted(self.imports)
		return '\n'.join(sorted_imports)
	
	def _add_line(self, line: str = ""):
		"""Add a line with proper indentation"""
		if line:
			self.output.append("    " * self.indent_level + line)
		else:
			self.output.append("")
	
	def _indent(self):
		"""Increase indentation level"""
		self.indent_level += 1
	
	def _dedent(self):
		"""Decrease indentation level"""
		self.indent_level = max(0, self.indent_level - 1)


# ========================================
# Main Code Generator Class
# ========================================

class CodeGenerator:
	"""
	Main code generator that orchestrates different target language generators.
	Currently supports Python, with extensibility for other languages.
	"""

	SUPPORTED_TARGETS = ("python",)
	
	def __init__(self, config: CodeGenConfig = None):
		self.config = config or CodeGenConfig()
		self.generators = {
			'python': PythonCodeGenerator(config)
		}
	
	def generate(self, ast: ModuleDeclaration, target_language: str = None) -> Dict[str, str]:
		"""
		Generate code for the specified target language.
		
		Args:
			ast: Root AST node
			target_language: Target language ('python', etc.)
			
		Returns:
			Dictionary mapping file names to generated code
		"""
		requested_target = target_language or self.config.target_language
		target = self.normalize_target(requested_target)
		
		if target not in self.generators:
			supported = ", ".join(self.SUPPORTED_TARGETS)
			raise ValueError(f"Unsupported target language: {requested_target}. Supported targets: {supported}")
		
		generator = self.generators[target]
		return generator.generate(ast)

	@classmethod
	def normalize_target(cls, target_language: str) -> str:
		"""Normalize a user-provided APG code-generation target."""
		return (target_language or "python").lower()
	
	def write_files(self, generated_files: Dict[str, str], output_dir: Path):
		"""Write generated files to disk"""
		output_dir.mkdir(parents=True, exist_ok=True)
		
		for filename, content in generated_files.items():
			file_path = output_dir / filename
			with open(file_path, 'w', encoding='utf-8') as f:
				f.write(content)
			
			print(f"Generated: {file_path}")


def test_code_generator():
	"""Test the code generator"""
	print("Code Generator module loaded successfully")
	print("Classes available:", [
		'CodeGenerator', 'PythonCodeGenerator', 'CodeGenConfig'
	])


if __name__ == "__main__":
	test_code_generator()
