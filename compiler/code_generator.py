"""
APG Code Generator Module
=========================

Generates Python code from APG Abstract Syntax Trees.
Transforms APG entities, workflows, and other constructs into executable Python code
with proper imports, type hints, and runtime support.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Import AST nodes
from .ast_builder import (
	ModuleDeclaration, Expression,
	LiteralExpression, IdentifierExpression, BinaryExpression, CallExpression,
	UnaryExpression, MemberExpression, IndexExpression, ListExpression,
	DictExpression,
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
	use_composable_templates: bool = False
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
	- Dependency-free application manifests
	- First-class AI agent composition metadata
	- Capability contracts with configuration, rules, UI, and theme metadata
	- Composable template integration with Python-first fallback output
	"""
	
	def __init__(self, config: CodeGenConfig = None):
		self.config = config or CodeGenConfig()
		
		# Code generation state
		self.current_module: Optional[ModuleDeclaration] = None
	
	def generate(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""
		Generate executable Python artifacts from an APG AST.
		
		Args:
			ast: Root AST node (ModuleDeclaration)
			
		Returns:
			Dictionary mapping file names to generated code content
		"""
		self.current_module = ast
		
		if self.config.use_composable_templates:
			return self._generate_with_composable_templates(ast)
		return self._generate_python_application(ast)
	
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
				# Combine template system with dependency-free entity metadata.
				template_files = generated_files
				template_files.update(self._generate_python_entity_catalog_files(ast))
				return template_files
			else:
				# Return complete application
				return generated_files
				
		except Exception as e:
			print(f"Error in composable template generation: {e}")
			print("Falling back to dependency-free Python generation...")
			return self._generate_python_application(ast)

	def _generate_python_application(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate a dependency-free Python application manifest."""
		files = {
			"app.py": self._generate_python_app(ast),
			"__init__.py": self._generate_package_init(ast),
			"requirements.txt": self._generate_python_requirements(),
		}
		files.update(self._generate_ai_agent_files(ast))
		files.update(self._generate_capability_files(ast))
		return files

	def _generate_python_app(self, module: ModuleDeclaration) -> str:
		"""Generate a framework-neutral Python app.py entrypoint."""
		entity_specs = [
			{
				"name": entity.name,
				"type": entity.entity_type.value,
				"properties": [property.name for property in entity.properties],
				"methods": [method.name for method in entity.methods],
			}
			for entity in module.entities
		]
		return f'''"""
{module.name} - APG Python Application
{"=" * (len(module.name) + 25)}

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional


MODULE_NAME = {module.name!r}
MODULE_VERSION = {module.version!r}
MODULE_DESCRIPTION = {module.description!r}
ENTITIES = {entity_specs!r}
ENTITY_NAMES = {{entity["name"] for entity in ENTITIES}}
RECORD_STORE: Dict[str, list[Dict[str, Any]]] = {{entity["name"]: [] for entity in ENTITIES}}
NEXT_RECORD_IDS: Dict[str, int] = {{entity["name"]: 1 for entity in ENTITIES}}


def _optional_module(name: str) -> Optional[Any]:
    if __package__:
        try:
            return importlib.import_module(f".{{name}}", __package__)
        except ImportError:
            pass
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


AI_AGENTS = _optional_module("ai_agents")
APG_CAPABILITIES = _optional_module("apg_capabilities")


def list_entities() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES]


def list_records(entity_name: str | None = None) -> Dict[str, list[Dict[str, Any]]] | list[Dict[str, Any]]:
    if entity_name is None:
        return {{
            name: [dict(record) for record in records]
            for name, records in RECORD_STORE.items()
        }}
    return [dict(record) for record in RECORD_STORE[entity_name]]


def describe_application() -> Dict[str, Any]:
    description: Dict[str, Any] = {{
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "entities": list_entities(),
    }}
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agents"] = AI_AGENTS.list_agents()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_agent") and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agent_descriptions"] = {{
            name: AI_AGENTS.describe_agent(name)
            for name in AI_AGENTS.list_agents()
        }}
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_teams"] = AI_AGENTS.list_agent_teams()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_team") and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_team_descriptions"] = {{
            name: AI_AGENTS.describe_team(name)
            for name in AI_AGENTS.list_agent_teams()
        }}
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities"):
        description["capabilities"] = APG_CAPABILITIES.list_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities"):
        description["capability_descriptions"] = APG_CAPABILITIES.describe_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities_by_erp_module"):
        description["capability_descriptions_by_erp_module"] = APG_CAPABILITIES.describe_capabilities_by_erp_module()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_dependency_graph"):
        description["capability_dependency_graph"] = APG_CAPABILITIES.capability_dependency_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_load_order"):
        description["capability_load_order"] = APG_CAPABILITIES.capability_load_order()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "ui_route_index"):
        description["ui_routes"] = APG_CAPABILITIES.ui_route_index()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "composition_graph"):
        description["composition_graph"] = APG_CAPABILITIES.composition_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "streaming_processor_index"):
        description["streaming_processors"] = APG_CAPABILITIES.streaming_processor_index()
    return description


def _record_validation(report: Dict[str, Any], name: str, validation: Dict[str, Any]) -> None:
    check = dict(validation)
    errors = [str(error) for error in check.get("errors", [])]
    warnings = [str(warning) for warning in check.get("warnings", [])]
    report["checks"][name] = check
    report["errors"].extend(f"{{name}}: {{error}}" for error in errors)
    report["warnings"].extend(f"{{name}}: {{warning}}" for warning in warnings)


def validate_application(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {{
        "name": MODULE_NAME,
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {{}},
    }}
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        _record_validation(
            report,
            "ai_agent_runtimes",
            AI_AGENTS.validate_agent_runtimes(available_agent_runtimes),
        )
    if APG_CAPABILITIES is not None:
        for check_name, function_name in (
            ("capability_contracts", "validate_capability_contracts"),
            ("capability_dependencies", "validate_capability_dependencies"),
            ("component_contracts", "validate_component_contracts"),
            ("master_data_contracts", "validate_master_data_contracts"),
            ("capability_i18n", "validate_capability_i18n"),
            ("streaming_contracts", "validate_streaming_contracts"),
        ):
            validator = getattr(APG_CAPABILITIES, function_name, None)
            if validator is not None:
                _record_validation(report, check_name, validator())
    report["valid"] = not report["errors"]
    return report


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _record_route(path: str) -> Dict[str, str | None] | None:
    parts = [part for part in path.split("/") if part]
    if parts == ["records"]:
        return {{"entity": None, "record_id": None}}
    if len(parts) in {{2, 3}} and parts[0] == "records":
        return {{
            "entity": parts[1],
            "record_id": parts[2] if len(parts) == 3 else None,
        }}
    if len(parts) in {{3, 4}} and parts[0] == "entities" and parts[2] == "records":
        return {{
            "entity": parts[1],
            "record_id": parts[3] if len(parts) == 4 else None,
        }}
    return None


def _record_by_id(entity_name: str, record_id: str) -> Dict[str, Any] | None:
    for record in RECORD_STORE[entity_name]:
        if str(record.get("id")) == str(record_id):
            return dict(record)
    return None


def _records_payload(path: str) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name is None:
        return 200, {{"records": list_records()}}
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    if record_id is None:
        records = list_records(entity_name)
        return 200, {{"entity": entity_name, "records": records, "count": len(records)}}
    record = _record_by_id(entity_name, record_id)
    if record is None:
        return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}
    return 200, {{"entity": entity_name, "record": record}}


def _route_payload(path: str) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path in {{"/", "/manifest", "/application"}}:
        return 200, describe_application()
    if path == "/health":
        validation = validate_application()
        return 200, {{
            "status": "ok" if validation["valid"] else "warning",
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "valid": validation["valid"],
            "warnings": validation["warnings"],
        }}
    if path == "/validate":
        validation = validate_application()
        return (200 if validation["valid"] else 422), validation
    if path == "/entities":
        return 200, {{"entities": list_entities()}}
    if path == "/records" or path.startswith("/records/") or (
        path.startswith("/entities/") and "/records" in path
    ):
        return _records_payload(path)
    if path == "/agents":
        return 200, {{
            "agents": describe_application().get("ai_agent_descriptions", {{}}),
            "teams": describe_application().get("ai_agent_team_descriptions", {{}}),
        }}
    if path == "/capabilities":
        app = describe_application()
        return 200, {{
            "capabilities": app.get("capability_descriptions", {{}}),
            "by_erp_module": app.get("capability_descriptions_by_erp_module", {{}}),
            "dependency_graph": app.get("capability_dependency_graph", {{}}),
            "load_order": app.get("capability_load_order", {{}}),
        }}
    if path == "/routes":
        return 200, {{"routes": describe_application().get("ui_routes", {{}})}}
    if path == "/composition":
        return 200, describe_application().get("composition_graph", {{"nodes": [], "edges": []}})
    return 404, {{"error": "not_found", "path": path}}


def _rule_evaluation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if path.startswith("/capabilities/") and path.endswith("/rules/evaluate"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3:
            capability_name = parts[1]
    if not capability_name:
        return 400, {{"error": "missing_capability"}}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "evaluate_capability_rules"):
        return 404, {{"error": "capability_rules_unavailable"}}
    context = payload.get("context", {{}})
    if not isinstance(context, dict):
        return 400, {{"error": "context_must_be_object"}}
    try:
        return 200, APG_CAPABILITIES.evaluate_capability_rules(str(capability_name), context)
    except KeyError:
        return 404, {{"error": "unknown_capability", "capability": str(capability_name)}}


def _capability_name_from_payload_or_path(path: str, payload: Dict[str, Any]) -> str | None:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if capability_name:
        return str(capability_name)
    if path.startswith("/capabilities/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            return parts[1]
    return None


def _configuration_payload(path: str, payload: Dict[str, Any], validate: bool = False) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {{"error": "missing_capability"}}
    if APG_CAPABILITIES is None:
        return 404, {{"error": "capabilities_unavailable"}}
    configuration = payload.get("configuration", payload.get("overrides"))
    if configuration is not None and not isinstance(configuration, dict):
        return 400, {{"error": "configuration_must_be_object"}}
    try:
        if validate:
            validator = getattr(APG_CAPABILITIES, "validate_capability_configuration", None)
            if validator is None:
                return 404, {{"error": "configuration_validation_unavailable"}}
            return 200, validator(str(capability_name), configuration)
        resolver = getattr(APG_CAPABILITIES, "capability_configuration", None)
        if resolver is None:
            return 404, {{"error": "configuration_resolution_unavailable"}}
        return 200, {{
            "capability": str(capability_name),
            "configuration": resolver(str(capability_name), configuration),
        }}
    except KeyError:
        return 404, {{"error": "unknown_capability", "capability": str(capability_name)}}


def _approval_plan_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {{"error": "missing_capability"}}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "approval_plan"):
        return 404, {{"error": "approval_planning_unavailable"}}
    context = payload.get("context", {{}})
    if not isinstance(context, dict):
        return 400, {{"error": "context_must_be_object"}}
    try:
        return 200, APG_CAPABILITIES.approval_plan(str(capability_name), context)
    except KeyError:
        return 404, {{"error": "unknown_capability", "capability": str(capability_name)}}


def _create_record_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is not None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {{"error": "record_must_be_object"}}
    record = dict(raw_record)
    if record.get("id") in (None, ""):
        record["id"] = NEXT_RECORD_IDS[entity_name]
        NEXT_RECORD_IDS[entity_name] += 1
    elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
        return 409, {{"error": "duplicate_record_id", "entity": entity_name, "id": record["id"]}}
    RECORD_STORE[entity_name].append(record)
    return 201, {{
        "entity": entity_name,
        "record": dict(record),
        "count": len(RECORD_STORE[entity_name]),
    }}


def _post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path.startswith("/records/") or (
        path.startswith("/entities/") and path.endswith("/records")
    ):
        return _create_record_payload(path, payload)
    if path in {{"/rules/evaluate", "/capabilities/rules/evaluate"}} or (
        path.startswith("/capabilities/") and path.endswith("/rules/evaluate")
    ):
        return _rule_evaluation_payload(path, payload)
    if path in {{"/configuration/resolve", "/capabilities/configuration/resolve"}} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/resolve")
    ):
        return _configuration_payload(path, payload)
    if path in {{"/configuration/validate", "/capabilities/configuration/validate"}} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/validate")
    ):
        return _configuration_payload(path, payload, validate=True)
    if path in {{"/approval/plan", "/capabilities/approval/plan"}} or (
        path.startswith("/capabilities/") and path.endswith("/approval/plan")
    ):
        return _approval_plan_payload(path, payload)
    return 404, {{"error": "not_found", "path": path}}


class ApplicationRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        status, payload = _route_payload(path)
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b"{{}}"
            payload = json.loads(raw_body.decode("utf-8") or "{{}}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            status, response = _post_payload(path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            status, response = 400, {{"error": "invalid_json", "message": str(error)}}
        body = _json_bytes(response)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("APG_DEBUG") == "1":
            super().log_message(format, *args)


def _arg_value(argv: list[str], name: str, default: str) -> str:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def run_server(host: str | None = None, port: int | str | None = None) -> None:
    resolved_host = host or os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1"
    resolved_port = int(port or os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    server = HTTPServer((resolved_host, resolved_port), ApplicationRequestHandler)
    print(f"{{MODULE_NAME}} listening on http://{{resolved_host}}:{{resolved_port}}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--describe" in args:
        print(json.dumps(describe_application(), indent=2, sort_keys=True))
        return
    if "--validate" in args:
        print(json.dumps(validate_application(), indent=2, sort_keys=True))
        return
    host = _arg_value(args, "--host", os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1")
    port = _arg_value(args, "--port", os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    run_server(host, port)


if __name__ == "__main__":
    main()
'''

	def _generate_python_requirements(self) -> str:
		"""Generate requirements for the dependency-free Python target."""
		return """# APG generated Python application requirements
# The default compiler target uses only the Python standard library.
"""

	def _generate_python_entity_catalog_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate dependency-free entity metadata for hybrid template mode."""
		return {"entities.py": self._generate_python_entity_catalog(ast)}

	def _generate_python_entity_catalog(self, module: ModuleDeclaration) -> str:
		"""Generate a framework-neutral entity catalog module."""
		entity_specs = [
			{
				"name": entity.name,
				"type": entity.entity_type.value,
				"properties": [property.name for property in entity.properties],
				"methods": [method.name for method in entity.methods],
			}
			for entity in module.entities
		]
		return f'''"""
{module.name} Entity Catalog
{"=" * (len(module.name) + 15)}

Generated APG entity metadata for composable hybrid output.
"""

from __future__ import annotations

from typing import Any, Dict


ENTITIES = {entity_specs!r}


def list_entities() -> list[Dict[str, Any]]:
    """Return APG entity metadata for composition adapters."""
    return [dict(entity) for entity in ENTITIES]
'''
	
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
				"screens": capability.screens,
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
    screens: Any


CAPABILITY_DATA: Dict[str, Dict[str, Any]] = {capability_specs!r}
CAPABILITIES: Dict[str, CapabilitySpec] = {{
    name: CapabilitySpec(name=name, **data)
    for name, data in CAPABILITY_DATA.items()
}}


def list_capabilities() -> List[str]:
    return sorted(CAPABILITIES)


def get_capability(name: str) -> CapabilitySpec:
    return CAPABILITIES[name]


def describe_capability(name: str) -> Dict[str, Any]:
    capability = get_capability(name)
    return {{
        "name": capability.name,
        "contract": dict(capability.contract),
        "provides": list(capability.provides),
        "requires": list(capability.requires),
        "configuration": dict(capability.configuration),
        "rules": [dict(rule) for rule in capability.rules],
        "rule_engine": dict(capability.rule_engine),
        "ui": dict(capability.ui),
        "theme": dict(capability.theme),
        "runtime": dict(capability.runtime),
        "erp_modules": list(capability.erp_modules),
        "components": capability.components,
        "business_rules": [dict(rule) for rule in capability.business_rules],
        "approvals": capability.approvals,
        "master_data": capability.master_data,
        "i18n": dict(capability.i18n),
        "streaming": dict(capability.streaming),
        "screens": capability.screens,
    }}


def describe_capabilities() -> Dict[str, Dict[str, Any]]:
    return {{
        name: describe_capability(name)
        for name in list_capabilities()
    }}


def capabilities_by_erp_module() -> Dict[str, List[CapabilitySpec]]:
    grouped: Dict[str, List[CapabilitySpec]] = {{}}
    for capability in CAPABILITIES.values():
        for module_name in capability.erp_modules:
            grouped.setdefault(module_name, []).append(capability)
    return grouped


def capability_names_by_erp_module() -> Dict[str, List[str]]:
    return {{
        module_name: sorted(capability.name for capability in capabilities)
        for module_name, capabilities in sorted(capabilities_by_erp_module().items())
    }}


def describe_capabilities_by_erp_module() -> Dict[str, List[Dict[str, Any]]]:
    return {{
        module_name: [describe_capability(name) for name in capability_names]
        for module_name, capability_names in capability_names_by_erp_module().items()
    }}


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
        integer_parse_failed = True
    try:
        return float(value)
    except ValueError:
        float_parse_failed = True
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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _screen_relationships(value: Any) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            relationships.append(dict(item))
            continue
        text = str(item).strip()
        if not text:
            continue
        relation = {{"type": "relates_to"}}
        if "->" in text:
            source, target = [part.strip() for part in text.split("->", 1)]
            relation.update({{"from": source, "to": target}})
        else:
            relation["to"] = text
        relationships.append(relation)
    return relationships


def _normalize_screen(
    capability: CapabilitySpec,
    name: str,
    spec: Any,
    index: int = 0,
) -> Dict[str, Any]:
    screen_spec = dict(spec) if isinstance(spec, dict) else {{"component": spec or name}}
    route = screen_spec.get("route", screen_spec.get("path", ""))
    component = screen_spec.get("component", name)
    return {{
        "id": f"{{capability.name}}.{{name}}",
        "capability": capability.name,
        "name": name,
        "path": route,
        "route": route,
        "layout": screen_spec.get("layout"),
        "component": component,
        "contains": _as_list(screen_spec.get("contains")),
        "composes": _as_list(screen_spec.get("composes")),
        "binds": _as_list(screen_spec.get("binds")),
        "actions": _as_list(screen_spec.get("actions")),
        "events": _as_list(screen_spec.get("events")),
        "relationships": _screen_relationships(screen_spec.get("relationships")),
        "permission": screen_spec.get("permission"),
        "permissions": _as_list(screen_spec.get("permissions")),
        "rules": _as_list(screen_spec.get("rules")),
        "nav_group": screen_spec.get("nav_group"),
        "shell": capability.ui.get("shell"),
        "theme": screen_spec.get("theme", capability.theme.get("name")),
        "spec": screen_spec,
    }}


def _declared_screen_specs(capability: CapabilitySpec) -> Any:
    if capability.screens:
        return capability.screens
    ui_screens = capability.ui.get("screens")
    return ui_screens if ui_screens else {{}}


def capability_screens(capability_name: str) -> List[Dict[str, Any]]:
    capability = get_capability(capability_name)
    screens: List[Dict[str, Any]] = []
    declared = _declared_screen_specs(capability)
    if isinstance(declared, dict):
        for index, (name, spec) in enumerate(declared.items()):
            screens.append(_normalize_screen(capability, str(name), spec, index))
    elif isinstance(declared, list):
        for index, item in enumerate(declared):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or item.get("component") or f"screen_{{index + 1}}")
                screens.append(_normalize_screen(capability, name, item, index))
            else:
                name = str(item)
                screens.append(_normalize_screen(capability, name, {{"component": name}}, index))

    known_names = {{screen["name"] for screen in screens}}
    routes = capability.ui.get("routes", [])
    if isinstance(routes, list):
        for index, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            name = str(route.get("name") or route.get("component") or f"screen_{{index + 1}}")
            if name in known_names:
                continue
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
            for contained in screen.get("contains", []):
                contained_id = f"component:{{contained}}"
                node(contained_id, "component", name=str(contained))
                edge(screen_id, contained_id, "contains")
            for composed in screen.get("composes", []):
                composed_id = f"component:{{composed}}"
                node(composed_id, "component", name=str(composed))
                edge(screen_id, composed_id, "composes")
            for binding in screen.get("binds", []):
                binding_id = f"binding:{{binding}}"
                node(binding_id, "binding", name=str(binding))
                edge(screen_id, binding_id, "binds_to")
            for relationship in screen.get("relationships", []):
                if not isinstance(relationship, dict):
                    continue
                source = relationship.get("from")
                target = relationship.get("to")
                if not source or not target:
                    continue
                source_id = f"component:{{source}}"
                target_id = f"component:{{target}}"
                relation = str(relationship.get("via") or relationship.get("type") or "relates_to")
                node(source_id, "component", name=str(source))
                node(target_id, "component", name=str(target))
                edge(source_id, target_id, relation)

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


def describe_agent(name: str) -> Dict[str, Any]:
    agent = get_agent(name)
    return {{
        "name": agent.name,
        "role": agent.role,
        "model": agent.model,
        "runtime": agent.runtime,
        "system": agent.system,
        "capabilities": list(agent.capabilities),
        "tools": list(agent.tools),
        "memory": dict(agent.memory) if agent.memory else None,
        "inputs": list(agent.inputs),
        "outputs": list(agent.outputs),
        "handoffs": [dict(edge) for edge in agent.handoffs],
        "configuration": dict(agent.configuration),
        "rules": [dict(rule) for rule in agent.rules],
        "ui": dict(agent.ui),
        "theme": dict(agent.theme),
    }}


def list_agents() -> List[str]:
    return sorted(AI_AGENTS)


def list_agent_teams() -> List[str]:
    return sorted(AI_AGENT_TEAMS)


def list_teams() -> List[str]:
    return list_agent_teams()


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
        "agents": [describe_agent(agent) for agent in team.agents],
        "agent_names": list(team.agents),
        "capabilities": list(team.capabilities),
        "flow": [dict(edge) for edge in team.flow],
        "policy": dict(team.policy),
        "configuration": dict(team.configuration),
        "rules": [dict(rule) for rule in team.rules],
        "ui": dict(team.ui),
        "theme": dict(team.theme),
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
	
	def _generate_package_init(self, module: ModuleDeclaration) -> str:
		"""Generate package __init__.py file."""
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
			'from .app import describe_application, list_entities, list_records, main, validate_application',
			'',
			'__all__ = [',
			'    "__version__",',
			'    "describe_application",',
			'    "list_entities",',
			'    "list_records",',
			'    "main",',
			'    "validate_application",',
			']',
			'',
			'try:',
			'    from .ai_agents import (',
			'        get_agent,',
			'        get_team,',
			'        list_agent_runtimes,',
			'        list_agent_teams,',
			'        list_agents,',
			'        list_teams,',
			'        validate_agent_runtimes,',
			'    )',
			'except ImportError:',
			'    pass',
			'else:',
			'    __all__.extend([',
			'        "get_agent",',
			'        "get_team",',
			'        "list_agent_runtimes",',
			'        "list_agent_teams",',
			'        "list_agents",',
			'        "list_teams",',
			'        "validate_agent_runtimes",',
			'    ])',
			'',
			'try:',
			'    from .apg_capabilities import (',
			'        capability_dependency_graph,',
			'        capability_load_order,',
			'        describe_capabilities,',
			'        describe_capabilities_by_erp_module,',
			'        describe_capability,',
			'        capability_names_by_erp_module,',
			'        composition_graph,',
			'        get_capability,',
			'        list_capabilities,',
			'        streaming_processor_index,',
			'        ui_route_index,',
			'    )',
			'except ImportError:',
			'    pass',
			'else:',
			'    __all__.extend([',
			'        "capability_dependency_graph",',
			'        "capability_load_order",',
			'        "describe_capabilities",',
			'        "describe_capabilities_by_erp_module",',
			'        "describe_capability",',
			'        "capability_names_by_erp_module",',
			'        "composition_graph",',
			'        "get_capability",',
			'        "list_capabilities",',
			'        "streaming_processor_index",',
			'        "ui_route_index",',
			'    ])',
		])
		
		return '\n'.join(lines)
	

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
