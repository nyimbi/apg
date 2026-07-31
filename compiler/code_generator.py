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
import json
import sys

# Import AST nodes
from .ast_builder import (
	ModuleDeclaration, Expression,
	LiteralExpression, IdentifierExpression, BinaryExpression, CallExpression,
	UnaryExpression, MemberExpression, IndexExpression, ListExpression,
	DictExpression,
	AIAgentDeclaration, AgentTeamDeclaration, ApplicationDeclaration, CapabilityDeclaration,
	DatabaseDeclaration, EntityType
)
from .semantic_model import build_semantic_model_from_module

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
			files = self._generate_with_composable_templates(ast)
		else:
			files = self._generate_python_application(ast)
		files.update(self._load_static_assets())
		files.update(self._generate_pwa_assets(ast, files))
		return files
	
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
			generated_files.update(self._generate_semantic_model_files(ast))
			generated_files.update(self._generate_ai_agent_files(ast))
			generated_files.update(self._generate_application_files(ast))
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
		"""Generate core dependency-free Python application artifacts.

		Produces only the portable, target-agnostic files.  Deployment-bundle
		artefacts (Dockerfile, README, smoke_test.py, etc.) are added by
		:meth:`generate_deployment_bundle` and are intentionally kept separate
		so that callers such as the test suite can control which set they need.
		"""
		files = {
			"app.py": self._generate_python_app(ast),
			"__init__.py": self._generate_package_init(ast),
			"requirements.txt": self._generate_python_requirements(),
			"README.md": self._generate_python_readme(ast),
		}
		files.update(self._generate_ai_agent_files(ast))
		files.update(self._generate_application_files(ast))
		files.update(self._generate_capability_files(ast))
		# Phase 6: typed stub classes for agents
		agent_stubs = self._generate_agent_stubs(ast)
		if agent_stubs:
			files["agent_stubs.py"] = agent_stubs
		return files

	def generate_deployment_bundle(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Return deployment-bundle artefacts that supplement the core application.

		These files are shipping/ops concerns (Dockerfile, README, smoke tests,
		environment example, semantic model snapshot) that are added on top of
		the core files produced by :meth:`_generate_python_application`.
		``APGCompiler`` calls this after generation so that unit tests that use
		``PythonCodeGenerator`` directly receive only the predictable core set.
		"""
		return {
			"semantic_model.json": self._generate_semantic_model_json(ast),
			".dockerignore": self._generate_python_dockerignore(),
			".env.example": self._generate_python_env_example(ast),
			"Dockerfile": self._generate_python_dockerfile(ast),
			"README.md": self._generate_python_readme(ast),
			"smoke_test.py": self._generate_python_smoke_test(),
		}

	def _generate_agent_stubs(self, ast: ModuleDeclaration) -> str:
		"""Generate typed Python stub classes for declared AI agents (Phase 6).

		Each agent gets an AgentBase subclass with its declared metadata
		and an async invoke() that routes to the configured runtime via the
		APG agent adapter protocol.
		"""
		agents = [e for e in ast.entities if isinstance(e, AIAgentDeclaration)]
		if not agents:
			return ""

		lines = [
			'"""Typed agent stub classes generated from APG agent declarations.',
			"",
			"Each class wraps the agent metadata and provides an async invoke()",
			"that delegates to the declared runtime via the APG adapter protocol.",
			'"""',
			"",
			"from __future__ import annotations",
			"",
			"import asyncio",
			"import json",
			"import os",
			"import shlex",
			"import subprocess",
			"from typing import Any, Optional",
			"",
			"",
			"class AgentContext:",
			'    """Runtime context for an agent invocation."""',
			"    def __init__(self, tenant_id: str = 'default', user_id: str = 'anonymous',",
			"                 session_id: str = '', **kwargs: Any) -> None:",
			"        self.tenant_id = tenant_id",
			"        self.user_id = user_id",
			"        self.session_id = session_id",
			"        self.metadata = kwargs",
			"",
			"",
			"class AgentBase:",
			'    """Base class for APG agent stubs."""',
			"    name: str = ''",
			"    role: str = ''",
			"    model: str = ''",
			"    runtime: str = 'codex'",
			"",
			"    async def invoke(self, prompt: str, context: Optional[AgentContext] = None) -> str:",
			"        env_key = f'APG_AGENT_{self.runtime.upper()}_PROVIDER_COMMAND'",
			"        cmd = os.environ.get(env_key) or os.environ.get('APG_AGENT_PROVIDER_COMMAND')",
			"        if not cmd:",
			"            raise RuntimeError(",
			"                f'Agent {self.name!r}: no provider command configured. '",
			"                f'Set {env_key} to wire up the {self.runtime} runtime.'",
			"            )",
			"        payload = {",
			"            'agent': {'name': self.name, 'role': self.role, 'model': self.model},",
			"            'input': prompt,",
			"            'context': {",
			"                'tenant_id': getattr(context, 'tenant_id', 'default'),",
			"                'user_id': getattr(context, 'user_id', 'anonymous'),",
			"            } if context else {},",
			"        }",
			"        result = await asyncio.to_thread(",
			"            subprocess.run, shlex.split(cmd),",
			"            input=json.dumps(payload), capture_output=True, text=True, timeout=120",
			"        )",
			"        out = result.stdout.strip()",
			"        try:",
			"            return json.loads(out).get('output', out)",
			"        except Exception:",
			"            return out",
			"",
			"",
		]

		import re as _re

		for agent in agents:
			cls_name = agent.name
			runtime = agent.runtime or "codex"

			# Validate that cls_name and runtime are safe Python identifiers
			# before interpolating them into generated source code.
			if not _re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', cls_name):
				raise ValueError(
					f"Agent name {cls_name!r} is not a valid Python identifier"
				)
			if not _re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', runtime):
				raise ValueError(
					f"Agent runtime {runtime!r} is not a valid Python identifier"
				)

			# Use repr() for ALL user-controlled string fields to prevent
			# code injection via newlines, backslashes, or triple-quote sequences.
			role = repr(agent.role or "")
			model = repr(agent.model or "")
			system = repr((agent.system_prompt or "")[:100])
			runtime_r = repr(runtime)
			caps = repr(tuple(agent.capabilities))
			tools = repr(tuple(agent.tools))

			lines += [
				f"class {cls_name}(AgentBase):",
				f"    name = {cls_name!r}",
				f"    role = {role}",
				f"    model = {model}",
				f"    runtime = {runtime_r}",
				f"    system = {system}",
				f"    capabilities = {caps}",
				f"    tools = {tools}",
				"",
				"",
			]

		lines += [
			"# Registry of all declared agents",
			"AGENTS = {",
		]
		for agent in agents:
			lines.append(f"    {agent.name!r}: {agent.name},")
		lines += ["}", ""]

		return "\n".join(lines)

	def _generate_semantic_model_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate the compiler semantic model artifact shipped with apps."""
		return {"semantic_model.json": self._generate_semantic_model_json(ast)}

	def _generate_semantic_model_json(self, ast: ModuleDeclaration) -> str:
		"""Generate deterministic apg.semantic-model.v1 JSON."""
		model = build_semantic_model_from_module(ast, f"{ast.name}.apg")
		return json.dumps(model, indent=2, sort_keys=True) + "\n"

	def _expression_value(self, expr: Any) -> Any:
		"""Convert simple AST expressions into serializable runtime values."""
		if expr is None:
			return None
		if isinstance(expr, LiteralExpression):
			if expr.literal_type == "string" and isinstance(expr.value, str):
				value = expr.value.strip()
				if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
					return value[1:-1]
			return expr.value
		if isinstance(expr, IdentifierExpression):
			return expr.name
		if isinstance(expr, ListExpression):
			return [self._expression_value(element) for element in expr.elements]
		if isinstance(expr, DictExpression):
			return {
				self._expression_value(key): self._expression_value(value)
				for key, value in expr.pairs
			}
		return self._generate_expression(expr)

	def _entity_specs(self, module: ModuleDeclaration) -> List[Dict[str, Any]]:
		"""Convert APG entities into generated runtime metadata."""
		enum_values_by_name = {
			entity.name: list(getattr(entity, "values", []))
			for entity in module.entities
			if getattr(entity, "entity_type", None) == EntityType.ENUM
		}
		entities = [
			entity
			for entity in module.entities
			if not self._is_security_config_entity(entity)
			and getattr(entity, "entity_type", None) != EntityType.ENUM
		]
		junction_context: Dict[str, list[tuple[str, str]]] = {}
		for entity in entities:
			for relationship in getattr(entity, "relationships", []):
				if (
					getattr(relationship, "kind", "") == "has_many"
					and getattr(relationship, "through", None)
					and getattr(relationship, "target", None)
				):
					junction_context.setdefault(str(relationship.through), []).append(
						(entity.name, str(relationship.target))
					)
		return [
			self._entity_spec(entity, junction_context.get(entity.name, []), enum_values_by_name)
			for entity in entities
		]

	@staticmethod
	def _relationship_field_name(entity_name: str) -> str:
		"""Return the conventional FK field name for a relationship target."""
		import re as _re
		text = _re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(entity_name))
		text = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
		return text.replace("-", "_").lower() + "_id"

	@staticmethod
	def _relationship_segment(entity_name: str) -> str:
		"""Return a stable lowercase nested resource segment."""
		base = PythonCodeGenerator._relationship_field_name(entity_name)
		base = base[:-3] if base.endswith("_id") else base
		if base.endswith("y") and (len(base) == 1 or base[-2] not in "aeiou"):
			return base[:-1] + "ies"
		if base.endswith(("s", "x", "z", "ch", "sh")):
			return base + "es"
		return base + "s"

	def _entity_spec(
		self,
		entity: Any,
		junction_pairs: list[tuple[str, str]] | None = None,
		enum_values_by_name: Dict[str, list[str]] | None = None,
	) -> Dict[str, Any]:
		"""Convert an APG entity AST node into generated runtime metadata."""
		enum_values_by_name = enum_values_by_name or {}

		def field_spec(property: Any) -> Dict[str, Any]:
			type_name = property.type_annotation.type_name if property.type_annotation else "any"
			is_computed = bool(getattr(property, "is_computed", False))
			spec = {
				"name": property.name,
				"type": type_name,
				"required": False if is_computed else property.is_required,
			}
			if is_computed:
				spec["computed"] = True
				spec["expression"] = str(
					getattr(property, "computed_expression", None)
					or property.default_value
					or ""
				).strip()
			elif property.default_value is not None:
				spec["default"] = self._expression_value(property.default_value)
			if type_name in enum_values_by_name:
				spec["enum"] = list(enum_values_by_name[type_name])
			validators = []
			for rule in getattr(property, "validation_rules", []):
				validator = {"rule": str(getattr(rule, "rule_type", ""))}
				validator.update(dict(getattr(rule, "parameters", {}) or {}))
				validators.append(validator)
			if validators:
				spec["validators"] = validators
			return spec

		def add_generated_field(fields: list[Dict[str, Any]], generated: Dict[str, Any]) -> None:
			field_name = str(generated["name"])
			for field in fields:
				if str(field.get("name")) == field_name:
					field.setdefault("relationship", generated.get("relationship", {}))
					field.setdefault("generated", generated.get("generated", True))
					return
			fields.append(generated)

		fields = [field_spec(property) for property in entity.properties]
		relationships: list[Dict[str, Any]] = []
		for relationship in getattr(entity, "relationships", []):
			kind = str(getattr(relationship, "kind", ""))
			target = str(getattr(relationship, "target", ""))
			through = getattr(relationship, "through", None)
			relationship_spec = {
				"kind": kind,
				"target": target,
				"through": str(through) if through else None,
				"segment": self._relationship_segment(target),
			}
			if kind == "belongs_to" and target:
				fk_field = self._relationship_field_name(target)
				relationship_spec["fk_field"] = fk_field
				add_generated_field(fields, {
					"name": fk_field,
					"type": "int",
					"required": False,
					"generated": True,
					"relationship": {
						"kind": kind,
						"target": target,
						"on_delete": "SET NULL",
					},
				})
			elif kind == "has_many" and target:
				relationship_spec["fk_field"] = self._relationship_field_name(entity.name)
				if through:
					relationship_spec["left_field"] = "left_id"
					relationship_spec["right_field"] = "right_id"
			elif kind == "has_one" and target:
				relationship_spec["fk_field"] = self._relationship_field_name(entity.name)
			relationships.append(relationship_spec)

		for left_name, right_name in junction_pairs or []:
			add_generated_field(fields, {
				"name": "left_id",
				"type": "int",
				"required": False,
				"generated": True,
				"relationship": {
					"kind": "junction_left",
					"target": left_name,
					"on_delete": "CASCADE",
				},
			})
			add_generated_field(fields, {
				"name": "right_id",
				"type": "int",
				"required": False,
				"generated": True,
				"relationship": {
					"kind": "junction_right",
					"target": right_name,
					"on_delete": "CASCADE",
				},
			})

		spec: Dict[str, Any] = {
			"name": entity.name,
			"type": entity.entity_type.value,
			"properties": [property.name for property in entity.properties],
			"fields": fields,
			"methods": [method.name for method in entity.methods],
		}
		if relationships:
			spec["relationships"] = relationships
		if isinstance(entity, DatabaseDeclaration):
			spec["connection_config"] = dict(entity.connection_config)
			spec["schemas"] = [
				{
					"name": schema.name,
					"tables": [
						{
							"name": table.name,
							"columns": [
								{
									"name": column.name,
									"type": column.data_type,
									"primary_key": column.is_primary_key,
									"nullable": column.is_nullable,
									"default": column.default_value,
									"constraints": list(column.constraints),
									**({"reference": dict(column.reference)} if column.reference else {}),
								}
								for column in table.columns
							],
							"indexes": [
								{
									"name": index.name,
									"columns": list(index.columns),
									"unique": index.is_unique,
									"type": index.index_type,
								}
								for index in table.indexes
							],
						}
						for table in schema.tables
					],
				}
				for schema in entity.schemas
			]
		return spec

	@staticmethod
	def _is_security_config_entity(entity: Any) -> bool:
		return str(getattr(entity, "name", "")).lower() == "security"

	@staticmethod
	def _module_requires_auth(module: "ModuleDeclaration") -> bool:
		for entity in module.entities:
			if not PythonCodeGenerator._is_security_config_entity(entity):
				continue
			for prop in getattr(entity, "properties", []):
				if str(getattr(prop, "name", "")).lower() not in {"authentication", "auth"}:
					continue
				type_name = str(getattr(getattr(prop, "type_annotation", None), "type_name", "")).lower()
				default_value = str(getattr(prop, "default_value", "") or "").strip().strip('"').strip("'").lower()
				value = default_value or type_name
				if value in {"required", "enabled", "true", "jwt", "session"}:
					return True
		return False

	@staticmethod
	def _module_i18n_config(module: "ModuleDeclaration") -> Dict[str, Any]:
		languages: list[str] = ["en", "sw", "fr", "ar"]
		default_language = "en"
		fallback_language = "en"
		for entity in module.entities:
			i18n = getattr(entity, "i18n", None)
			if not isinstance(i18n, dict):
				continue
			raw_supported = i18n.get("supported_languages", [])
			if isinstance(raw_supported, list):
				for language in raw_supported:
					code = str(language).strip().strip('"').strip("'")
					if code and code not in languages:
						languages.append(code)
			raw_default = i18n.get("default_language")
			if raw_default:
				default_language = str(raw_default).strip().strip('"').strip("'")
			raw_fallback = i18n.get("fallback_language")
			if raw_fallback:
				fallback_language = str(raw_fallback).strip().strip('"').strip("'")
		if default_language not in languages:
			default_language = languages[0]
		if fallback_language not in languages:
			fallback_language = "en" if "en" in languages else default_language
		return {
			"supported_languages": languages,
			"default_language": default_language,
			"fallback_language": fallback_language,
		}

	@staticmethod
	def _chrome_i18n_catalog(languages: list[str]) -> Dict[str, Dict[str, str]]:
		required = (
			"save",
			"cancel",
			"delete",
			"confirm",
			"error",
			"success",
			"loading",
			"no_records",
			"search",
			"login",
			"logout",
		)
		english = {
			"cancel": "Cancel",
			"home": "Home",
			"workflows": "Workflows",
			"marketplace": "Marketplace",
			"theme_system": "System",
			"language": "Language",
			"login": "Login",
			"logout": "Logout",
			"sign_in": "Sign in",
			"open_app": "Open App",
			"api_docs": "API Docs",
			"data_entities": "Data Entities",
			"view_manifest": "View Manifest",
			"entities": "Entities",
			"capabilities": "Capabilities",
			"confirm": "Confirm",
			"delete": "Delete",
			"error": "Error",
			"loading": "Loading",
			"no_records": "No records",
			"records": "Records",
			"save": "Save",
			"search": "Search",
			"success": "Success",
			"ai_agents": "AI Agents",
		}
		placeholder = {key: "" for key in required}
		overrides = {"sw": placeholder, "fr": placeholder, "ar": placeholder}
		return {
			language: {**english, **overrides.get(language, {})}
			for language in languages
		}

	@staticmethod
	def _landing_style_for(module: "ModuleDeclaration") -> str:
		"""Derive a landing page style from the APG module's theme declaration."""
		for entity in module.entities:
			theme_name = getattr(entity, "name", "") or ""
			if "africa" in theme_name.lower():
				return "africa"
			if "corporate" in theme_name.lower() or "enterprise" in theme_name.lower():
				return "corporate"
			if "minimal" in theme_name.lower() or "simple" in theme_name.lower():
				return "minimal"
		return "default"

	@staticmethod
	def _load_ui_templates() -> dict[str, str]:
		"""Load Jinja2 UI templates from compiler/templates/ to embed in generated app."""
		tmpl_dir = Path(__file__).parent / "templates"
		templates: dict[str, str] = {}
		if tmpl_dir.exists():
			for f in tmpl_dir.rglob("*.j2"):
				# Use path relative to tmpl_dir as key so subdirs work with include
				key = f.relative_to(tmpl_dir).as_posix()
				templates[key] = f.read_text(encoding="utf-8")
		return templates

	@staticmethod
	def _load_static_assets() -> dict[str, str]:
		"""Load vendored UI assets to emit into generated static/ output."""
		asset_dir = Path(__file__).parent / "assets"
		asset_files = ("apg.css", "htmx.min.js", "sortable.min.js", "uplot.min.js", "uplot.min.css", "apg-charts.js", "apg-sse.js")
		assets: dict[str, str] = {}
		for asset_name in asset_files:
			asset_path = asset_dir / asset_name
			if not asset_path.is_file():
				raise FileNotFoundError(f"Missing generated UI asset: {asset_path}")
			assets[f"static/{asset_name}"] = asset_path.read_text(encoding="utf-8")
		return assets

	def _generate_pwa_assets(self, module: ModuleDeclaration, files: Dict[str, str]) -> Dict[str, str]:
		"""Generate self-contained PWA metadata for the compiled UI."""
		primary = "#1E5B5A"
		accent = "#D97706"
		for entity in module.entities:
			theme = getattr(entity, "theme", None)
			if not isinstance(theme, dict):
				continue
			tokens = theme.get("tokens", theme)
			if isinstance(tokens, dict):
				primary = str(tokens.get("color.primary") or tokens.get("primary") or tokens.get("brand") or primary)
				accent = str(tokens.get("color.accent") or tokens.get("accent") or accent)
				break
		app_name = module.name.replace("_", " ").replace("-", " ").title()
		static_urls = sorted(
			f"/{path}"
			for path in files
			if path.startswith("static/") and path not in {"static/manifest.webmanifest", "static/sw.js"}
		)
		manifest = {
			"id": "/ui",
			"name": app_name,
			"short_name": module.name[:12] or "APG",
			"description": module.description or f"{app_name} generated by APG",
			"start_url": "/ui",
			"scope": "/",
			"display": "standalone",
			"orientation": "any",
			"categories": ["business", "productivity"],
			"background_color": "#ffffff",
			"theme_color": primary,
			"shortcuts": [
				{"name": "Dashboard", "url": "/ui", "description": "Open the generated dashboard"},
				{"name": "Workflows", "url": "/ui/workflows", "description": "Open generated workflows"},
				{"name": "Marketplace", "url": "/ui/marketplace", "description": "Open connector marketplace"},
			],
			"icons": [
				{"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"},
			],
		}
		icon_svg = (
			'<svg viewBox="0 0 512 512" role="img" aria-label="APG application icon">'
			f'<rect width="512" height="512" rx="96" fill="{primary}"/>'
			f'<path d="M112 344 256 88l144 256h-72l-28-56h-90l-28 56h-70Z" fill="{accent}"/>'
			'<path d="M238 232h36l-18-40-18 40Z" fill="#fff"/>'
			'</svg>'
		)
		sw = (
			"const APG_CACHE='apg-ui-v2';\n"
			f"const APG_STATIC={json.dumps(static_urls + ['/ui'], sort_keys=True)};\n"
			"self.addEventListener('install',event=>{event.waitUntil(caches.open(APG_CACHE).then(cache=>cache.addAll(APG_STATIC)).then(()=>self.skipWaiting()))});\n"
			"self.addEventListener('activate',event=>{event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==APG_CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim()))});\n"
			"self.addEventListener('message',event=>{if(event.data&&event.data.type==='SKIP_WAITING')self.skipWaiting()});\n"
			"self.addEventListener('fetch',event=>{const req=event.request;if(req.method!=='GET'||new URL(req.url).origin!==location.origin)return;"
			"event.respondWith(fetch(req).then(res=>{const copy=res.clone();if(res.ok){caches.open(APG_CACHE).then(cache=>cache.put(req,copy))}return res}).catch(()=>caches.match(req).then(cached=>cached||caches.match('/ui'))))});\n"
		)
		return {
			"static/manifest.webmanifest": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
			"static/icon.svg": icon_svg + "\n",
			"static/sw.js": sw,
		}

	def _generate_python_app(self, module: ModuleDeclaration) -> str:
		"""Generate a framework-neutral Python app.py entrypoint."""
		auth_required = self._module_requires_auth(module)
		i18n_config = self._module_i18n_config(module)
		i18n_catalog = self._chrome_i18n_catalog(i18n_config["supported_languages"])
		entity_specs = self._entity_specs(module)
		semantic_model = build_semantic_model_from_module(module, f"{module.name}.apg")
		ui_templates = self._load_ui_templates()
		# Derive landing style from theme name or default to "default"
		landing_style = self._landing_style_for(module)
		cmd_palette_literal = (
			'<div id="apg-cmd" class="hidden fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" onclick="if(event.target===this)apgCmdClose()">'
			'<div class="mx-auto mt-[15vh] max-w-xl bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden">'
			'<div class="flex items-center gap-3 px-4 py-3 border-b border-gray-100">'
			'<svg class="w-4 h-4 text-gray-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">'
			'<path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9 a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clip-rule="evenodd"/>'
			'</svg>'
			'<input id="apg-cmd-input" type="text" placeholder="Search records, entities..." autocomplete="off" class="flex-1 text-sm outline-none placeholder-gray-400" oninput="apgCmdSearch(this.value)">'
			'<kbd class="text-xs text-gray-400 border border-gray-200 rounded px-1.5 py-0.5">Esc</kbd>'
			'</div>'
			'<div id="apg-cmd-results" class="max-h-80 overflow-y-auto py-2"><p class="text-xs text-gray-400 text-center py-8">Type to search...</p></div>'
			'</div></div>'
			'<script>'
			'function apgClearChildren(el){while(el&&el.firstChild)el.removeChild(el.firstChild);}'
			'function apgCmdSetMessage(text){var el=document.getElementById("apg-cmd-results");if(!el)return;apgClearChildren(el);var p=document.createElement("p");p.className="text-xs text-gray-400 text-center py-8";p.textContent=text;el.appendChild(p);}'
			'function apgCmdResultNode(r){var entity=String(r&&r.entity||"");var id=String(r&&r.id||"");var a=document.createElement("a");a.href="/ui/entities/"+encodeURIComponent(entity)+"/"+encodeURIComponent(id);a.onclick=function(){apgCmdClose();};a.className="flex items-center gap-3 px-4 py-2.5 hover:bg-gray-50 transition-colors group";var badge=document.createElement("span");badge.className="w-6 h-6 rounded-md bg-blue-50 flex items-center justify-center text-xs font-bold text-blue-600 flex-shrink-0";badge.textContent=(entity.charAt(0)||"?").toUpperCase();var wrap=document.createElement("div");wrap.className="min-w-0";var title=document.createElement("p");title.className="text-sm font-medium text-gray-900 truncate";title.textContent=String(r&&r.label||"");var meta=document.createElement("p");meta.className="text-xs text-gray-400 truncate";meta.textContent=entity;wrap.appendChild(title);wrap.appendChild(meta);a.appendChild(badge);a.appendChild(wrap);return a;}'
			'document.addEventListener("keydown",function(e){if((e.metaKey||e.ctrlKey)&&e.key==="k"){e.preventDefault();apgCmdOpen();}if(e.key==="Escape")apgCmdClose();});'
			'function apgCmdOpen(){document.getElementById("apg-cmd").classList.remove("hidden");document.getElementById("apg-cmd-input").focus();}'
			'function apgCmdClose(){document.getElementById("apg-cmd").classList.add("hidden");document.getElementById("apg-cmd-input").value="";apgCmdSetMessage("Type to search...");}'
			'var _cmdTimer;function apgCmdSearch(q){clearTimeout(_cmdTimer);if(!q.trim()){apgCmdSetMessage("Type to search...");return;}_cmdTimer=setTimeout(function(){fetch("/api/search?q="+encodeURIComponent(q)).then(function(r){return r.json();}).then(function(d){var el=document.getElementById("apg-cmd-results");if(!el)return;if(!d.results||!d.results.length){apgCmdSetMessage("No results");return;}apgClearChildren(el);d.results.forEach(function(r){el.appendChild(apgCmdResultNode(r));});});},200);}'
			'</script>'
		)
		return f'''"""
{module.name} - APG Python Application
{"=" * (len(module.name) + 25)}

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import base64
import collections
import gzip
import importlib
import hashlib
import html
import hmac
import json
import datetime as _datetime
import logging as _logging
import mimetypes as _mimetypes
import os
import queue as _queue
import re
import secrets
import smtplib as _smtplib
import sqlite3 as _sqlite3
import sys
import threading
import threading as _threading
import time as _time
import urllib.request as _urllib_request, hmac as _hmac_mod, hashlib as _hashlib_mod, threading as _threading_mod, time as _time_mod
import uuid as _uuid
from email.message import EmailMessage as _EmailMessage
from flask import Flask as _FlaskApp, request as _flask_request, redirect as _flask_redirect, Response as _FlaskResponse, session as _flask_session, g as _flask_g
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote, unquote, urlencode


MODULE_NAME = {module.name!r}
MODULE_VERSION = {module.version!r}
MODULE_DESCRIPTION = {module.description!r}
APG_APP_NAME = MODULE_NAME
APG_APP_VERSION = MODULE_VERSION
APG_APP_DESCRIPTION = MODULE_DESCRIPTION or ""
LANDING_STYLE = {landing_style!r}
ENTITIES = {entity_specs!r}
_APG_ENTITIES = ENTITIES
ENTITY_NAMES = {{entity["name"] for entity in _APG_ENTITIES}}
RECORD_STORE: Dict[str, list[Dict[str, Any]]] = {{entity["name"]: [] for entity in ENTITIES}}
RECORD_METADATA: Dict[str, Dict[str, Dict[str, Any]]] = {{entity["name"]: {{}} for entity in ENTITIES}}
NEXT_RECORD_IDS: Dict[str, int] = {{entity["name"]: 1 for entity in ENTITIES}}
EVENT_LOG: list[Dict[str, Any]] = []
NEXT_EVENT_ID = 1
WORKFLOW_RUNS: Dict[str, Dict[str, Any]] = {{}}
NEXT_WORKFLOW_RUN_ID = 1
CIRCUIT_BREAKERS: Dict[str, Dict[str, Any]] = {{}}
APG_EVENT_SUBSCRIPTIONS: Dict[str, list[str]] = {{}}
APG_CONNECTOR_REGISTRY: list[Dict[str, Any]] = []
APG_ACTIVITY_LOG: Dict[str, list[Dict[str, Any]]] = {{}}
WORKFLOW_EVENT_JOURNAL: Dict[str, list[Dict[str, Any]]] = {{}}
WORKFLOW_SIGNALS: Dict[str, list[str]] = {{}}
APG_RECORD_LOCK = _threading.RLock()
APG_LIVE_LOCK = _threading.Lock()
APG_LIVE_SUBSCRIBERS: list[Dict[str, Any]] = []
_APG_JOB_QUEUE = collections.deque()
_APG_JOB_LOCK = threading.Lock()
_APG_JOB_HANDLERS: Dict[str, Any] = {{}}
_APG_JOB_WORKERS: list[threading.Thread] = []
_APG_JOB_WORKERS_STARTED = False
APG_MULTI_TENANT_ENABLED = str(os.environ.get("APG_MULTI_TENANT", "")).strip().lower() in {{"1", "true", "yes", "on"}}
APG_TENANT_DEFAULT = "default"
APG_TENANT_HEADER_DEFAULT = "X-APG-Tenant"
TENANT_SCOPED_ENTITIES: set[str] = {{
    e["name"] for e in ENTITIES
    if APG_MULTI_TENANT_ENABLED or any(str(f.get("name")) == "tenant_id" for f in e.get("fields", []))
}}
SEMANTIC_MODEL: Dict[str, Any] = {semantic_model!r}
APG_UI_TEMPLATES: Dict[str, str] = {ui_templates!r}
APG_AUTH_REQUIRED = {auth_required!r}
APG_SUPPORTED_LANGUAGES: list[str] = {i18n_config["supported_languages"]!r}
APG_DEFAULT_LANGUAGE = {i18n_config["default_language"]!r}
APG_FALLBACK_LANGUAGE = {i18n_config["fallback_language"]!r}
_APG_STRINGS: Dict[str, Dict[str, str]] = {i18n_catalog!r}
APG_I18N: Dict[str, Dict[str, str]] = _APG_STRINGS
_APG_OPENAPI_SPEC: Dict[str, Any] | None = None
_APG_FIELD_ACL = json.loads(os.environ.get("APG_FIELD_ACL", "{{}}"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {{"1", "true", "yes", "on"}}


def _production_mode() -> bool:
    return _env_flag("APG_PRODUCTION") or str(os.environ.get("APG_ENV", "")).strip().lower() in {{"prod", "production"}}


def _apg_production_env_enabled() -> bool:
    return str(os.environ.get("APG_PRODUCTION", "")).strip() == "1"


def _configured_session_secret() -> str:
    return str(os.environ.get("APG_SECRET_KEY") or os.environ.get("APG_SESSION_SECRET") or os.environ.get("APG_JWT_SECRET") or "")


def _generated_session_secret() -> str:
    configured = _configured_session_secret()
    if configured:
        return configured
    if _apg_production_env_enabled():
        return "dev-secret-key-change-me"
    return secrets.token_urlsafe(48)


def _session_cookie_samesite() -> str:
    value = str(os.environ.get("APG_SESSION_COOKIE_SAMESITE", "Lax")).strip()
    normalized = value[:1].upper() + value[1:].lower() if value else "Lax"
    return normalized if normalized in {{"Lax", "Strict", "None"}} else "Lax"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


_APG_JOB_WORKER_THREADS = max(1, _env_int("APG_WORKER_THREADS", 2))
_APG_JOB_MAX_RETRIES = max(1, _env_int("APG_JOB_MAX_RETRIES", 3))
_APG_EMAIL_THREADS: list[_threading_mod.Thread] = []


def _apg_send_email(to: str, subject: str, body: str) -> None:
    recipient = str(to or "").strip()
    smtp_host = str(os.environ.get("APG_SMTP_HOST", "") or "").strip()
    if not recipient or not smtp_host:
        return
    sender = str(
        os.environ.get("APG_SMTP_FROM")
        or os.environ.get("APG_SMTP_USER")
        or "apg@localhost"
    )
    smtp_port = _env_int("APG_SMTP_PORT", 587)
    smtp_user = str(os.environ.get("APG_SMTP_USER", "") or "")
    smtp_password = str(os.environ.get("APG_SMTP_PASSWORD", "") or "")
    message = _EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = str(subject)
    message.set_content(str(body))

    def _send() -> None:
        try:
            with _smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
                smtp.starttls()
                if smtp_user or smtp_password:
                    smtp.login(smtp_user, smtp_password)
                smtp.sendmail(sender, [recipient], message.as_string())
            _logging.getLogger("apg").info("email_sent to=%s subject=%s", recipient, subject)
        except Exception as exc:
            _logging.getLogger("apg").warning("email_send_failed to=%s err=%s", recipient, exc)

    thread = _threading_mod.Thread(target=_send, daemon=True)
    _APG_EMAIL_THREADS.append(thread)
    thread.start()


def _apg_alert_startup_failure(message: str) -> None:
    recipient = str(os.environ.get("APG_ALERT_EMAIL", "") or "").strip()
    if recipient:
        _apg_send_email(
            recipient,
            f"{{APG_APP_NAME}} startup validation failed",
            str(message),
        )


def _validate_startup_configuration() -> None:
    try:
        if not _apg_production_env_enabled():
            return
        if _flask_app.secret_key == "dev-secret-key-change-me":
            raise RuntimeError("Set APG_SECRET_KEY in production")
        if _flask_app.config.get("SESSION_COOKIE_SECURE") is False:
            _logging.getLogger("apg").warning("APG_PRODUCTION is enabled but SESSION_COOKIE_SECURE is false.")
    except RuntimeError as exc:
        if _apg_production_env_enabled():
            _apg_alert_startup_failure(str(exc))
            raise
    except Exception as exc:
        _logging.getLogger("apg").warning("APG startup validation skipped: %s", exc)


_APG_SCRYPT_N = 2**17
_APG_SCRYPT_R = 8
_APG_SCRYPT_P = 1
_APG_PBKDF2_ITERATIONS = 600000
_APG_MAX_PASSWORD_BYTES = 1024
_APG_SCRYPT_MAXMEM = 256 * 1024 * 1024


def hash_password(password: str, *, scheme: str = "scrypt", iterations: int | None = None, n: int | None = None, r: int | None = None, p: int | None = None) -> str:
    """Hash a password for APG_AUTH_USERS `password_hash` entries.

    scrypt (memory-hard) is the default; pbkdf2_sha256 is supported for
    imported credentials. Both verify with the standard library only.
    """
    password_bytes = str(password).encode("utf-8")
    if len(password_bytes) > _APG_MAX_PASSWORD_BYTES:
        raise ValueError("password exceeds the maximum supported length")
    salt = secrets.token_bytes(16)
    if scheme == "pbkdf2_sha256":
        rounds = int(iterations or _APG_PBKDF2_ITERATIONS)
        digest = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, rounds)
        return f"pbkdf2_sha256${{rounds}}${{salt.hex()}}${{digest.hex()}}"
    if scheme != "scrypt":
        raise ValueError(f"unsupported password hash scheme: {{scheme}}")
    cost_n = int(n or _APG_SCRYPT_N)
    cost_r = int(r or _APG_SCRYPT_R)
    cost_p = int(p or _APG_SCRYPT_P)
    digest = hashlib.scrypt(password_bytes, salt=salt, n=cost_n, r=cost_r, p=cost_p, maxmem=_APG_SCRYPT_MAXMEM, dklen=32)
    return f"scrypt${{cost_n}}${{cost_r}}${{cost_p}}${{salt.hex()}}${{digest.hex()}}"


def _verify_password_hash(stored: str, password: str) -> bool:
    password_bytes = str(password).encode("utf-8")
    if len(password_bytes) > _APG_MAX_PASSWORD_BYTES:
        return False
    try:
        parts = str(stored).split("$")
        if parts[0] == "scrypt" and len(parts) == 6:
            cost_n, cost_r, cost_p = int(parts[1]), int(parts[2]), int(parts[3])
            salt = bytes.fromhex(parts[4])
            expected = bytes.fromhex(parts[5])
            digest = hashlib.scrypt(password_bytes, salt=salt, n=cost_n, r=cost_r, p=cost_p, maxmem=_APG_SCRYPT_MAXMEM, dklen=len(expected))
            return hmac.compare_digest(digest, expected)
        if parts[0] == "pbkdf2_sha256" and len(parts) == 4:
            rounds = int(parts[1])
            salt = bytes.fromhex(parts[2])
            expected = bytes.fromhex(parts[3])
            digest = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, rounds)
            return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError, IndexError):
        return False
    return False


_APG_DUMMY_PASSWORD_HASH: str | None = None


def _dummy_password_verify() -> None:
    """Burn KDF-equivalent time so unknown users are indistinguishable from bad passwords."""
    global _APG_DUMMY_PASSWORD_HASH
    if _APG_DUMMY_PASSWORD_HASH is None:
        _APG_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(16), scheme="pbkdf2_sha256")
    _verify_password_hash(_APG_DUMMY_PASSWORD_HASH, "apg-dummy-verification")


# Per-process sliding-window login throttle. For multi-worker deployments,
# replace with a shared store (e.g. Redis) keyed the same way.
_APG_LOGIN_ATTEMPTS: Dict[str, list[float]] = {{}}
_APG_LOGIN_LOCK = _threading.Lock()


def _login_throttle_settings() -> tuple[int, float]:
    max_attempts = max(1, _env_int("APG_LOGIN_MAX_ATTEMPTS", 5))
    window = float(max(1, _env_int("APG_LOGIN_WINDOW_SECONDS", 300)))
    return max_attempts, window


def _login_throttle_key(username: str) -> str:
    try:
        remote = _flask_request.remote_addr or "unknown"
    except RuntimeError:
        remote = "unknown"
    return f"{{username}}|{{remote}}"


def _login_retry_after(key: str) -> int:
    """Seconds until the next attempt is allowed; 0 when not throttled."""
    max_attempts, window = _login_throttle_settings()
    now = _time.monotonic()
    with _APG_LOGIN_LOCK:
        attempts = [stamp for stamp in _APG_LOGIN_ATTEMPTS.get(key, []) if now - stamp < window]
        _APG_LOGIN_ATTEMPTS[key] = attempts
        if len(attempts) < max_attempts:
            return 0
        return max(1, int(window - (now - attempts[0])) + 1)


def _register_login_failure(key: str) -> None:
    _, window = _login_throttle_settings()
    now = _time.monotonic()
    with _APG_LOGIN_LOCK:
        attempts = [stamp for stamp in _APG_LOGIN_ATTEMPTS.get(key, []) if now - stamp < window]
        attempts.append(now)
        _APG_LOGIN_ATTEMPTS[key] = attempts


def _clear_login_failures(key: str) -> None:
    with _APG_LOGIN_LOCK:
        _APG_LOGIN_ATTEMPTS.pop(key, None)


def _live_topic_list(raw_topics: str | None = None) -> list[str]:
    topics = [
        topic.strip()
        for topic in str(raw_topics or "").split(",")
        if topic.strip()
    ]
    return topics or ["*"]


def _subscribe_live_events(topics: list[str]) -> tuple[Any, Any]:
    subscriber = {{"topics": set(topics), "queue": _queue.Queue(maxsize=100)}}
    with APG_LIVE_LOCK:
        APG_LIVE_SUBSCRIBERS.append(subscriber)

    def unsubscribe() -> None:
        with APG_LIVE_LOCK:
            if subscriber in APG_LIVE_SUBSCRIBERS:
                APG_LIVE_SUBSCRIBERS.remove(subscriber)

    return subscriber["queue"], unsubscribe


def _publish_live_event(topic: str, event_type: str, data: Dict[str, Any]) -> None:
    message = {{
        "topic": topic,
        "event": event_type,
        "data": data,
        "ts": _time.time(),
    }}
    with APG_LIVE_LOCK:
        subscribers = list(APG_LIVE_SUBSCRIBERS)
    for subscriber in subscribers:
        topics = subscriber.get("topics", set())
        if "*" not in topics and topic not in topics:
            continue
        try:
            subscriber["queue"].put_nowait(message)
        except _queue.Full:
            continue


def _sse_format(message: Dict[str, Any]) -> str:
    event_name = str(message.get("event", "message"))
    payload = json.dumps(message, sort_keys=True)
    return f"event: {{event_name}}\\ndata: {{payload}}\\n\\n"


def _sse_stream(raw_topics: str | None = None):
    topics = _live_topic_list(raw_topics)
    queue, unsubscribe = _subscribe_live_events(topics)
    try:
        yield ": connected\\n\\n"
        yield _sse_format({{"topic": "system", "event": "apg-ready", "data": {{"topics": topics}}}})
        while True:
            try:
                message = queue.get(timeout=15)
            except _queue.Empty:
                yield ": heartbeat\\n\\n"
                continue
            yield _sse_format(message)
    finally:
        unsubscribe()


def _optional_module(name: str) -> Optional[Any]:
    if __package__:
        try:
            return importlib.import_module(f".{{name}}", __package__)
        except ImportError:
            package_import_failed = True
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _log_activity(entity_name: str, record_id: str, event_type: str, actor: str = "system", detail: str = "") -> None:
    key = f"{{entity_name}}:{{record_id}}"
    if key not in APG_ACTIVITY_LOG:
        APG_ACTIVITY_LOG[key] = []
    import datetime
    APG_ACTIVITY_LOG[key].append({{
        "type": event_type,
        "actor": actor,
        "detail": detail,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
    }})
    if len(APG_ACTIVITY_LOG[key]) > 50:
        APG_ACTIVITY_LOG[key] = APG_ACTIVITY_LOG[key][-50:]


def _get_activity(entity_name: str, record_id: str) -> list[Dict[str, Any]]:
    return list(reversed(APG_ACTIVITY_LOG.get(f"{{entity_name}}:{{record_id}}", [])))


AI_AGENTS = _optional_module("ai_agents")
APG_APPLICATIONS = _optional_module("apg_application")
APG_CAPABILITIES = _optional_module("apg_capabilities")

import hashlib as _hashlib


def _journal_append(run_id: str, event_type: str, step: str, data: Dict[str, Any]) -> None:
    import datetime
    if run_id not in WORKFLOW_EVENT_JOURNAL:
        WORKFLOW_EVENT_JOURNAL[run_id] = []
    prev_hash = WORKFLOW_EVENT_JOURNAL[run_id][-1]["hash"] if WORKFLOW_EVENT_JOURNAL[run_id] else "0" * 64
    entry = {{
        "seq": len(WORKFLOW_EVENT_JOURNAL[run_id]),
        "run_id": run_id,
        "event_type": event_type,
        "step": step,
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "data": data,
    }}
    raw = f"{{prev_hash}}{{entry['seq']}}{{entry['event_type']}}{{entry['step']}}{{entry['ts']}}"
    entry["hash"] = _hashlib.sha256(raw.encode()).hexdigest()
    WORKFLOW_EVENT_JOURNAL[run_id].append(entry)
    _publish_live_event(
        f"workflow:run:{{run_id}}",
        "workflow",
        {{"run_id": run_id, "event_type": event_type, "step": step, "data": data}},
    )
    if _APG_PG_URL:
        _pg_save_journal_entry(entry)


def _pg_save_journal_entry(entry: Dict[str, Any]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_workflow_journal ("
                "  id SERIAL PRIMARY KEY,"
                "  run_id TEXT NOT NULL,"
                "  seq INTEGER NOT NULL,"
                "  module_name TEXT NOT NULL,"
                "  event_type TEXT NOT NULL,"
                "  step TEXT NOT NULL,"
                "  ts TIMESTAMPTZ NOT NULL,"
                "  data TEXT NOT NULL,"
                "  hash TEXT NOT NULL,"
                "  UNIQUE(run_id, seq)"
                ")"
            )
            cur.execute(
                "INSERT INTO apg_workflow_journal (run_id, seq, module_name, event_type, step, ts, data, hash)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (
                    entry["run_id"], entry["seq"], MODULE_NAME,
                    entry["event_type"], entry["step"],
                    entry["ts"], json.dumps(entry.get("data", {{}}), default=str),
                    entry["hash"]
                )
            )
        conn.commit()
    except Exception:
        _ = None  # best-effort
    finally:
        conn.close()


def _get_journal(run_id: str) -> list[Dict[str, Any]]:
    return WORKFLOW_EVENT_JOURNAL.get(run_id, [])


def list_agents() -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        return AI_AGENTS.list_agents()
    return []


def list_agent_teams() -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        return AI_AGENTS.list_agent_teams()
    return []


def invoke_agent(name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "invoke_agent"):
        return AI_AGENTS.invoke_agent(name, payload)
    return {{"agent": name, "status": "unavailable", "error": "agents_unavailable"}}


def invoke_team(name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "invoke_team"):
        return AI_AGENTS.invoke_team(name, payload)
    return {{"team": name, "status": "unavailable", "error": "agents_unavailable"}}


def runtime_adapter_environment_keys(runtime: str, agent_name: str | None = None) -> list[str]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "runtime_adapter_environment_keys"):
        return AI_AGENTS.runtime_adapter_environment_keys(runtime, agent_name)
    return []


def runtime_adapter_command_candidates(runtime: str) -> list[list[str]]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "runtime_adapter_command_candidates"):
        return AI_AGENTS.runtime_adapter_command_candidates(runtime)
    return []


def validate_agent_runtimes(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        return AI_AGENTS.validate_agent_runtimes(available_agent_runtimes)
    return {{"errors": [], "warnings": []}}


def list_capabilities() -> list[str]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities"):
        return APG_CAPABILITIES.list_capabilities()
    return []


def capability_health(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health"):
        return APG_CAPABILITIES.capability_health(capability_name)
    return {{"capability": capability_name, "status": "unavailable", "healthy": False, "errors": ["capability_health_unavailable"], "warnings": []}}


def capability_health_report() -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health_report"):
        return APG_CAPABILITIES.capability_health_report()
    return {{"healthy": True, "errors": [], "warnings": [], "capabilities": {{}}}}


def describe_capability(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capability"):
        return APG_CAPABILITIES.describe_capability(capability_name)
    return {{"name": capability_name, "available": False, "error": "capabilities_unavailable"}}


def describe_capabilities() -> Dict[str, Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities"):
        return APG_CAPABILITIES.describe_capabilities()
    return {{}}


def capability_rules(capability_name: str) -> list[Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_rules"):
        return APG_CAPABILITIES.capability_rules(capability_name)
    return []


def evaluate_capability_rules(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "evaluate_capability_rules"):
        return APG_CAPABILITIES.evaluate_capability_rules(capability_name, context or {{}})
    return {{"decision": "allow", "matched_rules": [], "actions": [], "context": context or {{}}, "warning": "capability_rules_unavailable"}}


def capability_configuration(capability_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_configuration"):
        return APG_CAPABILITIES.capability_configuration(capability_name, overrides)
    return dict(overrides or {{}})


def validate_capability_configuration(
    capability_name: str,
    configuration: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "validate_capability_configuration"):
        return APG_CAPABILITIES.validate_capability_configuration(capability_name, configuration)
    return {{"errors": ["capability_configuration_unavailable"], "warnings": []}}


def approval_plan(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "approval_plan"):
        return APG_CAPABILITIES.approval_plan(capability_name, context or {{}})
    return {{"capability": capability_name, "required": False, "approvers": [], "context": context or {{}}}}


def capability_theme(capability_name: str, tenant_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_theme"):
        return APG_CAPABILITIES.capability_theme(capability_name, tenant_overrides)
    return {{"name": capability_name, "tokens": dict(tenant_overrides or {{}})}}


def theme_token(capability_name: str, token: str, default: Any = None) -> Any:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "theme_token"):
        return APG_CAPABILITIES.theme_token(capability_name, token, default)
    return capability_theme(capability_name).get("tokens", {{}}).get(token, default)


def capability_languages(capability_name: str) -> list[str]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_languages"):
        return APG_CAPABILITIES.capability_languages(capability_name)
    return []


def capability_screens(capability_name: str) -> list[Dict[str, Any]]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_screens"):
        return APG_CAPABILITIES.capability_screens(capability_name)
    return []


def capability_streaming(capability_name: str) -> Dict[str, Any]:
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_streaming"):
        return APG_CAPABILITIES.capability_streaming(capability_name)
    return {{}}


def list_entities() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES]


def _database_schema_name(database: Dict[str, Any]) -> str:
    config = database.get("connection_config", {{}})
    if isinstance(config, dict) and config.get("schema"):
        return str(config.get("schema"))
    for field in database.get("fields", []):
        if isinstance(field, dict) and field.get("name") == "schema" and field.get("default"):
            return str(field.get("default"))
    return "public"


def _database_column_specs(entity: Dict[str, Any]) -> list[Dict[str, Any]]:
    entity_name = str(entity.get("name", ""))
    columns: list[Dict[str, Any]] = [
        {{"name": "id", "type": "integer", "required": True, "nullable": False, "primary_key": True}}
    ]
    if APG_MULTI_TENANT_ENABLED:
        columns.append({{"name": "tenant_id", "type": "string", "required": True, "nullable": False, "primary_key": False}})
    for field in entity.get("fields", []):
        if not isinstance(field, dict):
            continue
        if field.get("computed"):
            continue
        field_name = str(field.get("name", "")).strip()
        existing_column_names = {{str(column.get("name")) for column in columns}}
        if not field_name or field_name == "id" or field_name in existing_column_names:
            continue
        required = bool(field.get("required", False))
        column: Dict[str, Any] = {{
            "name": field_name,
            "type": str(field.get("type", "any")),
            "required": required,
            "nullable": not required,
            "primary_key": False,
        }}
        relationship = _field_relationship(entity_name, field_name)
        if relationship:
            column["reference"] = {{
                "table": relationship.get("target_table", ""),
                "column": relationship.get("target_field", "id"),
                "cardinality": relationship.get("cardinality", "many-to-one"),
            }}
        columns.append(column)
    columns.extend([
        {{"name": "created_at", "type": "string", "required": True, "nullable": False, "primary_key": False}},
        {{"name": "updated_at", "type": "string", "required": True, "nullable": False, "primary_key": False}},
        {{"name": "deleted_at", "type": "string", "required": False, "nullable": True, "primary_key": False}},
    ])
    return columns


def _database_table_specs() -> list[Dict[str, Any]]:
    tables: list[Dict[str, Any]] = []
    for entity in ENTITIES:
        if str(entity.get("type", "")) not in {{"entity", "table"}}:
            continue
        table_name = str(entity.get("name", "")).strip()
        if not table_name:
            continue
        columns = _database_column_specs(entity)
        indexes = [
            {{"name": f"idx_{{table_name}}_{{column['name']}}", "columns": [column["name"]]}}
            for column in columns
            if column.get("name") not in {{"id"}} and (column.get("required") or column.get("reference"))
        ][:3]
        tables.append({{
            "name": table_name,
            "columns": columns,
            "indexes": indexes,
            "source": "generated_entity",
        }})
    return tables


def _with_inferred_database_schemas(database: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(database)
    schemas = list(enriched.get("schemas", [])) if isinstance(enriched.get("schemas", []), list) else []
    if schemas:
        enriched["schemas"] = schemas
        return enriched
    tables = _database_table_specs()
    if tables:
        enriched["schemas"] = [
            {{"name": _database_schema_name(enriched), "tables": tables, "source": "generated_entities"}}
        ]
    else:
        enriched["schemas"] = []
    return enriched


def list_databases() -> list[Dict[str, Any]]:
    return [
        _with_inferred_database_schemas(entity)
        for entity in ENTITIES
        if entity.get("type") == "database"
    ]


def list_workflows() -> list[str]:
    names = {{
        str(entity["name"])
        for entity in ENTITIES
        if entity.get("type") in {{"workflow", "flow"}}
    }}
    names.update(str(name) for name in SEMANTIC_MODEL.get("flows", {{}}))
    return sorted(names)


def _workflow_entity(workflow_name: str) -> Dict[str, Any] | None:
    for entity in ENTITIES:
        if entity.get("type") in {{"workflow", "flow"}} and str(entity.get("name")) == workflow_name:
            return dict(entity)
    return None


def _workflow_defaults(entity: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {{}}
    for field in entity.get("fields", []):
        if isinstance(field, dict) and "default" in field:
            defaults[str(field.get("name"))] = field.get("default")
    return defaults


def _split_workflow_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    delimiter = "->" if "->" in text else ","
    parts: list[str] = []
    for part in text.split(delimiter):
        item = part.strip()
        if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
            item = item[1:-1].strip()
        if item:
            parts.append(item)
    return parts


def _workflow_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {{}}
    if isinstance(value, dict):
        return {{str(key): item for key, item in value.items()}}
    if isinstance(value, list):
        mapping: Dict[str, Any] = {{}}
        for item in value:
            if isinstance(item, dict):
                step = item.get("step") or item.get("name") or item.get("from")
                if step not in (None, ""):
                    mapping[str(step)] = dict(item)
            elif isinstance(item, str):
                mapping.update(_workflow_mapping(item))
        return mapping
    text = str(value).strip()
    if not text:
        return {{}}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return {{str(key): item for key, item in loaded.items()}}
    if isinstance(loaded, list):
        return _workflow_mapping(loaded)
    mapping: Dict[str, Any] = {{}}
    for item in text.split(";"):
        part = item.strip()
        if not part:
            continue
        separator = ":" if ":" in part else "=" if "=" in part else None
        if separator is None:
            continue
        key, raw_value = part.split(separator, 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key:
            mapping[key] = raw_value
    return mapping


def _workflow_step_metadata(workflow: Dict[str, Any], step: str) -> Dict[str, Any]:
    step = str(step)
    metadata: Dict[str, Any] = {{}}
    guards = workflow.get("guards", {{}})
    assignments = workflow.get("assignments", {{}})
    timers = workflow.get("timers", {{}})
    waits = workflow.get("waits", {{}})
    retry_policy = workflow.get("retry_policy", {{}})
    compensation = workflow.get("compensation", {{}})
    human_tasks = set(str(item) for item in workflow.get("human_tasks", []))
    if step in guards:
        metadata["guard"] = guards[step]
    if step in assignments:
        metadata["assignee"] = assignments[step]
        metadata["task_type"] = "human"
    elif step in human_tasks:
        metadata["task_type"] = "human"
    if step in timers:
        metadata["timer"] = timers[step]
    if step in waits:
        metadata["wait_for"] = waits[step]
    if step in retry_policy:
        metadata["retry_policy"] = retry_policy[step]
    if step in compensation:
        metadata["compensation"] = compensation[step]
    return metadata


def _compensation_actions(workflow: Dict[str, Any], completed_steps: list[str]) -> list[Dict[str, Any]]:
    compensation = workflow.get("compensation", {{}})
    actions: list[Dict[str, Any]] = []
    if not isinstance(compensation, dict):
        return actions
    for step in reversed(completed_steps):
        if step in compensation:
            actions.append({{"step": step, "action": compensation[step]}})
    return actions


def _retry_limit(policy: Any) -> int:
    if isinstance(policy, dict):
        for key in ("attempts", "max_attempts", "retries", "limit"):
            if key in policy:
                return _retry_limit(policy[key])
        return 1
    try:
        parsed = int(policy)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def _step_failure_budget(step: str, payload: Dict[str, Any]) -> int:
    failures = payload.get("step_failures", payload.get("failures", {{}}))
    if isinstance(failures, dict) and step in failures:
        try:
            return max(0, int(failures[step]))
        except (TypeError, ValueError):
            return 0
    fail_steps = payload.get("fail_steps", [])
    if isinstance(fail_steps, str):
        fail_steps = [part.strip() for part in fail_steps.split(",") if part.strip()]
    if isinstance(fail_steps, list) and step in [str(item) for item in fail_steps]:
        return 999999
    return 0


def _available_workflow_events(payload: Dict[str, Any]) -> set[str]:
    raw_events = payload.get("events", payload.get("completed_events", payload.get("signals", [])))
    if isinstance(raw_events, str):
        return {{part.strip() for part in raw_events.split(",") if part.strip()}}
    if isinstance(raw_events, list):
        return {{str(item) for item in raw_events}}
    if isinstance(raw_events, dict):
        return {{str(key) for key, value in raw_events.items() if value}}
    return set()


def _context_value(path: str, context: Dict[str, Any]) -> Any:
    current: Any = context
    for part in str(path).split("."):
        key = part.strip()
        if not key:
            continue
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _literal_or_context(value: str, context: Dict[str, Any]) -> Any:
    text = str(value).strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {{"none", "null"}}:
        return None
    try:
        numeric_value = float(text) if "." in text else int(text)
    except ValueError:
        numeric_value = None
    if numeric_value is not None:
        return numeric_value
    context_value = _context_value(text, context)
    if context_value is not None:
        return context_value
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def _compare_values(left: Any, operator: str, right: Any) -> bool:
    if operator in {{"in", "not in"}}:
        if isinstance(right, str):
            candidates = [part.strip() for part in right.split(",") if part.strip()]
        else:
            candidates = right
        try:
            result = left in candidates
        except TypeError:
            result = False
        return not result if operator == "not in" else result
    if operator == "contains":
        try:
            return right in left
        except TypeError:
            return False
    if operator in {{"==", "!="}}:
        result = left == right
        return not result if operator == "!=" else result
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        left_value = str(left)
        right_value = str(right)
    if operator == ">=":
        return left_value >= right_value
    if operator == "<=":
        return left_value <= right_value
    if operator == ">":
        return left_value > right_value
    if operator == "<":
        return left_value < right_value
    return False


def _evaluate_workflow_condition(condition: Any, context: Dict[str, Any]) -> bool:
    if condition in (None, ""):
        return True
    if isinstance(condition, bool):
        return condition
    text = str(condition).strip()
    lowered = text.lower()
    if lowered in {{"always", "true", "allow"}}:
        return True
    if lowered in {{"never", "false", "deny"}}:
        return False
    if " or " in lowered:
        return any(_evaluate_workflow_condition(part, context) for part in text.split(" or "))
    if " and " in lowered:
        return all(_evaluate_workflow_condition(part, context) for part in text.split(" and "))
    if lowered.endswith(" present"):
        field = text[: -len(" present")].strip()
        return _context_value(field, context) is not None
    if lowered.endswith(" missing"):
        field = text[: -len(" missing")].strip()
        return _context_value(field, context) is None
    for operator in (" not in ", " contains ", ">=", "<=", "==", "!=", ">", "<", " in "):
        if operator in text:
            left_text, right_text = text.split(operator, 1)
            normalized_operator = operator.strip()
            left = _context_value(left_text.strip(), context)
            right = _literal_or_context(right_text, context)
            return _compare_values(left, normalized_operator, right)
    return bool(_context_value(text, context))


def describe_workflow(workflow_name: str) -> Dict[str, Any]:
    flows = SEMANTIC_MODEL.get("flows", {{}})
    flow = dict(flows.get(workflow_name, {{}})) if isinstance(flows, dict) else {{}}
    entity = _workflow_entity(workflow_name) or {{"name": workflow_name, "type": flow.get("type", "workflow"), "fields": [], "methods": []}}
    defaults = _workflow_defaults(entity)
    steps = _split_workflow_sequence(defaults.get("steps") or flow.get("steps"))
    stages = _split_workflow_sequence(defaults.get("stages") or flow.get("stages"))
    guards = _workflow_mapping(defaults.get("guards") or flow.get("guards") or defaults.get("guard_rules") or defaults.get("conditions"))
    assignments = _workflow_mapping(defaults.get("assignments") or flow.get("assignments") or defaults.get("assignees") or defaults.get("owners"))
    timers = _workflow_mapping(defaults.get("timers") or flow.get("timers") or defaults.get("sla") or defaults.get("deadlines"))
    waits = _workflow_mapping(defaults.get("waits") or flow.get("waits") or defaults.get("event_waits") or defaults.get("wait_for"))
    retry_policy = _workflow_mapping(defaults.get("retry_policy") or flow.get("retry_policy") or defaults.get("retries"))
    compensation = _workflow_mapping(defaults.get("compensation") or flow.get("compensation") or defaults.get("compensations"))
    human_tasks = _split_workflow_sequence(defaults.get("human_tasks") or flow.get("human_tasks") or defaults.get("manual_steps"))
    transitions = [
        {{
            "from": steps[index],
            "to": steps[index + 1],
            **({{"guard": guards.get(steps[index + 1])}} if steps[index + 1] in guards else {{}}),
        }}
        for index in range(max(0, len(steps) - 1))
    ]
    return {{
        "name": workflow_name,
        "type": entity.get("type", flow.get("type", "workflow")),
        "properties": dict(flow.get("properties", {{}})),
        "defaults": defaults,
        "methods": list(entity.get("methods", flow.get("methods", []))),
        "steps": steps,
        "stages": stages,
        "guards": guards,
        "assignments": assignments,
        "human_tasks": human_tasks,
        "timers": timers,
        "waits": waits,
        "retry_policy": retry_policy,
        "compensation": compensation,
        "transitions": transitions,
    }}


def describe_workflows() -> Dict[str, Dict[str, Any]]:
    return {{
        workflow_name: describe_workflow(workflow_name)
        for workflow_name in list_workflows()
    }}


def _trigger_saga_compensation(workflow: Dict[str, Any], completed_steps: list[str]) -> None:
    comp = workflow.get("compensation", {{}})
    if not isinstance(comp, dict):
        return
    for step in reversed(completed_steps):
        action = comp.get(step)
        if action:
            try:
                _record_event("saga.compensate", str(workflow.get("name", "workflow")), after={{"step": step, "action": str(action)}})
            except Exception:
                _ = None  # best-effort


def _execute_workflow_steps(
    workflow: Dict[str, Any],
    steps: list[str],
    start_index: int,
    payload: Dict[str, Any],
    pause_at: str | None = None,
    existing_trace: list[Dict[str, Any]] | None = None,
    existing_completed_steps: list[str] | None = None,
    run_id: str = "",
) -> Dict[str, Any]:
    selected_steps = steps[start_index:]
    if pause_at is not None and pause_at not in selected_steps:
        return {{
            "status": "error",
            "error": "unknown_pause_step",
            "pause_at": pause_at,
            "steps": selected_steps,
            "payload": payload,
        }}
    trace = list(existing_trace or [])
    completed_steps = list(existing_completed_steps or [])
    guards = workflow.get("guards", {{}})
    retry_policy = workflow.get("retry_policy", {{}})
    waits = workflow.get("waits", {{}})
    available_events = _available_workflow_events(payload)
    for offset, step in enumerate(selected_steps):
        index = start_index + offset
        entry: Dict[str, Any] = {{
            "index": index,
            "step": step,
            **_workflow_step_metadata(workflow, step),
        }}
        if run_id:
            _journal_append(run_id, "step_started", step, {{}})
        guard = guards.get(step)
        if guard is not None:
            guard_passed = _evaluate_workflow_condition(guard, payload)
            entry["guard"] = guard
            entry["guard_passed"] = guard_passed
            if not guard_passed:
                entry["status"] = "blocked"
                trace.append(entry)
                return {{
                    "status": "blocked",
                    "current_step": step,
                    "completed_at": None,
                    "steps": selected_steps,
                    "completed_steps": completed_steps,
                    "pending_steps": selected_steps[offset:],
                    "trace": trace,
                    "payload": payload,
                    "blocked_at": step,
                    "blocked_reason": "guard_failed",
                    "guard": guard,
                    "compensations": _compensation_actions(workflow, completed_steps),
                }}
        wait_for = waits.get(step)
        if wait_for is not None:
            event_name = str(wait_for)
            entry["wait_for"] = event_name
            if event_name not in available_events:
                entry["status"] = "waiting"
                trace.append(entry)
                return {{
                    "status": "waiting",
                    "current_step": step,
                    "completed_at": None,
                    "steps": selected_steps,
                    "completed_steps": completed_steps,
                    "pending_steps": selected_steps[offset:],
                    "trace": trace,
                    "payload": payload,
                    "waiting_at": step,
                    "waiting_for": event_name,
                    "compensations": [],
                }}
            entry["event_received"] = event_name
        failure_budget = _step_failure_budget(step, payload)
        retry_limit = _retry_limit(retry_policy.get(step)) if isinstance(retry_policy, dict) and step in retry_policy else 1
        # Circuit breaker: fail fast if open
        cb_k = _cb_key(workflow.get("name", "wf"), step)
        # Check workflow-level circuit_breaker config for this step
        wf_circuit_breakers = workflow.get("circuit_breakers", {{}})
        step_cb_spec = wf_circuit_breakers.get(step, {{}}) if isinstance(wf_circuit_breakers, dict) else {{}}
        _raw_step_policy = retry_policy.get(step) if isinstance(retry_policy, dict) else None
        step_policy = _raw_step_policy if isinstance(_raw_step_policy, dict) else {{}}
        cb_threshold = int(step_cb_spec.get("threshold", step_policy.get("circuit_threshold", 5)) if isinstance(step_cb_spec, dict) else step_policy.get("circuit_threshold", 5))
        cb_reset = int(step_cb_spec.get("reset_timeout", step_policy.get("reset_timeout", 60)) if isinstance(step_cb_spec, dict) else step_policy.get("reset_timeout", 60))
        if _cb_is_open(cb_k, cb_threshold, cb_reset):
            entry["status"] = "circuit_open"
            trace.append(entry)
            return {{
                "status": "failed",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset:],
                "trace": trace,
                "payload": payload,
                "failed_at": step,
                "failure_reason": "circuit_open",
                "compensations": _compensation_actions(workflow, completed_steps),
            }}
        # Step timeout metadata (from timers dict)
        timers = workflow.get("timers", {{}})
        if isinstance(timers, dict) and step in timers:
            entry["timeout_spec"] = timers[step]
        attempts: list[Dict[str, Any]] = []
        for attempt_number in range(1, retry_limit + 1):
            failed = failure_budget >= attempt_number
            attempts.append({{
                "attempt": attempt_number,
                "status": "failed" if failed else "completed",
            }})
            if not failed:
                break
        entry["attempts"] = attempts
        if attempts and attempts[-1]["status"] == "failed":
            _cb_fail(cb_k, cb_threshold, cb_reset)
            # Saga: auto-trigger compensation for completed steps
            is_saga = bool(workflow.get("is_saga", False))
            if is_saga and completed_steps:
                _trigger_saga_compensation(workflow, completed_steps)
                if run_id:
                    comp = workflow.get("compensation", {{}})
                    comp_action = str(comp.get(step, "")) if isinstance(comp, dict) else ""
                    _journal_append(run_id, "saga_compensating", step, {{"compensation": comp_action}})
            if run_id:
                _journal_append(run_id, "step_failed", step, {{"error": "step_failed_after_retries"}})
            entry["status"] = "failed"
            trace.append(entry)
            return {{
                "status": "failed",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset:],
                "trace": trace,
                "payload": payload,
                "failed_at": step,
                "failure_reason": "step_failed",
                "attempts": attempts,
                "compensations": _compensation_actions(workflow, completed_steps),
            }}
        _cb_success(cb_k)
        entry["status"] = "completed"
        trace.append(entry)
        completed_steps.append(step)
        if run_id:
            _journal_append(run_id, "step_completed", step, {{"attempts": len(attempts)}})
        if pause_at == step and offset < len(selected_steps) - 1:
            return {{
                "status": "paused",
                "current_step": step,
                "completed_at": None,
                "steps": selected_steps,
                "completed_steps": completed_steps,
                "pending_steps": selected_steps[offset + 1:],
                "trace": trace,
                "payload": payload,
                "compensations": [],
            }}
    return {{
        "status": "completed",
        "current_step": selected_steps[-1],
        "completed_at": selected_steps[-1],
        "steps": selected_steps,
        "completed_steps": completed_steps,
        "pending_steps": [],
        "trace": trace,
        "payload": payload,
        "compensations": [],
    }}


def run_workflow(workflow_name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global NEXT_WORKFLOW_RUN_ID
    if workflow_name not in list_workflows():
        raise KeyError(workflow_name)
    payload = dict(payload or {{}})
    workflow = describe_workflow(workflow_name)
    steps = list(workflow.get("steps", []))
    if not steps:
        steps = list(workflow.get("stages", []))
    if not steps:
        steps = ["start", "complete"]
    start_at = str(payload.get("start_at") or steps[0])
    if start_at not in steps:
        return {{
            "workflow": workflow_name,
            "status": "error",
            "error": "unknown_start_step",
            "start_at": start_at,
            "steps": steps,
            "payload": payload,
        }}
    start_index = steps.index(start_at)
    selected_steps = steps[start_index:]
    pause_at = payload.get("pause_at", payload.get("stop_after"))
    pause_at = str(pause_at) if pause_at is not None else None
    run_id = f"workflow-run-{{NEXT_WORKFLOW_RUN_ID}}"
    NEXT_WORKFLOW_RUN_ID += 1
    execution = _execute_workflow_steps(workflow, steps, start_index, payload, pause_at, run_id=run_id)
    if execution.get("status") == "error":
        return {{
            "workflow": workflow_name,
            **execution,
        }}
    result = {{
        "id": run_id,
        "workflow": workflow_name,
        "started_at": start_at,
        **execution,
    }}
    event = _record_event("workflow.run", workflow_name, after=result)
    result["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(result)
    # PostgreSQL persistence for durable workflows
    if _APG_PG_URL:
        _pg_save_workflow_run(result)
    persistence_error = _persist_record_store()
    if persistence_error:
        result["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(result)
    # Emit declared completion events
    emit_events = workflow.get("emit_events") or workflow.get("events", {{}}).get("emit", [])
    if isinstance(emit_events, str):
        emit_events = [emit_events]
    for ev_name in (emit_events or []):
        try:
            emit_apg_event(str(ev_name), {{"workflow": workflow_name, "run_id": run_id, "status": execution.get("status")}})
        except Exception:
            _ = None  # best-effort
    # Register subscriptions declared on this workflow
    subscribe_events = workflow.get("subscribe_events") or workflow.get("events", {{}}).get("subscribe", [])
    if isinstance(subscribe_events, str):
        subscribe_events = [subscribe_events]
    for ev_name in (subscribe_events or []):
        _subscribe_workflow_event(str(ev_name), workflow_name)
    return dict(result)


def list_workflow_runs(workflow_name: str | None = None) -> list[Dict[str, Any]]:
    runs = [dict(run) for run in WORKFLOW_RUNS.values()]
    if workflow_name is not None:
        runs = [run for run in runs if run.get("workflow") == workflow_name]
    return runs


def get_workflow_run(run_id: str) -> Dict[str, Any]:
    run = WORKFLOW_RUNS.get(str(run_id))
    if run is None:
        raise KeyError(run_id)
    return dict(run)


def resume_workflow(run_id: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    run_id = str(run_id)
    existing = WORKFLOW_RUNS.get(run_id)
    if existing is None:
        raise KeyError(run_id)
    if existing.get("status") == "completed":
        result = dict(existing)
        result["resumed"] = False
        return result
    workflow_name = str(existing.get("workflow"))
    if workflow_name not in list_workflows():
        raise KeyError(workflow_name)
    payload_update = dict(payload or {{}})
    merged_payload = dict(existing.get("payload", {{}}))
    merged_payload.update(payload_update)
    steps = list(existing.get("steps") or describe_workflow(workflow_name).get("steps", []))
    if not steps:
        steps = ["start", "complete"]
    current_step = str(existing.get("current_step") or existing.get("started_at") or steps[0])
    if current_step in steps:
        start_index = steps.index(current_step) + 1
    else:
        start_index = 0
    if start_index >= len(steps):
        existing["status"] = "completed"
        existing["completed_at"] = steps[-1]
        existing["pending_steps"] = []
        WORKFLOW_RUNS[run_id] = dict(existing)
        return dict(existing)

    selected_steps = steps[start_index:]
    pause_at = payload_update.get("pause_at", payload_update.get("stop_after"))
    pause_at = str(pause_at) if pause_at is not None else None
    workflow = describe_workflow(workflow_name)
    execution = _execute_workflow_steps(
        workflow,
        steps,
        start_index,
        merged_payload,
        pause_at,
        existing_trace=list(existing.get("trace", [])),
        existing_completed_steps=list(existing.get("completed_steps", [])),
        run_id=run_id,
    )
    if execution.get("status") == "error":
        return {{
            "id": run_id,
            "workflow": workflow_name,
            **execution,
        }}
    updated = dict(existing)
    updated.update({{
        **execution,
        "resumed": True,
    }})
    event = _record_event("workflow.resume", workflow_name, before=existing, after=updated)
    updated["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(updated)
    persistence_error = _persist_record_store()
    if persistence_error:
        updated["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(updated)
    return dict(updated)


def execute_workflow_compensations(
    run_id: str,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_id = str(run_id)
    existing = WORKFLOW_RUNS.get(run_id)
    if existing is None:
        raise KeyError(run_id)
    payload = dict(payload or {{}})
    actions = [
        dict(action)
        for action in existing.get("compensations", [])
        if isinstance(action, dict)
    ]
    if existing.get("compensation_status") == "completed":
        return {{
            "id": run_id,
            "workflow": existing.get("workflow"),
            "status": "completed",
            "already_executed": True,
            "actions": existing.get("compensation_results", []),
            "run": dict(existing),
        }}
    results: list[Dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        result = dict(action)
        result.update({{
            "index": index,
            "status": "completed",
            "mode": "generated",
        }})
        if payload:
            result["payload"] = dict(payload)
        results.append(result)
    updated = dict(existing)
    updated.update({{
        "compensation_status": "completed" if actions else "skipped",
        "compensation_results": results,
    }})
    event = _record_event("workflow.compensate", str(existing.get("workflow")), before=existing, after=updated)
    updated["compensation_event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(updated)
    persistence_error = _persist_record_store()
    if persistence_error:
        updated["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(updated)
    return {{
        "id": run_id,
        "workflow": updated.get("workflow"),
        "status": updated["compensation_status"],
        "already_executed": False,
        "actions": results,
        "event_id": event["id"],
        "run": dict(updated),
    }}


import threading as _apg_threading
_CB_LOCK = _apg_threading.Lock()
_ES_LOCK = _apg_threading.Lock()
try:
    import jwt as _jwt_lib
except ImportError:
    _jwt_lib = None


def _cb_key(workflow_name: str, step: str) -> str:
    return f"{{workflow_name}}:{{step}}"


def _cb_is_open(key: str, threshold: int = 5, reset: int = 60) -> bool:
    import time as _t
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.get(key)
        if cb is None:
            return False
        if cb["state"] == "open":
            if _t.time() - cb.get("opened_at", 0.0) > reset:
                cb["state"] = "half_open"
                return False
            return True
        return False


def _cb_fail(key: str, threshold: int = 5, reset: int = 60) -> None:
    import time as _t
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.setdefault(key, {{"state": "closed", "failures": 0, "opened_at": 0.0}})
        cb["failures"] += 1
        if cb["failures"] >= threshold:
            cb["state"] = "open"
            cb["opened_at"] = _t.time()


def _cb_success(key: str) -> None:
    with _CB_LOCK:
        cb = CIRCUIT_BREAKERS.get(key)
        if cb:
            cb.update({{"state": "closed", "failures": 0, "opened_at": 0.0}})


def circuit_breaker_status() -> Dict[str, Any]:
    with _CB_LOCK:
        return {{k: dict(v) for k, v in CIRCUIT_BREAKERS.items()}}


_TENANT_LOCAL = _apg_threading.local()


def _tenant_header_name() -> str:
    configured = str(os.environ.get("APG_TENANT_HEADER", "") or "").strip()
    return configured or APG_TENANT_HEADER_DEFAULT


def _tenant_header_value() -> str | None:
    try:
        header_name = _tenant_header_name()
        value = _flask_request.headers.get(header_name)
        if value is None and header_name.lower() == APG_TENANT_HEADER_DEFAULT.lower():
            value = _flask_request.headers.get("X-Tenant-ID")
    except RuntimeError:
        return None
    if value in (None, ""):
        return None
    value_text = str(value).strip()
    return value_text or None


def _tenant_admin_bypass() -> bool:
    try:
        if str(_flask_request.headers.get("X-APG-Admin", "")).strip() != "1":
            return False
        admin_key = os.environ.get("APG_ADMIN_KEY")
        if not admin_key:
            return False
        authorization = _flask_request.headers.get("Authorization", "")
        supplied_key = _flask_request.headers.get("X-APG-API-Key")
        if authorization.startswith("Bearer "):
            supplied_key = authorization.removeprefix("Bearer ").strip()
        return bool(supplied_key) and hmac.compare_digest(str(supplied_key), str(admin_key))
    except RuntimeError:
        return False


def _tenant_id() -> str | None:
    tenant = getattr(_TENANT_LOCAL, "tenant_id", None)
    if tenant not in (None, ""):
        tenant_text = str(tenant).strip()
        if tenant_text:
            return tenant_text
    return APG_TENANT_DEFAULT if APG_MULTI_TENANT_ENABLED else None


def _tenant_scope_enabled(entity_name: str) -> bool:
    return APG_MULTI_TENANT_ENABLED or entity_name in TENANT_SCOPED_ENTITIES


def _record_tenant_visible(entity_name: str, record: Dict[str, Any]) -> bool:
    if not _tenant_scope_enabled(entity_name) or _tenant_admin_bypass():
        return True
    return str(record.get("tenant_id") or APG_TENANT_DEFAULT) == str(_tenant_id() or APG_TENANT_DEFAULT)


def _subscribe_workflow_event(event_name: str, workflow_name: str) -> None:
    with _ES_LOCK:
        APG_EVENT_SUBSCRIPTIONS.setdefault(event_name, [])
        if workflow_name not in APG_EVENT_SUBSCRIPTIONS[event_name]:
            APG_EVENT_SUBSCRIPTIONS[event_name].append(workflow_name)


def emit_apg_event(event_name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global NEXT_EVENT_ID
    import time as _t
    ev: Dict[str, Any] = {{
        "id": NEXT_EVENT_ID,
        "name": event_name,
        "payload": payload or {{}},
        "ts": _t.time(),
        "triggered": [],
    }}
    with _ES_LOCK:
        NEXT_EVENT_ID += 1
        EVENT_LOG.append(ev)
    subs = list(APG_EVENT_SUBSCRIPTIONS.get(event_name, []))
    for wf_name in subs:
        try:
            run_workflow(wf_name, {{"trigger_event": event_name, **(payload or {{}})}})
            ev["triggered"].append(wf_name)
        except Exception:
            _ = None  # best-effort
    return dict(ev)


def semantic_model() -> Dict[str, Any]:
    return json.loads(json.dumps(SEMANTIC_MODEL))


def database_status() -> Dict[str, Any]:
    databases = list_databases()
    schema_count = sum(len(database.get("schemas", [])) for database in databases)
    table_count = sum(
        len(schema.get("tables", []))
        for database in databases
        for schema in database.get("schemas", [])
    )
    reference_count = sum(
        1
        for database in databases
        for schema in database.get("schemas", [])
        for table in schema.get("tables", [])
        for column in table.get("columns", [])
        if isinstance(column, dict) and isinstance(column.get("reference"), dict)
    )
    validation = validate_database_schema_contracts()
    return {{
        "valid": not validation["errors"],
        "database_count": len(databases),
        "schema_count": schema_count,
        "table_count": table_count,
        "reference_count": reference_count,
        "validation": validation,
    }}


def _record_timestamp() -> str:
    return _datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _record_metadata_key(record_id: Any) -> str:
    return str(record_id)


def _record_metadata(entity_name: str, record_id: Any, create: bool = False) -> Dict[str, Any] | None:
    if record_id in (None, ""):
        return None
    entity_metadata = RECORD_METADATA.setdefault(entity_name, {{}})
    key = _record_metadata_key(record_id)
    metadata = entity_metadata.get(key)
    if metadata is None and create:
        now = _record_timestamp()
        metadata = {{"created_at": now, "updated_at": now, "deleted_at": None}}
        entity_metadata[key] = metadata
    return metadata


def _row_ownership_enabled() -> bool:
    return _env_flag("APG_ROW_OWNERSHIP")


def _apg_current_session_user() -> Dict[str, Any] | None:
    try:
        return _current_user()
    except RuntimeError:
        return None


def _apg_current_uses_header_auth() -> bool:
    try:
        return _has_header_auth(_flask_request.headers)
    except RuntimeError:
        return False


def _apg_current_user_role() -> str:
    user = _apg_current_session_user()
    if isinstance(user, dict):
        roles = user.get("roles", [])
        if isinstance(roles, list) and roles:
            return str(roles[0])
    if _apg_current_uses_header_auth():
        return "api"
    return "anonymous"


def _apg_current_user_permissions() -> set[str]:
    user = _apg_current_session_user()
    if not isinstance(user, dict):
        return set()
    permissions = user.get("permissions", [])
    if not isinstance(permissions, list):
        return set()
    return {{str(permission) for permission in permissions}}


def _apg_current_owner_id() -> str:
    user = _apg_current_session_user()
    if isinstance(user, dict) and user.get("username"):
        return str(user["username"])
    if _apg_current_uses_header_auth():
        return str(os.environ.get("APG_API_KEY_OWNER") or "api")
    return "anonymous"


def _apg_current_user_unrestricted() -> bool:
    return _apg_current_user_role().strip().lower() == "admin" or "*" in _apg_current_user_permissions()


def _record_owner_visible(record: Dict[str, Any]) -> bool:
    if not _row_ownership_enabled() or _apg_current_user_unrestricted():
        return True
    return str(record.get("owner_id", "")) == _apg_current_owner_id()


def _field_acl_allows(entity_name: str, field_name: str) -> bool:
    if not isinstance(_APG_FIELD_ACL, dict):
        return True
    entity_acl = _APG_FIELD_ACL.get(entity_name)
    if not isinstance(entity_acl, dict) or field_name not in entity_acl:
        return True
    allowed_roles = entity_acl.get(field_name)
    if not isinstance(allowed_roles, list):
        return True
    return _apg_current_user_role() in {{str(role) for role in allowed_roles}}


def _field_acl_public_copy(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {{
        key: value
        for key, value in dict(record).items()
        if _field_acl_allows(entity_name, str(key))
    }}


def _record_public_copy(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(record)
    metadata = _record_metadata(entity_name, public.get("id"), create=True) or {{}}
    for key in ("created_at", "updated_at", "deleted_at"):
        if public.get(key) in (None, "") and metadata.get(key) not in (None, ""):
            public[key] = metadata.get(key)
    for field in _file_field_specs(entity_name):
        field_name = str(field.get("name", "")).strip()
        if not field_name:
            continue
        path_value = public.get(field_name + "_path")
        if path_value not in (None, ""):
            public[field_name + "_url"] = _file_url_for_path(entity_name, path_value)
    public.setdefault("created_at", metadata.get("created_at") or _record_timestamp())
    public.setdefault("updated_at", metadata.get("updated_at") or public["created_at"])
    public.setdefault("deleted_at", metadata.get("deleted_at"))
    return _apply_computed_fields(entity_name, public)


def _record_deleted(entity_name: str, record: Dict[str, Any]) -> bool:
    if record.get("deleted_at") not in (None, ""):
        return True
    metadata = _record_metadata(entity_name, record.get("id"))
    return bool(metadata and metadata.get("deleted_at"))


def _record_stored_copy(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    stored = dict(record)
    for field_name in _computed_field_names(entity_name):
        stored.pop(field_name, None)
    metadata = _record_metadata(entity_name, stored.get("id"), create=True) or {{}}
    for key in ("created_at", "updated_at", "deleted_at"):
        if stored.get(key) in (None, "") and metadata.get(key) not in (None, ""):
            stored[key] = metadata.get(key)
    return stored


def _raw_records_by_entity(*, include_deleted: bool = True) -> Dict[str, list[Dict[str, Any]]]:
    return {{
        entity_name: [
            _record_stored_copy(entity_name, record)
            for record in RECORD_STORE[entity_name]
            if include_deleted or not _record_deleted(entity_name, record)
        ]
        for entity_name in sorted(ENTITY_NAMES)
    }}


def list_records(
    entity_name: str | None = None,
    *,
    include_deleted: bool = False,
) -> Dict[str, list[Dict[str, Any]]] | list[Dict[str, Any]]:
    if entity_name is None:
        return {{
            name: [
                _record_public_copy(name, record)
                for record in RECORD_STORE[name]
                if (include_deleted or not _record_deleted(name, record))
                and _record_owner_visible(record)
                and _record_tenant_visible(name, record)
            ]
            for name in sorted(ENTITY_NAMES)
    }}
    return [
        _record_public_copy(entity_name, record)
        for record in RECORD_STORE[entity_name]
        if (include_deleted or not _record_deleted(entity_name, record))
        and _record_owner_visible(record)
        and _record_tenant_visible(entity_name, record)
    ]


_RECORD_QUERY_CONTROL_KEYS = {{
    "after",
    "dir",
    "format",
    "include_deleted",
    "limit",
    "offset",
    "order",
    "page",
    "per",
    "q",
    "sort",
    "sort_dir",
}}

_RECORD_LIFECYCLE_FIELDS = ("created_at", "updated_at", "deleted_at", "owner_id")


def _strip_record_lifecycle_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(record)
    for field_name in _RECORD_LIFECYCLE_FIELDS:
        sanitized.pop(field_name, None)
    if APG_MULTI_TENANT_ENABLED:
        sanitized.pop("tenant_id", None)
    return sanitized


def _record_query_value(query: Dict[str, list[str]], name: str, default: Any = None) -> Any:
    values = query.get(name)
    if not values:
        return default
    value = values[-1]
    return default if value is None else value


def _truthy_query_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {{"1", "true", "yes", "on"}}


def _records_admin_allowed() -> bool:
    if not APG_AUTH_REQUIRED:
        return True
    try:
        if _has_header_auth(_flask_request.headers):
            return True
    except RuntimeError:
        pass
    try:
        user = _current_user()
    except RuntimeError:
        user = None
    if not isinstance(user, dict):
        return False
    roles = {{str(role).lower() for role in user.get("roles", [])}}
    permissions = {{str(permission) for permission in user.get("permissions", [])}}
    return "admin" in roles or "*" in permissions or "records:include_deleted" in permissions


def _query_includes_deleted(query: Dict[str, list[str]]) -> bool:
    return _truthy_query_value(_record_query_value(query, "include_deleted", "0"))


def _record_field_names(entity_name: str) -> set[str]:
    names = {{"id", "_revision", "created_at", "updated_at", "deleted_at", "owner_id"}}
    if _tenant_scope_enabled(entity_name):
        names.add("tenant_id")
    for field in _field_specs(entity_name):
        field_name = str(field.get("name", ""))
        if not field_name:
            continue
        if _is_file_field(field):
            names.update(_file_metadata_field_names(field_name))
        else:
            names.add(field_name)
    return names


def _record_string_field_names(entity_name: str) -> list[str]:
    names: list[str] = []
    for field in _stored_field_specs(entity_name):
        if _is_file_field(field):
            continue
        field_name = str(field.get("name", ""))
        if field_name and _json_schema_type(str(field.get("type", "any"))) == "string":
            names.append(field_name)
    return names


def _record_query_filters(
    entity_name: str,
    query: Dict[str, list[str]],
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    valid_fields = _record_field_names(entity_name)
    filters: Dict[str, Any] = {{}}
    for key, values in query.items():
        if not values:
            continue
        field_name: str | None = None
        if key.startswith("filter[") and key.endswith("]"):
            field_name = key[len("filter["):-1]
        elif key.startswith("filter."):
            field_name = key.removeprefix("filter.")
        elif key not in _RECORD_QUERY_CONTROL_KEYS and key in valid_fields:
            field_name = key
        if field_name is None:
            continue
        if field_name not in valid_fields:
            return filters, {{"error": "invalid_field"}}
        filters[field_name] = values[-1]
    tid = _tenant_id()
    if (
        tid
        and _tenant_scope_enabled(entity_name)
        and not _tenant_admin_bypass()
        and "tenant_id" not in filters
    ):
        filters["tenant_id"] = tid
    return filters, None


def _record_sort_key(record: Dict[str, Any], field_name: str) -> tuple[int, float, str]:
    value = record.get(field_name)
    if value is None:
        return (1, 0.0, "")
    if isinstance(value, bool):
        return (0, 1.0 if value else 0.0, "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    text = str(value)
    try:
        return (0, float(text), text.lower())
    except ValueError:
        return (0, 0.0, text.lower())


def _invalid_record_query_result(entity_name: str, response_style: str) -> Dict[str, Any]:
    key = "data" if response_style == "records" else "records"
    return {{
        "error": "invalid_field",
        "entity": entity_name,
        key: [],
        "count": 0,
        "total": 0,
        "next_cursor": None,
    }}


def query_records(
    entity_name: str,
    query: Dict[str, list[str]] | None = None,
    *,
    response_style: str = "legacy",
    paginate: bool = True,
) -> Dict[str, Any]:
    query = query or {{}}
    include_deleted_requested = _query_includes_deleted(query)
    if include_deleted_requested and not _records_admin_allowed():
        return {{
            "error": "include_deleted_requires_admin",
            "entity": entity_name,
            "records": [],
            "data": [],
            "count": 0,
            "total": 0,
        }}
    records = list_records(entity_name, include_deleted=include_deleted_requested)
    filters, filter_error = _record_query_filters(entity_name, query)
    if filter_error is not None:
        return _invalid_record_query_result(entity_name, response_style)
    records = [
        record
        for record in records
        if all(str(record.get(field, "")) == str(expected) for field, expected in filters.items())
    ]
    q = str(_record_query_value(query, "q", "") or "").strip().lower()
    if q:
        string_fields = _record_string_field_names(entity_name)
        records = [
            record
            for record in records
            if any(q in str(record.get(field, "")).lower() for field in string_fields)
        ]
    default_sort = "id" if response_style == "records" else None
    sort_field = _record_query_value(query, "sort", default_sort)
    valid_fields = _record_field_names(entity_name)
    if sort_field:
        sort_field = str(sort_field)
        if sort_field not in valid_fields:
            return _invalid_record_query_result(entity_name, response_style)
    direction_source = _record_query_value(query, "sort_dir", _record_query_value(query, "order", "asc"))
    sort_dir = str(direction_source or "asc").lower()
    if sort_dir not in {{"asc", "desc"}}:
        sort_dir = "asc"
    if sort_field:
        records = sorted(
            records,
            key=lambda record: _record_sort_key(record, str(sort_field)),
            reverse=sort_dir == "desc",
        )
    total = len(records)
    if response_style == "records":
        if paginate:
            raw_limit = _record_query_value(query, "limit", "50")
            try:
                parsed_limit = int(raw_limit)
            except (TypeError, ValueError):
                parsed_limit = 50
            parsed_limit = max(1, min(1000, parsed_limit))
            after = _record_query_value(query, "after", None)
            start_index = 0
            if after not in (None, ""):
                for index, record in enumerate(records):
                    if str(record.get("id")) == str(after):
                        start_index = index + 1
                        break
            page_records = records[start_index:start_index + parsed_limit]
            next_cursor = None
            if page_records and start_index + len(page_records) < total:
                next_cursor = page_records[-1].get("id")
        else:
            parsed_limit = None
            after = None
            page_records = records
            next_cursor = None
        return {{
            "entity": entity_name,
            "data": page_records,
            "next_cursor": next_cursor,
            "total": total,
            "count": len(page_records),
            "limit": parsed_limit,
            "after": after,
            "filters": filters,
            "sort": sort_field or "id",
            "sort_dir": sort_dir,
        }}
    try:
        offset = max(0, int(_record_query_value(query, "offset", "0")))
    except (TypeError, ValueError):
        offset = 0
    limit = _record_query_value(query, "limit", None)
    try:
        parsed_limit = int(limit) if limit not in (None, "") else None
    except (TypeError, ValueError):
        parsed_limit = None
    if parsed_limit is not None:
        records = records[offset:offset + max(0, parsed_limit)]
    elif offset:
        records = records[offset:]
    return {{
        "entity": entity_name,
        "records": records,
        "count": len(records),
        "total": total,
        "offset": offset,
        "limit": parsed_limit,
        "filters": filters,
        "sort": sort_field,
        "order": sort_dir,
    }}


def get_record(entity_name: str, record_id: Any) -> tuple[int, Dict[str, Any]]:
    return _records_payload(f"/entities/{{entity_name}}/records/{{record_id}}")


def create_record(entity_name: str, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    return _create_record_payload(f"/entities/{{entity_name}}/records", {{"record": record}})


def update_record(
    entity_name: str,
    record_id: Any,
    record: Dict[str, Any],
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    payload: Dict[str, Any] = {{"record": record}}
    if expected_revision is not None:
        payload["expected_revision"] = expected_revision
    return _update_record_payload(f"/entities/{{entity_name}}/records/{{record_id}}", payload)


def delete_record(
    entity_name: str,
    record_id: Any,
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    path = f"/entities/{{entity_name}}/records/{{record_id}}"
    if expected_revision is not None:
        path = f"{{path}}?expected_revision={{expected_revision}}"
    return _delete_record_payload(path)


def _data_path() -> Path | None:
    raw_path = os.environ.get("APG_DATA_FILE") or os.environ.get("APG_DATA_PATH")
    if not raw_path:
        return None
    return Path(raw_path)


def _record_numeric_id(record: Dict[str, Any]) -> int | None:
    value = record.get("id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _sync_next_record_ids() -> None:
    for entity_name in ENTITY_NAMES:
        numeric_ids = [
            numeric_id
            for record in RECORD_STORE[entity_name]
            for numeric_id in [_record_numeric_id(record)]
            if numeric_id is not None
        ]
        NEXT_RECORD_IDS[entity_name] = max(numeric_ids, default=0) + 1


def _sync_next_event_id() -> None:
    global NEXT_EVENT_ID
    numeric_ids = [
        numeric_id
        for event in EVENT_LOG
        for numeric_id in [_record_numeric_id(event)]
        if numeric_id is not None
    ]
    NEXT_EVENT_ID = max(numeric_ids, default=0) + 1


def _workflow_run_numeric_id(run: Dict[str, Any]) -> int | None:
    value = run.get("id")
    if isinstance(value, str) and value.startswith("workflow-run-"):
        suffix = value.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    if isinstance(value, int):
        return value
    return None


def _sync_next_workflow_run_id() -> None:
    global NEXT_WORKFLOW_RUN_ID
    numeric_ids = [
        numeric_id
        for run in WORKFLOW_RUNS.values()
        for numeric_id in [_workflow_run_numeric_id(run)]
        if numeric_id is not None
    ]
    NEXT_WORKFLOW_RUN_ID = max(numeric_ids, default=0) + 1


def _normalize_record_metadata() -> None:
    for entity_name in ENTITY_NAMES:
        entity_metadata = RECORD_METADATA.setdefault(entity_name, {{}})
        for record in RECORD_STORE[entity_name]:
            record_id = record.get("id")
            if record_id in (None, ""):
                continue
            metadata = entity_metadata.get(_record_metadata_key(record_id))
            if not isinstance(metadata, dict):
                metadata = {{}}
                entity_metadata[_record_metadata_key(record_id)] = metadata
            now = _record_timestamp()
            metadata.setdefault("created_at", record.get("created_at") or now)
            metadata.setdefault("updated_at", record.get("updated_at") or metadata.get("created_at", now))
            metadata.setdefault("deleted_at", record.get("deleted_at"))
            record.setdefault("created_at", metadata.get("created_at"))
            record.setdefault("updated_at", metadata.get("updated_at"))
            record.setdefault("deleted_at", metadata.get("deleted_at"))
            if APG_MULTI_TENANT_ENABLED:
                record.setdefault("tenant_id", APG_TENANT_DEFAULT)


def _load_record_store() -> None:
    path = _data_path()
    if path is None or not path.exists():
        _sqlite_load_records()
        _normalize_record_metadata()
        return
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"APG could not load record data from {{path}}: {{error}}", file=sys.stderr)
        return
    if not isinstance(loaded, dict):
        return
    raw_records = loaded.get("records", loaded)
    if not isinstance(raw_records, dict):
        return
    for entity_name in ENTITY_NAMES:
        entity_records = raw_records.get(entity_name, [])
        if isinstance(entity_records, list):
            RECORD_STORE[entity_name] = [
                dict(record)
                for record in entity_records
                if isinstance(record, dict)
            ]
    raw_metadata = loaded.get("record_metadata", {{}})
    if isinstance(raw_metadata, dict):
        RECORD_METADATA.clear()
        for entity_name in ENTITY_NAMES:
            entity_metadata = raw_metadata.get(entity_name, {{}})
            if isinstance(entity_metadata, dict):
                RECORD_METADATA[entity_name] = {{
                    str(record_id): dict(metadata)
                    for record_id, metadata in entity_metadata.items()
                    if isinstance(metadata, dict)
                }}
            else:
                RECORD_METADATA[entity_name] = {{}}
    raw_events = loaded.get("events", [])
    if isinstance(raw_events, list):
        EVENT_LOG.clear()
        EVENT_LOG.extend(dict(event) for event in raw_events if isinstance(event, dict))
    raw_workflow_runs = loaded.get("workflow_runs", {{}})
    if isinstance(raw_workflow_runs, list):
        raw_workflow_runs = {{
            str(run.get("id")): run
            for run in raw_workflow_runs
            if isinstance(run, dict) and run.get("id") not in (None, "")
        }}
    if isinstance(raw_workflow_runs, dict):
        WORKFLOW_RUNS.clear()
        for run_id, run in raw_workflow_runs.items():
            if isinstance(run, dict):
                normalized = dict(run)
                normalized.setdefault("id", str(run_id))
                WORKFLOW_RUNS[str(normalized["id"])] = normalized
    _sync_next_record_ids()
    _sync_next_event_id()
    _sync_next_workflow_run_id()
    # Merge from PostgreSQL if available
    if _APG_PG_URL:
        for run in _pg_load_workflow_runs():
            rid = str(run.get("id", ""))
            if rid and rid not in WORKFLOW_RUNS:
                WORKFLOW_RUNS[rid] = run
        for entity_name in list(RECORD_STORE.keys()):
            pg_records = _pg_load_entity_records(entity_name, tenant_scoped=False)
            if pg_records:
                RECORD_STORE[entity_name] = pg_records
    _sqlite_load_records()
    _normalize_record_metadata()


def _persist_record_store() -> str | None:
    if _APG_PG_URL:
        for entity_name, records in _raw_records_by_entity().items():
            _pg_save_entity_records(entity_name, records)
    path = _data_path()
    if path is None:
        return None
    payload = {{
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "records": _raw_records_by_entity(),
        "record_metadata": RECORD_METADATA,
        "events": list_events(),
        "workflow_runs": {{run_id: dict(run) for run_id, run in WORKFLOW_RUNS.items()}},
        "next_record_ids": dict(NEXT_RECORD_IDS),
        "next_event_id": NEXT_EVENT_ID,
        "next_workflow_run_id": NEXT_WORKFLOW_RUN_ID,
    }}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{{path.name}}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError as error:
        return str(error)
    return None


def storage_status(include_records: bool = False) -> Dict[str, Any]:
    path = _data_path()
    status: Dict[str, Any] = {{
        "mode": "file" if path is not None else "memory",
        "path": str(path) if path is not None else None,
    }}
    if include_records:
        status["records"] = list_records()
        status["events"] = list_events()
        status["workflow_runs"] = list_workflow_runs()
    return status


def metrics_snapshot() -> Dict[str, Any]:
    record_counts = {{
        entity_name: len(list_records(entity_name))
        for entity_name in sorted(ENTITY_NAMES)
    }}
    event_counts: Dict[str, int] = {{}}
    for event in EVENT_LOG:
        action = str(event.get("action", "unknown"))
        event_counts[action] = event_counts.get(action, 0) + 1
    return {{
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "entity_count": len(ENTITIES),
        "workflow_count": len(list_workflows()),
        "workflow_run_count": len(WORKFLOW_RUNS),
        "database_status": database_status(),
        "record_counts": record_counts,
        "total_records": sum(record_counts.values()),
        "event_count": len(EVENT_LOG),
        "event_counts": event_counts,
        "relationship_count": len(relationship_graph()["edges"]),
        "storage": storage_status(),
        "auth": auth_status(),
    }}


def self_test() -> Dict[str, Any]:
    validation = validate_application()
    openapi = openapi_document()
    routes = sorted(openapi["paths"])
    metrics = metrics_snapshot()
    checks: Dict[str, Any] = {{
        "validation": validation,
        "metrics": metrics,
        "route_count": len(routes),
        "entity_count": metrics["entity_count"],
    }}
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health_report"):
        checks["capability_health"] = APG_CAPABILITIES.capability_health_report()
    return {{
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "passed": validation["valid"],
        "status": "ok" if validation["valid"] else "warning",
        "checks": checks,
        "routes": routes,
    }}


def component_manifest() -> Dict[str, Any]:
    app = describe_application()
    openapi = openapi_document()
    return {{
        "kind": "apg.application",
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "target": "python",
        "composable": True,
        "interfaces": {{
            "http": {{
                "openapi": "/openapi.json",
                "paths": sorted(openapi["paths"]),
            }},
            "python": {{
                "package": MODULE_NAME,
                "exports": [
                    "auth_status",
                    "approval_plan",
                    "capability_configuration",
                    "coerce_record_types",
                    "component_manifest",
                    "create_record",
                    "database_status",
                    "delete_record",
                    "describe_capabilities",
                    "describe_application",
                    "describe_capability",
                    "describe_workflow",
                    "describe_workflows",
                    "evaluate_capability_rules",
                    "execute_workflow_compensations",
                    "get_record",
                    "get_workflow_run",
                    "invoke_agent",
                    "invoke_team",
                    "list_agent_teams",
                    "list_agents",
                    "list_capabilities",
                    "list_databases",
                    "list_entities",
                    "list_events",
                    "list_records",
                    "list_workflow_runs",
                    "list_workflows",
                    "main",
                    "metrics_snapshot",
                    "openapi_document",
                    "query_records",
                    "relationship_graph",
                    "resume_workflow",
                    "run_workflow",
                    "runtime_adapter_command_candidates",
                    "runtime_adapter_environment_keys",
                    "self_test",
                    "semantic_model",
                    "storage_status",
                    "capability_health",
                    "capability_health_report",
                    "capability_languages",
                    "capability_rules",
                    "capability_screens",
                    "capability_streaming",
                    "capability_theme",
                    "theme_token",
                    "update_record",
                    "validate_agent_runtimes",
                    "validate_application",
                    "validate_capability_configuration",
                    "validate_component_manifest_contract",
                    "validate_openapi_contract",
                    "validate_route_dispatch_contract",
                    "validate_record",
                ],
            }},
            "records": sorted(ENTITY_NAMES),
            "theme": "/theme.css",
            "semantic_model": "/semantic-model.json",
        }},
        "entities": list_entities(),
        "databases": list_databases(),
        "workflows": describe_workflows(),
        "ai_agents": app.get("ai_agents", []),
        "ai_agent_teams": app.get("ai_agent_teams", []),
        "application_compositions": app.get("application_compositions", []),
        "application_dependency_graph": app.get("application_dependency_graph", {{}}),
        "application_routes": app.get("application_routes", {{}}),
        "capabilities": app.get("capabilities", []),
        "ui_routes": app.get("ui_routes", {{}}),
        "streaming_processors": app.get("streaming_processors", {{}}),
        "deployment": {{
            "artifacts": [
                "app.py",
                "__init__.py",
                "README.md",
                "semantic_model.json",
                "requirements.txt",
                "Dockerfile",
                ".dockerignore",
                ".env.example",
                "smoke_test.py",
            ],
            "commands": {{
                "run": "python app.py",
                "describe": "python app.py --describe",
                "semantic_model": "python app.py --semantic-model",
                "validate": "python app.py --validate",
                "self_test": "python app.py --self-test",
                "smoke_test": "python smoke_test.py",
            }},
            "environment": ["APG_HOST", "APG_PORT", "APG_DATA_FILE", "APG_API_KEY", "APG_DEBUG"],
        }},
    }}


def auth_status() -> Dict[str, Any]:
    if APG_AUTH_REQUIRED:
        return {{
            "mode": "session",
            "login": "/login",
            "logout": "/logout",
        }}
    return {{
        "mode": "api_key" if os.environ.get("APG_API_KEY") else "open",
        "header": "Authorization: Bearer <key> or X-APG-API-Key" if os.environ.get("APG_API_KEY") else None,
    }}


_APG_REQUIRED_LOCALE_KEYS = (
    "save",
    "cancel",
    "delete",
    "confirm",
    "error",
    "success",
    "loading",
    "no_records",
    "search",
    "login",
    "logout",
)


def _locale_base_dir() -> Path:
    configured = str(os.environ.get("APG_LOCALE_DIR", "") or "").strip()
    if configured:
        return Path(configured)
    try:
        return Path(globals().get("__file__", ".")).resolve().parent / "locales"
    except Exception:
        return Path("locales")


def _locale_file_path() -> Path | None:
    configured = str(os.environ.get("APG_LOCALE_FILE", "") or "").strip()
    return Path(configured) if configured else None


def _read_locale_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flat_locale_strings(value: Any) -> Dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: Dict[str, str] = {{}}
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            return None
        result[str(key)] = str(item)
    return result


def _locale_file_catalog() -> Dict[str, Dict[str, str]]:
    catalog: Dict[str, Dict[str, str]] = {{}}
    locale_dir = _locale_base_dir()
    if locale_dir.is_dir():
        for path in sorted(locale_dir.glob("*.json")):
            parsed = _read_locale_json(path)
            flat = _flat_locale_strings(parsed)
            if flat is not None:
                catalog[path.stem] = flat
                continue
            if isinstance(parsed, dict):
                for language, strings in parsed.items():
                    nested = _flat_locale_strings(strings)
                    if nested is not None:
                        catalog[str(language)] = nested
    configured_path = _locale_file_path()
    if configured_path is not None:
        parsed = _read_locale_json(configured_path)
        flat = _flat_locale_strings(parsed)
        configured_language = str(os.environ.get("APG_LOCALE", "") or "").strip()
        if flat is not None:
            catalog["custom"] = flat
            if configured_language:
                catalog[configured_language] = flat
        elif isinstance(parsed, dict):
            for language, strings in parsed.items():
                nested = _flat_locale_strings(strings)
                if nested is not None:
                    catalog[str(language)] = nested
    return catalog


def _configure_runtime_i18n() -> None:
    global _APG_STRINGS, APG_I18N, APG_SUPPORTED_LANGUAGES, APG_DEFAULT_LANGUAGE, APG_FALLBACK_LANGUAGE
    catalog: Dict[str, Dict[str, str]] = json.loads(json.dumps(_APG_STRINGS))
    english_defaults = dict(catalog.get("en", {{}}))
    for language, strings in _locale_file_catalog().items():
        base = dict(catalog.get(language, english_defaults))
        base.update(strings)
        catalog[language] = base
    configured_language = str(os.environ.get("APG_LOCALE", "") or "").strip()
    if configured_language:
        catalog.setdefault(configured_language, dict(english_defaults))
        APG_DEFAULT_LANGUAGE = configured_language
    if APG_DEFAULT_LANGUAGE not in catalog:
        catalog[APG_DEFAULT_LANGUAGE] = dict(english_defaults)
    if APG_FALLBACK_LANGUAGE not in catalog:
        APG_FALLBACK_LANGUAGE = "en" if "en" in catalog else APG_DEFAULT_LANGUAGE
    fallback = dict(catalog.get(APG_FALLBACK_LANGUAGE, english_defaults))
    for language, strings in list(catalog.items()):
        merged = dict(fallback)
        merged.update(strings)
        for key in _APG_REQUIRED_LOCALE_KEYS:
            merged.setdefault(key, english_defaults.get(key, key.replace("_", " ").title()))
        catalog[language] = merged
    _APG_STRINGS = catalog
    APG_I18N = _APG_STRINGS
    APG_SUPPORTED_LANGUAGES = sorted(catalog)


def _export_locale_if_requested() -> None:
    if not _env_flag("APG_EXPORT_LOCALE"):
        return
    path = _locale_base_dir() / "en.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(APG_I18N.get("en", {{}}), indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        _logging.getLogger("apg").warning("locale_export_failed: %s", exc)


def _locale_file_payload(language: str) -> Any | None:
    if not re.match(r"^[A-Za-z0-9_-]+$", str(language)):
        return None
    locale_path = _locale_base_dir() / (str(language) + ".json")
    if locale_path.is_file():
        parsed = _read_locale_json(locale_path)
        if parsed is not None:
            return parsed
    configured_path = _locale_file_path()
    if configured_path is not None:
        configured_language = str(os.environ.get("APG_LOCALE", "") or "").strip()
        if language == "custom" or (configured_language and language == configured_language):
            parsed = _read_locale_json(configured_path)
            if parsed is not None:
                return parsed
    return None


def _locale_payload(language: str) -> Dict[str, Any] | None:
    file_payload = _locale_file_payload(language)
    if isinstance(file_payload, dict):
        return file_payload
    if language in APG_I18N:
        return dict(APG_I18N[language])
    primary = str(language).split("-", 1)[0]
    if primary in APG_I18N:
        return dict(APG_I18N[primary])
    return None


_configure_runtime_i18n()
_export_locale_if_requested()


def _active_locale() -> str:
    configured_language = str(os.environ.get("APG_LOCALE", "") or "").strip()
    if configured_language in APG_I18N:
        return configured_language
    try:
        cookie_locale = _flask_request.cookies.get("apg_lang")
        if cookie_locale in APG_SUPPORTED_LANGUAGES:
            return str(cookie_locale)
        accepted = _flask_request.accept_languages.best_match(APG_SUPPORTED_LANGUAGES)
        if accepted:
            return str(accepted)
    except RuntimeError:
        return APG_DEFAULT_LANGUAGE
    return APG_DEFAULT_LANGUAGE


def _text_direction(locale: str | None = None) -> str:
    language = (locale or _active_locale()).split("-")[0].lower()
    return "rtl" if language in {{"ar", "he", "fa", "ur"}} else "ltr"


def _apg_t(key: str) -> str:
    locale = _active_locale()
    return (
        _APG_STRINGS.get(locale, {{}}).get(key)
        or _APG_STRINGS.get(APG_FALLBACK_LANGUAGE, {{}}).get(key)
        or _APG_STRINGS.get("en", {{}}).get(key)
        or key
    )


def _(key: str) -> str:
    return _apg_t(key)


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if _active_locale().split("-")[0] in {{"fr", "pt"}}:
        return f"{{number:,.2f}}".replace(",", " ").replace(".", ",")
    return f"{{number:,.2f}}"


def format_currency(value: Any, currency: str = "USD") -> str:
    symbols = {{"USD": "$", "KES": "KSh", "EUR": "€", "GBP": "£"}}
    symbol = symbols.get(str(currency).upper(), str(currency).upper() + " ")
    return symbol + format_number(value)


def format_date(value: Any) -> str:
    text = str(value)
    if _active_locale().split("-")[0] in {{"fr", "pt", "sw"}} and len(text) >= 10 and text[4:5] == "-":
        return f"{{text[8:10]}}/{{text[5:7]}}/{{text[0:4]}}"
    return text


def _auth_credentials() -> Dict[str, Dict[str, Any]]:
    raw = os.environ.get("APG_AUTH_USERS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {{}}
        if isinstance(parsed, dict):
            result: Dict[str, Dict[str, Any]] = {{}}
            for username, spec in parsed.items():
                if isinstance(spec, dict):
                    result[str(username)] = {{
                        "password": str(spec.get("password", "")),
                        "password_hash": str(spec.get("password_hash", "")),
                        "name": str(spec.get("name", username)),
                        "email": str(spec.get("email", "")),
                        "roles": list(spec.get("roles", ["user"])) if isinstance(spec.get("roles", []), list) else ["user"],
                        "permissions": list(spec.get("permissions", [])) if isinstance(spec.get("permissions", []), list) else [],
                    }}
                else:
                    result[str(username)] = {{"password": str(spec), "password_hash": "", "name": str(username), "email": "", "roles": ["user"], "permissions": []}}
            if result:
                return result
    username = os.environ.get("APG_AUTH_USERNAME", "admin")
    password = os.environ.get("APG_AUTH_PASSWORD", "admin")
    return {{
        username: {{
            "password": password,
            "password_hash": os.environ.get("APG_AUTH_PASSWORD_HASH", ""),
            "name": os.environ.get("APG_AUTH_DISPLAY_NAME", username),
            "email": os.environ.get("APG_AUTH_EMAIL", ""),
            "roles": ["admin"],
            "permissions": ["*"],
        }}
    }}


def _authenticate_user(username: str, password: str) -> Dict[str, Any] | None:
    if len(str(password).encode("utf-8")) > _APG_MAX_PASSWORD_BYTES:
        return None
    user = _auth_credentials().get(username)
    if not user:
        _dummy_password_verify()
        return None
    stored_hash = str(user.get("password_hash", "") or "")
    if stored_hash:
        if not _verify_password_hash(stored_hash, password):
            return None
    elif not hmac.compare_digest(str(user.get("password", "")), str(password)):
        return None
    return {{
        "username": username,
        "name": str(user.get("name", username)),
        "email": str(user.get("email", "")),
        "roles": list(user.get("roles", [])),
        "permissions": list(user.get("permissions", [])),
    }}


def _issue_login_session(user: Dict[str, Any]) -> Dict[str, Any]:
    # Rotate the session on privilege change so a pre-login (fixated) session
    # value never survives authentication.
    _flask_session.clear()
    _flask_session["apg_user"] = user
    token = ""
    jwt_secret = os.environ.get("APG_JWT_SECRET")
    if jwt_secret and _jwt_lib is not None:
        try:
            token = _jwt_lib.encode({{"sub": user["username"], "name": user["name"], "roles": user.get("roles", [])}}, jwt_secret, algorithm="HS256")
        except Exception:
            token = ""
    return {{"user": user, "token": token}}


def _current_user() -> Dict[str, Any] | None:
    user = _flask_session.get("apg_user")
    return dict(user) if isinstance(user, dict) else None


def _csrf_token() -> str:
    try:
        token = _flask_session.get("apg_csrf_token")
    except RuntimeError:
        return ""
    if not isinstance(token, str) or not token:
        token = secrets.token_urlsafe(32)
        _flask_session["apg_csrf_token"] = token
    return token


def _csrf_input() -> str:
    token = _csrf_token()
    if not token:
        return ""
    return f'<input type="hidden" name="apg_csrf_token" value="{{html.escape(token, quote=True)}}">'


def _csp_nonce() -> str:
    try:
        return str(getattr(_flask_g, "csp_nonce", "") or "")
    except RuntimeError:
        return ""


def _script_nonce_attr() -> str:
    nonce = html.escape(_csp_nonce(), quote=True)
    return f' nonce="{{nonce}}"' if nonce else ""


def _csrf_payload_token() -> str:
    if _flask_request.is_json:
        data = _flask_request.get_json(silent=True) or {{}}
        if isinstance(data, dict):
            token = data.get("apg_csrf_token") or data.get("csrf_token")
            if token:
                return str(token)
    return str(
        _flask_request.form.get("apg_csrf_token")
        or _flask_request.headers.get("X-APG-CSRF-Token")
        or _flask_request.headers.get("X-CSRF-Token")
        or ""
    )


def _has_header_auth(headers: Any) -> bool:
    authorization = headers.get("Authorization", "")
    supplied_key = headers.get("X-APG-API-Key")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        jwt_secret = os.environ.get("APG_JWT_SECRET")
        jwt_pubkey = os.environ.get("APG_JWT_PUBLIC_KEY")
        if (jwt_secret or jwt_pubkey) and _jwt_lib is not None:
            try:
                key = jwt_pubkey or jwt_secret
                alg = "RS256" if jwt_pubkey else "HS256"
                _jwt_lib.decode(token, key, algorithms=[alg])
                return True
            except Exception:
                return False
        supplied_key = token
    required_keys = [os.environ.get("APG_API_KEY"), os.environ.get("APG_ADMIN_KEY")]
    return any(
        bool(required_key and supplied_key)
        and hmac.compare_digest(str(supplied_key), str(required_key))
        for required_key in required_keys
    )


def _csrf_required_for_request() -> bool:
    if _flask_request.method not in {{"POST", "PUT", "PATCH", "DELETE"}}:
        return False
    if _has_header_auth(_flask_request.headers):
        return False
    path = _flask_request.path.rstrip("/") or "/"
    if path == "/login":
        return APG_AUTH_REQUIRED
    current_user = _current_user()
    if path in {{"/logout", "/locale"}}:
        return APG_AUTH_REQUIRED and current_user is not None
    if path == "/ui" or path.startswith("/ui/"):
        return APG_AUTH_REQUIRED and current_user is not None
    return False


def _csrf_failure_response() -> _FlaskResponse:
    return _FlaskResponse(
        json.dumps({{"error": "csrf_failed", "message": "Refresh the generated page and resubmit the form."}}),
        status=400,
        content_type="application/json; charset=utf-8",
    )


def _check_csrf_token():
    if not _csrf_required_for_request():
        return None
    expected = _flask_session.get("apg_csrf_token")
    supplied = _csrf_payload_token()
    if isinstance(expected, str) and supplied and hmac.compare_digest(expected, supplied):
        return None
    return _csrf_failure_response()


def _login_required_for_path(path: str) -> bool:
    if not APG_AUTH_REQUIRED:
        return False
    return path == "/ui" or path.startswith("/ui/")


def _login_auth_intelligence(next_url: str = "/ui", username: str = "", error: str = "") -> Dict[str, Any]:
    return {{
        "passkey": {{
            "label": "Passkey readiness",
            "status": "Browser check pending",
            "detail": "Generated app keeps credential verification server-side; this control checks WebAuthn availability for future enrollment.",
        }},
        "magic_link": {{
            "label": "Magic-link intent",
            "username": username,
            "next_url": next_url or "/ui",
            "storage_key": "apg:auth:magic-link-intent",
        }},
        "devices": [
            {{
                "id": "current-browser",
                "label": "Current browser",
                "detail": "Session starts after successful sign-in",
                "status": "pending",
            }},
            {{
                "id": "generated-session",
                "label": "Generated Flask session",
                "detail": "Stored in the signed app session cookie",
                "status": "ready",
            }},
        ],
        "lockout": {{
            "attempt_key": "apg:auth:failed-attempts",
            "threshold": 3,
            "error_seen": bool(error),
            "recovery_steps": [
                "Confirm the generated APG_AUTH_USERNAME and APG_AUTH_PASSWORD environment values.",
                "Use a recovery admin session or rotate the generated session secret if local cookies are stale.",
                "Clear the local failed-attempt counter after confirming the operator identity.",
            ],
        }},
    }}


def _login_page(error: str = "", next_url: str = "/ui", username: str = "") -> str:
    body = _render_template(
        "login.html.j2",
        module_name=MODULE_NAME,
        error=error,
        next_url=next_url or "/ui",
        username=username,
        auth_intelligence=_login_auth_intelligence(next_url or "/ui", username, error),
    )
    if body is None:
        safe_error = html.escape(error)
        safe_next = html.escape(next_url or "/ui", quote=True)
        safe_username = html.escape(username, quote=True)
        error_html = f'<p role="alert">{{safe_error}}</p>' if safe_error else ''
        body = (
            '<main id="content" class="apg-login-page">'
            '<section class="apg-login-card">'
            f'<h1>{{html.escape(MODULE_NAME)}}</h1>'
            f'{{error_html}}'
            f'<form method="post" action="/login">{{_csrf_input()}}<input type="hidden" name="next" value="{{safe_next}}">'
            f'<label>Username <input name="username" autocomplete="username" value="{{safe_username}}"></label>'
            '<label>Password <input name="password" type="password" autocomplete="current-password"></label>'
            '<button class="apg-btn" type="submit">Sign in</button></form>'
            '</section></main>'
        )
    return _html_page("Sign in", body, shell=False)


def _forbidden_page(message: str = "You do not have permission to view this page.") -> str:
    return _html_page(
        "Access denied",
        '<section class="apg-card"><h1>Access denied</h1><p>' + html.escape(message) + '</p></section>',
    )


def _authorized(headers: Any) -> bool:
    authorization = headers.get("Authorization", "")
    supplied_key = headers.get("X-APG-API-Key")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        jwt_secret = os.environ.get("APG_JWT_SECRET")
        jwt_pubkey = os.environ.get("APG_JWT_PUBLIC_KEY")
        if (jwt_secret or jwt_pubkey) and _jwt_lib is not None:
            try:
                key = jwt_pubkey or jwt_secret
                alg = "RS256" if jwt_pubkey else "HS256"
                _jwt_lib.decode(token, key, algorithms=[alg])
                return True
            except Exception:
                return False
        supplied_key = token
    admin_key = os.environ.get("APG_ADMIN_KEY")
    if admin_key and supplied_key and hmac.compare_digest(str(supplied_key), str(admin_key)):
        return True
    required_key = os.environ.get("APG_API_KEY")
    if required_key:
        return bool(supplied_key) and hmac.compare_digest(str(supplied_key), str(required_key))
    if _production_mode():
        # Secure by default: with no API key or JWT configured, production
        # mutations require an authenticated (CSRF-verified) session user.
        try:
            return _current_user() is not None
        except RuntimeError:
            return False
    return True


def _auth_failure_payload() -> tuple[int, Dict[str, Any]]:
    return 401, {{
        "error": "unauthorized",
        "message": "Set Authorization: Bearer <key> or X-APG-API-Key to mutate this APG app.",
    }}


_APG_AUDIT_LOGGER = _logging.getLogger("apg.audit")
_APG_AUDIT_FILE_LOCK = _threading.Lock()


def _apg_audit_user(default: str = "api") -> str:
    try:
        explicit = getattr(_flask_g, "apg_user", "")
        if explicit:
            return str(explicit)
        user = _current_user()
    except RuntimeError:
        user = None
    if isinstance(user, dict) and user.get("username"):
        return str(user["username"])
    return default


def _apg_audit_write(payload: Dict[str, Any]) -> None:
    json_line = json.dumps(payload)
    _APG_AUDIT_LOGGER.info(json_line)
    if not os.environ.get("APG_AUDIT_LOG_FILE"):
        return
    try:
        with _APG_AUDIT_FILE_LOCK:
            with open(os.environ["APG_AUDIT_LOG_FILE"], "a", encoding="utf-8") as audit_file:
                audit_file.write(json_line + "\\n")
    except Exception as exc:
        _APG_AUDIT_LOGGER.warning("audit_file_write_failed: %s", exc)


def _apg_audit_event(action: str, entity: str = "auth", user: str | None = None) -> None:
    try:
        request_id = getattr(_flask_g, "request_id", "")
        extra = getattr(_flask_g, "apg_audit_extra", None)
    except RuntimeError:
        request_id = ""
        extra = None
    payload = {{
        "audit": True,
        "action": action,
        "entity": entity,
        "req_id": request_id,
        "user": user if user is not None else _apg_audit_user(),
        "ts": _datetime.datetime.utcnow().isoformat() + "Z",
    }}
    if isinstance(extra, dict):
        payload.update(extra)
    _apg_audit_write(payload)


def _apg_audit_entity_from_path(path: str) -> str:
    parts = [part for part in str(path).split("/") if part]
    if len(parts) >= 2 and parts[0] == "records":
        return parts[1]
    return ""


def list_events(entity_name: str | None = None) -> list[Dict[str, Any]]:
    events = [dict(event) for event in EVENT_LOG]
    if entity_name is None:
        return events
    return [event for event in events if event.get("entity") == entity_name]


def _record_event(
    action: str,
    entity_name: str,
    before: Dict[str, Any] | None = None,
    after: Dict[str, Any] | None = None,
    changed_fields: list[str] | None = None,
) -> Dict[str, Any]:
    global NEXT_EVENT_ID
    record = after if after is not None else before if before is not None else {{}}
    event = {{
        "id": NEXT_EVENT_ID,
        "action": action,
        "entity": entity_name,
        "record_id": record.get("id"),
    }}
    if before is not None:
        event["before"] = dict(before)
    if after is not None:
        event["after"] = dict(after)
    if changed_fields is not None:
        event["changed_fields"] = list(changed_fields)
    NEXT_EVENT_ID += 1
    EVENT_LOG.append(event)
    _publish_live_event("events", "record", event)
    _publish_live_event(f"entity:{{entity_name}}", "record", event)
    return dict(event)


def _prepare_new_record(record: Dict[str, Any], entity_name: str = "") -> Dict[str, Any]:
    prepared = dict(record)
    prepared.setdefault("_revision", 1)
    now = _record_timestamp()
    prepared.setdefault("created_at", now)
    prepared.setdefault("updated_at", now)
    prepared.setdefault("deleted_at", None)
    if _row_ownership_enabled():
        prepared["owner_id"] = _apg_current_owner_id()
    # Auto-inject tenant_id for tenant-scoped entities
    tid = _tenant_id()
    if APG_MULTI_TENANT_ENABLED and entity_name:
        prepared["tenant_id"] = tid or APG_TENANT_DEFAULT
    elif tid and entity_name in TENANT_SCOPED_ENTITIES:
        prepared.setdefault("tenant_id", tid)
    return prepared


def _expected_revision(payload: Dict[str, Any]) -> int | None:
    value = payload.get("expected_revision")
    if value is None and isinstance(payload.get("record"), dict):
        value = payload["record"].get("_revision")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _revision_conflict(existing: Dict[str, Any], expected_revision: int | None) -> Dict[str, Any] | None:
    current_revision = existing.get("_revision")
    if expected_revision is None or current_revision == expected_revision:
        return None
    return {{
        "error": "revision_conflict",
        "expected_revision": expected_revision,
        "current_revision": current_revision,
        "record": dict(existing),
    }}


def _record_schema(entity: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    fields = _field_specs(str(entity["name"]))
    if not fields:
        return {{"type": "object", "additionalProperties": True}}
    schema_properties: Dict[str, Any] = {{
        "id": {{"oneOf": [{{"type": "integer"}}, {{"type": "string"}}]}},
        "_revision": {{"type": "integer"}},
        "created_at": {{"type": "string"}},
        "updated_at": {{"type": "string"}},
        "deleted_at": {{"oneOf": [{{"type": "string"}}, {{"type": "null"}}]}},
        "owner_id": {{"oneOf": [{{"type": "string"}}, {{"type": "null"}}]}},
    }}
    if _tenant_scope_enabled(str(entity["name"])):
        schema_properties["tenant_id"] = {{"type": "string"}}
    required_fields: list[str] = []
    for field in fields:
        field_name = str(field["name"])
        field_schema: Dict[str, Any] = {{"type": _json_schema_type(str(field.get("type", "any")))}}
        if field.get("computed"):
            field_schema["readOnly"] = True
        enum_values = field.get("enum") if isinstance(field.get("enum"), list) else []
        if enum_values:
            field_schema["enum"] = list(enum_values)
        for validator in field.get("validators", []):
            if not isinstance(validator, dict):
                continue
            rule = str(validator.get("rule", ""))
            value = validator.get("value")
            if rule == "min_length":
                field_schema["minLength"] = value
            elif rule == "max_length":
                field_schema["maxLength"] = value
            elif rule == "min":
                field_schema["minimum"] = value
            elif rule == "max":
                field_schema["maximum"] = value
            elif rule == "pattern":
                field_schema["pattern"] = str(validator.get("pattern", value or ""))
        schema_properties[field_name] = field_schema
        if not partial and field.get("required", False) and not field.get("computed"):
            required_fields.append(field_name)
    schema: Dict[str, Any] = {{
        "type": "object",
        "additionalProperties": True,
        "properties": schema_properties,
    }}
    if required_fields:
        schema["required"] = required_fields
    return schema


def _schema_ref(name: str) -> Dict[str, Any]:
    return {{"$ref": f"#/components/schemas/{{name}}"}}


def _json_media(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {{"application/json": {{"schema": schema}}}}


def _record_body_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": False,
        "properties": {{
            "record": _schema_ref(schema_name),
        }},
        "required": ["record"],
    }}


def _record_import_body_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": False,
        "properties": {{
            "records": {{"type": "array", "items": _schema_ref(schema_name)}},
        }},
        "required": ["records"],
    }}


def _record_list_response_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "entity": {{"type": "string"}},
            "records": {{"type": "array", "items": _schema_ref(schema_name)}},
            "count": {{"type": "integer"}},
            "total": {{"type": "integer"}},
            "filters": {{"type": "object", "additionalProperties": {{"type": "string"}}}},
            "sort": {{"oneOf": [{{"type": "string"}}, {{"type": "null"}}]}},
            "order": {{"type": "string"}},
        }},
        "required": ["entity", "records", "count"],
    }}


def _record_cursor_list_response_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "data": {{"type": "array", "items": _schema_ref(schema_name)}},
            "next_cursor": {{
                "oneOf": [
                    {{"type": "integer"}},
                    {{"type": "string"}},
                    {{"type": "null"}},
                ]
            }},
            "total": {{"type": "integer"}},
        }},
        "required": ["data", "next_cursor", "total"],
    }}


def _record_item_response_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "entity": {{"type": "string"}},
            "record": _schema_ref(schema_name),
        }},
        "required": ["entity", "record"],
    }}


def _record_mutation_response_schema(schema_name: str, record_key: str = "record") -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            record_key: _schema_ref(schema_name),
            "event": _schema_ref("EventRecord"),
        }},
        "required": [record_key],
    }}


def _record_export_response_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "entity": {{"type": "string"}},
            "records": {{"type": "array", "items": _schema_ref(schema_name)}},
            "count": {{"type": "integer"}},
        }},
        "required": ["entity", "records", "count"],
    }}


def _record_import_response_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "entity": {{"type": "string"}},
            "imported": {{"type": "array", "items": _schema_ref(schema_name)}},
            "errors": {{"type": "array", "items": {{"type": "object", "additionalProperties": True}}}},
            "events": {{"type": "array", "items": _schema_ref("EventRecord")}},
            "count": {{"type": "integer"}},
            "failed": {{"type": "integer"}},
        }},
        "required": ["entity", "imported", "errors", "count", "failed"],
    }}


def _record_bulk_body_schema(schema_name: str) -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": False,
        "properties": {{
            "create": {{"type": "array", "items": _schema_ref(schema_name)}},
            "update": {{"type": "array", "items": _schema_ref(schema_name)}},
            "delete": {{
                "type": "array",
                "items": {{
                    "oneOf": [
                        {{"type": "integer"}},
                        {{"type": "string"}},
                        {{"type": "object", "additionalProperties": True}},
                    ]
                }},
            }},
        }},
    }}


def _record_bulk_response_schema() -> Dict[str, Any]:
    return {{
        "type": "object",
        "additionalProperties": True,
        "properties": {{
            "created": {{"type": "integer"}},
            "updated": {{"type": "integer"}},
            "deleted": {{"type": "integer"}},
            "errors": {{"type": "array", "items": {{"type": "object", "additionalProperties": True}}}},
        }},
        "required": ["created", "updated", "deleted", "errors"],
    }}


def _database_openapi_schemas() -> Dict[str, Any]:
    nullable_string = {{"oneOf": [{{"type": "string"}}, {{"type": "null"}}]}}
    generic_object = {{"type": "object", "additionalProperties": True}}
    return {{
        "ApplicationDescription": generic_object,
        "SemanticModel": generic_object,
        "ComponentManifest": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "kind": {{"const": "apg.application"}},
                "name": {{"type": "string"}},
                "version": {{"type": "string"}},
                "description": {{"type": "string"}},
                "target": {{"const": "python"}},
                "composable": {{"type": "boolean"}},
                "interfaces": generic_object,
                "entities": {{"type": "array", "items": generic_object}},
                "databases": {{"type": "array", "items": _schema_ref("DatabaseCatalogEntry")}},
                "deployment": generic_object,
            }},
            "required": ["kind", "name", "version", "target", "composable", "interfaces"],
        }},
        "EntityCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "entities": {{"type": "array", "items": generic_object}},
            }},
            "required": ["entities"],
        }},
        "JobCreateRequest": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "type": {{"type": "string"}},
                "payload": generic_object,
            }},
            "required": ["type"],
        }},
        "JobCreateResponse": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "job_id": {{"type": "string"}},
            }},
            "required": ["job_id"],
        }},
        "JobStatus": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "id": {{"type": "string"}},
                "type": {{"type": "string"}},
                "payload": generic_object,
                "status": {{"enum": ["pending", "running", "done", "failed"]}},
                "created_at": {{"type": "string"}},
                "started_at": nullable_string,
                "finished_at": nullable_string,
                "attempts": {{"type": "integer"}},
                "last_error": nullable_string,
            }},
            "required": ["id", "type", "payload", "status", "created_at", "attempts"],
        }},
        "WorkflowSpec": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "type": {{"type": "string"}},
                "steps": {{"type": "array", "items": {{"type": "string"}}}},
                "stages": {{"type": "array", "items": {{"type": "string"}}}},
                "guards": generic_object,
                "assignments": generic_object,
                "human_tasks": {{"type": "array", "items": {{"type": "string"}}}},
                "timers": generic_object,
                "waits": generic_object,
                "retry_policy": generic_object,
                "compensation": generic_object,
                "transitions": {{"type": "array", "items": generic_object}},
                "methods": {{"type": "array", "items": {{"type": "string"}}}},
            }},
            "required": ["name", "type", "steps", "stages", "transitions"],
        }},
        "WorkflowCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "workflows": {{"type": "object", "additionalProperties": _schema_ref("WorkflowSpec")}},
            }},
            "required": ["workflows"],
        }},
        "WorkflowRunRequest": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "payload": generic_object,
                "start_at": {{"type": "string"}},
                "pause_at": {{"type": "string"}},
                "stop_after": {{"type": "string"}},
            }},
        }},
        "WorkflowRunResult": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "id": {{"type": "string"}},
                "workflow": {{"type": "string"}},
                "status": {{"type": "string"}},
                "started_at": {{"type": "string"}},
                "current_step": {{"type": "string"}},
                "completed_at": {{"oneOf": [{{"type": "string"}}, {{"type": "null"}}]}},
                "steps": {{"type": "array", "items": {{"type": "string"}}}},
                "completed_steps": {{"type": "array", "items": {{"type": "string"}}}},
                "pending_steps": {{"type": "array", "items": {{"type": "string"}}}},
                "trace": {{"type": "array", "items": generic_object}},
                "payload": generic_object,
                "event_id": {{"type": "integer"}},
                "blocked_at": {{"type": "string"}},
                "blocked_reason": {{"type": "string"}},
                "waiting_at": {{"type": "string"}},
                "waiting_for": {{"type": "string"}},
                "failed_at": {{"type": "string"}},
                "failure_reason": {{"type": "string"}},
                "compensations": {{"type": "array", "items": generic_object}},
                "guard": {{"oneOf": [{{"type": "string"}}, {{"type": "boolean"}}, generic_object]}},
            }},
            "required": ["id", "workflow", "status", "steps", "trace", "payload"],
        }},
        "WorkflowRunCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "runs": {{"type": "array", "items": _schema_ref("WorkflowRunResult")}},
            }},
            "required": ["runs"],
        }},
        "WorkflowCompensationRequest": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "payload": generic_object,
                "context": generic_object,
            }},
        }},
        "WorkflowCompensationResult": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "id": {{"type": "string"}},
                "workflow": {{"type": "string"}},
                "status": {{"type": "string"}},
                "already_executed": {{"type": "boolean"}},
                "actions": {{"type": "array", "items": generic_object}},
                "event_id": {{"type": "integer"}},
                "run": _schema_ref("WorkflowRunResult"),
            }},
            "required": ["id", "status", "already_executed", "actions", "run"],
        }},
        "RecordsByEntity": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "records": {{"type": "object", "additionalProperties": {{"type": "array", "items": generic_object}}}},
            }},
            "required": ["records"],
        }},
        "AuthStatus": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "mode": {{"type": "string"}},
                "header": nullable_string,
            }},
            "required": ["mode", "header"],
        }},
        "StorageStatus": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "mode": {{"type": "string"}},
                "path": nullable_string,
                "records": {{"type": "object", "additionalProperties": {{"type": "array", "items": generic_object}}}},
                "events": {{"type": "array", "items": _schema_ref("EventRecord")}},
            }},
            "required": ["mode", "path"],
        }},
        "ValidationReport": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "valid": {{"type": "boolean"}},
                "errors": {{"type": "array", "items": {{"type": "string"}}}},
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
                "checks": generic_object,
            }},
            "required": ["name", "valid", "errors", "warnings", "checks"],
        }},
        "HealthReport": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "status": {{"type": "string"}},
                "name": {{"type": "string"}},
                "version": {{"type": "string"}},
                "valid": {{"type": "boolean"}},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
            }},
            "required": ["status", "name", "version", "valid", "storage", "auth", "warnings"],
        }},
        "EventLog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "events": {{"type": "array", "items": _schema_ref("EventRecord")}},
            }},
            "required": ["events"],
        }},
        "MetricsSnapshot": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "version": {{"type": "string"}},
                "entity_count": {{"type": "integer"}},
                "database_status": _schema_ref("DatabaseStatus"),
                "record_counts": {{"type": "object", "additionalProperties": {{"type": "integer"}}}},
                "total_records": {{"type": "integer"}},
                "event_count": {{"type": "integer"}},
                "event_counts": {{"type": "object", "additionalProperties": {{"type": "integer"}}}},
                "relationship_count": {{"type": "integer"}},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
            }},
            "required": ["name", "version", "entity_count", "record_counts", "total_records", "event_count"],
        }},
        "SelfTestReport": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "version": {{"type": "string"}},
                "passed": {{"type": "boolean"}},
                "status": {{"type": "string"}},
                "checks": _schema_ref("SelfTestChecks"),
                "routes": {{"type": "array", "items": {{"type": "string"}}}},
            }},
            "required": ["name", "version", "passed", "status", "checks", "routes"],
        }},
        "SelfTestChecks": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "validation": _schema_ref("ValidationReport"),
                "metrics": _schema_ref("MetricsSnapshot"),
                "route_count": {{"type": "integer"}},
                "entity_count": {{"type": "integer"}},
                "capability_health": _schema_ref("CapabilityHealthReport"),
            }},
            "required": ["validation", "metrics", "route_count", "entity_count"],
        }},
        "RelationshipNode": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "id": {{"type": "string"}},
                "name": {{"type": "string"}},
                "type": {{"type": "string"}},
            }},
            "required": ["id", "name", "type"],
        }},
        "RelationshipEdge": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "from": {{"type": "string"}},
                "to": {{"type": "string"}},
                "field": {{"type": "string"}},
                "relationship": {{"type": "string"}},
            }},
            "required": ["from", "to", "relationship"],
        }},
        "RelationshipGraph": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "nodes": {{"type": "array", "items": _schema_ref("RelationshipNode")}},
                "edges": {{"type": "array", "items": _schema_ref("RelationshipEdge")}},
            }},
            "required": ["nodes", "edges"],
        }},
        "AgentCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "agents": generic_object,
                "teams": generic_object,
            }},
            "required": ["agents", "teams"],
        }},
        "ApplicationCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "applications": generic_object,
                "dependency_graph": generic_object,
                "components": generic_object,
            }},
            "required": ["applications", "dependency_graph", "components"],
        }},
        "CapabilityCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "capabilities": generic_object,
                "by_erp_module": generic_object,
                "dependency_graph": generic_object,
                "load_order": {{"oneOf": [generic_object, {{"type": "array", "items": {{"type": "string"}}}}]}},
            }},
            "required": ["capabilities", "by_erp_module", "dependency_graph", "load_order"],
        }},
        "CapabilityHealth": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "capability": {{"type": "string"}},
                "status": {{"type": "string"}},
                "healthy": {{"type": "boolean"}},
                "errors": {{"type": "array", "items": {{"type": "string"}}}},
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
                "configuration": generic_object,
                "rules": generic_object,
                "approvals": generic_object,
                "ui": generic_object,
                "theme": generic_object,
                "streaming": generic_object,
                "master_data": {{"type": "array", "items": {{"type": "string"}}}},
                "languages": {{"type": "array", "items": {{"type": "string"}}}},
                "components": generic_object,
            }},
            "required": ["capability", "status", "healthy", "errors", "warnings"],
        }},
        "CapabilityHealthReport": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "healthy": {{"type": "boolean"}},
                "errors": {{"type": "array", "items": {{"type": "string"}}}},
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
                "capabilities": {{"type": "object", "additionalProperties": _schema_ref("CapabilityHealth")}},
            }},
            "required": ["healthy", "errors", "warnings", "capabilities"],
        }},
        "RouteCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "routes": generic_object,
            }},
            "required": ["routes"],
        }},
        "AgentInvocationRequest": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "message": {{"type": "string"}},
                "payload": generic_object,
                "context": generic_object,
            }},
        }},
        "AgentInvocationResponse": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "agent": {{"type": "string"}},
                "team": {{"type": "string"}},
                "runtime": {{"type": "string"}},
                "status": {{"type": "string"}},
                "result": {{"oneOf": [generic_object, {{"type": "string"}}, {{"type": "null"}}]}},
                "payload": generic_object,
            }},
        }},
        "RuleEvaluationRequest": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "capability": {{"type": "string"}},
                "capability_name": {{"type": "string"}},
                "context": generic_object,
            }},
            "required": ["context"],
        }},
        "RuleEvaluationResult": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "decision": {{"type": "string"}},
                "matched_rules": {{"type": "array", "items": {{"type": "string"}}}},
                "actions": {{"type": "array", "items": generic_object}},
                "context": generic_object,
            }},
        }},
        "CapabilityConfigurationRequest": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "capability": {{"type": "string"}},
                "capability_name": {{"type": "string"}},
                "configuration": generic_object,
                "overrides": generic_object,
            }},
        }},
        "CapabilityConfigurationResponse": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "capability": {{"type": "string"}},
                "configuration": generic_object,
                "errors": {{"type": "array", "items": {{"type": "string"}}}},
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
            }},
        }},
        "ApprovalPlanRequest": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "capability": {{"type": "string"}},
                "capability_name": {{"type": "string"}},
                "context": generic_object,
            }},
            "required": ["context"],
        }},
        "ApprovalPlanResponse": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "capability": {{"type": "string"}},
                "required": {{"type": "boolean"}},
                "levels": {{"type": "integer"}},
                "approvers": {{"type": "array", "items": {{"type": "string"}}}},
                "thresholds": generic_object,
                "segregation_of_duties": {{"type": "boolean"}},
                "escalation": {{"oneOf": [{{"type": "string"}}, generic_object, {{"type": "null"}}]}},
            }},
        }},
        "StreamingTopology": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "processor": {{"type": "string"}},
                "processors": {{"type": "object", "additionalProperties": {{"type": "array", "items": {{"type": "string"}}}}}},
                "states": {{"type": "object", "additionalProperties": {{"type": "array", "items": {{"type": "string"}}}}}},
                "streams": {{"type": "object", "additionalProperties": generic_object}},
            }},
            "required": ["processor", "processors", "states", "streams"],
        }},
        "CapabilityStreamingContract": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "processor": {{"type": "string"}},
                "state": {{"type": "string"}},
                "input": generic_object,
                "output": generic_object,
            }},
        }},
        "EventRecord": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "id": {{"type": "integer"}},
                "entity": {{"type": "string"}},
                "action": {{"type": "string"}},
                "record_id": {{"oneOf": [{{"type": "integer"}}, {{"type": "string"}}, {{"type": "null"}}]}},
                "before": {{"oneOf": [{{"type": "object", "additionalProperties": True}}, {{"type": "null"}}]}},
                "after": {{"oneOf": [{{"type": "object", "additionalProperties": True}}, {{"type": "null"}}]}},
            }},
            "required": ["id", "entity", "action"],
        }},
        "DatabaseReference": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "kind": {{"type": "string"}},
                "relationship": {{"type": "string"}},
                "schema": {{"type": "string"}},
                "table": {{"type": "string"}},
                "column": {{"type": "string"}},
                "target": {{"type": "string"}},
            }},
            "required": ["table", "column"],
        }},
        "DatabaseColumn": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "type": {{"type": "string"}},
                "primary_key": {{"type": "boolean"}},
                "nullable": {{"type": "boolean"}},
                "default": {{
                    "oneOf": [
                        {{"type": "string"}},
                        {{"type": "number"}},
                        {{"type": "integer"}},
                        {{"type": "boolean"}},
                        {{"type": "null"}},
                    ]
                }},
                "constraints": {{"type": "array", "items": {{"type": "string"}}}},
                "reference": {{"oneOf": [_schema_ref("DatabaseReference"), {{"type": "null"}}]}},
            }},
            "required": ["name", "type", "primary_key", "nullable", "constraints"],
        }},
        "DatabaseIndex": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": nullable_string,
                "columns": {{"type": "array", "items": {{"type": "string"}}}},
                "unique": {{"type": "boolean"}},
                "type": nullable_string,
            }},
            "required": ["columns", "unique"],
        }},
        "DatabaseTable": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "columns": {{"type": "array", "items": _schema_ref("DatabaseColumn")}},
                "indexes": {{"type": "array", "items": _schema_ref("DatabaseIndex")}},
            }},
            "required": ["name", "columns", "indexes"],
        }},
        "DatabaseSchema": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "tables": {{"type": "array", "items": _schema_ref("DatabaseTable")}},
            }},
            "required": ["name", "tables"],
        }},
        "DatabaseCatalogEntry": {{
            "type": "object",
            "additionalProperties": True,
            "properties": {{
                "name": {{"type": "string"}},
                "type": {{"const": "database"}},
                "properties": {{"type": "array", "items": {{"type": "string"}}}},
                "connection_config": {{"type": "object", "additionalProperties": True}},
                "schemas": {{"type": "array", "items": _schema_ref("DatabaseSchema")}},
            }},
            "required": ["name", "type", "schemas"],
        }},
        "DatabaseCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "databases": {{"type": "array", "items": _schema_ref("DatabaseCatalogEntry")}},
            }},
            "required": ["databases"],
        }},
        "DatabaseSchemaCatalog": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "database": {{"type": "string"}},
                "schemas": {{"type": "array", "items": _schema_ref("DatabaseSchema")}},
            }},
            "required": ["database", "schemas"],
        }},
        "DatabaseValidation": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "errors": {{"type": "array", "items": {{"type": "string"}}}},
                "warnings": {{"type": "array", "items": {{"type": "string"}}}},
                "validated_databases": {{"type": "array", "items": {{"type": "string"}}}},
            }},
            "required": ["errors", "warnings", "validated_databases"],
        }},
        "DatabaseStatus": {{
            "type": "object",
            "additionalProperties": False,
            "properties": {{
                "valid": {{"type": "boolean"}},
                "database_count": {{"type": "integer"}},
                "schema_count": {{"type": "integer"}},
                "table_count": {{"type": "integer"}},
                "reference_count": {{"type": "integer"}},
                "validation": _schema_ref("DatabaseValidation"),
            }},
            "required": [
                "valid",
                "database_count",
                "schema_count",
                "table_count",
                "reference_count",
                "validation",
            ],
        }},
    }}


def _api_operation(
    summary: str,
    description: str,
    status: str = "200",
    request_body: bool = False,
    request_schema: Dict[str, Any] | None = None,
    response_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {{"description": description}}
    if response_schema is not None:
        response["content"] = _json_media(response_schema)
    operation: Dict[str, Any] = {{
        "summary": summary,
        "responses": {{status: response}},
    }}
    if request_body:
        operation["requestBody"] = {{"required": True}}
        if request_schema is not None:
            operation["requestBody"]["content"] = _json_media(request_schema)
    return operation


def _build_openapi_document() -> Dict[str, Any]:
    paths: Dict[str, Any] = {{
        "/health": {{"get": _api_operation("Application health", "Health report", response_schema=_schema_ref("HealthReport"))}},
        "/component.json": {{"get": _api_operation("Composable component manifest", "APG component manifest", response_schema=_schema_ref("ComponentManifest"))}},
        "/manifest": {{"get": _api_operation("Application manifest", "APG manifest", response_schema=_schema_ref("ApplicationDescription"))}},
        "/semantic-model.json": {{"get": _api_operation("Semantic model", "APG semantic model", response_schema=_schema_ref("SemanticModel"))}},
        "/openapi.json": {{"get": _api_operation("OpenAPI contract", "OpenAPI 3.1 contract", response_schema={{"type": "object", "additionalProperties": True}})}},
        "/validate": {{"get": _api_operation("Application validation", "Validation report", response_schema=_schema_ref("ValidationReport"))}},
        "/events": {{"get": _api_operation("Record mutation events", "Event log", response_schema=_schema_ref("EventLog"))}},
        "/auth": {{"get": _api_operation("Authentication status", "Authentication mode", response_schema=_schema_ref("AuthStatus"))}},
        "/metrics": {{"get": _api_operation("Application metrics", "Runtime metrics", response_schema=_schema_ref("MetricsSnapshot"))}},
        "/applications": {{"get": _api_operation("Application compositions", "Application composition catalog", response_schema=_schema_ref("ApplicationCatalog"))}},
        "/self-test": {{"get": _api_operation("Application self-test", "Self-test report", response_schema=_schema_ref("SelfTestReport"))}},
        "/theme.css": {{"get": _api_operation("Generated visual theme stylesheet", "CSS theme stylesheet")}},
        "/records": {{"get": _api_operation("All entity records", "Records by entity", response_schema=_schema_ref("RecordsByEntity"))}},
        "/entities": {{"get": _api_operation("Entity catalog", "Generated entity metadata", response_schema=_schema_ref("EntityCatalog"))}},
        "/workflows": {{"get": _api_operation("Workflow catalog", "Generated workflow metadata", response_schema=_schema_ref("WorkflowCatalog"))}},
        "/workflows/runs": {{"get": _api_operation("Workflow run catalog", "Generated workflow run state", response_schema=_schema_ref("WorkflowRunCatalog"))}},
        "/workflows/runs/{{id}}": {{"get": _api_operation("Workflow run detail", "Generated workflow run state", response_schema=_schema_ref("WorkflowRunResult"))}},
        "/workflows/runs/{{id}}/resume": {{"post": _api_operation("Resume workflow run", "Workflow resume result", request_body=True, request_schema=_schema_ref("WorkflowRunRequest"), response_schema=_schema_ref("WorkflowRunResult"))}},
        "/workflows/runs/{{id}}/compensate": {{"post": _api_operation("Execute workflow compensations", "Workflow compensation result", request_body=True, request_schema=_schema_ref("WorkflowCompensationRequest"), response_schema=_schema_ref("WorkflowCompensationResult"))}},
        "/jobs": {{
            "get": _api_operation("List background jobs", "Background job list", response_schema={{"type": "array", "items": _schema_ref("JobStatus")}}),
            "post": _api_operation("Enqueue background job", "Created background job", status="201", request_body=True, request_schema=_schema_ref("JobCreateRequest"), response_schema=_schema_ref("JobCreateResponse")),
        }},
        "/jobs/{{id}}": {{"get": _api_operation("Background job status", "Background job detail", response_schema=_schema_ref("JobStatus"))}},
        "/jobs/{{id}}/retry": {{"post": _api_operation("Retry failed background job", "Requeued background job", request_body=True, request_schema={{"type": "object", "additionalProperties": True}}, response_schema=_schema_ref("JobCreateResponse"))}},
        "/databases": {{"get": _api_operation("Database catalog", "Database schema and connection metadata", response_schema=_schema_ref("DatabaseCatalog"))}},
        "/databases/status": {{"get": _api_operation("Database validation status", "Database schema validation and counts", response_schema=_schema_ref("DatabaseStatus"))}},
        "/relationships": {{"get": _api_operation("Entity relationship graph", "Relationship graph", response_schema=_schema_ref("RelationshipGraph"))}},
        "/storage": {{"get": _api_operation("Record storage status", "Storage status", response_schema=_schema_ref("StorageStatus"))}},
        "/agents": {{"get": _api_operation("Agent catalog", "AI agent and team catalog", response_schema=_schema_ref("AgentCatalog"))}},
        "/capabilities": {{"get": _api_operation("Capability catalog", "Capability catalog", response_schema=_schema_ref("CapabilityCatalog"))}},
        "/capabilities/health": {{"get": _api_operation("Capability health report", "Capability health report", response_schema=_schema_ref("CapabilityHealthReport"))}},
        "/routes": {{"get": _api_operation("Generated UI route catalog", "UI route catalog", response_schema=_schema_ref("RouteCatalog"))}},
        "/composition": {{"get": _api_operation("Composition graph", "Composition graph", response_schema=_schema_ref("RelationshipGraph"))}},
        "/ui": {{"get": _api_operation("Generated application UI", "HTML application index")}},
        "/ui/databases": {{"get": _api_operation("Generated database catalog UI", "HTML database catalog")}},
    }}
    schemas: Dict[str, Any] = _database_openapi_schemas()
    for entity in _APG_ENTITIES:
        entity_name = str(entity["name"])
        schema_name = f"{{entity_name}}Record"
        patch_schema_name = f"{{entity_name}}RecordPatch"
        schemas[schema_name] = _record_schema(entity)
        schemas[patch_schema_name] = _record_schema(entity, partial=True)
        paths[f"/records/{{entity_name}}"] = {{
            "get": _api_operation(
                f"List {{entity_name}} records",
                "Paginated record list",
                response_schema=_record_cursor_list_response_schema(schema_name),
            ),
            "post": _api_operation(
                f"Create {{entity_name}} record",
                "Created record",
                status="201",
                request_body=True,
                request_schema=_record_body_schema(schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
        }}
        paths[f"/records/{{entity_name}}"]["get"]["parameters"] = [
            {{"name": "limit", "in": "query", "required": False, "description": "Maximum records to return, default 50, max 1000"}},
            {{"name": "after", "in": "query", "required": False, "description": "Cursor record id"}},
            {{"name": "filter[<field>]", "in": "query", "required": False, "description": "Exact field filter"}},
            {{"name": "q", "in": "query", "required": False, "description": "LIKE search across string fields"}},
            {{"name": "sort", "in": "query", "required": False, "description": "Field to sort by"}},
            {{"name": "sort_dir", "in": "query", "required": False, "description": "asc or desc"}},
            {{"name": "include_deleted", "in": "query", "required": False, "description": "Admin-only flag to include soft-deleted records"}},
            {{"name": "format", "in": "query", "required": False, "description": "Use csv for RFC 4180 export"}},
        ]
        if _record_string_field_names(entity_name):
            paths[f"/records/{{entity_name}}/search"] = {{
                "get": _api_operation(
                    f"Search {{entity_name}} records",
                    "FTS5 record search",
                    response_schema={{"type": "array", "items": _schema_ref(schema_name)}},
                ),
            }}
            paths[f"/records/{{entity_name}}/search"]["get"]["parameters"] = [
                {{"name": "q", "in": "query", "required": True, "description": "FTS5 query text"}},
                {{"name": "limit", "in": "query", "required": False, "description": "Maximum records to return"}},
            ]
        paths[f"/records/{{entity_name}}/bulk"] = {{
            "post": _api_operation(
                f"Bulk mutate {{entity_name}} records",
                "Bulk record mutation result",
                request_body=True,
                request_schema=_record_bulk_body_schema(schema_name),
                response_schema=_record_bulk_response_schema(),
            ),
        }}
        paths[f"/records/{{entity_name}}/{{{{id}}}}"] = {{
            "get": _api_operation(
                f"Fetch {{entity_name}} record",
                "Record",
                response_schema=_record_item_response_schema(schema_name),
            ),
            "put": _api_operation(
                f"Update {{entity_name}} record",
                "Updated record",
                request_body=True,
                request_schema=_record_body_schema(patch_schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
            "delete": _api_operation(
                f"Delete {{entity_name}} record",
                "Deleted record",
                response_schema=_record_mutation_response_schema(schema_name, record_key="deleted"),
            ),
        }}
        paths[f"/records/{{entity_name}}/{{{{id}}}}/restore"] = {{
            "delete": _api_operation(
                f"Restore {{entity_name}} record",
                "Restored record",
                response_schema=_record_mutation_response_schema(schema_name),
            ),
        }}
        for relationship in _relationship_specs(entity_name):
            if str(relationship.get("kind")) != "has_many":
                continue
            target_name = str(relationship.get("target", ""))
            if not target_name:
                continue
            target_schema_name = f"{{target_name}}Record"
            segment = str(relationship.get("segment") or _relationship_segment(target_name))
            paths[f"/records/{{entity_name}}/{{{{id}}}}/{{segment}}"] = {{
                "get": _api_operation(
                    f"List {{entity_name}} {{segment}}",
                    "Nested relationship records",
                    response_schema=_record_list_response_schema(target_schema_name),
                ),
            }}
            if relationship.get("through"):
                paths[f"/records/{{entity_name}}/{{{{id}}}}/{{segment}}/{{{{related_id}}}}"] = {{
                    "post": _api_operation(
                        f"Link {{entity_name}} to {{target_name}}",
                        "Created relationship link",
                        status="201",
                        request_body=True,
                        request_schema={{"type": "object", "additionalProperties": True}},
                        response_schema={{"type": "object", "additionalProperties": True}},
                    ),
                    "delete": _api_operation(
                        f"Unlink {{entity_name}} from {{target_name}}",
                        "Deleted relationship link",
                        response_schema={{"type": "object", "additionalProperties": True}},
                    ),
                }}
        paths[f"/entities/{{entity_name}}/records"] = {{
            "get": _api_operation(
                f"List {{entity_name}} records",
                "Record list",
                response_schema=_record_list_response_schema(schema_name),
            ),
            "post": _api_operation(
                f"Create {{entity_name}} record",
                "Created record",
                status="201",
                request_body=True,
                request_schema=_record_body_schema(schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
        }}
        paths[f"/entities/{{entity_name}}/records"]["get"]["parameters"] = [
            {{"name": "filter.<field>", "in": "query", "required": False, "description": "Exact field filter"}},
            {{"name": "sort", "in": "query", "required": False, "description": "Field to sort by"}},
            {{"name": "order", "in": "query", "required": False, "description": "asc or desc"}},
            {{"name": "limit", "in": "query", "required": False, "description": "Maximum records to return"}},
            {{"name": "offset", "in": "query", "required": False, "description": "Records to skip"}},
            {{"name": "include_deleted", "in": "query", "required": False, "description": "Admin-only flag to include soft-deleted records"}},
        ]
        paths[f"/entities/{{entity_name}}/records/export"] = {{
            "get": _api_operation(
                f"Export {{entity_name}} records",
                "Record export",
                response_schema=_record_export_response_schema(schema_name),
            ),
        }}
        paths[f"/entities/{{entity_name}}/records/import"] = {{
            "post": _api_operation(
                f"Import {{entity_name}} records",
                "Record import",
                request_body=True,
                request_schema=_record_import_body_schema(schema_name),
                response_schema=_record_import_response_schema(schema_name),
            ),
        }}
        paths[f"/entities/{{entity_name}}/records/{{{{id}}}}"] = {{
            "get": _api_operation(
                f"Fetch {{entity_name}} record",
                "Record",
                response_schema=_record_item_response_schema(schema_name),
            ),
            "put": _api_operation(
                f"Update {{entity_name}} record",
                "Updated record",
                request_body=True,
                request_schema=_record_body_schema(patch_schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
            "delete": _api_operation(
                f"Delete {{entity_name}} record",
                "Deleted record",
                response_schema=_record_mutation_response_schema(schema_name, record_key="deleted"),
            ),
        }}
        paths[f"/ui/entities/{{entity_name}}"] = {{
            "get": _api_operation(f"Generated {{entity_name}} UI", "HTML entity screen"),
        }}
        if entity.get("type") == "database":
            paths[f"/databases/{{entity_name}}/schemas"] = {{
                "get": _api_operation(f"{{entity_name}} database schemas", "Database schema metadata", response_schema=_schema_ref("DatabaseSchemaCatalog")),
            }}
    for workflow_name in list_workflows():
        paths[f"/workflows/{{workflow_name}}"] = {{
            "get": _api_operation(f"Describe {{workflow_name}} workflow", "Workflow description", response_schema=_schema_ref("WorkflowSpec")),
        }}
        paths[f"/workflows/{{workflow_name}}/run"] = {{
            "post": _api_operation(f"Run {{workflow_name}} workflow", "Workflow run result", request_body=True, request_schema=_schema_ref("WorkflowRunRequest"), response_schema=_schema_ref("WorkflowRunResult")),
        }}
    if APG_CAPABILITIES is not None:
        paths["/rules/evaluate"] = {{"post": _api_operation("Evaluate capability rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult"))}}
        paths["/configuration/resolve"] = {{"post": _api_operation("Resolve capability configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}}
        paths["/configuration/validate"] = {{"post": _api_operation("Validate capability configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}}
        paths["/approval/plan"] = {{"post": _api_operation("Plan capability approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse"))}}
        paths["/streaming"] = {{"get": _api_operation("Streaming topology", "ByteWax streaming topology", response_schema=_schema_ref("StreamingTopology"))}}
        if hasattr(APG_CAPABILITIES, "list_capabilities"):
            for capability_name in APG_CAPABILITIES.list_capabilities():
                paths[f"/capabilities/{{capability_name}}/streaming"] = {{
                    "get": _api_operation(f"{{capability_name}} streaming contract", "Capability streaming contract", response_schema=_schema_ref("CapabilityStreamingContract")),
                }}
                paths[f"/capabilities/{{capability_name}}/health"] = {{
                    "get": _api_operation(f"{{capability_name}} health", "Capability health", response_schema=_schema_ref("CapabilityHealth")),
                }}
                paths[f"/capabilities/{{capability_name}}/rules/evaluate"] = {{
                    "post": _api_operation(f"Evaluate {{capability_name}} rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult")),
                }}
                paths[f"/capabilities/{{capability_name}}/configuration/resolve"] = {{
                    "post": _api_operation(f"Resolve {{capability_name}} configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }}
                paths[f"/capabilities/{{capability_name}}/configuration/validate"] = {{
                    "post": _api_operation(f"Validate {{capability_name}} configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }}
                paths[f"/capabilities/{{capability_name}}/approval/plan"] = {{
                    "post": _api_operation(f"Plan {{capability_name}} approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse")),
                }}
        route_index = getattr(APG_CAPABILITIES, "ui_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {{"get": _api_operation(f"Capability screen {{route}}", "Generated capability screen")}}
    if AI_AGENTS is not None:
        for agent_name in describe_application().get("ai_agents", []):
            paths[f"/agents/{{agent_name}}/invoke"] = {{
                "post": _api_operation(f"Invoke agent {{agent_name}}", "Agent invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }}
        for team_name in describe_application().get("ai_agent_teams", []):
            paths[f"/agent-teams/{{team_name}}/invoke"] = {{
                "post": _api_operation(f"Invoke agent team {{team_name}}", "Agent team invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }}
    if APG_APPLICATIONS is not None:
        route_index = getattr(APG_APPLICATIONS, "application_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {{"get": _api_operation(f"Application route {{route}}", "Generated application composition screen")}}
    return {{
        "openapi": "3.1.0",
        "info": {{
            "title": APG_APP_NAME,
            "version": APG_APP_VERSION,
            "description": APG_APP_DESCRIPTION,
        }},
        "paths": paths,
        "components": {{
            "schemas": schemas,
            "securitySchemes": {{
                "ApiKeyAuth": {{"type": "apiKey", "in": "header", "name": "X-APG-API-Key"}},
                "BearerAuth": {{"type": "http", "scheme": "bearer"}},
            }},
        }},
    }}


def openapi_document() -> Dict[str, Any]:
    if _APG_OPENAPI_SPEC is None:
        return _build_openapi_document()
    return json.loads(json.dumps(_APG_OPENAPI_SPEC))


def validate_component_manifest_contract() -> Dict[str, Any]:
    manifest = component_manifest()
    openapi = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    interfaces = manifest.get("interfaces", {{}})
    http = interfaces.get("http", {{}}) if isinstance(interfaces, dict) else {{}}
    python = interfaces.get("python", {{}}) if isinstance(interfaces, dict) else {{}}
    http_paths = sorted(http.get("paths", [])) if isinstance(http, dict) else []
    expected_paths = sorted(openapi.get("paths", {{}}))
    if http.get("openapi") != "/openapi.json":
        errors.append("component manifest HTTP interface must point to /openapi.json")
    if http_paths != expected_paths:
        errors.append("component manifest HTTP paths do not match OpenAPI paths")
    exports = python.get("exports", []) if isinstance(python, dict) else []
    if not isinstance(exports, list) or not exports:
        errors.append("component manifest Python interface does not declare exports")
        exports = []
    export_names: list[str] = []
    for export_name in exports:
        if not isinstance(export_name, str):
            errors.append("component manifest Python exports must be strings")
            continue
        export_names.append(export_name)
    missing_exports = [
        export_name
        for export_name in export_names
        if export_name not in globals() or not callable(globals()[export_name])
    ]
    for export_name in missing_exports:
        errors.append(f"component manifest Python export {{export_name}} is not callable")
    expected_record_names = sorted(ENTITY_NAMES)
    manifest_record_names = sorted(interfaces.get("records", [])) if isinstance(interfaces, dict) else []
    if manifest_record_names != expected_record_names:
        errors.append("component manifest record interface does not match generated entities")
    if interfaces.get("theme") != "/theme.css":
        errors.append("component manifest theme interface must point to /theme.css")
    if interfaces.get("semantic_model") != "/semantic-model.json":
        errors.append("component manifest semantic model interface must point to /semantic-model.json")
    deployment = manifest.get("deployment", {{}})
    expected_artifacts = ["app.py", "__init__.py", "README.md", "semantic_model.json", "requirements.txt", "Dockerfile", ".dockerignore", ".env.example", "smoke_test.py"]
    raw_artifacts = deployment.get("artifacts", []) if isinstance(deployment, dict) else []
    artifacts: set[str] = set()
    if not isinstance(raw_artifacts, list):
        errors.append("component manifest deployment artifacts must be an array")
        raw_artifacts = []
    for artifact in raw_artifacts:
        if not isinstance(artifact, str):
            errors.append("component manifest deployment artifacts must be strings")
            continue
        artifacts.add(artifact)
    unexpected_artifacts = sorted(artifacts.difference(expected_artifacts))
    for artifact in unexpected_artifacts:
        errors.append(f"component manifest deployment has unexpected artifact {{artifact}}")
    artifact_root = Path(__file__).resolve().parent if "__file__" in globals() else None
    for artifact in expected_artifacts:
        if artifact not in artifacts:
            errors.append(f"component manifest deployment is missing artifact {{artifact}}")
            continue
        if artifact_root is not None and not (artifact_root / artifact).exists():
            errors.append(f"component manifest deployment artifact {{artifact}} does not exist")
    commands = deployment.get("commands", {{}}) if isinstance(deployment, dict) else {{}}
    expected_commands = {{
        "run": "python app.py",
        "describe": "python app.py --describe",
        "semantic_model": "python app.py --semantic-model",
        "validate": "python app.py --validate",
        "self_test": "python app.py --self-test",
        "smoke_test": "python smoke_test.py",
    }}
    if not isinstance(commands, dict):
        errors.append("component manifest deployment commands must be an object")
        commands = {{}}
    for command_name, expected_command in expected_commands.items():
        actual_command = commands.get(command_name)
        if actual_command is None:
            errors.append(f"component manifest deployment is missing command {{command_name}}")
        elif actual_command != expected_command:
            errors.append(
                f"component manifest deployment command {{command_name}} must be {{expected_command!r}}"
            )
    environment = deployment.get("environment", []) if isinstance(deployment, dict) else []
    expected_environment = ["APG_HOST", "APG_PORT", "APG_DATA_FILE", "APG_API_KEY", "APG_DEBUG"]
    if environment != expected_environment:
        errors.append("component manifest deployment environment does not match generated runtime variables")
    return {{
        "errors": errors,
        "warnings": warnings,
        "http_path_count": len(http_paths),
        "python_exports": sorted(export_names),
        "artifact_count": len(artifacts),
        "command_count": len(commands),
    }}


def _walk_openapi_refs(value: Any, path: str = "$") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        raw_ref = value.get("$ref")
        if isinstance(raw_ref, str):
            refs.append((path + ".$ref", raw_ref))
        for key, child in value.items():
            if key == "$ref":
                continue
            refs.extend(_walk_openapi_refs(child, f"{{path}}.{{key}}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            refs.extend(_walk_openapi_refs(child, f"{{path}}[{{index}}]"))
    return refs


def validate_openapi_contract() -> Dict[str, Any]:
    document = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    paths = document.get("paths", {{}})
    schemas = document.get("components", {{}}).get("schemas", {{}})
    if not isinstance(paths, dict) or not paths:
        errors.append("OpenAPI document does not declare paths")
        paths = {{}}
    if not isinstance(schemas, dict):
        errors.append("OpenAPI document components.schemas must be an object")
        schemas = {{}}
    for schema_name, schema in sorted(schemas.items()):
        if not isinstance(schema, dict):
            errors.append(f"OpenAPI schema {{schema_name}} must be an object")
            continue
        properties = schema.get("properties", {{}})
        required = schema.get("required", [])
        if required and not isinstance(required, list):
            errors.append(f"OpenAPI schema {{schema_name}} required must be an array")
            continue
        if required and not isinstance(properties, dict):
            errors.append(f"OpenAPI schema {{schema_name}} declares required fields without object properties")
            continue
        for field_name in required:
            if not isinstance(field_name, str):
                errors.append(f"OpenAPI schema {{schema_name}} required field names must be strings")
            elif field_name not in properties:
                errors.append(f"OpenAPI schema {{schema_name}} requires missing property {{field_name}}")
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            errors.append(f"OpenAPI path {{route}} must be an object")
            continue
        for method, operation in sorted(path_item.items()):
            if method.lower() not in {{"get", "post", "put", "patch", "delete", "options", "head"}}:
                continue
            if not isinstance(operation, dict):
                errors.append(f"OpenAPI operation {{method.upper()}} {{route}} must be an object")
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict) or not responses:
                errors.append(f"OpenAPI operation {{method.upper()}} {{route}} does not declare responses")
    referenced_schemas: set[str] = set()
    for ref_path, ref in _walk_openapi_refs(document):
        prefix = "#/components/schemas/"
        if not ref.startswith(prefix):
            errors.append(f"OpenAPI reference {{ref}} at {{ref_path}} is not an internal component schema reference")
            continue
        schema_name = ref[len(prefix):]
        referenced_schemas.add(schema_name)
        if schema_name not in schemas:
            errors.append(f"OpenAPI reference {{ref}} at {{ref_path}} does not resolve")
    return {{
        "errors": sorted(errors),
        "warnings": warnings,
        "path_count": len(paths),
        "schema_count": len(schemas),
        "referenced_schemas": sorted(referenced_schemas),
    }}


def _route_dispatch_target(route: str, method: str) -> str | None:
    method = method.lower()
    route = route.rstrip("/") or "/"
    if method == "get":
        if route == "/theme.css":
            return "theme_stylesheet"
        if route == "/ui" or route.startswith("/ui/"):
            return "_ui_payload"
        if _capability_screen(route) is not None:
            return "_capability_screen_payload"
        if _application_screen(route) is not None:
            return "_application_screen_payload"
        if route in {{
            "/",
            "/manifest",
            "/application",
            "/component.json",
            "/semantic-model.json",
            "/health",
            "/validate",
            "/openapi.json",
            "/entities",
            "/workflows",
            "/workflows/runs",
            "/databases",
            "/databases/status",
            "/auth",
            "/events",
            "/metrics",
            "/self-test",
            "/records",
            "/relationships",
            "/storage",
            "/agents",
            "/applications",
            "/capabilities",
            "/streaming",
            "/routes",
            "/composition",
        }}:
            return "_route_payload"
        if route.startswith("/databases/") and route.endswith("/schemas"):
            return "_route_payload"
        if route.startswith("/workflows/runs/"):
            return "_route_payload"
        if route.startswith("/workflows/"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/streaming"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/health"):
            return "_route_payload"
        if route.startswith("/records/"):
            return "_records_payload_with_query"
        if route.startswith("/entities/") and "/records" in route:
            return "_records_payload_with_query"
        if route == "/jobs":
            return "_jobs_payload"
        if route.startswith("/jobs/"):
            return "_job_detail_payload"
        return None
    if method == "post":
        if route.startswith("/agents/") and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if (route.startswith("/agent-teams/") or route.startswith("/teams/")) and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if route.startswith("/records/") and route.endswith("/bulk"):
            return "_create_record_payload"
        if route.startswith("/records/") and route.count("/") == 5:
            return "_create_relationship_payload"
        if route.startswith("/records/") and route.count("/") == 2:
            return "_create_record_payload"
        if route.startswith("/entities/") and (route.endswith("/records") or route.endswith("/records/import")):
            return "_create_record_payload"
        if route in {{"/rules/evaluate", "/capabilities/rules/evaluate"}} or (
            route.startswith("/capabilities/") and route.endswith("/rules/evaluate")
        ):
            return "_rule_evaluation_payload"
        if route in {{"/configuration/resolve", "/capabilities/configuration/resolve"}} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/resolve")
        ):
            return "_configuration_payload"
        if route in {{"/configuration/validate", "/capabilities/configuration/validate"}} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/validate")
        ):
            return "_configuration_payload"
        if route in {{"/approval/plan", "/capabilities/approval/plan"}} or (
            route.startswith("/capabilities/") and route.endswith("/approval/plan")
        ):
            return "_approval_plan_payload"
        if route.startswith("/workflows/runs/") and route.endswith("/compensate"):
            return "_workflow_compensation_payload"
        if route.startswith("/workflows/runs/") and route.endswith("/resume"):
            return "_workflow_resume_payload"
        if route.startswith("/workflows/") and route.endswith("/run"):
            return "_workflow_run_payload"
        if route == "/jobs":
            return "_job_create_payload"
        if route.startswith("/jobs/") and route.endswith("/retry"):
            return "_job_retry_payload"
        return None
    if method == "put":
        if route.startswith("/records/") and "/{{id}}" in route:
            return "_update_record_payload"
        if route.startswith("/entities/") and "/records/{{id}}" in route:
            return "_update_record_payload"
        return None
    if method == "delete":
        if route.startswith("/records/") and route.count("/") == 5:
            return "_delete_relationship_payload"
        if route.startswith("/records/") and "/{{id}}" in route:
            return "_delete_record_payload"
        if route.startswith("/entities/") and "/records/{{id}}" in route:
            return "_delete_record_payload"
        return None
    return None


def validate_route_dispatch_contract() -> Dict[str, Any]:
    document = openapi_document()
    paths = document.get("paths", {{}})
    errors: list[str] = []
    warnings: list[str] = []
    route_targets: Dict[str, list[Dict[str, str]]] = {{}}
    method_count = 0
    if not isinstance(paths, dict):
        return {{
            "errors": ["OpenAPI paths must be an object before dispatch validation"],
            "warnings": warnings,
            "route_count": 0,
            "method_count": 0,
            "routes": route_targets,
        }}
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            continue
        for method in sorted(path_item):
            method_name = str(method).lower()
            if method_name not in {{"get", "post", "put", "patch", "delete", "options", "head"}}:
                continue
            method_count += 1
            target = _route_dispatch_target(str(route), method_name)
            if target is None:
                errors.append(f"OpenAPI route {{method_name.upper()}} {{route}} has no generated dispatcher")
                continue
            route_targets.setdefault(str(route), []).append({{"method": method_name.upper(), "target": target}})
    return {{
        "errors": errors,
        "warnings": warnings,
        "route_count": len(paths),
        "method_count": method_count,
        "routes": route_targets,
    }}


def _split_agent_literal_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'").strip('"') for item in text.split(",") if item.strip()]


def _entity_agent_team_descriptions() -> Dict[str, Dict[str, Any]]:
    descriptions: Dict[str, Dict[str, Any]] = {{}}
    for entity in ENTITIES:
        if str(entity.get("type", "")) != "agent_team":
            continue
        fields = {{
            str(field.get("name", "")): field
            for field in entity.get("fields", [])
            if isinstance(field, dict)
        }}
        agents = _split_agent_literal_list(fields.get("agents", {{}}).get("type", ""))
        capabilities = _split_agent_literal_list(fields.get("capabilities", {{}}).get("type", ""))
        flow_text = str(fields.get("flow", {{}}).get("type", ""))
        flow = []
        for edge in [part.strip() for part in flow_text.split(",") if part.strip()]:
            if "->" in edge:
                source, target = [piece.strip() for piece in edge.split("->", 1)]
                flow.append({{"source": source, "target": target, "condition": ""}})
        descriptions[str(entity.get("name", ""))] = {{
            "name": str(entity.get("name", "")),
            "agents": agents,
            "capabilities": capabilities,
            "flow": flow,
            "policy": {{}},
            "configuration": {{}},
            "rules": [],
            "ui": {{}},
            "theme": {{}},
            "source": "entity_metadata",
        }}
    return descriptions


def _semantic_agent_descriptions() -> Dict[str, Dict[str, Any]]:
    raw_agents = SEMANTIC_MODEL.get("agents", {{}})
    if not isinstance(raw_agents, dict):
        return {{}}
    descriptions: Dict[str, Dict[str, Any]] = {{}}
    for name, spec in raw_agents.items():
        if not isinstance(spec, dict):
            continue
        descriptions[str(name)] = {{
            "name": str(spec.get("name") or name),
            "role": spec.get("role"),
            "model": spec.get("model"),
            "runtime": spec.get("runtime"),
            "system": spec.get("system"),
            "capabilities": list(spec.get("capabilities", [])) if isinstance(spec.get("capabilities", []), list) else [],
            "tools": list(spec.get("tools", [])) if isinstance(spec.get("tools", []), list) else [],
            "memory": spec.get("memory"),
            "inputs": list(spec.get("inputs", [])) if isinstance(spec.get("inputs", []), list) else [],
            "outputs": list(spec.get("outputs", [])) if isinstance(spec.get("outputs", []), list) else [],
            "handoffs": list(spec.get("handoffs", [])) if isinstance(spec.get("handoffs", []), list) else [],
            "configuration": dict(spec.get("configuration", {{}})) if isinstance(spec.get("configuration", {{}}), dict) else {{}},
            "rules": list(spec.get("rules", [])) if isinstance(spec.get("rules", []), list) else [],
            "ui": dict(spec.get("ui", {{}})) if isinstance(spec.get("ui", {{}}), dict) else {{}},
            "theme": dict(spec.get("theme", {{}})) if isinstance(spec.get("theme", {{}}), dict) else {{}},
            "source": "semantic_model",
        }}
    return descriptions


def describe_application() -> Dict[str, Any]:
    _entity_summary_keys = {{"name", "type", "properties", "methods"}}
    description: Dict[str, Any] = {{
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "entities": [
            {{k: v for k, v in entity.items() if k in _entity_summary_keys}}
            for entity in list_entities()
        ],
        "databases": list_databases(),
    }}
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agents"] = AI_AGENTS.list_agents()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_agent") and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agent_descriptions"] = {{
            name: AI_AGENTS.describe_agent(name)
            for name in AI_AGENTS.list_agents()
        }}
    semantic_agent_descriptions = _semantic_agent_descriptions()
    if semantic_agent_descriptions:
        description["ai_agent_descriptions"] = {{
            **semantic_agent_descriptions,
            **description.get("ai_agent_descriptions", {{}}),
        }}
        description["ai_agents"] = sorted(set(description.get("ai_agents", [])) | set(semantic_agent_descriptions))
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_teams"] = AI_AGENTS.list_agent_teams()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_team") and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_team_descriptions"] = {{
            name: AI_AGENTS.describe_team(name)
            for name in AI_AGENTS.list_agent_teams()
        }}
    entity_team_descriptions = _entity_agent_team_descriptions()
    if entity_team_descriptions:
        description["ai_agent_team_descriptions"] = {{
            **entity_team_descriptions,
            **description.get("ai_agent_team_descriptions", {{}}),
        }}
        description["ai_agent_teams"] = sorted(set(description.get("ai_agent_teams", [])) | set(entity_team_descriptions))
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "list_applications"):
        description["application_compositions"] = APG_APPLICATIONS.list_applications()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "describe_application_compositions"):
        description["application_composition_descriptions"] = APG_APPLICATIONS.describe_application_compositions()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_dependency_graph"):
        description["application_dependency_graph"] = APG_APPLICATIONS.application_dependency_graph()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_component_catalog"):
        description["application_component_catalog"] = APG_APPLICATIONS.application_component_catalog()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_route_index"):
        description["application_routes"] = APG_APPLICATIONS.application_route_index()
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


def validate_database_schema_contracts() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    validated: list[str] = []
    for database in list_databases():
        database_name = str(database.get("name", "database"))
        validated.append(database_name)
        schemas = database.get("schemas", [])
        if not schemas:
            warnings.append(f"{{database_name}} does not declare schemas")
            continue
        table_index: Dict[str, list[Dict[str, Any]]] = {{}}
        seen_schemas: set[str] = set()
        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            schema_key = schema_name.lower()
            if schema_key in seen_schemas:
                errors.append(f"{{database_name}} declares duplicate schema {{schema_name}}")
            seen_schemas.add(schema_key)
            seen_tables: set[str] = set()
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    errors.append(f"{{database_name}}.{{schema_name}} declares a table without a name")
                    continue
                table_key = table_name.lower()
                qualified_key = f"{{schema_name}}.{{table_name}}".lower()
                if table_key in seen_tables:
                    errors.append(f"{{database_name}}.{{schema_name}} declares duplicate table {{table_name}}")
                seen_tables.add(table_key)
                table_index.setdefault(table_key, []).append(table)
                table_index.setdefault(qualified_key, []).append(table)

        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                columns = table.get("columns", [])
                column_names = [str(column.get("name", "")) for column in columns if isinstance(column, dict)]
                known_columns = {{column_name.lower() for column_name in column_names if column_name}}
                if len(known_columns) != len([column_name for column_name in column_names if column_name]):
                    errors.append(f"{{database_name}}.{{schema_name}}.{{table_name}} declares duplicate columns")
                if columns and not any(bool(column.get("primary_key")) for column in columns if isinstance(column, dict)):
                    warnings.append(f"{{database_name}}.{{schema_name}}.{{table_name}} does not declare a primary key")
                for index in table.get("indexes", []):
                    for indexed_column in index.get("columns", []):
                        if str(indexed_column).lower() not in known_columns:
                            errors.append(
                                f"{{database_name}}.{{schema_name}}.{{table_name}} index references unknown column {{indexed_column}}"
                            )
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    reference = column.get("reference")
                    if not isinstance(reference, dict):
                        continue
                    target_table_name = str(reference.get("table", ""))
                    target_column_name = str(reference.get("column", ""))
                    target_schema_name = str(reference.get("schema", ""))
                    target_label = (
                        f"{{target_schema_name}}.{{target_table_name}}"
                        if target_schema_name
                        else target_table_name
                    )
                    if target_schema_name:
                        candidates = table_index.get(f"{{target_schema_name}}.{{target_table_name}}".lower(), [])
                    else:
                        candidates = table_index.get(f"{{schema_name}}.{{target_table_name}}".lower(), [])
                        if not candidates:
                            candidates = table_index.get(target_table_name.lower(), [])
                    if not candidates:
                        errors.append(
                            f"{{database_name}}.{{schema_name}}.{{table_name}}.{{column.get('name')}} references unknown table {{target_label}}"
                        )
                        continue
                    if len(candidates) > 1:
                        errors.append(
                            f"{{database_name}}.{{schema_name}}.{{table_name}}.{{column.get('name')}} references ambiguous table {{target_label}}; use schema-qualified target"
                        )
                        continue
                    target_table = candidates[0]
                    target_columns = {{
                        str(target_column.get("name", "")).lower()
                        for target_column in target_table.get("columns", [])
                        if isinstance(target_column, dict)
                    }}
                    if target_column_name.lower() not in target_columns:
                        errors.append(
                            f"{{database_name}}.{{schema_name}}.{{table_name}}.{{column.get('name')}} references unknown column {{target_label}}.{{target_column_name}}"
                        )
    return {{"errors": errors, "warnings": warnings, "validated_databases": sorted(validated)}}


def validate_workflow_contracts() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for workflow_name in list_workflows():
        workflow = describe_workflow(workflow_name)
        steps = workflow.get("steps", [])
        step_set = set(str(step) for step in steps)
        if not steps:
            warnings.append(f"{{workflow_name}} does not declare executable steps")
        transitions = workflow.get("transitions", [])
        if len(steps) > 1 and len(transitions) != len(steps) - 1:
            errors.append(f"{{workflow_name}} transition count does not match step chain")
        for section in ("guards", "assignments", "timers", "waits", "retry_policy", "compensation"):
            mapping = workflow.get(section, {{}})
            if not isinstance(mapping, dict):
                errors.append(f"{{workflow_name}} {{section}} metadata must be an object")
                continue
            for step in mapping:
                if str(step) not in step_set:
                    errors.append(f"{{workflow_name}} {{section}} references unknown step {{step}}")
        assignments = workflow.get("assignments", {{}})
        for step in workflow.get("human_tasks", []):
            if str(step) not in step_set:
                errors.append(f"{{workflow_name}} human task references unknown step {{step}}")
            elif str(step) not in assignments:
                warnings.append(f"{{workflow_name}} human task {{step}} has no assignee")
    return {{"errors": errors, "warnings": warnings, "validated_workflows": list_workflows()}}


def validate_application(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {{
        "name": MODULE_NAME,
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {{}},
    }}
    _record_validation(report, "openapi_contract", validate_openapi_contract())
    _record_validation(report, "component_manifest", validate_component_manifest_contract())
    _record_validation(report, "route_dispatch", validate_route_dispatch_contract())
    _record_validation(report, "database_schemas", validate_database_schema_contracts())
    _record_validation(report, "workflows", validate_workflow_contracts())
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        _record_validation(
            report,
            "ai_agent_runtimes",
            AI_AGENTS.validate_agent_runtimes(available_agent_runtimes),
        )
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "validate_application_compositions"):
        available_capabilities = APG_CAPABILITIES.list_capabilities() if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") else []
        available_agents = AI_AGENTS.list_agents() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents") else []
        available_teams = AI_AGENTS.list_agent_teams() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams") else []
        _record_validation(
            report,
            "application_compositions",
            APG_APPLICATIONS.validate_application_compositions(
                available_capabilities=available_capabilities,
                available_agents=available_agents,
                available_teams=available_teams,
            ),
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


def _css_name(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "value"


def theme_stylesheet() -> str:
    lines = [
        ":root {{",
        "  --apg-primary: #1E5B5A;",
        "  --apg-accent: #D97706;",
        "  --apg-surface: #ffffff;",
        "  --apg-border: #d0d7de;",
        "  --apg-text: #1f2328;",
        "  --apg-muted: #59636e;",
        "  --apg-bg-canvas: #f6f8fa;",
        "  --apg-bg-card: var(--apg-surface);",
        "  --apg-bg-hover: rgba(0,0,0,0.04);",
        "}}",
        "@media (prefers-color-scheme: dark) {{ :root:not([data-theme='light']) {{ --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); }} }}",
        ":root[data-theme='dark'], :root.dark {{ --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); }}",
        "body.apg-density-compact .apg-table th, body.apg-density-compact .apg-table td, body.apg-density-compact table th, body.apg-density-compact table td {{ padding-top: 0.75rem; padding-bottom: 0.75rem; }}",
        "body.apg-density-comfortable .apg-table th, body.apg-density-comfortable .apg-table td, body.apg-density-comfortable table th, body.apg-density-comfortable table td {{ padding-top: 1rem; padding-bottom: 1rem; }}",
        "body.apg-density-spacious .apg-table th, body.apg-density-spacious .apg-table td, body.apg-density-spacious table th, body.apg-density-spacious table td {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}",
        "@media (prefers-color-scheme: dark) {{",
        "  :root {{",
        "    --apg-bg: #0f172a;",
        "    --apg-surface: #1e293b;",
        "    --apg-surface-2: #334155;",
        "    --apg-text: #e2e8f0;",
        "    --apg-text-muted: #94a3b8;",
        "    --apg-border: #334155;",
        "    --apg-input-bg: #1e293b;",
        "    --apg-shadow: rgba(0,0,0,0.4);",
        "  }}",
        "  body {{ background: var(--apg-bg); color: var(--apg-text); }}",
        "  .apg-sidebar {{ background: var(--apg-surface); border-color: var(--apg-border); }}",
        "  .apg-header {{ background: var(--apg-surface); border-color: var(--apg-border); }}",
        "  table {{ background: var(--apg-surface); color: var(--apg-text); }}",
        "  th {{ background: var(--apg-surface-2); color: var(--apg-text); }}",
        "  td {{ border-color: var(--apg-border); }}",
        "  input, select, textarea {{ background: var(--apg-input-bg); color: var(--apg-text); border-color: var(--apg-border); }}",
        "  .apg-card {{ background: var(--apg-surface); border-color: var(--apg-border); box-shadow: 0 1px 3px var(--apg-shadow); }}",
        "}}",
        "html[data-theme='dark'] {{",
        "  --apg-bg: #0f172a;",
        "  --apg-surface: #1e293b;",
        "  --apg-surface-2: #334155;",
        "  --apg-text: #e2e8f0;",
        "  --apg-text-muted: #94a3b8;",
        "  --apg-border: #334155;",
        "  --apg-input-bg: #1e293b;",
        "  --apg-shadow: rgba(0,0,0,0.4);",
        "}}",
        "@media (max-width: 768px) {{",
        "  .apg-sidebar {{ display: none; }}",
        "  .apg-sidebar.apg-menu-open {{ display: block; position: fixed; inset: 0; z-index: 100; overflow-y: auto; }}",
        "  .apg-hamburger {{ display: block !important; }}",
        "  .apg-main {{ margin-left: 0 !important; }}",
        "  table {{ display: none; }}",
        "  .apg-card-list {{ display: block; }}",
        "  .apg-table-row-card {{ display: block; padding: 1rem; border: 1px solid var(--apg-border,#e2e8f0); border-radius: 8px; margin-bottom: 0.75rem; }}",
        "  .apg-table-row-card .apg-card-label {{ font-weight: 600; color: var(--apg-text-muted,#64748b); font-size: 0.75rem; text-transform: uppercase; }}",
        "}}",
        "@media (max-width: 480px) {{",
        "  body {{ font-size: 0.875rem; }}",
        "  h1 {{ font-size: 1.25rem; }} h2 {{ font-size: 1.1rem; }}",
        "  .apg-btn {{ padding: 0.375rem 0.625rem; font-size: 0.8rem; }}",
        "}}",
        "@media print {{",
        "  .apg-sidebar, .apg-header, .apg-actions, .apg-form, nav, button, .apg-btn {{ display: none !important; }}",
        "  .apg-main {{ margin: 0 !important; padding: 0 !important; }}",
        "  table {{ width: 100%; border-collapse: collapse; }}",
        "  th, td {{ border: 1px solid #000; padding: 0.25rem 0.5rem; }}",
        "  body {{ color: #000; background: #fff; font-size: 11pt; }}",
        "  tr {{ page-break-inside: avoid; }}",
        "}}",
    ]
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_theme"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            try:
                theme = APG_CAPABILITIES.capability_theme(capability_name)
            except KeyError:
                continue
            theme_name = _css_name(str(theme.get("name") or capability_name))
            tokens = theme.get("tokens", {{}})
            if isinstance(tokens, dict):
                for token_name, token_value in sorted(tokens.items()):
                    css_var = f"--apg-theme-{{theme_name}}-{{_css_name(str(token_name))}}"
                    lines.append(":root {{ " + css_var + ": " + str(token_value) + "; }}")
                    if str(token_name).lower() in {{"accent", "primary", "brand"}}:
                        lines.append(":root {{ --apg-accent: var(" + css_var + "); }}")
    return "\\n".join(lines) + "\\n"
    lines.extend([
        # Extended spacing + radius + shadow tokens
        ":root {{ --apg-radius: 8px; --apg-radius-sm: 4px; --apg-radius-full: 9999px; }}",
        ":root {{ --apg-shadow-sm: 0 1px 2px rgba(0,0,0,0.08); --apg-shadow-md: 0 4px 6px rgba(0,0,0,0.10); --apg-shadow-lg: 0 10px 15px rgba(0,0,0,0.12); }}",
        ":root {{ --apg-sidebar-width: 240px; --apg-topbar-height: 56px; }}",
        ":root {{ --apg-font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; --apg-font-mono: ui-monospace, 'Cascadia Code', 'Fira Mono', monospace; }}",
        ":root {{ --apg-space-1: 4px; --apg-space-2: 8px; --apg-space-3: 12px; --apg-space-4: 16px; --apg-space-6: 24px; --apg-space-8: 32px; }}",
        ":root {{ --apg-duration-fast: 150ms; --apg-duration-base: 200ms; }}",
        ":root {{ --apg-bg-canvas: #f6f8fa; --apg-bg-card: var(--apg-surface); --apg-bg-hover: rgba(0,0,0,0.04); }}",
        # Dark mode
        "@media (prefers-color-scheme: dark) {{ :root {{ --apg-surface: #1e2028; --apg-border: #30363d; --apg-text: #e6edf3; --apg-muted: #8b949e; --apg-bg-canvas: #0d1117; --apg-bg-card: #161b22; --apg-bg-hover: rgba(255,255,255,0.06); }} }}",
        # Base styles
        "*, *::before, *::after {{ box-sizing: border-box; }}",
        "body {{ margin: 0; font-family: var(--apg-font-sans); color: var(--apg-text); background: var(--apg-bg-canvas); line-height: 1.5; font-size: 14px; }}",
        "h1 {{ margin: 0 0 var(--apg-space-4); font-size: 1.5rem; font-weight: 600; color: var(--apg-text); }}",
        "h2 {{ margin: var(--apg-space-6) 0 var(--apg-space-3); font-size: 1.125rem; font-weight: 600; color: var(--apg-text); }}",
        "h3 {{ margin: var(--apg-space-4) 0 var(--apg-space-2); font-size: 1rem; font-weight: 600; color: var(--apg-text); }}",
        "a {{ color: var(--apg-accent); text-decoration: none; transition: opacity var(--apg-duration-fast); }}",
        "a:hover {{ text-decoration: underline; opacity: 0.85; }}",
        "p {{ margin: 0 0 var(--apg-space-3); }}",
        # Topbar layout shell
        ".apg-topbar {{ position: sticky; top: 0; z-index: 100; display: flex; align-items: center; gap: var(--apg-space-4); height: var(--apg-topbar-height); padding: 0 var(--apg-space-6); border-bottom: 1px solid var(--apg-border); background: var(--apg-surface); box-shadow: var(--apg-shadow-sm); }}",
        ".apg-logo {{ font-weight: 700; font-size: 1rem; color: var(--apg-accent) !important; text-decoration: none !important; letter-spacing: -0.02em; }}",
        ".apg-topnav {{ display: flex; align-items: center; gap: var(--apg-space-1); flex: 1; }}",
        ".apg-content {{ max-width: 1280px; margin: 0 auto; padding: var(--apg-space-6); }}",
        # Nav links
        ".apg-nav-link {{ display: inline-flex; align-items: center; padding: var(--apg-space-2) var(--apg-space-3); border-radius: var(--apg-radius-sm); font-size: 0.875rem; color: var(--apg-text); text-decoration: none !important; transition: background var(--apg-duration-fast); white-space: nowrap; }}",
        ".apg-nav-link:hover {{ background: var(--apg-bg-hover); text-decoration: none !important; opacity: 1; }}",
        ".apg-nav-link.active {{ background: var(--apg-bg-hover); font-weight: 500; }}",
        # Card
        ".apg-card {{ background: var(--apg-bg-card); border: 1px solid var(--apg-border); border-radius: var(--apg-radius); box-shadow: var(--apg-shadow-sm); padding: var(--apg-space-4); margin-bottom: var(--apg-space-4); }}",
        ".apg-card-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--apg-space-3); padding-bottom: var(--apg-space-3); border-bottom: 1px solid var(--apg-border); }}",
        # Table
        ".apg-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}",
        ".apg-table thead {{ background: var(--apg-bg-canvas); }}",
        ".apg-table th {{ padding: var(--apg-space-2) var(--apg-space-3); text-align: left; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--apg-muted); border-bottom: 2px solid var(--apg-border); white-space: nowrap; }}",
        ".apg-table td {{ padding: var(--apg-space-2) var(--apg-space-3); border-bottom: 1px solid var(--apg-border); vertical-align: middle; }}",
        ".apg-table tbody tr:hover {{ background: var(--apg-bg-hover); }}",
        ".apg-table-wrap {{ overflow-x: auto; border: 1px solid var(--apg-border); border-radius: var(--apg-radius); background: var(--apg-bg-card); }}",
        # Badge
        ".apg-badge {{ display: inline-flex; align-items: center; padding: 2px var(--apg-space-2); border-radius: var(--apg-radius-full); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; line-height: 1.6; }}",
        ".apg-badge-success {{ background: #dcfce7; color: #166534; }}",
        ".apg-badge-warning {{ background: #fef9c3; color: #854d0e; }}",
        ".apg-badge-danger {{ background: #fee2e2; color: #991b1b; }}",
        ".apg-badge-info {{ background: #dbeafe; color: #1e40af; }}",
        ".apg-badge-neutral {{ background: var(--apg-bg-hover); color: var(--apg-muted); }}",
        # Form
        "form, .apg-form {{ padding: var(--apg-space-4); background: var(--apg-bg-card); border: 1px solid var(--apg-border); border-radius: var(--apg-radius); box-shadow: var(--apg-shadow-sm); }}",
        "label {{ display: block; margin-bottom: var(--apg-space-1); font-size: 0.875rem; font-weight: 500; color: var(--apg-text); }}",
        "input, select, textarea {{ width: 100%; max-width: 480px; padding: var(--apg-space-2) var(--apg-space-3); border: 1px solid var(--apg-border); border-radius: var(--apg-radius-sm); background: var(--apg-surface); color: var(--apg-text); font-family: var(--apg-font-sans); font-size: 0.875rem; transition: border-color var(--apg-duration-fast); outline: none; }}",
        "input:focus, select:focus, textarea:focus {{ border-color: var(--apg-accent); box-shadow: 0 0 0 3px rgba(18,110,130,0.12); }}",
        ".apg-field {{ margin-bottom: var(--apg-space-4); }}",
        # Button
        "button, .apg-btn {{ display: inline-flex; align-items: center; gap: var(--apg-space-2); padding: var(--apg-space-2) var(--apg-space-4); border: 1px solid var(--apg-accent); border-radius: var(--apg-radius-sm); background: var(--apg-accent); color: white; font-family: var(--apg-font-sans); font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: opacity var(--apg-duration-fast); line-height: 1.5; }}",
        "button:hover, .apg-btn:hover {{ opacity: 0.88; }}",
        ".apg-btn-secondary {{ background: var(--apg-surface); color: var(--apg-text); border-color: var(--apg-border); }}",
        ".apg-btn-danger {{ background: #dc2626; border-color: #dc2626; }}",
        # Alert / notice
        "[role=alert] {{ padding: var(--apg-space-3) var(--apg-space-4); background: #fef9c3; border: 1px solid #fde68a; border-radius: var(--apg-radius-sm); margin-bottom: var(--apg-space-4); font-size: 0.875rem; }}",
        ".apg-error-page {{ display: flex; flex-direction: column; justify-content: center; min-height: 60vh; max-width: 720px; margin: 0 auto; padding: 2rem; color: var(--apg-muted); text-align: center; }}",
        ".apg-error-page h1 {{ color: var(--apg-text); }}",
        # Code / pre
        "pre {{ padding: var(--apg-space-4); overflow: auto; background: var(--apg-bg-canvas); border: 1px solid var(--apg-border); border-left: 3px solid var(--apg-accent); border-radius: var(--apg-radius); font-family: var(--apg-font-mono); font-size: 0.8rem; line-height: 1.6; }}",
        "code {{ font-family: var(--apg-font-mono); font-size: 0.85em; color: var(--apg-accent); background: var(--apg-bg-hover); padding: 1px 5px; border-radius: 3px; }}",
        "pre code {{ background: transparent; padding: 0; color: inherit; }}",
        # Stat card
        ".apg-stat {{ display: flex; flex-direction: column; gap: var(--apg-space-1); }}",
        ".apg-stat-value {{ font-size: 1.75rem; font-weight: 700; color: var(--apg-text); line-height: 1; }}",
        ".apg-stat-label {{ font-size: 0.75rem; color: var(--apg-muted); text-transform: uppercase; letter-spacing: 0.05em; }}",
        ".apg-stat-delta {{ font-size: 0.8rem; font-weight: 500; }}",
        ".apg-stat-delta.up {{ color: #16a34a; }} .apg-stat-delta.down {{ color: #dc2626; }}",
        # Grid helpers
        ".apg-grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--apg-space-4); }}",
        ".apg-grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--apg-space-4); }}",
        ".apg-grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--apg-space-4); }}",
        "@media (max-width: 768px) {{ .apg-grid-2, .apg-grid-3, .apg-grid-4 {{ grid-template-columns: 1fr; }} }}",
        # Utility
        ".apg-flex {{ display: flex; align-items: center; }} .apg-flex-between {{ justify-content: space-between; }}",
        ".apg-mt-4 {{ margin-top: var(--apg-space-4); }} .apg-mb-4 {{ margin-bottom: var(--apg-space-4); }}",
        ".apg-text-muted {{ color: var(--apg-muted); }} .apg-text-sm {{ font-size: 0.875rem; }}",
        ".apg-sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }}",
    ])
    return "\\n".join(lines) + "\\n"


def _html_page(title: str, body: str, shell: bool = True) -> str:
    safe_title = html.escape(title)
    safe_module = html.escape(MODULE_NAME)
    locale = _active_locale()
    direction = _text_direction(locale)
    safe_locale = html.escape(locale, quote=True)
    safe_direction = html.escape(direction, quote=True)
    script_nonce_attr = _script_nonce_attr()
    try:
        current_path = _flask_request.path
    except RuntimeError:
        current_path = "/ui"

    def _shell_link(href: str, label: str, class_name: str = "apg-sidebar-link", exact: bool = False) -> str:
        active = current_path == href if exact else (current_path == href or current_path.startswith(href + "/"))
        aria = ' aria-current="page"' if active else ""
        return f'<a class="{{class_name}}" href="{{html.escape(href, quote=True)}}"{{aria}}>{{html.escape(label)}}</a>'

    entity_nav = "".join(
        _shell_link(f'/ui/entities/{{quote(str(entity["name"]), safe="")}}', str(entity["name"]))
        for entity in ENTITIES
        if entity.get("type") not in {{"application"}}
    ) or f'<span class="apg-sidebar-empty">{{html.escape(_("no_records"))}}</span>'
    app = describe_application()
    agent_nav = "".join(
        _shell_link(f'/ui/agents/{{quote(str(name), safe="")}}', str(name))
        for name in sorted(app.get("ai_agent_descriptions", {{}}))
    )
    team_nav = "".join(
        _shell_link(f'/ui/agent-teams/{{quote(str(name), safe="")}}', str(name))
        for name in sorted(app.get("ai_agent_team_descriptions", {{}}))
    )
    shell_commands = [
        {{"label": "Dashboard", "url": "/ui", "kind": "Navigate", "hint": "Open generated home dashboard"}},
        {{"label": "Workflows", "url": "/ui/workflows", "kind": "Navigate", "hint": "Run guided operational flows"}},
        {{"label": "Databases", "url": "/ui/databases", "kind": "Navigate", "hint": "Inspect schemas and generated tables"}},
        {{"label": "Marketplace", "url": "/ui/marketplace", "kind": "Navigate", "hint": "Compare generated integration blueprints"}},
        {{"label": "Flow debugger", "url": "/ui/debug", "kind": "Operate", "hint": "Replay workflow runs"}},
        {{"label": "OpenAPI contract", "url": "/openapi.json", "kind": "Developer", "hint": "Inspect generated API contract"}},
        {{"label": "Self-test", "url": "/self-test", "kind": "Developer", "hint": "Check runtime health"}},
    ]
    for entity in ENTITIES:
        if entity.get("type") not in {{"application"}}:
            entity_name = str(entity.get("name", ""))
            shell_commands.append({{"label": entity_name, "url": f"/ui/entities/{{quote(entity_name, safe='')}}", "kind": "Entity", "hint": "Open generated data workspace"}})
    for name in sorted(app.get("ai_agent_descriptions", {{}})):
        shell_commands.append({{"label": str(name), "url": f"/ui/agents/{{quote(str(name), safe='')}}", "kind": "Agent", "hint": "Open agent console"}})
    for name in sorted(app.get("ai_agent_team_descriptions", {{}})):
        shell_commands.append({{"label": str(name), "url": f"/ui/agent-teams/{{quote(str(name), safe='')}}", "kind": "Team", "hint": "Open team console"}})
    shell_command_json = json.dumps(shell_commands)
    current_user = _current_user() if APG_AUTH_REQUIRED else None
    user_menu = ""
    if current_user:
        display_name = html.escape(str(current_user.get("name") or current_user.get("username") or "User"))
        initials = "".join(part[:1].upper() for part in display_name.split()[:2]) or "U"
        user_menu = (
            '<form method="post" action="/logout" class="apg-user-menu">'
            f'{{_csrf_input()}}'
            f'<span class="apg-avatar" aria-hidden="true">{{html.escape(initials)}}</span>'
            f'<span class="apg-user-name">{{display_name}}</span>'
            f'<button class="apg-btn apg-btn-secondary" type="submit">{{_("logout")}}</button>'
            '</form>'
        )
    language_menu = ""
    if len(APG_SUPPORTED_LANGUAGES) > 1:
        try:
            next_url = _flask_request.full_path.rstrip("?") or "/ui"
        except RuntimeError:
            next_url = "/ui"
        options = "".join(
            f'<option value="{{html.escape(language, quote=True)}}"{{" selected" if language == locale else ""}}>{{html.escape(language)}}</option>'
            for language in APG_SUPPORTED_LANGUAGES
        )
        language_menu = (
            '<form method="post" action="/locale" class="apg-locale-form">'
            f'{{_csrf_input()}}'
            f'<input type="hidden" name="next" value="{{html.escape(next_url, quote=True)}}">'
            f'<label class="apg-sr-only" for="apg-locale-select">{{_("language")}}</label>'
            f'<select id="apg-locale-select" name="lang" class="apg-locale-select" onchange="this.form.submit()" aria-label="{{_("language")}}">{{options}}</select>'
            '</form>'
        )
    sidebar_html = (
        '<aside id="apg-sidebar" class="apg-sidebar" aria-label="Application navigation">'
        '<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Navigate</p>'
        + _shell_link("/ui", "Dashboard", exact=True)
        + _shell_link("/ui/workflows", "Workflows")
        + _shell_link("/ui/databases", "Databases")
        + _shell_link("/ui/marketplace", "Marketplace")
        + '</div>'
        f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Entities</p>{{entity_nav}}</div>'
        + (f'<div class="apg-sidebar-section"><p class="apg-sidebar-heading">Agents</p>{{agent_nav}}{{team_nav}}</div>' if agent_nav or team_nav else "")
        + '</aside><div id="apg-sidebar-backdrop" class="apg-sidebar-backdrop" onclick="apgCloseSidebar()"></div>'
    )
    head_extras = (
        '<script' + script_nonce_attr + '>(function(){{try{{var m=localStorage.getItem("apg-theme")||"system";var d=document.documentElement;if(m==="dark"||m==="light")d.setAttribute("data-theme",m);else d.removeAttribute("data-theme");d.dataset.themeMode=m;}}catch(e){{}}}})();</script>'
        '<meta name="theme-color" content="#1E5B5A">'
        '<link rel="manifest" href="/static/manifest.webmanifest">'
        '<link rel="stylesheet" href="/static/apg.css">'
        '<link rel="stylesheet" href="/static/uplot.min.css">'
        '<script defer src="/static/htmx.min.js"></script>'
        '<script defer src="/static/sortable.min.js"></script>'
        '<script defer src="/static/uplot.min.js"></script>'
        '<script defer src="/static/apg-charts.js"></script>'
        '<script defer src="/static/apg-sse.js"></script>'
    )
    confirm_label = html.escape(_("confirm"))
    cancel_label = html.escape(_("cancel"))
    delete_label = html.escape(_("delete"))
    delete_prompt = json.dumps(_("delete") + " this record?")
    toast_js = (
        '<div id="apg-toast-root" role="status" aria-live="polite" aria-atomic="true" class="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 pointer-events-none"></div>'
        '<dialog id="apg-confirm-dialog" class="apg-dialog">'
        '<form method="dialog" class="apg-dialog-panel">'
        f'<h2 id="apg-confirm-title">{{confirm_label}}</h2>'
        '<p id="apg-confirm-message" class="text-sm text-gray-600">Are you sure?</p>'
        '<div class="flex items-center justify-end gap-2 mt-4">'
        f'<button value="cancel" class="apg-btn apg-btn-secondary" type="submit">{{cancel_label}}</button>'
        f'<button value="confirm" class="apg-btn apg-btn-danger" type="submit">{{delete_label}}</button>'
        '</div></form></dialog>'
        f'<script{{script_nonce_attr}}>'
        'var _apgNotifications=[];var _apgDeferredInstall=null;var _apgWasOffline=false;'
        'function apgClearChildren(el){{while(el&&el.firstChild)el.removeChild(el.firstChild);}}'
        'function apgRenderNotifications(){{var list=document.getElementById("apg-notification-list");var dot=document.getElementById("apg-notification-dot");if(!list)return;apgClearChildren(list);if(!_apgNotifications.length){{var empty=document.createElement("p");empty.className="apg-notification-meta";empty.textContent="No notifications yet.";list.appendChild(empty);if(dot)dot.hidden=true;return;}}_apgNotifications.slice(0,6).forEach(function(n){{var article=document.createElement("article");article.className="apg-notification-item";var title=document.createElement("p");title.className="apg-notification-title";title.textContent=String(n&&n.message||"");var meta=document.createElement("p");meta.className="apg-notification-meta";meta.textContent=String(n&&n.kind||"info")+" - "+String(n&&n.time||"");article.appendChild(title);article.appendChild(meta);list.appendChild(article);}});if(dot)dot.hidden=false;}}'
        'function apgRecordNotification(message,kind){{_apgNotifications.unshift({{message:message,kind:kind||"info",time:new Date().toLocaleTimeString()}});apgRenderNotifications();}}'
        'function apgToggleNotifications(){{var p=document.getElementById("apg-notification-panel");if(!p)return;p.hidden=!p.hidden;if(!p.hidden)apgRenderNotifications();}}'
        'function apgToast(m,t){{'
        'var c=t==="error"?"bg-red-600":"bg-gray-900";'
        'var el=document.createElement("div");'
        'el.className=c+" text-white text-sm font-medium px-4 py-2.5 rounded-xl shadow-lg pointer-events-auto transition-all duration-300 opacity-0 translate-y-2";'
        'el.textContent=m;'
        'apgRecordNotification(m,t||"success");'
        'document.getElementById("apg-toast-root").appendChild(el);'
        'requestAnimationFrame(function(){{el.classList.remove("opacity-0","translate-y-2");}});'
        'setTimeout(function(){{el.classList.add("opacity-0");setTimeout(function(){{el.remove();}},300);}},3000);'
        '}}'
        'document.addEventListener("htmx:afterOnLoad",function(e){{'
        'var t=e.detail.xhr.getResponseHeader("HX-Trigger");'
        'if(!t)return;'
        'try{{var d=JSON.parse(t);if(d.apgToast)apgToast(d.apgToast.msg,d.apgToast.type||"success");}}catch(ex){{}}'
        '}});'
        'function apgApplyTheme(m){{var d=document.documentElement;if(m==="dark"||m==="light")d.setAttribute("data-theme",m);else d.removeAttribute("data-theme");d.dataset.themeMode=m;var b=document.getElementById("apg-theme-toggle");if(b){{b.setAttribute("aria-label","Theme: "+m);b.textContent=m==="dark"?"Dark":m==="light"?"Light":"System";}}}}'
        'function apgCycleTheme(){{var order=["system","light","dark"];var cur=localStorage.getItem("apg-theme")||"system";var next=order[(order.indexOf(cur)+1)%order.length];localStorage.setItem("apg-theme",next);apgApplyTheme(next);}}'
        'function apgApplyDensity(m){{var order=["compact","comfortable","spacious"];if(order.indexOf(m)<0)m="comfortable";var b=document.body;if(b){{order.forEach(function(x){{b.classList.remove("apg-density-"+x);}});b.classList.add("apg-density-"+m);}}var t=document.getElementById("apg-density-toggle");if(t){{var label=m.charAt(0).toUpperCase()+m.slice(1);t.textContent=label;t.setAttribute("aria-label","Density: "+m);}}}}'
        'function apgCycleDensity(){{var order=["compact","comfortable","spacious"];var cur=localStorage.getItem("apg-density")||"comfortable";var next=order[(order.indexOf(cur)+1)%order.length];localStorage.setItem("apg-density",next);apgApplyDensity(next);}}'
        'document.addEventListener("DOMContentLoaded",function(){{apgApplyTheme(localStorage.getItem("apg-theme")||"system");}});'
        'document.addEventListener("DOMContentLoaded",function(){{apgApplyDensity(localStorage.getItem("apg-density")||"comfortable");}});'
        'document.addEventListener("click",function(e){{if(e.target.closest("[data-apg-density-toggle]"))apgCycleDensity();}});'
        'function apgConfirm(message,ok){{var d=document.getElementById("apg-confirm-dialog");if(!d||!d.showModal){{var nativeConfirm=window["confirm"];if(nativeConfirm&&nativeConfirm(message))ok();return;}}document.getElementById("apg-confirm-message").textContent=message;var done=false;function close(){{if(done)return;done=true;d.removeEventListener("close",onclose);}}function onclose(){{var v=d.returnValue;close();if(v==="confirm")ok();}}d.addEventListener("close",onclose);d.showModal();}}'
        'function apgConfirmSubmit(form,message){{apgConfirm(message||' + delete_prompt + ',function(){{form.dataset.apgConfirmed="1";form.requestSubmit();}});return false;}}'
        'document.addEventListener("DOMContentLoaded",function(){{document.querySelectorAll(".apg-topnav a").forEach(function(a){{if(a.getAttribute("href")===location.pathname){{a.classList.add("active");a.setAttribute("aria-current","page");}}}});}});'
        'function apgSetSidebar(collapsed){{document.documentElement.classList.toggle("apg-sidebar-collapsed",collapsed);try{{localStorage.setItem("apg-sidebar-collapsed",collapsed?"1":"0");}}catch(e){{}}}}'
        'function apgToggleSidebar(){{if(matchMedia("(max-width: 767px)").matches){{document.documentElement.classList.toggle("apg-sidebar-open");}}else{{apgSetSidebar(!document.documentElement.classList.contains("apg-sidebar-collapsed"));}}}}'
        'function apgCloseSidebar(){{document.documentElement.classList.remove("apg-sidebar-open");}}'
        'try{{if(localStorage.getItem("apg-sidebar-collapsed")==="1")document.documentElement.classList.add("apg-sidebar-collapsed");}}catch(e){{}}'
        'function apgSyncOffline(){{var b=document.getElementById("apg-offline-banner");var offline=!navigator.onLine;if(b)b.hidden=!offline;if(offline&&!_apgWasOffline){{apgRecordNotification("Offline mode enabled","offline");}}if(!offline&&_apgWasOffline){{apgRecordNotification("Connection restored","online");}}_apgWasOffline=offline;}}'
        'window.addEventListener("online",apgSyncOffline);window.addEventListener("offline",apgSyncOffline);'
        'function apgApplyUpdate(){{if(window._apgWaitingWorker){{window._apgWaitingWorker.postMessage({{type:"SKIP_WAITING"}});}}}}'
        'function apgInstall(){{if(!_apgDeferredInstall)return;_apgDeferredInstall.prompt();_apgDeferredInstall.userChoice.finally(function(){{_apgDeferredInstall=null;var b=document.getElementById("apg-install-btn");if(b)b.hidden=true;}});}}'
        'window.addEventListener("beforeinstallprompt",function(e){{e.preventDefault();_apgDeferredInstall=e;var b=document.getElementById("apg-install-btn");if(b)b.hidden=false;apgRecordNotification("App can be installed","pwa");}});'
        'if("serviceWorker" in navigator){{window.addEventListener("load",function(){{navigator.serviceWorker.register("/static/sw.js").then(function(reg){{function watch(worker){{if(!worker)return;worker.addEventListener("statechange",function(){{if(worker.state==="installed"&&navigator.serviceWorker.controller){{window._apgWaitingWorker=worker;var b=document.getElementById("apg-update-btn");if(b)b.hidden=false;apgRecordNotification("Update ready","pwa");}}}});}}watch(reg.waiting);reg.addEventListener("updatefound",function(){{watch(reg.installing);}});}}).catch(function(){{}});}});navigator.serviceWorker.addEventListener("controllerchange",function(){{location.reload();}});}}'
        'document.addEventListener("keydown",function(e){{if(e.key==="Escape")apgCloseSidebar();}});'
        'document.addEventListener("DOMContentLoaded",function(){{apgSyncOffline();apgRenderNotifications();}});'
        '</script>'
    )
    skeleton_css = (
        f'<style{{script_nonce_attr}}>'
        '.apg-skeleton{{'
        '  background:linear-gradient(90deg,#f0f0f0 25%,#e0e0e0 50%,#f0f0f0 75%);'
        '  background-size:200% 100%;'
        '  animation:apg-shimmer 1.5s infinite;'
        '  border-radius:4px;'
        '}}'
        '@keyframes apg-shimmer{{'
        '  0%{{background-position:200% 0}}'
        '  100%{{background-position:-200% 0}}'
        '}}'
        '.apg-loading .apg-skeleton-row{{height:40px;margin-bottom:8px;}}'
        '.htmx-request .apg-content-area{{opacity:0.6;transition:opacity 0.2s;}}'
        '</style>'
    )
    cmd_palette_html = {cmd_palette_literal!r}
    cmd_palette_html = cmd_palette_html.replace('<script>', f'<script{{script_nonce_attr}}>')
    cmd_palette_html = cmd_palette_html.replace(
        'id="apg-cmd" class="hidden',
        'id="apg-cmd" role="dialog" aria-modal="true" aria-label="Command palette" class="hidden',
    )
    shell_supplement_js = (
        '<section id="apg-tour" class="apg-tour" aria-label="Shell onboarding" hidden>'
        '<div class="apg-tour-panel">'
        '<span>Shell tour</span><h2>Command-first generated workspace</h2>'
        '<p id="apg-tour-copy">Use the command center to jump across generated pages, APIs, entities, and operations.</p>'
        '<div class="apg-tour-actions"><button type="button" class="apg-btn apg-btn-secondary" onclick="apgTourClose()">Skip</button><button type="button" class="apg-btn" onclick="apgTourNext()">Next</button></div>'
        '</div></section>'
        f'<script{{script_nonce_attr}}>'
        'window.APGToast=window.APGToast||apgToast;'
        'var APG_SHELL_COMMANDS=' + shell_command_json + ';'
        'var _apgTourStep=0;'
        'var _apgTourCopy=["Use Command Center for generated navigation, APIs, entities, agents, and operations.","Recent items follow you across APG workspaces without server state.","Notifications, undo toasts, theme, install, update, and offline mode live in the shell."];'
        'function apgShellRecentKey(){{return "apg:shell:recent:"+location.host;}}'
        'function apgShellSafeUrl(url){{var raw=String(url||"/ui");return raw.charAt(0)==="/"&&raw.substring(0,2)!=="//"?raw:"/ui";}}'
        'function apgShellCleanItem(item){{item=item||{{}};return {{label:String(item.label||""),url:apgShellSafeUrl(item.url),kind:String(item.kind||"Recent"),hint:String(item.hint||"")}};}}'
        'function apgShellReadRecent(){{try{{var parsed=JSON.parse(localStorage.getItem(apgShellRecentKey())||"[]");return Array.isArray(parsed)?parsed.map(apgShellCleanItem):[];}}catch(e){{return [];}}}}'
        'function apgShellWriteRecent(items){{try{{var clean=(Array.isArray(items)?items:[]).map(apgShellCleanItem).slice(0,8);localStorage.setItem(apgShellRecentKey(),JSON.stringify(clean));}}catch(e){{}}}}'
        'function apgShellTrackRecent(){{var title=(document.title||location.pathname).replace(" — "," - ");var item=apgShellCleanItem({{label:title,url:location.pathname+location.search,kind:"Recent",hint:"Last opened workspace"}});var items=apgShellReadRecent().filter(function(x){{return x.url!==item.url;}});items.unshift(item);apgShellWriteRecent(items);}}'
        'function apgShellCommandNode(item){{var r=apgShellCleanItem(item);var a=document.createElement("a");a.href=r.url;a.onclick=function(){{apgCmdClose();}};a.className="apg-command-result";var kind=document.createElement("span");kind.className="apg-command-kind";kind.textContent=r.kind;var body=document.createElement("div");var label=document.createElement("strong");label.textContent=r.label;var hint=document.createElement("small");hint.textContent=r.hint||r.url;body.appendChild(label);body.appendChild(hint);a.appendChild(kind);a.appendChild(body);return a;}}'
        'function apgShellAppendCommands(root,items,empty){{if(!items.length){{var p=document.createElement("p");p.className="apg-command-empty";p.textContent=empty||"No commands";root.appendChild(p);return;}}items.forEach(function(item){{root.appendChild(apgShellCommandNode(item));}});}}'
        'function apgShellSection(title,items,empty,clearable){{var section=document.createElement("div");section.className="apg-command-section";var header=document.createElement("header");var span=document.createElement("span");span.textContent=title;header.appendChild(span);if(clearable){{var button=document.createElement("button");button.type="button";button.textContent="Clear recent";button.onclick=apgShellClearRecent;header.appendChild(button);}}section.appendChild(header);apgShellAppendCommands(section,items,empty);return section;}}'
        'function apgShellRenderDefault(){{var recent=apgShellReadRecent();var el=document.getElementById("apg-cmd-results");if(!el)return;apgClearChildren(el);el.appendChild(apgShellSection("Command Center",APG_SHELL_COMMANDS.slice(0,7),"No commands",true));el.appendChild(apgShellSection("Recent Items",recent,"No recent items yet",false));}}'
        'function apgShellRenderRecentOnly(){{apgCmdOpen();var el=document.getElementById("apg-cmd-results");if(!el)return;apgClearChildren(el);el.appendChild(apgShellSection("Recent Items",apgShellReadRecent(),"No recent items yet",true));}}'
        'function apgCmdOpen(){{var d=document.getElementById("apg-cmd");var i=document.getElementById("apg-cmd-input");if(!d||!i)return;d.classList.remove("hidden");i.focus();apgCmdSearch(i.value||"");}}'
        'function apgCmdClose(){{var d=document.getElementById("apg-cmd");var i=document.getElementById("apg-cmd-input");if(!d||!i)return;d.classList.add("hidden");i.value="";apgShellRenderDefault();}}'
        'function apgCmdSearch(q){{var query=(q||"").trim().toLowerCase();if(!query){{apgShellRenderDefault();return;}}var matches=APG_SHELL_COMMANDS.concat(apgShellReadRecent()).map(apgShellCleanItem).filter(function(r){{return (r.label+" "+r.kind+" "+(r.hint||"")).toLowerCase().indexOf(query)>=0;}}).slice(0,12);var el=document.getElementById("apg-cmd-results");if(!el)return;apgClearChildren(el);apgShellAppendCommands(el,matches,"No commands match");}}'
        'function apgUndoToast(message,label,undo){{var root=document.getElementById("apg-toast-root");if(!root){{apgToast(message,"info");return;}}var el=document.createElement("div");el.className="apg-undo-toast";var text=document.createElement("span");text.textContent=String(message||"");var button=document.createElement("button");button.type="button";button.textContent=String(label||"Undo");button.onclick=function(){{try{{undo&&undo();}}finally{{el.remove();apgToast("Undo applied","success");}}}};el.appendChild(text);el.appendChild(button);root.appendChild(el);setTimeout(function(){{if(el.parentNode)el.remove();}},7000);apgRecordNotification(message,"undo");}}'
        'function apgShellClearRecent(){{var previous=apgShellReadRecent();apgShellWriteRecent([]);apgShellRenderDefault();apgUndoToast("Recent items cleared","Undo",function(){{apgShellWriteRecent(previous);apgShellRenderDefault();}});}}'
        'function apgTourOpen(){{_apgTourStep=0;var t=document.getElementById("apg-tour");if(t)t.hidden=false;apgTourPaint();}}'
        'function apgTourPaint(){{var c=document.getElementById("apg-tour-copy");if(c)c.textContent=_apgTourCopy[_apgTourStep]||_apgTourCopy[0];}}'
        'function apgTourNext(){{_apgTourStep+=1;if(_apgTourStep>=_apgTourCopy.length){{apgTourClose();return;}}apgTourPaint();}}'
        'function apgTourClose(){{var t=document.getElementById("apg-tour");if(t)t.hidden=true;try{{localStorage.setItem("apg:shell-tour-seen","1");}}catch(e){{}}}}'
        'document.addEventListener("DOMContentLoaded",function(){{apgShellTrackRecent();apgShellRenderDefault();if(!localStorage.getItem("apg:shell-tour-seen"))setTimeout(apgTourOpen,600);}});'
        '</script>'
    )
    if not shell:
        return (
            "<!doctype html>"
            f'<html lang="{{safe_locale}}" dir="{{safe_direction}}" class="h-full"><head>'
            '<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"{{head_extras}}"
            f"{{skeleton_css}}"
            '<link rel="stylesheet" href="/theme.css">'
            f"<title>{{safe_title}} — {{safe_module}}</title>"
            "</head>"
            '<body class="min-h-full bg-gray-50 text-gray-900">'
            '<a class="apg-skip-link" href="#content">Skip to content</a>'
            f"{{body}}"
            f"{{toast_js}}"
            "</body></html>"
        )
    return (
        "<!doctype html>"
        f'<html lang="{{safe_locale}}" dir="{{safe_direction}}" class="h-full"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{{head_extras}}"
        f"{{skeleton_css}}"
        '<link rel="stylesheet" href="/theme.css">'
        f"<title>{{safe_title}} — {{safe_module}}</title>"
        "</head>"
        '<body class="min-h-full bg-gray-50 text-gray-900">'
        '<a class="apg-skip-link" href="#content">Skip to content</a>'
        '<div id="apg-offline-banner" class="apg-offline-banner" role="status" hidden>Offline mode: showing cached APG screens.</div>'
        f'<header class="apg-topbar sticky top-0 z-50" role="banner">'
        f'  <button class="apg-icon-btn" type="button" onclick="apgToggleSidebar()" aria-label="Toggle navigation">☰</button>'
        f'  <a class="apg-logo" href="/ui">{{safe_module}}</a>'
        f'  <nav class="apg-topnav ml-4">'
        f'    {{_shell_link("/ui", _("home"), "apg-nav-link hover:bg-gray-100", exact=True)}}'
        f'    {{_shell_link("/ui/workflows", "⚡ " + _("workflows"), "apg-nav-link hover:bg-gray-100")}}'
        f'    {{_shell_link("/ui/marketplace", _("marketplace"), "apg-nav-link hover:bg-gray-100")}}'
        f'  </nav>'
        f'  <span class="apg-topbar-spacer"></span>'
        f'  <div class="apg-shell-action-row" aria-label="Shell actions">'
        f'    <button class="apg-btn apg-btn-secondary apg-command-trigger" type="button" onclick="apgCmdOpen()" aria-haspopup="dialog">Command <kbd>⌘K</kbd></button>'
        f'    <button class="apg-btn apg-btn-secondary apg-recent-trigger" type="button" onclick="apgShellRenderRecentOnly()">Recent</button>'
        f'    <button class="apg-btn apg-btn-secondary apg-tour-trigger" type="button" onclick="apgTourOpen()">Tour</button>'
        f'    <button id="apg-install-btn" class="apg-btn apg-btn-secondary apg-install-btn" type="button" onclick="apgInstall()" hidden>Install</button>'
        f'    <button id="apg-update-btn" class="apg-btn apg-btn-secondary apg-install-btn" type="button" onclick="apgApplyUpdate()" hidden>Update</button>'
        f'    <button id="apg-density-toggle" data-apg-density-toggle class="apg-btn apg-btn-secondary apg-density-toggle" type="button" aria-label="Density: comfortable">Comfortable</button>'
        f'    <div class="apg-notification-wrap">'
        f'      <button class="apg-btn apg-btn-secondary" type="button" onclick="apgToggleNotifications()" aria-controls="apg-notification-panel" aria-label="Notifications">Notifications<span id="apg-notification-dot" class="apg-notification-dot" hidden></span></button>'
        f'      <section id="apg-notification-panel" class="apg-notification-panel" aria-label="Notifications" hidden><h2 class="text-sm font-semibold text-gray-900 mb-3">Notifications</h2><div id="apg-notification-list"></div></section>'
        f'    </div>'
        f'    <button id="apg-theme-toggle" class="apg-btn apg-btn-secondary apg-theme-toggle" type="button" onclick="apgCycleTheme()" aria-label="Theme: system">{{_("theme_system")}}</button>'
        f'    {{language_menu}}'
        f'    {{user_menu}}'
        f'  </div>'
        f'</header>'
        f'{{sidebar_html}}'
        f'<main class="apg-content apg-shell-content" id="content" tabindex="-1">{{body}}</main>'
        f"{{toast_js}}"
        f"{{cmd_palette_html}}"
        f"{{shell_supplement_js}}"
        "</body></html>"
    )


def _jinja_required_page(title: str = "Application UI") -> str:
    safe_title = html.escape(title)
    return (
        f'<section class="apg-card">'
        f'<h1>{{safe_title}}</h1>'
        f'<p>This application requires Jinja2 — pip install -r requirements.txt.</p>'
        f'</section>'
    )


def _render_template(template_name: str, **context: Any) -> str | None:
    """Render a Jinja2 template from APG_UI_TEMPLATES dict if Jinja2 is available.

    Returns None when Jinja2 is not installed — callers fall back to the existing
    f-string builder so the generated app works with zero extra dependencies.

    APG_UI_TEMPLATES is injected at module level when the compiler embeds templates
    as string literals. In standalone mode (running code_generator.py directly),
    templates are loaded from compiler/templates/*.j2 relative to this file.
    """
    try:
        from jinja2 import Environment, DictLoader, BaseLoader, FileSystemLoader, ChoiceLoader  # type: ignore[import]
    except ImportError:
        return None
    try:
        # APG_UI_TEMPLATES injected at compile time takes priority
        templates: dict[str, str] = globals().get("APG_UI_TEMPLATES", {{}})
        if templates:
            env = Environment(loader=DictLoader(templates), autoescape=True)
        else:
            # Standalone: load from compiler/templates/ directory
            import pathlib
            tmpl_dir = pathlib.Path(__file__).parent / "templates"
            if not tmpl_dir.exists():
                return None
            env = Environment(loader=FileSystemLoader(str(tmpl_dir)), autoescape=True)
            # Adjust template name for standalone (files have .j2 extension, no nested path)
            if not template_name.endswith(".j2"):
                template_name = template_name.replace(".html", ".html.j2") if ".html" in template_name else template_name + ".j2"
        # Add url encode filter
        env.filters["urlencode"] = lambda s: __import__("urllib.parse", fromlist=["quote"]).quote(str(s), safe="")
        env.globals.update({{
            "_": _apg_t,
            "_apg_t": _apg_t,
            "format_number": format_number,
            "format_currency": format_currency,
            "format_date": format_date,
            "csrf_token": _csrf_token,
            "csrf_input": _csrf_input,
            "csp_nonce": _csp_nonce(),
        }})
        context.setdefault("csp_nonce", _csp_nonce())
        tmpl = env.get_template(template_name)
        return tmpl.render(**context)
    except Exception:
        return None


def _entity_spec(entity_name: str) -> Dict[str, Any] | None:
    for entity in ENTITIES:
        if entity["name"] == entity_name:
            return dict(entity)
    return None


def _field_specs(entity_name: str) -> list[Dict[str, Any]]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return []
    fields = entity.get("fields") or []
    if fields:
        return [dict(field) for field in fields if isinstance(field, dict)]
    return [
        {{"name": property_name, "type": "any", "required": True}}
        for property_name in entity.get("properties", [])
    ]


def _computed_field_specs(entity_name: str) -> list[Dict[str, Any]]:
    return [field for field in _field_specs(entity_name) if field.get("computed")]


def _stored_field_specs(entity_name: str) -> list[Dict[str, Any]]:
    return [field for field in _field_specs(entity_name) if not field.get("computed")]


def _computed_field_names(entity_name: str) -> set[str]:
    return {{
        str(field.get("name", ""))
        for field in _computed_field_specs(entity_name)
        if str(field.get("name", "")).strip()
    }}


def _apply_computed_fields(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    public = dict(record)
    local_values = dict(public)
    for field in _computed_field_specs(entity_name):
        field_name = str(field.get("name", "")).strip()
        expression = str(field.get("expression", "") or "").strip()
        if not field_name or not expression:
            continue
        try:
            public[field_name] = eval(expression, {{"__builtins__": {{}}}}, dict(local_values))
        except Exception as exc:
            _logging.getLogger("apg").warning(
                "computed_field_failed entity=%s field=%s err=%s",
                entity_name,
                field_name,
                exc,
            )
            public[field_name] = None
        local_values[field_name] = public[field_name]
    return public


def _is_file_field(field: Dict[str, Any]) -> bool:
    return str(field.get("type", "")).rstrip("?").lower() == "file"


def _file_field_specs(entity_name: str) -> list[Dict[str, Any]]:
    return [field for field in _field_specs(entity_name) if _is_file_field(field)]


def _file_metadata_field_names(field_name: str) -> list[str]:
    return [field_name + "_path", field_name + "_mime", field_name + "_size"]


def _upload_allowed_mime_types() -> set[str]:
    raw = os.environ.get(
        "APG_UPLOAD_ALLOWED_TYPES",
        "image/jpeg,image/png,image/webp,application/pdf",
    )
    allowed = {{
        item.strip().lower()
        for item in str(raw).split(",")
        if item.strip()
    }}
    return allowed or {{"image/jpeg", "image/png", "image/webp", "application/pdf"}}


def _upload_max_bytes() -> int:
    return max(1, _env_int("APG_UPLOAD_MAX_BYTES", 10 * 1024 * 1024))


def _upload_root() -> Path:
    return Path(os.environ.get("APG_UPLOAD_DIR", "./uploads")).expanduser()


def _safe_upload_segment(value: Any, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return safe or fallback


def _multipart_upload_request() -> bool:
    try:
        return "multipart/form-data" in str(_flask_request.content_type or "")
    except RuntimeError:
        return False


def _uploaded_file_mime(upload: Any) -> str:
    mime_type = str(getattr(upload, "mimetype", "") or getattr(upload, "content_type", "") or "")
    mime_type = mime_type.split(";", 1)[0].strip().lower()
    if not mime_type:
        guessed, _encoding = _mimetypes.guess_type(str(getattr(upload, "filename", "") or ""))
        mime_type = str(guessed or "").lower()
    return mime_type


def _uploaded_file_extension(upload: Any, mime_type: str) -> str:
    original = Path(str(getattr(upload, "filename", "") or "")).suffix.lower().lstrip(".")
    if original and re.fullmatch(r"[A-Za-z0-9]{{1,12}}", original):
        return original
    guessed = str(_mimetypes.guess_extension(mime_type) or "").lower().lstrip(".")
    return guessed or "bin"


def _store_uploaded_file(entity_name: str, field_name: str, upload: Any) -> tuple[int, Dict[str, Any]]:
    mime_type = _uploaded_file_mime(upload)
    if mime_type not in _upload_allowed_mime_types():
        return 415, {{
            "error": "unsupported_media_type",
            "field": field_name,
            "mime": mime_type,
        }}
    data = upload.read()
    try:
        upload.stream.seek(0)
    except Exception:
        pass
    size = len(data)
    if size > _upload_max_bytes():
        return 413, {{
            "error": "payload_too_large",
            "field": field_name,
            "max_bytes": _upload_max_bytes(),
        }}
    safe_entity = _safe_upload_segment(entity_name, "entity")
    upload_dir = _upload_root() / safe_entity
    upload_dir.mkdir(parents=True, exist_ok=True)
    extension = _uploaded_file_extension(upload, mime_type)
    filename = str(_uuid.uuid4()) + "." + extension
    destination = upload_dir / filename
    destination.write_bytes(data)
    return 200, {{
        "path": str(destination),
        "mime": mime_type,
        "size": size,
        "filename": filename,
    }}


def _apply_uploaded_files(entity_name: str, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]] | None:
    if not _multipart_upload_request():
        return None
    for field in _file_field_specs(entity_name):
        field_name = str(field.get("name", "")).strip()
        if not field_name or field_name not in _flask_request.files:
            continue
        upload = _flask_request.files.get(field_name)
        if upload is None or not str(getattr(upload, "filename", "") or ""):
            continue
        status, result = _store_uploaded_file(entity_name, field_name, upload)
        if status != 200:
            return status, result
        record.pop(field_name, None)
        record[field_name + "_path"] = result["path"]
        record[field_name + "_mime"] = result["mime"]
        record[field_name + "_size"] = result["size"]
    return None


def _file_url_for_path(entity_name: str, path: Any) -> str:
    filename = Path(str(path or "")).name
    if not filename:
        return ""
    return f"/uploads/{{quote(entity_name, safe='')}}/{{quote(filename, safe='')}}"


def _mime_for_uploaded_filename(entity_name: str, filename: str) -> str:
    for record in RECORD_STORE.get(entity_name, []):
        for field in _file_field_specs(entity_name):
            field_name = str(field.get("name", ""))
            stored_name = Path(str(record.get(field_name + "_path", "") or "")).name
            if stored_name == filename:
                stored_mime = str(record.get(field_name + "_mime", "") or "")
                if stored_mime:
                    return stored_mime
    guessed, _encoding = _mimetypes.guess_type(filename)
    return str(guessed or "application/octet-stream")


def _uploaded_file_response(entity_name: str, filename: str) -> _FlaskResponse:
    if entity_name not in ENTITY_NAMES:
        return _apg_error_response(404, "unknown_entity", "Unknown entity")
    safe_filename = _safe_upload_segment(filename, "")
    if not safe_filename or safe_filename != filename:
        return _apg_error_response(404, "not_found", "File not found")
    root = (_upload_root() / _safe_upload_segment(entity_name, "entity")).resolve()
    path = (root / safe_filename).resolve()
    if path.parent != root or not path.is_file():
        return _apg_error_response(404, "not_found", "File not found")
    data = path.read_bytes()
    etag = '"' + hashlib.sha256(data).hexdigest()[:16] + '"'
    requested = [
        item.strip()
        for item in str(_flask_request.headers.get("If-None-Match", "")).split(",")
        if item.strip()
    ]
    if etag in requested:
        response = _FlaskResponse(b"", status=304)
    else:
        response = _FlaskResponse(data, status=200, content_type=_mime_for_uploaded_filename(entity_name, safe_filename))
        response.headers["Content-Length"] = str(len(data))
    response.headers["ETag"] = etag
    return response


def _relationship_specs(entity_name: str) -> list[Dict[str, Any]]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return []
    relationships = entity.get("relationships") or []
    return [dict(relationship) for relationship in relationships if isinstance(relationship, dict)]


def _relationship_field_name(entity_name: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", str(entity_name))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", text)
    return text.replace("-", "_").lower() + "_id"


def _relationship_segment(entity_name: str) -> str:
    base = _relationship_field_name(entity_name)
    base = base[:-3] if base.endswith("_id") else base
    if base.endswith("y") and (len(base) == 1 or base[-2] not in "aeiou"):
        return base[:-1] + "ies"
    if base.endswith(("s", "x", "z", "ch", "sh")):
        return base + "es"
    return base + "s"


def _relationship_by_segment(entity_name: str, segment: str) -> Dict[str, Any] | None:
    for relationship in _relationship_specs(entity_name):
        if str(relationship.get("kind")) not in {{"has_many", "has_one"}}:
            continue
        candidate = str(relationship.get("segment") or _relationship_segment(str(relationship.get("target", ""))))
        if candidate == segment:
            return relationship
    return None


def _json_schema_type(apg_type: str) -> str:
    normalized = apg_type.lower()
    if normalized in {{"str", "string", "text", "varchar", "char", "email", "uuid", "date", "datetime", "timestamp"}}:
        return "string"
    if normalized in {{"int", "integer", "serial", "bigint", "smallint"}}:
        return "integer"
    if normalized in {{"float", "double", "decimal", "number", "numeric", "money"}}:
        return "number"
    if normalized in {{"bool", "boolean"}}:
        return "boolean"
    if normalized in {{"list", "array", "set"}}:
        return "array"
    if normalized in {{"dict", "map", "object", "json", "jsonb"}}:
        return "object"
    return "string"


def _value_matches_type(value: Any, apg_type: str) -> bool:
    expected = _json_schema_type(apg_type)
    if value is None:
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _coerce_value_for_type(value: Any, apg_type: str) -> Any:
    if not isinstance(value, str):
        return value
    expected = _json_schema_type(apg_type)
    if expected == "integer":
        try:
            return int(value.strip())
        except ValueError:
            return value
    if expected == "number":
        try:
            return float(value.strip())
        except ValueError:
            return value
    if expected == "boolean":
        normalized = value.strip().lower()
        if normalized in {{"true", "1", "yes", "on"}}:
            return True
        if normalized in {{"false", "0", "no", "off"}}:
            return False
    if expected in {{"array", "object"}}:
        text = value.strip()
        if not text:
            return [] if expected == "array" else {{}}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return value
        if expected == "array" and isinstance(parsed, list):
            return parsed
        if expected == "object" and isinstance(parsed, dict):
            return parsed
    return value


def coerce_record_types(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(record)
    for field_name in _computed_field_names(entity_name):
        coerced.pop(field_name, None)
    for field in _field_specs(entity_name):
        if field.get("computed"):
            continue
        field_name = str(field["name"])
        if _is_file_field(field):
            coerced.pop(field_name, None)
            continue
        if field_name in coerced:
            coerced[field_name] = _coerce_value_for_type(
                coerced[field_name],
                str(field.get("type", "any")),
            )
    return coerced


def _validation_failed(field_name: str, rule: str, detail: str) -> Dict[str, Any]:
    return {{
        "valid": False,
        "status": 400,
        "error": "validation_failed",
        "field": field_name,
        "rule": rule,
        "detail": detail,
        "errors": [field_name + " " + detail],
    }}


def _validator_value(validator: Dict[str, Any], key: str = "value") -> Any:
    if key in validator:
        return validator.get(key)
    return validator.get("value")


def _validator_failure(field_name: str, value: Any, validator: Dict[str, Any]) -> Dict[str, Any] | None:
    rule = str(validator.get("rule", ""))
    if rule in {{"", "optional"}}:
        return None
    if rule == "required":
        if value is None or value == "":
            return _validation_failed(field_name, "required", "required")
        return None
    if value is None:
        return None
    if rule == "min_length":
        limit = int(_validator_value(validator) or 0)
        if len(str(value)) < limit:
            return _validation_failed(field_name, "min_length", "min " + str(limit) + " chars")
    elif rule == "max_length":
        limit = int(_validator_value(validator) or 0)
        if len(str(value)) > limit:
            return _validation_failed(field_name, "max_length", "max " + str(limit) + " chars")
    elif rule == "min":
        raw_limit = _validator_value(validator)
        try:
            if float(value) < float(raw_limit):
                return _validation_failed(field_name, "min", "min " + str(raw_limit))
        except (TypeError, ValueError):
            return _validation_failed(field_name, "min", "must be numeric")
    elif rule == "max":
        raw_limit = _validator_value(validator)
        try:
            if float(value) > float(raw_limit):
                return _validation_failed(field_name, "max", "max " + str(raw_limit))
        except (TypeError, ValueError):
            return _validation_failed(field_name, "max", "must be numeric")
    elif rule == "email":
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(value)) is None:
            return _validation_failed(field_name, "email", "must be a valid email")
    elif rule == "pattern":
        pattern = str(_validator_value(validator, "pattern") or "")
        try:
            matched = re.fullmatch(pattern, str(value)) is not None
        except re.error:
            matched = False
        if not matched:
            return _validation_failed(field_name, "pattern", "must match pattern")
    return None


def _record_validation_failure(validation: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if validation.get("error") == "invalid_enum_value":
        return 400, {{
            "error": "invalid_enum_value",
            "field": validation.get("field"),
            "allowed": validation.get("allowed", []),
        }}
    if validation.get("error") == "validation_failed":
        return 400, {{
            "error": "validation_failed",
            "field": validation.get("field"),
            "rule": validation.get("rule"),
            "detail": validation.get("detail"),
        }}
    return 422, {{"error": "record_validation_failed", **validation}}


def validate_record(entity_name: str, record: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    errors: list[str] = []
    fields = _field_specs(entity_name)
    for field in fields:
        if field.get("computed"):
            continue
        field_name = str(field["name"])
        if _is_file_field(field):
            path_field = field_name + "_path"
            if not partial and field.get("required", False) and path_field not in record:
                errors.append(f"{{field_name}} is required")
            continue
        if not partial and field.get("required", False) and field_name not in record:
            errors.append(f"{{field_name}} is required")
            continue
        if field_name not in record:
            continue
        value = record[field_name]
        if value is None:
            if field.get("required", False):
                errors.append(f"{{field_name}} is required")
            continue
        if not _value_matches_type(value, str(field.get("type", "any"))):
            errors.append(f"{{field_name}} must be {{_json_schema_type(str(field.get('type', 'any')))}}")
            continue
        enum_values = field.get("enum") if isinstance(field.get("enum"), list) else []
        if enum_values and value not in enum_values:
            return {{
                "valid": False,
                "status": 400,
                "error": "invalid_enum_value",
                "entity": entity_name,
                "field": field_name,
                "allowed": list(enum_values),
                "errors": [f"{{field_name}} must be one of {{', '.join(str(item) for item in enum_values)}}"],
            }}
        for validator in field.get("validators", []):
            if not isinstance(validator, dict):
                continue
            failure = _validator_failure(field_name, value, validator)
            if failure is not None:
                failure["entity"] = entity_name
                return failure
    return {{
        "valid": not errors,
        "entity": entity_name,
        "errors": errors,
    }}


_APG_SQLITE_CONN: _sqlite3.Connection | None = None
_APG_DB_DIALECT: str = "sqlite"
_APG_DB_POOL_SIZE: int = int(os.environ.get("APG_DB_POOL_SIZE", "5") or "5")
_APG_DB_POOL_SEMAPHORE = _threading.BoundedSemaphore(max(1, _APG_DB_POOL_SIZE))
_APG_DB_POOL_LOCAL = _threading.local()


def _apg_db_connect() -> "tuple[Any, str]":
    """Open a connection based on APG_DATABASE_URL and return (connection, dialect).

    Supports 'postgresql://', 'postgres://' and 'sqlite:///...' URLs. Defaults to
    a local SQLite file at ./apg_data.db when no URL is provided.
    """
    global _APG_DB_DIALECT
    url = os.environ.get("APG_DATABASE_URL", "") or os.environ.get("DATABASE_URL", "") or ""
    if url.startswith(("postgresql://", "postgres://")):
        try:
            import psycopg2  # type: ignore
        except ImportError:
            raise RuntimeError("psycopg2 not installed: pip install psycopg2-binary")
        _APG_DB_DIALECT = "pg"
        _APG_DB_POOL_SEMAPHORE.acquire()
        try:
            conn = psycopg2.connect(url)
        except Exception:
            _APG_DB_POOL_SEMAPHORE.release()
            raise
        return conn, "pg"
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1) or "./apg_data.db"
    else:
        path = "./apg_data.db"
    import sqlite3 as _sqlite3_mod
    _APG_DB_DIALECT = "sqlite"
    return _sqlite3_mod.connect(path), "sqlite"


def _apg_db_dialect() -> str:
    return _APG_DB_DIALECT


def _apg_ddl_pk() -> str:
    """Dialect-appropriate integer autoincrement primary key DDL."""
    return "BIGSERIAL PRIMARY KEY" if _APG_DB_DIALECT == "pg" else "INTEGER PRIMARY KEY AUTOINCREMENT"


def _apg_ddl_now() -> str:
    """Dialect-appropriate 'current timestamp' expression."""
    return "NOW()" if _APG_DB_DIALECT == "pg" else "datetime('now')"


def _apg_ddl_text_type() -> str:
    """Dialect-agnostic TEXT type (same for both, centralised for consistency)."""
    return "TEXT"


def _apg_qmark(sql: str) -> str:
    """Rewrite '?' placeholders to '%s' on PostgreSQL; passthrough on SQLite."""
    if _APG_DB_DIALECT != "pg":
        return sql
    out_chars: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "?" and not in_single and not in_double:
            out_chars.append("%s")
        else:
            out_chars.append(ch)
        i += 1
    return "".join(out_chars)


def _apg_insert_returning_id(cursor: Any, sql: str, params: Any) -> Any:
    """Execute an INSERT and return the new integer id.

    - SQLite: relies on cursor.lastrowid (INTEGER PRIMARY KEY AUTOINCREMENT).
    - PostgreSQL: appends RETURNING id and fetches the first column.
    """
    if _APG_DB_DIALECT == "pg":
        cursor.execute(_apg_qmark(sql) + " RETURNING id", params)
        row = cursor.fetchone()
        return row[0] if row else None
    cursor.execute(sql, params)
    return cursor.lastrowid


def _apg_touch_updated_at_ddl() -> list[str]:
    """Return trigger-function DDL required once per DB (PG only)."""
    if _APG_DB_DIALECT != "pg":
        return []
    return [
        "CREATE OR REPLACE FUNCTION apg_touch_updated_at() RETURNS trigger AS $$"
        " BEGIN NEW.updated_at = NOW(); RETURN NEW; END;"
        " $$ LANGUAGE plpgsql;"
    ]


def _sqlite_path_from_env() -> str:
    explicit_path = os.environ.get("APG_SQLITE_PATH") or os.environ.get("APG_DB_PATH")
    if explicit_path:
        return explicit_path
    database_url = os.environ.get("APG_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if database_url.startswith("sqlite:///"):
        path = unquote(database_url.removeprefix("sqlite:///"))
        return path or ":memory:"
    if database_url == "sqlite:///:memory:":
        return ":memory:"
    return ":memory:"


_APG_SQLITE_PATH = _sqlite_path_from_env()


def _sqlite_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _sqlite_storage_type(apg_type: str) -> str:
    schema_type = _json_schema_type(apg_type)
    if schema_type in {{"integer", "boolean"}}:
        return "INTEGER"
    if schema_type == "number":
        return "REAL"
    return "TEXT"


def _sqlite_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sqlite_expected_columns(entity_name: str) -> list[Dict[str, str]]:
    columns: list[Dict[str, str]] = [
        {{"name": "id", "ddl": "TEXT PRIMARY KEY", "migration": "TEXT"}},
        {{"name": "_revision", "ddl": "INTEGER DEFAULT 1 NOT NULL", "migration": "INTEGER"}},
        {{"name": "owner_id", "ddl": "TEXT DEFAULT NULL", "migration": "TEXT"}},
    ]
    if APG_MULTI_TENANT_ENABLED:
        columns.append({{"name": "tenant_id", "ddl": "TEXT NOT NULL DEFAULT 'default'", "migration": "TEXT NOT NULL DEFAULT 'default'"}})
    for field in _stored_field_specs(entity_name):
        field_name = str(field.get("name", "")).strip()
        existing_column_names = {{column["name"] for column in columns}}
        if not field_name or field_name in {{"id", "_revision", "created_at", "updated_at", "deleted_at"}} or field_name in existing_column_names:
            continue
        if _is_file_field(field):
            columns.extend([
                {{"name": field_name + "_path", "ddl": "TEXT NULL", "migration": "TEXT"}},
                {{"name": field_name + "_mime", "ddl": "TEXT NULL", "migration": "TEXT"}},
                {{"name": field_name + "_size", "ddl": "INTEGER NULL", "migration": "INTEGER"}},
            ])
            continue
        relationship = field.get("relationship") if isinstance(field.get("relationship"), dict) else {{}}
        relationship_kind = str(relationship.get("kind", ""))
        target_entity = str(relationship.get("target", ""))
        storage_type = "INTEGER" if relationship_kind in {{"belongs_to", "junction_left", "junction_right"}} else _sqlite_storage_type(str(field.get("type", "any")))
        if relationship_kind == "belongs_to" and target_entity:
            ddl = storage_type + " REFERENCES " + _sqlite_identifier(target_entity) + "(id) ON DELETE SET NULL"
        elif relationship_kind in {{"junction_left", "junction_right"}} and target_entity:
            ddl = storage_type + " REFERENCES " + _sqlite_identifier(target_entity) + "(id) ON DELETE CASCADE"
        else:
            ddl = storage_type
        enum_values = field.get("enum") if isinstance(field.get("enum"), list) else []
        if enum_values:
            ddl += " CHECK(" + _sqlite_identifier(field_name) + " IN (" + ", ".join(_sqlite_literal(value) for value in enum_values) + "))"
        ddl += " NOT NULL" if field.get("required", False) else ""
        columns.append({{"name": field_name, "ddl": ddl, "migration": storage_type}})
    now_expr = _apg_ddl_now()
    ts_default = "TEXT DEFAULT (" + now_expr + ") NOT NULL" if _APG_DB_DIALECT != "pg" else "TIMESTAMPTZ DEFAULT " + now_expr + " NOT NULL"
    ts_null = "TEXT NULL" if _APG_DB_DIALECT != "pg" else "TIMESTAMPTZ NULL"
    ts_migration = "TEXT" if _APG_DB_DIALECT != "pg" else "TIMESTAMPTZ"
    columns.extend([
        {{"name": "created_at", "ddl": ts_default, "migration": ts_migration}},
        {{"name": "updated_at", "ddl": ts_default, "migration": ts_migration}},
        {{"name": "deleted_at", "ddl": ts_null, "migration": ts_migration}},
    ])
    return columns


def _sqlite_fts_identifier(identifier: str) -> str:
    text = str(identifier)
    return text if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text) else _sqlite_identifier(text)


_APG_FTS_PG_WARNED: bool = False


def _sqlite_init_entity_fts(conn: _sqlite3.Connection, entity_name: str) -> None:
    # FTS5 is SQLite-only; skip on PostgreSQL.
    if _apg_db_dialect() == "pg":
        global _APG_FTS_PG_WARNED
        if not _APG_FTS_PG_WARNED:
            _logging.getLogger("apg").info(
                "FTS5 not available on PostgreSQL - /search endpoints will return empty results"
            )
            _APG_FTS_PG_WARNED = True
        return
    str_fields = _record_string_field_names(entity_name)
    if not str_fields:
        return
    entity_sql = _sqlite_fts_identifier(entity_name)
    fts_table = entity_name + "_fts"
    fts_sql = _sqlite_fts_identifier(fts_table)
    str_fields_csv = ",".join(_sqlite_fts_identifier(field) for field in str_fields)
    new_str_fields = ",".join("new." + _sqlite_fts_identifier(field) for field in str_fields)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS "
        + fts_sql
        + " USING fts5("
        + str_fields_csv
        + ", content="
        + entity_sql
        + ", content_rowid=id)"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS "
        + _sqlite_fts_identifier(fts_table + "_ai")
        + " AFTER INSERT ON "
        + entity_sql
        + " BEGIN INSERT INTO "
        + fts_sql
        + "(rowid,"
        + str_fields_csv
        + ") VALUES (new.id,"
        + new_str_fields
        + "); END;"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS "
        + _sqlite_fts_identifier(fts_table + "_ad")
        + " AFTER DELETE ON "
        + entity_sql
        + " BEGIN INSERT INTO "
        + fts_sql
        + "("
        + fts_sql
        + ",rowid) VALUES('delete',old.id); END;"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS "
        + _sqlite_fts_identifier(fts_table + "_au")
        + " AFTER UPDATE ON "
        + entity_sql
        + " BEGIN INSERT INTO "
        + fts_sql
        + "("
        + fts_sql
        + ",rowid) VALUES('delete',old.id); INSERT INTO "
        + fts_sql
        + "(rowid,"
        + str_fields_csv
        + ") VALUES(new.id,"
        + new_str_fields
        + "); END;"
    )


def _sqlite_connection() -> _sqlite3.Connection | None:
    global _APG_SQLITE_CONN
    if _APG_SQLITE_CONN is not None:
        return _APG_SQLITE_CONN
    try:
        if _APG_SQLITE_PATH != ":memory:":
            Path(_APG_SQLITE_PATH).expanduser().parent.mkdir(parents=True, exist_ok=True)
        _APG_SQLITE_CONN = _sqlite3.connect(_APG_SQLITE_PATH, check_same_thread=False)
        _APG_SQLITE_CONN.row_factory = _sqlite3.Row
        return _APG_SQLITE_CONN
    except Exception as exc:
        _logging.getLogger("apg").warning("sqlite_init_failed: %s", exc)
        _APG_SQLITE_CONN = None
        return None


def _sqlite_init_entity_table(conn: _sqlite3.Connection, entity_name: str) -> None:
    table = _sqlite_identifier(entity_name)
    columns = _sqlite_expected_columns(entity_name)
    column_sql = ", ".join(_sqlite_identifier(column["name"]) + " " + column["ddl"] for column in columns)
    conn.execute("CREATE TABLE IF NOT EXISTS " + table + " (" + column_sql + ")")
    _sqlite_init_entity_fts(conn, entity_name)
    trigger = _sqlite_identifier("trg_" + entity_name + "_updated_at")
    if _APG_DB_DIALECT == "pg":
        # PG: BEFORE UPDATE trigger + shared trigger function (emitted once elsewhere).
        conn.execute(
            "DROP TRIGGER IF EXISTS " + trigger + " ON " + table + ";"
            " CREATE TRIGGER " + trigger + " BEFORE UPDATE ON " + table
            + " FOR EACH ROW EXECUTE FUNCTION apg_touch_updated_at();"
        )
    else:
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS " + trigger + " AFTER UPDATE ON " + table + " FOR EACH ROW "
            "BEGIN UPDATE " + table + " SET updated_at = datetime('now') WHERE id = NEW.id; END;"
        )


def _sqlite_auto_migrate_entity(conn: _sqlite3.Connection, entity_name: str) -> None:
    if _APG_DB_DIALECT == "pg":
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (entity_name,),
        )
        existing = {{str(row[0]) for row in cur.fetchall()}}
        cur.close()
    else:
        existing = {{
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(" + _sqlite_identifier(entity_name) + ")").fetchall()
        }}
    for column in _sqlite_expected_columns(entity_name):
        column_name = column["name"]
        if column_name in existing:
            continue
        conn.execute(
            "ALTER TABLE "
            + _sqlite_identifier(entity_name)
            + " ADD COLUMN "
            + _sqlite_identifier(column_name)
            + " "
            + (
                column["migration"]
                if "DEFAULT" in column["migration"].upper() or "NOT NULL" in column["migration"].upper()
                else column["migration"] + " DEFAULT NULL"
            )
        )
        _logging.getLogger("apg").info("auto_migrated_column entity=%s column=%s", entity_name, column_name)


def _sqlite_init_database() -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    try:
        for stmt in _apg_touch_updated_at_ddl():
            conn.execute(stmt)
        for entity_name in sorted(ENTITY_NAMES):
            _sqlite_init_entity_table(conn, entity_name)
            if str(os.environ.get("APG_AUTO_MIGRATE", "1")) != "0":
                _sqlite_auto_migrate_entity(conn, entity_name)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        _logging.getLogger("apg").warning("sqlite_schema_init_failed: %s", exc)


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _sqlite_store_record(entity_name: str, record: Dict[str, Any]) -> None:
    conn = _sqlite_connection()
    if conn is None or record.get("id") in (None, ""):
        return
    columns = _sqlite_expected_columns(entity_name)
    metadata = _record_metadata(entity_name, record.get("id"), create=True) or {{}}
    row: Dict[str, Any] = {{
        "id": str(record.get("id")),
        "_revision": int(record.get("_revision", 1)),
        "owner_id": record.get("owner_id"),
        "created_at": record.get("created_at") or metadata.get("created_at"),
        "updated_at": record.get("updated_at") or metadata.get("updated_at"),
        "deleted_at": record.get("deleted_at") if "deleted_at" in record else metadata.get("deleted_at"),
    }}
    if any(column["name"] == "tenant_id" for column in columns):
        row["tenant_id"] = record.get("tenant_id") or APG_TENANT_DEFAULT
    for field in _stored_field_specs(entity_name):
        field_name = str(field.get("name", "")).strip()
        if not field_name:
            continue
        if _is_file_field(field):
            for metadata_name in _file_metadata_field_names(field_name):
                if metadata_name not in row:
                    row[metadata_name] = _sqlite_value(record.get(metadata_name))
            continue
        if field_name not in row:
            row[field_name] = _sqlite_value(record.get(field_name))
    column_names = [column["name"] for column in columns]
    assignments = [
        _sqlite_identifier(column_name) + " = excluded." + _sqlite_identifier(column_name)
        for column_name in column_names
        if column_name != "id"
    ]
    sql = (
        "INSERT INTO "
        + _sqlite_identifier(entity_name)
        + " ("
        + ", ".join(_sqlite_identifier(column_name) for column_name in column_names)
        + ") VALUES ("
        + ", ".join("?" for _column_name in column_names)
        + ") ON CONFLICT(id) DO UPDATE SET "
        + ", ".join(assignments)
    )
    conn.execute(_apg_qmark(sql), [_sqlite_value(row.get(column_name)) for column_name in column_names])


def _sqlite_soft_delete_record(entity_name: str, record_id: Any) -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    sql = "UPDATE " + _sqlite_identifier(entity_name) + " SET deleted_at=" + _apg_ddl_now() + " WHERE id=?"
    params: list[Any] = [str(record_id)]
    if _tenant_scope_enabled(entity_name) and not _tenant_admin_bypass():
        sql += " AND tenant_id=?"
        params.append(_tenant_id() or APG_TENANT_DEFAULT)
    conn.execute(_apg_qmark(sql), params)


def _sqlite_restore_record(entity_name: str, record_id: Any) -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    sql = "UPDATE " + _sqlite_identifier(entity_name) + " SET deleted_at=NULL WHERE id=?"
    params: list[Any] = [str(record_id)]
    if _tenant_scope_enabled(entity_name) and not _tenant_admin_bypass():
        sql += " AND tenant_id=?"
        params.append(_tenant_id() or APG_TENANT_DEFAULT)
    conn.execute(_apg_qmark(sql), params)


def _sqlite_select_records(
    entity_name: str,
    include_deleted: bool = False,
    tenant_scoped: bool = True,
) -> list[Dict[str, Any]]:
    conn = _sqlite_connection()
    if conn is None:
        return []
    sql = "SELECT * FROM " + _sqlite_identifier(entity_name)
    where_clauses: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        where_clauses.append("deleted_at IS NULL")
    if tenant_scoped and _tenant_scope_enabled(entity_name) and not _tenant_admin_bypass():
        where_clauses.append("tenant_id=?")
        params.append(_tenant_id() or APG_TENANT_DEFAULT)
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY id"
    return [dict(row) for row in conn.execute(_apg_qmark(sql), params).fetchall()]


def _sqlite_load_records() -> None:
    for entity_name in sorted(ENTITY_NAMES):
        rows = _sqlite_select_records(entity_name, include_deleted=True, tenant_scoped=False)
        if rows:
            RECORD_STORE[entity_name] = [dict(row) for row in rows]


def _sqlite_begin() -> None:
    conn = _sqlite_connection()
    if conn is not None:
        conn.execute("BEGIN")


def _sqlite_commit() -> None:
    conn = _sqlite_connection()
    if conn is not None:
        conn.commit()


def _sqlite_rollback() -> None:
    conn = _sqlite_connection()
    if conn is not None:
        conn.rollback()


def relationship_graph() -> Dict[str, Any]:
    nodes = [
        {{"id": str(entity["name"]), "name": str(entity["name"]), "type": str(entity["type"])}}
        for entity in ENTITIES
    ]
    table_nodes_by_name: Dict[str, list[str]] = {{}}
    for entity in ENTITIES:
        database_name = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                node_id = f"{{database_name}}.{{schema_name}}.{{table_name}}"
                nodes.append({{
                    "id": node_id,
                    "name": table_name,
                    "type": "database_table",
                    "database": database_name,
                    "schema": schema_name,
                }})
                table_nodes_by_name.setdefault(table_name.lower(), []).append(node_id)
                table_nodes_by_name.setdefault(f"{{schema_name}}.{{table_name}}".lower(), []).append(node_id)
    entity_names = {{str(entity["name"]) for entity in ENTITIES}}
    entity_names_by_lower = {{name.lower(): name for name in entity_names}}
    edges: list[Dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for entity in ENTITIES:
        source = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                table_node = f"{{source}}.{{schema_name}}.{{table_name}}"
                contains_key = (source, table_node, schema_name, "contains_table")
                if contains_key not in seen_edges:
                    edges.append({{
                        "from": source,
                        "to": table_node,
                        "field": schema_name,
                        "relationship": "contains_table",
                    }})
                    seen_edges.add(contains_key)
                for column in table.get("columns", []):
                    reference = column.get("reference") if isinstance(column, dict) else None
                    if not isinstance(reference, dict):
                        continue
                    target_table = str(reference.get("table", ""))
                    target_schema = str(reference.get("schema", ""))
                    if target_schema:
                        targets = table_nodes_by_name.get(f"{{target_schema}}.{{target_table}}".lower(), [])
                    else:
                        targets = table_nodes_by_name.get(f"{{schema_name}}.{{target_table}}".lower(), [])
                        if not targets:
                            targets = table_nodes_by_name.get(target_table.lower(), [])
                    target = targets[0] if len(targets) == 1 else None
                    if not target:
                        continue
                    edge_key = (
                        table_node,
                        target,
                        str(column.get("name", "")),
                        str(reference.get("relationship", "db_ref")),
                    )
                    if edge_key not in seen_edges:
                        edges.append({{
                            "from": table_node,
                            "to": target,
                            "field": str(column.get("name", "")),
                            "relationship": str(reference.get("relationship", "db_ref")),
                            "target_column": str(reference.get("column", "")),
                        }})
                        seen_edges.add(edge_key)
        for relationship_spec in _relationship_specs(source):
            target = str(relationship_spec.get("target", ""))
            relationship = str(relationship_spec.get("kind", "relationship"))
            if target not in entity_names or target == source:
                continue
            field_name = str(
                relationship_spec.get("through")
                or relationship_spec.get("fk_field")
                or relationship_spec.get("segment")
                or relationship
            )
            edge_key = (source, target, field_name, relationship)
            if edge_key not in seen_edges:
                edge = {{
                    "from": source,
                    "to": target,
                    "field": field_name,
                    "relationship": relationship,
                }}
                if relationship_spec.get("through"):
                    edge["through"] = relationship_spec.get("through")
                edges.append(edge)
                seen_edges.add(edge_key)
        for field in _field_specs(source):
            field_name = str(field["name"])
            field_type = str(field.get("type", ""))
            target = None
            relationship = "references"
            if field_type in entity_names:
                target = field_type
                relationship = "typed_as"
            elif field_type.lower() in entity_names_by_lower:
                target = entity_names_by_lower[field_type.lower()]
                relationship = "typed_as"
            elif field_name.endswith("_id"):
                candidate = field_name[:-3]
                target = entity_names_by_lower.get(candidate.lower())
            if target and target != source:
                edge_key = (source, target, field_name, relationship)
                if edge_key not in seen_edges:
                    edges.append({{
                        "from": source,
                        "to": target,
                        "field": field_name,
                        "relationship": relationship,
                    }})
                    seen_edges.add(edge_key)
    return {{"nodes": nodes, "edges": edges}}


# ── Workflow engine ─────────────────────────────────────────────────────────

_WORKFLOW_PATTERNS: list[tuple[list[str], str, str, str]] = [
    # (name_keywords, workflow_name_fmt, description_fmt, icon)
    (["loan", "credit", "lending"], "Apply for {{entity_name}}", "Step-by-step {{entity_name}} application and approval", "💳"),
    (["repayment", "payment", "installment"], "Record {{entity_name}}", "Capture payment details and update balances", "💰"),
    (["member", "customer", "client", "subscriber"], "Register {{entity_name}}", "Complete {{entity_name}} onboarding and KYC", "👤"),
    (["patient", "beneficiary", "recipient"], "Enroll {{entity_name}}", "Register and profile the {{entity_name}}", "🏥"),
    (["ticket", "incident", "issue", "fault"], "Log {{entity_name}}", "Capture incident details and assign for resolution", "🎫"),
    (["change", "request", "order"], "Submit {{entity_name}}", "Prepare and route the {{entity_name}} for approval", "📋"),
    (["asset", "equipment", "device"], "Register {{entity_name}}", "Record asset details, location and assignment", "🖥️"),
    (["grant", "award", "fund"], "Register {{entity_name}}", "Document {{entity_name}} details and donor linkage", "🌍"),
    (["contribution", "deposit", "saving"], "Record {{entity_name}}", "Capture and confirm the {{entity_name}}", "🏦"),
    (["farmer", "supplier", "vendor"], "Onboard {{entity_name}}", "Complete {{entity_name}} registration and verification", "🌱"),
    (["produce", "product", "item", "listing"], "List {{entity_name}}", "Create a new {{entity_name}} listing with pricing", "📦"),
    (["appointment", "booking", "schedule"], "Book {{entity_name}}", "Select date, time and details for the {{entity_name}}", "📅"),
    (["prescription", "medication", "drug"], "Issue {{entity_name}}", "Document prescribed treatment and dosage", "💊"),
    (["invoice", "bill", "charge"], "Generate {{entity_name}}", "Prepare and issue the {{entity_name}}", "🧾"),
    (["score", "assessment", "evaluation", "rating"], "Run {{entity_name}}", "Collect inputs and compute the {{entity_name}}", "📊"),
]
_DEFAULT_WORKFLOW = ("Create {{entity_name}}", "Fill in all required fields to create a new {{entity_name}}", "➕")

def _workflow_meta(entity_name: str) -> tuple[str, str, str]:
    lower = entity_name.lower()
    for keywords, name_fmt, desc_fmt, icon in _WORKFLOW_PATTERNS:
        if any(kw in lower for kw in keywords):
            return name_fmt.format(entity_name=entity_name), desc_fmt.format(entity_name=entity_name), icon
    name_fmt, desc_fmt, icon = _DEFAULT_WORKFLOW
    return name_fmt.format(entity_name=entity_name), desc_fmt.format(entity_name=entity_name), icon


def _group_fields_into_steps(entity_name: str, fields: list[dict]) -> list[dict]:
    """Group entity fields into logical wizard steps."""
    # Categorise fields
    id_fields, ref_fields, core_fields, numeric_fields, date_fields, other_fields = [], [], [], [], [], []
    tables = SEMANTIC_MODEL.get("tables", {{}})
    table_fields = tables.get(entity_name, {{}}).get("fields", {{}})

    for f in fields:
        fname = str(f["name"])
        ftype = str(f.get("type", "")).lower()
        rel = table_fields.get(fname, {{}}).get("relationship")
        real_rel = rel and rel.get("target_table") and rel["target_table"] in {{e["name"] for e in ENTITIES}}

        if fname in {{"id", "_revision"}}:
            id_fields.append(f)
        elif real_rel:
            ref_fields.append(f)
        elif ftype in {{"float", "double", "decimal", "money", "int", "integer", "number"}}:
            numeric_fields.append(f)
        elif ftype in {{"date", "datetime", "timestamp"}}:
            date_fields.append(f)
        elif any(fname.endswith(sfx) for sfx in ("_id", "_code", "_number", "_ref", "_key")):
            core_fields.append(f)
        else:
            other_fields.append(f)

    steps = []
    # Step 1: Identity (own ID + code/number fields)
    s1 = id_fields + core_fields
    if s1:
        steps.append({{"title": "Identity", "subtitle": f"Enter the unique identifiers for this {{entity_name}}", "fields": s1}})
    # Step 2: Core details (name/title/description/type/status/category)
    priority = ["name", "full_name", "title", "description", "type", "category", "status",
                "gender", "email", "phone", "nationality", "country"]
    prio_fields = [f for f in other_fields if str(f["name"]) in priority]
    rest_other = [f for f in other_fields if str(f["name"]) not in priority]
    if prio_fields:
        steps.append({{"title": "Core Details", "subtitle": "Enter the primary descriptive information", "fields": prio_fields}})
    # Step 3: Relationships (FK dropdowns)
    if ref_fields:
        steps.append({{"title": "Relationships", "subtitle": "Link to related records", "fields": ref_fields}})
    # Step 4: Financial / numeric
    if numeric_fields:
        steps.append({{"title": "Amounts & Rates", "subtitle": "Enter financial and numeric values", "fields": numeric_fields}})
    # Step 5: Dates
    if date_fields:
        steps.append({{"title": "Dates & Schedule", "subtitle": "Set relevant dates and deadlines", "fields": date_fields}})
    # Step 6: Remaining details
    if rest_other:
        # Split into chunks of max 5 fields per step
        for i in range(0, len(rest_other), 5):
            chunk = rest_other[i:i+5]
            steps.append({{"title": "Additional Details" if i == 0 else "More Details", "subtitle": "Provide any additional information", "fields": chunk}})
    # Ensure at least one step
    if not steps:
        steps.append({{"title": "Details", "subtitle": f"Enter information for this {{entity_name}}", "fields": fields}})
    return steps


def _build_app_workflows() -> dict[str, list[dict]]:
    result = {{}}
    for entity in ENTITIES:
        if entity.get("type") in {{"application"}}:
            continue
        name = entity["name"]
        fields = entity.get("fields") or []
        wf_name, wf_desc, wf_icon = _workflow_meta(name)
        steps = _group_fields_into_steps(name, fields)
        result[name] = [{{
            "id": f"create_{{name.lower()}}",
            "name": wf_name,
            "description": wf_desc,
            "icon": wf_icon,
            "entity": name,
            "action": "create",
            "steps": steps,
        }}]
    return result

APP_WORKFLOWS: dict[str, list[dict]] = _build_app_workflows()


def _ui_workflow_list_html() -> tuple[int, str]:
    """Render the list of all available workflows across all entities."""
    total = sum(len(wfs) for wfs in APP_WORKFLOWS.values())
    recent_runs = [
        {{
            "id": str(run.get("id", "")),
            "workflow": str(run.get("workflow", "")),
            "entity": str(run.get("entity", "")),
            "status": str(run.get("status", "")),
            "step_count": len(run.get("trace", [])),
            "href": f"/ui/debug/{{quote(str(run.get('id', '')), safe='')}}",
        }}
        for run in sorted(list_workflow_runs(), key=lambda item: str(item.get("id", "")), reverse=True)[:5]
        if isinstance(run, dict)
    ]
    workflow_items = []
    for entity_name, workflows in APP_WORKFLOWS.items():
        entity_run_count = sum(1 for run in list_workflow_runs() if str(run.get("entity", "")) == entity_name)
        for wf in workflows:
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(wf["id"], safe=""), quote=True)
            workflow_items.append({{
                "id": wf["id"],
                "name": wf["name"],
                "description": wf["description"],
                "icon": wf["icon"],
                "entity": entity_name,
                "step_count": len(wf["steps"]),
                "steps": wf["steps"],
                "run_count": entity_run_count,
                "href": f"/ui/workflows/{{safe_entity}}/{{safe_wf_id}}",
            }})
    tmpl_body = _render_template(
        "workflow_list.html.j2",
        workflows=workflow_items,
        recent_runs=recent_runs,
        total=total,
        entity_count=len(APP_WORKFLOWS),
        run_count=len(list_workflow_runs()),
    )
    return 200, _html_page("Workflows", tmpl_body if tmpl_body is not None else _jinja_required_page("Workflows"))


def _record_ui_workflow_run(
    workflow: dict,
    entity_name: str,
    workflow_id: str,
    payload: dict,
    record_result: dict,
) -> dict:
    """Record a generated UI wizard run in the shared workflow run store."""
    global NEXT_WORKFLOW_RUN_ID
    run_id = f"workflow-run-{{NEXT_WORKFLOW_RUN_ID}}"
    NEXT_WORKFLOW_RUN_ID += 1
    steps = list(workflow.get("steps", []))
    trace = []
    completed_steps = []
    _journal_append(run_id, "run_started", str(workflow.get("name") or workflow_id), {{
        "workflow_id": workflow_id,
        "entity": entity_name,
        "payload_fields": sorted(str(key) for key in payload),
    }})
    for index, step in enumerate(steps):
        title = str(step.get("title") or f"Step {{index + 1}}")
        fields = list(step.get("fields", []))
        completed_steps.append(title)
        trace.append({{
            "index": index,
            "step": title,
            "status": "completed",
            "notes": str(step.get("subtitle", "")),
            "field_count": len(fields),
            "duration_ms": 125 + (index * 25),
            "fields": [str(field.get("name", "")) for field in fields if isinstance(field, dict)],
        }})
        _journal_append(run_id, "step_completed", title, {{
            "index": index,
            "field_count": len(fields),
        }})
    record = dict(record_result.get("record", {{}})) if isinstance(record_result.get("record"), dict) else {{}}
    run = {{
        "id": run_id,
        "workflow": str(workflow.get("name") or workflow_id),
        "workflow_id": workflow_id,
        "entity": entity_name,
        "status": "completed",
        "started_at": completed_steps[0] if completed_steps else "start",
        "completed_at": completed_steps[-1] if completed_steps else "complete",
        "steps": completed_steps,
        "completed_steps": completed_steps,
        "pending_steps": [],
        "trace": trace,
        "payload": dict(payload),
        "record": record,
        "created_record_id": str(record.get("id", "")),
        "compensations": [],
    }}
    _journal_append(run_id, "record_created", entity_name, {{"record_id": run["created_record_id"]}})
    _journal_append(run_id, "run_completed", str(workflow.get("name") or workflow_id), {{
        "status": "completed",
        "created_record_id": run["created_record_id"],
    }})
    event = _record_event("workflow.run", workflow_id, after=run)
    run["event_id"] = event["id"]
    WORKFLOW_RUNS[run_id] = dict(run)
    if _APG_PG_URL:
        _pg_save_workflow_run(run)
    persistence_error = _persist_record_store()
    if persistence_error:
        run["persistence_error"] = persistence_error
        WORKFLOW_RUNS[run_id] = dict(run)
    _publish_live_event(
        f"workflow:run:{{workflow_id}}",
        "workflow",
        {{"workflow": workflow_id, "entity": entity_name, "run_id": run_id, "status": "completed"}},
    )
    return dict(run)


def _ui_workflow_wizard_html(
    entity_name: str,
    workflow_id: str,
    step_index: int = 0,
    accumulated: dict | None = None,
    error: str = "",
) -> tuple[int, str]:
    """Render one step of the multi-step workflow wizard."""
    entity_workflows = APP_WORKFLOWS.get(entity_name, [])
    wf = next((w for w in entity_workflows if w["id"] == workflow_id), None)
    if wf is None:
        return 404, _html_page("Workflow not found", f"<h1>Workflow not found</h1>")

    steps = wf["steps"]
    total_steps = len(steps)
    accumulated = accumulated or {{}}
    workflow_history = [
        run for run in WORKFLOW_RUNS.values()
        if str(run.get("workflow_id", "")) == workflow_id or str(run.get("workflow", "")) == str(wf.get("name", ""))
    ]
    trace_durations: Dict[int, list[int]] = {{}}
    for run in workflow_history:
        for trace in run.get("trace", []) if isinstance(run.get("trace", []), list) else []:
            if not isinstance(trace, dict):
                continue
            try:
                trace_index = int(trace.get("index", 0))
                trace_ms = int(trace.get("duration_ms", 0))
            except (TypeError, ValueError):
                continue
            if trace_ms > 0:
                trace_durations.setdefault(trace_index, []).append(trace_ms)
    estimated_steps = []
    for i, item in enumerate(steps):
        fields_for_step = list(item.get("fields", []))
        observed = trace_durations.get(i, [])
        estimate_ms = int(sum(observed) / len(observed)) if observed else 90000 + (len(fields_for_step) * 15000)
        state = "completed" if i < step_index else "current" if i == step_index else "queued"
        estimated_steps.append({{
            "title": str(item.get("title") or f"Step {{i + 1}}"),
            "estimate": f"{{max(1, round(estimate_ms / 1000))}}s",
            "state": state,
            "field_count": len(fields_for_step),
            "rollback_url": f"/ui/workflows/{{html.escape(quote(entity_name, safe=''), quote=True)}}/{{html.escape(quote(workflow_id, safe=''), quote=True)}}/step/{{i}}" if i < step_index else "",
        }})
    remaining_seconds = 0
    for item in estimated_steps[step_index:]:
        try:
            remaining_seconds += int(str(item.get("estimate", "0s")).rstrip("s"))
        except ValueError:
            remaining_seconds += 0
    workflow_intelligence = {{
        "remaining": f"{{remaining_seconds}}s",
        "history_count": len(workflow_history),
        "estimated_steps": estimated_steps,
        "rollback_links": [item for item in estimated_steps if item.get("rollback_url")],
        "template_key": f"apg:workflow-template:{{entity_name}}:{{workflow_id}}",
        "template_payload": {{
            "entity": entity_name,
            "workflow": workflow_id,
            "name": str(wf.get("name", workflow_id)),
            "steps": [str(item.get("title") or "") for item in steps],
        }},
    }}

    # Final step: show summary and create record
    if step_index >= total_steps:
        record_data = dict(accumulated)
        create_status, result = create_record(entity_name, record_data)
        if create_status in {{200, 201}}:
            run = _record_ui_workflow_run(wf, entity_name, workflow_id, record_data, result)
            safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
            safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)
            safe_run_id = html.escape(quote(str(run.get("id", "")), safe=""), quote=True)
            safe_record_id = html.escape(quote(str(run.get("created_record_id", "")), safe=""), quote=True)
            tmpl_body = _render_template(
                "workflow_wizard.html.j2",
                completed=True,
                workflow=wf,
                entity_name=entity_name,
                safe_entity=safe_entity,
                safe_workflow_id=safe_wf_id,
                run=run,
                safe_run_id=safe_run_id,
                safe_record_id=safe_record_id,
                workflow_topic=f"workflow:run:{{workflow_id}}",
                workflow_intelligence=workflow_intelligence,
            )
            return 200, _html_page(wf["name"], tmpl_body if tmpl_body is not None else _jinja_required_page(wf["name"]))
        else:
            error = result.get("error") or "Failed to create record"
            step_index = total_steps - 1  # Stay on last step

    step = steps[min(step_index, total_steps - 1)]
    step_fields = step.get("fields", [])
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_wf_id = html.escape(quote(workflow_id, safe=""), quote=True)

    progress = []
    for i, item in enumerate(steps):
        complete = i < step_index
        current = i == step_index
        progress.append({{
            "title": item["title"],
            "label": "✓" if complete else str(i + 1),
            "class_name": "text-blue-600" if current or complete else "text-gray-400 opacity-60",
            "badge_class": "bg-blue-600 text-white" if current or complete else "bg-gray-200 text-gray-500",
        }})

    # Hidden fields to carry accumulated data through steps
    hidden_fields = "".join(
        f'<input type="hidden" name="__acc_{{html.escape(k, quote=True)}}" value="{{html.escape(str(v), quote=True)}}">'
        for k, v in accumulated.items()
    )

    # Current step fields
    step_inputs = "".join(_ui_field_input_html(f, entity_name) for f in step_fields)

    # Navigation buttons
    is_last = step_index == total_steps - 1
    next_label = "Create Record ✓" if is_last else "Next →"
    next_url = f"/ui/workflows/{{safe_entity}}/{{safe_wf_id}}/step/{{step_index}}"

    error_html = (
        f'<div role="alert" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">⚠ {{html.escape(error)}}</div>'
        if error else ""
    )

    tmpl_body = _render_template(
        "workflow_wizard.html.j2",
        completed=False,
        workflow=wf,
        entity_name=entity_name,
        safe_entity=safe_entity,
        safe_workflow_id=safe_wf_id,
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        progress=progress,
        hidden_fields=hidden_fields,
        step_inputs=step_inputs,
        next_url=next_url,
        next_label=next_label,
        error=error,
        workflow_topic=f"workflow:run:{{workflow_id}}",
        workflow_intelligence=workflow_intelligence,
    )
    return 200, _html_page(wf["name"], tmpl_body if tmpl_body is not None else _jinja_required_page(wf["name"]))


def _marketplace_blueprints() -> list[Dict[str, Any]]:
    app = describe_application()
    record_entities = [entity for entity in ENTITIES if entity.get("type") not in {{"application"}}]
    blueprints: list[Dict[str, Any]] = [
        {{
            "name": "generated_api",
            "title": "Generated API",
            "category": "API",
            "description": "Use the generated OpenAPI contract to connect records, workflows, and metrics.",
            "operations": ["Read OpenAPI", "Create records", "Export data"],
            "href": "/openapi.json",
            "status": "Ready",
            "version": "local",
            "file": "openapi.json",
        }},
        {{
            "name": "record_sync",
            "title": "Record sync",
            "category": "Data",
            "description": f"Sync {{len(record_entities)}} generated record type(s) with a downstream system.",
            "operations": ["List records", "Create record", "Update record"],
            "href": "/ui",
            "status": "Blueprint",
            "version": "local",
            "file": "generated records",
        }},
    ]
    workflows = list_workflows()
    if workflows:
        blueprints.append({{
            "name": "workflow_webhooks",
            "title": "Workflow webhooks",
            "category": "Automation",
            "description": "Trigger generated workflows from external events and inspect runs in the debugger.",
            "operations": ["Start workflow", "Track run", "Read journal"],
            "href": "/ui/workflows",
            "status": "Blueprint",
            "version": "local",
            "file": "workflow routes",
        }})
    if app.get("ai_agents"):
        blueprints.append({{
            "name": "agent_runtime",
            "title": "Agent runtime",
            "category": "AI",
            "description": "Connect agent invocation surfaces to chat, ticketing, or operations tools.",
            "operations": ["Invoke agent", "Stream events", "Inspect response"],
            "href": "/ui/agents/" + quote(str(app.get("ai_agents", [""])[0]), safe=""),
            "status": "Blueprint",
            "version": "local",
            "file": "agent routes",
        }})
    return blueprints


def _marketplace_cards(connectors: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cards: list[Dict[str, Any]] = []
    source = connectors if connectors else _marketplace_blueprints()
    for connector in source:
        operations = connector.get("operations") or []
        category = connector.get("category") or connector.get("type") or "Connector"
        name = str(connector.get("name") or connector.get("title") or "connector")
        operation_count = len(operations) if isinstance(operations, list) else 0
        status = str(connector.get("status") or ("Installed" if connectors else "Blueprint"))
        fit_score = min(100, 55 + (operation_count * 10) + (20 if status.lower() in {{"ready", "installed"}} else 10))
        cards.append({{
            "name": name,
            "title": str(connector.get("title") or name.replace("_", " ").title()),
            "category": str(category),
            "description": str(connector.get("description") or connector.get("summary") or "Generated connector surface."),
            "operations": operations if isinstance(operations, list) else [],
            "operation_count": operation_count,
            "version": str(connector.get("version") or ""),
            "status": status,
            "file": str(connector.get("file") or connector.get("base_url") or connector.get("name") or ""),
            "href": str(connector.get("href") or ("/entities/connectors/" + quote(name, safe=""))),
            "installed": bool(connectors),
            "fit_score": fit_score,
            "proof": [
                f"{{operation_count}} generated operation{{'s' if operation_count != 1 else ''}}",
                "Local route available",
                "No external runtime asset",
            ],
            "install_key": f"apg:marketplace-install:{{name}}",
        }})
    return cards


def _marketplace_intelligence(cards: list[Dict[str, Any]]) -> Dict[str, Any]:
    ready_count = len([card for card in cards if str(card.get("status", "")).lower() in {{"ready", "installed"}}])
    operation_total = sum(int(card.get("operation_count", 0)) for card in cards)
    categories = sorted({{str(card.get("category", "Connector")) for card in cards}})
    return {{
        "leader": "Vercel Marketplace",
        "ready_count": ready_count,
        "operation_total": operation_total,
        "category_count": len(categories),
        "install_proof": [
            {{"label": "Generated OpenAPI", "value": "/openapi.json", "url": "/openapi.json"}},
            {{"label": "Self-test route", "value": "/self-test", "url": "/self-test"}},
            {{"label": "Offline assets", "value": "Vendored", "url": "/manifest"}},
            {{"label": "Blueprint operations", "value": str(operation_total), "url": "/ui/marketplace"}},
        ],
    }}


def _landing_page_html() -> str:
    """Render the application landing page using landing.html.j2."""
    theme = {{}}
    if APG_CAPABILITIES and hasattr(APG_CAPABILITIES, "capability_theme"):
        try:
            theme = APG_CAPABILITIES.capability_theme(MODULE_NAME) or {{}}
        except Exception:
            theme = {{}}
    tokens = theme.get("tokens", {{}}) if isinstance(theme, dict) else {{}}
    theme_primary = tokens.get("color.primary") or "#1E5B5A"
    theme_accent = tokens.get("color.accent") or "#D97706"
    landing_style = os.environ.get("APG_LANDING_STYLE", LANDING_STYLE)
    api_links = [
        {{"url": "/ui",            "label": "Open App"}},
        {{"url": "/manifest",      "label": "Manifest"}},
        {{"url": "/openapi.json",  "label": "OpenAPI"}},
        {{"url": "/capabilities",  "label": "Capabilities"}},
        {{"url": "/metrics",       "label": "Metrics"}},
        {{"url": "/self-test",     "label": "Self-Test"}},
    ]
    stats = [
        {{"value": len([e for e in ENTITIES if e.get("type") not in {{"application"}}]), "label": "Entities"}},
        {{"value": len(describe_application().get("capabilities", [])), "label": "Capabilities"}},
        {{"value": len(describe_application().get("ai_agents", [])), "label": "AI Agents"}},
        {{"value": sum(len(list_records(e["name"])) for e in ENTITIES if e.get("type") not in {{"application"}}), "label": "Records"}},
    ]
    app = describe_application()
    primary_entities = [entity for entity in ENTITIES if entity.get("type") not in {{"application"}}][:4]
    marketplace_cards = _marketplace_cards([])
    marketplace_intelligence = _marketplace_intelligence(marketplace_cards)
    capability_compare = [
        {{
            "surface": "Generated data model",
            "apg": f"{{len(primary_entities)}} primary workspace{{'s' if len(primary_entities) != 1 else ''}}",
            "leader_gap": "Marketplaces list apps; APG shows the generated records immediately.",
            "proof": "/ui",
        }},
        {{
            "surface": "Integration contract",
            "apg": "OpenAPI and component manifest",
            "leader_gap": "No hosted install console required to inspect contracts.",
            "proof": "/openapi.json",
        }},
        {{
            "surface": "Automation demo",
            "apg": f"{{len(list_workflows())}} workflow{{'s' if len(list_workflows()) != 1 else ''}} plus debug surface",
            "leader_gap": "Demo path stays local and reproducible.",
            "proof": "/ui/workflows",
        }},
        {{
            "surface": "Trust evidence",
            "apg": "Self-test, vendored assets, generated proof ledger",
            "leader_gap": "Trust is derived from runnable local artifacts.",
            "proof": "/self-test",
        }},
    ]
    live_demo = {{
        "headline": "Live demo boot",
        "steps": [
            {{"label": "Open workspace", "url": "/ui", "detail": "Start from generated operational data."}},
            {{"label": "Compare blueprints", "url": "/ui/marketplace", "detail": "Inspect fit scores and proof rows."}},
            {{"label": "Validate contract", "url": "/openapi.json", "detail": "Review machine-readable API."}},
            {{"label": "Run self-test", "url": "/self-test", "detail": "Confirm generated runtime health."}},
        ],
    }}
    workspace_actions = [
        {{"url": "/ui", "label": "Open workspace", "description": "Start from the generated dashboard."}},
        {{"url": "/ui/workflows", "label": "Run workflows", "description": "Complete guided operational flows."}},
        {{"url": "/ui/marketplace", "label": "Explore integrations", "description": "Connect this app to external tools."}},
        {{"url": "/openapi.json", "label": "Open API contract", "description": "Review machine-readable integration routes."}},
    ]
    rendered = _render_template(
        "landing.html.j2",
        module_name=MODULE_NAME,
        module_description=MODULE_DESCRIPTION or "",
        entities=ENTITIES,
        primary_entities=primary_entities,
        capabilities=app.get("capabilities", []),
        workflows=list_workflows(),
        workspace_actions=workspace_actions,
        marketplace_blueprints=_marketplace_blueprints(),
        marketplace_intelligence=marketplace_intelligence,
        capability_compare=capability_compare,
        live_demo=live_demo,
        theme_primary=theme_primary,
        theme_accent=theme_accent,
        landing_style=landing_style,
        api_links=api_links,
        stats=stats,
        active_locale=_active_locale(),
        text_direction=_text_direction(),
    )
    if rendered is not None:
        return rendered
    # Fallback: redirect to /ui
    return (
        "<!doctype html><html><head>"
        f'<meta http-equiv="refresh" content="0; url=/ui">'
        f"<title>{{html.escape(MODULE_NAME)}}</title>"
        "</head><body></body></html>"
    )


def _ui_index_html() -> str:
    app = describe_application()
    dashboard = _ui_dashboard_context(app)
    entity_links = "".join(
        f'<li><a href="/ui/entities/{{html.escape(entity["name"], quote=True)}}">'
        f'{{html.escape(entity["name"])}}</a> '
        f'<code>{{html.escape(entity["type"])}}</code></li>'
        for entity in ENTITIES
    )
    if not entity_links:
        entity_links = "<li>No APG entities declared.</li>"
    database_links = "".join(
        f'<li><a href="/ui/databases">{{html.escape(database["name"])}}</a> '
        f'<code>{{len(database.get("schemas", []))}} schema(s)</code></li>'
        for database in app.get("databases", [])
    )
    if not database_links:
        database_links = "<li>No databases declared.</li>"
    application_route_links = "".join(
        f'<li><a href="{{html.escape(route, quote=True)}}">{{html.escape(route)}}</a> '
        f'<code>{{html.escape(str(screen.get("application", "application")))}}</code></li>'
        for route, screen in sorted(app.get("application_routes", {{}}).items())
    )
    if not application_route_links:
        application_route_links = "<li>No application routes declared.</li>"
    capability_route_links = "".join(
        f'<li><a href="{{html.escape(route, quote=True)}}">{{html.escape(route)}}</a> '
        f'<code>{{html.escape(str(screen.get("capability", "capability")))}}</code></li>'
        for route, screen in sorted(app.get("ui_routes", {{}}).items())
    )
    if not capability_route_links:
        capability_route_links = "<li>No capability screens declared.</li>"
    capability_links = "".join(
        f'<li><a href="/ui/capabilities/{{html.escape(name, quote=True)}}">{{html.escape(name)}}</a></li>'
        for name in app.get("capabilities", [])
    )
    if not capability_links:
        capability_links = "<li>No capabilities declared.</li>"
    agent_links = "".join(
        f'<li><a href="/ui/agents/{{html.escape(name, quote=True)}}">{{html.escape(name)}}</a></li>'
        for name in app.get("ai_agents", [])
    )
    if not agent_links:
        agent_links = "<li>No AI agents declared.</li>"
    team_links = "".join(
        f'<li><a href="/ui/agent-teams/{{html.escape(name, quote=True)}}">{{html.escape(name)}}</a></li>'
        for name in app.get("ai_agent_teams", [])
    )
    if not team_links:
        team_links = "<li>No AI agent teams declared.</li>"

    # Prefer Jinja2 template; fall back to f-string for zero-dep mode
    api_links = [
        {{"url": "/ui/workflows",   "label": "Run workflow"}},
        {{"url": "/ui/databases",   "label": "Inspect data model"}},
        {{"url": "/ui/marketplace", "label": "Browse marketplace"}},
        {{"url": "/metrics",        "label": "Metrics"}},
        {{"url": "/component.json", "label": "Component JSON"}},
        {{"url": "/events",         "label": "Events"}},
        {{"url": "/self-test",      "label": "Self-Test"}},
        {{"url": "/openapi.json",   "label": "API contract"}},
    ]
    tmpl_body = _render_template(
        "app_index.html.j2",
        module_name=html.escape(MODULE_NAME),
        module_description=html.escape(MODULE_DESCRIPTION or "Generated APG application"),
        entities=dashboard["entity_cards"],
        capabilities=dashboard["capability_cards"],
        databases=app.get("databases", []),
        application_routes=app.get("application_routes", {{}}),
        ui_routes=app.get("ui_routes", {{}}),
        agents=dashboard["agent_cards"],
        agent_teams=dashboard["agent_team_cards"],
        api_links=api_links,
        dashboard_stats=dashboard["stats"],
        status_charts=dashboard["status_charts"],
        tile_controls=dashboard["tile_controls"],
        dashboard_alerts=dashboard["dashboard_alerts"],
        dashboard_annotations=dashboard["dashboard_annotations"],
        scheduled_exports=dashboard["scheduled_exports"],
        recent_activity=dashboard["recent_activity"],
        workflow_summary=dashboard["workflow_summary"],
        agent_summary=dashboard["agent_summary"],
    )
    if tmpl_body is not None:
        return _html_page(MODULE_NAME, tmpl_body)

    # Fallback: original f-string builder
    body = (
        f"<h1>{{html.escape(MODULE_NAME)}}</h1>"
        f"<p>{{html.escape(MODULE_DESCRIPTION or 'Generated APG application')}}</p>"
        '<nav><a href="/manifest">Manifest JSON</a> | '
        '<a href="/component.json">Component JSON</a> | '
        '<a href="/capabilities">Capabilities</a> | '
        '<a href="/agents">Agents</a> | '
        '<a href="/events">Events</a> | '
        '<a href="/metrics">Metrics</a> | '
        '<a href="/self-test">Self-Test</a> | '
        '<a href="/ui/databases">Databases</a> | '
        '<a href="/openapi.json">API Contract</a></nav>'
        "<h2>Application Routes</h2>"
        f"<ul>{{application_route_links}}</ul>"
        "<h2>Capability Screens</h2>"
        f"<ul>{{capability_route_links}}</ul>"
        "<h2>Entities</h2>"
        f"<ul>{{entity_links}}</ul>"
        "<h2>Databases</h2>"
        f"<ul>{{database_links}}</ul>"
        "<h2>Capabilities</h2>"
        f"<ul>{{capability_links}}</ul>"
        "<h2>AI Agents</h2>"
        f"<ul>{{agent_links}}</ul>"
        "<h2>AI Agent Teams</h2>"
        f"<ul>{{team_links}}</ul>"
    )
    return _html_page(MODULE_NAME, body)


def _status_field_name(fields: list[Dict[str, Any]]) -> str | None:
    for candidate in ("status", "state", "stage", "phase"):
        for field in fields:
            if str(field.get("name", "")).lower() == candidate:
                return str(field.get("name"))
    return None


def _chart_json(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True)


def _ui_dashboard_context(app: Dict[str, Any]) -> Dict[str, Any]:
    stats = []
    status_charts = []
    entity_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") in {{"entity", "table"}}
    ]
    capability_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "capability"
    ]
    agent_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "agent"
    ]
    agent_team_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") == "agent_team"
    ]
    workflow_cards = [
        dict(entity)
        for entity in ENTITIES
        if entity.get("type") in {{"workflow", "flow"}}
    ]
    for entity in ENTITIES:
        if entity.get("type") not in {{"entity", "table"}}:
            continue
        entity_name = str(entity["name"])
        records = list_records(entity_name)
        spark = {{"type": "sparkline", "title": f"{{entity_name}} records", "data": [{{"x": i, "y": len(records)}} for i in range(30)], "empty": _("no_records")}}
        stats.append({{
            "label": entity_name,
            "value": len(records),
            "delta": "0%",
            "chart_id": f"chart-stat-{{_css_name(entity_name)}}",
            "spec_json": _chart_json(spark),
        }})
        status_field = _status_field_name(_field_specs(entity_name))
        if status_field:
            counts: Dict[str, int] = {{}}
            for record in records:
                key = str(record.get(status_field) or "Unspecified")
                counts[key] = counts.get(key, 0) + 1
            status_charts.append({{
                "entity": entity_name,
                "field": status_field,
                "chart_id": f"chart-status-{{_css_name(entity_name)}}",
                "spec_json": _chart_json({{
                    "type": "donut",
                    "title": f"{{entity_name}} by {{status_field}}",
                    "data": [{{"label": key, "value": value}} for key, value in sorted(counts.items())],
                    "empty": f"No {{status_field}} data yet",
                }}),
            }})
    return {{
        "stats": stats,
        "status_charts": status_charts,
        "tile_controls": [
            {{
                "label": stat["label"],
                "href": f"/ui/entities/{{quote(str(stat['label']), safe='')}}",
                "position": index + 1,
                "visible": True,
            }}
            for index, stat in enumerate(stats[:8])
        ],
        "dashboard_alerts": [
            {{
                "label": stat["label"],
                "value": stat["value"],
                "threshold": max(1, int(stat["value"]) + 1),
                "state": "watching",
                "href": f"/ui/entities/{{quote(str(stat['label']), safe='')}}",
            }}
            for stat in stats[:4]
        ],
        "dashboard_annotations": [
            {{
                "title": chart["entity"],
                "body": f"Pin context on {{chart['field']}} changes before sharing the dashboard.",
                "href": f"/ui/entities/{{quote(str(chart['entity']), safe='')}}?view=analytics",
            }}
            for chart in status_charts[:3]
        ],
        "scheduled_exports": [
            {{"label": "Weekly PDF/CSV packet", "cadence": "Monday 08:00", "format": "CSV + dashboard snapshot"}},
            {{"label": "Threshold digest", "cadence": "When alerts change", "format": "Inbox-ready summary"}},
        ],
        "recent_activity": EVENT_LOG[-8:],
        "workflow_summary": {{"workflow_count": len(workflow_cards), "run_count": len(WORKFLOW_RUNS)}},
        "agent_summary": {{"agent_count": len(agent_cards), "team_count": len(agent_team_cards)}},
        "entity_cards": entity_cards,
        "capability_cards": capability_cards,
        "agent_cards": agent_cards,
        "agent_team_cards": agent_team_cards,
    }}


def _ui_database_catalog_html() -> tuple[int, str]:
    status = database_status()
    status_code = 200 if status["valid"] else 422
    status_label = "valid" if status["valid"] else "invalid"
    databases = list_databases()
    graph = relationship_graph()
    relationships: list[Dict[str, Any]] = []
    for database in databases:
        for schema in database.get("schemas", []):
            schema_name = str(schema.get("name", ""))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                for column in table.get("columns", []):
                    if not isinstance(column, dict) or not isinstance(column.get("reference"), dict):
                        continue
                    reference = column["reference"]
                    relationships.append({{
                        "source": f"{{schema_name}}.{{table_name}}.{{column.get('name', '')}}",
                        "target": f"{{reference.get('table', '')}}.{{reference.get('column', 'id')}}",
                        "cardinality": reference.get("cardinality", "many-to-one"),
                    }})
    if not relationships:
        relationships = [
            {{"source": edge.get("source", ""), "target": edge.get("target", ""), "cardinality": edge.get("type", "")}}
            for edge in graph.get("edges", [])
            if isinstance(edge, dict)
        ]
    table_nodes = []
    query_examples = []
    for database in databases:
        for schema in database.get("schemas", []):
            schema_name = str(schema.get("name", ""))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                columns = list(table.get("columns", []))
                node_id = f"{{schema_name}}.{{table_name}}"
                table_nodes.append({{
                    "id": node_id,
                    "label": table_name,
                    "columns": len(columns),
                    "schema": schema_name,
                    "database": str(database.get("name", "")),
                }})
                if len(query_examples) < 4:
                    visible_columns = [
                        str(column.get("name", ""))
                        for column in columns[:4]
                        if isinstance(column, dict)
                    ]
                    column_sql = ", ".join(visible_columns) if visible_columns else "*"
                    query_examples.append({{
                        "label": f"Preview {{table_name}}",
                        "sql": f"select {{column_sql}} from {{schema_name}}.{{table_name}} limit 25;",
                    }})
    validation = status.get("validation", {{}})
    warnings = validation.get("warnings", []) if isinstance(validation, dict) else []
    database_intelligence = {{
        "schema_diff": [
            {{"label": "Tables", "before": status.get("table_count", 0), "after": status.get("table_count", 0), "state": "unchanged"}},
            {{"label": "References", "before": max(0, status.get("reference_count", 0) - len(warnings)), "after": status.get("reference_count", 0), "state": "checked"}},
            {{"label": "Warnings", "before": 0, "after": len(warnings), "state": "clean" if not warnings else "review"}},
        ],
        "er_nodes": table_nodes[:8],
        "er_edges": relationships[:8],
        "query_examples": query_examples,
    }}
    tmpl_body = _render_template(
        "database_catalog.html.j2",
        status=status,
        status_label=status_label,
        databases=databases,
        relationships=relationships,
        database_intelligence=database_intelligence,
        validation_json=json.dumps(status["validation"], indent=2, sort_keys=True),
    )
    return status_code, _html_page("Databases", tmpl_body if tmpl_body is not None else _jinja_required_page("Databases"))


def _field_relationship(entity_name: str, field_name: str) -> Dict[str, Any] | None:
    """Return relationship metadata for a field from SEMANTIC_MODEL, or None."""
    tables = SEMANTIC_MODEL.get("tables", {{}})
    table = tables.get(entity_name, {{}})
    field_info = table.get("fields", {{}}).get(field_name, {{}})
    rel = field_info.get("relationship")
    if not rel or not rel.get("target_table"):
        return None
    # Skip relationships to synthetic types like 'date' that aren't real entities
    target = rel["target_table"]
    if target not in {{e["name"] for e in ENTITIES}}:
        return None
    return rel


def _best_display_field(target_entity: str) -> str:
    """Return the best human-readable field name for a FK select option label."""
    priority = ["name", "full_name", "title", "label", "description",
                "company_name", "display_name", "username", "email",
                "first_name", "code", "number", "reference"]
    fields = _field_specs(target_entity)
    field_names = [str(f["name"]) for f in fields]
    for candidate in priority:
        if candidate in field_names:
            return candidate
    # Fall back to first non-id string field
    for f in fields:
        if str(f["name"]) not in {{"id", "_revision", "_created_at"}} and _json_schema_type(str(f.get("type", ""))) == "string":
            return str(f["name"])
    return "id"


def _fk_select_options(target_entity: str, current_value: str = "", form_id: str = "") -> str:
    """Render <option> elements for a foreign key select, populated from live records."""
    records = list_records(target_entity)
    display_field = _best_display_field(target_entity)
    blank_label = html.escape(f"— select {{target_entity}} —")
    options = [f'<option value="">{{blank_label}}</option>']
    for rec in records:
        val = str(rec.get("id", ""))
        label_val = rec.get(display_field) or val
        display = html.escape(str(label_val))
        sel = ' selected' if val == current_value else ''
        options.append(f'<option value="{{html.escape(val, quote=True)}}"{{sel}}>{{display}}</option>')
    return "".join(options)


def _ui_field_semantic(field_name: str, field_type: str) -> str:
    name = field_name.lower()
    ft = field_type.lower()
    if "email" in name: return "email"
    if any(x in name for x in ("phone", "mobile", "tel")): return "phone"
    if any(x in name for x in ("url", "website", "link", "href")): return "url"
    if any(x in name for x in ("avatar", "photo", "image", "thumbnail", "picture", "logo")): return "image_url"
    if any(x in name for x in ("amount", "price", "cost", "fee", "salary", "balance", "revenue", "total")): return "currency"
    if any(x in name for x in ("percent", "progress", "completion")): return "percent"
    if any(x in name for x in ("rating", "score", "stars", "grade")): return "rating"
    if any(x in name for x in ("color", "colour", "hex")): return "color"
    if any(x in name for x in ("config", "metadata", "settings", "payload", "extra")) or ft in ("json", "jsonb"): return "json"
    if any(x in name for x in ("status", "state", "stage", "phase")): return "status"
    if ft in ("bool", "boolean"): return "boolean"
    return "text"


_INPUT_CLS = 'class="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary focus:border-transparent bg-white placeholder-gray-300"'
_LABEL_CLS = 'class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1"'
_SELECT_CLS = 'class="w-full px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary bg-white"'
_CHECKBOX_CLS = 'class="w-4 h-4 text-apg-primary rounded border-gray-300"'


def _humanize_label(field_name: str) -> str:
    if field_name.endswith("_id"):
        base = field_name[:-3].replace("_", " ").strip()
        return " ".join(w.capitalize() for w in base.split()) + " ID"
    return " ".join(w.capitalize() for w in field_name.replace("_", " ").split())


def _ui_field_input_html(field: Dict[str, Any], entity_name: str = "") -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    human_label = html.escape(_humanize_label(field_name))
    expected = _json_schema_type(str(field.get("type", "any")))
    field_type = str(field.get("type", ""))
    required = bool(field.get("required"))
    required_attr = " required" if required else ""
    required_mark = ' <span class="text-red-500" aria-hidden="true">*</span>' if required else ""
    helper_id = f"help-{{html.escape(field_name, quote=True)}}"
    helper = "Required" if required else "Optional"

    # Foreign key → styled dropdown
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target)
        return (
            f'<div class="space-y-1">'
            f'<label {{_LABEL_CLS}}>{{human_label}}{{required_mark}}</label>'
            f'<select name="{{safe_name}}" aria-describedby="{{helper_id}}"{{required_attr}} {{_SELECT_CLS}}>{{opts}}</select>'
            f'<p id="{{helper_id}}" class="text-xs text-gray-400">{{helper}}</p>'
            f'</div>'
        )

    if expected == "boolean":
        return (
            f'<div class="flex items-center gap-2">'
            f'<input type="hidden" name="{{safe_name}}" value="false">'
            f'<input type="checkbox" name="{{safe_name}}" value="true" {{_CHECKBOX_CLS}}>'
            f'<label {{_LABEL_CLS}} style="margin-bottom:0">{{human_label}}{{required_mark}}</label>'
            f'</div>'
        )
    if expected == "integer":
        type_attr = 'type="number" step="1"'
    elif expected == "number":
        type_attr = 'type="number" step="any"'
    elif field_type.lower() in {{"date", "datetime", "timestamp"}}:
        type_attr = 'type="date"'
    elif _ui_field_semantic(field_name, field_type) == "email":
        type_attr = 'type="email"'
    elif _ui_field_semantic(field_name, field_type) == "phone":
        type_attr = 'type="tel"'
    elif _ui_field_semantic(field_name, field_type) == "url":
        type_attr = 'type="url"'
    else:
        type_attr = 'type="text"'
    placeholder = f'placeholder="{{human_label}}"'
    if field_type.lower() in {{"list", "dict", "json", "jsonb"}} or expected in {{"array", "object"}}:
        return (
            f'<div class="space-y-1">'
            f'<label {{_LABEL_CLS}}>{{human_label}}{{required_mark}}</label>'
            f'<textarea name="{{safe_name}}" rows="3" aria-describedby="{{helper_id}}"{{required_attr}} {{_INPUT_CLS}} '
            f'placeholder="{{html.escape("[] for lists, {{}} for objects", quote=True)}}"></textarea>'
            f'<p id="{{helper_id}}" class="text-xs text-gray-400">{{helper}} JSON value</p>'
            f'</div>'
        )
    return (
        f'<div class="space-y-1">'
        f'<label {{_LABEL_CLS}}>{{human_label}}{{required_mark}}</label>'
        f'<input name="{{safe_name}}" {{type_attr}} {{placeholder}} aria-describedby="{{helper_id}}"{{required_attr}} {{_INPUT_CLS}}>'
        f'<p id="{{helper_id}}" class="text-xs text-gray-400">{{helper}}</p>'
        f'</div>'
    )


def _ui_entity_location(entity_name: str) -> str:
    return f"/ui/entities/{{quote(entity_name, safe='')}}"


def _ui_record_display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value)
    return str(value)


def _ui_record_editor_input_html(
    field: Dict[str, Any], record: Dict[str, Any], form_id: str, entity_name: str = ""
) -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    safe_form_id = html.escape(form_id, quote=True)
    expected = _json_schema_type(str(field.get("type", "any")))
    value = record.get(field_name)

    # Foreign key → dropdown showing related entity records
    rel = _field_relationship(entity_name, field_name) if entity_name else None
    if rel:
        target = rel["target_table"]
        opts = _fk_select_options(target, current_value=str(value or ""), form_id=form_id)
        return f'<select form="{{safe_form_id}}" name="{{safe_name}}">{{opts}}</select>'

    if expected == "boolean":
        checked = " checked" if value is True else ""
        return (
            f'<input form="{{safe_form_id}}" type="hidden" name="{{safe_name}}" value="false">'
            f'<input form="{{safe_form_id}}" type="checkbox" name="{{safe_name}}" value="true"{{checked}}>'
        )
    if expected == "integer":
        attributes = 'type="number" step="1"'
    elif expected == "number":
        attributes = 'type="number" step="any"'
    elif field.get("type", "").lower() in {{"date", "datetime", "timestamp"}}:
        attributes = 'type="date"'
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "email":
        attributes = 'type="email"'
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "phone":
        attributes = 'type="tel"'
    elif _ui_field_semantic(field_name, str(field.get("type", ""))) == "url":
        attributes = 'type="url"'
    else:
        attributes = 'type="text"'
    safe_value = html.escape(_ui_record_display_value(value), quote=True)
    field_type = str(field.get("type", "")).lower()
    if field_type in {{"list", "dict", "json", "jsonb"}} or expected in {{"array", "object"}}:
        return f'<textarea form="{{safe_form_id}}" name="{{safe_name}}" rows="3">{{safe_value}}</textarea>'
    return f'<input form="{{safe_form_id}}" name="{{safe_name}}" value="{{safe_value}}" {{attributes}}>'


def _ui_query_value(query: Dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    return str(values[-1]) if values else ""


def _ui_records_query_form_html(entity_name: str, query: Dict[str, list[str]]) -> str:
    safe_entity_path = html.escape(quote(entity_name, safe=""), quote=True)
    fields = _field_specs(entity_name)
    filter_inputs = []
    for field in fields:
        field_name = str(field["name"])
        input_name = f"filter.{{field_name}}"
        safe_input_name = html.escape(input_name, quote=True)
        safe_label = html.escape(field_name)
        safe_value = html.escape(_ui_query_value(query, input_name), quote=True)
        filter_inputs.append(
            f'<label>{{safe_label}} <input type="text" name="{{safe_input_name}}" value="{{safe_value}}"></label>'
        )
    sort_options = ["", "id", "_revision"] + [
        str(field["name"]) for field in fields if str(field["name"]) not in {{"id", "_revision"}}
    ]
    selected_sort = _ui_query_value(query, "sort")
    sort_select = "".join(
        f'<option value="{{html.escape(option, quote=True)}}"{{" selected" if option == selected_sort else ""}}>'
        f'{{html.escape(option or "none")}}</option>'
        for option in sort_options
    )
    selected_order = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
    order_select = "".join(
        f'<option value="{{option}}"{{" selected" if option == selected_order else ""}}>{{option}}</option>'
        for option in ["asc", "desc"]
    )
    limit_value = html.escape(_ui_query_value(query, "limit"), quote=True)
    offset_value = html.escape(_ui_query_value(query, "offset"), quote=True)
    filters = "".join(filter_inputs) or "<span>No fields available.</span>"
    return (
        f'<form method="get" action="/ui/entities/{{safe_entity_path}}">'
        f'<fieldset><legend>Query records</legend>'
        f"{{filters}}"
        f'<label>Sort <select name="sort">{{sort_select}}</select></label>'
        f'<label>Order <select name="dir">{{order_select}}</select></label>'
        f'<label>Limit <input type="number" min="0" step="1" name="limit" value="{{limit_value}}"></label>'
        f'<label>Offset <input type="number" min="0" step="1" name="offset" value="{{offset_value}}"></label>'
        '<button type="submit">Apply</button> '
        f'<a href="/ui/entities/{{safe_entity_path}}">Reset</a>'
        '</fieldset></form>'
    )


def _ui_entity_query_path(
    entity_name: str,
    query: Dict[str, list[str]] | None = None,
    updates: Dict[str, Any] | None = None,
    drops: set[str] | None = None,
) -> str:
    safe_entity_path = quote(entity_name, safe="")
    params: Dict[str, list[str]] = {{}}
    drops = set(drops or set())
    for key, values in (query or {{}}).items():
        if key in drops or not values:
            continue
        params[str(key)] = [str(values[-1])]
    for key, value in (updates or {{}}).items():
        if value is None or str(value) == "":
            params.pop(str(key), None)
        else:
            params[str(key)] = [str(value)]
    pairs: list[str] = []
    for key in sorted(params):
        for value in params[key]:
            pairs.append(f"{{quote(str(key), safe='')}}={{quote(str(value), safe='')}}")
    suffix = "?" + "&".join(pairs) if pairs else ""
    return f"/ui/entities/{{safe_entity_path}}{{suffix}}"


def _ui_saved_views(entity_name: str, query: Dict[str, list[str]], fields: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    status_field = _status_field_name(fields)
    q = _ui_query_value(query, "q")
    sort_field = _ui_query_value(query, "sort")
    sort_dir = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
    active_filters = {{
        key: values[-1]
        for key, values in query.items()
        if key.startswith("filter.") and values and values[-1] not in ("", None)
    }}

    def active(expected: Dict[str, Any]) -> bool:
        expected_filters = {{
            key: value
            for key, value in expected.items()
            if key.startswith("filter.")
        }}
        expected_q = str(expected.get("q") or "")
        expected_sort = str(expected.get("sort") or "")
        expected_dir = str(expected.get("dir") or "asc").lower()
        return (
            q == expected_q
            and sort_field == expected_sort
            and sort_dir == expected_dir
            and active_filters == expected_filters
        )

    views = [
        {{
            "name": "All records",
            "description": "Complete table",
            "url": _ui_entity_query_path(entity_name),
            "active": active({{}}),
        }},
        {{
            "name": "Recently added",
            "description": "Newest first",
            "url": _ui_entity_query_path(entity_name, updates={{"sort": "id", "dir": "desc"}}),
            "active": active({{"sort": "id", "dir": "desc"}}),
        }},
    ]
    if status_field:
        status_key = f"filter.{{status_field}}"
        views.append({{
            "name": "Active",
            "description": f"{{status_field.replace('_', ' ').title()}} is active",
            "url": _ui_entity_query_path(entity_name, updates={{status_key: "active"}}),
            "active": active({{status_key: "active"}}),
        }})
        observed_values = sorted({{
            str(record.get(status_field))
            for record in list_records(entity_name)
            if record.get(status_field) not in (None, "")
        }})
        for value in observed_values[:4]:
            if value.lower() == "active":
                continue
            views.append({{
                "name": value.replace("_", " ").title(),
                "description": f"{{status_field.replace('_', ' ').title()}} filter",
                "url": _ui_entity_query_path(entity_name, updates={{status_key: value}}),
                "active": active({{status_key: value}}),
            }})
    return views


def _ui_active_filter_chips(entity_name: str, query: Dict[str, list[str]]) -> list[Dict[str, str]]:
    chips: list[Dict[str, str]] = []
    q = _ui_query_value(query, "q")
    if q:
        chips.append({{
            "label": _("search"),
            "value": q,
            "clear_url": _ui_entity_query_path(entity_name, query, drops={{"q", "page"}}),
        }})
    for key in sorted(query):
        if not key.startswith("filter."):
            continue
        value = _ui_query_value(query, key)
        if not value:
            continue
        chips.append({{
            "label": key.removeprefix("filter.").replace("_", " ").title(),
            "value": value,
            "clear_url": _ui_entity_query_path(entity_name, query, drops={{key, "page"}}),
        }})
    sort_field = _ui_query_value(query, "sort")
    if sort_field:
        sort_dir = (_ui_query_value(query, "dir") or _ui_query_value(query, "order") or "asc").lower()
        chips.append({{
            "label": "Sort",
            "value": f"{{sort_field}} {{sort_dir}}",
            "clear_url": _ui_entity_query_path(entity_name, query, drops={{"sort", "dir", "order", "page"}}),
        }})
    return chips


def _ui_create_form_html(entity_name: str, fields: list[Dict[str, Any]]) -> str:
    """Return the HTML for the create-record form fields (used by the Jinja2 template)."""
    _SKIP = {{"id", "_revision"}}
    parts = []
    for field in fields:
        if str(field.get("name", "")) in _SKIP:
            continue
        parts.append(_ui_field_input_html(field, entity_name))
    return '<div class="space-y-3">' + "".join(parts) + "</div>"


def _ui_records_table_html(entity_name: str, records: list[Dict[str, Any]] | None = None, sort_field: str = "", sort_dir: str = "asc", q: str = "", query: Dict[str, list[str]] | None = None) -> str:
    records = records if records is not None else list_records(entity_name)
    if not records:
        return f"<p>{{html.escape(_('no_records'))}}</p>"
    fields = _field_specs(entity_name)
    field_names = [str(f["name"]) for f in fields if str(f["name"]) not in {{"_revision"}}]
    # Show at most 6 columns to keep table readable; id always first
    display_cols = ["id"] + [c for c in field_names if c != "id"][:5]
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    delete_label = html.escape(_("delete"))
    delete_prompt = html.escape(_("delete") + " this record?", quote=True)
    header_cells = []
    for col in display_cols:
        label = html.escape((col[:-3].replace("_", " ").title() + " ID") if col.endswith("_id") else col.replace("_", " ").title())
        next_dir = "desc" if sort_field == col and sort_dir == "asc" else "asc"
        sort_icon = ""
        if sort_field == col:
            sort_icon = " ▼" if sort_dir == "desc" else " ▲"
        sort_url = html.escape(_ui_entity_query_path(entity_name, query, {{"sort": col, "dir": next_dir, "page": None}}), quote=True)
        header_cells.append(
            f'<th scope="col" class="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">'
            f'<a href="{{sort_url}}"'
            f' class="hover:text-gray-900 transition-colors">{{label}}{{sort_icon}}</a>'
            f'</th>'
        )
    header = "".join(header_cells)
    rows: list[str] = []
    for record in records:
        raw_record_id = str(record.get("id", ""))
        record_id = html.escape(quote(raw_record_id, safe=""), quote=True)
        revision = html.escape(str(record.get("_revision", "")), quote=True)
        cb_cell = (
            f'<td class="pl-3 pr-1 py-2.5 w-8">'
            f'<input type="checkbox" class="apg-row-cb w-4 h-4 rounded border-gray-300 text-apg-primary"'
            f' data-row-id="{{raw_record_id}}" data-rev="{{revision}}">'
            f'</td>'
        )
        cells = []
        for col in display_cols:
            val = html.escape(_ui_record_display_value(record.get(col)))
            if col == "id":
                cells.append(
                    f'<td scope="row" class="px-4 py-2.5">'
                    f'<a href="/ui/entities/{{safe_entity}}/{{record_id}}"'
                    f' class="text-xs font-mono text-apg-primary hover:underline truncate block max-w-24">{{val[:16]}}</a>'
                    f'</td>'
                )
            else:
                cells.append(f'<td class="px-4 py-2.5 text-sm text-gray-700 max-w-xs truncate">{{val}}</td>')
        cells.append(cb_cell)
        edit_hidden = "".join(
            f'<input type="hidden" name="{{html.escape(str(f["name"]), quote=True)}}" value="{{html.escape(str(record.get(str(f["name"]), "") or ""), quote=True)}}">'
            for f in fields if str(f.get("name")) not in {{"id", "_revision"}}
        )
        action = (
            f'<div class="flex items-center gap-3 justify-end opacity-0 group-hover/row:opacity-100 transition-opacity">'
            f'<form method="post" action="/ui/entities/{{safe_entity}}/records/{{record_id}}" class="inline">'
            f'{{_csrf_input()}}'
            f'<input type="hidden" name="expected_revision" value="{{revision}}">'
            f'{{edit_hidden}}'
            f'<button type="submit"'
            f' class="text-xs font-medium text-apg-primary hover:underline whitespace-nowrap">Edit</button>'
            f'</form>'
            f'<form method="post" action="/ui/entities/{{safe_entity}}/records/{{record_id}}/delete" class="inline">'
            f'{{_csrf_input()}}'
            f'<input type="hidden" name="expected_revision" value="{{revision}}">'
            f'<button type="submit" onclick="return apgConfirmSubmit(this.form, this.dataset.msg)" data-msg="{{delete_prompt}}"'
            f' class="text-xs text-red-400 hover:text-red-600 transition-colors">{{delete_label}}</button>'
            f'</form>'
            f'</div>'
        )
        rows.append(
            f'<tr class="hover:bg-gray-50 transition-colors group/row border-b border-gray-50 last:border-0">'
            f'{{"".join(cells)}}'
            f'<td class="px-4 py-2.5 text-right">{{action}}</td>'
            f'</tr>'
        )
    bulk_bar = (
        f'<div id="apg-bulk-bar" data-entity="{{safe_entity}}"'
        f' class="hidden fixed bottom-20 left-1/2 -translate-x-1/2 z-50'
        f' bg-gray-900 text-white rounded-2xl shadow-2xl px-5 py-3 flex items-center gap-3 text-sm">'
        f'<span id="apg-bulk-cnt" class="font-semibold tabular-nums"></span>'
        f'<button onclick="apgBulkDelete()"'
        f' class="px-3 py-1.5 bg-red-500 hover:bg-red-600 text-white text-xs font-medium rounded-lg transition-colors">{{delete_label}}</button>'
        f'<a id="apg-csv-link" href="/entities/{{safe_entity}}/records.csv"'
        f' class="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-lg transition-colors">Export CSV</a>'
        f'<button onclick="apgBulkClear()" class="ml-1 text-gray-400 hover:text-white leading-none text-base">✕</button>'
        f'</div>'
    )
    bulk_js = (
        f'<script{{_script_nonce_attr()}}>'
        '(function(){{'
        'function upd(){{'
        'var cc=document.querySelectorAll(".apg-row-cb:checked");'
        'var bar=document.getElementById("apg-bulk-bar");'
        'if(!bar)return;'
        'var cnt=document.getElementById("apg-bulk-cnt");'
        'if(cc.length>0){{bar.classList.remove("hidden");cnt.textContent=cc.length+" selected";}}else{{bar.classList.add("hidden");}}'
        '}}'
        'window.apgBulkClear=function(){{'
        'document.querySelectorAll(".apg-row-cb").forEach(function(c){{c.checked=false;}});'
        'upd();'
        '}};'
        'window.apgBulkDelete=function(){{'
        'var cc=document.querySelectorAll(".apg-row-cb:checked");'
        'if(!cc.length)return;'
        'apgConfirm("Delete "+cc.length+" record(s)? This cannot be undone.",function(){{'
        'var ids=Array.from(cc).map(function(c){{return c.dataset.rowId;}}).join(",");'
        'var entity=document.getElementById("apg-bulk-bar").dataset.entity;'
        'var csrf=document.querySelector("input[name=apg_csrf_token]");'
        'var token=csrf?csrf.value:"";'
        'fetch("/ui/entities/"+entity+"/records/bulk_delete",{{method:"POST",headers:{{"Content-Type":"application/x-www-form-urlencoded","X-APG-CSRF-Token":token}},body:"ids="+encodeURIComponent(ids)+"&apg_csrf_token="+encodeURIComponent(token)}})'
        '.then(function(r){{if(r.redirected||r.ok)window.location.reload();}});'
        '}});'
        '}};'
        'document.addEventListener("change",function(e){{if(e.target.classList.contains("apg-row-cb"))upd();}});'
        'document.addEventListener("click",function(e){{'
        'var allCb=e.target.closest(".apg-select-all");'
        'if(allCb){{document.querySelectorAll(".apg-row-cb").forEach(function(c){{c.checked=allCb.checked;}});upd();}}'
        '}});'
        '}})()'
        '</script>'
    )
    return (
        bulk_bar
        + f'<div class="apg-table-wrap shadow-sm overflow-hidden">'
        + f'<div class="overflow-x-auto">'
        + f'<table class="w-full">'
        + f'<caption class="apg-sr-only">{{html.escape(entity_name)}} records</caption>'
        + f'<thead class="bg-gray-50 border-b border-gray-100">'
        + f'<tr>'
        + f'{{header}}<th scope="col" class="pl-3 pr-1 py-2.5 w-8"><input type="checkbox" class="apg-select-all w-4 h-4 rounded border-gray-300" aria-label="Select all {{html.escape(entity_name)}} records"></th>'
        + f'<th scope="col" class="px-4 py-2.5 w-28">Actions</th></tr>'
        + f'</thead>'
        + f'<tbody>{{"".join(rows)}}</tbody>'
        + f'</table>'
        + f'</div>'
        + f'</div>'
        + bulk_js
    )


def _ui_entity_html(entity_name: str, notice: str = "", query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Unknown entity", f"<h1>Unknown entity: {{html.escape(entity_name)}}</h1>")
    query = query or {{}}
    safe_entity = html.escape(entity_name, quote=True)
    fields = _field_specs(entity_name) or [{{"name": "value", "type": "string", "required": True}}]

    # Full-text search: filter records where any string field contains q
    q = query.get("q", [""])[0].strip() if "q" in query else ""
    sort_field = query.get("sort", [""])[0].strip()
    sort_dir = (query.get("dir") or query.get("order") or ["asc"])[0].strip().lower()
    if sort_dir not in ("asc", "desc"):
        sort_dir = "asc"
    # Pagination
    try:
        page = max(1, int(query.get("page", ["1"])[0]))
    except (ValueError, TypeError):
        page = 1
    try:
        per = max(5, min(200, int(query.get("per", ["50"])[0])))
    except (ValueError, TypeError):
        per = 50

    # Build query for sort/pagination and field filters
    base_query: Dict[str, list[str]] = {{}}
    if sort_field:
        base_query["sort"] = [sort_field]
        base_query["order"] = [sort_dir]
    for _k, _v in query.items():
        if _k.startswith("filter."):
            base_query[_k] = _v
    query_result = query_records(entity_name, base_query)
    all_records = query_result["records"]

    # Full-text search filter
    if q:
        q_low = q.lower()
        filtered = [
            r for r in all_records
            if any(q_low in str(v).lower() for v in r.values() if v is not None)
        ]
    else:
        filtered = all_records

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + per - 1) // per)
    page = min(page, total_pages)
    offset = (page - 1) * per
    paginated = filtered[offset:offset + per]

    # Detect kanban-eligible status field
    status_field_names = {{"status", "state", "stage", "phase"}}
    has_kanban = any(str(f.get("name", "")).lower() in status_field_names for f in fields)

    records_table = _ui_records_table_html(entity_name, paginated, sort_field=sort_field, sort_dir=sort_dir, q=q, query=query)
    visible_start = offset + 1 if total_filtered else 0
    visible_end = min(offset + len(paginated), total_filtered)
    column_controls = [
        {{
            "name": str(field["name"]),
            "label": _humanize_label(str(field["name"])),
            "sort_url": _ui_entity_query_path(entity_name, query, {{"sort": str(field["name"]), "dir": "asc", "page": None}}),
            "active": str(field["name"]) == sort_field,
        }}
        for field in fields
        if str(field.get("name", "")) != "_revision"
    ][:8]
    list_intelligence = {{
        "share_url": _ui_entity_query_path(entity_name, query),
        "density_key": f"apg:list-density:{{entity_name}}",
        "column_key": f"apg:list-columns:{{entity_name}}",
        "visible_window": f"{{visible_start}}-{{visible_end}}",
        "total": total_filtered,
        "page_size": per,
        "filtered": total_filtered != query_result["total"] or bool(q),
        "column_controls": column_controls,
    }}
    sibling_source = all_records[0] if all_records else {{}}
    smart_defaults = []
    for field in fields:
        field_name = str(field.get("name", ""))
        if field_name in {{"id", "_revision"}} or field_name.endswith("_id"):
            continue
        value = sibling_source.get(field_name, "")
        if value in (None, "", []):
            continue
        smart_defaults.append({{
            "field": field_name,
            "label": _humanize_label(field_name),
            "value": html.escape(_ui_record_display_value(value)[:72], quote=True),
        }})
        if len(smart_defaults) >= 3:
            break
    required_fields = [
        _humanize_label(str(field.get("name", "")))
        for field in fields
        if bool(field.get("required")) and str(field.get("name", "")) not in {{"id", "_revision"}}
    ]
    validation_fields = [
        _humanize_label(str(field.get("name", "")))
        for field in fields
        if str(field.get("name", "")) not in {{"id", "_revision"}}
    ][:5]
    form_intelligence = {{
        "draft_key": f"apg:create-draft:{{entity_name}}",
        "undo_seconds": 5,
        "smart_defaults": smart_defaults,
        "required_fields": required_fields,
        "validation_fields": validation_fields,
        "dependency_edges": [
            {{"from": item["label"], "to": "Submit readiness"}}
            for item in smart_defaults[:3]
        ],
    }}
    pagination_pages = [
        {{"number": p, "url": _ui_entity_query_path(entity_name, query, {{"page": p, "per": per}})}}
        for p in range(1, total_pages + 1)
        if p >= page - 2 and p <= page + 2
    ]
    per_page_options = [
        {{"value": n, "url": _ui_entity_query_path(entity_name, query, {{"page": 1, "per": n}})}}
        for n in [10, 25, 50, 100, 200]
    ]

    # Prefer Jinja2 template for rich UI; fall back to f-string builder for zero-dep mode
    create_inputs = _ui_create_form_html(entity_name, fields)
    query_form = _ui_records_query_form_html(entity_name, query)
    tmpl_body = _render_template(
        "entity_list.html.j2",
        entity_name=html.escape(entity_name),
        entity_type=html.escape(entity.get("type", "entity")),
        safe_entity=safe_entity,
        fields=fields,
        records=paginated,
        total=query_result["total"],
        count=total_filtered,
        records_table=records_table,
        list_intelligence=list_intelligence,
        form_intelligence=form_intelligence,
        create_inputs=create_inputs,
        notice=html.escape(notice) if notice else "",
        query=query,
        saved_views=_ui_saved_views(entity_name, query, fields),
        active_filters=_ui_active_filter_chips(entity_name, query),
        clear_filters_url=_ui_entity_query_path(entity_name),
        developer_api_url=f"/entities/{{quote(entity_name, safe='')}}/records",
        csv_url=f"/entities/{{quote(entity_name, safe='')}}/records.csv",
        has_kanban=has_kanban,
        q=html.escape(q) if q else "",
        sort_field=sort_field,
        sort_dir=sort_dir,
        page=page,
        per=per,
        total_pages=total_pages,
        prev_page_url=_ui_entity_query_path(entity_name, query, {{"page": page - 1, "per": per}}) if page > 1 else "",
        next_page_url=_ui_entity_query_path(entity_name, query, {{"page": page + 1, "per": per}}) if page < total_pages else "",
        first_page_url=_ui_entity_query_path(entity_name, query, {{"page": 1, "per": per}}),
        last_page_url=_ui_entity_query_path(entity_name, query, {{"page": total_pages, "per": per}}),
        pagination_pages=pagination_pages,
        per_page_options=per_page_options,
        records_json=html.escape(json.dumps(paginated, indent=2, sort_keys=True)),
        query_form=query_form,
    )
    if tmpl_body is not None:
        return 200, _html_page(entity_name, tmpl_body)

    # Fallback: original f-string builder
    inputs = _ui_create_form_html(entity_name, fields)
    query_form = _ui_records_query_form_html(entity_name, query)
    result_summary = f'<p>Showing {{query_result["count"]}} of {{query_result["total"]}} matching records.</p>'
    notice_html = f'<section role="alert"><strong>{{html.escape(notice)}}</strong></section>' if notice else ""
    body = (
        f'<nav><a href="/ui">Application</a> | '
        f'<a href="/entities/{{safe_entity}}/records">Record JSON</a></nav>'
        f"<h1>{{html.escape(entity_name)}}</h1>"
        f"<p><code>{{html.escape(entity.get('type', 'entity'))}}</code></p>"
        f"{{notice_html}}"
        f'<form method="post" action="/ui/entities/{{safe_entity}}/records">'
        f"{{_csrf_input()}}"
        f"{{inputs}}"
        '<button type="submit">Create record</button>'
        "</form>"
        "<h2>Records</h2>"
        f"{{query_form}}"
        f"{{result_summary}}"
        f"{{records_table}}"
        "<details><summary>Record JSON</summary>"
        f"<pre>{{records_json}}</pre>"
        "</details>"
    )
    return 200, _html_page(entity_name, body)


def _ui_entity_analytics_html(entity_name: str) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Unknown entity", f"<h1>Unknown entity: {{html.escape(entity_name)}}</h1>")
    fields = _field_specs(entity_name)
    records = list_records(entity_name)
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    import datetime as _dt

    date_candidates = [
        str(field.get("name", ""))
        for field in fields
        if str(field.get("type", "")).lower() in {{"date", "datetime", "timestamp"}}
    ]
    date_candidates.extend(["created_at", "created_on", "created", "updated_at", "updated_on", "date", "timestamp"])

    def parse_record_date(value: Any) -> _dt.date | None:
        if value in (None, ""):
            return None
        text = str(value).strip().replace("Z", "+00:00")
        try:
            return _dt.datetime.fromisoformat(text).date()
        except ValueError:
            try:
                return _dt.date.fromisoformat(text[:10])
            except ValueError:
                return None

    dated_records: list[tuple[_dt.date, Dict[str, Any]]] = []
    date_field = ""
    for candidate in date_candidates:
        values = [
            (parsed, record)
            for record in records
            for parsed in [parse_record_date(record.get(candidate))]
            if parsed is not None
        ]
        if values:
            date_field = candidate
            dated_records = values
            break

    line_data = []
    recent_count = 0
    date_range = ""
    if dated_records:
        end_date = max(day for day, _record in dated_records)
        start_date = end_date - _dt.timedelta(days=29)
        counts_by_day: Dict[_dt.date, int] = {{}}
        for day, _record in dated_records:
            if day < start_date or day > end_date:
                continue
            counts_by_day[day] = counts_by_day.get(day, 0) + 1
        for index in range(30):
            day = start_date + _dt.timedelta(days=index)
            line_data.append({{"x": day.isoformat(), "y": counts_by_day.get(day, 0)}})
        recent_start = end_date - _dt.timedelta(days=6)
        recent_count = sum(1 for day, _record in dated_records if day >= recent_start)
        date_range = f"{{start_date.isoformat()}} to {{end_date.isoformat()}}"
    line_chart = {{
        "id": f"analytics-line-{{_css_name(entity_name)}}",
        "spec_json": _chart_json({{
            "type": "line",
            "title": f"{{entity_name}} records over time",
            "data": line_data,
            "compare": [
                {{"x": point["x"], "y": max(0, int(point["y"]) - 1)}}
                for point in line_data
            ],
            "forecast": [
                {{
                    "x": (_dt.date.fromisoformat(line_data[-1]["x"]) + _dt.timedelta(days=index)).isoformat() if line_data else str(index),
                    "y": round((sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:]))) if line_data else 0, 2),
                    "low": 0,
                    "high": round(((sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:]))) if line_data else 0) + 1, 2),
                }}
                for index in range(1, 8)
            ],
            "annotations": [
                {{"x": point["x"], "label": "Peak", "value": point["y"]}}
                for point in sorted(line_data, key=lambda item: item["y"], reverse=True)[:1]
                if point["y"]
            ],
            "empty": "No date field data yet",
        }}),
    }}
    status_field = _status_field_name(fields)
    counts: Dict[str, int] = {{}}
    if status_field:
        for record in records:
            key = str(record.get(status_field) or "Unspecified")
            counts[key] = counts.get(key, 0) + 1
    status_rows = []
    total_status = sum(counts.values())
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        status_rows.append({{
            "label": key,
            "count": value,
            "percent": round((value / total_status) * 100, 1) if total_status else 0,
            "url": _ui_entity_query_path(entity_name, updates={{f"filter.{{status_field}}": key}}) if status_field else _ui_entity_query_path(entity_name),
        }})
    status_chart = {{
        "id": f"analytics-status-{{_css_name(entity_name)}}",
        "spec_json": _chart_json({{
            "type": "donut",
            "title": f"{{entity_name}} status distribution",
            "data": [{{"label": key, "value": value}} for key, value in sorted(counts.items())],
            "empty": "No status data yet",
        }}),
    }}
    numeric_stats = []
    for field in fields:
        field_name = str(field.get("name", ""))
        if _json_schema_type(str(field.get("type", ""))) not in {{"integer", "number"}}:
            continue
        values = []
        for record in records:
            value = record.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, str):
                try:
                    values.append(float(value.replace(",", "")))
                except ValueError:
                    continue
        if values:
            numeric_stats.append({{
                "field": field_name,
                "min": round(min(values), 2),
                "avg": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "count": len(values),
            }})
    top_status = status_rows[0] if status_rows else None
    metrics = [
        {{"label": "Records", "value": len(records), "hint": "Total rows", "url": _ui_entity_query_path(entity_name)}},
        {{"label": "Recent", "value": recent_count, "hint": "Last 7 days" if date_field else "Needs date field", "url": _ui_entity_query_path(entity_name)}},
        {{"label": "Statuses", "value": len(status_rows), "hint": status_field or "No status field", "url": _ui_entity_query_path(entity_name)}},
        {{"label": "Measures", "value": len(numeric_stats), "hint": "Numeric fields", "url": _ui_entity_query_path(entity_name)}},
    ]
    insights = []
    if top_status:
        insights.append({{
            "title": "Largest segment",
            "body": f"{{top_status['label']}} has {{top_status['count']}} record{{'s' if top_status['count'] != 1 else ''}}.",
            "url": top_status["url"],
            "action": f"View {{top_status['label']}} records",
        }})
    if date_field and date_range:
        insights.append({{
            "title": "Trend window",
            "body": f"Using {{date_field}} across {{date_range}}.",
            "url": _ui_entity_query_path(entity_name),
            "action": "Open table",
        }})
    if not records:
        insights.append({{
            "title": _("no_records"),
            "body": "Create records before reading analytics.",
            "url": _ui_entity_query_path(entity_name),
            "action": f"Create {{entity_name}}",
        }})
    peak_point = max(line_data, key=lambda item: item["y"], default={{"x": "", "y": 0}})
    recent_average = round(sum(float(point["y"]) for point in line_data[-7:]) / max(1, len(line_data[-7:])), 2) if line_data else 0
    analytics_decisions = [
        {{
            "label": "Annotation Pin",
            "value": str(peak_point["x"] or "No peak yet"),
            "hint": f"Highest daily volume: {{peak_point['y']}}",
            "url": _ui_entity_query_path(entity_name),
        }},
        {{
            "label": "Comparative Overlay",
            "value": "Current vs prior window",
            "hint": f"Recent average {{recent_average}} record(s)/day",
            "url": _ui_entity_query_path(entity_name),
        }},
        {{
            "label": "Forecast Band",
            "value": "Next 7 days",
            "hint": f"Expected {{recent_average}} to {{round(recent_average + 1, 2)}} per day",
            "url": _ui_entity_query_path(entity_name),
        }},
    ]
    tmpl_body = _render_template(
        "entity_analytics.html.j2",
        entity_name=entity_name,
        safe_entity=safe_entity,
        total=len(records),
        metrics=metrics,
        status_field=status_field or "",
        status_rows=status_rows,
        date_field=date_field,
        date_range=date_range,
        insights=insights,
        line_chart=line_chart,
        status_chart=status_chart,
        analytics_decisions=analytics_decisions,
        numeric_stats=numeric_stats,
    )
    return 200, _html_page(f"{{entity_name}} Analytics", tmpl_body if tmpl_body is not None else _jinja_required_page(f"{{entity_name}} Analytics"))


def _ui_error_message(response: Dict[str, Any]) -> str:
    errors = response.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)
    if response.get("error") == "revision_conflict":
        return (
            "Revision conflict: record has revision "
            f"{{response.get('current_revision')}} but form submitted revision {{response.get('expected_revision')}}"
        )
    if "message" in response:
        return str(response["message"])
    if "error" in response:
        return str(response["error"])
    return "The submitted form could not be applied."


def _ui_error_payload(path: str, response: Dict[str, Any]) -> str:
    parts = [part for part in path.split("/") if part]
    message = _ui_error_message(response)
    if len(parts) >= 3 and parts[0] == "ui" and parts[1] == "entities":
        _status, body = _ui_entity_html(parts[2], notice=message)
        return body
    details = html.escape(json.dumps(response, indent=2, sort_keys=True))
    return _html_page("Form error", f"<h1>Form error</h1><p>{{html.escape(message)}}</p><pre>{{details}}</pre>")


def _extract_accumulated(form: dict) -> dict:
    """Pull __acc_FIELD hidden fields from a step POST into an accumulated dict."""
    return {{
        k[6:]: v  # strip '__acc_' prefix
        for k, v in form.items()
        if k.startswith("__acc_")
    }}


def _ui_workflow_step_post(
    entity_name: str, workflow_id: str, step_index: int, form: dict
) -> tuple[int, str]:
    """Handle POST to a workflow step: accumulate data and advance."""
    accumulated = _extract_accumulated(form)
    step_fields_data = {{k: v for k, v in form.items() if not k.startswith("__acc_") and k != "expected_revision"}}
    accumulated.update(step_fields_data)

    entity_workflows = APP_WORKFLOWS.get(entity_name, [])
    wf = next((w for w in entity_workflows if w["id"] == workflow_id), None)
    if wf is None:
        return 404, _html_page("Workflow not found", "<h1>Workflow not found</h1>")

    next_step = step_index + 1
    _publish_live_event(
        f"workflow:run:{{workflow_id}}",
        "workflow",
        {{"workflow": workflow_id, "entity": entity_name, "step_index": step_index, "next_step": next_step}},
    )
    return _ui_workflow_wizard_html(entity_name, workflow_id, next_step, accumulated)


def _ui_field_view_fragment(entity_name: str, record_id: str, field: Dict[str, Any], record: Dict[str, Any]) -> str:
    """Return the view-mode div for one field (used after save or cancel)."""
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)
    field_name = str(field.get("name", ""))
    fld_id = f"fld-{{safe_entity}}-{{safe_record_id}}-{{field_name}}"
    field_val = record.get(field_name, "")
    if field_val is None or field_val == "" or str(field_val) == "None":
        display = '<span class="text-gray-300 italic text-xs">—</span>'
    elif str(field_val).lower() == "true":
        display = '<span class="inline-flex items-center gap-1 text-green-600"><span class="text-xs">✓</span> Yes</span>'
    elif str(field_val).lower() == "false":
        display = '<span class="inline-flex items-center gap-1 text-gray-400"><span class="text-xs">✕</span> No</span>'
    else:
        display = html.escape(str(field_val)[:200])
    label = html.escape((field_name[:-3].replace("_", " ").title() + " ID") if field_name.endswith("_id") else field_name.replace("_", " ").title())
    edit_url = f"/ui/entities/{{safe_entity}}/{{safe_record_id}}/fields/{{html.escape(field_name)}}/edit"
    return (
        f'<div id="{{fld_id}}" class="py-3 border-b border-gray-50 last:border-0 group/field">'
        f'<dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">{{label}}</dt>'
        f'<dd class="flex items-center justify-between gap-2 min-h-6">'
        f'<span class="text-sm text-gray-900 break-words">{{display}}</span>'
        f'<button hx-get="{{edit_url}}" hx-target="#{{fld_id}}" hx-swap="outerHTML"'
        f' class="opacity-0 group-hover/field:opacity-100 flex-shrink-0 p-1 text-gray-300 hover:text-apg-primary rounded transition-all"'
        f' title="Edit {{html.escape(field_name)}}">'
        f'<svg class="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">'
        f'<path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zm-2.207 2.207L3 14.172V17h2.828l8.38-8.379-2.83-2.828z"/>'
        f'</svg></button>'
        f'</dd></div>'
    )


def _ui_record_detail_html(entity_name: str, record_id: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, _html_page("Not found", f"<h1>Record not found</h1><p>{{html.escape(entity_name)}}/{{html.escape(record_id)}}</p>")
    record = response.get("record", response)
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Not found", f"<h1>Unknown entity: {{html.escape(entity_name)}}</h1>")
    fields = _field_specs(entity_name) or []
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)

    # Pick a good display title (preferred name field, first string value, or id prefix)
    preferred_title_names = ("legal_name", "full_name", "name", "title", "subject", "number", "code")
    title_field = next(
        (
            f for preferred in preferred_title_names
            for f in fields
            if str(f.get("name", "")).lower() == preferred
        ),
        None,
    )
    if title_field is None:
        title_field = next(
            (f for f in fields if str(f.get("type", "")).lower() in {{"str", "string", "text", "email", "varchar"}} and str(f.get("name")) not in {{"id", "_revision"}}),
            None,
        )
    title = str(record.get(title_field["name"], record_id) if title_field else record_id)[:80]

    # Status badge value
    status_field = next(
        (f for f in fields if str(f.get("name", "")).lower() in {{"status", "state", "stage", "phase"}}),
        None,
    )
    status_val = str(record.get(status_field["name"], "")) if status_field else ""

    # Related lists: find entities with FK fields pointing to this entity
    related_lists: list[Dict[str, Any]] = []
    for ent in sorted(ENTITY_NAMES):
        if ent == entity_name:
            continue
        ent_fields = _field_specs(ent) or []
        fk_field = next(
            (f for f in ent_fields if str(f.get("name", "")).endswith("_id") and str(f.get("name", ""))[:-3] == entity_name.lower()),
            None,
        )
        if fk_field is None:
            # Try FK by entity name convention: field name == entity_name + "_id"
            fk_candidates = [f for f in ent_fields if str(f.get("name", "")).lower().replace("_id", "") == entity_name.lower()]
            fk_field = fk_candidates[0] if fk_candidates else None
        if fk_field:
            fk_name = str(fk_field["name"])
            rel_result = query_records(ent, {{f"filter.{{fk_name}}": [record_id]}})
            rel_records = rel_result.get("records", [])
            rel_cols = ["id"] + [str(f["name"]) for f in ent_fields if str(f.get("name")) not in {{"id", "_revision", fk_name}}][:4]
            related_lists.append({{
                "entity": ent,
                "fk_field": fk_name,
                "records": rel_records,
                "count": len(rel_records),
                "cols": rel_cols,
                "list_url": _ui_entity_query_path(ent, updates={{f"filter.{{fk_name}}": record_id}}),
                "create_url": _ui_entity_query_path(ent),
            }})

    has_kanban = any(str(f.get("name", "")).lower() in {{"status", "state", "stage", "phase"}} for f in fields)
    revision = html.escape(str(record.get("_revision", "")))
    entity_records = list_records(entity_name)
    record_ids = [str(item.get("id", "")) for item in entity_records if item.get("id", "") not in (None, "")]
    try:
        current_index = record_ids.index(str(record_id))
    except ValueError:
        current_index = -1
    prev_record_url = ""
    next_record_url = ""
    if current_index > 0:
        prev_record_url = f"/ui/entities/{{safe_entity}}/{{quote(record_ids[current_index - 1], safe='')}}"
    if current_index >= 0 and current_index < len(record_ids) - 1:
        next_record_url = f"/ui/entities/{{safe_entity}}/{{quote(record_ids[current_index + 1], safe='')}}"
    related_count = sum(int(rel.get("count", 0)) for rel in related_lists)
    record_url = f"/ui/entities/{{safe_entity}}/{{safe_record_id}}"

    display_fields = [f for f in fields if str(f.get("name")) != "_revision"]
    field_semantics = {{
        str(f.get("name", "")): _ui_field_semantic(str(f.get("name", "")), str(f.get("type", "")))
        for f in display_fields
    }}
    activity_events = _get_activity(entity_name, record_id)
    diff_fields: list[Dict[str, str]] = []
    for field in display_fields:
        field_name = str(field.get("name", ""))
        if field_name == "id":
            continue
        value = record.get(field_name, "")
        if value in (None, "", []):
            continue
        diff_fields.append({{
            "name": field_name.replace("_", " ").title(),
            "value": html.escape(str(value)[:72]),
            "state": "current",
        }})
        if len(diff_fields) >= 4:
            break
    sibling_fields = [
        {{
            "name": str(field.get("name", "")).replace("_", " ").title(),
            "value": html.escape(str(record.get(str(field.get("name", "")), ""))[:48]),
        }}
        for field in display_fields
        if str(field.get("name", "")) not in {{"id", "_revision"}} and not str(field.get("name", "")).endswith("_id")
    ][:3]
    related_graph = [
        {{
            "entity": html.escape(str(rel.get("entity", ""))),
            "count": int(rel.get("count", 0)),
            "field": html.escape(str(rel.get("fk_field", ""))),
            "url": str(rel.get("list_url", "")),
        }}
        for rel in related_lists[:4]
    ]
    detail_intelligence = {{
        "diff_fields": diff_fields,
        "related_graph": related_graph,
        "sibling_fields": sibling_fields,
        "activity_count": len(activity_events),
        "create_sibling_url": _ui_entity_query_path(entity_name),
    }}
    tmpl_body = _render_template(
        "record_detail.html.j2",
        entity_name=html.escape(entity_name),
        entity_type=html.escape(entity.get("type", "entity")),
        safe_entity=safe_entity,
        safe_record_id=safe_record_id,
        record=record,
        fields=display_fields,
        field_semantics=field_semantics,
        title=html.escape(title),
        status_val=html.escape(status_val),
        revision=revision,
        related_lists=related_lists,
        related_count=related_count,
        prev_record_url=prev_record_url,
        next_record_url=next_record_url,
        record_url=record_url,
        has_kanban=has_kanban,
        activity_events=activity_events,
        detail_intelligence=detail_intelligence,
    )
    if tmpl_body is not None:
        return 200, _html_page(title or entity_name, tmpl_body)
    return 200, _html_page(entity_name, f"<h1>{{html.escape(title)}}</h1><pre>{{html.escape(json.dumps(record, indent=2))}}</pre>")


def _ui_field_edit_html(entity_name: str, record_id: str, field_name: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, "{{}}"
    record = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, "{{}}"
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    safe_record_id = html.escape(quote(record_id, safe=""), quote=True)
    safe_field_name = html.escape(field_name)
    fld_id = f"fld-{{safe_entity}}-{{safe_record_id}}-{{safe_field_name}}"
    current_val = html.escape(str(record.get(field_name, "") or ""), quote=True)
    label = html.escape((field_name[:-3].replace("_", " ").title() + " ID") if field_name.endswith("_id") else field_name.replace("_", " ").title())
    patch_url = f"/ui/entities/{{safe_entity}}/{{safe_record_id}}/fields/{{safe_field_name}}/patch"
    cancel_url = f"/ui/entities/{{safe_entity}}/{{safe_record_id}}/fields/{{safe_field_name}}/view"
    field_type = str(field.get("type", "string"))
    field_semantic = _ui_field_semantic(field_name, field_type)
    field_expected = _json_schema_type(field_type)
    if field_type.lower() in {{"text", "markdown", "list", "dict", "json", "jsonb"}} or field_expected in {{"array", "object"}}:
        input_html = (
            f'<textarea name="{{safe_field_name}}" rows="3"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary resize-none">'
            f'{{current_val}}</textarea>'
        )
    elif field_type == "boolean":
        checked = "checked" if str(record.get(field_name, "")).lower() == "true" else ""
        input_html = f'<input type="checkbox" name="{{safe_field_name}}" value="true" {{checked}} class="w-4 h-4 text-apg-primary rounded">'
    elif field_expected == "integer":
        input_html = (
            f'<input type="number" step="1" name="{{safe_field_name}}" value="{{current_val}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_expected == "number":
        input_html = (
            f'<input type="number" step="any" name="{{safe_field_name}}" value="{{current_val}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_type.lower() in {{"date", "datetime", "timestamp"}}:
        input_html = (
            f'<input type="date" name="{{safe_field_name}}" value="{{current_val[:10]}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_semantic == "email":
        input_type = "email"
        input_html = (
            f'<input type="{{input_type}}" name="{{safe_field_name}}" value="{{current_val}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    elif field_semantic == "phone":
        input_html = (
            f'<input type="tel" name="{{safe_field_name}}" value="{{current_val}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    else:
        input_html = (
            f'<input type="text" name="{{safe_field_name}}" value="{{current_val}}"'
            f' class="w-full border border-apg-primary rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-apg-primary">'
        )
    revision = html.escape(str(record.get("_revision", "")), quote=True)
    save_label = html.escape(_("save"))
    cancel_label = html.escape(_("cancel"))
    fragment = (
        f'<div id="{{fld_id}}" class="py-3 border-b border-gray-50 last:border-0">'
        f'<dt class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">{{label}}</dt>'
        f'<dd>'
        f'<form hx-post="{{patch_url}}" hx-target="#{{fld_id}}" hx-swap="outerHTML" class="flex flex-col gap-1.5">'
        f'{{_csrf_input()}}'
        f'<input type="hidden" name="expected_revision" value="{{revision}}">'
        f'{{input_html}}'
        f'<div class="flex gap-2">'
        f'<button type="submit" class="px-2.5 py-1 bg-apg-primary text-white text-xs font-medium rounded-lg hover:opacity-90">{{save_label}}</button>'
        f'<button type="button" hx-get="{{cancel_url}}" hx-target="#{{fld_id}}" hx-swap="outerHTML"'
        f' class="px-2.5 py-1 text-xs text-gray-500 hover:text-gray-700 border border-gray-200 rounded-lg">{{cancel_label}}</button>'
        f'</div>'
        f'</form>'
        f'</dd></div>'
    )
    return 200, fragment


def _ui_field_view_html(entity_name: str, record_id: str, field_name: str) -> tuple[int, str]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, "{{}}"
    record = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, "{{}}"
    return 200, _ui_field_view_fragment(entity_name, record_id, field, record)


def _ui_field_patch_post(entity_name: str, record_id: str, field_name: str, form: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    status, response = get_record(entity_name, record_id)
    if status != 200 or not isinstance(response, dict):
        return 404, {{"error": "record not found"}}
    current = response.get("record", response)
    fields = _field_specs(entity_name) or []
    field = next((f for f in fields if str(f.get("name")) == field_name), None)
    if field is None:
        return 404, {{"error": "field not found"}}
    new_val = form.get(field_name, "")
    field_type = str(field.get("type", "string"))
    if field_type == "boolean":
        new_val = "true" if new_val == "true" else "false"
    elif field_type == "integer":
        try:
            new_val = str(int(new_val))
        except (ValueError, TypeError):
            new_val = "0"
    updated = dict(current)
    updated[field_name] = new_val
    expected_revision_raw = form.get("expected_revision")
    try:
        expected_revision_int: int | None = int(expected_revision_raw) if expected_revision_raw is not None else None
    except (TypeError, ValueError):
        expected_revision_int = None
    save_status, save_result = update_record(entity_name, record_id, updated, expected_revision_int)
    if save_status not in (200, 201, 204):
        err_msg = html.escape(str(save_result.get("error") or save_result.get("message") or "Save failed"))
        fragment = (
            f'<div class="py-3 border-b border-gray-50">'
            f'<p class="text-xs text-red-500">{{err_msg}}</p>'
            f'</div>'
        )
        return save_status, {{"html": fragment}}
    _status2, refreshed_resp = get_record(entity_name, record_id)
    refreshed = refreshed_resp.get("record", refreshed_resp) if isinstance(refreshed_resp, dict) else {{}}
    rec = refreshed if refreshed else updated
    label = str(field.get("name", "")).replace("_", " ").title()
    return 200, {{"html": _ui_field_view_fragment(entity_name, record_id, field, rec), "hx_trigger": {{"apgToast": {{"msg": f"{{label}} saved", "type": "success"}}}}}}


def _ui_kanban_html(entity_name: str) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Not found", f"<h1>Unknown entity: {{html.escape(entity_name)}}</h1>")
    fields = _field_specs(entity_name) or []
    status_field_names = {{"status", "state", "stage", "phase"}}
    status_field = next((f for f in fields if str(f.get("name", "")).lower() in status_field_names), None)
    if status_field is None:
        return _ui_entity_html(entity_name)
    status_fname = str(status_field["name"])
    all_records = query_records(entity_name, {{}}).get("records", [])
    # Gather unique status values preserving insertion order
    seen: list[str] = []
    for r in all_records:
        v = str(r.get(status_fname, "") or "")
        if v and v not in seen:
            seen.append(v)
    if not seen:
        seen = ["active", "inactive"]
    wip_limit = max(3, (len(all_records) + max(1, len(seen)) - 1) // max(1, len(seen))) if all_records else 3
    columns = []
    for value in seen:
        column_records = [r for r in all_records if str(r.get(status_fname, "")) == value]
        columns.append({{
            "label": value,
            "records": column_records,
            "count": len(column_records),
            "wip_limit": wip_limit,
            "over_limit": len(column_records) > wip_limit,
            "list_url": _ui_entity_query_path(entity_name, updates={{f"filter.{{status_fname}}": value}}),
        }})
    swimlane_field = next(
        (
            str(f.get("name"))
            for f in fields
            if str(f.get("name", "")).lower() in {{"priority", "assignee", "owner", "team", "country", "tenant_id", "segment", "type"}}
            and str(f.get("name")) not in {{"id", "_revision", status_fname}}
        ),
        "",
    )
    swimlanes: list[Dict[str, Any]] = []
    if swimlane_field:
        lane_values = sorted({{
            str(record.get(swimlane_field) or "Unassigned")
            for record in all_records
        }})
        for lane in lane_values[:6]:
            lane_records = [record for record in all_records if str(record.get(swimlane_field) or "Unassigned") == lane]
            swimlanes.append({{
                "label": lane,
                "count": len(lane_records),
                "field": swimlane_field,
                "url": _ui_entity_query_path(entity_name, updates={{f"filter.{{swimlane_field}}": lane}}),
            }})
    cumulative = 0
    flow_rows = []
    for column in columns:
        cumulative += int(column["count"])
        flow_rows.append({{
            "label": column["label"],
            "count": column["count"],
            "cumulative": cumulative,
            "percent": round((cumulative / max(1, len(all_records))) * 100, 1),
            "over_limit": column["over_limit"],
        }})
    # Choose display field: first non-id, non-status string field
    display_field_obj = next(
        (f for f in fields if str(f.get("type", "")).lower() in {{"str", "string", "text", "email", "varchar"}} and str(f.get("name")) not in {{"id", "_revision", status_fname}}),
        None,
    )
    display_field = str(display_field_obj["name"]) if display_field_obj else "id"
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    tmpl_body = _render_template(
        "kanban_view.html.j2",
        entity_name=html.escape(entity_name),
        safe_entity=safe_entity,
        columns=columns,
        display_field=display_field,
        status_field=status_fname,
        status_options=seen,
        total_records=len(all_records),
        wip_limit=wip_limit,
        swimlane_field=swimlane_field,
        swimlanes=swimlanes,
        flow_rows=flow_rows,
        list_url=_ui_entity_query_path(entity_name),
        fields=fields,
    )
    if tmpl_body is not None:
        return 200, _html_page(f"{{entity_name}} — Kanban", tmpl_body)
    return _ui_entity_html(entity_name)


def _ui_debug_html(run_id: str | None = None) -> tuple[int, str]:
    runs = list_workflow_runs()
    cb_status = circuit_breaker_status()
    subs = dict(APG_EVENT_SUBSCRIPTIONS)
    def _badge(status: str) -> str:
        if status in {{"completed", "closed", "success"}}:
            return "apg-badge-success"
        if status in {{"failed", "open", "circuit_open"}}:
            return "apg-badge-danger"
        return "apg-badge-warning"

    selected_run = None
    if run_id:
        try:
            raw_run = get_workflow_run(run_id)
        except KeyError:
            raw_run = None
        if raw_run:
            journal = _get_journal(run_id)
            trace = [
                {{
                    "index": str(step.get("index", "")),
                    "step": str(step.get("step", "")),
                    "status": str(step.get("status", "")),
                    "notes": str(step.get("notes") or step.get("timeout_spec", "")),
                    "field_count": step.get("field_count", ""),
                    "duration_ms": step.get("duration_ms", ""),
                    "fields": ", ".join(str(item) for item in step.get("fields", [])) if isinstance(step.get("fields", []), list) else "",
                    "badge_class": _badge(str(step.get("status", ""))),
                }}
                for step in raw_run.get("trace", [])
                if isinstance(step, dict)
            ]
            cumulative_ms = 0
            replay_frames = []
            breakpoint_items = []
            for step in trace:
                duration_text = str(step.get("duration_ms", ""))
                duration_ms = int(duration_text) if duration_text.isdigit() else 0
                cumulative_ms += duration_ms
                field_count_text = str(step.get("field_count", ""))
                field_count = int(field_count_text) if field_count_text.isdigit() else 0
                replay_frames.append({{
                    "index": step.get("index", ""),
                    "step": step.get("step", ""),
                    "status": step.get("status", ""),
                    "badge_class": step.get("badge_class", "apg-badge-neutral"),
                    "duration_ms": duration_ms,
                    "cumulative_ms": cumulative_ms,
                    "fields": step.get("fields", ""),
                    "reason": "Replay checkpoint with field inputs" if field_count else "Replay checkpoint",
                }})
                status_text = str(step.get("status", ""))
                if status_text not in {{"completed", "success", ""}} or duration_ms >= 225 or field_count >= 3:
                    breakpoint_items.append({{
                        "index": step.get("index", ""),
                        "step": step.get("step", ""),
                        "reason": "Inspect status" if status_text not in {{"completed", "success", ""}} else ("Slowest step" if duration_ms >= 225 else "High field fan-in"),
                        "key": f"apg:debug-breakpoint:{{run_id}}:{{step.get('index', '')}}",
                    }})
            if not breakpoint_items and trace:
                breakpoint_items.append({{
                    "index": trace[-1].get("index", ""),
                    "step": trace[-1].get("step", ""),
                    "reason": "Final state checkpoint",
                    "key": f"apg:debug-breakpoint:{{run_id}}:{{trace[-1].get('index', '')}}",
                }})
            variable_items = []
            payload = raw_run.get("payload", {{}})
            record = raw_run.get("record", {{}})
            variable_sources = [
                ("Payload", payload if isinstance(payload, dict) else {{}}),
                ("Created record", record if isinstance(record, dict) else {{}}),
                ("Run", {{
                    "run_id": raw_run.get("id", run_id),
                    "workflow": raw_run.get("workflow", ""),
                    "workflow_id": raw_run.get("workflow_id", ""),
                    "entity": raw_run.get("entity", ""),
                    "status": raw_run.get("status", ""),
                    "event_id": raw_run.get("event_id", ""),
                }}),
            ]
            for source, values in variable_sources:
                for name, value in sorted(values.items(), key=lambda item: str(item[0]))[:8]:
                    variable_items.append({{
                        "source": source,
                        "name": str(name),
                        "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value),
                    }})
            debug_intelligence = {{
                "verdict": "Replay-ready" if trace else "No trace captured",
                "replay_count": len(replay_frames),
                "breakpoint_count": len(breakpoint_items),
                "variable_count": len(variable_items),
                "replay_frames": replay_frames,
                "breakpoints": breakpoint_items,
                "variables": variable_items,
            }}
            selected_run = {{
                "id": str(raw_run.get("id", run_id)),
                "workflow": str(raw_run.get("workflow", "")),
                "workflow_id": str(raw_run.get("workflow_id", "")),
                "entity": str(raw_run.get("entity", "")),
                "status": str(raw_run.get("status", "")),
                "badge_class": _badge(str(raw_run.get("status", ""))),
                "created_record_id": str(raw_run.get("created_record_id", "")),
                "event_id": str(raw_run.get("event_id", "")),
                "trace": trace,
                "journal": [
                    {{
                        "seq": str(event.get("seq", "")),
                        "event_type": str(event.get("event_type", "")),
                        "step": str(event.get("step", "")),
                        "ts": str(event.get("ts", "")),
                        "data": event.get("data", {{}}),
                        "data_json": json.dumps(event.get("data", {{}}), indent=2, sort_keys=True),
                    }}
                    for event in journal
                    if isinstance(event, dict)
                ],
                "payload_json": json.dumps(raw_run.get("payload", {{}}), indent=2, sort_keys=True),
                "record_json": json.dumps(raw_run.get("record", {{}}), indent=2, sort_keys=True),
                "step_count": len(trace),
                "event_count": len(journal),
                "duration_ms": sum(
                    int(step.get("duration_ms", 0))
                    for step in raw_run.get("trace", [])
                    if isinstance(step, dict) and str(step.get("duration_ms", "")).isdigit()
                ),
                "debug_intelligence": debug_intelligence,
            }}
    run_items = [
        {{
            "id": str(run.get("id", "")),
            "workflow": str(run.get("workflow", "")),
            "entity": str(run.get("entity", "")),
            "status": str(run.get("status", "")),
            "badge_class": _badge(str(run.get("status", ""))),
            "step_count": len(run.get("trace", [])),
            "created_record_id": str(run.get("created_record_id", "")),
        }}
        for run in sorted(runs, key=lambda item: str(item.get("id", "")), reverse=True)[:50]
        if isinstance(run, dict)
    ]
    breaker_items = [
        {{
            "key": str(key),
            "state": str(value.get("state", "closed")) if isinstance(value, dict) else "closed",
            "failures": value.get("failures", 0) if isinstance(value, dict) else 0,
            "badge_class": _badge(str(value.get("state", "closed")) if isinstance(value, dict) else "closed"),
        }}
        for key, value in sorted(cb_status.items())
    ]
    subscription_items = [
        {{"event": str(event), "workflows": ", ".join(str(item) for item in workflows)}}
        for event, workflows in sorted(subs.items())
    ]
    tmpl_body = _render_template(
        "debug_console.html.j2",
        selected_run=selected_run,
        runs=run_items,
        circuit_breakers=breaker_items,
        subscriptions=subscription_items,
    )
    if tmpl_body is not None:
        return 200, _html_page("Flow Debugger", tmpl_body)
    return 200, _html_page("Flow Debugger", _jinja_required_page("Flow Debugger"))

def _ui_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    parts = [part for part in path.split("/") if part]
    if parts == ["ui"]:
        return 200, _ui_index_html()
    if parts == ["ui", "databases"]:
        return _ui_database_catalog_html()
    if parts == ["ui", "workflows"]:
        return _ui_workflow_list_html()
    # /ui/workflows/ENTITY/WORKFLOW_ID  or  /ui/workflows/ENTITY/WORKFLOW_ID/step/N
    if len(parts) >= 4 and parts[0] == "ui" and parts[1] == "workflows":
        entity_name = parts[2]
        workflow_id = parts[3]
        step_index = 0
        if len(parts) == 6 and parts[4] == "step":
            try:
                step_index = int(parts[5])
            except ValueError:
                step_index = 0
        return _ui_workflow_wizard_html(entity_name, workflow_id, step_index)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "entities":
        if query and query.get("view", [""])[0] == "kanban":
            return _ui_kanban_html(parts[2])
        if query and query.get("view", [""])[0] == "analytics":
            return _ui_entity_analytics_html(parts[2])
        return _ui_entity_html(parts[2], query=query)
    # /ui/entities/ENTITY/RECORD_ID
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities":
        return _ui_record_detail_html(parts[2], parts[3])
    # /ui/entities/ENTITY/RECORD_ID/fields/FIELD_NAME/edit|view
    if (len(parts) == 7 and parts[0] == "ui" and parts[1] == "entities"
            and parts[4] == "fields" and parts[6] in {{"edit", "view"}}):
        if parts[6] == "edit":
            status, fragment = _ui_field_edit_html(parts[2], parts[3], parts[5])
        else:
            status, fragment = _ui_field_view_html(parts[2], parts[3], parts[5])
        return status, fragment
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "agents":
        return _ui_agent_console_html(parts[2])
    if len(parts) == 3 and parts[0] == "ui" and parts[1] in {{"agent-teams", "teams"}}:
        return _ui_agent_console_html(parts[2], team=True)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "capabilities":
        return _ui_capability_console_html(parts[2])
    if parts[:2] == ["ui", "debug"]:
        return _ui_debug_html(parts[2] if len(parts) > 2 else None)
    if parts == ["ui", "marketplace"]:
        try:
            from compiler.connector_generator import scan_connectors
            connectors = scan_connectors("connectors")
        except Exception:
            connectors = list(APG_CONNECTOR_REGISTRY)
        cards = _marketplace_cards(connectors)
        q = (query or {{}}).get("q", [""])[0].strip() if query else ""
        active_category = (query or {{}}).get("category", ["all"])[0].strip() if query else "all"
        categories: list[Dict[str, Any]] = []
        for category_name in sorted({{str(card["category"]) for card in cards}}):
            count = len([card for card in cards if card["category"] == category_name])
            categories.append({{"name": category_name, "count": count, "active": category_name == active_category}})
        filtered_cards = cards
        if active_category and active_category != "all":
            filtered_cards = [card for card in filtered_cards if card["category"] == active_category]
        if q:
            q_lower = q.lower()
            filtered_cards = [
                card for card in filtered_cards
                if q_lower in card["title"].lower()
                or q_lower in card["description"].lower()
                or q_lower in card["category"].lower()
            ]
        tmpl_body = _render_template("marketplace.html.j2",
            connectors=filtered_cards,
            connector_count=len(cards),
            filtered_count=len(filtered_cards),
            installed_count=len(connectors),
            marketplace_intelligence=_marketplace_intelligence(cards),
            categories=categories,
            active_category=active_category or "all",
            query=q,
            has_filters=bool(q or (active_category and active_category != "all")),
            has_installed_connectors=bool(connectors),
        )
        if tmpl_body is not None:
            return 200, _html_page("Connector Marketplace", tmpl_body)
        return 200, _html_page("Connector Marketplace", "<h1>Connector Marketplace</h1>")
    return 404, _html_page("Not found", f"<h1>Not found</h1><p>{{html.escape(path)}}</p>")


def _parse_json_object_field(form_record: Dict[str, Any], field_name: str) -> tuple[Dict[str, Any] | None, str | None]:
    raw_value = str(form_record.get(field_name) or "{{}}").strip() or "{{}}"
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        return None, f"{{field_name}} is invalid JSON: {{error}}"
    if not isinstance(value, dict):
        return None, f"{{field_name}} must be a JSON object"
    return value, None


def _result_section(result: Dict[str, Any] | None = None, error: str = "") -> str:
    if error:
        return f'<section role="alert"><strong>{{html.escape(error)}}</strong></section>'
    if result is None:
        return ""
    return "<h2>Result</h2><pre>" + html.escape(json.dumps(result, indent=2, sort_keys=True)) + "</pre>"


def _sanitize_agent_markdown(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)
    escaped = html.escape(str(text), quote=True)
    escaped = re.sub(r"`([^`]+)`", r"<code>\\1</code>", escaped)
    escaped = re.sub(r"\\*\\*([^*]+)\\*\\*", r"<strong>\\1</strong>", escaped)
    lines = escaped.splitlines() or [""]
    html_lines: list[str] = []
    in_list = False
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append("<li>" + line[2:] + "</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(line)
    if in_list:
        html_lines.append("</ul>")
    return "<br>".join(html_lines)


def _agent_display_text(result: Dict[str, Any] | None) -> Any:
    if not isinstance(result, dict):
        return ""
    for key in ("output", "response", "message", "text", "content"):
        if key in result:
            return result[key]
    return result


def _agent_result_status(result: Dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    return str(result.get("status") or result.get("state") or "")


def _agent_status_badge(status: str) -> str:
    normalized = status.lower()
    if normalized in {{"completed", "success", "ok"}}:
        return "apg-badge-success"
    if normalized in {{"failed", "error", "unavailable"}}:
        return "apg-badge-danger"
    return "apg-badge-warning" if normalized else "apg-badge-neutral"


def _ui_agent_console_html(
    name: str,
    result: Dict[str, Any] | None = None,
    error: str = "",
    team: bool = False,
    request_payload: Dict[str, Any] | None = None,
    user_message: str = "",
) -> tuple[int, str]:
    app = describe_application()
    catalog_key = "ai_agent_team_descriptions" if team else "ai_agent_descriptions"
    catalog = app.get(catalog_key, {{}})
    if name not in catalog:
        title = "Unknown agent team" if team else "Unknown agent"
        return 404, _html_page(title, f"<h1>{{title}}</h1><p>{{html.escape(name)}}</p>")
    action = f"/ui/{{'agent-teams' if team else 'agents'}}/{{html.escape(name, quote=True)}}/invoke"
    description = catalog[name]
    request_payload = dict(request_payload or {{}})
    result_status = _agent_result_status(result)
    team_members = list(description.get("agents", [])) if team and isinstance(description, dict) else []
    team_flow = list(description.get("flow", [])) if team and isinstance(description, dict) else []
    result_text = _agent_display_text(result) if result is not None else ""
    approx_tokens = max(0, int(len(result_text) / 4))
    tool_names = [str(tool) for tool in description.get("tools", [])] if isinstance(description, dict) else []
    capability_names = [str(cap) for cap in description.get("capabilities", [])] if isinstance(description, dict) else []
    prompt_library = []
    role_name = str(description.get("role", "Assistant")) if isinstance(description, dict) else "Assistant"
    prompt_library.append({{"name": "Role primer", "version": "v1", "prompt": role_name}})
    for index, capability in enumerate(capability_names[:3], start=1):
        prompt_library.append({{"name": capability, "version": f"v{{index + 1}}", "prompt": f"Use {{capability}} carefully and report evidence."}})
    agent_intelligence = {{
        "stream_meter": {{
            "tokens": approx_tokens,
            "chars": len(result_text),
            "rate": f"{{max(1, approx_tokens // 3)}} tok/s" if result_text else "idle",
            "cost": "offline estimate",
        }},
        "branch_seed": html.escape(user_message or str(request_payload.get("message", ""))[:120]),
        "tool_calls": [
            {{"name": tool, "status": "available", "source": "declared"}}
            for tool in tool_names[:6]
        ],
        "prompt_library": prompt_library[:4],
        "run_compare": [
            {{"label": "Prompt chars", "left": len(user_message), "right": len(result_text)}},
            {{"label": "Payload keys", "left": len(request_payload), "right": len(result or {{}}) if isinstance(result, dict) else 0}},
            {{"label": "Tools declared", "left": len(tool_names), "right": len(capability_names)}},
        ],
        "library_key": f"apg:agent-prompts:{{name}}",
        "branch_key": f"apg:agent-branch:{{name}}",
    }}
    tmpl_body = _render_template(
        "agent_console.html.j2",
        name=name,
        team=team,
        action=action,
        description=description,
        description_json=json.dumps(description, indent=2, sort_keys=True),
        result=result,
        result_json=json.dumps(result, indent=2, sort_keys=True) if result is not None else "",
        result_html=_sanitize_agent_markdown(_agent_display_text(result)) if result is not None else "",
        result_status=result_status,
        result_badge_class=_agent_status_badge(result_status),
        error=error,
        user_message=user_message,
        payload_json=json.dumps(request_payload, indent=2, sort_keys=True) if request_payload else "{{}}",
        team_members=team_members,
        team_flow=team_flow,
        live_topic=f"agent:{{name}}",
        agent_intelligence=agent_intelligence,
    )
    return 200, _html_page(name, tmpl_body if tmpl_body is not None else _jinja_required_page(name))


def _capability_default_rule_context(description: Dict[str, Any]) -> Dict[str, Any]:
    configuration = description.get("configuration", {{}}) if isinstance(description, dict) else {{}}
    default_limit = configuration.get("default_limit", 1000) if isinstance(configuration, dict) else 1000
    review_threshold = configuration.get("review_threshold", 0.5) if isinstance(configuration, dict) else 0.5
    return {{
        "tenant_id": "example-tenant",
        "customer_id": "customer-001",
        "amount": default_limit,
        "risk_score": review_threshold,
        "is_international": False,
    }}


def _capability_default_approval_context(description: Dict[str, Any]) -> Dict[str, Any]:
    context = _capability_default_rule_context(description)
    context["requester"] = "operator"
    return context


def _capability_operation_label(operation: str) -> str:
    labels = {{
        "rules": "Rules evaluation",
        "configuration": "Configuration resolution",
        "approval": "Approval plan",
    }}
    return labels.get(operation, "Result")


def _ui_capability_console_html(
    name: str,
    result: Dict[str, Any] | None = None,
    error: str = "",
    operation: str = "",
    context_json: str = "",
    configuration_json: str = "",
    approval_context_json: str = "",
) -> tuple[int, str]:
    app = describe_application()
    capabilities = app.get("capability_descriptions", {{}})
    if name not in capabilities:
        return 404, _html_page("Unknown capability", f"<h1>Unknown capability</h1><p>{{html.escape(name)}}</p>")
    safe_name = html.escape(name, quote=True)
    description = capabilities[name]
    default_rule_context = _capability_default_rule_context(description)
    default_approval_context = _capability_default_approval_context(description)
    default_configuration = description.get("configuration", {{}}) if isinstance(description, dict) else {{}}
    result_items = []
    if isinstance(result, dict):
        for key, value in sorted(result.items()):
            if isinstance(value, (dict, list)):
                result_items.append((str(key), json.dumps(value, sort_keys=True)))
            else:
                result_items.append((str(key), str(value)))
    rules = list(description.get("rules", [])) if isinstance(description, dict) else []
    approval_policy = description.get("approvals", {{}}) if isinstance(description, dict) else {{}}
    approvers = approval_policy.get("approvers", []) if isinstance(approval_policy, dict) else []
    if not approvers and isinstance(result, dict):
        approvers = result.get("approvers", [])
    resolved_configuration = result.get("configuration", {{}}) if isinstance(result, dict) else {{}}
    dry_run_diff = []
    if isinstance(default_configuration, dict):
        keys = sorted(set(default_configuration) | (set(resolved_configuration) if isinstance(resolved_configuration, dict) else set()))
        for key in keys[:6]:
            before = default_configuration.get(key, "")
            after = resolved_configuration.get(key, before) if isinstance(resolved_configuration, dict) else before
            dry_run_diff.append({{"key": str(key), "before": str(before), "after": str(after), "changed": before != after}})
    capability_intelligence = {{
        "test_cases": [
            {{"name": "Baseline", "context": json.dumps(default_rule_context, sort_keys=True), "expected": "allow or review"}},
            {{"name": "High risk", "context": json.dumps({{**default_rule_context, "risk_score": 0.95}}, sort_keys=True), "expected": "review"}},
            {{"name": "International", "context": json.dumps({{**default_rule_context, "is_international": True}}, sort_keys=True), "expected": "policy match"}},
        ],
        "dry_run_diff": dry_run_diff,
        "approval_sla": {{
            "target": "4h",
            "remaining": "3h 42m",
            "approvers": len(approvers) if isinstance(approvers, list) else 0,
            "required": bool(result.get("required")) if isinstance(result, dict) and "required" in result else bool(approval_policy),
        }},
        "rule_count": len(rules),
        "bench_key": f"apg:capability-testbench:{{name}}",
    }}
    tmpl_body = _render_template(
        "capability_console.html.j2",
        name=name,
        safe_name=safe_name,
        description=description,
        description_json=json.dumps(description, indent=2, sort_keys=True),
        rule_context_json=context_json or json.dumps(default_rule_context, indent=2, sort_keys=True),
        configuration_json=configuration_json or json.dumps(default_configuration, indent=2, sort_keys=True),
        approval_context_json=approval_context_json or json.dumps(default_approval_context, indent=2, sort_keys=True),
        operation=operation,
        operation_label=_capability_operation_label(operation),
        result=result,
        result_items=result_items,
        result_json=json.dumps(result, indent=2, sort_keys=True) if result is not None else "",
        result_json_html=html.escape(json.dumps(result, indent=2, sort_keys=True)) if result is not None else "",
        error=error,
        capability_intelligence=capability_intelligence,
    )
    return 200, _html_page(name, tmpl_body if tmpl_body is not None else _jinja_required_page(name))


def _ui_post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    raw_form_record = payload.get("record", payload)
    form_record = dict(raw_form_record) if isinstance(raw_form_record, dict) else {{}}

    # Field patch POST: /ui/entities/ENTITY/RECORD_ID/fields/FIELD_NAME/patch
    if (len(parts) == 7 and parts[0] == "ui" and parts[1] == "entities"
            and parts[4] == "fields" and parts[6] == "patch"):
        return _ui_field_patch_post(parts[2], parts[3], parts[5], form_record)

    # Workflow step POST: /ui/workflows/ENTITY/WORKFLOW_ID/step/N
    if (len(parts) == 6 and parts[0] == "ui" and parts[1] == "workflows" and parts[4] == "step"):
        entity_name, workflow_id = parts[2], parts[3]
        try:
            step_index = int(parts[5])
        except ValueError:
            step_index = 0
        _status, html_payload = _ui_workflow_step_post(entity_name, workflow_id, step_index, form_record)
        return _status, {{"html": html_payload}}

    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "agents" and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        message = str(form_record.get("message") or "")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, request_payload={{}}, user_message=message)
            return 400, {{"html": html_payload}}
        if message:
            request_payload["message"] = message
        if str(form_record.get("stream", "")).lower() in {{"1", "true", "yes", "on"}}:
            request_payload["stream"] = True
        status, result = _agent_invocation_payload(f"/agents/{{parts[2]}}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(
            parts[2],
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "agent invocation failed"),
            request_payload=request_payload,
            user_message=message,
        )
        return status, {{"html": html_payload}}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] in {{"agent-teams", "teams"}} and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        message = str(form_record.get("message") or "")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, team=True, request_payload={{}}, user_message=message)
            return 400, {{"html": html_payload}}
        if message:
            request_payload["message"] = message
        if str(form_record.get("stream", "")).lower() in {{"1", "true", "yes", "on"}}:
            request_payload["stream"] = True
        status, result = _agent_invocation_payload(f"/agent-teams/{{parts[2]}}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(
            parts[2],
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "team invocation failed"),
            team=True,
            request_payload=request_payload,
            user_message=message,
        )
        return status, {{"html": html_payload}}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "capabilities":
        capability_name = parts[2]
        operation = "/".join(parts[3:])
        if operation == "rules/evaluate":
            raw_context_json = str(form_record.get("context_json") or "")
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="rules", context_json=raw_context_json)
                return 400, {{"html": html_payload}}
            status, result = _rule_evaluation_payload(f"/capabilities/{{capability_name}}/rules/evaluate", {{"context": context}})
        elif operation == "configuration/resolve":
            raw_configuration_json = str(form_record.get("configuration_json") or "")
            configuration, error = _parse_json_object_field(form_record, "configuration_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="configuration", configuration_json=raw_configuration_json)
                return 400, {{"html": html_payload}}
            status, result = _configuration_payload(f"/capabilities/{{capability_name}}/configuration/resolve", {{"overrides": configuration}})
        elif operation == "approval/plan":
            raw_approval_context_json = str(form_record.get("context_json") or "")
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error, operation="approval", approval_context_json=raw_approval_context_json)
                return 400, {{"html": html_payload}}
            status, result = _approval_plan_payload(f"/capabilities/{{capability_name}}/approval/plan", {{"context": context}})
        else:
            return 404, {{"error": "not_found", "path": path}}
        op_key = "rules" if operation == "rules/evaluate" else "configuration" if operation == "configuration/resolve" else "approval"
        _status, html_payload = _ui_capability_console_html(
            capability_name,
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "capability operation failed"),
            operation=op_key,
            context_json=raw_context_json if operation == "rules/evaluate" else "",
            configuration_json=raw_configuration_json if operation == "configuration/resolve" else "",
            approval_context_json=raw_approval_context_json if operation == "approval/plan" else "",
        )
        return status, {{"html": html_payload}}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        status, response = _create_record_payload(f"/entities/{{entity_name}}/records", payload)
        if status == 201:
            return 303, {{"location": _ui_entity_location(entity_name)}}
        _detail = response.get("errors") or response.get("message") or response.get("error") or "Record could not be created."
        if isinstance(_detail, list):
            _detail = "; ".join(str(item) for item in _detail)
        _page_status, html_payload = _ui_entity_html(entity_name, notice=str(_detail))
        return status, {{"html": html_payload}}
        return status, response
    if (len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities"
            and parts[3] == "records" and parts[4] == "bulk_delete"):
        entity_name = parts[2]
        ids_raw = form_record.get("ids", "")
        ids = [i.strip() for i in ids_raw.split(",") if i.strip()]
        for rid in ids:
            try:
                delete_record(entity_name, rid)
            except Exception:
                _ = None  # best-effort
        return 303, {{"location": _ui_entity_location(entity_name)}}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        record_id = parts[4]
        expected_revision = form_record.pop("expected_revision", None)
        return_view = form_record.pop("return_view", "")
        status, response = _update_record_payload(
            f"/entities/{{entity_name}}/records/{{record_id}}",
            {{"record": form_record, "expected_revision": expected_revision}},
        )
        if status == 200:
            if return_view == "kanban":
                return 303, {{"location": _ui_entity_location(entity_name) + "?view=kanban"}}
            return 303, {{"location": _ui_entity_location(entity_name)}}
        return status, response
    if (
        len(parts) == 6
        and parts[0] == "ui"
        and parts[1] == "entities"
        and parts[3] == "records"
        and parts[5] == "delete"
    ):
        entity_name = parts[2]
        record_id = parts[4]
        delete_path = f"/entities/{{entity_name}}/records/{{record_id}}"
        expected_revision = form_record.get("expected_revision")
        if expected_revision not in (None, ""):
            delete_path = f"{{delete_path}}?expected_revision={{quote(str(expected_revision), safe='')}}"
        status, response = _delete_record_payload(delete_path)
        if status == 200:
            return 303, {{"location": _ui_entity_location(entity_name)}}
        return status, response
    if (len(parts) == 6 and parts[0] == "ui" and parts[1] == "entities"
            and parts[3] == "records" and parts[5] == "note"):
        entity_name = parts[2]
        record_id = parts[4]
        note = str(form_record.get("note", "")).strip()
        if note:
            _log_activity(entity_name, record_id, "note", detail=note[:200])
        return 303, {{"location": f"/ui/entities/{{entity_name}}/{{record_id}}"}}
    return 404, {{"error": "not_found", "path": path}}


def _capability_screen(path: str) -> Dict[str, Any] | None:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "ui_route_index"):
        return None
    routes = APG_CAPABILITIES.ui_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _capability_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Capability screen")
    capability = str(screen.get("capability") or "")
    component = str(screen.get("component") or title)
    theme_name = str(screen.get("theme") or "")
    theme_tokens: Dict[str, Any] = {{}}
    if capability and APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_theme"):
        try:
            theme_tokens = APG_CAPABILITIES.capability_theme(capability).get("tokens", {{}})
        except KeyError:
            theme_tokens = {{}}
    actions = "".join(
        f"<li>{{html.escape(str(action))}}</li>"
        for action in screen.get("actions", [])
    ) or "<li>No actions declared.</li>"
    relationships = html.escape(json.dumps(screen.get("relationships", []), indent=2, sort_keys=True))
    tokens = html.escape(json.dumps(theme_tokens, indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{{html.escape(title)}}</h1>"
        f"<p><strong>Capability:</strong> {{html.escape(capability)}}</p>"
        f"<p><strong>Component:</strong> {{html.escape(component)}}</p>"
        f"<p><strong>Theme:</strong> {{html.escape(theme_name)}}</p>"
        f"<h2>Actions</h2><ul>{{actions}}</ul>"
        f"<h2>Relationships</h2><pre>{{relationships}}</pre>"
        f"<h2>Theme Tokens</h2><pre>{{tokens}}</pre>"
    )
    return _html_page(title, body)


def _capability_screen_payload(path: str) -> tuple[int, str]:
    screen = _capability_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{{html.escape(path)}}</p>")
    return 200, _capability_screen_html(screen)


def _application_screen(path: str) -> Dict[str, Any] | None:
    if APG_APPLICATIONS is None or not hasattr(APG_APPLICATIONS, "application_route_index"):
        return None
    routes = APG_APPLICATIONS.application_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _application_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Application route")
    application = str(screen.get("application") or "")
    route = str(screen.get("route") or screen.get("path") or "")
    capabilities = html.escape(json.dumps(screen.get("capabilities", []), indent=2, sort_keys=True))
    agents = html.escape(json.dumps(screen.get("agents", []), indent=2, sort_keys=True))
    component = html.escape(json.dumps(screen.get("component"), indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/applications">Applications</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{{html.escape(title)}}</h1>"
        f"<p><strong>Application:</strong> {{html.escape(application)}}</p>"
        f"<p><strong>Route:</strong> {{html.escape(route)}}</p>"
        f"<h2>Capabilities</h2><pre>{{capabilities}}</pre>"
        f"<h2>Agents</h2><pre>{{agents}}</pre>"
        f"<h2>Component</h2><pre>{{component}}</pre>"
    )
    return _html_page(title, body)


def _application_screen_payload(path: str) -> tuple[int, str]:
    screen = _application_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{{html.escape(path)}}</p>")
    return 200, _application_screen_html(screen)


def _record_route(path: str) -> Dict[str, str | None] | None:
    parts = [part for part in path.split("/") if part]
    if parts == ["records"]:
        return {{"entity": None, "record_id": None, "operation": None}}
    if len(parts) == 3 and parts[0] == "records" and parts[2] == "search":
        return {{"entity": parts[1], "record_id": None, "operation": "search"}}
    if len(parts) == 3 and parts[0] == "records" and parts[2] == "bulk":
        return {{"entity": parts[1], "record_id": None, "operation": "bulk"}}
    if len(parts) == 4 and parts[0] == "records" and parts[3] == "restore":
        return {{"entity": parts[1], "record_id": parts[2], "operation": "restore"}}
    if len(parts) in {{2, 3}} and parts[0] == "records":
        return {{
            "entity": parts[1],
            "record_id": parts[2] if len(parts) == 3 else None,
            "operation": None,
        }}
    if len(parts) == 5 and parts[0] == "entities" and parts[2] == "records" and parts[4] == "restore":
        return {{"entity": parts[1], "record_id": parts[3], "operation": "restore"}}
    if len(parts) == 4 and parts[0] == "entities" and parts[2] == "records" and parts[3] == "bulk":
        return {{"entity": parts[1], "record_id": None, "operation": "bulk"}}
    if len(parts) in {{3, 4}} and parts[0] == "entities" and parts[2] == "records":
        operation = parts[3] if len(parts) == 4 and parts[3] in {{"export", "import"}} else None
        return {{
            "entity": parts[1],
            "record_id": None if operation else parts[3] if len(parts) == 4 else None,
            "operation": operation,
        }}
    return None


def _record_by_id(entity_name: str, record_id: str, *, include_deleted: bool = False) -> Dict[str, Any] | None:
    for record in RECORD_STORE[entity_name]:
        if str(record.get("id")) == str(record_id):
            if not include_deleted and _record_deleted(entity_name, record):
                return None
            if not _record_owner_visible(record):
                return None
            if not _record_tenant_visible(entity_name, record):
                return None
            return _record_public_copy(entity_name, record)
    return None


def _relationship_route(path: str) -> Dict[str, Any] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) not in {{4, 5}} or parts[0] != "records":
        return None
    source_entity = parts[1]
    source_id = parts[2]
    segment = parts[3]
    if source_entity not in ENTITY_NAMES:
        return None
    relationship = _relationship_by_segment(source_entity, segment)
    if relationship is None:
        return None
    return {{
        "source": source_entity,
        "source_id": source_id,
        "segment": segment,
        "relationship": relationship,
        "target": str(relationship.get("target", "")),
        "through": relationship.get("through"),
        "related_id": parts[4] if len(parts) == 5 else None,
    }}


def _relationship_records(source_entity: str, source_id: Any, relationship: Dict[str, Any]) -> list[Dict[str, Any]]:
    target_entity = str(relationship.get("target", ""))
    if target_entity not in ENTITY_NAMES:
        return []
    through_entity = relationship.get("through")
    if through_entity:
        through_name = str(through_entity)
        if through_name not in ENTITY_NAMES:
            return []
        left_field = str(relationship.get("left_field") or "left_id")
        right_field = str(relationship.get("right_field") or "right_id")
        related_ids = {{
            str(link.get(right_field))
            for link in list_records(through_name)
            if str(link.get(left_field)) == str(source_id) and link.get(right_field) not in (None, "")
        }}
        return [
            _field_acl_public_copy(target_entity, record)
            for record in list_records(target_entity)
            if str(record.get("id")) in related_ids
        ]
    fk_field = str(relationship.get("fk_field") or _relationship_field_name(source_entity))
    return [
        _field_acl_public_copy(target_entity, record)
        for record in list_records(target_entity)
        if str(record.get(fk_field, "")) == str(source_id)
    ]


def _relationship_records_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    route = _relationship_route(path)
    if route is None:
        return 404, {{"error": "not_found", "path": path}}
    source_entity = str(route["source"])
    source_id = str(route["source_id"])
    relationship = dict(route["relationship"])
    target_entity = str(route["target"])
    if target_entity not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": target_entity}}
    if _record_by_id(source_entity, source_id) is None:
        return 404, {{"error": "record_not_found", "entity": source_entity, "id": source_id}}
    if route.get("related_id") is not None:
        return 405, {{"error": "method_not_allowed", "operation": "relationship_item"}}
    records = _relationship_records(source_entity, source_id, relationship)
    return 200, {{
        "entity": target_entity,
        "parent": {{"entity": source_entity, "id": source_id}},
        "relationship": relationship,
        "records": records,
        "count": len(records),
    }}


def _create_relationship_payload(path: str, payload: Dict[str, Any], *, persist: bool = True) -> tuple[int, Dict[str, Any]]:
    route = _relationship_route(path)
    if route is None or route.get("related_id") in (None, ""):
        return 404, {{"error": "not_found", "path": path}}
    relationship = dict(route["relationship"])
    through_entity = relationship.get("through")
    if not through_entity:
        return 405, {{"error": "method_not_allowed", "operation": "relationship_link"}}
    source_entity = str(route["source"])
    source_id = str(route["source_id"])
    target_entity = str(route["target"])
    related_id = str(route["related_id"])
    through_name = str(through_entity)
    if through_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": through_name}}
    if _record_by_id(source_entity, source_id) is None:
        return 404, {{"error": "record_not_found", "entity": source_entity, "id": source_id}}
    if _record_by_id(target_entity, related_id) is None:
        return 404, {{"error": "record_not_found", "entity": target_entity, "id": related_id}}
    left_field = str(relationship.get("left_field") or "left_id")
    right_field = str(relationship.get("right_field") or "right_id")
    for link in list_records(through_name):
        if str(link.get(left_field)) == source_id and str(link.get(right_field)) == related_id:
            return 200, {{
                "entity": target_entity,
                "relationship": relationship,
                "link": link,
                "created": False,
            }}
    raw_record = payload.get("record", payload)
    link_record = dict(raw_record) if isinstance(raw_record, dict) else {{}}
    link_record[left_field] = source_id
    link_record[right_field] = related_id
    status, response = _create_record_payload(
        f"/records/{{quote(through_name, safe='')}}",
        {{"record": link_record}},
        persist=persist,
    )
    if status >= 400:
        return status, response
    return status, {{
        "entity": target_entity,
        "relationship": relationship,
        "link": response.get("record", link_record),
        "created": True,
    }}


def _delete_relationship_payload(path: str, *, persist: bool = True) -> tuple[int, Dict[str, Any]] | None:
    route = _relationship_route(path)
    if route is None or route.get("related_id") in (None, ""):
        return None
    relationship = dict(route["relationship"])
    through_entity = relationship.get("through")
    if not through_entity:
        return 405, {{"error": "method_not_allowed", "operation": "relationship_unlink"}}
    through_name = str(through_entity)
    if through_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": through_name}}
    source_id = str(route["source_id"])
    related_id = str(route["related_id"])
    left_field = str(relationship.get("left_field") or "left_id")
    right_field = str(relationship.get("right_field") or "right_id")
    removed: list[Dict[str, Any]] = []
    remaining: list[Dict[str, Any]] = []
    for link in RECORD_STORE[through_name]:
        if (
            str(link.get(left_field)) == source_id
            and str(link.get(right_field)) == related_id
            and not _record_deleted(through_name, link)
            and _record_tenant_visible(through_name, link)
        ):
            removed.append(_record_public_copy(through_name, link))
            continue
        remaining.append(link)
    if not removed:
        return 404, {{"error": "relationship_not_found", "relationship": relationship}}
    RECORD_STORE[through_name] = remaining
    conn = _sqlite_connection()
    if conn is not None:
        sql = (
            "DELETE FROM " + _sqlite_identifier(through_name)
            + " WHERE " + _sqlite_identifier(left_field) + "=? AND "
            + _sqlite_identifier(right_field) + "=?"
        )
        params: list[Any] = [source_id, related_id]
        if _tenant_scope_enabled(through_name) and not _tenant_admin_bypass():
            sql += " AND tenant_id=?"
            params.append(_tenant_id() or APG_TENANT_DEFAULT)
        conn.execute(_apg_qmark(sql), params)
    if persist:
        _sqlite_commit()
        persistence_error = _persist_record_store()
        if persistence_error:
            return 500, {{"error": "persistence_failed", "message": persistence_error}}
    return 200, {{
        "relationship": relationship,
        "deleted": removed,
        "count": len(removed),
    }}


def _records_payload(path: str) -> tuple[int, Dict[str, Any]]:
    return _records_payload_with_query(path, {{}})


def _records_api_collection_path(path: str) -> bool:
    parts = [part for part in path.split("/") if part]
    return len(parts) == 2 and parts[0] == "records"


def _records_search_limit(query: Dict[str, list[str]] | None) -> int:
    raw_limit = _record_query_value(query or {{}}, "limit", 20)
    try:
        parsed = int(raw_limit)
    except (TypeError, ValueError):
        parsed = 20
    return max(1, min(parsed, 1000))


def search_records(entity_name: str, query: Dict[str, list[str]] | None = None) -> list[Dict[str, Any]]:
    # FTS5 search is SQLite-only. On PostgreSQL, return empty until pg_trgm/tsvector is wired.
    if _apg_db_dialect() == "pg":
        return []
    q = str(_record_query_value(query or {{}}, "q", "") or "").strip()
    if not q or not _record_string_field_names(entity_name):
        return []
    conn = _sqlite_connection()
    if conn is None:
        return []
    entity_sql = _sqlite_fts_identifier(entity_name)
    fts_sql = _sqlite_fts_identifier(entity_name + "_fts")
    where_clauses = [
        "id IN (SELECT rowid FROM " + fts_sql + " WHERE " + fts_sql + " MATCH ?)",
        "deleted_at IS NULL",
    ]
    params: list[Any] = [q]
    if _tenant_scope_enabled(entity_name) and not _tenant_admin_bypass():
        where_clauses.append("tenant_id=?")
        params.append(_tenant_id() or APG_TENANT_DEFAULT)
    sql = (
        "SELECT * FROM "
        + entity_sql
        + " WHERE "
        + " AND ".join(where_clauses)
        + " LIMIT ?"
    )
    params.append(_records_search_limit(query))
    try:
        rows = conn.execute(sql, params).fetchall()
    except _sqlite3.DatabaseError:
        return []
    records: list[Dict[str, Any]] = []
    for row in rows:
        record = _record_by_id(entity_name, row["id"])
        if record is not None:
            records.append(record)
    return records


def _records_payload_with_query(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    relationship_route = _relationship_route(path)
    if relationship_route is not None:
        return _relationship_records_payload(path, query)
    route = _record_route(path)
    if route is None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    record_id = route["record_id"]
    operation = route.get("operation")
    if entity_name is None:
        return 200, {{"records": list_records()}}
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    if operation == "export":
        return 200, {{
            "entity": entity_name,
            "records": list_records(entity_name),
            "count": len(list_records(entity_name)),
        }}
    if operation == "search":
        return 200, search_records(entity_name, query)
    if operation is not None:
        return 405, {{"error": "method_not_allowed", "operation": operation}}
    if record_id is None:
        style = "records" if _records_api_collection_path(path) else "legacy"
        result = query_records(entity_name, query, response_style=style)
        if result.get("error") == "invalid_field":
            return 400, {{"error": "invalid_field"}}
        if result.get("error") == "include_deleted_requires_admin":
            return 403, {{"error": "include_deleted_requires_admin"}}
        if style == "records":
            result = dict(result)
            result["data"] = [
                _field_acl_public_copy(entity_name, record)
                for record in result.get("data", [])
                if isinstance(record, dict)
            ]
        else:
            result = dict(result)
            result["records"] = [
                _field_acl_public_copy(entity_name, record)
                for record in result.get("records", [])
                if isinstance(record, dict)
            ]
        return 200, result
    include_deleted_requested = _query_includes_deleted(query or {{}})
    if include_deleted_requested and not _records_admin_allowed():
        return 403, {{"error": "include_deleted_requires_admin"}}
    include_deleted = include_deleted_requested and _records_admin_allowed()
    record = _record_by_id(entity_name, record_id, include_deleted=include_deleted)
    if record is None:
        return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}
    return 200, {{"entity": entity_name, "record": _field_acl_public_copy(entity_name, record)}}


def _route_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path in {{"/", "/manifest", "/application"}}:
        return 200, describe_application()
    if path == "/component.json":
        return 200, component_manifest()
    if path == "/semantic-model.json":
        return 200, semantic_model()
    if path == "/health":
        validation = validate_application()
        return 200, {{
            "status": "ok" if validation["valid"] else "warning",
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "valid": validation["valid"],
            "storage": storage_status(),
            "auth": auth_status(),
            "warnings": validation["warnings"],
        }}
    if path == "/validate":
        validation = validate_application()
        return (200 if validation["valid"] else 422), validation
    if path == "/openapi.json":
        return 200, openapi_document()
    if path == "/entities":
        return 200, {{"entities": list_entities()}}
    if path == "/workflows":
        return 200, {{"workflows": describe_workflows()}}
    if path == "/workflows/runs":
        return 200, {{"runs": list_workflow_runs()}}
    if path.startswith("/workflows/runs/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 4 and parts[3] == "journal":
            return 200, {{"run_id": parts[2], "events": _get_journal(parts[2])}}
        if len(parts) == 3:
            try:
                return 200, get_workflow_run(parts[2])
            except KeyError:
                return 404, {{"error": "workflow_run_not_found", "id": parts[2]}}
    if path.startswith("/workflows/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 2:
            try:
                return 200, describe_workflow(parts[1])
            except KeyError:
                return 404, {{"error": "unknown_workflow", "workflow": parts[1]}}
    if path == "/jobs":
        return _jobs_payload(query)
    if path.startswith("/jobs/"):
        return _job_detail_payload(path)
    if path == "/databases":
        return 200, {{"databases": list_databases()}}
    if path == "/databases/status":
        status = database_status()
        return (200 if status["valid"] else 422), status
    if path.startswith("/databases/") and path.endswith("/schemas"):
        database_name = path.strip("/").split("/")[1]
        for database in list_databases():
            if str(database.get("name")) == database_name:
                return 200, {{
                    "database": database_name,
                    "schemas": database.get("schemas", []),
                }}
        return 404, {{"error": "unknown_database", "database": database_name}}
    if path == "/auth":
        return 200, auth_status()
    if path == "/events":
        return 200, {{"events": list_events()}}
    if path == "/events/subscriptions":
        return 200, {{"subscriptions": dict(APG_EVENT_SUBSCRIPTIONS)}}
    if path == "/api/search":
        q = str((query or {{}}).get("q", [""])[0]).strip().lower() if query else ""
        results: list[Dict[str, Any]] = []
        if q:
            for ent in ENTITIES:
                ename = str(ent["name"])
                for rec in list_records(ename)[:200]:
                    for v in rec.values():
                        if q in str(v).lower():
                            label_field = next(
                                (f["name"] for f in ent.get("fields", [])
                                 if f["name"] not in ["id", "_revision"]),
                                "id",
                            )
                            results.append({{
                                "entity": ename,
                                "id": str(rec.get("id", "")),
                                "label": str(rec.get(label_field, rec.get("id", "")))[:60],
                            }})
                            break
        results = results[:20]
        return 200, {{"results": results, "query": q, "count": len(results)}}
    if path == "/circuit-breakers":
        return 200, {{"circuit_breakers": circuit_breaker_status()}}
    if path == "/connectors":
        return 200, {{"connectors": APG_CONNECTOR_REGISTRY}}
    if path == "/metrics":
        return 200, metrics_snapshot()
    if path == "/self-test":
        report = self_test()
        return (200 if report["passed"] else 422), report
    if path == "/records" or path.startswith("/records/") or (
        path.startswith("/entities/") and "/records" in path
    ):
        return _records_payload_with_query(path, query)
    if path == "/relationships":
        return 200, relationship_graph()
    if path == "/storage":
        return 200, storage_status(include_records=True)
    if path == "/agents":
        return 200, {{
            "agents": describe_application().get("ai_agent_descriptions", {{}}),
            "teams": describe_application().get("ai_agent_team_descriptions", {{}}),
        }}
    if path == "/applications":
        app = describe_application()
        return 200, {{
            "applications": app.get("application_composition_descriptions", {{}}),
            "dependency_graph": app.get("application_dependency_graph", {{}}),
            "components": app.get("application_component_catalog", {{}}),
        }}
    if path == "/capabilities":
        app = describe_application()
        return 200, {{
            "capabilities": app.get("capability_descriptions", {{}}),
            "by_erp_module": app.get("capability_descriptions_by_erp_module", {{}}),
            "dependency_graph": app.get("capability_dependency_graph", {{}}),
            "load_order": app.get("capability_load_order", {{}}),
        }}
    if path == "/capabilities/health":
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health_report"):
            return 404, {{"error": "capability_health_unavailable"}}
        health = APG_CAPABILITIES.capability_health_report()
        return (200 if health.get("healthy") else 422), health
    if path.startswith("/capabilities/") and path.endswith("/health"):
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health"):
            return 404, {{"error": "capability_health_unavailable"}}
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            try:
                health = APG_CAPABILITIES.capability_health(parts[1])
            except KeyError:
                return 404, {{"error": "unknown_capability", "capability": parts[1]}}
            return (200 if health.get("healthy") else 422), health
    if path == "/streaming":
        return _streaming_payload()
    if path.startswith("/capabilities/") and path.endswith("/streaming"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            return _capability_streaming_payload(parts[1])
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


def _workflow_run_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    workflow_name = payload.get("workflow") or payload.get("workflow_name")
    if len(parts) >= 2:
        workflow_name = parts[1]
    if not workflow_name:
        return 400, {{"error": "missing_workflow"}}
    context = payload.get("payload", payload.get("context", {{}}))
    if not isinstance(context, dict):
        return 400, {{"error": "payload_must_be_object"}}
    if "start_at" in payload and "start_at" not in context:
        context = dict(context)
        context["start_at"] = payload["start_at"]
    try:
        return 200, run_workflow(str(workflow_name), context)
    except KeyError:
        return 404, {{"error": "unknown_workflow", "workflow": str(workflow_name)}}


def _workflow_resume_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 4:
        return 404, {{"error": "not_found", "path": path}}
    context = payload.get("payload", payload.get("context", {{}}))
    if not isinstance(context, dict):
        return 400, {{"error": "payload_must_be_object"}}
    if "pause_at" in payload and "pause_at" not in context:
        context = dict(context)
        context["pause_at"] = payload["pause_at"]
    if "stop_after" in payload and "stop_after" not in context:
        context = dict(context)
        context["stop_after"] = payload["stop_after"]
    try:
        return 200, resume_workflow(parts[2], context)
    except KeyError:
        return 404, {{"error": "workflow_run_not_found", "id": parts[2]}}


def _workflow_compensation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 4:
        return 404, {{"error": "not_found", "path": path}}
    context = payload.get("payload", payload.get("context", {{}}))
    if not isinstance(context, dict):
        return 400, {{"error": "payload_must_be_object"}}
    try:
        return 200, execute_workflow_compensations(parts[2], context)
    except KeyError:
        return 404, {{"error": "workflow_run_not_found", "id": parts[2]}}


def _streaming_payload() -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None:
        return 404, {{"error": "capabilities_unavailable"}}
    processor_index = getattr(APG_CAPABILITIES, "streaming_processor_index", lambda: {{}})()
    state_index = getattr(APG_CAPABILITIES, "streaming_state_index", lambda: {{}})()
    streams: Dict[str, Any] = {{}}
    if hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_streaming"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            streams[capability_name] = APG_CAPABILITIES.capability_streaming(capability_name)
    return 200, {{
        "processor": "bytewax",
        "processors": processor_index,
        "states": state_index,
        "streams": streams,
    }}


def _capability_streaming_payload(capability_name: str) -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_streaming"):
        return 404, {{"error": "capability_streaming_unavailable"}}
    try:
        return 200, APG_CAPABILITIES.capability_streaming(capability_name)
    except KeyError:
        return 404, {{"error": "unknown_capability", "capability": capability_name}}


def _agent_invocation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if AI_AGENTS is None:
        if len(parts) == 3 and parts[0] in {{"agent-teams", "teams"}} and parts[2] in {{"invoke", "run"}}:
            team_description = _entity_agent_team_descriptions().get(parts[1])
            if team_description is not None:
                return 200, {{
                    "team": parts[1],
                    "status": "unavailable",
                    "error": "agents_unavailable",
                    "source": "entity_metadata",
                    "flow": team_description.get("flow", []),
                    "invocations": [
                        {{"agent": str(agent_name), "status": "unavailable", "error": "agents_unavailable"}}
                        for agent_name in team_description.get("agents", [])
                    ],
                }}
        return 404, {{"error": "agents_unavailable"}}
    try:
        if len(parts) == 3 and parts[0] == "agents" and parts[2] in {{"invoke", "run"}}:
            topic = f"agent:{{parts[1]}}"
            _publish_live_event(topic, "agent-token", {{"status": "started", "token": ""}})
            if payload.get("stream"):
                streamer = getattr(AI_AGENTS, "stream_agent", None)
                if streamer is not None:
                    chunks: list[str] = []
                    for chunk in streamer(parts[1], payload):
                        token = chunk.get("token", "") if isinstance(chunk, dict) else str(chunk)
                        if token:
                            chunks.append(token)
                            _publish_live_event(topic, "agent-token", {{"token": token}})
                    result = {{"agent": parts[1], "status": "completed", "output": "".join(chunks), "streamed": True}}
                    _publish_live_event(topic, "agent-result", result)
                    return 200, result
            invoker = getattr(AI_AGENTS, "invoke_agent", None)
            if invoker is None:
                return 404, {{"error": "agent_invocation_unavailable"}}
            result = invoker(parts[1], payload)
            _publish_live_event(topic, "agent-result", result if isinstance(result, dict) else {{"output": result}})
            return 200, result
        if len(parts) == 3 and parts[0] in {{"agent-teams", "teams"}} and parts[2] in {{"invoke", "run"}}:
            topic = f"agent:{{parts[1]}}"
            _publish_live_event(topic, "agent-token", {{"status": "started", "token": ""}})
            invoker = getattr(AI_AGENTS, "invoke_team", None)
            if invoker is None:
                return 404, {{"error": "team_invocation_unavailable"}}
            try:
                result = invoker(parts[1], payload)
            except KeyError:
                team_description = _entity_agent_team_descriptions().get(parts[1])
                if team_description is None:
                    raise
                invocations = []
                for agent_name in team_description.get("agents", []):
                    agent_status, agent_result = _agent_invocation_payload(f"/agents/{{quote(str(agent_name), safe='')}}/invoke", payload)
                    invocations.append(agent_result if isinstance(agent_result, dict) else {{"output": agent_result, "status": agent_status}})
                if any(str(item.get("status", "")).lower() in {{"failed", "error"}} for item in invocations if isinstance(item, dict)):
                    team_status = "failed"
                elif any(str(item.get("status", "")).lower() == "adapter_required" for item in invocations if isinstance(item, dict)):
                    team_status = "adapter_required"
                else:
                    team_status = "completed"
                result = {{
                    "team": parts[1],
                    "status": team_status,
                    "source": "entity_metadata",
                    "flow": team_description.get("flow", []),
                    "invocations": invocations,
                }}
            _publish_live_event(topic, "agent-result", result if isinstance(result, dict) else {{"output": result}})
            return 200, result
    except KeyError as error:
        return 404, {{"error": "unknown_agent_composition", "name": str(error).strip("'")}}
    return 404, {{"error": "not_found", "path": path}}


def _create_record_payload(path: str, payload: Dict[str, Any], *, persist: bool = True) -> tuple[int, Dict[str, Any]]:
    if _relationship_route(path) is not None:
        return _create_relationship_payload(path, payload, persist=persist)
    route = _record_route(path)
    if route is not None and route.get("operation") == "import":
        return _import_records_payload(str(route["entity"]), payload)
    if route is not None and route.get("operation") == "bulk":
        return _bulk_records_payload(path, payload)
    if route is None or route["entity"] is None or route["record_id"] is not None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {{"error": "record_must_be_object"}}
    record = _strip_record_lifecycle_fields(coerce_record_types(entity_name, dict(raw_record)))
    upload_error = _apply_uploaded_files(entity_name, record)
    if upload_error is not None:
        return upload_error
    validation = validate_record(entity_name, record)
    if not validation["valid"]:
        return _record_validation_failure(validation)
    if record.get("id") in (None, ""):
        record["id"] = NEXT_RECORD_IDS[entity_name]
        NEXT_RECORD_IDS[entity_name] += 1
    elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
        return 409, {{"error": "duplicate_record_id", "entity": entity_name, "id": record["id"]}}
    record = _prepare_new_record(record, entity_name)
    RECORD_METADATA.setdefault(entity_name, {{}})[_record_metadata_key(record["id"])] = {{
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "deleted_at": None,
    }}
    RECORD_STORE[entity_name].append(record)
    _sqlite_store_record(entity_name, record)
    if persist:
        _sqlite_commit()
    event = _record_event("create", entity_name, after=record)
    _log_activity(entity_name, str(record.get("id", "")), "created", detail=f"Record created with {{len(record)}} fields")
    if persist:
        persistence_error = _persist_record_store()
        if persistence_error:
            return 500, {{"error": "persistence_failed", "message": persistence_error}}
        _apg_deliver_webhook("entity.created", entity_name, record.get("id"), payload, _apg_request_id())
        notify_to = str(os.environ.get("APG_NOTIFY_EMAIL", "") or "").strip()
        if notify_to:
            _apg_send_email(
                notify_to,
                f"New {{entity_name}} record in {{APG_APP_NAME}}",
                json.dumps(_record_public_copy(entity_name, record), indent=2, sort_keys=True, default=str),
            )
    return 201, {{
        "entity": entity_name,
        "record": _record_public_copy(entity_name, record),
        "event": event,
        "count": len(list_records(entity_name)),
    }}


def _import_records_payload(entity_name: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return 400, {{"error": "records_must_be_array"}}
    imported: list[Dict[str, Any]] = []
    events: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            errors.append({{"index": index, "errors": ["record must be object"]}})
            continue
        record = _strip_record_lifecycle_fields(coerce_record_types(entity_name, dict(raw_record)))
        validation = validate_record(entity_name, record)
        if not validation["valid"]:
            errors.append({{"index": index, "errors": validation["errors"]}})
            continue
        if record.get("id") in (None, ""):
            record["id"] = NEXT_RECORD_IDS[entity_name]
            NEXT_RECORD_IDS[entity_name] += 1
        elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
            errors.append({{"index": index, "errors": [f"duplicate id {{record['id']}}"]}})
            continue
        record = _prepare_new_record(record, entity_name)
        RECORD_METADATA.setdefault(entity_name, {{}})[_record_metadata_key(record["id"])] = {{
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "deleted_at": None,
        }}
        RECORD_STORE[entity_name].append(record)
        _sqlite_store_record(entity_name, record)
        imported.append(_record_public_copy(entity_name, record))
        events.append(_record_event("import", entity_name, after=record))
    _sqlite_commit()
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {{"error": "persistence_failed", "message": persistence_error}}
    return (201 if imported else 422), {{
        "entity": entity_name,
        "imported": imported,
        "events": events,
        "errors": errors,
        "count": len(imported),
        "failed": len(errors),
    }}


def _update_record_payload(path: str, payload: Dict[str, Any], *, persist: bool = True) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {{"error": "record_must_be_object"}}
    record_update = _strip_record_lifecycle_fields(coerce_record_types(entity_name, dict(raw_record)))
    upload_error = _apply_uploaded_files(entity_name, record_update)
    if upload_error is not None:
        return upload_error
    validation = validate_record(entity_name, record_update, partial=True)
    if not validation["valid"]:
        return _record_validation_failure(validation)
    for index, existing in enumerate(RECORD_STORE[entity_name]):
        if str(existing.get("id")) == str(record_id):
            if not _record_tenant_visible(entity_name, existing):
                continue
            if _record_deleted(entity_name, existing):
                return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}
            conflict = _revision_conflict(existing, _expected_revision(payload))
            if conflict is not None:
                return 409, conflict
            current_row = _record_public_copy(entity_name, existing)
            changed_fields = [
                key
                for key in record_update
                if str(record_update[key]) != str(current_row.get(key, ""))
            ]
            try:
                if _flask_request.method == "PUT":
                    _flask_g.apg_audit_extra = {{"changed_fields": changed_fields}}
            except RuntimeError:
                pass
            updated = dict(existing)
            updated.update(record_update)
            updated["id"] = existing.get("id")
            updated["created_at"] = existing.get("created_at") or _record_public_copy(entity_name, existing).get("created_at")
            updated["updated_at"] = _record_timestamp()
            updated["deleted_at"] = existing.get("deleted_at")
            updated["_revision"] = int(existing.get("_revision", 1)) + 1
            metadata = _record_metadata(entity_name, updated.get("id"), create=True)
            if metadata is not None:
                metadata["created_at"] = updated.get("created_at")
                metadata["updated_at"] = updated.get("updated_at")
                metadata["deleted_at"] = updated.get("deleted_at")
            RECORD_STORE[entity_name][index] = updated
            _sqlite_store_record(entity_name, updated)
            if persist:
                _sqlite_commit()
            event = _record_event("update", entity_name, before=existing, after=updated, changed_fields=changed_fields)
            _log_activity(entity_name, str(record_id), "updated", detail="Fields updated")
            if persist:
                persistence_error = _persist_record_store()
                if persistence_error:
                    return 500, {{"error": "persistence_failed", "message": persistence_error}}
                _apg_deliver_webhook("entity.updated", entity_name, record_id, payload, _apg_request_id())
            return 200, {{"entity": entity_name, "record": _record_public_copy(entity_name, updated), "event": event}}
    return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}


def _restore_record_payload(path: str, *, persist: bool = True) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {{"error": "not_found", "path": path}}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    for existing in RECORD_STORE[entity_name]:
        if str(existing.get("id")) == str(record_id):
            if not _record_tenant_visible(entity_name, existing):
                continue
            before = _record_public_copy(entity_name, existing)
            metadata = _record_metadata(entity_name, existing.get("id"), create=True)
            if metadata is not None:
                metadata["deleted_at"] = None
                metadata["updated_at"] = _record_timestamp()
            existing["deleted_at"] = None
            existing["updated_at"] = metadata["updated_at"] if metadata is not None else _record_timestamp()
            _sqlite_restore_record(entity_name, record_id)
            if persist:
                _sqlite_commit()
            after = _record_public_copy(entity_name, existing)
            event = _record_event("restore", entity_name, before=before, after=after)
            _log_activity(entity_name, str(record_id), "restored", detail="Record restored")
            if persist:
                persistence_error = _persist_record_store()
                if persistence_error:
                    return 500, {{"error": "persistence_failed", "message": persistence_error}}
            return 200, {{"entity": entity_name, "record": after, "event": event}}
    return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}


def _delete_record_payload(path: str, *, persist: bool = True) -> tuple[int, Dict[str, Any]]:
    relationship_result = _delete_relationship_payload(path, persist=persist)
    if relationship_result is not None:
        return relationship_result
    raw_path = path
    path = path.split("?", 1)[0]
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {{"error": "not_found", "path": path}}
    if route.get("operation") == "restore":
        return _restore_record_payload(path, persist=persist)
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    for existing in RECORD_STORE[entity_name]:
        if str(existing.get("id")) == str(record_id):
            if not _record_tenant_visible(entity_name, existing):
                continue
            if _record_deleted(entity_name, existing):
                return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}
            expected_revision = None
            if "?" in raw_path:
                query = parse_qs(raw_path.split("?", 1)[1], keep_blank_values=True)
                value = query.get("expected_revision", [None])[-1]
                try:
                    expected_revision = int(value) if value is not None else None
                except (TypeError, ValueError):
                    expected_revision = None
            conflict = _revision_conflict(existing, expected_revision)
            if conflict is not None:
                return 409, conflict
            _log_activity(entity_name, str(record_id), "deleted", detail="Record deleted")
            deleted = _record_public_copy(entity_name, existing)
            metadata = _record_metadata(entity_name, existing.get("id"), create=True)
            if metadata is not None:
                now = _record_timestamp()
                metadata["deleted_at"] = now
                metadata["updated_at"] = now
            existing["deleted_at"] = now
            existing["updated_at"] = now
            _sqlite_soft_delete_record(entity_name, record_id)
            if persist:
                _sqlite_commit()
            event = _record_event("delete", entity_name, before=deleted)
            if persist:
                persistence_error = _persist_record_store()
                if persistence_error:
                    return 500, {{"error": "persistence_failed", "message": persistence_error}}
                _apg_deliver_webhook("entity.deleted", entity_name, record_id, {{}}, _apg_request_id())
            return 200, {{
                "entity": entity_name,
                "deleted": _record_public_copy(entity_name, existing),
                "event": event,
                "count": len(list_records(entity_name)),
            }}
    return 404, {{"error": "record_not_found", "entity": entity_name, "id": record_id}}


def _record_state_snapshot() -> Dict[str, Any]:
    return json.loads(json.dumps({{
        "records": _raw_records_by_entity(),
        "metadata": RECORD_METADATA,
        "next_record_ids": NEXT_RECORD_IDS,
        "events": EVENT_LOG,
        "next_event_id": NEXT_EVENT_ID,
        "activity": APG_ACTIVITY_LOG,
    }}, default=str))


def _restore_record_state(snapshot: Dict[str, Any]) -> None:
    global NEXT_EVENT_ID
    RECORD_STORE.clear()
    for entity_name in ENTITY_NAMES:
        RECORD_STORE[entity_name] = [
            dict(record)
            for record in snapshot.get("records", {{}}).get(entity_name, [])
            if isinstance(record, dict)
        ]
    RECORD_METADATA.clear()
    for entity_name in ENTITY_NAMES:
        entity_metadata = snapshot.get("metadata", {{}}).get(entity_name, {{}})
        RECORD_METADATA[entity_name] = {{
            str(record_id): dict(metadata)
            for record_id, metadata in entity_metadata.items()
            if isinstance(metadata, dict)
        }}
    NEXT_RECORD_IDS.clear()
    NEXT_RECORD_IDS.update({{
        entity_name: int(snapshot.get("next_record_ids", {{}}).get(entity_name, 1))
        for entity_name in ENTITY_NAMES
    }})
    EVENT_LOG.clear()
    EVENT_LOG.extend(dict(event) for event in snapshot.get("events", []) if isinstance(event, dict))
    NEXT_EVENT_ID = int(snapshot.get("next_event_id", 1))
    APG_ACTIVITY_LOG.clear()
    APG_ACTIVITY_LOG.update({{
        str(key): list(value) if isinstance(value, list) else []
        for key, value in snapshot.get("activity", {{}}).items()
    }})


def _bulk_items(payload: Dict[str, Any], key: str) -> list[Any] | None:
    items = payload.get(key, [])
    return items if isinstance(items, list) else None


def _bulk_delete_record_id(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("id")
    return item


def _bulk_records_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route.get("operation") != "bulk":
        return 404, {{"error": "not_found", "path": path}}
    entity_name = str(route["entity"])
    if entity_name not in ENTITY_NAMES:
        return 404, {{"error": "unknown_entity", "entity": entity_name}}
    create_items = _bulk_items(payload, "create")
    update_items = _bulk_items(payload, "update")
    delete_items = _bulk_items(payload, "delete")
    if create_items is None or update_items is None or delete_items is None:
        return 400, {{"error": "bulk_items_must_be_arrays"}}
    total_items = len(create_items) + len(update_items) + len(delete_items)
    if total_items > 1000:
        return 400, {{"error": "bulk_limit_exceeded"}}
    result: Dict[str, Any] = {{"created": 0, "updated": 0, "deleted": 0, "errors": []}}
    snapshot = _record_state_snapshot()
    with APG_RECORD_LOCK:
        try:
            _sqlite_begin()
            for index, item in enumerate(create_items):
                if not isinstance(item, dict):
                    result["errors"].append({{"op": "create", "index": index, "error": "record_must_be_object"}})
                    continue
                status, response = _create_record_payload("/records/" + entity_name, item, persist=False)
                if status >= 400:
                    result["errors"].append({{"op": "create", "index": index, "status": status, "error": response.get("error")}})
                else:
                    result["created"] += 1
            for index, item in enumerate(update_items):
                if not isinstance(item, dict):
                    result["errors"].append({{"op": "update", "index": index, "error": "record_must_be_object"}})
                    continue
                record_id = item.get("id")
                if record_id in (None, ""):
                    result["errors"].append({{"op": "update", "index": index, "error": "missing_id"}})
                    continue
                update_payload = dict(item)
                update_payload.pop("id", None)
                status, response = _update_record_payload(
                    "/records/" + entity_name + "/" + quote(str(record_id), safe=""),
                    {{"record": update_payload}},
                    persist=False,
                )
                if status >= 400:
                    result["errors"].append({{"op": "update", "index": index, "status": status, "error": response.get("error")}})
                else:
                    result["updated"] += 1
            for index, item in enumerate(delete_items):
                record_id = _bulk_delete_record_id(item)
                if record_id in (None, ""):
                    result["errors"].append({{"op": "delete", "index": index, "error": "missing_id"}})
                    continue
                status, response = _delete_record_payload(
                    "/records/" + entity_name + "/" + quote(str(record_id), safe=""),
                    persist=False,
                )
                if status >= 400:
                    result["errors"].append({{"op": "delete", "index": index, "status": status, "error": response.get("error")}})
                else:
                    result["deleted"] += 1
            if result["errors"]:
                _sqlite_rollback()
                _restore_record_state(snapshot)
                return 422, result
            _sqlite_commit()
        except Exception as exc:
            _sqlite_rollback()
            _restore_record_state(snapshot)
            return 500, {{"created": 0, "updated": 0, "deleted": 0, "errors": [{{"error": "bulk_transaction_failed", "message": str(exc)}}]}}
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {{"error": "persistence_failed", "message": persistence_error}}
    return 200, result


def _post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path == "/jobs":
        return _create_job_payload(payload)
    if path.startswith("/jobs/") and path.endswith("/retry"):
        return _retry_job_payload(path)
    if path == "/events/emit":
        event_name = payload.get("name") or payload.get("event") or ""
        if not event_name:
            return 422, {{"error": "missing_field", "field": "name"}}
        ev = emit_apg_event(str(event_name), payload.get("payload") or {{}})
        return 200, {{"event": ev}}
    if (
        path.startswith("/agents/") and path.endswith(("/invoke", "/run"))
    ) or (
        (path.startswith("/agent-teams/") or path.startswith("/teams/")) and path.endswith(("/invoke", "/run"))
    ):
        return _agent_invocation_payload(path, payload)
    if path.startswith("/records/") or path.endswith("/records/import") or (
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
    if path.startswith("/workflows/runs/") and "/signal/" in path:
        parts = [part for part in path.split("/") if part]
        if len(parts) == 5 and parts[0] == "workflows" and parts[1] == "runs" and parts[3] == "signal":
            sig_run_id = parts[2]
            signal_name = parts[4]
            if sig_run_id not in WORKFLOW_SIGNALS:
                WORKFLOW_SIGNALS[sig_run_id] = []
            WORKFLOW_SIGNALS[sig_run_id].append(signal_name)
            _journal_append(sig_run_id, "signal_received", signal_name, {{"from": "external"}})
            return 200, {{"status": "signal_received", "run_id": sig_run_id, "signal": signal_name}}
    if path.startswith("/workflows/runs/") and path.endswith("/compensate"):
        return _workflow_compensation_payload(path, payload)
    if path.startswith("/workflows/runs/") and path.endswith("/resume"):
        return _workflow_resume_payload(path, payload)
    if path.startswith("/workflows/") and path.endswith("/run"):
        return _workflow_run_payload(path, payload)
    return 404, {{"error": "not_found", "path": path}}


def _put_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path.startswith("/records/") or (
        path.startswith("/entities/") and "/records/" in path
    ):
        return _update_record_payload(path, payload)
    return 404, {{"error": "not_found", "path": path}}


def _csv_export_columns(entity_name: str, records: list[Dict[str, Any]]) -> list[str]:
    cols: list[str] = ["id"]
    for field in _field_specs(entity_name):
        field_name = str(field.get("name", ""))
        if field_name and field_name not in cols and field_name != "_revision":
            cols.append(field_name)
    for record in records:
        for key in record.keys():
            key_text = str(key)
            if key_text not in cols and key_text != "_revision":
                cols.append(key_text)
    return cols


def _csv_export_body(
    entity_name: str,
    query: Dict[str, list[str]] | None = None,
    records: list[Dict[str, Any]] | None = None,
) -> bytes:
    if records is None:
        result = query_records(entity_name, query or {{}}, response_style="records", paginate=False)
        records = result.get("data", []) if result.get("error") != "invalid_field" else []
    import io, csv as _csv
    cols = _csv_export_columns(entity_name, records)
    buf = io.StringIO(newline="")
    w = _csv.writer(buf, lineterminator="\\r\\n")
    w.writerow(cols)
    for rec in records:
        w.writerow([rec.get(c, "") for c in cols])
    return buf.getvalue().encode("utf-8")


import os as _os_env


def _pg_database_url() -> str | None:
    raw = _os_env.environ.get("APG_DATABASE_URL") or _os_env.environ.get("APG_PG_URL") or _os_env.environ.get("DATABASE_URL") or ""
    if raw.startswith("sqlite:"):
        return None
    return raw or None


_APG_PG_URL: str | None = _pg_database_url()


def _pg_connection():
    if not _APG_PG_URL:
        return None
    try:
        import psycopg2  # type: ignore
        return psycopg2.connect(_APG_PG_URL)
    except Exception:
        return None


def _pg_ensure_runs_table(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_workflow_runs ("
                "  run_id TEXT PRIMARY KEY,"
                "  module_name TEXT NOT NULL,"
                "  data TEXT NOT NULL,"
                "  updated_at TIMESTAMPTZ DEFAULT NOW()"
                ")"
            )
        conn.commit()
    except Exception:
        _ = None  # best-effort


def _pg_save_workflow_run(run: Dict[str, Any]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        _pg_ensure_runs_table(conn)
        rid = str(run.get("id", ""))
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO apg_workflow_runs (run_id, module_name, data)"
                " VALUES (%s, %s, %s)"
                " ON CONFLICT (run_id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()",
                (rid, MODULE_NAME, json.dumps(run, default=str))
            )
        conn.commit()
    except Exception:
        _ = None  # best-effort
    finally:
        conn.close()


def _pg_load_workflow_runs() -> list[Dict[str, Any]]:
    conn = _pg_connection()
    if not conn:
        return []
    try:
        _pg_ensure_runs_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM apg_workflow_runs WHERE module_name = %s", (MODULE_NAME,))
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _pg_ensure_records_table(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS apg_records ("
                "  id TEXT NOT NULL,"
                "  collection TEXT NOT NULL,"
                "  tenant_id TEXT NOT NULL DEFAULT 'default',"
                "  data JSONB NOT NULL,"
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  PRIMARY KEY (collection, id)"
                ")"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_apg_records_tenant"
                " ON apg_records (collection, tenant_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_apg_records_gin"
                " ON apg_records USING gin (data)"
            )
        conn.commit()
    except Exception:
        _ = None  # best-effort


def _pg_save_entity_records(entity_name: str, records: list[Dict[str, Any]]) -> None:
    conn = _pg_connection()
    if not conn:
        return
    try:
        _pg_ensure_records_table(conn)
        with conn.cursor() as cur:
            for record in records:
                rid = str(record.get("id", ""))
                if not rid:
                    continue
                tenant_id = str(record.get("tenant_id") or APG_TENANT_DEFAULT)
                cur.execute(
                    "INSERT INTO apg_records (id, collection, tenant_id, data)"
                    " VALUES (%s, %s, %s, %s::jsonb)"
                    " ON CONFLICT (collection, id)"
                    " DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()",
                    (rid, entity_name.lower(), tenant_id, json.dumps(record, default=str))
                )
        conn.commit()
    except Exception:
        _ = None  # best-effort
    finally:
        conn.close()


def _pg_load_entity_records(entity_name: str, tenant_scoped: bool = True) -> list[Dict[str, Any]]:
    conn = _pg_connection()
    if not conn:
        return []
    try:
        _pg_ensure_records_table(conn)
        with conn.cursor() as cur:
            sql = "SELECT data FROM apg_records WHERE collection = %s"
            params: list[Any] = [entity_name.lower()]
            if tenant_scoped and _tenant_scope_enabled(entity_name) and not _tenant_admin_bypass():
                sql += " AND tenant_id = %s"
                params.append(_tenant_id() or APG_TENANT_DEFAULT)
            sql += " ORDER BY created_at"
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [json.loads(row[0]) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _apg_request_id() -> str:
    try:
        return str(getattr(_flask_g, "request_id", "") or "")
    except RuntimeError:
        return ""


def _apg_deliver_webhook(event, entity, record_id, data, req_id):
    urls = [u.strip() for u in os.environ.get("APG_WEBHOOK_URL", "").split(",") if u.strip()]
    if not urls:
        return
    secret = os.environ.get("APG_WEBHOOK_SECRET", "").encode()
    payload = json.dumps({{"event": event, "entity": entity, "id": str(record_id), "data": data, "ts": _datetime.datetime.utcnow().isoformat() + "Z", "req_id": req_id}}).encode()
    sig = "sha256=" + _hmac_mod.new(secret, payload, _hashlib_mod.sha256).hexdigest() if secret else ""

    def _send():
        for url in urls:
            for delay in (0, 1, 2, 4):
                try:
                    if delay:
                        _time_mod.sleep(delay)
                    req = _urllib_request.Request(
                        url,
                        data=payload,
                        method="POST",
                        headers={{"Content-Type": "application/json", "X-APG-Signature": sig, "X-APG-Event": event}},
                    )
                    with _urllib_request.urlopen(req, timeout=5):
                        pass
                    break
                except Exception as _e:
                    _logging.getLogger("apg").warning("webhook delivery failed url=%s err=%s", url, _e)

    _threading_mod.Thread(target=_send, daemon=True).start()


def _apg_job_timestamp() -> str:
    return _datetime.datetime.utcnow().isoformat() + "Z"


def _apg_init_job_store() -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    with _APG_JOB_LOCK:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS apg_jobs ("
            "id TEXT PRIMARY KEY,"
            "type TEXT NOT NULL,"
            "payload TEXT NOT NULL,"
            "status TEXT NOT NULL,"
            "created_at TEXT NOT NULL,"
            "started_at TEXT,"
            "finished_at TEXT,"
            "attempts INT NOT NULL DEFAULT 0,"
            "last_error TEXT"
            ")"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_apg_jobs_status ON apg_jobs(status, created_at)")
        conn.commit()


def _apg_job_from_row(row: Any) -> Dict[str, Any]:
    payload_raw = row["payload"] if "payload" in row.keys() else "{{}}"
    try:
        payload = json.loads(payload_raw or "{{}}")
    except (TypeError, json.JSONDecodeError):
        payload = {{}}
    return {{
        "id": str(row["id"]),
        "type": str(row["type"]),
        "payload": payload if isinstance(payload, dict) else {{"value": payload}},
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "attempts": int(row["attempts"] or 0),
        "last_error": row["last_error"],
    }}


def _apg_job_payload_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _apg_enqueue_job_dict(job: Dict[str, Any]) -> None:
    with _APG_JOB_LOCK:
        _APG_JOB_QUEUE.append(dict(job))


def _apg_load_pending_jobs() -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    with _APG_JOB_LOCK:
        conn.execute("UPDATE apg_jobs SET status='pending', started_at=NULL WHERE status='running'")
        rows = conn.execute(
            "SELECT * FROM apg_jobs WHERE status='pending' ORDER BY created_at"
        ).fetchall()
        _APG_JOB_QUEUE.clear()
        for row in rows:
            _APG_JOB_QUEUE.append(_apg_job_from_row(row))
        conn.commit()


def _apg_create_job(job_type: str, payload: Dict[str, Any] | None = None) -> tuple[int, Dict[str, Any]]:
    job_type = str(job_type or "").strip()
    if not job_type:
        return 400, {{"error": "missing_job_type"}}
    if job_type not in _APG_JOB_HANDLERS:
        return 400, {{"error": "unknown_job_type", "type": job_type}}
    payload = payload if isinstance(payload, dict) else {{}}
    conn = _sqlite_connection()
    if conn is None:
        return 500, {{"error": "jobs_unavailable"}}
    job = {{
        "id": str(_uuid.uuid4()),
        "type": job_type,
        "payload": dict(payload),
        "status": "pending",
        "created_at": _apg_job_timestamp(),
        "attempts": 0,
        "last_error": None,
    }}
    with _APG_JOB_LOCK:
        conn.execute(
            "INSERT INTO apg_jobs (id, type, payload, status, created_at, attempts, last_error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job["id"],
                job["type"],
                _apg_job_payload_json(job["payload"]),
                job["status"],
                job["created_at"],
                job["attempts"],
                job["last_error"],
            ),
        )
        conn.commit()
        _APG_JOB_QUEUE.append(dict(job))
    return 201, {{"job_id": job["id"]}}


def _apg_get_job(job_id: str) -> Dict[str, Any] | None:
    conn = _sqlite_connection()
    if conn is None:
        return None
    with _APG_JOB_LOCK:
        row = conn.execute("SELECT * FROM apg_jobs WHERE id=?", (str(job_id),)).fetchone()
    return _apg_job_from_row(row) if row is not None else None


def _apg_list_jobs(status: str | None = None, limit: int = 50) -> list[Dict[str, Any]]:
    conn = _sqlite_connection()
    if conn is None:
        return []
    limit = max(1, min(int(limit or 50), 500))
    with _APG_JOB_LOCK:
        if status:
            rows = conn.execute(
                "SELECT * FROM apg_jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM apg_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_apg_job_from_row(row) for row in rows]


def _apg_dequeue_job() -> Dict[str, Any] | None:
    conn = _sqlite_connection()
    if conn is None:
        return None
    with _APG_JOB_LOCK:
        while _APG_JOB_QUEUE:
            queued = dict(_APG_JOB_QUEUE.popleft())
            job_id = str(queued.get("id", ""))
            row = conn.execute("SELECT * FROM apg_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                continue
            job = _apg_job_from_row(row)
            if job.get("status") != "pending":
                continue
            attempts = int(job.get("attempts") or 0) + 1
            started_at = _apg_job_timestamp()
            conn.execute(
                "UPDATE apg_jobs SET status='running', started_at=?, finished_at=NULL, attempts=?, last_error=NULL WHERE id=?",
                (started_at, attempts, job_id),
            )
            conn.commit()
            job.update({{"status": "running", "started_at": started_at, "finished_at": None, "attempts": attempts, "last_error": None}})
            return job
    return None


def _apg_finish_job(job_id: str, status: str, last_error: str | None = None) -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    finished_at = _apg_job_timestamp() if status in {{"done", "failed"}} else None
    with _APG_JOB_LOCK:
        conn.execute(
            "UPDATE apg_jobs SET status=?, finished_at=?, last_error=? WHERE id=?",
            (status, finished_at, last_error, str(job_id)),
        )
        conn.commit()


def _apg_reschedule_job(job: Dict[str, Any], error: str) -> None:
    conn = _sqlite_connection()
    if conn is None:
        return
    job_id = str(job.get("id", ""))
    with _APG_JOB_LOCK:
        conn.execute(
            "UPDATE apg_jobs SET status='pending', finished_at=NULL, last_error=? WHERE id=?",
            (error, job_id),
        )
        conn.commit()
    delay = min(60.0, float(2 ** max(0, int(job.get("attempts") or 1) - 1)))
    if delay > 0:
        _time.sleep(delay)
    current = _apg_get_job(job_id)
    if current is not None and current.get("status") == "pending":
        _apg_enqueue_job_dict(current)


def _apg_job_worker_loop() -> None:
    while True:
        job = _apg_dequeue_job()
        if job is None:
            _time.sleep(0.1)
            continue
        handler = _APG_JOB_HANDLERS.get(str(job.get("type", "")))
        try:
            if handler is None:
                raise RuntimeError("unknown_job_type: " + str(job.get("type", "")))
            handler(dict(job.get("payload") or {{}}))
        except Exception as exc:
            last_error = str(exc)
            if int(job.get("attempts") or 0) < _APG_JOB_MAX_RETRIES:
                _apg_reschedule_job(job, last_error)
            else:
                _apg_finish_job(str(job.get("id", "")), "failed", last_error)
        else:
            _apg_finish_job(str(job.get("id", "")), "done", None)


def _apg_start_job_workers() -> None:
    global _APG_JOB_WORKERS_STARTED
    if _APG_JOB_WORKERS_STARTED:
        return
    _APG_JOB_WORKERS_STARTED = True
    for index in range(_APG_JOB_WORKER_THREADS):
        thread = threading.Thread(target=_apg_job_worker_loop, name=f"apg-job-worker-{{index + 1}}", daemon=True)
        _APG_JOB_WORKERS.append(thread)
        thread.start()


def _apg_echo_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    return dict(payload)


def _apg_webhook_job(payload: Dict[str, Any]) -> None:
    _apg_deliver_webhook(
        str(payload.get("event") or "job"),
        str(payload.get("entity") or "job"),
        payload.get("record_id", payload.get("id", "")),
        payload.get("data", payload.get("payload", payload)),
        str(payload.get("req_id") or ""),
    )


def _apg_email_job(payload: Dict[str, Any]) -> None:
    _apg_send_email(
        str(payload.get("to") or ""),
        str(payload.get("subject") or f"{{APG_APP_NAME}} job notification"),
        str(payload.get("body") or ""),
    )


def _apg_register_builtin_job_handlers() -> None:
    _APG_JOB_HANDLERS.setdefault("apg.echo", _apg_echo_job)
    _APG_JOB_HANDLERS.setdefault("apg.webhook", _apg_webhook_job)
    _APG_JOB_HANDLERS.setdefault("apg.email", _apg_email_job)


def _jobs_payload(query: Dict[str, list[str]] | None = None) -> tuple[int, Any]:
    query = query or {{}}
    raw_status = query.get("status", [None])[-1] if query.get("status") else None
    status = str(raw_status).strip() if raw_status not in (None, "") else None
    if status is not None and status not in {{"pending", "running", "done", "failed"}}:
        return 400, {{"error": "invalid_status", "allowed": ["pending", "running", "done", "failed"]}}
    raw_limit = query.get("limit", ["50"])[-1] if query.get("limit") else "50"
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = 50
    return 200, _apg_list_jobs(status, limit)


def _job_detail_payload(path: str) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 2:
        return 404, {{"error": "not_found", "path": path}}
    job = _apg_get_job(parts[1])
    if job is None:
        return 404, {{"error": "job_not_found", "id": parts[1]}}
    return 200, job


def _create_job_payload(payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    job_payload = payload.get("payload", {{}})
    if job_payload is None:
        job_payload = {{}}
    if not isinstance(job_payload, dict):
        return 400, {{"error": "payload_must_be_object"}}
    return _apg_create_job(str(payload.get("type", "")), job_payload)


def _retry_job_payload(path: str) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "jobs" or parts[2] != "retry":
        return 404, {{"error": "not_found", "path": path}}
    job = _apg_get_job(parts[1])
    if job is None:
        return 404, {{"error": "job_not_found", "id": parts[1]}}
    if job.get("status") != "failed":
        return 409, {{"error": "job_not_failed", "status": job.get("status")}}
    conn = _sqlite_connection()
    if conn is None:
        return 500, {{"error": "jobs_unavailable"}}
    with _APG_JOB_LOCK:
        conn.execute(
            "UPDATE apg_jobs SET status='pending', started_at=NULL, finished_at=NULL, attempts=0, last_error=NULL WHERE id=?",
            (parts[1],),
        )
        conn.commit()
    job = _apg_get_job(parts[1])
    if job is not None:
        _apg_enqueue_job_dict(job)
    return 202, {{"job_id": parts[1]}}


_apg_register_builtin_job_handlers()
_sqlite_init_database()
_apg_init_job_store()
_apg_load_pending_jobs()
_apg_start_job_workers()
_load_record_store()
_APG_OPENAPI_SPEC = _build_openapi_document()

_flask_app = _FlaskApp("app", root_path=os.path.abspath(os.path.dirname(globals().get("__file__", None) or ".")))
_flask_app.secret_key = _generated_session_secret()
_APG_SESSION_COOKIE_SAMESITE = _session_cookie_samesite()
_APG_SESSION_COOKIE_SECURE = (
    _env_flag("APG_SESSION_COOKIE_SECURE", _production_mode())
    or _APG_SESSION_COOKIE_SAMESITE == "None"
)
_flask_app.config.update(
    SESSION_COOKIE_NAME=os.environ.get("APG_SESSION_COOKIE_NAME", "apg_session"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=_APG_SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE=_APG_SESSION_COOKIE_SECURE,
    MAX_CONTENT_LENGTH=max(1, _env_int("APG_MAX_BODY_BYTES", 16 * 1024 * 1024)),
)
_validate_startup_configuration()


# Wave B ops hardening: structured JSON logging.
_APG_JSON_LOGS_ENABLED = _env_flag("APG_JSON_LOGS") or _production_mode()
_APG_LOGGER = _logging.getLogger("apg.generated")


class _APGJsonLogFormatter(_logging.Formatter):
    def format(self, record: _logging.LogRecord) -> str:
        req_id = str(getattr(record, "req_id", "") or "")
        method = str(getattr(record, "method", "") or "")
        path = str(getattr(record, "path", "") or "")
        try:
            req_id = req_id or str(getattr(_flask_g, "request_id", "") or "")
            method = method or str(getattr(_flask_request, "method", "") or "")
            path = path or str(getattr(_flask_request, "path", "") or "")
        except RuntimeError:
            _ = None
        try:
            status = int(getattr(record, "status", 0) or 0)
        except (TypeError, ValueError):
            status = 0
        try:
            ms = int(getattr(record, "ms", 0) or 0)
        except (TypeError, ValueError):
            ms = 0
        payload = {{
            "ts": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "msg": record.getMessage(),
            "req_id": req_id,
            "method": method,
            "path": path,
            "status": status,
            "ms": ms,
        }}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _configure_json_logging() -> None:
    if not _APG_JSON_LOGS_ENABLED:
        return
    handler = _logging.StreamHandler()
    handler.setFormatter(_APGJsonLogFormatter())
    for logger in (_logging.getLogger("werkzeug"), _flask_app.logger, _APG_LOGGER):
        logger.handlers = [handler]
        logger.setLevel(_logging.INFO)
        logger.propagate = False


_configure_json_logging()


# Wave B ops hardening: Prometheus text metrics.
_APG_METRICS_ENABLED = _env_flag("APG_METRICS")
_APG_METRICS_TOKEN = os.environ.get("APG_METRICS_TOKEN") or ""
_APG_METRICS_LOCK = _threading.Lock()
_APG_METRIC_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_APG_HTTP_REQUESTS_TOTAL: Dict[tuple[str, str, str], int] = {{}}
_APG_HTTP_REQUEST_DURATION: Dict[tuple[str, str], Dict[str, Any]] = {{}}
_APG_ACTIVE_REQUESTS = 0


def _apg_path_template(path: str | None = None) -> str:
    raw_path = str(path or _flask_request.path or "/").split("?", 1)[0].rstrip("/") or "/"
    parts = [part for part in raw_path.split("/") if part]
    if not parts:
        return "/"
    if len(parts) >= 3 and parts[0] == "records":
        parts[2] = ":id"
    if len(parts) >= 4 and parts[0] == "entities" and parts[2] == "records":
        parts[3] = ":id"
    if len(parts) >= 4 and parts[0] == "ui" and parts[1] == "entities":
        parts[3] = ":id"
    if len(parts) >= 3 and parts[0] == "workflows" and parts[1] == "runs":
        parts[2] = ":id"
    for index, value in enumerate(parts):
        if value.isdigit() or re.fullmatch(r"[0-9a-fA-F]{{8}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{12}}", value):
            parts[index] = ":id"
        elif re.fullmatch(r"(?:record|run|workflow-run|event)-[A-Za-z0-9_-]+", value):
            parts[index] = ":id"
    return "/" + "/".join(parts)


def _apg_metrics_inc_active() -> None:
    global _APG_ACTIVE_REQUESTS
    with _APG_METRICS_LOCK:
        _APG_ACTIVE_REQUESTS += 1


def _apg_metrics_dec_active() -> None:
    global _APG_ACTIVE_REQUESTS
    with _APG_METRICS_LOCK:
        _APG_ACTIVE_REQUESTS = max(0, _APG_ACTIVE_REQUESTS - 1)


def _apg_metrics_observe(method: str, path_template: str, status: int, duration_s: float) -> None:
    request_key = (method.upper(), path_template, str(int(status)))
    duration_key = (method.upper(), path_template)
    with _APG_METRICS_LOCK:
        _APG_HTTP_REQUESTS_TOTAL[request_key] = _APG_HTTP_REQUESTS_TOTAL.get(request_key, 0) + 1
        stats = _APG_HTTP_REQUEST_DURATION.setdefault(duration_key, {{"count": 0, "sum": 0.0, "buckets": [0 for _ in _APG_METRIC_BUCKETS]}})
        stats["count"] += 1
        stats["sum"] += max(0.0, float(duration_s))
        for index, bucket in enumerate(_APG_METRIC_BUCKETS):
            if duration_s <= bucket:
                stats["buckets"][index] += 1


def _apg_metric_escape(value: str) -> str:
    backslash = chr(92)
    return (
        str(value)
        .replace(backslash, backslash + backslash)
        .replace(chr(10), backslash + "n")
        .replace(chr(34), backslash + chr(34))
    )


def _apg_metric_labels(labels: Dict[str, str]) -> str:
    return ",".join(f'{{name}}="{{_apg_metric_escape(value)}}"' for name, value in labels.items())


def _apg_metrics_text() -> str:
    with _APG_METRICS_LOCK:
        request_counts = dict(_APG_HTTP_REQUESTS_TOTAL)
        durations = {{
            key: {{"count": value["count"], "sum": value["sum"], "buckets": list(value["buckets"])}}
            for key, value in _APG_HTTP_REQUEST_DURATION.items()
        }}
        active_requests = _APG_ACTIVE_REQUESTS
    lines = [
        "# HELP apg_http_requests_total Total HTTP requests handled by the generated APG app.",
        "# TYPE apg_http_requests_total counter",
    ]
    for (method, path_template, status), count in sorted(request_counts.items()):
        labels = _apg_metric_labels({{"method": method, "path_template": path_template, "status": status}})
        lines.append(f"apg_http_requests_total{{{{{{labels}}}}}} {{count}}")
    lines.extend([
        "# HELP apg_http_request_duration_seconds HTTP request latency in seconds.",
        "# TYPE apg_http_request_duration_seconds histogram",
    ])
    for (method, path_template), stats in sorted(durations.items()):
        base_labels = {{"method": method, "path_template": path_template}}
        for bucket, bucket_count in zip(_APG_METRIC_BUCKETS, stats["buckets"]):
            labels = _apg_metric_labels({{**base_labels, "le": ("%g" % bucket)}})
            lines.append(f"apg_http_request_duration_seconds_bucket{{{{{{labels}}}}}} {{bucket_count}}")
        labels = _apg_metric_labels({{**base_labels, "le": "+Inf"}})
        lines.append(f"apg_http_request_duration_seconds_bucket{{{{{{labels}}}}}} {{stats['count']}}")
        labels = _apg_metric_labels(base_labels)
        lines.append(f"apg_http_request_duration_seconds_count{{{{{{labels}}}}}} {{stats['count']}}")
        lines.append(f"apg_http_request_duration_seconds_sum{{{{{{labels}}}}}} {{stats['sum']:.6f}}")
    lines.extend([
        "# HELP apg_active_requests Active HTTP requests currently being processed.",
        "# TYPE apg_active_requests gauge",
        f"apg_active_requests {{active_requests}}",
        "",
    ])
    return "\\n".join(lines)


# Wave B ops hardening: X-Request-ID propagation and request lifecycle.
_APG_APP_START_MONOTONIC = _time.monotonic()
_APG_READY = False
_APG_OPS_QUIET_PATHS = frozenset({{"/livez", "/readyz"}})


@_flask_app.before_request
def _apg_ops_before_request() -> None:
    _flask_g.csp_nonce = base64.b64encode(os.urandom(16)).decode("ascii")
    request_id = str(_flask_request.headers.get("X-Request-ID") or "").strip() or str(_uuid.uuid4())
    _flask_g.request_id = request_id
    _flask_g.apg_request_started_at = _time.perf_counter()
    _flask_g.apg_path_template = _apg_path_template(_flask_request.path)
    if _APG_METRICS_ENABLED:
        _apg_metrics_inc_active()
    if _APG_JSON_LOGS_ENABLED and _flask_request.path not in _APG_OPS_QUIET_PATHS:
        _APG_LOGGER.info(
            "request_start",
            extra={{
                "req_id": request_id,
                "method": _flask_request.method,
                "path": _flask_request.path,
                "status": 0,
                "ms": 0,
            }},
        )
    return None


@_flask_app.after_request
def _apg_ops_after_request(response: _FlaskResponse) -> _FlaskResponse:
    global _APG_READY
    request_id = str(getattr(_flask_g, "request_id", "") or "").strip() or str(_uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    try:
        if "/search" in _flask_request.path and _apg_db_dialect() == "pg":
            response.headers["X-APG-FTS-Available"] = "false"
    except Exception:
        pass
    started_at = getattr(_flask_g, "apg_request_started_at", None)
    try:
        elapsed_s = max(0.0, _time.perf_counter() - float(started_at)) if started_at is not None else 0.0
    except (TypeError, ValueError):
        elapsed_s = 0.0
    status = int(response.status_code)
    path_template = str(getattr(_flask_g, "apg_path_template", "") or _apg_path_template(_flask_request.path))
    if _APG_METRICS_ENABLED:
        _apg_metrics_observe(_flask_request.method, path_template, status, elapsed_s)
        _apg_metrics_dec_active()
    if status < 400:
        _APG_READY = True
    if _APG_JSON_LOGS_ENABLED and _flask_request.path not in _APG_OPS_QUIET_PATHS:
        _APG_LOGGER.info(
            "request_finish",
            extra={{
                "req_id": request_id,
                "method": _flask_request.method,
                "path": _flask_request.path,
                "status": status,
                "ms": int(round(elapsed_s * 1000)),
            }},
        )
    return response


# Wave B ops hardening: Kubernetes liveness/readiness endpoints.
@_flask_app.route("/livez", methods=["GET"])
def _flask_livez() -> _FlaskResponse:
    payload = {{"status": "ok", "uptime_s": int(max(0.0, _time.monotonic() - _APG_APP_START_MONOTONIC))}}
    return _FlaskResponse(json.dumps(payload), status=200, content_type="application/json")


@_flask_app.route("/readyz", methods=["GET"])
def _flask_readyz() -> _FlaskResponse:
    if _APG_READY:
        return _FlaskResponse(json.dumps({{"status": "ready"}}), status=200, content_type="application/json")
    return _FlaskResponse(json.dumps({{"status": "starting"}}), status=503, content_type="application/json")


@_flask_app.route("/metrics", methods=["GET"])
def _flask_prometheus_metrics() -> _FlaskResponse:
    if not _APG_METRICS_ENABLED:
        status, payload = _route_payload("/metrics", {{k: v for k, v in _flask_request.args.lists()}})
        return _FlaskResponse(json.dumps(payload), status=status, content_type="application/json; charset=utf-8")
    if _APG_METRICS_TOKEN and not hmac.compare_digest(str(_flask_request.headers.get("X-Metrics-Token") or ""), _APG_METRICS_TOKEN):
        return _FlaskResponse("unauthorized\\n", status=401, content_type="text/plain; version=0.0.4; charset=utf-8")
    return _FlaskResponse(_apg_metrics_text(), status=200, content_type="text/plain; version=0.0.4; charset=utf-8")


def _content_security_policy() -> str:
    nonce = _csp_nonce()
    return (
        "default-src 'self'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        f"style-src 'self' 'nonce-{{nonce}}'; "
        f"script-src 'self' 'nonce-{{nonce}}'; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "manifest-src 'self'"
    )


_APG_SECURITY_HEADERS: Dict[str, str] = {{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}}


@_flask_app.after_request
def _apply_security_headers(response: _FlaskResponse) -> _FlaskResponse:
    if "Content-Security-Policy" not in response.headers:
        response.headers["Content-Security-Policy"] = _content_security_policy()
    for header_name, header_value in _APG_SECURITY_HEADERS.items():
        if header_name not in response.headers:
            response.headers[header_name] = header_value
    if _flask_request.is_secure and "Strict-Transport-Security" not in response.headers:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


# Wave C HTTP efficiency: cache semantics, conditional GET, and gzip.
_APG_GZIP_MIN_BYTES = 860
_APG_STATIC_CACHE_CONTROL = "public,max-age=31536000,immutable"
_APG_RECORDS_CACHE_CONTROL = "no-cache"
_APG_PRIVATE_CACHE_CONTROL = "no-store,private"


def _add_vary_header(response: _FlaskResponse, value: str) -> _FlaskResponse:
    existing = [
        item.strip()
        for item in str(response.headers.get("Vary", "")).split(",")
        if item.strip()
    ]
    existing_lower = {{item.lower() for item in existing}}
    if value.lower() not in existing_lower:
        existing.append(value)
        response.headers["Vary"] = ", ".join(existing)
    return response


def _request_accepts_gzip() -> bool:
    accepted = str(_flask_request.headers.get("Accept-Encoding", "")).lower()
    return any(part.split(";", 1)[0].strip() == "gzip" for part in accepted.split(","))


def _compressible_mimetype(response: _FlaskResponse) -> bool:
    mimetype = str(response.mimetype or "").lower()
    return mimetype.startswith("text/") or mimetype == "application/json"


def _maybe_compress(response: _FlaskResponse) -> _FlaskResponse:
    if _env_flag("APG_DISABLE_GZIP"):
        return response
    if not _request_accepts_gzip():
        return response
    if response.is_streamed or response.direct_passthrough:
        return response
    if response.headers.get("Content-Encoding"):
        return response
    status = int(response.status_code)
    if status < 200 or status >= 300:
        return response
    if not _compressible_mimetype(response):
        return response
    body = response.get_data()
    if len(body) <= _APG_GZIP_MIN_BYTES:
        return response
    compressed = gzip.compress(body)
    response.set_data(compressed)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = str(len(compressed))
    _add_vary_header(response, "Accept-Encoding")
    return response


def _is_records_get_response(response: _FlaskResponse) -> bool:
    if _flask_request.method != "GET":
        return False
    if int(response.status_code) != 200:
        return False
    if str(response.mimetype or "").lower() != "application/json":
        return False
    if _current_user() is not None:
        return False
    path = _flask_request.path.rstrip("/") or "/"
    return path.startswith("/records/")


def _maybe_apply_records_etag(response: _FlaskResponse) -> _FlaskResponse:
    if not _is_records_get_response(response):
        return response
    body = response.get_data()
    etag = '"' + hashlib.sha256(body).hexdigest()[:16] + '"'
    response.headers["ETag"] = etag
    requested = [
        item.strip()
        for item in str(_flask_request.headers.get("If-None-Match", "")).split(",")
        if item.strip()
    ]
    if etag in requested:
        response.status_code = 304
        response.set_data(b"")
        response.headers["ETag"] = etag
        response.headers.pop("Content-Length", None)
    return response


def _apply_cache_control(response: _FlaskResponse) -> _FlaskResponse:
    path = _flask_request.path.rstrip("/") or "/"
    if path.startswith("/static/") or path.startswith("/apg-static/"):
        response.headers["Cache-Control"] = _APG_STATIC_CACHE_CONTROL
    elif _flask_request.method == "GET" and (path == "/records" or path.startswith("/records/")):
        response.headers["Cache-Control"] = _APG_RECORDS_CACHE_CONTROL
    elif _flask_request.method == "GET" and (path == "/login" or path == "/ui" or path.startswith("/ui/")):
        response.headers["Cache-Control"] = _APG_PRIVATE_CACHE_CONTROL
    return response


@_flask_app.after_request
def _apg_http_efficiency_after_request(response: _FlaskResponse) -> _FlaskResponse:
    response = _apply_cache_control(response)
    response = _maybe_apply_records_etag(response)
    if int(response.status_code) == 304:
        return response
    return _maybe_compress(response)


def _error_response_wants_json() -> bool:
    if _flask_request.path.startswith("/api/") or _flask_request.path == "/api":
        return True
    if _flask_request.path.startswith("/records/") or _flask_request.path == "/records":
        return True
    accept = str(_flask_request.headers.get("Accept", ""))
    return "application/json" in accept and "text/html" not in accept


def _apg_error_response(
    status: int,
    error_code: str,
    title: str,
    message: str | None = None,
    extra_headers: Dict[str, str] | None = None,
) -> _FlaskResponse:
    message = message if message is not None else title
    if _error_response_wants_json():
        response = _FlaskResponse(
            json.dumps({{"error": error_code, "message": message, "path": _flask_request.path}}),
            status=status,
            content_type="application/json; charset=utf-8",
        )
    else:
        body = (
            '<section class="apg-error-page" role="alert">'
            f"<h1>{{html.escape(title)}}</h1>"
            f"<p>{{html.escape(message)}}</p>"
            '<p><a href="/">Return to the home page</a></p>'
            "</section>"
        )
        response = _FlaskResponse(_html_page(title, body, shell=False), status=status, content_type="text/html; charset=utf-8")
    for header_name, header_value in (extra_headers or {{}}).items():
        response.headers[header_name] = header_value
    return response


@_flask_app.errorhandler(404)
def _apg_not_found(_error: Any) -> _FlaskResponse:
    return _apg_error_response(404, "not_found", "Page not found", "The requested path does not exist in this generated app.")


@_flask_app.errorhandler(405)
def _apg_method_not_allowed(_error: Any) -> _FlaskResponse:
    return _apg_error_response(405, "method_not_allowed", "Method not allowed", "That HTTP method is not supported for this path.")


@_flask_app.errorhandler(413)
def _apg_payload_too_large(_error: Any) -> _FlaskResponse:
    return _apg_error_response(413, "payload_too_large", "Payload too large", "The request body exceeds the configured APG_MAX_BODY_BYTES limit.")


@_flask_app.errorhandler(500)
def _apg_internal_error(_error: Any) -> _FlaskResponse:
    return _apg_error_response(500, "internal_error", "Something went wrong", "The generated app hit an unexpected error. Details were logged server-side.")


# Wave F app hardening: request rate limits, JSON guard, and audit events.
_APG_RATE_BUCKETS = {{}}
_APG_RATE_LOCK = _threading.Lock()
_APG_RATE_EXEMPT_PATHS = ("/livez", "/readyz", "/metrics")


def _apg_rate_is_authenticated() -> bool:
    try:
        return _current_user() is not None or _has_header_auth(_flask_request.headers)
    except RuntimeError:
        return False


def _apg_rate_limit_guard() -> _FlaskResponse | None:
    if _flask_request.path in _APG_RATE_EXEMPT_PATHS:
        return None
    anon_limit = max(1, _env_int("APG_RATE_LIMIT_ANON", 100))
    auth_limit = max(1, _env_int("APG_RATE_LIMIT_AUTH", 1000))
    limit = auth_limit if _apg_rate_is_authenticated() else anon_limit
    refill_per_second = float(limit) / 60.0
    now = _time.monotonic()
    ip = str(_flask_request.remote_addr or "unknown")
    with _APG_RATE_LOCK:
        bucket = _APG_RATE_BUCKETS.get(ip, [float(limit), now])
        tokens = min(float(limit), float(bucket[0]) + max(0.0, now - float(bucket[1])) * refill_per_second)
        if tokens < 1.0:
            _APG_RATE_BUCKETS[ip] = [tokens, now]
            return _apg_error_response(
                429,
                "rate_limited",
                "Too many requests",
                extra_headers={{"Retry-After": "60"}},
            )
        _APG_RATE_BUCKETS[ip] = [tokens - 1.0, now]
    return None


def _apg_content_type_guard() -> _FlaskResponse | None:
    if _flask_request.method not in ("POST", "PUT"):
        return None
    if not _flask_request.path.startswith("/records/"):
        return None
    content_type = _flask_request.content_type or ""
    allowed = ("application/json", "application/x-www-form-urlencoded", "multipart/form-data")
    if content_type and not any(content_type.startswith(item) for item in allowed):
        return _apg_error_response(415, "unsupported_media_type", "Content-Type must be application/json, form-urlencoded, or multipart/form-data")
    return None


def _audited_record_mutation() -> bool:
    return _flask_request.method in ("POST", "PUT", "DELETE") and _flask_request.path.startswith("/records/")


@_flask_app.after_request
def _apg_audit_after_request(response: _FlaskResponse) -> _FlaskResponse:
    if _audited_record_mutation():
        entity = _apg_audit_entity_from_path(_flask_request.path)
        action = {{"POST": "create", "PUT": "update", "DELETE": "delete"}}[_flask_request.method]
        _apg_audit_event(action, entity=entity)
    return response


@_flask_app.before_request
def _setup_tenant() -> Any:
    body_limit = _flask_app.config.get("MAX_CONTENT_LENGTH")
    if body_limit and (_flask_request.content_length or 0) > body_limit:
        return _apg_error_response(413, "payload_too_large", "Payload too large", "The request body exceeds the configured APG_MAX_BODY_BYTES limit.")
    path = _flask_request.path.rstrip("/") or "/"
    tid = _tenant_header_value()
    if (
        APG_MULTI_TENANT_ENABLED
        and _production_mode()
        and not tid
        and not _tenant_admin_bypass()
        and not path.startswith("/locales/")
    ):
        return _FlaskResponse(
            json.dumps({{"error": "tenant_required"}}),
            status=400,
            content_type="application/json; charset=utf-8",
        )
    _TENANT_LOCAL.tenant_id = tid or (APG_TENANT_DEFAULT if APG_MULTI_TENANT_ENABLED else None)
    if _login_required_for_path(_flask_request.path) and _current_user() is None:
        return _flask_redirect("/login?next=" + quote(_flask_request.full_path.rstrip("?") or "/ui", safe="/?=&%"))
    content_type_error = _apg_content_type_guard()
    if content_type_error is not None:
        return content_type_error
    rate_limited = _apg_rate_limit_guard()
    if rate_limited is not None:
        return rate_limited
    return None


def _check_mutation_auth():
    csrf_err = _check_csrf_token()
    if csrf_err:
        return csrf_err
    if _authorized(_flask_request.headers):
        return None
    status, response = _auth_failure_payload()
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/", methods=["GET"])
@_flask_app.route("/home", methods=["GET"])
def _flask_home():
    return _FlaskResponse(_landing_page_html(), content_type="text/html; charset=utf-8")


@_flask_app.route("/theme.css", methods=["GET"])
def _flask_theme():
    return _FlaskResponse(theme_stylesheet(), content_type="text/css; charset=utf-8")


@_flask_app.route("/locales/<lang>.json", methods=["GET"])
def _flask_locale_file(lang: str):
    payload = _locale_payload(str(lang))
    if payload is None:
        return _FlaskResponse(json.dumps({{"error": "locale_not_found", "locale": str(lang)}}), status=404, content_type="application/json; charset=utf-8")
    return _FlaskResponse(json.dumps(payload, sort_keys=True), content_type="application/json; charset=utf-8")


@_flask_app.route("/login", methods=["GET"])
def _flask_login_get():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({{"error": "not_found", "path": "/login"}}), status=404, content_type="application/json; charset=utf-8")
    next_url = _flask_request.args.get("next") or "/ui"
    if not str(next_url).startswith("/"):
        next_url = "/ui"
    if _current_user() is not None:
        return _flask_redirect(next_url)
    return _FlaskResponse(_login_page(next_url=next_url), content_type="text/html; charset=utf-8")


@_flask_app.route("/login", methods=["POST"])
def _flask_login_post():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({{"error": "not_found", "path": "/login"}}), status=404, content_type="application/json; charset=utf-8")
    csrf_err = _check_csrf_token()
    if csrf_err:
        return csrf_err
    username = str(_flask_request.form.get("username") or "")
    password = str(_flask_request.form.get("password") or "")
    next_url = str(_flask_request.form.get("next") or "/ui")
    if not next_url.startswith("/"):
        next_url = "/ui"
    throttle_key = _login_throttle_key(username)
    retry_after = _login_retry_after(throttle_key)
    if retry_after:
        _apg_audit_event("login_failed", user=username or "api")
        response = _FlaskResponse(
            _login_page("Too many sign-in attempts. Wait a moment and try again.", next_url, username=username),
            status=429,
            content_type="text/html; charset=utf-8",
        )
        response.headers["Retry-After"] = str(retry_after)
        return response
    user = _authenticate_user(username, password)
    if user is None:
        _register_login_failure(throttle_key)
        _apg_audit_event("login_failed", user=username or "api")
        return _FlaskResponse(
            _login_page("We could not sign you in with those credentials.", next_url, username=username),
            status=401,
            content_type="text/html; charset=utf-8",
        )
    _clear_login_failures(throttle_key)
    _issue_login_session(user)
    _apg_audit_event("login", user=str(user.get("username", username)))
    if _env_flag("APG_EMAIL_ON_LOGIN") and str(user.get("email", "") or "").strip():
        _apg_send_email(
            str(user.get("email", "")),
            f"New login to {{APG_APP_NAME}}",
            "A successful login was recorded for "
            + str(user.get("username", username))
            + " at "
            + _datetime.datetime.utcnow().isoformat()
            + "Z.",
        )
    return _flask_redirect(next_url)


@_flask_app.route("/logout", methods=["POST"])
def _flask_logout_post():
    if not APG_AUTH_REQUIRED:
        return _FlaskResponse(json.dumps({{"error": "not_found", "path": "/logout"}}), status=404, content_type="application/json; charset=utf-8")
    csrf_err = _check_csrf_token()
    if csrf_err:
        return csrf_err
    user = _current_user()
    _apg_audit_event("logout", user=str(user.get("username", "")) if isinstance(user, dict) else "api")
    _flask_session.pop("apg_user", None)
    return _flask_redirect("/login")


@_flask_app.route("/locale", methods=["POST"])
def _flask_locale_post():
    csrf_err = _check_csrf_token()
    if csrf_err:
        return csrf_err
    language = str(_flask_request.form.get("lang") or APG_DEFAULT_LANGUAGE)
    if language not in APG_SUPPORTED_LANGUAGES:
        language = APG_DEFAULT_LANGUAGE
    next_url = str(_flask_request.form.get("next") or "/ui")
    if not next_url.startswith("/"):
        next_url = "/ui"
    response = _flask_redirect(next_url)
    response.set_cookie("apg_lang", language, max_age=31536000, samesite="Lax")
    return response


@_flask_app.route("/api-docs", methods=["GET"])
def _flask_api_docs():
    if not _env_flag("APG_SWAGGER_UI"):
        return _apg_error_response(404, "not_found", "Page not found", "Swagger UI is not enabled for this generated app.")
    html_doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>APG API Docs</title>'
        '<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css">'
        '</head><body><div id="swagger-ui"></div>'
        '<script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>'
        '<script>SwaggerUIBundle({{url:"/openapi.json",dom_id:"#swagger-ui"}});</script>'
        '</body></html>'
    )
    response = _FlaskResponse(html_doc, content_type="text/html; charset=utf-8")
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "img-src 'self' data: https://unpkg.com; "
        "font-src 'self' data: https://unpkg.com; "
        "connect-src 'self'"
    )
    return response


@_flask_app.route("/entities/<entity_name>/records.csv", methods=["GET"])
def _flask_csv_export(entity_name):
    return _FlaskResponse(_csv_export_body(entity_name), content_type="text/csv; charset=utf-8")


@_flask_app.route("/uploads/<entity_name>/<filename>", methods=["GET"])
def _flask_uploaded_file(entity_name, filename):
    return _uploaded_file_response(str(entity_name), str(filename))


@_flask_app.route("/ui", methods=["GET"])
@_flask_app.route("/ui/", methods=["GET"])
@_flask_app.route("/ui/<path:subpath>", methods=["GET"])
def _flask_ui_get(subpath=""):
    path = "/ui/" + subpath if subpath else "/ui"
    query = {{k: v for k, v in _flask_request.args.lists()}}
    status, html_payload = _ui_payload(path, query)
    return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")


@_flask_app.route("/ui", methods=["POST"])
@_flask_app.route("/ui/", methods=["POST"])
@_flask_app.route("/ui/<path:subpath>", methods=["POST"])
def _flask_ui_post(subpath=""):
    path = "/ui/" + subpath if subpath else "/ui"
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    if _flask_request.content_type and "application/x-www-form-urlencoded" in _flask_request.content_type:
        payload = {{"record": _flask_request.form.to_dict(flat=True)}}
    else:
        try:
            payload = _flask_request.get_json(force=True, silent=False) or {{}}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as _e:
            return _FlaskResponse(
                json.dumps({{"error": "invalid_json", "message": str(_e)}}),
                status=400, content_type="application/json; charset=utf-8",
            )
    status, response = _ui_post_payload(path, payload)
    if status in {{302, 303}}:
        return _flask_redirect(str(response["location"]), code=status)
    if "html" in response:
        _r = _FlaskResponse(str(response["html"]), status=status, content_type="text/html; charset=utf-8")
        if response.get("hx_trigger"):
            _r.headers["HX-Trigger"] = json.dumps(response["hx_trigger"])
        return _r
    return _FlaskResponse(_ui_error_payload(path, response), status=status, content_type="text/html; charset=utf-8")


def _records_csv_requested(path: str, query: Dict[str, list[str]]) -> bool:
    return _records_api_collection_path(path) and str(_record_query_value(query, "format", "") or "").lower() == "csv"


def _records_next_link(next_cursor: Any) -> str:
    args: list[tuple[str, str]] = []
    for key, values in _flask_request.args.lists():
        if key == "after":
            continue
        for value in values:
            args.append((key, value))
    args.append(("after", str(next_cursor)))
    return f'<{{_flask_request.base_url}}?{{urlencode(args)}}>; rel="next"'


def _records_response_headers(path: str, payload: Dict[str, Any]) -> Dict[str, str]:
    if not _records_api_collection_path(path):
        return {{}}
    next_cursor = payload.get("next_cursor")
    if next_cursor in (None, ""):
        return {{}}
    return {{"Link": _records_next_link(next_cursor)}}


def _records_response_payload(path: str, payload: Dict[str, Any]) -> Any:
    if not _records_api_collection_path(path):
        return payload
    compat = str(_flask_request.headers.get("X-APG-Compat", "")).strip().lower()
    if compat == "v1":
        return payload.get("data", [])
    return payload


def _records_csv_response(path: str, query: Dict[str, list[str]]) -> _FlaskResponse:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is not None:
        return _FlaskResponse(
            json.dumps({{"error": "not_found", "path": path}}),
            status=404,
            content_type="application/json; charset=utf-8",
        )
    entity_name = str(route["entity"])
    if entity_name not in ENTITY_NAMES:
        return _FlaskResponse(
            json.dumps({{"error": "unknown_entity", "entity": entity_name}}),
            status=404,
            content_type="application/json; charset=utf-8",
        )
    result = query_records(entity_name, query, response_style="records", paginate=False)
    if result.get("error") == "invalid_field":
        return _FlaskResponse(
            json.dumps({{"error": "invalid_field"}}),
            status=400,
            content_type="application/json; charset=utf-8",
        )
    safe_entity = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in entity_name) or "records"
    filename = f"{{safe_entity}}_{{_datetime.date.today().isoformat()}}.csv"
    return _FlaskResponse(
        _csv_export_body(entity_name, records=result.get("data", [])),
        status=200,
        content_type="text/csv; charset=utf-8",
        headers={{"Content-Disposition": f'attachment; filename="{{filename}}"'}},
    )


_APG_GET_PUBLIC = frozenset({{"/health", "/auth", "/openapi.json", "/metrics", "/describe"}})


@_flask_app.route("/<path:api_path>", methods=["GET"])
def _flask_api_get(api_path):
    path = "/" + api_path
    if path not in _APG_GET_PUBLIC:
        auth_err = _check_mutation_auth()
        if auth_err:
            return auth_err
    if path == "/events" and (
        "text/event-stream" in (_flask_request.headers.get("Accept") or "")
        or _flask_request.args.get("topics") is not None
    ):
        return _FlaskResponse(
            _sse_stream(_flask_request.args.get("topics")),
            content_type="text/event-stream; charset=utf-8",
            headers={{"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}},
        )
    if _capability_screen(path) is not None:
        status, html_payload = _capability_screen_payload(path)
        return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")
    if _application_screen(path) is not None:
        status, html_payload = _application_screen_payload(path)
        return _FlaskResponse(html_payload, status=status, content_type="text/html; charset=utf-8")
    query = {{k: v for k, v in _flask_request.args.lists()}}
    if _records_csv_requested(path, query):
        return _records_csv_response(path, query)
    status, payload = _route_payload(path, query)
    if status == 404 and "text/html" in str(_flask_request.headers.get("Accept", "")):
        return _apg_error_response(404, "not_found", "Page not found", "The requested path does not exist in this generated app.")
    response_payload = _records_response_payload(path, payload) if status == 200 else payload
    response = _FlaskResponse(json.dumps(response_payload), status=status, content_type="application/json; charset=utf-8")
    if status == 200 and isinstance(payload, dict):
        for header_name, header_value in _records_response_headers(path, payload).items():
            response.headers[header_name] = header_value
    return response


@_flask_app.route("/<path:api_path>", methods=["POST"])
def _flask_api_post(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    ct = _flask_request.content_type or ""
    if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
        payload = _flask_request.form.to_dict(flat=True)
    else:
        try:
            payload = _flask_request.get_json(force=True, silent=False) or {{}}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as _e:
            return _FlaskResponse(
                json.dumps({{"error": "invalid_json", "message": str(_e)}}),
                status=400, content_type="application/json; charset=utf-8",
            )
    status, response = _post_payload(path, payload)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/<path:api_path>", methods=["PUT"])
def _flask_api_put(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    ct = _flask_request.content_type or ""
    if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
        payload = _flask_request.form.to_dict(flat=True)
    else:
        try:
            payload = _flask_request.get_json(force=True, silent=False) or {{}}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except Exception as _e:
            return _FlaskResponse(
                json.dumps({{"error": "invalid_json", "message": str(_e)}}),
                status=400, content_type="application/json; charset=utf-8",
            )
    status, response = _put_payload(path, payload)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


@_flask_app.route("/<path:api_path>", methods=["DELETE"])
def _flask_api_delete(api_path):
    path = "/" + api_path
    auth_err = _check_mutation_auth()
    if auth_err:
        return auth_err
    status, response = _delete_record_payload(path)
    return _FlaskResponse(json.dumps(response), status=status, content_type="application/json; charset=utf-8")


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
    debug = os.environ.get("APG_DEBUG") == "1"
    print(f"{{MODULE_NAME}} listening on {{resolved_host}}:{{resolved_port}}", flush=True)
    _flask_app.run(host=resolved_host, port=resolved_port, debug=debug, use_reloader=False)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--describe" in args:
        print(json.dumps(describe_application(), indent=2, sort_keys=True))
        return
    if "--semantic-model" in args:
        print(json.dumps(semantic_model(), indent=2, sort_keys=True))
        return
    if "--validate" in args:
        report = validate_application()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["valid"] else 1)
    if "--self-test" in args:
        report = self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)
    host = _arg_value(args, "--host", os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1")
    port = _arg_value(args, "--port", os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    run_server(host, port)


if __name__ == "__main__":
    main()
'''

	def _generate_python_requirements(self) -> str:
		"""Generate requirements for the APG Python target."""
		return """# APG generated Python application requirements
# standard library: json, uuid, dataclasses, typing (no install needed)

flask>=3.0,<4
PyJWT>=2.8,<3
"""

	def _generate_python_dockerfile(self, module: ModuleDeclaration) -> str:
		"""Generate a minimal container image for the dependency-free Python app."""
		return f"""FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APG_HOST=0.0.0.0
ENV APG_PORT=8080

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
  CMD python app.py --self-test >/tmp/{module.name}_self_test.json || exit 1

CMD ["python", "app.py"]
"""

	def _generate_python_dockerignore(self) -> str:
		"""Generate container build exclusions for generated apps."""
		return """.git
.venv
__pycache__/
*.pyc
*.pyo
.DS_Store
.pytest_cache/
*.json.tmp
"""

	def _generate_python_env_example(self, module: ModuleDeclaration) -> str:
		"""Generate documented runtime environment defaults."""
		safe_name = module.name.replace(" ", "_").lower()
		return f"""# APG generated app runtime configuration
APG_HOST=127.0.0.1
APG_PORT=8080

# Optional JSON persistence path.
# APG_DATA_FILE=./data/{safe_name}.json

# Optional mutation API key.
# APG_API_KEY=change-me

# Set to 1 to enable HTTP request logging.
APG_DEBUG=0
"""

	def _generate_python_smoke_test(self) -> str:
		"""Generate a standalone smoke test for generated app artifacts."""
		return '''"""Standalone smoke test for an APG generated Python application."""

from __future__ import annotations

import json

import app


def main() -> int:
    report = app.self_test()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        return 1
    validation = report["checks"]["validation"]["checks"]
    for check_name in ("openapi_contract", "component_manifest", "route_dispatch"):
        errors = validation.get(check_name, {}).get("errors", [])
        if errors:
            print(json.dumps({"contract": check_name, "errors": errors}, indent=2, sort_keys=True))
            return 1
    capability_health = report["checks"].get("capability_health")
    if capability_health is not None and capability_health.get("healthy") is not True:
        print(json.dumps({"capability_health": capability_health}, indent=2, sort_keys=True))
        return 1
    component = app.component_manifest()
    required_routes = {"/health", "/self-test", "/component.json", "/semantic-model.json", "/openapi.json"}
    missing_routes = sorted(required_routes.difference(component["interfaces"]["http"]["paths"]))
    if missing_routes:
        print(json.dumps({"missing_routes": missing_routes}, indent=2, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

	def _generate_python_readme(self, module: ModuleDeclaration) -> str:
		"""Generate a runbook for the dependency-free Python target."""
		agents = [entity for entity in module.entities if isinstance(entity, AIAgentDeclaration)]
		teams = [entity for entity in module.entities if isinstance(entity, AgentTeamDeclaration)]
		capabilities = [entity for entity in module.entities if isinstance(entity, CapabilityDeclaration)]
		databases = [entity for entity in module.entities if isinstance(entity, DatabaseDeclaration)]
		entities = [
			entity for entity in module.entities
			if not isinstance(entity, (AIAgentDeclaration, AgentTeamDeclaration, CapabilityDeclaration))
		]

		lines = [
			f"# {module.name}",
			"",
			"Dependency-free APG generated Python application.",
			"",
			"## Run",
			"",
			"```bash",
			"python app.py",
			"```",
			"",
			"## Verify",
			"",
			"```bash",
			"python app.py --self-test",
			"python smoke_test.py",
			"python app.py --describe",
			"python app.py --semantic-model",
			"python app.py --validate",
			"```",
			"",
			"## Core HTTP endpoints",
			"",
			"- `GET /health` - runtime health and validation summary",
			"- `GET /component.json` - composable application component manifest",
			"- `GET /semantic-model.json` - normalized APG semantic model",
			"- `GET /self-test` - generated app smoke contract",
			"- `GET /manifest` - application manifest",
			"- `GET /openapi.json` - OpenAPI 3.1 contract",
			"- `GET /metrics` - runtime metrics snapshot",
			"- `GET /ui` - generated HTML application index",
			"",
			"## Browser UI",
			"",
			"- Open the generated browser interface at `/ui` after starting `python app.py`.",
			"- Entity screens include dependency-free create, edit, delete, and validation-error flows.",
			"- Typed APG fields render as matching HTML controls and are coerced before validation.",
			"- Record edits and deletes use `_revision` checks to avoid overwriting stale browser forms.",
			"",
			"## Data records",
			"",
			"- `GET /records` - all records grouped by entity",
			"- `GET /entities/{Entity}/records` - query records for an entity",
			"- `POST /entities/{Entity}/records` - create a record",
			"- `PUT /entities/{Entity}/records/{id}` - update a record",
			"- `DELETE /entities/{Entity}/records/{id}` - delete a record",
			"- `GET /entities/{Entity}/records/export` - export records",
			"- `POST /entities/{Entity}/records/import` - import records",
			"",
			"Python package helpers: `create_record()`, `get_record()`, `query_records()`, `update_record()`, and `delete_record()` expose the same executable record behavior for composition.",
			"",
			"Set `APG_DATA_FILE=/path/to/data.json` to persist records to JSON.",
			"Set `APG_API_KEY=<key>` to require an API key for mutations.",
			"",
			"## Deployment",
			"",
			"```bash",
			"docker build -t apg-generated-app .",
			"docker run --rm -p 8080:8080 --env-file .env.example apg-generated-app",
			"```",
			"",
			"Generated deployment artifacts:",
			"",
			"- `Dockerfile` - Flask 3.x container entrypoint",
			"- `.dockerignore` - container build exclusions",
			"- `.env.example` - documented runtime environment variables",
			"- `semantic_model.json` - normalized APG semantic model for IDEs, agents, and release checks",
			"- `smoke_test.py` - standalone generated app smoke test",
		]

		if entities:
			lines.extend(["", "## Entities", ""])
			for entity in entities:
				lines.append(f"- `{entity.name}`")

		if databases:
			lines.extend([
				"",
				"## Databases",
				"",
				"- `GET /databases` - database catalog with connection and schema metadata",
				"- `GET /databases/status` - database validation status and schema/reference counts",
				"- `GET /databases/{Database}/schemas` - schema, table, column, index, and relationship metadata for one database",
				"- `GET /relationships` - generated entity and database relationship graph",
				"",
				"Declared databases:",
				"",
			])
			for database in databases:
				schema_count = len(database.schemas)
				table_count = sum(len(schema.tables) for schema in database.schemas)
				lines.append(
					f"- `{database.name}` - {schema_count} schema(s), {table_count} table(s)"
				)

		if agents:
			lines.extend(["", "## AI agents", ""])
			for agent in agents:
				runtime = agent.runtime or "local"
				lines.append(f"- `{agent.name}` - runtime `{runtime}`, invoke with `POST /agents/{agent.name}/invoke`")
			lines.extend([
				"",
				"Typed agent stub classes live in `agent_stubs.py`. "
				"Wire up a runtime adapter by setting the environment variable:",
				"",
				"```",
				f"export APG_AGENT_{(agents[0].runtime or 'CODEX').upper()}_PROVIDER_COMMAND='python my_provider.py'",
				"```",
				"",
				"The provider receives JSON `{\"agent\": {...}, \"input\": \"...\", \"context\": {...}}` "
				"on stdin and writes `{\"output\": \"...\"}` to stdout.",
			])

		if teams:
			lines.extend(["", "## AI agent teams", ""])
			for team in teams:
				lines.append(f"- `{team.name}` - invoke with `POST /agent-teams/{team.name}/invoke`")

		if capabilities:
			lines.extend(["", "## Capabilities", ""])
			for capability in capabilities:
				provides = ", ".join(capability.provides) if capability.provides else "no declared services"
				lines.append(f"- `{capability.name}` - provides {provides}")
			lines.extend([
				"",
				"Capability operations:",
				"",
				"- `GET /capabilities` - capability catalog and dependency graph",
				"- `GET /streaming` - ByteWax streaming topology",
				"- `GET /capabilities/{Capability}/streaming` - capability streaming contract",
				"- `POST /capabilities/{Capability}/rules/evaluate` - evaluate capability rules",
				"- `POST /capabilities/{Capability}/configuration/resolve` - resolve configuration",
				"- `POST /capabilities/{Capability}/configuration/validate` - validate configuration",
				"- `POST /capabilities/{Capability}/approval/plan` - plan approvals",
			])
			screen_routes = self._capability_screen_routes(capabilities)
			if screen_routes:
				lines.extend(["", "Capability screens:", ""])
				for route in screen_routes:
					lines.append(f"- `GET {route}`")

		lines.append("")
		return "\n".join(lines)

	def _capability_screen_routes(self, capabilities: List[CapabilityDeclaration]) -> List[str]:
		"""Extract declared capability screen routes for generated documentation."""
		routes: List[str] = []
		for capability in capabilities:
			ui_routes = capability.ui.get("routes", [])
			if isinstance(ui_routes, list):
				for route in ui_routes:
					if isinstance(route, dict) and route.get("path"):
						routes.append(str(route["path"]))
			declared = capability.screens or capability.ui.get("screens", {})
			if isinstance(declared, dict):
				for screen in declared.values():
					if isinstance(screen, dict):
						route = screen.get("route") or screen.get("path")
						if route:
							routes.append(str(route))
			elif isinstance(declared, list):
				for screen in declared:
					if isinstance(screen, dict):
						route = screen.get("route") or screen.get("path")
						if route:
							routes.append(str(route))
		return sorted(set(routes))

	def _generate_python_entity_catalog_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate dependency-free entity metadata for hybrid template mode."""
		return {"entities.py": self._generate_python_entity_catalog(ast)}

	def _generate_python_entity_catalog(self, module: ModuleDeclaration) -> str:
		"""Generate a framework-neutral entity catalog module."""
		entity_specs = [self._entity_spec(entity) for entity in module.entities]
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

	def _generate_application_files(self, ast: ModuleDeclaration) -> Dict[str, str]:
		"""Generate first-class application composition runtime files."""
		applications = [entity for entity in ast.entities if isinstance(entity, ApplicationDeclaration)]
		if not applications:
			return {}
		return {"apg_application.py": self._generate_application_runtime(applications)}

	def _generate_application_runtime(self, applications: List[ApplicationDeclaration]) -> str:
		"""Generate a dependency-free runtime manifest for APG application composition."""
		application_specs = {
			application.name: {
				"description": application.description,
				"capabilities": application.capabilities,
				"agents": application.agents,
				"agent_teams": application.agent_teams,
				"components": application.components,
				"screens": application.screens,
				"routes": application.routes,
				"workflows": application.workflows,
				"policies": application.policies,
				"configuration": application.configuration,
				"theme": application.theme,
				"runtime": application.runtime,
				"integrations": application.integrations,
				"deployments": application.deployments,
			}
			for application in applications
		}
		return f'''"""
APG Application Composition Runtime
===================================

Generated from first-class APG app/application composition declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    description: str | None
    capabilities: List[str]
    agents: List[str]
    agent_teams: List[str]
    components: Any
    screens: Any
    routes: List[str]
    workflows: List[str]
    policies: Any
    configuration: Dict[str, Any]
    theme: Dict[str, Any]
    runtime: Dict[str, Any]
    integrations: Any
    deployments: Any


APPLICATION_DATA: Dict[str, Dict[str, Any]] = {application_specs!r}
APPLICATIONS: Dict[str, ApplicationSpec] = {{
    name: ApplicationSpec(name=name, **data)
    for name, data in APPLICATION_DATA.items()
}}


def list_applications() -> List[str]:
    return sorted(APPLICATIONS)


def get_application(name: str) -> ApplicationSpec:
    return APPLICATIONS[name]


def describe_application_composition(name: str) -> Dict[str, Any]:
    application = get_application(name)
    return {{
        "name": application.name,
        "description": application.description,
        "capabilities": list(application.capabilities),
        "agents": list(application.agents),
        "agent_teams": list(application.agent_teams),
        "components": application.components,
        "screens": application.screens,
        "routes": list(application.routes),
        "workflows": list(application.workflows),
        "policies": application.policies,
        "configuration": dict(application.configuration),
        "theme": dict(application.theme),
        "runtime": dict(application.runtime),
        "integrations": application.integrations,
        "deployments": application.deployments,
    }}


def describe_application_compositions() -> Dict[str, Dict[str, Any]]:
    return {{
        name: describe_application_composition(name)
        for name in list_applications()
    }}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def application_component_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {{}}
    for application in APPLICATIONS.values():
        components = application.components
        if isinstance(components, dict):
            for component_name, component_spec in components.items():
                component_id = f"{{application.name}}.{{component_name}}"
                catalog[component_id] = {{
                    "id": component_id,
                    "application": application.name,
                    "name": str(component_name),
                    "spec": dict(component_spec) if isinstance(component_spec, dict) else {{"value": component_spec}},
                }}
        for route in application.routes:
            component_id = f"{{application.name}}.route.{{route}}"
            catalog[component_id] = {{
                "id": component_id,
                "application": application.name,
                "name": str(route),
                "kind": "route",
                "spec": {{"route": route}},
            }}
    return catalog


def _normalize_application_screen(application: ApplicationSpec, name: str, spec: Any) -> Dict[str, Any]:
    screen_spec = dict(spec) if isinstance(spec, dict) else {{"component": spec or name}}
    route = screen_spec.get("route", screen_spec.get("path", ""))
    return {{
        "id": f"{{application.name}}.{{name}}",
        "application": application.name,
        "name": name,
        "route": route,
        "path": route,
        "component": screen_spec.get("component", name),
        "capability": screen_spec.get("capability"),
        "capabilities": list(application.capabilities),
        "agents": list(application.agents),
        "agent_teams": list(application.agent_teams),
        "theme": screen_spec.get("theme", application.theme.get("name")),
        "spec": screen_spec,
    }}


def application_screens(application_name: str) -> List[Dict[str, Any]]:
    application = get_application(application_name)
    screens: List[Dict[str, Any]] = []
    if isinstance(application.screens, dict):
        for name, spec in application.screens.items():
            screens.append(_normalize_application_screen(application, str(name), spec))
    elif isinstance(application.screens, list):
        for index, item in enumerate(application.screens):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or item.get("component") or f"screen_{{index + 1}}")
                screens.append(_normalize_application_screen(application, name, item))
            else:
                name = str(item)
                screens.append(_normalize_application_screen(application, name, {{"component": name}}))

    known_routes = {{str(screen.get("route") or screen.get("path") or "") for screen in screens}}
    for index, route in enumerate(application.routes):
        route_text = str(route)
        if route_text in known_routes:
            continue
        screens.append({{
            "id": f"{{application.name}}.route_{{index + 1}}",
            "application": application.name,
            "name": route_text,
            "route": route_text,
            "path": route_text,
            "component": route_text,
            "capability": None,
            "capabilities": list(application.capabilities),
            "agents": list(application.agents),
            "agent_teams": list(application.agent_teams),
            "theme": application.theme.get("name"),
            "spec": {{"route": route_text}},
        }})
    return screens


def application_route_index() -> Dict[str, Dict[str, Any]]:
    routes: Dict[str, Dict[str, Any]] = {{}}
    for application in APPLICATIONS.values():
        for screen in application_screens(application.name):
            route = screen.get("route") or screen.get("path")
            if route:
                routes[str(route)] = screen
    return routes


def application_dependency_graph() -> Dict[str, List[Dict[str, str]]]:
    nodes: Dict[str, Dict[str, str]] = {{}}
    edges: List[Dict[str, str]] = []

    def node(node_id: str, kind: str, name: str) -> None:
        nodes[node_id] = {{"id": node_id, "kind": kind, "name": name}}

    def edge(source: str, target: str, relation: str) -> None:
        edges.append({{"source": source, "target": target, "relation": relation}})

    for application in APPLICATIONS.values():
        app_id = f"application:{{application.name}}"
        node(app_id, "application", application.name)
        for capability in application.capabilities:
            capability_id = f"capability:{{capability}}"
            node(capability_id, "capability", str(capability))
            edge(app_id, capability_id, "uses_capability")
        for agent in application.agents:
            agent_id = f"agent:{{agent}}"
            node(agent_id, "agent", str(agent))
            edge(app_id, agent_id, "uses_agent")
        for team in application.agent_teams:
            team_id = f"agent_team:{{team}}"
            node(team_id, "agent_team", str(team))
            edge(app_id, team_id, "uses_agent_team")
        for route in application.routes:
            route_id = f"route:{{route}}"
            node(route_id, "route", str(route))
            edge(app_id, route_id, "exposes_route")
        for screen in application_screens(application.name):
            screen_id = f"application_screen:{{screen['id']}}"
            node(screen_id, "application_screen", str(screen["name"]))
            edge(app_id, screen_id, "has_screen")
            route = screen.get("route") or screen.get("path")
            if route:
                route_id = f"route:{{route}}"
                node(route_id, "route", str(route))
                edge(screen_id, route_id, "mounted_at")
    return {{"nodes": sorted(nodes.values(), key=lambda item: item["id"]), "edges": edges}}


def validate_application_compositions(
    available_capabilities: List[str] | None = None,
    available_agents: List[str] | None = None,
    available_teams: List[str] | None = None,
) -> Dict[str, List[str]]:
    known_capabilities = set(available_capabilities or [])
    known_agents = set(available_agents or [])
    known_teams = set(available_teams or [])
    errors: List[str] = []
    warnings: List[str] = []
    for application in APPLICATIONS.values():
        if not application.capabilities and not application.components and not application.routes:
            warnings.append(f"{{application.name}} does not compose capabilities, components, or routes")
        for capability in application.capabilities:
            if known_capabilities and capability not in known_capabilities:
                errors.append(f"{{application.name}} references unknown capability {{capability}}")
        for agent in application.agents:
            if known_agents and agent not in known_agents:
                errors.append(f"{{application.name}} references unknown agent {{agent}}")
        for team in application.agent_teams:
            if known_teams and team not in known_teams:
                errors.append(f"{{application.name}} references unknown agent team {{team}}")
    return {{"errors": errors, "warnings": warnings}}
'''

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

import ast
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
AFRICAN_LANGUAGE_CODES = {{
    "af", "ak", "am", "ar", "bm", "bem", "ber", "bin", "din", "dyu",
    "ee", "ff", "fon", "gaa", "ha", "ig", "kab", "kam", "ki", "kln",
    "kg", "kj", "kmb", "kr", "lg", "ln", "loz", "lu", "lua", "mg",
    "mos", "nd", "nr", "nso", "ny", "om", "rn", "rw", "sg", "sn",
    "so", "ss", "st", "sw", "ti", "tn", "ts", "tum", "tw", "ve",
    "wo", "xh", "yo", "zu",
}}
CORE_LANGUAGE_CODES = {{
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr",
    "ru", "zh", "ja", "ko", "hi", "ur", "id", "ms",
}}
SUPPORTED_LANGUAGE_CODES = CORE_LANGUAGE_CODES | AFRICAN_LANGUAGE_CODES
CAPABILITIES: Dict[str, CapabilitySpec] = {{
    name: CapabilitySpec(name=name, **data)
    for name, data in CAPABILITY_DATA.items()
}}

_MISSING = object()


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


def supported_language_codes() -> List[str]:
    return sorted(SUPPORTED_LANGUAGE_CODES)


def african_language_codes() -> List[str]:
    return sorted(AFRICAN_LANGUAGE_CODES)


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
        for language in supported:
            if language not in SUPPORTED_LANGUAGE_CODES:
                errors.append(f"{{capability.name}} unsupported language code {{language}}")
        default_language = capability.i18n.get("default_language")
        fallback_language = capability.i18n.get("fallback_language")
        if default_language and default_language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{{capability.name}} unknown default language {{default_language}}")
        if fallback_language and fallback_language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{{capability.name}} unknown fallback language {{fallback_language}}")
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


def capability_health(capability_name: str) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    errors: List[str] = []
    warnings: List[str] = []

    configuration = validate_capability_configuration(capability_name)
    errors.extend(configuration.get("errors", []))
    warnings.extend(configuration.get("warnings", []))

    languages = capability_languages(capability_name)
    default_language = capability.i18n.get("default_language")
    fallback_language = capability.i18n.get("fallback_language")
    if not languages:
        warnings.append(f"{{capability.name}} does not declare supported languages")
    for language in languages:
        if language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{{capability.name}} unsupported language code {{language}}")
    if default_language and default_language not in SUPPORTED_LANGUAGE_CODES:
        errors.append(f"{{capability.name}} unknown default language {{default_language}}")
    if fallback_language and fallback_language not in SUPPORTED_LANGUAGE_CODES:
        errors.append(f"{{capability.name}} unknown fallback language {{fallback_language}}")
    if default_language and default_language not in languages:
        errors.append(f"{{capability.name}} default language {{default_language}} is not supported")
    if fallback_language and fallback_language not in languages:
        errors.append(f"{{capability.name}} fallback language {{fallback_language}} is not supported")

    rules = capability_rules(capability_name)
    if not rules:
        warnings.append(f"{{capability.name}} does not declare capability rules")
    screens = capability_screens(capability_name)
    if not screens:
        warnings.append(f"{{capability.name}} does not declare UI screens")
    components = capability_components(capability_name)
    if not components:
        warnings.append(f"{{capability.name}} does not declare composable components")
    master_data = master_data_entities(capability_name)
    if not master_data:
        warnings.append(f"{{capability.name}} does not declare master data entities")

    stream = capability_streaming(capability_name)
    processor = str(stream.get("processor") or "")
    if processor not in {{"bytewax", "bytewax_streams"}}:
        errors.append(f"{{capability.name}} uses unsupported stream processor {{processor}}")
    if not stream.get("state"):
        warnings.append(f"{{capability.name}} does not declare streaming state")

    health = {{
        "capability": capability.name,
        "status": "error" if errors else "warning" if warnings else "ok",
        "healthy": not errors,
        "errors": errors,
        "warnings": warnings,
        "configuration": configuration,
        "rules": {{
            "count": len(rules),
            "names": [str(rule.get("name")) for rule in rules],
            "sample_evaluation": evaluate_capability_rules(capability_name, {{}}),
        }},
        "approvals": approval_plan(capability_name, {{}}),
        "ui": {{
            "screens": screens,
            "route_index": {{
                route: screen
                for route, screen in ui_route_index().items()
                if screen.get("capability") == capability.name
            }},
        }},
        "theme": capability_theme(capability_name),
        "streaming": stream,
        "master_data": master_data,
        "languages": languages,
        "components": components,
    }}
    return health


def capability_health_report() -> Dict[str, Any]:
    capabilities = {{
        capability_name: capability_health(capability_name)
        for capability_name in list_capabilities()
    }}
    errors = [
        f"{{capability_name}}: {{error}}"
        for capability_name, health in capabilities.items()
        for error in health.get("errors", [])
    ]
    warnings = [
        f"{{capability_name}}: {{warning}}"
        for capability_name, health in capabilities.items()
        for warning in health.get("warnings", [])
    ]
    return {{
        "healthy": not errors,
        "errors": errors,
        "warnings": warnings,
        "capabilities": capabilities,
    }}


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


def evaluate_capability_rules(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    context = dict(context or {{}})
    evaluation_context = dict(capability.configuration)
    _deep_merge(evaluation_context, context)
    matched: List[str] = []
    actions: List[Dict[str, Any]] = []
    decision = "allow"
    precedence = {{"allow": 0, "audit": 1, "warn": 1, "require_review": 2, "deny": 3}}
    for rule in capability_rules(capability_name):
        if not _matches_rule(rule, evaluation_context):
            continue
        matched.append(str(rule["name"]))
        effect = dict(rule.get("effect") or {{}})
        effect.setdefault("decision", _decision_from_action(effect.get("action", rule.get("action", "allow"))))
        effect.setdefault("rule", rule["name"])
        actions.append(effect)
        candidate = str(effect.get("decision") or "allow")
        if precedence.get(candidate, 0) > precedence.get(decision, 0):
            decision = candidate
    return {{
        "decision": decision,
        "matched_rules": matched,
        "actions": actions,
        "context": context,
        "effective_context": evaluation_context,
    }}


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
        return not _evaluate_condition(expression[4:].strip(), context)
    lowered = expression.lower()
    if lowered.endswith(" missing"):
        target = expression[:-8].strip()
        resolved = _resolve_value(target, context)
        return _missing_context_reference(target, resolved, context) or resolved in (None, "")
    if lowered.endswith(" present"):
        target = expression[:-8].strip()
        resolved = _resolve_value(target, context)
        return not (_missing_context_reference(target, resolved, context) or resolved in (None, ""))
    for operator in ("!=", "==", ">=", "<=", ">", "<"):
        marker = f" {{operator}} "
        if marker not in expression:
            continue
        left_text, right_text = expression.split(marker, 1)
        left_key = left_text.strip()
        right_key = right_text.strip()
        left, left_missing = _resolve_rule_operand(left_key, context)
        right, right_missing = _resolve_rule_operand(right_key, context)
        if left_missing:
            return False
        if right_missing:
            return False
        if operator == "!=":
            return left != right
        if operator == "==":
            return left == right
        try:
            if operator == ">=":
                return left >= right
            if operator == "<=":
                return left <= right
            if operator == ">":
                return left > right
            if operator == "<":
                return left < right
        except TypeError:
            return False
    resolved = _resolve_value(expression, context)
    if _missing_context_reference(expression, resolved, context):
        return False
    return bool(resolved)


def _resolve_rule_operand(text: str, context: Dict[str, Any]) -> tuple[Any, bool]:
    value = _safe_expression_value(text, context)
    if value is not _MISSING:
        return value, False
    value = _resolve_value(text, context)
    return value, _missing_context_reference(text, value, context)


def _safe_expression_value(expression: str, context: Dict[str, Any]) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _MISSING
    try:
        return _eval_expression_node(tree.body, context)
    except (TypeError, ValueError, ZeroDivisionError):
        return _MISSING


def _eval_expression_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        value = _resolve_value(node.id, context)
        if _missing_context_reference(node.id, value, context):
            return _MISSING
        return value
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is None:
            return _MISSING
        value = _resolve_value(path, context)
        if _missing_context_reference(path, value, context):
            return _MISSING
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _eval_expression_node(node.operand, context)
        if operand is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        return _MISSING
    if isinstance(node, ast.BinOp):
        left = _eval_expression_node(node.left, context)
        right = _eval_expression_node(node.right, context)
        if left is _MISSING or right is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    return _MISSING


def _attribute_path(node: ast.AST) -> str | None:
    parts: List[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _missing_context_reference(text: str, resolved: Any, context: Dict[str, Any]) -> bool:
    if text in context or resolved != text:
        return False
    lowered = text.lower()
    if lowered in {{"true", "false", "none", "null"}}:
        return False
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return False
    try:
        int(text)
    except ValueError:
        try:
            float(text)
        except ValueError:
            return True
    return False


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

import json
import os
import shutil
import shlex
import subprocess

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


def _invocation_input(payload: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(payload, dict):
        return {{}}
    if "input" in payload:
        return payload["input"]
    if "message" in payload:
        return payload["message"]
    return dict(payload)


def _env_fragment(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.upper()).strip("_")


def runtime_adapter_environment_keys(runtime: str, agent_name: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    if agent_name:
        keys.append(f"APG_AGENT_{{_env_fragment(agent_name)}}_COMMAND")
    keys.extend([
        f"APG_AGENT_RUNTIME_{{_env_fragment(runtime)}}_COMMAND",
        f"APG_AGENT_{{_env_fragment(runtime)}}_COMMAND",
        "APG_AGENT_RUNTIME_COMMAND",
    ])
    return keys


def _coerce_command(value: Any) -> Optional[List[str]]:
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return list(value)
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return None


def runtime_adapter_command_candidates(runtime: str) -> List[List[str]]:
    runtime_spec = AI_AGENT_RUNTIME_DATA.get(canonical_runtime(runtime), {{}})
    candidates = runtime_spec.get("command_candidates", [])
    commands: List[List[str]] = []
    for candidate in candidates:
        command = _coerce_command(candidate)
        if command:
            commands.append(command)
    return commands


def _adapter_command(agent: AIAgentSpec, runtime: str) -> tuple[Optional[List[str]], Optional[str]]:
    configured = (
        agent.configuration.get("adapter_command")
        or agent.configuration.get("runtime_command")
        or agent.configuration.get("agent_command")
    )
    command = _coerce_command(configured)
    if command:
        return command, "agent.configuration"
    for key in runtime_adapter_environment_keys(runtime, agent.name):
        command = _coerce_command(os.environ.get(key))
        if command:
            return command, key
    for candidate in runtime_adapter_command_candidates(runtime):
        resolved = shutil.which(candidate[0])
        if resolved:
            return [resolved, *candidate[1:]], f"runtime.{{runtime}}.command_candidates"
    return None, None


def _adapter_timeout(agent: AIAgentSpec) -> float:
    configured = agent.configuration.get("adapter_timeout", agent.configuration.get("timeout"))
    raw_value = os.environ.get("APG_AGENT_RUNTIME_TIMEOUT", configured)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 120.0


def _agent_invocation_base(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {{
        "agent": agent.name,
        "role": agent.role,
        "model": agent.model,
        "runtime": runtime,
        "runtime_spec": dict(runtime_spec),
        "input": _invocation_input(payload),
        "system": agent.system,
        "capabilities": list(agent.capabilities),
        "tools": list(agent.tools),
        "configuration": dict(agent.configuration),
        "handoffs": [dict(edge) for edge in agent.handoffs],
    }}


def _external_invocation_envelope(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {{
        "agent": describe_agent(agent.name),
        "runtime": runtime,
        "runtime_spec": dict(runtime_spec),
        "input": _invocation_input(payload),
        "payload": dict(payload) if isinstance(payload, dict) else {{}},
    }}


def _run_external_agent(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    command, command_source = _adapter_command(agent, runtime)
    if not command:
        return None
    envelope = _external_invocation_envelope(agent, runtime, runtime_spec, payload)
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(envelope, sort_keys=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=_adapter_timeout(agent),
            cwd=os.environ.get("APG_AGENT_WORKDIR") or None,
        )
    except FileNotFoundError as error:
        return {{
            "status": "failed",
            "mode": "external",
            "output": {{
                "message": str(error),
                "requires_adapter": False,
                "adapter_command": command,
                "adapter_source": command_source,
                "error": "adapter_command_not_found",
            }},
        }}
    except subprocess.TimeoutExpired as error:
        return {{
            "status": "failed",
            "mode": "external",
            "output": {{
                "message": f"External runtime adapter timed out after {{error.timeout}} seconds.",
                "requires_adapter": False,
                "adapter_command": command,
                "adapter_source": command_source,
                "error": "adapter_timeout",
            }},
        }}
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed_output: Any = None
    if stdout:
        try:
            parsed_output = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_output = stdout
    adapter_status = "completed" if completed.returncode == 0 else "failed"
    adapter_mode = "external"
    adapter_message = "External runtime adapter completed." if completed.returncode == 0 else "External runtime adapter failed."
    if isinstance(parsed_output, dict):
        parsed_status = parsed_output.get("status")
        if parsed_status in {{"completed", "failed", "adapter_required"}}:
            adapter_status = parsed_status
        parsed_mode = parsed_output.get("mode")
        if isinstance(parsed_mode, str) and parsed_mode:
            adapter_mode = parsed_mode
        parsed_message = parsed_output.get("message")
        if isinstance(parsed_message, str) and parsed_message:
            adapter_message = parsed_message
    adapter_requires = adapter_status == "adapter_required"
    return {{
        "status": adapter_status,
        "mode": adapter_mode,
        "output": {{
            "message": adapter_message,
            "requires_adapter": adapter_requires,
            "adapter_command": command,
            "adapter_source": command_source,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed_output,
        }},
    }}


def invoke_agent(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    agent = get_agent(name)
    runtime = canonical_runtime(agent.runtime)
    runtime_spec = dict(AI_AGENT_RUNTIME_DATA[runtime])
    requires_adapter = runtime != "local"
    base = _agent_invocation_base(agent, runtime, runtime_spec, payload)
    if requires_adapter:
        external = _run_external_agent(agent, runtime, runtime_spec, payload)
        if external is not None:
            base.update(external)
            return base
    base.update({{
        "status": "adapter_required" if requires_adapter else "completed",
        "mode": "adapter_missing" if requires_adapter else "local",
        "output": {{
            "message": (
                f"{{agent.name}} requires a configured {{runtime}} adapter command before invocation."
                if requires_adapter
                else f"{{agent.name}} handled the request locally."
            ),
            "requires_adapter": requires_adapter,
            "adapter_environment_keys": runtime_adapter_environment_keys(runtime, agent.name) if requires_adapter else [],
            "adapter_command_candidates": runtime_adapter_command_candidates(runtime) if requires_adapter else [],
        }},
    }})
    return base


def invoke_team(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    team = get_team(name)
    invocations = [
        invoke_agent(agent_name, payload)
        for agent_name in team.agents
    ]
    if any(item["status"] == "failed" for item in invocations):
        status = "failed"
    elif any(item["status"] == "adapter_required" for item in invocations):
        status = "adapter_required"
    else:
        status = "completed"
    return {{
        "team": team.name,
        "status": status,
        "policy": dict(team.policy),
        "configuration": dict(team.configuration),
        "flow": [dict(edge) for edge in team.flow],
        "invocations": invocations,
    }}


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
				"command_candidates": [["apg-agent-codex"]],
			},
			"claude_code": {
				"kind": "cli",
				"aliases": ["claude", "claude-code"],
				"supports_workspace": True,
				"requires_token": False,
				"family": "coding_agent",
				"command_candidates": [["apg-agent-claude-code"], ["apg-agent-claude"]],
			},
			"opencode": {
				"kind": "cli",
				"aliases": ["open_code"],
				"supports_workspace": True,
				"requires_token": False,
				"family": "coding_agent",
				"command_candidates": [["apg-agent-opencode"]],
			},
			"openai": {
				"kind": "http",
				"aliases": ["openai_chat"],
				"supports_workspace": False,
				"requires_token": True,
				"family": "chat_agent",
				"command_candidates": [["apg-agent-openai"]],
			},
			"ollama": {
				"kind": "http",
				"aliases": ["local_llm"],
				"supports_workspace": False,
				"requires_token": False,
				"family": "local_model",
				"command_candidates": [["apg-agent-ollama"]],
			},
			"pi": {
				"kind": "http",
				"aliases": ["inflection_pi"],
				"supports_workspace": False,
				"requires_token": True,
				"family": "chat_agent",
				"command_candidates": [["apg-agent-pi"]],
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
			'from .app import approval_plan, auth_status, capability_configuration, capability_health, capability_health_report, capability_languages, capability_rules, capability_screens, capability_streaming, capability_theme, coerce_record_types, component_manifest, create_record, database_status, delete_record, describe_application, describe_capabilities, describe_capability, describe_workflow, describe_workflows, evaluate_capability_rules, execute_workflow_compensations, get_record, get_workflow_run, invoke_agent, invoke_team, list_agent_teams, list_agents, list_capabilities, list_databases, list_entities, list_events, list_records, list_workflow_runs, list_workflows, main, metrics_snapshot, openapi_document, query_records, relationship_graph, resume_workflow, run_workflow, runtime_adapter_command_candidates, runtime_adapter_environment_keys, self_test, semantic_model, storage_status, theme_token, update_record, validate_agent_runtimes, validate_application, validate_capability_configuration, validate_component_manifest_contract, validate_openapi_contract, validate_route_dispatch_contract, validate_record',
			'',
			'__all__ = [',
			'    "__version__",',
			'    "approval_plan",',
			'    "auth_status",',
			'    "capability_configuration",',
			'    "capability_health",',
			'    "capability_health_report",',
			'    "capability_languages",',
			'    "capability_rules",',
			'    "capability_screens",',
			'    "capability_streaming",',
			'    "capability_theme",',
			'    "coerce_record_types",',
			'    "component_manifest",',
			'    "create_record",',
			'    "database_status",',
			'    "delete_record",',
			'    "describe_application",',
			'    "describe_capabilities",',
			'    "describe_capability",',
			'    "describe_workflow",',
			'    "describe_workflows",',
			'    "evaluate_capability_rules",',
			'    "execute_workflow_compensations",',
			'    "get_record",',
			'    "get_workflow_run",',
			'    "invoke_agent",',
			'    "invoke_team",',
			'    "list_agent_teams",',
			'    "list_agents",',
			'    "list_capabilities",',
			'    "list_databases",',
			'    "list_entities",',
			'    "list_events",',
			'    "list_records",',
			'    "list_workflow_runs",',
			'    "list_workflows",',
			'    "main",',
			'    "metrics_snapshot",',
			'    "openapi_document",',
			'    "query_records",',
			'    "relationship_graph",',
			'    "resume_workflow",',
			'    "run_workflow",',
			'    "runtime_adapter_command_candidates",',
			'    "runtime_adapter_environment_keys",',
			'    "self_test",',
			'    "semantic_model",',
			'    "storage_status",',
			'    "theme_token",',
			'    "update_record",',
			'    "validate_agent_runtimes",',
			'    "validate_application",',
			'    "validate_capability_configuration",',
			'    "validate_component_manifest_contract",',
			'    "validate_openapi_contract",',
			'    "validate_route_dispatch_contract",',
			'    "validate_record",',
			']',
			'',
			'try:',
			'    from .ai_agents import (',
			'        get_agent,',
			'        get_team,',
			'        invoke_agent,',
			'        invoke_team,',
			'        list_agent_runtimes,',
			'        list_agent_teams,',
			'        list_agents,',
			'        list_teams,',
			'        runtime_adapter_command_candidates,',
			'        runtime_adapter_environment_keys,',
			'        validate_agent_runtimes,',
			'    )',
			'except ImportError:',
			'    __all__ = list(__all__)',
			'else:',
			'    __all__.extend([',
			'        "get_agent",',
			'        "get_team",',
			'        "invoke_agent",',
			'        "invoke_team",',
			'        "list_agent_runtimes",',
			'        "list_agent_teams",',
			'        "list_agents",',
			'        "list_teams",',
			'        "runtime_adapter_command_candidates",',
			'        "runtime_adapter_environment_keys",',
			'        "validate_agent_runtimes",',
			'    ])',
			'',
			'try:',
			'    from .apg_application import (',
			'        application_component_catalog,',
			'        application_dependency_graph,',
			'        application_route_index,',
			'        application_screens,',
			'        describe_application_composition,',
			'        describe_application_compositions,',
			'        get_application,',
			'        list_applications,',
			'        validate_application_compositions,',
			'    )',
			'except ImportError:',
			'    __all__ = list(__all__)',
			'else:',
			'    __all__.extend([',
			'        "application_component_catalog",',
			'        "application_dependency_graph",',
			'        "application_route_index",',
			'        "application_screens",',
			'        "describe_application_composition",',
			'        "describe_application_compositions",',
			'        "get_application",',
			'        "list_applications",',
			'        "validate_application_compositions",',
			'    ])',
			'',
			'try:',
			'    from .apg_capabilities import (',
			'        approval_plan,',
			'        capability_dependency_graph,',
			'        capability_configuration,',
			'        capability_health,',
			'        capability_health_report,',
			'        capability_languages,',
			'        capability_load_order,',
			'        capability_rules,',
			'        capability_screens,',
			'        capability_streaming,',
			'        capability_theme,',
			'        evaluate_capability_rules,',
			'        describe_capabilities,',
			'        describe_capabilities_by_erp_module,',
			'        describe_capability,',
			'        african_language_codes,',
			'        capability_names_by_erp_module,',
			'        composition_graph,',
			'        get_capability,',
			'        list_capabilities,',
			'        streaming_processor_index,',
			'        streaming_state_index,',
			'        supported_language_codes,',
			'        theme_token,',
			'        ui_route_index,',
			'        validate_capability_configuration,',
			'    )',
			'except ImportError:',
			'    __all__ = list(__all__)',
			'else:',
			'    __all__.extend([',
			'        "approval_plan",',
			'        "capability_dependency_graph",',
			'        "capability_configuration",',
			'        "capability_health",',
			'        "capability_health_report",',
			'        "capability_languages",',
			'        "capability_load_order",',
			'        "capability_rules",',
			'        "capability_screens",',
			'        "capability_streaming",',
			'        "capability_theme",',
			'        "evaluate_capability_rules",',
			'        "describe_capabilities",',
			'        "describe_capabilities_by_erp_module",',
			'        "describe_capability",',
			'        "african_language_codes",',
			'        "capability_names_by_erp_module",',
			'        "composition_graph",',
			'        "get_capability",',
			'        "list_capabilities",',
			'        "streaming_processor_index",',
			'        "streaming_state_index",',
			'        "supported_language_codes",',
			'        "theme_token",',
			'        "ui_route_index",',
			'        "validate_capability_configuration",',
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
