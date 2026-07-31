"""
APG Semantic Analyzer Module
============================

Performs semantic analysis and type checking on APG Abstract Syntax Trees.
Validates program semantics, resolves symbols, performs type inference and checking,
and reports semantic errors before code generation.
"""

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import difflib
import sys
from pathlib import Path

# Import AST nodes
from .ast_builder import (
	ASTNode, ModuleDeclaration, EntityDeclaration, PropertyDeclaration,
	MethodDeclaration, Parameter, TypeAnnotation, Expression, Statement,
	LiteralExpression, IdentifierExpression, BinaryExpression, CallExpression,
	AssignmentStatement, ReturnStatement, BlockStatement, EntityType,
	AIAgentDeclaration, AgentTeamDeclaration, ApplicationDeclaration, CapabilityDeclaration,
	DatabaseDeclaration, ListExpression, DictExpression, RelationshipNode
)


# Module-level MANIFEST cache — keyed by mtime_ns to auto-invalidate on updates
_MANIFEST_KNOWN_SYSTEM_CACHE: "tuple[frozenset[str], int] | None" = None


def _load_known_system_capabilities() -> frozenset[str]:
	"""Load capability IDs from MANIFEST.json, cached by file mtime."""
	global _MANIFEST_KNOWN_SYSTEM_CACHE
	from pathlib import Path
	manifest_path = Path(__file__).resolve().parents[1] / "capabilities" / "MANIFEST.json"
	try:
		mtime = manifest_path.stat().st_mtime_ns
	except OSError:
		return frozenset()
	if _MANIFEST_KNOWN_SYSTEM_CACHE is not None and _MANIFEST_KNOWN_SYSTEM_CACHE[1] == mtime:
		return _MANIFEST_KNOWN_SYSTEM_CACHE[0]
	known: set[str] = set()
	try:
		import json
		raw = json.loads(manifest_path.read_text(encoding="utf-8"))
		for cap in raw.get("capabilities", {}).values():
			for key in ("id", "code"):
				v = str(cap.get(key, "") or "")
				if v:
					known.add(v)
	except Exception:
		pass
	result = frozenset(known)
	_MANIFEST_KNOWN_SYSTEM_CACHE = (result, mtime)
	return result


KNOWN_AI_AGENT_RUNTIMES = {
	"local", "offline", "test",
	"codex", "codex_cli", "openai_codex",
	"claude_code", "claude", "claude-code",
	"opencode", "open_code",
	"openai", "openai_chat",
	"ollama", "local_llm",
	"pi", "inflection_pi",
}


# ========================================
# Type System and Symbol Table
# ========================================

class APGType(Enum):
	"""Built-in APG types"""
	STRING = "str"
	INTEGER = "int"
	FLOAT = "float"
	BOOLEAN = "bool"
	LIST = "list"
	DICT = "dict"
	VOID = "void"
	ANY = "any"
	NULL = "null"
	FUNCTION = "function"
	ENTITY = "entity"


VALID_APG_FIELD_TYPES = (
	"str", "int", "float", "bool", "date", "datetime", "text", "uuid", "json", "file",
	"decimal", "time", "bytes", "any", "Any",
	"list", "List", "dict", "Dict", "set", "Set",
)

COMMON_TYPE_HINTS = {
	"boolean": "bool",
	"integer": "int",
	"number": "float",
	"string": "str",
}

RESERVED_FIELD_NAMES = {"id", "created_at", "updated_at", "deleted_at", "owner_id"}
VALID_FIELD_VALIDATORS = {
	"min_length", "max_length", "min", "max", "email", "pattern", "required", "optional",
}
NUMERIC_VALIDATOR_FIELD_TYPES = {
	"int", "integer", "serial", "bigint", "smallint",
	"float", "double", "decimal", "number", "numeric", "money",
}


@dataclass
class Symbol:
	"""Symbol table entry"""
	name: str
	symbol_type: APGType
	declared_type: Optional[TypeAnnotation] = None
	value: Any = None
	is_mutable: bool = True
	declaration_node: Optional[ASTNode] = None
	scope_level: int = 0


@dataclass
class FunctionSignature:
	"""Function signature for type checking"""
	name: str
	parameters: List[Parameter]
	return_type: Optional[TypeAnnotation]
	is_async: bool = False


class SymbolTable:
	"""Hierarchical symbol table for scope management"""
	
	def __init__(self, parent: Optional['SymbolTable'] = None):
		self.parent = parent
		self.symbols: Dict[str, Symbol] = {}
		self.level = parent.level + 1 if parent else 0
	
	def define(self, symbol: Symbol) -> bool:
		"""Define a new symbol in current scope"""
		if symbol.name in self.symbols:
			return False  # Symbol already exists in current scope
		
		symbol.scope_level = self.level
		self.symbols[symbol.name] = symbol
		return True
	
	def lookup(self, name: str) -> Optional[Symbol]:
		"""Look up symbol in current scope and parent scopes"""
		if name in self.symbols:
			return self.symbols[name]
		
		if self.parent:
			return self.parent.lookup(name)
		
		return None
	
	def lookup_local(self, name: str) -> Optional[Symbol]:
		"""Look up symbol only in current scope"""
		return self.symbols.get(name)
	
	def get_all_symbols(self) -> Dict[str, Symbol]:
		"""Get all symbols visible in current scope"""
		all_symbols = {}
		if self.parent:
			all_symbols.update(self.parent.get_all_symbols())
		all_symbols.update(self.symbols)
		return all_symbols


# ========================================
# Semantic Error Reporting
# ========================================

@dataclass
class SemanticError:
	"""Semantic analysis error"""
	message: str
	node: Optional[ASTNode]
	error_type: str = "semantic"

	@property
	def line(self) -> int:
		if self.node is None:
			return 1
		return self.node.line if self.node.line > 0 else 1

	@property
	def column(self) -> int:
		if self.node is None:
			return 1
		return self.node.column + 1 if self.node.column >= 0 else 1

	@property
	def source_file(self) -> Optional[str]:
		if self.node is None:
			return None
		return self.node.source_file
	
	def __str__(self) -> str:
		if self.error_type == "warning":
			return f"line {self.line}, col {self.column}: warning: {self.message}"
		if self.error_type != "semantic":
			return f"line {self.line}, col {self.column}: {self.error_type} error: {self.message}"
		return f"line {self.line}, col {self.column}: {self.message}"


# ========================================
# Semantic Analyzer Implementation
# ========================================

class SemanticAnalyzer:
	"""
	Performs semantic analysis on APG AST.
	
	Key responsibilities:
	- Symbol resolution and scope management
	- Type checking and inference
	- Entity relationship validation
	- Method signature validation
	- Dead code detection
	- Semantic error reporting
	"""
	
	def __init__(self):
		self.symbol_table = SymbolTable()
		self.current_scope = self.symbol_table
		self.errors: List[SemanticError] = []
		self.warnings: List[SemanticError] = []
		
		# Analysis state
		self.current_module: Optional[ModuleDeclaration] = None
		self.current_entity: Optional[EntityDeclaration] = None
		self.current_method: Optional[MethodDeclaration] = None
		
		# Built-in types and functions
		self.builtin_types = {t.value for t in APGType}
		# Extend with all common APG primitive and collection types
		self.builtin_types.update({
			"str", "int", "float", "bool", "bytes", "datetime", "date", "time",
			"decimal", "Decimal", "duration", "Any", "None", "Optional",
			"list", "List", "dict", "Dict", "set", "Set", "tuple", "Tuple",
			"str?", "int?", "float?", "bool?", "bytes?", "datetime?", "decimal?",
			"vector", "embedding", "json", "uuid", "url",
			"string", "number", "file",
		})
		self.builtin_functions = self._initialize_builtins()
	
	def analyze(self, ast: ModuleDeclaration, collect_all_errors: bool = False) -> Dict[str, Any]:
		"""
		Perform semantic analysis on the AST.

		Args:
			ast: Root AST node (ModuleDeclaration)
			collect_all_errors: When True, run all phases even if earlier phases
			    produced errors, collecting every diagnostic before returning.

		Returns:
			Analysis results including errors, warnings, and symbol table
		"""
		self.errors.clear()
		self.warnings.clear()
		self.symbol_table = SymbolTable()
		self.current_scope = self.symbol_table
		self.current_module = ast
		self.current_entity = None
		self.current_method = None

		phases = [
			("symbol_declaration", lambda: self._declare_module_symbols(ast)),
			("type_resolution",    lambda: self._resolve_types(ast)),
			("semantic_validation",lambda: self._validate_semantics(ast)),
			("dead_code_analysis", lambda: self._analyze_dead_code(ast)),
			("reference_resolution", lambda: self._resolve_references(ast)),
		]

		for phase_name, phase_fn in phases:
			try:
				phase_fn()
			except Exception as exc:
				self.errors.append(SemanticError(
					f"Internal analyzer error in {phase_name}: {exc}",
					ast,
					"internal",
				))
				# Without --all-errors stop after the first phase failure so
				# later phases don't cascade on broken state.
				if not collect_all_errors:
					break

		return {
			'success': len(self.errors) == 0,
			'errors': self.errors.copy(),
			'warnings': self.warnings.copy(),
			'symbol_table': self.symbol_table,
			'module': ast,
		}
	
	# ========================================
	# Phase 1: Symbol Declaration
	# ========================================
	
	def _declare_module_symbols(self, module: ModuleDeclaration):
		"""Declare all module-level symbols"""
		# Declare the module itself
		module_symbol = Symbol(
			name=module.name,
			symbol_type=APGType.ENTITY,
			declaration_node=module
		)
		self.symbol_table.define(module_symbol)
		
		# Declare all entities
		for entity in module.entities:
			self._declare_entity_symbols(entity)
	
	def _declare_entity_symbols(self, entity: EntityDeclaration):
		"""Declare entity and its members"""
		# Declare the entity
		entity_symbol = Symbol(
			name=entity.name,
			symbol_type=APGType.ENTITY,
			declaration_node=entity
		)
		
		if not self.symbol_table.define(entity_symbol):
			self.errors.append(SemanticError(
				f"Duplicate entity name: {entity.name}",
				entity
			))
			return
		
		# Create new scope for entity members
		entity_scope = SymbolTable(self.current_scope)
		previous_scope = self.current_scope
		self.current_scope = entity_scope
		self.current_entity = entity
		
		try:
			# Declare properties
			for prop in entity.properties:
				self._declare_property_symbol(prop)
			
			# Declare methods
			for method in entity.methods:
				self._declare_method_symbol(method)
		
		finally:
			self.current_scope = previous_scope
			self.current_entity = None
	
	def _declare_property_symbol(self, prop: PropertyDeclaration):
		"""Declare property symbol"""
		if prop.name in RESERVED_FIELD_NAMES:
			self.warnings.append(SemanticError(
				f"Reserved field name '{prop.name}' is declared explicitly; APG generates this field automatically",
				prop,
				"warning",
			))

		prop_symbol = Symbol(
			name=prop.name,
			symbol_type=self._apg_type_from_annotation(prop.type_annotation),
			declared_type=prop.type_annotation,
			declaration_node=prop,
			is_mutable=True  # APG properties are mutable by default
		)
		
		if not self.current_scope.define(prop_symbol):
			self.errors.append(SemanticError(
				f"Duplicate field name: {prop.name} in entity {self.current_entity.name if self.current_entity else 'unknown'}",
				prop
			))
	
	def _declare_method_symbol(self, method: MethodDeclaration):
		"""Declare method symbol"""
		# Create function signature
		signature = FunctionSignature(
			name=method.name,
			parameters=method.parameters,
			return_type=method.return_type,
			is_async=method.is_async
		)
		
		method_symbol = Symbol(
			name=method.name,
			symbol_type=APGType.FUNCTION,
			declared_type=method.return_type,
			value=signature,
			declaration_node=method,
			is_mutable=False
		)
		
		if not self.current_scope.define(method_symbol):
			self.errors.append(SemanticError(
				f"Method '{method.name}' is already defined in this entity",
				method
			))
	
	# ========================================
	# Phase 2: Type Resolution
	# ========================================
	
	def _resolve_types(self, module: ModuleDeclaration):
		"""Resolve all type references in the module"""
		for entity in module.entities:
			self._resolve_entity_types(entity)
	
	def _resolve_entity_types(self, entity: EntityDeclaration):
		"""Resolve types within an entity"""
		self.current_entity = entity
		
		# Resolve property types — only for table/entity types, not agent/workflow/capability config
		_TYPED_ENTITY_TYPES = {EntityType.ENTITY, EntityType.FORM, EntityType.UI_COMPONENT}
		is_config_entity = entity.name.lower() in {"security"}
		if entity.entity_type in _TYPED_ENTITY_TYPES and not is_config_entity:
			for prop in entity.properties:
				if not self._is_valid_field_type(prop.type_annotation):
					self.errors.append(SemanticError(
						self._unknown_type_message(
							prop.type_annotation.type_name,
							f"field '{prop.name}'",
						),
						prop
					))
		
		# Resolve method types
		for method in entity.methods:
			self._resolve_method_types(method)
	
	def _resolve_method_types(self, method: MethodDeclaration):
		"""Resolve types within a method"""
		self.current_method = method
		
		# Resolve parameter types
		for param in method.parameters:
			if not self._is_valid_type(param.type_annotation):
				self.errors.append(SemanticError(
					self._unknown_type_message(
						param.type_annotation.type_name,
						f"parameter '{param.name}'",
					),
					param
				))
		
		# Resolve return type
		if method.return_type and not self._is_valid_type(method.return_type):
			self.errors.append(SemanticError(
				self._unknown_type_message(
					method.return_type.type_name,
					f"return type for method '{method.name}'",
				),
				method
			))
	
	def _is_valid_type(self, type_annotation: TypeAnnotation) -> bool:
		"""Check if a type annotation is valid"""
		type_name = type_annotation.type_name or ""
		
		# Check built-in types
		if type_name in self.builtin_types:
			return True
		
		# Check if it's a defined entity
		symbol = self.symbol_table.lookup(type_name)
		if symbol and symbol.symbol_type == APGType.ENTITY:
			return True
		
		# Permissive: allow quoted strings (config values, not type names)
		if type_name.startswith(('"', "'")):
			return True
		
		# Permissive: allow list/dict literals, complex expressions, optional types
		if type_name.startswith('[') or type_name.startswith('{'):
			return True
		if type_name.endswith('?') or '->' in type_name:
			return True
		if type_name.startswith('vector ') or ' ' in type_name:
			return True
		
		return False

	def _is_valid_field_type(self, type_annotation: TypeAnnotation) -> bool:
		"""Check whether a field uses a valid APG field type or declared entity type."""
		type_name = type_annotation.type_name or ""
		base_type = type_name[:-1] if type_name.endswith("?") else type_name

		if base_type in VALID_APG_FIELD_TYPES:
			return True

		symbol = self.symbol_table.lookup(base_type)
		if symbol and symbol.symbol_type == APGType.ENTITY:
			return True

		# Keep non-type field values and complex relationship syntax from being
		# reported as primitive-type typos.
		if base_type.startswith(('"', "'")):
			return True
		if base_type.startswith('[') or base_type.startswith('{'):
			return True
		if '->' in base_type:
			return True
		if base_type.startswith('vector ') or ' ' in base_type:
			return True

		return False

	def _unknown_type_message(self, type_name: str, subject: str) -> str:
		"""Return an actionable unknown-type diagnostic message."""
		valid_types = ", ".join(VALID_APG_FIELD_TYPES)
		suggestion = COMMON_TYPE_HINTS.get(type_name)
		if suggestion is None:
			prefix_matches = [
				valid_type for valid_type in VALID_APG_FIELD_TYPES
				if valid_type.startswith(type_name) or type_name.startswith(valid_type)
			]
			matches = prefix_matches or difflib.get_close_matches(
				type_name,
				VALID_APG_FIELD_TYPES,
				n=1,
				cutoff=0.5,
			)
			suggestion = matches[0] if matches else None
		if suggestion:
			return (
				f"Unknown type '{type_name}' for {subject}. "
				f"Did you mean: {suggestion}? Valid types: {valid_types}"
			)
		return f"Unknown type '{type_name}' for {subject}. Valid types: {valid_types}"
	
	# ========================================
	# Phase 3: Semantic Validation
	# ========================================
	
	def _validate_semantics(self, module: ModuleDeclaration):
		"""Validate semantic rules"""
		for entity in module.entities:
			self._validate_entity_semantics(entity)
		self._validate_relationship_semantics(module)
		self._validate_agent_composition(module)
	
	def _validate_entity_semantics(self, entity: EntityDeclaration):
		"""Validate entity-specific semantic rules"""
		self.current_entity = entity

		if (
			entity.entity_type in {EntityType.ENTITY, EntityType.FORM, EntityType.UI_COMPONENT}
			and entity.name.lower() != "security"
			and not entity.properties
			and not getattr(entity, "relationships", [])
		):
			self.warnings.append(SemanticError(
				f"Entity {entity.name} has no fields",
				entity,
				"warning",
			))
		
		# Validate entity type constraints
		self._validate_entity_type_constraints(entity)

		# Validate field decorators
		self._validate_property_validation_rules(entity)
		
		# Validate methods
		for method in entity.methods:
			self._validate_method_semantics(method)

	def _validate_property_validation_rules(self, entity: EntityDeclaration) -> None:
		"""Validate field validation decorators."""
		for prop in entity.properties:
			type_name = (prop.type_annotation.type_name or "") if prop.type_annotation else ""
			base_type = type_name[:-1] if type_name.endswith("?") else type_name
			for rule in getattr(prop, "validation_rules", []):
				rule_type = str(getattr(rule, "rule_type", ""))
				if rule_type not in VALID_FIELD_VALIDATORS:
					self.warnings.append(SemanticError(
						f"Unknown validator '@{rule_type}' on field {entity.name}.{prop.name}",
						rule,
						"warning",
					))
					continue
				if rule_type == "email" and base_type != "str":
					self.errors.append(SemanticError(
						f"Validator '@email' can only be used on str fields; {entity.name}.{prop.name} is {base_type}",
						rule,
					))
				if rule_type in {"min", "max"} and base_type not in NUMERIC_VALIDATOR_FIELD_TYPES:
					self.errors.append(SemanticError(
						f"Validator '@{rule_type}' can only be used on int or float fields; {entity.name}.{prop.name} is {base_type}",
						rule,
					))

	def _validate_relationship_semantics(self, module: ModuleDeclaration) -> None:
		"""Validate entity relationship declarations."""
		entity_names = {entity.name for entity in module.entities}
		valid_kinds = {"has_many", "belongs_to", "has_one"}
		for entity in module.entities:
			for relationship in getattr(entity, "relationships", []):
				if not isinstance(relationship, RelationshipNode):
					continue
				if relationship.kind not in valid_kinds:
					self.errors.append(SemanticError(
						f"Unknown relationship kind '{relationship.kind}' in entity {entity.name}",
						relationship,
					))
				if relationship.target not in entity_names:
					self.errors.append(SemanticError(
						f"Relationship target entity '{relationship.target}' does not exist",
						relationship,
					))
				if relationship.through:
					if relationship.kind != "has_many":
						self.errors.append(SemanticError(
							f"Only has_many relationships can use through in entity {entity.name}",
							relationship,
						))
					if relationship.through not in entity_names:
						self.errors.append(SemanticError(
							f"Relationship junction entity '{relationship.through}' does not exist",
							relationship,
						))
	
	def _validate_entity_type_constraints(self, entity: EntityDeclaration):
		"""Validate constraints specific to entity types"""
		if entity.entity_type in {EntityType.AGENT, EntityType.AI_AGENT}:
			# Agent-specific validations
			self._validate_agent_constraints(entity)
		elif entity.entity_type == EntityType.AGENT_TEAM:
			self._validate_agent_team_constraints(entity)
		elif entity.entity_type == EntityType.CAPABILITY:
			self._validate_capability_constraints(entity)
		elif isinstance(entity, ApplicationDeclaration):
			self._validate_application_constraints(entity)
		elif entity.entity_type == EntityType.DIGITAL_TWIN:
			# Digital twin-specific validations
			self._validate_digital_twin_constraints(entity)
		elif entity.entity_type == EntityType.WORKFLOW:
			# Workflow-specific validations
			self._validate_workflow_constraints(entity)
		elif entity.entity_type == EntityType.DATABASE:
			# Database-specific validations
			self._validate_database_constraints(entity)
	
	def _validate_agent_constraints(self, entity: EntityDeclaration):
		"""Validate agent-specific constraints"""
		if isinstance(entity, AIAgentDeclaration):
			if not entity.model:
				self.errors.append(SemanticError(
					f"AI agent '{entity.name}' must declare a model",
					entity
				))
			if not entity.system_prompt and not entity.role:
				self.warnings.append(SemanticError(
					f"AI agent '{entity.name}' should declare a role or system prompt",
					entity,
					"warning"
				))
			if entity.runtime and entity.runtime not in KNOWN_AI_AGENT_RUNTIMES:
				self.warnings.append(SemanticError(
					f"AI agent '{entity.name}' uses custom runtime '{entity.runtime}'; ensure an adapter is registered",
					entity,
					"warning"
				))
			if len(set(entity.capabilities)) != len(entity.capabilities):
				self.errors.append(SemanticError(
					f"AI agent '{entity.name}' declares duplicate capabilities",
					entity
				))
			return

		# Check for required 'process' method
		has_process_method = any(
			method.name == 'process' for method in entity.methods
		)
		
		if not has_process_method:
			self.warnings.append(SemanticError(
				f"Agent '{entity.name}' should have a 'process' method",
				entity,
				"warning"
			))

	def _validate_capability_constraints(self, entity: EntityDeclaration):
		"""Validate first-class capability declaration shape."""
		if not isinstance(entity, CapabilityDeclaration):
			return
		if not entity.contract:
			self.errors.append(SemanticError(
				f"Capability '{entity.name}' must declare a contract",
				entity
			))
		if not entity.provides:
			self.errors.append(SemanticError(
				f"Capability '{entity.name}' must declare at least one provided capability or service",
				entity
			))
		if len(set(entity.provides)) != len(entity.provides):
			self.errors.append(SemanticError(
				f"Capability '{entity.name}' declares duplicate provided services",
				entity
			))
		if len(set(entity.requires)) != len(entity.requires):
			self.errors.append(SemanticError(
				f"Capability '{entity.name}' declares duplicate required services",
				entity
			))
		for rule_set_name, rules in {
			"rules": entity.rules,
			"business_rules": entity.business_rules,
		}.items():
			for rule in rules:
				if "name" not in rule:
					self.errors.append(SemanticError(
						f"Capability '{entity.name}' has a {rule_set_name} entry without a name",
						entity
					))
				# Phase 2: parse rule `when` conditions and store the AST
				when_cond = rule.get("when") or rule.get("condition")
				if when_cond and isinstance(when_cond, str):
					try:
						from .rule_expr import parse_rule_expr, expr_to_dict
						expr_ast = parse_rule_expr(when_cond)
						if expr_ast is not None:
							rule["when_ast"] = expr_to_dict(expr_ast)
					except Exception:
						pass  # parse failure → no AST, no error (lenient)

	def _validate_workflow_constraints(self, entity: EntityDeclaration) -> None:
		"""Validate WorkflowDeclaration state graph consistency."""
		from .ast_builder import WorkflowDeclaration
		if not isinstance(entity, WorkflowDeclaration):
			return
		if not entity.states:
			# Warn when a workflow body exists but declares no step transitions
			if not entity.steps_raw:
				self.warnings.append(SemanticError(
					f"Workflow '{entity.name}' defines no steps or states",
					entity, "warning"
				))
			return
		state_set = set(entity.states)
		for task in entity.human_tasks:
			if task and task not in state_set:
				self.warnings.append(SemanticError(
					f"Workflow '{entity.name}': human_task '{task}' is not a declared state",
					entity, "workflow"
				))
		for state_key in entity.guards:
			if state_key and state_key not in state_set:
				self.warnings.append(SemanticError(
					f"Workflow '{entity.name}': guard key '{state_key}' is not a declared state",
					entity, "workflow"
				))

	def _validate_agent_team_constraints(self, entity: EntityDeclaration):
		"""Validate AI agent team shape before cross-reference checks."""
		if isinstance(entity, AgentTeamDeclaration):
			if not entity.agents:
				self.errors.append(SemanticError(
					f"Agent team '{entity.name}' must include at least one agent",
					entity
				))
			if len(set(entity.capabilities)) != len(entity.capabilities):
				self.errors.append(SemanticError(
					f"Agent team '{entity.name}' declares duplicate capabilities",
					entity
				))

	def _validate_application_constraints(self, entity: EntityDeclaration):
		"""Validate first-class application composition shape."""
		if not isinstance(entity, ApplicationDeclaration):
			return
		if not (entity.capabilities or entity.components or entity.routes):
			self.warnings.append(SemanticError(
				f"Application '{entity.name}' should compose capabilities, components, or routes",
				entity,
				"warning"
			))
		for field_name, values in {
			"capabilities": entity.capabilities,
			"agents": entity.agents,
			"agent_teams": entity.agent_teams,
			"routes": entity.routes,
		}.items():
			if len(set(values)) != len(values):
				self.errors.append(SemanticError(
					f"Application '{entity.name}' declares duplicate {field_name}",
					entity
				))

	def _validate_agent_composition(self, module: ModuleDeclaration):
		"""Validate references between first-class AI agents and teams."""
		agent_names = {
			entity.name for entity in module.entities
			if isinstance(entity, AIAgentDeclaration) or entity.entity_type == EntityType.AI_AGENT
		}

		for entity in module.entities:
			if isinstance(entity, AIAgentDeclaration):
				for edge in entity.handoffs:
					self._validate_agent_edge(edge.source, edge.target, agent_names, entity)
			elif isinstance(entity, AgentTeamDeclaration):
				for agent_name in entity.agents:
					if agent_name not in agent_names:
						self.errors.append(SemanticError(
							f"Agent team '{entity.name}' references unknown agent '{agent_name}'",
							entity
						))
				for edge in entity.flow:
					self._validate_agent_edge(edge.source, edge.target, agent_names, entity)

	def _validate_agent_edge(self, source: str, target: str, agent_names: Set[str], node: ASTNode):
		"""Validate one handoff edge."""
		if source not in agent_names:
			self.errors.append(SemanticError(
				f"Agent handoff references unknown source agent '{source}'",
				node
			))
		if target not in agent_names:
			self.errors.append(SemanticError(
				f"Agent handoff references unknown target agent '{target}'",
				node
			))
	
	def _validate_digital_twin_constraints(self, entity: EntityDeclaration):
		"""Validate digital twin-specific constraints"""
		# Check for state-related properties
		has_state = any(
			'state' in prop.name.lower() for prop in entity.properties
		)
		
		if not has_state:
			self.warnings.append(SemanticError(
				f"Digital twin '{entity.name}' should have state-related properties",
				entity,
				"warning"
			))
	
	def _validate_database_constraints(self, entity: EntityDeclaration):
		"""Validate database-specific constraints"""
		# Check for connection properties
		has_connection_config = (
			isinstance(entity, DatabaseDeclaration)
			and any(entity.connection_config.get(key) is not None for key in ["url", "host", "port", "database"])
		)
		has_connection_property = any(
			prop.name in ['url', 'host', 'port', 'database'] 
			for prop in entity.properties
		)
		
		if not (has_connection_config or has_connection_property):
			self.warnings.append(SemanticError(
				f"Database '{entity.name}' should have connection configuration",
				entity,
				"warning"
			))
	
	def _validate_method_semantics(self, method: MethodDeclaration):
		"""Validate method semantic rules"""
		self.current_method = method
		
		# Create method scope
		method_scope = SymbolTable(self.current_scope)
		previous_scope = self.current_scope
		self.current_scope = method_scope
		
		try:
			# Declare parameters in method scope
			for param in method.parameters:
				param_symbol = Symbol(
					name=param.name,
					symbol_type=self._apg_type_from_annotation(param.type_annotation),
					declared_type=param.type_annotation,
					declaration_node=param,
					is_mutable=True
				)
				self.current_scope.define(param_symbol)
			
			# Validate method body if present
			if method.body:
				self._validate_statement_semantics(method.body)
		
		finally:
			self.current_scope = previous_scope
			self.current_method = None
	
	def _validate_statement_semantics(self, stmt: Statement):
		"""Validate statement semantics"""
		if isinstance(stmt, BlockStatement):
			for s in stmt.statements:
				self._validate_statement_semantics(s)
		
		elif isinstance(stmt, AssignmentStatement):
			# Check if target exists and is mutable
			symbol = self.current_scope.lookup(stmt.target)
			if symbol and not symbol.is_mutable:
				self.errors.append(SemanticError(
					f"Cannot assign to immutable symbol '{stmt.target}'",
					stmt
				))
		
		elif isinstance(stmt, ReturnStatement):
			# Check return type compatibility
			if self.current_method and self.current_method.return_type:
				expected_type = self._apg_type_from_annotation(self.current_method.return_type)
				actual_type = APGType.VOID if stmt.value is None else self._infer_expression_type(stmt.value)
				if not self._types_compatible(expected_type, actual_type):
					self.errors.append(SemanticError(
						f"Method '{self.current_method.name}' returns {actual_type.value}, expected {expected_type.value}",
						stmt
					))
	
	# ========================================
	# Phase 4: Dead Code Analysis
	# ========================================
	
	def _analyze_dead_code(self, module: ModuleDeclaration):
		"""Analyze for dead code and unused symbols"""
		for entity in module.entities:
			self._analyze_entity_dead_code(entity)
	
	def _analyze_entity_dead_code(self, entity: EntityDeclaration):
		"""Analyze dead code within an entity"""
		# Check for unused properties (simple heuristic)
		for prop in entity.properties:
			if not self._is_property_used(prop, entity):
				self.warnings.append(SemanticError(
					f"Property '{prop.name}' appears to be unused",
					prop,
					"warning"
				))
		
		# Check for unused methods
		for method in entity.methods:
			if not self._is_method_used(method, entity):
				self.warnings.append(SemanticError(
					f"Method '{method.name}' appears to be unused",
					method,
					"warning"
				))
	
	def _is_property_used(self, prop: PropertyDeclaration, entity: EntityDeclaration) -> bool:
		"""Check if a property is used (simplified heuristic)"""
		# APG properties are usually declarative table, form, screen, capability,
		# and configuration fields. Treat them as contract surface, not dead code.
		return True
	
	def _is_method_used(self, method: MethodDeclaration, entity: EntityDeclaration) -> bool:
		"""Check if a method is used (simplified heuristic)"""
		# Special methods are always considered used
		special_methods = {'process', 'init', 'main', 'setup', 'teardown'}
		return method.name in special_methods
	
	# ========================================
	# Utility Methods
	# ========================================
	
	def _apg_type_from_annotation(self, annotation: TypeAnnotation) -> APGType:
		"""Convert type annotation to APGType"""
		type_map = {
			'str': APGType.STRING,
			'int': APGType.INTEGER,
			'float': APGType.FLOAT,
			'bool': APGType.BOOLEAN,
			'list': APGType.LIST,
			'dict': APGType.DICT,
			'void': APGType.VOID,
			'any': APGType.ANY
		}
		
		return type_map.get(annotation.type_name, APGType.ANY)

	def _infer_expression_type(self, expression: Expression) -> APGType:
		"""Infer a conservative APG type for executable semantic checks."""
		if isinstance(expression, LiteralExpression):
			literal_map = {
				'string': APGType.STRING,
				'integer': APGType.INTEGER,
				'int': APGType.INTEGER,
				'float': APGType.FLOAT,
				'boolean': APGType.BOOLEAN,
				'bool': APGType.BOOLEAN,
				'null': APGType.NULL,
			}
			return literal_map.get(expression.literal_type, APGType.ANY)
		if isinstance(expression, IdentifierExpression):
			symbol = self.current_scope.lookup(expression.name)
			return symbol.symbol_type if symbol else APGType.ANY
		if isinstance(expression, ListExpression):
			return APGType.LIST
		if isinstance(expression, DictExpression):
			return APGType.DICT
		if isinstance(expression, CallExpression):
			if isinstance(expression.function, IdentifierExpression):
				signature = self.builtin_functions.get(expression.function.name)
				if signature and signature.return_type:
					return self._apg_type_from_annotation(signature.return_type)
			return APGType.ANY
		if isinstance(expression, BinaryExpression):
			left_type = self._infer_expression_type(expression.left)
			right_type = self._infer_expression_type(expression.right)
			if expression.operator in {'==', '!=', '<', '>', '<=', '>=', '&&', '||', 'in'}:
				return APGType.BOOLEAN
			if APGType.FLOAT in {left_type, right_type}:
				return APGType.FLOAT
			if left_type == right_type == APGType.INTEGER:
				return APGType.INTEGER
			if expression.operator == '+' and left_type == right_type == APGType.STRING:
				return APGType.STRING
			return APGType.ANY
		return APGType.ANY

	def _types_compatible(self, expected: APGType, actual: APGType) -> bool:
		"""Return whether an inferred type can satisfy a declared type."""
		if expected == APGType.ANY or actual == APGType.ANY:
			return True
		if actual == APGType.NULL:
			return expected in {APGType.ANY, APGType.NULL}
		if expected == APGType.FLOAT and actual == APGType.INTEGER:
			return True
		return expected == actual
	
	def _initialize_builtins(self) -> Dict[str, FunctionSignature]:
		"""Initialize built-in functions"""
		return {
			'print': FunctionSignature('print', [], None),
			'len': FunctionSignature('len', [], TypeAnnotation(type_name='int')),
			'str': FunctionSignature('str', [], TypeAnnotation(type_name='str')),
			'int': FunctionSignature('int', [], TypeAnnotation(type_name='int')),
			'float': FunctionSignature('float', [], TypeAnnotation(type_name='float')),
			'bool': FunctionSignature('bool', [], TypeAnnotation(type_name='bool')),
		}
	
	def has_errors(self) -> bool:
		"""Check if analysis found any errors"""
		return len(self.errors) > 0
	
	def has_warnings(self) -> bool:
		"""Check if analysis found any warnings"""
		return len(self.warnings) > 0
	
	def _resolve_references(self, module: ModuleDeclaration) -> None:
		"""Phase 5: validate cross-entity references (warnings, not errors).

		Checks that `requires`, `provides`, `capabilities`, `agents`, and
		`binds` identifiers refer to entities declared in the module.
		Unknown references from installed system capabilities produce warnings
		rather than errors because those packages may not be in scope.
		"""
		# Build local name sets
		local_names: frozenset[str] = frozenset(e.name for e in module.entities)
		capability_provides: frozenset[str] = frozenset(
			name
			for e in module.entities
			if isinstance(e, CapabilityDeclaration)
			for name in (e.provides or [])
		)

		# Load known system capability IDs from MANIFEST.json (best-effort, mtime-cached)
		known_system = _load_known_system_capabilities()

		# Pre-compute union once; membership tests are O(1) on frozenset
		all_known: frozenset[str] = local_names | capability_provides | known_system

		def _is_known(ref: str) -> bool:
			return ref in all_known

		for entity in module.entities:
			# Capability requires/provides cross-check
			if isinstance(entity, CapabilityDeclaration):
				for ref in (entity.requires or []):
					if ref and not _is_known(ref):
						self.warnings.append(SemanticError(
							f"Capability '{entity.name}' requires '{ref}' which is not declared in this module",
							entity, "reference"
						))

			# Application capabilities/agents cross-check (same rule as capability.requires:
			# system capabilities from MANIFEST are accepted without local declaration)
			if isinstance(entity, ApplicationDeclaration):
				for ref in (entity.capabilities or []):
					if ref and not _is_known(ref):
						self.warnings.append(SemanticError(
							f"Application '{entity.name}' references capability '{ref}' not declared in this module",
							entity, "reference"
						))
				for ref in (entity.agents or []):
					if ref and ref not in local_names:
						self.warnings.append(SemanticError(
							f"Application '{entity.name}' references agent '{ref}' not declared in this module",
							entity, "reference"
						))

			# Agent capabilities cross-check
			if isinstance(entity, AIAgentDeclaration):
				for ref in (entity.capabilities or []):
					if ref and not _is_known(ref):
						self.warnings.append(SemanticError(
							f"Agent '{entity.name}' references capability '{ref}' not declared or provided in this module",
							entity, "reference"
						))

			if isinstance(entity, DatabaseDeclaration):
				self._validate_database_relationship_references(entity)

	def _validate_database_relationship_references(self, database: DatabaseDeclaration) -> None:
		"""Validate DBML column references point at declared tables."""
		for schema in database.schemas:
			table_names = {table.name for table in schema.tables}
			for table in schema.tables:
				for column in table.columns:
					reference = column.reference or {}
					target_table = reference.get("table")
					if target_table and target_table not in table_names:
						self.errors.append(SemanticError(
							f"Relationship in database '{database.name}' references undefined entity/table '{target_table}'",
							database,
							"reference",
						))

	def print_errors(self):
		"""Print all errors and warnings"""
		if self.errors:
			print(f"✗ Found {len(self.errors)} semantic error(s):")
			for error in self.errors:
				print(f"  {error}")
		
		if self.warnings:
			print(f"⚠ Found {len(self.warnings)} warning(s):")
			for warning in self.warnings:
				print(f"  {warning}")
		
		if not self.errors and not self.warnings:
			print("✓ No semantic errors or warnings")


def test_semantic_analyzer():
	"""Test the semantic analyzer"""
	print("Semantic Analyzer module loaded successfully")
	print("Classes available:", [
		'SemanticAnalyzer', 'SymbolTable', 'Symbol', 
		'SemanticError', 'APGType', 'FunctionSignature'
	])


if __name__ == "__main__":
	test_semantic_analyzer()
