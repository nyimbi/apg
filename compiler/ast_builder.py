"""
APG AST Builder Module
======================

Converts ANTLR parse trees into structured Abstract Syntax Tree (AST) nodes.
Provides a clean, typed representation of APG programs for semantic analysis and code generation.
"""

import sys
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ── module-level grammar keyword cache ──────────────────────────────────────
# Reads spec/apg.g4 once per process; shared across all ASTBuilder instances.
_GRAMMAR_ENTITY_KEYWORDS_CACHE: Optional[frozenset[str]] = None


def _grammar_entity_keywords() -> frozenset[str]:
	"""Return the set of APG entity-type keywords from spec/apg.g4.

	Result is cached process-wide after the first call.
	"""
	global _GRAMMAR_ENTITY_KEYWORDS_CACHE
	if _GRAMMAR_ENTITY_KEYWORDS_CACHE is not None:
		return _GRAMMAR_ENTITY_KEYWORDS_CACHE
	keywords: set[str] = {
		"module", "agent", "capability", "digital_twin", "workflow",
		"database", "db",
	}
	grammar_path = Path(__file__).resolve().parent.parent / "spec" / "apg.g4"
	try:
		grammar = grammar_path.read_text(encoding="utf-8")
		m = re.search(r"^entity_type\s*\n\s*:(.*?)\n\s*;", grammar, flags=re.MULTILINE | re.DOTALL)
		if m:
			keywords.update(re.findall(r"'([^']+)'", m.group(1)))
	except OSError:
		keywords.update({"app", "flow", "screen", "twin", "agent_runtime"})
	_GRAMMAR_ENTITY_KEYWORDS_CACHE = frozenset(keywords)
	return _GRAMMAR_ENTITY_KEYWORDS_CACHE


# Import generated ANTLR parsers
sys.path.append(str(Path(__file__).parent.parent / "spec"))

try:
	from apgParser import apgParser
	from apgVisitor import apgVisitor
except ImportError:
	apgParser = apgVisitor = None


# ========================================
# AST Node Type Definitions
# ========================================

@dataclass(kw_only=True)
class ASTNode:
	"""Base class for all AST nodes"""
	line: int = 0
	column: int = 0
	source_file: Optional[str] = None


@dataclass
class ModuleDeclaration(ASTNode):
	"""APG module declaration"""
	name: str = ""
	version: str = "1.0.0"
	description: Optional[str] = None
	author: Optional[str] = None
	license: Optional[str] = None
	imports: List['ImportDeclaration'] = field(default_factory=list)
	exports: List['ExportDeclaration'] = field(default_factory=list)
	entities: List['EntityDeclaration'] = field(default_factory=list)
	workflows: List[Any] = field(default_factory=list)
	module_name: Optional[str] = None

	def __post_init__(self) -> None:
		if self.module_name and not self.name:
			self.name = self.module_name


@dataclass
class ImportDeclaration(ASTNode):
	"""Import statement"""
	module_name: str
	import_items: List[str] = field(default_factory=list)  # Empty means import all
	alias: Optional[str] = None


@dataclass
class ExportDeclaration(ASTNode):
	"""Export statement"""
	export_items: List[str]


class EntityType(Enum):
	"""Types of APG entities"""
	ENTITY = "entity"
	AGENT = "agent"
	AI_AGENT = "ai_agent"
	AGENT_TEAM = "agent_team"
	SWARM = "swarm"
	APP = "app"
	APPLICATION = "application"
	CAPABILITY = "capability"
	DIGITAL_TWIN = "digital_twin"
	WORKFLOW = "workflow"
	FLOW = "flow"
	DATABASE = "database"
	API = "api"
	FORM = "form"
	SCREEN = "screen"
	UI_COMPONENT = "ui_component"
	RULE = "rule"
	RULE_SET = "rule_set"
	POLICY = "policy"
	AGENT_RUNTIME = "agent_runtime"
	NOTIFICATION = "notification"
	ANALYTICS = "analytics"
	# New types added for enum, statemachine, migration, deployment, marketplace, event sourcing
	ENUM = "enum"
	STATEMACHINE = "statemachine"
	MIGRATION = "migration"
	DEPLOYMENT = "deployment"
	MARKETPLACE = "marketplace"
	EVENT_STORE = "event_store"


@dataclass
class EntityDeclaration(ASTNode):
	"""Base class for all entity declarations"""
	entity_type: EntityType
	name: str
	properties: List['PropertyDeclaration'] = field(default_factory=list)
	methods: List['MethodDeclaration'] = field(default_factory=list)


@dataclass
class AgentMemory(ASTNode):
	"""Memory attached to an AI agent"""
	kind: str
	name: Optional[str] = None


@dataclass
class AgentHandoff(ASTNode):
	"""Directed handoff between two AI agents"""
	source: str
	target: str
	condition: str = "done"


@dataclass
class AIAgentDeclaration(EntityDeclaration):
	"""First-class AI agent declaration"""
	role: Optional[str] = None
	model: Optional[str] = None
	runtime: Optional[str] = None
	system_prompt: Optional[str] = None
	capabilities: List[str] = field(default_factory=list)
	tools: List[str] = field(default_factory=list)
	memory: Optional[AgentMemory] = None
	inputs: List[str] = field(default_factory=list)
	outputs: List[str] = field(default_factory=list)
	handoffs: List[AgentHandoff] = field(default_factory=list)
	configuration: Dict[str, Any] = field(default_factory=dict)
	rules: List[Dict[str, Any]] = field(default_factory=list)
	ui: Dict[str, Any] = field(default_factory=dict)
	theme: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTeamDeclaration(EntityDeclaration):
	"""Composition of first-class AI agents"""
	agents: List[str] = field(default_factory=list)
	capabilities: List[str] = field(default_factory=list)
	flow: List[AgentHandoff] = field(default_factory=list)
	policy: Dict[str, Any] = field(default_factory=dict)
	configuration: Dict[str, Any] = field(default_factory=dict)
	rules: List[Dict[str, Any]] = field(default_factory=list)
	ui: Dict[str, Any] = field(default_factory=dict)
	theme: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition(ASTNode):
	"""A single state-machine transition within a workflow."""
	source: str
	target: str
	guard: Optional[str] = None  # raw condition string from guards dict


@dataclass
class WorkflowDeclaration(EntityDeclaration):
	"""First-class workflow declaration with a typed state graph."""
	steps_raw: str = ""                              # original "a -> b -> c" string
	states: List[str] = field(default_factory=list)  # ordered list of state names
	transitions: List[Transition] = field(default_factory=list)
	human_tasks: List[str] = field(default_factory=list)
	guards: Dict[str, str] = field(default_factory=dict)
	assignments: Dict[str, str] = field(default_factory=dict)
	timers: Dict[str, str] = field(default_factory=dict)
	waits: Dict[str, str] = field(default_factory=dict)
	retry_policy: Dict[str, str] = field(default_factory=dict)
	compensation: Dict[str, str] = field(default_factory=dict)


@dataclass
class CapabilityDeclaration(EntityDeclaration):
	"""First-class composable APG capability declaration"""
	contract: Dict[str, Any] = field(default_factory=dict)
	provides: List[str] = field(default_factory=list)
	requires: List[str] = field(default_factory=list)
	configuration: Dict[str, Any] = field(default_factory=dict)
	rules: List[Dict[str, Any]] = field(default_factory=list)
	rule_engine: Dict[str, Any] = field(default_factory=dict)
	ui: Dict[str, Any] = field(default_factory=dict)
	theme: Dict[str, Any] = field(default_factory=dict)
	runtime: Dict[str, Any] = field(default_factory=dict)
	erp_modules: List[str] = field(default_factory=list)
	components: Any = field(default_factory=dict)
	business_rules: List[Dict[str, Any]] = field(default_factory=list)
	approvals: Any = field(default_factory=dict)
	master_data: Any = field(default_factory=dict)
	i18n: Dict[str, Any] = field(default_factory=dict)
	streaming: Dict[str, Any] = field(default_factory=dict)
	screens: Any = field(default_factory=dict)


@dataclass
class ApplicationDeclaration(EntityDeclaration):
	"""First-class APG application composition declaration"""
	description: Optional[str] = None
	capabilities: List[str] = field(default_factory=list)
	agents: List[str] = field(default_factory=list)
	agent_teams: List[str] = field(default_factory=list)
	components: Any = field(default_factory=dict)
	screens: Any = field(default_factory=dict)
	routes: List[str] = field(default_factory=list)
	workflows: List[str] = field(default_factory=list)
	policies: Any = field(default_factory=dict)
	configuration: Dict[str, Any] = field(default_factory=dict)
	theme: Dict[str, Any] = field(default_factory=dict)
	runtime: Dict[str, Any] = field(default_factory=dict)
	integrations: Any = field(default_factory=dict)
	deployments: Any = field(default_factory=dict)


@dataclass
class PropertyDeclaration(ASTNode):
	"""Property/field declaration within an entity"""
	name: str
	type_annotation: 'TypeAnnotation'
	default_value: Optional['Expression'] = None
	is_required: bool = False
	validation_rules: List['ValidationRule'] = field(default_factory=list)


@dataclass
class MethodDeclaration(ASTNode):
	"""Method declaration within an entity"""
	name: str
	parameters: List['Parameter'] = field(default_factory=list)
	return_type: Optional['TypeAnnotation'] = None
	body: Optional['BlockStatement'] = None
	is_async: bool = False


@dataclass
class Parameter(ASTNode):
	"""Method parameter"""
	name: str
	type_annotation: 'TypeAnnotation'
	default_value: Optional['Expression'] = None


@dataclass
class TypeAnnotation(ASTNode):
	"""Type annotation for variables, parameters, etc."""
	type_name: str
	generic_args: List['TypeAnnotation'] = field(default_factory=list)
	is_optional: bool = False
	is_list: bool = False
	is_dict: bool = False

	def __post_init__(self) -> None:
		if isinstance(self.generic_args, bool):
			self.is_optional = self.generic_args
			self.generic_args = []


@dataclass
class ValidationRule(ASTNode):
	"""Validation rule for properties"""
	rule_type: str  # e.g., "range", "pattern", "required"
	parameters: Dict[str, Any] = field(default_factory=dict)


# ========================================
# Statement AST Nodes
# ========================================

@dataclass
class Statement(ASTNode):
	"""Base class for all statements"""
	node_category: str = field(default="statement", init=False)


@dataclass
class BlockStatement(Statement):
	"""Block of statements enclosed in braces"""
	statements: List[Statement] = field(default_factory=list)


@dataclass
class ExpressionStatement(Statement):
	"""Statement that contains a single expression"""
	expression: 'Expression'


@dataclass
class AssignmentStatement(Statement):
	"""Assignment statement"""
	target: str
	value: 'Expression'
	operator: str = "="  # =, +=, -=, etc.


@dataclass
class ReturnStatement(Statement):
	"""Return statement"""
	value: Optional['Expression'] = None


@dataclass
class IfStatement(Statement):
	"""Conditional statement"""
	condition: 'Expression'
	then_branch: Statement
	else_branch: Optional[Statement] = None


@dataclass
class ForStatement(Statement):
	"""For loop statement"""
	variable: str
	iterable: 'Expression'
	body: Statement


@dataclass
class WhileStatement(Statement):
	"""While loop statement"""
	condition: 'Expression'
	body: Statement


# ========================================
# Expression AST Nodes
# ========================================

@dataclass
class Expression(ASTNode):
	"""Base class for all expressions"""
	node_category: str = field(default="expression", init=False)


@dataclass
class LiteralExpression(Expression):
	"""Literal value (string, number, boolean, etc.)"""
	value: Any
	literal_type: str  # "string", "integer", "float", "boolean", "null"


@dataclass
class IdentifierExpression(Expression):
	"""Variable/identifier reference"""
	name: str


@dataclass
class BinaryExpression(Expression):
	"""Binary operation expression"""
	left: Expression
	operator: str
	right: Expression


@dataclass
class UnaryExpression(Expression):
	"""Unary operation expression"""
	operator: str
	operand: Expression


@dataclass
class CallExpression(Expression):
	"""Function/method call expression"""
	function: Expression
	arguments: List[Expression] = field(default_factory=list)


@dataclass
class MemberExpression(Expression):
	"""Member access expression (obj.property)"""
	object: Expression
	property: str


@dataclass
class IndexExpression(Expression):
	"""Index access expression (obj[index])"""
	object: Expression
	index: Expression


@dataclass
class ListExpression(Expression):
	"""List literal expression"""
	elements: List[Expression] = field(default_factory=list)


@dataclass
class DictExpression(Expression):
	"""Dictionary literal expression"""
	pairs: List[tuple[Expression, Expression]] = field(default_factory=list)


@dataclass
class LambdaExpression(Expression):
	"""Lambda/anonymous function expression"""
	parameters: List[Parameter] = field(default_factory=list)
	body: Optional[Expression] = None


# ========================================
# Database-specific AST Nodes
# ========================================

@dataclass
class DatabaseDeclaration(EntityDeclaration):
	"""Database entity declaration"""
	connection_config: Dict[str, Any] = field(default_factory=dict)
	schemas: List['DatabaseSchema'] = field(default_factory=list)


@dataclass
class DatabaseSchema(ASTNode):
	"""Database schema definition"""
	name: str
	tables: List['TableDeclaration'] = field(default_factory=list)
	views: List['ViewDeclaration'] = field(default_factory=list)
	procedures: List['ProcedureDeclaration'] = field(default_factory=list)
	triggers: List['TriggerDeclaration'] = field(default_factory=list)


@dataclass
class TableDeclaration(ASTNode):
	"""Database table declaration"""
	name: str
	columns: List['ColumnDeclaration'] = field(default_factory=list)
	indexes: List['IndexDeclaration'] = field(default_factory=list)
	constraints: List['ConstraintDeclaration'] = field(default_factory=list)


@dataclass
class ColumnDeclaration(ASTNode):
	"""Database column declaration"""
	name: str
	data_type: str
	is_primary_key: bool = False
	is_nullable: bool = True
	default_value: Optional[Any] = None
	constraints: List[str] = field(default_factory=list)
	reference: Optional[Dict[str, str]] = None


@dataclass
class IndexDeclaration(ASTNode):
	"""Database index declaration"""
	name: Optional[str]
	columns: List[str]
	is_unique: bool = False
	index_type: Optional[str] = None  # btree, hash, gin, gist, etc.


@dataclass
class TriggerDeclaration(ASTNode):
	"""Database trigger declaration"""
	name: str
	timing: str  # before, after, instead_of
	events: List[str]  # insert, update, delete
	table_name: str
	body: BlockStatement


@dataclass
class ProcedureDeclaration(ASTNode):
	"""Database stored procedure/function declaration"""
	name: str
	parameters: List[Parameter] = field(default_factory=list)
	return_type: Optional[TypeAnnotation] = None
	body: BlockStatement = field(default_factory=BlockStatement)
	language: str = "sql"


# ========================================
# AST Builder Visitor
# ========================================

class ASTBuilder(apgVisitor if apgVisitor else object):
	"""
	Converts ANTLR parse trees to APG AST nodes.
	Visits the parse tree and constructs a clean, typed AST representation.
	"""
	
	def __init__(self):
		super().__init__()
		self.current_source_file: Optional[str] = None
		self.errors: List[str] = []
		self._source_entity_keyword_cache: Optional[set[str]] = None
	
	def build_ast(self, parse_tree, source_file: Optional[str] = None) -> Optional[ModuleDeclaration]:
		"""
		Build AST from parse tree.
		
		Args:
			parse_tree: ANTLR parse tree from parser
			source_file: Source file path for error reporting
			
		Returns:
			Root AST node (ModuleDeclaration) or None if parsing failed
		"""
		self.current_source_file = source_file
		self.errors.clear()
		
		try:
			if isinstance(parse_tree, ModuleDeclaration):
				parse_tree.source_file = source_file
				return parse_tree
			if hasattr(parse_tree, "source_code"):
				# Prefer the ANTLR visitor when the real parse tree is available
				# Use ANTLR visitor only when the ANTLR parse was error-free
				antlr_tree = getattr(parse_tree, "antlr_tree", None)
				antlr_clean = getattr(parse_tree, "antlr_clean", False)
				if antlr_tree is not None and antlr_clean:
					from .antlr_ast_visitor import build_ast_from_antlr
					antlr_src = getattr(parse_tree, "antlr_source", parse_tree.source_code)
					result = build_ast_from_antlr(antlr_tree, antlr_src, source_file)
					if result is not None:
						return result
				# Fall back to the regex-based source parser
				return self._build_source_ast(parse_tree.source_code, source_file)
			return self.visit(parse_tree)
		except Exception as e:
			self.errors.append(f"AST building failed: {e}")
			return None

	def _build_source_ast(self, source_code: str, source_file: Optional[str]) -> ModuleDeclaration:
		"""Build a lightweight AST from APG source text for legacy grammar coverage."""

		cleaned = self._strip_comments(source_code)
		module_match = re.search(
			r"\bmodule\s+(?P<name>[^\s{]+)\s+version\s+(?P<version>[^\s{]+)\s*\{",
			cleaned,
			re.UNICODE,
		)
		module = ModuleDeclaration(
			name=module_match.group("name") if module_match else "main",
			version=module_match.group("version") if module_match else "1.0.0",
			source_file=source_file,
		)

		# Parse import statements: "import foo.bar;" or "from foo.bar import Baz;"
		for m in re.finditer(
			r"\bfrom\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([A-Za-z_*][A-Za-z0-9_,\s*]*);",
			cleaned,
		):
			module_name = m.group(1).strip()
			items_raw = m.group(2).strip()
			items = [i.strip() for i in items_raw.split(",") if i.strip() and i.strip() != "*"]
			module.imports.append(ImportDeclaration(
				module_name=module_name,
				import_items=items,
			))
		for m in re.finditer(
			r"(?<!['\"])(?:^|\n)\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)\s*(?:as\s+([A-Za-z_][A-Za-z0-9_]*))?\s*;",
			cleaned,
		):
			module_name = m.group(1).strip()
			alias = m.group(2).strip() if m.group(2) else None
			module.imports.append(ImportDeclaration(
				module_name=module_name,
				import_items=[],
				alias=alias,
			))

		for kind, name, body in self._iter_source_entities(cleaned):
			if kind == "module":
				continue
			if kind in {"app", "application", "composition"}:
				module.entities.append(self._parse_source_application(kind, name, body, source_file))
				continue
			if kind == "capability":
				module.entities.append(self._parse_source_capability(name, body, source_file))
				continue
			if kind in {"db", "database"}:
				module.entities.append(self._parse_source_database(name, body, source_file))
				continue
			if kind == "agent" and self._is_agent_config_body(body):
				module.entities.append(self._parse_source_agent(name, body, source_file))
				continue
			if kind in {"workflow", "flow"}:
				module.entities.append(self._parse_source_workflow(name, body, source_file, kind=kind))
				continue
			# Form and business-logic entities contain nested sub-block syntax
			# (widget properties, inline object literals, async def method bodies)
			# that the flat line scanner misreads as top-level entity properties,
			# causing spurious duplicate-property errors. Register these with no
			# flat properties — the full ANTLR grammar path handles them correctly.
			if kind in {
				# Form layout / UI
				"form", "screen", "view", "ui", "component", "widget",
				# Business logic (async def bodies contain inline object literals)
				"biz", "service", "logic", "controller",
				# Test / notification / metrics (also contain nested object syntax)
				"test", "notification", "notify", "metrics", "logger",
			}:
				entity_type = self._entity_type_for_source_kind(kind)
				module.entities.append(EntityDeclaration(
					entity_type=entity_type,
					name=name,
					properties=[],
					methods=[],
					source_file=source_file,
				))
				continue
			properties, methods = self._parse_source_members(body, source_file)
			entity_type = self._entity_type_for_source_kind(kind)
			module.entities.append(EntityDeclaration(
				entity_type=entity_type,
				name=name,
				properties=properties,
				methods=methods,
				source_file=source_file,
			))

		return module

	def _strip_comments(self, source_code: str) -> str:
		"""Strip APG comments while preserving comment markers inside strings."""
		result: List[str] = []
		index = 0
		quote: Optional[str] = None
		escaped = False

		while index < len(source_code):
			char = source_code[index]
			next_char = source_code[index + 1:index + 2]

			if quote:
				result.append(char)
				if escaped:
					escaped = False
				elif char == "\\":
					escaped = True
				elif char == quote:
					quote = None
				index += 1
				continue

			if char in {"'", '"'}:
				quote = char
				result.append(char)
				index += 1
				continue

			if char == "/" and next_char == "/":
				index += 2
				while index < len(source_code) and source_code[index] != "\n":
					index += 1
				continue

			if char == "/" and next_char == "*":
				index += 2
				while index + 1 < len(source_code) and source_code[index:index + 2] != "*/":
					if source_code[index] == "\n":
						result.append("\n")
					index += 1
				index += 2
				continue

			result.append(char)
			index += 1

		return "".join(result)

	def _iter_source_entities(self, source_code: str):
		keywords = self._source_entity_keywords()
		pattern = re.compile(
			r"\b(" + "|".join(
				re.escape(keyword)
				for keyword in sorted(keywords, key=len, reverse=True)
			) + r")\s+([^\s{]+)\s*(?:version\s+[^\s{]+)?\s*\{",
			re.UNICODE,
		)
		position = 0
		while True:
			match = pattern.search(source_code, position)
			if not match:
				break
			body_start = match.end()
			depth = 1
			index = body_start
			while index < len(source_code) and depth:
				if source_code[index] == "{":
					depth += 1
				elif source_code[index] == "}":
					depth -= 1
				index += 1
			yield match.group(1), match.group(2), source_code[body_start:index - 1]
			position = index

	def _source_entity_keywords(self) -> set[str]:
		"""Return grammar-backed entity keywords for source-backed AST building.

		Result is cached process-wide in _GRAMMAR_ENTITY_KEYWORDS so the
		apg.g4 file is read at most once per process regardless of how many
		ASTBuilder instances are created.
		"""
		return _grammar_entity_keywords()

	def _entity_type_for_source_kind(self, kind: str) -> EntityType:
		"""Map source keywords to AST entity categories while preserving key APG surfaces."""
		return {
			"agent": EntityType.AGENT,
			"team": EntityType.AGENT_TEAM,
			"agent_team": EntityType.AGENT_TEAM,
			"swarm": EntityType.SWARM,
			"app": EntityType.APP,
			"application": EntityType.APPLICATION,
			"digital_twin": EntityType.DIGITAL_TWIN,
			"twin": EntityType.DIGITAL_TWIN,
			"workflow": EntityType.WORKFLOW,
			"flow": EntityType.FLOW,
			"db": EntityType.DATABASE,
			"database": EntityType.DATABASE,
			"api": EntityType.API,
			"form": EntityType.FORM,
			"screen": EntityType.SCREEN,
			"view": EntityType.SCREEN,
			"ui": EntityType.UI_COMPONENT,
			"component": EntityType.UI_COMPONENT,
			"widget": EntityType.UI_COMPONENT,
			"rule": EntityType.RULE,
			"rule_set": EntityType.RULE_SET,
			"policy": EntityType.POLICY,
			"agent_runtime": EntityType.AGENT_RUNTIME,
			"report": EntityType.ANALYTICS,
			"dashboard": EntityType.ANALYTICS,
			"notify": EntityType.NOTIFICATION,
			# Enum and type system
			"enum": EntityType.ENUM,
			"interface": EntityType.ENTITY,
			"type_alias": EntityType.ENTITY,
			"struct": EntityType.ENTITY,
			# State machines
			"statemachine": EntityType.STATEMACHINE,
			"state_machine": EntityType.STATEMACHINE,
			"fsm": EntityType.STATEMACHINE,
			# Event sourcing
			"event_schema": EntityType.EVENT_STORE,
			"event_store": EntityType.EVENT_STORE,
			"projection": EntityType.EVENT_STORE,
			"aggregate": EntityType.EVENT_STORE,
			# Database lifecycle
			"migration": EntityType.MIGRATION,
			"seed": EntityType.MIGRATION,
			"fixture_data": EntityType.MIGRATION,
			# Deployment and platform
			"deployment_strategy": EntityType.DEPLOYMENT,
			"deployment_pattern": EntityType.DEPLOYMENT,
			"marketplace": EntityType.MARKETPLACE,
			"ecommerce": EntityType.MARKETPLACE,
			"platform": EntityType.ENTITY,
			# Reporting and analytics
			"pipeline": EntityType.ENTITY,
			"etl": EntityType.ENTITY,
			"dbt_model": EntityType.ENTITY,
		}.get(kind, EntityType.ENTITY)

	def _parse_source_capability(self, name: str, body: str, source_file: Optional[str]) -> CapabilityDeclaration:
		"""Parse the first-class capability contract surface from source text."""
		from .ai_agent_composition import _dict_value, _parse_properties, _rule_list, _string_list

		props = _parse_properties(body)
		contract = _dict_value(props.get("contract", props.get("capability_contract")))

		def contract_value(key: str, default: Any = None) -> Any:
			return contract.get(key, props.get(key, default))

		return CapabilityDeclaration(
			entity_type=EntityType.CAPABILITY,
			name=name,
			contract=contract,
			provides=_string_list(contract_value("provides")),
			requires=_string_list(contract_value("requires")),
			configuration=_dict_value(contract_value("configuration", props.get("config"))),
			rules=_rule_list(contract_value("rules")),
			rule_engine=_dict_value(contract_value("rule_engine")),
			ui=_dict_value(contract_value("ui")),
			theme=_dict_value(contract_value("theme")),
			runtime=_dict_value(contract_value("runtime")),
			erp_modules=_string_list(props.get("erp_modules")),
			components=props.get("components", {}),
			business_rules=_rule_list(props.get("business_rules")),
			approvals=props.get("approvals", {}),
			master_data=props.get("master_data", {}),
			i18n=_dict_value(props.get("i18n", props.get("localization"))),
			streaming=_dict_value(props.get("streaming")),
			screens=contract_value("screens", props.get("screens", {})),
			source_file=source_file,
		)

	def _parse_source_application(
		self,
		kind: str,
		name: str,
		body: str,
		source_file: Optional[str],
	) -> ApplicationDeclaration:
		"""Parse first-class application composition metadata from source text."""
		from .ai_agent_composition import _dict_value, _optional_string, _parse_properties, _string_list

		props = _parse_properties(body)
		description = _optional_string(props.get("description"))
		return ApplicationDeclaration(
			entity_type=self._entity_type_for_source_kind(kind),
			name=name,
			description=description,
			capabilities=_string_list(props.get("capabilities", props.get("capability"))),
			agents=_string_list(props.get("agents", props.get("agent"))),
			agent_teams=_string_list(props.get("agent_teams", props.get("teams"))),
			components=props.get("components", {}),
			screens=props.get("screens", {}),
			routes=_string_list(props.get("routes", props.get("route"))),
			workflows=_string_list(props.get("workflows", props.get("flows"))),
			policies=props.get("policies", props.get("policy", {})),
			configuration=_dict_value(props.get("config", props.get("configuration"))),
			theme=_dict_value(props.get("theme")),
			runtime=_dict_value(props.get("runtime")),
			integrations=props.get("integrations", {}),
			deployments=props.get("deployments", props.get("deployment", {})),
			source_file=source_file,
		)

	def _parse_source_database(
		self,
		name: str,
		body: str,
		source_file: Optional[str],
	) -> DatabaseDeclaration:
		"""Parse DB connection metadata and DBML schemas from source text."""
		from .ai_agent_composition import _parse_value, _split_statements

		body_without_schemas = self._remove_source_blocks(body, {"schema"})
		connection_config: Dict[str, Any] = {}
		properties: List[PropertyDeclaration] = []

		for statement in _split_statements(body_without_schemas):
			if ":" not in statement:
				continue
			key, value = statement.split(":", 1)
			config_key = key.strip()
			if not config_key:
				continue
			parsed_value = _parse_value(value.strip())
			connection_config[config_key] = parsed_value
			properties.append(PropertyDeclaration(
				name=config_key,
				type_annotation=self._parse_source_type(type(parsed_value).__name__, source_file),
				default_value=parsed_value,
				source_file=source_file,
			))

		return DatabaseDeclaration(
			entity_type=EntityType.DATABASE,
			name=name,
			properties=properties,
			connection_config=connection_config,
			schemas=[
				self._parse_source_database_schema(schema_name, schema_body, source_file)
				for schema_name, schema_body in self._iter_named_source_blocks(body, {"schema"})
			],
			source_file=source_file,
		)

	def _parse_source_database_schema(
		self,
		name: str,
		body: str,
		source_file: Optional[str],
	) -> DatabaseSchema:
		"""Parse a DBML schema body into database AST nodes."""
		return DatabaseSchema(
			name=name,
			tables=[
				self._parse_source_table(table_name, table_body, source_file)
				for table_name, table_body in self._iter_named_source_blocks(body, {"table"})
			],
			source_file=source_file,
		)

	def _parse_source_table(
		self,
		name: str,
		body: str,
		source_file: Optional[str],
	) -> TableDeclaration:
		"""Parse a DBML table body into column and index declarations."""
		return TableDeclaration(
			name=name,
			columns=self._parse_source_columns(self._remove_source_blocks(body, {"indexes"}), source_file),
			indexes=self._parse_source_indexes(body, source_file),
			source_file=source_file,
		)

	def _parse_source_columns(
		self,
		body: str,
		source_file: Optional[str],
	) -> List[ColumnDeclaration]:
		"""Parse DBML column declarations from a table body."""
		columns: List[ColumnDeclaration] = []
		column_pattern = re.compile(
			r"^\s*(?P<name>[^\W\d]\w*)\s+"
			r"(?P<type>[^\s\[]+(?:\([^)]*\))?)"
			r"(?:\s+(?P<nullable>not\s+null|null))?"
			r"(?:\s*\[(?P<constraints>[^\]]*)\])?\s*$",
			re.UNICODE,
		)

		for line in body.splitlines():
			stripped = line.strip()
			if not stripped or stripped.startswith(("constraint ", "trigger ", "vector_index ")):
				continue
			match = column_pattern.match(stripped)
			if not match:
				continue
			constraints = self._split_source_options(match.group("constraints") or "")
			nullable_text = (match.group("nullable") or "").lower()
			constraint_text = " ".join(constraints).lower()
			columns.append(ColumnDeclaration(
				name=match.group("name"),
				data_type=match.group("type"),
				is_primary_key="pk" in constraints or "primary key" in constraint_text,
				is_nullable=not ("not null" in nullable_text or "not null" in constraint_text),
				default_value=self._extract_source_option_value(constraints, "default"),
				constraints=constraints,
				reference=self._extract_source_reference(constraints),
				source_file=source_file,
			))
		return columns

	def _parse_source_indexes(
		self,
		body: str,
		source_file: Optional[str],
	) -> List[IndexDeclaration]:
		"""Parse DBML indexes blocks from a table body."""
		indexes: List[IndexDeclaration] = []
		for _block_name, indexes_body in self._iter_named_source_blocks(body, {"indexes"}):
			for line in indexes_body.splitlines():
				stripped = line.strip()
				if not stripped:
					continue
				options_match = re.search(r"\[(?P<options>[^\]]*)\]\s*$", stripped)
				options = self._split_source_options(options_match.group("options") if options_match else "")
				columns_text = stripped[:options_match.start()].strip() if options_match else stripped
				columns_text = columns_text.strip("()")
				columns = [column.strip() for column in columns_text.split(",") if column.strip()]
				if not columns:
					continue
				indexes.append(IndexDeclaration(
					name=self._extract_source_option_value(options, "name"),
					columns=columns,
					is_unique="unique" in options,
					index_type=self._extract_source_option_value(options, "type"),
					source_file=source_file,
				))
		return indexes

	_TYPED_PROP_RE = re.compile(r":\s*(str|int|float|bool|bytes|list|dict|Any)\b|\s*->\s*\w+\s*=\s*\{")

	def _is_agent_config_body(self, body: str) -> bool:
		"""Return True when body uses agent config syntax (role:/model:) not typed entity properties."""
		if self._TYPED_PROP_RE.search(body):
			return False
		return bool(re.search(r"\b(role|model|system|capabilities|tools|memory)\s*:", body))

	def _parse_source_agent(self, name: str, body: str, source_file: Optional[str]) -> AIAgentDeclaration:
		"""Parse agent { role:...; model:...; ... } into AIAgentDeclaration."""
		from .ai_agent_composition import _parse_properties, _string_list, _rule_list

		props = _parse_properties(body)

		def _strval(key: str) -> str:
			v = props.get(key, "")
			return str(v).strip('"\'') if v else ""

		mem_raw = props.get("memory", "")
		memory: Optional[AgentMemory] = None
		if mem_raw:
			parts = str(mem_raw).split()
			if len(parts) >= 2:
				memory = AgentMemory(kind=parts[0], name=parts[-1])

		return AIAgentDeclaration(
			entity_type=EntityType.AGENT,
			name=name,
			source_file=source_file,
			role=_strval("role"),
			model=_strval("model"),
			system_prompt=_strval("system"),
			capabilities=_string_list(props.get("capabilities", [])),
			tools=_string_list(props.get("tools", [])),
			memory=memory,
			configuration=props.get("configuration", {}),
			rules=_rule_list(props.get("rules", [])),
		)

	def _parse_source_workflow(self, name: str, body: str, source_file: Optional[str], kind: str = "workflow") -> WorkflowDeclaration:
		"""Parse workflow { steps: ...; human_tasks: ...; guards: ...; } into a typed state graph."""
		from .ai_agent_composition import _parse_properties, _string_list

		props = _parse_properties(body)

		# Parse steps string "a -> b -> c" into states + transitions
		# Handle `steps: str = "..."` — strip any `str =` type annotation prefix
		steps_val = str(props.get("steps", "")).strip()
		import re as _re
		steps_val = _re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*', '', steps_val)
		steps_raw = steps_val.strip().strip('"\'')
		states: List[str] = []
		transitions: List[Transition] = []
		if steps_raw:
			parts = [s.strip() for s in steps_raw.split("->") if s.strip()]
			states = parts
			for i in range(len(parts) - 1):
				transitions.append(Transition(source=parts[i], target=parts[i + 1]))

		# Parse guards: {state: "condition"} dict
		# Strip "dict =" prefix when written as "guards: dict = {...}"
		guards_raw = props.get("guards", {})
		if isinstance(guards_raw, str):
			guards_raw = _re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*', '', guards_raw.strip())
			from .ai_agent_composition import _parse_value as _pv
			guards_raw = _pv(guards_raw)
		guards: Dict[str, str] = {}
		if isinstance(guards_raw, dict):
			for k, v in guards_raw.items():
				guards[str(k)] = str(v).strip('"\'')
			# Attach guard conditions to transitions
			for t in transitions:
				if t.target in guards:
					t.guard = guards[t.target]

		# Parse human_tasks: [state, state, ...] or "str = state, state"
		human_tasks_raw = props.get("human_tasks", [])
		if isinstance(human_tasks_raw, str):
			human_tasks_raw = _re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*["\']*', '', human_tasks_raw.strip()).strip('"\'')
			human_tasks_raw = [t.strip() for t in human_tasks_raw.split(",") if t.strip()]
		human_tasks = _string_list(human_tasks_raw)

		# Parse assignments: {state: role} or "dict = {...}"
		assignments_raw = props.get("assignments", {})
		if isinstance(assignments_raw, str):
			assignments_raw = _re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*', '', assignments_raw.strip())
			from .ai_agent_composition import _parse_value as _pv2
			assignments_raw = _pv2(assignments_raw)
		assignments: Dict[str, str] = {}
		if isinstance(assignments_raw, dict):
			for k, v in assignments_raw.items():
				assignments[str(k)] = str(v)

		def _dict_prop(key: str) -> Dict[str, str]:
			v = props.get(key, {})
			if isinstance(v, str):
				v = _re.sub(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s*', '', v.strip())
				from .ai_agent_composition import _parse_value as _pv3
				v = _pv3(v)
			return {str(k): str(val) for k, val in v.items()} if isinstance(v, dict) else {}

		return WorkflowDeclaration(
			entity_type=self._entity_type_for_source_kind(kind),
			name=name,
			source_file=source_file,
			steps_raw=steps_raw,
			states=states,
			transitions=transitions,
			human_tasks=human_tasks,
			guards=guards,
			assignments=assignments,
			timers=_dict_prop("timers"),
			waits=_dict_prop("waits"),
			retry_policy=_dict_prop("retry_policy"),
			compensation=_dict_prop("compensation"),
		)

	def _parse_source_members(self, body: str, source_file: Optional[str]):
		properties: List[PropertyDeclaration] = []
		methods: List[MethodDeclaration] = []

		method_pattern = re.compile(
			r"(?P<name>[^\W\d]\w*)\s*:\s*(?P<async>async\s*)?\((?P<params>[^)]*)\)\s*->\s*(?P<return>[^\s={;]+)",
			re.UNICODE,
		)
		method_spans = []
		for match in method_pattern.finditer(body):
			method_spans.append(self._source_method_span(body, match))
			methods.append(MethodDeclaration(
				name=match.group("name"),
				parameters=self._parse_source_parameters(match.group("params"), source_file),
				return_type=self._parse_source_type(match.group("return"), source_file),
				body=None,
				is_async=bool(match.group("async")),
				source_file=source_file,
			))

		property_body = body
		for start, end in reversed(method_spans):
			property_body = property_body[:start] + property_body[end:]

		for line in property_body.splitlines():
			match = re.match(
				r"\s*(?P<name>[^\W\d]\w*)\s*:\s*(?P<type>[^=;{]+?)\s*(?:=\s*(?P<default>.*?))?\s*;\s*$",
				line,
				re.UNICODE,
			)
			if not match:
				continue
			type_text = match.group("type").strip()
			if type_text.startswith("(") or type_text.startswith("async"):
				continue
			properties.append(PropertyDeclaration(
				name=match.group("name"),
				type_annotation=self._parse_source_type(type_text, source_file),
				default_value=match.group("default"),
				source_file=source_file,
			))

		return properties, methods

	def _source_method_span(self, body: str, match: re.Match) -> tuple[int, int]:
		brace_start = body.find("{", match.end())
		if brace_start == -1:
			return match.span()

		depth = 1
		index = brace_start + 1
		while index < len(body) and depth:
			if body[index] == "{":
				depth += 1
			elif body[index] == "}":
				depth -= 1
			index += 1

		while index < len(body) and body[index].isspace():
			index += 1
		if index < len(body) and body[index] == ";":
			index += 1
		return match.start(), index

	def _iter_named_source_blocks(self, body: str, block_names: set[str]):
		"""Yield named source blocks such as schema/table blocks."""
		header = re.compile(
			r"\b(" + "|".join(re.escape(name) for name in sorted(block_names)) + r")\b(?:\s+([^\s{\[]+))?[^{]*\{",
			re.UNICODE,
		)
		position = 0
		while True:
			match = header.search(body, position)
			if not match:
				break
			open_brace = body.find("{", match.end() - 1)
			close_brace = self._find_source_matching_brace(body, open_brace)
			if close_brace < 0:
				break
			yield match.group(2) or match.group(1), body[open_brace + 1:close_brace]
			position = close_brace + 1

	def _remove_source_blocks(self, body: str, block_names: set[str]) -> str:
		"""Remove named nested blocks before parsing top-level statements."""
		header = re.compile(
			r"\b(" + "|".join(re.escape(name) for name in sorted(block_names)) + r")\b(?:\s+[^\s{\[]+)?[^{]*\{",
			re.UNICODE,
		)
		ranges: List[tuple[int, int]] = []
		for match in header.finditer(body):
			open_brace = body.find("{", match.end() - 1)
			close_brace = self._find_source_matching_brace(body, open_brace)
			if close_brace >= 0:
				ranges.append((match.start(), close_brace + 1))
		if not ranges:
			return body

		result: List[str] = []
		start = 0
		for range_start, range_end in ranges:
			result.append(body[start:range_start])
			start = range_end
		result.append(body[start:])
		return "".join(result)

	def _find_source_matching_brace(self, source: str, open_brace: int) -> int:
		"""Find a matching brace while respecting quoted strings."""
		if open_brace < 0:
			return -1
		depth = 0
		quote: Optional[str] = None
		escaped = False
		for index in range(open_brace, len(source)):
			char = source[index]
			if quote:
				if escaped:
					escaped = False
				elif char == "\\":
					escaped = True
				elif char == quote:
					quote = None
				continue
			if char in {"'", '"'}:
				quote = char
			elif char == "{":
				depth += 1
			elif char == "}":
				depth -= 1
				if depth == 0:
					return index
		return -1

	def _split_source_options(self, options_text: str) -> List[str]:
		"""Split DBML bracket options while preserving option values."""
		if not options_text.strip():
			return []
		options: List[str] = []
		start = 0
		quote: Optional[str] = None
		for index, char in enumerate(options_text):
			if quote:
				if char == quote and options_text[index - 1:index] != "\\":
					quote = None
				continue
			if char in {"'", '"'}:
				quote = char
			elif char == ",":
				option = options_text[start:index].strip()
				if option:
					options.append(option)
				start = index + 1
		last_option = options_text[start:].strip()
		if last_option:
			options.append(last_option)
		return options

	def _extract_source_option_value(self, options: List[str], key: str) -> Optional[str]:
		"""Extract a `key: value` DBML option value."""
		prefix = f"{key}:"
		for option in options:
			if option.startswith(prefix):
				return option[len(prefix):].strip().strip("'\"")
		return None

	def _extract_source_reference(self, options: List[str]) -> Optional[Dict[str, str]]:
		"""Extract a DBML column reference option."""
		for option in options:
			match = re.match(
				r"ref\s*:\s*(?P<kind><>|>|<|-)\s*(?:(?P<schema>[^\W\d]\w*)\.)?(?P<table>[^\W\d]\w*)\.(?P<column>[^\W\d]\w*)$",
				option,
				flags=re.UNICODE,
			)
			if not match:
				continue
			reference = {
				"kind": match.group("kind"),
				"relationship": {
					">": "many_to_one",
					"<": "one_to_many",
					"-": "one_to_one",
					"<>": "many_to_many",
				}.get(match.group("kind"), "references"),
				"table": match.group("table"),
				"column": match.group("column"),
				"target": (
					f"{match.group('schema')}.{match.group('table')}.{match.group('column')}"
					if match.group("schema")
					else f"{match.group('table')}.{match.group('column')}"
				),
			}
			if match.group("schema"):
				reference["schema"] = match.group("schema")
			return reference
		return None

	def _parse_source_parameters(self, params: str, source_file: Optional[str]) -> List[Parameter]:
		parsed: List[Parameter] = []
		for raw_param in [part.strip() for part in params.split(",") if part.strip()]:
			name_type, _, default = raw_param.partition("=")
			name, _, type_text = name_type.partition(":")
			if name.strip() and type_text.strip():
				parsed.append(Parameter(
					name=name.strip(),
					type_annotation=self._parse_source_type(type_text.strip(), source_file),
					default_value=default.strip() or None,
					source_file=source_file,
				))
		return parsed

	def _parse_source_type(self, type_text: str, source_file: Optional[str]) -> TypeAnnotation:
		type_text = type_text.strip()
		generic_match = re.match(r"(?P<name>[^\[]+)\[(?P<args>.*)\]$", type_text)
		if not generic_match:
			return TypeAnnotation(type_name=type_text, source_file=source_file)
		args = [
			self._parse_source_type(part.strip(), source_file)
			for part in generic_match.group("args").split(",")
			if part.strip()
		]
		return TypeAnnotation(
			type_name=generic_match.group("name").strip(),
			generic_args=args,
			is_list=generic_match.group("name").strip() == "list",
			is_dict=generic_match.group("name").strip() == "dict",
			source_file=source_file,
		)
	
	def _get_position(self, ctx) -> tuple[int, int]:
		"""Extract line and column position from parse tree context"""
		if hasattr(ctx, 'start') and ctx.start:
			return ctx.start.line, ctx.start.column
		return 0, 0
	
	def _create_node(self, node_class, ctx, **kwargs):
		"""Create AST node with position information"""
		line, column = self._get_position(ctx)
		return node_class(
			line=line,
			column=column,
			source_file=self.current_source_file,
			**kwargs
		)
	
	# ========================================
	# Visit Methods for Core Language Constructs
	# ========================================
	
	def visitProgram(self, ctx):
		"""Visit the root program node"""
		# Find module declaration
		module_ctx = None
		entities = []
		
		for child in ctx.children:
			if hasattr(child, 'getRuleIndex'):
				rule_name = self._get_rule_name(child.getRuleIndex())
				if rule_name == 'module_declaration':
					module_ctx = child
				elif rule_name in ['agent', 'digital_twin', 'workflow', 'database', 'api']:
					entity = self.visit(child)
					if entity:
						entities.append(entity)
		
		# Create module or default one
		if module_ctx:
			module = self.visit(module_ctx)
			if module:
				module.entities.extend(entities)
				return module
		
		# Create default module if none declared
		return self._create_node(ModuleDeclaration, ctx,
			name="main",
			version="1.0.0",
			entities=entities
		)
	
	def visitModule_declaration(self, ctx):
		"""Visit module declaration"""
		name = self._extract_module_name(ctx)
		version = self._extract_module_version(ctx)
		
		# Extract optional properties
		description = self._extract_module_property(ctx, 'description')
		author = self._extract_module_property(ctx, 'author')
		license_prop = self._extract_module_property(ctx, 'license')
		
		return self._create_node(ModuleDeclaration, ctx,
			name=name,
			version=version,
			description=description,
			author=author,
			license=license_prop
		)
	
	def visitAgent(self, ctx):
		"""Visit agent declaration"""
		name = self._extract_identifier(ctx)
		properties, methods = self._extract_entity_members(ctx)
		
		return self._create_node(EntityDeclaration, ctx,
			entity_type=EntityType.AGENT,
			name=name,
			properties=properties,
			methods=methods
		)
	
	def visitDigital_twin(self, ctx):
		"""Visit digital twin declaration"""
		name = self._extract_identifier(ctx)
		properties, methods = self._extract_entity_members(ctx)
		
		return self._create_node(EntityDeclaration, ctx,
			entity_type=EntityType.DIGITAL_TWIN,
			name=name,
			properties=properties,
			methods=methods
		)
	
	def visitWorkflow(self, ctx):
		"""Visit workflow declaration"""
		name = self._extract_identifier(ctx)
		properties, methods = self._extract_entity_members(ctx)
		
		return self._create_node(EntityDeclaration, ctx,
			entity_type=EntityType.WORKFLOW,
			name=name,
			properties=properties,
			methods=methods
		)
	
	def visitDatabase(self, ctx):
		"""Visit database declaration"""
		name = self._extract_identifier(ctx)
		connection_config = self._extract_database_config(ctx)
		schemas = self._extract_database_schemas(ctx)
		
		return self._create_node(DatabaseDeclaration, ctx,
			entity_type=EntityType.DATABASE,
			name=name,
			connection_config=connection_config,
			schemas=schemas
		)
	
	# ========================================
	# Expression Visitors
	# ========================================
	
	def visitLiteral(self, ctx):
		"""Visit literal expression"""
		text = ctx.getText()
		
		# Determine literal type and value
		if text.startswith('"') and text.endswith('"'):
			return self._create_node(LiteralExpression, ctx,
				value=text[1:-1],  # Remove quotes
				literal_type="string"
			)
		elif text.startswith("'") and text.endswith("'"):
			return self._create_node(LiteralExpression, ctx,
				value=text[1:-1],  # Remove quotes
				literal_type="string"
			)
		elif text.isdigit():
			return self._create_node(LiteralExpression, ctx,
				value=int(text),
				literal_type="integer"
			)
		elif self._is_float(text):
			return self._create_node(LiteralExpression, ctx,
				value=float(text),
				literal_type="float"
			)
		elif text in ['true', 'false']:
			return self._create_node(LiteralExpression, ctx,
				value=text == 'true',
				literal_type="boolean"
			)
		elif text == 'null':
			return self._create_node(LiteralExpression, ctx,
				value=None,
				literal_type="null"
			)
		else:
			# Default to string
			return self._create_node(LiteralExpression, ctx,
				value=text,
				literal_type="string"
			)
	
	def visitIdentifier(self, ctx):
		"""Visit identifier expression"""
		name = ctx.getText()
		return self._create_node(IdentifierExpression, ctx, name=name)
	
	# ========================================
	# Helper Methods
	# ========================================
	
	def _get_rule_name(self, rule_index: int) -> str:
		"""Get rule name from parser rule index"""
		if apgParser and hasattr(apgParser, 'ruleNames'):
			if 0 <= rule_index < len(apgParser.ruleNames):
				return apgParser.ruleNames[rule_index]
		return f"rule_{rule_index}"
	
	def _extract_identifier(self, ctx) -> str:
		"""Extract identifier name from context"""
		for child in ctx.children:
			if hasattr(child, 'symbol') and child.symbol.type == apgParser.IDENTIFIER:
				return child.getText()
		return "unknown"
	
	def _extract_module_name(self, ctx) -> str:
		"""Extract module name from module declaration"""
		# Look for module name after 'module' keyword
		for i, child in enumerate(ctx.children):
			if child.getText() == 'module' and i + 1 < len(ctx.children):
				return ctx.children[i + 1].getText()
		return "unnamed"
	
	def _extract_module_version(self, ctx) -> str:
		"""Extract module version from module declaration"""
		# Look for version after 'version' keyword
		for i, child in enumerate(ctx.children):
			if child.getText() == 'version' and i + 1 < len(ctx.children):
				version_text = ctx.children[i + 1].getText()
				return version_text.strip('"\'')  # Remove quotes
		return "1.0.0"
	
	def _extract_module_property(self, ctx, property_name: str) -> Optional[str]:
		"""Extract optional module property"""
		for i, child in enumerate(ctx.children):
			if child.getText() == property_name and i + 2 < len(ctx.children):
				if ctx.children[i + 1].getText() == ':':
					value_text = ctx.children[i + 2].getText()
					return value_text.strip('";\'')  # Remove quotes and semicolon
		return None
	
	def _extract_entity_members(self, ctx) -> tuple[List[PropertyDeclaration], List[MethodDeclaration]]:
		"""Extract properties and methods from entity body"""
		properties = []
		methods = []
		
		# This is a simplified implementation
		# In practice, you'd need to traverse the entity body and identify property/method declarations
		
		return properties, methods
	
	def _extract_database_config(self, ctx) -> Dict[str, Any]:
		"""Extract database connection configuration"""
		# Simplified implementation
		return {}
	
	def _extract_database_schemas(self, ctx) -> List[DatabaseSchema]:
		"""Extract database schemas"""
		# Simplified implementation
		return []
	
	def _is_float(self, text: str) -> bool:
		"""Check if text represents a float"""
		try:
			float(text)
			return '.' in text
		except ValueError:
			return False


def test_ast_builder():
	"""Test the AST builder with sample APG code"""
	# This would require the parser to be available
	if not apgParser:
		print("ANTLR parsers not available - skipping AST builder test")
		return
	
	print("AST Builder module loaded successfully")
	print("Classes available:", [
		'ModuleDeclaration', 'EntityDeclaration', 'PropertyDeclaration',
		'MethodDeclaration', 'Expression', 'Statement', 'ASTBuilder'
	])


if __name__ == "__main__":
	test_ast_builder()
