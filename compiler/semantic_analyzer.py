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
import sys
from pathlib import Path

# Import AST nodes
from .ast_builder import (
	ASTNode, ModuleDeclaration, EntityDeclaration, PropertyDeclaration,
	MethodDeclaration, Parameter, TypeAnnotation, Expression, Statement,
	LiteralExpression, IdentifierExpression, BinaryExpression, CallExpression,
	AssignmentStatement, ReturnStatement, BlockStatement, EntityType,
	AIAgentDeclaration, AgentTeamDeclaration, ListExpression, DictExpression
)


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
	node: ASTNode
	error_type: str = "semantic"
	
	def __str__(self) -> str:
		location = f"{self.node.source_file or 'unknown'}:{self.node.line}:{self.node.column}"
		return f"{location}: {self.error_type} error: {self.message}"


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
		self.builtin_functions = self._initialize_builtins()
	
	def analyze(self, ast: ModuleDeclaration) -> Dict[str, Any]:
		"""
		Perform semantic analysis on the AST.
		
		Args:
			ast: Root AST node (ModuleDeclaration)
			
		Returns:
			Analysis results including errors, warnings, and symbol table
		"""
		self.errors.clear()
		self.warnings.clear()
		self.current_module = ast
		
		try:
			# Phase 1: Symbol declaration - declare all entities and their members
			self._declare_module_symbols(ast)
			
			# Phase 2: Type resolution - resolve all type references
			self._resolve_types(ast)
			
			# Phase 3: Semantic validation - check method bodies, expressions, etc.
			self._validate_semantics(ast)
			
			# Phase 4: Dead code analysis
			self._analyze_dead_code(ast)
			
			return {
				'success': len(self.errors) == 0,
				'errors': self.errors.copy(),
				'warnings': self.warnings.copy(),
				'symbol_table': self.symbol_table,
				'module': ast
			}
			
		except Exception as e:
			self.errors.append(SemanticError(
				f"Internal analyzer error: {e}",
				ast,
				"internal"
			))
			return {
				'success': False,
				'errors': self.errors.copy(),
				'warnings': self.warnings.copy(),
				'symbol_table': self.symbol_table,
				'module': ast
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
				f"Entity '{entity.name}' is already defined",
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
		prop_symbol = Symbol(
			name=prop.name,
			symbol_type=self._apg_type_from_annotation(prop.type_annotation),
			declared_type=prop.type_annotation,
			declaration_node=prop,
			is_mutable=True  # APG properties are mutable by default
		)
		
		if not self.current_scope.define(prop_symbol):
			self.errors.append(SemanticError(
				f"Property '{prop.name}' is already defined in this entity",
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
		
		# Resolve property types
		for prop in entity.properties:
			if not self._is_valid_type(prop.type_annotation):
				self.errors.append(SemanticError(
					f"Unknown type '{prop.type_annotation.type_name}' for property '{prop.name}'",
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
					f"Unknown type '{param.type_annotation.type_name}' for parameter '{param.name}'",
					param
				))
		
		# Resolve return type
		if method.return_type and not self._is_valid_type(method.return_type):
			self.errors.append(SemanticError(
				f"Unknown return type '{method.return_type.type_name}' for method '{method.name}'",
				method
			))
	
	def _is_valid_type(self, type_annotation: TypeAnnotation) -> bool:
		"""Check if a type annotation is valid"""
		# Check built-in types
		if type_annotation.type_name in self.builtin_types:
			return True
		
		# Check if it's a defined entity
		symbol = self.symbol_table.lookup(type_annotation.type_name)
		if symbol and symbol.symbol_type == APGType.ENTITY:
			return True
		
		return False
	
	# ========================================
	# Phase 3: Semantic Validation
	# ========================================
	
	def _validate_semantics(self, module: ModuleDeclaration):
		"""Validate semantic rules"""
		for entity in module.entities:
			self._validate_entity_semantics(entity)
		self._validate_agent_composition(module)
	
	def _validate_entity_semantics(self, entity: EntityDeclaration):
		"""Validate entity-specific semantic rules"""
		self.current_entity = entity
		
		# Validate entity type constraints
		self._validate_entity_type_constraints(entity)
		
		# Validate methods
		for method in entity.methods:
			self._validate_method_semantics(method)
	
	def _validate_entity_type_constraints(self, entity: EntityDeclaration):
		"""Validate constraints specific to entity types"""
		if entity.entity_type in {EntityType.AGENT, EntityType.AI_AGENT}:
			# Agent-specific validations
			self._validate_agent_constraints(entity)
		elif entity.entity_type == EntityType.AGENT_TEAM:
			self._validate_agent_team_constraints(entity)
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
	
	def _validate_workflow_constraints(self, entity: EntityDeclaration):
		"""Validate workflow-specific constraints"""
		# Check for steps or stages
		has_steps = any(
			'step' in prop.name.lower() or 'stage' in prop.name.lower() 
			for prop in entity.properties
		)
		
		if not has_steps:
			self.warnings.append(SemanticError(
				f"Workflow '{entity.name}' should define steps or stages",
				entity,
				"warning"
			))
	
	def _validate_database_constraints(self, entity: EntityDeclaration):
		"""Validate database-specific constraints"""
		# Check for connection properties
		has_connection = any(
			prop.name in ['url', 'host', 'port', 'database'] 
			for prop in entity.properties
		)
		
		if not has_connection:
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
		# This is a simplified check - in practice, would need full usage analysis
		return len(entity.methods) > 0  # Assume used if entity has methods
	
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
