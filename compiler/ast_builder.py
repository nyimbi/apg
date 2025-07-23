"""
APG AST Builder Module
======================

Converts ANTLR parse trees into structured Abstract Syntax Tree (AST) nodes.
Provides a clean, typed representation of APG programs for semantic analysis and code generation.
"""

import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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

@dataclass
class ASTNode:
	"""Base class for all AST nodes"""
	line: int = 0
	column: int = 0
	source_file: Optional[str] = None


@dataclass
class ModuleDeclaration(ASTNode):
	"""APG module declaration"""
	name: str
	version: str
	description: Optional[str] = None
	author: Optional[str] = None
	license: Optional[str] = None
	imports: List['ImportDeclaration'] = field(default_factory=list)
	exports: List['ExportDeclaration'] = field(default_factory=list)
	entities: List['EntityDeclaration'] = field(default_factory=list)


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
	AGENT = "agent"
	DIGITAL_TWIN = "digital_twin"
	WORKFLOW = "workflow"
	DATABASE = "database"
	API = "api"
	FORM = "form"
	UI_COMPONENT = "ui_component"
	NOTIFICATION = "notification"
	ANALYTICS = "analytics"


@dataclass
class EntityDeclaration(ASTNode):
	"""Base class for all entity declarations"""
	entity_type: EntityType
	name: str
	properties: List['PropertyDeclaration'] = field(default_factory=list)
	methods: List['MethodDeclaration'] = field(default_factory=list)


@dataclass
class PropertyDeclaration(ASTNode):
	"""Property/field declaration within an entity"""
	name: str
	type_annotation: 'TypeAnnotation'
	default_value: Optional['Expression'] = None
	is_required: bool = True
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
	pass


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
	pass


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
	body: Expression


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
	body: BlockStatement
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
			return self.visit(parse_tree)
		except Exception as e:
			self.errors.append(f"AST building failed: {e}")
			return None
	
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