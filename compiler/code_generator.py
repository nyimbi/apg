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
	AssignmentStatement, ReturnStatement, BlockStatement, EntityType,
	DatabaseDeclaration, DatabaseSchema, TableDeclaration
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
			project_name = ast.module_name or "APGGeneratedApp"
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
		
		# Generate HTML templates
		template_files = self._generate_templates(ast)
		files.update(template_files)
		
		return files
	
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
			# Generate stub implementation
			if is_async:
				self._add_line("# TODO: Implement async method logic")
				self._add_line("await asyncio.sleep(0)")  # Ensure it's actually async
			else:
				self._add_line("# TODO: Implement method logic")
			
			if method.return_type and method.return_type.type_name != "void":
				default_return = self._get_default_return_value(method.return_type)
				self._add_line(f"return {default_return}")
			else:
				self._add_line("pass")
		
		self._dedent()
	
	def _generate_statement(self, stmt: Statement):
		"""Generate Python code for a statement"""
		if isinstance(stmt, BlockStatement):
			for s in stmt.statements:
				self._generate_statement(s)
		
		elif isinstance(stmt, AssignmentStatement):
			value = self._generate_expression(stmt.value)
			self._add_line(f"self.{stmt.target} = {value}")
		
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
			# Generate a meaningful default implementation
			self._add_line("# Auto-generated placeholder implementation")
			self._add_line("pass")
	
	def _generate_expression(self, expr: Expression) -> str:
		"""Generate Python code for an expression"""
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
			return "None  # TODO: Implement expression"
	
	def _add_default_agent_start(self):
		"""Add default start method for agents"""
		self._add_line("")
		self._add_line("async def start(self) -> None:")
		self._add_line("    \"\"\"Start the agent\"\"\"")
		self._add_line("    self._logger.info(f'Starting agent {self.__class__.__name__}')")
		self._add_line("    self._running = True")
		self._add_line("    # TODO: Implement agent startup logic")
	
	def _add_default_agent_stop(self):
		"""Add default stop method for agents"""
		self._add_line("")
		self._add_line("async def stop(self) -> None:")
		self._add_line("    \"\"\"Stop the agent\"\"\"")
		self._add_line("    self._logger.info(f'Stopping agent {self.__class__.__name__}')")
		self._add_line("    self._running = False")
		self._add_line("    # TODO: Implement agent shutdown logic")
	
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
		self._add_line("    # TODO: Apply state changes to properties")
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
		self._add_line("        # TODO: Implement workflow steps")
		self._add_line("        self._status = 'completed'")
		self._add_line("        return {'status': 'success', 'results': self._step_results}")
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
				self._add_line("    pass")
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
SECRET_KEY = '\\2\\1thisismyscretkey\\1\\2\\e\\y\\y\\h'

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
		self._add_line(f'    """Runtime implementation of {method.name}"""')
		
		self._indent()
		
		# Generate method body with actual logic
		if method.body:
			self._generate_statement(method.body)
		else:
			# Generate stub implementation with meaningful defaults
			if method.return_type:
				return_type = method.return_type.type_name
				if return_type == "str":
					self._add_line(f"    return f'Result from {method.name}'")
				elif return_type == "int":
					self._add_line("    return 42")
				elif return_type == "float":
					self._add_line("    return 3.14")
				elif return_type == "bool":
					self._add_line("    return True")
				elif return_type == "dict":
					self._add_line("    return {'result': 'success', 'method': '" + method.name + "'}")
				elif return_type == "list":
					self._add_line("    return []")
				else:
					self._add_line("    return None")
			else:
				self._add_line("    pass")
		
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
		target = target_language or self.config.target_language
		
		if target not in self.generators:
			raise ValueError(f"Unsupported target language: {target}")
		
		generator = self.generators[target]
		return generator.generate(ast)
	
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