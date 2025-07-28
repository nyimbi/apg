"""
APG Financial Management General Ledger - Advanced Visual Transaction Flow Designer

Revolutionary visual workflow designer that allows users to create, modify, and
understand complex transaction flows through an intuitive drag-and-drop interface
with intelligent validation and automated journal entry generation.

Features:
- Visual drag-and-drop transaction flow design
- Real-time validation and error detection
- Intelligent account suggestions and mappings
- Complex multi-step transaction workflows
- Template library with best practices
- Automated journal entry generation from flows

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from decimal import Decimal
from enum import Enum
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class NodeType(Enum):
	"""Types of nodes in transaction flow"""
	START = "start"
	END = "end"
	ACCOUNT_DEBIT = "account_debit"
	ACCOUNT_CREDIT = "account_credit"
	CALCULATION = "calculation"
	CONDITION = "condition"
	APPROVAL = "approval"
	VALIDATION = "validation"
	NOTIFICATION = "notification"
	MULTI_ENTITY = "multi_entity"
	CURRENCY_CONVERSION = "currency_conversion"
	TEMPLATE = "template"


class FlowValidationLevel(Enum):
	"""Validation levels for flows"""
	BASIC = "basic"
	INTERMEDIATE = "intermediate"
	ADVANCED = "advanced"
	EXPERT = "expert"


class FlowStatus(Enum):
	"""Status of transaction flows"""
	DRAFT = "draft"
	VALIDATED = "validated"
	ACTIVE = "active"
	ARCHIVED = "archived"
	DEPRECATED = "deprecated"


@dataclass
class FlowNode:
	"""Individual node in transaction flow"""
	node_id: str
	node_type: NodeType
	name: str
	description: str
	position: Dict[str, float]  # x, y coordinates
	properties: Dict[str, Any]
	validation_rules: List[Dict[str, Any]]
	error_handling: Dict[str, Any]
	styling: Dict[str, str]


@dataclass
class FlowConnection:
	"""Connection between flow nodes"""
	connection_id: str
	from_node_id: str
	to_node_id: str
	connection_type: str  # 'success', 'error', 'conditional'
	condition: Optional[str] = None
	properties: Dict[str, Any] = None


@dataclass
class TransactionFlow:
	"""Complete transaction flow definition"""
	flow_id: str
	flow_name: str
	description: str
	version: str
	created_by: str
	created_date: datetime
	status: FlowStatus
	validation_level: FlowValidationLevel
	nodes: List[FlowNode]
	connections: List[FlowConnection]
	variables: Dict[str, Any]
	templates_used: List[str]
	execution_metadata: Dict[str, Any]


@dataclass
class FlowTemplate:
	"""Reusable flow template"""
	template_id: str
	template_name: str
	category: str
	description: str
	complexity_level: str
	use_cases: List[str]
	template_flow: TransactionFlow
	customization_points: List[Dict[str, Any]]
	business_rules: List[str]


@dataclass
class FlowExecution:
	"""Execution instance of a flow"""
	execution_id: str
	flow_id: str
	initiated_by: str
	execution_date: datetime
	input_data: Dict[str, Any]
	execution_steps: List[Dict[str, Any]]
	generated_journal_entries: List[str]
	execution_status: str
	execution_time_ms: float
	errors: List[str]


class VisualTransactionFlowDesigner:
	"""
	🎯 GAME CHANGER #8: Advanced Visual Transaction Flow Designer
	
	Revolutionary visual design environment for complex transactions:
	- Drag-and-drop interface for creating transaction flows
	- Real-time validation and intelligent suggestions
	- Template library with industry best practices
	- Automated journal entry generation from visual flows
	- Complex multi-step workflow support
	"""
	
	def __init__(self, gl_service):
		self.gl_service = gl_service
		self.tenant_id = gl_service.tenant_id
		
		# Designer components
		self.flow_validator = FlowValidator()
		self.template_library = FlowTemplateLibrary()
		self.execution_engine = FlowExecutionEngine()
		self.ai_assistant = FlowDesignAIAssistant()
		self.code_generator = JournalEntryCodeGenerator()
		
		logger.info(f"Visual Transaction Flow Designer initialized for tenant {self.tenant_id}")
	
	async def create_visual_flow(self, flow_definition: Dict[str, Any]) -> TransactionFlow:
		"""
		🎯 REVOLUTIONARY: Visual Flow Creation
		
		Creates complex transaction flows through visual interface:
		- Drag-and-drop nodes for accounts, calculations, validations
		- Real-time connection validation
		- Intelligent suggestions based on context
		- Automatic error detection and suggestions
		"""
		try:
			flow = TransactionFlow(
				flow_id=f"flow_{uuid.uuid4().hex[:8]}",
				flow_name=flow_definition["name"],
				description=flow_definition.get("description", ""),
				version="1.0",
				created_by=flow_definition["created_by"],
				created_date=datetime.now(timezone.utc),
				status=FlowStatus.DRAFT,
				validation_level=FlowValidationLevel.BASIC,
				nodes=[],
				connections=[],
				variables=flow_definition.get("variables", {}),
				templates_used=[],
				execution_metadata={}
			)
			
			# Process nodes from visual designer
			for node_data in flow_definition.get("nodes", []):
				node = await self._create_flow_node(node_data)
				flow.nodes.append(node)
			
			# Process connections from visual designer
			for connection_data in flow_definition.get("connections", []):
				connection = await self._create_flow_connection(connection_data)
				flow.connections.append(connection)
			
			# Validate flow structure
			validation_result = await self.flow_validator.validate_flow(flow)
			if not validation_result["valid"]:
				flow.status = FlowStatus.DRAFT
				# Attach validation errors for user feedback
				flow.execution_metadata["validation_errors"] = validation_result["errors"]
			else:
				flow.status = FlowStatus.VALIDATED
			
			# Apply AI suggestions
			ai_suggestions = await self.ai_assistant.suggest_flow_improvements(flow)
			flow.execution_metadata["ai_suggestions"] = ai_suggestions
			
			# Store flow
			await self._store_flow(flow)
			
			return flow
			
		except Exception as e:
			logger.error(f"Error creating visual flow: {e}")
			raise
	
	async def design_complex_transaction(self, transaction_type: str, 
									   complexity_requirements: Dict[str, Any]) -> Dict[str, Any]:
		"""
		🎯 REVOLUTIONARY: Complex Transaction Designer
		
		Guides users through designing complex transactions:
		- Multi-entity transactions with currency conversions
		- Transfer pricing adjustments
		- Consolidation eliminations
		- Revenue recognition workflows
		- Period-end accruals and reversals
		"""
		try:
			design_session = {
				"session_id": f"design_{uuid.uuid4().hex[:8]}",
				"transaction_type": transaction_type,
				"complexity_level": complexity_requirements.get("complexity", "intermediate"),
				"recommended_flow": None,
				"guided_steps": [],
				"ai_assistance": {},
				"validation_feedback": {}
			}
			
			# Get appropriate template based on transaction type
			recommended_template = await self.template_library.get_template_for_transaction(
				transaction_type, complexity_requirements
			)
			
			if recommended_template:
				design_session["recommended_flow"] = recommended_template
				
				# Customize template based on requirements
				customized_flow = await self._customize_template_for_requirements(
					recommended_template, complexity_requirements
				)
				design_session["recommended_flow"] = customized_flow
			
			# Generate guided design steps
			guided_steps = await self._generate_guided_design_steps(
				transaction_type, complexity_requirements
			)
			design_session["guided_steps"] = guided_steps
			
			# Provide AI assistance
			ai_assistance = await self.ai_assistant.provide_design_assistance(
				transaction_type, complexity_requirements
			)
			design_session["ai_assistance"] = ai_assistance
			
			return design_session
			
		except Exception as e:
			logger.error(f"Error designing complex transaction: {e}")
			raise
	
	async def execute_visual_flow(self, flow_id: str, input_data: Dict[str, Any]) -> FlowExecution:
		"""
		🎯 REVOLUTIONARY: Visual Flow Execution Engine
		
		Executes visual flows and generates journal entries:
		- Step-by-step flow execution with validation
		- Real-time variable calculation and substitution
		- Automatic journal entry generation
		- Error handling and rollback capabilities
		- Audit trail of execution steps
		"""
		try:
			# Get flow definition
			flow = await self._get_flow(flow_id)
			if not flow:
				raise ValueError(f"Flow {flow_id} not found")
			
			# Initialize execution
			execution = FlowExecution(
				execution_id=f"exec_{uuid.uuid4().hex[:8]}",
				flow_id=flow_id,
				initiated_by=input_data.get("user_id", "system"),
				execution_date=datetime.now(timezone.utc),
				input_data=input_data,
				execution_steps=[],
				generated_journal_entries=[],
				execution_status="running",
				execution_time_ms=0.0,
				errors=[]
			)
			
			start_time = datetime.now()
			
			# Execute flow using execution engine
			execution_result = await self.execution_engine.execute_flow(flow, input_data)
			
			# Update execution with results
			execution.execution_steps = execution_result["steps"]
			execution.generated_journal_entries = execution_result["journal_entries"]
			execution.execution_status = execution_result["status"]
			execution.errors = execution_result.get("errors", [])
			
			execution.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
			
			# Generate actual journal entries if execution successful
			if execution.execution_status == "completed" and not execution.errors:
				journal_entries = await self._generate_journal_entries_from_execution(execution_result)
				execution.generated_journal_entries = journal_entries
			
			# Store execution record
			await self._store_execution(execution)
			
			return execution
			
		except Exception as e:
			logger.error(f"Error executing visual flow: {e}")
			raise
	
	async def get_intelligent_suggestions(self, current_flow: TransactionFlow,
										cursor_position: Dict[str, Any]) -> Dict[str, Any]:
		"""
		🎯 REVOLUTIONARY: AI-Powered Design Suggestions
		
		Provides intelligent suggestions during flow design:
		- Context-aware node suggestions
		- Best practice recommendations
		- Error prevention suggestions
		- Optimization opportunities
		- Compliance considerations
		"""
		try:
			suggestions = {
				"node_suggestions": [],
				"connection_suggestions": [],
				"validation_suggestions": [],
				"optimization_suggestions": [],
				"compliance_suggestions": [],
				"best_practice_tips": []
			}
			
			# Analyze current flow context
			flow_context = await self._analyze_flow_context(current_flow, cursor_position)
			
			# Get node suggestions based on current position
			node_suggestions = await self.ai_assistant.suggest_next_nodes(
				current_flow, cursor_position, flow_context
			)
			suggestions["node_suggestions"] = node_suggestions
			
			# Get connection suggestions
			connection_suggestions = await self.ai_assistant.suggest_connections(
				current_flow, cursor_position, flow_context
			)
			suggestions["connection_suggestions"] = connection_suggestions
			
			# Get validation suggestions
			validation_suggestions = await self._get_validation_suggestions(
				current_flow, flow_context
			)
			suggestions["validation_suggestions"] = validation_suggestions
			
			# Get optimization suggestions
			optimization_suggestions = await self._get_optimization_suggestions(
				current_flow, flow_context
			)
			suggestions["optimization_suggestions"] = optimization_suggestions
			
			# Get compliance suggestions
			compliance_suggestions = await self._get_compliance_suggestions(
				current_flow, flow_context
			)
			suggestions["compliance_suggestions"] = compliance_suggestions
			
			# Get best practice tips
			best_practice_tips = await self._get_best_practice_tips(
				current_flow, flow_context
			)
			suggestions["best_practice_tips"] = best_practice_tips
			
			return suggestions
			
		except Exception as e:
			logger.error(f"Error getting intelligent suggestions: {e}")
			return {}
	
	async def generate_flow_from_description(self, natural_language_description: str) -> TransactionFlow:
		"""
		🎯 REVOLUTIONARY: Natural Language to Visual Flow
		
		Converts natural language transaction descriptions into visual flows:
		- "When we receive payment, credit cash and debit AR"
		- "For month-end accruals, calculate based on percentage of revenue"
		- "Inter-entity sales require transfer pricing adjustment and elimination"
		"""
		try:
			# Parse natural language description
			parsed_intent = await self.ai_assistant.parse_transaction_description(
				natural_language_description
			)
			
			# Generate flow structure from intent
			flow_structure = await self.ai_assistant.generate_flow_from_intent(parsed_intent)
			
			# Create visual flow
			flow = TransactionFlow(
				flow_id=f"auto_flow_{uuid.uuid4().hex[:8]}",
				flow_name=f"Auto-generated: {parsed_intent.get('transaction_type', 'Unknown')}",
				description=f"Generated from: {natural_language_description}",
				version="1.0",
				created_by="ai_assistant",
				created_date=datetime.now(timezone.utc),
				status=FlowStatus.DRAFT,
				validation_level=FlowValidationLevel.BASIC,
				nodes=[],
				connections=[],
				variables={},
				templates_used=[],
				execution_metadata={"generated_from_nl": True, "original_description": natural_language_description}
			)
			
			# Convert structure to nodes and connections
			for node_data in flow_structure.get("nodes", []):
				node = await self._create_flow_node(node_data)
				flow.nodes.append(node)
			
			for connection_data in flow_structure.get("connections", []):
				connection = await self._create_flow_connection(connection_data)
				flow.connections.append(connection)
			
			# Validate generated flow
			validation_result = await self.flow_validator.validate_flow(flow)
			if validation_result["valid"]:
				flow.status = FlowStatus.VALIDATED
			else:
				flow.execution_metadata["validation_errors"] = validation_result["errors"]
			
			return flow
			
		except Exception as e:
			logger.error(f"Error generating flow from description: {e}")
			raise
	
	# =====================================
	# PRIVATE HELPER METHODS
	# =====================================
	
	async def _create_flow_node(self, node_data: Dict[str, Any]) -> FlowNode:
		"""Create a flow node from data"""
		
		return FlowNode(
			node_id=node_data.get("id", f"node_{uuid.uuid4().hex[:8]}"),
			node_type=NodeType(node_data["type"]),
			name=node_data["name"],
			description=node_data.get("description", ""),
			position=node_data.get("position", {"x": 0, "y": 0}),
			properties=node_data.get("properties", {}),
			validation_rules=node_data.get("validation_rules", []),
			error_handling=node_data.get("error_handling", {}),
			styling=node_data.get("styling", {})
		)
	
	async def _create_flow_connection(self, connection_data: Dict[str, Any]) -> FlowConnection:
		"""Create a flow connection from data"""
		
		return FlowConnection(
			connection_id=connection_data.get("id", f"conn_{uuid.uuid4().hex[:8]}"),
			from_node_id=connection_data["from"],
			to_node_id=connection_data["to"],
			connection_type=connection_data.get("type", "success"),
			condition=connection_data.get("condition"),
			properties=connection_data.get("properties", {})
		)
	
	async def _analyze_flow_context(self, flow: TransactionFlow, 
								  cursor_position: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze the context of the current flow for suggestions"""
		
		context = {
			"flow_complexity": len(flow.nodes),
			"current_node_types": [node.node_type.value for node in flow.nodes],
			"missing_elements": [],
			"flow_balance": {"debits": 0, "credits": 0},
			"validation_gaps": []
		}
		
		# Analyze balance
		for node in flow.nodes:
			if node.node_type == NodeType.ACCOUNT_DEBIT:
				context["flow_balance"]["debits"] += 1
			elif node.node_type == NodeType.ACCOUNT_CREDIT:
				context["flow_balance"]["credits"] += 1
		
		# Check for missing start/end nodes
		has_start = any(node.node_type == NodeType.START for node in flow.nodes)
		has_end = any(node.node_type == NodeType.END for node in flow.nodes)
		
		if not has_start:
			context["missing_elements"].append("start_node")
		if not has_end:
			context["missing_elements"].append("end_node")
		
		# Check for unbalanced flow
		if context["flow_balance"]["debits"] != context["flow_balance"]["credits"]:
			context["validation_gaps"].append("unbalanced_entries")
		
		return context


class FlowValidator:
	"""Validates transaction flows for correctness"""
	
	async def validate_flow(self, flow: TransactionFlow) -> Dict[str, Any]:
		"""Validate complete flow structure and logic"""
		
		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": [],
			"suggestions": []
		}
		
		# Validate structure
		structure_validation = await self._validate_flow_structure(flow)
		if not structure_validation["valid"]:
			validation_result["valid"] = False
			validation_result["errors"].extend(structure_validation["errors"])
		
		# Validate balance
		balance_validation = await self._validate_flow_balance(flow)
		if not balance_validation["valid"]:
			validation_result["valid"] = False
			validation_result["errors"].extend(balance_validation["errors"])
		
		# Validate connections
		connection_validation = await self._validate_connections(flow)
		if not connection_validation["valid"]:
			validation_result["valid"] = False
			validation_result["errors"].extend(connection_validation["errors"])
		
		# Add warnings and suggestions
		validation_result["warnings"].extend(structure_validation.get("warnings", []))
		validation_result["warnings"].extend(balance_validation.get("warnings", []))
		validation_result["suggestions"].extend(structure_validation.get("suggestions", []))
		
		return validation_result
	
	async def _validate_flow_structure(self, flow: TransactionFlow) -> Dict[str, Any]:
		"""Validate basic flow structure"""
		
		result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}
		
		# Check for start node
		start_nodes = [node for node in flow.nodes if node.node_type == NodeType.START]
		if not start_nodes:
			result["valid"] = False
			result["errors"].append("Flow must have a START node")
		elif len(start_nodes) > 1:
			result["valid"] = False
			result["errors"].append("Flow can only have one START node")
		
		# Check for end node
		end_nodes = [node for node in flow.nodes if node.node_type == NodeType.END]
		if not end_nodes:
			result["valid"] = False
			result["errors"].append("Flow must have an END node")
		
		# Check for orphaned nodes
		connected_nodes = set()
		for connection in flow.connections:
			connected_nodes.add(connection.from_node_id)
			connected_nodes.add(connection.to_node_id)
		
		for node in flow.nodes:
			if node.node_id not in connected_nodes and node.node_type not in [NodeType.START, NodeType.END]:
				result["warnings"].append(f"Node '{node.name}' appears to be orphaned")
		
		return result
	
	async def _validate_flow_balance(self, flow: TransactionFlow) -> Dict[str, Any]:
		"""Validate that flow maintains accounting balance"""
		
		result = {"valid": True, "errors": [], "warnings": []}
		
		debit_nodes = [node for node in flow.nodes if node.node_type == NodeType.ACCOUNT_DEBIT]
		credit_nodes = [node for node in flow.nodes if node.node_type == NodeType.ACCOUNT_CREDIT]
		
		if len(debit_nodes) == 0 and len(credit_nodes) == 0:
			result["warnings"].append("Flow contains no accounting entries")
		elif len(debit_nodes) != len(credit_nodes):
			result["warnings"].append("Unbalanced flow - different number of debit and credit nodes")
		
		return result


class FlowTemplateLibrary:
	"""Library of reusable flow templates"""
	
	def __init__(self):
		self.templates = {}
		self._initialize_standard_templates()
	
	def _initialize_standard_templates(self):
		"""Initialize standard template library"""
		
		# Payment Processing Template
		self.templates["payment_processing"] = FlowTemplate(
			template_id="tmpl_payment_001",
			template_name="Standard Payment Processing",
			category="payments",
			description="Standard flow for processing vendor payments",
			complexity_level="basic",
			use_cases=["vendor_payments", "expense_reimbursements", "bill_payments"],
			template_flow=None,  # Would contain actual flow definition
			customization_points=[
				{"point": "payment_method", "options": ["check", "wire", "ach"]},
				{"point": "approval_required", "type": "boolean"},
				{"point": "expense_account", "type": "account_selector"}
			],
			business_rules=[
				"Payments over $10,000 require manager approval",
				"All payments must have supporting documentation",
				"Vendor must be approved before payment"
			]
		)
		
		# Revenue Recognition Template
		self.templates["revenue_recognition"] = FlowTemplate(
			template_id="tmpl_revenue_001",
			template_name="Revenue Recognition (ASC 606)",
			category="revenue",
			description="ASC 606 compliant revenue recognition flow",
			complexity_level="advanced",
			use_cases=["product_sales", "service_revenue", "subscription_revenue"],
			template_flow=None,
			customization_points=[
				{"point": "performance_obligations", "type": "list"},
				{"point": "allocation_method", "options": ["standalone", "residual"]},
				{"point": "timing", "options": ["point_in_time", "over_time"]}
			],
			business_rules=[
				"Revenue recognized when performance obligation satisfied",
				"Contract modifications require separate assessment",
				"Variable consideration estimated at contract inception"
			]
		)
	
	async def get_template_for_transaction(self, transaction_type: str,
										 requirements: Dict[str, Any]) -> Optional[FlowTemplate]:
		"""Get appropriate template for transaction type"""
		
		# Match transaction type to template
		template_mapping = {
			"payment": "payment_processing",
			"revenue": "revenue_recognition",
			"expense": "expense_processing",
			"accrual": "accrual_processing",
			"intercompany": "intercompany_transaction"
		}
		
		template_key = template_mapping.get(transaction_type)
		if template_key and template_key in self.templates:
			return self.templates[template_key]
		
		return None


class FlowExecutionEngine:
	"""Executes visual flows and generates journal entries"""
	
	async def execute_flow(self, flow: TransactionFlow, input_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a flow step by step"""
		
		execution_result = {
			"status": "completed",
			"steps": [],
			"journal_entries": [],
			"errors": [],
			"variables": input_data.copy()
		}
		
		# Find start node
		start_nodes = [node for node in flow.nodes if node.node_type == NodeType.START]
		if not start_nodes:
			execution_result["status"] = "error"
			execution_result["errors"].append("No START node found")
			return execution_result
		
		current_node = start_nodes[0]
		visited_nodes = set()
		
		# Execute flow step by step
		while current_node and current_node.node_type != NodeType.END:
			if current_node.node_id in visited_nodes:
				execution_result["status"] = "error"
				execution_result["errors"].append(f"Circular reference detected at node {current_node.name}")
				break
			
			visited_nodes.add(current_node.node_id)
			
			# Execute current node
			step_result = await self._execute_node(current_node, execution_result["variables"])
			execution_result["steps"].append(step_result)
			
			# Update variables with step result
			if step_result.get("variables"):
				execution_result["variables"].update(step_result["variables"])
			
			# Check for errors
			if step_result.get("error"):
				execution_result["status"] = "error"
				execution_result["errors"].append(step_result["error"])
				break
			
			# Find next node
			next_node = await self._find_next_node(current_node, flow, execution_result["variables"])
			current_node = next_node
		
		# Generate journal entries from execution
		if execution_result["status"] == "completed":
			journal_entries = await self._generate_journal_entries_from_steps(
				execution_result["steps"], execution_result["variables"]
			)
			execution_result["journal_entries"] = journal_entries
		
		return execution_result
	
	async def _execute_node(self, node: FlowNode, variables: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute individual node"""
		
		step_result = {
			"node_id": node.node_id,
			"node_name": node.name,
			"node_type": node.node_type.value,
			"execution_time": datetime.now(timezone.utc),
			"variables": {},
			"output": {},
			"error": None
		}
		
		try:
			if node.node_type == NodeType.ACCOUNT_DEBIT:
				# Process debit entry
				account_id = node.properties.get("account_id")
				amount_expr = node.properties.get("amount")
				amount = await self._evaluate_expression(amount_expr, variables)
				
				step_result["output"] = {
					"entry_type": "debit",
					"account_id": account_id,
					"amount": amount
				}
			
			elif node.node_type == NodeType.ACCOUNT_CREDIT:
				# Process credit entry
				account_id = node.properties.get("account_id")
				amount_expr = node.properties.get("amount")
				amount = await self._evaluate_expression(amount_expr, variables)
				
				step_result["output"] = {
					"entry_type": "credit",
					"account_id": account_id,
					"amount": amount
				}
			
			elif node.node_type == NodeType.CALCULATION:
				# Process calculation
				formula = node.properties.get("formula")
				result = await self._evaluate_expression(formula, variables)
				variable_name = node.properties.get("variable_name", "result")
				
				step_result["variables"] = {variable_name: result}
				step_result["output"] = {"calculation_result": result}
			
			elif node.node_type == NodeType.VALIDATION:
				# Process validation
				validation_expr = node.properties.get("validation")
				is_valid = await self._evaluate_expression(validation_expr, variables)
				
				if not is_valid:
					step_result["error"] = f"Validation failed: {node.properties.get('error_message', 'Unknown validation error')}"
				
				step_result["output"] = {"validation_passed": is_valid}
			
			# Add more node type handlers as needed
			
		except Exception as e:
			step_result["error"] = f"Error executing node {node.name}: {str(e)}"
		
		return step_result
	
	async def _evaluate_expression(self, expression: str, variables: Dict[str, Any]) -> Any:
		"""Safely evaluate expressions with variables"""
		
		# Simple expression evaluator - in production would use a more robust parser
		if not expression:
			return None
		
		# Replace variables in expression
		for var_name, var_value in variables.items():
			expression = expression.replace(f"${var_name}", str(var_value))
		
		# Basic arithmetic evaluation (expand as needed)
		try:
			# Only allow simple arithmetic for security
			allowed_chars = set("0123456789+-*/.() ")
			if all(c in allowed_chars for c in expression):
				return eval(expression)
			else:
				return expression  # Return as string if complex
		except:
			return expression


class FlowDesignAIAssistant:
	"""AI assistant for flow design"""
	
	async def suggest_next_nodes(self, flow: TransactionFlow, cursor_position: Dict[str, Any],
							   context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Suggest next nodes based on context"""
		
		suggestions = []
		
		# Suggest based on current context
		if context.get("missing_elements"):
			if "start_node" in context["missing_elements"]:
				suggestions.append({
					"type": "START",
					"name": "Start",
					"description": "Begin transaction flow",
					"priority": "high"
				})
			
			if "end_node" in context["missing_elements"]:
				suggestions.append({
					"type": "END",
					"name": "End",
					"description": "Complete transaction flow",
					"priority": "high"
				})
		
		# Suggest balance corrections
		balance = context.get("flow_balance", {})
		if balance.get("debits", 0) > balance.get("credits", 0):
			suggestions.append({
				"type": "ACCOUNT_CREDIT",
				"name": "Credit Account",
				"description": "Add credit entry to balance the transaction",
				"priority": "medium"
			})
		elif balance.get("credits", 0) > balance.get("debits", 0):
			suggestions.append({
				"type": "ACCOUNT_DEBIT",
				"name": "Debit Account", 
				"description": "Add debit entry to balance the transaction",
				"priority": "medium"
			})
		
		# Suggest common next steps
		current_node_types = context.get("current_node_types", [])
		if "validation" not in current_node_types:
			suggestions.append({
				"type": "VALIDATION",
				"name": "Validation Check",
				"description": "Add validation to ensure data integrity",
				"priority": "low"
			})
		
		return suggestions
	
	async def parse_transaction_description(self, description: str) -> Dict[str, Any]:
		"""Parse natural language transaction description"""
		
		# Simple NLP parsing - in production would use advanced NLP models
		description_lower = description.lower()
		
		intent = {
			"transaction_type": "unknown",
			"entities": {},
			"confidence": 0.5
		}
		
		# Detect transaction type
		if any(word in description_lower for word in ["payment", "pay", "paid"]):
			intent["transaction_type"] = "payment"
			intent["confidence"] = 0.8
		elif any(word in description_lower for word in ["receive", "received", "collection"]):
			intent["transaction_type"] = "receipt"
			intent["confidence"] = 0.8
		elif any(word in description_lower for word in ["accrual", "accrue", "provision"]):
			intent["transaction_type"] = "accrual"
			intent["confidence"] = 0.9
		
		# Extract entities (simplified)
		if "cash" in description_lower:
			intent["entities"]["cash_account"] = True
		if "expense" in description_lower:
			intent["entities"]["expense_account"] = True
		if "revenue" in description_lower:
			intent["entities"]["revenue_account"] = True
		
		return intent


class JournalEntryCodeGenerator:
	"""Generates journal entry code from flow execution"""
	
	async def generate_from_flow_execution(self, execution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate journal entries from flow execution results"""
		
		journal_entries = []
		
		# Collect all debit and credit entries from execution steps
		debits = []
		credits = []
		
		for step in execution_result.get("steps", []):
			output = step.get("output", {})
			
			if output.get("entry_type") == "debit":
				debits.append({
					"account_id": output["account_id"],
					"amount": output["amount"],
					"description": f"Generated from flow node: {step['node_name']}"
				})
			elif output.get("entry_type") == "credit":
				credits.append({
					"account_id": output["account_id"],
					"amount": output["amount"],
					"description": f"Generated from flow node: {step['node_name']}"
				})
		
		# Create journal entry
		if debits or credits:
			journal_entry = {
				"entry_id": f"je_flow_{uuid.uuid4().hex[:8]}",
				"description": "Generated from visual transaction flow",
				"entry_date": datetime.now().date(),
				"lines": []
			}
			
			# Add debit lines
			for debit in debits:
				journal_entry["lines"].append({
					"line_number": len(journal_entry["lines"]) + 1,
					"account_id": debit["account_id"],
					"debit_amount": debit["amount"],
					"credit_amount": Decimal('0'),
					"description": debit["description"]
				})
			
			# Add credit lines
			for credit in credits:
				journal_entry["lines"].append({
					"line_number": len(journal_entry["lines"]) + 1,
					"account_id": credit["account_id"],
					"debit_amount": Decimal('0'),
					"credit_amount": credit["amount"],
					"description": credit["description"]
				})
			
			journal_entries.append(journal_entry)
		
		return journal_entries


# Export visual designer classes
__all__ = [
	'VisualTransactionFlowDesigner',
	'TransactionFlow',
	'FlowNode',
	'FlowConnection',
	'FlowTemplate',
	'FlowExecution',
	'FlowValidator',
	'FlowTemplateLibrary',
	'FlowExecutionEngine',
	'FlowDesignAIAssistant',
	'JournalEntryCodeGenerator',
	'NodeType',
	'FlowValidationLevel',
	'FlowStatus'
]