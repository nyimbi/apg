#!/usr/bin/env python3
"""
APG Application Template Manager
================================

Manages complete application templates for generating world-class, domain-specific
Flask-AppBuilder applications from APG source code.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .application_templates import get_template_directory, list_available_templates, get_template_by_id


@dataclass
class TemplateContext:
	"""Context information for template processing"""
	project_name: str
	project_description: str
	author: str = "APG Developer"
	version: str = "1.0.0"
	database_url: str = "sqlite:///app.db"
	secret_key: str = "dev-secret-key-change-in-production"
	
	# APG-specific context
	agents: List[Dict[str, Any]] = None
	digital_twins: List[Dict[str, Any]] = None
	workflows: List[Dict[str, Any]] = None
	databases: List[Dict[str, Any]] = None
	
	# Template-specific variables
	custom_variables: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.agents is None:
			self.agents = []
		if self.digital_twins is None:
			self.digital_twins = []
		if self.workflows is None:
			self.workflows = []
		if self.databases is None:
			self.databases = []
		if self.custom_variables is None:
			self.custom_variables = {}
	
	def to_template_variables(self) -> Dict[str, Any]:
		"""Convert to template variable dictionary"""
		variables = {
			'project_name': self.project_name,
			'project_description': self.project_description,
			'author': self.author,
			'version': self.version,
			'database_url': self.database_url,
			'secret_key': self.secret_key,
			'current_date': datetime.now().strftime('%Y-%m-%d'),
			'current_year': datetime.now().year
		}
		
		# Add APG-specific variables
		variables.update({
			'agents': self.agents,
			'digital_twins': self.digital_twins,
			'workflows': self.workflows,
			'databases': self.databases,
			'agent_count': len(self.agents),
			'has_digital_twins': len(self.digital_twins) > 0,
			'has_workflows': len(self.workflows) > 0
		})
		
		# Add custom variables
		variables.update(self.custom_variables)
		
		return variables


class ApplicationTemplateManager:
	"""Manages application templates for APG code generation"""
	
	def __init__(self):
		self.templates_dir = get_template_directory()
		self.variable_pattern = re.compile(r'\{\{(\w+)\}\}')
		self.block_pattern = re.compile(r'\{\%\s*(if|for|endif|endfor).*?\%\}', re.DOTALL)
	
	def list_templates(self) -> List[Dict[str, Any]]:
		"""List all available application templates"""
		return list_available_templates()
	
	def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
		"""Get template by ID"""
		return get_template_by_id(template_id)
	
	def detect_template_from_apg_ast(self, apg_ast) -> str:
		"""Detect the best template based on APG AST analysis"""
		
		# Analyze AST to determine application characteristics
		characteristics = self._analyze_apg_ast(apg_ast)
		
		# Template detection logic
		if characteristics['has_ai_features']:
			if characteristics['has_knowledge_base']:
				return 'intelligence/knowledge_base'
			elif characteristics['has_chat_interface']:
				return 'intelligence/chat_assistant'
			elif characteristics['has_nlp_processing']:
				return 'intelligence/nlp_processor'
			else:
				return 'intelligence/ai_platform'
		
		elif characteristics['has_marketplace']:
			if characteristics['has_b2b_features']:
				return 'marketplace/b2b_platform'
			elif characteristics['has_services']:
				return 'marketplace/service_marketplace'
			elif characteristics['has_gig_features']:
				return 'marketplace/gig_economy'
			else:
				return 'marketplace/ecommerce'
		
		elif characteristics['has_iot_devices']:
			if characteristics['has_manufacturing']:
				return 'iot/smart_factory'
			elif characteristics['has_fleet_management']:
				return 'iot/fleet_management'
			elif characteristics['has_environmental_sensors']:
				return 'iot/environmental_sensor'
			else:
				return 'iot/device_monitor'
		
		elif characteristics['has_financial_features']:
			if characteristics['has_trading']:
				return 'fintech/trading_platform'
			elif characteristics['has_payments']:
				return 'fintech/payment_processor'
			elif characteristics['has_lending']:
				return 'fintech/loan_origination'
			else:
				return 'fintech/compliance_monitor'
		
		elif characteristics['has_healthcare']:
			if characteristics['has_telemedicine']:
				return 'healthcare/telemedicine'
			elif characteristics['has_clinical_trials']:
				return 'healthcare/clinical_trials'
			elif characteristics['has_health_analytics']:
				return 'healthcare/health_analytics'
			else:
				return 'healthcare/patient_management'
		
		elif characteristics['has_logistics']:
			if characteristics['has_warehouse']:
				return 'logistics/warehouse_management'
			elif characteristics['has_shipping']:
				return 'logistics/shipping_tracker'
			elif characteristics['has_inventory']:
				return 'logistics/inventory_optimizer'
			else:
				return 'logistics/supply_chain'
		
		elif characteristics['has_enterprise']:
			if characteristics['has_crm']:
				return 'enterprise/crm_platform'
			elif characteristics['has_hr']:
				return 'enterprise/hr_management'
			elif characteristics['has_bi']:
				return 'enterprise/business_intelligence'
			else:
				return 'enterprise/erp_system'
		
		elif characteristics['has_analytics']:
			return 'basic/dashboard'
		
		elif characteristics['has_crud']:
			return 'basic/crud_app'
		
		else:
			return 'basic/simple_agent'
	
	def _analyze_apg_ast(self, apg_ast) -> Dict[str, bool]:
		"""Analyze APG AST to determine application characteristics"""
		characteristics = {
			'has_ai_features': False,
			'has_knowledge_base': False,
			'has_chat_interface': False,
			'has_nlp_processing': False,
			'has_marketplace': False,
			'has_b2b_features': False,
			'has_services': False,
			'has_gig_features': False,
			'has_iot_devices': False,
			'has_manufacturing': False,
			'has_fleet_management': False,
			'has_environmental_sensors': False,
			'has_financial_features': False,
			'has_trading': False,
			'has_payments': False,
			'has_lending': False,
			'has_healthcare': False,
			'has_telemedicine': False,
			'has_clinical_trials': False,
			'has_health_analytics': False,
			'has_logistics': False,
			'has_warehouse': False,
			'has_shipping': False,
			'has_inventory': False,
			'has_enterprise': False,
			'has_crm': False,
			'has_hr': False,
			'has_bi': False,
			'has_analytics': False,
			'has_crud': False
		}
		
		# Analyze agents, digital twins, and databases for keywords
		keywords_found = set()
		
		# Extract keywords from AST (simplified - would be more sophisticated in real implementation)
		if hasattr(apg_ast, 'entities'):
			for entity in apg_ast.entities:
				if hasattr(entity, 'name'):
					keywords_found.add(entity.name.lower())
				if hasattr(entity, 'properties'):
					for prop in entity.properties:
						if hasattr(prop, 'name'):
							keywords_found.add(prop.name.lower())
		
		# Keyword-based detection
		ai_keywords = {'ai', 'ml', 'model', 'inference', 'nlp', 'chat', 'knowledge', 'semantic'}
		marketplace_keywords = {'product', 'order', 'payment', 'cart', 'customer', 'seller', 'buyer'}
		iot_keywords = {'device', 'sensor', 'monitor', 'iot', 'twin', 'factory', 'fleet'}
		fintech_keywords = {'trading', 'finance', 'payment', 'loan', 'credit', 'compliance'}
		healthcare_keywords = {'patient', 'medical', 'health', 'clinical', 'telemedicine'}
		logistics_keywords = {'logistics', 'warehouse', 'shipping', 'inventory', 'supply'}
		enterprise_keywords = {'erp', 'crm', 'hr', 'employee', 'business', 'analytics'}
		
		# Check characteristics
		characteristics['has_ai_features'] = bool(keywords_found & ai_keywords)
		characteristics['has_knowledge_base'] = 'knowledge' in keywords_found
		characteristics['has_chat_interface'] = 'chat' in keywords_found
		characteristics['has_nlp_processing'] = 'nlp' in keywords_found
		
		characteristics['has_marketplace'] = bool(keywords_found & marketplace_keywords)
		characteristics['has_b2b_features'] = 'b2b' in keywords_found or 'business' in keywords_found
		characteristics['has_services'] = 'service' in keywords_found
		characteristics['has_gig_features'] = 'gig' in keywords_found or 'freelance' in keywords_found
		
		characteristics['has_iot_devices'] = bool(keywords_found & iot_keywords)
		characteristics['has_manufacturing'] = 'factory' in keywords_found or 'manufacturing' in keywords_found
		characteristics['has_fleet_management'] = 'fleet' in keywords_found or 'vehicle' in keywords_found
		characteristics['has_environmental_sensors'] = 'environmental' in keywords_found
		
		characteristics['has_financial_features'] = bool(keywords_found & fintech_keywords)
		characteristics['has_trading'] = 'trading' in keywords_found or 'trade' in keywords_found
		characteristics['has_payments'] = 'payment' in keywords_found or 'pay' in keywords_found
		characteristics['has_lending'] = 'loan' in keywords_found or 'credit' in keywords_found
		
		characteristics['has_healthcare'] = bool(keywords_found & healthcare_keywords)
		characteristics['has_telemedicine'] = 'telemedicine' in keywords_found
		characteristics['has_clinical_trials'] = 'clinical' in keywords_found or 'trial' in keywords_found
		characteristics['has_health_analytics'] = 'health' in keywords_found and 'analytics' in keywords_found
		
		characteristics['has_logistics'] = bool(keywords_found & logistics_keywords)
		characteristics['has_warehouse'] = 'warehouse' in keywords_found
		characteristics['has_shipping'] = 'shipping' in keywords_found or 'delivery' in keywords_found
		characteristics['has_inventory'] = 'inventory' in keywords_found
		
		characteristics['has_enterprise'] = bool(keywords_found & enterprise_keywords)
		characteristics['has_crm'] = 'crm' in keywords_found or 'customer' in keywords_found
		characteristics['has_hr'] = 'hr' in keywords_found or 'employee' in keywords_found
		characteristics['has_bi'] = 'bi' in keywords_found or 'intelligence' in keywords_found
		
		characteristics['has_analytics'] = 'analytics' in keywords_found or 'dashboard' in keywords_found
		characteristics['has_crud'] = any(word in keywords_found for word in ['create', 'read', 'update', 'delete', 'crud'])
		
		return characteristics
	
	def generate_application(self, template_id: str, context: TemplateContext) -> Dict[str, str]:
		"""Generate complete application from template"""
		
		template_info = self.get_template(template_id)
		if not template_info:
			raise ValueError(f"Template not found: {template_id}")
		
		template_path = Path(template_info['template_path'])
		template_variables = context.to_template_variables()
		
		generated_files = {}
		
		# Process all template files
		for file_path in template_path.rglob('*.template'):
			relative_path = file_path.relative_to(template_path)
			output_filename = str(relative_path).replace('.template', '')
			
			# Skip certain directories/files
			if any(part.startswith('.') for part in relative_path.parts):
				continue
			
			# Read and process template file
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					template_content = f.read()
				
				processed_content = self._process_template_content(template_content, template_variables)
				generated_files[output_filename] = processed_content
				
			except Exception as e:
				print(f"Warning: Could not process template file {file_path}: {e}")
				continue
		
		return generated_files
	
	def _process_template_content(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process template content with variable substitution and logic blocks"""
		
		# Process conditional blocks first
		content = self._process_conditional_blocks(content, variables)
		
		# Process loop blocks
		content = self._process_loop_blocks(content, variables)
		
		# Replace simple variables
		for var_name, var_value in variables.items():
			placeholder = f'{{{{{var_name}}}}}'
			if isinstance(var_value, bool):
				var_value = 'true' if var_value else 'false'
			elif var_value is None:
				var_value = 'null'
			elif isinstance(var_value, (list, dict)):
				var_value = json.dumps(var_value)
			
			content = content.replace(placeholder, str(var_value))
		
		return content
	
	def _process_conditional_blocks(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process {% if condition %} blocks"""
		
		if_pattern = re.compile(r'\{\%\s*if\s+(\w+)\s*\%\}(.*?)\{\%\s*endif\s*\%\}', re.DOTALL)
		
		def replace_if_block(match):
			condition_var = match.group(1)
			block_content = match.group(2)
			
			condition_value = variables.get(condition_var, False)
			if isinstance(condition_value, str):
				condition_value = condition_value.lower() in ['true', '1', 'yes', 'on']
			elif isinstance(condition_value, (list, dict)):
				condition_value = len(condition_value) > 0
			
			return block_content if condition_value else ''
		
		return if_pattern.sub(replace_if_block, content)
	
	def _process_loop_blocks(self, content: str, variables: Dict[str, Any]) -> str:
		"""Process {% for item in list %} blocks"""
		
		for_pattern = re.compile(r'\{\%\s*for\s+(\w+)\s+in\s+(\w+)\s*\%\}(.*?)\{\%\s*endfor\s*\%\}', re.DOTALL)
		
		def replace_for_block(match):
			item_var = match.group(1)
			list_var = match.group(2)
			block_content = match.group(3)
			
			list_value = variables.get(list_var, [])
			if not isinstance(list_value, list):
				return ''
			
			result = []
			for i, item in enumerate(list_value):
				item_content = block_content
				
				# Replace item variable
				if isinstance(item, dict):
					for key, value in item.items():
						item_content = item_content.replace(f'{{{{{item_var}.{key}}}}}', str(value))
				else:
					item_content = item_content.replace(f'{{{{{item_var}}}}}', str(item))
				
				# Replace loop variables
				item_content = item_content.replace(f'{{{{{item_var}_index}}}}', str(i))
				item_content = item_content.replace(f'{{{{{item_var}_first}}}}', str(i == 0).lower())
				item_content = item_content.replace(f'{{{{{item_var}_last}}}}', str(i == len(list_value) - 1).lower())
				
				result.append(item_content)
			
			return ''.join(result)
		
		return for_pattern.sub(replace_for_block, content)
	
	def get_template_recommendations(self, apg_ast) -> List[Tuple[str, float]]:
		"""Get ranked template recommendations based on APG AST"""
		
		characteristics = self._analyze_apg_ast(apg_ast)
		templates = self.list_templates()
		
		recommendations = []
		
		for template in templates:
			score = self._calculate_template_score(template, characteristics)
			if score > 0.1:  # Only include relevant templates
				recommendations.append((template['template_id'], score))
		
		# Sort by score descending
		recommendations.sort(key=lambda x: x[1], reverse=True)
		
		return recommendations
	
	def _calculate_template_score(self, template: Dict[str, Any], characteristics: Dict[str, bool]) -> float:
		"""Calculate how well a template matches the detected characteristics"""
		
		domain = template.get('domain', '').lower()
		features = [f.lower() for f in template.get('features', [])]
		agents = [a.lower() for a in template.get('agents', [])]
		
		score = 0.0
		
		# Domain matching
		domain_weights = {
			'artificial intelligence': characteristics.get('has_ai_features', False),
			'e-commerce': characteristics.get('has_marketplace', False),
			'internet of things': characteristics.get('has_iot_devices', False),
			'financial services': characteristics.get('has_financial_features', False),
			'healthcare': characteristics.get('has_healthcare', False),
			'supply chain': characteristics.get('has_logistics', False),
			'enterprise management': characteristics.get('has_enterprise', False)
		}
		
		if domain in domain_weights:
			score += 0.5 if domain_weights[domain] else 0.0
		
		# Feature matching
		feature_keywords = {
			'ai': characteristics.get('has_ai_features', False),
			'marketplace': characteristics.get('has_marketplace', False),
			'iot': characteristics.get('has_iot_devices', False),
			'payment': characteristics.get('has_payments', False),
			'analytics': characteristics.get('has_analytics', False),
			'crud': characteristics.get('has_crud', False)
		}
		
		for keyword, has_feature in feature_keywords.items():
			if has_feature and any(keyword in feature for feature in features):
				score += 0.2
		
		# Base score for general templates
		if template['template_id'].startswith('basic/'):
			score += 0.1
		
		return min(score, 1.0)  # Cap at 1.0