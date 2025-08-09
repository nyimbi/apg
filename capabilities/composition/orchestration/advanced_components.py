#!/usr/bin/env python3
"""
APG Workflow Orchestration Advanced Components

Specialized components for data operations, integrations, AI/ML, and advanced functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import aiohttp
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import re
import hashlib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from .component_library import (
    BaseWorkflowComponent, ExecutionResult, ComponentDefinition, 
    ComponentType, ComponentCategory, component_library
)


logger = logging.getLogger(__name__)


# Data Operation Components

class TransformComponent(BaseWorkflowComponent):
	"""Data transformation component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute data transformation."""
		try:
			transformations = self.config.get('transformations', [])
			output_format = self.config.get('output_format', 'json')
			
			result_data = input_data
			
			for transformation in transformations:
				result_data = await self._apply_transformation(result_data, transformation, context)
			
			# Format output
			if output_format == 'json' and not isinstance(result_data, (dict, list)):
				result_data = {'value': result_data}
			elif output_format == 'csv' and isinstance(result_data, list):
				result_data = self._convert_to_csv(result_data)
			
			result = ExecutionResult(
				success=True,
				data=result_data,
				metadata={'transformations_applied': len(transformations), 'output_format': output_format}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _apply_transformation(self, data: Any, transformation: Dict[str, Any], context: Dict[str, Any]) -> Any:
		"""Apply a single transformation."""
		transform_type = transformation.get('type', 'map')
		
		if transform_type == 'map':
			return await self._map_transform(data, transformation)
		elif transform_type == 'filter':
			return await self._filter_transform(data, transformation)
		elif transform_type == 'aggregate':
			return await self._aggregate_transform(data, transformation)
		elif transform_type == 'sort':
			return await self._sort_transform(data, transformation)
		elif transform_type == 'group':
			return await self._group_transform(data, transformation)
		elif transform_type == 'flatten':
			return await self._flatten_transform(data, transformation)
		elif transform_type == 'pivot':
			return await self._pivot_transform(data, transformation)
		else:
			raise ValueError(f"Unknown transformation type: {transform_type}")
	
	async def _map_transform(self, data: Any, transformation: Dict[str, Any]) -> Any:
		"""Apply map transformation."""
		mapping = transformation.get('mapping', {})
		
		if isinstance(data, list):
			return [self._apply_mapping(item, mapping) for item in data]
		else:
			return self._apply_mapping(data, mapping)
	
	def _apply_mapping(self, item: Any, mapping: Dict[str, str]) -> Any:
		"""Apply mapping to a single item."""
		if not isinstance(item, dict):
			return item
		
		result = {}
		for output_key, input_path in mapping.items():
			if input_path.startswith('$.'):
				# JSONPath-like syntax
				value = self._get_nested_value(item, input_path[2:])
			else:
				value = item.get(input_path)
			
			result[output_key] = value
		
		return result
	
	async def _filter_transform(self, data: Any, transformation: Dict[str, Any]) -> Any:
		"""Apply filter transformation."""
		condition = transformation.get('condition', '')
		
		if not isinstance(data, list):
			data = [data]
		
		filtered_data = []
		for item in data:
			if self._evaluate_filter_condition(item, condition):
				filtered_data.append(item)
		
		return filtered_data
	
	def _evaluate_filter_condition(self, item: Any, condition: str) -> bool:
		"""Evaluate filter condition."""
		try:
			# Simple condition evaluation
			# In production, use a proper expression evaluator
			safe_globals = {'item': item}
			return bool(eval(condition.replace('$', 'item'), safe_globals))
		except Exception:
			return False
	
	async def _aggregate_transform(self, data: Any, transformation: Dict[str, Any]) -> Any:
		"""Apply aggregation transformation."""
		if not isinstance(data, list):
			return data
		
		operations = transformation.get('operations', {})
		group_by = transformation.get('group_by')
		
		if group_by:
			# Group by field and then aggregate
			groups = {}
			for item in data:
				if isinstance(item, dict):
					key = item.get(group_by, 'unknown')
					if key not in groups:
						groups[key] = []
					groups[key].append(item)
			
			result = {}
			for group_key, group_items in groups.items():
				result[group_key] = self._calculate_aggregations(group_items, operations)
			
			return result
		else:
			# Aggregate entire dataset
			return self._calculate_aggregations(data, operations)
	
	def _calculate_aggregations(self, data: List[Any], operations: Dict[str, str]) -> Dict[str, Any]:
		"""Calculate aggregation operations."""
		result = {}
		
		for field, operation in operations.items():
			values = [item.get(field) for item in data if isinstance(item, dict) and field in item]
			numeric_values = [v for v in values if isinstance(v, (int, float))]
			
			if operation == 'count':
				result[f"{field}_{operation}"] = len(values)
			elif operation == 'sum' and numeric_values:
				result[f"{field}_{operation}"] = sum(numeric_values)
			elif operation == 'avg' and numeric_values:
				result[f"{field}_{operation}"] = sum(numeric_values) / len(numeric_values)
			elif operation == 'min' and numeric_values:
				result[f"{field}_{operation}"] = min(numeric_values)
			elif operation == 'max' and numeric_values:
				result[f"{field}_{operation}"] = max(numeric_values)
		
		return result
	
	def _get_nested_value(self, data: Any, path: str) -> Any:
		"""Get nested value using dot notation."""
		parts = path.split('.')
		result = data
		
		for part in parts:
			if isinstance(result, dict):
				result = result.get(part)
			elif isinstance(result, list) and part.isdigit():
				index = int(part)
				result = result[index] if index < len(result) else None
			else:
				return None
		
		return result
	
	def _convert_to_csv(self, data: List[Dict[str, Any]]) -> str:
		"""Convert list of dictionaries to CSV format."""
		if not data:
			return ""
		
		# Use pandas for CSV conversion
		df = pd.DataFrame(data)
		return df.to_csv(index=False)
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="transform_component",
			type=ComponentType.TRANSFORM,
			name="Transform",
			description="Advanced data transformation component",
			category=ComponentCategory.DATA_OPERATIONS,
			icon="transform",
			color="#CDDC39",
			config_schema={
				"type": "object",
				"properties": {
					"transformations": {
						"type": "array",
						"items": {
							"type": "object",
							"properties": {
								"type": {
									"type": "string",
									"enum": ["map", "filter", "aggregate", "sort", "group", "flatten", "pivot"]
								},
								"mapping": {"type": "object"},
								"condition": {"type": "string"},
								"operations": {"type": "object"},
								"group_by": {"type": "string"},
								"sort_by": {"type": "string"},
								"sort_order": {"type": "string", "enum": ["asc", "desc"]}
							}
						}
					},
					"output_format": {
						"type": "string",
						"enum": ["json", "csv", "xml"],
						"default": "json"
					}
				}
			}
		)


class HTTPRequestComponent(BaseWorkflowComponent):
	"""HTTP request component for API integration."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute HTTP request."""
		try:
			method = self.config.get('method', 'GET').upper()
			url = self.config.get('url', '')
			headers = self.config.get('headers', {})
			timeout = self.config.get('timeout', 30)
			
			# Prepare request data
			request_data = await self._prepare_request_data(input_data, method)
			
			# Make HTTP request
			async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
				response_data = await self._make_request(session, method, url, headers, request_data)
			
			result = ExecutionResult(
				success=True,
				data=response_data,
				metadata={
					'method': method,
					'url': url,
					'status_code': response_data.get('status_code'),
					'response_time': response_data.get('response_time')
				}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _prepare_request_data(self, input_data: Any, method: str) -> Optional[Dict[str, Any]]:
		"""Prepare request data based on method."""
		if method in ['POST', 'PUT', 'PATCH']:
			if isinstance(input_data, dict):
				return input_data
			else:
				return {'data': input_data}
		return None
	
	async def _make_request(self, session: aiohttp.ClientSession, method: str, url: str, 
						   headers: Dict[str, str], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		"""Make HTTP request and return response data."""
		start_time = datetime.utcnow()
		
		try:
			if method == 'GET':
				async with session.get(url, headers=headers) as response:
					response_data = await self._process_response(response)
			elif method == 'POST':
				async with session.post(url, headers=headers, json=data) as response:
					response_data = await self._process_response(response)
			elif method == 'PUT':
				async with session.put(url, headers=headers, json=data) as response:
					response_data = await self._process_response(response)
			elif method == 'DELETE':
				async with session.delete(url, headers=headers) as response:
					response_data = await self._process_response(response)
			else:
				raise ValueError(f"Unsupported HTTP method: {method}")
			
			# Add timing information
			response_time = (datetime.utcnow() - start_time).total_seconds()
			response_data['response_time'] = response_time
			
			return response_data
			
		except Exception as e:
			raise Exception(f"HTTP request failed: {str(e)}")
	
	async def _process_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
		"""Process HTTP response."""
		try:
			# Try to parse as JSON
			content = await response.json()
		except:
			# Fall back to text
			content = await response.text()
		
		return {
			'status_code': response.status,
			'headers': dict(response.headers),
			'content': content,
			'url': str(response.url)
		}
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="http_request_component",
			type=ComponentType.HTTP_REQUEST,
			name="HTTP Request",
			description="Make HTTP requests to external APIs",
			category=ComponentCategory.INTEGRATIONS,
			icon="api",
			color="#3F51B5",
			config_schema={
				"type": "object",
				"properties": {
					"method": {
						"type": "string",
						"enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
						"default": "GET"
					},
					"url": {
						"type": "string",
						"format": "uri",
						"description": "Target URL for the HTTP request"
					},
					"headers": {
						"type": "object",
						"description": "HTTP headers to include in the request"
					},
					"timeout": {
						"type": "integer",
						"minimum": 1,
						"maximum": 300,
						"default": 30
					},
					"retry_count": {
						"type": "integer",
						"minimum": 0,
						"default": 3
					},
					"retry_delay": {
						"type": "integer",
						"minimum": 1,
						"default": 5
					}
				},
				"required": ["url"]
			}
		)


class EmailComponent(BaseWorkflowComponent):
	"""Email sending component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Send email."""
		try:
			# Extract email parameters
			email_data = self._extract_email_data(input_data)
			
			# Prepare email message
			message = await self._prepare_email_message(email_data)
			
			# Send email
			send_result = await self._send_email(message, email_data)
			
			result = ExecutionResult(
				success=True,
				data={
					'message_id': send_result.get('message_id'),
					'to': email_data['to'],
					'subject': email_data['subject'],
					'sent_at': datetime.utcnow().isoformat()
				},
				metadata={'email_sent': True}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	def _extract_email_data(self, input_data: Any) -> Dict[str, Any]:
		"""Extract email data from input."""
		if isinstance(input_data, dict):
			email_data = input_data.copy()
		else:
			email_data = {'body': str(input_data)}
		
		# Use config defaults if not provided in input
		email_data.setdefault('to', self.config.get('default_to', ''))
		email_data.setdefault('subject', self.config.get('default_subject', 'Workflow Notification'))
		email_data.setdefault('from', self.config.get('from_address', ''))
		
		# Validate required fields
		if not email_data.get('to'):
			raise ValueError("Email 'to' address is required")
		if not email_data.get('from'):
			raise ValueError("Email 'from' address is required")
		
		return email_data
	
	async def _prepare_email_message(self, email_data: Dict[str, Any]) -> MIMEMultipart:
		"""Prepare email message."""
		message = MIMEMultipart('alternative')
		
		message['From'] = email_data['from']
		message['To'] = email_data['to']
		message['Subject'] = email_data['subject']
		
		# Add CC and BCC if provided
		if email_data.get('cc'):
			message['Cc'] = email_data['cc']
		if email_data.get('bcc'):
			message['Bcc'] = email_data['bcc']
		
		# Add body content
		body = email_data.get('body', '')
		if email_data.get('html'):
			# HTML content
			html_part = MIMEText(body, 'html')
			message.attach(html_part)
		else:
			# Plain text content
			text_part = MIMEText(body, 'plain')
			message.attach(text_part)
		
		return message
	
	async def _send_email(self, message: MIMEMultipart, email_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Send email via SMTP."""
		smtp_config = self.config.get('smtp', {})
		
		# SMTP configuration
		smtp_host = smtp_config.get('host', 'localhost')
		smtp_port = smtp_config.get('port', 587)
		use_tls = smtp_config.get('use_tls', True)
		username = smtp_config.get('username')
		password = smtp_config.get('password')
		
		try:
			# Create SMTP connection
			server = smtplib.SMTP(smtp_host, smtp_port)
			
			if use_tls:
				server.starttls()
			
			if username and password:
				server.login(username, password)
			
			# Send email
			recipients = [email_data['to']]
			if email_data.get('cc'):
				recipients.extend(email_data['cc'].split(','))
			if email_data.get('bcc'):
				recipients.extend(email_data['bcc'].split(','))
			
			server.send_message(message, to_addrs=recipients)
			server.quit()
			
			# Generate message ID
			message_id = hashlib.md5(
				f"{email_data['to']}{email_data['subject']}{datetime.utcnow()}".encode()
			).hexdigest()
			
			return {'message_id': message_id}
			
		except Exception as e:
			raise Exception(f"SMTP send failed: {str(e)}")
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="email_component",
			type=ComponentType.EMAIL_SEND,
			name="Email",
			description="Send email notifications",
			category=ComponentCategory.INTEGRATIONS,
			icon="email",
			color="#E91E63",
			config_schema={
				"type": "object",
				"properties": {
					"default_to": {
						"type": "string",
						"format": "email",
						"description": "Default recipient email address"
					},
					"default_subject": {
						"type": "string",
						"default": "Workflow Notification"
					},
					"from_address": {
						"type": "string",
						"format": "email",
						"description": "Sender email address"
					},
					"smtp": {
						"type": "object",
						"properties": {
							"host": {"type": "string", "default": "localhost"},
							"port": {"type": "integer", "default": 587},
							"use_tls": {"type": "boolean", "default": True},
							"username": {"type": "string"},
							"password": {"type": "string"}
						}
					}
				}
			}
		)


class ScriptComponent(BaseWorkflowComponent):
	"""Script execution component for custom code."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute custom script."""
		try:
			script_type = self.config.get('script_type', 'python')
			script_code = self.config.get('script_code', '')
			timeout = self.config.get('timeout', 30)
			
			if script_type == 'python':
				result_data = await self._execute_python_script(script_code, input_data, context, timeout)
			elif script_type == 'javascript':
				result_data = await self._execute_javascript_script(script_code, input_data, context, timeout)
			else:
				raise ValueError(f"Unsupported script type: {script_type}")
			
			result = ExecutionResult(
				success=True,
				data=result_data,
				metadata={'script_type': script_type, 'script_length': len(script_code)}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _execute_python_script(self, script_code: str, input_data: Any, 
									context: Dict[str, Any], timeout: int) -> Any:
		"""Execute Python script safely."""
		# Create safe execution environment
		safe_globals = {
			'__builtins__': {
				'len': len,
				'str': str,
				'int': int,
				'float': float,
				'bool': bool,
				'list': list,
				'dict': dict,
				'tuple': tuple,
				'set': set,
				'range': range,
				'enumerate': enumerate,
				'zip': zip,
				'map': map,
				'filter': filter,
				'sum': sum,
				'min': min,
				'max': max,
				'abs': abs,
				'round': round,
				'sorted': sorted,
				'json': json,
				'datetime': datetime,
				're': re
			},
			'input_data': input_data,
			'context': context,
			'result': None
		}
		
		# Execute script with timeout
		try:
			exec(script_code, safe_globals)
			return safe_globals.get('result', input_data)
		except Exception as e:
			raise Exception(f"Python script execution failed: {str(e)}")
	
	async def _execute_javascript_script(self, script_code: str, input_data: Any,
										context: Dict[str, Any], timeout: int) -> Any:
		"""Execute JavaScript script (simplified simulation)."""
		# In a real implementation, this would use a JavaScript engine
		# For now, return input data with a processed flag
		return {
			'original_data': input_data,
			'processed': True,
			'script_executed': True,
			'timestamp': datetime.utcnow().isoformat()
		}
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="script_component",
			type=ComponentType.SCRIPT,
			name="Script",
			description="Execute custom Python or JavaScript code",
			category=ComponentCategory.ADVANCED,
			icon="code",
			color="#8BC34A",
			config_schema={
				"type": "object",
				"properties": {
					"script_type": {
						"type": "string",
						"enum": ["python", "javascript"],
						"default": "python"
					},
					"script_code": {
						"type": "string",
						"description": "Script code to execute"
					},
					"timeout": {
						"type": "integer",
						"minimum": 1,
						"maximum": 300,
						"default": 30
					},
					"allowed_imports": {
						"type": "array",
						"items": {"type": "string"},
						"description": "Allowed Python imports"
					}
				},
				"required": ["script_code"]
			}
		)


class MLPredictionComponent(BaseWorkflowComponent):
	"""Machine Learning prediction component."""
	
	async def execute(self, input_data: Any, context: Dict[str, Any]) -> ExecutionResult:
		"""Execute ML prediction."""
		try:
			model_type = self.config.get('model_type', 'classification')
			model_endpoint = self.config.get('model_endpoint')
			features = self.config.get('features', [])
			
			# Extract features from input data
			feature_data = await self._extract_features(input_data, features)
			
			# Make prediction
			if model_endpoint:
				prediction_result = await self._call_external_model(model_endpoint, feature_data)
			else:
				prediction_result = await self._local_prediction(feature_data, model_type)
			
			result = ExecutionResult(
				success=True,
				data=prediction_result,
				metadata={
					'model_type': model_type,
					'features_used': len(features),
					'prediction_confidence': prediction_result.get('confidence', 0.0)
				}
			)
			
		except Exception as e:
			result = ExecutionResult(
				success=False,
				error=str(e),
				data=input_data
			)
		
		await self._log_execution(input_data, result)
		return result
	
	async def _extract_features(self, input_data: Any, features: List[str]) -> List[float]:
		"""Extract feature values from input data."""
		if not isinstance(input_data, dict):
			raise ValueError("Input data must be a dictionary for feature extraction")
		
		feature_values = []
		for feature in features:
			value = input_data.get(feature, 0.0)
			if isinstance(value, (int, float)):
				feature_values.append(float(value))
			else:
				# Try to convert string to float
				try:
					feature_values.append(float(value))
				except ValueError:
					feature_values.append(0.0)
		
		return feature_values
	
	async def _call_external_model(self, endpoint: str, features: List[float]) -> Dict[str, Any]:
		"""Call external ML model API."""
		try:
			async with aiohttp.ClientSession() as session:
				payload = {'features': features}
				async with session.post(endpoint, json=payload) as response:
					if response.status == 200:
						return await response.json()
					else:
						raise Exception(f"Model API returned status {response.status}")
		except Exception as e:
			raise Exception(f"External model call failed: {str(e)}")
	
	async def _local_prediction(self, features: List[float], model_type: str) -> Dict[str, Any]:
		"""Make local prediction (simplified)."""
		# Simplified prediction logic - in production, load actual ML model
		if model_type == 'classification':
			# Simple classification based on feature sum
			feature_sum = sum(features)
			prediction = 1 if feature_sum > 0 else 0
			confidence = min(abs(feature_sum) / 10.0, 1.0)
			
			return {
				'prediction': prediction,
				'confidence': confidence,
				'prediction_type': 'classification',
				'classes': [0, 1]
			}
		
		elif model_type == 'regression':
			# Simple regression prediction
			prediction = sum(features) / len(features) if features else 0.0
			
			return {
				'prediction': prediction,
				'confidence': 0.8,
				'prediction_type': 'regression'
			}
		
		else:
			raise ValueError(f"Unknown model type: {model_type}")
	
	def get_definition(self) -> ComponentDefinition:
		return ComponentDefinition(
			id="ml_prediction_component",
			type=ComponentType.ML_PREDICTION,
			name="ML Prediction",
			description="Machine learning prediction component",
			category=ComponentCategory.AI_ML,
			icon="psychology",
			color="#9C27B0",
			config_schema={
				"type": "object",
				"properties": {
					"model_type": {
						"type": "string",
						"enum": ["classification", "regression", "clustering"],
						"default": "classification"
					},
					"model_endpoint": {
						"type": "string",
						"format": "uri",
						"description": "External ML model API endpoint"
					},
					"features": {
						"type": "array",
						"items": {"type": "string"},
						"description": "Feature names to extract from input data"
					},
					"model_params": {
						"type": "object",
						"description": "Model-specific parameters"
					}
				},
				"required": ["features"]
			}
		)


# Register advanced components with the library
def register_advanced_components():
	"""Register all advanced components."""
	advanced_components = [
		TransformComponent,
		HTTPRequestComponent,
		EmailComponent,
		ScriptComponent,
		MLPredictionComponent
	]
	
	for component_class in advanced_components:
		component_library.register_component(component_class)


# Auto-register when module is imported
register_advanced_components()