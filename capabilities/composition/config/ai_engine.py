"""
APG Central Configuration - AI Engine

Revolutionary AI-powered configuration management with intelligent optimization,
natural language processing, and predictive analytics using only Ollama models.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import yaml

from .models import RecommendationType


@dataclass
class AIInsight:
	"""AI-generated insight."""
	insight_type: str
	title: str
	description: str
	confidence: float
	impact: str
	recommendations: List[str]
	data: Dict[str, Any]


@dataclass
class OptimizationSuggestion:
	"""Configuration optimization suggestion."""
	field: str
	current_value: Any
	suggested_value: Any
	reasoning: str
	confidence: float
	expected_improvement: str


class CentralConfigurationAI:
	"""AI engine for intelligent configuration management."""
	
	def __init__(self, ollama_base_url: str = "http://localhost:11434"):
		self.ollama_base_url = ollama_base_url
		self.client = httpx.AsyncClient(timeout=300.0)
		
		# Available models
		self.language_model = "llama3.2:3b"
		self.code_model = "codellama:7b"
		self.embedding_model = "nomic-embed-text"
		
		# Local models for offline functionality
		self.embedding_engine = None
		self.sentiment_analyzer = None
		
		# Configuration patterns and best practices
		self.config_patterns = self._load_configuration_patterns()
		self.security_rules = self._load_security_rules()
		self.performance_rules = self._load_performance_rules()
		
		# Analytics and learning
		self.optimization_history = []
		self.anomaly_patterns = []
		
	async def initialize(self):
		"""Initialize AI engine with models."""
		try:
			# Check Ollama availability
			await self._check_ollama_health()
			
			# Pull required models if not available
			await self._ensure_models_available()
			
			# Initialize local models for offline processing
			await self._initialize_local_models()
			
			print("🧠 AI Engine initialized successfully with Ollama models")
			
		except Exception as e:
			print(f"❌ AI Engine initialization failed: {e}")
			raise

	async def _check_ollama_health(self):
		"""Check if Ollama service is available."""
		try:
			response = await self.client.get(f"{self.ollama_base_url}/api/version")
			response.raise_for_status()
			version_info = response.json()
			print(f"✅ Ollama service available - version {version_info.get('version', 'unknown')}")
		except Exception as e:
			raise ConnectionError(f"Ollama service not available at {self.ollama_base_url}: {e}")

	async def _ensure_models_available(self):
		"""Ensure required models are available."""
		models_to_check = [self.language_model, self.code_model, self.embedding_model]
		
		for model in models_to_check:
			try:
				# Check if model exists
				response = await self.client.post(
					f"{self.ollama_base_url}/api/show",
					json={"name": model}
				)
				
				if response.status_code == 404:
					print(f"📥 Pulling model {model}...")
					await self._pull_model(model)
				else:
					print(f"✅ Model {model} is available")
					
			except Exception as e:
				print(f"⚠️ Error checking model {model}: {e}")

	async def _pull_model(self, model_name: str):
		"""Pull a model from Ollama."""
		try:
			response = await self.client.post(
				f"{self.ollama_base_url}/api/pull",
				json={"name": model_name}
			)
			
			if response.status_code == 200:
				print(f"✅ Successfully pulled model {model_name}")
			else:
				print(f"❌ Failed to pull model {model_name}: {response.text}")
				
		except Exception as e:
			print(f"❌ Error pulling model {model_name}: {e}")

	async def _initialize_local_models(self):
		"""Initialize local models for offline processing."""
		try:
			# Initialize embedding model for semantic similarity
			self.embedding_engine = SentenceTransformer('all-MiniLM-L6-v2')
			
			# Initialize sentiment analyzer for configuration context analysis
			self.sentiment_analyzer = pipeline(
				"sentiment-analysis",
				model="distilbert-base-uncased-finetuned-sst-2-english"
			)
			
			print("✅ Local AI models initialized")
		except Exception as e:
			print(f"⚠️ Local model initialization failed: {e}")

	# ==================== Configuration Optimization ====================

	async def optimize_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Optimize configuration using AI analysis."""
		try:
			# Analyze current configuration
			analysis = await self._analyze_configuration_structure(config_data)
			
			# Generate optimization suggestions
			suggestions = await self._generate_optimization_suggestions(config_data, analysis)
			
			# Apply high-confidence optimizations
			optimized_config = self._apply_optimizations(config_data, suggestions)
			
			return optimized_config
			
		except Exception as e:
			print(f"⚠️ Configuration optimization failed: {e}")
			return config_data

	async def _analyze_configuration_structure(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze configuration structure and patterns."""
		analysis = {
			'complexity_score': self._calculate_complexity_score(config_data),
			'security_issues': self._detect_security_issues(config_data),
			'performance_issues': self._detect_performance_issues(config_data),
			'best_practices': self._check_best_practices(config_data),
			'dependencies': self._analyze_dependencies(config_data)
		}
		
		return analysis

	def _calculate_complexity_score(self, config_data: Dict[str, Any]) -> float:
		"""Calculate configuration complexity score."""
		def count_nested_items(obj, depth=0):
			if depth > 10:  # Prevent infinite recursion
				return 0, 0
			
			if isinstance(obj, dict):
				count = len(obj)
				max_depth = depth
				for value in obj.values():
					nested_count, nested_depth = count_nested_items(value, depth + 1)
					count += nested_count
					max_depth = max(max_depth, nested_depth)
				return count, max_depth
			elif isinstance(obj, list):
				count = len(obj)
				max_depth = depth
				for item in obj:
					nested_count, nested_depth = count_nested_items(item, depth + 1)
					count += nested_count
					max_depth = max(max_depth, nested_depth)
				return count, max_depth
			else:
				return 1, depth
		
		total_items, max_depth = count_nested_items(config_data)
		
		# Normalize complexity score (0-1, where 1 is most complex)
		complexity = min(1.0, (total_items * 0.01) + (max_depth * 0.1))
		return complexity

	def _detect_security_issues(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect potential security issues in configuration."""
		issues = []
		
		def scan_for_secrets(obj, path=""):
			if isinstance(obj, dict):
				for key, value in obj.items():
					current_path = f"{path}.{key}" if path else key
					
					# Check for potential secrets
					if any(secret_word in key.lower() for secret_word in ['password', 'secret', 'key', 'token', 'api_key']):
						if isinstance(value, str) and len(value) > 5:
							issues.append({
								'type': 'potential_secret',
								'path': current_path,
								'severity': 'high',
								'description': f'Potential secret detected in field: {key}'
							})
					
					# Check for insecure defaults
					if key.lower() in ['ssl_verify', 'verify_ssl', 'tls_verify'] and value is False:
						issues.append({
							'type': 'insecure_default',
							'path': current_path,
							'severity': 'medium',
							'description': 'SSL/TLS verification disabled'
						})
					
					# Recursively scan nested objects
					scan_for_secrets(value, current_path)
			elif isinstance(obj, list):
				for i, item in enumerate(obj):
					scan_for_secrets(item, f"{path}[{i}]")
		
		scan_for_secrets(config_data)
		return issues

	def _detect_performance_issues(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect potential performance issues."""
		issues = []
		
		def scan_for_performance_issues(obj, path=""):
			if isinstance(obj, dict):
				for key, value in obj.items():
					current_path = f"{path}.{key}" if path else key
					
					# Check for performance-related settings
					if 'timeout' in key.lower() and isinstance(value, (int, float)):
						if value > 300:  # 5 minutes
							issues.append({
								'type': 'high_timeout',
								'path': current_path,
								'severity': 'low',
								'description': f'High timeout value: {value}s'
							})
					
					if 'pool_size' in key.lower() and isinstance(value, int):
						if value > 100:
							issues.append({
								'type': 'large_pool_size',
								'path': current_path,
								'severity': 'medium',
								'description': f'Large connection pool size: {value}'
							})
					
					# Recursively scan
					scan_for_performance_issues(value, current_path)
			elif isinstance(obj, list):
				for i, item in enumerate(obj):
					scan_for_performance_issues(item, f"{path}[{i}]")
		
		scan_for_performance_issues(config_data)
		return issues

	def _check_best_practices(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check configuration against best practices."""
		recommendations = []
		
		# Check for environment-specific configurations
		if not any(env in str(config_data).lower() for env in ['dev', 'test', 'prod', 'staging']):
			recommendations.append({
				'type': 'environment_awareness',
				'severity': 'medium',
				'description': 'Configuration should be environment-aware'
			})
		
		# Check for logging configuration  
		if not any(log_key in str(config_data).lower() for log_key in ['log', 'logging', 'logger']):
			recommendations.append({
				'type': 'logging_missing',
				'severity': 'low',
				'description': 'Consider adding logging configuration'
			})
		
		return recommendations

	def _analyze_dependencies(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze configuration dependencies."""
		dependencies = {
			'external_services': [],
			'database_connections': [],
			'api_endpoints': [],
			'file_paths': []
		}
		
		def scan_dependencies(obj, path=""):
			if isinstance(obj, dict):
				for key, value in obj.items():
					current_path = f"{path}.{key}" if path else key
					
					# Detect database connections
					if 'db' in key.lower() or 'database' in key.lower():
						if isinstance(value, str) and ('://' in value or 'jdbc:' in value):
							dependencies['database_connections'].append({
								'path': current_path,
								'connection_string': value
							})
					
					# Detect API endpoints
					if isinstance(value, str) and value.startswith(('http://', 'https://')):
						dependencies['api_endpoints'].append({
							'path': current_path,
							'url': value
						})
					
					# Detect file paths
					if isinstance(value, str) and ('/' in value or '\\' in value) and len(value) > 3:
						if any(ext in value.lower() for ext in ['.log', '.json', '.yaml', '.xml', '.conf']):
							dependencies['file_paths'].append({
								'path': current_path,
								'file_path': value
							})
					
					scan_dependencies(value, current_path)
			elif isinstance(obj, list):
				for i, item in enumerate(obj):
					scan_dependencies(item, f"{path}[{i}]")
		
		scan_dependencies(config_data)
		return dependencies

	async def _generate_optimization_suggestions(
		self,
		config_data: Dict[str, Any],
		analysis: Dict[str, Any]
	) -> List[OptimizationSuggestion]:
		"""Generate AI-powered optimization suggestions."""
		suggestions = []
		
		# Use Ollama language model for advanced suggestions
		try:
			prompt = self._build_optimization_prompt(config_data, analysis)
			ai_suggestions = await self._query_ollama(self.language_model, prompt)
			
			# Parse AI suggestions
			parsed_suggestions = self._parse_ai_suggestions(ai_suggestions)
			suggestions.extend(parsed_suggestions)
			
		except Exception as e:
			print(f"⚠️ AI optimization suggestions failed: {e}")
		
		# Add rule-based suggestions
		rule_based_suggestions = self._generate_rule_based_suggestions(config_data, analysis)
		suggestions.extend(rule_based_suggestions)
		
		return suggestions

	def _build_optimization_prompt(self, config_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
		"""Build optimization prompt for language model."""
		return f"""
You are an expert configuration management consultant. Analyze this configuration and provide specific optimization suggestions.

Configuration:
{json.dumps(config_data, indent=2)}

Analysis:
- Complexity Score: {analysis['complexity_score']:.2f}
- Security Issues: {len(analysis['security_issues'])} found
- Performance Issues: {len(analysis['performance_issues'])} found
- Dependencies: {len(analysis['dependencies']['external_services'])} external services

Please provide specific optimization suggestions in JSON format:
{{
  "suggestions": [
    {{
      "field": "path.to.field",
      "current_value": "current value",
      "suggested_value": "optimized value", 
      "reasoning": "why this optimization helps",
      "confidence": 0.85,
      "expected_improvement": "performance/security/maintainability"
    }}
  ]
}}

Focus on:
1. Security improvements (secrets management, SSL/TLS settings)
2. Performance optimizations (timeouts, pool sizes, caching)
3. Maintainability improvements (environment variables, documentation)
4. Resource optimization (memory, CPU, network)
"""

	async def _query_ollama(self, model: str, prompt: str) -> str:
		"""Query Ollama model with prompt."""
		try:
			response = await self.client.post(
				f"{self.ollama_base_url}/api/generate",
				json={
					"model": model,
					"prompt": prompt,
					"stream": False,
					"options": {
						"temperature": 0.3,
						"top_p": 0.9,
						"num_predict": 2000
					}
				}
			)
			
			if response.status_code == 200:
				result = response.json()
				return result.get('response', '')
			else:
				print(f"❌ Ollama query failed: {response.status_code} - {response.text}")
				return ""
				
		except Exception as e:
			print(f"❌ Error querying Ollama: {e}")
			return ""

	def _parse_ai_suggestions(self, ai_response: str) -> List[OptimizationSuggestion]:
		"""Parse AI response into optimization suggestions."""
		suggestions = []
		
		try:
			# Try to extract JSON from response
			json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
			if json_match:
				json_str = json_match.group(0)
				parsed = json.loads(json_str)
				
				for suggestion_data in parsed.get('suggestions', []):
					suggestions.append(OptimizationSuggestion(
						field=suggestion_data.get('field', ''),
						current_value=suggestion_data.get('current_value'),
						suggested_value=suggestion_data.get('suggested_value'),
						reasoning=suggestion_data.get('reasoning', ''),
						confidence=float(suggestion_data.get('confidence', 0.5)),
						expected_improvement=suggestion_data.get('expected_improvement', '')
					))
		except Exception as e:
			print(f"⚠️ Failed to parse AI suggestions: {e}")
		
		return suggestions

	def _generate_rule_based_suggestions(
		self,
		config_data: Dict[str, Any],
		analysis: Dict[str, Any]
	) -> List[OptimizationSuggestion]:
		"""Generate rule-based optimization suggestions."""
		suggestions = []
		
		# Security-based suggestions
		for issue in analysis['security_issues']:
			if issue['type'] == 'potential_secret':
				suggestions.append(OptimizationSuggestion(
					field=issue['path'],
					current_value="<redacted>",
					suggested_value="${SECRET_" + issue['path'].upper().replace('.', '_') + "}",
					reasoning="Replace hardcoded secret with environment variable",
					confidence=0.90,
					expected_improvement="security"
				))
		
		# Performance-based suggestions
		for issue in analysis['performance_issues']:
			if issue['type'] == 'high_timeout':
				suggestions.append(OptimizationSuggestion(
					field=issue['path'],
					current_value=issue.get('current_value'),
					suggested_value=30,  # 30 seconds default
					reasoning="Reduce timeout to prevent resource exhaustion",
					confidence=0.75,
					expected_improvement="performance"
				))
		
		return suggestions

	def _apply_optimizations(
		self,
		config_data: Dict[str, Any],
		suggestions: List[OptimizationSuggestion]
	) -> Dict[str, Any]:
		"""Apply high-confidence optimization suggestions."""
		optimized_config = config_data.copy()
		
		for suggestion in suggestions:
			if suggestion.confidence >= 0.8:  # Only apply high-confidence suggestions
				try:
					# Apply the optimization
					self._set_nested_value(optimized_config, suggestion.field, suggestion.suggested_value)
				except Exception as e:
					print(f"⚠️ Failed to apply optimization for {suggestion.field}: {e}")
		
		return optimized_config

	def _set_nested_value(self, obj: Dict[str, Any], path: str, value: Any):
		"""Set nested dictionary value by path."""
		keys = path.split('.')
		current = obj
		
		for key in keys[:-1]:
			if key not in current:
				current[key] = {}
			current = current[key]
		
		current[keys[-1]] = value

	# ==================== Natural Language Processing ====================

	async def parse_natural_language_query(self, query: str) -> Dict[str, Any]:
		"""Parse natural language query into configuration filters."""
		try:
			# Use language model to understand intent
			prompt = f"""
Parse this configuration search query into structured filters:
Query: "{query}"

Return JSON format:
{{
  "intent": "search/filter/action",
  "filters": {{
    "key_pattern": "*pattern*",
    "status": "active/draft/deprecated",
    "tags": ["tag1", "tag2"],
    "date_range": {{"after": "2025-01-01", "before": "2025-12-31"}},
    "security_level": "public/internal/confidential"
  }},
  "confidence": 0.85
}}

Examples:
- "show me all database configurations" → {{"key_pattern": "*database*"}}
- "find production secrets" → {{"key_pattern": "*secret*", "tags": ["production"]}}
- "configs updated this week" → {{"date_range": {{"after": "2025-01-24"}}}}
"""
			
			ai_response = await self._query_ollama(self.language_model, prompt)
			return self._parse_query_response(ai_response)
			
		except Exception as e:
			print(f"⚠️ Natural language parsing failed: {e}")
			return {}

	def _parse_query_response(self, ai_response: str) -> Dict[str, Any]:
		"""Parse AI response for query filters."""
		try:
			json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
			if json_match:
				return json.loads(json_match.group(0))
		except:
			pass
		
		# Fallback to keyword-based parsing
		filters = {}
		query_lower = ai_response.lower()
		
		if 'database' in query_lower:
			filters['key_pattern'] = '*database*'
		if 'secret' in query_lower:
			filters['key_pattern'] = '*secret*'
		if 'production' in query_lower:
			filters['tags'] = ['production']
		
		return {'filters': filters, 'confidence': 0.6}

	# ==================== Recommendation Engine ====================

	async def generate_recommendations(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate AI-powered configuration recommendations."""
		recommendations = []
		
		try:
			# Analyze configuration patterns
			pattern_analysis = await self._analyze_configuration_patterns(config_data)
			
			# Generate recommendations based on analysis
			ai_recommendations = await self._generate_ai_recommendations(config_data, pattern_analysis)
			recommendations.extend(ai_recommendations)
			
			# Add rule-based recommendations
			rule_recommendations = self._generate_rule_based_recommendations(config_data)
			recommendations.extend(rule_recommendations)
			
			# Sort by priority and confidence
			recommendations.sort(key=lambda x: (x['priority'], -x['confidence']))
			
			return recommendations[:10]  # Return top 10
			
		except Exception as e:
			print(f"⚠️ Recommendation generation failed: {e}")
			return []

	async def _analyze_configuration_patterns(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze configuration patterns using ML."""
		analysis = {
			'structure_similarity': 0.0,
			'common_patterns': [],
			'anomalies': [],
			'optimization_opportunities': []
		}
		
		try:
			# Use embedding model to analyze semantic similarity
			if self.embedding_engine:
				config_text = json.dumps(config_data, sort_keys=True)
				embeddings = self.embedding_engine.encode([config_text])
				
				# Compare with known good patterns
				similarities = []
				for pattern in self.config_patterns:
					pattern_text = json.dumps(pattern, sort_keys=True)
					pattern_embeddings = self.embedding_engine.encode([pattern_text])
					similarity = np.dot(embeddings[0], pattern_embeddings[0])
					similarities.append(similarity)
				
				analysis['structure_similarity'] = float(np.mean(similarities))
		
		except Exception as e:
			print(f"⚠️ Pattern analysis failed: {e}")
		
		return analysis

	async def _generate_ai_recommendations(
		self,
		config_data: Dict[str, Any],
		pattern_analysis: Dict[str, Any]
	) -> List[Dict[str, Any]]:
		"""Generate AI-powered recommendations."""
		recommendations = []
		
		try:
			prompt = f"""
As a configuration management expert, analyze this configuration and provide improvement recommendations:

Configuration:
{json.dumps(config_data, indent=2)}

Pattern Analysis:
- Structure Similarity: {pattern_analysis['structure_similarity']:.2f}
- Common Patterns: {len(pattern_analysis['common_patterns'])}

Provide recommendations in JSON format:
{{
  "recommendations": [
    {{
      "type": "performance/security/cost/reliability/compliance",
      "title": "Brief title",
      "description": "Detailed description",
      "model": "llama3.2:3b",
      "confidence": 0.85,
      "impact": 0.75,
      "priority": 1,
      "current_config": {{"relevant": "fields"}},
      "recommended_config": {{"optimized": "fields"}},
      "benefits": ["benefit1", "benefit2"],
      "steps": ["step1", "step2"]
    }}
  ]
}}

Focus on actionable improvements that provide measurable benefits.
"""
			
			ai_response = await self._query_ollama(self.language_model, prompt)
			parsed_recommendations = self._parse_recommendations_response(ai_response)
			
			for rec in parsed_recommendations:
				recommendations.append({
					'type': rec.get('type', 'general'),
					'title': rec.get('title', 'AI Recommendation'),
					'description': rec.get('description', ''),
					'model': self.language_model,
					'confidence': float(rec.get('confidence', 0.5)),
					'impact': float(rec.get('impact', 0.5)),
					'priority': int(rec.get('priority', 3)),
					'current_config': rec.get('current_config', {}),
					'recommended_config': rec.get('recommended_config', {}),
					'benefits': rec.get('benefits', []),
					'steps': rec.get('steps', [])
				})
				
		except Exception as e:
			print(f"⚠️ AI recommendation generation failed: {e}")
		
		return recommendations

	def _parse_recommendations_response(self, ai_response: str) -> List[Dict[str, Any]]:
		"""Parse AI response into recommendations."""
		try:
			json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
			if json_match:
				parsed = json.loads(json_match.group(0))
				return parsed.get('recommendations', [])
		except:
			pass
		
		return []

	def _generate_rule_based_recommendations(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate rule-based recommendations."""
		recommendations = []
		
		# Check for common optimization opportunities
		if self._has_database_config(config_data):
			recommendations.append({
				'type': 'performance',
				'title': 'Database Connection Pooling',
				'description': 'Configure connection pooling for better database performance',
				'model': 'rule-based',
				'confidence': 0.80,
				'impact': 0.70,
				'priority': 2,
				'current_config': {},
				'recommended_config': {'pool_size': 10, 'max_overflow': 20},
				'benefits': ['Improved performance', 'Reduced connection overhead'],
				'steps': ['Configure connection pool', 'Set appropriate pool size', 'Monitor performance']
			})
		
		if self._has_caching_opportunity(config_data):
			recommendations.append({
				'type': 'performance',
				'title': 'Enable Caching',
				'description': 'Add caching layer to improve response times',
				'model': 'rule-based',
				'confidence': 0.75,
				'impact': 0.65,
				'priority': 2,
				'current_config': {},
				'recommended_config': {'cache_enabled': True, 'cache_ttl': 300},
				'benefits': ['Faster response times', 'Reduced load'],
				'steps': ['Configure cache backend', 'Set cache policies', 'Monitor hit rates']
			})
		
		return recommendations

	def _has_database_config(self, config_data: Dict[str, Any]) -> bool:
		"""Check if configuration has database settings."""
		config_str = json.dumps(config_data).lower()
		return any(db_term in config_str for db_term in ['database', 'postgres', 'mysql', 'mongo', 'redis'])

	def _has_caching_opportunity(self, config_data: Dict[str, Any]) -> bool:
		"""Check if configuration could benefit from caching."""
		config_str = json.dumps(config_data).lower()
		return 'cache' not in config_str and any(term in config_str for term in ['api', 'http', 'web'])

	# ==================== Anomaly Detection ====================

	async def detect_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect configuration anomalies using AI."""
		anomalies = []
		
		try:
			# Statistical anomaly detection
			statistical_anomalies = self._detect_statistical_anomalies(metrics_data)
			anomalies.extend(statistical_anomalies)
			
			# Pattern-based anomaly detection
			pattern_anomalies = await self._detect_pattern_anomalies(metrics_data)
			anomalies.extend(pattern_anomalies)
			
			# Behavioral anomaly detection
			behavioral_anomalies = self._detect_behavioral_anomalies(metrics_data)
			anomalies.extend(behavioral_anomalies)
			
			return anomalies
			
		except Exception as e:
			print(f"⚠️ Anomaly detection failed: {e}")
			return []

	def _detect_statistical_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect statistical anomalies in metrics."""
		anomalies = []
		
		for service_id, metrics in metrics_data.items():
			try:
				# Check error rate anomalies
				error_rate = metrics.get('error_rate', 0)
				if error_rate > 10:  # > 10% error rate
					anomalies.append({
						'type': 'high_error_rate',
						'service': service_id,
						'title': 'High Error Rate Detected',
						'description': f'Service {service_id} has {error_rate:.1f}% error rate',
						'confidence': 0.90,
						'severity': 'high' if error_rate > 20 else 'medium',
						'value': error_rate,
						'threshold': 10
					})
				
				# Check response time anomalies
				response_time = metrics.get('avg_response_time', 0)
				if response_time > 2000:  # > 2 seconds
					anomalies.append({
						'type': 'high_latency',
						'service': service_id,
						'title': 'High Latency Detected',
						'description': f'Service {service_id} has {response_time:.0f}ms average response time',
						'confidence': 0.85,
						'severity': 'high' if response_time > 5000 else 'medium',
						'value': response_time,
						'threshold': 2000
					})
				
			except Exception as e:
				print(f"⚠️ Statistical anomaly detection failed for {service_id}: {e}")
		
		return anomalies

	async def _detect_pattern_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect pattern-based anomalies using AI."""
		anomalies = []
		
		try:
			# Use AI to analyze patterns
			prompt = f"""
Analyze these service metrics for anomalies and unusual patterns:

Metrics Data:
{json.dumps(metrics_data, indent=2)}

Identify anomalies in JSON format:
{{
  "anomalies": [
    {{
      "type": "anomaly_type",
      "service": "service_id", 
      "title": "Anomaly Title",
      "description": "Detailed description",
      "confidence": 0.85,
      "severity": "low/medium/high",
      "pattern": "description of unusual pattern"
    }}
  ]
}}

Look for:
1. Unusual traffic patterns
2. Sudden changes in behavior  
3. Resource utilization anomalies
4. Performance degradation patterns
"""
			
			ai_response = await self._query_ollama(self.language_model, prompt)
			parsed_anomalies = self._parse_anomalies_response(ai_response)
			anomalies.extend(parsed_anomalies)
			
		except Exception as e:
			print(f"⚠️ Pattern anomaly detection failed: {e}")
		
		return anomalies

	def _parse_anomalies_response(self, ai_response: str) -> List[Dict[str, Any]]:
		"""Parse AI response for anomalies."""
		try:
			json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
			if json_match:
				parsed = json.loads(json_match.group(0))
				return parsed.get('anomalies', [])
		except:
			pass
		
		return []

	def _detect_behavioral_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect behavioral anomalies based on historical patterns."""
		anomalies = []
		
		# This would typically compare against historical data
		# For now, implement basic heuristics
		
		for service_id, metrics in metrics_data.items():
			# Check for zero traffic (potential service down)
			if metrics.get('request_count', 0) == 0:
				anomalies.append({
					'type': 'no_traffic',
					'service': service_id,
					'title': 'No Traffic Detected',
					'description': f'Service {service_id} has no incoming requests',
					'confidence': 0.95,
					'severity': 'high'
				})
		
		return anomalies

	# ==================== Traffic Prediction ====================

	async def predict_traffic(self, metrics_data: Dict[str, Any], horizon_hours: int = 4) -> Dict[str, Any]:
		"""Predict traffic patterns using AI."""
		predictions = {}
		
		try:
			for service_id, metrics in metrics_data.items():
				# Simple trend-based prediction
				current_rps = metrics.get('current_rps', 0)
				request_count = metrics.get('request_count', 0)
				
				# Basic linear prediction (would be replaced with ML model)
				predicted_rps = self._predict_service_traffic(current_rps, request_count, horizon_hours)
				
				predictions[service_id] = {
					'current_rps': current_rps,
					'predicted_rps': predicted_rps,
					'confidence': 0.70,
					'trend': 'increasing' if predicted_rps > current_rps else 'decreasing',
					'horizon_hours': horizon_hours
				}
			
			return predictions
			
		except Exception as e:
			print(f"⚠️ Traffic prediction failed: {e}")
			return {}

	def _predict_service_traffic(self, current_rps: float, request_count: int, horizon_hours: int) -> float:
		"""Predict traffic for a service."""
		# Simple trend analysis (would be replaced with proper ML model)
		if request_count > 1000:  # High traffic service
			growth_rate = 0.05  # 5% growth
		else:
			growth_rate = 0.02  # 2% growth
		
		predicted_rps = current_rps * (1 + growth_rate * horizon_hours)
		return max(0, predicted_rps)

	# ==================== Configuration Patterns ====================

	def _load_configuration_patterns(self) -> List[Dict[str, Any]]:
		"""Load common configuration patterns."""
		return [
			{
				'name': 'database_config',
				'pattern': {
					'host': 'string',
					'port': 'integer',
					'database': 'string',
					'pool_size': 'integer',
					'timeout': 'integer'
				}
			},
			{
				'name': 'cache_config', 
				'pattern': {
					'enabled': 'boolean',
					'ttl': 'integer',
					'max_size': 'integer',
					'backend': 'string'
				}
			},
			{
				'name': 'api_config',
				'pattern': {
					'base_url': 'string',
					'timeout': 'integer',
					'retry_attempts': 'integer',
					'rate_limit': 'integer'
				}
			}
		]

	def _load_security_rules(self) -> List[Dict[str, Any]]:
		"""Load security validation rules."""
		return [
			{
				'rule': 'no_hardcoded_secrets',
				'description': 'Detect hardcoded passwords, keys, tokens'
			},
			{
				'rule': 'ssl_enabled',
				'description': 'Ensure SSL/TLS is enabled for external connections'
			},
			{
				'rule': 'authentication_required',
				'description': 'Verify authentication is configured'
			}
		]

	def _load_performance_rules(self) -> List[Dict[str, Any]]:
		"""Load performance optimization rules."""
		return [
			{
				'rule': 'connection_pooling',
				'description': 'Use connection pooling for database connections'
			},
			{
				'rule': 'caching_enabled',
				'description': 'Enable caching for frequently accessed data'
			},
			{
				'rule': 'timeout_settings',
				'description': 'Configure appropriate timeout values'
			}
		]

	async def close(self):
		"""Close AI engine and cleanup resources."""
		try:
			await self.client.aclose()
			print("✅ AI Engine closed successfully")
		except Exception as e:
			print(f"⚠️ AI Engine cleanup failed: {e}")


class TrafficPredictionModel:
	"""Traffic prediction model using time series analysis."""
	
	def __init__(self):
		self.model_initialized = False
	
	async def predict_traffic(self, metrics_data: Dict[str, Any], horizon_hours: int = 4) -> Dict[str, Any]:
		"""Predict traffic patterns."""
		predictions = {}
		
		for service_id, metrics in metrics_data.items():
			current_rps = metrics.get('current_rps', 0)
			trend = self._calculate_trend(metrics.get('response_time_trend', []))
			
			predictions[service_id] = {
				'predicted_rps': current_rps * (1 + trend * horizon_hours),
				'confidence': 0.75,
				'trend_direction': 'up' if trend > 0 else 'down',
				'horizon_hours': horizon_hours
			}
		
		return predictions
	
	def _calculate_trend(self, time_series_data: List[float]) -> float:
		"""Calculate trend from time series data."""
		if len(time_series_data) < 2:
			return 0.0
		
		# Simple linear trend calculation
		x = list(range(len(time_series_data)))
		y = time_series_data
		
		n = len(x)
		sum_x = sum(x)
		sum_y = sum(y)
		sum_xy = sum(xi * yi for xi, yi in zip(x, y))
		sum_x2 = sum(xi * xi for xi in x)
		
		slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
		return slope / 100.0  # Normalize


class AnomalyDetectionModel:
	"""Anomaly detection model for configuration monitoring."""
	
	def __init__(self):
		self.baseline_metrics = {}
		self.anomaly_threshold = 2.0  # Standard deviations
	
	async def detect_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect anomalies in metrics data."""
		anomalies = []
		
		for service_id, metrics in metrics_data.items():
			service_anomalies = self._check_service_anomalies(service_id, metrics)
			anomalies.extend(service_anomalies)
		
		return anomalies
	
	def _check_service_anomalies(self, service_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Check for anomalies in service metrics."""
		anomalies = []
		
		# Check error rate anomaly
		error_rate = metrics.get('error_rate', 0)
		if error_rate > 5:  # > 5% error rate is anomalous
			anomalies.append({
				'service': service_id,
				'type': 'error_rate_spike',
				'title': 'High Error Rate',
				'description': f'Error rate is {error_rate:.1f}%',
				'confidence': 0.90,
				'severity': 'high' if error_rate > 10 else 'medium'
			})
		
		# Check response time anomaly
		response_time = metrics.get('avg_response_time', 0)
		if response_time > 1000:  # > 1 second
			anomalies.append({
				'service': service_id,
				'type': 'latency_spike',
				'title': 'High Latency',
				'description': f'Average response time is {response_time:.0f}ms',
				'confidence': 0.85,
				'severity': 'high' if response_time > 5000 else 'medium'
			})
		
		return anomalies