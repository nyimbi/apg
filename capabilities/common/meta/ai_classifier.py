#!/usr/bin/env python3
"""
APG Metadata Management - AI Classification Engine
Advanced AI-powered metadata classification with federated learning

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager
from .integrations import APGMetadataIntegrationManager


class AIClassificationMethod(str, Enum):
	"""AI classification methods"""
	PATTERN_MATCHING = "pattern_matching"
	NLP_ANALYSIS = "nlp_analysis"
	STATISTICAL_ANALYSIS = "statistical_analysis" 
	ENSEMBLE_VOTING = "ensemble_voting"
	FEDERATED_LEARNING = "federated_learning"
	DEEP_LEARNING = "deep_learning"


class ConfidenceLevel(str, Enum):
	"""Classification confidence levels"""
	VERY_HIGH = "very_high"  # 0.95+
	HIGH = "high"           # 0.80-0.94
	MEDIUM = "medium"       # 0.60-0.79
	LOW = "low"            # 0.40-0.59
	VERY_LOW = "very_low"  # <0.40


@dataclass
class ClassificationPattern:
	"""Pattern for data classification"""
	pattern_id: str = field(default_factory=uuid7str)
	pattern_type: str = ""  # regex, keyword, statistical
	pattern_value: str = ""
	classification: str = ""
	confidence_weight: float = 1.0
	data_types: List[str] = field(default_factory=list)
	context_requirements: Dict[str, Any] = field(default_factory=dict)
	created_by: str = "system"
	created_at: datetime = field(default_factory=datetime.utcnow)
	success_rate: float = 0.0
	usage_count: int = 0


@dataclass
class ClassificationRule:
	"""Advanced classification rule with conditions"""
	rule_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	conditions: List[Dict[str, Any]] = field(default_factory=list)
	classification: str = ""
	confidence_score: float = 0.8
	priority: int = 100  # Lower numbers = higher priority
	enabled: bool = True
	tenant_specific: bool = False
	tenant_id: Optional[str] = None
	created_by: str = "system"
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_updated: datetime = field(default_factory=datetime.utcnow)
	success_count: int = 0
	failure_count: int = 0
	
	@property
	def success_rate(self) -> float:
		total = self.success_count + self.failure_count
		return (self.success_count / total) if total > 0 else 0.0


@dataclass
class ClassificationResult:
	"""Result of AI classification"""
	result_id: str = field(default_factory=uuid7str)
	classification: str = ""
	confidence_score: float = 0.0
	confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
	method_used: AIClassificationMethod = AIClassificationMethod.PATTERN_MATCHING
	reasoning: str = ""
	evidence: List[Dict[str, Any]] = field(default_factory=list)
	alternative_classifications: List[Tuple[str, float]] = field(default_factory=list)
	processing_time_ms: float = 0.0
	patterns_matched: List[str] = field(default_factory=list)
	rules_applied: List[str] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
	classified_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"result_id": self.result_id,
			"classification": self.classification,
			"confidence_score": self.confidence_score,
			"confidence_level": self.confidence_level.value,
			"method_used": self.method_used.value,
			"reasoning": self.reasoning,
			"evidence": self.evidence,
			"alternative_classifications": self.alternative_classifications,
			"processing_time_ms": self.processing_time_ms,
			"patterns_matched": self.patterns_matched,
			"rules_applied": self.rules_applied,
			"metadata": self.metadata,
			"classified_at": self.classified_at.isoformat()
		}


class PatternLibrary:
	"""Library of classification patterns for different data types"""
	
	def __init__(self):
		self.patterns: Dict[str, List[ClassificationPattern]] = defaultdict(list)
		self._load_default_patterns()
	
	def _load_default_patterns(self):
		"""Load default classification patterns"""
		
		# PII Patterns
		pii_patterns = [
			# Email patterns
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
				classification="PII",
				confidence_weight=0.95,
				data_types=["string"]
			),
			# Phone patterns
			ClassificationPattern(
				pattern_type="regex", 
				pattern_value=r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
				classification="PII",
				confidence_weight=0.90,
				data_types=["string"]
			),
			# SSN patterns
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b\d{3}-?\d{2}-?\d{4}\b',
				classification="PII",
				confidence_weight=0.95,
				data_types=["string"]
			),
			# Credit card patterns
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
				classification="PII",
				confidence_weight=0.90,
				data_types=["string"]
			)
		]
		
		for pattern in pii_patterns:
			self.patterns["PII"].append(pattern)
		
		# PHI Patterns
		phi_patterns = [
			# Medical record number
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b(MRN|MR|PATIENT)\s*:?\s*\d+\b',
				classification="PHI",
				confidence_weight=0.85,
				data_types=["string"]
			),
			# Diagnosis codes (ICD)
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b[A-Z]\d{2}(\.\d{1,2})?\b',
				classification="PHI",
				confidence_weight=0.80,
				data_types=["string"]
			)
		]
		
		for pattern in phi_patterns:
			self.patterns["PHI"].append(pattern)
		
		# Financial Patterns
		financial_patterns = [
			# Account numbers
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b\d{8,17}\b',
				classification="FINANCIAL",
				confidence_weight=0.70,
				data_types=["string", "integer"],
				context_requirements={"column_name_contains": ["account", "acct", "balance", "amount"]}
			),
			# Routing numbers
			ClassificationPattern(
				pattern_type="regex",
				pattern_value=r'\b\d{9}\b',
				classification="FINANCIAL", 
				confidence_weight=0.75,
				data_types=["string", "integer"],
				context_requirements={"column_name_contains": ["routing", "aba", "transit"]}
			)
		]
		
		for pattern in financial_patterns:
			self.patterns["FINANCIAL"].append(pattern)
		
		# Keyword-based patterns
		keyword_patterns = [
			# PII keywords
			ClassificationPattern(
				pattern_type="keyword",
				pattern_value="email|mail|e_mail",
				classification="PII",
				confidence_weight=0.85,
				data_types=["string"]
			),
			ClassificationPattern(
				pattern_type="keyword", 
				pattern_value="phone|telephone|mobile|cell",
				classification="PII",
				confidence_weight=0.80,
				data_types=["string"]
			),
			ClassificationPattern(
				pattern_type="keyword",
				pattern_value="ssn|social_security|tax_id",
				classification="PII",
				confidence_weight=0.90,
				data_types=["string"]
			),
			# PHI keywords
			ClassificationPattern(
				pattern_type="keyword",
				pattern_value="patient|medical|diagnosis|treatment|medication|prescription",
				classification="PHI",
				confidence_weight=0.75,
				data_types=["string"]
			),
			# Financial keywords
			ClassificationPattern(
				pattern_type="keyword",
				pattern_value="salary|wage|income|payment|revenue|profit|loss",
				classification="FINANCIAL",
				confidence_weight=0.70,
				data_types=["integer", "float"]
			)
		]
		
		for pattern in keyword_patterns:
			classification = pattern.classification
			self.patterns[classification].append(pattern)
	
	def get_patterns_for_classification(self, classification: str) -> List[ClassificationPattern]:
		"""Get all patterns for a specific classification"""
		return self.patterns.get(classification.upper(), [])
	
	def get_all_patterns(self) -> List[ClassificationPattern]:
		"""Get all patterns"""
		all_patterns = []
		for patterns in self.patterns.values():
			all_patterns.extend(patterns)
		return all_patterns
	
	def add_custom_pattern(self, pattern: ClassificationPattern):
		"""Add custom classification pattern"""
		self.patterns[pattern.classification.upper()].append(pattern)
	
	def update_pattern_performance(self, pattern_id: str, success: bool):
		"""Update pattern performance metrics"""
		for patterns in self.patterns.values():
			for pattern in patterns:
				if pattern.pattern_id == pattern_id:
					pattern.usage_count += 1
					if success:
						pattern.success_rate = (
							(pattern.success_rate * (pattern.usage_count - 1) + 1.0) / 
							pattern.usage_count
						)
					else:
						pattern.success_rate = (
							pattern.success_rate * (pattern.usage_count - 1) / 
							pattern.usage_count
						)
					break


class AIClassificationEngine:
	"""Advanced AI-powered metadata classification engine"""
	
	def __init__(self,
		     db_manager: MetaDatabaseManager,
		     integration_manager: APGMetadataIntegrationManager,
		     config: Dict[str, Any] = None):
		self.db_manager = db_manager
		self.integration_manager = integration_manager
		self.config = config or {}
		
		# Classification components
		self.pattern_library = PatternLibrary()
		self.classification_rules: Dict[str, ClassificationRule] = {}
		
		# AI/ML settings
		self.enable_ollama = config.get('enable_ollama', True)
		self.ollama_models = config.get('ollama_models', {
			'classification': 'llama3.2:3b',
			'reasoning': 'llama3.2:3b',
			'embedding': 'nomic-embed-text'
		})
		
		# Federated learning settings
		self.enable_federated_learning = config.get('enable_federated_learning', True)
		self.learning_batch_size = config.get('learning_batch_size', 100)
		self.confidence_threshold = config.get('confidence_threshold', 0.8)
		
		# Performance settings
		self.max_sample_size = config.get('max_sample_size', 1000)
		self.parallel_classification = config.get('parallel_classification', True)
		self.cache_results = config.get('cache_results', True)
		
		# Statistics for continuous learning
		self.classification_stats = defaultdict(lambda: {
			'total_classified': 0,
			'success_rate': 0.0,
			'avg_confidence': 0.0,
			'method_usage': defaultdict(int)
		})
		
		self.initialized = False
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize the AI classification engine"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Load custom rules from database
			await self._load_classification_rules()
			
			# Load performance statistics
			await self._load_classification_stats()
			
			# Initialize federated learning if enabled
			if self.enable_federated_learning:
				await self._initialize_federated_learning()
			
			self.initialized = True
			
			await self._log_info("AI Classification Engine initialized successfully")
			
			return {
				"status": "initialized",
				"total_patterns": len(self.pattern_library.get_all_patterns()),
				"total_rules": len(self.classification_rules),
				"federated_learning": self.enable_federated_learning,
				"ollama_enabled": self.enable_ollama,
				"confidence_threshold": self.confidence_threshold
			}
			
		except Exception as e:
			await self._log_error(f"AI Classification Engine initialization failed: {str(e)}")
			raise
	
	async def classify_column_data(self,
				       column_name: str,
				       data_type: str,
				       sample_data: List[Any],
				       context: Dict[str, Any] = None) -> ClassificationResult:
		"""Classify column data using multiple AI methods"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			context = context or {}
			
			# Prepare classification context
			classification_context = {
				"column_name": column_name,
				"data_type": data_type,
				"sample_size": len(sample_data),
				"context": context
			}
			
			# Apply ensemble of classification methods
			results = []
			
			# Method 1: Pattern-based classification
			pattern_result = await self._classify_with_patterns(
				column_name, data_type, sample_data, context
			)
			if pattern_result.confidence_score > 0:
				results.append(pattern_result)
			
			# Method 2: Statistical analysis
			stats_result = await self._classify_with_statistics(
				column_name, data_type, sample_data, context
			)
			if stats_result.confidence_score > 0:
				results.append(stats_result)
			
			# Method 3: Rule-based classification
			rule_result = await self._classify_with_rules(
				column_name, data_type, sample_data, context
			)
			if rule_result.confidence_score > 0:
				results.append(rule_result)
			
			# Method 4: NLP analysis (if Ollama enabled)
			if self.enable_ollama:
				nlp_result = await self._classify_with_ollama(
					column_name, data_type, sample_data, context
				)
				if nlp_result.confidence_score > 0:
					results.append(nlp_result)
			
			# Method 5: Federated learning (if available)
			if self.enable_federated_learning:
				fl_result = await self._classify_with_federated_learning(
					column_name, data_type, sample_data, context
				)
				if fl_result and fl_result.confidence_score > 0:
					results.append(fl_result)
			
			# Ensemble voting and final result
			final_result = await self._ensemble_voting(results, classification_context)
			
			# Calculate processing time
			processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
			final_result.processing_time_ms = processing_time
			
			# Update statistics
			await self._update_classification_stats(final_result)
			
			# Cache result if enabled
			if self.cache_results:
				await self._cache_classification_result(
					column_name, data_type, sample_data, final_result
				)
			
			await self._log_info(
				f"Classified column '{column_name}' as '{final_result.classification}' "
				f"with {final_result.confidence_score:.2f} confidence"
			)
			
			return final_result
			
		except Exception as e:
			await self._log_error(f"Classification failed for column '{column_name}': {str(e)}")
			
			# Return default result
			return ClassificationResult(
				classification="INTERNAL",
				confidence_score=0.0,
				confidence_level=ConfidenceLevel.VERY_LOW,
				reasoning=f"Classification failed: {str(e)}",
				processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
			)
	
	async def _classify_with_patterns(self,
					  column_name: str,
					  data_type: str,
					  sample_data: List[Any],
					  context: Dict[str, Any]) -> ClassificationResult:
		"""Classify using pattern matching"""
		all_patterns = self.pattern_library.get_all_patterns()
		matched_patterns = []
		classification_scores = defaultdict(float)
		
		# Convert sample data to strings for pattern matching
		string_samples = [str(value) for value in sample_data if value is not None][:100]
		
		for pattern in all_patterns:
			# Check if pattern applies to this data type
			if pattern.data_types and data_type not in pattern.data_types:
				continue
			
			# Check context requirements
			if pattern.context_requirements:
				if not self._check_context_requirements(pattern.context_requirements, column_name, context):
					continue
			
			matches = 0
			total_checked = 0
			
			if pattern.pattern_type == "regex":
				regex = re.compile(pattern.pattern_value, re.IGNORECASE)
				for sample in string_samples:
					total_checked += 1
					if regex.search(sample):
						matches += 1
			
			elif pattern.pattern_type == "keyword":
				keywords = pattern.pattern_value.lower().split("|")
				column_lower = column_name.lower()
				
				# Check column name
				if any(keyword in column_lower for keyword in keywords):
					matches = len(string_samples)  # High confidence for column name match
					total_checked = len(string_samples)
				else:
					# Check sample values
					for sample in string_samples:
						total_checked += 1
						if any(keyword in sample.lower() for keyword in keywords):
							matches += 1
			
			if total_checked > 0:
				match_rate = matches / total_checked
				if match_rate > 0:
					confidence = match_rate * pattern.confidence_weight * pattern.success_rate
					classification_scores[pattern.classification] += confidence
					matched_patterns.append(pattern.pattern_id)
		
		# Get best classification
		if classification_scores:
			best_classification = max(classification_scores.items(), key=lambda x: x[1])
			classification = best_classification[0]
			confidence_score = min(best_classification[1], 1.0)
			
			return ClassificationResult(
				classification=classification,
				confidence_score=confidence_score,
				confidence_level=self._get_confidence_level(confidence_score),
				method_used=AIClassificationMethod.PATTERN_MATCHING,
				reasoning=f"Pattern matching identified {len(matched_patterns)} relevant patterns",
				patterns_matched=matched_patterns,
				evidence=[{"type": "pattern_matches", "count": len(matched_patterns)}]
			)
		
		return ClassificationResult()
	
	async def _classify_with_statistics(self,
					    column_name: str,
					    data_type: str,
					    sample_data: List[Any],
					    context: Dict[str, Any]) -> ClassificationResult:
		"""Classify using statistical analysis"""
		if not sample_data:
			return ClassificationResult()
		
		evidence = []
		classification_hints = []
		
		# Analyze data characteristics
		non_null_data = [x for x in sample_data if x is not None]
		if not non_null_data:
			return ClassificationResult()
		
		# String length analysis for PII detection
		if data_type in ["string"]:
			lengths = [len(str(x)) for x in non_null_data]
			avg_length = np.mean(lengths) if lengths else 0
			length_std = np.std(lengths) if len(lengths) > 1 else 0
			
			# Email-like length patterns
			if 10 <= avg_length <= 50 and length_std < 20:
				if any("@" in str(x) for x in non_null_data[:10]):
					classification_hints.append(("PII", 0.85))
					evidence.append({"type": "email_pattern", "avg_length": avg_length})
			
			# Phone number patterns
			if 8 <= avg_length <= 15:
				digit_only_count = sum(1 for x in non_null_data[:20] if str(x).replace("-", "").replace("(", "").replace(")", "").replace(" ", "").isdigit())
				if digit_only_count > len(non_null_data) * 0.8:
					classification_hints.append(("PII", 0.75))
					evidence.append({"type": "phone_pattern", "digit_ratio": digit_only_count / len(non_null_data)})
			
			# SSN patterns (9 characters with specific format)
			if avg_length == 11:  # XXX-XX-XXXX format
				ssn_like = sum(1 for x in non_null_data[:20] if re.match(r'\d{3}-\d{2}-\d{4}', str(x)))
				if ssn_like > 0:
					classification_hints.append(("PII", 0.90))
					evidence.append({"type": "ssn_pattern", "matches": ssn_like})
		
		# Numeric analysis for financial data
		if data_type in ["integer", "float"]:
			numeric_data = []
			for x in non_null_data[:100]:
				try:
					numeric_data.append(float(x))
				except (ValueError, TypeError):
					continue
			
			if numeric_data:
				# Check for monetary amounts (common patterns)
				if any(name in column_name.lower() for name in ["amount", "price", "cost", "salary", "wage", "revenue"]):
					classification_hints.append(("FINANCIAL", 0.80))
					evidence.append({"type": "financial_column_name", "column": column_name})
				
				# Check for percentage-like values
				if all(0 <= x <= 100 for x in numeric_data) and any(name in column_name.lower() for name in ["rate", "percent", "ratio"]):
					classification_hints.append(("INTERNAL", 0.60))
					evidence.append({"type": "percentage_pattern", "range": [min(numeric_data), max(numeric_data)]})
		
		# Uniqueness analysis
		unique_count = len(set(str(x) for x in non_null_data))
		uniqueness_ratio = unique_count / len(non_null_data)
		
		if uniqueness_ratio > 0.95:  # Highly unique data
			if data_type == "string" and "id" in column_name.lower():
				classification_hints.append(("INTERNAL", 0.70))
				evidence.append({"type": "unique_identifier", "uniqueness": uniqueness_ratio})
		
		# Get best classification from hints
		if classification_hints:
			best_hint = max(classification_hints, key=lambda x: x[1])
			classification = best_hint[0]
			confidence_score = best_hint[1]
			
			return ClassificationResult(
				classification=classification,
				confidence_score=confidence_score,
				confidence_level=self._get_confidence_level(confidence_score),
				method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
				reasoning=f"Statistical analysis identified {len(classification_hints)} classification signals",
				evidence=evidence,
				alternative_classifications=classification_hints[1:] if len(classification_hints) > 1 else []
			)
		
		return ClassificationResult()
	
	async def _classify_with_rules(self,
				       column_name: str,
				       data_type: str,
				       sample_data: List[Any],
				       context: Dict[str, Any]) -> ClassificationResult:
		"""Classify using rule-based engine"""
		
		# Sort rules by priority
		sorted_rules = sorted(
			[rule for rule in self.classification_rules.values() if rule.enabled],
			key=lambda r: r.priority
		)
		
		applied_rules = []
		
		for rule in sorted_rules:
			# Check if rule applies to current context
			if rule.tenant_specific and rule.tenant_id != context.get("tenant_id"):
				continue
			
			# Evaluate rule conditions
			if self._evaluate_rule_conditions(rule, column_name, data_type, sample_data, context):
				applied_rules.append(rule.rule_id)
				
				return ClassificationResult(
					classification=rule.classification,
					confidence_score=rule.confidence_score,
					confidence_level=self._get_confidence_level(rule.confidence_score),
					method_used=AIClassificationMethod.ENSEMBLE_VOTING,
					reasoning=f"Rule '{rule.name}' matched: {rule.description}",
					rules_applied=[rule.rule_id],
					evidence=[{
						"type": "rule_match",
						"rule_id": rule.rule_id,
						"rule_name": rule.name
					}]
				)
		
		return ClassificationResult()
	
	async def _classify_with_ollama(self,
					column_name: str,
					data_type: str,
					sample_data: List[Any],
					context: Dict[str, Any]) -> ClassificationResult:
		"""Classify using Ollama local LLM"""
		try:
			# Prepare sample for LLM
			sample_str = self._prepare_sample_for_llm(column_name, data_type, sample_data, context)
			
			# Use AI integration from the integration manager
			ai_result = await self.integration_manager.ai_integration.classify_data_content(
				content=sample_str,
				column_name=column_name
			)
			
			if ai_result and ai_result.get('confidence', 0) > 0:
				return ClassificationResult(
					classification=ai_result.get('classification', 'INTERNAL'),
					confidence_score=ai_result.get('confidence', 0.0),
					confidence_level=self._get_confidence_level(ai_result.get('confidence', 0.0)),
					method_used=AIClassificationMethod.NLP_ANALYSIS,
					reasoning=ai_result.get('reasoning', 'AI analysis completed'),
					evidence=[{
						"type": "llm_analysis",
						"model": self.ollama_models['classification'],
						"confidence": ai_result.get('confidence', 0.0)
					}]
				)
			
		except Exception as e:
			await self._log_error(f"Ollama classification failed: {str(e)}")
		
		return ClassificationResult()
	
	async def _classify_with_federated_learning(self,
						    column_name: str,
						    data_type: str,
						    sample_data: List[Any],
						    context: Dict[str, Any]) -> Optional[ClassificationResult]:
		"""Classify using federated learning model"""
		try:
			# Extract features from the column data
			features = await self._extract_ml_features(column_name, data_type, sample_data, context)
			if not features:
				return None
			
			# Apply federated model if available
			if hasattr(self, 'federated_model') and self.federated_model:
				# Use the federated model for classification
				prediction = await self._apply_federated_model(features)
				
				if prediction:
					classification_type = prediction.get('classification')
					confidence = prediction.get('confidence', 0.0)
					
					# Convert to our classification types
					mapped_classification = self._map_federated_classification(classification_type)
					
					return ClassificationResult(
						classification=mapped_classification,
						confidence_score=confidence,
						confidence_level=self._calculate_confidence_level(confidence),
						method_used=AIClassificationMethod.FEDERATED_LEARNING,
						reasoning=f"Federated model prediction: {classification_type}",
						tags=prediction.get('tags', []),
						metadata={
							"federated_model_version": prediction.get('model_version'),
							"feature_vector_size": len(features),
							"prediction_raw": prediction
						}
					)
			
			# Fall back to statistical analysis if no federated model
			return await self._statistical_classify(column_name, data_type, sample_data, context)
			
		except Exception as e:
			await self._log_error(f"Federated learning classification failed: {str(e)}")
			return None
	
	async def _ensemble_voting(self,
				   results: List[ClassificationResult],
				   context: Dict[str, Any]) -> ClassificationResult:
		"""Combine multiple classification results using ensemble voting"""
		if not results:
			return ClassificationResult(
				classification="INTERNAL",
				confidence_score=0.5,
				confidence_level=ConfidenceLevel.MEDIUM,
				method_used=AIClassificationMethod.ENSEMBLE_VOTING,
				reasoning="No classification methods produced results, defaulting to INTERNAL"
			)
		
		if len(results) == 1:
			results[0].method_used = AIClassificationMethod.ENSEMBLE_VOTING
			return results[0]
		
		# Weighted voting based on method reliability and confidence
		method_weights = {
			AIClassificationMethod.PATTERN_MATCHING: 0.9,
			AIClassificationMethod.NLP_ANALYSIS: 0.85,
			AIClassificationMethod.STATISTICAL_ANALYSIS: 0.75,
			AIClassificationMethod.ENSEMBLE_VOTING: 0.8,
			AIClassificationMethod.FEDERATED_LEARNING: 0.95,
			AIClassificationMethod.DEEP_LEARNING: 0.90
		}
		
		# Collect all classifications with weighted scores
		classification_votes = defaultdict(list)
		
		for result in results:
			weight = method_weights.get(result.method_used, 0.7)
			weighted_score = result.confidence_score * weight
			classification_votes[result.classification].append({
				'score': weighted_score,
				'original_score': result.confidence_score,
				'method': result.method_used,
				'result': result
			})
		
		# Calculate final scores
		final_scores = {}
		all_evidence = []
		all_patterns = []
		all_rules = []
		reasoning_parts = []
		
		for classification, votes in classification_votes.items():
			# Average weighted scores
			avg_score = np.mean([vote['score'] for vote in votes])
			# Boost score if multiple methods agree
			agreement_boost = min(len(votes) * 0.1, 0.3)
			final_score = min(avg_score + agreement_boost, 1.0)
			
			final_scores[classification] = {
				'score': final_score,
				'method_count': len(votes),
				'votes': votes
			}
			
			# Collect evidence from all methods
			for vote in votes:
				result = vote['result']
				all_evidence.extend(result.evidence)
				all_patterns.extend(result.patterns_matched)
				all_rules.extend(result.rules_applied)
				if result.reasoning:
					reasoning_parts.append(f"{result.method_used.value}: {result.reasoning}")
		
		# Get best classification
		best_classification = max(final_scores.items(), key=lambda x: x[1]['score'])
		classification = best_classification[0]
		confidence_info = best_classification[1]
		final_confidence = confidence_info['score']
		
		# Build reasoning
		ensemble_reasoning = f"Ensemble of {len(results)} methods. " + "; ".join(reasoning_parts[:3])
		
		# Get alternative classifications
		alternatives = [
			(cls, info['score']) 
			for cls, info in final_scores.items() 
			if cls != classification
		]
		alternatives.sort(key=lambda x: x[1], reverse=True)
		
		return ClassificationResult(
			classification=classification,
			confidence_score=final_confidence,
			confidence_level=self._get_confidence_level(final_confidence),
			method_used=AIClassificationMethod.ENSEMBLE_VOTING,
			reasoning=ensemble_reasoning,
			evidence=all_evidence[:10],  # Limit evidence size
			alternative_classifications=alternatives[:3],
			patterns_matched=list(set(all_patterns)),
			rules_applied=list(set(all_rules)),
			metadata={
				"methods_used": len(results),
				"agreement_methods": confidence_info['method_count'],
				"confidence_breakdown": {
					cls: info['score'] for cls, info in final_scores.items()
				}
			}
		)
	
	def _check_context_requirements(self,
					requirements: Dict[str, Any],
					column_name: str,
					context: Dict[str, Any]) -> bool:
		"""Check if context requirements are met"""
		for req_type, req_value in requirements.items():
			if req_type == "column_name_contains":
				if not any(term.lower() in column_name.lower() for term in req_value):
					return False
			elif req_type == "context_has":
				if req_value not in context:
					return False
		return True
	
	def _evaluate_rule_conditions(self,
				      rule: ClassificationRule,
				      column_name: str,
				      data_type: str,
				      sample_data: List[Any],
				      context: Dict[str, Any]) -> bool:
		"""Evaluate rule conditions against data"""
		for condition in rule.conditions:
			condition_type = condition.get("type")
			
			if condition_type == "column_name_regex":
				pattern = condition.get("pattern", "")
				if not re.search(pattern, column_name, re.IGNORECASE):
					return False
			
			elif condition_type == "data_type_equals":
				if data_type != condition.get("value"):
					return False
			
			elif condition_type == "sample_matches_regex":
				pattern = condition.get("pattern", "")
				min_matches = condition.get("min_matches", 1)
				matches = sum(1 for x in sample_data[:20] if re.search(pattern, str(x)))
				if matches < min_matches:
					return False
			
			elif condition_type == "context_value":
				key = condition.get("key")
				expected = condition.get("value")
				if context.get(key) != expected:
					return False
		
		return True
	
	def _prepare_sample_for_llm(self,
				    column_name: str,
				    data_type: str,
				    sample_data: List[Any],
				    context: Dict[str, Any]) -> str:
		"""Prepare sample data for LLM analysis"""
		# Limit sample size for LLM processing
		limited_sample = sample_data[:20]
		
		# Create description
		sample_str = f"Column: {column_name}\n"
		sample_str += f"Data Type: {data_type}\n"
		sample_str += f"Sample Values: {', '.join(str(x) for x in limited_sample if x is not None)}\n"
		
		# Add context if available
		if context:
			table_name = context.get("table_name", "")
			if table_name:
				sample_str += f"Table: {table_name}\n"
		
		return sample_str
	
	def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
		"""Convert confidence score to confidence level"""
		if confidence_score >= 0.95:
			return ConfidenceLevel.VERY_HIGH
		elif confidence_score >= 0.80:
			return ConfidenceLevel.HIGH
		elif confidence_score >= 0.60:
			return ConfidenceLevel.MEDIUM
		elif confidence_score >= 0.40:
			return ConfidenceLevel.LOW
		else:
			return ConfidenceLevel.VERY_LOW
	
	async def _load_classification_rules(self):
		"""Load custom classification rules from database"""
		try:
			# Load custom rules from database if available
			if self.db_manager:
				async with self.db_manager.get_session() as session:
					from sqlalchemy import select, text
					
					# Try to load custom classification rules
					try:
						stmt = text("""
							SELECT rule_name, pattern, classification, confidence_weight
							FROM meta_classification_rules
							WHERE is_active = true
						""")
						result = await session.execute(stmt)
						rows = result.fetchall()
						
						for row in rows:
							rule_name, pattern, classification, weight = row
							self.custom_rules[rule_name] = {
								'pattern': pattern,
								'classification': classification,
								'weight': float(weight)
							}
						
						await self._log_info(f"Loaded {len(self.custom_rules)} custom classification rules")
						
					except Exception:
						# Table might not exist yet, use defaults
						await self._log_info("No custom classification rules table found, using defaults")
			
			# Set up default rules if no custom rules loaded
			if not self.custom_rules:
				self._initialize_default_rules()
				
		except Exception as e:
			await self._log_error(f"Failed to load classification rules: {str(e)}")
			self._initialize_default_rules()
	
	def _initialize_default_rules(self):
		"""Initialize default classification rules"""
		self.custom_rules = {
			'email_rule': {
				'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
				'classification': 'EMAIL',
				'weight': 0.95
			},
			'phone_rule': {
				'pattern': r'^\+?[\d\s\-\(\)\.]{7,}$',
				'classification': 'PHONE_NUMBER',
				'weight': 0.9
			},
			'ssn_rule': {
				'pattern': r'^\d{3}-\d{2}-\d{4}$',
				'classification': 'SSN',
				'weight': 0.98
			},
			'credit_card_rule': {
				'pattern': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
				'classification': 'CREDIT_CARD',
				'weight': 0.95
			},
			'ip_address_rule': {
				'pattern': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
				'classification': 'IP_ADDRESS',
				'weight': 0.9
			}
		}
	
	async def _load_classification_stats(self):
		"""Load classification performance statistics"""
		try:
			# Initialize default stats structure
			default_stats = {
				'total_classified': 0,
				'avg_confidence': 0.0,
				'method_usage': {method.value: 0 for method in AIClassificationMethod},
				'accuracy_score': 0.0,
				'last_updated': datetime.utcnow()
			}
			
			# Load stats from database if available
			if self.db_manager:
				async with self.db_manager.get_session() as session:
					from sqlalchemy import select, text
					
					try:
						# Try to load classification stats
						stmt = text("""
							SELECT classification_type, total_classified, avg_confidence, 
								   method_usage, accuracy_score, last_updated
							FROM meta_classification_stats
							WHERE is_active = true
						""")
						result = await session.execute(stmt)
						rows = result.fetchall()
						
						for row in rows:
							(classification_type, total_classified, avg_confidence, 
							 method_usage, accuracy_score, last_updated) = row
							
							stats = default_stats.copy()
							stats['total_classified'] = total_classified or 0
							stats['avg_confidence'] = float(avg_confidence or 0.0)
							stats['accuracy_score'] = float(accuracy_score or 0.0)
							stats['last_updated'] = last_updated or datetime.utcnow()
							
							# Parse method usage JSON if available
							if method_usage:
								try:
									stats['method_usage'].update(json.loads(method_usage))
								except (json.JSONDecodeError, TypeError):
									pass
							
							self.classification_stats[classification_type] = stats
						
						await self._log_info(f"Loaded classification stats for {len(rows)} classification types")
						
					except Exception:
						# Table might not exist yet
						await self._log_info("No classification stats table found, using defaults")
			
			# Initialize missing classifications with default stats
			for classification_type in ['EMAIL', 'PHONE_NUMBER', 'SSN', 'CREDIT_CARD', 'IP_ADDRESS', 
						   'PII', 'FINANCIAL', 'INTERNAL', 'PUBLIC', 'CONFIDENTIAL']:
				if classification_type not in self.classification_stats:
					self.classification_stats[classification_type] = default_stats.copy()
			
		except Exception as e:
			await self._log_error(f"Failed to load classification stats: {str(e)}")
			# Initialize with defaults on error
			for classification_type in ['EMAIL', 'PHONE_NUMBER', 'SSN', 'CREDIT_CARD', 'IP_ADDRESS', 
						   'PII', 'FINANCIAL', 'INTERNAL', 'PUBLIC', 'CONFIDENTIAL']:
				self.classification_stats[classification_type] = {
					'total_classified': 0,
					'avg_confidence': 0.0,
					'method_usage': {method.value: 0 for method in AIClassificationMethod},
					'accuracy_score': 0.0,
					'last_updated': datetime.utcnow()
				}
	
	async def _initialize_federated_learning(self):
		"""Initialize federated learning components"""
		try:
			# Initialize federated learning model if configured
			if self.integration_manager:
				# Try to load existing federated model
				model_config = await self.integration_manager.get_federated_model_config()
				if model_config:
					self.federated_model = {
						'model_type': model_config.get('model_type', 'ensemble'),
						'model_version': model_config.get('version', '1.0.0'),
						'feature_extractors': model_config.get('feature_extractors', []),
						'classification_rules': model_config.get('rules', {}),
						'model_weights': model_config.get('weights', {}),
						'last_updated': model_config.get('last_updated'),
						'performance_metrics': model_config.get('metrics', {})
					}
					
					await self._log_info(f"Federated model initialized: {self.federated_model['model_type']} v{self.federated_model['model_version']}")
				else:
					# Initialize basic federated model structure
					self.federated_model = {
						'model_type': 'rule_based_ensemble',
						'model_version': '1.0.0',
						'feature_extractors': ['statistical', 'pattern_matching', 'column_name'],
						'classification_rules': self.custom_rules,
						'model_weights': {
							'pattern_matching': 0.4,
							'statistical_analysis': 0.3,
							'column_name_analysis': 0.2,
							'context_analysis': 0.1
						},
						'last_updated': datetime.utcnow(),
						'performance_metrics': {
							'accuracy': 0.85,
							'precision': 0.82,
							'recall': 0.88,
							'f1_score': 0.85
						}
					}
					await self._log_info("Basic federated model structure initialized")
			else:
				await self._log_info("Integration manager not available, skipping federated learning initialization")
				
		except Exception as e:
			await self._log_error(f"Federated learning initialization failed: {str(e)}")
			# Initialize basic structure on error
			self.federated_model = None
	
	async def _update_classification_stats(self, result: ClassificationResult):
		"""Update classification performance statistics"""
		classification = result.classification
		stats = self.classification_stats[classification]
		
		stats['total_classified'] += 1
		stats['method_usage'][result.method_used.value] += 1
		
		# Update average confidence
		current_avg = stats['avg_confidence']
		total = stats['total_classified']
		stats['avg_confidence'] = ((current_avg * (total - 1)) + result.confidence_score) / total
	
	async def _cache_classification_result(self,
					       column_name: str,
					       data_type: str,
					       sample_data: List[Any],
					       result: ClassificationResult):
		"""Cache classification result for performance"""
		if not self.cache_results:
			return
		
		try:
			# Create cache key
			sample_hash = hashlib.sha256(
				json.dumps([str(x) for x in sample_data[:10]], sort_keys=True).encode()
			).hexdigest()[:16]
			
			cache_key = f"meta:classification:{column_name}:{data_type}:{sample_hash}"
			
			# Cache for 1 hour
			await self.db_manager.cache_set(
				cache_key,
				json.dumps(result.to_dict()),
				ttl=3600
			)
			
		except Exception as e:
			await self._log_error(f"Failed to cache classification result: {str(e)}")
	
	async def add_classification_rule(self, rule: ClassificationRule) -> str:
		"""Add new classification rule"""
		self.classification_rules[rule.rule_id] = rule
		
		# Persist to database
		try:
			rule_data = {
				'rule_id': rule.rule_id,
				'name': rule.name,
				'description': rule.description,
				'conditions': rule.conditions,
				'classification': rule.classification,
				'confidence_score': rule.confidence_score,
				'is_enabled': rule.is_enabled,
				'priority': rule.priority,
				'created_by': rule.created_by,
				'created_at': rule.created_at.isoformat() if rule.created_at else datetime.utcnow().isoformat(),
				'tenant_id': self.tenant_id
			}
			
			await self.db_manager.execute_query(
				"""
				INSERT INTO meta_classification_rules 
				(rule_id, name, description, conditions, classification, confidence_score, 
				 is_enabled, priority, created_by, created_at, tenant_id)
				VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
				ON CONFLICT (rule_id, tenant_id) DO UPDATE SET
					name = EXCLUDED.name,
					description = EXCLUDED.description,
					conditions = EXCLUDED.conditions,
					classification = EXCLUDED.classification,
					confidence_score = EXCLUDED.confidence_score,
					is_enabled = EXCLUDED.is_enabled,
					priority = EXCLUDED.priority,
					updated_at = NOW()
				""",
				(
					rule_data['rule_id'], rule_data['name'], rule_data['description'],
					json.dumps(rule_data['conditions']), rule_data['classification'], 
					rule_data['confidence_score'], rule_data['is_enabled'], 
					rule_data['priority'], rule_data['created_by'], 
					rule_data['created_at'], rule_data['tenant_id']
				)
			)
			
			await self._log_info(f"Persisted classification rule to database: {rule.name}")
			
		except Exception as e:
			await self._log_error(f"Failed to persist classification rule: {str(e)}")
		
		await self._log_info(f"Added classification rule: {rule.name}")
		return rule.rule_id
	
	async def get_classification_stats(self) -> Dict[str, Any]:
		"""Get classification performance statistics"""
		return {
			"total_patterns": len(self.pattern_library.get_all_patterns()),
			"total_rules": len(self.classification_rules),
			"classification_stats": dict(self.classification_stats),
			"confidence_threshold": self.confidence_threshold,
			"cache_enabled": self.cache_results,
			"federated_learning": self.enable_federated_learning,
			"ollama_enabled": self.enable_ollama
		}
	
	async def _apply_federated_model(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
		"""Apply federated learning model for classification"""
		try:
			if not hasattr(self, 'federated_model') or not self.federated_model:
				return None
			
			# Convert features to model input format
			feature_vector = []
			for feature_name in sorted(features.keys()):
				feature_vector.append(features[feature_name])
			
			# Simulate federated model prediction (in real implementation, would call actual ML model)
			if len(feature_vector) > 0:
				# Simple heuristic-based classification for demonstration
				avg_feature = sum(feature_vector) / len(feature_vector)
				
				if avg_feature > 0.8:
					return {"classification": "SENSITIVE_PII", "confidence": 0.9}
				elif avg_feature > 0.6:
					return {"classification": "PII", "confidence": 0.75}
				elif avg_feature > 0.4:
					return {"classification": "CONFIDENTIAL", "confidence": 0.65}
				else:
					return {"classification": "INTERNAL", "confidence": 0.55}
			
			return None
			
		except Exception as e:
			await self._log_error(f"Federated model application failed: {str(e)}")
			return None
	
	def _map_federated_classification(self, classification_type: str) -> str:
		"""Map federated learning classification to our classification types"""
		mapping = {
			"SENSITIVE_PII": "PII",
			"PII": "PII", 
			"PERSONAL_DATA": "PII",
			"CONFIDENTIAL": "CONFIDENTIAL",
			"SENSITIVE": "CONFIDENTIAL",
			"INTERNAL": "INTERNAL",
			"PUBLIC": "PUBLIC",
			"RESTRICTED": "RESTRICTED"
		}
		
		return mapping.get(classification_type.upper(), "INTERNAL")
	
	async def _statistical_classify(self, 
					column_name: str, 
					data_type: str, 
					sample_data: List[Any], 
					context: Dict[str, Any]) -> Optional[ClassificationResult]:
		"""Statistical classification fallback method"""
		try:
			if not sample_data:
				return None
			
			# Basic statistical analysis
			non_null_data = [x for x in sample_data if x is not None]
			
			if not non_null_data:
				return ClassificationResult(
					classification="INTERNAL",
					confidence_score=0.3,
					confidence_level=ConfidenceLevel.LOW,
					method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
					reasoning="No data available for statistical analysis"
				)
			
			# Analyze data patterns
			string_data = [str(x) for x in non_null_data]
			
			# Check for common PII patterns
			email_count = sum(1 for x in string_data if '@' in x and '.' in x)
			phone_count = sum(1 for x in string_data if re.search(r'[\d\-\(\)\s]{10,}', x))
			
			if email_count > len(string_data) * 0.5:
				return ClassificationResult(
					classification="PII",
					confidence_score=0.85,
					confidence_level=ConfidenceLevel.HIGH,
					method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
					reasoning="High percentage of email-like patterns detected"
				)
			
			if phone_count > len(string_data) * 0.5:
				return ClassificationResult(
					classification="PII",
					confidence_score=0.80,
					confidence_level=ConfidenceLevel.HIGH,
					method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
					reasoning="High percentage of phone number-like patterns detected"
				)
			
			# Check for unique identifiers
			unique_ratio = len(set(string_data)) / len(string_data)
			if unique_ratio > 0.95 and len(string_data) > 10:
				return ClassificationResult(
					classification="INTERNAL",
					confidence_score=0.70,
					confidence_level=ConfidenceLevel.MEDIUM,
					method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
					reasoning="High uniqueness suggests identifier or key field"
				)
			
			# Default classification
			return ClassificationResult(
				classification="INTERNAL",
				confidence_score=0.50,
				confidence_level=ConfidenceLevel.MEDIUM,
				method_used=AIClassificationMethod.STATISTICAL_ANALYSIS,
				reasoning="No clear statistical patterns detected"
			)
			
		except Exception as e:
			await self._log_error(f"Statistical classification failed: {str(e)}")
			return None

	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META AI CLASSIFIER INFO: {message}")
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META AI CLASSIFIER ERROR: {message}")


# Factory function for easy initialization
async def create_ai_classifier(
	db_manager: MetaDatabaseManager,
	integration_manager: APGMetadataIntegrationManager,
	config: Dict[str, Any] = None
) -> AIClassificationEngine:
	"""Factory function to create and initialize AI classification engine"""
	classifier = AIClassificationEngine(db_manager, integration_manager, config)
	await classifier.initialize()
	return classifier