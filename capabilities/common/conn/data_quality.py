"""
APG Connection Management Data Quality Checks
Advanced data validation, profiling, and quality monitoring

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import re
import math

# Statistical analysis imports
try:
	import numpy as np
	import pandas as pd
	ADVANCED_STATS_AVAILABLE = True
except ImportError:
	ADVANCED_STATS_AVAILABLE = False
	logging.warning("NumPy/Pandas not available. Using basic statistics only.")

# Data validation imports
try:
	import jsonschema
	from cerberus import Validator
	VALIDATION_LIBS_AVAILABLE = True
except ImportError:
	VALIDATION_LIBS_AVAILABLE = False
	logging.warning("Advanced validation libraries not available.")

from .error_handling import APGError, ErrorContext
from .monitoring import global_metrics_collector, monitor_performance
from .performance import cached

logger = logging.getLogger(__name__)


class DataQualityLevel(str, Enum):
	"""Data quality assessment levels"""
	EXCELLENT = "excellent"
	GOOD = "good"
	FAIR = "fair"
	POOR = "poor"
	CRITICAL = "critical"


class DataQualityDimension(str, Enum):
	"""Data quality dimensions"""
	COMPLETENESS = "completeness"
	ACCURACY = "accuracy"
	CONSISTENCY = "consistency"
	VALIDITY = "validity"
	UNIQUENESS = "uniqueness"
	TIMELINESS = "timeliness"
	INTEGRITY = "integrity"


class IssueType(str, Enum):
	"""Data quality issue types"""
	MISSING_VALUE = "missing_value"
	INVALID_FORMAT = "invalid_format"
	OUTLIER = "outlier"
	DUPLICATE = "duplicate"
	INCONSISTENT = "inconsistent"
	STALE_DATA = "stale_data"
	CONSTRAINT_VIOLATION = "constraint_violation"
	SCHEMA_MISMATCH = "schema_mismatch"


class IssueSeverity(str, Enum):
	"""Issue severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class DataQualityIssue:
	"""Individual data quality issue"""
	issue_type: IssueType
	severity: IssueSeverity
	dimension: DataQualityDimension
	description: str
	field_name: str
	record_id: Optional[str] = None
	expected_value: Optional[Any] = None
	actual_value: Optional[Any] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DataQualityMetrics:
	"""Comprehensive data quality metrics"""
	total_records: int
	valid_records: int
	completeness_score: float
	accuracy_score: float
	consistency_score: float
	validity_score: float
	uniqueness_score: float
	timeliness_score: float
	integrity_score: float
	overall_score: float
	quality_level: DataQualityLevel
	issues: List[DataQualityIssue] = field(default_factory=list)
	profiling_stats: Dict[str, Any] = field(default_factory=dict)
	assessment_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FieldProfile:
	"""Statistical profile for a data field"""
	field_name: str
	data_type: str
	total_count: int
	null_count: int
	unique_count: int
	duplicate_count: int
	completeness_rate: float
	uniqueness_rate: float
	min_value: Optional[Any] = None
	max_value: Optional[Any] = None
	mean_value: Optional[float] = None
	median_value: Optional[float] = None
	std_deviation: Optional[float] = None
	quartiles: Optional[List[float]] = None
	top_values: List[Tuple[Any, int]] = field(default_factory=list)
	outliers: List[Any] = field(default_factory=list)
	patterns: List[str] = field(default_factory=list)


@dataclass
class DataQualityRule:
	"""Data quality validation rule"""
	rule_id: str
	name: str
	description: str
	dimension: DataQualityDimension
	severity: IssueSeverity
	field_names: List[str]
	rule_type: str
	parameters: Dict[str, Any] = field(default_factory=dict)
	is_active: bool = True
	created_by: str = "system"
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataProfiler:
	"""Advanced data profiling and statistical analysis"""

	def __init__(self):
		self.profile_cache = {}

	def profile_dataset(self, data: Union[List[Dict], Any],
						dataset_name: str = "default") -> Dict[str, FieldProfile]:
		"""Generate comprehensive statistical profiles for all fields"""
		try:
			if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
				return self._profile_dict_list(data)
			elif ADVANCED_STATS_AVAILABLE and hasattr(data, 'columns'):
				return self._profile_dataframe(data)
			else:
				logger.warning(f"Unsupported data format for profiling: {type(data)}")
				return {}
		except Exception as e:
			logger.error(f"Error profiling dataset {dataset_name}: {e}")
			return {}

	def _profile_dict_list(self, data: List[Dict]) -> Dict[str, FieldProfile]:
		"""Profile list of dictionaries"""
		profiles = {}
		all_fields = set()

		# Collect all field names
		for record in data:
			all_fields.update(record.keys())

		total_records = len(data)

		for field_name in all_fields:
			values = []
			null_count = 0

			# Extract values for this field
			for record in data:
				if field_name in record:
					value = record[field_name]
					if value is None or value == "" or value == "null":
						null_count += 1
					else:
						values.append(value)
				else:
					null_count += 1

			# Generate field profile
			profiles[field_name] = self._generate_field_profile(
				field_name, values, null_count, total_records
			)

		return profiles

	def _profile_dataframe(self, df) -> Dict[str, FieldProfile]:
		"""Profile pandas DataFrame"""
		profiles = {}

		for column in df.columns:
			series = df[column]
			values = series.dropna().tolist()
			null_count = series.isnull().sum()
			total_records = len(df)

			profiles[column] = self._generate_field_profile(
				column, values, null_count, total_records
			)

		return profiles

	def _generate_field_profile(self, field_name: str, values: List[Any],
							   null_count: int, total_records: int) -> FieldProfile:
		"""Generate statistical profile for a field"""
		if not values:
			return FieldProfile(
				field_name=field_name,
				data_type="unknown",
				total_count=total_records,
				null_count=null_count,
				unique_count=0,
				duplicate_count=0,
				completeness_rate=0.0,
				uniqueness_rate=0.0
			)

		# Basic counts and rates
		unique_values = list(set(values))
		unique_count = len(unique_values)
		duplicate_count = len(values) - unique_count
		completeness_rate = (total_records - null_count) / total_records
		uniqueness_rate = unique_count / len(values) if values else 0

		# Determine data type
		data_type = self._determine_data_type(values)

		# Statistical measures for numeric data
		min_val = max_val = mean_val = median_val = std_dev = None
		quartiles = None

		if data_type in ["int", "float"]:
			try:
				numeric_values = [float(v) for v in values if self._is_numeric(v)]
				if numeric_values:
					min_val = min(numeric_values)
					max_val = max(numeric_values)
					mean_val = statistics.mean(numeric_values)
					median_val = statistics.median(numeric_values)

					if len(numeric_values) > 1:
						std_dev = statistics.stdev(numeric_values)
						quartiles = self._calculate_quartiles(numeric_values)
			except Exception as e:
				logger.warning(f"Error calculating statistics for {field_name}: {e}")

		# Top values (most frequent)
		value_counts = Counter(values)
		top_values = value_counts.most_common(10)

		# Detect outliers
		outliers = self._detect_outliers(values, data_type)

		# Pattern analysis
		patterns = self._analyze_patterns(values, data_type)

		return FieldProfile(
			field_name=field_name,
			data_type=data_type,
			total_count=total_records,
			null_count=null_count,
			unique_count=unique_count,
			duplicate_count=duplicate_count,
			completeness_rate=completeness_rate,
			uniqueness_rate=uniqueness_rate,
			min_value=min_val,
			max_value=max_val,
			mean_value=mean_val,
			median_value=median_val,
			std_deviation=std_dev,
			quartiles=quartiles,
			top_values=top_values,
			outliers=outliers,
			patterns=patterns
		)

	def _determine_data_type(self, values: List[Any]) -> str:
		"""Determine the predominant data type"""
		type_counts = defaultdict(int)

		for value in values[:100]:  # Sample first 100 values
			if isinstance(value, bool):
				type_counts["bool"] += 1
			elif isinstance(value, int):
				type_counts["int"] += 1
			elif isinstance(value, float):
				type_counts["float"] += 1
			elif self._is_numeric(value):
				if '.' in str(value):
					type_counts["float"] += 1
				else:
					type_counts["int"] += 1
			elif self._is_datetime(value):
				type_counts["datetime"] += 1
			elif isinstance(value, str):
				type_counts["string"] += 1
			else:
				type_counts["unknown"] += 1

		return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"

	def _is_numeric(self, value: Any) -> bool:
		"""Check if value is numeric"""
		try:
			float(value)
			return True
		except (ValueError, TypeError):
			return False

	def _is_datetime(self, value: Any) -> bool:
		"""Check if value looks like a datetime"""
		if isinstance(value, datetime):
			return True

		if isinstance(value, str):
			# Common datetime patterns
			datetime_patterns = [
				r'\d{4}-\d{2}-\d{2}',
				r'\d{2}/\d{2}/\d{4}',
				r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
				r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
			]
			return any(re.match(pattern, value) for pattern in datetime_patterns)

		return False

	def _calculate_quartiles(self, values: List[float]) -> List[float]:
		"""Calculate quartiles for numeric data"""
		try:
			if ADVANCED_STATS_AVAILABLE:
				return np.percentile(values, [25, 50, 75]).tolist()
			else:
				sorted_values = sorted(values)
				n = len(sorted_values)
				return [
					sorted_values[n // 4],
					sorted_values[n // 2],
					sorted_values[3 * n // 4]
				]
		except Exception:
			return [0, 0, 0]

	def _detect_outliers(self, values: List[Any], data_type: str) -> List[Any]:
		"""Detect outliers using statistical methods"""
		outliers = []

		if data_type in ["int", "float"] and len(values) > 3:
			try:
				numeric_values = [float(v) for v in values if self._is_numeric(v)]
				if len(numeric_values) < 4:
					return outliers

				# Use IQR method
				if ADVANCED_STATS_AVAILABLE:
					q1, q3 = np.percentile(numeric_values, [25, 75])
				else:
					sorted_values = sorted(numeric_values)
					n = len(sorted_values)
					q1 = sorted_values[n // 4]
					q3 = sorted_values[3 * n // 4]

				iqr = q3 - q1
				lower_bound = q1 - 1.5 * iqr
				upper_bound = q3 + 1.5 * iqr

				outliers = [v for v in numeric_values if v < lower_bound or v > upper_bound]

			except Exception as e:
				logger.warning(f"Error detecting outliers: {e}")

		return outliers[:10]  # Return max 10 outliers

	def _analyze_patterns(self, values: List[Any], data_type: str) -> List[str]:
		"""Analyze common patterns in the data"""
		patterns = []

		if data_type == "string" and values:
			# Sample values for pattern analysis
			sample_values = values[:50]

			# Check for common patterns
			pattern_checks = [
				(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', 'email'),
				(r'^\+?1?\d{9,15}$', 'phone'),
				(r'^https?://', 'url'),
				(r'^\d{4}-\d{2}-\d{2}$', 'date_iso'),
				(r'^\d+$', 'numeric_string'),
				(r'^[A-Z]{2,3}$', 'country_code'),
				(r'^\d{5}(-\d{4})?$', 'zip_code'),
			]

			for pattern, name in pattern_checks:
				matches = sum(1 for v in sample_values if re.match(pattern, str(v)))
				if matches >= max(2, len(sample_values) * 0.6):
					patterns.append(name)

		return patterns


class DataQualityValidator:
	"""Advanced data quality validation engine"""

	def __init__(self):
		self.rules: Dict[str, DataQualityRule] = {}
		self.custom_validators = {}
		self._load_default_rules()

	def _load_default_rules(self):
		"""Load default data quality rules"""
		default_rules = [
			DataQualityRule(
				rule_id="completeness_check",
				name="Completeness Check",
				description="Check for missing values in required fields",
				dimension=DataQualityDimension.COMPLETENESS,
				severity=IssueSeverity.HIGH,
				field_names=["*"],
				rule_type="completeness",
				parameters={"required_fields": [], "max_null_rate": 0.1}
			),
			DataQualityRule(
				rule_id="email_format",
				name="Email Format Validation",
				description="Validate email address format",
				dimension=DataQualityDimension.VALIDITY,
				severity=IssueSeverity.MEDIUM,
				field_names=["email", "email_address", "contact_email"],
				rule_type="regex",
				parameters={"pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
			),
			DataQualityRule(
				rule_id="duplicate_check",
				name="Duplicate Record Check",
				description="Check for duplicate records",
				dimension=DataQualityDimension.UNIQUENESS,
				severity=IssueSeverity.MEDIUM,
				field_names=["id", "uuid", "key"],
				rule_type="uniqueness",
				parameters={"unique_fields": ["id"]}
			),
			DataQualityRule(
				rule_id="data_freshness",
				name="Data Freshness Check",
				description="Check if data is recent enough",
				dimension=DataQualityDimension.TIMELINESS,
				severity=IssueSeverity.LOW,
				field_names=["created_at", "updated_at", "timestamp"],
				rule_type="timeliness",
				parameters={"max_age_days": 30}
			),
		]

		for rule in default_rules:
			self.rules[rule.rule_id] = rule

	def add_rule(self, rule: DataQualityRule):
		"""Add custom data quality rule"""
		self.rules[rule.rule_id] = rule

	def remove_rule(self, rule_id: str):
		"""Remove data quality rule"""
		if rule_id in self.rules:
			del self.rules[rule_id]

	def validate_data(self, data: Union[List[Dict], Any],
					  rule_ids: Optional[List[str]] = None) -> List[DataQualityIssue]:
		"""Validate data against quality rules"""
		issues = []

		# Convert data to standard format
		if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
			records = data
		else:
			logger.warning("Unsupported data format for validation")
			return issues

		# Select rules to apply
		rules_to_apply = []
		if rule_ids:
			rules_to_apply = [self.rules[rid] for rid in rule_ids if rid in self.rules]
		else:
			rules_to_apply = [rule for rule in self.rules.values() if rule.is_active]

		# Apply each rule
		for rule in rules_to_apply:
			try:
				rule_issues = self._apply_rule(rule, records)
				issues.extend(rule_issues)
			except Exception as e:
				logger.error(f"Error applying rule {rule.rule_id}: {e}")

		return issues

	def _apply_rule(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Apply specific validation rule"""
		if rule.rule_type == "completeness":
			return self._check_completeness(rule, records)
		elif rule.rule_type == "regex":
			return self._check_regex_pattern(rule, records)
		elif rule.rule_type == "uniqueness":
			return self._check_uniqueness(rule, records)
		elif rule.rule_type == "timeliness":
			return self._check_timeliness(rule, records)
		elif rule.rule_type == "custom":
			return self._check_custom_rule(rule, records)
		else:
			logger.warning(f"Unknown rule type: {rule.rule_type}")
			return []

	def _check_completeness(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Check data completeness"""
		issues = []
		required_fields = rule.parameters.get("required_fields", rule.field_names)
		max_null_rate = rule.parameters.get("max_null_rate", 0.1)
		if not required_fields or required_fields == ["*"]:
			required_fields = sorted({field for record in records for field in record.keys()})

		for field_name in required_fields:
			null_count = 0
			total_count = len(records)

			for i, record in enumerate(records):
				value = record.get(field_name)
				if value is None or value == "" or value == "null":
					null_count += 1
					issues.append(DataQualityIssue(
						issue_type=IssueType.MISSING_VALUE,
						severity=rule.severity,
						dimension=rule.dimension,
						description=f"Missing value in required field '{field_name}'",
						field_name=field_name,
						record_id=str(i),
						expected_value="non-null",
						actual_value=value
					))

			# Check overall null rate
			null_rate = null_count / total_count if total_count > 0 else 0
			if null_rate > max_null_rate:
				issues.append(DataQualityIssue(
					issue_type=IssueType.MISSING_VALUE,
					severity=IssueSeverity.HIGH,
					dimension=rule.dimension,
					description=f"High null rate ({null_rate:.2%}) in field '{field_name}' exceeds threshold ({max_null_rate:.2%})",
					field_name=field_name,
					metadata={"null_rate": null_rate, "threshold": max_null_rate}
				))

		return issues

	def _check_regex_pattern(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Check regex pattern matching"""
		issues = []
		pattern = rule.parameters.get("pattern")

		if not pattern:
			return issues

		try:
			compiled_pattern = re.compile(pattern)
		except re.error as e:
			logger.error(f"Invalid regex pattern in rule {rule.rule_id}: {e}")
			return issues

		for field_name in rule.field_names:
			for i, record in enumerate(records):
				if field_name in record:
					value = record[field_name]
					if value is not None and not compiled_pattern.match(str(value)):
						issues.append(DataQualityIssue(
							issue_type=IssueType.INVALID_FORMAT,
							severity=rule.severity,
							dimension=rule.dimension,
							description=f"Value '{value}' in field '{field_name}' does not match pattern '{pattern}'",
							field_name=field_name,
							record_id=str(i),
							actual_value=value,
							metadata={"pattern": pattern}
						))

		return issues

	def _check_uniqueness(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Check data uniqueness"""
		issues = []
		unique_fields = rule.parameters.get("unique_fields", rule.field_names)

		for field_name in unique_fields:
			seen_values = {}

			for i, record in enumerate(records):
				if field_name in record:
					value = record[field_name]
					if value is not None:
						if value in seen_values:
							issues.append(DataQualityIssue(
								issue_type=IssueType.DUPLICATE,
								severity=rule.severity,
								dimension=rule.dimension,
								description=f"Duplicate value '{value}' found in field '{field_name}'",
								field_name=field_name,
								record_id=str(i),
								actual_value=value,
								metadata={"first_occurrence": seen_values[value]}
							))
						else:
							seen_values[value] = i

		return issues

	def _check_timeliness(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Check data timeliness"""
		issues = []
		max_age_days = rule.parameters.get("max_age_days", 30)
		cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

		for field_name in rule.field_names:
			for i, record in enumerate(records):
				if field_name in record:
					value = record[field_name]
					if value is not None:
						try:
							if isinstance(value, str):
								date_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
							elif isinstance(value, datetime):
								date_value = value
							else:
								continue

							if date_value < cutoff_date:
								issues.append(DataQualityIssue(
									issue_type=IssueType.STALE_DATA,
									severity=rule.severity,
									dimension=rule.dimension,
									description=f"Stale data in field '{field_name}': {date_value} is older than {max_age_days} days",
									field_name=field_name,
									record_id=str(i),
									actual_value=value,
									metadata={"age_days": (datetime.now(timezone.utc) - date_value).days}
								))
						except (ValueError, TypeError):
							continue

		return issues

	def _check_custom_rule(self, rule: DataQualityRule, records: List[Dict]) -> List[DataQualityIssue]:
		"""Apply custom validation rule"""
		if rule.rule_id in self.custom_validators:
			try:
				validator_func = self.custom_validators[rule.rule_id]
				return validator_func(rule, records)
			except Exception as e:
				logger.error(f"Error in custom validator {rule.rule_id}: {e}")

		return []


class DataQualityAssessment:
	"""Comprehensive data quality assessment engine"""

	def __init__(self):
		self.profiler = DataProfiler()
		self.validator = DataQualityValidator()

	@monitor_performance("data_quality_assessment")
	async def assess_data_quality(self, data: Union[List[Dict], Any],
								  dataset_name: str = "default",
								  custom_rules: Optional[List[DataQualityRule]] = None) -> DataQualityMetrics:
		"""Perform comprehensive data quality assessment"""

		try:
			# Add custom rules if provided
			if custom_rules:
				for rule in custom_rules:
					self.validator.add_rule(rule)

			# Convert data to standard format
			if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
				records = data
			else:
				raise APGError(
					message=f"Unsupported data format: {type(data)}",
					context=ErrorContext(tenant_id="system", operation="assess_data_quality")
				)

			total_records = len(records)

			# Generate data profiles
			field_profiles = self.profiler.profile_dataset(data, dataset_name)

			# Validate data quality
			issues = self.validator.validate_data(records)

			# Calculate quality scores
			quality_scores = self._calculate_quality_scores(records, field_profiles, issues)

			# Determine overall quality level
			overall_score = self._calculate_overall_score(quality_scores)
			quality_level = self._determine_quality_level(overall_score)

			# Count valid records (records without critical issues)
			critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
			records_with_critical_issues = len(set(i.record_id for i in critical_issues if i.record_id))
			valid_records = total_records - records_with_critical_issues

			return DataQualityMetrics(
				total_records=total_records,
				valid_records=valid_records,
				completeness_score=quality_scores["completeness"],
				accuracy_score=quality_scores["accuracy"],
				consistency_score=quality_scores["consistency"],
				validity_score=quality_scores["validity"],
				uniqueness_score=quality_scores["uniqueness"],
				timeliness_score=quality_scores["timeliness"],
				integrity_score=quality_scores["integrity"],
				overall_score=overall_score,
				quality_level=quality_level,
				issues=issues,
				profiling_stats={
					"field_count": len(field_profiles),
					"profiles": {name: self._profile_to_dict(profile)
							   for name, profile in field_profiles.items()}
				}
			)

		except Exception as e:
			logger.error(f"Error in data quality assessment: {e}")
			raise APGError(
				message=f"Data quality assessment failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="assess_data_quality"),
				cause=e
			)

	def _calculate_quality_scores(self, records: List[Dict],
								  profiles: Dict[str, FieldProfile],
								  issues: List[DataQualityIssue]) -> Dict[str, float]:
		"""Calculate quality scores by dimension"""

		total_records = len(records)
		if total_records == 0:
			return {dim.value: 0.0 for dim in DataQualityDimension}

		# Group issues by dimension
		issues_by_dimension = defaultdict(list)
		for issue in issues:
			issues_by_dimension[issue.dimension].append(issue)

		scores = {}

		# Completeness score
		completeness_issues = len(issues_by_dimension[DataQualityDimension.COMPLETENESS])
		completeness_score = max(0, 1 - (completeness_issues / (total_records * len(profiles))))
		scores["completeness"] = completeness_score * 100

		# Validity score
		validity_issues = len(issues_by_dimension[DataQualityDimension.VALIDITY])
		validity_score = max(0, 1 - (validity_issues / total_records))
		scores["validity"] = validity_score * 100

		# Uniqueness score
		uniqueness_issues = len(issues_by_dimension[DataQualityDimension.UNIQUENESS])
		uniqueness_score = max(0, 1 - (uniqueness_issues / total_records))
		scores["uniqueness"] = uniqueness_score * 100

		# Timeliness score
		timeliness_issues = len(issues_by_dimension[DataQualityDimension.TIMELINESS])
		timeliness_score = max(0, 1 - (timeliness_issues / total_records))
		scores["timeliness"] = timeliness_score * 100

		# Accuracy score (based on outliers and format issues)
		accuracy_issues = len([i for i in issues if i.issue_type in [IssueType.OUTLIER, IssueType.INVALID_FORMAT]])
		accuracy_score = max(0, 1 - (accuracy_issues / total_records))
		scores["accuracy"] = accuracy_score * 100

		# Consistency score (based on inconsistent values)
		consistency_issues = len(issues_by_dimension[DataQualityDimension.CONSISTENCY])
		consistency_score = max(0, 1 - (consistency_issues / total_records))
		scores["consistency"] = consistency_score * 100

		# Integrity score (based on constraint violations)
		integrity_issues = len(issues_by_dimension[DataQualityDimension.INTEGRITY])
		integrity_score = max(0, 1 - (integrity_issues / total_records))
		scores["integrity"] = integrity_score * 100

		return scores

	def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
		"""Calculate weighted overall quality score"""
		weights = {
			"completeness": 0.20,
			"accuracy": 0.18,
			"consistency": 0.15,
			"validity": 0.17,
			"uniqueness": 0.12,
			"timeliness": 0.10,
			"integrity": 0.08
		}

		overall_score = sum(scores.get(dim, 0) * weight for dim, weight in weights.items())
		return round(overall_score, 2)

	def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
		"""Determine quality level based on overall score"""
		if overall_score >= 90:
			return DataQualityLevel.EXCELLENT
		elif overall_score >= 80:
			return DataQualityLevel.GOOD
		elif overall_score >= 60:
			return DataQualityLevel.FAIR
		elif overall_score >= 30:
			return DataQualityLevel.POOR
		else:
			return DataQualityLevel.CRITICAL

	def _profile_to_dict(self, profile: FieldProfile) -> Dict[str, Any]:
		"""Convert field profile to dictionary"""
		return {
			"data_type": profile.data_type,
			"completeness_rate": profile.completeness_rate,
			"uniqueness_rate": profile.uniqueness_rate,
			"null_count": profile.null_count,
			"unique_count": profile.unique_count,
			"min_value": profile.min_value,
			"max_value": profile.max_value,
			"mean_value": profile.mean_value,
			"std_deviation": profile.std_deviation,
			"outliers_count": len(profile.outliers),
			"top_values": profile.top_values[:5],
			"patterns": profile.patterns
		}


class BoundedHistory(list):
	"""List with an enforced maximum length for quality history."""

	def __init__(self, maxlen: int = 100):
		super().__init__()
		self.maxlen = maxlen

	def _trim(self) -> None:
		if len(self) > self.maxlen:
			del self[:-self.maxlen]

	def append(self, item: Any) -> None:
		super().append(item)
		self._trim()

	def extend(self, items) -> None:
		super().extend(items)
		self._trim()


class DataQualityMonitor:
	"""Continuous data quality monitoring"""

	def __init__(self):
		self.assessment_engine = DataQualityAssessment()
		self.quality_history: List[DataQualityMetrics] = BoundedHistory(maxlen=100)
		self.alert_thresholds = {
			"overall_score": 70,
			"completeness_score": 80,
			"validity_score": 85,
			"critical_issues": 5
		}

	async def monitor_data_quality(self, connection_id: str, data: Union[List[Dict], Any],
								   alert_callback: Optional[callable] = None) -> DataQualityMetrics:
		"""Monitor data quality with alerting"""

		# Perform assessment
		metrics = await self.assessment_engine.assess_data_quality(
			data, dataset_name=f"connection_{connection_id}"
		)

		# Store in history
		self.quality_history.append(metrics)

		# Keep only last 100 assessments
		if len(self.quality_history) > 100:
			self.quality_history = self.quality_history[-100:]

		# Check for quality alerts
		alerts = self._check_quality_alerts(metrics)

		# Trigger alerts if callback provided
		if alert_callback and alerts:
			try:
				await alert_callback(connection_id, metrics, alerts)
			except Exception as e:
				logger.error(f"Error triggering quality alerts: {e}")

		# Update metrics through the local collector API shape.
		if hasattr(global_metrics_collector, "increment_counter"):
			global_metrics_collector.increment_counter(
				"data_quality_assessments_total",
				tags={"quality_level": metrics.quality_level.value}
			)
		elif hasattr(global_metrics_collector, "record_counter"):
			global_metrics_collector.record_counter(
				"data_quality_assessments_total",
				1,
				{"quality_level": metrics.quality_level.value}
			)

		global_metrics_collector.record_gauge(
			"data_quality_score",
			metrics.overall_score,
			{"connection_id": connection_id}
		)

		return metrics

	def _check_quality_alerts(self, metrics: DataQualityMetrics) -> List[Dict[str, Any]]:
		"""Check for quality threshold violations"""
		alerts = []

		# Overall score alert
		if metrics.overall_score < self.alert_thresholds["overall_score"]:
			alerts.append({
				"type": "low_overall_quality",
				"severity": "high",
				"message": f"Overall data quality score ({metrics.overall_score:.1f}%) below threshold ({self.alert_thresholds['overall_score']}%)",
				"current_value": metrics.overall_score,
				"threshold": self.alert_thresholds["overall_score"]
			})

		# Completeness alert
		if metrics.completeness_score < self.alert_thresholds["completeness_score"]:
			alerts.append({
				"type": "low_completeness",
				"severity": "medium",
				"message": f"Data completeness ({metrics.completeness_score:.1f}%) below threshold ({self.alert_thresholds['completeness_score']}%)",
				"current_value": metrics.completeness_score,
				"threshold": self.alert_thresholds["completeness_score"]
			})

		# Critical issues alert
		critical_issues = len([i for i in metrics.issues if i.severity == IssueSeverity.CRITICAL])
		if critical_issues >= self.alert_thresholds["critical_issues"]:
			alerts.append({
				"type": "critical_issues",
				"severity": "critical",
				"message": f"Found {critical_issues} critical data quality issues (threshold: {self.alert_thresholds['critical_issues']})",
				"current_value": critical_issues,
				"threshold": self.alert_thresholds["critical_issues"]
			})

		return alerts

	def get_quality_trends(self, lookback_hours: int = 24) -> Dict[str, Any]:
		"""Get data quality trends over time"""
		cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

		recent_metrics = [
			m for m in self.quality_history
			if m.assessment_timestamp >= cutoff_time
		]

		if not recent_metrics:
			return {"error": "No recent quality metrics available"}

		# Calculate trends
		scores = [m.overall_score for m in recent_metrics]
		completeness_scores = [m.completeness_score for m in recent_metrics]
		issue_counts = [len(m.issues) for m in recent_metrics]

		return {
			"period_hours": lookback_hours,
			"assessments_count": len(recent_metrics),
			"overall_score": {
				"current": scores[-1] if scores else 0,
				"average": statistics.mean(scores) if scores else 0,
				"min": min(scores) if scores else 0,
				"max": max(scores) if scores else 0,
				"trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable"
			},
			"completeness_trend": {
				"current": completeness_scores[-1] if completeness_scores else 0,
				"average": statistics.mean(completeness_scores) if completeness_scores else 0
			},
			"issues_trend": {
				"current": issue_counts[-1] if issue_counts else 0,
				"average": statistics.mean(issue_counts) if issue_counts else 0
			}
		}


# Global data quality monitor
global_data_quality_monitor = DataQualityMonitor()


# Convenience functions
@cached(ttl=300)
async def assess_connection_data_quality(connection_id: str,
										 data: Union[List[Dict], Any]) -> DataQualityMetrics:
	"""Assess data quality for connection data"""
	return await global_data_quality_monitor.monitor_data_quality(connection_id, data)


async def get_data_quality_report(connection_id: str) -> Dict[str, Any]:
	"""Get comprehensive data quality report"""
	trends = global_data_quality_monitor.get_quality_trends()

	return {
		"connection_id": connection_id,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"quality_trends": trends,
		"alert_thresholds": global_data_quality_monitor.alert_thresholds,
		"assessment_history_count": len(global_data_quality_monitor.quality_history)
	}
