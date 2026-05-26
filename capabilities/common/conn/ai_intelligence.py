"""
APG Connection Management - AI Intelligence Engine

AI-powered schema detection, intelligent data mapping, and predictive analytics
for automatic data integration and optimization.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

@dataclass
class SchemaAnalyzer:
	"""
	AI-powered schema analyzer for automatic field detection,
	type inference, and relationship discovery.
	"""

	# ML Models (placeholders for production models)
	field_classifier_model: Optional[Any] = None
	type_detector_model: Optional[Any] = None
	relationship_model: Optional[Any] = None

	# Learning Data
	schema_patterns: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
	field_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

	def __post_init__(self):
		"""Initialize with common field patterns."""
		self._initialize_field_patterns()

	def _log_ai_operation(self, operation: str) -> None:
		"""Log AI operations following APG patterns."""
		print(f"Schema analyzer: {operation}")

	def _initialize_field_patterns(self) -> None:
		"""Initialize common field patterns for recognition."""
		self.field_patterns = {
			"id_fields": {
				"patterns": [r".*id$", r".*_id$", r"^id", r"identifier", r"uuid"],
				"type": "string",
				"key_field": True,
				"confidence_boost": 0.9
			},
			"email_fields": {
				"patterns": [r"email", r"e_mail", r"mail"],
				"type": "string",
				"format": "email",
				"validation": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
			},
			"date_fields": {
				"patterns": [r".*date.*", r".*_at$", r".*_on$", r"created", r"updated", r"timestamp"],
				"type": "string",
				"format": "date-time",
				"confidence_boost": 0.8
			},
			"name_fields": {
				"patterns": [r".*name.*", r"title", r"label", r"description"],
				"type": "string",
				"searchable": True
			},
			"numeric_fields": {
				"patterns": [r"count", r"amount", r"price", r"total", r"sum", r"quantity"],
				"type": "number",
				"aggregatable": True
			},
			"phone_fields": {
				"patterns": [r"phone", r"mobile", r"tel"],
				"type": "string",
				"format": "phone"
			},
			"address_fields": {
				"patterns": [r"address", r"street", r"city", r"state", r"zip", r"postal"],
				"type": "string",
				"location": True
			}
		}

	async def analyze_sample_data(
		self,
		sample_records: List[Dict[str, Any]],
		source_name: str = "unknown"
	) -> Dict[str, Any]:
		"""Analyze sample data to infer comprehensive schema."""
		assert sample_records, "Sample records are required"

		self._log_ai_operation(f"Analyzing {len(sample_records)} sample records")

		field_analysis = {}
		schema_insights = {
			"record_count": len(sample_records),
			"field_count": 0,
			"data_quality_score": 0.0,
			"suggested_keys": [],
			"relationships": [],
			"confidence_score": 0.0
		}

		# Analyze each field
		all_fields = set()
		for record in sample_records:
			all_fields.update(record.keys())

		for field_name in all_fields:
			analysis = await self._analyze_field(field_name, sample_records)
			field_analysis[field_name] = analysis

		schema_insights["field_count"] = len(field_analysis)

		# Generate JSON schema
		json_schema = await self._generate_json_schema(field_analysis)

		# Detect relationships
		relationships = await self._detect_relationships(field_analysis, sample_records)
		schema_insights["relationships"] = relationships

		# Calculate confidence scores
		confidence_scores = [field["confidence"] for field in field_analysis.values()]
		schema_insights["confidence_score"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

		# Suggest primary keys
		schema_insights["suggested_keys"] = await self._suggest_primary_keys(field_analysis, sample_records)

		# Data quality assessment
		schema_insights["data_quality_score"] = await self._assess_data_quality(field_analysis, sample_records)

		return {
			"source_name": source_name,
			"analysis_timestamp": datetime.now(timezone.utc).isoformat(),
			"schema_insights": schema_insights,
			"field_analysis": field_analysis,
			"json_schema": json_schema,
			"recommendations": await self._generate_schema_recommendations(field_analysis)
		}

	async def _analyze_field(
		self,
		field_name: str,
		sample_records: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Analyze individual field characteristics."""
		values = []
		null_count = 0
		total_count = 0

		for record in sample_records:
			total_count += 1
			if field_name in record:
				value = record[field_name]
				if value is None or value == "":
					null_count += 1
				else:
					values.append(value)

		if not values:
			return {
				"field_name": field_name,
				"type": "unknown",
				"nullable": True,
				"confidence": 0.0,
				"null_percentage": 100.0
			}

		# Type detection
		type_analysis = await self._detect_field_type(field_name, values)

		# Pattern matching
		pattern_analysis = await self._match_field_patterns(field_name, values)

		# Statistical analysis
		stats = await self._calculate_field_statistics(values)

		# Combine analyses
		analysis = {
			"field_name": field_name,
			"type": type_analysis["type"],
			"format": type_analysis.get("format"),
			"nullable": null_count > 0,
			"null_percentage": (null_count / total_count) * 100,
			"unique_values": len(set(str(v) for v in values)),
			"sample_values": values[:5],
			"confidence": type_analysis["confidence"],
			"patterns": pattern_analysis,
			"statistics": stats,
			"recommendations": []
		}

		# Add pattern-based enhancements
		if pattern_analysis["matched_pattern"]:
			pattern_info = self.field_patterns[pattern_analysis["matched_pattern"]]
			analysis.update({
				"semantic_type": pattern_analysis["matched_pattern"],
				"validation_regex": pattern_info.get("validation"),
				"searchable": pattern_info.get("searchable", False),
				"key_field": pattern_info.get("key_field", False)
			})

			# Boost confidence for pattern matches
			analysis["confidence"] = min(1.0, analysis["confidence"] + pattern_info.get("confidence_boost", 0.0))

		return analysis

	async def _detect_field_type(
		self,
		field_name: str,
		values: List[Any]
	) -> Dict[str, Any]:
		"""Detect field data type using multiple heuristics."""
		if not values:
			return {"type": "unknown", "confidence": 0.0}

		type_votes = {"string": 0, "integer": 0, "number": 0, "boolean": 0, "date": 0}

		for value in values:
			value_str = str(value).strip()

			# Boolean detection
			if value_str.lower() in ("true", "false", "yes", "no", "1", "0"):
				type_votes["boolean"] += 1
				continue

			# Integer detection
			try:
				int(value_str)
				type_votes["integer"] += 1
				continue
			except ValueError:
				pass

			# Float detection
			try:
				float(value_str)
				type_votes["number"] += 1
				continue
			except ValueError:
				pass

			# Date detection
			if self._is_date_string(value_str):
				type_votes["date"] += 1
				continue

			# Default to string
			type_votes["string"] += 1

		# Determine winning type
		total_votes = sum(type_votes.values())
		winning_type = max(type_votes.items(), key=lambda x: x[1])
		confidence = winning_type[1] / total_votes if total_votes > 0 else 0.0

		result = {
			"type": winning_type[0],
			"confidence": confidence,
			"type_distribution": type_votes
		}

		# Add format for dates
		if winning_type[0] == "date":
			result["format"] = "date-time"

		return result

	def _is_date_string(self, value: str) -> bool:
		"""Check if string represents a date."""
		date_patterns = [
			r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
			r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
			r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
		]

		return any(re.match(pattern, value) for pattern in date_patterns)

	async def _match_field_patterns(
		self,
		field_name: str,
		values: List[Any]
	) -> Dict[str, Any]:
		"""Match field against known patterns."""
		field_lower = field_name.lower()

		best_match = None
		best_score = 0.0

		for pattern_name, pattern_info in self.field_patterns.items():
			score = 0.0

			# Name-based matching
			for pattern in pattern_info["patterns"]:
				if re.search(pattern, field_lower, re.IGNORECASE):
					score += 0.8
					break

			# Value-based validation
			if pattern_info.get("validation") and values:
				valid_count = 0
				for value in values[:10]:  # Check first 10 values
					if re.match(pattern_info["validation"], str(value)):
						valid_count += 1

				if valid_count > len(values[:10]) * 0.7:  # 70% match threshold
					score += 0.6

			if score > best_score:
				best_score = score
				best_match = pattern_name

		return {
			"matched_pattern": best_match,
			"confidence": best_score,
			"all_scores": {name: 0.0 for name in self.field_patterns.keys()}  # Simplified
		}

	async def _calculate_field_statistics(self, values: List[Any]) -> Dict[str, Any]:
		"""Calculate statistical properties of field values."""
		stats = {
			"count": len(values),
			"unique_count": len(set(str(v) for v in values)),
			"uniqueness_ratio": 0.0
		}

		if stats["count"] > 0:
			stats["uniqueness_ratio"] = stats["unique_count"] / stats["count"]

		# For numeric fields
		numeric_values = []
		for value in values:
			try:
				numeric_values.append(float(value))
			except (ValueError, TypeError):
				continue

		if numeric_values:
			stats.update({
				"min": min(numeric_values),
				"max": max(numeric_values),
				"avg": sum(numeric_values) / len(numeric_values),
				"numeric_ratio": len(numeric_values) / len(values)
			})

		# For string fields
		string_values = [str(v) for v in values]
		if string_values:
			lengths = [len(s) for s in string_values]
			stats.update({
				"avg_length": sum(lengths) / len(lengths),
				"min_length": min(lengths),
				"max_length": max(lengths)
			})

		return stats

	async def _detect_relationships(
		self,
		field_analysis: Dict[str, Dict[str, Any]],
		sample_records: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Detect relationships between fields."""
		relationships = []

		# Look for foreign key patterns
		id_fields = [
			name for name, analysis in field_analysis.items()
			if analysis.get("key_field", False) or "id" in name.lower()
		]

		for field_name, analysis in field_analysis.items():
			# Skip if this is already identified as an ID field
			if analysis.get("key_field"):
				continue

			# Look for potential foreign key references
			if field_name.endswith("_id") and field_name != "id":
				referenced_entity = field_name.replace("_id", "")
				relationships.append({
					"type": "foreign_key",
					"source_field": field_name,
					"target_entity": referenced_entity,
					"confidence": 0.8
				})

		# Detect composition relationships (nested objects)
		for record in sample_records[:5]:  # Check first few records
			for field_name, value in record.items():
				if isinstance(value, dict):
					relationships.append({
						"type": "composition",
						"source_field": field_name,
						"nested_fields": list(value.keys()),
						"confidence": 1.0
					})

		return relationships

	async def _suggest_primary_keys(
		self,
		field_analysis: Dict[str, Dict[str, Any]],
		sample_records: List[Dict[str, Any]]
	) -> List[str]:
		"""Suggest potential primary key fields."""
		candidates = []

		for field_name, analysis in field_analysis.items():
			score = 0.0

			# High uniqueness
			if analysis.get("statistics", {}).get("uniqueness_ratio", 0) > 0.95:
				score += 0.5

			# Identified as key field by pattern
			if analysis.get("key_field"):
				score += 0.4

			# Name suggests it's an ID
			if "id" in field_name.lower():
				score += 0.3

			# Non-nullable
			if analysis.get("null_percentage", 100) < 5:
				score += 0.2

			if score > 0.7:
				candidates.append((field_name, score))

		# Sort by score and return field names
		candidates.sort(key=lambda x: x[1], reverse=True)
		return [field_name for field_name, score in candidates[:3]]

	async def _assess_data_quality(
		self,
		field_analysis: Dict[str, Dict[str, Any]],
		sample_records: List[Dict[str, Any]]
	) -> float:
		"""Assess overall data quality score."""
		if not field_analysis:
			return 0.0

		quality_factors = []

		# Completeness (low null percentages)
		null_percentages = [analysis.get("null_percentage", 0) for analysis in field_analysis.values()]
		avg_null = sum(null_percentages) / len(null_percentages)
		completeness_score = max(0, 1 - (avg_null / 100))
		quality_factors.append(completeness_score)

		# Consistency (high confidence in type detection)
		confidences = [analysis.get("confidence", 0) for analysis in field_analysis.values()]
		avg_confidence = sum(confidences) / len(confidences)
		quality_factors.append(avg_confidence)

		# Validity (fields match expected patterns)
		pattern_matches = sum(1 for analysis in field_analysis.values() if analysis.get("patterns", {}).get("matched_pattern"))
		validity_score = pattern_matches / len(field_analysis)
		quality_factors.append(validity_score)

		# Overall score
		return sum(quality_factors) / len(quality_factors)

	async def _generate_json_schema(
		self,
		field_analysis: Dict[str, Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Generate JSON schema from field analysis."""
		schema = {
			"type": "object",
			"properties": {},
			"required": []
		}

		for field_name, analysis in field_analysis.items():
			field_schema = {"type": analysis["type"]}

			# Add format if available
			if analysis.get("format"):
				field_schema["format"] = analysis["format"]

			# Add description
			field_schema["description"] = f"Field: {field_name}"
			if analysis.get("semantic_type"):
				field_schema["description"] += f" ({analysis['semantic_type']})"

			schema["properties"][field_name] = field_schema

			# Add to required if low null percentage
			if analysis.get("null_percentage", 100) < 10:
				schema["required"].append(field_name)

		return schema

	async def _generate_schema_recommendations(
		self,
		field_analysis: Dict[str, Dict[str, Any]]
	) -> List[str]:
		"""Generate recommendations for schema optimization."""
		recommendations = []

		# Check for potential indexing opportunities
		high_unique_fields = [
			name for name, analysis in field_analysis.items()
			if analysis.get("statistics", {}).get("uniqueness_ratio", 0) > 0.8
		]

		if high_unique_fields:
			recommendations.append(f"Consider indexing high-uniqueness fields: {', '.join(high_unique_fields)}")

		# Check for data validation opportunities
		pattern_fields = [
			name for name, analysis in field_analysis.items()
			if analysis.get("patterns", {}).get("matched_pattern")
		]

		if pattern_fields:
			recommendations.append(f"Add validation rules for structured fields: {', '.join(pattern_fields)}")

		# Check for normalization opportunities
		low_unique_fields = [
			name for name, analysis in field_analysis.items()
			if analysis.get("statistics", {}).get("uniqueness_ratio", 1) < 0.1
		]

		if low_unique_fields:
			recommendations.append(f"Consider normalizing low-uniqueness fields: {', '.join(low_unique_fields)}")

		return recommendations

@dataclass
class IntelligentMapper:
	"""
	AI-powered data mapping engine for automatic field mapping
	and transformation suggestions between schemas.
	"""

	# Learning Models
	mapping_model: Optional[Any] = None
	similarity_model: Optional[Any] = None

	# Historical Data
	mapping_history: List[Dict[str, Any]] = field(default_factory=list)
	successful_mappings: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

	def _log_mapping_operation(self, operation: str) -> None:
		"""Log mapping operations following APG patterns."""
		print(f"Intelligent mapper: {operation}")

	async def suggest_field_mappings(
		self,
		source_schema: Dict[str, Any],
		target_schema: Dict[str, Any],
		source_sample_data: Optional[List[Dict[str, Any]]] = None,
		context: Optional[Dict[str, Any]] = None
	) -> List[Dict[str, Any]]:
		"""Generate intelligent field mapping suggestions."""
		assert source_schema, "Source schema is required"
		assert target_schema, "Target schema is required"

		self._log_mapping_operation("Generating intelligent field mappings")

		source_fields = source_schema.get("properties", {})
		target_fields = target_schema.get("properties", {})

		mapping_suggestions = []

		for target_field, target_props in target_fields.items():
			best_matches = await self._find_best_source_matches(
				target_field,
				target_props,
				source_fields,
				source_sample_data,
				context
			)

			for match in best_matches:
				mapping_suggestions.append({
					"source_field": match["source_field"],
					"target_field": target_field,
					"confidence": match["confidence"],
					"mapping_type": match["mapping_type"],
					"transformation_required": match["transformation_required"],
					"transformation_suggestion": match.get("transformation_suggestion"),
					"reasoning": match["reasoning"]
				})

		# Sort by confidence
		mapping_suggestions.sort(key=lambda x: x["confidence"], reverse=True)

		# Learn from this mapping session
		await self._record_mapping_session(source_schema, target_schema, mapping_suggestions)

		return mapping_suggestions

	async def _find_best_source_matches(
		self,
		target_field: str,
		target_props: Dict[str, Any],
		source_fields: Dict[str, Dict[str, Any]],
		source_sample_data: Optional[List[Dict[str, Any]]],
		context: Optional[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Find best source field matches for a target field."""
		matches = []

		for source_field, source_props in source_fields.items():
			similarity_score = await self._calculate_field_similarity(
				source_field,
				source_props,
				target_field,
				target_props,
				source_sample_data
			)

			if similarity_score > 0.3:  # Minimum threshold
				match = {
					"source_field": source_field,
					"confidence": similarity_score,
					"mapping_type": "direct",
					"transformation_required": False,
					"reasoning": []
				}

				# Determine if transformation is needed
				transformation_info = await self._analyze_transformation_needs(
					source_props,
					target_props,
					source_sample_data,
					source_field
				)

				match.update(transformation_info)
				matches.append(match)

		# Sort by confidence and return top matches
		matches.sort(key=lambda x: x["confidence"], reverse=True)
		return matches[:3]  # Top 3 matches

	async def _calculate_field_similarity(
		self,
		source_field: str,
		source_props: Dict[str, Any],
		target_field: str,
		target_props: Dict[str, Any],
		source_sample_data: Optional[List[Dict[str, Any]]]
	) -> float:
		"""Calculate similarity between source and target fields."""
		similarity_factors = []

		# Name similarity
		name_sim = await self._calculate_name_similarity(source_field, target_field)
		similarity_factors.append(("name", name_sim, 0.4))

		# Type compatibility
		type_compat = await self._calculate_type_compatibility(source_props, target_props)
		similarity_factors.append(("type", type_compat, 0.3))

		# Semantic similarity
		semantic_sim = await self._calculate_semantic_similarity(source_field, target_field)
		similarity_factors.append(("semantic", semantic_sim, 0.2))

		# Data pattern similarity (if sample data available)
		if source_sample_data:
			pattern_sim = await self._calculate_pattern_similarity(
				source_field,
				target_field,
				source_sample_data
			)
			similarity_factors.append(("pattern", pattern_sim, 0.1))

		# Weighted average
		total_score = sum(score * weight for name, score, weight in similarity_factors)
		total_weight = sum(weight for name, score, weight in similarity_factors)

		return total_score / total_weight if total_weight > 0 else 0.0

	async def _calculate_name_similarity(self, source_field: str, target_field: str) -> float:
		"""Calculate name-based similarity between fields."""
		source_clean = source_field.lower().replace("_", "").replace("-", "")
		target_clean = target_field.lower().replace("_", "").replace("-", "")

		# Exact match
		if source_clean == target_clean:
			return 1.0

		# Substring match
		if source_clean in target_clean or target_clean in source_clean:
			return 0.8

		# Common prefixes/suffixes
		common_patterns = [
			(r"(.*)_id$", r"\1$"),  # user_id -> user
			(r"^is_(.*)", r"\1$"),   # is_active -> active
			(r"(.*)_name$", r"\1$"), # first_name -> first
		]

		for source_pattern, target_pattern in common_patterns:
			source_match = re.match(source_pattern, source_field)
			target_match = re.match(target_pattern, target_field)

			if source_match and target_match:
				if source_match.group(1) == target_match.group(1):
					return 0.7

		# Levenshtein-like similarity (simplified)
		max_len = max(len(source_clean), len(target_clean))
		common_chars = len(set(source_clean) & set(target_clean))
		return common_chars / max_len if max_len > 0 else 0.0

	async def _calculate_type_compatibility(
		self,
		source_props: Dict[str, Any],
		target_props: Dict[str, Any]
	) -> float:
		"""Calculate type compatibility between fields."""
		source_type = source_props.get("type", "string")
		target_type = target_props.get("type", "string")

		# Exact type match
		if source_type == target_type:
			return 1.0

		# Compatible types
		compatible_types = {
			("integer", "number"): 0.9,
			("number", "integer"): 0.8,  # Potential data loss
			("string", "integer"): 0.6,  # If string contains numbers
			("string", "number"): 0.6,
			("boolean", "string"): 0.7,
			("integer", "string"): 0.8,
			("number", "string"): 0.8,
		}

		return compatible_types.get((source_type, target_type), 0.1)

	async def _calculate_semantic_similarity(self, source_field: str, target_field: str) -> float:
		"""Calculate semantic similarity using field patterns."""
		# Simplified semantic matching based on common business concepts
		semantic_groups = {
			"identity": ["id", "identifier", "uuid", "key"],
			"personal": ["name", "first", "last", "email", "phone"],
			"temporal": ["date", "time", "created", "updated", "timestamp"],
			"location": ["address", "city", "state", "country", "zip"],
			"financial": ["price", "amount", "cost", "total", "balance"],
			"status": ["status", "state", "active", "enabled", "valid"]
		}

		source_group = None
		target_group = None

		for group_name, keywords in semantic_groups.items():
			if any(keyword in source_field.lower() for keyword in keywords):
				source_group = group_name
			if any(keyword in target_field.lower() for keyword in keywords):
				target_group = group_name

		if source_group and target_group:
			return 1.0 if source_group == target_group else 0.3

		return 0.5  # Neutral score if no semantic group detected

	async def _calculate_pattern_similarity(
		self,
		source_field: str,
		target_field: str,
		source_sample_data: List[Dict[str, Any]]
	) -> float:
		"""Calculate similarity based on data patterns."""
		if not source_sample_data or source_field not in source_sample_data[0]:
			return 0.5

		source_values = [
			record.get(source_field) for record in source_sample_data[:10]
			if source_field in record and record[source_field] is not None
		]

		if not source_values:
			return 0.5

		# Analyze patterns in source data
		source_patterns = {
			"all_numeric": all(isinstance(v, (int, float)) for v in source_values),
			"all_string": all(isinstance(v, str) for v in source_values),
			"email_like": any("@" in str(v) for v in source_values),
			"date_like": any(self._looks_like_date(str(v)) for v in source_values),
			"id_like": any(len(str(v)) > 10 and str(v).replace("-", "").isalnum() for v in source_values)
		}

		# Match patterns with target field expectations
		target_lower = target_field.lower()
		pattern_matches = 0

		if source_patterns["email_like"] and "email" in target_lower:
			pattern_matches += 1
		if source_patterns["date_like"] and any(word in target_lower for word in ["date", "time", "created", "updated"]):
			pattern_matches += 1
		if source_patterns["id_like"] and "id" in target_lower:
			pattern_matches += 1
		if source_patterns["all_numeric"] and any(word in target_lower for word in ["count", "amount", "price", "total"]):
			pattern_matches += 1

		return min(1.0, pattern_matches * 0.5)

	def _looks_like_date(self, value: str) -> bool:
		"""Check if string looks like a date."""
		date_patterns = [
			r'\d{4}-\d{2}-\d{2}',
			r'\d{2}/\d{2}/\d{4}',
			r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
		]
		return any(re.search(pattern, value) for pattern in date_patterns)

	async def _analyze_transformation_needs(
		self,
		source_props: Dict[str, Any],
		target_props: Dict[str, Any],
		source_sample_data: Optional[List[Dict[str, Any]]],
		source_field: str
	) -> Dict[str, Any]:
		"""Analyze what transformations are needed for mapping."""
		transformation_info = {
			"transformation_required": False,
			"transformation_suggestion": None,
			"reasoning": []
		}

		source_type = source_props.get("type", "string")
		target_type = target_props.get("type", "string")

		# Type conversion needed
		if source_type != target_type:
			transformation_info["transformation_required"] = True
			transformation_info["transformation_suggestion"] = f"Convert {source_type} to {target_type}"
			transformation_info["reasoning"].append(f"Type conversion required: {source_type} -> {target_type}")

		# Format conversion needed
		source_format = source_props.get("format")
		target_format = target_props.get("format")

		if source_format != target_format and target_format:
			transformation_info["transformation_required"] = True
			if not transformation_info["transformation_suggestion"]:
				transformation_info["transformation_suggestion"] = f"Format conversion to {target_format}"
			transformation_info["reasoning"].append(f"Format conversion required: {source_format or 'none'} -> {target_format}")

		return transformation_info

	async def _record_mapping_session(
		self,
		source_schema: Dict[str, Any],
		target_schema: Dict[str, Any],
		suggestions: List[Dict[str, Any]]
	) -> None:
		"""Record mapping session for learning."""
		session_record = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"source_schema_hash": hash(json.dumps(source_schema, sort_keys=True)),
			"target_schema_hash": hash(json.dumps(target_schema, sort_keys=True)),
			"suggestions_count": len(suggestions),
			"avg_confidence": sum(s["confidence"] for s in suggestions) / len(suggestions) if suggestions else 0.0
		}

		self.mapping_history.append(session_record)

		# Keep only last 1000 sessions
		if len(self.mapping_history) > 1000:
			self.mapping_history = self.mapping_history[-1000:]