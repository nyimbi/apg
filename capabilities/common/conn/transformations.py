"""
APG Connection Management - Data Transformations Engine

Advanced data transformation capabilities with support for JSON, CSV, XML processing,
data type conversions, field mapping, and intelligent filtering.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Union, Callable
from xml.dom import minidom

from .models import TransformationRule

@dataclass
class DataTransformationEngine:
	"""
	Advanced data transformation engine supporting multiple formats
	and intelligent field mapping with validation.
	"""

	# Transformation Registry
	transformation_rules: Dict[str, TransformationRule] = field(default_factory=dict)
	custom_functions: Dict[str, Callable] = field(default_factory=dict)

	# Processing Statistics
	transformation_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

	def _log_transformation_operation(self, operation: str) -> None:
		"""Log transformation operations following APG patterns."""
		print(f"Data transformation engine: {operation}")

	async def transform_json_to_json(
		self,
		source_data: Dict[str, Any],
		transformation_rules: List[str],
		jq_expressions: Optional[List[str]] = None
	) -> Dict[str, Any]:
		"""Advanced JSON-to-JSON transformation with jq-like syntax support."""
		assert source_data, "Source data is required"

		self._log_transformation_operation("Executing JSON-to-JSON transformation")

		result = source_data.copy()

		# Apply transformation rules
		for rule_id in transformation_rules:
			if rule_id in self.transformation_rules:
				rule = self.transformation_rules[rule_id]
				result = await rule.apply(result)

		# Apply jq-like expressions
		if jq_expressions:
			for expression in jq_expressions:
				result = await self._apply_jq_expression(result, expression)

		return result

	async def _apply_jq_expression(self, data: Dict[str, Any], expression: str) -> Dict[str, Any]:
		"""Apply jq-like transformation expression to data."""
		# Simplified jq-like operations - in production, use actual jq library
		if expression.startswith("."):
			# Field selection: .field_name
			field_name = expression[1:]
			if field_name in data:
				return {field_name: data[field_name]}
		elif "|" in expression:
			# Pipe operations: .field | map(func)
			parts = expression.split("|")
			current_data = data
			for part in parts:
				part = part.strip()
				if part.startswith("."):
					field_name = part[1:]
					if isinstance(current_data, dict) and field_name in current_data:
						current_data = current_data[field_name]
				elif part.startswith("map("):
					# Simple map operation
					continue
			return current_data if isinstance(current_data, dict) else data

		return data

	async def parse_csv_data(
		self,
		csv_content: str,
		delimiter: str = ",",
		has_header: bool = True,
		encoding: str = "utf-8"
	) -> List[Dict[str, Any]]:
		"""Parse CSV data with configurable options."""
		assert csv_content.strip(), "CSV content is required"

		self._log_transformation_operation(f"Parsing CSV data with delimiter '{delimiter}'")

		try:
			csv_file = StringIO(csv_content)
			reader = csv.DictReader(csv_file, delimiter=delimiter) if has_header else csv.reader(csv_file, delimiter=delimiter)

			records = []
			for row in reader:
				if has_header:
					# Clean field names and convert values
					cleaned_row = {}
					for key, value in row.items():
						clean_key = key.strip().replace(" ", "_").lower()
						cleaned_row[clean_key] = self._convert_csv_value(value)
					records.append(cleaned_row)
				else:
					# Use column indexes as keys
					record = {}
					for i, value in enumerate(row):
						record[f"column_{i}"] = self._convert_csv_value(value)
					records.append(record)

			return records

		except Exception as e:
			self._log_transformation_operation(f"CSV parsing error: {e}")
			return []

	def _convert_csv_value(self, value: str) -> Union[str, int, float, bool]:
		"""Convert CSV string value to appropriate Python type."""
		if not value or value.strip() == "":
			return None

		value = value.strip()

		# Boolean conversion
		if value.lower() in ("true", "false"):
			return value.lower() == "true"

		# Numeric conversion
		try:
			if "." in value:
				return float(value)
			else:
				return int(value)
		except ValueError:
			pass

		return value

	async def format_csv_data(
		self,
		records: List[Dict[str, Any]],
		delimiter: str = ",",
		include_header: bool = True
	) -> str:
		"""Format data as CSV string."""
		assert records, "Records are required"

		self._log_transformation_operation("Formatting data as CSV")

		if not records:
			return ""

		output = StringIO()
		fieldnames = list(records[0].keys())
		writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)

		if include_header:
			writer.writeheader()

		for record in records:
			# Convert all values to strings for CSV output
			str_record = {k: str(v) if v is not None else "" for k, v in record.items()}
			writer.writerow(str_record)

		return output.getvalue()

	async def parse_xml_data(
		self,
		xml_content: str,
		root_element: Optional[str] = None
	) -> Dict[str, Any]:
		"""Parse XML data into dictionary structure."""
		assert xml_content.strip(), "XML content is required"

		self._log_transformation_operation("Parsing XML data")

		try:
			root = ET.fromstring(xml_content)

			if root_element:
				# Find specific root element
				target_root = root.find(root_element)
				if target_root is not None:
					root = target_root

			return self._xml_to_dict(root)

		except ET.ParseError as e:
			self._log_transformation_operation(f"XML parsing error: {e}")
			return {}

	def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
		"""Convert XML element to dictionary."""
		result = {}

		# Add attributes
		if element.attrib:
			result["@attributes"] = element.attrib

		# Add text content
		if element.text and element.text.strip():
			if len(element) == 0:
				# Leaf element with only text
				return element.text.strip()
			else:
				result["@text"] = element.text.strip()

		# Add child elements
		children = {}
		for child in element:
			child_data = self._xml_to_dict(child)

			if child.tag in children:
				# Multiple elements with same tag - create array
				if not isinstance(children[child.tag], list):
					children[child.tag] = [children[child.tag]]
				children[child.tag].append(child_data)
			else:
				children[child.tag] = child_data

		result.update(children)
		return result

	async def convert_xml_to_json(self, xml_content: str) -> str:
		"""Convert XML to JSON string."""
		xml_dict = await self.parse_xml_data(xml_content)
		return json.dumps(xml_dict, indent=2, default=str)

	async def convert_data_types(
		self,
		data: Dict[str, Any],
		type_mappings: Dict[str, str]
	) -> Dict[str, Any]:
		"""Convert data types based on field mappings."""
		assert data, "Data is required"
		assert type_mappings, "Type mappings are required"

		self._log_transformation_operation("Converting data types")

		result = data.copy()

		for field_name, target_type in type_mappings.items():
			if field_name in result:
				try:
					original_value = result[field_name]
					converted_value = await self._convert_value_type(original_value, target_type)
					result[field_name] = converted_value
				except Exception as e:
					self._log_transformation_operation(f"Type conversion error for {field_name}: {e}")
					# Keep original value on conversion error

		return result

	async def _convert_value_type(self, value: Any, target_type: str) -> Any:
		"""Convert a single value to target type."""
		if value is None:
			return None

		if target_type == "string":
			return str(value)
		elif target_type == "integer":
			if isinstance(value, str):
				# Handle numeric strings
				return int(float(value))
			return int(value)
		elif target_type == "float":
			return float(value)
		elif target_type == "boolean":
			if isinstance(value, str):
				return value.lower() in ("true", "1", "yes", "on")
			return bool(value)
		elif target_type == "datetime":
			if isinstance(value, str):
				# Try common datetime formats
				formats = [
					"%Y-%m-%d %H:%M:%S",
					"%Y-%m-%d",
					"%Y-%m-%dT%H:%M:%S",
					"%Y-%m-%dT%H:%M:%SZ",
					"%d/%m/%Y",
					"%m/%d/%Y"
				]
				for fmt in formats:
					try:
						return datetime.strptime(value, fmt)
					except ValueError:
						continue
				# If no format matches, return original
				return value
			return value
		else:
			return value

	async def map_fields(
		self,
		data: Dict[str, Any],
		field_mappings: Dict[str, str],
		remove_unmapped: bool = False
	) -> Dict[str, Any]:
		"""Map field names according to mapping rules."""
		assert data, "Data is required"
		assert field_mappings, "Field mappings are required"

		self._log_transformation_operation(f"Mapping fields: {len(field_mappings)} mappings")

		result = {} if remove_unmapped else data.copy()

		# Apply field mappings
		for source_field, target_field in field_mappings.items():
			if source_field in data:
				result[target_field] = data[source_field]
				# Remove original field if different name
				if source_field != target_field and source_field in result:
					del result[source_field]

		return result

	async def filter_records(
		self,
		records: List[Dict[str, Any]],
		filter_conditions: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Filter records based on conditions."""
		assert records, "Records are required"
		assert filter_conditions, "Filter conditions are required"

		self._log_transformation_operation(f"Filtering {len(records)} records")

		filtered_records = []

		for record in records:
			include_record = True

			for condition in filter_conditions:
				field = condition.get("field")
				operator = condition.get("operator", "equals")
				value = condition.get("value")

				if not field or field not in record:
					continue

				record_value = record[field]

				if operator == "equals" and record_value != value:
					include_record = False
					break
				elif operator == "not_equals" and record_value == value:
					include_record = False
					break
				elif operator == "greater_than" and not (record_value > value):
					include_record = False
					break
				elif operator == "less_than" and not (record_value < value):
					include_record = False
					break
				elif operator == "contains" and str(value) not in str(record_value):
					include_record = False
					break
				elif operator == "regex" and not re.match(str(value), str(record_value)):
					include_record = False
					break

			if include_record:
				filtered_records.append(record)

		self._log_transformation_operation(f"Filtered to {len(filtered_records)} records")
		return filtered_records

	async def aggregate_data(
		self,
		records: List[Dict[str, Any]],
		group_by: List[str],
		aggregations: Dict[str, Dict[str, str]]
	) -> List[Dict[str, Any]]:
		"""Aggregate data with grouping and aggregation functions."""
		assert records, "Records are required"
		assert group_by, "Group by fields are required"
		assert aggregations, "Aggregations are required"

		self._log_transformation_operation(f"Aggregating data by {group_by}")

		# Group records
		groups = {}
		for record in records:
			# Create group key
			group_key = tuple(record.get(field, None) for field in group_by)

			if group_key not in groups:
				groups[group_key] = []
			groups[group_key].append(record)

		# Calculate aggregations
		aggregated_records = []
		for group_key, group_records in groups.items():
			aggregated_record = {}

			# Add group fields
			for i, field in enumerate(group_by):
				aggregated_record[field] = group_key[i]

			# Calculate aggregations
			for field_name, agg_config in aggregations.items():
				source_field = agg_config.get("field", field_name)
				operation = agg_config.get("operation", "count")

				values = [r.get(source_field) for r in group_records if r.get(source_field) is not None]

				if operation == "count":
					aggregated_record[field_name] = len(group_records)
				elif operation == "sum" and values:
					aggregated_record[field_name] = sum(float(v) for v in values if isinstance(v, (int, float)) or str(v).replace('.', '').isdigit())
				elif operation == "avg" and values:
					numeric_values = [float(v) for v in values if isinstance(v, (int, float)) or str(v).replace('.', '').isdigit()]
					aggregated_record[field_name] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
				elif operation == "min" and values:
					aggregated_record[field_name] = min(values)
				elif operation == "max" and values:
					aggregated_record[field_name] = max(values)
				else:
					aggregated_record[field_name] = len(group_records)

			aggregated_records.append(aggregated_record)

		return aggregated_records

	async def validate_data_schema(
		self,
		records: List[Dict[str, Any]],
		schema: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validate records against a JSON schema."""
		assert records, "Records are required"
		assert schema, "Schema is required"

		self._log_transformation_operation(f"Validating {len(records)} records against schema")

		validation_results = {
			"valid_records": [],
			"invalid_records": [],
			"validation_errors": [],
			"summary": {
				"total_records": len(records),
				"valid_count": 0,
				"invalid_count": 0,
				"validation_rate": 0.0
			}
		}

		required_fields = schema.get("required", [])
		field_types = schema.get("properties", {})

		for i, record in enumerate(records):
			is_valid = True
			record_errors = []

			# Check required fields
			for field in required_fields:
				if field not in record or record[field] is None:
					is_valid = False
					record_errors.append(f"Missing required field: {field}")

			# Check field types
			for field_name, field_schema in field_types.items():
				if field_name in record:
					expected_type = field_schema.get("type", "string")
					field_value = record[field_name]

					if not self._validate_field_type(field_value, expected_type):
						is_valid = False
						record_errors.append(f"Invalid type for {field_name}: expected {expected_type}")

			if is_valid:
				validation_results["valid_records"].append(record)
				validation_results["summary"]["valid_count"] += 1
			else:
				validation_results["invalid_records"].append({
					"record_index": i,
					"record": record,
					"errors": record_errors
				})
				validation_results["validation_errors"].extend(record_errors)
				validation_results["summary"]["invalid_count"] += 1

		validation_results["summary"]["validation_rate"] = (
			validation_results["summary"]["valid_count"] /
			validation_results["summary"]["total_records"]
		) if validation_results["summary"]["total_records"] > 0 else 0

		return validation_results

	def _validate_field_type(self, value: Any, expected_type: str) -> bool:
		"""Validate if value matches expected type."""
		if value is None:
			return True  # Allow null values unless specified otherwise

		if expected_type == "string":
			return isinstance(value, str)
		elif expected_type == "integer":
			return isinstance(value, int)
		elif expected_type == "number":
			return isinstance(value, (int, float))
		elif expected_type == "boolean":
			return isinstance(value, bool)
		elif expected_type == "array":
			return isinstance(value, list)
		elif expected_type == "object":
			return isinstance(value, dict)

		return True

	async def get_transformation_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive transformation statistics."""
		return {
			"total_rules": len(self.transformation_rules),
			"custom_functions": len(self.custom_functions),
			"transformation_stats": self.transformation_stats.copy(),
			"supported_formats": ["json", "csv", "xml"],
			"supported_operations": [
				"field_mapping", "type_conversion", "filtering",
				"aggregation", "validation", "transformation"
			]
		}

@dataclass
class TransformationRuleBuilder:
	"""Builder for creating complex transformation rules programmatically."""

	def __init__(self):
		self.rule_config = {
			"transformations": [],
			"validations": [],
			"filters": []
		}

	def add_field_mapping(self, source_field: str, target_field: str) -> 'TransformationRuleBuilder':
		"""Add field mapping transformation."""
		self.rule_config["transformations"].append({
			"type": "field_mapping",
			"source_field": source_field,
			"target_field": target_field
		})
		return self

	def add_type_conversion(self, field: str, target_type: str) -> 'TransformationRuleBuilder':
		"""Add type conversion transformation."""
		self.rule_config["transformations"].append({
			"type": "type_conversion",
			"field": field,
			"target_type": target_type
		})
		return self

	def add_filter(self, field: str, operator: str, value: Any) -> 'TransformationRuleBuilder':
		"""Add filter condition."""
		self.rule_config["filters"].append({
			"field": field,
			"operator": operator,
			"value": value
		})
		return self

	def add_validation(self, field: str, validation_type: str, **kwargs) -> 'TransformationRuleBuilder':
		"""Add validation rule."""
		validation_rule = {
			"field": field,
			"type": validation_type,
			**kwargs
		}
		self.rule_config["validations"].append(validation_rule)
		return self

	def build(self, name: str, tenant_id: str, created_by: str) -> TransformationRule:
		"""Build the transformation rule."""
		return TransformationRule(
			tenant_id=tenant_id,
			name=name,
			description=f"Generated transformation rule with {len(self.rule_config['transformations'])} transformations",
			rule_type="composite",
			source_field="*",
			target_field="*",
			transformation_config=self.rule_config,
			created_by=created_by
		)