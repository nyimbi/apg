"""
Dependency Validator

Validates capability compositions for dependencies, conflicts, and compatibility.
Provides conflict resolution strategies and detailed validation reporting.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from.registry import CapabilityRegistry, CapabilityMetadata, SubCapabilityMetadata, DependencyInfo, get_registry

logger = logging.getLogger(__name__)

class ValidationErrorType(Enum):
	"""Types of validation errors."""
	MISSING_DEPENDENCY = "missing_dependency"
	CIRCULAR_DEPENDENCY = "circular_dependency"
	VERSION_CONFLICT = "version_conflict"
	INCOMPATIBLE_CAPABILITIES = "incompatible_capabilities"
	MISSING_CAPABILITY = "missing_capability"
	CONFIGURATION_CONFLICT = "configuration_conflict"
	RESOURCE_CONFLICT = "resource_conflict"
	SECURITY_CONFLICT = "security_conflict"

class ValidationWarningType(Enum):
	"""Types of validation warnings."""
	RECOMMENDED_DEPENDENCY = "recommended_dependency"
	VERSION_MISMATCH = "version_mismatch"
	PERFORMANCE_IMPACT = "performance_impact"
	COMPATIBILITY_CONCERN = "compatibility_concern"
	CONFIGURATION_OVERRIDE = "configuration_override"
	DEPRECATED_CAPABILITY = "deprecated_capability"

class ConflictResolutionStrategy(Enum):
	"""Strategies for resolving conflicts."""
	AUTO_ADD_DEPENDENCIES = "auto_add_dependencies"
	USE_LATEST_VERSION = "use_latest_version"
	PRIORITIZE_REQUIRED = "prioritize_required"
	MANUAL_RESOLUTION = "manual_resolution"
	EXCLUDE_CONFLICTING = "exclude_conflicting"

class ValidationError(BaseModel):
	"""Represents a validation error."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	id: str = Field(default_factory=uuid7str, description="Unique error ID")
	error_type: ValidationErrorType = Field(..., description="Type of error")
	capability: str = Field(..., description="Capability code causing error")
	subcapability: Optional[str] = Field(default=None, description="Sub-capability if applicable")
	message: str = Field(..., description="Human-readable error message")
	details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
	suggested_resolution: Optional[str] = Field(default=None, description="Suggested resolution")
	severity: str = Field(default="error", description="Error severity")

	def __str__(self) -> str:
		location = f"{self.capability}.{self.subcapability}" if self.subcapability else self.capability
		return f"[{self.error_type.value}] {location}: {self.message}"

class ValidationWarning(BaseModel):
	"""Represents a validation warning."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	id: str = Field(default_factory=uuid7str, description="Unique warning ID")
	warning_type: ValidationWarningType = Field(..., description="Type of warning")
	capability: str = Field(..., description="Capability code")
	subcapability: Optional[str] = Field(default=None, description="Sub-capability if applicable")
	message: str = Field(..., description="Human-readable warning message")
	details: dict[str, Any] = Field(default_factory=dict, description="Additional warning details")
	suggested_action: Optional[str] = Field(default=None, description="Suggested action")

	def __str__(self) -> str:
		location = f"{self.capability}.{self.subcapability}" if self.subcapability else self.capability
		return f"[{self.warning_type.value}] {location}: {self.message}"

class ConflictResolution(BaseModel):
	"""Represents a conflict resolution."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	conflict_id: str = Field(..., description="ID of the conflict being resolved")
	strategy: ConflictResolutionStrategy = Field(..., description="Resolution strategy")
	actions: list[str] = Field(default_factory=list, description="Actions to take")
	capabilities_added: list[str] = Field(default_factory=list, description="Capabilities to add")
	capabilities_removed: list[str] = Field(default_factory=list, description="Capabilities to remove")
	configuration_changes: dict[str, Any] = Field(default_factory=dict, description="Configuration changes")
	rationale: str = Field(default="", description="Explanation of resolution")

class ValidationResult(BaseModel):
	"""Result of capability composition validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	valid: bool = Field(..., description="Whether composition is valid")
	composition_id: str = Field(default_factory=uuid7str, description="Unique validation ID")
	validated_at: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")

	# Validation details
	capabilities_validated: list[str] = Field(..., description="Capabilities that were validated")
	errors: list[ValidationError] = Field(default_factory=list, description="Validation errors")
	warnings: list[ValidationWarning] = Field(default_factory=list, description="Validation warnings")

	# Dependency resolution
	resolved_dependencies: list[str] = Field(default_factory=list, description="Resolved dependencies")
	dependency_order: list[str] = Field(default_factory=list, description="Dependency initialization order")
	circular_dependencies: list[list[str]] = Field(default_factory=list, description="Circular dependency chains")

	# Conflict resolution
	conflicts_found: list[dict[str, Any]] = Field(default_factory=list, description="Conflicts found")
	resolutions: list[ConflictResolution] = Field(default_factory=list, description="Conflict resolutions")

	# Metadata
	validation_time_ms: int = Field(default=0, description="Validation time in milliseconds")
	total_capabilities: int = Field(default=0, description="Total capabilities validated")
	total_subcapabilities: int = Field(default=0, description="Total sub-capabilities validated")

	@property
	def error_messages(self) -> list[str]:
		"""Get list of error messages."""
		return [str(error) for error in self.errors]

	@property
	def warning_messages(self) -> list[str]:
		"""Get list of warning messages."""
		return [str(warning) for warning in self.warnings]

class DependencyValidator:
	"""
	Validates capability compositions for dependencies, conflicts, and compatibility.

	Performs comprehensive validation including:
	- Dependency resolution and circular dependency detection
	- Version compatibility checking
	- Resource and configuration conflict detection
	- Security and compliance validation
	- Performance impact analysis
	"""

	def __init__(self, registry: Optional[CapabilityRegistry] = None):
		"""Initialize the dependency validator."""
		self.registry = registry or get_registry()

		# Version comparison regex
		self.version_pattern = re.compile(r'([><=!]+)?(\d+\.\d+\.\d+)')

		logger.info("DependencyValidator initialized")

	def validate_composition(self, capabilities: List[str]) -> ValidationResult:
		"""
		Validate a capability composition.

		Args:
			capabilities: List of capability codes to validate

		Returns:
			ValidationResult with validation status and details
		"""
		start_time = datetime.utcnow()

		try:
			# Ensure registry is populated
			if not self.registry.capabilities:
				self.registry.discover_all()

			result = ValidationResult(
				valid=True,
				capabilities_validated=capabilities.copy()
			)

			logger.info(f"Starting validation for capabilities: {capabilities}")

			# Step 1: Validate capability existence
			self._validate_capability_existence(capabilities, result)

			# Step 2: Resolve dependencies
			self._resolve_dependencies(capabilities, result)

			# Step 3: Check for circular dependencies
			self._check_circular_dependencies(result)

			# Step 4: Validate versions
			self._validate_versions(result)

			# Step 5: Check for conflicts
			self._check_conflicts(result)

			# Step 6: Validate compatibility
			self._validate_compatibility(result)

			# Step 7: Check security requirements
			self._validate_security(result)

			# Step 8: Performance analysis
			self._analyze_performance_impact(result)

			# Determine final validation status
			result.valid = len(result.errors) == 0

			# Calculate validation time
			end_time = datetime.utcnow()
			result.validation_time_ms = int((end_time - start_time).total_seconds() * 1000)

			logger.info(f"Validation completed in {result.validation_time_ms}ms. Valid: {result.valid}")

			return result

		except Exception as e:
			logger.error(f"Validation failed: {e}")

			# Create error result
			end_time = datetime.utcnow()
			validation_time = int((end_time - start_time).total_seconds() * 1000)

			error_result = ValidationResult(
				valid=False,
				capabilities_validated=capabilities,
				validation_time_ms=validation_time
			)

			error_result.errors.append(ValidationError(
				error_type=ValidationErrorType.MISSING_CAPABILITY,
				capability="SYSTEM",
				message=f"Validation process failed: {str(e)}",
				severity="critical"
			))

			return error_result

	def _validate_capability_existence(self, capabilities: List[str], result: ValidationResult) -> None:
		"""Validate that all requested capabilities exist."""
		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				result.errors.append(ValidationError(
					error_type=ValidationErrorType.MISSING_CAPABILITY,
					capability=cap_code,
					message=f"Capability '{cap_code}' not found in registry",
					suggested_resolution=f"Check capability code spelling or ensure capability is installed"
				))

	def _resolve_dependencies(self, capabilities: List[str], result: ValidationResult) -> None:
		"""Resolve all dependencies for the given capabilities."""
		all_capabilities = set(capabilities)
		to_process = list(capabilities)
		processed = set()

		while to_process:
			cap_code = to_process.pop(0)
			if cap_code in processed:
				continue

			processed.add(cap_code)
			capability = self.registry.get_capability(cap_code)

			if capability:
				for dep in capability.dependencies:
					if dep.required:
						if dep.capability not in all_capabilities:
							all_capabilities.add(dep.capability)
							to_process.append(dep.capability)
							result.resolved_dependencies.append(dep.capability)

						# Check if dependency exists
						dep_capability = self.registry.get_capability(dep.capability)
						if not dep_capability:
							result.errors.append(ValidationError(
								error_type=ValidationErrorType.MISSING_DEPENDENCY,
								capability=cap_code,
								message=f"Required dependency '{dep.capability}' not found",
								details={'dependency': dep.capability, 'version_requirement': dep.version_requirement},
								suggested_resolution=f"Install capability '{dep.capability}' or remove '{cap_code}'"
							))
					else:
						# Optional dependency - add warning if missing
						dep_capability = self.registry.get_capability(dep.capability)
						if not dep_capability:
							result.warnings.append(ValidationWarning(
								warning_type=ValidationWarningType.RECOMMENDED_DEPENDENCY,
								capability=cap_code,
								message=f"Recommended dependency '{dep.capability}' not available",
								details={'dependency': dep.capability},
								suggested_action=f"Consider installing '{dep.capability}' for enhanced functionality"
							))

		# Build dependency order using topological sort
		result.dependency_order = self._topological_sort(all_capabilities, result)

	def _topological_sort(self, capabilities: Set[str], result: ValidationResult) -> List[str]:
		"""Sort capabilities by dependency order."""
		# Build dependency graph
		graph = {}
		in_degree = {}

		for cap_code in capabilities:
			graph[cap_code] = []
			in_degree[cap_code] = 0

		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if capability:
				for dep in capability.dependencies:
					if dep.required and dep.capability in capabilities:
						graph[dep.capability].append(cap_code)
						in_degree[cap_code] += 1

		# Kahn's algorithm
		queue = [cap for cap, degree in in_degree.items() if degree == 0]
		sorted_caps = []

		while queue:
			cap = queue.pop(0)
			sorted_caps.append(cap)

			for neighbor in graph[cap]:
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					queue.append(neighbor)

		return sorted_caps

	def _check_circular_dependencies(self, result: ValidationResult) -> None:
		"""Check for circular dependencies."""
		if len(result.dependency_order) < len(result.capabilities_validated) + len(result.resolved_dependencies):
			# Circular dependency detected
			all_caps = set(result.capabilities_validated + result.resolved_dependencies)
			remaining = all_caps - set(result.dependency_order)

			if remaining:
				# Find the circular dependency chain
				circular_chain = self._find_circular_chain(remaining)
				if circular_chain:
					result.circular_dependencies.append(circular_chain)

					result.errors.append(ValidationError(
						error_type=ValidationErrorType.CIRCULAR_DEPENDENCY,
						capability=circular_chain[0],
						message=f"Circular dependency detected: {' -> '.join(circular_chain)}",
						details={'circular_chain': circular_chain},
						suggested_resolution="Remove one of the dependencies to break the cycle"
					))

	def _find_circular_chain(self, remaining_caps: Set[str]) -> List[str]:
		"""Find a circular dependency chain."""
		for start_cap in remaining_caps:
			visited = set()
			path = []

			if self._dfs_find_cycle(start_cap, visited, path, remaining_caps):
				return path

		return list(remaining_caps)  # Fallback

	def _dfs_find_cycle(self, cap: str, visited: Set[str], path: List[str], remaining: Set[str]) -> bool:
		"""Depth-first search to find cycle."""
		if cap in path:
			# Found cycle
			cycle_start = path.index(cap)
			return True

		if cap in visited:
			return False

		visited.add(cap)
		path.append(cap)

		capability = self.registry.get_capability(cap)
		if capability:
			for dep in capability.dependencies:
				if dep.required and dep.capability in remaining:
					if self._dfs_find_cycle(dep.capability, visited, path, remaining):
						return True

		path.pop()
		return False

	def _validate_versions(self, result: ValidationResult) -> None:
		"""Validate version requirements and compatibility."""
		all_caps = result.capabilities_validated + result.resolved_dependencies

		for cap_code in all_caps:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				continue

			for dep in capability.dependencies:
				if dep.capability in all_caps:
					dep_capability = self.registry.get_capability(dep.capability)
					if dep_capability:
						if not self._check_version_compatibility(dep_capability.version, dep.version_requirement):
							result.errors.append(ValidationError(
								error_type=ValidationErrorType.VERSION_CONFLICT,
								capability=cap_code,
								message=f"Version conflict: {dep.capability} v{dep_capability.version} does not satisfy requirement {dep.version_requirement}",
								details={
									'dependency': dep.capability,
									'available_version': dep_capability.version,
									'required_version': dep.version_requirement
								},
								suggested_resolution=f"Update {dep.capability} to satisfy version requirement"
							))

	def _check_version_compatibility(self, available_version: str, requirement: str) -> bool:
		"""Check if available version satisfies requirement."""
		try:
			# Parse requirement
			match = self.version_pattern.match(requirement)
			if not match:
				return True  # No specific requirement

			operator = match.group(1) or '>='
			required_version = match.group(2)

			# Simple version comparison (assumes semantic versioning)
			available_parts = [int(x) for x in available_version.split('.')]
			required_parts = [int(x) for x in required_version.split('.')]

			# Pad to same length
			max_len = max(len(available_parts), len(required_parts))
			available_parts.extend([0] * (max_len - len(available_parts)))
			required_parts.extend([0] * (max_len - len(required_parts)))

			# Compare
			if operator == '>=':
				return available_parts >= required_parts
			elif operator == '>':
				return available_parts > required_parts
			elif operator == '==':
				return available_parts == required_parts
			elif operator == '<=':
				return available_parts <= required_parts
			elif operator == '<':
				return available_parts < required_parts
			elif operator == '!=':
				return available_parts != required_parts

			return True

		except Exception:
			return True  # Default to compatible if parsing fails

	def _check_conflicts(self, result: ValidationResult) -> None:
		"""Check for various types of conflicts between capabilities."""
		all_caps = result.capabilities_validated + result.resolved_dependencies

		# Check for database table conflicts
		self._check_database_conflicts(all_caps, result)

		# Check for configuration conflicts
		self._check_configuration_conflicts(all_caps, result)

		# Check for resource conflicts
		self._check_resource_conflicts(all_caps, result)

	def _check_database_conflicts(self, capabilities: List[str], result: ValidationResult) -> None:
		"""Check for database table name conflicts."""
		table_owners = {}

		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				continue

			for subcap in capability.subcapabilities.values():
				for table in subcap.database_tables:
					if table in table_owners:
						existing_owner = table_owners[table]
						result.errors.append(ValidationError(
							error_type=ValidationErrorType.RESOURCE_CONFLICT,
							capability=cap_code,
							subcapability=subcap.code,
							message=f"Database table '{table}' conflict with {existing_owner}",
							details={'table': table, 'existing_owner': existing_owner},
							suggested_resolution=f"Use table prefixes or rename tables to avoid conflict"
						))
					else:
						table_owners[table] = f"{cap_code}.{subcap.code}"

	def _check_configuration_conflicts(self, capabilities: List[str], result: ValidationResult) -> None:
		"""Check for configuration conflicts."""
		config_keys = {}

		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				continue

			for config_key in capability.configuration_schema:
				if config_key in config_keys:
					existing_cap = config_keys[config_key]

					# Check if configurations are compatible
					existing_config = self.registry.get_capability(existing_cap).configuration_schema[config_key]
					current_config = capability.configuration_schema[config_key]

					if existing_config != current_config:
						result.warnings.append(ValidationWarning(
							warning_type=ValidationWarningType.CONFIGURATION_OVERRIDE,
							capability=cap_code,
							message=f"Configuration key '{config_key}' conflicts with {existing_cap}",
							details={'config_key': config_key, 'existing_capability': existing_cap},
							suggested_action="Review configuration to ensure compatibility"
						))
				else:
					config_keys[config_key] = cap_code

	def _check_resource_conflicts(self, capabilities: List[str], result: ValidationResult) -> None:
		"""Check for resource conflicts (ports, endpoints, etc.)."""
		endpoint_owners: list[tuple[str, str, str, str | None]] = []
		port_owners: dict[int, str] = {}
		service_owners: dict[str, str] = {}
		file_path_owners: dict[str, str] = {}
		queue_owners: dict[str, str] = {}
		topic_owners: dict[str, str] = {}
		env_var_owners: dict[str, tuple[str, Any]] = {}

		for cap_code in capabilities:
			capability = self.registry.get_capability(cap_code)
			if not capability:
				continue

			for owner, resource_spec in self._capability_resource_specs(capability):
				for method, path in self._normalize_resource_endpoints(self._resource_values(resource_spec, "api_endpoints", "endpoints", "routes")):
					conflicting_owner = self._find_endpoint_owner(endpoint_owners, method, path)
					if conflicting_owner and conflicting_owner != owner:
						self._add_resource_conflict(
							result,
							cap_code,
							owner,
							conflicting_owner,
							"api_endpoint",
							f"{method} {path}",
							"Move one endpoint under a capability-specific route namespace"
						)
					else:
						endpoint_owners.append((method, path, owner, cap_code))

				for port in self._normalize_resource_ports(resource_spec):
					self._check_unique_resource(
						port_owners,
						port,
						result,
						cap_code,
						owner,
						"port",
						str(port),
						"Assign a distinct port or route one capability through the composition gateway"
					)

				for service in self._normalize_string_resources(self._resource_values(resource_spec, "services", "service_names", "provides_services")):
					self._check_unique_resource(
						service_owners,
						service,
						result,
						cap_code,
						owner,
						"service_name",
						service,
						"Use unique service aliases in the composition manifest"
					)

				for path in self._normalize_string_resources(self._resource_values(resource_spec, "file_paths", "filesystem_paths", "files")):
					self._check_unique_resource(
						file_path_owners,
						path,
						result,
						cap_code,
						owner,
						"file_path",
						path,
						"Move one capability to a capability-specific storage path"
					)

				for queue in self._normalize_string_resources(self._resource_values(resource_spec, "queues", "message_queues")):
					self._check_unique_resource(
						queue_owners,
						queue,
						result,
						cap_code,
						owner,
						"queue",
						queue,
						"Use a shared broker exchange with capability-specific queues"
					)

				for topic in self._normalize_string_resources(self._resource_values(resource_spec, "topics", "event_topics")):
					self._check_unique_resource(
						topic_owners,
						topic,
						result,
						cap_code,
						owner,
						"topic",
						topic,
						"Declare a shared event contract or use capability-specific topics"
					)

				for env_key, env_default in self._normalize_env_vars(resource_spec):
					existing = env_var_owners.get(env_key)
					if existing and existing[0] != owner and existing[1] != env_default:
						self._add_resource_conflict(
							result,
							cap_code,
							owner,
							existing[0],
							"environment_variable",
							env_key,
							"Align the required value or use capability-specific environment variable names"
						)
					else:
						env_var_owners[env_key] = (owner, env_default)

	def _capability_resource_specs(self, capability: Any) -> list[tuple[str, dict[str, Any]]]:
		"""Collect resource declarations from capability and sub-capability metadata."""
		specs: list[tuple[str, dict[str, Any]]] = []
		cap_code = str(getattr(capability, "code", getattr(capability, "capability_code", "")) or "")
		cap_spec = self._extract_resource_spec(capability)
		if cap_spec:
			specs.append((cap_code, cap_spec))

		subcapabilities = getattr(capability, "subcapabilities", {}) or {}
		subcap_values = subcapabilities.values() if isinstance(subcapabilities, dict) else subcapabilities
		for subcapability in subcap_values:
			subcap_code = str(getattr(subcapability, "code", getattr(subcapability, "capability_code", "")) or "")
			owner = f"{cap_code}.{subcap_code}" if subcap_code else cap_code
			subcap_spec = self._extract_resource_spec(subcapability)
			if subcap_spec:
				specs.append((owner, subcap_spec))

		return specs

	def _extract_resource_spec(self, source: Any) -> dict[str, Any]:
		"""Extract resource metadata without assuming one canonical schema yet."""
		spec: dict[str, Any] = {}

		configuration = getattr(source, "configuration_schema", {}) or {}
		if isinstance(configuration, dict):
			resource_config = configuration.get("resources", {})
			if isinstance(resource_config, dict):
				spec.update(resource_config)
			for key in (
				"api_endpoints",
				"endpoints",
				"routes",
				"ports",
				"services",
				"service_names",
				"provides_services",
				"file_paths",
				"filesystem_paths",
				"files",
				"queues",
				"message_queues",
				"topics",
				"event_topics",
				"environment_variables",
				"env_vars",
			):
				if key in configuration:
					spec[key] = configuration[key]

		for attr in (
			"api_endpoints",
			"ports",
			"services",
			"service_names",
			"provides_services",
			"file_paths",
			"filesystem_paths",
			"queues",
			"topics",
			"environment_variables",
			"env_vars",
		):
			value = getattr(source, attr, None)
			if value:
				spec[attr] = value

		return spec

	def _resource_values(self, resource_spec: dict[str, Any], *keys: str) -> list[Any]:
		values: list[Any] = []
		for key in keys:
			raw_value = resource_spec.get(key)
			if raw_value is None:
				continue
			if isinstance(raw_value, list):
				values.extend(raw_value)
			else:
				values.append(raw_value)
		return values

	def _normalize_resource_endpoints(self, endpoints: list[Any]) -> list[tuple[str, str]]:
		normalized: list[tuple[str, str]] = []
		for endpoint in endpoints:
			if isinstance(endpoint, str):
				path = endpoint.strip()
				if path:
					normalized.append(("ANY", path))
			elif isinstance(endpoint, dict):
				method = str(endpoint.get("method", endpoint.get("verb", "ANY"))).upper()
				path = str(endpoint.get("path", endpoint.get("route", ""))).strip()
				if path:
					normalized.append((method, path))
		return normalized

	def _find_endpoint_owner(self, endpoint_owners: list[tuple[str, str, str, str | None]], method: str, path: str) -> str | None:
		for existing_method, existing_path, owner, _cap_code in endpoint_owners:
			if existing_path == path and (existing_method == method or existing_method == "ANY" or method == "ANY"):
				return owner
		return None

	def _normalize_resource_ports(self, resource_spec: dict[str, Any]) -> list[int]:
		raw_ports = self._resource_values(resource_spec, "ports")
		for endpoint in self._resource_values(resource_spec, "api_endpoints", "endpoints", "routes"):
			if isinstance(endpoint, dict) and endpoint.get("port") is not None:
				raw_ports.append(endpoint["port"])

		ports: list[int] = []
		for port in raw_ports:
			try:
				ports.append(int(port))
			except (TypeError, ValueError):
				continue
		return ports

	def _normalize_string_resources(self, values: list[Any]) -> list[str]:
		normalized: list[str] = []
		for value in values:
			name = ""
			if isinstance(value, str):
				name = value
			elif isinstance(value, dict):
				name = str(value.get("name") or value.get("code") or value.get("service") or value.get("path") or value.get("queue") or value.get("topic") or "")
			name = name.strip()
			if name:
				normalized.append(name)
		return normalized

	def _normalize_env_vars(self, resource_spec: dict[str, Any]) -> list[tuple[str, Any]]:
		raw_values = self._resource_values(resource_spec, "environment_variables", "env_vars")
		normalized: list[tuple[str, Any]] = []
		for raw_value in raw_values:
			if isinstance(raw_value, dict):
				if "name" in raw_value:
					normalized.append((str(raw_value["name"]), raw_value.get("default", raw_value.get("value"))))
				else:
					for key, value in raw_value.items():
						if isinstance(value, dict):
							normalized.append((str(key), value.get("default", value.get("value"))))
						else:
							normalized.append((str(key), value))
			elif isinstance(raw_value, str):
				normalized.append((raw_value, None))
		return normalized

	def _check_unique_resource(
		self,
		owners: dict[Any, str],
		resource_key: Any,
		result: ValidationResult,
		capability: str,
		owner: str,
		resource_type: str,
		resource_value: str,
		suggested_resolution: str
	) -> None:
		existing_owner = owners.get(resource_key)
		if existing_owner and existing_owner != owner:
			self._add_resource_conflict(
				result,
				capability,
				owner,
				existing_owner,
				resource_type,
				resource_value,
				suggested_resolution
			)
		else:
			owners[resource_key] = owner

	def _add_resource_conflict(
		self,
		result: ValidationResult,
		capability: str,
		owner: str,
		existing_owner: str,
		resource_type: str,
		resource_value: str,
		suggested_resolution: str
	) -> None:
		message = f"{resource_type.replace('_', ' ').title()} '{resource_value}' conflicts with {existing_owner}"
		result.errors.append(ValidationError(
			error_type=ValidationErrorType.RESOURCE_CONFLICT,
			capability=capability,
			subcapability=owner.split(".", 1)[1] if "." in owner else None,
			message=message,
			details={
				"resource_type": resource_type,
				"resource_value": resource_value,
				"owner": owner,
				"existing_owner": existing_owner
			},
			suggested_resolution=suggested_resolution
		))
		result.conflicts_found.append({
			"conflict_type": "resource_conflict",
			"resource_type": resource_type,
			"resource_value": resource_value,
			"owners": [existing_owner, owner],
			"severity": "error",
			"suggested_resolution": suggested_resolution
		})

	def _validate_compatibility(self, result: ValidationResult) -> None:
		"""Validate compatibility between capabilities."""
		all_caps = result.capabilities_validated + result.resolved_dependencies

		# Check for known incompatible combinations
		incompatible_pairs = [
			# Example incompatible pairs (would be configured externally)
			# ('LEGACY_SYSTEM', 'MODERN_API'),
		]

		for cap1, cap2 in incompatible_pairs:
			if cap1 in all_caps and cap2 in all_caps:
				result.errors.append(ValidationError(
					error_type=ValidationErrorType.INCOMPATIBLE_CAPABILITIES,
					capability=cap1,
					message=f"Capabilities '{cap1}' and '{cap2}' are incompatible",
					details={'incompatible_with': cap2},
					suggested_resolution=f"Remove either '{cap1}' or '{cap2}' from the composition"
				))

	def _validate_security(self, result: ValidationResult) -> None:
		"""Validate security requirements and compatibility."""
		all_caps = result.capabilities_validated + result.resolved_dependencies

		# Check if security capability is included when needed
		security_required = False
		has_security = False

		for cap_code in all_caps:
			capability = self.registry.get_capability(cap_code)
			if capability:
				# Check if capability requires security
				if 'requires_authentication' in capability.composition_keywords:
					security_required = True

				# Check if security capability is present
				if cap_code == 'AUTH_RBAC' or 'security' in capability.composition_keywords:
					has_security = True

		if security_required and not has_security:
			result.warnings.append(ValidationWarning(
				warning_type=ValidationWarningType.COMPATIBILITY_CONCERN,
				capability="SYSTEM",
				message="Security-sensitive capabilities detected but no authentication system included",
				suggested_action="Consider adding AUTH_RBAC capability for security"
			))

	def _analyze_performance_impact(self, result: ValidationResult) -> None:
		"""Analyze potential performance impact of the composition."""
		all_caps = result.capabilities_validated + result.resolved_dependencies

		# Count database-heavy capabilities
		db_heavy_count = 0
		api_heavy_count = 0

		for cap_code in all_caps:
			capability = self.registry.get_capability(cap_code)
			if capability:
				# Count sub-capabilities with databases
				for subcap in capability.subcapabilities.values():
					if subcap.has_models and len(subcap.database_tables) > 5:
						db_heavy_count += 1
					if subcap.has_api:
						api_heavy_count += 1

		# Warn about potential performance issues
		if db_heavy_count > 10:
			result.warnings.append(ValidationWarning(
				warning_type=ValidationWarningType.PERFORMANCE_IMPACT,
				capability="SYSTEM",
				message=f"High number of database-heavy capabilities ({db_heavy_count}) may impact performance",
				suggested_action="Consider database optimization and indexing strategies"
			))

		if api_heavy_count > 20:
			result.warnings.append(ValidationWarning(
				warning_type=ValidationWarningType.PERFORMANCE_IMPACT,
				capability="SYSTEM",
				message=f"High number of API endpoints ({api_heavy_count}) may require load balancing",
				suggested_action="Consider API gateway and caching strategies"
			))

	def suggest_resolutions(self, result: ValidationResult) -> List[ConflictResolution]:
		"""Suggest automatic resolutions for conflicts."""
		resolutions = []

		for error in result.errors:
			if error.error_type == ValidationErrorType.MISSING_DEPENDENCY:
				# Suggest adding missing dependency
				resolution = ConflictResolution(
					conflict_id=error.id,
					strategy=ConflictResolutionStrategy.AUTO_ADD_DEPENDENCIES,
					actions=[f"Add dependency '{error.details.get('dependency')}'"],
					capabilities_added=[error.details.get('dependency', '')],
					rationale=f"Automatically add required dependency for {error.capability}"
				)
				resolutions.append(resolution)

			elif error.error_type == ValidationErrorType.VERSION_CONFLICT:
				# Suggest version update
				resolution = ConflictResolution(
					conflict_id=error.id,
					strategy=ConflictResolutionStrategy.USE_LATEST_VERSION,
					actions=[f"Update {error.details.get('dependency')} to compatible version"],
					configuration_changes={
						'version_overrides': {
							error.details.get('dependency'): error.details.get('required_version')
						}
					},
					rationale="Use version that satisfies all requirements"
				)
				resolutions.append(resolution)

		return resolutions

# Global validator instance
_validator_instance: Optional[DependencyValidator] = None

def get_validator() -> DependencyValidator:
	"""Get the global validator instance."""
	global _validator_instance
	if _validator_instance is None:
		_validator_instance = DependencyValidator()
	return _validator_instance
