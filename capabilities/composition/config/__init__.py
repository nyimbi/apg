"""APG central configuration capability package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .capability_contract import (
	CONFIG_EVENT_STREAM,
	SUPPORTED_CONFIG_AGENT_ROLES,
	SUPPORTED_CONFIG_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .models import (
	ConfigAgentRecord,
	ConfigAuditEventRecord,
	ConfigDeploymentRecord,
	ConfigDriftRecord,
	ConfigNamespaceRecord,
	ConfigTemplateRecord,
	ConfigurationRecord,
)
from .service import CentralConfigurationService, CompositionConfigService


class ConfigurationScope(str, Enum):
	"""Compatibility configuration scope values used by composition imports."""

	GLOBAL = "global"
	TENANT = "tenant"
	USER = "user"
	CAPABILITY = "capability"
	ENVIRONMENT = "environment"


class ConfigurationDataType(str, Enum):
	"""Compatibility configuration data types used by applet definitions."""

	STRING = "string"
	INTEGER = "integer"
	FLOAT = "float"
	BOOLEAN = "boolean"
	JSON = "json"
	ARRAY = "array"
	SECRET = "secret"
	FILE = "file"


@dataclass
class ConfigurationField:
	"""Configuration field definition for compatibility applets."""

	key: str
	label: str
	data_type: ConfigurationDataType
	default_value: Any = None
	required: bool = False
	description: str = ""
	validation_rules: dict[str, Any] | None = None
	depends_on: list[str] | None = None
	scope: ConfigurationScope = ConfigurationScope.TENANT


class ConfigurationApplet:
	"""Base class for capability-owned configuration applets."""

	applet_id: str = ""
	capability_name: str = ""
	display_name: str = ""
	description: str = ""

	def get_configuration_fields(self) -> list[ConfigurationField]:
		return []


class CentralConfigurationManager:
	"""Small compatibility manager for registering configuration applets."""

	def __init__(self) -> None:
		self._applets: dict[str, ConfigurationApplet] = {}

	def register_applet(self, applet: ConfigurationApplet) -> bool:
		if not applet.applet_id or applet.applet_id in self._applets:
			return False
		self._applets[applet.applet_id] = applet
		return True

	def unregister_applet(self, applet_id: str) -> bool:
		return self._applets.pop(applet_id, None) is not None

	def get_applets(self) -> list[ConfigurationApplet]:
		return list(self._applets.values())


_CONFIGURATION_MANAGER = CentralConfigurationManager()


def get_configuration_manager() -> CentralConfigurationManager:
	"""Return the process-local compatibility manager."""
	return _CONFIGURATION_MANAGER


def register_configuration_applet(applet: ConfigurationApplet) -> bool:
	"""Register a configuration applet with the process-local manager."""
	return _CONFIGURATION_MANAGER.register_applet(applet)


__version__ = "2.1.0"
__capability_id__ = "composition_config"
__apg_dependencies__ = ["auth", "audl", "ntfy", "registry", "composition_access"]
__apg_optional_dependencies__ = ["i18n", "mchn", "secrets"]


def register_capability() -> dict[str, object]:
	"""Return package metadata used by APG capability discovery."""
	contract = get_capability_contract()
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"version": __version__,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"CONFIG_EVENT_STREAM",
	"SUPPORTED_CONFIG_AGENT_ROLES",
	"SUPPORTED_CONFIG_AGENT_RUNTIMES",
	"CentralConfigurationService",
	"CentralConfigurationManager",
	"CompositionConfigService",
	"ConfigurationApplet",
	"ConfigurationDataType",
	"ConfigurationField",
	"ConfigurationScope",
	"ConfigAgentRecord",
	"ConfigAuditEventRecord",
	"ConfigDeploymentRecord",
	"ConfigDriftRecord",
	"ConfigNamespaceRecord",
	"ConfigTemplateRecord",
	"ConfigurationRecord",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_configuration_manager",
	"get_capability_contract",
	"register_capability",
	"register_configuration_applet",
	"streaming_manifest",
]
