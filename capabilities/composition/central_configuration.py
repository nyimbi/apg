"""Compatibility facade for composition central-configuration imports."""

from .config import (
	CentralConfigurationManager,
	ConfigurationApplet,
	ConfigurationDataType,
	ConfigurationField,
	ConfigurationScope,
	get_configuration_manager,
	register_configuration_applet,
)

__all__ = [
	"CentralConfigurationManager",
	"ConfigurationApplet",
	"ConfigurationField",
	"ConfigurationScope",
	"ConfigurationDataType",
	"get_configuration_manager",
	"register_configuration_applet",
]
