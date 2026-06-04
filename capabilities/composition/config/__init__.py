"""APG Central Configuration Management capability.

Standalone package: ``pip install apg-composition-config``

Quick start::

    from apg_composition_config import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_config
Provides      : configuration_namespace_registry, configuration_value_lifecycle, configuration_schema_validation, configuration_release_workflows, configuration_template_library, configuration_drift_monitoring
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-config"
__capability_id__ = "composition_config"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]

# ── Backward-compatibility stubs ──────────────────────────────────────────────
from typing import Any as _Any
from enum import Enum


class ConfigFormat(str, Enum):
    JSON = "json"; YAML = "yaml"; ENV = "env"; TOML = "toml"


class ConfigurationDSL:
    @staticmethod
    def parse(text: str, fmt: ConfigFormat = ConfigFormat.JSON) -> dict:
        import json
        try: return json.loads(text)
        except Exception: return {}


class ConfigurationScope(str, Enum):
    GLOBAL = "global"; TENANT = "tenant"; USER = "user"; SERVICE = "service"


class ConfigurationDataType(str, Enum):
    STRING = "string"; INTEGER = "integer"; FLOAT = "float"; BOOLEAN = "boolean"
    JSON = "json"; LIST = "list"


class ConfigurationField:
    def __init__(
        self,
        key: str,
        data_type: ConfigurationDataType = ConfigurationDataType.STRING,
        default: _Any = None,
        required: bool = False,
        description: str = "",
    ) -> None:
        self.key = key
        self.data_type = data_type
        self.default = default
        self.required = required
        self.description = description


class ConfigurationApplet:
    """Declarative configuration namespace for a service or capability."""

    def __init__(self, namespace: str, fields: list[ConfigurationField] | None = None) -> None:
        self.namespace = namespace
        self.fields: list[ConfigurationField] = fields or []

    def add_field(self, field: ConfigurationField) -> None:
        self.fields.append(field)


class CentralConfigurationManager:
    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id
        self._store: dict[str, _Any] = {}

    def get(self, key: str, default: _Any = None) -> _Any:
        return self._store.get(key, default)

    def set(self, key: str, value: _Any) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]


_managers: dict[str, CentralConfigurationManager] = {}
_applets: dict[str, ConfigurationApplet] = {}


def get_configuration_manager(tenant_id: str = "default") -> CentralConfigurationManager:
    if tenant_id not in _managers:
        _managers[tenant_id] = CentralConfigurationManager(tenant_id=tenant_id)
    return _managers[tenant_id]


def register_configuration_applet(applet: ConfigurationApplet) -> None:
    _applets[applet.namespace] = applet
