"""Discovery and validation registry for APG capability contracts."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


CONTRACT_FILENAME = "capability_contract.py"
REQUIRED_CONTRACT_KEYS = {"configuration", "configuration_schema", "rule_engine", "ui", "theme"}
REQUIRED_SCHEMA_KEYS = {"tenant_id", "ui", "theme"}
REQUIRED_RULE_KEYS = {"name", "condition", "effect"}
REQUIRED_ROUTE_KEYS = {"name", "path", "component", "permission"}
REQUIRED_THEME_TOKENS = {"border.radius"}
PYTHON_UI_SHELL = "apg_python"
LEGACY_UI_SHELL_ALIASES = {
	"flask_appbuilder",
	"fastapi_flask_appbuilder",
	"flask",
	"fastapi",
	"django",
}


@dataclass(frozen=True)
class CapabilityContractRecord:
	"""Loaded capability contract with its repository location."""

	capability_id: str
	display_name: str
	path: Path
	module_name: str
	contract: dict[str, Any]
	module: ModuleType


def discover_contract_paths(root: Path | str | None = None) -> list[Path]:
	"""Return all capability contract files under the capabilities tree."""
	base = Path(root) if root is not None else Path(__file__).resolve().parent
	return sorted(
		path for path in base.glob("**/" + CONTRACT_FILENAME)
		if path.name == CONTRACT_FILENAME and "__pycache__" not in path.parts
	)


def load_contract_registry(
	root: Path | str | None = None,
	tenant_id: str = "default",
) -> dict[str, CapabilityContractRecord]:
	"""Load every discovered capability contract, keyed by capability id."""
	records: dict[str, CapabilityContractRecord] = {}
	for path in discover_contract_paths(root):
		module = _load_contract_module(path)
		if not hasattr(module, "get_capability_contract"):
			raise ValueError(f"{path} does not expose get_capability_contract")
		contract = normalize_contract(module.get_capability_contract(tenant_id))
		validate_contract_shape(contract, path)
		capability_id = str(contract["capability"])
		records[capability_id] = CapabilityContractRecord(
			capability_id=capability_id,
			display_name=str(contract.get("display_name") or capability_id),
			path=path,
			module_name=module.__name__,
			contract=contract,
			module=module,
		)
	return records


def validate_contract_registry(
	root: Path | str | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Validate all discovered contracts and return a structured report."""
	errors: list[str] = []
	records: dict[str, CapabilityContractRecord] = {}
	for path in discover_contract_paths(root):
		try:
			module = _load_contract_module(path)
			if not hasattr(module, "get_capability_contract"):
				raise ValueError(f"{path} does not expose get_capability_contract")
			contract = normalize_contract(module.get_capability_contract(tenant_id))
			validate_contract_shape(contract, path)
			capability_id = str(contract["capability"])
			records[capability_id] = CapabilityContractRecord(
				capability_id=capability_id,
				display_name=str(contract.get("display_name") or capability_id),
				path=path,
				module_name=module.__name__,
				contract=contract,
				module=module,
			)
		except Exception as exc:
			errors.append(f"{path}: {exc}")
	return {
		"valid": not errors,
		"contract_count": len(records),
		"error_count": len(errors),
		"errors": errors,
		"capabilities": sorted(records),
	}


def get_contract(
	capability_id: str,
	root: Path | str | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Return one capability contract by id."""
	registry = load_contract_registry(root, tenant_id)
	if capability_id not in registry:
		raise KeyError(f"Unknown capability contract: {capability_id}")
	return registry[capability_id].contract


def evaluate_rules(
	capability_id: str,
	context: dict[str, Any],
	root: Path | str | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Evaluate a capability's deterministic rules through the registry."""
	record = load_contract_registry(root, tenant_id)[capability_id]
	if hasattr(record.module, "evaluate_capability_rules"):
		result = record.module.evaluate_capability_rules(context)
	else:
		result = _evaluate_default(record.contract["rule_engine"]["rules"], context)
	return _normalize_rule_evaluation_result(result, context)


def validate_contract_shape(contract: dict[str, Any], source: Path | str = "<contract>") -> None:
	"""Raise ValueError if a capability contract is missing executable APG surfaces."""
	missing = sorted(REQUIRED_CONTRACT_KEYS - set(contract))
	if missing:
		raise ValueError(f"{source} missing contract keys: {', '.join(missing)}")
	if not contract.get("capability"):
		raise ValueError(f"{source} missing capability id")
	if not isinstance(contract["configuration"], dict):
		raise ValueError(f"{source} configuration must be a dict")
	if not isinstance(contract["configuration_schema"], dict):
		raise ValueError(f"{source} configuration_schema must be a dict")
	_validate_configuration(contract["configuration"], contract["configuration_schema"], source)
	_validate_rule_engine(contract["rule_engine"], source)
	_validate_ui(contract["ui"], source)
	_validate_theme(contract["theme"], source)


def normalize_contract(contract: dict[str, Any]) -> dict[str, Any]:
	"""Return a runtime-normalized APG contract without mutating module globals."""
	normalized = _copy_contract(contract)
	ui = normalized.setdefault("ui", {})
	shell = ui.get("shell")
	if isinstance(shell, str) and shell.lower() in LEGACY_UI_SHELL_ALIASES:
		ui["legacy_shell"] = shell
		ui["shell"] = PYTHON_UI_SHELL
	return normalized


def _validate_configuration(configuration: Any, schema: Any, source: Path | str) -> None:
	if not isinstance(configuration.get("tenant_id"), str) or not configuration["tenant_id"]:
		raise ValueError(f"{source} configuration.tenant_id must be a non-empty string")
	required = set(schema.get("required", []))
	missing = sorted(REQUIRED_SCHEMA_KEYS - required)
	if missing:
		raise ValueError(f"{source} configuration_schema.required missing: {', '.join(missing)}")


def _load_contract_module(path: Path) -> ModuleType:
	module_name = "apg_contract_registry_" + "_".join(path.with_suffix("").parts[-4:])
	spec = importlib.util.spec_from_file_location(module_name, path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Cannot load capability contract: {path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


def _validate_rule_engine(rule_engine: Any, source: Path | str) -> None:
	if not isinstance(rule_engine, dict):
		raise ValueError(f"{source} rule_engine must be a dict")
	if rule_engine.get("type") != "deterministic":
		raise ValueError(f"{source} rule_engine.type must be deterministic")
	rules = rule_engine.get("rules")
	if not isinstance(rules, list) or not rules:
		raise ValueError(f"{source} rule_engine.rules must be a non-empty list")
	for index, rule in enumerate(rules):
		if not isinstance(rule, dict):
			raise ValueError(f"{source} rule_engine.rules[{index}] must be a dict")
		missing = sorted(REQUIRED_RULE_KEYS - set(rule))
		if missing:
			raise ValueError(f"{source} rule_engine.rules[{index}] missing: {', '.join(missing)}")
		if not isinstance(rule["name"], str) or not rule["name"]:
			raise ValueError(f"{source} rule_engine.rules[{index}].name must be a non-empty string")
		if not isinstance(rule["condition"], dict):
			raise ValueError(f"{source} rule_engine.rules[{index}].condition must be a dict")
		if not isinstance(rule["effect"], dict):
			raise ValueError(f"{source} rule_engine.rules[{index}].effect must be a dict")
		if not rule["effect"].get("decision"):
			raise ValueError(f"{source} rule_engine.rules[{index}].effect.decision is required")


def _validate_ui(ui: Any, source: Path | str) -> None:
	if not isinstance(ui, dict):
		raise ValueError(f"{source} ui must be a dict")
	if ui.get("requires_theme") is not True:
		raise ValueError(f"{source} ui.requires_theme must be true")
	if not isinstance(ui.get("shell"), str) or not ui["shell"]:
		raise ValueError(f"{source} ui.shell must be a non-empty string")
	if not isinstance(ui.get("template_roots"), list) or not ui["template_roots"]:
		raise ValueError(f"{source} ui.template_roots must be a non-empty list")
	if not isinstance(ui.get("routes"), list) or not ui["routes"]:
		raise ValueError(f"{source} ui.routes must be a non-empty list")
	for index, route in enumerate(ui["routes"]):
		if not isinstance(route, dict):
			raise ValueError(f"{source} ui.routes[{index}] must be a dict")
		missing = sorted(REQUIRED_ROUTE_KEYS - set(route))
		if missing:
			raise ValueError(f"{source} ui.routes[{index}] missing: {', '.join(missing)}")
		for key in REQUIRED_ROUTE_KEYS:
			if not isinstance(route[key], str) or not route[key]:
				raise ValueError(f"{source} ui.routes[{index}].{key} must be a non-empty string")
		if not route["path"].startswith("/"):
			raise ValueError(f"{source} ui.routes[{index}].path must start with /")


def _validate_theme(theme: Any, source: Path | str) -> None:
	if not isinstance(theme, dict):
		raise ValueError(f"{source} theme must be a dict")
	if not isinstance(theme.get("name"), str) or not theme["name"]:
		raise ValueError(f"{source} theme.name must be a non-empty string")
	if not isinstance(theme.get("tokens"), dict) or not theme["tokens"]:
		raise ValueError(f"{source} theme.tokens must be a non-empty dict")
	missing_tokens = sorted(REQUIRED_THEME_TOKENS - set(theme["tokens"]))
	if missing_tokens:
		raise ValueError(f"{source} theme.tokens missing: {', '.join(missing_tokens)}")
	if not isinstance(theme.get("components"), dict) or not theme["components"]:
		raise ValueError(f"{source} theme.components must be a non-empty dict")


def _evaluate_default(rules: Iterable[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in rules:
		if _matches(rule.get("condition", {}), context):
			matched.append(str(rule.get("name", "unnamed_rule")))
			effect = dict(rule.get("effect", {}))
			actions.append(effect)
			if effect.get("decision") == "deny":
				decision = "deny"
			elif effect.get("decision") == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _normalize_rule_evaluation_result(result: Any, context: dict[str, Any]) -> dict[str, Any]:
	"""Normalize package-specific rule evaluators to the registry result contract."""
	if not isinstance(result, dict):
		return {
			"decision": None,
			"matched_rules": [],
			"actions": [],
			"context": context,
			"errors": ["rule evaluator returned a non-object result"],
		}

	normalized = dict(result)
	matched_rules = normalized.get("matched_rules")
	if not isinstance(matched_rules, list):
		matched_rules = normalized.get("matched", [])
	if isinstance(matched_rules, list):
		normalized["matched_rules"] = [
			str(rule.get("name", "unnamed_rule")) if isinstance(rule, dict) else str(rule)
			for rule in matched_rules
		]
	else:
		normalized["matched_rules"] = []

	actions = normalized.get("actions")
	if not isinstance(actions, list):
		actions = normalized.get("effects", [])
	if isinstance(actions, list):
		normalized["actions"] = [
			dict(action) if isinstance(action, dict) else {"value": action}
			for action in actions
		]
	else:
		normalized["actions"] = []

	normalized.setdefault("context", context)
	return normalized


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _copy_contract(contract: dict[str, Any]) -> dict[str, Any]:
	"""Copy plain contract data while keeping dependency surface minimal."""
	copied: dict[str, Any] = {}
	for key, value in contract.items():
		if isinstance(value, dict):
			copied[key] = _copy_contract(value)
		elif isinstance(value, list):
			copied[key] = [_copy_contract(item) if isinstance(item, dict) else item for item in value]
		else:
			copied[key] = value
	return copied
