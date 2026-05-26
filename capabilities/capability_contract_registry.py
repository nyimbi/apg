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
		contract = module.get_capability_contract(tenant_id)
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
		return record.module.evaluate_capability_rules(context)
	return _evaluate_default(record.contract["rule_engine"]["rules"], context)


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
	_validate_rule_engine(contract["rule_engine"], source)
	_validate_ui(contract["ui"], source)
	_validate_theme(contract["theme"], source)


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


def _validate_ui(ui: Any, source: Path | str) -> None:
	if not isinstance(ui, dict):
		raise ValueError(f"{source} ui must be a dict")
	if ui.get("requires_theme") is not True:
		raise ValueError(f"{source} ui.requires_theme must be true")
	if not isinstance(ui.get("routes"), list) or not ui["routes"]:
		raise ValueError(f"{source} ui.routes must be a non-empty list")


def _validate_theme(theme: Any, source: Path | str) -> None:
	if not isinstance(theme, dict):
		raise ValueError(f"{source} theme must be a dict")
	if not isinstance(theme.get("tokens"), dict) or not theme["tokens"]:
		raise ValueError(f"{source} theme.tokens must be a non-empty dict")


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
