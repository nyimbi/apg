"""
APG Capability Composition Runtime
==================================

Generated from first-class APG capability declarations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    contract: Dict[str, Any]
    provides: List[str]
    requires: List[str]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    rule_engine: Dict[str, Any]
    ui: Dict[str, Any]
    theme: Dict[str, Any]
    runtime: Dict[str, Any]
    erp_modules: List[str]
    components: Any
    business_rules: List[Dict[str, Any]]
    approvals: Any
    master_data: Any
    i18n: Dict[str, Any]
    streaming: Dict[str, Any]
    screens: Any


CAPABILITY_DATA: Dict[str, Dict[str, Any]] = {'CreditControl': {'contract': {'id': 'credit_control', 'name': 'Credit Control', 'version': '1.0.0', 'provides': ['credit_limit_checks', 'hold_release'], 'requires': ['audit_events'], 'configuration': {'default_limit': 10000, 'currency': 'KES'}, 'rules': [{'name': 'over_limit', 'when': 'order_total > credit_limit', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'order_total', 'op': '>', 'value': 'credit_limit'}}, {'name': 'manual_review', 'when': 'risk_score > 0.75', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'risk_score', 'op': '>', 'value': 0.75}}]}, 'provides': ['credit_limit_checks', 'hold_release'], 'requires': ['audit_events'], 'configuration': {'default_limit': 10000, 'currency': 'KES'}, 'rules': [{'name': 'over_limit', 'when': 'order_total > credit_limit', 'action': 'deny', 'when_ast': {'type': 'compare', 'field': 'order_total', 'op': '>', 'value': 'credit_limit'}}, {'name': 'manual_review', 'when': 'risk_score > 0.75', 'action': 'require_review', 'when_ast': {'type': 'compare', 'field': 'risk_score', 'op': '>', 'value': 0.75}}], 'rule_engine': {}, 'ui': {}, 'theme': {}, 'runtime': {}, 'erp_modules': [], 'components': {}, 'business_rules': [], 'approvals': {'levels': 2, 'approvers': ['credit_manager', 'finance_controller']}, 'master_data': {'entities': ['customer_credit_profile']}, 'i18n': {}, 'streaming': {}, 'screens': {}}}
AFRICAN_LANGUAGE_CODES = {
    "af", "ak", "am", "ar", "bm", "bem", "ber", "bin", "din", "dyu",
    "ee", "ff", "fon", "gaa", "ha", "ig", "kab", "kam", "ki", "kln",
    "kg", "kj", "kmb", "kr", "lg", "ln", "loz", "lu", "lua", "mg",
    "mos", "nd", "nr", "nso", "ny", "om", "rn", "rw", "sg", "sn",
    "so", "ss", "st", "sw", "ti", "tn", "ts", "tum", "tw", "ve",
    "wo", "xh", "yo", "zu",
}
CORE_LANGUAGE_CODES = {
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr",
    "ru", "zh", "ja", "ko", "hi", "ur", "id", "ms",
}
SUPPORTED_LANGUAGE_CODES = CORE_LANGUAGE_CODES | AFRICAN_LANGUAGE_CODES
CAPABILITIES: Dict[str, CapabilitySpec] = {
    name: CapabilitySpec(name=name, **data)
    for name, data in CAPABILITY_DATA.items()
}

_MISSING = object()


def list_capabilities() -> List[str]:
    return sorted(CAPABILITIES)


def get_capability(name: str) -> CapabilitySpec:
    return CAPABILITIES[name]


def describe_capability(name: str) -> Dict[str, Any]:
    capability = get_capability(name)
    return {
        "name": capability.name,
        "contract": dict(capability.contract),
        "provides": list(capability.provides),
        "requires": list(capability.requires),
        "configuration": dict(capability.configuration),
        "rules": [dict(rule) for rule in capability.rules],
        "rule_engine": dict(capability.rule_engine),
        "ui": dict(capability.ui),
        "theme": dict(capability.theme),
        "runtime": dict(capability.runtime),
        "erp_modules": list(capability.erp_modules),
        "components": capability.components,
        "business_rules": [dict(rule) for rule in capability.business_rules],
        "approvals": capability.approvals,
        "master_data": capability.master_data,
        "i18n": dict(capability.i18n),
        "streaming": dict(capability.streaming),
        "screens": capability.screens,
    }


def describe_capabilities() -> Dict[str, Dict[str, Any]]:
    return {
        name: describe_capability(name)
        for name in list_capabilities()
    }


def supported_language_codes() -> List[str]:
    return sorted(SUPPORTED_LANGUAGE_CODES)


def african_language_codes() -> List[str]:
    return sorted(AFRICAN_LANGUAGE_CODES)


def capabilities_by_erp_module() -> Dict[str, List[CapabilitySpec]]:
    grouped: Dict[str, List[CapabilitySpec]] = {}
    for capability in CAPABILITIES.values():
        for module_name in capability.erp_modules:
            grouped.setdefault(module_name, []).append(capability)
    return grouped


def capability_names_by_erp_module() -> Dict[str, List[str]]:
    return {
        module_name: sorted(capability.name for capability in capabilities)
        for module_name, capabilities in sorted(capabilities_by_erp_module().items())
    }


def describe_capabilities_by_erp_module() -> Dict[str, List[Dict[str, Any]]]:
    return {
        module_name: [describe_capability(name) for name in capability_names]
        for module_name, capability_names in capability_names_by_erp_module().items()
    }


def provided_services() -> Dict[str, List[str]]:
    services: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        for service in capability.provides:
            services.setdefault(service, []).append(capability.name)
    return services


def service_providers(service_name: str) -> List[str]:
    return sorted(provided_services().get(service_name, []))


def required_services(capability_name: str) -> List[str]:
    return list(get_capability(capability_name).requires)


def capability_dependency_graph() -> Dict[str, List[str]]:
    providers = provided_services()
    graph: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        dependencies: List[str] = []
        for service in capability.requires:
            for provider in providers.get(service, []):
                if provider != capability.name and provider not in dependencies:
                    dependencies.append(provider)
        graph[capability.name] = sorted(dependencies)
    return graph


def unresolved_required_services() -> Dict[str, List[str]]:
    providers = provided_services()
    unresolved: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        missing = [
            service for service in capability.requires
            if service not in providers and service not in CAPABILITIES
        ]
        if missing:
            unresolved[capability.name] = sorted(missing)
    return unresolved


def capability_load_order() -> Dict[str, Any]:
    graph = capability_dependency_graph()
    visited: set[str] = set()
    visiting: set[str] = set()
    order: List[str] = []
    cycles: List[List[str]] = []

    def visit(name: str, stack: List[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            cycle_start = stack.index(name) if name in stack else 0
            cycles.append([*stack[cycle_start:], name])
            return
        visiting.add(name)
        for dependency in graph.get(name, []):
            visit(dependency, [*stack, name])
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for capability_name in sorted(CAPABILITIES):
        visit(capability_name, [])

    return {
        "order": order,
        "cycles": cycles,
        "unresolved": unresolved_required_services(),
    }


def validate_capability_dependencies() -> Dict[str, List[str]]:
    plan = capability_load_order()
    errors: List[str] = []
    warnings: List[str] = []
    for cycle in plan["cycles"]:
        errors.append("Capability dependency cycle: " + " -> ".join(cycle))
    for capability_name, services in plan["unresolved"].items():
        for service in services:
            warnings.append(f"{capability_name} requires external service {service}")
    return {"errors": errors, "warnings": warnings}


def validate_capability_contracts() -> Dict[str, Any]:
    providers = provided_services()
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        if not capability.contract:
            errors.append(f"{capability.name} is missing a contract")
        if not capability.provides:
            errors.append(f"{capability.name} does not provide any services")
        for service in capability.requires:
            if service not in providers and service not in CAPABILITIES:
                warnings.append(f"{capability.name} requires external service {service}")
        if len(set(capability.provides)) != len(capability.provides):
            errors.append(f"{capability.name} declares duplicate provided services")
        if len(set(capability.requires)) != len(capability.requires):
            errors.append(f"{capability.name} declares duplicate required services")
    return {"errors": errors, "warnings": warnings}


def capability_components(capability_name: str) -> Dict[str, Dict[str, Any]]:
    components = get_capability(capability_name).components
    if not isinstance(components, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for component_name, component_spec in components.items():
        if isinstance(component_spec, dict):
            normalized[str(component_name)] = dict(component_spec)
        else:
            normalized[str(component_name)] = {"value": component_spec}
    return normalized


def component_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for capability in CAPABILITIES.values():
        for component_name, component_spec in capability_components(capability.name).items():
            component_id = f"{capability.name}.{component_name}"
            permissions = component_spec.get("permissions", [])
            if isinstance(permissions, list):
                normalized_permissions = list(permissions)
            elif permissions:
                normalized_permissions = [str(permissions)]
            else:
                normalized_permissions = []
            catalog[component_id] = {
                "id": component_id,
                "capability": capability.name,
                "name": component_name,
                "service": component_spec.get("capability"),
                "permissions": normalized_permissions,
                "spec": component_spec,
            }
    return catalog


def component_permissions(capability_name: str, component_name: str) -> List[str]:
    component = component_catalog().get(f"{capability_name}.{component_name}")
    if component is None:
        return []
    return list(component["permissions"])


def component_service_bindings() -> Dict[str, List[str]]:
    bindings: Dict[str, List[str]] = {}
    for component_id, component in component_catalog().items():
        service = component.get("service")
        if service:
            bindings.setdefault(str(service), []).append(component_id)
    return {
        service: sorted(component_ids)
        for service, component_ids in sorted(bindings.items())
    }


def validate_component_contracts() -> Dict[str, List[str]]:
    provided = provided_services()
    errors: List[str] = []
    warnings: List[str] = []
    for component_id, component in component_catalog().items():
        service = component.get("service")
        if not service:
            warnings.append(f"{component_id} does not declare a service binding")
        elif service not in provided and service not in CAPABILITIES:
            warnings.append(f"{component_id} binds to external service {service}")
        for permission in component.get("permissions", []):
            if not permission:
                errors.append(f"{component_id} declares an empty permission")
    return {"errors": errors, "warnings": warnings}


def capability_configuration(capability_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(get_capability(capability_name).configuration or {})
    if overrides:
        _deep_merge(config, overrides)
    return config


def configuration_value(
    capability_name: str,
    key: str,
    default: Any = None,
    overrides: Dict[str, Any] | None = None,
) -> Any:
    return capability_configuration(capability_name, overrides).get(key, default)


def validate_capability_configuration(
    capability_name: str,
    configuration: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    config = capability_configuration(capability_name, configuration or {})
    schema = capability.contract.get("configuration_schema", {})
    required = schema.get("required", list(capability.configuration)) if isinstance(schema, dict) else list(capability.configuration)
    errors: List[str] = []
    warnings: List[str] = []
    for key in required:
        if key not in config:
            errors.append(f"{capability.name} missing required configuration {key}")
    for key in config:
        if capability.configuration and key not in capability.configuration:
            warnings.append(f"{capability.name} has undeclared configuration {key}")
    return {"errors": errors, "warnings": warnings, "configuration": config}


def approval_policy(capability_name: str) -> Dict[str, Any]:
    approvals = get_capability(capability_name).approvals
    if isinstance(approvals, dict):
        return {
            "levels": int(approvals.get("levels") or 0),
            "approvers": [str(approver) for approver in approvals.get("approvers", [])],
            "thresholds": dict(approvals.get("thresholds") or {}),
            "segregation_of_duties": bool(approvals.get("segregation_of_duties", False)),
            "escalation": approvals.get("escalation"),
        }
    if isinstance(approvals, list):
        return {"levels": len(approvals), "approvers": [str(item) for item in approvals], "thresholds": {}, "segregation_of_duties": False, "escalation": None}
    return {"levels": 0, "approvers": [], "thresholds": {}, "segregation_of_duties": False, "escalation": None}


def approval_plan(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    policy = approval_policy(capability_name)
    context = context or {}
    amount = context.get("amount")
    thresholds = policy.get("thresholds", {})
    levels = policy["levels"]
    if isinstance(amount, (int, float)):
        for threshold_name, threshold_value in thresholds.items():
            if isinstance(threshold_value, (int, float)) and amount >= threshold_value:
                levels = max(levels, int(str(threshold_name).split("_")[-1]) if str(threshold_name).split("_")[-1].isdigit() else levels)
    return {
        "capability": capability_name,
        "required": levels > 0,
        "levels": levels,
        "approvers": policy["approvers"][:levels] if levels else [],
        "segregation_of_duties": policy["segregation_of_duties"],
        "escalation": policy["escalation"],
    }


def master_data_entities(capability_name: str) -> List[str]:
    master_data = get_capability(capability_name).master_data
    if isinstance(master_data, dict):
        entities = master_data.get("entities", [])
        if isinstance(entities, list):
            return [str(entity) for entity in entities]
    if isinstance(master_data, list):
        return [str(entity) for entity in master_data]
    return []


def master_data_index() -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        for entity in master_data_entities(capability.name):
            index.setdefault(entity, []).append(capability.name)
    return {
        entity: sorted(capability_names)
        for entity, capability_names in index.items()
    }


def validate_master_data_contracts() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        entities = master_data_entities(capability.name)
        if not entities:
            warnings.append(f"{capability.name} does not declare master data entities")
        if len(set(entities)) != len(entities):
            errors.append(f"{capability.name} declares duplicate master data entities")
    return {"errors": errors, "warnings": warnings}


def capability_theme(capability_name: str, tenant_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    theme = dict(capability.theme or {})
    resolved = {
        "name": theme.get("name", f"{capability.name}_theme"),
        "tokens": dict(theme.get("tokens") or {}),
        "components": dict(theme.get("components") or {}),
        "allow_tenant_overrides": bool(theme.get("allow_tenant_overrides", True)),
    }
    if tenant_overrides and resolved["allow_tenant_overrides"]:
        _deep_merge(resolved, tenant_overrides)
    return resolved


def theme_token(
    capability_name: str,
    token_name: str,
    default: Any = None,
    tenant_overrides: Dict[str, Any] | None = None,
) -> Any:
    return capability_theme(capability_name, tenant_overrides)["tokens"].get(token_name, default)


def capability_languages(capability_name: str) -> List[str]:
    languages = get_capability(capability_name).i18n.get("supported_languages", [])
    if not isinstance(languages, list):
        return []
    return [str(language) for language in languages]


def resolve_language(capability_name: str, requested_language: str | None = None) -> str:
    capability = get_capability(capability_name)
    supported = capability_languages(capability_name)
    default_language = str(capability.i18n.get("default_language") or (supported[0] if supported else "en"))
    fallback_language = str(capability.i18n.get("fallback_language") or default_language)
    if requested_language and requested_language in supported:
        return requested_language
    if default_language in supported:
        return default_language
    if fallback_language in supported:
        return fallback_language
    return supported[0] if supported else fallback_language


def validate_capability_i18n() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for capability in CAPABILITIES.values():
        supported = capability_languages(capability.name)
        if not supported:
            warnings.append(f"{capability.name} does not declare supported languages")
            continue
        for language in supported:
            if language not in SUPPORTED_LANGUAGE_CODES:
                errors.append(f"{capability.name} unsupported language code {language}")
        default_language = capability.i18n.get("default_language")
        fallback_language = capability.i18n.get("fallback_language")
        if default_language and default_language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{capability.name} unknown default language {default_language}")
        if fallback_language and fallback_language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{capability.name} unknown fallback language {fallback_language}")
        if default_language and default_language not in supported:
            errors.append(f"{capability.name} default language {default_language} is not supported")
        if fallback_language and fallback_language not in supported:
            errors.append(f"{capability.name} fallback language {fallback_language} is not supported")
    return {"errors": errors, "warnings": warnings}


def capability_streaming(capability_name: str) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    runtime_streaming = capability.runtime.get("streaming", {})
    stream = dict(runtime_streaming) if isinstance(runtime_streaming, dict) else {}
    if isinstance(capability.streaming, dict):
        _deep_merge(stream, capability.streaming)
    return {
        "capability": capability.name,
        "processor": stream.get("processor", "bytewax"),
        "input": stream.get("input"),
        "output": stream.get("output"),
        "state": stream.get("state"),
        "window": stream.get("window"),
        "config": stream,
    }


def streaming_processor_index() -> Dict[str, List[str]]:
    processors: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        stream = capability_streaming(capability.name)
        processor = str(stream.get("processor") or "bytewax")
        processors.setdefault(processor, []).append(capability.name)
    return {
        processor: sorted(capability_names)
        for processor, capability_names in processors.items()
    }


def streaming_state_index() -> Dict[str, List[str]]:
    states: Dict[str, List[str]] = {}
    for capability in CAPABILITIES.values():
        state = capability_streaming(capability.name).get("state")
        if state:
            states.setdefault(str(state), []).append(capability.name)
    return {
        state: sorted(capability_names)
        for state, capability_names in states.items()
    }


def validate_streaming_contracts() -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    allowed_processors = {"bytewax", "bytewax_streams"}
    for capability in CAPABILITIES.values():
        stream = capability_streaming(capability.name)
        processor = str(stream.get("processor") or "")
        if processor not in allowed_processors:
            errors.append(f"{capability.name} uses unsupported stream processor {processor}")
        if not stream.get("state"):
            warnings.append(f"{capability.name} does not declare streaming state")
    return {"errors": errors, "warnings": warnings}


def capability_health(capability_name: str) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    errors: List[str] = []
    warnings: List[str] = []

    configuration = validate_capability_configuration(capability_name)
    errors.extend(configuration.get("errors", []))
    warnings.extend(configuration.get("warnings", []))

    languages = capability_languages(capability_name)
    default_language = capability.i18n.get("default_language")
    fallback_language = capability.i18n.get("fallback_language")
    if not languages:
        warnings.append(f"{capability.name} does not declare supported languages")
    for language in languages:
        if language not in SUPPORTED_LANGUAGE_CODES:
            errors.append(f"{capability.name} unsupported language code {language}")
    if default_language and default_language not in SUPPORTED_LANGUAGE_CODES:
        errors.append(f"{capability.name} unknown default language {default_language}")
    if fallback_language and fallback_language not in SUPPORTED_LANGUAGE_CODES:
        errors.append(f"{capability.name} unknown fallback language {fallback_language}")
    if default_language and default_language not in languages:
        errors.append(f"{capability.name} default language {default_language} is not supported")
    if fallback_language and fallback_language not in languages:
        errors.append(f"{capability.name} fallback language {fallback_language} is not supported")

    rules = capability_rules(capability_name)
    if not rules:
        warnings.append(f"{capability.name} does not declare capability rules")
    screens = capability_screens(capability_name)
    if not screens:
        warnings.append(f"{capability.name} does not declare UI screens")
    components = capability_components(capability_name)
    if not components:
        warnings.append(f"{capability.name} does not declare composable components")
    master_data = master_data_entities(capability_name)
    if not master_data:
        warnings.append(f"{capability.name} does not declare master data entities")

    stream = capability_streaming(capability_name)
    processor = str(stream.get("processor") or "")
    if processor not in {"bytewax", "bytewax_streams"}:
        errors.append(f"{capability.name} uses unsupported stream processor {processor}")
    if not stream.get("state"):
        warnings.append(f"{capability.name} does not declare streaming state")

    health = {
        "capability": capability.name,
        "status": "error" if errors else "warning" if warnings else "ok",
        "healthy": not errors,
        "errors": errors,
        "warnings": warnings,
        "configuration": configuration,
        "rules": {
            "count": len(rules),
            "names": [str(rule.get("name")) for rule in rules],
            "sample_evaluation": evaluate_capability_rules(capability_name, {}),
        },
        "approvals": approval_plan(capability_name, {}),
        "ui": {
            "screens": screens,
            "route_index": {
                route: screen
                for route, screen in ui_route_index().items()
                if screen.get("capability") == capability.name
            },
        },
        "theme": capability_theme(capability_name),
        "streaming": stream,
        "master_data": master_data,
        "languages": languages,
        "components": components,
    }
    return health


def capability_health_report() -> Dict[str, Any]:
    capabilities = {
        capability_name: capability_health(capability_name)
        for capability_name in list_capabilities()
    }
    errors = [
        f"{capability_name}: {error}"
        for capability_name, health in capabilities.items()
        for error in health.get("errors", [])
    ]
    warnings = [
        f"{capability_name}: {warning}"
        for capability_name, health in capabilities.items()
        for warning in health.get("warnings", [])
    ]
    return {
        "healthy": not errors,
        "errors": errors,
        "warnings": warnings,
        "capabilities": capabilities,
    }


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def capability_rules(capability_name: str) -> List[Dict[str, Any]]:
    capability = get_capability(capability_name)
    rules: List[Dict[str, Any]] = []
    for source, source_rules in (
        ("contract", capability.rules),
        ("business", capability.business_rules),
        ("engine", capability.rule_engine.get("rules", [])),
    ):
        if not isinstance(source_rules, list):
            continue
        for index, rule in enumerate(source_rules):
            if not isinstance(rule, dict):
                continue
            normalized = dict(rule)
            normalized.setdefault("name", f"{source}_rule_{index + 1}")
            normalized.setdefault("source", source)
            normalized.setdefault("priority", 0)
            if "condition" not in normalized and "when" in normalized:
                normalized["condition"] = normalized["when"]
            if "effect" not in normalized:
                action = normalized.get("action", "allow")
                normalized["effect"] = {
                    "decision": _decision_from_action(action),
                    "action": action,
                }
            rules.append(normalized)
    return sorted(rules, key=lambda rule: int(rule.get("priority") or 0), reverse=True)


def evaluate_capability_rules(capability_name: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    capability = get_capability(capability_name)
    context = dict(context or {})
    evaluation_context = dict(capability.configuration)
    _deep_merge(evaluation_context, context)
    matched: List[str] = []
    actions: List[Dict[str, Any]] = []
    decision = "allow"
    precedence = {"allow": 0, "audit": 1, "warn": 1, "require_review": 2, "deny": 3}
    for rule in capability_rules(capability_name):
        if not _matches_rule(rule, evaluation_context):
            continue
        matched.append(str(rule["name"]))
        effect = dict(rule.get("effect") or {})
        effect.setdefault("decision", _decision_from_action(effect.get("action", rule.get("action", "allow"))))
        effect.setdefault("rule", rule["name"])
        actions.append(effect)
        candidate = str(effect.get("decision") or "allow")
        if precedence.get(candidate, 0) > precedence.get(decision, 0):
            decision = candidate
    return {
        "decision": decision,
        "matched_rules": matched,
        "actions": actions,
        "context": context,
        "effective_context": evaluation_context,
    }


def _matches_rule(rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
    condition = rule.get("condition")
    if condition is None:
        return False
    if isinstance(condition, dict):
        for key, expected in condition.items():
            if _resolve_value(str(key), context) != expected:
                return False
        return True
    if isinstance(condition, bool):
        return condition
    return _evaluate_condition(str(condition), context)


def _evaluate_condition(expression: str, context: Dict[str, Any]) -> bool:
    expression = expression.strip()
    if not expression:
        return False
    if expression.startswith("not "):
        return not _evaluate_condition(expression[4:].strip(), context)
    lowered = expression.lower()
    if lowered.endswith(" missing"):
        target = expression[:-8].strip()
        resolved = _resolve_value(target, context)
        return _missing_context_reference(target, resolved, context) or resolved in (None, "")
    if lowered.endswith(" present"):
        target = expression[:-8].strip()
        resolved = _resolve_value(target, context)
        return not (_missing_context_reference(target, resolved, context) or resolved in (None, ""))
    for operator in ("!=", "==", ">=", "<=", ">", "<"):
        marker = f" {operator} "
        if marker not in expression:
            continue
        left_text, right_text = expression.split(marker, 1)
        left_key = left_text.strip()
        right_key = right_text.strip()
        left, left_missing = _resolve_rule_operand(left_key, context)
        right, right_missing = _resolve_rule_operand(right_key, context)
        if left_missing:
            return False
        if right_missing:
            return False
        if operator == "!=":
            return left != right
        if operator == "==":
            return left == right
        try:
            if operator == ">=":
                return left >= right
            if operator == "<=":
                return left <= right
            if operator == ">":
                return left > right
            if operator == "<":
                return left < right
        except TypeError:
            return False
    resolved = _resolve_value(expression, context)
    if _missing_context_reference(expression, resolved, context):
        return False
    return bool(resolved)


def _resolve_rule_operand(text: str, context: Dict[str, Any]) -> tuple[Any, bool]:
    value = _safe_expression_value(text, context)
    if value is not _MISSING:
        return value, False
    value = _resolve_value(text, context)
    return value, _missing_context_reference(text, value, context)


def _safe_expression_value(expression: str, context: Dict[str, Any]) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _MISSING
    try:
        return _eval_expression_node(tree.body, context)
    except (TypeError, ValueError, ZeroDivisionError):
        return _MISSING


def _eval_expression_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        value = _resolve_value(node.id, context)
        if _missing_context_reference(node.id, value, context):
            return _MISSING
        return value
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is None:
            return _MISSING
        value = _resolve_value(path, context)
        if _missing_context_reference(path, value, context):
            return _MISSING
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _eval_expression_node(node.operand, context)
        if operand is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        return _MISSING
    if isinstance(node, ast.BinOp):
        left = _eval_expression_node(node.left, context)
        right = _eval_expression_node(node.right, context)
        if left is _MISSING or right is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    return _MISSING


def _attribute_path(node: ast.AST) -> str | None:
    parts: List[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _missing_context_reference(text: str, resolved: Any, context: Dict[str, Any]) -> bool:
    if text in context or resolved != text:
        return False
    lowered = text.lower()
    if lowered in {"true", "false", "none", "null"}:
        return False
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return False
    try:
        int(text)
    except ValueError:
        try:
            float(text)
        except ValueError:
            return True
    return False


def _resolve_value(value: str, context: Dict[str, Any]) -> Any:
    value = value.strip()
    if value in context:
        return context[value]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"none", "null"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        integer_parse_failed = True
    try:
        return float(value)
    except ValueError:
        float_parse_failed = True
    current: Any = context
    for part in value.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return value
    return current


def _decision_from_action(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("decision", "allow"))
    action_text = str(action)
    if action_text in {"allow", "deny", "require_review", "warn", "audit"}:
        return action_text
    return "allow"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _screen_relationships(value: Any) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            relationships.append(dict(item))
            continue
        text = str(item).strip()
        if not text:
            continue
        relation = {"type": "relates_to"}
        if "->" in text:
            source, target = [part.strip() for part in text.split("->", 1)]
            relation.update({"from": source, "to": target})
        else:
            relation["to"] = text
        relationships.append(relation)
    return relationships


def _normalize_screen(
    capability: CapabilitySpec,
    name: str,
    spec: Any,
    index: int = 0,
) -> Dict[str, Any]:
    screen_spec = dict(spec) if isinstance(spec, dict) else {"component": spec or name}
    route = screen_spec.get("route", screen_spec.get("path", ""))
    component = screen_spec.get("component", name)
    return {
        "id": f"{capability.name}.{name}",
        "capability": capability.name,
        "name": name,
        "path": route,
        "route": route,
        "layout": screen_spec.get("layout"),
        "component": component,
        "contains": _as_list(screen_spec.get("contains")),
        "composes": _as_list(screen_spec.get("composes")),
        "binds": _as_list(screen_spec.get("binds")),
        "actions": _as_list(screen_spec.get("actions")),
        "events": _as_list(screen_spec.get("events")),
        "relationships": _screen_relationships(screen_spec.get("relationships")),
        "permission": screen_spec.get("permission"),
        "permissions": _as_list(screen_spec.get("permissions")),
        "rules": _as_list(screen_spec.get("rules")),
        "nav_group": screen_spec.get("nav_group"),
        "shell": capability.ui.get("shell"),
        "theme": screen_spec.get("theme", capability.theme.get("name")),
        "spec": screen_spec,
    }


def _declared_screen_specs(capability: CapabilitySpec) -> Any:
    if capability.screens:
        return capability.screens
    ui_screens = capability.ui.get("screens")
    return ui_screens if ui_screens else {}


def capability_screens(capability_name: str) -> List[Dict[str, Any]]:
    capability = get_capability(capability_name)
    screens: List[Dict[str, Any]] = []
    declared = _declared_screen_specs(capability)
    if isinstance(declared, dict):
        for index, (name, spec) in enumerate(declared.items()):
            screens.append(_normalize_screen(capability, str(name), spec, index))
    elif isinstance(declared, list):
        for index, item in enumerate(declared):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or item.get("component") or f"screen_{index + 1}")
                screens.append(_normalize_screen(capability, name, item, index))
            else:
                name = str(item)
                screens.append(_normalize_screen(capability, name, {"component": name}, index))

    known_names = {screen["name"] for screen in screens}
    routes = capability.ui.get("routes", [])
    if isinstance(routes, list):
        for index, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            name = str(route.get("name") or route.get("component") or f"screen_{index + 1}")
            if name in known_names:
                continue
            component = route.get("component", name)
            screens.append({
                "id": f"{capability.name}.{name}",
                "capability": capability.name,
                "name": name,
                "path": route.get("path", ""),
                "component": component,
                "permission": route.get("permission"),
                "nav_group": route.get("nav_group"),
                "shell": capability.ui.get("shell"),
                "theme": capability.theme.get("name"),
            })
    return screens


def ui_route_index() -> Dict[str, Dict[str, Any]]:
    routes: Dict[str, Dict[str, Any]] = {}
    for capability in CAPABILITIES.values():
        for screen in capability_screens(capability.name):
            path = screen.get("path")
            if path:
                routes[str(path)] = screen
    return routes


def composition_graph() -> Dict[str, List[Dict[str, Any]]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def node(node_id: str, kind: str, **attrs: Any) -> None:
        nodes[node_id] = {"id": node_id, "kind": kind, **attrs}

    def edge(source: str, target: str, relation: str) -> None:
        edges.append({"source": source, "target": target, "relation": relation})

    for capability in CAPABILITIES.values():
        cap_id = f"capability:{capability.name}"
        node(cap_id, "capability", name=capability.name)

        for service in capability.provides:
            service_id = f"service:{service}"
            node(service_id, "service", name=service)
            edge(cap_id, service_id, "provides")

        for service in capability.requires:
            service_id = f"service:{service}"
            node(service_id, "service", name=service)
            edge(cap_id, service_id, "requires")

        for module_name in capability.erp_modules:
            module_id = f"erp_module:{module_name}"
            node(module_id, "erp_module", name=module_name)
            edge(cap_id, module_id, "belongs_to")

        theme_name = capability.theme.get("name")
        if theme_name:
            theme_id = f"theme:{theme_name}"
            node(theme_id, "theme", name=theme_name)
            edge(cap_id, theme_id, "uses_theme")

        for screen in capability_screens(capability.name):
            screen_id = f"screen:{screen['id']}"
            node(screen_id, "screen", **screen)
            edge(cap_id, screen_id, "has_screen")
            component = screen.get("component")
            if component:
                component_id = f"component:{component}"
                node(component_id, "component", name=str(component))
                edge(screen_id, component_id, "renders")
            for contained in screen.get("contains", []):
                contained_id = f"component:{contained}"
                node(contained_id, "component", name=str(contained))
                edge(screen_id, contained_id, "contains")
            for composed in screen.get("composes", []):
                composed_id = f"component:{composed}"
                node(composed_id, "component", name=str(composed))
                edge(screen_id, composed_id, "composes")
            for binding in screen.get("binds", []):
                binding_id = f"binding:{binding}"
                node(binding_id, "binding", name=str(binding))
                edge(screen_id, binding_id, "binds_to")
            for relationship in screen.get("relationships", []):
                if not isinstance(relationship, dict):
                    continue
                source = relationship.get("from")
                target = relationship.get("to")
                if not source or not target:
                    continue
                source_id = f"component:{source}"
                target_id = f"component:{target}"
                relation = str(relationship.get("via") or relationship.get("type") or "relates_to")
                node(source_id, "component", name=str(source))
                node(target_id, "component", name=str(target))
                edge(source_id, target_id, relation)

        if isinstance(capability.components, dict):
            for component_name, component_spec in capability_components(capability.name).items():
                component_id = f"component:{component_name}"
                node(component_id, "component", name=str(component_name), spec=component_spec)
                edge(cap_id, component_id, "has_component")
                for permission in component_permissions(capability.name, component_name):
                    permission_id = f"permission:{permission}"
                    node(permission_id, "permission", name=str(permission))
                    edge(component_id, permission_id, "requires_permission")
                if component_spec.get("capability"):
                    service_id = f"service:{component_spec['capability']}"
                    node(service_id, "service", name=str(component_spec["capability"]))
                    edge(component_id, service_id, "binds_to")

        stream = capability_streaming(capability.name)
        processor = stream.get("processor")
        if processor:
            processor_id = f"stream_processor:{processor}"
            node(processor_id, "stream_processor", name=str(processor))
            edge(cap_id, processor_id, "streams_with")
        state = stream.get("state")
        if state:
            state_id = f"stream_state:{state}"
            node(state_id, "stream_state", name=str(state))
            edge(cap_id, state_id, "stores_stream_state")

    return {"nodes": sorted(nodes.values(), key=lambda item: item["id"]), "edges": edges}
