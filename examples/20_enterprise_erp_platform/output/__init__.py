"""
enterprise_erp_platform - APG Generated Package
================================================

Version: 1.0.0

This package was automatically generated from APG source code.
"""

__version__ = "1.0.0"

from .app import approval_plan, auth_status, capability_configuration, capability_health, capability_health_report, capability_languages, capability_rules, capability_screens, capability_streaming, capability_theme, coerce_record_types, component_manifest, create_record, database_status, delete_record, describe_application, describe_capabilities, describe_capability, describe_workflow, describe_workflows, evaluate_capability_rules, get_record, get_workflow_run, invoke_agent, invoke_team, list_agent_teams, list_agents, list_capabilities, list_databases, list_entities, list_events, list_records, list_workflow_runs, list_workflows, main, metrics_snapshot, openapi_document, query_records, relationship_graph, resume_workflow, run_workflow, runtime_adapter_command_candidates, runtime_adapter_environment_keys, self_test, semantic_model, storage_status, theme_token, update_record, validate_agent_runtimes, validate_application, validate_capability_configuration, validate_component_manifest_contract, validate_openapi_contract, validate_route_dispatch_contract, validate_record

__all__ = [
    "__version__",
    "approval_plan",
    "auth_status",
    "capability_configuration",
    "capability_health",
    "capability_health_report",
    "capability_languages",
    "capability_rules",
    "capability_screens",
    "capability_streaming",
    "capability_theme",
    "coerce_record_types",
    "component_manifest",
    "create_record",
    "database_status",
    "delete_record",
    "describe_application",
    "describe_capabilities",
    "describe_capability",
    "describe_workflow",
    "describe_workflows",
    "evaluate_capability_rules",
    "get_record",
    "get_workflow_run",
    "invoke_agent",
    "invoke_team",
    "list_agent_teams",
    "list_agents",
    "list_capabilities",
    "list_databases",
    "list_entities",
    "list_events",
    "list_records",
    "list_workflow_runs",
    "list_workflows",
    "main",
    "metrics_snapshot",
    "openapi_document",
    "query_records",
    "relationship_graph",
    "resume_workflow",
    "run_workflow",
    "runtime_adapter_command_candidates",
    "runtime_adapter_environment_keys",
    "self_test",
    "semantic_model",
    "storage_status",
    "theme_token",
    "update_record",
    "validate_agent_runtimes",
    "validate_application",
    "validate_capability_configuration",
    "validate_component_manifest_contract",
    "validate_openapi_contract",
    "validate_route_dispatch_contract",
    "validate_record",
]

try:
    from .ai_agents import (
        get_agent,
        get_team,
        invoke_agent,
        invoke_team,
        list_agent_runtimes,
        list_agent_teams,
        list_agents,
        list_teams,
        runtime_adapter_command_candidates,
        runtime_adapter_environment_keys,
        validate_agent_runtimes,
    )
except ImportError:
    __all__ = list(__all__)
else:
    __all__.extend([
        "get_agent",
        "get_team",
        "invoke_agent",
        "invoke_team",
        "list_agent_runtimes",
        "list_agent_teams",
        "list_agents",
        "list_teams",
        "runtime_adapter_command_candidates",
        "runtime_adapter_environment_keys",
        "validate_agent_runtimes",
    ])

try:
    from .apg_application import (
        application_component_catalog,
        application_dependency_graph,
        application_route_index,
        application_screens,
        describe_application_composition,
        describe_application_compositions,
        get_application,
        list_applications,
        validate_application_compositions,
    )
except ImportError:
    __all__ = list(__all__)
else:
    __all__.extend([
        "application_component_catalog",
        "application_dependency_graph",
        "application_route_index",
        "application_screens",
        "describe_application_composition",
        "describe_application_compositions",
        "get_application",
        "list_applications",
        "validate_application_compositions",
    ])

try:
    from .apg_capabilities import (
        approval_plan,
        capability_dependency_graph,
        capability_configuration,
        capability_health,
        capability_health_report,
        capability_languages,
        capability_load_order,
        capability_rules,
        capability_screens,
        capability_streaming,
        capability_theme,
        evaluate_capability_rules,
        describe_capabilities,
        describe_capabilities_by_erp_module,
        describe_capability,
        african_language_codes,
        capability_names_by_erp_module,
        composition_graph,
        get_capability,
        list_capabilities,
        streaming_processor_index,
        streaming_state_index,
        supported_language_codes,
        theme_token,
        ui_route_index,
        validate_capability_configuration,
    )
except ImportError:
    __all__ = list(__all__)
else:
    __all__.extend([
        "approval_plan",
        "capability_dependency_graph",
        "capability_configuration",
        "capability_health",
        "capability_health_report",
        "capability_languages",
        "capability_load_order",
        "capability_rules",
        "capability_screens",
        "capability_streaming",
        "capability_theme",
        "evaluate_capability_rules",
        "describe_capabilities",
        "describe_capabilities_by_erp_module",
        "describe_capability",
        "african_language_codes",
        "capability_names_by_erp_module",
        "composition_graph",
        "get_capability",
        "list_capabilities",
        "streaming_processor_index",
        "streaming_state_index",
        "supported_language_codes",
        "theme_token",
        "ui_route_index",
        "validate_capability_configuration",
    ])