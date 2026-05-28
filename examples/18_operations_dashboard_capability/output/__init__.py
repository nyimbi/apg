"""
operations_dashboard_capability - APG Generated Package
========================================================

Version: 1.0.0

This package was automatically generated from APG source code.
"""

__version__ = "1.0.0"

from .app import auth_status, coerce_record_types, component_manifest, create_record, delete_record, describe_application, get_record, list_entities, list_events, list_records, main, metrics_snapshot, openapi_document, query_records, relationship_graph, self_test, storage_status, update_record, validate_application, validate_record

__all__ = [
    "__version__",
    "auth_status",
    "coerce_record_types",
    "component_manifest",
    "create_record",
    "delete_record",
    "describe_application",
    "get_record",
    "list_entities",
    "list_events",
    "list_records",
    "main",
    "metrics_snapshot",
    "openapi_document",
    "query_records",
    "relationship_graph",
    "self_test",
    "storage_status",
    "update_record",
    "validate_application",
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
        "validate_agent_runtimes",
    ])

try:
    from .apg_application import (
        application_component_catalog,
        application_dependency_graph,
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
        "describe_application_composition",
        "describe_application_compositions",
        "get_application",
        "list_applications",
        "validate_application_compositions",
    ])

try:
    from .apg_capabilities import (
        capability_dependency_graph,
        capability_load_order,
        capability_screens,
        capability_streaming,
        capability_theme,
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
    )
except ImportError:
    __all__ = list(__all__)
else:
    __all__.extend([
        "capability_dependency_graph",
        "capability_load_order",
        "capability_screens",
        "capability_streaming",
        "capability_theme",
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
    ])