"""
support_agent - APG Python Application
======================================

Generated from APG source as dependency-free Python artifacts.
"""

from __future__ import annotations

import importlib
import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote


MODULE_NAME = 'support_agent'
MODULE_VERSION = '1.0.0'
MODULE_DESCRIPTION = 'Single support triage agent'
ENTITIES = [{'name': 'TriageAgent', 'type': 'ai_agent', 'properties': [], 'fields': [], 'methods': []}]
ENTITY_NAMES = {entity["name"] for entity in ENTITIES}
RECORD_STORE: Dict[str, list[Dict[str, Any]]] = {entity["name"]: [] for entity in ENTITIES}
NEXT_RECORD_IDS: Dict[str, int] = {entity["name"]: 1 for entity in ENTITIES}
EVENT_LOG: list[Dict[str, Any]] = []
NEXT_EVENT_ID = 1


def _optional_module(name: str) -> Optional[Any]:
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            package_import_failed = True
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


AI_AGENTS = _optional_module("ai_agents")
APG_APPLICATIONS = _optional_module("apg_application")
APG_CAPABILITIES = _optional_module("apg_capabilities")


def list_entities() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES]


def list_databases() -> list[Dict[str, Any]]:
    return [dict(entity) for entity in ENTITIES if entity.get("type") == "database"]


def database_status() -> Dict[str, Any]:
    databases = list_databases()
    schema_count = sum(len(database.get("schemas", [])) for database in databases)
    table_count = sum(
        len(schema.get("tables", []))
        for database in databases
        for schema in database.get("schemas", [])
    )
    reference_count = sum(
        1
        for database in databases
        for schema in database.get("schemas", [])
        for table in schema.get("tables", [])
        for column in table.get("columns", [])
        if isinstance(column, dict) and isinstance(column.get("reference"), dict)
    )
    validation = validate_database_schema_contracts()
    return {
        "valid": not validation["errors"],
        "database_count": len(databases),
        "schema_count": schema_count,
        "table_count": table_count,
        "reference_count": reference_count,
        "validation": validation,
    }


def list_records(entity_name: str | None = None) -> Dict[str, list[Dict[str, Any]]] | list[Dict[str, Any]]:
    if entity_name is None:
        return {
            name: [dict(record) for record in records]
            for name, records in RECORD_STORE.items()
    }
    return [dict(record) for record in RECORD_STORE[entity_name]]


def query_records(entity_name: str, query: Dict[str, list[str]] | None = None) -> Dict[str, Any]:
    query = query or {}
    records = list_records(entity_name)
    filters = {
        key.removeprefix("filter."): values[-1]
        for key, values in query.items()
        if values and key not in {"limit", "offset", "sort", "order"}
    }
    records = [
        record
        for record in records
        if all(str(record.get(field, "")) == str(expected) for field, expected in filters.items())
    ]
    sort_field = query.get("sort", [None])[-1]
    if sort_field:
        reverse = query.get("order", ["asc"])[-1].lower() == "desc"
        records = sorted(records, key=lambda record: str(record.get(sort_field, "")), reverse=reverse)
    total = len(records)
    try:
        offset = max(0, int(query.get("offset", ["0"])[-1]))
    except (TypeError, ValueError):
        offset = 0
    limit = query.get("limit", [None])[-1]
    try:
        parsed_limit = int(limit) if limit not in (None, "") else None
    except (TypeError, ValueError):
        parsed_limit = None
    if parsed_limit is not None:
        records = records[offset:offset + max(0, parsed_limit)]
    elif offset:
        records = records[offset:]
    return {
        "entity": entity_name,
        "records": records,
        "count": len(records),
        "total": total,
        "offset": offset,
        "limit": parsed_limit,
        "filters": filters,
        "sort": sort_field,
        "order": query.get("order", ["asc"])[-1],
    }


def get_record(entity_name: str, record_id: Any) -> tuple[int, Dict[str, Any]]:
    return _records_payload(f"/entities/{entity_name}/records/{record_id}")


def create_record(entity_name: str, record: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    return _create_record_payload(f"/entities/{entity_name}/records", {"record": record})


def update_record(
    entity_name: str,
    record_id: Any,
    record: Dict[str, Any],
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    payload: Dict[str, Any] = {"record": record}
    if expected_revision is not None:
        payload["expected_revision"] = expected_revision
    return _update_record_payload(f"/entities/{entity_name}/records/{record_id}", payload)


def delete_record(
    entity_name: str,
    record_id: Any,
    expected_revision: int | None = None,
) -> tuple[int, Dict[str, Any]]:
    path = f"/entities/{entity_name}/records/{record_id}"
    if expected_revision is not None:
        path = f"{path}?expected_revision={expected_revision}"
    return _delete_record_payload(path)


def _data_path() -> Path | None:
    raw_path = os.environ.get("APG_DATA_FILE") or os.environ.get("APG_DATA_PATH")
    if not raw_path:
        return None
    return Path(raw_path)


def _record_numeric_id(record: Dict[str, Any]) -> int | None:
    value = record.get("id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _sync_next_record_ids() -> None:
    for entity_name in ENTITY_NAMES:
        numeric_ids = [
            numeric_id
            for record in RECORD_STORE[entity_name]
            for numeric_id in [_record_numeric_id(record)]
            if numeric_id is not None
        ]
        NEXT_RECORD_IDS[entity_name] = max(numeric_ids, default=0) + 1


def _sync_next_event_id() -> None:
    global NEXT_EVENT_ID
    numeric_ids = [
        numeric_id
        for event in EVENT_LOG
        for numeric_id in [_record_numeric_id(event)]
        if numeric_id is not None
    ]
    NEXT_EVENT_ID = max(numeric_ids, default=0) + 1


def _load_record_store() -> None:
    path = _data_path()
    if path is None or not path.exists():
        return
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"APG could not load record data from {path}: {error}", file=sys.stderr)
        return
    if not isinstance(loaded, dict):
        return
    raw_records = loaded.get("records", loaded)
    if not isinstance(raw_records, dict):
        return
    for entity_name in ENTITY_NAMES:
        entity_records = raw_records.get(entity_name, [])
        if isinstance(entity_records, list):
            RECORD_STORE[entity_name] = [
                dict(record)
                for record in entity_records
                if isinstance(record, dict)
            ]
    raw_events = loaded.get("events", [])
    if isinstance(raw_events, list):
        EVENT_LOG.clear()
        EVENT_LOG.extend(dict(event) for event in raw_events if isinstance(event, dict))
    _sync_next_record_ids()
    _sync_next_event_id()


def _persist_record_store() -> str | None:
    path = _data_path()
    if path is None:
        return None
    payload = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "records": list_records(),
        "events": list_events(),
        "next_record_ids": dict(NEXT_RECORD_IDS),
        "next_event_id": NEXT_EVENT_ID,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, path)
    except OSError as error:
        return str(error)
    return None


def storage_status(include_records: bool = False) -> Dict[str, Any]:
    path = _data_path()
    status: Dict[str, Any] = {
        "mode": "file" if path is not None else "memory",
        "path": str(path) if path is not None else None,
    }
    if include_records:
        status["records"] = list_records()
        status["events"] = list_events()
    return status


def metrics_snapshot() -> Dict[str, Any]:
    record_counts = {
        entity_name: len(RECORD_STORE[entity_name])
        for entity_name in sorted(ENTITY_NAMES)
    }
    event_counts: Dict[str, int] = {}
    for event in EVENT_LOG:
        action = str(event.get("action", "unknown"))
        event_counts[action] = event_counts.get(action, 0) + 1
    return {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "entity_count": len(ENTITIES),
        "database_status": database_status(),
        "record_counts": record_counts,
        "total_records": sum(record_counts.values()),
        "event_count": len(EVENT_LOG),
        "event_counts": event_counts,
        "relationship_count": len(relationship_graph()["edges"]),
        "storage": storage_status(),
        "auth": auth_status(),
    }


def self_test() -> Dict[str, Any]:
    validation = validate_application()
    openapi = openapi_document()
    routes = sorted(openapi["paths"])
    metrics = metrics_snapshot()
    checks: Dict[str, Any] = {
        "validation": validation,
        "metrics": metrics,
        "route_count": len(routes),
        "entity_count": metrics["entity_count"],
    }
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_health_report"):
        checks["capability_health"] = APG_CAPABILITIES.capability_health_report()
    return {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "passed": validation["valid"],
        "status": "ok" if validation["valid"] else "warning",
        "checks": checks,
        "routes": routes,
    }


def component_manifest() -> Dict[str, Any]:
    app = describe_application()
    openapi = openapi_document()
    return {
        "kind": "apg.application",
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "target": "python",
        "composable": True,
        "interfaces": {
            "http": {
                "openapi": "/openapi.json",
                "paths": sorted(openapi["paths"]),
            },
            "python": {
                "package": MODULE_NAME,
                "exports": [
                    "auth_status",
                    "coerce_record_types",
                    "component_manifest",
                    "create_record",
                    "database_status",
                    "delete_record",
                    "describe_application",
                    "get_record",
                    "list_databases",
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
                    "validate_component_manifest_contract",
                    "validate_openapi_contract",
                    "validate_route_dispatch_contract",
                    "validate_record",
                ],
            },
            "records": sorted(ENTITY_NAMES),
            "theme": "/theme.css",
        },
        "entities": list_entities(),
        "databases": list_databases(),
        "ai_agents": app.get("ai_agents", []),
        "ai_agent_teams": app.get("ai_agent_teams", []),
        "application_compositions": app.get("application_compositions", []),
        "application_dependency_graph": app.get("application_dependency_graph", {}),
        "application_routes": app.get("application_routes", {}),
        "capabilities": app.get("capabilities", []),
        "ui_routes": app.get("ui_routes", {}),
        "streaming_processors": app.get("streaming_processors", {}),
        "deployment": {
            "artifacts": [
                "app.py",
                "__init__.py",
                "README.md",
                "requirements.txt",
                "Dockerfile",
                ".dockerignore",
                ".env.example",
                "smoke_test.py",
            ],
            "commands": {
                "run": "python app.py",
                "describe": "python app.py --describe",
                "validate": "python app.py --validate",
                "self_test": "python app.py --self-test",
                "smoke_test": "python smoke_test.py",
            },
            "environment": ["APG_HOST", "APG_PORT", "APG_DATA_FILE", "APG_API_KEY", "APG_DEBUG"],
        },
    }


def auth_status() -> Dict[str, Any]:
    return {
        "mode": "api_key" if os.environ.get("APG_API_KEY") else "open",
        "header": "Authorization: Bearer <key> or X-APG-API-Key" if os.environ.get("APG_API_KEY") else None,
    }


def _authorized(headers: Any) -> bool:
    required_key = os.environ.get("APG_API_KEY")
    if not required_key:
        return True
    supplied_key = headers.get("X-APG-API-Key")
    authorization = headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        supplied_key = authorization.removeprefix("Bearer ").strip()
    return supplied_key == required_key


def _auth_failure_payload() -> tuple[int, Dict[str, Any]]:
    return 401, {
        "error": "unauthorized",
        "message": "Set Authorization: Bearer <key> or X-APG-API-Key to mutate this APG app.",
    }


def list_events(entity_name: str | None = None) -> list[Dict[str, Any]]:
    events = [dict(event) for event in EVENT_LOG]
    if entity_name is None:
        return events
    return [event for event in events if event.get("entity") == entity_name]


def _record_event(
    action: str,
    entity_name: str,
    before: Dict[str, Any] | None = None,
    after: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    global NEXT_EVENT_ID
    record = after if after is not None else before if before is not None else {}
    event = {
        "id": NEXT_EVENT_ID,
        "action": action,
        "entity": entity_name,
        "record_id": record.get("id"),
    }
    if before is not None:
        event["before"] = dict(before)
    if after is not None:
        event["after"] = dict(after)
    NEXT_EVENT_ID += 1
    EVENT_LOG.append(event)
    return dict(event)


def _prepare_new_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(record)
    prepared.setdefault("_revision", 1)
    return prepared


def _expected_revision(payload: Dict[str, Any]) -> int | None:
    value = payload.get("expected_revision")
    if value is None and isinstance(payload.get("record"), dict):
        value = payload["record"].get("_revision")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _revision_conflict(existing: Dict[str, Any], expected_revision: int | None) -> Dict[str, Any] | None:
    current_revision = existing.get("_revision")
    if expected_revision is None or current_revision == expected_revision:
        return None
    return {
        "error": "revision_conflict",
        "expected_revision": expected_revision,
        "current_revision": current_revision,
        "record": dict(existing),
    }


def _record_schema(entity: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    fields = _field_specs(str(entity["name"]))
    if not fields:
        return {"type": "object", "additionalProperties": True}
    schema_properties: Dict[str, Any] = {
        "id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
        "_revision": {"type": "integer"},
    }
    required_fields: list[str] = []
    for field in fields:
        field_name = str(field["name"])
        schema_properties[field_name] = {"type": _json_schema_type(str(field.get("type", "any")))}
        if not partial and field.get("required", True):
            required_fields.append(field_name)
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": schema_properties,
    }
    if required_fields:
        schema["required"] = required_fields
    return schema


def _schema_ref(name: str) -> Dict[str, Any]:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_media(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"application/json": {"schema": schema}}


def _record_body_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "record": _schema_ref(schema_name),
        },
        "required": ["record"],
    }


def _record_import_body_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "records": {"type": "array", "items": _schema_ref(schema_name)},
        },
        "required": ["records"],
    }


def _record_list_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "records": {"type": "array", "items": _schema_ref(schema_name)},
            "count": {"type": "integer"},
            "total": {"type": "integer"},
            "filters": {"type": "object", "additionalProperties": {"type": "string"}},
            "sort": {"oneOf": [{"type": "string"}, {"type": "null"}]},
            "order": {"type": "string"},
        },
        "required": ["entity", "records", "count"],
    }


def _record_item_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "record": _schema_ref(schema_name),
        },
        "required": ["entity", "record"],
    }


def _record_mutation_response_schema(schema_name: str, record_key: str = "record") -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            record_key: _schema_ref(schema_name),
            "event": _schema_ref("EventRecord"),
        },
        "required": [record_key],
    }


def _record_export_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "records": {"type": "array", "items": _schema_ref(schema_name)},
            "count": {"type": "integer"},
        },
        "required": ["entity", "records", "count"],
    }


def _record_import_response_schema(schema_name: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "entity": {"type": "string"},
            "imported": {"type": "array", "items": _schema_ref(schema_name)},
            "errors": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "events": {"type": "array", "items": _schema_ref("EventRecord")},
            "count": {"type": "integer"},
            "failed": {"type": "integer"},
        },
        "required": ["entity", "imported", "errors", "count", "failed"],
    }


def _database_openapi_schemas() -> Dict[str, Any]:
    nullable_string = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    generic_object = {"type": "object", "additionalProperties": True}
    return {
        "ApplicationDescription": generic_object,
        "ComponentManifest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "kind": {"const": "apg.application"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "target": {"const": "python"},
                "composable": {"type": "boolean"},
                "interfaces": generic_object,
                "entities": {"type": "array", "items": generic_object},
                "databases": {"type": "array", "items": _schema_ref("DatabaseCatalogEntry")},
                "deployment": generic_object,
            },
            "required": ["kind", "name", "version", "target", "composable", "interfaces"],
        },
        "EntityCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entities": {"type": "array", "items": generic_object},
            },
            "required": ["entities"],
        },
        "RecordsByEntity": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "records": {"type": "object", "additionalProperties": {"type": "array", "items": generic_object}},
            },
            "required": ["records"],
        },
        "AuthStatus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string"},
                "header": nullable_string,
            },
            "required": ["mode", "header"],
        },
        "StorageStatus": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "mode": {"type": "string"},
                "path": nullable_string,
                "records": {"type": "object", "additionalProperties": {"type": "array", "items": generic_object}},
                "events": {"type": "array", "items": _schema_ref("EventRecord")},
            },
            "required": ["mode", "path"],
        },
        "ValidationReport": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "valid": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "checks": generic_object,
            },
            "required": ["name", "valid", "errors", "warnings", "checks"],
        },
        "HealthReport": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "valid": {"type": "boolean"},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "name", "version", "valid", "storage", "auth", "warnings"],
        },
        "EventLog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "events": {"type": "array", "items": _schema_ref("EventRecord")},
            },
            "required": ["events"],
        },
        "MetricsSnapshot": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "entity_count": {"type": "integer"},
                "database_status": _schema_ref("DatabaseStatus"),
                "record_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                "total_records": {"type": "integer"},
                "event_count": {"type": "integer"},
                "event_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                "relationship_count": {"type": "integer"},
                "storage": _schema_ref("StorageStatus"),
                "auth": _schema_ref("AuthStatus"),
            },
            "required": ["name", "version", "entity_count", "record_counts", "total_records", "event_count"],
        },
        "SelfTestReport": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "passed": {"type": "boolean"},
                "status": {"type": "string"},
                "checks": generic_object,
                "routes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "version", "passed", "status", "checks", "routes"],
        },
        "RelationshipNode": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["id", "name", "type"],
        },
        "RelationshipEdge": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "field": {"type": "string"},
                "relationship": {"type": "string"},
            },
            "required": ["from", "to", "relationship"],
        },
        "RelationshipGraph": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nodes": {"type": "array", "items": _schema_ref("RelationshipNode")},
                "edges": {"type": "array", "items": _schema_ref("RelationshipEdge")},
            },
            "required": ["nodes", "edges"],
        },
        "AgentCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "agents": generic_object,
                "teams": generic_object,
            },
            "required": ["agents", "teams"],
        },
        "ApplicationCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "applications": generic_object,
                "dependency_graph": generic_object,
                "components": generic_object,
            },
            "required": ["applications", "dependency_graph", "components"],
        },
        "CapabilityCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capabilities": generic_object,
                "by_erp_module": generic_object,
                "dependency_graph": generic_object,
                "load_order": {"oneOf": [generic_object, {"type": "array", "items": {"type": "string"}}]},
            },
            "required": ["capabilities", "by_erp_module", "dependency_graph", "load_order"],
        },
        "CapabilityHealth": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "status": {"type": "string"},
                "healthy": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "configuration": generic_object,
                "rules": generic_object,
                "approvals": generic_object,
                "ui": generic_object,
                "theme": generic_object,
                "streaming": generic_object,
                "master_data": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "components": generic_object,
            },
            "required": ["capability", "status", "healthy", "errors", "warnings"],
        },
        "CapabilityHealthReport": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "healthy": {"type": "boolean"},
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "capabilities": {"type": "object", "additionalProperties": _schema_ref("CapabilityHealth")},
            },
            "required": ["healthy", "errors", "warnings", "capabilities"],
        },
        "RouteCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "routes": generic_object,
            },
            "required": ["routes"],
        },
        "AgentInvocationRequest": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "message": {"type": "string"},
                "payload": generic_object,
                "context": generic_object,
            },
        },
        "AgentInvocationResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "agent": {"type": "string"},
                "team": {"type": "string"},
                "runtime": {"type": "string"},
                "status": {"type": "string"},
                "result": {"oneOf": [generic_object, {"type": "string"}, {"type": "null"}]},
                "payload": generic_object,
            },
        },
        "RuleEvaluationRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "context": generic_object,
            },
            "required": ["context"],
        },
        "RuleEvaluationResult": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "decision": {"type": "string"},
                "matched_rules": {"type": "array", "items": {"type": "string"}},
                "actions": {"type": "array", "items": generic_object},
                "context": generic_object,
            },
        },
        "CapabilityConfigurationRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "configuration": generic_object,
                "overrides": generic_object,
            },
        },
        "CapabilityConfigurationResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "configuration": generic_object,
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        "ApprovalPlanRequest": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "capability": {"type": "string"},
                "capability_name": {"type": "string"},
                "context": generic_object,
            },
            "required": ["context"],
        },
        "ApprovalPlanResponse": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "capability": {"type": "string"},
                "required": {"type": "boolean"},
                "levels": {"type": "integer"},
                "approvers": {"type": "array", "items": {"type": "string"}},
                "thresholds": generic_object,
                "segregation_of_duties": {"type": "boolean"},
                "escalation": {"oneOf": [{"type": "string"}, generic_object, {"type": "null"}]},
            },
        },
        "StreamingTopology": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "processor": {"type": "string"},
                "processors": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                "states": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                "streams": {"type": "object", "additionalProperties": generic_object},
            },
            "required": ["processor", "processors", "states", "streams"],
        },
        "CapabilityStreamingContract": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "processor": {"type": "string"},
                "state": {"type": "string"},
                "input": generic_object,
                "output": generic_object,
            },
        },
        "EventRecord": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "id": {"type": "integer"},
                "entity": {"type": "string"},
                "action": {"type": "string"},
                "record_id": {"oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]},
                "before": {"oneOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}]},
                "after": {"oneOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}]},
            },
            "required": ["id", "entity", "action"],
        },
        "DatabaseReference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string"},
                "relationship": {"type": "string"},
                "schema": {"type": "string"},
                "table": {"type": "string"},
                "column": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["table", "column"],
        },
        "DatabaseColumn": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "primary_key": {"type": "boolean"},
                "nullable": {"type": "boolean"},
                "default": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "integer"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "constraints": {"type": "array", "items": {"type": "string"}},
                "reference": {"oneOf": [_schema_ref("DatabaseReference"), {"type": "null"}]},
            },
            "required": ["name", "type", "primary_key", "nullable", "constraints"],
        },
        "DatabaseIndex": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": nullable_string,
                "columns": {"type": "array", "items": {"type": "string"}},
                "unique": {"type": "boolean"},
                "type": nullable_string,
            },
            "required": ["columns", "unique"],
        },
        "DatabaseTable": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "columns": {"type": "array", "items": _schema_ref("DatabaseColumn")},
                "indexes": {"type": "array", "items": _schema_ref("DatabaseIndex")},
            },
            "required": ["name", "columns", "indexes"],
        },
        "DatabaseSchema": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "tables": {"type": "array", "items": _schema_ref("DatabaseTable")},
            },
            "required": ["name", "tables"],
        },
        "DatabaseCatalogEntry": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "name": {"type": "string"},
                "type": {"const": "database"},
                "properties": {"type": "array", "items": {"type": "string"}},
                "connection_config": {"type": "object", "additionalProperties": True},
                "schemas": {"type": "array", "items": _schema_ref("DatabaseSchema")},
            },
            "required": ["name", "type", "schemas"],
        },
        "DatabaseCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "databases": {"type": "array", "items": _schema_ref("DatabaseCatalogEntry")},
            },
            "required": ["databases"],
        },
        "DatabaseSchemaCatalog": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "database": {"type": "string"},
                "schemas": {"type": "array", "items": _schema_ref("DatabaseSchema")},
            },
            "required": ["database", "schemas"],
        },
        "DatabaseValidation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "errors": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "validated_databases": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["errors", "warnings", "validated_databases"],
        },
        "DatabaseStatus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "valid": {"type": "boolean"},
                "database_count": {"type": "integer"},
                "schema_count": {"type": "integer"},
                "table_count": {"type": "integer"},
                "reference_count": {"type": "integer"},
                "validation": _schema_ref("DatabaseValidation"),
            },
            "required": [
                "valid",
                "database_count",
                "schema_count",
                "table_count",
                "reference_count",
                "validation",
            ],
        },
    }


def _api_operation(
    summary: str,
    description: str,
    status: str = "200",
    request_body: bool = False,
    request_schema: Dict[str, Any] | None = None,
    response_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {"description": description}
    if response_schema is not None:
        response["content"] = _json_media(response_schema)
    operation: Dict[str, Any] = {
        "summary": summary,
        "responses": {status: response},
    }
    if request_body:
        operation["requestBody"] = {"required": True}
        if request_schema is not None:
            operation["requestBody"]["content"] = _json_media(request_schema)
    return operation


def openapi_document() -> Dict[str, Any]:
    paths: Dict[str, Any] = {
        "/health": {"get": _api_operation("Application health", "Health report", response_schema=_schema_ref("HealthReport"))},
        "/component.json": {"get": _api_operation("Composable component manifest", "APG component manifest", response_schema=_schema_ref("ComponentManifest"))},
        "/manifest": {"get": _api_operation("Application manifest", "APG manifest", response_schema=_schema_ref("ApplicationDescription"))},
        "/openapi.json": {"get": _api_operation("OpenAPI contract", "OpenAPI 3.1 contract", response_schema={"type": "object", "additionalProperties": True})},
        "/validate": {"get": _api_operation("Application validation", "Validation report", response_schema=_schema_ref("ValidationReport"))},
        "/events": {"get": _api_operation("Record mutation events", "Event log", response_schema=_schema_ref("EventLog"))},
        "/auth": {"get": _api_operation("Authentication status", "Authentication mode", response_schema=_schema_ref("AuthStatus"))},
        "/metrics": {"get": _api_operation("Application metrics", "Runtime metrics", response_schema=_schema_ref("MetricsSnapshot"))},
        "/applications": {"get": _api_operation("Application compositions", "Application composition catalog", response_schema=_schema_ref("ApplicationCatalog"))},
        "/self-test": {"get": _api_operation("Application self-test", "Self-test report", response_schema=_schema_ref("SelfTestReport"))},
        "/theme.css": {"get": _api_operation("Generated visual theme stylesheet", "CSS theme stylesheet")},
        "/records": {"get": _api_operation("All entity records", "Records by entity", response_schema=_schema_ref("RecordsByEntity"))},
        "/entities": {"get": _api_operation("Entity catalog", "Generated entity metadata", response_schema=_schema_ref("EntityCatalog"))},
        "/databases": {"get": _api_operation("Database catalog", "Database schema and connection metadata", response_schema=_schema_ref("DatabaseCatalog"))},
        "/databases/status": {"get": _api_operation("Database validation status", "Database schema validation and counts", response_schema=_schema_ref("DatabaseStatus"))},
        "/relationships": {"get": _api_operation("Entity relationship graph", "Relationship graph", response_schema=_schema_ref("RelationshipGraph"))},
        "/storage": {"get": _api_operation("Record storage status", "Storage status", response_schema=_schema_ref("StorageStatus"))},
        "/agents": {"get": _api_operation("Agent catalog", "AI agent and team catalog", response_schema=_schema_ref("AgentCatalog"))},
        "/capabilities": {"get": _api_operation("Capability catalog", "Capability catalog", response_schema=_schema_ref("CapabilityCatalog"))},
        "/capabilities/health": {"get": _api_operation("Capability health report", "Capability health report", response_schema=_schema_ref("CapabilityHealthReport"))},
        "/routes": {"get": _api_operation("Generated UI route catalog", "UI route catalog", response_schema=_schema_ref("RouteCatalog"))},
        "/composition": {"get": _api_operation("Composition graph", "Composition graph", response_schema=_schema_ref("RelationshipGraph"))},
        "/ui": {"get": _api_operation("Generated application UI", "HTML application index")},
        "/ui/databases": {"get": _api_operation("Generated database catalog UI", "HTML database catalog")},
    }
    schemas: Dict[str, Any] = _database_openapi_schemas()
    for entity in ENTITIES:
        entity_name = str(entity["name"])
        schema_name = f"{entity_name}Record"
        patch_schema_name = f"{entity_name}RecordPatch"
        schemas[schema_name] = _record_schema(entity)
        schemas[patch_schema_name] = _record_schema(entity, partial=True)
        paths[f"/entities/{entity_name}/records"] = {
            "get": _api_operation(
                f"List {entity_name} records",
                "Record list",
                response_schema=_record_list_response_schema(schema_name),
            ),
            "post": _api_operation(
                f"Create {entity_name} record",
                "Created record",
                status="201",
                request_body=True,
                request_schema=_record_body_schema(schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records"]["get"]["parameters"] = [
            {"name": "filter.<field>", "in": "query", "required": False, "description": "Exact field filter"},
            {"name": "sort", "in": "query", "required": False, "description": "Field to sort by"},
            {"name": "order", "in": "query", "required": False, "description": "asc or desc"},
            {"name": "limit", "in": "query", "required": False, "description": "Maximum records to return"},
            {"name": "offset", "in": "query", "required": False, "description": "Records to skip"},
        ]
        paths[f"/entities/{entity_name}/records/export"] = {
            "get": _api_operation(
                f"Export {entity_name} records",
                "Record export",
                response_schema=_record_export_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records/import"] = {
            "post": _api_operation(
                f"Import {entity_name} records",
                "Record import",
                request_body=True,
                request_schema=_record_import_body_schema(schema_name),
                response_schema=_record_import_response_schema(schema_name),
            ),
        }
        paths[f"/entities/{entity_name}/records/{{id}}"] = {
            "get": _api_operation(
                f"Fetch {entity_name} record",
                "Record",
                response_schema=_record_item_response_schema(schema_name),
            ),
            "put": _api_operation(
                f"Update {entity_name} record",
                "Updated record",
                request_body=True,
                request_schema=_record_body_schema(patch_schema_name),
                response_schema=_record_mutation_response_schema(schema_name),
            ),
            "delete": _api_operation(
                f"Delete {entity_name} record",
                "Deleted record",
                response_schema=_record_mutation_response_schema(schema_name, record_key="deleted"),
            ),
        }
        paths[f"/ui/entities/{entity_name}"] = {
            "get": _api_operation(f"Generated {entity_name} UI", "HTML entity screen"),
        }
        if entity.get("type") == "database":
            paths[f"/databases/{entity_name}/schemas"] = {
                "get": _api_operation(f"{entity_name} database schemas", "Database schema metadata", response_schema=_schema_ref("DatabaseSchemaCatalog")),
            }
    if APG_CAPABILITIES is not None:
        paths["/rules/evaluate"] = {"post": _api_operation("Evaluate capability rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult"))}
        paths["/configuration/resolve"] = {"post": _api_operation("Resolve capability configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}
        paths["/configuration/validate"] = {"post": _api_operation("Validate capability configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse"))}
        paths["/approval/plan"] = {"post": _api_operation("Plan capability approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse"))}
        paths["/streaming"] = {"get": _api_operation("Streaming topology", "ByteWax streaming topology", response_schema=_schema_ref("StreamingTopology"))}
        if hasattr(APG_CAPABILITIES, "list_capabilities"):
            for capability_name in APG_CAPABILITIES.list_capabilities():
                paths[f"/capabilities/{capability_name}/streaming"] = {
                    "get": _api_operation(f"{capability_name} streaming contract", "Capability streaming contract", response_schema=_schema_ref("CapabilityStreamingContract")),
                }
                paths[f"/capabilities/{capability_name}/health"] = {
                    "get": _api_operation(f"{capability_name} health", "Capability health", response_schema=_schema_ref("CapabilityHealth")),
                }
                paths[f"/capabilities/{capability_name}/rules/evaluate"] = {
                    "post": _api_operation(f"Evaluate {capability_name} rules", "Rule decision", request_body=True, request_schema=_schema_ref("RuleEvaluationRequest"), response_schema=_schema_ref("RuleEvaluationResult")),
                }
                paths[f"/capabilities/{capability_name}/configuration/resolve"] = {
                    "post": _api_operation(f"Resolve {capability_name} configuration", "Resolved configuration", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }
                paths[f"/capabilities/{capability_name}/configuration/validate"] = {
                    "post": _api_operation(f"Validate {capability_name} configuration", "Configuration validation", request_body=True, request_schema=_schema_ref("CapabilityConfigurationRequest"), response_schema=_schema_ref("CapabilityConfigurationResponse")),
                }
                paths[f"/capabilities/{capability_name}/approval/plan"] = {
                    "post": _api_operation(f"Plan {capability_name} approvals", "Approval plan", request_body=True, request_schema=_schema_ref("ApprovalPlanRequest"), response_schema=_schema_ref("ApprovalPlanResponse")),
                }
        route_index = getattr(APG_CAPABILITIES, "ui_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {"get": _api_operation(f"Capability screen {route}", "Generated capability screen")}
    if AI_AGENTS is not None:
        for agent_name in describe_application().get("ai_agents", []):
            paths[f"/agents/{agent_name}/invoke"] = {
                "post": _api_operation(f"Invoke agent {agent_name}", "Agent invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }
        for team_name in describe_application().get("ai_agent_teams", []):
            paths[f"/agent-teams/{team_name}/invoke"] = {
                "post": _api_operation(f"Invoke agent team {team_name}", "Agent team invocation result", request_body=True, request_schema=_schema_ref("AgentInvocationRequest"), response_schema=_schema_ref("AgentInvocationResponse")),
            }
    if APG_APPLICATIONS is not None:
        route_index = getattr(APG_APPLICATIONS, "application_route_index", None)
        if route_index is not None:
            for route in sorted(route_index()):
                paths[str(route)] = {"get": _api_operation(f"Application route {route}", "Generated application composition screen")}
    return {
        "openapi": "3.1.0",
        "info": {
            "title": MODULE_NAME,
            "version": MODULE_VERSION,
            "description": MODULE_DESCRIPTION,
        },
        "paths": paths,
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-APG-API-Key"},
                "BearerAuth": {"type": "http", "scheme": "bearer"},
            },
        },
    }


def validate_component_manifest_contract() -> Dict[str, Any]:
    manifest = component_manifest()
    openapi = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    interfaces = manifest.get("interfaces", {})
    http = interfaces.get("http", {}) if isinstance(interfaces, dict) else {}
    python = interfaces.get("python", {}) if isinstance(interfaces, dict) else {}
    http_paths = sorted(http.get("paths", [])) if isinstance(http, dict) else []
    expected_paths = sorted(openapi.get("paths", {}))
    if http.get("openapi") != "/openapi.json":
        errors.append("component manifest HTTP interface must point to /openapi.json")
    if http_paths != expected_paths:
        errors.append("component manifest HTTP paths do not match OpenAPI paths")
    exports = python.get("exports", []) if isinstance(python, dict) else []
    if not isinstance(exports, list) or not exports:
        errors.append("component manifest Python interface does not declare exports")
        exports = []
    export_names: list[str] = []
    for export_name in exports:
        if not isinstance(export_name, str):
            errors.append("component manifest Python exports must be strings")
            continue
        export_names.append(export_name)
    missing_exports = [
        export_name
        for export_name in export_names
        if export_name not in globals() or not callable(globals()[export_name])
    ]
    for export_name in missing_exports:
        errors.append(f"component manifest Python export {export_name} is not callable")
    expected_record_names = sorted(ENTITY_NAMES)
    manifest_record_names = sorted(interfaces.get("records", [])) if isinstance(interfaces, dict) else []
    if manifest_record_names != expected_record_names:
        errors.append("component manifest record interface does not match generated entities")
    if interfaces.get("theme") != "/theme.css":
        errors.append("component manifest theme interface must point to /theme.css")
    deployment = manifest.get("deployment", {})
    artifacts = set(deployment.get("artifacts", [])) if isinstance(deployment, dict) else set()
    for artifact in ["app.py", "__init__.py", "README.md", "requirements.txt", "Dockerfile", ".dockerignore", ".env.example", "smoke_test.py"]:
        if artifact not in artifacts:
            errors.append(f"component manifest deployment is missing artifact {artifact}")
    commands = deployment.get("commands", {}) if isinstance(deployment, dict) else {}
    for command_name in ["run", "describe", "validate", "self_test", "smoke_test"]:
        if command_name not in commands:
            errors.append(f"component manifest deployment is missing command {command_name}")
    return {
        "errors": errors,
        "warnings": warnings,
        "http_path_count": len(http_paths),
        "python_exports": sorted(export_names),
        "artifact_count": len(artifacts),
    }


def _walk_openapi_refs(value: Any, path: str = "$") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        raw_ref = value.get("$ref")
        if isinstance(raw_ref, str):
            refs.append((path + ".$ref", raw_ref))
        for key, child in value.items():
            if key == "$ref":
                continue
            refs.extend(_walk_openapi_refs(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            refs.extend(_walk_openapi_refs(child, f"{path}[{index}]"))
    return refs


def validate_openapi_contract() -> Dict[str, Any]:
    document = openapi_document()
    errors: list[str] = []
    warnings: list[str] = []
    paths = document.get("paths", {})
    schemas = document.get("components", {}).get("schemas", {})
    if not isinstance(paths, dict) or not paths:
        errors.append("OpenAPI document does not declare paths")
        paths = {}
    if not isinstance(schemas, dict):
        errors.append("OpenAPI document components.schemas must be an object")
        schemas = {}
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            errors.append(f"OpenAPI path {route} must be an object")
            continue
        for method, operation in sorted(path_item.items()):
            if method.lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            if not isinstance(operation, dict):
                errors.append(f"OpenAPI operation {method.upper()} {route} must be an object")
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict) or not responses:
                errors.append(f"OpenAPI operation {method.upper()} {route} does not declare responses")
    referenced_schemas: set[str] = set()
    for ref_path, ref in _walk_openapi_refs(document):
        prefix = "#/components/schemas/"
        if not ref.startswith(prefix):
            errors.append(f"OpenAPI reference {ref} at {ref_path} is not an internal component schema reference")
            continue
        schema_name = ref[len(prefix):]
        referenced_schemas.add(schema_name)
        if schema_name not in schemas:
            errors.append(f"OpenAPI reference {ref} at {ref_path} does not resolve")
    return {
        "errors": sorted(errors),
        "warnings": warnings,
        "path_count": len(paths),
        "schema_count": len(schemas),
        "referenced_schemas": sorted(referenced_schemas),
    }


def _route_dispatch_target(route: str, method: str) -> str | None:
    method = method.lower()
    route = route.rstrip("/") or "/"
    if method == "get":
        if route == "/theme.css":
            return "theme_stylesheet"
        if route == "/ui" or route.startswith("/ui/"):
            return "_ui_payload"
        if _capability_screen(route) is not None:
            return "_capability_screen_payload"
        if _application_screen(route) is not None:
            return "_application_screen_payload"
        if route in {
            "/",
            "/manifest",
            "/application",
            "/component.json",
            "/health",
            "/validate",
            "/openapi.json",
            "/entities",
            "/databases",
            "/databases/status",
            "/auth",
            "/events",
            "/metrics",
            "/self-test",
            "/records",
            "/relationships",
            "/storage",
            "/agents",
            "/applications",
            "/capabilities",
            "/streaming",
            "/routes",
            "/composition",
        }:
            return "_route_payload"
        if route.startswith("/databases/") and route.endswith("/schemas"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/streaming"):
            return "_route_payload"
        if route.startswith("/capabilities/") and route.endswith("/health"):
            return "_route_payload"
        if route.startswith("/entities/") and "/records" in route:
            return "_records_payload_with_query"
        return None
    if method == "post":
        if route.startswith("/agents/") and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if (route.startswith("/agent-teams/") or route.startswith("/teams/")) and route.endswith(("/invoke", "/run")):
            return "_agent_invocation_payload"
        if route.startswith("/entities/") and (route.endswith("/records") or route.endswith("/records/import")):
            return "_create_record_payload"
        if route in {"/rules/evaluate", "/capabilities/rules/evaluate"} or (
            route.startswith("/capabilities/") and route.endswith("/rules/evaluate")
        ):
            return "_rule_evaluation_payload"
        if route in {"/configuration/resolve", "/capabilities/configuration/resolve"} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/resolve")
        ):
            return "_configuration_payload"
        if route in {"/configuration/validate", "/capabilities/configuration/validate"} or (
            route.startswith("/capabilities/") and route.endswith("/configuration/validate")
        ):
            return "_configuration_payload"
        if route in {"/approval/plan", "/capabilities/approval/plan"} or (
            route.startswith("/capabilities/") and route.endswith("/approval/plan")
        ):
            return "_approval_plan_payload"
        return None
    if method == "put":
        if route.startswith("/entities/") and "/records/{id}" in route:
            return "_update_record_payload"
        return None
    if method == "delete":
        if route.startswith("/entities/") and "/records/{id}" in route:
            return "_delete_record_payload"
        return None
    return None


def validate_route_dispatch_contract() -> Dict[str, Any]:
    document = openapi_document()
    paths = document.get("paths", {})
    errors: list[str] = []
    warnings: list[str] = []
    route_targets: Dict[str, list[Dict[str, str]]] = {}
    method_count = 0
    if not isinstance(paths, dict):
        return {
            "errors": ["OpenAPI paths must be an object before dispatch validation"],
            "warnings": warnings,
            "route_count": 0,
            "method_count": 0,
            "routes": route_targets,
        }
    for route, path_item in sorted(paths.items()):
        if not isinstance(path_item, dict):
            continue
        for method in sorted(path_item):
            method_name = str(method).lower()
            if method_name not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            method_count += 1
            target = _route_dispatch_target(str(route), method_name)
            if target is None:
                errors.append(f"OpenAPI route {method_name.upper()} {route} has no generated dispatcher")
                continue
            route_targets.setdefault(str(route), []).append({"method": method_name.upper(), "target": target})
    return {
        "errors": errors,
        "warnings": warnings,
        "route_count": len(paths),
        "method_count": method_count,
        "routes": route_targets,
    }


def describe_application() -> Dict[str, Any]:
    description: Dict[str, Any] = {
        "name": MODULE_NAME,
        "version": MODULE_VERSION,
        "description": MODULE_DESCRIPTION,
        "entities": list_entities(),
        "databases": list_databases(),
    }
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agents"] = AI_AGENTS.list_agents()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_agent") and hasattr(AI_AGENTS, "list_agents"):
        description["ai_agent_descriptions"] = {
            name: AI_AGENTS.describe_agent(name)
            for name in AI_AGENTS.list_agents()
        }
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_teams"] = AI_AGENTS.list_agent_teams()
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "describe_team") and hasattr(AI_AGENTS, "list_agent_teams"):
        description["ai_agent_team_descriptions"] = {
            name: AI_AGENTS.describe_team(name)
            for name in AI_AGENTS.list_agent_teams()
        }
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "list_applications"):
        description["application_compositions"] = APG_APPLICATIONS.list_applications()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "describe_application_compositions"):
        description["application_composition_descriptions"] = APG_APPLICATIONS.describe_application_compositions()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_dependency_graph"):
        description["application_dependency_graph"] = APG_APPLICATIONS.application_dependency_graph()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_component_catalog"):
        description["application_component_catalog"] = APG_APPLICATIONS.application_component_catalog()
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "application_route_index"):
        description["application_routes"] = APG_APPLICATIONS.application_route_index()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities"):
        description["capabilities"] = APG_CAPABILITIES.list_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities"):
        description["capability_descriptions"] = APG_CAPABILITIES.describe_capabilities()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "describe_capabilities_by_erp_module"):
        description["capability_descriptions_by_erp_module"] = APG_CAPABILITIES.describe_capabilities_by_erp_module()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_dependency_graph"):
        description["capability_dependency_graph"] = APG_CAPABILITIES.capability_dependency_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_load_order"):
        description["capability_load_order"] = APG_CAPABILITIES.capability_load_order()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "ui_route_index"):
        description["ui_routes"] = APG_CAPABILITIES.ui_route_index()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "composition_graph"):
        description["composition_graph"] = APG_CAPABILITIES.composition_graph()
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "streaming_processor_index"):
        description["streaming_processors"] = APG_CAPABILITIES.streaming_processor_index()
    return description


def _record_validation(report: Dict[str, Any], name: str, validation: Dict[str, Any]) -> None:
    check = dict(validation)
    errors = [str(error) for error in check.get("errors", [])]
    warnings = [str(warning) for warning in check.get("warnings", [])]
    report["checks"][name] = check
    report["errors"].extend(f"{name}: {error}" for error in errors)
    report["warnings"].extend(f"{name}: {warning}" for warning in warnings)


def validate_database_schema_contracts() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    validated: list[str] = []
    for database in list_databases():
        database_name = str(database.get("name", "database"))
        validated.append(database_name)
        schemas = database.get("schemas", [])
        if not schemas:
            warnings.append(f"{database_name} does not declare schemas")
            continue
        table_index: Dict[str, list[Dict[str, Any]]] = {}
        seen_schemas: set[str] = set()
        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            schema_key = schema_name.lower()
            if schema_key in seen_schemas:
                errors.append(f"{database_name} declares duplicate schema {schema_name}")
            seen_schemas.add(schema_key)
            seen_tables: set[str] = set()
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    errors.append(f"{database_name}.{schema_name} declares a table without a name")
                    continue
                table_key = table_name.lower()
                qualified_key = f"{schema_name}.{table_name}".lower()
                if table_key in seen_tables:
                    errors.append(f"{database_name}.{schema_name} declares duplicate table {table_name}")
                seen_tables.add(table_key)
                table_index.setdefault(table_key, []).append(table)
                table_index.setdefault(qualified_key, []).append(table)

        for schema in schemas:
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                columns = table.get("columns", [])
                column_names = [str(column.get("name", "")) for column in columns if isinstance(column, dict)]
                known_columns = {column_name.lower() for column_name in column_names if column_name}
                if len(known_columns) != len([column_name for column_name in column_names if column_name]):
                    errors.append(f"{database_name}.{schema_name}.{table_name} declares duplicate columns")
                if columns and not any(bool(column.get("primary_key")) for column in columns if isinstance(column, dict)):
                    warnings.append(f"{database_name}.{schema_name}.{table_name} does not declare a primary key")
                for index in table.get("indexes", []):
                    for indexed_column in index.get("columns", []):
                        if str(indexed_column).lower() not in known_columns:
                            errors.append(
                                f"{database_name}.{schema_name}.{table_name} index references unknown column {indexed_column}"
                            )
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    reference = column.get("reference")
                    if not isinstance(reference, dict):
                        continue
                    target_table_name = str(reference.get("table", ""))
                    target_column_name = str(reference.get("column", ""))
                    target_schema_name = str(reference.get("schema", ""))
                    target_label = (
                        f"{target_schema_name}.{target_table_name}"
                        if target_schema_name
                        else target_table_name
                    )
                    if target_schema_name:
                        candidates = table_index.get(f"{target_schema_name}.{target_table_name}".lower(), [])
                    else:
                        candidates = table_index.get(f"{schema_name}.{target_table_name}".lower(), [])
                        if not candidates:
                            candidates = table_index.get(target_table_name.lower(), [])
                    if not candidates:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references unknown table {target_label}"
                        )
                        continue
                    if len(candidates) > 1:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references ambiguous table {target_label}; use schema-qualified target"
                        )
                        continue
                    target_table = candidates[0]
                    target_columns = {
                        str(target_column.get("name", "")).lower()
                        for target_column in target_table.get("columns", [])
                        if isinstance(target_column, dict)
                    }
                    if target_column_name.lower() not in target_columns:
                        errors.append(
                            f"{database_name}.{schema_name}.{table_name}.{column.get('name')} references unknown column {target_label}.{target_column_name}"
                        )
    return {"errors": errors, "warnings": warnings, "validated_databases": sorted(validated)}


def validate_application(available_agent_runtimes: list[str] | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "name": MODULE_NAME,
        "valid": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }
    _record_validation(report, "openapi_contract", validate_openapi_contract())
    _record_validation(report, "component_manifest", validate_component_manifest_contract())
    _record_validation(report, "route_dispatch", validate_route_dispatch_contract())
    _record_validation(report, "database_schemas", validate_database_schema_contracts())
    if AI_AGENTS is not None and hasattr(AI_AGENTS, "validate_agent_runtimes"):
        _record_validation(
            report,
            "ai_agent_runtimes",
            AI_AGENTS.validate_agent_runtimes(available_agent_runtimes),
        )
    if APG_APPLICATIONS is not None and hasattr(APG_APPLICATIONS, "validate_application_compositions"):
        available_capabilities = APG_CAPABILITIES.list_capabilities() if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") else []
        available_agents = AI_AGENTS.list_agents() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agents") else []
        available_teams = AI_AGENTS.list_agent_teams() if AI_AGENTS is not None and hasattr(AI_AGENTS, "list_agent_teams") else []
        _record_validation(
            report,
            "application_compositions",
            APG_APPLICATIONS.validate_application_compositions(
                available_capabilities=available_capabilities,
                available_agents=available_agents,
                available_teams=available_teams,
            ),
        )
    if APG_CAPABILITIES is not None:
        for check_name, function_name in (
            ("capability_contracts", "validate_capability_contracts"),
            ("capability_dependencies", "validate_capability_dependencies"),
            ("component_contracts", "validate_component_contracts"),
            ("master_data_contracts", "validate_master_data_contracts"),
            ("capability_i18n", "validate_capability_i18n"),
            ("streaming_contracts", "validate_streaming_contracts"),
        ):
            validator = getattr(APG_CAPABILITIES, function_name, None)
            if validator is not None:
                _record_validation(report, check_name, validator())
    report["valid"] = not report["errors"]
    return report


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _css_name(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "value"


def theme_stylesheet() -> str:
    lines = [
        ":root {",
        "  --apg-accent: #126e82;",
        "  --apg-surface: #ffffff;",
        "  --apg-border: #d0d7de;",
        "  --apg-text: #1f2328;",
        "  --apg-muted: #59636e;",
        "}",
    ]
    if APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_theme"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            try:
                theme = APG_CAPABILITIES.capability_theme(capability_name)
            except KeyError:
                continue
            theme_name = _css_name(str(theme.get("name") or capability_name))
            tokens = theme.get("tokens", {})
            if isinstance(tokens, dict):
                for token_name, token_value in sorted(tokens.items()):
                    css_var = f"--apg-theme-{theme_name}-{_css_name(str(token_name))}"
                    lines.append(":root { " + css_var + ": " + str(token_value) + "; }")
                    if str(token_name).lower() in {"accent", "primary", "brand"}:
                        lines.append(":root { --apg-accent: var(" + css_var + "); }")
    lines.extend([
        "body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: var(--apg-text); background: #f6f8fa; line-height: 1.5; }",
        "body > * { max-width: 1100px; margin-left: auto; margin-right: auto; }",
        "h1 { margin-top: 24px; color: var(--apg-text); }",
        "h2 { margin-top: 24px; color: var(--apg-text); }",
        "nav { margin: 16px auto; padding: 10px 0; border-bottom: 1px solid var(--apg-border); }",
        "a { color: var(--apg-accent); text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "form { padding: 16px; background: var(--apg-surface); border: 1px solid var(--apg-border); border-radius: 8px; }",
        "label { display: block; margin: 8px 0; color: var(--apg-muted); }",
        "input { min-width: 280px; padding: 8px; border: 1px solid var(--apg-border); border-radius: 6px; }",
        "button { padding: 8px 12px; border: 1px solid var(--apg-accent); border-radius: 6px; background: var(--apg-accent); color: white; cursor: pointer; }",
        "pre { padding: 16px; overflow: auto; background: var(--apg-surface); border: 1px solid var(--apg-border); border-left: 4px solid var(--apg-accent); border-radius: 8px; }",
        "code { color: var(--apg-accent); }",
    ])
    return "\n".join(lines) + "\n"


def _html_page(title: str, body: str) -> str:
    safe_title = html.escape(title)
    return (
        "<!doctype html>"
        "<html><head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="stylesheet" href="/theme.css">'
        f"<title>{safe_title}</title>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )


def _entity_spec(entity_name: str) -> Dict[str, Any] | None:
    for entity in ENTITIES:
        if entity["name"] == entity_name:
            return dict(entity)
    return None


def _field_specs(entity_name: str) -> list[Dict[str, Any]]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return []
    fields = entity.get("fields") or []
    if fields:
        return [dict(field) for field in fields if isinstance(field, dict)]
    return [
        {"name": property_name, "type": "any", "required": True}
        for property_name in entity.get("properties", [])
    ]


def _json_schema_type(apg_type: str) -> str:
    normalized = apg_type.lower()
    if normalized in {"str", "string", "text", "varchar", "char", "email", "uuid", "date", "datetime", "timestamp"}:
        return "string"
    if normalized in {"int", "integer", "serial", "bigint", "smallint"}:
        return "integer"
    if normalized in {"float", "double", "decimal", "number", "numeric", "money"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"list", "array", "set"}:
        return "array"
    if normalized in {"dict", "map", "object", "json", "jsonb"}:
        return "object"
    return "string"


def _value_matches_type(value: Any, apg_type: str) -> bool:
    expected = _json_schema_type(apg_type)
    if value is None:
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _coerce_value_for_type(value: Any, apg_type: str) -> Any:
    if not isinstance(value, str):
        return value
    expected = _json_schema_type(apg_type)
    if expected == "integer":
        try:
            return int(value.strip())
        except ValueError:
            return value
    if expected == "number":
        try:
            return float(value.strip())
        except ValueError:
            return value
    if expected == "boolean":
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return value


def coerce_record_types(entity_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(record)
    for field in _field_specs(entity_name):
        field_name = str(field["name"])
        if field_name in coerced:
            coerced[field_name] = _coerce_value_for_type(
                coerced[field_name],
                str(field.get("type", "any")),
            )
    return coerced


def validate_record(entity_name: str, record: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    errors: list[str] = []
    fields = _field_specs(entity_name)
    for field in fields:
        field_name = str(field["name"])
        if not partial and field.get("required", True) and field_name not in record:
            errors.append(f"{field_name} is required")
            continue
        if field_name in record and not _value_matches_type(record[field_name], str(field.get("type", "any"))):
            errors.append(f"{field_name} must be {_json_schema_type(str(field.get('type', 'any')))}")
    return {
        "valid": not errors,
        "entity": entity_name,
        "errors": errors,
    }


def relationship_graph() -> Dict[str, Any]:
    nodes = [
        {"id": str(entity["name"]), "name": str(entity["name"]), "type": str(entity["type"])}
        for entity in ENTITIES
    ]
    table_nodes_by_name: Dict[str, list[str]] = {}
    for entity in ENTITIES:
        database_name = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                node_id = f"{database_name}.{schema_name}.{table_name}"
                nodes.append({
                    "id": node_id,
                    "name": table_name,
                    "type": "database_table",
                    "database": database_name,
                    "schema": schema_name,
                })
                table_nodes_by_name.setdefault(table_name.lower(), []).append(node_id)
                table_nodes_by_name.setdefault(f"{schema_name}.{table_name}".lower(), []).append(node_id)
    entity_names = {str(entity["name"]) for entity in ENTITIES}
    entity_names_by_lower = {name.lower(): name for name in entity_names}
    edges: list[Dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for entity in ENTITIES:
        source = str(entity["name"])
        for schema in entity.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            for table in schema.get("tables", []):
                table_name = str(table.get("name", ""))
                if not table_name:
                    continue
                table_node = f"{source}.{schema_name}.{table_name}"
                contains_key = (source, table_node, schema_name, "contains_table")
                if contains_key not in seen_edges:
                    edges.append({
                        "from": source,
                        "to": table_node,
                        "field": schema_name,
                        "relationship": "contains_table",
                    })
                    seen_edges.add(contains_key)
                for column in table.get("columns", []):
                    reference = column.get("reference") if isinstance(column, dict) else None
                    if not isinstance(reference, dict):
                        continue
                    target_table = str(reference.get("table", ""))
                    target_schema = str(reference.get("schema", ""))
                    if target_schema:
                        targets = table_nodes_by_name.get(f"{target_schema}.{target_table}".lower(), [])
                    else:
                        targets = table_nodes_by_name.get(f"{schema_name}.{target_table}".lower(), [])
                        if not targets:
                            targets = table_nodes_by_name.get(target_table.lower(), [])
                    target = targets[0] if len(targets) == 1 else None
                    if not target:
                        continue
                    edge_key = (
                        table_node,
                        target,
                        str(column.get("name", "")),
                        str(reference.get("relationship", "db_ref")),
                    )
                    if edge_key not in seen_edges:
                        edges.append({
                            "from": table_node,
                            "to": target,
                            "field": str(column.get("name", "")),
                            "relationship": str(reference.get("relationship", "db_ref")),
                            "target_column": str(reference.get("column", "")),
                        })
                        seen_edges.add(edge_key)
        for field in _field_specs(source):
            field_name = str(field["name"])
            field_type = str(field.get("type", ""))
            target = None
            relationship = "references"
            if field_type in entity_names:
                target = field_type
                relationship = "typed_as"
            elif field_type.lower() in entity_names_by_lower:
                target = entity_names_by_lower[field_type.lower()]
                relationship = "typed_as"
            elif field_name.endswith("_id"):
                candidate = field_name[:-3]
                target = entity_names_by_lower.get(candidate.lower())
            if target and target != source:
                edge_key = (source, target, field_name, relationship)
                if edge_key not in seen_edges:
                    edges.append({
                        "from": source,
                        "to": target,
                        "field": field_name,
                        "relationship": relationship,
                    })
                    seen_edges.add(edge_key)
    return {"nodes": nodes, "edges": edges}


def _ui_index_html() -> str:
    app = describe_application()
    entity_links = "".join(
        f'<li><a href="/ui/entities/{html.escape(entity["name"], quote=True)}">'
        f'{html.escape(entity["name"])}</a> '
        f'<code>{html.escape(entity["type"])}</code></li>'
        for entity in ENTITIES
    )
    if not entity_links:
        entity_links = "<li>No APG entities declared.</li>"
    database_links = "".join(
        f'<li><a href="/ui/databases">{html.escape(database["name"])}</a> '
        f'<code>{len(database.get("schemas", []))} schema(s)</code></li>'
        for database in app.get("databases", [])
    )
    if not database_links:
        database_links = "<li>No databases declared.</li>"
    application_route_links = "".join(
        f'<li><a href="{html.escape(route, quote=True)}">{html.escape(route)}</a> '
        f'<code>{html.escape(str(screen.get("application", "application")))}</code></li>'
        for route, screen in sorted(app.get("application_routes", {}).items())
    )
    if not application_route_links:
        application_route_links = "<li>No application routes declared.</li>"
    capability_route_links = "".join(
        f'<li><a href="{html.escape(route, quote=True)}">{html.escape(route)}</a> '
        f'<code>{html.escape(str(screen.get("capability", "capability")))}</code></li>'
        for route, screen in sorted(app.get("ui_routes", {}).items())
    )
    if not capability_route_links:
        capability_route_links = "<li>No capability screens declared.</li>"
    capability_links = "".join(
        f'<li><a href="/ui/capabilities/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("capabilities", [])
    )
    if not capability_links:
        capability_links = "<li>No capabilities declared.</li>"
    agent_links = "".join(
        f'<li><a href="/ui/agents/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("ai_agents", [])
    )
    if not agent_links:
        agent_links = "<li>No AI agents declared.</li>"
    team_links = "".join(
        f'<li><a href="/ui/agent-teams/{html.escape(name, quote=True)}">{html.escape(name)}</a></li>'
        for name in app.get("ai_agent_teams", [])
    )
    if not team_links:
        team_links = "<li>No AI agent teams declared.</li>"
    body = (
        f"<h1>{html.escape(MODULE_NAME)}</h1>"
        f"<p>{html.escape(MODULE_DESCRIPTION or 'Generated APG application')}</p>"
        '<nav><a href="/manifest">Manifest JSON</a> | '
        '<a href="/component.json">Component JSON</a> | '
        '<a href="/applications">Applications</a> | '
        '<a href="/capabilities">Capabilities</a> | '
        '<a href="/agents">Agents</a> | '
        '<a href="/events">Events</a> | '
        '<a href="/metrics">Metrics</a> | '
        '<a href="/self-test">Self-Test</a> | '
        '<a href="/records">Record JSON</a> | '
        '<a href="/ui/databases">Databases</a> | '
        '<a href="/relationships">Relationships</a> | '
        '<a href="/openapi.json">API Contract</a></nav>'
        "<h2>Application Routes</h2>"
        f"<ul>{application_route_links}</ul>"
        "<h2>Capability Screens</h2>"
        f"<ul>{capability_route_links}</ul>"
        "<h2>Entities</h2>"
        f"<ul>{entity_links}</ul>"
        "<h2>Databases</h2>"
        f"<ul>{database_links}</ul>"
        "<h2>Capabilities</h2>"
        f"<ul>{capability_links}</ul>"
        "<h2>AI Agents</h2>"
        f"<ul>{agent_links}</ul>"
        "<h2>AI Agent Teams</h2>"
        f"<ul>{team_links}</ul>"
    )
    return _html_page(MODULE_NAME, body)


def _ui_database_catalog_html() -> tuple[int, str]:
    status = database_status()
    status_code = 200 if status["valid"] else 422
    status_label = "valid" if status["valid"] else "invalid"
    database_items: list[str] = []
    for database in list_databases():
        database_name = str(database.get("name", "database"))
        schema_rows: list[str] = []
        for schema in database.get("schemas", []):
            schema_name = str(schema.get("name", "default"))
            table_names = ", ".join(
                html.escape(str(table.get("name", "table")))
                for table in schema.get("tables", [])
            ) or "no tables"
            schema_rows.append(
                f"<li><strong>{html.escape(schema_name)}</strong>: {table_names}</li>"
            )
        schemas_html = "".join(schema_rows) or "<li>No schemas declared.</li>"
        database_items.append(
            f"<section><h2>{html.escape(database_name)}</h2>"
            f'<p><a href="/databases/{html.escape(database_name, quote=True)}/schemas">'
            "Schema JSON</a></p>"
            f"<ul>{schemas_html}</ul></section>"
        )
    databases_html = "".join(database_items) or "<p>No databases declared.</p>"
    validation_html = html.escape(json.dumps(status["validation"], indent=2, sort_keys=True))
    body = (
        "<h1>Databases</h1>"
        f"<p>Status: <strong>{html.escape(status_label)}</strong>; "
        f"{status['database_count']} database(s), "
        f"{status['schema_count']} schema(s), "
        f"{status['table_count']} table(s), "
        f"{status['reference_count']} reference(s).</p>"
        '<nav><a href="/ui">Application UI</a> | '
        '<a href="/databases">Database JSON</a> | '
        '<a href="/databases/status">Status JSON</a> | '
        '<a href="/relationships">Relationships</a></nav>'
        f"{databases_html}"
        f"<h2>Validation</h2><pre>{validation_html}</pre>"
    )
    return status_code, _html_page("Databases", body)


def _ui_field_input_html(field: Dict[str, Any]) -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    safe_label = html.escape(field_name)
    expected = _json_schema_type(str(field.get("type", "any")))
    if expected == "boolean":
        return (
            f'<input type="hidden" name="{safe_name}" value="false">'
            f'<label>{safe_label} '
            f'<input type="checkbox" name="{safe_name}" value="true"></label><br>'
        )
    if expected == "integer":
        attributes = 'type="number" step="1"'
    elif expected == "number":
        attributes = 'type="number" step="any"'
    else:
        attributes = 'type="text"'
    return f'<label>{safe_label} <input name="{safe_name}" {attributes}></label><br>'


def _ui_entity_location(entity_name: str) -> str:
    return f"/ui/entities/{quote(entity_name, safe='')}"


def _ui_record_display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value)
    return str(value)


def _ui_record_editor_input_html(field: Dict[str, Any], record: Dict[str, Any], form_id: str) -> str:
    field_name = str(field["name"])
    safe_name = html.escape(field_name, quote=True)
    safe_form_id = html.escape(form_id, quote=True)
    expected = _json_schema_type(str(field.get("type", "any")))
    value = record.get(field_name)
    if expected == "boolean":
        checked = " checked" if value is True else ""
        return (
            f'<input form="{safe_form_id}" type="hidden" name="{safe_name}" value="false">'
            f'<input form="{safe_form_id}" type="checkbox" name="{safe_name}" value="true"{checked}>'
        )
    if expected == "integer":
        attributes = 'type="number" step="1"'
    elif expected == "number":
        attributes = 'type="number" step="any"'
    else:
        attributes = 'type="text"'
    safe_value = html.escape(_ui_record_display_value(value), quote=True)
    return f'<input form="{safe_form_id}" name="{safe_name}" value="{safe_value}" {attributes}>'


def _ui_query_value(query: Dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    return str(values[-1]) if values else ""


def _ui_records_query_form_html(entity_name: str, query: Dict[str, list[str]]) -> str:
    safe_entity_path = html.escape(quote(entity_name, safe=""), quote=True)
    fields = _field_specs(entity_name)
    filter_inputs = []
    for field in fields:
        field_name = str(field["name"])
        input_name = f"filter.{field_name}"
        safe_input_name = html.escape(input_name, quote=True)
        safe_label = html.escape(field_name)
        safe_value = html.escape(_ui_query_value(query, input_name), quote=True)
        filter_inputs.append(
            f'<label>{safe_label} <input type="text" name="{safe_input_name}" value="{safe_value}"></label>'
        )
    sort_options = ["", "id", "_revision"] + [
        str(field["name"]) for field in fields if str(field["name"]) not in {"id", "_revision"}
    ]
    selected_sort = _ui_query_value(query, "sort")
    sort_select = "".join(
        f'<option value="{html.escape(option, quote=True)}"{" selected" if option == selected_sort else ""}>'
        f'{html.escape(option or "none")}</option>'
        for option in sort_options
    )
    selected_order = (_ui_query_value(query, "order") or "asc").lower()
    order_select = "".join(
        f'<option value="{option}"{" selected" if option == selected_order else ""}>{option}</option>'
        for option in ["asc", "desc"]
    )
    limit_value = html.escape(_ui_query_value(query, "limit"), quote=True)
    offset_value = html.escape(_ui_query_value(query, "offset"), quote=True)
    filters = "".join(filter_inputs) or "<span>No fields available.</span>"
    return (
        f'<form method="get" action="/ui/entities/{safe_entity_path}">'
        f'<fieldset><legend>Query records</legend>'
        f"{filters}"
        f'<label>Sort <select name="sort">{sort_select}</select></label>'
        f'<label>Order <select name="order">{order_select}</select></label>'
        f'<label>Limit <input type="number" min="0" step="1" name="limit" value="{limit_value}"></label>'
        f'<label>Offset <input type="number" min="0" step="1" name="offset" value="{offset_value}"></label>'
        '<button type="submit">Apply</button> '
        f'<a href="/ui/entities/{safe_entity_path}">Reset</a>'
        '</fieldset></form>'
    )


def _ui_records_table_html(entity_name: str, records: list[Dict[str, Any]] | None = None) -> str:
    records = records if records is not None else list_records(entity_name)
    if not records:
        return "<p>No records yet.</p>"
    fields = _field_specs(entity_name)
    field_by_name = {str(field["name"]): field for field in fields}
    field_names = list(field_by_name)
    columns = ["id", "_revision"] + [
        field_name for field_name in field_names if field_name not in {"id", "_revision"}
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    safe_entity = html.escape(quote(entity_name, safe=""), quote=True)
    rows: list[str] = []
    for record in records:
        raw_record_id = str(record.get("id", ""))
        record_id = html.escape(quote(raw_record_id, safe=""), quote=True)
        form_id = f"apg-update-{entity_name}-{raw_record_id}"
        safe_form_id = html.escape(form_id, quote=True)
        cells = []
        for column in columns:
            if column in field_by_name:
                cell_value = _ui_record_editor_input_html(field_by_name[column], record, form_id)
            else:
                cell_value = html.escape(_ui_record_display_value(record.get(column)))
            cells.append(f"<td>{cell_value}</td>")
        revision = html.escape(str(record.get("_revision", "")), quote=True)
        action = (
            f'<form id="{safe_form_id}" method="post" action="/ui/entities/{safe_entity}/records/{record_id}"></form>'
            f'<input form="{safe_form_id}" type="hidden" name="expected_revision" value="{revision}">'
            f'<button form="{safe_form_id}" type="submit">Save</button> '
            f'<form method="post" action="/ui/entities/{safe_entity}/records/{record_id}/delete">'
            f'<input type="hidden" name="expected_revision" value="{revision}">'
            '<button type="submit">Delete</button>'
            '</form>'
        )
        rows.append(f"<tr>{''.join(cells)}<td>{action}</td></tr>")
    return f"<table><thead><tr>{header}<th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _ui_entity_html(entity_name: str, notice: str = "", query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    entity = _entity_spec(entity_name)
    if entity is None:
        return 404, _html_page("Unknown entity", f"<h1>Unknown entity: {html.escape(entity_name)}</h1>")
    query = query or {}
    query_result = query_records(entity_name, query)
    safe_entity = html.escape(entity_name, quote=True)
    fields = _field_specs(entity_name) or [{"name": "value", "type": "string", "required": True}]
    inputs = "".join(_ui_field_input_html(field) for field in fields)
    query_form = _ui_records_query_form_html(entity_name, query)
    records_table = _ui_records_table_html(entity_name, query_result["records"])
    records_json = html.escape(json.dumps(query_result["records"], indent=2, sort_keys=True))
    result_summary = (
        f'<p>Showing {query_result["count"]} of {query_result["total"]} matching records.</p>'
    )
    notice_html = f'<section role="alert"><strong>{html.escape(notice)}</strong></section>' if notice else ""
    body = (
        f'<nav><a href="/ui">Application</a> | '
        f'<a href="/entities/{safe_entity}/records">Record JSON</a></nav>'
        f"<h1>{html.escape(entity_name)}</h1>"
        f"<p><code>{html.escape(entity.get('type', 'entity'))}</code></p>"
        f"{notice_html}"
        f'<form method="post" action="/ui/entities/{safe_entity}/records">'
        f"{inputs}"
        '<button type="submit">Create record</button>'
        "</form>"
        "<h2>Records</h2>"
        f"{query_form}"
        f"{result_summary}"
        f"{records_table}"
        "<details><summary>Record JSON</summary>"
        f"<pre>{records_json}</pre>"
        "</details>"
    )
    return 200, _html_page(entity_name, body)


def _ui_error_message(response: Dict[str, Any]) -> str:
    errors = response.get("errors")
    if isinstance(errors, list) and errors:
        return "; ".join(str(error) for error in errors)
    if response.get("error") == "revision_conflict":
        return (
            "Revision conflict: record has revision "
            f"{response.get('current_revision')} but form submitted revision {response.get('expected_revision')}"
        )
    if "message" in response:
        return str(response["message"])
    if "error" in response:
        return str(response["error"])
    return "The submitted form could not be applied."


def _ui_error_payload(path: str, response: Dict[str, Any]) -> str:
    parts = [part for part in path.split("/") if part]
    message = _ui_error_message(response)
    if len(parts) >= 3 and parts[0] == "ui" and parts[1] == "entities":
        _status, body = _ui_entity_html(parts[2], notice=message)
        return body
    details = html.escape(json.dumps(response, indent=2, sort_keys=True))
    return _html_page("Form error", f"<h1>Form error</h1><p>{html.escape(message)}</p><pre>{details}</pre>")


def _ui_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, str]:
    parts = [part for part in path.split("/") if part]
    if parts == ["ui"]:
        return 200, _ui_index_html()
    if parts == ["ui", "databases"]:
        return _ui_database_catalog_html()
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "entities":
        return _ui_entity_html(parts[2], query=query)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "agents":
        return _ui_agent_console_html(parts[2])
    if len(parts) == 3 and parts[0] == "ui" and parts[1] in {"agent-teams", "teams"}:
        return _ui_agent_console_html(parts[2], team=True)
    if len(parts) == 3 and parts[0] == "ui" and parts[1] == "capabilities":
        return _ui_capability_console_html(parts[2])
    return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")


def _parse_json_object_field(form_record: Dict[str, Any], field_name: str) -> tuple[Dict[str, Any] | None, str | None]:
    raw_value = str(form_record.get(field_name) or "{}").strip() or "{}"
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        return None, f"{field_name} is invalid JSON: {error}"
    if not isinstance(value, dict):
        return None, f"{field_name} must be a JSON object"
    return value, None


def _result_section(result: Dict[str, Any] | None = None, error: str = "") -> str:
    if error:
        return f'<section role="alert"><strong>{html.escape(error)}</strong></section>'
    if result is None:
        return ""
    return "<h2>Result</h2><pre>" + html.escape(json.dumps(result, indent=2, sort_keys=True)) + "</pre>"


def _ui_agent_console_html(name: str, result: Dict[str, Any] | None = None, error: str = "", team: bool = False) -> tuple[int, str]:
    app = describe_application()
    catalog_key = "ai_agent_team_descriptions" if team else "ai_agent_descriptions"
    catalog = app.get(catalog_key, {})
    if name not in catalog:
        title = "Unknown agent team" if team else "Unknown agent"
        return 404, _html_page(title, f"<h1>{title}</h1><p>{html.escape(name)}</p>")
    action = f"/ui/{'agent-teams' if team else 'agents'}/{html.escape(name, quote=True)}/invoke"
    description = html.escape(json.dumps(catalog[name], indent=2, sort_keys=True))
    result_html = _result_section(result, error)
    body = (
        '<nav><a href="/ui">Application</a> | <a href="/agents">Agent catalog</a></nav>'
        f"<h1>{html.escape(name)}</h1>"
        f"<pre>{description}</pre>"
        f'<form method="post" action="{action}">'
        '<label>Message <input name="message" type="text"></label><br>'
        '<label>Payload JSON<br><textarea name="payload_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Invoke</button>'
        '</form>'
        f"{result_html}"
    )
    return 200, _html_page(name, body)


def _ui_capability_console_html(name: str, result: Dict[str, Any] | None = None, error: str = "") -> tuple[int, str]:
    app = describe_application()
    capabilities = app.get("capability_descriptions", {})
    if name not in capabilities:
        return 404, _html_page("Unknown capability", f"<h1>Unknown capability</h1><p>{html.escape(name)}</p>")
    description = html.escape(json.dumps(capabilities[name], indent=2, sort_keys=True))
    safe_name = html.escape(name, quote=True)
    result_html = _result_section(result, error)
    body = (
        '<nav><a href="/ui">Application</a> | <a href="/capabilities">Capability catalog</a></nav>'
        f"<h1>{html.escape(name)}</h1>"
        f"<pre>{description}</pre>"
        f'<form method="post" action="/ui/capabilities/{safe_name}/rules/evaluate">'
        '<h2>Evaluate Rules</h2>'
        '<label>Context JSON<br><textarea name="context_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Evaluate</button>'
        '</form>'
        f'<form method="post" action="/ui/capabilities/{safe_name}/configuration/resolve">'
        '<h2>Resolve Configuration</h2>'
        '<label>Overrides JSON<br><textarea name="configuration_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Resolve</button>'
        '</form>'
        f'<form method="post" action="/ui/capabilities/{safe_name}/approval/plan">'
        '<h2>Plan Approval</h2>'
        '<label>Context JSON<br><textarea name="context_json" rows="8" cols="80">{}</textarea></label><br>'
        '<button type="submit">Plan</button>'
        '</form>'
        f"{result_html}"
    )
    return 200, _html_page(name, body)


def _ui_post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    parts = [part for part in path.split("/") if part]
    raw_form_record = payload.get("record", payload)
    form_record = dict(raw_form_record) if isinstance(raw_form_record, dict) else {}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "agents" and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error)
            return 400, {"html": html_payload}
        message = form_record.get("message")
        if message:
            request_payload["message"] = message
        status, result = _agent_invocation_payload(f"/agents/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(parts[2], result=result if status == 200 else None, error="" if status == 200 else result.get("error", "agent invocation failed"))
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] in {"agent-teams", "teams"} and parts[3] == "invoke":
        request_payload, error = _parse_json_object_field(form_record, "payload_json")
        if error:
            _status, html_payload = _ui_agent_console_html(parts[2], error=error, team=True)
            return 400, {"html": html_payload}
        message = form_record.get("message")
        if message:
            request_payload["message"] = message
        status, result = _agent_invocation_payload(f"/agent-teams/{parts[2]}/invoke", request_payload)
        _status, html_payload = _ui_agent_console_html(parts[2], result=result if status == 200 else None, error="" if status == 200 else result.get("error", "team invocation failed"), team=True)
        return status, {"html": html_payload}
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "capabilities":
        capability_name = parts[2]
        operation = "/".join(parts[3:])
        if operation == "rules/evaluate":
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _rule_evaluation_payload(f"/capabilities/{capability_name}/rules/evaluate", {"context": context})
        elif operation == "configuration/resolve":
            configuration, error = _parse_json_object_field(form_record, "configuration_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _configuration_payload(f"/capabilities/{capability_name}/configuration/resolve", {"overrides": configuration})
        elif operation == "approval/plan":
            context, error = _parse_json_object_field(form_record, "context_json")
            if error:
                _status, html_payload = _ui_capability_console_html(capability_name, error=error)
                return 400, {"html": html_payload}
            status, result = _approval_plan_payload(f"/capabilities/{capability_name}/approval/plan", {"context": context})
        else:
            return 404, {"error": "not_found", "path": path}
        _status, html_payload = _ui_capability_console_html(
            capability_name,
            result=result if status == 200 else None,
            error="" if status == 200 else result.get("error", "capability operation failed"),
        )
        return status, {"html": html_payload}
    if len(parts) == 4 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        status, response = _create_record_payload(f"/entities/{entity_name}/records", payload)
        if status == 201:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    if len(parts) == 5 and parts[0] == "ui" and parts[1] == "entities" and parts[3] == "records":
        entity_name = parts[2]
        record_id = parts[4]
        expected_revision = form_record.pop("expected_revision", None)
        status, response = _update_record_payload(
            f"/entities/{entity_name}/records/{record_id}",
            {"record": form_record, "expected_revision": expected_revision},
        )
        if status == 200:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    if (
        len(parts) == 6
        and parts[0] == "ui"
        and parts[1] == "entities"
        and parts[3] == "records"
        and parts[5] == "delete"
    ):
        entity_name = parts[2]
        record_id = parts[4]
        delete_path = f"/entities/{entity_name}/records/{record_id}"
        expected_revision = form_record.get("expected_revision")
        if expected_revision not in (None, ""):
            delete_path = f"{delete_path}?expected_revision={quote(str(expected_revision), safe='')}"
        status, response = _delete_record_payload(delete_path)
        if status == 200:
            return 303, {"location": _ui_entity_location(entity_name)}
        return status, response
    return 404, {"error": "not_found", "path": path}


def _capability_screen(path: str) -> Dict[str, Any] | None:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "ui_route_index"):
        return None
    routes = APG_CAPABILITIES.ui_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _capability_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Capability screen")
    capability = str(screen.get("capability") or "")
    component = str(screen.get("component") or title)
    theme_name = str(screen.get("theme") or "")
    theme_tokens: Dict[str, Any] = {}
    if capability and APG_CAPABILITIES is not None and hasattr(APG_CAPABILITIES, "capability_theme"):
        try:
            theme_tokens = APG_CAPABILITIES.capability_theme(capability).get("tokens", {})
        except KeyError:
            theme_tokens = {}
    actions = "".join(
        f"<li>{html.escape(str(action))}</li>"
        for action in screen.get("actions", [])
    ) or "<li>No actions declared.</li>"
    relationships = html.escape(json.dumps(screen.get("relationships", []), indent=2, sort_keys=True))
    tokens = html.escape(json.dumps(theme_tokens, indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>Capability:</strong> {html.escape(capability)}</p>"
        f"<p><strong>Component:</strong> {html.escape(component)}</p>"
        f"<p><strong>Theme:</strong> {html.escape(theme_name)}</p>"
        f"<h2>Actions</h2><ul>{actions}</ul>"
        f"<h2>Relationships</h2><pre>{relationships}</pre>"
        f"<h2>Theme Tokens</h2><pre>{tokens}</pre>"
    )
    return _html_page(title, body)


def _capability_screen_payload(path: str) -> tuple[int, str]:
    screen = _capability_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")
    return 200, _capability_screen_html(screen)


def _application_screen(path: str) -> Dict[str, Any] | None:
    if APG_APPLICATIONS is None or not hasattr(APG_APPLICATIONS, "application_route_index"):
        return None
    routes = APG_APPLICATIONS.application_route_index()
    screen = routes.get(path)
    return dict(screen) if isinstance(screen, dict) else None


def _application_screen_html(screen: Dict[str, Any]) -> str:
    title = str(screen.get("name") or screen.get("component") or "Application route")
    application = str(screen.get("application") or "")
    route = str(screen.get("route") or screen.get("path") or "")
    capabilities = html.escape(json.dumps(screen.get("capabilities", []), indent=2, sort_keys=True))
    agents = html.escape(json.dumps(screen.get("agents", []), indent=2, sort_keys=True))
    component = html.escape(json.dumps(screen.get("component"), indent=2, sort_keys=True))
    body = (
        '<nav><a href="/ui">Application</a> | '
        '<a href="/applications">Applications</a> | '
        '<a href="/routes">Routes</a> | '
        '<a href="/composition">Composition</a></nav>'
        f"<h1>{html.escape(title)}</h1>"
        f"<p><strong>Application:</strong> {html.escape(application)}</p>"
        f"<p><strong>Route:</strong> {html.escape(route)}</p>"
        f"<h2>Capabilities</h2><pre>{capabilities}</pre>"
        f"<h2>Agents</h2><pre>{agents}</pre>"
        f"<h2>Component</h2><pre>{component}</pre>"
    )
    return _html_page(title, body)


def _application_screen_payload(path: str) -> tuple[int, str]:
    screen = _application_screen(path)
    if screen is None:
        return 404, _html_page("Not found", f"<h1>Not found</h1><p>{html.escape(path)}</p>")
    return 200, _application_screen_html(screen)


def _record_route(path: str) -> Dict[str, str | None] | None:
    parts = [part for part in path.split("/") if part]
    if parts == ["records"]:
        return {"entity": None, "record_id": None, "operation": None}
    if len(parts) in {2, 3} and parts[0] == "records":
        return {
            "entity": parts[1],
            "record_id": parts[2] if len(parts) == 3 else None,
            "operation": None,
        }
    if len(parts) in {3, 4} and parts[0] == "entities" and parts[2] == "records":
        operation = parts[3] if len(parts) == 4 and parts[3] in {"export", "import"} else None
        return {
            "entity": parts[1],
            "record_id": None if operation else parts[3] if len(parts) == 4 else None,
            "operation": operation,
        }
    return None


def _record_by_id(entity_name: str, record_id: str) -> Dict[str, Any] | None:
    for record in RECORD_STORE[entity_name]:
        if str(record.get("id")) == str(record_id):
            return dict(record)
    return None


def _records_payload(path: str) -> tuple[int, Dict[str, Any]]:
    return _records_payload_with_query(path, {})


def _records_payload_with_query(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    operation = route.get("operation")
    if entity_name is None:
        return 200, {"records": list_records()}
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    if operation == "export":
        return 200, {
            "entity": entity_name,
            "records": list_records(entity_name),
            "count": len(list_records(entity_name)),
        }
    if operation is not None:
        return 405, {"error": "method_not_allowed", "operation": operation}
    if record_id is None:
        return 200, query_records(entity_name, query)
    record = _record_by_id(entity_name, record_id)
    if record is None:
        return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}
    return 200, {"entity": entity_name, "record": record}


def _route_payload(path: str, query: Dict[str, list[str]] | None = None) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path in {"/", "/manifest", "/application"}:
        return 200, describe_application()
    if path == "/component.json":
        return 200, component_manifest()
    if path == "/health":
        validation = validate_application()
        return 200, {
            "status": "ok" if validation["valid"] else "warning",
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "valid": validation["valid"],
            "storage": storage_status(),
            "auth": auth_status(),
            "warnings": validation["warnings"],
        }
    if path == "/validate":
        validation = validate_application()
        return (200 if validation["valid"] else 422), validation
    if path == "/openapi.json":
        return 200, openapi_document()
    if path == "/entities":
        return 200, {"entities": list_entities()}
    if path == "/databases":
        return 200, {"databases": list_databases()}
    if path == "/databases/status":
        status = database_status()
        return (200 if status["valid"] else 422), status
    if path.startswith("/databases/") and path.endswith("/schemas"):
        database_name = path.strip("/").split("/")[1]
        for database in list_databases():
            if str(database.get("name")) == database_name:
                return 200, {
                    "database": database_name,
                    "schemas": database.get("schemas", []),
                }
        return 404, {"error": "unknown_database", "database": database_name}
    if path == "/auth":
        return 200, auth_status()
    if path == "/events":
        return 200, {"events": list_events()}
    if path == "/metrics":
        return 200, metrics_snapshot()
    if path == "/self-test":
        report = self_test()
        return (200 if report["passed"] else 422), report
    if path == "/records" or path.startswith("/records/") or (
        path.startswith("/entities/") and "/records" in path
    ):
        return _records_payload_with_query(path, query)
    if path == "/relationships":
        return 200, relationship_graph()
    if path == "/storage":
        return 200, storage_status(include_records=True)
    if path == "/agents":
        return 200, {
            "agents": describe_application().get("ai_agent_descriptions", {}),
            "teams": describe_application().get("ai_agent_team_descriptions", {}),
        }
    if path == "/applications":
        app = describe_application()
        return 200, {
            "applications": app.get("application_composition_descriptions", {}),
            "dependency_graph": app.get("application_dependency_graph", {}),
            "components": app.get("application_component_catalog", {}),
        }
    if path == "/capabilities":
        app = describe_application()
        return 200, {
            "capabilities": app.get("capability_descriptions", {}),
            "by_erp_module": app.get("capability_descriptions_by_erp_module", {}),
            "dependency_graph": app.get("capability_dependency_graph", {}),
            "load_order": app.get("capability_load_order", {}),
        }
    if path == "/capabilities/health":
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health_report"):
            return 404, {"error": "capability_health_unavailable"}
        health = APG_CAPABILITIES.capability_health_report()
        return (200 if health.get("healthy") else 422), health
    if path.startswith("/capabilities/") and path.endswith("/health"):
        if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_health"):
            return 404, {"error": "capability_health_unavailable"}
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            try:
                health = APG_CAPABILITIES.capability_health(parts[1])
            except KeyError:
                return 404, {"error": "unknown_capability", "capability": parts[1]}
            return (200 if health.get("healthy") else 422), health
    if path == "/streaming":
        return _streaming_payload()
    if path.startswith("/capabilities/") and path.endswith("/streaming"):
        parts = [part for part in path.split("/") if part]
        if len(parts) == 3:
            return _capability_streaming_payload(parts[1])
    if path == "/routes":
        return 200, {"routes": describe_application().get("ui_routes", {})}
    if path == "/composition":
        return 200, describe_application().get("composition_graph", {"nodes": [], "edges": []})
    return 404, {"error": "not_found", "path": path}


def _rule_evaluation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if path.startswith("/capabilities/") and path.endswith("/rules/evaluate"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3:
            capability_name = parts[1]
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "evaluate_capability_rules"):
        return 404, {"error": "capability_rules_unavailable"}
    context = payload.get("context", {})
    if not isinstance(context, dict):
        return 400, {"error": "context_must_be_object"}
    try:
        return 200, APG_CAPABILITIES.evaluate_capability_rules(str(capability_name), context)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _capability_name_from_payload_or_path(path: str, payload: Dict[str, Any]) -> str | None:
    capability_name = payload.get("capability") or payload.get("capability_name")
    if capability_name:
        return str(capability_name)
    if path.startswith("/capabilities/"):
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            return parts[1]
    return None


def _configuration_payload(path: str, payload: Dict[str, Any], validate: bool = False) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None:
        return 404, {"error": "capabilities_unavailable"}
    configuration = payload.get("configuration", payload.get("overrides"))
    if configuration is not None and not isinstance(configuration, dict):
        return 400, {"error": "configuration_must_be_object"}
    try:
        if validate:
            validator = getattr(APG_CAPABILITIES, "validate_capability_configuration", None)
            if validator is None:
                return 404, {"error": "configuration_validation_unavailable"}
            return 200, validator(str(capability_name), configuration)
        resolver = getattr(APG_CAPABILITIES, "capability_configuration", None)
        if resolver is None:
            return 404, {"error": "configuration_resolution_unavailable"}
        return 200, {
            "capability": str(capability_name),
            "configuration": resolver(str(capability_name), configuration),
        }
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _approval_plan_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    capability_name = _capability_name_from_payload_or_path(path, payload)
    if not capability_name:
        return 400, {"error": "missing_capability"}
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "approval_plan"):
        return 404, {"error": "approval_planning_unavailable"}
    context = payload.get("context", {})
    if not isinstance(context, dict):
        return 400, {"error": "context_must_be_object"}
    try:
        return 200, APG_CAPABILITIES.approval_plan(str(capability_name), context)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": str(capability_name)}


def _streaming_payload() -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None:
        return 404, {"error": "capabilities_unavailable"}
    processor_index = getattr(APG_CAPABILITIES, "streaming_processor_index", lambda: {})()
    state_index = getattr(APG_CAPABILITIES, "streaming_state_index", lambda: {})()
    streams: Dict[str, Any] = {}
    if hasattr(APG_CAPABILITIES, "list_capabilities") and hasattr(APG_CAPABILITIES, "capability_streaming"):
        for capability_name in APG_CAPABILITIES.list_capabilities():
            streams[capability_name] = APG_CAPABILITIES.capability_streaming(capability_name)
    return 200, {
        "processor": "bytewax",
        "processors": processor_index,
        "states": state_index,
        "streams": streams,
    }


def _capability_streaming_payload(capability_name: str) -> tuple[int, Dict[str, Any]]:
    if APG_CAPABILITIES is None or not hasattr(APG_CAPABILITIES, "capability_streaming"):
        return 404, {"error": "capability_streaming_unavailable"}
    try:
        return 200, APG_CAPABILITIES.capability_streaming(capability_name)
    except KeyError:
        return 404, {"error": "unknown_capability", "capability": capability_name}


def _agent_invocation_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if AI_AGENTS is None:
        return 404, {"error": "agents_unavailable"}
    parts = [part for part in path.split("/") if part]
    try:
        if len(parts) == 3 and parts[0] == "agents" and parts[2] in {"invoke", "run"}:
            invoker = getattr(AI_AGENTS, "invoke_agent", None)
            if invoker is None:
                return 404, {"error": "agent_invocation_unavailable"}
            return 200, invoker(parts[1], payload)
        if len(parts) == 3 and parts[0] in {"agent-teams", "teams"} and parts[2] in {"invoke", "run"}:
            invoker = getattr(AI_AGENTS, "invoke_team", None)
            if invoker is None:
                return 404, {"error": "team_invocation_unavailable"}
            return 200, invoker(parts[1], payload)
    except KeyError as error:
        return 404, {"error": "unknown_agent_composition", "name": str(error).strip("'")}
    return 404, {"error": "not_found", "path": path}


def _create_record_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is not None and route.get("operation") == "import":
        return _import_records_payload(str(route["entity"]), payload)
    if route is None or route["entity"] is None or route["record_id"] is not None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {"error": "record_must_be_object"}
    record = coerce_record_types(entity_name, dict(raw_record))
    validation = validate_record(entity_name, record)
    if not validation["valid"]:
        return 422, {"error": "record_validation_failed", **validation}
    if record.get("id") in (None, ""):
        record["id"] = NEXT_RECORD_IDS[entity_name]
        NEXT_RECORD_IDS[entity_name] += 1
    elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
        return 409, {"error": "duplicate_record_id", "entity": entity_name, "id": record["id"]}
    record = _prepare_new_record(record)
    RECORD_STORE[entity_name].append(record)
    event = _record_event("create", entity_name, after=record)
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {"error": "persistence_failed", "message": persistence_error}
    return 201, {
        "entity": entity_name,
        "record": dict(record),
        "event": event,
        "count": len(RECORD_STORE[entity_name]),
    }


def _import_records_payload(entity_name: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return 400, {"error": "records_must_be_array"}
    imported: list[Dict[str, Any]] = []
    events: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            errors.append({"index": index, "errors": ["record must be object"]})
            continue
        record = coerce_record_types(entity_name, dict(raw_record))
        validation = validate_record(entity_name, record)
        if not validation["valid"]:
            errors.append({"index": index, "errors": validation["errors"]})
            continue
        if record.get("id") in (None, ""):
            record["id"] = NEXT_RECORD_IDS[entity_name]
            NEXT_RECORD_IDS[entity_name] += 1
        elif any(str(existing.get("id")) == str(record["id"]) for existing in RECORD_STORE[entity_name]):
            errors.append({"index": index, "errors": [f"duplicate id {record['id']}"]})
            continue
        record = _prepare_new_record(record)
        RECORD_STORE[entity_name].append(record)
        imported.append(dict(record))
        events.append(_record_event("import", entity_name, after=record))
    persistence_error = _persist_record_store()
    if persistence_error:
        return 500, {"error": "persistence_failed", "message": persistence_error}
    return (201 if imported else 422), {
        "entity": entity_name,
        "imported": imported,
        "events": events,
        "errors": errors,
        "count": len(imported),
        "failed": len(errors),
    }


def _update_record_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    raw_record = payload.get("record", payload)
    if not isinstance(raw_record, dict):
        return 400, {"error": "record_must_be_object"}
    record_update = coerce_record_types(entity_name, dict(raw_record))
    validation = validate_record(entity_name, record_update, partial=True)
    if not validation["valid"]:
        return 422, {"error": "record_validation_failed", **validation}
    for index, existing in enumerate(RECORD_STORE[entity_name]):
        if str(existing.get("id")) == str(record_id):
            conflict = _revision_conflict(existing, _expected_revision(payload))
            if conflict is not None:
                return 409, conflict
            updated = dict(existing)
            updated.update(record_update)
            updated["id"] = existing.get("id")
            updated["_revision"] = int(existing.get("_revision", 1)) + 1
            RECORD_STORE[entity_name][index] = updated
            event = _record_event("update", entity_name, before=existing, after=updated)
            persistence_error = _persist_record_store()
            if persistence_error:
                return 500, {"error": "persistence_failed", "message": persistence_error}
            return 200, {"entity": entity_name, "record": dict(updated), "event": event}
    return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}


def _delete_record_payload(path: str) -> tuple[int, Dict[str, Any]]:
    raw_path = path
    path = path.split("?", 1)[0]
    route = _record_route(path)
    if route is None or route["entity"] is None or route["record_id"] is None:
        return 404, {"error": "not_found", "path": path}
    entity_name = route["entity"]
    record_id = route["record_id"]
    if entity_name not in ENTITY_NAMES:
        return 404, {"error": "unknown_entity", "entity": entity_name}
    for index, existing in enumerate(RECORD_STORE[entity_name]):
        if str(existing.get("id")) == str(record_id):
            expected_revision = None
            if "?" in raw_path:
                query = parse_qs(raw_path.split("?", 1)[1], keep_blank_values=True)
                value = query.get("expected_revision", [None])[-1]
                try:
                    expected_revision = int(value) if value is not None else None
                except (TypeError, ValueError):
                    expected_revision = None
            conflict = _revision_conflict(existing, expected_revision)
            if conflict is not None:
                return 409, conflict
            deleted = RECORD_STORE[entity_name].pop(index)
            event = _record_event("delete", entity_name, before=deleted)
            persistence_error = _persist_record_store()
            if persistence_error:
                return 500, {"error": "persistence_failed", "message": persistence_error}
            return 200, {
                "entity": entity_name,
                "deleted": dict(deleted),
                "event": event,
                "count": len(RECORD_STORE[entity_name]),
            }
    return 404, {"error": "record_not_found", "entity": entity_name, "id": record_id}


def _post_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if (
        path.startswith("/agents/") and path.endswith(("/invoke", "/run"))
    ) or (
        (path.startswith("/agent-teams/") or path.startswith("/teams/")) and path.endswith(("/invoke", "/run"))
    ):
        return _agent_invocation_payload(path, payload)
    if path.startswith("/records/") or path.endswith("/records/import") or (
        path.startswith("/entities/") and path.endswith("/records")
    ):
        return _create_record_payload(path, payload)
    if path in {"/rules/evaluate", "/capabilities/rules/evaluate"} or (
        path.startswith("/capabilities/") and path.endswith("/rules/evaluate")
    ):
        return _rule_evaluation_payload(path, payload)
    if path in {"/configuration/resolve", "/capabilities/configuration/resolve"} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/resolve")
    ):
        return _configuration_payload(path, payload)
    if path in {"/configuration/validate", "/capabilities/configuration/validate"} or (
        path.startswith("/capabilities/") and path.endswith("/configuration/validate")
    ):
        return _configuration_payload(path, payload, validate=True)
    if path in {"/approval/plan", "/capabilities/approval/plan"} or (
        path.startswith("/capabilities/") and path.endswith("/approval/plan")
    ):
        return _approval_plan_payload(path, payload)
    return 404, {"error": "not_found", "path": path}


def _put_payload(path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    path = path.rstrip("/") or "/"
    if path.startswith("/records/") or (
        path.startswith("/entities/") and "/records/" in path
    ):
        return _update_record_payload(path, payload)
    return 404, {"error": "not_found", "path": path}


_load_record_store()


class ApplicationRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, response: Dict[str, Any]) -> None:
        body = _json_bytes(response)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html_payload: str) -> None:
        body = html_payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, status: int, location: str) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorize_mutation(self) -> bool:
        if _authorized(self.headers):
            return True
        status, response = _auth_failure_payload()
        self._send_json(status, response)
        return False

    def do_GET(self) -> None:
        path, _, raw_query = self.path.partition("?")
        query = parse_qs(raw_query, keep_blank_values=True)
        if path == "/theme.css":
            body = theme_stylesheet().encode("utf-8")
            status = 200
            content_type = "text/css; charset=utf-8"
        elif path == "/ui" or path.startswith("/ui/"):
            status, html_payload = _ui_payload(path, query)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif _capability_screen(path) is not None:
            status, html_payload = _capability_screen_payload(path)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif _application_screen(path) is not None:
            status, html_payload = _application_screen_payload(path)
            body = html_payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            status, payload = _route_payload(path, query)
            body = _json_bytes(payload)
            content_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorize_mutation():
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b"{}"
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
            if content_type == "application/x-www-form-urlencoded":
                parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
                payload = {"record": {key: values[-1] if values else "" for key, values in parsed.items()}}
            else:
                payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            if path.startswith("/ui/") and content_type == "application/x-www-form-urlencoded":
                status, response = _ui_post_payload(path, payload)
                if status in {302, 303}:
                    self._send_redirect(status, str(response["location"]))
                    return
                if "html" in response:
                    self._send_html(status, str(response["html"]))
                    return
                self._send_html(status, _ui_error_payload(path, response))
                return
            else:
                status, response = _post_payload(path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            status, response = 400, {"error": "invalid_json", "message": str(error)}
        self._send_json(status, response)

    def do_PUT(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorize_mutation():
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            status, response = _put_payload(path, payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            status, response = 400, {"error": "invalid_json", "message": str(error)}
        self._send_json(status, response)

    def do_DELETE(self) -> None:
        path = self.path
        if not self._authorize_mutation():
            return
        status, response = _delete_record_payload(path)
        self._send_json(status, response)

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("APG_DEBUG") == "1":
            super().log_message(format, *args)


def _arg_value(argv: list[str], name: str, default: str) -> str:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def run_server(host: str | None = None, port: int | str | None = None) -> None:
    resolved_host = host or os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1"
    resolved_port = int(port or os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    server = HTTPServer((resolved_host, resolved_port), ApplicationRequestHandler)
    print(f"{MODULE_NAME} listening on http://{resolved_host}:{resolved_port}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--describe" in args:
        print(json.dumps(describe_application(), indent=2, sort_keys=True))
        return
    if "--validate" in args:
        report = validate_application()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["valid"] else 1)
    if "--self-test" in args:
        report = self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)
    host = _arg_value(args, "--host", os.environ.get("APG_HOST") or os.environ.get("HOST") or "127.0.0.1")
    port = _arg_value(args, "--port", os.environ.get("APG_PORT") or os.environ.get("PORT") or "8080")
    run_server(host, port)


if __name__ == "__main__":
    main()
