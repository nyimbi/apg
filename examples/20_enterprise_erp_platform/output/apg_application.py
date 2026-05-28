"""
APG Application Composition Runtime
===================================

Generated from first-class APG app/application composition declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    description: str | None
    capabilities: List[str]
    agents: List[str]
    agent_teams: List[str]
    components: Any
    screens: Any
    routes: List[str]
    workflows: List[str]
    policies: Any
    configuration: Dict[str, Any]
    theme: Dict[str, Any]
    runtime: Dict[str, Any]
    integrations: Any
    deployments: Any


APPLICATION_DATA: Dict[str, Dict[str, Any]] = {'EnterpriseERPPlatform': {'description': 'Composable ERP application shell that assembles finance, operations, and audit capabilities', 'capabilities': ['PlatformAudit', 'EnterpriseFinance', 'EnterpriseOperations'], 'agents': [], 'agent_teams': [], 'components': {'finance_workbench': {'capability': 'journal_entries', 'route': '/erp/finance'}, 'operations_workbench': {'capability': 'executive_kpis', 'route': '/erp/operations'}, 'audit_console': {'capability': 'audit_events', 'route': '/erp/audit'}}, 'screens': {'ExecutiveHome': {'route': '/erp', 'capability': 'EnterpriseOperations', 'component': 'ExecutiveHome'}}, 'routes': ['/erp/finance', '/erp/operations', '/erp/operations/dashboard'], 'workflows': [], 'policies': {}, 'configuration': {}, 'theme': {'name': 'enterprise_theme', 'tokens': {'accent': '#174EA6', 'surface': '#F8FAFC'}}, 'runtime': {'target': 'python', 'deployment': 'container', 'streaming': {'processor': 'bytewax'}}, 'integrations': {}, 'deployments': {'default': 'local', 'container': 'docker'}}}
APPLICATIONS: Dict[str, ApplicationSpec] = {
    name: ApplicationSpec(name=name, **data)
    for name, data in APPLICATION_DATA.items()
}


def list_applications() -> List[str]:
    return sorted(APPLICATIONS)


def get_application(name: str) -> ApplicationSpec:
    return APPLICATIONS[name]


def describe_application_composition(name: str) -> Dict[str, Any]:
    application = get_application(name)
    return {
        "name": application.name,
        "description": application.description,
        "capabilities": list(application.capabilities),
        "agents": list(application.agents),
        "agent_teams": list(application.agent_teams),
        "components": application.components,
        "screens": application.screens,
        "routes": list(application.routes),
        "workflows": list(application.workflows),
        "policies": application.policies,
        "configuration": dict(application.configuration),
        "theme": dict(application.theme),
        "runtime": dict(application.runtime),
        "integrations": application.integrations,
        "deployments": application.deployments,
    }


def describe_application_compositions() -> Dict[str, Dict[str, Any]]:
    return {
        name: describe_application_composition(name)
        for name in list_applications()
    }


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def application_component_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for application in APPLICATIONS.values():
        components = application.components
        if isinstance(components, dict):
            for component_name, component_spec in components.items():
                component_id = f"{application.name}.{component_name}"
                catalog[component_id] = {
                    "id": component_id,
                    "application": application.name,
                    "name": str(component_name),
                    "spec": dict(component_spec) if isinstance(component_spec, dict) else {"value": component_spec},
                }
        for route in application.routes:
            component_id = f"{application.name}.route.{route}"
            catalog[component_id] = {
                "id": component_id,
                "application": application.name,
                "name": str(route),
                "kind": "route",
                "spec": {"route": route},
            }
    return catalog


def application_dependency_graph() -> Dict[str, List[Dict[str, str]]]:
    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict[str, str]] = []

    def node(node_id: str, kind: str, name: str) -> None:
        nodes[node_id] = {"id": node_id, "kind": kind, "name": name}

    def edge(source: str, target: str, relation: str) -> None:
        edges.append({"source": source, "target": target, "relation": relation})

    for application in APPLICATIONS.values():
        app_id = f"application:{application.name}"
        node(app_id, "application", application.name)
        for capability in application.capabilities:
            capability_id = f"capability:{capability}"
            node(capability_id, "capability", str(capability))
            edge(app_id, capability_id, "uses_capability")
        for agent in application.agents:
            agent_id = f"agent:{agent}"
            node(agent_id, "agent", str(agent))
            edge(app_id, agent_id, "uses_agent")
        for team in application.agent_teams:
            team_id = f"agent_team:{team}"
            node(team_id, "agent_team", str(team))
            edge(app_id, team_id, "uses_agent_team")
        for route in application.routes:
            route_id = f"route:{route}"
            node(route_id, "route", str(route))
            edge(app_id, route_id, "exposes_route")
    return {"nodes": sorted(nodes.values(), key=lambda item: item["id"]), "edges": edges}


def validate_application_compositions(
    available_capabilities: List[str] | None = None,
    available_agents: List[str] | None = None,
    available_teams: List[str] | None = None,
) -> Dict[str, List[str]]:
    known_capabilities = set(available_capabilities or [])
    known_agents = set(available_agents or [])
    known_teams = set(available_teams or [])
    errors: List[str] = []
    warnings: List[str] = []
    for application in APPLICATIONS.values():
        if not application.capabilities and not application.components and not application.routes:
            warnings.append(f"{application.name} does not compose capabilities, components, or routes")
        for capability in application.capabilities:
            if known_capabilities and capability not in known_capabilities:
                errors.append(f"{application.name} references unknown capability {capability}")
        for agent in application.agents:
            if known_agents and agent not in known_agents:
                errors.append(f"{application.name} references unknown agent {agent}")
        for team in application.agent_teams:
            if known_teams and team not in known_teams:
                errors.append(f"{application.name} references unknown agent team {team}")
    return {"errors": errors, "warnings": warnings}
