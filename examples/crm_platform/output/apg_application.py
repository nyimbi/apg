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


APPLICATION_DATA: Dict[str, Dict[str, Any]] = {'CRMPlatform': {'description': 'Enterprise CRM composed from APG capabilities', 'capabilities': ['CRMCore'], 'agents': ['SalesAssistant'], 'agent_teams': [], 'components': {}, 'screens': {}, 'routes': ['/crm', '/crm/contacts', '/crm/accounts', '/crm/pipeline'], 'workflows': [], 'policies': {}, 'configuration': {}, 'theme': {'name': 'crm_platform_theme', 'tokens': {'accent': '#FF6D00', 'border.radius': '6px'}}, 'runtime': {'target': 'python', 'deployment': 'container', 'streaming': {'processor': 'bytewax'}}, 'integrations': {}, 'deployments': {'default': 'local', 'container': 'docker'}}}
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


def _normalize_application_screen(application: ApplicationSpec, name: str, spec: Any) -> Dict[str, Any]:
    screen_spec = dict(spec) if isinstance(spec, dict) else {"component": spec or name}
    route = screen_spec.get("route", screen_spec.get("path", ""))
    return {
        "id": f"{application.name}.{name}",
        "application": application.name,
        "name": name,
        "route": route,
        "path": route,
        "component": screen_spec.get("component", name),
        "capability": screen_spec.get("capability"),
        "capabilities": list(application.capabilities),
        "agents": list(application.agents),
        "agent_teams": list(application.agent_teams),
        "theme": screen_spec.get("theme", application.theme.get("name")),
        "spec": screen_spec,
    }


def application_screens(application_name: str) -> List[Dict[str, Any]]:
    application = get_application(application_name)
    screens: List[Dict[str, Any]] = []
    if isinstance(application.screens, dict):
        for name, spec in application.screens.items():
            screens.append(_normalize_application_screen(application, str(name), spec))
    elif isinstance(application.screens, list):
        for index, item in enumerate(application.screens):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or item.get("component") or f"screen_{index + 1}")
                screens.append(_normalize_application_screen(application, name, item))
            else:
                name = str(item)
                screens.append(_normalize_application_screen(application, name, {"component": name}))

    known_routes = {str(screen.get("route") or screen.get("path") or "") for screen in screens}
    for index, route in enumerate(application.routes):
        route_text = str(route)
        if route_text in known_routes:
            continue
        screens.append({
            "id": f"{application.name}.route_{index + 1}",
            "application": application.name,
            "name": route_text,
            "route": route_text,
            "path": route_text,
            "component": route_text,
            "capability": None,
            "capabilities": list(application.capabilities),
            "agents": list(application.agents),
            "agent_teams": list(application.agent_teams),
            "theme": application.theme.get("name"),
            "spec": {"route": route_text},
        })
    return screens


def application_route_index() -> Dict[str, Dict[str, Any]]:
    routes: Dict[str, Dict[str, Any]] = {}
    for application in APPLICATIONS.values():
        for screen in application_screens(application.name):
            route = screen.get("route") or screen.get("path")
            if route:
                routes[str(route)] = screen
    return routes


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
        for screen in application_screens(application.name):
            screen_id = f"application_screen:{screen['id']}"
            node(screen_id, "application_screen", str(screen["name"]))
            edge(app_id, screen_id, "has_screen")
            route = screen.get("route") or screen.get("path")
            if route:
                route_id = f"route:{route}"
                node(route_id, "route", str(route))
                edge(screen_id, route_id, "mounted_at")
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
