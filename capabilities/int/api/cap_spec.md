# int_api Capability Package

`int_api` is the APG Integration API Management capability. It supplies a
dependency-light, executable lifecycle for APIs, endpoints, policies, consumers,
API keys, subscriptions, deployments, usage analytics, and AI-agent review
teams.

## Contract Summary

- Capability: `int_api`
- Display name: `Integration API Management`
- Version: `2.1.0`
- Target: `python`
- UI shell: `apg_python`
- Theme: `int_api_control`
- Stream processor: `bytewax`
- Stream: `apg.int.api.lifecycle`

## Provides

- `api_registry_lifecycle`
- `api_endpoint_lifecycle`
- `api_policy_lifecycle`
- `api_consumer_lifecycle`
- `api_key_lifecycle`
- `api_subscription_lifecycle`
- `api_deployment_workflow`
- `api_gateway_route_catalog`
- `api_analytics_workflow`
- `api_dashboard_service`
- `api_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `policy_management`
- `service_discovery`
- `developer_portal`

## Primary Workflows

1. Register APIs.
2. Register endpoints.
3. Attach policies.
4. Register consumers.
5. Issue scoped API keys.
6. Create approved subscriptions.
7. Approve and deploy APIs.
8. Record usage analytics.
9. Register API agents for governed review and validation work.

## Runtime Files

- `capability_contract.py`: executable APG contract and deterministic rules.
- `service.py`: in-memory lifecycle facade.
- `api.py`: generated-app helper functions.
- `views.py`: framework-neutral screen models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package verification.

## UI Screens

- Dashboard
- APIs
- Endpoints
- Policies
- Consumers
- Keys
- Subscriptions
- Deployments
- Analytics
- Agents
- Settings

## Guardrail Scope

Rules cover tenant context, policy attachment, API completeness, protocol and
auth support, external-upstream review, endpoint validation, policy
configuration, consumer validation, key scope and expiration, subscription
approval, API approval, deployment identity, deployment approval, usage
analytics, Bytewax routing, supported agent runtimes, supported agent roles, and
privileged-agent human approval.
