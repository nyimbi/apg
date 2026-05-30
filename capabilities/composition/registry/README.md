# APG Capability Registry

`composition_registry` is the APG catalog and governance capability for registering capabilities, validating dependency graphs, preparing composition blueprints, governing version releases, preparing marketplace publication, and coordinating registry review agents.

## What It Provides

- Capability catalog lifecycle.
- Dependency graph management.
- Composition blueprint validation.
- Version compatibility governance.
- Marketplace publication governance.
- Registry discovery.
- Registry AI agents.

## How To Use It

Import the service for in-process generated applications:

```python
from capabilities.composition.registry import CompositionRegistryService

service = CompositionRegistryService()
capability = service.register_capability(
    "auth",
    "tenant-a",
    "Authentication",
    "platform-owner",
    "common",
    "1.0.0",
    ["authn", "session_management"],
    "capabilities/common/auth/capability_contract.py",
)
```

Create a composition from registered capabilities:

```python
composition = service.create_composition(
    "secure-app",
    "tenant-a",
    "Secure App",
    "platform-owner",
    ["auth"],
)
```

Inspect compiler-facing package evidence:

```bash
./.venv/bin/apg capabilities inspect composition_registry --json
./.venv/bin/apg capabilities publish-plan capabilities/composition/registry --json
```

## Lifecycle

1. Register capabilities with owner, category, version, provided surfaces, and contract reference.
2. Add dependency edges with version constraints.
3. Validate dependency graph cycles and composition completeness.
4. Create and publish composition blueprints with validation evidence.
5. Release capability versions with compatibility evidence.
6. Deprecate capabilities with migration plans.
7. Prepare marketplace publication with documentation and review evidence.
8. Register AI agents for registry review work.

## Screens

- Dashboard
- Catalog
- Dependencies
- Compositions
- Versions
- Marketplace
- Rules
- Agents
- Settings

## Guardrails

The deterministic rule engine blocks missing tenant context, writes without policy, incomplete capability records, incomplete dependency records, unvalidated composition publication, releases without compatibility evidence, deprecations without migration plans, marketplace publication without documentation, non-Bytewax registry imports, unsupported agent runtimes, unsupported agent roles, and privileged agent actions without human approval.

## AI Agent Support

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are capability curator, dependency reviewer, composition reviewer, version reviewer, marketplace reviewer, and security reviewer.
