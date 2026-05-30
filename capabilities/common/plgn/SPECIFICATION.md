# PLGN Plugin/Extension Framework Specification

## Purpose

PLGN is APG's common plugin and extension capability. It lets generated and
composed applications register plugin manifests, review permissions, bind
sandbox policy, publish curated marketplace listings, release signed packages,
install extensions, enable extensions, and govern plugin lifecycle work through
APG UI and API surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real plugin registries, artifact stores, signing services, security
scanners, sandbox providers, identity providers, audit stores, workflow
systems, and Bytewax workers later.

## Capability Identity

- Capability id: `plgn`
- Display name: `Plugin/Extension Framework`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.PlgnService`
- UI prefix: `/plgn`
- API prefix: `/plgn/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `plugin_registry`
- `extension_marketplace`
- `permission_review`
- `sandbox_policy`
- `plugin_release_lifecycle`
- `plgn_agents`

## Required Capabilities

- `auth` for identity, permissions, and installer authority.
- `secu` for permission review and package security posture.
- `conf` for tenant install policy and plugin configuration policy.
- `audl` for durable audit evidence.

Optional adapters include `regy`, `agnt`, `sbox`, and `wflo`.

## Domain Model

`PluginManifest`

- Tenant-local plugin id, name, owner, version, publisher, release channel,
  requested permissions, dependencies, external-plugin posture, signature
  state, manifest validation, dependency validation, scan evidence, lifecycle
  status, metadata, and timestamps.

`PermissionReview`

- Review record with tenant, plugin id, reviewer, approved scopes, denied
  scopes, secret-access posture, notes, and creation time.

`SandboxPolicy`

- Sandbox policy with tenant, plugin id, network access, filesystem access,
  secret access, tool allowlist, and creation time.

`MarketplaceListing`

- Curated listing with tenant, plugin id, title, publisher-verification state,
  curation state, install policy, lifecycle status, and creation time.

`PluginRelease`

- Signed release with tenant, plugin id, version, channel, signature reference,
  lifecycle status, and creation time.

`PluginInstallation`

- Tenant installation with plugin id, installer, status, enablement timestamp,
  and creation time.

`PlgnAuditEvent`

- Governance record for plugin lifecycle actions.

`PlgnAgent`

- Registered AI plugin agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every plugin operation;
- plugin owner identity;
- package signature evidence;
- manifest schema validation;
- dependency validation;
- supply-chain scan evidence;
- permission review for requested scopes;
- sandbox policy before enablement;
- external review for external plugins;
- verified publisher for marketplace listings;
- curation for marketplace listings;
- release signature reference;
- Bytewax event stream for release lifecycle events;
- tenant install policy before installation;
- registered AI plugin agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch plugin mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/plgn/dashboard`
- `/plgn/marketplace`
- `/plgn/plugins`
- `/plgn/manifests`
- `/plgn/permissions`
- `/plgn/sandbox`
- `/plgn/releases`
- `/plgn/agents`
- `/plgn/audit`
- `/plgn/settings`

View models must expose plugin summaries, marketplace listings, plugin
registry state, permission reviews, sandbox policies, releases, plugin agents,
rules, audit events, theme data, and Bytewax stream metadata.

## Theme

The default theme is `plgn_extension_marketplace`. Theme components cover
plugin cards, marketplace grids, permission review tables, release managers,
agent panels, and audit timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.plgn.lifecycle`
- state: plugins, permission reviews, sandbox policies, listings, releases,
  installations, PLGN agents, audit events
- events: plugin registered, permission review recorded, sandbox policy
  attached, listing published, plugin released, plugin installed, plugin
  enabled, agent registered
- guardrail: `batch_plugin_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports plugin manifests, permission reviews, sandbox policies,
  marketplace listings, releases, installations, enablement, AI-agent
  registration, audit events, tenant-local IDs, and Bytewax batch mutation
  validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
