# I18N Internationalization Specification

## Purpose

I18N is APG's common internationalization capability. It lets generated and
composed applications ship multilingual user interfaces, help content,
workflow labels, notifications, and regulated text while keeping tenant
boundaries, review gates, fallback behavior, and publication evidence explicit.

The capability is optimized for rapid application composition. A capability
consumer should be able to add I18N, configure supported languages and routes,
register localization agents, publish translations, and resolve text at
runtime without building a separate localization subsystem.

## Capability Identity

- Capability id: `i18n`
- Display name: `Internationalization`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.I18nService`
- UI prefix: `/i18n`
- API prefix: `/i18n/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `locale_management`
- `translation_memory`
- `content_localization`
- `language_fallbacks`
- `regional_formatting`
- `language_policy`
- `i18n_agents`

## Required Capabilities

- `conf` for tenant configuration and feature flags.
- `nlpc` for text analysis, terminology, and language support.
- `auth` for identity, permissions, and RBAC filtering.
- `audl` for durable audit evidence.

Optional adapters include `mchn`, `help`, and `them`.

## Domain Model

`LocaleDefinition`

- Tenant-local locale identity, language/region code, display name, owner,
  fallback locale, regional format, timezone, enabled flag, and creation time.

`GlossaryTerm`

- Tenant-local source term, localized variants, description, owner, and
  creation time.

`TranslationEntry`

- Tenant-local localization key, locale code, source text, translated text,
  lifecycle status, source type, reviewer, restricted-content flag, version,
  and timestamps.

`CoverageReport`

- Locale-specific coverage evidence: total required keys, published keys,
  missing keys, coverage percentage, and review requirement.

`PublishBatch`

- Approved publication set for a locale and list of translation ids.

`I18nAgent`

- Registered AI localization agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

`I18nAuditEvent`

- Audit record for localization lifecycle state changes.

## Language Policy

The default language policy includes core application languages and a broad
African-language set. The African list includes at least 40 language codes,
including `sw`, `am`, `ha`, `ig`, `yo`, `zu`, `rw`, `so`, `om`, `ti`, `sn`,
`xh`, `st`, `tn`, `ts`, `ve`, `wo`, `ak`, `ee`, `ff`, and others.

Locale creation must use a supported language subtag. Unsupported language
codes are denied before a locale is created.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every localization operation;
- accountable locale owner;
- supported language code;
- fallback locale policy;
- regional format metadata;
- glossary owner;
- translation key;
- localized text;
- review for machine-generated translations;
- RBAC filtering for restricted content;
- publication approval;
- publication approver;
- missing-key review before release;
- coverage review when coverage is below the threshold;
- registered AI localization agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch localization mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/i18n/dashboard`
- `/i18n/locales`
- `/i18n/translations`
- `/i18n/glossaries`
- `/i18n/coverage`
- `/i18n/publishing`
- `/i18n/agents`
- `/i18n/audit`
- `/i18n/policies`
- `/i18n/settings`

Each route is represented in the semantic model and has a permission and
component name. View models must expose enough data for dashboards, workbench
screens, policy screens, audit trails, and AI-agent panels.

## Theme

The default theme is `i18n_localization_workbench`. Theme components cover
locale matrices, translation editors, coverage dashboards, publish queues,
agent panels, audit trails, and language policy tables.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.i18n.lifecycle`
- state: locales, glossary terms, translations, coverage reports, publish
  batches, I18N agents, audit events
- events: locale created, glossary term added, translation upserted,
  translation published, coverage reported, agent registered
- guardrail: `batch_i18n_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports locale, glossary, translation, publication, coverage,
  text resolution, AI-agent registration, audit events, tenant-local IDs, and
  Bytewax batch mutation validation.
- At least 40 African language codes are exposed in configuration and tested.
- The tests prove the main lifecycle, policy denials, tenant isolation,
  generated evidence, and docs.
- Focused compile, pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
