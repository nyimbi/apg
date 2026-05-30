# Help and Knowledge Base Capability Specification

## Purpose

`help` is the APG common capability for governed support knowledge. It lets generated applications compose tenant-scoped source registries, help articles, article publication workflows, cited assisted answers, localization, feedback review, curation, audit trails, analytics, UI screens, visual theming, and event-stream policy.

## Scope

The capability must support:

- Source registration with owner, URI, visibility, approval state, and audit evidence.
- Article lifecycle from draft to publication with owner, title, body, topics, locale, source approval, publication approval, freshness review, and restricted-content filtering.
- Help search with query checks, RBAC filtering, and query-logging policy.
- Assisted answer generation with query checks, citations, confidence review, unsafe-answer blocking, and RAG/search adapter boundaries.
- Localization with supported locale, translator, source locale, fallback locale, and translated article body.
- Feedback with user identity, rating bounds, low-rating curation review, and article/answer references.
- Curation decisions with reviewer and evidence.
- Bytewax-backed event-stream configuration for batch help mutations.
- UI route contracts and dependency-light view models for generated applications.

## Dependencies

Required:

- `ragn` for retrieval-augmented generation composition.
- `srch` for search composition.
- `nlpc` for natural-language composition.

Optional:

- `auth`, `audl`, `chat`, `ntfy`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `content`
- `sources`
- `answers`
- `search`
- `feedback`
- `localization`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- source owner, source URI, and source approval
- article owner, title, body, publication approval, publication audit, and freshness review
- answer query, citations, confidence, and unsafe answer blocking
- search query, query logging, and restricted-content filtering
- feedback user, rating bounds, and low-rating review
- localization supported locale, translator, and fallback locale
- curation reviewer and evidence
- state-change audit evidence
- cross-tenant access denial
- Bytewax batch mutation enforcement

## Runtime

`service.HelpService` is the generated-application runtime. It stores deterministic in-memory state for:

- sources
- articles
- answers
- feedback
- localizations
- curation items
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine where the package can do so without live provider dependencies.

## UI

The UI contract exposes:

- dashboard
- home
- articles
- editor
- sources
- answers
- localization
- curation
- audit
- analytics
- settings

## Production Boundary

This packet does not operate live RAG providers, external search engines, production databases, notification services, identity services, or Bytewax workers. Those are production adapters behind the APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI, theme, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative guardrail behavior.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` match the current contract.
- Focused compile, pytest, implementation audit, publish-plan, stale-marker scan, and diff check pass.
