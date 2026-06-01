# ONTO Capability Plan

## 1. Specification

- Define the executable generated-app lifecycle for ontologies, namespaces, terms, taxonomy, mappings, validation, publication, export, and audit.
- Keep production RDF/OWL/SKOS, graph, metadata, NLP, approval, and audit systems behind adapters.
- Add first-class ontology-agent composition for Codex, Claude Code, opencode, and Pi style assistants behind provider-neutral AICR contracts.
- Add Bytewax lifecycle batch validation for ontology, namespace, term, taxonomy, mapping, validation, publication, exchange, and agent changes.

## 2. Contract

- Expand configuration to ontology, namespace, term, taxonomy, mapping, validation, publication, import/export, curation, security, governance, observability, adapter, UI, and theme sections.
- Expand configuration to ontology, namespace, term, taxonomy, mapping, validation, publication, import/export, curation, agents, streaming, security, governance, observability, adapter, UI, and theme sections.
- Expand deterministic rules beyond 55 lifecycle, agent, stream, and guardrail checks.
- Expose at least 15 UI routes and component theme hooks.
- Support numeric and inequality operators in rule matching.

## 3. Runtime

- Extend domain models with namespaces, validation reports, and exports.
- Extend `OntoService` to enforce namespace uniqueness, duplicate-term review, taxonomy integrity, validation reports, publication validation, and export formats.
- Persist review-required duplicate terms, breaking curation requests, deprecations, low-confidence mappings, validation issues, and privileged ontology-agent registrations as `pending_review` records with policy evidence.
- Add ontology-agent records and lifecycle-batch records with tenant-qualified storage, audit evidence, and dashboard counts.
- Preserve existing ontology, term, taxonomy, mapping, review, publication, audit, and compatibility behavior.

## 4. API And UI

- Add payload helpers for namespaces, term deprecation, validation, and exports.
- Add payload helpers for ontology-agent registration, lifecycle-batch validation, pending-review queues, and package evidence.
- Add view models for namespaces, validation, exchange, ontology agents, lifecycle batches, pending-review queues, audit, and settings.

## 5. Package Evidence

- Replace static semantic model data with contract-derived metadata.
- Refresh semantic model, package manifest, and release report.

## 6. Verification

- Compile the ONTO packet.
- Run focused ONTO tests only.
- Run `app.self_test()`.
- Run APG implementation audit and publish-plan.
- Scan primary files for stale markers and exaggerated language.
- Run whitespace diff checks.

## 7. Review And Commit

- Perform a direct review of the packet and fix discovered issues.
- Update the progress log.
- Commit and push using the Lore commit protocol.
