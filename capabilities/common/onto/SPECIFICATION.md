# ONTO Capability Specification

## Purpose

ONTO provides governed ontology and vocabulary management for generated APG applications. It standardizes business vocabulary, taxonomy structure, semantic mappings, publication readiness, and exchange artifacts so ERP, analytics, RAG, graph, metadata, search, and AI capabilities can share consistent meaning.

## Scope

ONTO owns the generated-app lifecycle for:

1. Ontology registration.
2. Namespace registration.
3. Term creation, synonym management, curation, and deprecation.
4. Taxonomy edge creation.
5. Semantic mapping to metadata, graph, search, application, or external concepts.
6. Validation reports.
7. Publication approval and versioning.
8. Export artifact records.
9. First-class ontology-agent composition for curation, mapping, validation, and publication review.
10. Bytewax lifecycle batch validation for ontology, namespace, term, taxonomy, mapping, validation, publication, exchange, and agent changes.
11. UI route metadata, view models, theme metadata, audit events, and package evidence.

Production RDF/OWL/SKOS parsers, graph stores, search indexes, NLP term extraction, approval workflows, and external audit stores are adapter concerns.
External AI-agent runtimes such as Codex, Claude Code, opencode, and Pi are adapter concerns. ONTO owns their provider-neutral registration contract, accountable scope, contribution disclosure, and privileged-role review gates.

## Configuration

The contract includes explicit sections for:

- `ontologies`
- `namespaces`
- `terms`
- `taxonomy`
- `mappings`
- `validation`
- `publication`
- `import_export`
- `curation`
- `agents`
- `streaming`
- `security`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

`observability.event_stream` and `adapters.event_stream` are `bytewax`.
`streaming.required_processor` is also `bytewax`; broker-specific queue or broker-core coupling is outside the ONTO generated-app contract.

## Runtime Records

The generated-app runtime uses these record types:

- `Ontology`
- `OntologyNamespace`
- `OntologyTerm`
- `TaxonomyEdge`
- `SemanticMapping`
- `CurationReview`
- `ValidationReport`
- `OntologyPublication`
- `OntologyExport`
- `OntologyAgentRecord`
- `OntoLifecycleBatchRecord`
- `OntoAuditEvent`

Records are tenant-scoped and exposed as dictionaries for generated applications. Persisted lifecycle records carry policy evidence fields: `decision`, `matched_rules`, `review_reasons`, and `audit_evidence`.

## Rule Engine

The deterministic rule engine returns `allow`, `require_review`, or `deny`. Runtime methods raise `PermissionError` for hard deny decisions. Review-required ontology outcomes are preserved as `pending_review` records with policy evidence so generated applications can compose approval queues instead of losing the attempted change as a transient exception.

Rules cover:

- Tenant context and cross-tenant access.
- Ontology id, name, owner, domain, and retirement review.
- Namespace prefix, URI, owner, ontology, and uniqueness.
- Term ontology, label, owner, status, duplicates, deprecation replacement, and deprecation review.
- Synonym term/value requirements.
- Taxonomy parent, child, self-relation, cycle, and relationship type constraints.
- Mapping term, target, type, confidence, and external review constraints.
- Breaking change and curation evidence.
- Validation report review for issue-bearing ontologies.
- Publication approval, validation, duplicate, cycle, draft-term, and low-confidence mapping gates.
- Import/export format and review gates.
- Ontology-agent runtime, role, scope, owner, purpose, contribution-disclosure, and human-review gates.
- Bytewax lifecycle batch processor and operation gates.
- Bytewax event-stream, audit, and tenant isolation requirements.

Review-required outcomes currently persisted as durable evidence include duplicate term submissions, breaking term-curation requests with evidence but no recorded review, term deprecation without review, low-confidence mappings, validation reports with unresolved issues, and privileged ontology-agent registration without human approval.

## UI Requirements

The generated UI exposes these route surfaces:

- Dashboard
- Ontologies
- Namespaces
- Terms
- Taxonomy
- Mappings
- Validation
- Imports
- Exports
- Publication
- Governance
- Agents
- Lifecycle
- Audit
- Settings

Theme components cover ontology cards, namespace panels, term cards, taxonomy trees, mapping panels, validation reports, publication queues, exchange panels, ontology-agent rosters, Bytewax lifecycle panels, and audit timelines.

## Acceptance Criteria

- Root README, specification, and plan exist.
- Contract exposes at least 55 deterministic rules, at least 15 routes, first-class ontology agents, Bytewax streaming metadata, Bytewax adapters, and theme metadata.
- Runtime executes ontology registration, namespace registration, term lifecycle, taxonomy edges, mappings, validation, publication, export, and audit.
- Runtime executes provider-neutral ontology-agent registration and Bytewax lifecycle-batch validation.
- Guardrails block missing tenant context, ownerless terms, duplicate prefixes, taxonomy cycles, publication without validation or approval, invalid export formats, unsupported ontology-agent runtimes or roles, missing agent scope/owner/purpose, undisclosed machine contribution, non-Bytewax lifecycle batches, unsupported lifecycle operations, and cross-tenant access.
- Review-required duplicate terms, breaking curation requests, deprecations, low-confidence mappings, issue-bearing validations, and privileged ontology agents persist as `pending_review` records with durable policy evidence.
- Package semantic evidence is generated from the current contract.
- Focused tests cover the lifecycle, agent guardrails, Bytewax lifecycle batches, views, package evidence, and import-light API.
