# Ontology Management

**Capability ID**: `onto` | **Domain**: `common` | **Version**: `1.0.0`

## Description

ONTO is the APG capability for governed ontologies, taxonomies, controlled vocabularies, semantic mappings, validation, publication, ontology exchange, first-class ontology agents, and Bytewax lifecycle batches. It gives generated applications an executable vocabulary workbench that can be composed with Knowledge Graph, Metadata, NLP, Search, Auth, Audit, AICR, Cache, Metrics, and Bytewax-backed event processing.

## Installation

```bash
pip install apg-common-onto
```

## Provides

- `ontology_management`
- `semantic_vocabulary_governance`
- `ontology_agent_composition`

## Requires

- `meta`
- `nlpc`
- `grph`
- `srch`
- `aicr`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/onto/dashboard` | `onto:view` | Overview |
| `/onto/ontologies` | `onto:view` | Registry |
| `/onto/namespaces` | `onto:edit` | Registry |
| `/onto/terms` | `onto:edit` | Vocabulary |
| `/onto/taxonomy` | `onto:edit` | Vocabulary |
| `/onto/mappings` | `onto:map` | Mappings |
| `/onto/validation` | `onto:govern` | Governance |
| `/onto/imports` | `onto:edit` | Exchange |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_ontology()`
- `register_namespace()`
- `create_term()`
- `curate_term()`
- `add_synonym()`
- `add_taxonomy_edge()`
- `create_mapping()`
- `deprecate_term()`

_(See `service.py` for complete API.)_

## Interoperability

`onto` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use onto;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ONTO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
