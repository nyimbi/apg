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

### Synchronous API

| Method | Description |
|--------|-------------|
| `describe()` | Return capability contract metadata |
| `evaluate()` | Run capability policy rules against a context |
| `register_ontology()` | Create a new tenant-scoped ontology |
| `register_namespace()` | Register a prefix/URI namespace |
| `create_term()` | Create a controlled vocabulary term |
| `curate_term()` | Apply a curation review to a term |
| `add_synonym()` | Add an alternative label to a term |
| `add_taxonomy_edge()` | Assert a hierarchical relationship between terms |
| `create_mapping()` | Map a term to an internal or external reference |
| `deprecate_term()` | Mark a term as deprecated with a replacement |
| `validate_ontology()` | Run publication-readiness validation |
| `publish_ontology()` | Publish a validated ontology (bumps version) |
| `export_ontology()` | Create an export record (RDF/OWL/JSON-LD/SKOS/CSV) |
| `import_owl()` | Import terms from OWL/RDF XML content |
| `export_turtle()` | Export as Turtle RDF serialisation |
| `sparql_query()` | Execute a SPARQL-like term query |
| `reasoner_run()` | Run a simulated OWL reasoner |
| `concept_define()` | High-level term creation with stable ID derivation |
| `axiom_assert()` | Assert a logical axiom (subClassOf, equivalentClass, …) |
| `ontology_merge()` | Merge source ontology terms into a target ontology |
| `consistency_check()` | Check for duplicate labels, cycles, and orphan terms |
| `ontology_visualise()` | Return graph nodes+edges for visualisation |
| `property_add()` | Add or update a metadata property on a term |
| `register_ontology_agent()` | Register an AI agent for ontology governance |
| `validate_onto_lifecycle_batch()` | Validate a Bytewax lifecycle batch |
| `dashboard_summary()` | Tenant-level stats across all registries |
| `ontology_package()` | Full tenant ontology snapshot |

### Async API (new)

| Method | Description |
|--------|-------------|
| `async_register_ontology()` | Async variant of `register_ontology` for pipeline use |
| `async_create_term()` | Async variant of `create_term` for pipeline use |
| `async_bulk_create_terms()` | Bulk-create many terms in one call with per-item error reporting |
| `find_similar_terms()` | Near-duplicate detection via token Jaccard similarity |
| `align_ontologies()` | Cross-ontology term alignment with optional auto-mapping |
| `compute_version_diff()` | Semantic diff since last publication; recommends bump type |
| `export_skos()` | W3C SKOS Turtle export with `skos:broader/narrower/altLabel` |
| `verify_audit_chain()` | SHA-256 Merkle chain integrity check on audit events |
| `suggest_definition()` | LLM-assisted definition generation via local Ollama |
| `sync_to_graph()` | Push nodes and edges to a `grph` capability adapter |

_(See `service.py` for full signatures and docstrings.)_

## Async Usage Examples

```python
import asyncio
from capabilities.common.onto.service import OntoService

service = OntoService()

async def main():
    onto = service.register_ontology("med-onto", "acme", "Medical Ontology", "stewards", "healthcare")

    # Bulk term import
    result = await service.async_bulk_create_terms(
        tenant_id="acme",
        ontology_id="med-onto",
        owner="stewards",
        terms=[
            {"label": "Patient", "definition": "An individual receiving medical care."},
            {"label": "Diagnosis", "definition": "Identification of a disease or condition."},
            {"label": "Treatment", "definition": "A procedure to remedy a condition."},
        ],
    )
    print(f"Created {result['created_count']} terms")

    # Detect near-duplicates before adding
    dupes = await service.find_similar_terms("acme", "med-onto", "Client", top_k=3)
    print(dupes["candidates"])

    # Get LLM-assisted definition
    suggestion = await service.suggest_definition("acme", "med-onto", "Prognosis", domain_hint="clinical medicine")
    for i, cand in enumerate(suggestion["candidates"], 1):
        print(f"{i}. {cand}")

    # Export as SKOS
    skos = await service.export_skos("acme", "med-onto")
    print(skos["turtle"][:400])

    # Check audit chain integrity
    chain = await service.verify_audit_chain("acme")
    assert chain["chain_valid"], chain["broken_links"]

    # Semantic version diff
    diff = await service.compute_version_diff("acme", "med-onto")
    print(f"Recommended bump: {diff['recommended_bump']}, breaking: {diff['breaking_change']}")

asyncio.run(main())
```

## Cross-Ontology Alignment

```python
async def align():
    service = OntoService()
    service.register_ontology("snomed", "acme", "SNOMED Subset", "stewards", "clinical")
    service.register_ontology("icd11", "acme", "ICD-11 Subset", "stewards", "clinical")
    # ... populate terms ...

    result = await service.align_ontologies(
        tenant_id="acme",
        source_ontology_id="snomed",
        target_ontology_id="icd11",
        strategy="synonym",
        confidence_cutoff=0.7,
        auto_create_above=0.9,   # auto-persist high-confidence mappings
    )
    print(f"{result['candidate_count']} candidates, {result['auto_created_count']} auto-mapped")
```

## Interoperability

`onto` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use onto;
```

Composition adapters:

| Capability | Integration method |
|------------|--------------------|
| `grph` | `sync_to_graph(graph_adapter)` — push nodes/edges to graph store |
| `srch` | Planned: `index_for_search()` |
| `nlpc` | Planned: `extract_nlp_entities()` |
| `aicr` | `suggest_definition()` — routes to local Ollama via AICR contract |

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ONTO_`.

| Key | Default | Description |
|-----|---------|-------------|
| `ONTO_CONFIDENCE_THRESHOLD` | `0.8` | Minimum mapping confidence before `require_review` |
| `ONTO_SIMILARITY_THRESHOLD` | `0.85` | Jaccard threshold for near-duplicate detection |
| `ONTO_OLLAMA_MODEL` | `llama3.2` | Local Ollama model for definition suggestions |
| `ONTO_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |

## Further Reading

- `service.py` — Business logic implementation (sync + async methods)
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Improvement proposals and roadmap
