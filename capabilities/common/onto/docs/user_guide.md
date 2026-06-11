# Ontology Management — User Guide

**Capability ID**: `onto` | **Domain**: `common` | **Version**: 1.1.0

---

## Description

ONTO is the APG capability for governed ontologies, taxonomies, controlled vocabularies, semantic mappings, validation, publication, ontology exchange, first-class ontology agents, and Bytewax lifecycle batches. It gives generated applications an executable vocabulary workbench that can be composed with Knowledge Graph, Metadata, NLP, Search, Auth, Audit, AICR, Cache, Metrics, and Bytewax-backed event processing.

---

## Installation

```bash
pip install apg-common-onto
```

---

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

---

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

---

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

### Async API

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
| `import_skos()` | Import W3C SKOS Turtle into an existing ontology |
| `advance_review()` | Drive a pending review through the governance workflow |
| `list_pending_reviews_by_sla()` | List reviews breaching a configured SLA threshold |
| `get_term_lineage()` | Full audit lineage for a single term |
| `batch_deprecate_terms()` | Bulk-deprecate terms with per-item error collection |
| `search_terms()` | Ranked term search (fulltext / semantic / hybrid) |
| `emit_delta()` | Publish a CloudEvents delta to downstream capabilities |
| `clone_ontology()` | Deep-clone an ontology into a new registry entry |
| `add_label_translation()` | Add a BCP 47 language-tagged label to a term |

---

## Quick Start

```python
from capabilities.common.onto.service import OntoService

service = OntoService()

ontology = service.register_ontology(
    "customer-ontology",
    "tenant-a",
    "Customer Ontology",
    "data-stewards",
    "crm",
)
namespace = service.register_namespace(
    "ns-cust",
    "tenant-a",
    ontology["id"],
    "cust",
    "https://example.com/ontology/customer#",
    "data-stewards",
)
customer = service.create_term(
    "term-customer",
    "tenant-a",
    ontology["id"],
    "Customer",
    "data-stewards",
    "A party that purchases products or services.",
)
service.add_synonym("tenant-a", customer["id"], "Client")
service.add_taxonomy_edge(
    "edge-customer-account",
    "tenant-a",
    ontology["id"],
    customer["id"],
    customer["id"],   # self-test -- use distinct IDs in production
)
service.curate_term("review-customer", "tenant-a", customer["id"], "chief-steward")
service.validate_ontology("validation-customer", "tenant-a", ontology["id"])
publication = service.publish_ontology(
    "publication-customer",
    "tenant-a",
    ontology["id"],
    approval_recorded=True,
    approval_ref="approval:onto-42",
)
```

---

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

    # Ranked term search
    hits = await service.search_terms("acme", "med-onto", "patient diagnosis", mode="hybrid", top_k=5)
    for h in hits["results"]:
        print(h["term"]["label"], h["score"])

    # Get LLM-assisted definition
    suggestion = await service.suggest_definition("acme", "med-onto", "Prognosis", domain_hint="clinical medicine")
    for i, cand in enumerate(suggestion["candidates"], 1):
        print(f"{i}. {cand}")

    # Export as SKOS
    skos = await service.export_skos("acme", "med-onto")
    print(skos["turtle"][:400])

    # Import SKOS back in
    skos_import = await service.import_skos("acme", "med-onto-v2", skos["turtle"], owner="stewards")
    print(f"SKOS import: {skos_import['terms_imported']} terms")

    # Check audit chain integrity
    chain = await service.verify_audit_chain("acme")
    assert chain["chain_valid"], chain["broken_links"]

    # Semantic version diff
    diff = await service.compute_version_diff("acme", "med-onto")
    print(f"Recommended bump: {diff['recommended_bump']}, breaking: {diff['breaking_change']}")

asyncio.run(main())
```

---

## Governance Workflow

```python
import asyncio
from capabilities.common.onto.service import OntoService

service = OntoService(confidence_threshold=0.75)

async def governance_demo():
    service.register_ontology("fin-onto", "acme", "Finance Ontology", "stewards", "finance")
    term = service.create_term("t-revenue", "acme", "fin-onto", "Revenue", "stewards",
                               "Income from normal business operations.")
    # Low-confidence mapping triggers pending_review
    m = service.create_mapping("m-rev-ext", "acme", "t-revenue",
                               "external:gaap.revenue", "exact", 0.60)
    assert m["status"] == "pending_review"

    # Advance through governance workflow
    review_id = service.list_reviews("acme")[0]["id"]
    result = await service.advance_review("acme", review_id, actor="chief-steward", action="approve",
                                          notes="Verified against GAAP codification.")
    print(result["new_status"])  # approved

    # Check SLA compliance
    sla_report = await service.list_pending_reviews_by_sla("acme", sla_hours=24.0)
    print(f"Overdue: {sla_report['overdue_count']}, within SLA: {sla_report['within_sla_count']}")

asyncio.run(governance_demo())
```

---

## Ontology Branching (Clone)

```python
async def branching_demo():
    service = OntoService()
    service.register_ontology("prod-onto", "acme", "Production Ontology", "stewards", "core")
    # ... populate ...

    # Branch for experimental work
    clone = await service.clone_ontology(
        tenant_id="acme",
        source_ontology_id="prod-onto",
        new_ontology_id="exp-onto-v2",
        new_name="Experimental Ontology v2",
        owner="researcher",
        include_terms=True,
        include_edges=True,
        include_mappings=False,
    )
    print(f"Cloned {clone['terms_cloned']} terms, {clone['edges_cloned']} edges")
```

---

## Multi-Language Labels

```python
async def i18n_demo():
    service = OntoService()
    service.register_ontology("global-onto", "acme", "Global Ontology", "stewards", "common")
    term = service.create_term("t-customer", "acme", "global-onto", "Customer", "stewards")

    await service.add_label_translation("acme", "t-customer", "fr", "Client")
    await service.add_label_translation("acme", "t-customer", "sw", "Mteja")
    await service.add_label_translation("acme", "t-customer", "ar", "عميل")

    term_dict = service._terms["t-customer"].to_dict()
    print(term_dict["metadata"]["labels"])
    # {"fr": "Client", "sw": "Mteja", "ar": "عميل"}
```

---

## Term Lineage

```python
async def lineage_demo():
    service = OntoService()
    service.register_ontology("reg-onto", "acme", "Regulatory Ontology", "stewards", "compliance")
    term = service.create_term("t-liability", "acme", "reg-onto", "Liability", "stewards",
                               "An obligation arising from a past transaction.")
    service.add_synonym("acme", "t-liability", "Obligation")
    service.curate_term("rev-liab", "acme", "t-liability", "chief-steward")

    lineage = await service.get_term_lineage("acme", "t-liability")
    print(f"Events: {lineage['event_count']}, Reviews: {lineage['review_count']}")
    for ev in lineage["lineage"]:
        print(ev["event_type"], ev["created_at"])
```

---

## Delta Streaming

```python
async def delta_demo():
    service = OntoService()
    service.register_ontology("evt-onto", "acme", "Event Ontology", "stewards", "events")
    term = service.create_term("t-order", "acme", "evt-onto", "Order", "stewards")

    # Without transport (dry-run — audit only)
    delta = await service.emit_delta(
        tenant_id="acme",
        event_type="term_curated",
        subject_id=term["id"],
        old_value={"status": "draft"},
        new_value={"status": "curated"},
        actor="steward-1",
    )
    print(delta["content_hash"], delta["published"])  # hash, False

    # With transport (e.g. Kafka adapter)
    # delta = await service.emit_delta(..., transport=kafka_adapter)
```

---

## Cross-Ontology Alignment

```python
async def align():
    service = OntoService()
    service.register_ontology("snomed", "acme", "SNOMED Subset", "stewards", "clinical")
    service.register_ontology("icd11", "acme", "ICD-11 Subset", "stewards", "clinical")

    result = await service.align_ontologies(
        tenant_id="acme",
        source_ontology_id="snomed",
        target_ontology_id="icd11",
        strategy="synonym",
        confidence_cutoff=0.7,
        auto_create_above=0.9,
    )
    print(f"{result['candidate_count']} candidates, {result['auto_created_count']} auto-mapped")
```

---

## Bulk Deprecation

```python
async def bulk_deprecation_demo():
    service = OntoService()
    service.register_ontology("old-onto", "acme", "Legacy Ontology", "stewards", "legacy")
    t1 = service.create_term("t-old1", "acme", "old-onto", "OldConcept1", "stewards")
    t2 = service.create_term("t-old2", "acme", "old-onto", "OldConcept2", "stewards")
    t3 = service.create_term("t-new", "acme", "old-onto", "NewConcept", "stewards")
    service.curate_term("r1", "acme", "t-old1", "steward")
    service.curate_term("r2", "acme", "t-old2", "steward")
    service.curate_term("r3", "acme", "t-new", "steward")

    result = await service.batch_deprecate_terms(
        tenant_id="acme",
        deprecations=[
            {"term_id": "t-old1", "replacement_term_id": "t-new", "notes": "Superseded by NewConcept"},
            {"term_id": "t-old2", "replacement_term_id": "t-new", "notes": "Merged into NewConcept"},
        ],
        reviewer="chief-steward",
    )
    print(f"Deprecated {result['succeeded_count']}, failed {result['failed_count']}")
```

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ONTO_`.

| Key | Default | Description |
|-----|---------|-------------|
| `ONTO_CONFIDENCE_THRESHOLD` | `0.8` | Minimum mapping confidence before `require_review` |
| `ONTO_SIMILARITY_THRESHOLD` | `0.85` | Jaccard threshold for near-duplicate detection |
| `ONTO_OLLAMA_MODEL` | `llama3.2` | Local Ollama model for definition suggestions |
| `ONTO_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `ONTO_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model for semantic search |
| `ONTO_SLA_HOURS` | `48` | Default SLA for pending-review items |

---

## Interoperability

`onto` integrates with other APG capabilities through the composition engine:

```apg
use onto;
```

| Capability | Integration method |
|------------|--------------------|
| `grph` | `sync_to_graph(graph_adapter)` — push nodes/edges to graph store |
| `srch` | `index_for_search(srch_adapter)` — planned |
| `nlpc` | `extract_nlp_entities(nlpc_adapter)` — planned |
| `aicr` | `suggest_definition()` — routes to local Ollama via AICR contract |
| `meta` | `emit_delta()` — CloudEvents delta streaming for metadata consistency |

---

## Further Reading

- `service.py` — Business logic implementation (sync + async methods)
- `models.py` — Data models
- `ontology_runtime.py` — Deterministic helper functions
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `SPECIFICATION.md` — Capability specification
- `WORLD_CLASS_IMPROVEMENTS.md` — Improvement proposals and roadmap
