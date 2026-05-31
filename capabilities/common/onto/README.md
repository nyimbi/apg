# ONTO - Ontology Management

ONTO is the APG capability for governed ontologies, taxonomies, controlled vocabularies, semantic mappings, validation, publication, ontology exchange, first-class ontology agents, and Bytewax lifecycle batches. It gives generated applications an executable vocabulary workbench that can be composed with Knowledge Graph, Metadata, NLP, Search, Auth, Audit, AICR, Cache, Metrics, and Bytewax-backed event processing.

The generated-app surface is dependency-light. It uses in-process domain records for ontologies, namespaces, terms, taxonomy edges, mappings, reviews, validation reports, publications, exports, ontology agents, lifecycle batches, and audit events. Production systems can attach persistent stores, RDF/OWL/SKOS processors, graph stores, approval workflows, NLP assistants, external AI-agent runtimes, and durable Bytewax topologies through the configured adapters.

## What ONTO Provides

- Tenant-scoped ontology registry with owners, domains, versions, and publication state.
- Namespace governance for ontology prefixes and URIs.
- Controlled vocabulary terms with owners, definitions, statuses, synonyms, external references, and deprecation links.
- Taxonomy edge management with self-relation and cycle prevention.
- Semantic mappings to metadata, graph, search, external ontology, or application concepts.
- Review gates for duplicate terms, breaking changes, low-confidence mappings, external mappings, deprecations, imports, and validation issues.
- Validation reports for publication readiness.
- Publication lifecycle with approval, duplicate checks, draft-term checks, mapping-review checks, and version bumps.
- Export records for RDF, OWL, JSON-LD, SKOS, and CSV style interchange.
- First-class ontology-agent registration for Codex, Claude Code, opencode, and Pi style assistants behind provider-neutral AICR adapter contracts.
- Bytewax lifecycle batch validation for ontology, namespace, term, taxonomy, mapping, validation, publication, exchange, and agent changes.
- UI route metadata, view models, theme tokens, and audit evidence for generated APG applications.

## How To Use It

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
account = service.create_term(
	"term-account",
	"tenant-a",
	ontology["id"],
	"Account",
	"data-stewards",
	"A commercial relationship with a customer.",
)
service.add_synonym("tenant-a", customer["id"], "Client")
service.add_taxonomy_edge(
	"edge-customer-account",
	"tenant-a",
	ontology["id"],
	customer["id"],
	account["id"],
)
service.curate_term("review-customer", "tenant-a", customer["id"], "chief-steward")
service.curate_term("review-account", "tenant-a", account["id"], "chief-steward")
service.create_mapping(
	"map-customer-meta",
	"tenant-a",
	customer["id"],
	"meta:party.customer",
	"exact",
	0.96,
)
service.validate_ontology("validation-customer", "tenant-a", ontology["id"])
publication = service.publish_ontology(
	"publication-customer",
	"tenant-a",
	ontology["id"],
	approval_recorded=True,
	approval_ref="approval:onto-42",
)
export = service.export_ontology("export-customer", "tenant-a", ontology["id"], "jsonld")

agent = service.register_ontology_agent(
	"agent-taxonomy-review",
	"tenant-a",
	"Taxonomy Reviewer",
	"codex",
	"taxonomy_reviewer",
	"customer ontology taxonomy",
	"data-stewards",
	"Review ontology hierarchy changes before publication",
	contribution_disclosed=True,
)
batch = service.validate_onto_lifecycle_batch(
	"tenant-a",
	"bytewax",
	4,
	"ontology_agent_batch",
	"ontobatch-customer-review",
)
```

Use `capability_contract.py` for compiler and composition metadata. Use `api.py` for generated endpoint-style helpers. Use `views.py` for dashboard, registry, namespace, term, taxonomy, mapping, validation, exchange, publication, governance, ontology-agent roster, lifecycle-batch monitor, audit, and settings view models.

## Guardrails

ONTO exposes deterministic rules for tenant context, ontology identity, namespace uniqueness, term ownership, term status, duplicate review, deprecation replacement, synonym values, taxonomy integrity, mapping confidence, external mapping review, breaking change review, curation evidence, validation, publication readiness, import/export controls, first-class ontology-agent governance, Bytewax lifecycle batches, tenant isolation, and audit evidence.

## Files

- `SPECIFICATION.md` defines the capability behavior and boundaries.
- `PLAN.md` records the packet implementation plan.
- `capability_contract.py` is the executable APG contract.
- `models.py` contains dependency-light domain records.
- `ontology_runtime.py` contains deterministic helper functions.
- `service.py` executes the ontology lifecycle.
- `api.py` exposes payload helper functions.
- `views.py` exposes generated-app UI models.
- `app.py` emits semantic package evidence and self-test results.
