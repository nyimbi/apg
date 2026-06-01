# Capability Spec: intel_analytics

## Summary

`intel_analytics` provides executable lawful authority, analytic workspace,
dataset, feature-set, model, run, insight, dashboard, narrative,
recommendation, review, and AI-agent workflows for APG intelligence-analytics
applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Graph Data Management,
Retrieval-Augmented Generation, and geospatial capabilities. It emits Bytewax
lifecycle metadata through `apg.intel.analytics.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live warehouses, ML engines, feature stores, model
registries, notebook runtimes, visualization renderers, graph writes, RAG
indexing, notification delivery, and durable Bytewax workers are adapter
responsibilities. Hallucinated insights, training-data leakage, privacy bypass,
unsupported automated decisions, unapproved model deployment, and autonomous
dissemination scopes are denied for AI-agent actions.
