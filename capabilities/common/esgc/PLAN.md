# ESGC ESG and Carbon Tracking Packet Plan

## Scope

Build `esgc` as a coherent lifecycle and guardrail packet for APG applications
that need emissions inventory, factor libraries, activity emissions,
sustainability reporting, target tracking, ESG evidence, AI-agent review, UI
metadata, theme metadata, Bytewax stream governance, and publishable package
evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Add `SPECIFICATION.md` for the normative behavior.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, agent metadata, and Bytewax streaming.

2. Dependency-light service
   - Preserve inventory, factor, activity, report, target, audit, and
     view-model behavior already present in `EsgcService`.
   - Add ESGC-agent data contracts and service methods.
   - Add batch mutation validation tied to the Bytewax stream guardrail.
   - Keep meters, geospatial providers, compliance filings, forecasting, audit
     stores, and stream workers behind adapters.

3. Package entrypoint and helper surfaces
   - Make `__init__.py` export the expanded contract, service, agent model, and
     stream metadata.
   - Extend API helpers and view models with ESGC-agent and batch mutation
     surfaces.

4. Documentation and generated evidence
   - Add root package `README.md` with practical usage and composition notes.
   - Refresh semantic model, package manifest, and release evidence from the
     live contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Extend focused contract/service tests without invoking live meter,
     geospatial, compliance, forecast, or stream-worker fixtures.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review tenant isolation, boundaries, factor evidence, activity evidence,
     anomaly review, report approval, target baseline, AI-agent boundaries,
     Bytewax guardrails, import behavior, and generated evidence consistency.

## Out Of Scope

- Live meter and source-system integrations.
- Compliance filing submission.
- Forecast model execution.
- Durable audit store writes.
- Live Bytewax topology deployment.
- Browser-rendered UI.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- Dependency-light package import does not start meter, geospatial, compliance,
  forecast, audit, or stream services.
- Inventories require owner and reporting boundary.
- Factors require approved source, evidence, version, and valid scope.
- Activities require matching units and evidence.
- Anomalies require review.
- Reports require approval, compliance mapping, audit evidence, and approver.
- Targets require baseline data.
- AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Batch mutation is rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
