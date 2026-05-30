# ETLP Improvement Backlog

This backlog lists high-value improvements for ETLP. Items here are not claims
of implemented behavior unless they are also present in the contract, service,
tests, and package evidence.

## Near-Term Packet Improvements

- Durable persistence adapter for lifecycle records.
- Datasource approval workflow integrated with AUTH and AUDL.
- Bytewax flow adapter for streaming executions.
- Metadata and lineage adapter integration with META.
- Quality profiling adapter with configurable dimensions.
- Rendered generated-app screens backed by `view_models.py`.
- Schedule, retry, replay, and backfill consoles.
- Secret-store adapter checks for datasource definitions.

## Runtime Improvements

- Connector registry plugin interface.
- Execution checkpoint and resume protocol.
- Dead-letter and quarantine handling through MQEB.
- Execution health metrics through MONI.
- Cost-estimation adapter for pipeline plans.
- Backpressure and concurrency control.

## AI-Assisted Improvements

- Mapping suggestions from schema and sample profiles.
- Pipeline plan linting.
- Failure classification and remediation suggestions.
- Quality-rule recommendations.
- Cost and resource optimization recommendations.

AI-assisted behavior must remain behind adapters and must not bypass
deterministic ETLP guardrails.
