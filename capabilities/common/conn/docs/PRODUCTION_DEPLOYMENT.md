# CONN Production Deployment Notes

Production CONN deployments are adapter-backed. The generated-app runtime
records lifecycle decisions, while production adapters perform external work
such as Singer tap execution, credential access, monitoring, audit writes,
lineage persistence, data-quality evaluation, registry publication, gateway
integration, and Bytewax event processing.

Before enabling live side effects, verify:

- connector package source and checksum evidence;
- credential vault and encryption wiring;
- connection test and secret rotation evidence;
- production activation reviews;
- flow mapping, lineage, quality gate, and PII policy evidence;
- sync monitoring and schema review paths;
- schedule timezone and replay idempotency controls;
- audit and metrics sinks;
- rollback and retirement impact review procedures.

Generated-app package checks do not prove live external integrations. Run full
adapter, network, secret-store, tap execution, UI, and performance validation in
a powered staging environment.
