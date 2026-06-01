# KEYM Capability Source Of Truth

The active Key Management specification is `SPECIFICATION.md`.

The current executable packet covers tenant-managed keys, key operation
decisions, dual-control export approvals, rotation exception review,
evidence-backed rotation, compromise response, first-class key-agent
composition, Bytewax lifecycle stream enforcement, cloud federation adapter
execution, API helpers, UI view models, semantic-model evidence, and focused
package verification.

The current review-evidence packet adds durable policy evidence for
review-required operations, export approvals, rotation exceptions, key
rotations, privileged key-agent review, denied Bytewax lifecycle batch routing,
and audit events.

Historical provider ideas for HSM, KMS, vault, blockchain audit, AI lifecycle,
security intelligence, compliance, SIEM, SOAR, DLP, monitoring, and
notification systems are adapter backlog items. They must preserve the
dependency-light service contract and fail-closed guardrails before becoming
runtime dependencies.

The cloud federation runtime now initializes AWS, Azure, GCP, IBM Cloud,
Oracle Cloud, Alibaba Cloud, DigitalOcean, and Vultr as deterministic APG
adapter descriptors. It can create federated references, synchronize replicas,
rotate references, fail over to healthy backups, migrate metadata-only custody
between providers, report federation status, estimate provider cost, and map
compliance coverage without importing provider SDKs into the package runtime.
