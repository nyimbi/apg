# Data Loss Prevention Capability Specification

- **Capability Name**: Data Loss Prevention
- **Capability ID**: `dlpd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`dlpd` provides tenant-scoped sensitive-data discovery, egress inspection,
large-export review, quarantine, incident response, and audit evidence for APG
applications. It turns the executable capability contract into a dependency-light
Python runtime that generated applications can compose without requiring a live
mail gateway, proxy, CASB, object-store scanner, SIEM, or ticketing platform.

## Current Runtime Behavior

The package currently provides:

- DLP policies binding owners, channels, classifier IDs, egress-policy state,
  large-export review, and default response action.
- Data classifier registration with custom-pattern review enforcement.
- Deterministic local classification for PII, PHI, PCI, secrets, financial
  records, and source-code signals.
- Egress inspection decisions with APG rule evaluation for tenant context,
  egress policy, classification labels, high-severity blocking/quarantine,
  encrypted quarantine, and large-export review.
- Encrypted quarantine vault entries for quarantined sensitive content.
- Incident opening and resolution for blocked or quarantined transfers.
- Dashboard, policy console, incident queue, quarantine, and audit view models.
- Append-only audit events with stable digests.

Runtime state is in-memory and dependency-light. Durable storage, inline network
enforcement, mail gateways, proxies, endpoint agents, object storage scanners,
SIEM/SOAR integration, ticketing, legal-hold systems, and notification delivery
are external integration boundaries.

## Provided Services

- `sensitive_data_discovery`
- `channel_inspection`
- `exfiltration_detection`
- `incident_response`
- `policy_enforcement`

## Required Services

- `secu` for security posture and access-policy composition.
- `encr` for production encryption and key handling behind quarantine.
- `nlpc` for richer NLP classification beyond deterministic local patterns.
- `anom` for anomaly context and exfiltration behavior scoring.

Optional composition points are `audl`, `mqeb`, `srch`, and `comp`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The active configuration includes data-pattern
classifiers, inspected channels, response settings, governance constraints, UI
route metadata, and theme metadata.

Tenant context is required for executable operations.

## Rules

- `tenant_context_required`
- `inspection_source_requires_policy`
- `sensitive_content_requires_classification`
- `high_severity_exfiltration_requires_block`
- `quarantine_requires_encryption`
- `large_export_requires_review`

The service evaluates these rules before state changes that inspect egress,
quarantine content, or create review gates.

## UI

The package exposes 8 APG Python UI route contracts through `views.py` and the
package semantic model:

- dashboard
- policies
- classifiers
- channels
- incidents
- quarantine
- analytics
- settings

## Theme

The package uses the `dlpd_data_protection_ops` APG theme contract, including
classifier-grid, channel-flow, incident-queue, and quarantine-vault component
tokens.

## Verification

Focused package verification should include:

```bash
./.venv/bin/python -m py_compile capabilities/common/dlpd/*.py capabilities/common/dlpd/tests/*.py
./.venv/bin/pytest -q capabilities/common/dlpd/test_capability_contract.py capabilities/common/dlpd/tests
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dlpd --json
```
