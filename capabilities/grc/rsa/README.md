# Risk and Security Assessment

## Overview

Risk and Security Assessment provides a world-class, standalone-deployable implementation of risk and security assessment capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`grc_rsa`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| `security_assessment_lifecycle` | Security Assessment Lifecycle workflow |
| `vulnerability_finding_workflow` | Vulnerability Finding Workflow workflow |
| `remediation_tracking_workflow` | Remediation Tracking Workflow workflow |
| `vendor_risk_assessment_workflow` | Vendor Risk Assessment Workflow workflow |
| `threat_modelling_workflow` | Threat Modelling Workflow workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `mten` | Mten services |
| `conf` | Conf services |
| `ntfy` | Ntfy services |


## Installation

```bash
pip install apg-grc-rsa
```

## Standalone Usage

```python
from apg_grc_rsa import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # grc_rsa
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-grc-rsa --port 8080

# With PostgreSQL persistence
apg-grc-rsa --db-url postgresql+asyncpg://user:pass@localhost/rsa --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-rsa/dashboard` | `grc_rsa:view` |
| assessments | `/grc-rsa/assessments` | `grc_rsa:manage_assessments` |
| assessment_detail | `/grc-rsa/assessments/:id` | `grc_rsa:view` |
| findings | `/grc-rsa/findings` | `grc_rsa:manage_findings` |
| finding_detail | `/grc-rsa/findings/:id` | `grc_rsa:view` |
| remediation | `/grc-rsa/remediation` | `grc_rsa:manage_remediation` |
| vendor_risk | `/grc-rsa/vendor-risk` | `grc_rsa:manage_vendor_risk` |
| threat_model | `/grc-rsa/threat-model` | `grc_rsa:view` |


## HTTP Endpoints

```
GET  /health           Liveness probe
GET  /contract         Full capability contract JSON
POST /evaluate         Evaluate governance rules
GET  /api/v1/...       Domain-specific REST API
```

## Composability

This capability integrates with the APG platform via the `apg.capabilities` entry-point group. It is auto-discovered by the capability registry when installed.

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["grc_rsa"].contract
```

## Development

```bash
# Run tests
pytest tests/ -q

# Build wheel
python -m build --wheel .

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## License

Proprietary — © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>
