# Quantum Computing

**Capability ID**: `quan` | **Domain**: `common` | **Version**: `1.0.0`

## Description

QUAN gives APG applications a tenant-scoped quantum lab runtime: backend registry, provider credentials posture, quota policies, circuit library, job submission, deterministic result capture, experiment workbench, quantum agents,

## Installation

```bash
pip install apg-common-quan
```

## Provides

- `quantum_backend_registry`
- `circuit_management`
- `quantum_job_orchestration`
- `result_analysis`
- `post_quantum_governance`

## Requires

- `aicr`
- `encr`
- `keym`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/quan/dashboard` | `quan:view` | Overview |
| `/quan/backends` | `quan:manage_backends` | Backends |
| `/quan/circuits` | `quan:experiment` | Circuits |
| `/quan/jobs` | `quan:run_jobs` | Jobs |
| `/quan/experiments` | `quan:experiment` | Experiments |
| `/quan/results` | `quan:view` | Results |
| `/quan/agents` | `quan:admin` | Operations |
| `/quan/audit` | `quan:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `submit_quantum_job()`
- `job_status()`
- `job_result()`
- `quantum_error_mitigation()`
- `variational_quantum_eigensolver()`
- `quantum_approximate_optimisation()`
- `quantum_key_distribution()`
- `post_quantum_encryption()`

_(See `service.py` for complete API.)_

## Interoperability

`quan` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use quan;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `QUAN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
