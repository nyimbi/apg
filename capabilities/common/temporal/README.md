# APG Temporal Durable Workflow (`temporal`)

**Version**: 1.0.0 | **Domain**: common

## Overview

Durable workflow execution via Temporal.io. Workflow state survives process restarts. Supports human tasks, guards, and timers.

## Usage

```python
from apg_common_temporal import *
```

## Governance Rules

- tenant_context_required
- operation_type_required  
- audit_logged
- access_controlled

## License

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — Temporal Capability
- **I2.** Batch Workflow Operations
- **I3.** Workflow Update (Temporal Update API)
- **I4.** Structured Workflow Metadata / Search Attributes
- **I5.** Saga / Compensation Pattern Support
- **I6.** Workflow Version / Patch Management
- **I7.** Durable Timers with Named Cancellation
- **I8.** Continue-As-New Lifecycle Hook
- **I9.** Worker Versioning / Build-ID Groups
- **I10.** Nexus Service Calls
- **I11.** Structured Concurrency via Child Workflows
- **I12.** Dead-Letter Queue / Poison Pill Handling
- **I13.** Workflow Tagging / Label System
- **I14.** Execution Replay / Test Harness Hooks
- **I15.** Observability: Structured OpenTelemetry Spans

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
