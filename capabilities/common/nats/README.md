# APG NATS JetStream Event Bus (`nats`)

**Version**: 1.0.0 | **Domain**: common

## Overview

Durable publish/subscribe event bus via NATS JetStream. Replaces in-process event dispatch with crash-resilient message delivery.

## Usage

```python
from apg_common_nats import *
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

- **I1.** NATS JetStream — World-Class Improvements
- **I2.** Key-Value Store API
- **I3.** Object Store API
- **I4.** Stream Mirror / Source Federation
- **I5.** Exactly-Once Publish with Idempotency Token
- **I6.** Pull Consumer with Batch Fetch and Backpressure
- **I7.** Push Consumer with Flow Control
- **I8.** Message Replay from Sequence or Timestamp
- **I9.** Dead-Letter Stream (DLQ)
- **I10.** Subject-Scoped Authorization Tokens
- **I11.** Metrics Exporter for Prometheus / OpenTelemetry
- **I12.** Circuit Breaker and Fallback Queue
- **I13.** Stream Snapshot and Restore
- **I14.** Header-Driven Event Routing
- **I15.** Ordered Consumer for Event Sourcing

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
