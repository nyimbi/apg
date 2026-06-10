# Process Mining (pmin)

Infer BPMN process models from NATS event streams, conformance checking, bottleneck analysis, and variant discovery.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/pmin/health | Service health |
| GET | /api/pmin/logs | List event logs |
| POST | /api/pmin/logs | Create event log |
| GET | /api/pmin/logs/{id} | Get event log |
| PUT | /api/pmin/logs/{id} | Update event log |
| DELETE | /api/pmin/logs/{id} | Delete event log |
| POST | /api/pmin/logs/{id}/events | Ingest events |
| POST | /api/pmin/logs/{id}/events/nats | Ingest from NATS |
| GET | /api/pmin/logs/{id}/events | Query events |
| GET | /api/pmin/logs/{id}/cases/{case_id} | Case trace |
| POST | /api/pmin/logs/{id}/discover | Discover BPMN model |
| POST | /api/pmin/logs/{id}/bottlenecks | Bottleneck analysis |
| POST | /api/pmin/logs/{id}/variants | Variant discovery |
| GET | /api/pmin/logs/{id}/performance | Performance metrics |
| GET | /api/pmin/models | List BPMN models |
| GET | /api/pmin/models/{id} | Get model |
| DELETE | /api/pmin/models/{id} | Delete model |
| GET | /api/pmin/models/{id}/xml | Export BPMN 2.0 XML |
| POST | /api/pmin/models/{id}/simulate | Process simulation |
| POST | /api/pmin/conformance | Check conformance |
| GET | /api/pmin/conformance | List results |
| POST | /api/pmin/conformance/deviating-cases | Deviating cases |
| GET | /api/pmin/bottlenecks | List bottleneck reports |
| GET | /api/pmin/bottlenecks/{id} | Get report |
| GET | /api/pmin/variants | List variant analyses |
| GET | /api/pmin/variants/{id} | Get analysis |
| GET | /api/pmin/dashboard | Dashboard |
| GET | /api/pmin/audit | Audit trail |
