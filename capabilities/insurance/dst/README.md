# Distribution & Agency Management (ins_dst)

Agent registry, commission management, performance tracking, compliance, bancassurance.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/dst/health | Health check |
| GET | /api/insurance/dst/describe | Capability description |
| GET | /api/insurance/dst/agents | List agents |
| POST | /api/insurance/dst/agents | Register agent |
| GET | /api/insurance/dst/agents/{id} | Get agent |
| PUT | /api/insurance/dst/agents/{id} | Update agent |
| DELETE | /api/insurance/dst/agents/{id} | Deregister agent |
| POST | /api/insurance/dst/agents/{id}/suspend | Suspend agent |
| POST | /api/insurance/dst/commissions | Compute commission |
| GET | /api/insurance/dst/commissions | List commissions |
| GET | /api/insurance/dst/commissions/{id} | Get commission |
| POST | /api/insurance/dst/commissions/{id}/approve | Approve commission |
| POST | /api/insurance/dst/commissions/{id}/pay | Pay commission |
| POST | /api/insurance/dst/compliance | Record compliance |
| GET | /api/insurance/dst/compliance | List compliance records |
| POST | /api/insurance/dst/bancassurance | Register partner |
| GET | /api/insurance/dst/bancassurance | List partners |
| POST | /api/insurance/dst/performance/{agent_id} | Performance report |
| GET | /api/insurance/dst/summary | Agency summary |
| GET | /api/insurance/dst/audit | Audit trail |
