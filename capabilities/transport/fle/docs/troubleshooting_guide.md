# Fleet Management — Troubleshooting Guide

## Common Errors

### `PermissionError: non_compliant_vehicle_dispatch_denied`
Vehicle has a failed inspection or expired COF. Check `/fle/compliance` and resolve outstanding items before dispatching.

### `PermissionError: vehicle_type_not_supported`
The vehicle type string is not in the supported list. Use one of: `rigid_truck`, `articulated_truck`, `van`, `pickup`, `tractor_unit`, `trailer`, `tanker`, `refrigerated_vehicle`, `flatbed`, `tipper`, `minibus`, `motorcycle`, `electric_vehicle`, `bus`, `crane_truck`.

### `ValidationError: vin → Value must be non-empty`
VIN is required and must be at least 11 characters.

### `RuleViolation: VEH-003 — VIN already registered`
Duplicate VIN within the tenant. Each vehicle must have a unique VIN.

### `RuleViolation: TACHO-001 — Continuous driving exceeds limit`
EU tachograph rule violation. Driving session exceeds 270 minutes (4h30m). Insert a break record before submitting more driving time.

### `RuleViolation: INC-002 — police reference required`
Fatal/critical incidents require a police OB (occurrence book) reference. Obtain the reference and re-submit.

### `AssertionError: Vehicle <id> not found`
Vehicle ID does not exist or belongs to a different tenant. Check the ID and the `X-Tenant-ID` header.

### `AssertionError: Tenant mismatch`
The `tenant_id` in the request body does not match the `X-Tenant-ID` header. They must match.

### `pydantic_core.ValidationError: Extra inputs are not permitted`
A field in the request body is not defined on the model. Check the API reference for the correct fields.

## Performance

- For high-volume telematics ingestion, use the batch API and ensure `fle_telematics_events` is partitioned by month in PostgreSQL.
- Add year-specific partitions: `CREATE TABLE fle_trips_2026 PARTITION OF fle_trips FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');`

## Database

Check table sizes:
```sql
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
WHERE relname LIKE 'fle_%'
ORDER BY pg_total_relation_size(relid) DESC;
```

Check compliance calendar data:
```sql
SELECT v.registration, i.expires_at, i.cof_number
FROM fle_cof_inspections i JOIN fle_vehicles v ON v.id = i.vehicle_id
WHERE i.expires_at < NOW() + INTERVAL '30 days'
  AND v.tenant_id = 'your_tenant_id'
ORDER BY i.expires_at;
```
