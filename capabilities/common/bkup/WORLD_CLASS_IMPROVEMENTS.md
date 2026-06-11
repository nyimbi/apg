# BKUP World-Class Improvements

15 high-impact improvements to elevate APG Backup & Recovery to production-grade.

---

## 1. Immutable Snapshot Ledger with Merkle Chain

**Category**: Data Integrity / Tamper Evidence

**Justification**: Backup data is only trustworthy if you can prove it hasn't been altered post-creation. Current implementation stores snapshots in mutable dict records — a compromised process can retroactively rewrite history. An append-only ledger with each entry hashing its predecessor produces cryptographic proof that the audit trail and snapshot chain are unbroken.

**Implementation**:
- Add `LedgerEntry` with `entry_hash: str` = SHA-256(`prev_hash + snapshot_id + created_at + size_bytes`)
- Store `_ledger: list[LedgerEntry]` separately from `_snapshots`
- `verify_ledger()` recomputes hashes end-to-end and returns break point if tampered
- `snapshot_hash` field on each snapshot record anchors it to the ledger

**Competitor Reference**: AWS Backup Vault Lock (WORM), Veeam Immutable Backup to S3 Object Lock

---

## 2. Multi-Region Replication with Quorum Tracking

**Category**: Availability / Disaster Recovery

**Justification**: Single-location backups fail silently when that region goes down. Quorum-based replication (e.g. 2-of-3 regions confirmed) is the standard for enterprise DR. Knowing precisely which regions hold a valid copy lets runbooks skip unavailable sites rather than fail the entire restore.

**Implementation**:
- `replicate_to_regions(snapshot_id, regions: list[str], quorum: int)` — records per-region `ReplicationStatus`
- `snapshot["region_copies"]: dict[str, str]` maps region → status (`pending|confirmed|failed`)
- `quorum_met(snapshot_id)` returns bool
- DR runbook gates RTO estimate on quorum confirmation

**Competitor Reference**: Zerto multi-site journaling, Cohesity SmartFiles geo-redundancy

---

## 3. Deduplication & Compression Savings Tracking

**Category**: Cost Optimisation

**Justification**: Storage is the dominant operational cost in large backup estates. Deduplication ratios and compression factors are headline KPIs that procurement and finance ask for quarterly. Tracking logical vs physical bytes per snapshot enables chargeback reports and storage tier decisions.

**Implementation**:
- `snapshot["logical_bytes"]` vs `snapshot["physical_bytes"]`
- `dedup_ratio: Decimal` = logical / physical (Decimal for precise cost arithmetic)
- `compression_ratio: Decimal`
- `storage_savings_report()` aggregates per-plan and tenant-wide savings in cost-basis dollars using `Decimal`

**Competitor Reference**: Veeam deduplication analytics, Commvault DeDup savings dashboard

---

## 4. Backup Cost Estimation with Decimal Precision

**Category**: FinOps / Cost Management

**Justification**: Organisations running hundreds of backup plans need accurate cost forecasting before retention policies are changed. Integer or float arithmetic accumulates rounding errors across thousands of snapshots. All monetary calculations must use `decimal.Decimal` with explicit rounding modes.

**Implementation**:
- `estimate_backup_cost(plan_id, storage_cost_per_gb: Decimal, egress_cost_per_gb: Decimal)` returns `Decimal` total
- `cost_breakdown_report()` produces per-plan `Decimal` monthly cost
- Uses `ROUND_HALF_UP` explicitly throughout
- Guard `tenant_id` via `guard_tenant_id` before any cost calculation

**Competitor Reference**: AWS Cost Explorer backup cost views, Azure Backup pricing calculator integration

---

## 5. SLA Breach Alerting with Configurable Thresholds

**Category**: Observability / Operations

**Justification**: RPO/RTO misses discovered during an incident are too late. Proactive SLA breach detection — raising alerts when gap_minutes approaches the RPO threshold — converts reactive firefighting into preventive operations. A breach event log with severity levels (warning/critical) enables escalation routing.

**Implementation**:
- `sla_breach_check(plan_id, warn_pct: float = 0.8)` raises `SLAWarning` at 80% RPO consumed, `SLABreach` at 100%
- `_sla_events: list[_R]` per-tenant breach log
- `list_sla_events(severity: str | None)` queries the log
- Audit event `sla_breach_detected` emitted on each breach

**Competitor Reference**: Rubrik SLA Policy Engine, Druva alert policies with severity tiers

---

## 6. Backup Simulation / Dry-Run Mode

**Category**: Reliability / Testing

**Justification**: Production backup jobs should be validated before first execution. A dry-run mode walks the full code path — plan lookup, source enumeration, retention check, encryption — but writes nothing durable. This is standard in enterprise schedulers and is the basis for pre-flight checks in runbooks.

**Implementation**:
- `backup_run(..., dry_run: bool = False)` — when `True`, compute all metadata, emit audit `backup_dryrun`, return record with `status="dry_run"` and do not persist to `_snapshots`
- `dry_run_report(plan_id)` lists what would be created/expired
- `dr_runbook_execute` supports `dry_run=True` for full rehearsal

**Competitor Reference**: Cohesity RunNow dry-run, Veeam "test job" execution

---

## 7. Snapshot Lifecycle State Machine with Explicit Transitions

**Category**: Correctness / Governance

**Justification**: Snapshots transitioning directly from `available` to `deleted` without intermediate states (`expiring`, `verifying`, `quarantined`) causes race conditions in concurrent workflows. Explicit FSM with allowed-transition guards prevents illegal state moves and makes lifecycle audits machine-readable.

**Implementation**:
- `SNAPSHOT_TRANSITIONS: dict[str, set[str]]` = `{"available": {"expiring","verifying","legal_hold","deleting"}, ...}`
- `_transition_snapshot(snapshot_id, new_status, actor)` validates transition, records event, raises `IllegalTransition` if invalid
- All existing methods route through `_transition_snapshot`
- `snapshot_lifecycle_history(snapshot_id)` returns ordered transition log

**Competitor Reference**: NetBackup snapshot lifecycle policies, Commvault lifecycle rules engine

---

## 8. Chunked Snapshot Transfer with Resume Support

**Category**: Resilience / Performance

**Justification**: Large snapshots (hundreds of GB) fail silently mid-transfer on flaky WAN links. Chunk-level tracking with a resume cursor allows retrying from the last confirmed chunk rather than retransmitting from byte zero. This is the mechanism behind every production cloud sync agent.

**Implementation**:
- `initiate_chunked_transfer(snapshot_id, destination, chunk_size_mb: int = 64)` → `ChunkedTransfer` with `transfer_id`, `total_chunks`, `confirmed_chunks: list[int]`
- `confirm_chunk(transfer_id, chunk_index)` — marks chunk confirmed, triggers completion when all done
- `resume_transfer(transfer_id)` — returns next unconfirmed chunk index
- `_chunked_transfers: dict[tuple, _R]` per-tenant

**Competitor Reference**: Veeam Cloud Connect transport with resumable transfers, Restic pack splitting

---

## 9. Cross-Tenant Backup Delegation

**Category**: Multi-Tenancy / Enterprise

**Justification**: MSPs and platform teams routinely manage backups on behalf of customer tenants. Delegation must be scoped (read-only vs restore-authorised vs full-admin) and time-bounded. Without explicit delegation records, privilege creep is unauditable and violates SOC 2 CC6.

**Implementation**:
- `delegate_backup_access(delegator_tenant_id, delegatee_tenant_id, scope: str, expires_at: str)` creates `DelegationRecord`
- `guard_delegation(actor_tenant, target_tenant, required_scope)` validates active non-expired record
- All cross-tenant reads/writes route through `guard_delegation`
- `list_delegations(tenant_id)` and `revoke_delegation(delegation_id)`

**Competitor Reference**: Commvault CommCell delegation, Druva inSync MSP portal

---

## 10. Automated Recovery Time Objective Calibration

**Category**: Intelligence / Continuous Improvement

**Justification**: RTO targets set at plan creation become stale as data volume grows. Historical restore durations should feed back into RTO targets automatically, surfacing plans where actual recovery consistently exceeds stated objective. This drives SLA renegotiation and infrastructure investment decisions.

**Implementation**:
- `calibrate_rto(plan_id)` computes rolling P95 from restore history; if P95 > `rpo_minutes * 2`, sets `rto_breach_flag=True` on plan
- `rto_trend_report(plan_id, window_days: int = 90)` returns weekly P50/P95 series
- `auto_rto_recommendations()` returns list of plans needing RTO uplift with suggested new values

**Competitor Reference**: Zerto Journal-based RTO analytics, Rubrik predictive analytics

---

## 11. Backup Anomaly Detection

**Category**: Security / Operations

**Justification**: A sudden 10x spike in snapshot size or a backup run at 3AM on a plan with a business-hours-only schedule are strong signals of ransomware staging or misconfiguration. Statistical baseline comparison against recent history catches anomalies before they become incidents.

**Implementation**:
- `detect_anomalies(plan_id, z_threshold: float = 3.0)` — computes rolling mean/stddev of `size_bytes` per backup type; flags snapshots where `|size - mean| > z * stddev`
- `_anomaly_log: list[_R]` stores detected anomalies with severity
- `list_anomalies(plan_id, severity: str | None)` query interface
- Audit event `anomaly_detected` emitted

**Competitor Reference**: Cohesity DataHawk ransomware detection, Rubrik anomaly detection

---

## 12. Backup Policy as Code (BPaC) Export/Import

**Category**: DevOps / GitOps

**Justification**: Manual GUI-configured backup policies are the first thing lost in a DR scenario. Exporting policy definitions as declarative YAML/JSON enables version control, peer review, and infrastructure-as-code pipelines. Import with conflict detection completes the round-trip.

**Implementation**:
- `export_policy_bundle(plan_ids: list[str])` → JSON blob including plan + schedule + retention policy for each ID
- `import_policy_bundle(bundle: str, conflict_mode: str = "skip")` → `ImportResult` with created/skipped/merged counts
- Schema includes `policy_version`, `exported_at`, `tenant_id` for validation on import
- Idempotent re-import (same `plan_id` + same config = no-op)

**Competitor Reference**: Veeam Backup as Code (PowerShell DSC), HashiCorp Vault snapshot policies as code

---

## 13. WORM (Write-Once Read-Many) Snapshot Locking

**Category**: Compliance / Ransomware Resilience

**Justification**: Regulatory frameworks (SEC 17a-4, FINRA, GDPR Article 5) require immutable audit records and, increasingly, immutable backup copies. A software-enforced WORM lock prevents deletion or modification of a snapshot before its lock expiry, even by administrators. This is distinct from legal hold — it is time-bounded and automatically releases.

**Implementation**:
- `worm_lock(snapshot_id, lock_until: str, reason: str)` sets `snapshot["worm_locked_until"]`; any delete/modify attempt before expiry raises `WORMLockActive`
- `_check_worm(snapshot_id)` guard called in `bulk_delete_snapshots`, `enforce_expiry`
- `list_worm_locked_snapshots()` for compliance reporting
- Audit event `worm_lock_applied`

**Competitor Reference**: NetApp SnapLock WORM, AWS S3 Object Lock with Compliance mode

---

## 14. Parallel Backup Execution with Concurrency Limits

**Category**: Performance / Throughput

**Justification**: Sequential backup runs across 50+ sources take unacceptably long when RPO windows are tight. `asyncio.gather` with a semaphore-bounded concurrency limit runs multiple source backups in parallel while preventing resource exhaustion. Completion events feed back into the catalogue atomically.

**Implementation**:
- `parallel_backup_run(plan_id, backup_type, max_concurrency: int = 4)` uses `asyncio.Semaphore(max_concurrency)` across source list
- Returns `ParallelRunResult` with per-source snapshot_id, duration_ms, status
- Failed sources logged without aborting the run; partial success surfaced in result
- Audit event `parallel_backup_completed` with concurrency stats

**Competitor Reference**: Veeam parallel processing, Cohesity distributed job scheduling

---

## 15. Continuous Data Protection (CDP) Journal

**Category**: Recovery Granularity / RPO Minimisation

**Justification**: Traditional scheduled backups have RPO equal to the backup interval. CDP journals every write as a micro-snapshot, enabling recovery to any second within the retention window. Even a lightweight in-memory journal of change events dramatically reduces RPO for critical plans and is the basis of modern cloud-native recovery products.

**Implementation**:
- `CDP_JOURNAL_ENABLED` flag on plan
- `journal_write_event(plan_id, source_id, change_summary: str, bytes_changed: int)` appends to `_cdp_journal: list[_R]`
- `cdp_restore_to_second(plan_id, target_datetime: str, target_environment: str)` replays journal from nearest full snapshot to exact second
- `cdp_journal_stats(plan_id)` returns event count, earliest/latest event, total bytes journaled

**Competitor Reference**: Zerto continuous journaling, Azure Continuous Backup for Cosmos DB
