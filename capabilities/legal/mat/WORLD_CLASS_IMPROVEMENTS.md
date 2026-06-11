# leg_mat — World-Class Improvement Plan

## I1. AI-Driven Conflict of Interest Detection
**Category**: AI/ML
**Justification**: Conflict checks are the #1 malpractice risk and new business bottleneck. Automating cross-matter conflict searches across parties, adverse interests, and prior representations eliminates 30–60 minutes of manual work per intake and drastically reduces missed-conflict liability — a known gap in most boutique and mid-size firm tools.
**Implementation**: On matter creation, extract all party names from `client_id`, opposing parties, and `metadata`. Run async fuzzy-match against all existing matters using `rapidfuzz` (token-sort ratio ≥ 85). Persist conflict candidates to a `conflict_checks` store with `status` (`clear` | `flagged` | `waived`). Expose `run_conflict_check()` and `waive_conflict()` methods. Auto-flag matters with unresolved conflicts; block closing without resolution.
**Competitor**: Clio Grow, Intapp Conflicts

---

## I2. Statute of Limitations Calendar Engine
**Category**: Compliance
**Justification**: SoL miscalculation is one of the most common and costly legal malpractice claims. A rules-based engine that calculates tolling, holiday offsets, and jurisdiction-specific SoL periods — and automatically creates deadline records with multi-tier reminders — removes manual calculation error entirely.
**Implementation**: Define a `SoLRule` registry keyed by `(jurisdiction, matter_type, cause_of_action)` storing base period (days), tolling rules (minor, discovery, fraudulent concealment), and public holiday calendars per jurisdiction. `calculate_sol_deadline()` ingests matter parameters, runs through the registry, applies tolling, skips weekends/holidays using `workalendar`, and auto-creates a `statute_of_limitations` deadline with reminders at 365, 90, 30, 14, 7 days.
**Competitor**: LexisNexis Deadline Assistant, CompuLaw

---

## I3. Multi-Level Budget Tracking with Burn Rate Forecasting
**Category**: Feature
**Justification**: Clio and PracticePanther surface budget burn in real time with trend lines. Most legal teams only discover overruns after the fact. Burn-rate forecasting using historical time entries lets partners intervene before clients receive shock invoices, improving realization rates and client retention.
**Implementation**: Extend `time_budgets` with a `time_entries` sub-store holding `{entry_id, attorney_id, hours, billable, rate, narrative, date}`. Add `log_time_entry()`, `list_time_entries()`, `get_budget_burn_report()`. Burn-rate forecast: linear regression on daily entry totals using `numpy` (or a pure-Python fallback), projecting to matter `closed_date`. Surface `days_to_overrun` and `projected_final_hours` in the dashboard.
**Competitor**: Clio, Bill4Time, TimeSolv

---

## I4. Court Filing Deadline Calculator with Rules Engine
**Category**: Compliance
**Justification**: Federal/local civil procedure rules govern response periods, service windows, and extension mechanics. Manual tracking causes missed filings. Competitors like CompuLaw charge $10k+ per seat for this. A rules-driven engine embedded in the capability closes the gap.
**Implementation**: Define `CourtRule` records per `(court, rule_type)` storing `base_days`, `service_method_additions` (e.g. +3 for mail), `weekend_rule` (next business day), `holiday_calendar`. `calculate_filing_deadline(matter_id, trigger_event, trigger_date, rule_type)` applies the chain and returns a proposed deadline that can be reviewed and auto-created. Persist rules to `court_rules` dict; seed common rules (FRCP 12(a), CPR 15, etc.).
**Competitor**: CompuLaw, Deadlines on Demand

---

## I5. Matter Lifecycle State Machine with Guards
**Category**: Feature
**Justification**: Ad-hoc status updates allow invalid transitions (e.g. reopening archived matters, closing matters with unpaid invoices). A formal FSM with guards enforces business rules consistently, produces a clean audit trail, and prevents data integrity issues that corrupt analytics.
**Implementation**: Define `MATTER_TRANSITIONS: dict[str, set[str]]` as the allowed FSM. Implement `transition_matter_status(tenant_id, matter_id, new_status, actor_id, reason)` that validates the transition, runs pre-condition guards (open task count, conflict resolution, budget approval), persists the transition with full metadata to `_audit_events`, and returns the updated matter. Expose `get_matter_history()` to return the full status timeline.
**Competitor**: Fulcrum (Matter FSM), Litify

---

## I6. Attorney Capacity Planning & Load Balancing
**Category**: Feature
**Justification**: Uneven workload distribution is the primary driver of burnout and attrition in law firms. Real-time capacity dashboards — showing hours committed vs. available, matter priority weighting, and suggested re-assignment — are a key differentiator in tools like Actionstep and PracticePanther.
**Implementation**: `get_team_capacity_report(tenant_id, date_range)` aggregates: open task count × estimated hours, active matter count, upcoming deadline density, and time budget remaining per attorney. Normalise to a `load_score` (0–100). `suggest_reassignment(tenant_id, matter_id, task_id)` returns the top 3 underloaded attorneys with compatible practice area and current availability. Store capacity preferences (max_hours_per_week, out_of_office dates) in an `attorney_profiles` dict.
**Competitor**: Actionstep, Clio

---

## I7. Document Checklist Templates per Matter Type
**Category**: Feature
**Justification**: Litigation matters require a predictable set of initial tasks (complaint, service, initial disclosures, scheduling order). Manually recreating these for each new matter wastes billable time and causes omissions. Template-driven task seeding is table-stakes in Clio and MyCase.
**Implementation**: Define `MATTER_TEMPLATES: dict[str, list[dict]]` mapping matter type → list of `{title, task_type, relative_due_days, priority}`. `apply_matter_template(tenant_id, matter_id, template_id, start_date, assigned_to_id)` bulk-creates tasks with due dates calculated from `start_date + relative_due_days`, skipping non-business days. Allow custom templates via a `matter_templates` store with CRUD methods.
**Competitor**: Clio, MyCase, Smokeball

---

## I8. Privilege Log Generation
**Category**: Compliance
**Justification**: In discovery-heavy litigation, producing a privilege log is mandatory and tedious. Auto-generating a formatted privilege log from `is_privileged=True` notes and documents — with required fields (author, date, recipient, basis, description) — eliminates hours of manual compilation and reduces waiver risk.
**Implementation**: Add `privilege_basis` and `recipients` fields to privileged notes on creation. `generate_privilege_log(tenant_id, matter_id, format)` filters `notes` where `is_privileged=True`, assembles log entries conforming to FRCP 26(b)(5) / CPR 31 field requirements, and returns structured list (or renders to CSV/JSON). Include `log_hash` (SHA-256 of sorted entries) for tamper detection.
**Competitor**: Relativity, Clio Discovery

---

## I9. Smart Deadline Chaining (Trigger → Derived)
**Category**: Feature
**Justification**: Court deadlines rarely exist in isolation — a complaint triggers a response deadline, which triggers a reply, which triggers a scheduling conference. Clio Scheduler and CompuLaw model these dependency chains. Manual entry of all downstream deadlines is error-prone and common source of malpractice.
**Implementation**: `DeadlineChain` registry maps `trigger_event_type → list[{title, offset_days, deadline_type, reminder_days}]`. `create_chained_deadlines(tenant_id, matter_id, trigger_event, trigger_date)` iterates the chain, creates each derived deadline with correct offset, and links them via `parent_deadline_id`. `get_deadline_chain(tenant_id, deadline_id)` returns the full tree. Support custom chain definitions stored in `deadline_chains` dict.
**Competitor**: CompuLaw, Clio

---

## I10. Matter Spend Analytics with Invoice Reconciliation
**Category**: Feature
**Justification**: Legal departments consistently rank cost visibility as top priority. Surfacing accrued fees vs. budget, by timekeeper category and task phase, enables proactive spend control. This is core functionality in Brightflag, SimpleLegal, and TeamConnect that boutique tools universally lack.
**Implementation**: Add `invoices` store with `{invoice_id, matter_id, vendor_id, amount, period, line_items, status}`. `reconcile_invoice(tenant_id, matter_id, invoice_id)` matches line items against time entries; flags discrepancies (rate variance > 5%, duplicate entries). `get_spend_report(tenant_id, matter_id)` returns: total invoiced, total accrued (unbilled time entries × rate), budget remaining, and variance by timekeeper. Use `decimal.Decimal` for monetary precision.
**Competitor**: Brightflag, SimpleLegal, TeamConnect

---

## I11. Integrated Court Date Sync (iCal / CalDAV Export)
**Category**: Integration
**Justification**: Attorneys live in calendar applications. Court dates and deadlines that exist only in the legal management system get missed. Clio, PracticePanther, and Smokeball all provide calendar sync. Without it, the system is a secondary tool requiring manual re-entry.
**Implementation**: `export_matter_ical(tenant_id, matter_id)` generates a valid RFC 5545 iCalendar string from all docket entries and deadlines. Each event includes VALARM components for `reminder_days`. `export_attorney_ical(tenant_id, attorney_id)` aggregates events across all assigned matters. Produce `VCALENDAR/VEVENT` manually (no external dependency) or via `icalendar` lib. Return UTF-8 bytes with correct MIME type `text/calendar`.
**Competitor**: Clio, PracticePanther, Smokeball

---

## I12. Risk Scoring per Matter
**Category**: AI/ML
**Justification**: Not all matters carry equal risk. Surfaces with high deadline density, approaching SoL, budget overrun, and open conflict flags are objectively more dangerous. A composite risk score lets managing partners triage attention and proactively avoid malpractice.
**Implementation**: `compute_matter_risk_score(tenant_id, matter_id)` produces a normalised 0–100 score from: overdue tasks (×10), unresolved conflicts (×25), SoL within 30 days (×20), budget burn > 80% (×15), overdue deadlines (×15), days since last activity (×15). Return `{score, risk_level, contributing_factors}`. Run scoring on demand or batch via `batch_risk_scores(tenant_id)` returning sorted list.
**Competitor**: Gallagher Bassett, Litify

---

## I13. Secure Client Portal Activity Feed
**Category**: Security
**Justification**: Clients increasingly demand self-service visibility into matter status without requiring attorney time. A structured, access-controlled activity feed (filtered to non-privileged events) provides real-time status while enforcing privilege. ClientSide (Clio's portal) is a significant competitive differentiator.
**Implementation**: `get_client_activity_feed(tenant_id, matter_id, client_token)` validates `client_token` (HMAC-SHA256 signed with matter ID + expiry) then returns non-privileged events: status changes, task completions, deadline status, docket entries. Privilege filter: exclude notes with `is_privileged=True` and audit events with `event_type` in `PRIVILEGED_EVENT_TYPES`. Return paginated feed with `cursor`-based pagination.
**Competitor**: Clio Client Portal, MyCase Client Portal

---

## I14. Bulk Matter Import / Migration
**Category**: Integration
**Justification**: Every firm migrating from Clio, PracticePanther, or spreadsheets needs to import historical matters. A validated bulk import with conflict detection, de-duplication, and rollback support reduces onboarding friction from weeks to hours — a primary adoption barrier for new clients.
**Implementation**: `bulk_import_matters(tenant_id, records, dry_run, conflict_resolution)` accepts a list of matter dicts, validates each against `MatMatterCreate` schema, checks for duplicates by `(client_id, title, opened_date)`, applies `conflict_resolution` strategy (`skip` | `overwrite` | `version`). Returns `{imported, skipped, errors}` report. In `dry_run=True` mode, validates and reports without writing. Use `asyncio.gather` with bounded semaphore for concurrent validation.
**Competitor**: Clio, PracticePanther

---

## I15. Automated Status Digest Notifications
**Category**: UX
**Justification**: Attorneys miss updates because they don't poll the system. Push notifications for upcoming deadlines, overdue tasks, and matter status changes — delivered via webhook, email, or Slack — move the system from passive repository to active co-worker. Clio Alerts and Smokeball Reminders are differentiating features that increase daily active usage.
**Implementation**: `register_notification_channel(tenant_id, channel_type, config)` stores webhook URL, email address, or Slack webhook per tenant. `dispatch_deadline_digest(tenant_id, lookahead_days)` generates a per-attorney digest of upcoming deadlines and overdue items, then fans out to registered channels via `httpx.AsyncClient.post`. Use exponential backoff (3 retries). `get_notification_log(tenant_id)` returns delivery history with status codes.
**Competitor**: Clio Alerts, Smokeball Reminders, Actionstep
