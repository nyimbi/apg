# USSD Engine — World-Class Improvement Proposals

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

### I1. Multi-Language Menu Fallback Chain
**Category:** Internationalization
**Justification:** East African operators serve Swahili, English, Kikuyu, Luo, and Sheng speakers. A static `en` fallback loses users when a locale-specific menu is missing. A priority chain (`sw` → `sw-KE` → `en`) dramatically improves retention in rural markets.
**Implementation:** Add `resolve_menu_chain(menu_id, service_code, language, tenant_id)` that tries decreasing specificity variants before raising KeyError. Store language priority lists per gateway config.
**Competitor Reference:** Yo! Uganda USSD platform supports multi-lingual menus with cascade fallback; Comviva MobiLytix uses locale chains across 30+ African networks.

---

### I2. Session Resumption After Timeout
**Category:** Session Resilience
**Justification:** Network drops are common on 2G/2.5G. When a session times out mid-flow but the user re-dials within a grace window, resuming from the last valid menu removes friction and prevents re-entering data (e.g. PIN, amount). M-Pesa STK push handles this with a 90-second grace window.
**Implementation:** `resume_session(phone_number, service_code, grace_seconds, tenant_id)` — find the most recent timed-out session within the grace window and re-activate it with `hop_count` preserved. Emit `session_resumed` audit event.
**Competitor Reference:** Africa's Talking USSD Gateway's session continuation header; Safaricom DARAJA resume semantics for C2B flows.

---

### I3. Idempotent Transaction Execution
**Category:** Financial Integrity
**Justification:** USSD networks can deliver duplicate callbacks (GSM DTAP retransmission). A debit triggered twice from a duplicate POST loses customer money. Idempotency keys scoped per session+hop prevent double execution.
**Implementation:** `execute_idempotent(session_id, hop_count, handler_name, payload, tenant_id)` — hash `session_id:hop_count:handler_name` as idempotency key; return cached result if already executed. Use `Decimal` for all monetary fields in cached results.
**Competitor Reference:** Stripe's idempotency-key header model; Interswitch TransAct idempotency for USSD banking flows.

---

### I4. Rate Limiting Per Phone Number
**Category:** Abuse Prevention / Security
**Justification:** USSD bots and fraudulent bulk-diallers can exhaust session capacity and trigger expensive per-session billing. Per-phone rate limits (e.g. 10 sessions/hour) block scraping and credential stuffing at the engine layer before reaching downstream services.
**Implementation:** `check_rate_limit(phone_number, service_code, tenant_id, window_seconds, max_sessions)` — bucket phone+service into a sliding window counter; return `{"allowed": bool, "remaining": int, "reset_at": str}`.
**Competitor Reference:** Vonage USSD API rate limiting docs; AT sandbox throttle at 1 req/s per phone.

---

### I5. Input Validation Schema per Menu Item
**Category:** Data Quality / Security
**Justification:** Free-text USSD inputs (amounts, account numbers, PINs) arrive as raw strings. Without validation, garbage propagates to downstream payment processors. Regex + type + range constraints per input item catch errors before handler execution, surfacing an inline error message instead of a failed transaction.
**Implementation:** `validate_input_against_schema(value, schema, tenant_id)` — schema dict with `type`, `pattern`, `min_value`, `max_value`, `max_length`. Returns `{"valid": bool, "error_message": str | None}`. Use `Decimal` for numeric range checks.
**Competitor Reference:** MicroFocus USSD Developer Toolkit input validation DSL; Comviva MobiLytix form-field constraint engine.

---

### I6. Menu Versioning and Rollback
**Category:** Operational Excellence
**Justification:** A bad menu deploy (broken navigation, wrong pricing text) during peak hours on a live service causes revenue loss. Versioned snapshots allow a single API call to atomically roll back to the last known-good version without a full redeploy.
**Implementation:** `create_menu_version(menu_id, service_code, tenant_id)` snapshots the current menu; `rollback_menu(menu_id, service_code, version, tenant_id)` restores it. Store versions keyed `{composite_key}:v{n}`.
**Competitor Reference:** Twilio Studio's flow versioning and rollback; Africa's Talking dashboard "revert to previous" menu action.

---

### I7. Bulk Session Import from Gateway
**Category:** Operations / Migration
**Justification:** When migrating live sessions from one gateway to another (e.g. AT sandbox → AT production, or AT → Safaricom), session continuity requires bulk importing gateway session records. Without this, all in-flight users lose context.
**Implementation:** `bulk_import_sessions(sessions_payload, source_gateway, tenant_id)` — validate, normalize, and insert with deduplication. Returns `{imported, skipped_duplicate, failed}`.
**Competitor Reference:** Infobip USSD session migration API; Vonage session handoff during gateway failover.

---

### I8. Conditional Menu Item Weighting and A/B Testing
**Category:** Growth / Conversion Optimization
**Justification:** Operators run promotions where different user segments see different menu options (e.g. loan product vs savings product). Hard-coded menu items cannot segment users without custom code per client. A/B weight assignment at the menu engine layer enables controlled experiments without app deploys.
**Implementation:** `assign_ab_variant(session_id, experiment_id, variants, weights, tenant_id)` — deterministically assigns a variant using `hash(session_id + experiment_id) % 100` vs cumulative weight buckets. Stores variant in session variables.
**Competitor Reference:** Tyntec USSD analytics with A/B test support; Branch.io mobile deep link A/B testing applied to USSD user paths.

---

### I9. Real-Time Session Broadcast / Webhook Delivery
**Category:** Integration / Real-Time Notifications
**Justification:** Downstream systems (CRM, fraud detection, loyalty platforms) need real-time session events. Polling the audit log is high-latency and expensive at scale. Webhook delivery per event type with retry-with-backoff is the industry standard integration pattern.
**Implementation:** `register_webhook(url, events, tenant_id, secret)` — stores webhook config. `deliver_webhook(event, tenant_id)` — async HTTP POST with HMAC-SHA256 signature header and exponential backoff (3 retries). Returns delivery receipt.
**Competitor Reference:** Africa's Talking webhook events for USSD sessions; Twilio webhook retry policy (4 attempts, exponential backoff).

---

### I10. Session Encryption for PII at Rest
**Category:** Security / Compliance
**Justification:** USSD sessions collect phone numbers, PINs (typed as free text), account numbers, and amounts. Storing these in plaintext violates GDPR, Kenya DPA 2019, and PCI-DSS. Encrypting session variable payloads with a per-tenant key satisfies data-at-rest requirements without changing the external API.
**Implementation:** `encrypt_session_variables(session_id, tenant_id)` / `decrypt_session_variables(session_id, tenant_id)` — use Fernet (AES-128-CBC + HMAC-SHA256) from `cryptography` package with a per-tenant key from environment/vault. Variables stored as `{"_enc": "<ciphertext>"}` sentinel dict.
**Competitor Reference:** Interswitch TransAct AES-256 session storage; Safaricom M-Pesa session token encryption standard.

---

### I11. Dead-Letter Queue for Failed Handler Executions
**Category:** Reliability / Observability
**Justification:** When an `execute` handler raises an exception (e.g. payment service down), the current code logs and silently continues. The user sees no error, and the failed operation is invisible to ops. A dead-letter store captures the failed context (session, menu, input, exception) for replay or alerting.
**Implementation:** `queue_dead_letter(session_id, handler_name, payload, error, tenant_id)` — appends to `self._dead_letters` with timestamp and retry count. `get_dead_letters(tenant_id, handler_name, limit)` exposes queue for ops dashboards and automated replay.
**Competitor Reference:** AWS SQS Dead Letter Queues; RabbitMQ dead-letter exchange pattern applied to USSD handler failures.

---

### I12. Paginated Session and Audit Log Queries
**Category:** Performance / Scalability
**Justification:** At production scale (10M+ sessions), `list_sessions()` returning all in-memory records causes O(n) memory copies and multi-second API responses. Cursor-based or offset pagination with filtering allows dashboards to load within SLA.
**Implementation:** `list_sessions_paginated(tenant_id, page, page_size, filters, sort_by, sort_dir)` — applies filters then slices the sorted list; returns `{"items": [...], "total": int, "page": int, "pages": int}`. Uses `guard_page` from reliability.
**Competitor Reference:** Stripe list API with `starting_after` cursor; Africa's Talking USSD report API page/limit parameters.

---

### I13. Phone Number Masking and Anonymization
**Category:** Privacy / Compliance
**Justification:** Analytics queries and audit logs should not expose raw MSISDNs to non-privileged roles. Masking (e.g. `+254712***678`) in analytics outputs satisfies the Kenya DPA 2019 data minimisation principle without removing analytical utility.
**Implementation:** `get_anonymized_analytics(tenant_id, service_code)` — wraps `get_session_analytics` but replaces phone numbers in any per-phone breakdowns with masked variants via `_mask_phone(phone)` utility. `_mask_phone` keeps country code + last 3 digits.
**Competitor Reference:** Jumia USSD analytics dashboard phone masking; PCI-DSS requirement 3.4 tokenization of PANs applied to MSISDNs.

---

### I14. Menu Import/Export (JSON Schema)
**Category:** Developer Experience / Portability
**Justification:** Building menus via individual API calls is tedious for large deployments (50+ menus). A JSON schema import/export enables version-controlled menu definitions in git, CI/CD pipeline deployment, and cross-environment migration. Export produces a deterministic, re-importable snapshot.
**Implementation:** `export_menu_tree(service_code, tenant_id)` — serializes all menus for a service code into a portable JSON document with schema version header. `import_menu_tree(payload, tenant_id, overwrite)` — validates schema and bulk-creates/updates menus atomically.
**Competitor Reference:** Twilio Studio flow import/export JSON; Africa's Talking USSD menu export in dashboard.

---

### I15. Session Replay for Debugging
**Category:** Developer Experience / Observability
**Justification:** Diagnosing a failed transaction in a 12-hop session requires reconstructing every input, menu transition, variable assignment, and handler call from the audit log. Manual reconstruction is error-prone and slow. A replay engine re-executes the session input chain against the current menu tree and produces a step-by-step trace.
**Implementation:** `replay_session(session_id, tenant_id, stop_at_hop)` — extracts `input_history` from a completed session, creates a shadow session, and drives `handle_ussd_request` through each recorded input, capturing the response at each hop. Returns `[{"hop": int, "input": str, "menu": str, "response_type": str, "body": str}]`.
**Competitor Reference:** Postman Collection Runner for API replay; Interswitch USSD simulator with step-through mode.
