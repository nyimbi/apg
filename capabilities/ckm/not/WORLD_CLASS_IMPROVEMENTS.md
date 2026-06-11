# World-Class Improvements: Notification System (ckm_not)

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## 1. Idempotent Delivery with Deduplication Keys

**Category**: Reliability / Correctness

**Justification**: Duplicate notifications destroy user trust. Network timeouts cause callers to retry, and without server-side idempotency, the same notification fires twice. AWS SNS, SendGrid, and Stripe all accept a caller-supplied idempotency key and silently deduplicate within a 24-hour window.

**Implementation**: Accept an optional `idempotency_key: str` on `send_notification`. Before dispatch, check `_idempotency_index: Dict[str, str]` keyed by `(tenant_id, idempotency_key)` mapping to `notification_id`. On hit, return the existing record immediately without re-sending. Expire keys after 24 hours using a TTL field or a periodic background sweep.

**Competitor Reference**: Stripe payment API — every POST accepts `Idempotency-Key` header; SendGrid X-Message-Id tracking; AWS SQS `MessageDeduplicationId`.

---

## 2. Adaptive Send-Time Optimisation (STO)

**Category**: Engagement / AI

**Justification**: Sending at the wrong time is the single biggest drag on open rates. Braze and Iterable model each recipient's historical open timestamps to predict the 30-minute window per day in which they are most likely to engage, lifting open rates 20-35%.

**Implementation**: Maintain a `_engagement_history: Dict[str, List[datetime]]` per `(tenant_id, recipient_id)`. On each `OPENED` event, append the UTC hour. Expose `async def optimal_send_window(self, recipient_id: str) -> datetime` that returns the next upcoming datetime matching the mode of the recipient's open-time histogram. `timezone_aware_send` delegates to this when `optimize_time=True`.

**Competitor Reference**: Braze Intelligent Timing; Iterable Send Time Optimization; OneSignal Intelligent Delivery.

---

## 3. Quiet-Hours Deferral with Timezone-Aware Scheduling

**Category**: Compliance / User Experience

**Justification**: TCPA in the US and GDPR recital 47 in the EU implicitly require not harassing recipients during sleep hours. The current `send_notification` has a timezone field but does not actually defer or block quiet-hours sends. Twilio Messaging and Customer.io enforce quiet hours at the execution layer, not just as metadata.

**Implementation**: Add `quiet_hours_start: int` (0-23) and `quiet_hours_end: int` to user preferences. In `send_notification`, convert `datetime.utcnow()` to the recipient's `zoneinfo` timezone, check if the local hour falls in the quiet window (handling midnight wrap-around), and if so, reschedule to `quiet_hours_end` in the recipient's timezone. Emit a `notification_delivery_deferred` audit event.

**Competitor Reference**: Customer.io Quiet Hours; Klaviyo Smart Send Time; Twilio Programmable SMS quiet-hours API.

---

## 4. A/B Template Testing with Statistical Significance Gating

**Category**: Optimisation / Analytics

**Justification**: Marketers spend weeks guessing which subject line works. Optimizely and Mailchimp run multivariate tests and auto-promote the winning variant only after reaching a configurable confidence threshold (default 95%), preventing premature promotion of noise.

**Implementation**: Add `async def create_ab_test(self, template_id: str, variant_body: str, variant_subject: str, traffic_split: float = 0.5, significance_threshold: float = 0.95) -> Dict`. Track opens/clicks per variant. Expose `async def evaluate_ab_test(self, test_id: str) -> Dict` that computes a chi-squared p-value against opens; auto-promote the winner when `p < (1 - significance_threshold)` and sample size >= 100.

**Competitor Reference**: Mailchimp A/B testing with confidence scoring; Braze multivariate experiments; Iterable experiments with auto-winner.

---

## 5. Provider Failover with Circuit Breaker Pattern

**Category**: Reliability / Resilience

**Justification**: A degraded Twilio account causes all SMS to fail silently. Implemented by Netflix (Hystrix), the circuit breaker opens after N consecutive failures, routes traffic to a secondary provider, and probes the primary after a reset timeout — without requiring operator intervention.

**Implementation**: Add `_circuit_breakers: Dict[str, Dict]` tracking `state` (CLOSED/OPEN/HALF_OPEN), `failure_count`, `opened_at`. In `_execute_multi_channel_delivery`, before dispatching to a provider, check the breaker. If OPEN and `opened_at + reset_timeout > now`, skip to the next provider. If HALF_OPEN, allow one probe; on success reset to CLOSED; on failure reopen. Expose `async def provider_circuit_status(self) -> Dict`.

**Competitor Reference**: Netflix Hystrix; AWS SDK retry with circuit breaker; Twilio helper libraries with fallback URL.

---

## 6. Webhook Delivery with Signed Payloads and Retry Queue

**Category**: Integration / Security

**Justification**: Webhook consumers need to verify the sender. GitHub, Stripe, and Shopify sign webhook payloads with HMAC-SHA256 using a shared secret; consumers reject unsigned or tampered requests. The current webhook channel has no signing and no retry queue.

**Implementation**: In `register_channel` for `channel_type="webhook"`, accept `signing_secret`. On dispatch, compute `HMAC-SHA256(secret, json_payload)` and attach as `X-APG-Signature-256` header. Add a `_webhook_retry_queue: asyncio.Queue` for 4xx/5xx responses; retry with exponential back-off up to 3 attempts. Expose `async def list_webhook_failures(self, channel_id: str) -> List[Dict]`.

**Competitor Reference**: GitHub webhook signatures; Stripe webhook verification; Shopify HMAC validation.

---

## 7. In-App Notification Feed with Read-State Management

**Category**: User Experience / Features

**Justification**: Users expect a persistent notification inbox, not just transient toasts. Intercom and Knock provide an in-app feed with unread counts, mark-as-read, mark-all-read, and archiving. The current service has no in-app feed concept beyond delivery tracking.

**Implementation**: Add `_inapp_feed: Dict[Tuple[str,str], List[Dict]]` keyed by `(tenant_id, recipient_id)`. `send_notification` with `channel="in_app"` appends to the feed. Expose `async def get_inapp_feed(self, recipient_id: str, unread_only: bool = False) -> List[Dict]`, `async def mark_read(self, recipient_id: str, notification_id: str) -> bool`, `async def mark_all_read(self, recipient_id: str) -> int`, and `async def get_unread_count(self, recipient_id: str) -> int`.

**Competitor Reference**: Intercom in-app messenger; Knock notification feed API; Novu in-app notification center.

---

## 8. Digest / Batching with Configurable Rollup Windows

**Category**: User Experience / Volume Control

**Justification**: High-frequency notification sources (monitoring alerts, CI pipelines) spam recipients. PagerDuty and Datadog support digest windows — buffer N notifications over T minutes and send a single summary. This cuts noise without losing information.

**Implementation**: Add `_digest_buffers: Dict[Tuple[str,str,str], List[Dict]]` keyed by `(tenant_id, recipient_id, digest_key)`. Expose `async def add_to_digest(self, recipient_id: str, digest_key: str, content: Dict, flush_after_seconds: int = 300) -> str` and `async def flush_digest(self, recipient_id: str, digest_key: str) -> Dict[str,Any]`. On flush, render a summary template and call `send_notification`. A background task fires `flush_digest` when the window expires.

**Competitor Reference**: PagerDuty alert grouping; Datadog event rollup; Slack notification digest.

---

## 9. Consent Lifecycle with GDPR/TCPA Evidence Trail

**Category**: Compliance / Legal

**Justification**: GDPR Article 7 requires demonstrable consent. TCPA requires written consent for marketing SMS. Without a timestamped, immutable consent record linked to every outbound message, the system cannot survive a regulatory audit. Sailthru and Listrak maintain consent provenance per channel.

**Implementation**: Add `_consent_registry: Dict[Tuple[str,str,str], Dict]` keyed by `(tenant_id, recipient_id, channel)` storing `consent_ref`, `consented_at`, `consent_method`, `ip_address`, `revoked_at`. Expose `async def record_consent(self, recipient_id: str, channel: str, consent_ref: str, method: str, ip: str) -> Dict`, `async def revoke_consent(self, recipient_id: str, channel: str, reason: str) -> bool`, and `async def verify_consent(self, recipient_id: str, channel: str) -> bool`. Block delivery if `verify_consent` returns False for external channels.

**Competitor Reference**: Sailthru consent management; Braze subscription groups with consent; OneTrust preference center integration.

---

## 10. Cost-Aware Channel Selection with Budget Caps

**Category**: Cost Management / Operations

**Justification**: SMS costs 75x email. Without budget guardrails, a misconfigured campaign drains budgets overnight. Braze and Customer.io expose per-campaign and per-channel spend caps with automatic fallback to cheaper channels when a cap is approached.

**Implementation**: Add `_channel_budgets: Dict[Tuple[str,str], Decimal]` (tenant, channel) and `_channel_spend: Dict[Tuple[str,str], Decimal]`. In `_execute_multi_channel_delivery`, before dispatching to each channel, check `_channel_spend[key] + estimated_cost <= _channel_budgets[key]`. If the cap would be exceeded, skip the channel and log a `budget_cap_exceeded` audit event. Expose `async def set_channel_budget(self, channel: str, budget: Decimal, period: str = "monthly") -> bool` and `async def get_spend_summary(self, period: Dict[str,str]) -> Dict`.

**Competitor Reference**: Braze spend caps; Customer.io channel budget controls; Twilio usage triggers.

---

## 11. Priority Lanes with Fair-Queue Scheduling

**Category**: Performance / SLAs

**Justification**: A marketing blast should not delay a transactional OTP notification. RabbitMQ and Kafka support priority queues. Without priority lanes, all notifications compete for the same concurrency slot regardless of urgency.

**Implementation**: Replace the single `asyncio.Semaphore` in `send_bulk` with four `asyncio.PriorityQueue` instances mapped to `CRITICAL/HIGH/NORMAL/LOW`. A dispatcher coroutine drains the highest-priority non-empty queue first, applying weighted round-robin to prevent starvation of lower lanes. Expose `async def queue_depth(self) -> Dict[str, int]` to surface per-priority backlog.

**Competitor Reference**: RabbitMQ priority queues; SQS FIFO with message groups; Celery task routing with priority workers.

---

## 12. Template Localisation with Locale Fallback Chain

**Category**: Internationalisation / UX

**Justification**: A Kenyan user receiving an email in US English when Swahili is preferred indicates a locale-blind system. Salesforce Marketing Cloud and Iterable maintain locale variants per template with a fallback chain: `sw_KE -> sw -> en_KE -> en` ensuring the most specific available locale is used.

**Implementation**: Extend `create_template` to accept `locale: str = "en"`. Store template records keyed by `(template_id, locale)`. In `render_template`, accept a `locale` parameter and resolve the fallback chain: split locale `sw_KE` → try `sw_KE`, then `sw`, then `en`. Expose `async def add_template_locale(self, template_id: str, locale: str, subject: str, body: str) -> Dict` and `async def list_template_locales(self, template_id: str) -> List[str]`.

**Competitor Reference**: Salesforce MC locale-aware sends; Iterable localisation; Klaviyo multi-language templates.

---

## 13. Real-Time Delivery Status Webhooks (Outbound)

**Category**: Integration / Developer Experience

**Justification**: Polling `track_delivery` is chatty and adds latency to downstream workflows. Twilio, SendGrid, and Mailgun push status events (delivered, opened, bounced) to caller-registered webhooks in real time, enabling event-driven downstream automation.

**Implementation**: Add `_status_webhooks: Dict[str, List[Dict]]` keyed by `notification_id` or `campaign_id`. Expose `async def subscribe_status_updates(self, entity_id: str, callback_url: str, events: List[str], signing_secret: str) -> Dict`. When delivery status changes in `send_notification` or `retry_failed`, fire signed POST requests to all registered callbacks via `asyncio.create_task` so the primary path is not blocked.

**Competitor Reference**: Twilio status callbacks; SendGrid Event Webhook; Mailgun webhook signing.

---

## 14. Notification Rate Limiting Per Recipient

**Category**: Compliance / Anti-Spam

**Justification**: Sending 20 notifications per hour to a single user constitutes harassment and violates CAN-SPAM and GDPR legitimate-interest criteria. Intercom and HubSpot apply per-recipient frequency caps: no more than N notifications per hour/day/week, configurable globally and overridable per campaign.

**Implementation**: Add `_recipient_rate_counters: Dict[Tuple[str,str,str], int]` keyed by `(tenant_id, recipient_id, window_key)` where `window_key` is `YYYY-MM-DD-HH`. In `send_notification`, before dispatch, check and increment the counter; reject with `RateLimitExceeded` if the hourly cap is exceeded. Expose `async def set_recipient_rate_limit(self, recipient_id: str, max_per_hour: int, max_per_day: int) -> bool` and `async def get_recipient_rate_status(self, recipient_id: str) -> Dict`.

**Competitor Reference**: Intercom frequency capping; HubSpot email sending limits; Iterable rate limiting.

---

## 15. Notification Archive with Full-Text Search

**Category**: Observability / Operations

**Justification**: Support teams need to answer "what did we send to customer X last week?" without querying raw database tables. Zendesk and Intercom maintain a searchable notification archive with filters by recipient, channel, status, template, and date range, with export to CSV/JSON.

**Implementation**: Extend `notification_history` to support `channel`, `status`, `template_id`, `date_from`, `date_to`, and `search_text` filters. Add `async def export_notifications(self, filters: Dict, format: str = "json") -> bytes` that serialises matching records to JSON or CSV. Add full-text `search_text` matching against rendered subject and body using simple substring search (upgrade path: PostgreSQL `tsvector` with `to_tsquery`).

**Competitor Reference**: Zendesk notification history; Intercom conversation search; SendGrid Activity Feed with export.
