# Video Conferencing (vidc) — World-Class Improvement Roadmap

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Adaptive Media Quality via SFU-Aware Simulcast Negotiation

Current service is transport-agnostic. Add a `negotiate_simulcast_layers` method that accepts per-participant bandwidth estimates and returns WebRTC SDP amendments (low/mid/high layer selection). This eliminates blunt-instrument quality drops under constrained networks and enables per-recipient adaptive bitrate without requiring a full SFU re-architecture.

## 2. End-to-End Encryption Key Distribution (E2EE Ratchet)

Integrate a per-meeting ratcheted key schedule (Double Ratchet or MLS-lite) so that recording, screen-share, and chat payloads are E2EE even from the SFU. The service layer issues key epoch records and rotates on participant join/leave events. Current `encrypted` flag on recordings is a boolean; replace with a structured `EncryptionManifest` that tracks key epoch, KDF algorithm, and rotation events.

## 3. AI-Powered Real-Time Transcription via Ollama (Local ASR)

`recording_transcript` currently returns an empty stub. Wire it to a locally-hosted Whisper model via the Ollama HTTP API (`/api/generate` with a custom ASR adapter). Return streaming token callbacks so the caller can incrementally render live captions. This eliminates cloud ASR dependency and keeps PII on-premises — critical for regulated tenants.

## 4. Persistent Store Adapter Interface

Replace the flat in-memory dicts with a `VidcStoreProtocol` (structural subtyping via `typing.Protocol`). Provide three concrete implementations: `MemoryVidcStore` (current), `PostgresVidcStore` (asyncpg), and `RedisVidcStore` (aioredis sorted sets for ordered audit events). Service constructor accepts any conforming store — zero service-layer changes needed to swap backends.

## 5. Composable Meeting Workflow DSL

Add a `MeetingWorkflow` dataclass that encodes a sequence of lifecycle steps (room_create → meeting_start → agent_register → recording_start → transcribe → end) as a declarative pipeline. `VidcService.run_workflow(workflow, tenant_id)` executes steps in order with per-step rollback hooks. This makes complex orchestrations reproducible, testable as a unit, and composable with other APG capabilities.

## 6. Granular RBAC with Attribute-Based Policy Evaluation

Current evaluation context is flat booleans. Extend `evaluate()` to accept an `actor_attributes` dict (department, clearance_level, device_trust_score) and evaluate ABAC rules. Enables: "only hosts with clearance_level >= 3 can start recordings involving external guests" — expressible without code changes.

## 7. Meeting Network Topology Optimisation (P2P ↔ SFU Switch)

Add `optimize_topology(tenant_id, meeting_id)` that reads current `participant_count` and decides whether the meeting should run peer-to-peer (≤3 participants) or route through an SFU. Emits a `topology_changed` audit event with the recommended media server target. Reduces infrastructure cost for small meetings by ~40% in practice.

## 8. Breakout Room Synchronisation Bus

Current `breakout_room_create` creates an isolated meeting. Add a `BreakoutBus` that propagates chat messages, reactions, and poll results between breakout rooms and the parent meeting in real-time via an in-process `asyncio.Queue` (swappable for NATS/Redis Streams in production). Enables facilitators to broadcast instructions to all breakout rooms simultaneously.

## 9. Meeting Noise & Engagement Metrics (Computer Vision Pipeline)

Add `compute_engagement_metrics(tenant_id, meeting_id, frame_batch)` that passes frame batches to a locally-served LLaVA model via Ollama. Returns engagement scores: speaker_activity ratio, camera-on ratio, reaction count. Stores results as time-series entries on `MeetingRecord`. Provides facilitators with objective engagement data without uploading video to cloud services.

## 10. Automated Post-Meeting Action Item Extraction

Add `extract_action_items(tenant_id, meeting_id, transcript_ref)` that calls a local Ollama model (`mistral` or `llama3`) with a structured extraction prompt. Returns a list of `ActionItem` records (assignee, description, due_date, confidence). Persists items as first-class entities — not just audit events — enabling downstream task-management capability integration.

## 11. Federated Multi-Tenant Meeting (Cross-Tenant Bridge)

Add `federate_meeting(host_tenant_id, guest_tenant_id, meeting_id, federated_room_ref)` that creates a cross-tenant participant record with tenant-isolation invariants enforced at the service layer. Enables partner organisations on separate APG tenants to join a single meeting while preserving per-tenant recording consent and retention policies.

## 12. Structured Meeting Minutes Generation

Add `generate_minutes(tenant_id, meeting_id, format)` that assembles a structured meeting-minutes document from captions, action items, poll results, and attendance records. Supports `markdown`, `docx`, and `pdf` output formats. Delegates to a locally-hosted model for narrative coherence while keeping all source data in-store. Returns a `MinutesRecord` with a download URL.

## 13. Webhook / Event Subscription System

Add `subscribe_events(tenant_id, webhook_url, event_types)` and `unsubscribe_events(subscription_id)`. On every `_record_event` call, dispatch matching subscriptions via `aiohttp` POST with HMAC-signed payloads. Enables external systems (CRM, JIRA, Slack) to react to meeting lifecycle events without polling the audit log.

## 14. Immutable Audit Trail with Merkle-Chain Integrity

Extend `MeetingAuditEventRecord` with a `prev_hash` field. Each new event hashes `(id + event_type + subject_id + actor + prev_hash)` — forming a tamper-evident chain per tenant. Add `verify_audit_chain(tenant_id)` that walks the chain and returns a verification report. Regulators can independently verify that audit records have not been altered.

## 15. Meeting Cost Attribution and Chargeback Model

Add `compute_meeting_cost(tenant_id, meeting_id, cost_model_ref)` that calculates infrastructure cost attribution based on participant-minutes, recording storage, and ASR compute. Returns a `MeetingCostRecord` with per-department breakdowns. Plugs into APG's financial capabilities (`finc`, `budg`) for automated chargeback reporting — closing the loop between operational telemetry and financial accountability.
