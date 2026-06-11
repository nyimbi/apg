# Chat Capability — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## 1. Semantic RAG Message Retrieval

**Category**: Intelligence / AI Integration

**Justification**: Keyword search (current `search_messages`) suffers from vocabulary mismatch. Users ask "what did we decide about the deployment?" and miss messages containing "we'll ship on Friday." Semantic search using vector embeddings closes this gap — the single largest quality-of-life delta in modern knowledge-worker chat.

**Implementation**: On `send_message`, emit each message body to an async background task that calls a locally-hosted Ollama embedding model (e.g. `nomic-embed-text`). Store the float32 vector alongside the message key in an in-process `numpy` array or `pgvector` column. On `search_messages`, when a `semantic=True` flag is set, embed the query and return cosine-ranked results instead of term-overlap results. Fall back to lexical when the embedding service is unavailable.

**Competitor Reference**: Slack AI (Slack, 2024), Microsoft Copilot in Teams — both use RAG over conversation history to surface answers, not just messages.

---

## 2. Conversation Summarisation via Local LLM

**Category**: Intelligence / Productivity

**Justification**: Long-running channels accumulate context that new members or returning participants cannot efficiently absorb. A `summarise_conversation` method that calls an Ollama-served model (e.g. `mistral`, `llama3`) to produce a bullet-point digest reduces catch-up time from 20 minutes to 30 seconds.

**Implementation**: Collect the `N` most recent non-deleted messages in a room, render them as a transcript string, and POST to `http://OLLAMA_BASE_URL/api/generate` with a summarisation system prompt. Stream the response and return both the full summary and a list of key decision points. Cache the result with a TTL keyed by `(tenant_id, room_id, last_message_id)` to avoid redundant model calls.

**Competitor Reference**: Slack AI channel summaries, Google Meet / Chat AI summaries (Google Workspace, 2024).

---

## 3. Structured Tool-Calling Agent Dispatch

**Category**: Intelligence / Automation

**Justification**: Ad-hoc bot commands (e.g. `/weather london`) are brittle string parsing. Structured tool-calling — where the LLM receives a JSON schema of registered bot capabilities and emits a structured `tool_call` object — produces reliable, type-safe agent dispatch that composes correctly with the existing `bot_registration` surface.

**Implementation**: Add `agent_tool_call` method that takes a `user_message`, fetches the tool schemas for bots registered in the room, calls Ollama in function-calling mode (Mistral-Nemo, Qwen2.5 support this), parses the structured response, dispatches to the correct registered bot handler, and injects the tool result back into the conversation as a system message.

**Competitor Reference**: OpenAI GPT-4o function calling in ChatGPT, Anthropic Claude tool use in Slack integration.

---

## 4. Per-Message Token Cost Accounting

**Category**: FinTech / Cost Governance

**Justification**: Multi-tenant SaaS must attribute LLM inference costs to tenants and optionally bill downstream. Without token-level accounting, the operator cannot enforce budgets or produce accurate invoices. This matters most in AI-heavy rooms where agent responses dominate traffic.

**Implementation**: On every LLM call initiated from chat (summarisation, semantic search, tool dispatch), capture `prompt_tokens`, `completion_tokens`, and `total_tokens` from the Ollama response. Multiply by configurable `Decimal` rates stored per tenant. Accumulate in a `_token_ledger: dict[str, Decimal]` keyed by `(tenant_id, agent_id, date)`. Expose via `token_usage_report` and `token_cost_summary` methods. Use `Decimal` throughout; never `float` for monetary values.

**Competitor Reference**: Azure OpenAI Service token metering, AWS Bedrock per-token billing dashboards.

---

## 5. Real-Time Sentiment and Toxicity Guardrails

**Category**: Safety / Moderation

**Justification**: Manual moderation queues are reactive — harm has already landed. Embedding a fast local classifier (e.g. a fine-tuned `DistilBERT` served via Ollama or a direct HuggingFace call) on the `send_message` hot path allows the service to intercept or flag toxic, harassing, or sentiment-negative messages before delivery, not after.

**Implementation**: Add `async_toxicity_screen` as a pre-commit hook inside `send_message`. If the score exceeds a configurable threshold (`toxicity_threshold` in the capability contract), route to the moderation queue with `auto_flagged=True` rather than delivering. Expose aggregate sentiment trends per room in `room_analytics`. Degrade gracefully when the classifier is unavailable.

**Competitor Reference**: Discord AutoMod ML classifier, Slack's harmful content detection (2023).

---

## 6. Retention Policy Enforcement Engine

**Category**: Compliance / Data Governance

**Justification**: Retention policies exist in the data model but are never enforced. In regulated industries (finance, healthcare, government) this creates legal exposure. An enforcement engine that actually expires messages based on policy is the difference between governance theatre and governance reality.

**Implementation**: Add `enforce_retention_policy(tenant_id, room_id)` that reads `retention_policy` from the `ChatRoom`, parses it into a `timedelta` (e.g. `retain-90-days` → 90 days), and soft-deletes messages older than the cutoff. Record an audit event per purge batch. Expose `retention_compliance_report` to show which rooms are within policy and which have overdue purges. Designed to be called by a scheduled job via the `lifecycle_batch` surface.

**Competitor Reference**: Microsoft Teams retention policies (Purview compliance), Slack Enterprise Grid message retention.

---

## 7. End-to-End Encrypted Direct Messages

**Category**: Security / Privacy

**Justification**: Direct messages in regulated B2B environments require confidentiality guarantees beyond TLS. Signal Protocol and Matrix MLS are the reference designs; even a simplified envelope-encryption scheme (per-conversation symmetric key wrapped by each recipient's public key) raises the bar meaningfully against server-side data breaches.

**Implementation**: Add `create_e2e_dm_session(tenant_id, from_user, to_user, from_public_key, to_public_key)` that generates a random AES-256 session key, wraps it once per recipient using their RSA/EC public key, and stores the wrapped keys. `send_e2e_direct_message` accepts a ciphertext blob (encrypted client-side) and stores it opaquely, never touching plaintext. `decrypt_e2e_dm` (client-side helper, not server-side) reconstructs the session key to decrypt locally.

**Competitor Reference**: Signal, WhatsApp Business, Apple iMessage Advanced Data Protection.

---

## 8. Adaptive Rate Limiting with Tenant Quotas

**Category**: Reliability / Multi-Tenancy

**Justification**: A noisy tenant — or a misconfigured bot flooding messages — can saturate the service for all other tenants. Without rate limiting, the shared-state `_messages` dict grows unboundedly and latency degrades for the whole fleet. Token-bucket rate limiting per `(tenant_id, user_id)` bounds worst-case abuse.

**Implementation**: Add a `_rate_limiter: dict[str, tuple[int, str]]` mapping `(tenant_id:user_id)` to `(tokens_remaining, last_refill_timestamp)`. `check_rate_limit(tenant_id, user_id, cost=1)` implements a token-bucket: refill at a configurable rate per tenant, deduct `cost` on each call, raise `RateLimitExceeded` when depleted. Wire into `send_message`. Expose `rate_limit_status(tenant_id, user_id)` for dashboards.

**Competitor Reference**: Slack API rate limits (Tier 1–4), Discord per-channel message rate limiting.

---

## 9. Multi-Modal Attachment Intelligence

**Category**: Intelligence / Accessibility

**Justification**: Attachments are currently opaque `storage_ref` strings. Vision-language models (e.g. `llava`, `bakllava` via Ollama) can describe images, extract text from screenshots, and summarise PDFs. This converts attachments from inert blobs into searchable, accessible content.

**Implementation**: Add `analyse_attachment(tenant_id, message_id, storage_ref, mime_type)` that routes to the appropriate Ollama model based on MIME type: image/* → `llava`, application/pdf → `nomic-embed-text` + text extraction, video/* → frame-sampled `llava`. Store the analysis result as structured metadata on the `ChatMessage`. Expose `attachment_analysis` in `search_messages` so queries match image content.

**Competitor Reference**: Microsoft Teams Copilot image analysis, Slack AI file summaries.

---

## 10. Conversation Graph and Knowledge Extraction

**Category**: Intelligence / Knowledge Management

**Justification**: Chat rooms accumulate institutional knowledge — decisions, commitments, action items — but it is buried in message history. Extracting a structured knowledge graph (entities, relationships, decisions) transforms the chat capability into an organisational memory system. This is the core differentiator of products like Notion AI and Confluence AI.

**Implementation**: Add `extract_knowledge_graph(tenant_id, room_id, lookback_hours=24)` that collects recent messages, calls an Ollama LLM with a structured-extraction prompt to identify named entities (people, projects, dates), relationships (assigned-to, depends-on), and decision points. Return a JSON-LD graph. Cache and incrementally update on new messages. Expose via `knowledge_graph_query(tenant_id, room_id, entity_type)`.

**Competitor Reference**: Notion AI knowledge base, Confluence AI, Mem.ai.

---

## 11. Workspace-Wide Cross-Room Search with Faceting

**Category**: Discovery / Navigation

**Justification**: `search_messages` is scoped to a single optional room. Enterprise users need cross-room search with facets: date range, sender, room, attachment presence, sentiment, thread context. This is standard in Slack, Teams, and every enterprise search product.

**Implementation**: Add `workspace_search(tenant_id, query, filters: dict)` that supports filter keys: `room_id`, `sender`, `after_date`, `before_date`, `has_attachment`, `moderation_status`, `thread_only`. Combine lexical scoring with a configurable `min_score` threshold. Return paginated results with a `facets` block showing hit counts per dimension. Wire semantic search as an opt-in flag.

**Competitor Reference**: Slack Enterprise Search, Microsoft Teams global search with filters, Elasticsearch faceted search.

---

## 12. Composable Notification Routing

**Category**: Reliability / Composability

**Justification**: Mention notifications (`mention_notification`) record a dict but never deliver it. Real-world notification routing must fan out across channels (email, push, SMS, webhook) with per-user preference, per-tenant routing rules, and delivery confirmation — none of which exist today.

**Implementation**: Add `route_notification(tenant_id, notification_type, recipient, payload, channels)` that calls the `ntfy` adapter (APG adapter boundary). Add per-user `notification_preferences` storage (`_notification_prefs`) with `get_notification_preferences` and `set_notification_preferences`. `mention_notification` and `direct_message` call `route_notification` automatically. Use a queue (`_notification_queue`) to decouple delivery from message send latency.

**Competitor Reference**: Slack notification preferences (per-channel, per-keyword, DND schedule), Teams notification management.

---

## 13. Federated Room Guest Access with Expiry

**Category**: Compliance / Multi-Tenancy

**Justification**: External guests exist in the data model but access control is a boolean flag. Production federated guest access requires: time-boxed tokens, explicit permission sets (read-only vs. send), audit of guest activity, and automatic revocation at expiry. Without this, the `external_guests` field is governance-unsafe.

**Implementation**: Add `grant_guest_access(tenant_id, room_id, guest_email, granted_by, expiry_hours, permissions)` that creates a `GuestAccessGrant` record with a cryptographic token, expiry timestamp, and explicit permission list. `verify_guest_token(token)` validates and returns the grant. `revoke_guest_access` and `expire_guest_access_pass(tenant_id)` (for batch enforcement) close the loop. All actions produce audit events.

**Competitor Reference**: Slack Connect (federated channel guest access), Microsoft Teams external access policies.

---

## 14. Intelligent Thread Auto-Routing

**Category**: Intelligence / UX

**Justification**: High-volume rooms become chaotic without automatic thread creation. When a new message is semantically similar to a recent message (cosine similarity > threshold), auto-creating a thread and linking them prevents topic fragmentation. LinkedIn, Discord, and Twist all default to threaded responses.

**Implementation**: Add `suggest_thread_parent(tenant_id, room_id, body)` that embeds the candidate message, computes cosine similarity against the 50 most recent messages' embeddings, and returns the top candidate `parent_message_id` if similarity exceeds `thread_similarity_threshold` (default 0.82). Call this inside `send_message` when `auto_thread=True`. Record the suggestion and whether it was accepted as a learning signal.

**Competitor Reference**: Twist (thread-first chat), Discord forum channels, Slack's thread recommendations.

---

## 15. LLM-Powered Intent Classification for Agent Dispatch

**Category**: Intelligence / Automation

**Justification**: Routing a user message to the correct handler — whether a bot command, an agent workflow, or a human — currently requires explicit `/command` syntax. LLM-based intent classification removes this friction: users speak naturally and the system routes automatically with explainable confidence scores.

**Implementation**: `classify_message_intent(tenant_id, message_id)` calls a local Ollama model (e.g. `phi3`, `mistral`) with a classification prompt listing registered agent capabilities and bot commands as candidate intents. Returns `{intent, confidence, handler_id, rationale}`. Wire into `send_message` when `ai_agent_participant=True` to automatically dispatch high-confidence intents (`> 0.85`) to the matched agent. Log all classifications for feedback loop training.

**Competitor Reference**: Google Dialogflow CX intent classification, Amazon Lex, Rasa NLU.
