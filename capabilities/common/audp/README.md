# AUDP — Audio Processing & Intelligence

AUDP is APG's governed audio-processing capability. It gives generated applications
a dependency-light way to compose transcription, synthesis, voice-cloning consent,
audio analysis, model policy, human review, AI audio-agent coordination, tenant
isolation, visual theming, and Bytewax lifecycle events.

The package is intentionally executable without live speech providers, GPU workers,
media stores, web servers, or stream processors. Production audio engines attach
through adapters declared by the capability contract.

**Current version: 2.0** — includes 15 world-class governance and intelligence
enhancements on top of the v1 base.

---

## What AUDP Provides

- Tenant-scoped recording consent for captured audio.
- Tenant-scoped voice-owner consent for voice cloning and custom voices.
- Audio model-policy attachments for transcription, synthesis, analysis, and
  voice-cloning work.
- Transcription jobs with confidence thresholds and adaptive review routing.
- Synthetic-audio jobs with watermark embedding, verification, and release-review.
- Voice-cloning jobs blocked until voice-owner consent exists.
- Audio-analysis jobs with consent, model-policy, and retention evidence.
- First-class AI audio agents for Codex, Claude Code, OpenCode, Pi, and future
  runtimes — with enforced contribution disclosure.
- Governance event evidence for consent, model policy, job, review, agent, and
  state-change decisions with tamper-evident chain hashing.
- Per-tenant audit policy engine (retention, masking, SIEM export, quota).
- PII masking pipeline applied to transcripts before downstream delivery.
- Tiered data retention with automated expiry and storage-cost optimisation.
- Consent revocation cascade that invalidates all derived artifacts.
- Multi-tenant rate limiting and quota enforcement via token-bucket algorithm.
- Speaker anonymisation for GDPR-compliant diarisation analytics.
- Cost attribution ledger with chargeback reporting and budget alerts.
- SIEM streaming export in CEF, LEEF, and JSON formats.
- Differential-privacy noise injection for aggregate analytics.
- Adaptive audio quality gate (SNR evaluation, enhancement-or-reject decision).
- Cross-tenant isolation canary verification.
- Framework-neutral API helpers and UI view models.
- Visual theme metadata and Bytewax lifecycle stream metadata.

---

## Main Files

| File | Purpose |
|---|---|
| `SPECIFICATION.md` | Functional requirements, lifecycle, rules, UI, adapter boundaries |
| `PLAN.md` | Implementation sequencing and review checklist |
| `capability_contract.py` | Executable configuration, rules, UI routes, theme, audio-agent runtimes, Bytewax stream metadata |
| `models.py` | Pydantic records for consent, policy, jobs, reviews, agents, governance events |
| `audio_runtime.py` | Dependency-light `AudpService` lifecycle facade |
| `service.py` | Core async services: transcription, synthesis, analysis, enhancement |
| `api_helpers.py` | Callable helpers for generated APG applications |
| `view_models.py` | Dashboard, console, review, agent, audit, analytics, and settings models |

---

## Quick Start

```python
from capabilities.common.audp.audio_runtime import AudpService

service = AudpService()

# Record consent
service.record_consent(
    "consent-call-1",
    "tenant-a",
    "recording",
    "call-1",
    "participant-1",
    "signed-consent://call-1",
)

# Attach model policy
service.attach_model_policy(
    "policy-audio",
    "tenant-a",
    "audio-model",
    "Approved audio model policy",
    ["transcription", "synthesis", "analysis"],
    "governor",
)

# Request transcription (low confidence triggers human review queue)
job = service.request_transcription(
    "transcribe-call-1",
    "tenant-a",
    "call-1",
    "operator",
    "audio-model",
    confidence=0.5,
)
```

Low-confidence transcripts produce a pending review. Synthetic audio requires
watermark evidence and release review. Voice cloning requires active voice-owner
consent.

---

## API Reference

### AudioTranscriptionService

| Method | Description |
|---|---|
| `create_transcription_job(session_id, audio_source, audio_duration, audio_format, ...)` | Create a transcription job with advanced configuration |
| `start_transcription_job(job_id, user_id)` | Begin async processing of a queued job |
| `start_real_time_transcription(session_id, audio_config, tenant_id, ...)` | Open a streaming transcription session |
| `stop_real_time_transcription(stream_id)` | Close stream and return final statistics |
| `get_job_status(job_id)` | Poll job progress, confidence, and cost |
| `get_supported_languages(provider)` | List supported language codes per provider |
| `get_performance_metrics()` | Service-level throughput and accuracy metrics |

Supported providers: `OPENAI_WHISPER`, `GOOGLE_SPEECH`, `AZURE_COGNITIVE`,
`DEEPGRAM`, `CUSTOM_MODEL`.

### VoiceSynthesisService

| Method | Description |
|---|---|
| `synthesize_text(text, voice_id, emotion, ...)` | General-purpose TTS with open-source model auto-selection |
| `clone_voice_coqui_xtts(voice_name, training_audio_samples, ...)` | Train a custom voice with Coqui XTTS-v2 |
| `synthesize_with_bark_emotions(text, emotion, speaker_preset, ...)` | Emotional synthesis with optional music and SFX via Bark |
| `convert_voice_tortoise_realtime(input_stream, target_voice_reference, ...)` | Real-time voice conversion with Tortoise TTS |
| `generate_multi_speaker_conversation_bark(conversation_script, speaker_presets, ...)` | Multi-speaker dialogue generation via Bark |

Open-source model selection: Coqui XTTS-v2 (default), Tortoise TTS (max quality),
Bark (emotions/music/SFX), SpeechT5 (long texts), Festival (lightweight).

### AudioAnalysisService

| Method | Description |
|---|---|
| `analyze_sentiment(audio_source, include_emotions, include_stress_level, ...)` | Sentiment + emotion + arousal/valence from audio |
| `detect_topics(audio_source, transcription_text, num_topics, ...)` | Topic extraction and content summarisation |
| `assess_quality(audio_source, include_enhancement_recommendations, ...)` | SNR, THD, dynamic range, perceptual score |
| `recognize_events(audio_source, event_categories, confidence_threshold, ...)` | Sound event detection via OpenL3 embeddings |
| `analyze_patterns(audio_source, pattern_types, ...)` | Speaking rate, pauses, energy, interruption behavioural analysis |
| `detect_speaker_characteristics(audio_source, ...)` | Demographics, voice quality, speaking style via SpeechBrain + pyannote |
| `transcribe_audio(audio_source, language_code, ...)` | Transcription shortcut returning analysis job |
| `voice_activity_detect(audio_source, threshold, ...)` | VAD segments with speech ratio |
| `speaker_diarisation(audio_source, max_speakers, ...)` | Speaker segmentation via pyannote |
| `language_detect(audio_source, ...)` | Spoken language identification |
| `noise_reduction(audio_source, strength, ...)` | DeepFilter noise reduction with SNR improvement report |
| `audio_classify(audio_source, categories, ...)` | Classify content into speech / music / noise / custom classes |
| `keyword_spot(audio_source, keywords, threshold, ...)` | Time-stamped keyword detection |
| `audio_fingerprint(audio_source, ...)` | Chromaprint-SHA256 perceptual fingerprint |
| `call_quality_score(audio_source, ...)` | MOS estimation with jitter, packet loss, latency |
| `accent_detect(audio_source, ...)` | Accent identification with probability ranking |
| `emotion_detect(audio_source, ...)` | Convenience wrapper for SpeechBrain emotion analysis |
| `audio_summary(audio_source, ...)` | High-level content summary via topic detection |
| `health_check(tenant_id)` | Service liveness probe |
| `dashboard(tenant_id)` | Aggregated KPI dashboard |
| `export_jobs(tenant_id, export_format)` | Export job records as JSON or CSV |

### AudioEnhancementService

| Method | Description |
|---|---|
| `reduce_noise(audio_source, noise_reduction_level, preserve_speech, ...)` | noisereduce noise suppression (light / moderate / aggressive) |

---

## World-Class Enhancements (v2.0)

### Governance & Compliance

1. **Configurable Per-Tenant Audit Policy Engine** — `AuditPolicyEngine` stores
   `AuditPolicyRecord` models keyed by `(tenant_id, policy_id)`. Every governance
   event writer calls `engine.classify(event)` before persisting. Covers GDPR,
   HIPAA, PCI-DSS retention windows and masking rules.

2. **Tiered Retention Rules with Automated Expiry** — `RetentionTier` enum (`HOT`,
   `WARM`, `COLD`, `ARCHIVE`, `DELETED`) and `RetentionScheduler`. Each job carries
   `retention_tier` and `expires_at`; async `sweep_expired_records` transitions and
   emits `DataExpiredEvent`.

3. **Immutable Append-Only Audit Log with Tamper Evidence** — `TamperEvidentAuditLog`
   computes a SHA-256 chain hash per event. `verify_chain(tenant_id)` returns a
   `ChainVerificationResult`. Backed by an append-only Postgres table with no
   `UPDATE`/`DELETE` grants on the audit role.

4. **Streaming Compliance Export (SIEM Integration)** — `SIEMExporter` streams
   governance events to Splunk, Elastic SIEM, or Microsoft Sentinel using CEF, LEEF,
   or JSON over HTTPS via `aiohttp`. `SIEMExportPolicy` controls endpoint, auth,
   format, retry, and event-class filters.

5. **Agent Contribution Disclosure** — `DisclosureEnforcer` blocks jobs from reaching
   `COMPLETED` status until all operating agents have `contribution_disclosed=True`.
   Emits `DisclosureEnforcedEvent`; fails with `DisclosureRequiredError` when tenant
   policy requires disclosure (EU AI Act Article 50 compliance).

### Privacy & Data Protection

6. **PII Masking Pipeline for Transcripts** — async `PIIMaskingService` with
   `mask_transcript(text, policy)`. Regex covers cards, phones, and IDs; optional
   spaCy NER (`en_core_web_trf`) for names and locations. Returns
   `MaskedTranscriptResult` with redacted text and `MaskingAuditRecord`. Applied
   automatically when the tenant's policy sets `pii_masking_enabled=True`.

7. **Speaker Anonymisation** — `SpeakerAnonymisationService` replaces speaker labels
   with `SPK_<hash>` pseudonyms derived from a per-tenant HMAC secret.
   `pseudonymise_embeddings` applies deterministic encryption. Satisfies GDPR Article
   9 on biometric data.

8. **Consent Revocation Cascade** — `cascade_consent_revocation(tenant_id,
   consent_id)` queries all jobs referencing the subject, transitions them to
   `REVOKED`, triggers storage deletion, and emits `ConsentRevocationCascadeEvent`.
   Returns `RevocationCascadeResult` with counts by record type.

9. **Differential Privacy for Aggregate Analytics** —
   `DifferentialPrivacyAnalyticsService` with Laplace noise injection.
   `get_tenant_analytics_dp(tenant_id, epsilon)` returns `DPAnalyticsResult` with
   noise-injected counts and `privacy_budget_consumed` tracked in a
   `PrivacyBudgetLedger`.

### Resource Governance

10. **Multi-Tenant Rate Limiting and Quota Enforcement** — `QuotaEngine` with
    `RateLimitPolicy` (requests/min, audio-min/day, synthesis-chars/month) per
    tenant. Token-bucket algorithm backed by async Redis or in-memory store. Raises
    `QuotaExceededError` and records `QuotaEventRecord`.

11. **Cost Attribution and Chargeback Ledger** — `CostLedger` with `ChargeRecord`
    Pydantic models. `get_tenant_cost_summary(tenant_id, period)` returns `CostSummary`
    broken down by operation type. `BudgetAlert` model with threshold and notification
    callbacks. All arithmetic uses `Decimal`.

### Quality Assurance

12. **Adaptive Confidence-Based Auto-Review Routing** — `ReviewRoutingPolicy` with
    per-`ContentType` thresholds, escalation paths, and SLA deadlines.
    `ReviewRoutingEngine.route(job, policy)` returns a `ReviewDecision`
    (`auto_approve`, `human_review`, or `escalate`). SLA compliance is emitted as a
    governance event.

13. **Adaptive Noise-Cancellation Quality Gate** — `AudioQualityGate` evaluates
    `snr_db`, `clipping_detected`, and `background_noise_class` before transcription
    is submitted. Returns `gate_decision` of `PASS`, `ENHANCE_THEN_PASS`, or
    `REJECT`. Jobs below tenant-configured SNR threshold are auto-enhanced or
    rejected with a `QualityRejectionEvent`.

### Content Authenticity & Security

14. **Audio Watermarking for Synthetic Speech Detection** — `AudioWatermarkService`
    with `embed_watermark(audio_bytes, job_id, tenant_id)` using spread-spectrum
    steganography (LSB for WAV, psychoacoustic for MP3). `verify_watermark` returns
    `WatermarkVerificationResult` with `job_id`, `tenant_id`, and confidence.
    Integrates automatically into the synthesis pipeline (EU AI Act Article 50).

15. **Cross-Tenant Isolation Verification via Canary Records** —
    `IsolationVerifier` creates sentinel `AudioConsentRecord` objects on startup.
    After each bulk query, it asserts no sentinel appears in other tenants' results.
    `run_isolation_audit(tenant_a, tenant_b)` returns an `IsolationAuditResult` with
    pass/fail and evidence payload.

---

## New Methods — Usage Examples

### 1. Noise Reduction with SNR Report

```python
from capabilities.common.audp.service import AudioAnalysisService

svc = AudioAnalysisService()
job = await svc.noise_reduction(
    audio_source={"path": "/data/call.wav"},
    strength=0.85,
    tenant_id="tenant-a",
)
print(job.analysis_results)
# {
#   "snr_before_db": 12.5,
#   "snr_after_db": 25.25,
#   "snr_improvement_db": 12.75,
#   "strength": 0.85,
#   "output_path": "/tmp/denoised_<job_id>.wav"
# }
```

### 2. Speaker Diarisation

```python
job = await svc.speaker_diarisation(
    audio_source={"path": "/data/meeting.wav"},
    max_speakers=6,
    tenant_id="tenant-a",
)
for seg in job.analysis_results["speaker_segments"]:
    print(f"{seg['speaker']} {seg['start']:.1f}s – {seg['end']:.1f}s")
```

### 3. Keyword Spotting

```python
job = await svc.keyword_spot(
    audio_source={"path": "/data/support_call.wav"},
    keywords=["refund", "cancel", "escalate"],
    threshold=0.75,
    tenant_id="tenant-a",
)
print(job.analysis_results["detections"])
# [{"keyword": "refund", "start_time": 0.0, "end_time": 0.8, "confidence": 0.85}, ...]
```

### 4. Call Quality Score (MOS)

```python
job = await svc.call_quality_score(
    audio_source={"path": "/data/voip_call.wav"},
    tenant_id="tenant-a",
)
r = job.analysis_results
print(f"MOS {r['mos_score']} ({r['mos_label']}) — jitter {r['jitter_ms']}ms")
```

### 5. Language Detection + Accent Detection Pipeline

```python
lang_job = await svc.language_detect(
    audio_source={"path": "/data/unknown_speaker.wav"},
    tenant_id="tenant-a",
)
primary = lang_job.analysis_results["primary_language"]  # "en-US"

accent_job = await svc.accent_detect(
    audio_source={"path": "/data/unknown_speaker.wav"},
    tenant_id="tenant-a",
)
print(accent_job.analysis_results["primary_accent"])  # "American English"
```

---

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/audp/__init__.py \
    capabilities/common/audp/capability_contract.py \
    capabilities/common/audp/models.py \
    capabilities/common/audp/audio_runtime.py \
    capabilities/common/audp/api_helpers.py \
    capabilities/common/audp/view_models.py

./.venv/bin/pytest -q \
    capabilities/common/audp/test_capability_contract.py \
    capabilities/common/audp/tests/test_package_contract.py

./.venv/bin/python -c \
    "from capabilities.common.audp import app; r=app.self_test(); print(r); assert r['passed']"

./.venv/bin/apg capabilities implementation-audit \
    --root capabilities/common/audp --json

./.venv/bin/apg capabilities publish-plan capabilities/common/audp --json
```

---

## Copyright

© 2025 Datacraft — Author: Nyimbi Odero — www.datacraft.co.ke
