# FREC - World Class Improvements

**Capability**: Facial Recognition (`frec`)
**Author**: Datacraft (nyimbi@gmail.com)
**Copyright**: © 2025 Datacraft

---

## 1. Adaptive Threshold Calibration

The current static `verification_threshold = 0.80` does not account for environmental conditions (lighting variance, image resolution, sensor type) or per-subject template age. Implement per-subject dynamic thresholds derived from historical match score distributions using Bayesian adaptive calibration. Thresholds should tighten when the subject's template is fresh/high-quality and relax slightly for aged templates — while never crossing the tenant-configured floor. This closes the performance gap between controlled enrollment conditions and uncontrolled operational conditions.

## 2. Morphing Attack Detection (MAD)

Face morphing (combining two identities into one template) is a documented attack vector in border control and access systems. Add a `morphing_attack_detect` method that applies frequency-domain analysis (FFT artifact detection on JPEG compression boundaries) and landmark asymmetry scoring to flag suspected morphs. This is distinct from PAD (presentation attack detection) and requires its own scoring pipeline.

## 3. Continuous Authentication Streaming

The current API is strictly request/response. High-security environments (attended workstations, secure rooms) need continuous ambient re-authentication: a subject is passively re-verified every N seconds using ambient camera frames, and access is revoked if confidence drops below a sustained threshold. Add an `async_continuous_auth_stream` method returning an `AsyncGenerator` of re-auth events so callers can subscribe to an ongoing identity signal.

## 4. Cross-Tenant Federated Identity Matching

Enterprises with multiple tenants (subsidiaries, partner organizations) need cross-tenant identity matching under explicit policy — e.g., a shared watchlist or a merged authentication domain. Add a `federated_identify` method that accepts a list of `(tenant_id, gallery_id)` pairs, fans out identification in parallel with per-tenant access checks, and merges ranked results. Each federated call must carry its own consent proof.

## 5. Template Aging and Re-enrollment Scheduling

Face templates degrade in accuracy over time as subjects age (facial hair, weight, scarring). The current `purge_expired` only hard-deletes by creation date; it does not identify templates whose match performance has silently drifted. Add a `template_aging_report` method that tracks match-score distribution over time per subject and flags templates whose rolling average confidence has dropped below a configurable drift threshold, scheduling re-enrollment proactively.

## 6. Privacy-Preserving Federated Learning for Model Updates

When FREC is deployed across multiple organizations, model improvement requires aggregating data that must never leave its origin. Implement a `federated_model_update_round` stub and integration contract that accepts encrypted gradient updates from edge nodes, aggregates using secure aggregation (sum without decryption), and applies the update to the local feature extractor — without any raw biometric data crossing tenant boundaries.

## 7. Demographic Bias Monitoring

Facial recognition systems have documented differential error rates across demographic groups. Add a `bias_audit_report` method that segments FAR/FRR metrics by demographic cohort (age band, gender expression, skin tone cluster) using ISO/IEC 19795-10 guidelines. This is not a politically optional feature — it is a regulatory requirement under the EU AI Act (GPAI obligations) and Kenya's Data Protection Act.

## 8. Deepfake and GAN-Generated Face Detection

The current `texture_analysis` and `presentation_attack_detect` methods use heuristic proxies (Laplacian variance, contrast). Deepfakes produced by modern diffusion and GAN models fool these proxies trivially. Add a `deepfake_detect` method that applies frequency-domain analysis (Fourier spectrum anomaly detection), temporal inconsistency scoring across video frames, and patch-level artifact detection — with a pluggable backend contract so production deployments can wire a dedicated model server (e.g., FaceForensics++-trained classifier).

## 9. Secure Enclave Template Storage with HSM Integration

Templates stored encrypted at rest using application-layer AES are only as secure as the key management. The current `FaceTemplateEncryption` class holds the key in process memory. Add an `HsmTemplateStore` adapter contract and `enroll_face_with_hsm` method that routes key operations to a PKCS#11-compatible HSM (or cloud KMS: AWS KMS, Azure Key Vault, GCP Cloud KMS). All template wrap/unwrap operations should be HSM-resident so the plaintext feature vector never exists in application memory.

## 10. Real-Time Watchlist Hit Streaming via Event Bus

The current watchlist matching happens synchronously inside identification requests. For high-volume surveillance use cases (airport gates, stadium entry), watchlist hits must propagate to downstream systems (SIEM, physical access control) in real time with sub-second latency. Add a `watchlist_hit_stream` async generator that publishes confirmed hits to a configurable event bus (Kafka, NATS, or APG's internal event bus via the `evtb` capability adapter contract).

## 11. Multi-Modal Fusion: Face + Voice + Iris

Single-modality biometrics have inherent FAR floors. Implement a `multimodal_verify` method that accepts face evidence, optional voice embedding, and optional iris hash, then fuses scores using a learned score-level fusion model (logistic regression trained on paired modality data). The fusion layer should degrade gracefully — if voice or iris data is absent, it falls back to face-only with an adjusted threshold and records the absent modalities in the audit event.

## 12. Explainability API for Verification Decisions

Under GDPR Article 22 and the EU AI Act, automated biometric decisions that produce legal or significant effects must be explainable. Add an `explain_verification` method that returns the top contributing factors to a verification outcome: which facial landmarks drove the match, what quality dimensions constrained confidence, whether liveness or occlusion was the binding constraint, and what the counterfactual threshold would need to be for the outcome to flip. Output should be human-readable and structured for downstream audit systems.

## 13. Incremental 1:N Search Using FAISS/HNSW Indexing

The current `identify_face` iterates linearly over all gallery subjects — O(N) per query. For galleries with thousands of subjects this becomes a latency bottleneck. Replace the linear scan with an approximate nearest-neighbor (ANN) index using FAISS HNSW or ScaNN. Add `build_gallery_index`, `update_gallery_index`, and `identify_face_indexed` methods. The index must support incremental updates (add/remove subjects without full rebuild) and must persist between service restarts via the database layer.

## 14. ISO/IEC 30107-3 Level 4 Liveness Compliance Report

The current liveness system is described as `level_4` in the constructor but the `liveness_score` aggregation uses fixed weights that do not implement the ISO/IEC 30107-3 presentation attack detection evaluation methodology. Implement a `liveness_compliance_report` method that evaluates a test dataset against the APCER (Attack Presentation Classification Error Rate) and BPCER (Bona fide Presentation Classification Error Rate) metrics as specified by ISO/IEC 30107-3, and certifies whether the current configuration meets the claimed Level 4 threshold.

## 15. Consent Portability and Cross-System Sync

Under GDPR Article 20 (data portability) and emerging global regulations, subjects have the right to receive their biometric consent records in a machine-readable format and to have those records transferred to other systems. Add `export_consent_portable` (returns a W3C Verifiable Credential-structured JSON-LD consent record) and `import_consent_portable` (ingests a VC-structured consent record from another system, validates its cryptographic signature, and activates it locally). This enables consent to follow the subject across organizational boundaries without requiring re-enrollment.
