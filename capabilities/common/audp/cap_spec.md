# AUDP Capability Specification Pointer

The current executable AUDP capability packet is defined by:

- `README.md` for usage and composition guidance;
- `SPECIFICATION.md` for functional requirements and guardrails;
- `PLAN.md` for implementation sequencing and review criteria;
- `capability_contract.py` for executable configuration, rules, UI, theme, and
  Bytewax streaming metadata;
- `audio_runtime.py`, `api_helpers.py`, and `view_models.py` for the
  dependency-light lifecycle used by generated APG applications.

Legacy provider-specific speech, synthesis, and media-processing ambitions are
adapter work. The local package proves governed consent, model policy,
transcription review, synthesis review, voice-cloning consent, audio agents,
tenant isolation, and audit evidence without requiring live audio providers.
