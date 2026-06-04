"""APG Audio Processing capability.

Standalone package: ``pip install apg-common-audp``

Quick start::

    from apg_common_audp import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : audp
Provides      : audio_transcription, voice_synthesis, audio_analysis, speaker_diarization, audio_enhancement, audio_consent_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-audp"
__capability_id__ = "audp"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]
