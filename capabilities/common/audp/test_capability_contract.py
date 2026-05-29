"""Regression coverage for the AUDP executable capability contract."""

import pytest

from capabilities.common.audp import api_helpers, view_models
from capabilities.common.audp import register_capability
from capabilities.common.audp.audio_runtime import AudpService
from capabilities.common.audp.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-audio", {"transcription": {"minimum_confidence": 0.85}})

	assert contract["capability"] == "audp"
	assert contract["configuration"]["tenant_id"] == "tenant-audio"
	assert contract["configuration"]["transcription"]["minimum_confidence"] == 0.85
	assert contract["configuration_schema"]["required"] == ["tenant_id", "transcription", "synthesis", "analysis", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "transcription", "synthesis", "analysis", "sessions", "models", "consents", "reviews", "quality", "settings"}
	assert contract["ui"]["api_prefix"] == "/audp/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "waveform_viewer" in contract["theme"]["components"]
	assert "consent_banner" in contract["theme"]["components"]


def test_rule_engine_enforces_audio_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_recording",
		"recording_consent_recorded": False,
		"voice_owner_consent_recorded": False,
		"synthetic_audio_requested": True,
		"watermark_applied": False,
		"model_invocation": True,
		"model_policy_attached": False,
		"transcription_confidence": 0.4,
		"human_review_recorded": False,
		"synthetic_release_reviewed": False,
	})
	clone_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "clone_voice", "voice_owner_consent_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "recording_consent_required", "synthetic_audio_requires_watermark", "synthetic_audio_requires_release_review", "audio_model_requires_policy", "low_transcription_confidence_requires_review"}
	assert clone_result["matched_rules"] == ["voice_cloning_requires_consent"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "audp"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "audp_audio_intelligence"
	assert registration["ui_components"]["transcription"] == "/audp/transcription"
	assert "aicr" in registration["dependencies"]
	assert "audp:transcribe" in registration["permissions"]
	assert "audio_consent_governance" in registration["capabilities"]


def test_service_runs_audio_governance_lifecycle():
	service = AudpService()
	consent = service.record_consent(
		consent_id="consent-recording",
		tenant_id="tenant-audio",
		consent_type="recording",
		subject_id="call-001",
		granted_by="participant-001",
		evidence="signed-consent://call-001",
	)
	voice_consent = service.record_consent(
		consent_id="consent-voice",
		tenant_id="tenant-audio",
		consent_type="voice_owner",
		subject_id="speaker-001",
		granted_by="speaker-001",
		evidence="signed-consent://speaker-001",
	)
	policy = service.attach_model_policy(
		policy_id="policy-audio",
		tenant_id="tenant-audio",
		model_id="audio-model",
		policy_name="Approved audio model policy",
		allowed_operations=["transcription", "synthesis", "analysis", "voice_cloning"],
		attached_by="model-governor",
	)
	transcription = service.request_transcription(
		job_id="transcribe-001",
		tenant_id="tenant-audio",
		audio_source_id=consent["subject_id"],
		requested_by="agent-supervisor",
		model_id=policy["model_id"],
		confidence=0.52,
		result={"transcript": "Low confidence transcript"},
	)
	review = service.decide_transcript_review(
		review_id=transcription["transcript_review"]["id"],
		tenant_id="tenant-audio",
		reviewer="qa-reviewer",
		decision="approved",
		notes="Transcript corrected and approved.",
	)
	synthesis = service.request_synthesis(
		job_id="synth-001",
		tenant_id="tenant-audio",
		text="Your request has been processed.",
		requested_by="agent-supervisor",
		model_id=policy["model_id"],
		watermark_applied=True,
	)
	synthesis_review = service.decide_synthesis_review(
		review_id=synthesis["synthesis_review"]["id"],
		tenant_id="tenant-audio",
		reviewer="release-reviewer",
		decision="approved",
		notes="Synthetic audio watermark and copy approved.",
	)
	clone = service.request_voice_clone(
		job_id="clone-001",
		tenant_id="tenant-audio",
		voice_owner_id=voice_consent["subject_id"],
		requested_by="agent-supervisor",
		model_id=policy["model_id"],
	)
	analysis = service.request_analysis(
		job_id="analysis-001",
		tenant_id="tenant-audio",
		audio_source_id=consent["subject_id"],
		requested_by="agent-supervisor",
		model_id=policy["model_id"],
		analysis_types=["sentiment", "quality"],
	)
	dashboard = view_models.dashboard_model(service, "tenant-audio")

	assert transcription["job"]["status"] == "pending_review"
	assert review["decision"] == "approved"
	assert {job["id"]: job["status"] for job in service.list_jobs("tenant-audio")}["transcribe-001"] == "completed"
	assert synthesis["job"]["watermark_applied"] is True
	assert synthesis["job"]["status"] == "pending_review"
	assert synthesis_review["decision"] == "approved"
	assert clone["job_type"] == "voice_cloning"
	assert analysis["job_type"] == "analysis"
	assert dashboard["summary"]["job_count"] == 4
	assert dashboard["summary"]["pending_review_count"] == 0
	assert {event["event_type"] for event in dashboard["governance_events"]} >= {
		"audio_consent_recorded",
		"audio_model_policy_attached",
		"transcription_requested",
		"transcript_review_requested",
		"transcript_review_decided",
		"synthesis_requested",
		"synthesis_review_requested",
		"synthesis_review_decided",
		"voice_clone_requested",
		"analysis_requested",
	}


def test_service_blocks_audio_guardrail_violations():
	service = AudpService()

	with pytest.raises(PermissionError, match="recording_consent_required"):
		service.request_transcription(
			job_id="no-consent",
			tenant_id="tenant-audio",
			audio_source_id="call-001",
			requested_by="agent-supervisor",
			model_id="audio-model",
		)

	service.record_consent(
		consent_id="consent-recording",
		tenant_id="tenant-audio",
		consent_type="recording",
		subject_id="call-001",
		granted_by="participant-001",
		evidence="signed-consent://call-001",
	)
	with pytest.raises(PermissionError, match="model_policy_required"):
		service.request_transcription(
			job_id="no-policy",
			tenant_id="tenant-audio",
			audio_source_id="call-001",
			requested_by="agent-supervisor",
			model_id="audio-model",
		)

	service.attach_model_policy(
		policy_id="policy-audio",
		tenant_id="tenant-audio",
		model_id="audio-model",
		policy_name="Approved audio model policy",
		allowed_operations=["transcription", "synthesis", "analysis"],
		attached_by="model-governor",
	)
	with pytest.raises(PermissionError, match="synthetic_audio_watermark_required"):
		service.request_synthesis(
			job_id="synth-no-watermark",
			tenant_id="tenant-audio",
			text="Generated speech.",
			requested_by="agent-supervisor",
			model_id="audio-model",
			watermark_applied=False,
		)

	with pytest.raises(PermissionError, match="voice_owner_consent_required"):
		service.request_voice_clone(
			job_id="clone-no-consent",
			tenant_id="tenant-audio",
			voice_owner_id="speaker-001",
			requested_by="agent-supervisor",
			model_id="audio-model",
		)

	transcription = service.request_transcription(
		job_id="transcribe-review",
		tenant_id="tenant-audio",
		audio_source_id="call-001",
		requested_by="agent-supervisor",
		model_id="audio-model",
		confidence=0.5,
	)
	with pytest.raises(ValueError, match="transcript reviewer notes are required"):
		service.decide_transcript_review(
			review_id=transcription["transcript_review"]["id"],
			tenant_id="tenant-audio",
			reviewer="qa-reviewer",
			decision="approved",
			notes="",
		)

	synthesis = service.request_synthesis(
		job_id="synth-review",
		tenant_id="tenant-audio",
		text="Generated speech.",
		requested_by="agent-supervisor",
		model_id="audio-model",
		watermark_applied=True,
	)
	with pytest.raises(ValueError, match="synthesis reviewer notes are required"):
		service.decide_synthesis_review(
			review_id=synthesis["synthesis_review"]["id"],
			tenant_id="tenant-audio",
			reviewer="release-reviewer",
			decision="approved",
			notes="",
		)


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = AudpService()
	for tenant_id, actor in [("tenant-a", "actor-a"), ("tenant-b", "actor-b")]:
		service.record_consent(
			consent_id="shared-consent",
			tenant_id=tenant_id,
			consent_type="recording",
			subject_id="shared-audio",
			granted_by=actor,
			evidence=f"evidence://{tenant_id}",
		)
		service.attach_model_policy(
			policy_id="shared-policy",
			tenant_id=tenant_id,
			model_id="shared-model",
			policy_name="Shared model policy",
			allowed_operations=["transcription"],
			attached_by=actor,
		)
		service.request_transcription(
			job_id="shared-job",
			tenant_id=tenant_id,
			audio_source_id="shared-audio",
			requested_by=actor,
			model_id="shared-model",
		)

	assert service.list_jobs("tenant-a")[0]["requested_by"] == "actor-a"
	assert service.list_jobs("tenant-b")[0]["requested_by"] == "actor-b"
	assert service.list_consents("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_consents("tenant-b")[0]["tenant_id"] == "tenant-b"

	with pytest.raises(ValueError, match="audio job already exists"):
		service.request_transcription(
			job_id="shared-job",
			tenant_id="tenant-a",
			audio_source_id="shared-audio",
			requested_by="actor-a",
			model_id="shared-model",
		)


def test_create_record_compatibility_path_fails_closed_and_persists_status():
	service = AudpService()

	with pytest.raises(PermissionError, match="recording_consent_required"):
		service.create_record("compat-no-consent", "tenant-compat", {"audio_source_id": "compat-audio"})

	service.record_consent(
		consent_id="compat-consent",
		tenant_id="tenant-compat",
		consent_type="recording",
		subject_id="compat-audio",
		granted_by="participant",
		evidence="signed-consent://compat-audio",
	)
	with pytest.raises(PermissionError, match="model_policy_required"):
		service.create_record("compat-no-policy", "tenant-compat", {"audio_source_id": "compat-audio", "model_id": "compat-model"})

	service.attach_model_policy(
		policy_id="compat-policy",
		tenant_id="tenant-compat",
		model_id="compat-model",
		policy_name="Compatibility model policy",
		allowed_operations=["transcription"],
		attached_by="model-governor",
	)
	pending = service.create_record(
		"compat-low-confidence",
		"tenant-compat",
		{
			"audio_source_id": "compat-audio",
			"model_id": "compat-model",
			"confidence": 0.1,
			"human_review_recorded": "false",
		},
	)
	blocked = service.create_record(
		"compat-blocked",
		"tenant-compat",
		{"audio_source_id": "compat-audio", "model_id": "compat-model"},
		status="blocked",
	)

	assert pending["status"] == "pending_review"
	assert service.list_transcript_reviews("tenant-compat")[0]["job_id"] == "compat-low-confidence"
	assert blocked["status"] == "blocked"
	assert {record["id"]: record["status"] for record in service.list_records("tenant-compat")}["compat-blocked"] == "blocked"


def test_api_helpers_and_view_models_expose_audio_lifecycle():
	consent = api_helpers.record_consent({
		"id": "api-consent",
		"tenant_id": "tenant-api-audio",
		"consent_type": "recording",
		"subject_id": "api-call",
		"granted_by": "api-participant",
		"evidence": "signed-consent://api-call",
	})
	policy = api_helpers.attach_model_policy({
		"id": "api-policy",
		"tenant_id": consent["tenant_id"],
		"model_id": "api-model",
		"policy_name": "API model policy",
		"allowed_operations": ["transcription", "synthesis"],
		"attached_by": "api-governor",
	})
	transcription = api_helpers.request_transcription({
		"id": "api-transcription",
		"tenant_id": consent["tenant_id"],
		"audio_source_id": consent["subject_id"],
		"requested_by": "api-user",
		"model_id": policy["model_id"],
		"confidence": 0.5,
	})
	synthesis = api_helpers.request_synthesis({
		"id": "api-synthesis",
		"tenant_id": consent["tenant_id"],
		"text": "Generated response.",
		"requested_by": "api-user",
		"model_id": policy["model_id"],
		"watermark_applied": True,
	})
	decision = api_helpers.decide_transcript_review({
		"id": transcription["transcript_review"]["id"],
		"tenant_id": consent["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Reviewed via API helper.",
	})
	synthesis_decision = api_helpers.decide_synthesis_review({
		"id": synthesis["synthesis_review"]["id"],
		"tenant_id": consent["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Synthetic output reviewed via API helper.",
	})
	model = view_models.transcription_console_model(api_helpers.SERVICE, consent["tenant_id"])
	synthesis_model = view_models.synthesis_studio_model(api_helpers.SERVICE, consent["tenant_id"])

	assert decision["decision"] == "approved"
	assert synthesis_decision["decision"] == "approved"
	assert api_helpers.capability_status(consent["tenant_id"])["job_count"] == 2
	assert model["jobs"][0]["id"] == "api-transcription"
	assert synthesis_model["synthesis_reviews"][0]["decision"] == "approved"
