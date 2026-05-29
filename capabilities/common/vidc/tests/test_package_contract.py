"""VIDC package contract and deterministic runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.vidc import api, views
from capabilities.common.vidc.service import VidcService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_is_valid():
	module = _load_module("materialized_contract_vidc", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "vidc"
	assert contract["ui"]["routes"]
	assert contract["theme"]["tokens"]["border.radius"]


def test_app_entrypoint_is_publishable():
	module = _load_module("materialized_app_vidc", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "vidc" in model["capabilities"]


def test_room_meeting_participant_recording_caption_and_end_lifecycle_executes():
	service = VidcService()

	room = service.create_room(
		tenant_id="tenant-a",
		room_name="Ops Room",
		owner="meeting-owner",
		guest_policy_ref="guest-policy://ops",
		moderation_policy_ref="moderation://ops",
	)
	meeting = service.start_meeting(
		tenant_id="tenant-a",
		room_id=room["id"],
		title="Operations Standup",
		host_id="host-1",
		participant_count=3,
		external_guest_count=1,
		recording_requested=True,
		recording_consent_ref="consent://meeting/1",
		recording_encrypted=True,
	)
	participant = service.add_participant(
		tenant_id="tenant-a",
		meeting_id=meeting["id"],
		user_ref="host-1",
		display_name="Host One",
		role="host",
	)
	recording = service.create_recording(
		tenant_id="tenant-a",
		meeting_id=meeting["id"],
		recording_ref="recording://meeting/1",
		consent_ref="consent://meeting/1",
		retention_policy_ref="retention://meetings/90d",
		encrypted=True,
		created_by="host-1",
	)
	captions = service.generate_captions(
		tenant_id="tenant-a",
		meeting_id=meeting["id"],
		language="en",
		transcript_ref="transcript://meeting/1",
		caption_count=42,
		generated_by="caption-service",
	)
	ended = service.end_meeting("tenant-a", meeting["id"], "host-1")
	summary = service.dashboard_summary("tenant-a")

	assert room["status"] == "ready"
	assert meeting["status"] == "active"
	assert participant["role"] == "host"
	assert recording["encrypted"] is True
	assert captions["caption_count"] == 42
	assert ended["status"] == "ended"
	assert summary["room_count"] == 1
	assert summary["meeting_count"] == 1
	assert summary["recording_count"] == 1
	assert summary["caption_count"] == 1


def test_video_guardrails_require_tenant_host_guest_policy_consent_encryption_and_capacity_review():
	service = VidcService()

	try:
		service.create_room("", "No tenant", "owner", "guest-policy://x", "moderation://x")
	except PermissionError as exc:
		assert str(exc) == "tenant_context_required"
	else:
		raise AssertionError("missing tenant was accepted")

	room = service.create_room("tenant-a", "Secure Room", "owner", "", "moderation://secure")

	try:
		service.start_meeting("tenant-a", room["id"], "No Host", "", 2)
	except PermissionError as exc:
		assert str(exc) == "host_required"
	else:
		raise AssertionError("meeting without host was accepted")

	try:
		service.start_meeting("tenant-a", room["id"], "No Guest Policy", "host-1", 2, external_guest_count=1)
	except PermissionError as exc:
		assert str(exc) == "guest_policy_required"
	else:
		raise AssertionError("external guest without policy was accepted")

	try:
		service.start_meeting("tenant-a", room["id"], "No Consent", "host-1", 2, recording_requested=True, recording_encrypted=True)
	except PermissionError as exc:
		assert str(exc) == "recording_consent_required"
	else:
		raise AssertionError("recording without consent was accepted")

	try:
		service.start_meeting("tenant-a", room["id"], "No Encryption", "host-1", 2, recording_requested=True, recording_consent_ref="consent://x")
	except PermissionError as exc:
		assert str(exc) == "recording_encryption_required"
	else:
		raise AssertionError("recording without encryption was accepted")

	large = service.start_meeting(
		tenant_id="tenant-a",
		room_id=room["id"],
		title="Large Meeting",
		host_id="host-1",
		participant_count=750,
		external_guest_count=0,
		recording_requested=False,
		capacity_review_recorded=False,
	)
	assert large["status"] == "review_required"
	assert large["required_actions"] == ["review_meeting_capacity"]


def test_api_and_view_models_expose_video_conferencing_surfaces():
	local_service = VidcService()
	api.SERVICE = local_service

	room = api.create_room({
		"tenant_id": "tenant-b",
		"room_name": "Customer Room",
		"owner": "meeting-owner",
		"guest_policy_ref": "guest-policy://customer",
		"moderation_policy_ref": "moderation://customer",
	})
	meeting = api.start_meeting({
		"tenant_id": "tenant-b",
		"room_id": room["id"],
		"title": "Customer Review",
		"host_id": "host-1",
		"participant_count": 2,
		"external_guest_count": 1,
	})
	api.add_participant({
		"tenant_id": "tenant-b",
		"meeting_id": meeting["id"],
		"user_ref": "guest-1",
		"display_name": "Guest One",
		"role": "guest",
		"external_guest": True,
	})
	api.create_recording({
		"tenant_id": "tenant-b",
		"meeting_id": meeting["id"],
		"recording_ref": "recording://customer",
		"consent_ref": "consent://customer",
		"retention_policy_ref": "retention://meetings",
		"encrypted": True,
		"created_by": "host-1",
	})
	api.generate_captions({
		"tenant_id": "tenant-b",
		"meeting_id": meeting["id"],
		"language": "en",
		"transcript_ref": "transcript://customer",
		"caption_count": 10,
		"generated_by": "caption-service",
	})

	status = api.capability_status("tenant-b")
	system = api.list_video_conferencing("tenant-b")
	dashboard = views.dashboard_model(local_service, "tenant-b")
	meetings = views.meeting_console_model(local_service, "tenant-b")
	rooms = views.room_manager_model(local_service, "tenant-b")
	participants = views.participant_panel_model(local_service, "tenant-b")
	recordings = views.recording_library_model(local_service, "tenant-b")
	captions = views.caption_workbench_model(local_service, "tenant-b")
	analytics = views.analytics_model(local_service, "tenant-b")
	settings = views.settings_model("tenant-b")

	assert status["meeting_count"] == 1
	assert system["summary"]["participant_count"] == 1
	assert dashboard["summary"]["recording_count"] == 1
	assert meetings["route"] == "/vidc/meetings"
	assert rooms["waiting_room_supported"] is True
	assert participants["roles"] == ["host", "cohost", "participant", "guest", "observer"]
	assert recordings["encryption_required"] is True
	assert captions["languages_supported"] == ["en", "fr", "sw", "ar"]
	assert analytics["review_required_meetings"] == []
	assert settings["theme"]["name"] == "vidc_meeting_room"
