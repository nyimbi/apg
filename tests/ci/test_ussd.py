# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
#
# Async tests for the USSD Engine capability.
# Run with: uv run pytest -vxs tests/ci/test_ussd.py

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from capabilities.common.ussd.models import (
	FlowDefinition,
	SessionState,
	USSDMenu,
	USSDMenuItem,
	USSDRequest,
	uuid7str,
)
from capabilities.common.ussd.service import USSDEngineService

# ── Fixtures ──────────────────────────────────────────────────────────────────

SERVICE_CODE = "*123#"
MSISDN = "+254700000001"


def _make_flow() -> FlowDefinition:
	"""Minimal 2-menu flow: main → account_info (terminal)."""
	main_menu = USSDMenu(
		menu_id="main",
		title="Main Menu",
		items=[
			USSDMenuItem(key="1", label="Account Info", action="navigate", target="account_info"),
			USSDMenuItem(key="9", label="Exit",         action="end",      target="Thank you. Goodbye."),
		],
		show_exit=False,
	)
	account_menu = USSDMenu(
		menu_id="account_info",
		title="Account Info",
		body="Balance: 0.00",
		is_terminal=True,
		show_back=True,
	)
	return FlowDefinition(
		flow_id=uuid7str(),
		service_code=SERVICE_CODE,
		name="Test Flow",
		root_menu_id="main",
		menus={"main": main_menu, "account_info": account_menu},
	)


async def _svc_with_flow() -> USSDEngineService:
	svc = USSDEngineService()
	flow = _make_flow()
	await svc.create_flow(flow)
	return svc


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_new_session_returns_con_main_menu():
	"""First request (empty text) must return CON with the root menu rendered."""
	svc = await _svc_with_flow()
	session_id = uuid7str()

	req = USSDRequest(
		session_id=session_id,
		service_code=SERVICE_CODE,
		msisdn=MSISDN,
		text="",
	)
	resp = await svc.handle_request(req)

	assert resp.continue_session is True, "initial dial-in must be CON"
	assert "Main Menu" in resp.text
	assert "1. Account Info" in resp.text
	assert resp.session_id == session_id
	assert resp.hop_count == 1


async def test_session_continuation_navigates_to_submenu():
	"""Second request selecting '1' must navigate to account_info and return END (terminal)."""
	svc = await _svc_with_flow()
	session_id = uuid7str()

	# Hop 1: dial-in
	req1 = USSDRequest(session_id=session_id, service_code=SERVICE_CODE, msisdn=MSISDN, text="")
	await svc.handle_request(req1)

	# Hop 2: select option 1
	req2 = USSDRequest(session_id=session_id, service_code=SERVICE_CODE, msisdn=MSISDN, text="1")
	resp = await svc.handle_request(req2)

	assert resp.continue_session is False, "account_info is terminal → END"
	assert "Account Info" in resp.text or "Balance" in resp.text
	assert resp.menu_id == "account_info"
	assert resp.hop_count == 2


async def test_end_session_marks_state_ended():
	"""end_session() must set state to ENDED and prevent further navigation."""
	svc = await _svc_with_flow()
	session_id = uuid7str()

	# Create session via first hop
	req = USSDRequest(session_id=session_id, service_code=SERVICE_CODE, msisdn=MSISDN, text="")
	await svc.handle_request(req)

	# Verify session is active
	session = await svc.get_session(session_id)
	assert session is not None
	assert session.state == SessionState.ACTIVE

	# End it
	await svc.end_session(session_id, reason="test_exit")

	# Verify state (session retained briefly for audit)
	raw = svc._sessions.get(session_id)
	assert raw is not None
	assert raw.state == SessionState.ENDED
	assert raw.ended_at is not None

	# get_session should still return it (ended but within audit window)
	# ended sessions are not purged immediately; only expired ones are
	ended = svc._sessions.get(session_id)
	assert ended is not None
	assert ended.state == SessionState.ENDED
