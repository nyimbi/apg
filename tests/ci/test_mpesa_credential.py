# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
#
# Tests for M-Pesa credential handling. The initiator password must never fall
# back to a value baked into source.
# Run with: uv run pytest -vxs tests/ci/test_mpesa_credential.py

from __future__ import annotations

import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from capabilities.fintech.gateway.mpesa_processor import MPESAPaymentProcessor

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _processor() -> MPESAPaymentProcessor:
	return MPESAPaymentProcessor(
		consumer_key="test-consumer-key",
		consumer_secret="test-consumer-secret",
		business_short_code="600000",
		lipa_na_mpesa_passkey="test-passkey",
	)


def _credential() -> str:
	return asyncio.get_event_loop().run_until_complete(
		_processor()._get_security_credential()
	)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_missing_initiator_password_fails_loudly():
	"""No default: an unset env var must raise, not silently use a baked-in password."""
	os.environ.pop("MPESA_INITIATOR_PASSWORD", None)

	try:
		_credential()
	except RuntimeError as exc:
		assert "MPESA_INITIATOR_PASSWORD" in str(exc)
	else:
		raise AssertionError("expected RuntimeError when MPESA_INITIATOR_PASSWORD is unset")


def test_blank_initiator_password_is_treated_as_missing():
	"""Whitespace is not a credential."""
	os.environ["MPESA_INITIATOR_PASSWORD"] = "   "

	try:
		_credential()
	except RuntimeError as exc:
		assert "MPESA_INITIATOR_PASSWORD" in str(exc)
	else:
		raise AssertionError("expected RuntimeError for a blank MPESA_INITIATOR_PASSWORD")


def test_configured_initiator_password_is_encoded():
	"""The configured password is base64-encoded, as the Daraja credential expects."""
	os.environ["MPESA_INITIATOR_PASSWORD"] = "s3cret-pass"

	assert _credential() == base64.b64encode(b"s3cret-pass").decode()


def test_source_contains_no_hardcoded_initiator_password():
	"""Guard against the default creeping back into the module."""
	source = (
		os.path.join(os.path.dirname(__file__), "..", "..")
		+ "/capabilities/fintech/gateway/mpesa_processor.py"
	)
	with open(source, encoding="utf-8") as fh:
		assert "Safcom001" not in fh.read()