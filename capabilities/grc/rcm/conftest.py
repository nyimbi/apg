"""Pytest fixtures for the executable APG RCM package."""

from __future__ import annotations

import pytest

from .service import GrcRcmService


@pytest.fixture
def rcm_service() -> GrcRcmService:
	"""Return a fresh dependency-light RCM service for focused package tests."""
	return GrcRcmService()
