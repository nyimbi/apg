"""Pytest fixtures for SACCO GL tests."""
from __future__ import annotations

import asyncio
import pytest

from capabilities.fintech.sacco.gl.service import SACCOGLService


@pytest.fixture()
def svc() -> SACCOGLService:
	return SACCOGLService()


@pytest.fixture()
def tenant() -> str:
	return "test_sacco"


@pytest.fixture()
async def initialised_svc(svc: SACCOGLService, tenant: str) -> SACCOGLService:
	await svc.initialise_sacco_coa(tenant)
	return svc
