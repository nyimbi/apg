"""Focused pytest configuration for gateway package contract tests."""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
	"""Create an event loop for legacy async tests without importing providers."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()
