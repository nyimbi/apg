"""Focused pytest configuration for the BFC package contract tests."""

from __future__ import annotations

import logging

import pytest


def pytest_configure(config: pytest.Config) -> None:
	"""Register local markers without importing optional provider test stacks."""
	logging.disable(logging.CRITICAL)
	config.addinivalue_line("markers", "integration: integration tests for provider-backed BFC flows")
	config.addinivalue_line("markers", "unit: dependency-light BFC unit tests")
	config.addinivalue_line("markers", "slow: longer BFC tests")
