"""Production validator reliability configuration regressions."""

from __future__ import annotations

import pytest

from capabilities.composition.gateway.production_validator import ReliabilityValidator


@pytest.mark.asyncio
async def test_reliability_validator_defaults_do_not_emit_canned_findings():
	validator = ReliabilityValidator()

	issues, score = await validator.validate_reliability()

	assert issues == []
	assert score == 100.0


@pytest.mark.asyncio
async def test_reliability_validator_reports_configured_posture():
	validator = ReliabilityValidator({
		"all_services": ["user-service", "payment-service", "notification-service"],
		"services_error_rates": {
			"user-service": 0.4,
			"payment-service": 2.4,
			"notification-service": 1.4,
		},
		"services_with_circuit_breakers": ["user-service"],
		"critical_services": ["user-service", "payment-service"],
		"services_with_retries": ["user-service"],
		"services_with_health_checks": ["user-service", "payment-service"],
		"health_check_interval_seconds": 90,
		"database_backup_enabled": False,
		"backup_frequency_hours": 48,
		"monitored_metrics": ["cpu", "memory", "disk"],
		"critical_metrics": ["cpu", "memory", "disk", "response_time", "error_rate"],
		"alert_channels": ["email"],
	})

	issues, score = await validator.validate_reliability()

	titles = {issue.title for issue in issues}
	assert "High error rate in payment-service" in titles
	assert "Elevated error rate in notification-service" in titles
	assert "Missing circuit breakers" in titles
	assert "Missing retry mechanisms" in titles
	assert "Missing health checks" in titles
	assert "Infrequent health checks" in titles
	assert "Database backups not enabled" in titles
	assert "Infrequent backups" in titles
	assert "Incomplete monitoring coverage" in titles
	assert "Limited alerting channels" in titles
	assert score < 100.0
