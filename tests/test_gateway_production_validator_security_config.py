"""Production validator security configuration regressions."""

from __future__ import annotations

import pytest

from capabilities.composition.gateway.production_validator import SecurityValidator


@pytest.mark.asyncio
async def test_security_validator_defaults_do_not_emit_mock_findings():
	validator = SecurityValidator()

	issues, score = await validator.validate_security()

	assert issues == []
	assert score == 100.0


@pytest.mark.asyncio
async def test_security_validator_reports_configured_posture():
	validator = SecurityValidator({
		"authentication_mechanisms": ["basic_auth"],
		"rbac_enabled": False,
		"admin_users_count": 5,
		"total_users": 10,
		"database_encrypted": False,
		"tls_enabled": True,
		"tls_version": "1.0",
		"firewall_enabled": False,
		"open_ports": [22, 80, 443, 5432],
		"allowed_public_ports": [80, 443],
		"sql_injection_protection": False,
		"xss_protection": False,
		"vulnerable_packages": [{
			"name": "real-lib",
			"version": "0.9.0",
			"vulnerability": "CVE-2026-0001",
			"severity": "HIGH",
		}],
		"hardcoded_secrets_found": True,
		"secrets_rotated_recently": False,
		"cert_valid": True,
		"cert_expires_soon": True,
	})

	issues, score = await validator.validate_security()

	titles = {issue.title for issue in issues}
	assert "Weak authentication mechanisms detected" in titles
	assert "Role-based access control not implemented" in titles
	assert "Too many administrative users" in titles
	assert "Database encryption not enabled" in titles
	assert "Outdated TLS version" in titles
	assert "Firewall not configured" in titles
	assert "Unnecessary ports open" in titles
	assert "SQL injection vulnerability" in titles
	assert "Cross-site scripting vulnerability" in titles
	assert "Vulnerable dependency: real-lib" in titles
	assert "Hardcoded secrets detected" in titles
	assert "Secrets not rotated recently" in titles
	assert "SSL certificate expires soon" in titles
	assert score < 100.0
