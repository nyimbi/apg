"""Production validator compliance and deployment configuration regressions."""

from __future__ import annotations

import pytest

from capabilities.composition.gateway.production_validator import ProductionReadinessValidator


@pytest.mark.asyncio
async def test_production_validator_defaults_do_not_emit_compliance_or_deployment_mocks():
	validator = ProductionReadinessValidator(db_session=None)

	compliance_issues, compliance_status = await validator._validate_compliance()
	deployment_issues = await validator._validate_deployment_readiness()

	assert compliance_issues == []
	assert compliance_status["pci_dss_compliant"] is True
	assert deployment_issues == []


@pytest.mark.asyncio
async def test_production_validator_reports_configured_compliance_and_deployment_posture(monkeypatch):
	monkeypatch.delenv("APG_REQUIRED_TEST_SECRET", raising=False)
	validator = ProductionReadinessValidator(
		db_session=None,
		validation_config={
			"compliance": {
				"status": {
					"gdpr_compliant": "false",
					"pci_dss_compliant": "false",
				},
			},
			"deployment": {
				"missing_env_vars": ["DATABASE_URL"],
				"required_env_vars": ["APG_REQUIRED_TEST_SECRET"],
				"migrations_applied": "false",
				"unavailable_services": ["redis", "auth-service"],
			},
		},
	)

	compliance_issues, compliance_status = await validator._validate_compliance()
	deployment_issues = await validator._validate_deployment_readiness()

	titles = {issue.title for issue in [*compliance_issues, *deployment_issues]}
	assert compliance_status["gdpr_compliant"] is False
	assert compliance_status["pci_dss_compliant"] is False
	assert "GDPR compliance violation" in titles
	assert "PCI DSS compliance required" in titles
	assert "Missing environment variables" in titles
	assert "Database migrations not applied" in titles
	assert "External service dependencies unavailable" in titles

	missing_env_issue = next(issue for issue in deployment_issues if issue.title == "Missing environment variables")
	assert "APG_REQUIRED_TEST_SECRET" in missing_env_issue.description
	assert "DATABASE_URL" in missing_env_issue.description
