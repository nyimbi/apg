"""Tenant context regressions for pharmaceutical default-data initialization."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHARMA_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "pharma" / "blueprint.py"
REGULATORY_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "pharma" / "rec" / "blueprint.py"


def test_pharma_default_data_initializers_use_request_context_tenant():
	source = PHARMA_BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "from ..common.request_context import get_tenant_id_from_context" in source
	assert "tenant_id = get_tenant_id_from_context()" in source
	assert "tenant_id='default_tenant'" not in source
	assert '"default_tenant"' not in source
	assert "tenant_id=tenant_id,\n\t\t\tframework_code=framework_data['framework_code']" in source
	assert "tenant_id=tenant_id,\n\t\t\tcontrol_code=control_data['control_code']" in source
	assert "tenant_id=tenant_id,\n\t\t\tstandard_code=standard_data['standard_code']" in source


def test_pharma_regulatory_initializer_scopes_defaults_to_current_tenant():
	source = REGULATORY_BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "from ...common.request_context import get_tenant_id_from_context" in source
	assert "tenant_id = get_tenant_id_from_context()" in source
	assert "tenant_id='default_tenant'" not in source
	assert '"default_tenant"' not in source
	assert "tenant_id=tenant_id,\n\t\t\t\tframework_code=framework_data['framework_code']" in source
	assert "tenant_id=tenant_id,\n\t\t\t\tcontrol_code=control_data['control_code']" in source
	assert "tenant_id=tenant_id,\n\t\t\t\t\tframework_code='FDA'" in source
