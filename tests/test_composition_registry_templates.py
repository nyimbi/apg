from capabilities.composition.registry.templates import (
	ComplianceFramework,
	TemplateManager,
	TemplateSize,
)


def test_template_search_rejects_invalid_enum_filters_without_broadening_results():
	manager = TemplateManager()
	assert manager.list_templates()

	assert manager.search_templates(size="tiny") == []
	assert manager.search_templates(compliance="pci_dss") == []

	diagnostics = manager.validate_search_filters(size="tiny", compliance="pci_dss")
	assert diagnostics["valid"] is False
	assert diagnostics["errors"] == [
		{
			"field": "size",
			"value": "tiny",
			"expected": ["small", "medium", "large", "enterprise"],
		},
		{
			"field": "compliance",
			"value": "pci_dss",
			"expected": [
				"fda_cfr_21",
				"iso_13485",
				"iso_9001",
				"iso_14001",
				"sox",
				"gdpr",
				"hipaa",
				"gmp",
				"haccp",
				"itar",
			],
		},
	]


def test_template_search_applies_valid_size_and_compliance_filters():
	manager = TemplateManager()

	medium_templates = manager.search_templates(size="medium")
	assert medium_templates
	assert all(template.template_size == TemplateSize.MEDIUM for template in medium_templates)

	gmp_templates = manager.search_templates(compliance="gmp")
	assert gmp_templates
	assert all(
		ComplianceFramework.GMP in template.compliance_frameworks
		for template in gmp_templates
	)
