from capabilities.common.conf.models import ConfigurationDSL


def test_configuration_dsl_exports_readable_nested_hcl() -> None:
	dsl = ConfigurationDSL(
		kind="WebApplication",
		metadata={
			"name": "Customer Portal",
			"owner-team": "commerce",
		},
		spec={
			"resources": {
				"cpu": "2",
				"memory": "4Gi",
			},
			"replicas": 3,
			"monitoring": {
				"enabled": True,
				"alerts": ["latency", "errors"],
			},
			"security": {
				"encryption_at_rest": True,
				"network_policy": None,
			},
		},
		dependencies=["auth-service", "catalog-db"],
		variables={"environment": "production"},
	)

	hcl = dsl.to_hcl()

	assert hcl.startswith('apg_configuration "customer_portal" {')
	assert 'kind = "WebApplication"' in hcl
	assert '"owner-team" = "commerce"' in hcl
	assert "replicas = 3" in hcl
	assert "enabled = true" in hcl
	assert 'alerts = [\n\t\t\t\t"latency",\n\t\t\t\t"errors"\n\t\t\t]' in hcl
	assert "network_policy = null" in hcl
	assert 'dependencies = [\n\t\t"auth-service",\n\t\t"catalog-db"\n\t]' in hcl
	assert hcl.endswith("}\n")
