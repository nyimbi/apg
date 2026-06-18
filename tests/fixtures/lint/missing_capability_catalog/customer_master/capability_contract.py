def get_capability_contract(tenant_id="default"):
	return {
		"capability": "customer_master",
		"display_name": "Customer Master",
		"configuration": {"tenant_id": tenant_id, "ui": {}, "theme": {}},
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"rules": [
				{"name": "allow_default", "condition": {"tenant_context_present": True}, "effect": {"decision": "allow"}},
			],
		},
		"ui": {
			"requires_theme": True,
			"shell": "apg_python",
			"template_roots": ["templates/"],
			"routes": [
				{"name": "dashboard", "path": "/customer-master/dashboard", "component": "Dashboard", "permission": "customer_master:view"},
			],
		},
		"theme": {
			"name": "customer_master_theme",
			"tokens": {
				"color.primary": "#1A3A5C",
				"color.accent": "#F59E0B",
				"color.success": "#10B981",
				"color.danger": "#EF4444",
				"surface.canvas": "#F8FAFC",
				"surface.panel": "#FFFFFF",
				"text.primary": "#111827",
				"border.radius": "8px",
			},
			"components": {"dashboard": {"density": "compact"}},
		},
	}
