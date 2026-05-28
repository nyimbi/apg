def get_capability_contract(tenant_id="default"):
	return {
		"capability": "audit_log",
		"display_name": "Audit Log",
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
				{"name": "dashboard", "path": "/audit-log/dashboard", "component": "Dashboard", "permission": "audit_log:view"},
			],
		},
		"theme": {
			"name": "audit_log_theme",
			"tokens": {"border.radius": "8px"},
			"components": {"dashboard": {"density": "compact"}},
		},
	}
