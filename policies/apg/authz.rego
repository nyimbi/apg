package apg.authz

import rego.v1

# Base APG authorization policy
# Input shape: {user: {id, tenant_id, roles: []}, action, resource, resource_type, context: {}}

default allow := false
default decision := "deny"
default matched_rules := []
default actions := []

# Superadmin bypasses all checks
allow if {
	input.user.roles[_] == "superadmin"
}

# Tenant admin has full access within their tenant
allow if {
	input.user.roles[_] == "admin"
	input.context.tenant_id == input.user.tenant_id
}

# Role-based permission check
allow if {
	some role in input.user.roles
	some perm in data.apg.roles[role].permissions
	perm == input.action
}

# Resource-specific owner access
allow if {
	input.context.owner_id == input.user.id
	input.action in {"read", "update", "delete"}
}

# Build decision string
decision := "allow" if allow
decision := "deny" if not allow

# Matched rules for audit trail
matched_rules := rules if {
	rules := [r |
		r := data.apg.rules[_]
		rule_matches(r)
	]
}

rule_matches(rule) if {
	input.user.roles[_] == rule.role
	input.action == rule.action
}

# Actions to take (used by capabilities for post-decision effects)
actions := ["audit"] if {
	allow
	input.action in {"create", "update", "delete"}
}

actions := ["audit", "alert"] if {
	not allow
	input.action in {"admin_override", "export_all"}
}
