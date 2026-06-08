package apg.capabilities.pharma

import rego.v1

# FDA 21 CFR Part 11 / GxP access policy for pharmaceutical capabilities
# Requires validated system access and electronic signature authorization.

# Roles with GxP-authorized access
gxp_authorized_roles := {
	"gxp_authorized", "qc_analyst", "qa_manager", "regulatory_affairs",
	"clinical_investigator", "pharmacovigilance_officer", "admin",
}

default gxp_access_allowed := false

gxp_access_allowed if {
	input.user.roles[_] in gxp_authorized_roles
}

# Electronic signature required for GxP-critical operations
esig_required_operations := {
	"approve_batch_record", "release_product", "close_deviation",
	"approve_change_control", "sign_protocol", "certify_analytical_result",
}

esig_required if {
	input.action in esig_required_operations
}

# 21 CFR Part 11: electronic signature must have three components
valid_esig(esig) if {
	esig.meaning != ""         # What the signer is agreeing to
	esig.signer_id != ""       # Identity of the signer (user ID)
	esig.timestamp != ""       # Date/time of signing
}

deny_reason := "Electronic signature required for GxP-critical operation" if {
	input.action in esig_required_operations
	not valid_esig(input.context.electronic_signature)
}

# Audit trail: all GxP operations must be audit-logged
require_audit if {
	input.capability_id in gxp_capability_ids
}

gxp_capability_ids := {
	"pharma_qlt", "pharma_mfg", "pharma_com", "pharma_ctr",
	"pharma_phl", "pharma_rlt",
}

# Data retention: 10 years for pharma audit records (21 CFR Part 11)
retention_years := 10
