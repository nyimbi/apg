package apg.capabilities.healthcare

import rego.v1

# HIPAA-compliant access policy for PHI-bearing healthcare capabilities
# Augments the base authz policy with healthcare-specific rules.

default phi_access_allowed := false

# PHI access requires explicit HIPAA authorization role
phi_access_allowed if {
	input.user.roles[_] in {"hipaa_authorized", "clinician", "provider", "admin"}
	input.context.purpose in {"treatment", "payment", "operations", "research_with_consent"}
}

# Deny PHI without documented purpose
deny_reason := "PHI access requires documented purpose (treatment/payment/operations)" if {
	not input.context.purpose
	input.resource_type in phi_resource_types
}

# PHI resource types that require HIPAA authorization
phi_resource_types := {
	"patient_record", "clinical_note", "prescription", "lab_result",
	"diagnosis", "treatment_plan", "insurance_claim", "emr_entry",
}

# Minimum necessary: restrict fields based on role
allowed_phi_fields["clinician"] := {
	"patient_id", "name", "dob", "diagnosis", "treatment_plan",
	"medications", "lab_results", "clinical_notes",
}

allowed_phi_fields["billing"] := {
	"patient_id", "insurance_id", "diagnosis_codes", "procedure_codes",
}

allowed_phi_fields["researcher"] := {
	"age_range", "diagnosis_category", "treatment_outcome",
}
