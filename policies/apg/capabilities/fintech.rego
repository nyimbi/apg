package apg.capabilities.fintech

import rego.v1

# PCI DSS scope isolation policy for fintech capabilities
# Augments the base authz policy with PCI DSS rules.

# Capabilities in PCI DSS scope
pci_scope_capabilities := {
	"fintech_gwy", "fintech_trx", "fintech_cre", "fintech_clc",
}

# Roles permitted to access PCI-scoped data
pci_authorized_roles := {
	"pci_authorized", "payment_processor", "fraud_analyst", "admin",
}

default pci_access_allowed := false

pci_access_allowed if {
	input.capability_id in pci_scope_capabilities
	input.user.roles[_] in pci_authorized_roles
}

# Deny cardholder data access to non-PCI roles
deny_reason := "Access to cardholder data requires PCI DSS authorization" if {
	input.capability_id in pci_scope_capabilities
	not pci_access_allowed
}

# AML-specific: flag high-value transactions for review
require_review if {
	input.action == "process_transaction"
	input.context.amount_usd > 10000
}

# AML-specific: block sanctioned countries
deny_reason := "Transaction blocked: sanctioned jurisdiction" if {
	input.context.destination_country in sanctioned_countries
}

sanctioned_countries := {"IR", "KP", "SY", "CU", "SD"}
