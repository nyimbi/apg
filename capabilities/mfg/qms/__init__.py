"""
Quality Management (QA/QC) Sub-Capability

Ensures products meet quality standards through inspections, testing,
non-conformance management, and corrective actions with full regulatory compliance.

Key Features:
- Quality Control Plans
- Inspection & Testing Management
- Non-Conformance Tracking
- Corrective & Preventive Actions (CAPA)
- Statistical Process Control (SPC)
- Regulatory Compliance Management
"""

from uuid_extensions import uuid7str

SUB_CAPABILITY_INFO = {
	"id": uuid7str(),
	"name": "Quality Management",
	"code": "MF_QM",
	"parent_capability": "MF",
	"version": "1.0.0",
	"description": "Ensures products meet quality standards through inspections, testing, and corrective actions",
	"features": [
		"Quality Control Plans",
		"Inspection & Testing Management",
		"Non-Conformance Tracking",
		"Corrective & Preventive Actions (CAPA)",
		"Statistical Process Control (SPC)",
		"Regulatory Compliance Management"
	]
}

__all__ = ["SUB_CAPABILITY_INFO"]