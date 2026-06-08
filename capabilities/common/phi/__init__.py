"""APG PHI (Protected Health Information) classifier — HIPAA compliance.

Provides field-level PHI detection and access-control tagging for healthcare
capabilities. Integrates with OPA healthcare policy and the audit service's
contains_pii field to ensure HIPAA audit trail requirements are met.

Usage::

    from capabilities.common.phi import PHIClassifier, phi_fields

    clf = PHIClassifier()
    result = clf.classify({"patient_id": "P-001", "diagnosis": "T2D", "name": "Jane"})
    print(result.contains_phi)      # True
    print(result.phi_fields)        # ["patient_id", "diagnosis", "name"]
    print(result.minimum_necessary) # fields allowed for the "billing" purpose
"""
from .classifier import PHIClassifier, PHIClassificationResult, phi_fields, is_phi_field

__all__ = [
    "PHIClassifier",
    "PHIClassificationResult",
    "phi_fields",
    "is_phi_field",
]
