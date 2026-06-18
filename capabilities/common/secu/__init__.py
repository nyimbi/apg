"""APG Security Framework capability.

Standalone package: ``pip install apg-common-secu``

Quick start::

    from apg_common_secu import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : secu
Provides      : risk_assessment, threat_detection, security_policies, compliance_automation, incident_response_governance, security_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-secu"
__capability_id__ = "secu"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]

# Backward-compatibility stub

from enum import Enum as _Enum

class SecurityLevel(str, _Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"

class RiskLevel(str, _Enum):
    LOW = "low"
    MEDIUM = "medium"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(str, _Enum):
    MALWARE = "malware"
    PHISHING = "phishing"
    BRUTE_FORCE = "brute_force"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    DDOS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNKNOWN = "unknown"

class ComplianceFramework(str, _Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    CCPA = "ccpa"
    SOX = "sox"

class SecurityAction(str, _Enum):
    BLOCK = "block"
    ALERT = "alert"
    LOG = "log"
    QUARANTINE = "quarantine"
    REQUIRE_MFA = "require_mfa"
    REVOKE_SESSION = "revoke_session"
    NOTIFY_ADMIN = "notify_admin"
    RATE_LIMIT = "rate_limit"

class DeviceTrustLevel(str, _Enum):
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FULLY_TRUSTED = "fully_trusted"

def get_apg_dependencies() -> dict:
    """Return APG dependency configuration for security capability."""
    return {
        "capability_id": "secu",
        "version": __version__,
        "requires": [],
    }
