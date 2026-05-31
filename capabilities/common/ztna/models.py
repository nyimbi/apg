"""Zero Trust Network Access domain models."""

from __future__ import annotations

from .zero_trust_runtime import (
	ZeroTrustAccessRequestRecord,
	ZeroTrustAgentRecord,
	ZeroTrustAuditEventRecord,
	ZeroTrustDeviceRecord,
	ZeroTrustIdentityRecord,
	ZeroTrustResourceRecord,
	ZeroTrustSessionRecord,
	ZtnaLifecycleBatchRecord,
)


ZtnaRecord = ZeroTrustAccessRequestRecord


__all__ = [
	"ZeroTrustAccessRequestRecord",
	"ZeroTrustAgentRecord",
	"ZeroTrustAuditEventRecord",
	"ZeroTrustDeviceRecord",
	"ZeroTrustIdentityRecord",
	"ZeroTrustResourceRecord",
	"ZeroTrustSessionRecord",
	"ZtnaLifecycleBatchRecord",
	"ZtnaRecord",
]
