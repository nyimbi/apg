"""Zero Trust Network Access domain models."""

from __future__ import annotations

from .zero_trust_runtime import (
	ZeroTrustAccessRequestRecord,
	ZeroTrustAuditEventRecord,
	ZeroTrustDeviceRecord,
	ZeroTrustIdentityRecord,
	ZeroTrustResourceRecord,
	ZeroTrustSessionRecord,
)


ZtnaRecord = ZeroTrustAccessRequestRecord


__all__ = [
	"ZeroTrustAccessRequestRecord",
	"ZeroTrustAuditEventRecord",
	"ZeroTrustDeviceRecord",
	"ZeroTrustIdentityRecord",
	"ZeroTrustResourceRecord",
	"ZeroTrustSessionRecord",
	"ZtnaRecord",
]
