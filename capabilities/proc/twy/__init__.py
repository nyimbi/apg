"""APG Three-Way Match Engine (proc_twy).

Matches Purchase Orders, Goods Receipts, and Vendor Invoices with
configurable tolerance rules, auto-approval, and exception management.

Primary public surface:

    from capabilities.proc.twy.service import ThreeWayMatchService
    from capabilities.proc.twy.models import (
        TwMatchDocument, TwMatchResult, TwMatchException,
        TwMatchToleranceRule, TwDocumentType, TwMatchOutcome,
    )
    from capabilities.proc.twy.capability_contract import get_capability_contract
"""

from .models import (
	TwDocumentLine,
	TwDocumentType,
	TwExceptionResolutionType,
	TwExceptionStatus,
	TwMatchAttempt,
	TwMatchDocument,
	TwMatchException,
	TwMatchOutcome,
	TwMatchResult,
	TwMatchStatus,
	TwMatchToleranceRule,
	TwToleranceScope,
	TwVarianceDetail,
	TwVarianceType,
)
from .service import ThreeWayMatchService

__all__ = [
	"ThreeWayMatchService",
	"TwDocumentLine",
	"TwDocumentType",
	"TwExceptionResolutionType",
	"TwExceptionStatus",
	"TwMatchAttempt",
	"TwMatchDocument",
	"TwMatchException",
	"TwMatchOutcome",
	"TwMatchResult",
	"TwMatchStatus",
	"TwMatchToleranceRule",
	"TwToleranceScope",
	"TwVarianceDetail",
	"TwVarianceType",
]
