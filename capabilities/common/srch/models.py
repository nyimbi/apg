"""Search Engine data models."""

from __future__ import annotations

from .search_runtime import (
	QueryRecord,
	SearchAuditEventRecord,
	SearchDocumentRecord,
	SearchIndexRecord,
)


SrchRecord = SearchIndexRecord


__all__ = [
	"QueryRecord",
	"SearchAuditEventRecord",
	"SearchDocumentRecord",
	"SearchIndexRecord",
	"SrchRecord",
]
