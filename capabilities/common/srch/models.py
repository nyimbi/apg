"""Search Engine data models."""

from __future__ import annotations

from .search_runtime import (
	QueryRecord,
	SearchAgentRecord,
	SearchAuditEventRecord,
	SearchDocumentRecord,
	SearchIndexRecord,
	SrchLifecycleBatchRecord,
)


SrchRecord = SearchIndexRecord


__all__ = [
	"QueryRecord",
	"SearchAgentRecord",
	"SearchAuditEventRecord",
	"SearchDocumentRecord",
	"SearchIndexRecord",
	"SrchRecord",
	"SrchLifecycleBatchRecord",
]
