"""GRC Document Management APG capability package."""

from __future__ import annotations

from .api import (
	approve_document,
	assign_retention_policy,
	capability_status,
	complete_processing_job,
	create_document,
	create_record,
	create_revision,
	dashboard_summary,
	grant_access,
	list_records,
	publish_document,
	register_doc_agent,
	register_processing_job,
	register_template,
	service,
)
from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, evaluate_capability_rules, get_capability_contract
from .service import APGDocumentService, DocService, DocumentService, GrcDocService


__version__ = CAPABILITY_VERSION

__all__ = [
	"APGDocumentService",
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"DocService",
	"DocumentService",
	"GrcDocService",
	"approve_document",
	"assign_retention_policy",
	"capability_status",
	"complete_processing_job",
	"create_document",
	"create_record",
	"create_revision",
	"dashboard_summary",
	"evaluate_capability_rules",
	"get_capability_contract",
	"grant_access",
	"list_records",
	"publish_document",
	"register_doc_agent",
	"register_processing_job",
	"register_template",
	"service",
]
