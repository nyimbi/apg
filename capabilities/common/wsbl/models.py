"""Website Builder domain models."""

from __future__ import annotations

from .website_runtime import (
	WebsiteAuditEventRecord,
	WebsiteComponentRecord,
	WebsiteDomainRecord,
	WebsitePageRecord,
	WebsitePublishRequestRecord,
	WebsiteSiteRecord,
)


WsblRecord = WebsiteSiteRecord


__all__ = [
	"WebsiteAuditEventRecord",
	"WebsiteComponentRecord",
	"WebsiteDomainRecord",
	"WebsitePageRecord",
	"WebsitePublishRequestRecord",
	"WebsiteSiteRecord",
	"WsblRecord",
]
