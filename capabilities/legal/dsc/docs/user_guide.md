# Document & eDiscovery (leg_dsc) — User Guide

## Overview

Provides a legal-grade document repository with version control, attorney-client privilege logging, litigation hold management, and eDiscovery production set generation.

## Use Cases

- **Litigation support**: issue hold, collect documents, produce to opposing counsel.
- **Privilege review**: maintain a privilege log for withheld documents.
- **Due diligence**: organize and search document repositories for transactions.
- **Regulatory response**: respond to regulator information requests.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Document | A versioned, metadata-rich file in the repository |
| Privilege Log | Formal record of attorney-client and work product assertions |
| Litigation Hold | A freeze on document modification/deletion for preservation |
| Production Set | A curated, Bates-numbered document package for disclosure |

## API Reference

### Upload a Document

```http
POST /api/legal/dsc/documents
{
  "tenant_id": "acme",
  "title": "Board Minutes 2026-01-15",
  "document_type": "internal",
  "owner_id": "atty-003",
  "file_reference": "s3://legal-docs/board-minutes-2026-01-15.pdf",
  "matter_id": "mat-002",
  "is_privileged": false
}
```

### Issue a Litigation Hold

```http
POST /api/legal/dsc/holds
{
  "tenant_id": "acme",
  "matter_id": "mat-002",
  "title": "Smith v Jones — Preservation Hold",
  "description": "All communications from 2024-01-01 to present",
  "custodian_ids": ["emp-001", "emp-007"],
  "issued_by_id": "atty-003",
  "scope_query": "smith jones contract"
}
```

### Create a Production Set

```http
POST /api/legal/dsc/productions
{
  "tenant_id": "acme",
  "matter_id": "mat-002",
  "title": "First Production — Claimant Request",
  "document_ids": ["doc-001", "doc-002", "doc-003"],
  "production_format": "pdf",
  "bates_prefix": "ACME-",
  "requesting_party": "Claimant's Counsel",
  "prepared_by_id": "atty-003"
}
```

### Log Privilege

```http
POST /api/legal/dsc/privilege-log
{
  "tenant_id": "acme",
  "document_id": "doc-099",
  "privilege_type": "attorney_client",
  "basis": "Confidential legal advice requested by CEO",
  "logged_by_id": "atty-003"
}
```
