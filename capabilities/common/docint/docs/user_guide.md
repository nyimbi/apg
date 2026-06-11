# Document Intelligence User Guide

## Overview

The Document Intelligence capability (`docint`) provides an end-to-end pipeline for extracting
structured data from unstructured documents. It combines OCR (Tesseract/EasyOCR) with
locally-hosted LLM extraction (Ollama) to parse invoices, contracts, ID documents, forms,
and bank statements without sending data to external services.

All models run on-premises via Ollama. The default extraction model is `llama3`; multilingual
documents are automatically routed to `aya:8b`. Embeddings use `nomic-embed-text`.

---

## Supported Document Types

| Type | Description |
|------|-------------|
| `invoice` | Tax invoices, purchase orders — extracts totals, line items, vendor |
| `contract` | Service/sale agreements — extracts parties, dates, obligations, clauses |
| `id_document` | National ID, passport — extracts identity fields, MRZ, verifies expiry |
| `form` | Generic structured forms — maps fields to key-value pairs |
| `receipt` | Point-of-sale receipts — totals, store, date |
| `bank_statement` | Bank statements — transactions, balances, cash flow metrics |
| `tax_form` | Tax returns and certificates |
| `other` | Unclassified documents |

## Pipelines

| Pipeline | Description |
|----------|-------------|
| `ocr_only` | OCR text extraction only, no LLM |
| `llm_only` | Direct LLM extraction (assumes text-layer PDF) |
| `ocr_llm` | OCR then LLM extraction (default, most robust) |

---

## Quickstart

### 1. Submit and process an invoice

```http
POST /api/docint/documents
{
  "tenant_id": "acme",
  "document_type": "invoice",
  "file_name": "invoice-001.pdf",
  "file_url": "s3://acme-docs/invoice-001.pdf",
  "pipeline": "ocr_llm"
}

POST /api/docint/documents/{id}/process/invoice
{"tenant_id": "acme"}
```

Response includes validated fields and `validation.totals_balanced`.

### 2. Verify an ID document

```http
POST /api/docint/documents/{id}/verify/id
{"tenant_id": "acme"}
```

Returns `verification_status: verified | failed` with per-check breakdown.

### 3. Batch processing

```http
POST /api/docint/batches
{
  "tenant_id": "acme",
  "pipeline": "ocr_llm",
  "documents": [
    {"document_type": "invoice", "file_name": "inv-001.pdf", "file_url": "..."},
    {"document_type": "invoice", "file_name": "inv-002.pdf", "file_url": "..."}
  ]
}

POST /api/docint/batches/{batch_id}/process
{"tenant_id": "acme"}
```

---

## New v2 Features

### Document Quality Assessment

Before submitting a document for expensive OCR+LLM processing, assess its quality:

```http
POST /api/docint/documents/{id}/quality
{"tenant_id": "acme"}
```

Response:
```json
{
  "quality_score": 0.82,
  "issues": ["scanned_image_detected"],
  "processable": true,
  "recommendations": []
}
```

Documents with `processable: false` should be re-scanned before pipeline submission.
Quality issues include: `low_resolution_below_150dpi`, `page_skew_detected`, `scanned_image_detected`.

---

### Multi-Language Detection

```http
POST /api/docint/documents/{id}/language
{"tenant_id": "acme"}
```

Response:
```json
{
  "detected_language": "sw",
  "script_type": "Latin",
  "confidence": 0.84,
  "recommended_model": "ollama/aya:8b"
}
```

Supported languages: English (`en`), Swahili (`sw`), Arabic (`ar`), French (`fr`),
Amharic (`am`), Portuguese (`pt`), Somali (`so`).

---

### Per-Field Confidence Analysis

After extraction, retrieve which fields are below a confidence threshold:

```http
GET /api/docint/documents/{id}/confidence?threshold=0.75
{"tenant_id": "acme"}
```

Response:
```json
{
  "low_confidence_fields": {"vendor_address": 0.61, "due_date": 0.72},
  "review_recommended": true
}
```

Fields below threshold should be routed to the review queue.

---

### Field Locking and Incremental Re-Extraction

When a human reviewer corrects a field, lock it so re-runs preserve the correction:

```http
POST /api/docint/documents/{id}/lock
{
  "tenant_id": "acme",
  "field_names": ["vendor_name", "total_amount"]
}
```

Then re-extract without overwriting locked fields:

```http
POST /api/docint/documents/{id}/reextract
{"tenant_id": "acme", "model": "ollama/llama3"}
```

---

### Contract Clause Classification

Get a risk-scored breakdown of every clause in a contract:

```http
POST /api/docint/documents/{id}/process/contract/clauses
{"tenant_id": "acme", "model": "ollama/llama3"}
```

Response:
```json
{
  "clauses": [
    {"clause_type": "indemnity", "risk_score": 0.72, "flags": ["high_risk_indemnity"]},
    {"clause_type": "liability_cap", "risk_score": 0.55, "flags": []}
  ],
  "aggregate_risk": "high",
  "max_clause_risk": 0.72
}
```

Clause types: `indemnity`, `liability_cap`, `termination`, `ip_assignment`,
`non_compete`, `confidentiality`, `force_majeure`, `governing_law`, `payment`.

---

### KYC/AML Watchlist Screening

After extracting an ID document, screen the identity against compliance watchlists:

```http
POST /api/docint/documents/{id}/screen
{
  "tenant_id": "acme",
  "lists": ["ofac", "un_sanctions", "cbk"]
}
```

Response:
```json
{
  "screen_status": "clear",
  "matched_entries": [],
  "lists_checked": ["ofac", "un_sanctions", "cbk"]
}
```

Status values: `clear` | `hit` | `review_required`. A `hit` sets document status to
`watchlist_hit` and should block downstream processing until investigated.

---

### Bank Statement Parsing

Parse a bank statement into normalised transaction records:

```http
POST /api/docint/documents/{id}/parse/bank
{"tenant_id": "acme"}
```

Response:
```json
{
  "account_summary": {
    "opening_balance": 62450.00,
    "closing_balance": 272450.00,
    "total_credits": 250000.00,
    "total_debits": 40000.00,
    "net_flow": 210000.00,
    "currency": "KES"
  },
  "transactions": [
    {
      "date": "2026-05-01",
      "description": "SALARY CREDIT - DATACRAFT LTD",
      "credit": 250000.00,
      "category": "salary",
      "channel": "eft"
    }
  ],
  "cash_flow_metrics": {
    "mpesa_transaction_count": 2
  }
}
```

Recognised transaction channels: `eft`, `mpesa`, `mpesa_paybill`, `atm`, `rtgs`, `cheque`.

---

### Semantic Search

Index documents and search across them with natural language queries:

```http
# Index a document after extraction
POST /api/docint/documents/{id}/embed
{"tenant_id": "acme", "embedding_model": "ollama/nomic-embed-text"}

# Search
POST /api/docint/search
{
  "tenant_id": "acme",
  "query": "contracts with high indemnity risk",
  "top_k": 5,
  "document_type_filter": "contract"
}
```

Response:
```json
{
  "results": [
    {"document_id": "doc-abc123", "similarity": 0.91, "file_name": "supplier-contract.pdf"},
    {"document_id": "doc-def456", "similarity": 0.84, "file_name": "nda-2026.pdf"}
  ]
}
```

In production, embeddings are stored in a PostgreSQL `pgvector` column for
sub-10ms retrieval across millions of documents.

---

### Human-in-the-Loop Review Queue

Flag specific fields for human correction when confidence is low:

```http
POST /api/docint/reviews
{
  "tenant_id": "acme",
  "document_id": "doc-abc123",
  "field_names": ["vendor_name", "total_amount"],
  "reason": "low_confidence"
}
```

A reviewer resolves the ticket with corrected values:

```http
POST /api/docint/reviews/{review_id}/resolve
{
  "tenant_id": "acme",
  "corrections": {"vendor_name": "Acme Corp (Verified)", "total_amount": 174000.00},
  "reviewer_note": "Cross-checked with original scan page 2"
}
```

Corrections are automatically applied to the extraction and the corrected fields are
locked to prevent overwrite. A `docint.review_required` event is published to NATS for
webhook/notification integrations.

---

## LLM Backend

| Use case | Default model |
|----------|--------------|
| English documents | `ollama/llama3` |
| Multilingual (SW/AR/FR) | `ollama/aya:8b` |
| Embeddings | `ollama/nomic-embed-text` |

Override per-request: `{"model": "ollama/mistral"}`. All inference runs locally.

## Event Streaming (NATS)

| Subject | Event |
|---------|-------|
| `docint.process.{tenant_id}` | Document submitted for processing |
| `docint.audit.{tenant_id}` | All pipeline events (immutable) |
| `docint.review_required` | Human review escalation |

Bytewax pipelines subscribe to `docint.audit.*` for real-time monitoring dashboards
and downstream workflow triggers.

## Error Codes

| Status | Meaning |
|--------|---------|
| `quality_failed` | Document below quality threshold — re-upload required |
| `watchlist_hit` | Identity matched a sanctions/compliance list |
| `review_pending` | Awaiting human review before further processing |
| `reviewed` | Human corrections applied |
| `processed` | Full pipeline completed |
| `verified` | ID document verification passed |
