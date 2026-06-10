# Document Intelligence User Guide

## Overview

The Document Intelligence capability (`docint`) provides an end-to-end pipeline for extracting structured data from unstructured documents. It combines OCR (Tesseract/EasyOCR) with locally-hosted LLM extraction (Ollama) to parse invoices, contracts, ID documents, and forms without sending data to external services.

## Supported Document Types

| Type | Description |
|------|-------------|
| `invoice` | Tax invoices, purchase orders — extracts totals, line items, vendor |
| `contract` | Service/sale agreements — extracts parties, dates, obligations |
| `id_document` | National ID, passport — extracts identity fields, MRZ, verifies expiry |
| `form` | Generic structured forms — maps fields to key-value pairs |
| `receipt` | Point-of-sale receipts — totals, store, date |
| `bank_statement` | Bank statements — transactions, balances |
| `tax_form` | Tax returns and certificates |
| `other` | Unclassified documents |

## Pipelines

| Pipeline | Description |
|----------|-------------|
| `ocr_only` | OCR text extraction only, no LLM |
| `llm_only` | Direct LLM extraction (assumes text-layer PDF) |
| `ocr_llm` | OCR then LLM extraction (default, most robust) |

## Quickstart

### Submit and process an invoice

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

### Verify an ID document

```http
POST /api/docint/documents/{id}/verify/id
{"tenant_id": "acme"}
```

Returns `verification_status: verified | failed` with per-check breakdown.

### Batch processing

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

## LLM Backend

By default uses `ollama/llama3`. Override per-request with `{"model": "ollama/mistral"}`.
All inference runs locally — no data leaves the platform.
