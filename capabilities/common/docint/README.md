# Document Intelligence (docint)

OCR pipeline + LLM extraction for invoice parsing, contract extraction, ID verification, form digitization, and batch processing of PDF/image inputs.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/docint/health | Service health |
| GET | /api/docint/documents | List documents |
| POST | /api/docint/documents | Submit document |
| GET | /api/docint/documents/{id} | Get document |
| DELETE | /api/docint/documents/{id} | Delete document |
| GET | /api/docint/documents/{id}/full | Document + extraction |
| POST | /api/docint/documents/{id}/ocr | Run OCR |
| POST | /api/docint/documents/{id}/extract | Extract fields (LLM) |
| POST | /api/docint/documents/{id}/process/invoice | Invoice pipeline |
| POST | /api/docint/documents/{id}/process/contract | Contract pipeline |
| POST | /api/docint/documents/{id}/verify/id | ID verification |
| POST | /api/docint/documents/{id}/digitize | Form digitization |
| GET | /api/docint/extractions | List extractions |
| GET | /api/docint/extractions/{id} | Get extraction |
| POST | /api/docint/templates | Create form template |
| GET | /api/docint/templates | List templates |
| GET | /api/docint/templates/{id} | Get template |
| DELETE | /api/docint/templates/{id} | Delete template |
| POST | /api/docint/batches | Submit batch |
| GET | /api/docint/batches/{id} | Get batch |
| POST | /api/docint/batches/{id}/process | Process batch |
| GET | /api/docint/statistics | Processing statistics |
| GET | /api/docint/audit | Audit trail |
