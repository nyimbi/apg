# Document Intelligence (docint)

OCR pipeline + LLM extraction for invoice parsing, contract extraction, ID verification, form
digitization, bank statement parsing, and semantic document search — all running on locally-hosted
open-weight models via Ollama.

## New in v2

| Feature | Method |
|---------|--------|
| Document quality gating | `assess_document_quality` |
| Multi-language detection (EN/SW/AR/FR) | `detect_language` |
| Per-field confidence analysis | `get_low_confidence_fields` |
| Field locking (correction-safe re-extraction) | `lock_fields`, `reextract_unlocked_fields` |
| Contract clause classification + risk scoring | `classify_contract_clauses` |
| KYC/AML watchlist screening (OFAC, UN, CBK) | `screen_against_watchlists` |
| Bank statement transaction normalization | `parse_bank_statement` |
| Semantic vector search (pgvector) | `embed_document`, `semantic_search` |
| Human-in-the-loop review queue | `flag_for_review`, `resolve_review` |

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
| POST | /api/docint/documents/{id}/quality | Assess document quality |
| POST | /api/docint/documents/{id}/language | Detect language |
| GET | /api/docint/documents/{id}/confidence | Get per-field confidence |
| POST | /api/docint/documents/{id}/lock | Lock fields |
| POST | /api/docint/documents/{id}/reextract | Re-extract unlocked fields |
| POST | /api/docint/documents/{id}/process/invoice | Invoice pipeline |
| POST | /api/docint/documents/{id}/process/contract | Contract pipeline |
| POST | /api/docint/documents/{id}/process/contract/clauses | Classify contract clauses |
| POST | /api/docint/documents/{id}/verify/id | ID verification |
| POST | /api/docint/documents/{id}/screen | Watchlist screening |
| POST | /api/docint/documents/{id}/parse/bank | Bank statement parsing |
| POST | /api/docint/documents/{id}/digitize | Form digitization |
| POST | /api/docint/documents/{id}/embed | Generate embedding |
| POST | /api/docint/search | Semantic search |
| GET | /api/docint/extractions | List extractions |
| GET | /api/docint/extractions/{id} | Get extraction |
| POST | /api/docint/templates | Create form template |
| GET | /api/docint/templates | List templates |
| GET | /api/docint/templates/{id} | Get template |
| DELETE | /api/docint/templates/{id} | Delete template |
| POST | /api/docint/batches | Submit batch |
| GET | /api/docint/batches/{id} | Get batch |
| POST | /api/docint/batches/{id}/process | Process batch |
| POST | /api/docint/reviews | Flag fields for review |
| POST | /api/docint/reviews/{id}/resolve | Resolve review with corrections |
| GET | /api/docint/statistics | Processing statistics |
| GET | /api/docint/audit | Audit trail |

## Supported Document Types

`invoice` | `contract` | `id_document` | `form` | `receipt` | `bank_statement` | `tax_form` | `other`

## Pipelines

`ocr_only` | `llm_only` | `ocr_llm` (default)

## LLM Backend

Default model: `ollama/llama3`. Override per-request. Multilingual documents automatically
route to `ollama/aya:8b`. All inference is local — no data leaves the platform.

## Streaming

Document processing events are published to NATS subjects:
- `docint.process.{tenant_id}` — processing queue
- `docint.audit.{tenant_id}` — immutable audit stream
- `docint.review_required` — human review escalations
