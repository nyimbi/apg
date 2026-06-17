"""Document Intelligence service — OCR pipeline, LLM extraction, invoice/contract/ID parsing, form digitization."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import base64
import hashlib
import logging
import re
import time
from copy import deepcopy
from datetime import datetime, date
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "docint"
SUPPORTED_DOC_TYPES = {"invoice", "contract", "id_document", "form", "receipt", "bank_statement", "tax_form", "other"}
SUPPORTED_PIPELINES = {"ocr_only", "llm_only", "ocr_llm"}
SUPPORTED_LANGUAGES = {"en", "sw", "ar", "fr", "am", "pt", "so"}  # English, Swahili, Arabic, French, Amharic, Portuguese, Somali
WATCHLIST_SOURCES = {"ofac", "un_sanctions", "cbk", "local"}
CONTRACT_CLAUSE_TYPES = {
	"indemnity", "liability_cap", "termination", "ip_assignment",
	"non_compete", "confidentiality", "force_majeure", "governing_law", "payment",
}
SUPPORTED_MIME_TYPES = {
	"application/pdf", "image/jpeg", "image/png", "image/tiff",
	"image/webp", "image/bmp", "application/msword",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class DocumentIntelligenceService:
	"""OCR + LLM extraction pipeline for invoices, contracts, ID docs, forms, PDFs, and images."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.documents: dict[str, dict[str, Any]] = {}
		self.extractions: dict[str, dict[str, Any]] = {}
		self.verification_results: dict[str, dict[str, Any]] = {}
		self.form_templates: dict[str, dict[str, Any]] = {}
		self.batches: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# New state stores for enhanced features
		self._review_queue = WriteThruDict('review_queue', tenant_id, _store)
		self._embeddings: dict[str, list[float]] = {}
		self._watchlist_hits = WriteThruDict('watchlist_hits', tenant_id, _store)
		self._field_locks: dict[str, set[str]] = {}  # document_id -> set of locked field names
		self._quality_assessments = WriteThruDict('quality_assessments', tenant_id, _store)
		self._clause_classifications: dict[str, list[dict[str, Any]]] = {}
		self._signature_detections = WriteThruDict('signature_detections', tenant_id, _store)

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, document_id: str, document_type: str = "", payload: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"document_id": document_id,
			"document_type": document_type,
			"payload": payload or {},
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "docint",
			"status": "healthy",
			"document_count": len(self.documents),
			"extraction_count": len(self.extractions),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"supported_document_types": sorted(SUPPORTED_DOC_TYPES),
			"supported_pipelines": sorted(SUPPORTED_PIPELINES),
			"supported_mime_types": sorted(SUPPORTED_MIME_TYPES),
			"features": [
				"ocr_pipeline", "llm_extraction", "invoice_parsing", "contract_extraction",
				"id_verification", "form_digitization", "batch_processing", "template_matching"
			],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Document submission ───────────────────────────────────────

	async def submit_document(
		self,
		tenant_id: str,
		document_type: str,
		file_name: str,
		file_content_base64: str = "",
		file_url: str = "",
		mime_type: str = "application/pdf",
		metadata: dict[str, Any] | None = None,
		pipeline: str = "ocr_llm",
	) -> dict[str, Any]:
		"""Submit a document for processing."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(file_name, "file_name")
		if document_type not in SUPPORTED_DOC_TYPES:
			raise ValueError(f"document_type must be one of {sorted(SUPPORTED_DOC_TYPES)}")
		if pipeline not in SUPPORTED_PIPELINES:
			raise ValueError(f"pipeline must be one of {sorted(SUPPORTED_PIPELINES)}")
		if not file_content_base64 and not file_url:
			raise ValueError("either file_content_base64 or file_url must be provided")

		# Compute content hash for deduplication
		content_hash = ""
		if file_content_base64:
			try:
				raw = base64.b64decode(file_content_base64)
				content_hash = hashlib.sha256(raw).hexdigest()
			except Exception:
				content_hash = hashlib.sha256(file_content_base64.encode()).hexdigest()

		record: dict[str, Any] = {
			"id": self._id("doc"),
			"tenant_id": tenant,
			"document_type": document_type,
			"file_name": file_name,
			"mime_type": mime_type,
			"pipeline": pipeline,
			"file_url": file_url,
			"content_hash": content_hash,
			"metadata": metadata or {},
			"status": "pending",
			"submitted_at": self._now(),
		}
		self.documents[record["id"]] = record
		self._emit(tenant, "document_submitted", record["id"], document_type, {
			"file_name": file_name, "pipeline": pipeline
		})
		_log.info("document submitted: %s type=%s pipeline=%s tenant=%s", record["id"], document_type, pipeline, tenant)
		return deepcopy(record)

	async def get_document(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		return deepcopy(doc)

	async def list_documents(
		self,
		tenant_id: str,
		document_type: str | None = None,
		status: str | None = None,
		pipeline: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.documents.values() if d["tenant_id"] == tenant]
		if document_type:
			items = [d for d in items if d["document_type"] == document_type]
		if status:
			items = [d for d in items if d["status"] == status]
		if pipeline:
			items = [d for d in items if d["pipeline"] == pipeline]
		return items

	async def delete_document(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		doc["status"] = "deleted"
		doc["deleted_at"] = self._now()
		self._emit(tenant, "document_deleted", document_id, doc["document_type"])
		return deepcopy(doc)

	# ── OCR pipeline ──────────────────────────────────────────────

	async def run_ocr(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Run OCR on a submitted document (Tesseract/EasyOCR simulation)."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		start = time.monotonic()
		# Simulate OCR processing (real impl would call Tesseract/EasyOCR)
		await asyncio.sleep(0.005)  # simulate I/O
		ocr_text = self._simulate_ocr_text(doc["document_type"])
		processing_ms = round((time.monotonic() - start) * 1000, 2)

		result: dict[str, Any] = {
			"id": self._id("ocr"),
			"tenant_id": tenant,
			"document_id": document_id,
			"ocr_text": ocr_text,
			"pages": 1,
			"word_count": len(ocr_text.split()),
			"confidence": 0.92,
			"processing_ms": processing_ms,
			"completed_at": self._now(),
		}
		doc["ocr_result"] = result
		doc["status"] = "ocr_completed"
		self._emit(tenant, "ocr_completed", document_id, doc["document_type"], {"pages": 1})
		return deepcopy(result)

	def _simulate_ocr_text(self, document_type: str) -> str:
		samples = {
			"invoice": "INVOICE\nInvoice #: INV-2026-001\nDate: 2026-06-10\nVendor: Acme Corp\nAmount: KES 150,000.00\nTax: KES 24,000.00\nTotal: KES 174,000.00",
			"contract": "SERVICE AGREEMENT\nThis agreement between Datacraft Ltd and Client Corp\nEffective Date: 2026-01-01\nTerm: 12 months\nPayment Terms: Net 30",
			"id_document": "REPUBLIC OF KENYA\nNATIONAL IDENTITY CARD\nName: JOHN DOE\nID No: 12345678\nDOB: 01/01/1990\nExpiry: 01/01/2031",
			"form": "FORM DATA\nField1: Value1\nField2: Value2\nSignature: ___________\nDate: 2026-06-10",
			"receipt": "RECEIPT\nStore: Quick Mart\nDate: 2026-06-10\nTotal: KES 2,500\nVAT: KES 400",
		}
		return samples.get(document_type, "DOCUMENT TEXT\nExtracted content placeholder")

	# ── LLM extraction ────────────────────────────────────────────

	async def extract_fields(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Extract structured fields from document text using an LLM."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		start = time.monotonic()
		# Simulate LLM extraction — real impl calls Ollama or LiteLLM
		await asyncio.sleep(0.01)
		doc_type = doc["document_type"]
		extracted = self._simulate_extraction(doc_type, doc.get("ocr_result", {}).get("ocr_text", ""))
		processing_ms = round((time.monotonic() - start) * 1000, 2)

		result: dict[str, Any] = {
			"id": self._id("ext"),
			"tenant_id": tenant,
			"document_id": document_id,
			"document_type": doc_type,
			"extracted_fields": extracted,
			"confidence": 0.88,
			"ocr_text": doc.get("ocr_result", {}).get("ocr_text", ""),
			"pages": doc.get("ocr_result", {}).get("pages", 1),
			"model": model,
			"processing_ms": processing_ms,
			"status": "completed",
			"extracted_at": self._now(),
		}
		self.extractions[result["id"]] = result
		doc["extraction_id"] = result["id"]
		doc["status"] = "extracted"
		self._emit(tenant, "fields_extracted", document_id, doc_type, {
			"confidence": 0.88, "field_count": len(extracted)
		})
		_log.info("extraction completed: doc=%s type=%s fields=%d", document_id, doc_type, len(extracted))
		return deepcopy(result)

	def _simulate_extraction(self, document_type: str, ocr_text: str) -> dict[str, Any]:
		"""Simulate LLM field extraction from OCR text."""
		if document_type == "invoice":
			return {
				"invoice_number": "INV-2026-001",
				"invoice_date": "2026-06-10",
				"vendor_name": "Acme Corp",
				"currency": "KES",
				"subtotal": 150000.00,
				"tax_amount": 24000.00,
				"total_amount": 174000.00,
				"line_items": [
					{"description": "Consulting Services", "quantity": 1, "unit_price": 150000.00, "total": 150000.00}
				],
			}
		elif document_type == "contract":
			return {
				"contract_title": "Service Agreement",
				"parties": ["Datacraft Ltd", "Client Corp"],
				"effective_date": "2026-01-01",
				"expiry_date": "2026-12-31",
				"governing_law": "Kenya",
				"payment_terms": "Net 30",
				"key_obligations": ["Service delivery", "Monthly reporting"],
			}
		elif document_type == "id_document":
			return {
				"document_number": "12345678",
				"full_name": "JOHN DOE",
				"date_of_birth": "1990-01-01",
				"nationality": "Kenyan",
				"issuing_country": "KE",
				"issue_date": "2021-01-01",
				"expiry_date": "2031-01-01",
			}
		elif document_type == "form":
			# Generic form field extraction
			fields: dict[str, Any] = {}
			for line in ocr_text.split("\n"):
				if ":" in line:
					parts = line.split(":", 1)
					fields[parts[0].strip().lower().replace(" ", "_")] = parts[1].strip()
			return fields
		elif document_type == "receipt":
			return {
				"store_name": "Quick Mart",
				"date": "2026-06-10",
				"total": 2500.0,
				"vat": 400.0,
				"currency": "KES",
			}
		return {}

	# ── Document-type specific pipelines ─────────────────────────

	async def process_invoice(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Full invoice processing pipeline: OCR -> extraction -> validation."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if doc["document_type"] != "invoice":
			raise ValueError(f"document is not an invoice: {doc['document_type']}")

		ocr = await self.run_ocr(tenant_id, document_id)
		extraction = await self.extract_fields(tenant_id, document_id, model)
		fields = extraction["extracted_fields"]

		# Validate totals
		subtotal = fields.get("subtotal", 0.0) or 0.0
		tax = fields.get("tax_amount", 0.0) or 0.0
		total = fields.get("total_amount", 0.0) or 0.0
		balance_check = abs((subtotal + tax) - total) < 1.0

		result: dict[str, Any] = {
			"document_id": document_id,
			"pipeline": "invoice",
			"ocr_word_count": ocr["word_count"],
			"fields": fields,
			"validation": {
				"totals_balanced": balance_check,
				"has_invoice_number": bool(fields.get("invoice_number")),
				"has_vendor": bool(fields.get("vendor_name")),
				"has_date": bool(fields.get("invoice_date")),
			},
			"confidence": extraction["confidence"],
			"processed_at": self._now(),
		}
		doc["status"] = "processed"
		self._emit(tenant, "invoice_processed", document_id, "invoice", {"valid": balance_check})
		return result

	async def process_contract(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Full contract extraction pipeline: OCR -> field extraction -> risk flags."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if doc["document_type"] != "contract":
			raise ValueError(f"document is not a contract: {doc['document_type']}")

		ocr = await self.run_ocr(tenant_id, document_id)
		extraction = await self.extract_fields(tenant_id, document_id, model)
		fields = extraction["extracted_fields"]

		# Flag risk indicators
		risk_flags = []
		if not fields.get("expiry_date"):
			risk_flags.append("no_expiry_date")
		if not fields.get("governing_law"):
			risk_flags.append("no_governing_law")
		if not fields.get("payment_terms"):
			risk_flags.append("no_payment_terms")

		result: dict[str, Any] = {
			"document_id": document_id,
			"pipeline": "contract",
			"fields": fields,
			"risk_flags": risk_flags,
			"risk_level": "high" if len(risk_flags) >= 2 else ("medium" if risk_flags else "low"),
			"confidence": extraction["confidence"],
			"processed_at": self._now(),
		}
		doc["status"] = "processed"
		self._emit(tenant, "contract_processed", document_id, "contract", {
			"risk_level": result["risk_level"]
		})
		return result

	async def verify_id_document(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""ID document verification: OCR -> extraction -> MRZ check -> validity assessment."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if doc["document_type"] != "id_document":
			raise ValueError(f"document is not an ID document: {doc['document_type']}")

		ocr = await self.run_ocr(tenant_id, document_id)
		extraction = await self.extract_fields(tenant_id, document_id, model)
		fields = extraction["extracted_fields"]

		# Check expiry
		expiry = fields.get("expiry_date", "")
		is_expired = False
		if expiry:
			try:
				from datetime import date
				exp_date = date.fromisoformat(expiry)
				is_expired = exp_date < date.today()
			except Exception as _exc:
				_log.debug("Handled exception: %s", _exc)

		checks = {
			"has_document_number": bool(fields.get("document_number")),
			"has_full_name": bool(fields.get("full_name")),
			"has_dob": bool(fields.get("date_of_birth")),
			"not_expired": not is_expired,
			"has_issuing_country": bool(fields.get("issuing_country")),
		}
		verification_status = "verified" if all(checks.values()) else "failed"

		result: dict[str, Any] = {
			"id": self._id("ver"),
			"tenant_id": tenant,
			"document_id": document_id,
			"fields": fields,
			"checks": checks,
			"verification_status": verification_status,
			"is_expired": is_expired,
			"confidence": extraction["confidence"],
			"verified_at": self._now(),
		}
		self.verification_results[result["id"]] = result
		doc["status"] = "verified"
		doc["verification_id"] = result["id"]
		self._emit(tenant, "id_document_verified", document_id, "id_document", {
			"status": verification_status
		})
		return result

	async def digitize_form(
		self,
		tenant_id: str,
		document_id: str,
		template_id: str | None = None,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Digitize a form document: OCR -> template matching -> structured data."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		ocr = await self.run_ocr(tenant_id, document_id)
		extraction = await self.extract_fields(tenant_id, document_id, model)

		template = None
		if template_id:
			template = self.form_templates.get(f"{tenant}:{template_id}")

		result: dict[str, Any] = {
			"document_id": document_id,
			"pipeline": "form_digitization",
			"template_id": template_id,
			"template_matched": template is not None,
			"fields": extraction["extracted_fields"],
			"ocr_word_count": ocr["word_count"],
			"confidence": extraction["confidence"],
			"digitized_at": self._now(),
		}
		doc["status"] = "digitized"
		self._emit(tenant, "form_digitized", document_id, doc["document_type"])
		return result

	# ── Form templates ────────────────────────────────────────────

	async def create_form_template(
		self,
		tenant_id: str,
		name: str,
		fields: list[dict[str, Any]],
		description: str = "",
	) -> dict[str, Any]:
		"""Create a form template for structured extraction."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		template_id = self._id("tmpl")
		record: dict[str, Any] = {
			"id": template_id,
			"tenant_id": tenant,
			"name": name,
			"description": description,
			"fields": fields,
			"created_at": self._now(),
		}
		self.form_templates[f"{tenant}:{template_id}"] = record
		self._emit(tenant, "form_template_created", template_id, "form_template")
		return deepcopy(record)

	async def list_form_templates(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		return [deepcopy(r) for k, r in self.form_templates.items() if k.startswith(prefix)]

	async def get_form_template(self, tenant_id: str, template_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.form_templates.get(f"{tenant}:{template_id}")
		if not record:
			raise KeyError(f"template not found: {template_id}")
		return deepcopy(record)

	async def delete_form_template(self, tenant_id: str, template_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{template_id}"
		record = self.form_templates.get(key)
		if not record:
			raise KeyError(f"template not found: {template_id}")
		del self.form_templates[key]
		self._emit(tenant, "form_template_deleted", template_id, "form_template")
		return deepcopy(record)

	# ── Batch processing ──────────────────────────────────────────

	async def submit_batch(
		self,
		tenant_id: str,
		documents: list[dict[str, Any]],
		pipeline: str = "ocr_llm",
	) -> dict[str, Any]:
		"""Submit multiple documents for batch processing."""
		tenant = self._tenant(tenant_id)
		batch_id = self._id("batch")
		submitted = []
		errors = []
		for doc_input in documents:
			try:
				doc = await self.submit_document(
					tenant_id=tenant_id,
					document_type=doc_input.get("document_type", "other"),
					file_name=doc_input.get("file_name", "unnamed"),
					file_content_base64=doc_input.get("file_content_base64", ""),
					file_url=doc_input.get("file_url", "placeholder://batch"),
					mime_type=doc_input.get("mime_type", "application/pdf"),
					metadata=doc_input.get("metadata"),
					pipeline=pipeline,
				)
				submitted.append(doc["id"])
			except Exception as exc:
				_log.error("batch submit failed for %s: %s", doc_input.get("file_name"), exc)
				errors.append({"file_name": doc_input.get("file_name"), "error": str(exc)})

		batch: dict[str, Any] = {
			"id": batch_id,
			"tenant_id": tenant,
			"pipeline": pipeline,
			"total": len(documents),
			"submitted": len(submitted),
			"failed": len(errors),
			"document_ids": submitted,
			"errors": errors,
			"status": "submitted",
			"created_at": self._now(),
		}
		self.batches[batch_id] = batch
		self._emit(tenant, "batch_submitted", batch_id, "", {
			"count": len(submitted), "pipeline": pipeline
		})
		return deepcopy(batch)

	async def get_batch(self, tenant_id: str, batch_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		batch = self.batches.get(batch_id)
		if not batch or batch["tenant_id"] != tenant:
			raise KeyError(f"batch not found: {batch_id}")
		return deepcopy(batch)

	async def process_batch(self, tenant_id: str, batch_id: str, model: str = "ollama/llama3") -> dict[str, Any]:
		"""Run the full pipeline for all documents in a batch."""
		tenant = self._tenant(tenant_id)
		batch = self.batches.get(batch_id)
		if not batch or batch["tenant_id"] != tenant:
			raise KeyError(f"batch not found: {batch_id}")

		tasks = [self.extract_fields(tenant_id, did, model) for did in batch["document_ids"]]
		results_raw = await asyncio.gather(*tasks, return_exceptions=True)
		processed = []
		errors = []
		for did, result in zip(batch["document_ids"], results_raw):
			if isinstance(result, Exception):
				_log.error("batch process failed for doc=%s: %s", did, result)
				errors.append({"document_id": did, "error": str(result)})
			else:
				processed.append(did)

		batch["status"] = "completed"
		batch["processed"] = len(processed)
		batch["errors"] += errors
		batch["completed_at"] = self._now()
		self._emit(tenant, "batch_processed", batch_id, "", {
			"processed": len(processed), "failed": len(errors)
		})
		return deepcopy(batch)

	# ── Extraction retrieval ──────────────────────────────────────

	async def get_extraction(self, tenant_id: str, extraction_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		extraction = self.extractions.get(extraction_id)
		if not extraction or extraction["tenant_id"] != tenant:
			raise KeyError(f"extraction not found: {extraction_id}")
		return deepcopy(extraction)

	async def list_extractions(self, tenant_id: str, document_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.extractions.values() if e["tenant_id"] == tenant]
		if document_type:
			items = [e for e in items if e["document_type"] == document_type]
		return items

	async def get_document_with_extraction(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Return document record with its extraction result if available."""
		tenant = self._tenant(tenant_id)
		doc = await self.get_document(tenant_id, document_id)
		extraction_id = doc.get("extraction_id")
		extraction = None
		if extraction_id:
			try:
				extraction = await self.get_extraction(tenant_id, extraction_id)
			except KeyError as _exc:
				_log.debug("Handled exception: %s", _exc)
		return {"document": doc, "extraction": extraction}

	# ── Statistics ────────────────────────────────────────────────

	async def processing_statistics(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		docs = [d for d in self.documents.values() if d["tenant_id"] == tenant]
		extractions = [e for e in self.extractions.values() if e["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for d in docs:
			by_type[d["document_type"]] = by_type.get(d["document_type"], 0) + 1
			by_status[d["status"]] = by_status.get(d["status"], 0) + 1
		avg_conf = (sum(e["confidence"] for e in extractions) / len(extractions)) if extractions else 0.0
		return {
			"tenant_id": tenant,
			"total_documents": len(docs),
			"total_extractions": len(extractions),
			"by_document_type": by_type,
			"by_status": by_status,
			"avg_extraction_confidence": round(avg_conf, 4),
			"generated_at": self._now(),
		}

	# ── Document quality assessment ───────────────────────────────

	async def assess_document_quality(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Assess image/PDF quality before OCR — DPI, blur, skew, page coverage.

		Returns a quality_score [0.0–1.0] and actionable issue list. Documents
		scoring below 0.5 are flagged ``processable: false`` and should be
		re-uploaded rather than wasting OCR/LLM compute.

		Real implementation would pass the page image through a Laplacian blur
		detector and DPI inspector. This simulation uses document metadata to
		produce a plausible response.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		await asyncio.sleep(0.002)  # simulate image analysis I/O

		# Simulate quality heuristics based on mime type and metadata
		issues: list[str] = []
		quality_score = 1.0
		mime = doc.get("mime_type", "")
		if "jpeg" in mime or "jpg" in mime:
			# JPEG compression artifacts reduce quality slightly
			quality_score -= 0.05
		if doc.get("metadata", {}).get("scanned"):
			quality_score -= 0.1
			issues.append("scanned_image_detected")
		if doc.get("metadata", {}).get("low_res"):
			quality_score -= 0.35
			issues.append("low_resolution_below_150dpi")
		if doc.get("metadata", {}).get("skewed"):
			quality_score -= 0.2
			issues.append("page_skew_detected")
		quality_score = max(0.0, round(quality_score, 3))
		processable = quality_score >= 0.5

		assessment: dict[str, Any] = {
			"id": self._id("qa"),
			"tenant_id": tenant,
			"document_id": document_id,
			"quality_score": quality_score,
			"issues": issues,
			"processable": processable,
			"recommendations": [
				"Re-scan at minimum 300 DPI" if "low_resolution_below_150dpi" in issues else None,
				"Apply deskew pre-processing" if "page_skew_detected" in issues else None,
			],
			"assessed_at": self._now(),
		}
		assessment["recommendations"] = [r for r in assessment["recommendations"] if r]
		self._quality_assessments[document_id] = assessment
		if not processable:
			doc["status"] = "quality_failed"
		self._emit(tenant, "quality_assessed", document_id, doc["document_type"], {
			"quality_score": quality_score, "processable": processable
		})
		_log.info("quality assessment: doc=%s score=%.3f processable=%s", document_id, quality_score, processable)
		return deepcopy(assessment)

	# ── Multi-language detection ──────────────────────────────────

	async def detect_language(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Detect primary language and script of a document's extracted text.

		Uses character frequency heuristics in simulation. Real implementation
		calls ``langdetect`` or an Ollama multilingual model (e.g. ``aya:8b``).
		Detected language is stored on the document record and used to select
		the appropriate extraction model variant.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		await asyncio.sleep(0.003)
		ocr_text: str = doc.get("ocr_result", {}).get("ocr_text", "")

		# Heuristic simulation — real impl calls langdetect.detect()
		arabic_chars = sum(1 for c in ocr_text if "؀" <= c <= "ۿ")
		swahili_keywords = {"kwa", "na", "ya", "wa", "ni", "au"}
		swahili_hits = sum(1 for w in ocr_text.lower().split() if w in swahili_keywords)

		if arabic_chars > len(ocr_text) * 0.15:
			language, script, confidence = "ar", "Arabic", 0.91
		elif swahili_hits >= 2:
			language, script, confidence = "sw", "Latin", 0.84
		else:
			language, script, confidence = "en", "Latin", 0.97

		result: dict[str, Any] = {
			"document_id": document_id,
			"detected_language": language,
			"script_type": script,
			"confidence": confidence,
			"recommended_model": "ollama/aya:8b" if language != "en" else "ollama/llama3",
			"detected_at": self._now(),
		}
		doc["detected_language"] = language
		doc["recommended_model"] = result["recommended_model"]
		self._emit(tenant, "language_detected", document_id, doc["document_type"], {
			"language": language, "script": script
		})
		return result

	# ── Per-field confidence and field locking ────────────────────

	async def get_low_confidence_fields(
		self,
		tenant_id: str,
		document_id: str,
		threshold: float = 0.75,
	) -> dict[str, Any]:
		"""Return fields from the latest extraction whose confidence is below ``threshold``.

		These fields are candidates for human-review escalation or targeted
		re-extraction. The service uses uniform confidence in simulation; real
		implementations receive per-field scores from the LLM response logprobs.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		extraction_id = doc.get("extraction_id")
		if not extraction_id:
			raise ValueError(f"no extraction found for document: {document_id}")
		extraction = self.extractions.get(extraction_id)
		if not extraction:
			raise KeyError(f"extraction record missing: {extraction_id}")

		# Simulate per-field confidence — real impl reads LLM logprob output
		field_confidences: dict[str, float] = {}
		base_conf: float = extraction.get("confidence", 0.88)
		for i, key in enumerate(extraction.get("extracted_fields", {}).keys()):
			# Vary confidence slightly per field for realistic simulation
			jitter = (hash(key) % 20 - 10) / 100.0
			field_confidences[key] = round(max(0.0, min(1.0, base_conf + jitter)), 3)

		low_confidence = {k: v for k, v in field_confidences.items() if v < threshold}
		return {
			"document_id": document_id,
			"extraction_id": extraction_id,
			"threshold": threshold,
			"all_field_confidences": field_confidences,
			"low_confidence_fields": low_confidence,
			"review_recommended": len(low_confidence) > 0,
			"assessed_at": self._now(),
		}

	async def lock_fields(self, tenant_id: str, document_id: str, field_names: list[str]) -> dict[str, Any]:
		"""Lock specific extracted fields so they survive re-extraction unchanged.

		Locked fields are preserved verbatim in ``reextract_unlocked_fields``,
		allowing users to correct a vendor name or total without triggering a
		full re-extraction that would overwrite the correction.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if not field_names:
			raise ValueError("field_names must not be empty")
		existing = self._field_locks.get(document_id, set())
		existing.update(field_names)
		self._field_locks[document_id] = existing
		self._emit(tenant, "fields_locked", document_id, doc["document_type"], {"locked": field_names})
		return {
			"document_id": document_id,
			"locked_fields": sorted(existing),
			"newly_locked": field_names,
			"locked_at": self._now(),
		}

	async def reextract_unlocked_fields(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Re-run LLM extraction but preserve any fields locked via ``lock_fields``.

		This is the surgical update pathway: corrections made by human reviewers
		survive a pipeline re-run. Only unlocked fields are re-inferred from OCR text.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		locked = self._field_locks.get(document_id, set())

		# Preserve locked field values from previous extraction
		locked_values: dict[str, Any] = {}
		prev_extraction_id = doc.get("extraction_id")
		if prev_extraction_id and prev_extraction_id in self.extractions:
			prev_fields = self.extractions[prev_extraction_id].get("extracted_fields", {})
			locked_values = {k: v for k, v in prev_fields.items() if k in locked}

		# Re-extract fresh fields
		new_extraction = await self.extract_fields(tenant_id, document_id, model)
		new_fields = new_extraction["extracted_fields"]

		# Overlay locked values — locked fields win
		new_fields.update(locked_values)
		new_extraction["extracted_fields"] = new_fields
		new_extraction["locked_fields_preserved"] = sorted(locked)
		self.extractions[new_extraction["id"]] = new_extraction

		self._emit(tenant, "fields_reextracted", document_id, doc["document_type"], {
			"locked_count": len(locked), "reextracted_count": len(new_fields) - len(locked)
		})
		_log.info("reextraction completed: doc=%s locked=%d", document_id, len(locked))
		return deepcopy(new_extraction)

	# ── Contract clause classification ────────────────────────────

	async def classify_contract_clauses(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Segment contract text into clauses and classify each by type and risk score.

		Calls an Ollama model with a clause taxonomy prompt in real implementation.
		Returns a list of classified clause objects with per-clause risk scores.
		Aggregate risk level is derived from the highest-risk clause found.

		Clause types: indemnity, liability_cap, termination, ip_assignment,
		non_compete, confidentiality, force_majeure, governing_law, payment.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if doc["document_type"] != "contract":
			raise ValueError(f"clause classification requires a contract document, got: {doc['document_type']}")

		start = time.monotonic()
		await asyncio.sleep(0.012)  # simulate LLM inference

		ocr_text: str = doc.get("ocr_result", {}).get("ocr_text", "")
		# Simulate clause detection from OCR text paragraphs
		clauses: list[dict[str, Any]] = []
		paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=\n)(?=[A-Z\d])", ocr_text) if p.strip()]
		if not paragraphs:
			paragraphs = [ocr_text] if ocr_text else ["Standard service agreement terms apply."]

		# Assign simulated clause types round-robin with plausible risk scores
		clause_cycle = [
			("payment", 0.15), ("confidentiality", 0.35), ("indemnity", 0.72),
			("liability_cap", 0.55), ("termination", 0.45), ("governing_law", 0.1),
		]
		for idx, paragraph in enumerate(paragraphs[:6]):
			ctype, base_risk = clause_cycle[idx % len(clause_cycle)]
			clauses.append({
				"clause_index": idx,
				"clause_type": ctype,
				"text_excerpt": paragraph[:200],
				"risk_score": round(base_risk + (hash(paragraph[:20]) % 10) / 100.0, 3),
				"flags": ["high_risk_indemnity"] if ctype == "indemnity" and base_risk > 0.65 else [],
			})

		max_risk = max((c["risk_score"] for c in clauses), default=0.0)
		aggregate_risk = "high" if max_risk >= 0.7 else ("medium" if max_risk >= 0.4 else "low")
		processing_ms = round((time.monotonic() - start) * 1000, 2)

		result: dict[str, Any] = {
			"id": self._id("cls"),
			"tenant_id": tenant,
			"document_id": document_id,
			"clauses": clauses,
			"clause_count": len(clauses),
			"aggregate_risk": aggregate_risk,
			"max_clause_risk": round(max_risk, 3),
			"model": model,
			"processing_ms": processing_ms,
			"classified_at": self._now(),
		}
		self._clause_classifications[document_id] = clauses
		doc["clause_classification_id"] = result["id"]
		self._emit(tenant, "clauses_classified", document_id, "contract", {
			"clause_count": len(clauses), "aggregate_risk": aggregate_risk
		})
		_log.info("clause classification: doc=%s clauses=%d risk=%s", document_id, len(clauses), aggregate_risk)
		return result

	# ── Watchlist screening ───────────────────────────────────────

	async def screen_against_watchlists(
		self,
		tenant_id: str,
		document_id: str,
		lists: list[str] | None = None,
	) -> dict[str, Any]:
		"""Cross-reference extracted names/document numbers against KYC/AML watchlists.

		Checks OFAC, UN sanctions, CBK, and locally configured lists by default.
		In production, queries a PostgreSQL watchlist table or a remote compliance
		API endpoint. Names are normalised (stripped diacritics, case-folded,
		double-metaphone phonetic match) before comparison.

		Returns screen_status: ``clear`` | ``hit`` | ``review_required``.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		lists_to_check = lists or ["ofac", "un_sanctions", "cbk"]
		invalid = [lst for lst in lists_to_check if lst not in WATCHLIST_SOURCES]
		if invalid:
			raise ValueError(f"unsupported watchlist sources: {invalid}. Valid: {sorted(WATCHLIST_SOURCES)}")

		await asyncio.sleep(0.004)  # simulate remote API latency

		# Extract identity fields from the latest extraction
		extraction_id = doc.get("extraction_id")
		extracted_fields: dict[str, Any] = {}
		if extraction_id and extraction_id in self.extractions:
			extracted_fields = self.extractions[extraction_id].get("extracted_fields", {})

		full_name = str(extracted_fields.get("full_name", "") or "").upper().strip()
		doc_number = str(extracted_fields.get("document_number", "") or "").strip()

		# Simulate screening — in prod would query watchlist DB/API
		matched_entries: list[dict[str, Any]] = []
		# Flagged test names for simulation purposes only
		flagged_names = {"JOHN DOE SANCTIONS"}
		if full_name in flagged_names:
			matched_entries.append({
				"list": "ofac",
				"matched_name": full_name,
				"matched_id": doc_number,
				"match_type": "exact",
				"risk_level": "high",
			})

		screen_status = "hit" if matched_entries else "clear"

		result: dict[str, Any] = {
			"id": self._id("scr"),
			"tenant_id": tenant,
			"document_id": document_id,
			"lists_checked": lists_to_check,
			"screened_name": full_name,
			"screened_id_number": doc_number,
			"screen_status": screen_status,
			"matched_entries": matched_entries,
			"screened_at": self._now(),
		}
		self._watchlist_hits[document_id] = result
		if screen_status == "hit":
			doc["status"] = "watchlist_hit"
		self._emit(tenant, "watchlist_screened", document_id, doc["document_type"], {
			"status": screen_status, "lists": lists_to_check
		})
		_log.info("watchlist screen: doc=%s status=%s hits=%d", document_id, screen_status, len(matched_entries))
		return deepcopy(result)

	# ── Bank statement parsing ────────────────────────────────────

	async def parse_bank_statement(
		self,
		tenant_id: str,
		document_id: str,
		model: str = "ollama/llama3",
	) -> dict[str, Any]:
		"""Parse a bank statement into normalised debit/credit transaction records.

		Handles MPESA, RTGS, ATM, EFT, and cheque transaction narrative formats
		common in East African bank statements. Transactions are categorised
		(utilities, salary, merchant, transfer, fee) using regex + LLM fallback.

		Returns account_summary, transactions list, and cash_flow_metrics.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if doc["document_type"] != "bank_statement":
			raise ValueError(f"bank statement parsing requires document_type=bank_statement, got: {doc['document_type']}")

		start = time.monotonic()
		await asyncio.sleep(0.015)  # simulate multi-page LLM extraction

		# Simulate a normalised transaction list
		transactions: list[dict[str, Any]] = [
			{
				"seq": 1, "date": "2026-05-01", "value_date": "2026-05-01",
				"description": "SALARY CREDIT - DATACRAFT LTD",
				"debit": None, "credit": 250000.00, "balance": 312450.00,
				"category": "salary", "channel": "eft", "ref": "EFT001",
			},
			{
				"seq": 2, "date": "2026-05-03", "value_date": "2026-05-03",
				"description": "MPESA-B2C-SAFARICOM 254700000000",
				"debit": 15000.00, "credit": None, "balance": 297450.00,
				"category": "transfer", "channel": "mpesa", "ref": "MP20260503",
			},
			{
				"seq": 3, "date": "2026-05-07", "value_date": "2026-05-07",
				"description": "ATM WITHDRAWAL WESTLANDS KE",
				"debit": 20000.00, "credit": None, "balance": 277450.00,
				"category": "cash", "channel": "atm", "ref": "ATM5503",
			},
			{
				"seq": 4, "date": "2026-05-15", "value_date": "2026-05-15",
				"description": "KPLC PREPAID TOKEN PAYBILL 888880",
				"debit": 5000.00, "credit": None, "balance": 272450.00,
				"category": "utilities", "channel": "mpesa_paybill", "ref": "PB8880",
			},
		]
		total_credits = sum(t["credit"] for t in transactions if t["credit"])
		total_debits = sum(t["debit"] for t in transactions if t["debit"])
		closing_balance = transactions[-1]["balance"] if transactions else 0.0
		processing_ms = round((time.monotonic() - start) * 1000, 2)

		result: dict[str, Any] = {
			"id": self._id("bsp"),
			"tenant_id": tenant,
			"document_id": document_id,
			"account_summary": {
				"opening_balance": transactions[0]["balance"] - (transactions[0]["credit"] or 0) if transactions else 0.0,
				"closing_balance": closing_balance,
				"total_credits": total_credits,
				"total_debits": total_debits,
				"net_flow": total_credits - total_debits,
				"currency": "KES",
			},
			"transactions": transactions,
			"transaction_count": len(transactions),
			"cash_flow_metrics": {
				"avg_monthly_credit": total_credits,
				"avg_monthly_debit": total_debits,
				"credit_debit_ratio": round(total_credits / total_debits, 3) if total_debits else None,
				"mpesa_transaction_count": sum(1 for t in transactions if t["channel"] == "mpesa"),
			},
			"model": model,
			"processing_ms": processing_ms,
			"parsed_at": self._now(),
		}
		doc["bank_statement_parse_id"] = result["id"]
		doc["status"] = "parsed"
		self._emit(tenant, "bank_statement_parsed", document_id, "bank_statement", {
			"transaction_count": len(transactions), "net_flow": result["account_summary"]["net_flow"]
		})
		_log.info("bank statement parsed: doc=%s txns=%d", document_id, len(transactions))
		return result

	# ── Semantic search via embeddings ────────────────────────────

	async def embed_document(
		self,
		tenant_id: str,
		document_id: str,
		embedding_model: str = "ollama/nomic-embed-text",
	) -> dict[str, Any]:
		"""Generate a text embedding for the document's extracted content.

		Real implementation calls ``ollama.embeddings(model=embedding_model,
		prompt=ocr_text)`` and stores the vector in a PostgreSQL ``pgvector``
		column for sub-second semantic retrieval across millions of documents.

		Returns a 768-dim embedding vector (simulated here as a short representative
		vector) and stores it internally keyed by document_id.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")

		await asyncio.sleep(0.006)  # simulate embedding inference

		ocr_text: str = doc.get("ocr_result", {}).get("ocr_text", "")
		extraction_id = doc.get("extraction_id")
		extracted_text = ""
		if extraction_id and extraction_id in self.extractions:
			fields = self.extractions[extraction_id].get("extracted_fields", {})
			extracted_text = " ".join(str(v) for v in fields.values() if isinstance(v, (str, int, float)))

		content = f"{ocr_text} {extracted_text}".strip()

		# Simulate a 16-dim stub vector using hash of content (prod uses 768-dim)
		seed = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
		vector: list[float] = [
			round((((seed >> i) & 0xFF) / 255.0) * 2 - 1, 4) for i in range(16)
		]

		self._embeddings[document_id] = vector
		result: dict[str, Any] = {
			"document_id": document_id,
			"embedding_model": embedding_model,
			"vector_dimensions": len(vector),
			"embedding": vector,
			"content_length": len(content),
			"embedded_at": self._now(),
		}
		self._emit(tenant, "document_embedded", document_id, doc["document_type"], {
			"model": embedding_model, "dims": len(vector)
		})
		return result

	async def semantic_search(
		self,
		tenant_id: str,
		query: str,
		top_k: int = 5,
		document_type_filter: str | None = None,
		embedding_model: str = "ollama/nomic-embed-text",
	) -> dict[str, Any]:
		"""Search across all embedded documents using cosine similarity.

		Encodes ``query`` with the same embedding model used during indexing,
		then returns the ``top_k`` closest documents by cosine distance.
		In production, this executes a pgvector ``<->`` nearest-neighbour query.

		Example queries:
		- "contracts with high indemnity risk"
		- "invoices from Acme Corp above KES 100000"
		- "expired ID documents"
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")
		if top_k < 1 or top_k > 100:
			raise ValueError("top_k must be between 1 and 100")

		await asyncio.sleep(0.005)

		# Build query vector stub (same deterministic method as embed_document)
		seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
		query_vector: list[float] = [
			round((((seed >> i) & 0xFF) / 255.0) * 2 - 1, 4) for i in range(16)
		]

		def _cosine_sim(a: list[float], b: list[float]) -> float:
			dot = sum(x * y for x, y in zip(a, b))
			mag_a = sum(x ** 2 for x in a) ** 0.5
			mag_b = sum(x ** 2 for x in b) ** 0.5
			return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

		# Score all embedded documents belonging to this tenant
		scored: list[dict[str, Any]] = []
		for did, vec in self._embeddings.items():
			doc = self.documents.get(did)
			if not doc or doc["tenant_id"] != tenant:
				continue
			if document_type_filter and doc["document_type"] != document_type_filter:
				continue
			score = _cosine_sim(query_vector, vec)
			scored.append({
				"document_id": did,
				"document_type": doc["document_type"],
				"file_name": doc["file_name"],
				"status": doc["status"],
				"similarity": round(score, 4),
			})

		scored.sort(key=lambda x: x["similarity"], reverse=True)
		hits = scored[:top_k]
		return {
			"query": query,
			"embedding_model": embedding_model,
			"document_type_filter": document_type_filter,
			"top_k": top_k,
			"total_indexed": len(self._embeddings),
			"results": hits,
			"searched_at": self._now(),
		}

	# ── Human-in-the-loop review queue ───────────────────────────

	async def flag_for_review(
		self,
		tenant_id: str,
		document_id: str,
		field_names: list[str],
		reason: str = "",
	) -> dict[str, Any]:
		"""Escalate specific low-confidence or suspicious fields for human review.

		Creates a review ticket in the review queue. In production, publishes a
		``docint.review_required`` event to NATS JetStream so a review worker or
		webhook can notify the appropriate reviewer. The review remains open until
		``resolve_review`` is called with corrected values.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document not found: {document_id}")
		if not field_names:
			raise ValueError("field_names must not be empty")

		review_id = self._id("rev")
		ticket: dict[str, Any] = {
			"id": review_id,
			"tenant_id": tenant,
			"document_id": document_id,
			"document_type": doc["document_type"],
			"field_names": field_names,
			"reason": reason or "low_confidence",
			"status": "pending",
			"created_at": self._now(),
			"resolved_at": None,
			"corrections": {},
		}
		self._review_queue[review_id] = ticket
		doc["review_id"] = review_id
		doc["status"] = "review_pending"
		self._emit(tenant, "review_flagged", document_id, doc["document_type"], {
			"review_id": review_id, "fields": field_names, "reason": reason
		})
		_log.info("review flagged: doc=%s review=%s fields=%s", document_id, review_id, field_names)
		return deepcopy(ticket)

	async def resolve_review(
		self,
		tenant_id: str,
		review_id: str,
		corrections: dict[str, Any],
		reviewer_note: str = "",
	) -> dict[str, Any]:
		"""Apply human corrections to a flagged review ticket and close it.

		Corrections are merged into the document's extracted fields, overriding
		LLM-inferred values. The corrected fields are automatically locked via
		``lock_fields`` to prevent re-extraction from discarding the corrections.

		Publishes ``docint.review_resolved`` to NATS for downstream audit.
		"""
		tenant = self._tenant(tenant_id)
		ticket = self._review_queue.get(review_id)
		if not ticket or ticket["tenant_id"] != tenant:
			raise KeyError(f"review not found: {review_id}")
		if ticket["status"] != "pending":
			raise ValueError(f"review already resolved: {ticket['status']}")

		document_id: str = ticket["document_id"]
		doc = self.documents.get(document_id)
		if not doc:
			raise KeyError(f"document missing for review: {document_id}")

		# Apply corrections to the latest extraction
		extraction_id = doc.get("extraction_id")
		if extraction_id and extraction_id in self.extractions:
			extraction = self.extractions[extraction_id]
			extraction["extracted_fields"].update(corrections)

		# Lock corrected fields to protect them from re-extraction
		if corrections:
			await self.lock_fields(tenant_id, document_id, list(corrections.keys()))

		ticket["corrections"] = corrections
		ticket["reviewer_note"] = reviewer_note
		ticket["status"] = "resolved"
		ticket["resolved_at"] = self._now()
		doc["status"] = "reviewed"

		self._emit(tenant, "review_resolved", document_id, doc["document_type"], {
			"review_id": review_id, "corrections": list(corrections.keys())
		})
		_log.info("review resolved: doc=%s review=%s corrections=%d", document_id, review_id, len(corrections))
		return deepcopy(ticket)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_review_queue', '_watchlist_hits', '_quality_assessments', '_signature_detections', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

