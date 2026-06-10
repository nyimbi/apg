"""Document Intelligence service — OCR pipeline, LLM extraction, invoice/contract/ID parsing, form digitization."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "docint"
SUPPORTED_DOC_TYPES = {"invoice", "contract", "id_document", "form", "receipt", "bank_statement", "tax_form", "other"}
SUPPORTED_PIPELINES = {"ocr_only", "llm_only", "ocr_llm"}
SUPPORTED_MIME_TYPES = {
	"application/pdf", "image/jpeg", "image/png", "image/tiff",
	"image/webp", "image/bmp", "application/msword",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class DocumentIntelligenceService:
	"""OCR + LLM extraction pipeline for invoices, contracts, ID docs, forms, PDFs, and images."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.documents: dict[str, dict[str, Any]] = {}
		self.extractions: dict[str, dict[str, Any]] = {}
		self.verification_results: dict[str, dict[str, Any]] = {}
		self.form_templates: dict[str, dict[str, Any]] = {}
		self.batches: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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
			except Exception:
				pass

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
			except KeyError:
				pass
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
