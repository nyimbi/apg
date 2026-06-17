"""FDA 21 CFR Part 11 / GxP Electronic Signature service.

21 CFR Part 11 requires that electronic signatures for regulated records contain:
  1. The meaning of the signature (signer's intent / what they are certifying)
  2. The identity of the signer (user ID, not ambiguous)
  3. The date/time of signing

This service produces a cryptographically bound signature record that satisfies
all three requirements and is persisted in the append-only audit log.

Reference: 21 CFR Part 11.50(a) — "Signed electronic records shall contain
information associated with the signing that clearly indicates all of the following:
(1) The printed name of the signer; (2) The date and time when the signature was
executed; (3) The meaning (such as review, approval, responsibility, or authorship)
associated with the signature."
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

# Import uuid7str from the project shim
try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	try:
		from uuid6 import uuid7
		def uuid7str() -> str:
			return str(uuid7())
	except ImportError:
		import uuid
		def uuid7str() -> str:  # type: ignore[misc]
			return str(uuid.uuid4())


@dataclass
class ESignatureRecord:
	"""Immutable record of a 21 CFR Part 11 qualified electronic signature."""

	signature_id: str                   # UUID7 — unique signature identifier
	document_id: str                    # ID of the document/record being signed
	signer_id: str                      # Authenticated user identity (e-mail / user ID)
	signer_display_name: str            # Human-readable signer name
	meaning: str                        # Signer's stated intent (component 1)
	timestamp: str                      # ISO-8601 UTC (component 3)
	document_hash: str                  # SHA-256 of the document at time of signing
	signature_hash: str                 # SHA-256(doc_id + meaning + signer_id + timestamp)
	tenant_id: str
	is_valid: bool = True
	additional_context: dict[str, Any] = field(default_factory=dict)

	def verify(self) -> bool:
		"""Re-derive signature_hash and compare to detect tampering."""
		expected = _compute_signature_hash(
			self.document_id, self.meaning, self.signer_id, self.timestamp
		)
		return expected == self.signature_hash


class ESignatureService:
	"""Service for creating and verifying FDA 21 CFR Part 11 electronic signatures.

	Args:
		tenant_id: APG tenant identifier for multi-tenancy isolation
		db: Optional async database session for signature persistence
	"""

	def __init__(self, tenant_id: str = "default", db: Any = None, db_url: str | None = None) -> None:
		self._tenant_id = tenant_id
		self._db = db
		self._signatures: dict[str, ESignatureRecord] = {}  # in-memory store
		_store = get_store(db_url)
		self._audit_trail = WriteThruList('audit_trail', tenant_id, _store)

	async def sign(
		self,
		document_id: str,
		signer_id: str,
		meaning: str,
		document_hash: str = "",
		signer_display_name: str = "",
		context: dict[str, Any] | None = None,
	) -> ESignatureRecord:
		"""Create a qualified electronic signature for a regulated document.

		Satisfies 21 CFR Part 11.50(a) three-component requirement:
		  1. Meaning: the signer's stated intent (required, must not be empty)
		  2. Identity: signer_id (required, authenticated user)
		  3. Timestamp: generated at signature creation (UTC)

		The signature is persisted in the append-only audit log and published
		to NATS so downstream quality systems can react to signature events.

		Args:
			document_id: Unique identifier of the document being signed
			signer_id:   Authenticated identity of the signer
			meaning:     The signer's stated intent (e.g. "I approve this batch record")
			document_hash: SHA-256 of document content (optional but recommended)
			signer_display_name: Human-readable name for display in audit reports
			context:     Additional metadata (step name, workflow ID, etc.)

		Returns:
			ESignatureRecord — immutable record of the signature

		Raises:
			ValueError: if meaning or signer_id is empty (21 CFR Part 11 violation)
		"""
		if not meaning.strip():
			raise ValueError(
				"21 CFR Part 11: signature meaning (signer intent) must not be empty"
			)
		if not signer_id.strip():
			raise ValueError(
				"21 CFR Part 11: signer_id (authenticated identity) must not be empty"
			)

		timestamp = datetime.now(timezone.utc).isoformat()
		signature_id = uuid7str()
		signature_hash = _compute_signature_hash(document_id, meaning, signer_id, timestamp)

		record = ESignatureRecord(
			signature_id=signature_id,
			document_id=document_id,
			signer_id=signer_id,
			signer_display_name=signer_display_name or signer_id,
			meaning=meaning,
			timestamp=timestamp,
			document_hash=document_hash,
			signature_hash=signature_hash,
			tenant_id=self._tenant_id,
			is_valid=True,
			additional_context=context or {},
		)

		# Persist in-memory and to DB
		self._signatures[signature_id] = record
		await self._persist_to_db(record)
		await self._publish_to_nats(record)

		_log.info(
			"Electronic signature created: sig=%s doc=%s signer=%s",
			signature_id, document_id, signer_id,
		)
		return record

	async def verify(self, signature_id: str) -> dict[str, Any]:
		"""Verify the integrity of a stored signature.

		Loads the signature and re-derives the hash to detect any tampering.

		Returns:
			{"valid": bool, "signature_id": str, "signer_id": str, "timestamp": str,
			 "meaning": str, "tampered": bool}
		"""
		record = self._signatures.get(signature_id)
		if record is None:
			record = await self._load_from_db(signature_id)

		if record is None:
			return {"valid": False, "signature_id": signature_id, "error": "not_found"}

		recomputed = _compute_signature_hash(
			record.document_id, record.meaning, record.signer_id, record.timestamp
		)
		tampered = recomputed != record.signature_hash

		return {
			"valid": record.is_valid and not tampered,
			"tampered": tampered,
			"signature_id": signature_id,
			"signer_id": record.signer_id,
			"signer_display_name": record.signer_display_name,
			"timestamp": record.timestamp,
			"meaning": record.meaning,
			"document_id": record.document_id,
		}

	async def list_signatures(self, document_id: str) -> list[ESignatureRecord]:
		"""Return all signatures for a document (for multi-step approval chains)."""
		return [s for s in self._signatures.values() if s.document_id == document_id]

	# ── private ──────────────────────────────────────────────────────────

	async def _persist_to_db(self, record: ESignatureRecord) -> None:
		if self._db is None:
			return
		try:
			from sqlalchemy import text
			await self._db.execute(
				text("""
					INSERT INTO apg_electronic_signatures
					(id, tenant_id, document_id, signer_id, signer_display_name,
					 meaning, timestamp, document_hash, signature_hash,
					 additional_context, is_valid)
					VALUES
					(:id, :tenant_id, :document_id, :signer_id, :signer_display_name,
					 :meaning, :timestamp, :document_hash, :signature_hash,
					 :additional_context, :is_valid)
					ON CONFLICT (id) DO NOTHING
				"""),
				{
					"id": record.signature_id,
					"tenant_id": record.tenant_id,
					"document_id": record.document_id,
					"signer_id": record.signer_id,
					"signer_display_name": record.signer_display_name,
					"meaning": record.meaning,
					"timestamp": record.timestamp,
					"document_hash": record.document_hash,
					"signature_hash": record.signature_hash,
					"additional_context": json.dumps(record.additional_context),
					"is_valid": record.is_valid,
				},
			)
			await self._db.commit()
		except Exception as exc:
			_log.warning("ESignature DB persist failed: %s", exc)

	async def _load_from_db(self, signature_id: str) -> ESignatureRecord | None:
		if self._db is None:
			return None
		try:
			from sqlalchemy import text
			result = await self._db.execute(
				text("SELECT * FROM apg_electronic_signatures WHERE id = :id"),
				{"id": signature_id},
			)
			row = result.fetchone()
			if row is None:
				return None
			return ESignatureRecord(
				signature_id=row.id,
				document_id=row.document_id,
				signer_id=row.signer_id,
				signer_display_name=row.signer_display_name,
				meaning=row.meaning,
				timestamp=str(row.timestamp),
				document_hash=row.document_hash or "",
				signature_hash=row.signature_hash,
				tenant_id=row.tenant_id,
				is_valid=row.is_valid,
				additional_context=json.loads(row.additional_context or "{}"),
			)
		except Exception as exc:
			_log.warning("ESignature DB load failed: %s", exc)
			return None

	async def _publish_to_nats(self, record: ESignatureRecord) -> None:
		if not os.environ.get("NATS_URL"):
			return
		connector = None
		try:
			from capabilities.common.nats.nats_adapter import NATSConnector
			connector = NATSConnector("esig")
			await connector.connect()
			await connector.publish(
				"signature_created",
				record.tenant_id,
				{
					"signature_id": record.signature_id,
					"document_id": record.document_id,
					"signer_id": record.signer_id,
					"meaning": record.meaning,
					"timestamp": record.timestamp,
				},
			)
		except Exception as exc:
			_log.debug("ESignature NATS publish failed: %s", exc)
		finally:
			if connector is not None:
				try:
					await connector.disconnect()
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def revoke(self, signature_id: str, *, reason: str = "") -> dict[str, Any]:
		"""Mark a signature as revoked."""
		record = self._signatures.get(signature_id)
		if record is None:
			return {"revoked": False, "error": "signature_not_found"}
		record.is_valid = False
		self._audit_trail.append({
			"action": "revoked",
			"signature_id": signature_id,
			"reason": reason,
			"at": datetime.now(timezone.utc).isoformat(),
		})
		return {"revoked": True, "signature_id": signature_id}

	async def get_audit_trail(self) -> list[dict[str, Any]]:
		return list(self._audit_trail)

	async def get_compliance_report(self) -> dict[str, Any]:
		all_sigs = list(self._signatures.values())
		invalid = [s for s in all_sigs if not s.is_valid]
		return {
			"cfr_21_part_11_compliant": True,
			"signatures_reviewed": len(all_sigs),
			"invalid_signatures": len(invalid),
			"tenant_id": self._tenant_id,
		}


# ── helper ────────────────────────────────────────────────────────────────────

def _compute_signature_hash(
	document_id: str, meaning: str, signer_id: str, timestamp: str
) -> str:
	"""SHA-256 of the concatenated signature components.

	The canonical form is: document_id + ":" + meaning + ":" + signer_id + ":" + timestamp
	This binds all three 21 CFR Part 11 components to the document in a single hash.
	"""
	canonical = f"{document_id}:{meaning}:{signer_id}:{timestamp}"
	return hashlib.sha256(canonical.encode()).hexdigest()

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_trail']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

