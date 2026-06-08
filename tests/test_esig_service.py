"""Tests for FDA 21 CFR Part 11 electronic signature service."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from capabilities.common.esig import ESignatureService, ESignatureRecord
from capabilities.common.esig.service import _compute_signature_hash


def make_svc(db=None):
	return ESignatureService(tenant_id="pharma-test", db=db)


# ── Signature creation ────────────────────────────────────────────────────────

async def test_sign_returns_record_with_all_fields():
	svc = make_svc()
	record = await svc.sign(
		document_id="BATCH-2025-001",
		signer_id="analyst@lab.com",
		meaning="I certify that all batch records are complete and accurate",
	)
	assert isinstance(record, ESignatureRecord)
	assert record.document_id == "BATCH-2025-001"
	assert record.signer_id == "analyst@lab.com"
	assert record.meaning == "I certify that all batch records are complete and accurate"
	assert record.signature_hash is not None and len(record.signature_hash) == 64
	assert record.is_valid is True
	assert record.tenant_id == "pharma-test"


async def test_sign_records_timestamp():
	svc = make_svc()
	record = await svc.sign("DOC-001", "user@lab.com", "I approve this document")
	assert record.timestamp is not None
	assert "T" in record.timestamp  # ISO-8601 format


async def test_sign_requires_non_empty_meaning():
	svc = make_svc()
	with pytest.raises(ValueError, match="meaning"):
		await svc.sign("DOC-001", "user@lab.com", "")


async def test_sign_requires_non_empty_signer_id():
	svc = make_svc()
	with pytest.raises(ValueError, match="signer_id"):
		await svc.sign("DOC-001", "", "I approve")


async def test_sign_stores_additional_context():
	svc = make_svc()
	record = await svc.sign(
		"DOC-001", "user@lab.com", "I approve",
		context={"workflow_step": "qc_review", "batch": "BATCH-001"}
	)
	assert record.additional_context["workflow_step"] == "qc_review"


# ── 21 CFR Part 11 three-component verification ──────────────────────────────

def test_signature_hash_binds_all_three_components():
	"""The hash must change if ANY of the three 21 CFR Part 11 components changes."""
	original = _compute_signature_hash("DOC-001", "I approve", "user@lab.com", "2025-06-01T10:00:00+00:00")

	# Change meaning
	assert _compute_signature_hash("DOC-001", "I REJECT", "user@lab.com", "2025-06-01T10:00:00+00:00") != original
	# Change signer
	assert _compute_signature_hash("DOC-001", "I approve", "other@lab.com", "2025-06-01T10:00:00+00:00") != original
	# Change timestamp
	assert _compute_signature_hash("DOC-001", "I approve", "user@lab.com", "2025-06-01T11:00:00+00:00") != original


# ── Signature verification ────────────────────────────────────────────────────

async def test_verify_returns_valid_for_good_signature():
	svc = make_svc()
	record = await svc.sign("DOC-001", "user@lab.com", "I certify")
	result = await svc.verify(record.signature_id)
	assert result["valid"] is True
	assert result["tampered"] is False
	assert result["signer_id"] == "user@lab.com"


async def test_verify_detects_tampered_signature():
	svc = make_svc()
	record = await svc.sign("DOC-001", "user@lab.com", "I certify")
	# Tamper with the stored signature hash
	svc._signatures[record.signature_id].signature_hash = "a" * 64
	result = await svc.verify(record.signature_id)
	assert result["valid"] is False
	assert result["tampered"] is True


async def test_verify_not_found_returns_error():
	svc = make_svc()
	result = await svc.verify("nonexistent-sig-id")
	assert result["valid"] is False
	assert "error" in result


# ── Record method ────────────────────────────────────────────────────────────

async def test_record_verify_method_is_consistent():
	svc = make_svc()
	record = await svc.sign("DOC-001", "user@lab.com", "I approve")
	assert record.verify() is True


# ── list_signatures ───────────────────────────────────────────────────────────

async def test_list_signatures_returns_all_for_document():
	svc = make_svc()
	await svc.sign("DOC-001", "user1@lab.com", "I review")
	await svc.sign("DOC-001", "user2@lab.com", "I approve")
	await svc.sign("DOC-002", "user1@lab.com", "I certify")

	sigs_doc1 = await svc.list_signatures("DOC-001")
	sigs_doc2 = await svc.list_signatures("DOC-002")

	assert len(sigs_doc1) == 2
	assert len(sigs_doc2) == 1
	assert all(s.document_id == "DOC-001" for s in sigs_doc1)


# ── DB persistence ────────────────────────────────────────────────────────────

async def test_db_persist_called_when_db_provided():
	mock_db = AsyncMock()
	mock_db.execute = AsyncMock(return_value=AsyncMock())
	mock_db.commit = AsyncMock()

	svc = make_svc(db=mock_db)
	await svc.sign("DOC-001", "user@lab.com", "I certify")

	mock_db.execute.assert_called_once()
	sql = str(mock_db.execute.call_args[0][0])
	assert "apg_electronic_signatures" in sql
	assert "INSERT" in sql


async def test_db_persist_skipped_when_no_db():
	svc = make_svc(db=None)
	# No DB provided — should not raise
	record = await svc.sign("DOC-001", "user@lab.com", "I approve")
	assert record.is_valid is True


# ── SQL migration file ────────────────────────────────────────────────────────

def test_esig_migration_file_exists():
	from pathlib import Path
	sql = Path("capabilities/common/esig/0001_electronic_signatures.sql")
	assert sql.exists()
	content = sql.read_text()
	assert "apg_electronic_signatures" in content
	assert "meaning" in content         # 21 CFR Part 11 component 1
	assert "timestamp" in content       # component 3
	assert "signature_hash" in content
	assert "DO INSTEAD NOTHING" in content  # append-only rule
