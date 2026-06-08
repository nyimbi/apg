"""APG Electronic Signature — FDA 21 CFR Part 11 / GxP compliance.

Implements qualified electronic signatures meeting FDA 21 CFR Part 11 requirements:
  - Three required components: meaning (signer intent), signer identity, timestamp
  - Cryptographic binding: SHA-256 hash of (document_id + meaning + signer_id + timestamp)
  - Audit trail integration: every signature emitted to NATS and persisted in audl
  - OPA authorization: signing requires 'gxp_authorized' role (via pharma.rego)

Usage::

    from capabilities.common.esig import ESignatureService

    esig = ESignatureService(tenant_id="pharma-co", db=db_session)

    record = await esig.sign(
        document_id="BATCH-2025-001",
        signer_id="analyst@lab.com",
        meaning="I certify that all batch records are complete and accurate",
        document_hash="sha256:...",  # hash of the document being signed
    )
    print(record.signature_id)  # UUID7
    print(record.is_valid)      # True

    # Verify later
    verified = await esig.verify(record.signature_id)
    print(verified.valid)
"""
from .service import ESignatureService, ESignatureRecord

__all__ = ["ESignatureService", "ESignatureRecord"]
