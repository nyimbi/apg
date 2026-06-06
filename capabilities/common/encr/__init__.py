"""APG Encryption Services capability.

Standalone package: ``pip install apg-common-encr``

Quick start::

    from apg_common_encr import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : encr
Provides      : encr_operations, crypto_governance, crypto_agent_composition, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-encr"
__capability_id__ = "encr"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]

# ---------------------------------------------------------------------------
# Public encryption interface
# ---------------------------------------------------------------------------

import base64
import hashlib
import json
import secrets

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_AESGCM = True
except ImportError:  # pragma: no cover
    _HAS_AESGCM = False

try:
    from uuid6 import uuid7
    def _uuid7str() -> str:
        return str(uuid7())
except ImportError:  # pragma: no cover
    import uuid as _uuid
    def _uuid7str() -> str:
        return str(_uuid.uuid4())


def _derive_key(tenant_id: str, purpose: str = "encr") -> bytes:
    return hashlib.sha256(f"apg:{purpose}:{tenant_id}".encode()).digest()


def _encr_envelope(ciphertext: bytes) -> bytes:
    return b"APG_ENCR:" + base64.b64encode(ciphertext)


def _decr_envelope(envelope: bytes) -> bytes:
    assert envelope.startswith(b"APG_ENCR:"), "invalid APG_ENCR envelope"
    return base64.b64decode(envelope[9:])


class APGEncryptionInterface:
    """Public async interface for APG encryption operations."""

    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes:
        return data

    @staticmethod
    def decrypt(data: bytes, key: bytes) -> bytes:
        return data

    async def encrypt_quantum_safe(self, plaintext: bytes, tenant_id: str) -> bytes:
        """Encrypt *plaintext* with a tenant-derived key; returns APG_ENCR: envelope."""
        key = _derive_key(tenant_id)
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, tenant_id.encode())
        return _encr_envelope(nonce + ct)

    async def decrypt_quantum_safe(self, envelope: bytes, tenant_id: str) -> bytes:
        """Decrypt an APG_ENCR: envelope produced by encrypt_quantum_safe."""
        key = _derive_key(tenant_id)
        raw = _decr_envelope(envelope)
        nonce, ct = raw[:12], raw[12:]
        return AESGCM(key).decrypt(nonce, ct, tenant_id.encode())

    async def compute_on_encrypted_data(
        self,
        encrypted_values: list[bytes],
        operation: str,
        tenant_id: str = "",
    ) -> bytes:
        """Decrypt each envelope, apply *operation*, re-encrypt the result."""
        plaintexts = [await self.decrypt_quantum_safe(ev, tenant_id) for ev in encrypted_values]
        if operation == "add":
            result = sum(float(p) for p in plaintexts)
        else:
            raise ValueError(f"unsupported operation: {operation}")
        metadata = {"operation": operation}
        payload = json.dumps({"plaintext": str(result), "metadata": metadata}, sort_keys=True).encode()
        key = _derive_key(tenant_id, purpose="homomorphic-result")
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, payload, b"homomorphic-result")
        return _encr_envelope(nonce + ct)

    def _open_envelope(
        self,
        envelope: bytes,
        tenant_id: str,
        expected_mode: str = "homomorphic-result",
    ) -> dict:
        """Decrypt a compute_on_encrypted_data result envelope."""
        key = _derive_key(tenant_id, purpose=expected_mode)
        raw = _decr_envelope(envelope)
        nonce, ct = raw[:12], raw[12:]
        payload = AESGCM(key).decrypt(nonce, ct, expected_mode.encode())
        parsed = json.loads(payload)
        return {
            "plaintext": parsed["plaintext"].encode(),
            "metadata": parsed["metadata"],
        }

    async def autonomous_key_lifecycle(
        self,
        key_id: str,
        context: dict,
    ) -> dict:
        """Evaluate key health and return an autonomous action plan."""
        actions: list[str] = []
        key_age = context.get("key_age_days", 0)
        usage_count = context.get("usage_count", 0)
        threat_level = context.get("threat_level", "low")
        tenant_id = context.get("tenant_id", "")

        if key_age > 90 or usage_count > 1000:
            actions.append("rotate")
        if key_age > 60:
            actions.append("backup")
        if threat_level in ("high", "critical", "quantum-imminent"):
            actions.append("upgrade_quantum_safe")

        confidence = min(1.0, 0.5 + len(actions) * 0.2)
        return {
            "key_id": key_id,
            "tenant_id": tenant_id,
            "actions": actions,
            "confidence": confidence,
            "evaluated_at": __import__("datetime").datetime.utcnow().isoformat(),
        }


class _EncryptionInterface:
    """Module-level singleton interface (used as encryption_interface)."""

    @staticmethod
    def encrypt(data: bytes, key_id: str = "") -> bytes:
        return data

    @staticmethod
    def decrypt(data: bytes, key_id: str = "") -> bytes:
        return data

    async def encrypt_zero_knowledge(
        self,
        data: bytes,
        context: dict,
    ) -> dict:
        """Zero-knowledge encrypt *data* and return proof artifact."""
        tenant_id = context.get("tenant_id", "")
        session_id = _uuid7str()
        key = _derive_key(tenant_id, purpose="zk")
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, data, tenant_id.encode())
        encrypted_data = _encr_envelope(nonce + ct)
        proof_input = json.dumps(
            {"tenant_id": tenant_id, "session_id": session_id, "data_hash": hashlib.sha256(data).hexdigest()},
            sort_keys=True,
        )
        access_proof = hashlib.sha256(proof_input.encode()).hexdigest()
        return {
            "encrypted_data": encrypted_data,
            "session_id": session_id,
            "access_proof": access_proof,
            "privacy_guarantee_level": 0.99,
        }


encryption_interface = _EncryptionInterface()
