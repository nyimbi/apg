"""Deterministic ledger hashing helpers for APG BCLG."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class LedgerEngine:
	"""Build deterministic transaction, contract, and block hashes."""

	def digest(self, payload: dict[str, Any]) -> str:
		canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
		return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

	def transaction_hash(self, payload: dict[str, Any]) -> str:
		return self.digest({"kind": "bclg.transaction", "payload": payload})

	def contract_deployment_hash(self, payload: dict[str, Any]) -> str:
		return self.digest({"kind": "bclg.contract", "payload": payload})

	def block_hash(
		self,
		ledger_id: str,
		transaction_hashes: list[str],
		previous_hash: str | None,
	) -> str:
		return self.digest({
			"kind": "bclg.block",
			"ledger_id": ledger_id,
			"previous_hash": previous_hash or "genesis",
			"transactions": list(transaction_hashes),
		})
