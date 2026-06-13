"""Async service layer for APG Product Lifecycle Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
	from .models import MfPlmProduct, MfPlmStageGate
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore
	from models import MfPlmProduct, MfPlmStageGate  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgPlmService:
	"""Product Lifecycle Management service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._products: dict[str, MfPlmProduct] = {}
		self._gates: dict[str, MfPlmStageGate] = {}

	async def create_product(self, product_code: str, product_name: str, product_type: str = "standard", description: str = "", owner_id: str | None = None, metadata: dict[str, Any] | None = None) -> MfPlmProduct:
		product = MfPlmProduct(tenant_id=self._tenant_id, product_code=product_code, product_name=product_name, product_type=product_type, description=description, owner_id=owner_id, metadata=metadata or {})
		self._products[product.id] = product
		return product

	async def advance_stage(self, product_id: str, to_stage: str, reviewer_id: str, decision: str = "pass", conditions: str = "") -> MfPlmStageGate:
		product = self._products.get(product_id)
		if not product:
			raise KeyError(f"Product not found: {product_id}")
		ctx = {"tenant_context_present": True, "operation": "record_gate_decision", "approval_present": bool(reviewer_id)}
		dec = evaluate_capability_rules(ctx)
		if dec["decision"] == "deny":
			raise ValueError(f"Stage gate denied: {dec['actions']}")
		gate_number = len([g for g in self._gates.values() if g.product_id == product_id]) + 1
		gate = MfPlmStageGate(tenant_id=self._tenant_id, product_id=product_id, gate_number=gate_number, gate_name=f"Gate {gate_number}", from_stage=product.lifecycle_stage, to_stage=to_stage, decision=decision, reviewer_id=reviewer_id, reviewed_at=_now(), conditions=conditions)
		if decision in ("pass", "conditional_pass"):
			product.lifecycle_stage = to_stage
			if to_stage == "production":
				product.released_at = _now()
			elif to_stage == "discontinued":
				product.discontinued_at = _now()
		self._gates[gate.id] = gate
		return gate

	async def list_products(self, lifecycle_stage: str | None = None, product_type: str | None = None) -> list[MfPlmProduct]:
		products = list(self._products.values())
		if lifecycle_stage:
			products = [p for p in products if p.lifecycle_stage == lifecycle_stage]
		if product_type:
			products = [p for p in products if p.product_type == product_type]
		return products

	async def get_product(self, product_id: str) -> MfPlmProduct:
		if product_id not in self._products:
			raise KeyError(f"Product not found: {product_id}")
		return self._products[product_id]

	async def list_gates(self, product_id: str) -> list[MfPlmStageGate]:
		return sorted([g for g in self._gates.values() if g.product_id == product_id], key=lambda g: g.gate_number)
