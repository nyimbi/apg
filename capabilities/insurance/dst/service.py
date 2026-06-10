"""Distribution & Agency Management Service (ins_dst).

Agent registry, commission management, performance tracking, compliance, bancassurance.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

AGENT_TYPES = {"tied", "independent", "bancassurance", "digital", "broker", "corporate"}
COMMISSION_STATUSES = {"pending", "approved", "paid", "withheld", "reversed"}
COMPLIANCE_TYPES = {"ira_licence", "cpd_training", "anti_money_laundering", "fit_and_proper", "professional_indemnity"}

DEFAULT_COMMISSION_RATES: dict[str, Decimal] = {
	"motor_comprehensive": Decimal("0.125"),
	"motor_third_party": Decimal("0.10"),
	"fire_industrial": Decimal("0.10"),
	"fire_domestic": Decimal("0.15"),
	"marine_cargo": Decimal("0.10"),
	"life_whole": Decimal("0.20"),
	"life_term": Decimal("0.25"),
	"health_individual": Decimal("0.15"),
	"health_group": Decimal("0.10"),
	"travel": Decimal("0.20"),
}


class DistributionAgencyService:
	"""In-memory executable service for Distribution & Agency Management."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.agents: dict[str, dict[str, Any]] = {}
		self.commissions: dict[str, dict[str, Any]] = {}
		self.performance_reports: dict[str, dict[str, Any]] = {}
		self.compliance_records: dict[str, dict[str, Any]] = {}
		self.bancassurance_partners: dict[str, dict[str, Any]] = {}
		self.commission_schedules: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"details": details or {},
			"created_at": self._now(),
		})

	def _get_agent(self, agent_id: str, tenant: str) -> dict[str, Any]:
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise KeyError(f"agent_not_found:{agent_id}")
		return agent

	# ── Agent Registry ────────────────────────────────────────────────────────

	async def register_agent(
		self,
		tenant_id: str,
		agent_code: str,
		agent_name: str,
		agent_type: str,
		id_number: str,
		ira_licence_number: str,
		phone: str,
		email: str,
		supervisor_id: str | None = None,
		branch_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a new agent in the distribution network."""
		tenant = self._tenant(tenant_id)
		if agent_type not in AGENT_TYPES:
			raise ValueError(f"unsupported_agent_type:{agent_type}")
		if any(a["agent_code"] == agent_code and a["tenant_id"] == tenant for a in self.agents.values()):
			raise ValueError(f"agent_code_duplicate:{agent_code}")
		record: dict[str, Any] = {
			"id": self._record_id("dst"),
			"type": "dst_agent",
			"agent_code": agent_code,
			"agent_name": agent_name,
			"agent_type": agent_type,
			"id_number": id_number,
			"ira_licence_number": ira_licence_number,
			"phone": phone,
			"email": email,
			"supervisor_id": supervisor_id,
			"branch_id": branch_id,
			"status": "active",
			"policies_count": 0,
			"premium_written": Decimal("0"),
			"commission_earned": Decimal("0"),
			"tenant_id": tenant,
			"created_at": self._now(),
			"metadata": deepcopy(metadata or {}),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "agent_registered", record["id"], "dst_agent", {"agent_code": agent_code, "type": agent_type})
		_log.info("Agent registered: %s type=%s tenant=%s", agent_code, agent_type, tenant)
		return deepcopy(record)

	async def get_agent(self, tenant_id: str, agent_id: str) -> dict[str, Any]:
		"""Retrieve agent details."""
		tenant = self._tenant(tenant_id)
		return deepcopy(self._get_agent(agent_id, tenant))

	async def get_agent_by_code(self, tenant_id: str, agent_code: str) -> dict[str, Any]:
		"""Find agent by agent code."""
		tenant = self._tenant(tenant_id)
		agent = next((a for a in self.agents.values() if a["agent_code"] == agent_code and a["tenant_id"] == tenant), None)
		if not agent:
			raise KeyError(f"agent_not_found:{agent_code}")
		return deepcopy(agent)

	async def update_agent(self, tenant_id: str, agent_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update agent profile."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		allowed = {"agent_name", "phone", "email", "status", "supervisor_id", "branch_id", "metadata"}
		for k, v in updates.items():
			if k in allowed:
				agent[k] = v
		agent["updated_at"] = self._now()
		self._emit(tenant, "agent_updated", agent_id, "dst_agent", {"fields": list(updates.keys())})
		return deepcopy(agent)

	async def delete_agent(self, tenant_id: str, agent_id: str, reason: str) -> dict[str, Any]:
		"""Deregister an agent."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		if agent["policies_count"] > 0:
			raise PermissionError("cannot_deregister_agent_with_active_policies")
		agent["status"] = "deregistered"
		agent["deregistration_reason"] = reason
		agent["deregistered_at"] = self._now()
		self._emit(tenant, "agent_deregistered", agent_id, "dst_agent", {"reason": reason})
		return deepcopy(agent)

	async def list_agents(self, tenant_id: str, agent_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List agents."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.agents.values() if a["tenant_id"] == tenant]
		if agent_type:
			items = [a for a in items if a["agent_type"] == agent_type]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	async def suspend_agent(self, tenant_id: str, agent_id: str, reason: str, suspended_by: str) -> dict[str, Any]:
		"""Suspend an agent's licence."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		if agent["status"] != "active":
			raise PermissionError("only_active_agents_can_be_suspended")
		agent["status"] = "suspended"
		agent["suspension_reason"] = reason
		agent["suspended_by"] = suspended_by
		agent["suspended_at"] = self._now()
		self._emit(tenant, "agent_suspended", agent_id, "dst_agent", {"reason": reason})
		return deepcopy(agent)

	async def reinstate_agent(self, tenant_id: str, agent_id: str, reinstated_by: str) -> dict[str, Any]:
		"""Reinstate a suspended agent."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		if agent["status"] != "suspended":
			raise PermissionError("agent_not_suspended")
		agent["status"] = "active"
		agent["reinstated_by"] = reinstated_by
		agent["reinstated_at"] = self._now()
		self._emit(tenant, "agent_reinstated", agent_id, "dst_agent", {})
		return deepcopy(agent)

	# ── Commission Management ─────────────────────────────────────────────────

	async def compute_commission(
		self,
		tenant_id: str,
		agent_id: str,
		policy_id: str,
		policy_number: str,
		product_code: str,
		premium_amount: Decimal,
		commission_rate: Decimal | None = None,
		period: str = "",
	) -> dict[str, Any]:
		"""Calculate and record agent commission for a policy."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		rate = commission_rate if commission_rate is not None else DEFAULT_COMMISSION_RATES.get(product_code, Decimal("0.10"))
		prem = Decimal(str(premium_amount))
		commission_amount = (prem * Decimal(str(rate))).quantize(Decimal("0.01"))
		record: dict[str, Any] = {
			"id": self._record_id("com"),
			"type": "dst_commission",
			"agent_id": agent_id,
			"agent_code": agent["agent_code"],
			"policy_id": policy_id,
			"policy_number": policy_number,
			"product_code": product_code,
			"premium_amount": prem,
			"commission_rate": Decimal(str(rate)),
			"commission_amount": commission_amount,
			"period": period,
			"status": "pending",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.commissions[record["id"]] = record
		agent["policies_count"] = agent.get("policies_count", 0) + 1
		agent["premium_written"] = agent.get("premium_written", Decimal("0")) + prem
		agent["commission_earned"] = agent.get("commission_earned", Decimal("0")) + commission_amount
		self._emit(tenant, "commission_computed", record["id"], "dst_commission", {"agent_id": agent_id, "amount": str(commission_amount)})
		return deepcopy(record)

	async def approve_commission(self, tenant_id: str, commission_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a pending commission record."""
		tenant = self._tenant(tenant_id)
		com = self.commissions.get(commission_id)
		if not com or com["tenant_id"] != tenant:
			raise KeyError(f"commission_not_found:{commission_id}")
		if com["status"] != "pending":
			raise PermissionError("only_pending_commissions_can_be_approved")
		com["status"] = "approved"
		com["approved_by"] = approved_by
		com["approved_at"] = self._now()
		self._emit(tenant, "commission_approved", commission_id, "dst_commission", {})
		return deepcopy(com)

	async def pay_commission(self, tenant_id: str, commission_id: str, payment_reference: str) -> dict[str, Any]:
		"""Mark a commission as paid."""
		tenant = self._tenant(tenant_id)
		com = self.commissions.get(commission_id)
		if not com or com["tenant_id"] != tenant:
			raise KeyError(f"commission_not_found:{commission_id}")
		if com["status"] != "approved":
			raise PermissionError("commission_must_be_approved_before_payment")
		com["status"] = "paid"
		com["payment_reference"] = payment_reference
		com["paid_at"] = self._now()
		self._emit(tenant, "commission_paid", commission_id, "dst_commission", {"payment_reference": payment_reference})
		return deepcopy(com)

	async def list_commissions(self, tenant_id: str, agent_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List commission records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.commissions.values() if c["tenant_id"] == tenant]
		if agent_id:
			items = [c for c in items if c["agent_id"] == agent_id]
		if status:
			items = [c for c in items if c["status"] == status]
		return items

	async def get_commission(self, tenant_id: str, commission_id: str) -> dict[str, Any]:
		"""Get commission detail."""
		tenant = self._tenant(tenant_id)
		com = self.commissions.get(commission_id)
		if not com or com["tenant_id"] != tenant:
			raise KeyError(f"commission_not_found:{commission_id}")
		return deepcopy(com)

	async def delete_commission(self, tenant_id: str, commission_id: str, reason: str) -> dict[str, Any]:
		"""Reverse a commission."""
		tenant = self._tenant(tenant_id)
		com = self.commissions.get(commission_id)
		if not com or com["tenant_id"] != tenant:
			raise KeyError(f"commission_not_found:{commission_id}")
		if com["status"] not in {"pending", "approved"}:
			raise PermissionError("only_pending_or_approved_commissions_can_be_reversed")
		com["status"] = "reversed"
		com["reversal_reason"] = reason
		com["reversed_at"] = self._now()
		self._emit(tenant, "commission_reversed", commission_id, "dst_commission", {"reason": reason})
		return deepcopy(com)

	# ── Performance Tracking ──────────────────────────────────────────────────

	async def generate_performance_report(
		self,
		tenant_id: str,
		agent_id: str,
		period_start: str,
		period_end: str,
		target_premium: Decimal | None = None,
	) -> dict[str, Any]:
		"""Generate a performance report for an agent."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		period_commissions = [
			c for c in self.commissions.values()
			if c["tenant_id"] == tenant
			and c["agent_id"] == agent_id
			and period_start <= c["created_at"][:10] <= period_end
		]
		policies_sold = len(period_commissions)
		premium_written = sum(c["premium_amount"] for c in period_commissions)
		commission_earned = sum(c["commission_amount"] for c in period_commissions if c["status"] in {"approved", "paid"})
		target = Decimal(str(target_premium)) if target_premium else None
		attainment = float((premium_written / target * 100).quantize(Decimal("0.01"))) if target and target > 0 else None
		record: dict[str, Any] = {
			"id": self._record_id("perf"),
			"type": "dst_performance",
			"agent_id": agent_id,
			"agent_code": agent["agent_code"],
			"agent_name": agent["agent_name"],
			"period_start": period_start,
			"period_end": period_end,
			"policies_sold": policies_sold,
			"premium_written": premium_written,
			"commission_earned": commission_earned,
			"target_premium": target,
			"target_attainment_pct": attainment,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.performance_reports[record["id"]] = record
		self._emit(tenant, "performance_report_generated", record["id"], "dst_performance", {"agent_id": agent_id})
		return deepcopy(record)

	async def list_performance_reports(self, tenant_id: str, agent_id: str | None = None) -> list[dict[str, Any]]:
		"""List performance reports."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.performance_reports.values() if r["tenant_id"] == tenant]
		if agent_id:
			items = [r for r in items if r["agent_id"] == agent_id]
		return items

	# ── Compliance ────────────────────────────────────────────────────────────

	async def record_compliance(
		self,
		tenant_id: str,
		agent_id: str,
		compliance_type: str,
		status: str,
		expiry_date: str | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Record a compliance event for an agent (licence, CPD, AML)."""
		tenant = self._tenant(tenant_id)
		agent = self._get_agent(agent_id, tenant)
		if compliance_type not in COMPLIANCE_TYPES:
			raise ValueError(f"unsupported_compliance_type:{compliance_type}")
		record: dict[str, Any] = {
			"id": self._record_id("cmp"),
			"type": "dst_compliance",
			"agent_id": agent_id,
			"agent_code": agent["agent_code"],
			"compliance_type": compliance_type,
			"status": status,
			"expiry_date": expiry_date,
			"notes": notes,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.compliance_records[record["id"]] = record
		self._emit(tenant, "compliance_recorded", record["id"], "dst_compliance", {"agent_id": agent_id, "type": compliance_type})
		return deepcopy(record)

	async def list_compliance_records(self, tenant_id: str, agent_id: str | None = None) -> list[dict[str, Any]]:
		"""List compliance records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.compliance_records.values() if c["tenant_id"] == tenant]
		if agent_id:
			items = [c for c in items if c["agent_id"] == agent_id]
		return items

	async def get_compliance_record(self, tenant_id: str, compliance_id: str) -> dict[str, Any]:
		"""Get compliance record."""
		tenant = self._tenant(tenant_id)
		rec = self.compliance_records.get(compliance_id)
		if not rec or rec["tenant_id"] != tenant:
			raise KeyError(f"compliance_record_not_found:{compliance_id}")
		return deepcopy(rec)

	async def list_expiring_licences(self, tenant_id: str, days_ahead: int = 30) -> list[dict[str, Any]]:
		"""Return compliance records expiring within the next N days."""
		from datetime import timedelta
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
		return [
			deepcopy(c) for c in self.compliance_records.values()
			if c["tenant_id"] == tenant and c.get("expiry_date") and today <= c["expiry_date"] <= cutoff
		]

	# ── Bancassurance ─────────────────────────────────────────────────────────

	async def register_bancassurance_partner(
		self,
		tenant_id: str,
		partner_name: str,
		partner_type: str,
		bank_code: str,
		products: list[str],
		commission_rate: Decimal,
		effective_date: str,
	) -> dict[str, Any]:
		"""Register a bancassurance distribution partner."""
		tenant = self._tenant(tenant_id)
		if any(p["bank_code"] == bank_code and p["tenant_id"] == tenant for p in self.bancassurance_partners.values()):
			raise ValueError(f"bank_code_duplicate:{bank_code}")
		record: dict[str, Any] = {
			"id": self._record_id("banca"),
			"type": "dst_bancassurance",
			"partner_name": partner_name,
			"partner_type": partner_type,
			"bank_code": bank_code,
			"products": list(products),
			"commission_rate": Decimal(str(commission_rate)),
			"effective_date": effective_date,
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.bancassurance_partners[record["id"]] = record
		self._emit(tenant, "bancassurance_partner_registered", record["id"], "dst_bancassurance", {"partner_name": partner_name})
		return deepcopy(record)

	async def list_bancassurance_partners(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List bancassurance partners."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.bancassurance_partners.values() if p["tenant_id"] == tenant]

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def agency_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Agency network summary metrics."""
		tenant = self._tenant(tenant_id)
		agents = [a for a in self.agents.values() if a["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		for a in agents:
			by_type[a["agent_type"]] = by_type.get(a["agent_type"], 0) + 1
		total_premium = sum(a.get("premium_written", Decimal("0")) for a in agents)
		total_commission = sum(a.get("commission_earned", Decimal("0")) for a in agents)
		return {
			"tenant_id": tenant,
			"total_agents": len(agents),
			"active_agents": sum(1 for a in agents if a["status"] == "active"),
			"by_type": by_type,
			"total_premium_written": str(total_premium),
			"total_commission_earned": str(total_commission),
			"bancassurance_partners": len([p for p in self.bancassurance_partners.values() if p["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ins_dst",
			"status": "healthy",
			"agent_count": len(self.agents),
			"commission_count": len(self.commissions),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"capability_id": "ins_dst",
			"name": "Distribution & Agency Management",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"agent_types": list(AGENT_TYPES),
			"compliance_types": list(COMPLIANCE_TYPES),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
