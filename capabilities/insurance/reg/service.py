"""Insurance Regulatory Reporting Service (ins_reg).

IRA/NAICOM/FSA returns, Solvency II reporting, statistical returns, market conduct filings.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

REGULATORS = {"IRA", "NAICOM", "FSA", "FCA", "IAIS", "AKI", "PRA"}
RETURN_TYPES = {
	"quarterly_statistical",
	"annual_statutory",
	"solvency_capital",
	"minimum_capital",
	"reinsurance_placement",
	"market_conduct",
	"claims_experience",
	"premium_levy",
	"policy_in_force",
	"anti_money_laundering",
}
RETURN_STATUSES = {"draft", "prepared", "reviewed", "submitted", "accepted", "rejected", "amended"}
SCR_MINIMUM_RATIO = Decimal("1.0")
MCR_MINIMUM_RATIO = Decimal("0.25")


class InsuranceRegulatoryReportingService:
	"""In-memory executable service for Insurance Regulatory Reporting."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.returns: dict[str, dict[str, Any]] = {}
		self.solvency_reports: dict[str, dict[str, Any]] = {}
		self.statistical_returns: dict[str, dict[str, Any]] = {}
		self.market_conduct_filings: dict[str, dict[str, Any]] = {}
		self.compliance_calendar: dict[str, dict[str, Any]] = {}
		self.return_seq: int = 0
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _return_reference(self, regulator: str, return_type: str, period_end: str) -> str:
		self.return_seq += 1
		return f"{regulator}/{return_type.upper()[:4]}/{period_end[:7]}/{self.return_seq:04d}"

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

	# ── Regulatory Returns ────────────────────────────────────────────────────

	async def create_return(
		self,
		tenant_id: str,
		return_type: str,
		regulator: str,
		period_start: str,
		period_end: str,
		prepared_by: str,
		data: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create a new regulatory return."""
		tenant = self._tenant(tenant_id)
		if return_type not in RETURN_TYPES:
			raise ValueError(f"unsupported_return_type:{return_type}")
		if regulator not in REGULATORS:
			raise ValueError(f"unsupported_regulator:{regulator}")
		ref = self._return_reference(regulator, return_type, period_end)
		record: dict[str, Any] = {
			"id": self._record_id("ret"),
			"type": "reg_return",
			"return_reference": ref,
			"return_type": return_type,
			"regulator": regulator,
			"period_start": period_start,
			"period_end": period_end,
			"prepared_by": prepared_by,
			"status": "draft",
			"data": deepcopy(data or {}),
			"submitted_by": None,
			"submitted_at": None,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.returns[record["id"]] = record
		self._emit(tenant, "regulatory_return_created", record["id"], "reg_return", {"ref": ref, "regulator": regulator})
		_log.info("Regulatory return created: %s tenant=%s", ref, tenant)
		return deepcopy(record)

	async def get_return(self, tenant_id: str, return_id: str) -> dict[str, Any]:
		"""Retrieve a regulatory return."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		return deepcopy(ret)

	async def list_returns(self, tenant_id: str, regulator: str | None = None, return_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List regulatory returns."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.returns.values() if r["tenant_id"] == tenant]
		if regulator:
			items = [r for r in items if r["regulator"] == regulator]
		if return_type:
			items = [r for r in items if r["return_type"] == return_type]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def update_return(self, tenant_id: str, return_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update return data (only draft/prepared status)."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		if ret["status"] not in {"draft", "prepared"}:
			raise PermissionError("cannot_update_submitted_return")
		allowed = {"data", "prepared_by", "status"}
		for k, v in updates.items():
			if k in allowed:
				ret[k] = v
		ret["updated_at"] = self._now()
		self._emit(tenant, "regulatory_return_updated", return_id, "reg_return", {})
		return deepcopy(ret)

	async def delete_return(self, tenant_id: str, return_id: str) -> dict[str, Any]:
		"""Cancel a draft return."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		if ret["status"] not in {"draft"}:
			raise PermissionError("only_draft_returns_can_be_deleted")
		ret["status"] = "cancelled"
		ret["cancelled_at"] = self._now()
		self._emit(tenant, "regulatory_return_cancelled", return_id, "reg_return", {})
		return deepcopy(ret)

	async def review_return(self, tenant_id: str, return_id: str, reviewed_by: str, notes: str = "") -> dict[str, Any]:
		"""Mark a return as reviewed."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		if ret["status"] != "prepared":
			raise PermissionError("return_must_be_prepared_for_review")
		ret["status"] = "reviewed"
		ret["reviewed_by"] = reviewed_by
		ret["review_notes"] = notes
		ret["reviewed_at"] = self._now()
		self._emit(tenant, "regulatory_return_reviewed", return_id, "reg_return", {})
		return deepcopy(ret)

	async def submit_return(self, tenant_id: str, return_id: str, submitted_by: str, submission_channel: str = "portal") -> dict[str, Any]:
		"""Submit a reviewed return to the regulator."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		if ret["status"] != "reviewed":
			raise PermissionError("return_must_be_reviewed_before_submission")
		ret["status"] = "submitted"
		ret["submitted_by"] = submitted_by
		ret["submitted_at"] = self._now()
		ret["submission_channel"] = submission_channel
		ret["acknowledgement_reference"] = f"ACK/{ret['return_reference']}"
		self._emit(tenant, "regulatory_return_submitted", return_id, "reg_return", {"regulator": ret["regulator"]})
		return deepcopy(ret)

	async def record_acceptance(self, tenant_id: str, return_id: str, regulator_reference: str) -> dict[str, Any]:
		"""Record regulator acceptance of a submitted return."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		if ret["status"] != "submitted":
			raise PermissionError("return_must_be_submitted_for_acceptance")
		ret["status"] = "accepted"
		ret["regulator_reference"] = regulator_reference
		ret["accepted_at"] = self._now()
		self._emit(tenant, "regulatory_return_accepted", return_id, "reg_return", {"regulator_reference": regulator_reference})
		return deepcopy(ret)

	async def record_rejection(self, tenant_id: str, return_id: str, reason: str) -> dict[str, Any]:
		"""Record regulator rejection of a return."""
		tenant = self._tenant(tenant_id)
		ret = self.returns.get(return_id)
		if not ret or ret["tenant_id"] != tenant:
			raise KeyError(f"return_not_found:{return_id}")
		ret["status"] = "rejected"
		ret["rejection_reason"] = reason
		ret["rejected_at"] = self._now()
		self._emit(tenant, "regulatory_return_rejected", return_id, "reg_return", {"reason": reason})
		return deepcopy(ret)

	# ── Solvency II / Solvency Reporting ─────────────────────────────────────

	async def prepare_solvency_report(
		self,
		tenant_id: str,
		valuation_date: str,
		total_assets: Decimal,
		total_liabilities: Decimal,
		eligible_own_funds: Decimal,
		scr: Decimal,
		mcr: Decimal,
		prepared_by: str,
	) -> dict[str, Any]:
		"""Prepare a solvency capital requirement report."""
		tenant = self._tenant(tenant_id)
		eof = Decimal(str(eligible_own_funds))
		scr_val = Decimal(str(scr))
		mcr_val = Decimal(str(mcr))
		scr_ratio = (eof / scr_val).quantize(Decimal("0.0001")) if scr_val > 0 else Decimal("0")
		mcr_ratio = (eof / mcr_val).quantize(Decimal("0.0001")) if mcr_val > 0 else Decimal("0")
		scr_breach = scr_ratio < SCR_MINIMUM_RATIO
		mcr_breach = mcr_ratio < MCR_MINIMUM_RATIO
		record: dict[str, Any] = {
			"id": self._record_id("solv"),
			"type": "reg_solvency_report",
			"valuation_date": valuation_date,
			"total_assets": Decimal(str(total_assets)),
			"total_liabilities": Decimal(str(total_liabilities)),
			"net_assets": Decimal(str(total_assets)) - Decimal(str(total_liabilities)),
			"eligible_own_funds": eof,
			"scr": scr_val,
			"mcr": mcr_val,
			"scr_ratio": scr_ratio,
			"mcr_ratio": mcr_ratio,
			"scr_breach": scr_breach,
			"mcr_breach": mcr_breach,
			"prepared_by": prepared_by,
			"status": "draft",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.solvency_reports[record["id"]] = record
		if scr_breach or mcr_breach:
			_log.warning("Solvency breach detected: tenant=%s scr_breach=%s mcr_breach=%s", tenant, scr_breach, mcr_breach)
		self._emit(tenant, "solvency_report_prepared", record["id"], "reg_solvency_report", {"scr_breach": scr_breach})
		return deepcopy(record)

	async def list_solvency_reports(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List solvency reports."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.solvency_reports.values() if r["tenant_id"] == tenant]

	async def get_solvency_report(self, tenant_id: str, report_id: str) -> dict[str, Any]:
		"""Retrieve a solvency report."""
		tenant = self._tenant(tenant_id)
		rep = self.solvency_reports.get(report_id)
		if not rep or rep["tenant_id"] != tenant:
			raise KeyError(f"solvency_report_not_found:{report_id}")
		return deepcopy(rep)

	# ── Statistical Returns ───────────────────────────────────────────────────

	async def compile_statistical_return(
		self,
		tenant_id: str,
		period: str,
		policies_in_force: int,
		gross_premium: Decimal,
		net_premium: Decimal,
		gross_claims: Decimal,
		net_claims: Decimal,
		prepared_by: str,
	) -> dict[str, Any]:
		"""Compile an industry statistical return."""
		tenant = self._tenant(tenant_id)
		gp = Decimal(str(gross_premium))
		gc = Decimal(str(gross_claims))
		loss_ratio = (gc / gp * 100).quantize(Decimal("0.01")) if gp > 0 else Decimal("0")
		record: dict[str, Any] = {
			"id": self._record_id("stat"),
			"type": "reg_statistical_return",
			"period": period,
			"policies_in_force": policies_in_force,
			"gross_premium": gp,
			"net_premium": Decimal(str(net_premium)),
			"gross_claims": gc,
			"net_claims": Decimal(str(net_claims)),
			"loss_ratio": loss_ratio,
			"prepared_by": prepared_by,
			"status": "prepared",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.statistical_returns[record["id"]] = record
		self._emit(tenant, "statistical_return_compiled", record["id"], "reg_statistical_return", {"period": period})
		return deepcopy(record)

	async def list_statistical_returns(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List statistical returns."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.statistical_returns.values() if r["tenant_id"] == tenant]

	# ── Market Conduct ────────────────────────────────────────────────────────

	async def file_market_conduct(
		self,
		tenant_id: str,
		filing_type: str,
		subject: str,
		description: str,
		submitted_by: str,
		attachments: list[str] | None = None,
	) -> dict[str, Any]:
		"""Submit a market conduct filing."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._record_id("mc"),
			"type": "reg_market_conduct",
			"filing_type": filing_type,
			"subject": subject,
			"description": description,
			"attachments": list(attachments or []),
			"submitted_by": submitted_by,
			"filing_reference": f"MC/{tenant}/{self._now()[:10]}",
			"status": "filed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.market_conduct_filings[record["id"]] = record
		self._emit(tenant, "market_conduct_filed", record["id"], "reg_market_conduct", {"filing_type": filing_type})
		return deepcopy(record)

	async def list_market_conduct_filings(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List market conduct filings."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(f) for f in self.market_conduct_filings.values() if f["tenant_id"] == tenant]

	async def get_market_conduct_filing(self, tenant_id: str, filing_id: str) -> dict[str, Any]:
		"""Retrieve a market conduct filing."""
		tenant = self._tenant(tenant_id)
		fld = self.market_conduct_filings.get(filing_id)
		if not fld or fld["tenant_id"] != tenant:
			raise KeyError(f"market_conduct_filing_not_found:{filing_id}")
		return deepcopy(fld)

	# ── Compliance Calendar ───────────────────────────────────────────────────

	async def add_compliance_deadline(
		self,
		tenant_id: str,
		return_type: str,
		regulator: str,
		due_date: str,
		frequency: str,
		responsible_party: str,
	) -> dict[str, Any]:
		"""Register a regulatory filing deadline."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._record_id("cal"),
			"type": "reg_compliance_calendar",
			"return_type": return_type,
			"regulator": regulator,
			"due_date": due_date,
			"frequency": frequency,
			"responsible_party": responsible_party,
			"status": "pending",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.compliance_calendar[record["id"]] = record
		self._emit(tenant, "compliance_deadline_added", record["id"], "reg_compliance_calendar", {})
		return deepcopy(record)

	async def list_upcoming_deadlines(self, tenant_id: str, days_ahead: int = 30) -> list[dict[str, Any]]:
		"""Return deadlines falling within the next N days."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
		return [
			deepcopy(d) for d in self.compliance_calendar.values()
			if d["tenant_id"] == tenant and d["status"] == "pending"
			and today <= d["due_date"] <= cutoff
		]

	async def list_compliance_calendar(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all compliance calendar entries."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.compliance_calendar.values() if c["tenant_id"] == tenant]

	# ── IRA-specific helpers ──────────────────────────────────────────────────

	async def ira_premium_levy_return(self, tenant_id: str, period: str, gross_premium: Decimal, prepared_by: str) -> dict[str, Any]:
		"""Compute and record the IRA premium levy (0.25% of gross premium)."""
		tenant = self._tenant(tenant_id)
		gp = Decimal(str(gross_premium))
		levy_rate = Decimal("0.0025")
		levy_amount = (gp * levy_rate).quantize(Decimal("0.01"))
		return await self.create_return(
			tenant_id=tenant,
			return_type="premium_levy",
			regulator="IRA",
			period_start=f"{period}-01",
			period_end=f"{period}-31",
			prepared_by=prepared_by,
			data={"gross_premium": str(gp), "levy_rate": "0.25%", "levy_amount": str(levy_amount)},
		)

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def regulatory_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Summary of regulatory reporting status."""
		tenant = self._tenant(tenant_id)
		rets = [r for r in self.returns.values() if r["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		by_regulator: dict[str, int] = {}
		for r in rets:
			by_status[r["status"]] = by_status.get(r["status"], 0) + 1
			by_regulator[r["regulator"]] = by_regulator.get(r["regulator"], 0) + 1
		overdue_deadlines = [
			d for d in self.compliance_calendar.values()
			if d["tenant_id"] == tenant and d["status"] == "pending" and d["due_date"] < date.today().isoformat()
		]
		return {
			"tenant_id": tenant,
			"total_returns": len(rets),
			"by_status": by_status,
			"by_regulator": by_regulator,
			"solvency_reports": len([s for s in self.solvency_reports.values() if s["tenant_id"] == tenant]),
			"market_conduct_filings": len([m for m in self.market_conduct_filings.values() if m["tenant_id"] == tenant]),
			"overdue_deadlines": len(overdue_deadlines),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ins_reg",
			"status": "healthy",
			"return_count": len(self.returns),
			"solvency_report_count": len(self.solvency_reports),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"capability_id": "ins_reg",
			"name": "Insurance Regulatory Reporting",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_regulators": list(REGULATORS),
			"return_types": list(RETURN_TYPES),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

