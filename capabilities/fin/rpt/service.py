"""Domain service for APG financial reporting."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_OUTPUT_FORMATS,
		SUPPORTED_RPT_AGENT_ROLES,
		SUPPORTED_RPT_AGENT_RUNTIMES,
		SUPPORTED_STATEMENT_TYPES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_OUTPUT_FORMATS,
		SUPPORTED_RPT_AGENT_ROLES,
		SUPPORTED_RPT_AGENT_RUNTIMES,
		SUPPORTED_STATEMENT_TYPES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class FinancialReportingService:
	"""Tenant-scoped report template, generation, publication, consolidation, and distribution coordinator."""

	def __init__(self) -> None:
		self._templates: dict[str, dict[str, Any]] = {}
		self._report_lines: dict[str, dict[str, Any]] = {}
		self._periods: dict[str, dict[str, Any]] = {}
		self._generations: dict[str, dict[str, Any]] = {}
		self._statements: dict[str, dict[str, Any]] = {}
		self._consolidations: dict[str, dict[str, Any]] = {}
		self._disclosures: dict[str, dict[str, Any]] = {}
		self._distributions: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		# new collections
		self._segment_reports: dict[str, dict[str, Any]] = {}
		self._eps_records: dict[str, dict[str, Any]] = {}
		self._notes_to_accounts: dict[str, dict[str, Any]] = {}
		self._xbrl_mappings: dict[str, dict[str, Any]] = {}
		self._regulatory_submissions: dict[str, dict[str, Any]] = {}
		self._ledger_entries: dict[str, dict[str, Any]] = {}  # synthetic GL for demos

	# ------------------------------------------------------------------ existing

	def create_template(self, template_id: str, tenant_id: str, name: str, statement_type: str, owner: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_template",
			"template_name_present": bool(name),
			"statement_type_supported": statement_type in SUPPORTED_STATEMENT_TYPES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_template", template_id),
			"template_id": template_id,
			"tenant_id": tenant_id,
			"name": name,
			"statement_type": statement_type,
			"owner": owner,
			"line_count": 0,
			"status": "draft",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._templates[record["id"]] = record
		self._emit("template_created", tenant_id, record["id"], {"statement_type": statement_type})
		return deepcopy(record)

	def add_report_line(self, line_id: str, tenant_id: str, template_record_id: str, label: str, account_mapping: str, sort_order: int | None, line_type: str = "detail") -> dict[str, Any]:
		template = self._require_template(template_record_id, tenant_id) if template_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "add_report_line",
			"template_present": template is not None,
			"account_mapping_present": bool(account_mapping),
			"sort_order_present": sort_order is not None,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_line", line_id),
			"line_id": line_id,
			"tenant_id": tenant_id,
			"template_record_id": template["id"],
			"label": label,
			"account_mapping": account_mapping,
			"sort_order": sort_order,
			"line_type": line_type,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._report_lines[record["id"]] = record
		template["line_count"] = len(self.list_report_lines(tenant_id, template["id"]))
		template["status"] = "mapped"
		template["updated_at"] = self._now()
		self._emit("report_line_added", tenant_id, record["id"], {"template_id": template["template_id"], "account_mapping": account_mapping})
		return deepcopy(record)

	def open_period(self, period_id: str, tenant_id: str, name: str, period_start: str, period_end: str, close_status: str = "open") -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_period",
			"period_name_present": bool(name),
			"period_dates_present": bool(period_start) and bool(period_end),
			"period_range_valid": self._period_range_valid(period_start, period_end),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_period", period_id),
			"period_id": period_id,
			"tenant_id": tenant_id,
			"name": name,
			"period_start": period_start,
			"period_end": period_end,
			"close_status": close_status,
			"status": "open",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._periods[record["id"]] = record
		self._emit("period_opened", tenant_id, record["id"], {"period_id": period_id})
		return deepcopy(record)

	def generate_report(self, generation_id: str, tenant_id: str, template_record_id: str, period_record_id: str, output_format: str, data_quality_score: float = 1.0, quality_reviewed_by: str | None = None) -> dict[str, Any]:
		template = self._require_template(template_record_id, tenant_id) if template_record_id else None
		period = self._require_period(period_record_id, tenant_id) if period_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "generate_report",
			"template_present": template is not None,
			"period_present": period is not None,
			"template_line_count": template["line_count"] if template else 0,
			"output_format_supported": output_format in SUPPORTED_OUTPUT_FORMATS,
			"data_quality_score": data_quality_score,
			"quality_review_recorded": bool(quality_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_generation", generation_id),
			"generation_id": generation_id,
			"tenant_id": tenant_id,
			"template_record_id": template["id"],
			"period_record_id": period["id"],
			"output_format": output_format,
			"data_quality_score": float(data_quality_score),
			"quality_reviewed_by": quality_reviewed_by,
			"status": "generated",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._generations[record["id"]] = record
		self._emit("report_generated", tenant_id, record["id"], {"output_format": output_format, "data_quality_score": data_quality_score})
		return deepcopy(record)

	def publish_statement(self, statement_id: str, tenant_id: str, generation_record_id: str, title: str, balance_check_passed: bool, approved_by: str, narrative_reviewed_by: str) -> dict[str, Any]:
		generation = self._require_generation(generation_record_id, tenant_id) if generation_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_statement",
			"generation_present": generation is not None,
			"balance_check_passed": balance_check_passed,
			"approval_recorded": bool(approved_by),
			"narrative_review_recorded": bool(narrative_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_statement", statement_id),
			"statement_id": statement_id,
			"tenant_id": tenant_id,
			"generation_record_id": generation["id"],
			"title": title,
			"balance_check_passed": balance_check_passed,
			"approved_by": approved_by,
			"narrative_reviewed_by": narrative_reviewed_by,
			"status": "published",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._statements[record["id"]] = record
		self._emit("statement_published", tenant_id, record["id"], {"title": title, "approved_by": approved_by})
		return deepcopy(record)

	def create_consolidation(self, consolidation_id: str, tenant_id: str, parent_entity: str, subsidiary_entity: str, method: str, ownership_percent: float, elimination_reviewed_by: str | None = None) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_consolidation",
			"parent_entity_present": bool(parent_entity),
			"subsidiary_entity_present": bool(subsidiary_entity),
			"ownership_out_of_bounds": ownership_percent < 0 or ownership_percent > 100,
			"elimination_review_recorded": bool(elimination_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_consolidation", consolidation_id),
			"consolidation_id": consolidation_id,
			"tenant_id": tenant_id,
			"parent_entity": parent_entity,
			"subsidiary_entity": subsidiary_entity,
			"method": method,
			"ownership_percent": float(ownership_percent),
			"elimination_reviewed_by": elimination_reviewed_by,
			"status": "reviewed",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._consolidations[record["id"]] = record
		self._emit("consolidation_created", tenant_id, record["id"], {"parent_entity": parent_entity, "subsidiary_entity": subsidiary_entity})
		return deepcopy(record)

	def record_disclosure(self, disclosure_id: str, tenant_id: str, statement_record_id: str, title: str, owner: str, reviewed_by: str) -> dict[str, Any]:
		statement = self._require_statement(statement_record_id, tenant_id) if statement_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_disclosure",
			"statement_present": statement is not None,
			"owner_present": bool(owner),
			"disclosure_review_recorded": bool(reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_disclosure", disclosure_id),
			"disclosure_id": disclosure_id,
			"tenant_id": tenant_id,
			"statement_record_id": statement["id"],
			"title": title,
			"owner": owner,
			"reviewed_by": reviewed_by,
			"status": "reviewed",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._disclosures[record["id"]] = record
		self._emit("disclosure_recorded", tenant_id, record["id"], {"title": title})
		return deepcopy(record)

	def distribute_statement(self, distribution_id: str, tenant_id: str, statement_record_id: str, recipients: list[str], output_format: str) -> dict[str, Any]:
		statement = self._require_statement(statement_record_id, tenant_id) if statement_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "distribute_statement",
			"statement_present": statement is not None,
			"statement_approved": statement is not None and bool(statement.get("approved_by")),
			"recipient_present": bool(recipients),
			"distribution_format_supported": output_format in SUPPORTED_OUTPUT_FORMATS,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_distribution", distribution_id),
			"distribution_id": distribution_id,
			"tenant_id": tenant_id,
			"statement_record_id": statement["id"],
			"recipients": list(recipients),
			"output_format": output_format,
			"status": "distributed",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._distributions[record["id"]] = record
		self._emit("statement_distributed", tenant_id, record["id"], {"recipient_count": len(recipients), "output_format": output_format})
		return deepcopy(record)

	def register_rpt_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_rpt_agent",
			"agent_runtime_supported": runtime in SUPPORTED_RPT_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_RPT_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("rpt_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("rpt_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_rpt_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown RPT agent: {agent_id}")
		context = {"tenant_context_present": bool(tenant_id), "operation": "agent_rpt_action", "action": action, "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded}
		return evaluate_capability_rules(context)

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "rpt_batch", "event_stream": event_stream}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	# ------------------------------------------------------------------ new methods

	def generate_ifrs_income_statement(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		period: str,
		revenue: dict[str, float] | None = None,
		expenses: dict[str, float] | None = None,
		prepared_by: str = "finance",
	) -> dict[str, Any]:
		"""Generate an IFRS-compliant income statement for an entity and period."""
		assert bool(entity_id), "entity_id required"
		assert bool(period), "period required"
		rev = dict(revenue or {"operating_revenue": 0.0})
		exp = dict(expenses or {"operating_expenses": 0.0, "cost_of_sales": 0.0})
		total_revenue = sum(rev.values())
		total_expenses = sum(exp.values())
		gross_profit = total_revenue - exp.get("cost_of_sales", 0.0)
		operating_profit = gross_profit - sum(v for k, v in exp.items() if k != "cost_of_sales" and "tax" not in k and "interest" not in k)
		finance_costs = exp.get("finance_costs", 0.0) + exp.get("interest_expense", 0.0)
		pbt = operating_profit - finance_costs
		tax = exp.get("income_tax", pbt * 0.3)
		pat = pbt - tax
		record = {
			"id": self._record_id("rpt_income_stmt", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"statement_type": "income_statement",
			"standard": "IFRS",
			"revenue_breakdown": rev,
			"expense_breakdown": exp,
			"total_revenue": round(total_revenue, 2),
			"total_expenses": round(total_expenses, 2),
			"gross_profit": round(gross_profit, 2),
			"operating_profit": round(operating_profit, 2),
			"profit_before_tax": round(pbt, 2),
			"income_tax": round(tax, 2),
			"profit_after_tax": round(pat, 2),
			"prepared_by": prepared_by,
			"status": "draft",
			"generated_at": self._now(),
		}
		self._ledger_entries[record["id"]] = record
		self._emit("ifrs_income_statement_generated", tenant_id, record["id"], {"entity_id": entity_id, "period": period, "pat": pat})
		return deepcopy(record)

	def generate_balance_sheet(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		period: str,
		assets: dict[str, float] | None = None,
		liabilities: dict[str, float] | None = None,
		equity: dict[str, float] | None = None,
		prepared_by: str = "finance",
	) -> dict[str, Any]:
		"""Generate an IFRS statement of financial position (balance sheet)."""
		assert bool(entity_id), "entity_id required"
		a = dict(assets or {"current_assets": 0.0, "non_current_assets": 0.0})
		l = dict(liabilities or {"current_liabilities": 0.0, "non_current_liabilities": 0.0})
		eq = dict(equity or {"share_capital": 0.0, "retained_earnings": 0.0})
		total_assets = sum(a.values())
		total_liabilities = sum(l.values())
		total_equity = sum(eq.values())
		balanced = abs(total_assets - (total_liabilities + total_equity)) < 0.01
		record = {
			"id": self._record_id("rpt_balance_sheet", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"statement_type": "balance_sheet",
			"standard": "IFRS",
			"assets": a,
			"liabilities": l,
			"equity": eq,
			"total_assets": round(total_assets, 2),
			"total_liabilities": round(total_liabilities, 2),
			"total_equity": round(total_equity, 2),
			"total_liabilities_and_equity": round(total_liabilities + total_equity, 2),
			"balanced": balanced,
			"prepared_by": prepared_by,
			"status": "draft",
			"generated_at": self._now(),
		}
		self._ledger_entries[record["id"]] = record
		self._emit("balance_sheet_generated", tenant_id, record["id"], {"entity_id": entity_id, "period": period, "balanced": balanced})
		return deepcopy(record)

	def generate_cash_flow_statement(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		period: str,
		operating_activities: dict[str, float] | None = None,
		investing_activities: dict[str, float] | None = None,
		financing_activities: dict[str, float] | None = None,
		prepared_by: str = "finance",
	) -> dict[str, Any]:
		"""Generate an IFRS cash flow statement using the indirect method."""
		assert bool(entity_id), "entity_id required"
		op = dict(operating_activities or {"net_profit": 0.0, "depreciation": 0.0, "working_capital_changes": 0.0})
		inv = dict(investing_activities or {"capex": 0.0, "asset_disposals": 0.0})
		fin = dict(financing_activities or {"dividends_paid": 0.0, "debt_repayments": 0.0, "new_borrowings": 0.0})
		net_operating = sum(op.values())
		net_investing = sum(inv.values())
		net_financing = sum(fin.values())
		net_change = net_operating + net_investing + net_financing
		record = {
			"id": self._record_id("rpt_cashflow", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"statement_type": "cash_flow_statement",
			"standard": "IFRS",
			"method": "indirect",
			"operating_activities": op,
			"investing_activities": inv,
			"financing_activities": fin,
			"net_cash_from_operations": round(net_operating, 2),
			"net_cash_from_investing": round(net_investing, 2),
			"net_cash_from_financing": round(net_financing, 2),
			"net_change_in_cash": round(net_change, 2),
			"prepared_by": prepared_by,
			"status": "draft",
			"generated_at": self._now(),
		}
		self._ledger_entries[record["id"]] = record
		self._emit("cash_flow_statement_generated", tenant_id, record["id"], {"entity_id": entity_id, "period": period, "net_change": net_change})
		return deepcopy(record)

	def consolidation(
		self,
		tenant_id: str,
		consolidation_id: str,
		parent_id: str,
		subsidiaries: list[str],
		period: str,
		method: str = "full",
		eliminations: dict[str, float] | None = None,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Produce a group consolidation for a parent entity and its subsidiaries."""
		assert bool(parent_id), "parent_id required"
		assert bool(subsidiaries), "at least one subsidiary required"
		assert method in {"full", "proportionate", "equity"}, f"invalid method: {method}"
		elim = dict(eliminations or {})
		total_eliminations = sum(elim.values())
		# create individual consolidation entries
		consolidation_entries: list[dict[str, Any]] = []
		for idx, sub in enumerate(subsidiaries):
			entry = self.create_consolidation(
				consolidation_id=f"{consolidation_id}:{sub}:{idx}",
				tenant_id=tenant_id,
				parent_entity=parent_id,
				subsidiary_entity=sub,
				method=method,
				ownership_percent=100.0 if method == "full" else 50.0,
				elimination_reviewed_by=approved_by,
			)
			consolidation_entries.append(entry)
		record = {
			"id": self._record_id("rpt_group_consolidation", consolidation_id),
			"consolidation_id": consolidation_id,
			"tenant_id": tenant_id,
			"parent_id": parent_id,
			"subsidiary_count": len(subsidiaries),
			"subsidiaries": list(subsidiaries),
			"period": period,
			"method": method,
			"eliminations": elim,
			"total_eliminations": round(total_eliminations, 2),
			"consolidation_entries": [e["id"] for e in consolidation_entries],
			"approved_by": approved_by,
			"status": "consolidated",
			"generated_at": self._now(),
		}
		self._ledger_entries[record["id"]] = record
		self._emit("group_consolidation_completed", tenant_id, record["id"], {"parent_id": parent_id, "subsidiary_count": len(subsidiaries)})
		return deepcopy(record)

	def segment_reporting(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		period: str,
		dimension: str,
		segments: dict[str, dict[str, float]] | None = None,
		prepared_by: str = "finance",
	) -> dict[str, Any]:
		"""Generate IFRS 8 segment reporting broken down by a reporting dimension."""
		assert bool(entity_id), "entity_id required"
		assert bool(dimension), "dimension required"
		segs = dict(segments or {"default_segment": {"revenue": 0.0, "profit": 0.0, "assets": 0.0}})
		summary: list[dict[str, Any]] = []
		for seg_name, metrics in segs.items():
			summary.append({
				"segment": seg_name,
				"revenue": metrics.get("revenue", 0.0),
				"profit": metrics.get("profit", 0.0),
				"assets": metrics.get("assets", 0.0),
				"revenue_pct": 0.0,  # computed below
			})
		total_rev = sum(s["revenue"] for s in summary)
		for s in summary:
			s["revenue_pct"] = round(s["revenue"] / max(total_rev, 1) * 100, 2)
		record = {
			"id": self._record_id("rpt_segment", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"dimension": dimension,
			"segment_count": len(segs),
			"total_revenue": round(total_rev, 2),
			"total_profit": round(sum(s["profit"] for s in summary), 2),
			"total_assets": round(sum(s["assets"] for s in summary), 2),
			"segments": summary,
			"prepared_by": prepared_by,
			"status": "draft",
			"generated_at": self._now(),
		}
		self._segment_reports[record["id"]] = record
		self._emit("segment_report_generated", tenant_id, record["id"], {"entity_id": entity_id, "dimension": dimension})
		return deepcopy(record)

	def earnings_per_share(
		self,
		tenant_id: str,
		report_id: str,
		entity_id: str,
		period: str,
		net_profit: float,
		weighted_avg_shares: float,
		diluted_shares: float | None = None,
		preferred_dividends: float = 0.0,
		prepared_by: str = "finance",
	) -> dict[str, Any]:
		"""Compute basic and diluted EPS per IAS 33."""
		assert weighted_avg_shares > 0, "weighted_avg_shares must be positive"
		earnings_attributable = net_profit - preferred_dividends
		basic_eps = earnings_attributable / weighted_avg_shares
		diluted_eps = earnings_attributable / (diluted_shares or weighted_avg_shares)
		record = {
			"id": self._record_id("rpt_eps", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"net_profit": round(net_profit, 2),
			"preferred_dividends": round(preferred_dividends, 2),
			"earnings_attributable": round(earnings_attributable, 2),
			"weighted_avg_shares": weighted_avg_shares,
			"diluted_shares": diluted_shares or weighted_avg_shares,
			"basic_eps": round(basic_eps, 4),
			"diluted_eps": round(diluted_eps, 4),
			"standard": "IAS 33",
			"prepared_by": prepared_by,
			"status": "draft",
			"generated_at": self._now(),
		}
		self._eps_records[record["id"]] = record
		self._emit("eps_calculated", tenant_id, record["id"], {"entity_id": entity_id, "basic_eps": basic_eps})
		return deepcopy(record)

	def notes_to_accounts(
		self,
		tenant_id: str,
		note_id: str,
		entity_id: str,
		period: str,
		notes: list[dict[str, Any]],
		approved_by: str = "finance",
	) -> dict[str, Any]:
		"""Record notes to the financial statements (accounting policies, estimates, disclosures)."""
		assert bool(notes), "at least one note required"
		assert bool(approved_by), "approver required"
		categorised: dict[str, list[str]] = {}
		for note in notes:
			cat = str(note.get("category", "general"))
			categorised.setdefault(cat, []).append(str(note.get("title", "Untitled")))
		record = {
			"id": self._record_id("rpt_notes", note_id),
			"note_id": note_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"note_count": len(notes),
			"notes": notes,
			"categories": categorised,
			"approved_by": approved_by,
			"status": "published",
			"generated_at": self._now(),
		}
		self._notes_to_accounts[record["id"]] = record
		self._emit("notes_to_accounts_recorded", tenant_id, record["id"], {"entity_id": entity_id, "note_count": len(notes)})
		return deepcopy(record)

	def xbrl_taxonomy_mapping(
		self,
		tenant_id: str,
		mapping_id: str,
		entity_id: str,
		period: str,
		taxonomy: str,
		line_mappings: dict[str, str] | None = None,
		validated_by: str = "system",
	) -> dict[str, Any]:
		"""Map report line items to XBRL taxonomy concepts for electronic filing."""
		assert bool(taxonomy), "taxonomy required"
		known_taxonomies = {"IFRS", "US-GAAP", "UK-GAAP", "ESRS", "FERC"}
		if taxonomy not in known_taxonomies:
			raise ValueError(f"unsupported_taxonomy:{taxonomy}; supported={known_taxonomies}")
		mappings = dict(line_mappings or {})
		lines = self.list_report_lines(tenant_id)
		unmapped = [ln["label"] for ln in lines if ln["label"] not in mappings]
		record = {
			"id": self._record_id("rpt_xbrl", mapping_id),
			"mapping_id": mapping_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"taxonomy": taxonomy,
			"line_mappings": mappings,
			"mapped_count": len(mappings),
			"unmapped_lines": unmapped[:20],
			"coverage_pct": round(len(mappings) / max(len(lines), 1) * 100, 2),
			"validated_by": validated_by,
			"status": "mapped",
			"generated_at": self._now(),
		}
		self._xbrl_mappings[record["id"]] = record
		self._emit("xbrl_mapping_created", tenant_id, record["id"], {"taxonomy": taxonomy, "coverage_pct": record["coverage_pct"]})
		return deepcopy(record)

	def regulatory_submission(
		self,
		tenant_id: str,
		submission_id: str,
		entity_id: str,
		report_type: str,
		regulator: str,
		period: str,
		submitted_by: str,
		statement_ids: list[str] | None = None,
		xbrl_mapping_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a regulatory filing submission with full traceability to underlying statements."""
		assert bool(regulator), "regulator required"
		assert bool(submitted_by), "submitter required"
		record = {
			"id": self._record_id("rpt_reg_submission", submission_id),
			"submission_id": submission_id,
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"report_type": report_type,
			"regulator": regulator,
			"period": period,
			"submitted_by": submitted_by,
			"statement_ids": list(statement_ids or []),
			"xbrl_mapping_id": xbrl_mapping_id,
			"status": "submitted",
			"submitted_at": self._now(),
		}
		self._regulatory_submissions[record["id"]] = record
		self._emit("regulatory_submission_filed", tenant_id, record["id"], {"regulator": regulator, "report_type": report_type})
		return deepcopy(record)

	def reporting_analytics(
		self,
		tenant_id: str,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return aggregated reporting KPIs for an entity across a period."""
		statements = self.list_statements(tenant_id)
		generations = self.list_generations(tenant_id)
		distributions = self.list_distributions(tenant_id)
		disclosures = self.list_disclosures(tenant_id)
		consolidations = self.list_consolidations(tenant_id)
		segments = [r for r in self._segment_reports.values() if r["tenant_id"] == tenant_id and r["entity_id"] == entity_id]
		eps_recs = [r for r in self._eps_records.values() if r["tenant_id"] == tenant_id and r["entity_id"] == entity_id]
		regulatory = [r for r in self._regulatory_submissions.values() if r["tenant_id"] == tenant_id and r["entity_id"] == entity_id]
		latest_eps = eps_recs[-1]["basic_eps"] if eps_recs else None
		return {
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"statement_count": len(statements),
			"published_statements": len([s for s in statements if s["status"] == "published"]),
			"generation_count": len(generations),
			"distribution_count": len(distributions),
			"total_recipients": sum(len(d["recipients"]) for d in distributions),
			"disclosure_count": len(disclosures),
			"consolidation_count": len(consolidations),
			"segment_report_count": len(segments),
			"eps_report_count": len(eps_recs),
			"latest_basic_eps": latest_eps,
			"regulatory_submission_count": len(regulatory),
			"xbrl_mapping_count": len([m for m in self._xbrl_mappings.values() if m["tenant_id"] == tenant_id]),
			"notes_to_accounts_count": len([n for n in self._notes_to_accounts.values() if n["tenant_id"] == tenant_id]),
			"computed_at": self._now(),
		}

	# ------------------------------------------------------------------ dashboard / list / compat

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"template_count": len(self.list_templates(tenant_id)),
			"report_line_count": len(self.list_report_lines(tenant_id)),
			"period_count": len(self.list_periods(tenant_id)),
			"generation_count": len(self.list_generations(tenant_id)),
			"published_statement_count": len([item for item in self.list_statements(tenant_id) if item["status"] == "published"]),
			"consolidation_count": len(self.list_consolidations(tenant_id)),
			"disclosure_count": len(self.list_disclosures(tenant_id)),
			"distribution_count": len(self.list_distributions(tenant_id)),
			"segment_report_count": len([r for r in self._segment_reports.values() if r["tenant_id"] == tenant_id]),
			"eps_report_count": len([r for r in self._eps_records.values() if r["tenant_id"] == tenant_id]),
			"regulatory_submission_count": len([r for r in self._regulatory_submissions.values() if r["tenant_id"] == tenant_id]),
			"xbrl_mapping_count": len([m for m in self._xbrl_mappings.values() if m["tenant_id"] == tenant_id]),
			"rpt_agent_count": len(self.list_rpt_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def statement_summary(self, tenant_id: str) -> dict[str, Any]:
		statements = self.list_statements(tenant_id)
		return {"tenant_id": tenant_id, "statement_count": len(statements), "published_count": len([item for item in statements if item["status"] == "published"])}

	def distribution_summary(self, tenant_id: str) -> dict[str, Any]:
		distributions = self.list_distributions(tenant_id)
		return {"tenant_id": tenant_id, "distribution_count": len(distributions), "recipient_count": sum(len(item["recipients"]) for item in distributions)}

	def list_templates(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._templates, tenant_id)

	def list_report_lines(self, tenant_id: str, template_record_id: str | None = None) -> list[dict[str, Any]]:
		records = self._tenant_records(self._report_lines, tenant_id)
		if template_record_id:
			records = [record for record in records if record["template_record_id"] == template_record_id]
		return records

	def list_periods(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._periods, tenant_id)

	def list_generations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._generations, tenant_id)

	def list_statements(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._statements, tenant_id)

	def list_consolidations(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._consolidations, tenant_id)

	def list_disclosures(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._disclosures, tenant_id)

	def list_distributions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._distributions, tenant_id)

	def list_rpt_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def list_segment_reports(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(r) for r in self._segment_reports.values() if r["tenant_id"] == tenant_id]

	def list_eps_records(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(r) for r in self._eps_records.values() if r["tenant_id"] == tenant_id]

	def list_regulatory_submissions(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(r) for r in self._regulatory_submissions.values() if r["tenant_id"] == tenant_id]

	def list_xbrl_mappings(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(r) for r in self._xbrl_mappings.values() if r["tenant_id"] == tenant_id]

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_template(data.get("template_id", data.get("id", "template")), data.get("tenant_id", "default"), data.get("name", "Statement Template"), data.get("statement_type", "income_statement"), data.get("owner", "finance"))

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_templates(tenant_id)

	# ------------------------------------------------------------------ internals

	def _require_template(self, template_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._templates, template_id, tenant_id, "template", "template_id")

	def _require_period(self, period_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._periods, period_id, tenant_id, "period", "period_id")

	def _require_generation(self, generation_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._generations, generation_id, tenant_id, "generation", "generation_id")

	def _require_statement(self, statement_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._statements, statement_id, tenant_id, "statement", "statement_id")

	def _require_record(self, records: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str, public_key: str) -> dict[str, Any]:
		for record in records.values():
			if record["tenant_id"] == tenant_id and (record["id"] == record_id or record[public_key] == record_id):
				return record
		raise KeyError(f"Unknown {label}: {record_id}")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({"event": event_name, "tenant_id": tenant_id, "record_id": record_id, "payload": deepcopy(payload), "processor": "bytewax", "stream": streaming_manifest()["stream"], "created_at": self._now()})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _period_range_valid(self, start: str, end: str) -> bool:
		if not start or not end:
			return False
		return str(end) > str(start)

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


RPTService = FinancialReportingService
