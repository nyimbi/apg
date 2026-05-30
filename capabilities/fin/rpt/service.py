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

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.create_template(data.get("template_id", data.get("id", "template")), data.get("tenant_id", "default"), data.get("name", "Statement Template"), data.get("statement_type", "income_statement"), data.get("owner", "finance"))

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_templates(tenant_id)

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
