"""Executable service layer for APG Scraper/Data Harvesting — expanded implementation."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_HARVEST_AGENT_ROLES,
	SUPPORTED_HARVEST_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .harvest_runtime import (
	classify_dlp_status,
	normalize_extractor_type,
	normalize_harvest_mode,
	normalize_source_type,
	normalize_tags,
	result_retention_until,
	run_status,
	stable_id,
	summarize_decision,
	utc_now,
)
from .models import ExtractorProfile, HarvestAgent, HarvestJob, HarvestResult, HarvestRun, HarvestSource, PipelineHandoff, ScrpAuditEvent
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _domain_from_url(url: str) -> str:
	match = re.search(r"https?://([^/]+)", url)
	return match.group(1) if match else url


class ScraperDataHarvestingService:
	"""
	Tenant-scoped source, extractor, harvest job, run, result, and pipeline runtime.

	Expanded with: schedule_scrape, run_scrape, scrape_result,
	extract_structured_data, javascript_rendered_scrape,
	rate_limit_management, proxy_rotation, captcha_handling,
	data_deduplication, scraping_analytics.
	"""

	def __init__(self) -> None:
		self._sources: dict[str, HarvestSource] = {}
		self._extractors: dict[str, ExtractorProfile] = {}
		self._jobs: dict[str, HarvestJob] = {}
		self._runs: dict[str, HarvestRun] = {}
		self._results: dict[str, HarvestResult] = {}
		self._handoffs: dict[str, PipelineHandoff] = {}
		self._agents: dict[str, HarvestAgent] = {}
		self._audit_events: list[ScrpAuditEvent] = []
		# New stores
		self._scheduled_tasks: dict[str, dict[str, Any]] = {}
		self._rate_limits: dict[str, dict[str, Any]] = {}
		self._proxy_pool: list[dict[str, Any]] = []
		self._proxy_index: int = 0
		self._captcha_records: list[dict[str, Any]] = []
		self._dedup_manifests: dict[str, dict[str, Any]] = {}
		self._structured_extractions: dict[str, dict[str, Any]] = {}
		self._js_renders: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# schedule_scrape
	# ------------------------------------------------------------------

	def schedule_scrape(
		self,
		url: str,
		depth: int,
		frequency: str,
		selectors: dict[str, str],
		auth_config: dict[str, Any],
		tenant_id: str = "default",
		owner: str = "system",
		tags: list[str] | None = None,
		pii_expected: bool = False,
	) -> dict[str, Any]:
		"""
		Schedule a recurring scrape task for a URL.

		Args:
			url: Target URL to scrape.
			depth: Link-follow depth (0 = single page).
			frequency: Cron-style or named frequency (e.g. 'hourly', 'daily').
			selectors: Dict of field_name -> CSS/XPath selector.
			auth_config: Authentication config dict (type, credentials vault ref).
			tenant_id: Owning tenant.
			owner: Responsible owner identity.
			tags: Optional classification tags.
			pii_expected: Whether PII may be encountered.
		"""
		self._require_tenant(tenant_id)
		if not url:
			raise ValueError("schedule_scrape_url_required")
		if depth < 0:
			raise ValueError("schedule_scrape_depth_must_be_non_negative")
		if not selectors:
			raise ValueError("schedule_scrape_selectors_required")
		domain = _domain_from_url(url)
		rate_policy = self._rate_limits.get(f"{tenant_id}:{domain}")
		task_id = stable_id("task", tenant_id, url, frequency, str(len(self._scheduled_tasks)))
		task = {
			"id": task_id,
			"tenant_id": tenant_id,
			"url": url,
			"domain": domain,
			"depth": depth,
			"frequency": frequency,
			"selectors": selectors,
			"auth_type": auth_config.get("type", "none"),
			"auth_vault_ref": auth_config.get("vault_ref"),
			"owner": owner,
			"tags": normalize_tags(tags),
			"pii_expected": pii_expected,
			"rate_limit_rpm": rate_policy["requests_per_minute"] if rate_policy else 60,
			"status": "scheduled",
			"next_run": _ts(),
			"created_at": _ts(),
		}
		self._scheduled_tasks[task_id] = task
		self._record_event(tenant_id, "scrape_task_scheduled", task_id, f"Task scheduled for {url}", owner)
		return dict(task)

	def run_scrape(
		self,
		task_id: str,
		tenant_id: str = "default",
		requested_by: str = "system",
	) -> dict[str, Any]:
		"""
		Execute a scheduled scrape task immediately.

		Validates rate limits, rotates proxy if needed, and returns a run record.
		"""
		self._require_tenant(tenant_id)
		task = self._scheduled_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant_id:
			raise KeyError(f"scheduled_task_not_found:{task_id}")
		domain = task["domain"]
		rate_policy = self._rate_limits.get(f"{tenant_id}:{domain}")
		if rate_policy and rate_policy.get("blocked"):
			raise PermissionError(f"rate_limit_exceeded_for_domain:{domain}")
		proxy = self._rotate_proxy_internal(tenant_id)
		run_id = stable_id("run", tenant_id, task_id, str(len(self._runs)))
		# Simulate run creation via harvest job/run mechanism
		run_record = {
			"id": run_id,
			"task_id": task_id,
			"tenant_id": tenant_id,
			"url": task["url"],
			"depth": task["depth"],
			"selectors": task["selectors"],
			"proxy_used": proxy.get("address") if proxy else None,
			"requested_by": requested_by,
			"status": "running",
			"records_found": 0,
			"started_at": _ts(),
			"completed_at": None,
		}
		task["status"] = "running"
		task["last_run"] = _ts()
		self._record_event(tenant_id, "scrape_run_started", run_id, f"Run started for task {task_id}", requested_by)
		return run_record

	def scrape_result(
		self,
		task_id: str,
		tenant_id: str = "default",
		records_found: int = 0,
		raw_bytes: int = 0,
		errors: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Record and return the result of a completed scrape task.

		Updates task status to 'completed' or 'failed'.
		"""
		self._require_tenant(tenant_id)
		task = self._scheduled_tasks.get(task_id)
		if not task or task["tenant_id"] != tenant_id:
			raise KeyError(f"scheduled_task_not_found:{task_id}")
		error_list = errors or []
		status = "failed" if error_list else "completed"
		task["status"] = "scheduled"  # reset for next run
		task["last_result"] = {
			"records_found": records_found,
			"raw_bytes": raw_bytes,
			"errors": error_list,
			"status": status,
			"completed_at": _ts(),
		}
		result = {
			"task_id": task_id,
			"tenant_id": tenant_id,
			"url": task["url"],
			"records_found": records_found,
			"raw_bytes": raw_bytes,
			"errors": error_list,
			"status": status,
			"completed_at": _ts(),
		}
		self._record_event(tenant_id, "scrape_result_recorded", task_id, f"Scrape {status} with {records_found} records", "system")
		return result

	# ------------------------------------------------------------------
	# extract_structured_data
	# ------------------------------------------------------------------

	def extract_structured_data(
		self,
		raw_html: str,
		extraction_schema: dict[str, Any],
		tenant_id: str = "default",
		extraction_id: str | None = None,
		source_url: str = "",
	) -> dict[str, Any]:
		"""
		Extract structured data from raw HTML using a field schema.

		extraction_schema: dict of field_name -> {"selector": ..., "type": "text|attr|list"}.
		Returns extracted field dict plus quality metrics.
		"""
		self._require_tenant(tenant_id)
		if not raw_html:
			raise ValueError("raw_html_required")
		if not extraction_schema:
			raise ValueError("extraction_schema_required")
		ext_id = extraction_id or stable_id("ext", tenant_id, source_url, str(len(self._structured_extractions)))
		# Synthetic extraction: use selector hints to find patterns
		extracted: dict[str, Any] = {}
		fields_attempted = len(extraction_schema)
		fields_extracted = 0
		for field_name, spec in extraction_schema.items():
			selector = spec.get("selector", "")
			field_type = spec.get("type", "text")
			# Simulate: if selector keyword appears in HTML, extract a value
			if selector and selector.lower()[:4] in raw_html.lower():
				if field_type == "list":
					extracted[field_name] = [f"item_{i}" for i in range(1, 4)]
				else:
					extracted[field_name] = f"extracted_{field_name}_value"
				fields_extracted += 1
			else:
				extracted[field_name] = None
		quality_score = round(fields_extracted / fields_attempted, 4) if fields_attempted else 0.0
		record = {
			"extraction_id": ext_id,
			"tenant_id": tenant_id,
			"source_url": source_url,
			"fields_attempted": fields_attempted,
			"fields_extracted": fields_extracted,
			"quality_score": quality_score,
			"extracted": extracted,
			"extracted_at": _ts(),
		}
		self._structured_extractions[ext_id] = record
		self._record_event(tenant_id, "structured_data_extracted", ext_id, f"Extracted {fields_extracted}/{fields_attempted} fields", "system")
		return record

	def javascript_rendered_scrape(
		self,
		url: str,
		wait_for_selector: str,
		tenant_id: str = "default",
		timeout_ms: int = 10000,
		owner: str = "system",
	) -> dict[str, Any]:
		"""
		Perform a JavaScript-rendered page scrape.

		Simulates browser rendering by tracking the wait_for_selector.
		Returns rendered HTML metadata and extraction readiness.
		"""
		self._require_tenant(tenant_id)
		if not url:
			raise ValueError("url_required")
		if not wait_for_selector:
			raise ValueError("wait_for_selector_required")
		render_id = stable_id("render", tenant_id, url, wait_for_selector)
		record = {
			"render_id": render_id,
			"tenant_id": tenant_id,
			"url": url,
			"wait_for_selector": wait_for_selector,
			"timeout_ms": timeout_ms,
			"owner": owner,
			"rendered": True,
			"selector_found": True,
			"render_time_ms": min(timeout_ms, 3500),
			"rendered_at": _ts(),
		}
		self._js_renders[render_id] = record
		self._record_event(tenant_id, "js_rendered_scrape", render_id, f"Rendered {url} awaiting {wait_for_selector}", owner)
		return record

	# ------------------------------------------------------------------
	# rate_limit_management
	# ------------------------------------------------------------------

	def rate_limit_management(
		self,
		domain: str,
		requests_per_minute: int,
		tenant_id: str = "default",
		burst_limit: int | None = None,
		backoff_seconds: int = 30,
		managed_by: str = "system",
	) -> dict[str, Any]:
		"""
		Set or update rate limiting policy for a domain.

		Policies are per (tenant_id, domain).  Setting requests_per_minute=0
		blocks the domain entirely.
		"""
		self._require_tenant(tenant_id)
		if not domain:
			raise ValueError("rate_limit_domain_required")
		if requests_per_minute < 0:
			raise ValueError("requests_per_minute_must_be_non_negative")
		key = f"{tenant_id}:{domain}"
		record = {
			"domain": domain,
			"tenant_id": tenant_id,
			"requests_per_minute": requests_per_minute,
			"burst_limit": burst_limit or requests_per_minute * 2,
			"backoff_seconds": backoff_seconds,
			"blocked": requests_per_minute == 0,
			"managed_by": managed_by,
			"updated_at": _ts(),
		}
		self._rate_limits[key] = record
		self._record_event(tenant_id, "rate_limit_configured", domain, f"Rate limit set to {requests_per_minute} rpm", managed_by)
		return record

	def proxy_rotation(
		self,
		request_id: str,
		tenant_id: str = "default",
		strategy: str = "round_robin",
		required_country: str | None = None,
	) -> dict[str, Any]:
		"""
		Select and return the next proxy for a request.

		Supports round_robin and random strategies.
		Filters by required_country if specified.
		"""
		self._require_tenant(tenant_id)
		proxy = self._rotate_proxy_internal(tenant_id, required_country=required_country, strategy=strategy)
		if not proxy:
			# Auto-generate a synthetic proxy entry
			proxy = self._add_default_proxy(tenant_id, required_country)
		record = {
			"request_id": request_id,
			"tenant_id": tenant_id,
			"proxy_address": proxy["address"],
			"proxy_country": proxy.get("country"),
			"strategy": strategy,
			"assigned_at": _ts(),
		}
		self._record_event(tenant_id, "proxy_rotated", request_id, f"Assigned proxy {proxy['address']}", "system")
		return record

	def captcha_handling(
		self,
		page_url: str,
		captcha_type: str,
		tenant_id: str = "default",
		solver_type: str = "third_party",
		solved_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record and handle a CAPTCHA challenge for a scraping task.

		captcha_type: 'recaptcha_v2', 'recaptcha_v3', 'hcaptcha', 'image_text'.
		solver_type: 'third_party', 'manual', 'ml_model'.
		"""
		self._require_tenant(tenant_id)
		if not page_url:
			raise ValueError("page_url_required")
		supported_types = {"recaptcha_v2", "recaptcha_v3", "hcaptcha", "image_text", "cloudflare"}
		if captcha_type not in supported_types:
			raise ValueError(f"unsupported_captcha_type:{captcha_type}")
		challenge_id = stable_id("captcha", tenant_id, page_url, captcha_type, str(len(self._captcha_records)))
		# Simulate success rate by type
		success_rates = {"recaptcha_v2": 0.92, "recaptcha_v3": 0.85, "hcaptcha": 0.88, "image_text": 0.97, "cloudflare": 0.75}
		success_rate = success_rates.get(captcha_type, 0.8)
		solved = success_rate > 0.7  # synthetic deterministic pass
		record = {
			"challenge_id": challenge_id,
			"tenant_id": tenant_id,
			"page_url": page_url,
			"captcha_type": captcha_type,
			"solver_type": solver_type,
			"solved_by": solved_by,
			"solved": solved,
			"success_rate": success_rate,
			"token_issued": solved,
			"handled_at": _ts(),
		}
		self._captcha_records.append(record)
		self._record_event(tenant_id, "captcha_handled", challenge_id, f"{captcha_type} {'solved' if solved else 'failed'}", solved_by)
		return record

	def data_deduplication(
		self,
		dataset_id: str,
		tenant_id: str = "default",
		field_keys: list[str] | None = None,
		strategy: str = "hash",
		owner: str = "system",
	) -> dict[str, Any]:
		"""
		Run deduplication analysis on a harvest dataset.

		strategy: 'hash' (exact), 'fuzzy' (similarity-based), 'key_fields'.
		field_keys: For key_fields strategy, the fields used as unique identifier.
		Returns dedup statistics and manifest.
		"""
		self._require_tenant(tenant_id)
		if not dataset_id:
			raise ValueError("dataset_id_required")
		supported_strategies = {"hash", "fuzzy", "key_fields"}
		if strategy not in supported_strategies:
			raise ValueError(f"unsupported_dedup_strategy:{strategy}")
		if strategy == "key_fields" and not field_keys:
			raise ValueError("key_fields_required_for_key_fields_strategy")
		# Gather runs/results for this dataset as a proxy for dedup scope
		results = list(self._results.values())
		tenant_results = [r for r in results if r.tenant_id == tenant_id]
		total_records = sum(r.record_count for r in tenant_results)
		# Synthetic dedup: assume 5-15% duplicates depending on strategy
		dup_rate = {"hash": 0.05, "fuzzy": 0.12, "key_fields": 0.08}.get(strategy, 0.05)
		duplicates_found = int(total_records * dup_rate)
		unique_records = total_records - duplicates_found
		manifest_id = stable_id("dedup", tenant_id, dataset_id, strategy)
		manifest = {
			"manifest_id": manifest_id,
			"dataset_id": dataset_id,
			"tenant_id": tenant_id,
			"strategy": strategy,
			"field_keys": field_keys or [],
			"total_records": total_records,
			"duplicates_found": duplicates_found,
			"unique_records": unique_records,
			"dedup_rate": round(dup_rate, 4),
			"owner": owner,
			"completed_at": _ts(),
		}
		self._dedup_manifests[manifest_id] = manifest
		self._record_event(tenant_id, "deduplication_run", manifest_id, f"Found {duplicates_found} duplicates in {total_records} records", owner)
		return manifest

	def scraping_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return aggregated scraping analytics for a tenant over a period.

		Covers tasks, runs, results, rate limits, proxy usage, captchas,
		deduplication, and structured extractions.
		"""
		sources = self.list_sources(tenant_id)
		runs = self.list_runs(tenant_id)
		results = self.list_results(tenant_id)
		period_tasks = [t for t in self._scheduled_tasks.values() if t["tenant_id"] == tenant_id]
		period_captchas = [c for c in self._captcha_records if c["tenant_id"] == tenant_id]
		period_dedup = [m for m in self._dedup_manifests.values() if m["tenant_id"] == tenant_id]
		period_renders = [r for r in self._js_renders.values() if r["tenant_id"] == tenant_id]
		rate_limited_domains = [
			v for v in self._rate_limits.values()
			if v["tenant_id"] == tenant_id
		]
		blocked_domains = [d for d in rate_limited_domains if d.get("blocked")]
		succeeded_runs = [r for r in runs if r.get("status") == "succeeded"]
		total_records = sum(r.get("record_count", 0) for r in results)
		captcha_solve_rate = (
			round(sum(1 for c in period_captchas if c["solved"]) / len(period_captchas), 4)
			if period_captchas else 0.0
		)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"source_count": len(sources),
			"scheduled_task_count": len(period_tasks),
			"run_count": len(runs),
			"succeeded_run_count": len(succeeded_runs),
			"run_success_rate": round(len(succeeded_runs) / len(runs), 4) if runs else 0.0,
			"total_records_harvested": total_records,
			"result_count": len(results),
			"rate_limited_domain_count": len(rate_limited_domains),
			"blocked_domain_count": len(blocked_domains),
			"captcha_encounter_count": len(period_captchas),
			"captcha_solve_rate": captcha_solve_rate,
			"dedup_run_count": len(period_dedup),
			"js_render_count": len(period_renders),
			"structured_extraction_count": len(self._structured_extractions),
			"pipeline_handoff_count": len(self.list_handoffs(tenant_id)),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# Original methods (retained unchanged for backward compat)
	# ------------------------------------------------------------------

	def register_source(
		self,
		tenant_id: str,
		name: str,
		source_type: str,
		endpoint: str,
		owner: str,
		terms_evidence: str,
		credential_vault_ref: str,
		rate_limit_per_minute: int,
		robots_policy_attached: bool = True,
		pii_expected: bool = False,
		pii_policy_attached: bool = False,
		sensitive_source: bool = False,
		source_review_recorded: bool = False,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not endpoint:
			raise ValueError("source_endpoint_required")
		source_type = normalize_source_type(source_type)
		policy_context = {
			"tenant_context_present": True,
			"operation": "register_source",
			"source_owner_assigned": bool(owner),
			"terms_evidence_present": bool(terms_evidence),
			"credential_vault_present": bool(credential_vault_ref),
			"rate_limit_per_minute": int(rate_limit_per_minute),
			"robots_policy_attached": bool(robots_policy_attached),
			"pii_expected": pii_expected,
			"pii_policy_attached": pii_policy_attached,
			"sensitive_source": sensitive_source,
			"source_review_recorded": source_review_recorded,
		}
		result = self.evaluate(policy_context)
		if result["decision"] == "deny" or (result["decision"] == "require_review" and not source_review_recorded):
			raise PermissionError(summarize_decision(result))
		source = HarvestSource(
			id=stable_id("src", tenant_id, name, endpoint),
			tenant_id=tenant_id,
			name=name,
			source_type=source_type,
			owner=owner,
			endpoint=endpoint,
			terms_evidence=terms_evidence,
			credential_vault_ref=credential_vault_ref,
			rate_limit_per_minute=rate_limit_per_minute,
			robots_policy_attached=robots_policy_attached,
			pii_expected=pii_expected,
			pii_policy_attached=pii_policy_attached,
			sensitive_source=sensitive_source,
			source_review_recorded=source_review_recorded,
			tags=normalize_tags(tags),
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._sources[source.id] = source
		self._record_event(tenant_id, "source_registered", source.id, f"Source {name} registered.", owner)
		return source.to_dict()

	def create_extractor_profile(
		self,
		tenant_id: str,
		name: str,
		extractor_type: str,
		owner: str,
		schema: dict[str, Any],
		output_mapping: dict[str, str] | None = None,
		incremental_cursor_field: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("extractor_owner_required")
		result = self.evaluate({"tenant_context_present": True, "operation": "create_extractor", "schema_present": bool(schema)})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		extractor = ExtractorProfile(
			id=stable_id("ext", tenant_id, name, extractor_type),
			tenant_id=tenant_id,
			name=name,
			extractor_type=normalize_extractor_type(extractor_type),
			owner=owner,
			schema=dict(schema),
			output_mapping=dict(output_mapping or {}),
			incremental_cursor_field=incremental_cursor_field,
			created_at=utc_now(),
		)
		self._extractors[extractor.id] = extractor
		self._record_event(tenant_id, "extractor_created", extractor.id, f"Extractor {name} created.", owner)
		return extractor.to_dict()

	def create_harvest_job(
		self,
		tenant_id: str,
		name: str,
		source_id: str,
		extractor_profile_id: str,
		owner: str,
		mode: str = "incremental",
		schedule_policy_attached: bool = True,
		pipeline_target: str | None = None,
		enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("harvest_job_owner_required")
		self._require_owned(self._sources, source_id, tenant_id, "source_not_found")
		self._require_owned(self._extractors, extractor_profile_id, tenant_id, "extractor_not_found")
		mode = normalize_harvest_mode(mode)
		result = self.evaluate({"tenant_context_present": True, "operation": "create_harvest_job", "pipeline_target_present": bool(pipeline_target)})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		job = HarvestJob(
			id=stable_id("job", tenant_id, name, source_id, extractor_profile_id),
			tenant_id=tenant_id,
			name=name,
			source_id=source_id,
			extractor_profile_id=extractor_profile_id,
			owner=owner,
			mode=mode,
			schedule_policy_attached=schedule_policy_attached,
			pipeline_target=pipeline_target,
			enabled=enabled,
			created_at=utc_now(),
			updated_at=utc_now(),
		)
		self._jobs[job.id] = job
		self._record_event(tenant_id, "harvest_job_created", job.id, f"Harvest job {name} created.", owner)
		return job.to_dict()

	def run_harvest(self, tenant_id: str, job_id: str, requested_by: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		job = self._require_owned(self._jobs, job_id, tenant_id, "harvest_job_not_found")
		if not job.enabled:
			raise PermissionError("harvest_job_disabled")
		source = self._require_owned(self._sources, job.source_id, tenant_id, "source_not_found")
		self._require_owned(self._extractors, job.extractor_profile_id, tenant_id, "extractor_not_found")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "run_harvest",
			"schedule_policy_attached": job.schedule_policy_attached,
			"terms_evidence_present": bool(source.terms_evidence),
			"pii_expected": source.pii_expected,
			"pii_policy_attached": source.pii_policy_attached,
			"sensitive_source": source.sensitive_source,
			"source_review_recorded": source.source_review_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		run = HarvestRun(
			id=stable_id("run", tenant_id, job_id, len(self._runs) + 1),
			tenant_id=tenant_id,
			job_id=job_id,
			source_id=job.source_id,
			extractor_profile_id=job.extractor_profile_id,
			requested_by=requested_by,
			status="running",
			dlp_status=classify_dlp_status(source.pii_expected, False),
			started_at=utc_now(),
			logs=[f"Started harvest job {job.name} for source {source.name}."],
		)
		self._runs[run.id] = run
		self._record_event(tenant_id, "harvest_run_started", run.id, f"Harvest run started for {job.name}.", requested_by)
		return run.to_dict()

	def complete_harvest_run(
		self,
		tenant_id: str,
		run_id: str,
		records_extracted: int,
		error_count: int = 0,
		dlp_scanned: bool = True,
		dlp_violations: int = 0,
		schema_valid: bool = True,
		storage_ref: str | None = None,
		logs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		run = self._require_owned(self._runs, run_id, tenant_id, "harvest_run_not_found")
		source = self._require_owned(self._sources, run.source_id, tenant_id, "source_not_found")
		job = self._require_owned(self._jobs, run.job_id, tenant_id, "harvest_job_not_found")
		if records_extracted < 0 or error_count < 0 or dlp_violations < 0:
			raise ValueError("harvest_counts_must_be_non_negative")
		result_policy = self.evaluate({"tenant_context_present": True, "operation": "complete_harvest_run", "pii_expected": source.pii_expected, "dlp_scanned": bool(dlp_scanned)})
		if result_policy["decision"] != "allow":
			raise PermissionError(summarize_decision(result_policy))
		run.records_extracted = records_extracted
		run.error_count = error_count
		run.dlp_status = classify_dlp_status(source.pii_expected, dlp_scanned, dlp_violations)
		run.dlp_violations = dlp_violations
		run.status = run_status(records_extracted, error_count, run.dlp_status == "failed")
		run.logs.extend(logs or [])
		run.completed_at = utc_now()
		result = HarvestResult(
			id=stable_id("res", tenant_id, run.id, records_extracted),
			tenant_id=tenant_id,
			run_id=run.id,
			record_count=records_extracted,
			schema_valid=schema_valid,
			retention_until=result_retention_until(get_capability_contract(tenant_id)["configuration"]["extraction"]["result_retention_days"]),
			storage_ref=storage_ref or f"memory://{run.id}",
			created_at=utc_now(),
		)
		self._results[result.id] = result
		if job.pipeline_target and run.status == "succeeded":
			handoff = PipelineHandoff(
				id=stable_id("pipe", tenant_id, result.id, job.pipeline_target),
				tenant_id=tenant_id,
				result_id=result.id,
				pipeline_target=job.pipeline_target,
				status="queued",
				created_at=utc_now(),
			)
			self._handoffs[handoff.id] = handoff
		self._record_event(tenant_id, "harvest_run_completed", run.id, f"Harvest completed with status {run.status}.", run.requested_by, "warning" if run.status != "succeeded" else "info")
		return run.to_dict() | {"result": result.to_dict()}

	def register_harvest_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool,
		policy_ref: str = "",
		registered: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_harvest_agent_runtime(runtime)
		normalized_role = _normalize_harvest_agent_role(role)
		result = self.evaluate({
			"tenant_context_present": True,
			"harvest_agent_present": True,
			"agent_registered": bool(registered),
			"agent_runtime_supported": bool(normalized_runtime),
			"agent_role_supported": bool(normalized_role),
			"agent_scope_present": bool(scope.strip()),
			"agent_contribution_disclosed": bool(contribution_disclosed),
		})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		agent_key = stable_id("agent", tenant_id, agent_id)
		if agent_key in self._agents:
			raise ValueError("harvest_agent_already_registered")
		agent = HarvestAgent(
			id=agent_id,
			tenant_id=tenant_id,
			name=name or agent_id,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			registered=registered,
			contribution_disclosed=contribution_disclosed,
			policy_ref=policy_ref or None,
			created_at=utc_now(),
		)
		self._agents[agent_key] = agent
		self._record_event(tenant_id, "harvest_agent_registered", agent.id, f"Harvest agent {agent.name} registered.", agent.name)
		return agent.to_dict()

	def change_harvest_job_state(self, tenant_id: str, job_id: str, enabled: bool, reason: str, audit_recorded: bool = True) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		job = self._require_owned(self._jobs, job_id, tenant_id, "harvest_job_not_found")
		result = self.evaluate({"tenant_context_present": True, "state_change_requested": True, "state_change_reason_present": bool(reason.strip()), "audit_event_recorded": bool(audit_recorded)})
		if result["decision"] != "allow":
			raise PermissionError(summarize_decision(result))
		job.enabled = bool(enabled)
		job.updated_at = utc_now()
		self._record_event(tenant_id, "harvest_job_state_changed", job.id, reason, job.owner)
		return job.to_dict()

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sources, tenant_id)

	def list_extractors(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._extractors, tenant_id)

	def list_jobs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._jobs, tenant_id)

	def list_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def list_results(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._results, tenant_id)

	def list_handoffs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._handoffs, tenant_id)

	def list_harvest_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [e for e in events if e.tenant_id == tenant_id]
		return [e.to_dict() for e in events]

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		sources = self.list_sources(tenant_id)
		runs = self.list_runs(tenant_id)
		tasks = [t for t in self._scheduled_tasks.values() if tenant_id is None or t["tenant_id"] == tenant_id]
		return {
			"source_count": len(sources),
			"sensitive_source_count": sum(1 for s in sources if s["sensitive_source"]),
			"extractor_count": len(self.list_extractors(tenant_id)),
			"job_count": len(self.list_jobs(tenant_id)),
			"run_count": len(runs),
			"succeeded_run_count": sum(1 for r in runs if r["status"] == "succeeded"),
			"blocked_run_count": sum(1 for r in runs if r["status"] == "blocked"),
			"result_count": len(self.list_results(tenant_id)),
			"pipeline_handoff_count": len(self.list_handoffs(tenant_id)),
			"scheduled_task_count": len(tasks),
			"agent_count": len(self.list_harvest_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
		}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		metadata = metadata or {}
		return self.register_source(
			tenant_id=tenant_id,
			name=record_id,
			source_type=metadata.get("source_type", "api"),
			endpoint=metadata.get("endpoint", f"https://example.invalid/{record_id}"),
			owner=metadata.get("owner", "system"),
			terms_evidence=metadata.get("terms_evidence", "internal_authorization"),
			credential_vault_ref=metadata.get("credential_vault_ref", "vault://scrp/default"),
			rate_limit_per_minute=metadata.get("rate_limit_per_minute", 60),
			pii_expected=metadata.get("pii_expected", False),
			pii_policy_attached=metadata.get("pii_policy_attached", metadata.get("pii_expected", False)),
			source_review_recorded=status == "active",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sources(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _rotate_proxy_internal(self, tenant_id: str, required_country: str | None = None, strategy: str = "round_robin") -> dict[str, Any] | None:
		candidates = [p for p in self._proxy_pool if p.get("tenant_id") in (tenant_id, None)]
		if required_country:
			candidates = [p for p in candidates if p.get("country") == required_country]
		if not candidates:
			return None
		if strategy == "round_robin":
			proxy = candidates[self._proxy_index % len(candidates)]
			self._proxy_index += 1
		else:
			import random
			proxy = random.choice(candidates)
		return proxy

	def _add_default_proxy(self, tenant_id: str, country: str | None) -> dict[str, Any]:
		proxy = {"address": f"proxy-{len(self._proxy_pool) + 1}.internal:8080", "country": country or "US", "tenant_id": tenant_id, "active": True}
		self._proxy_pool.append(proxy)
		return proxy

	def _require_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			self._raise_policy({"tenant_context_present": False})

	def _require_owned(self, store: dict[str, Any], object_id: str, tenant_id: str, missing_reason: str) -> Any:
		item = store.get(object_id)
		if item is None or item.tenant_id != tenant_id:
			raise KeyError(missing_reason)
		return item

	def _raise_policy(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		raise PermissionError(summarize_decision(result))

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str, severity: str = "info") -> None:
		event = ScrpAuditEvent(
			id=stable_id("evt", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			created_at=utc_now(),
		)
		self._audit_events.append(event)

	def _list(self, store: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = store.values()
		if tenant_id is not None:
			values = [v for v in values if v.tenant_id == tenant_id]
		return [v.to_dict() for v in values]


	# ------------------------------------------------------------------
	# Extended methods — 40+ total
	# ------------------------------------------------------------------

	def crawler_create(
		self,
		tenant_id: str,
		name: str,
		seed_urls: list[str],
		depth: int,
		owner: str,
		tags: list[str] | None = None,
		respect_robots: bool = True,
		pii_expected: bool = False,
		crawler_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Create a web crawler configuration.

		seed_urls: Starting URLs for the crawl.
		depth: Maximum link-follow depth.
		"""
		self._require_tenant(tenant_id)
		if not seed_urls:
			raise ValueError("seed_urls_required")
		if depth < 0:
			raise ValueError("depth_must_be_non_negative")
		cid = crawler_id or stable_id("crawler", tenant_id, name, str(len(self._scheduled_tasks)))
		record = {
			"crawler_id":      cid,
			"tenant_id":       tenant_id,
			"name":            name,
			"seed_urls":       seed_urls,
			"depth":           depth,
			"owner":           owner,
			"tags":            normalize_tags(tags),
			"respect_robots":  respect_robots,
			"pii_expected":    pii_expected,
			"status":          "created",
			"created_at":      _ts(),
		}
		if not hasattr(self, "_crawlers"):
			self._crawlers: dict[str, dict[str, Any]] = {}
		self._crawlers[cid] = record
		self._record_event(tenant_id, "crawler_created", cid, f"Crawler '{name}' created", owner)
		return record

	def crawler_schedule(
		self,
		tenant_id: str,
		crawler_id: str,
		frequency: str,
		selectors: dict[str, str] | None = None,
		auth_config: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Schedule an existing crawler to run at a given frequency.

		Wraps schedule_scrape using the first seed URL of the crawler.
		"""
		self._require_tenant(tenant_id)
		if not hasattr(self, "_crawlers"):
			self._crawlers: dict[str, dict[str, Any]] = {}
		crawler = self._crawlers.get(crawler_id)
		if crawler is None or crawler["tenant_id"] != tenant_id:
			raise KeyError(f"crawler_not_found:{crawler_id}")
		seed_url = crawler["seed_urls"][0]
		return self.schedule_scrape(
			url=seed_url,
			depth=crawler["depth"],
			frequency=frequency,
			selectors=selectors or {"content": "body"},
			auth_config=auth_config or {"type": "none"},
			tenant_id=tenant_id,
			owner=crawler["owner"],
			tags=crawler["tags"],
			pii_expected=crawler["pii_expected"],
		)

	def page_render_js(
		self,
		tenant_id: str,
		url: str,
		wait_for_selector: str,
		timeout_ms: int = 10000,
		owner: str = "system",
	) -> dict[str, Any]:
		"""JavaScript-rendered page scrape (alias for javascript_rendered_scrape)."""
		return self.javascript_rendered_scrape(
			url=url,
			wait_for_selector=wait_for_selector,
			tenant_id=tenant_id,
			timeout_ms=timeout_ms,
			owner=owner,
		)

	def element_extract(
		self,
		tenant_id: str,
		raw_html: str,
		schema: dict[str, Any],
		source_url: str = "",
		extraction_id: str | None = None,
	) -> dict[str, Any]:
		"""Extract structured elements from HTML (alias for extract_structured_data)."""
		return self.extract_structured_data(
			raw_html=raw_html,
			extraction_schema=schema,
			tenant_id=tenant_id,
			extraction_id=extraction_id,
			source_url=source_url,
		)

	def data_clean(
		self,
		tenant_id: str,
		dataset_id: str,
		operations: list[str],
		cleaned_by: str = "system",
	) -> dict[str, Any]:
		"""
		Apply a sequence of data cleaning operations to a dataset reference.

		operations: list of op names — 'trim_whitespace', 'remove_nulls',
		'normalise_unicode', 'drop_duplicates', 'validate_schema'.
		Returns a cleaning report.
		"""
		self._require_tenant(tenant_id)
		supported = {"trim_whitespace", "remove_nulls", "normalise_unicode", "drop_duplicates", "validate_schema", "lowercase_keys"}
		unsupported = [op for op in operations if op not in supported]
		if unsupported:
			raise ValueError(f"unsupported_operations:{unsupported}")
		clean_id = stable_id("clean", tenant_id, dataset_id, str(len(operations)))
		report = {
			"clean_id":        clean_id,
			"dataset_id":      dataset_id,
			"tenant_id":       tenant_id,
			"operations":      operations,
			"operations_count": len(operations),
			"cleaned_by":      cleaned_by,
			"records_affected": 0,  # synthetic — no live data in memory store
			"status":          "completed",
			"cleaned_at":      _ts(),
		}
		if not hasattr(self, "_clean_reports"):
			self._clean_reports: list[dict[str, Any]] = []
		self._clean_reports.append(report)
		self._record_event(tenant_id, "data_cleaned", clean_id, f"Cleaned {len(operations)} ops on {dataset_id}", cleaned_by)
		return report

	def entity_extract(
		self,
		tenant_id: str,
		text: str,
		entity_types: list[str],
		source_url: str = "",
		extract_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Extract named entities from text.

		entity_types: list of entity labels to detect — 'PERSON', 'ORG',
		'DATE', 'LOCATION', 'MONEY', 'EMAIL', 'PHONE'.
		Returns synthetic entity spans.
		"""
		self._require_tenant(tenant_id)
		if not text:
			raise ValueError("text_required")
		supported_types = {"PERSON", "ORG", "DATE", "LOCATION", "MONEY", "EMAIL", "PHONE", "URL"}
		unknown = [t for t in entity_types if t not in supported_types]
		if unknown:
			raise ValueError(f"unsupported_entity_types:{unknown}")
		# Synthetic extraction: detect keywords heuristically
		entities: list[dict[str, Any]] = []
		for etype in entity_types:
			probe = {"EMAIL": "@", "PHONE": "+", "URL": "http", "DATE": "/2", "MONEY": "$"}.get(etype)
			if probe and probe in text:
				start = text.index(probe)
				end = min(start + 20, len(text))
				entities.append({"type": etype, "value": text[start:end].strip(), "start": start, "end": end, "confidence": 0.85})
		eid = extract_id or stable_id("ent", tenant_id, source_url or text[:16])
		record = {
			"extract_id":    eid,
			"tenant_id":     tenant_id,
			"source_url":    source_url,
			"entity_count":  len(entities),
			"entities":      entities,
			"entity_types_requested": entity_types,
			"extracted_at":  _ts(),
		}
		if not hasattr(self, "_entity_extractions"):
			self._entity_extractions: dict[str, dict[str, Any]] = {}
		self._entity_extractions[eid] = record
		return record

	def dedup_check(
		self,
		tenant_id: str,
		dataset_id: str,
		field_keys: list[str] | None = None,
		strategy: str = "hash",
	) -> dict[str, Any]:
		"""Check for duplicates (alias for data_deduplication)."""
		return self.data_deduplication(
			dataset_id=dataset_id,
			tenant_id=tenant_id,
			field_keys=field_keys,
			strategy=strategy,
		)

	def robots_respect(
		self,
		tenant_id: str,
		domain: str,
		user_agent: str = "*",
		path: str = "/",
	) -> dict[str, Any]:
		"""
		Check whether a path is allowed for a user-agent per robots.txt rules.

		In production, fetch and parse the domain's robots.txt.
		Returns a synthetic allow/deny decision.
		"""
		self._require_tenant(tenant_id)
		if not domain:
			raise ValueError("domain_required")
		# Synthetic: disallow /admin and /private paths
		disallowed_prefixes = ("/admin", "/private", "/internal", "/.git")
		denied = any(path.startswith(p) for p in disallowed_prefixes)
		record = {
			"domain":       domain,
			"tenant_id":    tenant_id,
			"user_agent":   user_agent,
			"path":         path,
			"allowed":      not denied,
			"crawl_delay":  1.0,
			"checked_at":   _ts(),
		}
		self._record_event(tenant_id, "robots_checked", domain, f"{'denied' if denied else 'allowed'} {path}", user_agent)
		return record

	def proxy_rotate(
		self,
		tenant_id: str,
		request_id: str,
		strategy: str = "round_robin",
		required_country: str | None = None,
	) -> dict[str, Any]:
		"""Select next proxy (alias for proxy_rotation)."""
		return self.proxy_rotation(
			request_id=request_id,
			tenant_id=tenant_id,
			strategy=strategy,
			required_country=required_country,
		)

	def captcha_handle(
		self,
		tenant_id: str,
		page_url: str,
		captcha_type: str,
		solver_type: str = "third_party",
	) -> dict[str, Any]:
		"""Handle a CAPTCHA challenge (alias for captcha_handling)."""
		return self.captcha_handling(
			page_url=page_url,
			captcha_type=captcha_type,
			tenant_id=tenant_id,
			solver_type=solver_type,
		)

	def rate_throttle(
		self,
		tenant_id: str,
		domain: str,
		requests_per_minute: int,
		burst_limit: int | None = None,
		backoff_seconds: int = 30,
	) -> dict[str, Any]:
		"""Set rate throttling for a domain (alias for rate_limit_management)."""
		return self.rate_limit_management(
			domain=domain,
			requests_per_minute=requests_per_minute,
			tenant_id=tenant_id,
			burst_limit=burst_limit,
			backoff_seconds=backoff_seconds,
		)

	def data_store(
		self,
		tenant_id: str,
		dataset_id: str,
		records: list[dict[str, Any]],
		source_url: str = "",
		stored_by: str = "system",
		schema_ref: str = "",
	) -> dict[str, Any]:
		"""
		Store harvested records into the in-memory result store.

		Returns a storage manifest with record count, schema validation status,
		and a storage reference URI.
		"""
		self._require_tenant(tenant_id)
		if not dataset_id:
			raise ValueError("dataset_id_required")
		storage_ref = f"memory://{tenant_id}/{dataset_id}"
		store_id = stable_id("store", tenant_id, dataset_id, str(len(records)))
		manifest = {
			"store_id":       store_id,
			"dataset_id":     dataset_id,
			"tenant_id":      tenant_id,
			"source_url":     source_url,
			"record_count":   len(records),
			"schema_ref":     schema_ref,
			"storage_ref":    storage_ref,
			"stored_by":      stored_by,
			"schema_valid":   bool(schema_ref),
			"stored_at":      _ts(),
		}
		if not hasattr(self, "_data_stores"):
			self._data_stores: dict[str, dict[str, Any]] = {}
		self._data_stores[store_id] = manifest
		self._record_event(tenant_id, "data_stored", store_id, f"Stored {len(records)} records for {dataset_id}", stored_by)
		return manifest

	def source_monitor(
		self,
		tenant_id: str,
		source_id: str,
		check_interval_minutes: int = 60,
		alert_on_down: bool = True,
		monitored_by: str = "system",
	) -> dict[str, Any]:
		"""
		Set up monitoring for a registered harvest source.

		Returns a monitoring configuration record.
		"""
		self._require_tenant(tenant_id)
		source = self._require_owned(self._sources, source_id, tenant_id, "source_not_found")
		mon_key = f"{tenant_id}:{source_id}"
		record = {
			"monitor_id":            stable_id("mon", tenant_id, source_id),
			"source_id":             source_id,
			"source_name":           source.name,
			"tenant_id":             tenant_id,
			"check_interval_minutes": check_interval_minutes,
			"alert_on_down":         alert_on_down,
			"monitored_by":          monitored_by,
			"last_status":           "ok",
			"enabled":               True,
			"created_at":            _ts(),
		}
		if not hasattr(self, "_source_monitors"):
			self._source_monitors: dict[str, dict[str, Any]] = {}
		self._source_monitors[mon_key] = record
		self._record_event(tenant_id, "source_monitor_created", source_id, f"Monitor set up for {source.name}", monitored_by)
		return record

	def change_detect(
		self,
		tenant_id: str,
		source_id: str,
		snapshot_hash: str,
		detector_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Detect changes in a source by comparing the current snapshot hash
		to the previously stored one.

		Returns whether a change was detected and records the new snapshot.
		"""
		import hashlib
		self._require_tenant(tenant_id)
		self._require_owned(self._sources, source_id, tenant_id, "source_not_found")
		if not hasattr(self, "_change_snapshots"):
			self._change_snapshots: dict[str, str] = {}
		key = f"{tenant_id}:{source_id}"
		prev_hash = self._change_snapshots.get(key)
		changed = prev_hash is not None and prev_hash != snapshot_hash
		self._change_snapshots[key] = snapshot_hash
		did = detector_id or stable_id("chgdet", tenant_id, source_id, snapshot_hash[:8])
		record = {
			"detector_id":   did,
			"source_id":     source_id,
			"tenant_id":     tenant_id,
			"previous_hash": prev_hash,
			"current_hash":  snapshot_hash,
			"changed":       changed,
			"first_snapshot": prev_hash is None,
			"detected_at":   _ts(),
		}
		if changed:
			self._record_event(tenant_id, "source_change_detected", source_id, "Source content changed", "system")
		return record

	def scrape_analytics(
		self,
		tenant_id: str,
		period: str = "all_time",
	) -> dict[str, Any]:
		"""Return scraping analytics (alias for scraping_analytics)."""
		return self.scraping_analytics(period=period, tenant_id=tenant_id)


# Aliases
ScrpService = ScraperDataHarvestingService


def _normalize_harvest_agent_runtime(runtime: str) -> str:
	value = (runtime or "").strip().lower()
	return value if value in SUPPORTED_HARVEST_AGENT_RUNTIMES else ""


def _normalize_harvest_agent_role(role: str) -> str:
	value = (role or "").strip().lower()
	return value if value in SUPPORTED_HARVEST_AGENT_ROLES else ""
