"""Service layer for the Website Builder capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_WSBL_AGENT_ROLES,
	SUPPORTED_WSBL_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .website_runtime import (
	WebsiteAgentRecord,
	WebsiteAuditEventRecord,
	WebsiteComponentRecord,
	WebsiteDomainRecord,
	WebsitePageRecord,
	WebsitePublishRequestRecord,
	WebsiteSiteRecord,
	stable_id,
	utc_now,
)


class WsblService:
	"""Dependency-light website-builder runtime behind the capability contract."""

	def __init__(self) -> None:
		self._sites: dict[str, WebsiteSiteRecord] = {}
		self._domains: dict[str, WebsiteDomainRecord] = {}
		self._components: dict[str, WebsiteComponentRecord] = {}
		self._pages: dict[str, WebsitePageRecord] = {}
		self._publish_requests: dict[str, WebsitePublishRequestRecord] = {}
		self._agents: dict[str, WebsiteAgentRecord] = {}
		self._audit_events: list[WebsiteAuditEventRecord] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_site(
		self,
		site_key: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		primary_domain: str | None = None,
		locale: str = "en",
		public_site: bool = True,
		privacy_banner_required: bool = True,
		domain_validated: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation": "create_site", "site_owner_assigned": bool(owner_id)})
		site_id = stable_id("site", tenant_id, site_key)
		required_actions: list[str] = []
		domains: list[str] = []
		status = "ready"
		if primary_domain:
			domain = self.register_domain(
				site_id=site_id,
				tenant_id=tenant_id,
				domain=primary_domain,
				validated=domain_validated,
				actor_id=owner_id,
			)
			domains.append(domain["domain"])
			if not domain_validated:
				required_actions.append("validate_domain")
				status = "domain_pending"
		site = WebsiteSiteRecord(
			id=site_id,
			tenant_id=tenant_id,
			name=name,
			owner_id=owner_id,
			locale=locale,
			public_site=public_site,
			privacy_banner_required=privacy_banner_required,
			status=status,
			domains=domains,
			required_actions=required_actions,
			metadata=dict(metadata or {}),
		)
		self._sites[site_id] = site
		self._audit(tenant_id, "site_created", site_id, owner_id, {"status": status, "domain_count": len(domains)})
		return site.to_dict()

	def register_domain(
		self,
		site_id: str,
		tenant_id: str,
		domain: str,
		validated: bool = False,
		actor_id: str = "system",
		validation_method: str = "dns_txt",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		domain_id = stable_id("domain", tenant_id, site_id, domain)
		record = WebsiteDomainRecord(
			id=domain_id,
			tenant_id=tenant_id,
			site_id=site_id,
			domain=domain,
			validated=validated,
			validation_method=validation_method,
			validated_at=utc_now() if validated else None,
		)
		self._domains[domain_id] = record
		if site_id in self._sites and domain not in self._sites[site_id].domains:
			self._sites[site_id].domains.append(domain)
			self._sites[site_id].updated_at = utc_now()
		self._audit(tenant_id, "domain_registered", domain_id, actor_id, {"site_id": site_id, "validated": validated})
		return record.to_dict()

	def validate_domain(self, domain_id: str, actor_id: str) -> dict[str, Any]:
		domain = self._get_domain(domain_id)
		domain.validated = True
		domain.validated_at = utc_now()
		site = self._sites.get(domain.site_id)
		if site:
			site.required_actions = [action for action in site.required_actions if action != "validate_domain"]
			site.status = "ready" if not site.required_actions else site.status
			site.updated_at = utc_now()
		self._audit(domain.tenant_id, "domain_validated", domain.id, actor_id, {"site_id": domain.site_id})
		return domain.to_dict()

	def create_component(
		self,
		component_key: str,
		tenant_id: str,
		name: str,
		component_type: str = "section",
		custom: bool = False,
		reviewed: bool = False,
		reviewed_by: str | None = None,
		policy_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_component",
			"custom_component_present": bool(custom),
			"component_review_recorded": bool(reviewed),
		})
		self._raise_if_denied(result)
		component_id = stable_id("component", tenant_id, component_key)
		status = "approved" if custom and reviewed else "review_required" if result["decision"] == "require_review" else "available"
		record = WebsiteComponentRecord(
			id=component_id,
			tenant_id=tenant_id,
			name=name,
			component_type=component_type,
			custom=custom,
			status=status,
			reviewed_by=reviewed_by if reviewed else None,
			reviewed_at=utc_now() if reviewed else None,
			policy_id=policy_id,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result, reviewed),
			metadata=dict(metadata or {}),
		)
		self._components[component_id] = record
		self._audit(tenant_id, "component_created", component_id, reviewed_by or "system", {"custom": custom, "status": status}, policy_result=result)
		return record.to_dict()

	def review_component(self, component_id: str, reviewer_id: str, policy_id: str | None = None) -> dict[str, Any]:
		component = self._get_component(component_id)
		self._enforce_context({
			"tenant_context_present": bool(component.tenant_id),
			"operation": "review_component",
			"custom_component_present": bool(component.custom),
			"component_policy_attached": bool(policy_id or component.policy_id),
		})
		component.status = "approved"
		component.reviewed_by = reviewer_id
		component.reviewed_at = utc_now()
		component.policy_id = policy_id or component.policy_id
		component.decision = "allow"
		component.matched_rules = []
		component.review_reasons = []
		component.audit_evidence = {"required_actions": [], "reasons": [], "review_recorded": True}
		self._audit(component.tenant_id, "component_reviewed", component.id, reviewer_id, {"policy_id": component.policy_id})
		return component.to_dict()

	def create_page(
		self,
		site_id: str,
		slug: str,
		title: str,
		tenant_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		site = self._get_site(site_id)
		if tenant_id is not None and site.tenant_id != tenant_id:
			raise ValueError("site_tenant_mismatch")
		page_id = stable_id("page", site.tenant_id, site_id, slug)
		page = WebsitePageRecord(
			id=page_id,
			tenant_id=site.tenant_id,
			site_id=site_id,
			slug=slug,
			title=title,
			metadata=dict(metadata or {}),
		)
		self._pages[page_id] = page
		self._audit(site.tenant_id, "page_created", page_id, site.owner_id, {"site_id": site_id, "slug": slug})
		return page.to_dict()

	def add_page_section(
		self,
		page_id: str,
		component_id: str,
		content: dict[str, Any] | None = None,
		position: int | None = None,
		actor_id: str = "system",
	) -> dict[str, Any]:
		page = self._get_page(page_id)
		component = self._get_component(component_id)
		if component.tenant_id != page.tenant_id:
			raise ValueError("component_tenant_mismatch")
		if component.custom and component.status != "approved":
			self._enforce_context({
				"tenant_context_present": True,
				"operation": "add_page_section",
				"custom_component_present": True,
				"component_review_recorded": False,
			})
		section = {
			"id": stable_id("section", page.id, component.id, len(page.sections) + 1),
			"component_id": component.id,
			"component_name": component.name,
			"content": dict(content or {}),
			"position": position if position is not None else len(page.sections) + 1,
		}
		page.sections.append(section)
		page.version += 1
		page.status = "review_ready"
		page.updated_at = utc_now()
		self._audit(page.tenant_id, "page_section_added", page.id, actor_id, {"component_id": component.id})
		return page.to_dict()

	def create_publish_request(
		self,
		site_id: str,
		requested_by: str,
		environment: str = "production",
		approval_recorded: bool = False,
		accessibility_passed: bool = False,
		consent_policy_attached: bool = False,
		preview_evidence_present: bool = True,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		site = self._get_site(site_id)
		domain_validation_complete = all(
			domain.validated
			for domain in self._domains.values()
			if domain.tenant_id == site.tenant_id and domain.site_id == site.id
		)
		structured_sections_present = any(
			page.sections
			for page in self._pages.values()
			if page.tenant_id == site.tenant_id and page.site_id == site.id
		)
		result = self.evaluate({
			"tenant_context_present": bool(site.tenant_id),
			"operation": "publish_site",
			"domain_validation_complete": domain_validation_complete,
			"structured_sections_present": structured_sections_present,
			"preview_evidence_present": bool(preview_evidence_present),
			"approval_recorded": approval_recorded,
			"event_stream": self._normalize_token(event_stream),
			"public_site": site.public_site,
			"accessibility_passed": accessibility_passed,
			"privacy_banner_required": site.privacy_banner_required,
			"consent_policy_attached": consent_policy_attached,
		})
		deny_reasons = [action.get("reason", "capability_policy_blocked") for action in result["actions"] if action.get("decision") == "deny"]
		required_actions = [
			action.get("required_action", "review_required")
			for action in result["actions"]
			if action.get("required_action")
		]
		status = "denied" if deny_reasons else "review_required" if result["decision"] == "require_review" else "approved"
		request_id = stable_id("publish", site.tenant_id, site.id, environment, site.published_version + 1)
		record = WebsitePublishRequestRecord(
			id=request_id,
			tenant_id=site.tenant_id,
			site_id=site.id,
			requested_by=requested_by,
			environment=environment,
			status=status,
			approval_recorded=approval_recorded,
			accessibility_passed=accessibility_passed,
			consent_policy_attached=consent_policy_attached,
			required_actions=required_actions,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result, approval_recorded and accessibility_passed and consent_policy_attached),
		)
		self._publish_requests[request_id] = record
		self._audit(
			site.tenant_id,
			"publish_request_denied" if deny_reasons else "publish_request_created",
			request_id,
			requested_by,
			{"status": status, "required_actions": required_actions, "event_stream": self._normalize_token(event_stream)},
			policy_result=result,
		)
		if deny_reasons:
			raise PermissionError(", ".join(deny_reasons))
		return record.to_dict()

	def publish_site(self, publish_request_id: str, actor_id: str) -> dict[str, Any]:
		request = self._get_publish_request(publish_request_id)
		if request.status != "approved":
			raise PermissionError("publish_request_not_approved")
		site = self._get_site(request.site_id)
		site.published_version += 1
		site.status = "published"
		site.updated_at = utc_now()
		request.status = "published"
		request.published_version = site.published_version
		request.published_at = utc_now()
		for page in self._pages.values():
			if page.site_id == site.id and page.tenant_id == site.tenant_id:
				page.status = "published"
				page.updated_at = utc_now()
		self._audit(site.tenant_id, "site_published", site.id, actor_id, {"publish_request_id": request.id, "version": site.published_version})
		return {"site": site.to_dict(), "publish_request": request.to_dict()}

	def rollback_site(self, site_id: str, version: int, actor_id: str, event_stream: str = "bytewax") -> dict[str, Any]:
		site = self._get_site(site_id)
		self._enforce_context({
			"tenant_context_present": bool(site.tenant_id),
			"operation": "rollback_site",
			"event_stream": self._normalize_token(event_stream),
		})
		if version < 0 or version > site.published_version:
			raise ValueError("invalid_rollback_version")
		site.published_version = version
		site.status = "published" if version else "ready"
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "site_rolled_back", site.id, actor_id, {"version": version, "event_stream": self._normalize_token(event_stream)})
		return site.to_dict()

	def register_wsbl_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "platform",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_wsbl_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_WSBL_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_WSBL_AGENT_ROLES,
		}
		self._enforce_context(context)
		record = WebsiteAgentRecord(
			id=stable_id("agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
			audit_evidence={"required_actions": [], "reasons": [], "review_recorded": bool(human_approval_required)},
		)
		self._agents[record.id] = record
		self._audit(
			tenant_id,
			"wsbl_agent_registered",
			record.id,
			owner,
			{"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_publish_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool = False,
		human_approval_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self._agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"wsbl_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_publish_action",
			"agent_id": agent_id,
			"agent_role": agent.role,
			"action": action,
			"privileged_scope": bool(privileged_scope),
			"human_approval_recorded": bool(str(human_approval_ref or "").strip()),
		}
		result = self.evaluate(context)
		self._audit(
			tenant_id,
			"wsbl_agent_publish_action_validated",
			agent_id,
			agent.owner,
			{"action": action, "privileged_scope": bool(privileged_scope), "human_approval_recorded": bool(str(human_approval_ref or "").strip())},
			policy_result=result,
		)
		return self._policy_payload(result)

	def validate_batch_publish(
		self,
		tenant_id: str,
		site_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "batch_publish",
			"site_count": int(site_count),
			"event_stream": self._normalize_token(event_stream),
		}
		result = self.evaluate(context)
		self._audit(
			tenant_id,
			"batch_publish_validated",
			stable_id("batch_publish", tenant_id, site_count, self._normalize_token(event_stream)),
			"system",
			{"site_count": int(site_count), "event_stream": self._normalize_token(event_stream)},
			policy_result=result,
		)
		return self._policy_payload(result)

	def list_sites(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [site.to_dict() for site in sorted(self._filter(self._sites.values(), tenant_id), key=lambda item: item.name)]

	def list_pages(self, tenant_id: str | None = None, site_id: str | None = None) -> list[dict[str, Any]]:
		pages = self._filter(self._pages.values(), tenant_id)
		if site_id is not None:
			pages = [page for page in pages if page.site_id == site_id]
		return [page.to_dict() for page in sorted(pages, key=lambda item: (item.site_id, item.slug))]

	def list_components(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [component.to_dict() for component in sorted(self._filter(self._components.values(), tenant_id), key=lambda item: item.name)]

	def list_publish_requests(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [request.to_dict() for request in sorted(self._filter(self._publish_requests.values(), tenant_id), key=lambda item: item.created_at)]

	def list_domains(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [domain.to_dict() for domain in sorted(self._filter(self._domains.values(), tenant_id), key=lambda item: item.domain)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._filter(self._audit_events, tenant_id)
		return [event.to_dict() for event in sorted(events, key=lambda item: item.created_at)]

	def list_wsbl_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [agent.to_dict() for agent in sorted(self._filter(self._agents.values(), tenant_id), key=lambda item: item.name)]

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [
			item
			for item in (self.list_components(tenant_id) + self.list_publish_requests(tenant_id))
			if item.get("status") == "review_required"
		]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		sites = self.list_sites(tenant_id)
		pages = self.list_pages(tenant_id)
		components = self.list_components(tenant_id)
		requests = self.list_publish_requests(tenant_id)
		pending_reviews = self.list_pending_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"site_count": len(sites),
			"published_site_count": sum(1 for site in sites if site["status"] == "published"),
			"page_count": len(pages),
			"component_count": len(components),
			"custom_component_count": sum(1 for component in components if component["custom"]),
			"pending_component_review_count": sum(1 for component in components if component["status"] == "review_required"),
			"publish_request_count": len(requests),
			"publish_review_count": sum(1 for request in requests if request["status"] == "review_required"),
			"denied_publish_request_count": sum(1 for request in requests if request["status"] == "denied"),
			"pending_review_count": len(pending_reviews),
			"wsbl_agent_count": len(self.list_wsbl_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		return self.create_site(
			site_key=record_id,
			tenant_id=tenant_id,
			name=str((metadata or {}).get("name") or record_id),
			owner_id=str((metadata or {}).get("owner_id") or "system"),
			public_site=bool((metadata or {}).get("public_site", True)),
			privacy_banner_required=bool((metadata or {}).get("privacy_banner_required", True)),
			metadata={"compatibility_status": status, **dict(metadata or {})},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_sites(tenant_id)

	def _enforce_context(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		self._raise_if_denied(result)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] != "deny":
			return
		reasons = [action.get("reason", "capability_policy_blocked") for action in result["actions"]]
		raise PermissionError(", ".join(reasons) or "capability_policy_blocked")

	def _require_tenant(self, tenant_id: str) -> None:
		self._enforce_context({"tenant_context_present": bool(tenant_id)})

	def _get_site(self, site_id: str) -> WebsiteSiteRecord:
		try:
			return self._sites[site_id]
		except KeyError as exc:
			raise KeyError(f"site_not_found:{site_id}") from exc

	def _get_domain(self, domain_id: str) -> WebsiteDomainRecord:
		try:
			return self._domains[domain_id]
		except KeyError as exc:
			raise KeyError(f"domain_not_found:{domain_id}") from exc

	def _get_component(self, component_id: str) -> WebsiteComponentRecord:
		try:
			return self._components[component_id]
		except KeyError as exc:
			raise KeyError(f"component_not_found:{component_id}") from exc

	def _get_page(self, page_id: str) -> WebsitePageRecord:
		try:
			return self._pages[page_id]
		except KeyError as exc:
			raise KeyError(f"page_not_found:{page_id}") from exc

	def _get_publish_request(self, request_id: str) -> WebsitePublishRequestRecord:
		try:
			return self._publish_requests[request_id]
		except KeyError as exc:
			raise KeyError(f"publish_request_not_found:{request_id}") from exc

	def _audit(
		self,
		tenant_id: str,
		action: str,
		subject_id: str,
		actor_id: str,
		details: dict[str, Any] | None = None,
		policy_result: dict[str, Any] | None = None,
	) -> None:
		policy_result = policy_result or {"decision": "allow", "matched_rules": [], "actions": []}
		event = WebsiteAuditEventRecord(
			id=stable_id("audit", tenant_id, action, subject_id, len(self._audit_events) + 1),
			tenant_id=tenant_id,
			action=action,
			subject_id=subject_id,
			actor_id=actor_id,
			details=dict(details or {}),
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._review_reasons(policy_result),
			audit_evidence=self._audit_evidence(policy_result),
		)
		self._audit_events.append(event)

	def _policy_payload(self, result: dict[str, Any]) -> dict[str, Any]:
		payload = dict(result)
		payload["matched_rules"] = list(result.get("matched_rules", []))
		payload["actions"] = [dict(action) for action in result.get("actions", [])]
		payload["review_reasons"] = self._review_reasons(result)
		payload["audit_evidence"] = self._audit_evidence(result)
		return payload

	def _review_reasons(self, result: dict[str, Any]) -> list[str]:
		if result["decision"] == "allow":
			return []
		return [
			action.get("reason", "website_builder_policy_blocked")
			for action in result.get("actions", [])
		]

	def _audit_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				action["required_action"]
				for action in result.get("actions", [])
				if action.get("required_action")
			],
			"reasons": [
				action.get("reason", "website_builder_policy_blocked")
				for action in result.get("actions", [])
			],
			"review_recorded": bool(review_recorded),
		}

	@staticmethod
	def _filter(records: Any, tenant_id: str | None) -> list[Any]:
		items = list(records)
		if tenant_id is None:
			return items
		return [record for record in items if record.tenant_id == tenant_id]

	@staticmethod
	def _normalize_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

	# ── Extended methods ───────────────────────────────────────────────────────

	def page_create(
		self,
		site_id: str,
		slug: str,
		title: str,
		tenant_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Spec alias for create_page."""
		return self.create_page(site_id, slug, title, tenant_id, metadata)

	def page_publish(self, page_id: str, actor_id: str) -> dict[str, Any]:
		"""Mark a single page as published."""
		page = self._get_page(page_id)
		page.status = "published"
		page.updated_at = utc_now()
		self._audit(page.tenant_id, "page_published", page_id, actor_id, {"slug": page.slug})
		return page.to_dict()

	def page_unpublish(self, page_id: str, actor_id: str) -> dict[str, Any]:
		"""Revert a page to draft status."""
		page = self._get_page(page_id)
		page.status = "draft"
		page.updated_at = utc_now()
		self._audit(page.tenant_id, "page_unpublished", page_id, actor_id, {"slug": page.slug})
		return page.to_dict()

	def component_add(
		self,
		component_key: str,
		tenant_id: str,
		name: str,
		component_type: str = "section",
		custom: bool = False,
	) -> dict[str, Any]:
		"""Spec alias for create_component."""
		return self.create_component(component_key, tenant_id, name, component_type, custom)

	def template_apply(
		self,
		site_id: str,
		template_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Apply a named template to a site (recorded as metadata)."""
		site = self._get_site(site_id)
		site.metadata["applied_template"] = template_id
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "template_applied", site_id, actor_id, {"template_id": template_id})
		return site.to_dict()

	def domain_bind(
		self,
		site_id: str,
		domain: str,
		actor_id: str,
		validated: bool = False,
	) -> dict[str, Any]:
		"""Bind a domain to a site (alias for register_domain)."""
		site = self._get_site(site_id)
		return self.register_domain(site_id, site.tenant_id, domain, validated, actor_id)

	def seo_optimise(
		self,
		page_id: str,
		meta_title: str,
		meta_description: str,
		keywords: list[str],
		actor_id: str,
	) -> dict[str, Any]:
		"""Attach SEO metadata to a page."""
		page = self._get_page(page_id)
		page.metadata.update({
			"seo_meta_title": meta_title,
			"seo_meta_description": meta_description,
			"seo_keywords": keywords,
		})
		page.updated_at = utc_now()
		self._audit(page.tenant_id, "seo_optimised", page_id, actor_id, {"meta_title": meta_title})
		return page.to_dict()

	def form_embed(
		self,
		page_id: str,
		form_id: str,
		form_type: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Embed a form reference into a page's metadata."""
		page = self._get_page(page_id)
		page.metadata.setdefault("embedded_forms", []).append({"form_id": form_id, "form_type": form_type})
		page.updated_at = utc_now()
		self._audit(page.tenant_id, "form_embedded", page_id, actor_id, {"form_id": form_id})
		return page.to_dict()

	def analytics_embed(
		self,
		site_id: str,
		provider: str,
		tracking_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Attach an analytics provider tracking ID to a site."""
		site = self._get_site(site_id)
		site.metadata.update({"analytics_provider": provider, "analytics_tracking_id": tracking_id})
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "analytics_embedded", site_id, actor_id, {"provider": provider})
		return site.to_dict()

	def media_upload(
		self,
		site_id: str,
		media_id: str,
		file_ref: str,
		media_type: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Register a media asset reference against a site."""
		site = self._get_site(site_id)
		media: dict[str, Any] = {
			"id": media_id,
			"site_id": site_id,
			"tenant_id": site.tenant_id,
			"file_ref": file_ref,
			"media_type": media_type,
			"uploaded_at": utc_now(),
		}
		site.metadata.setdefault("media", []).append({"id": media_id, "file_ref": file_ref, "media_type": media_type})
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "media_uploaded", site_id, actor_id, {"media_id": media_id, "media_type": media_type})
		return media

	def css_customise(
		self,
		site_id: str,
		css_snippet: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Attach or replace custom CSS for a site."""
		site = self._get_site(site_id)
		site.metadata["custom_css"] = css_snippet
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "css_customised", site_id, actor_id, {"css_length": len(css_snippet)})
		return site.to_dict()

	def mobile_preview(self, site_id: str, actor_id: str) -> dict[str, Any]:
		"""Generate a mobile-preview record for a site."""
		site = self._get_site(site_id)
		preview: dict[str, Any] = {
			"site_id": site_id,
			"tenant_id": site.tenant_id,
			"viewport": "375x812",
			"generated_at": utc_now(),
			"status": "ready",
		}
		self._audit(site.tenant_id, "mobile_preview_generated", site_id, actor_id, {})
		return preview

	def ab_test_page(
		self,
		site_id: str,
		page_a_id: str,
		page_b_id: str,
		test_id: str,
		split_pct: int,
		actor_id: str,
	) -> dict[str, Any]:
		"""Register an A/B test between two pages."""
		site = self._get_site(site_id)
		self._get_page(page_a_id)
		self._get_page(page_b_id)
		ab_record: dict[str, Any] = {
			"id": test_id,
			"site_id": site_id,
			"tenant_id": site.tenant_id,
			"page_a_id": page_a_id,
			"page_b_id": page_b_id,
			"split_pct": max(0, min(100, split_pct)),
			"status": "running",
			"created_at": utc_now(),
		}
		site.metadata.setdefault("ab_tests", []).append(ab_record)
		site.updated_at = utc_now()
		self._audit(site.tenant_id, "ab_test_created", site_id, actor_id, {"test_id": test_id})
		return ab_record

	def sitemap_generate(self, site_id: str, actor_id: str) -> dict[str, Any]:
		"""Generate an XML sitemap record for a site."""
		site = self._get_site(site_id)
		pages = [p for p in self._pages.values() if p.site_id == site_id and p.tenant_id == site.tenant_id]
		urls = [{"loc": f"/{p.slug}", "lastmod": p.updated_at} for p in pages]
		sitemap: dict[str, Any] = {
			"site_id": site_id,
			"tenant_id": site.tenant_id,
			"url_count": len(urls),
			"urls": urls,
			"generated_at": utc_now(),
		}
		self._audit(site.tenant_id, "sitemap_generated", site_id, actor_id, {"url_count": len(urls)})
		return sitemap

	def site_export(self, site_id: str, actor_id: str, format: str = "json") -> dict[str, Any]:
		"""Export a full site bundle (pages + components + metadata)."""
		site = self._get_site(site_id)
		pages = self.list_pages(site.tenant_id, site_id)
		return {
			"site": site.to_dict(),
			"pages": pages,
			"format": format,
			"exported_at": utc_now(),
		}
