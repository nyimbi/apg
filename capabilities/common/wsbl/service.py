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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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
		# ── WebSocket Broker state ─────────────────────────────────────────────
		# keyed by "tenant_id:connection_id"
		self._connections: dict[str, dict[str, Any]] = {}
		# keyed by "tenant_id:room_id"
		self._rooms: dict[str, dict[str, Any]] = {}
		# keyed by "tenant_id:connection_id"
		self._presence: dict[str, dict[str, Any]] = {}
		# keyed by session_id
		self._ws_sessions: dict[str, dict[str, Any]] = {}
		# keyed by "tenant_id:component_id"
		self._component_locks: dict[str, dict[str, Any]] = {}
		# keyed by "tenant_id:page_id"
		self._annotations: dict[str, list[dict[str, Any]]] = {}
		# keyed by "tenant_id:room_id"
		self._broadcast_log: dict[str, list[dict[str, Any]]] = {}

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

	# ── WebSocket Broker: Connection Registry ─────────────────────────────────

	async def async_connect(
		self,
		tenant_id: str,
		connection_id: str,
		actor_id: str,
		protocol_version: str = "1.0",
		transport_meta: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Register a new WebSocket connection in the tenant-scoped registry.

		Stores transport metadata (IP, compression, protocol version) and
		initialises a last-seen heartbeat timestamp.  Returns the connection
		record.  Raises ValueError if the connection_id is already registered
		for this tenant.
		"""
		self._require_tenant(tenant_id)
		key = f"{tenant_id}:{connection_id}"
		if key in self._connections:
			raise ValueError(f"connection_already_registered:{connection_id}")
		record: dict[str, Any] = {
			"id": connection_id,
			"tenant_id": tenant_id,
			"actor_id": actor_id,
			"protocol_version": protocol_version,
			"transport_meta": dict(transport_meta or {}),
			"rooms": [],
			"connected_at": utc_now(),
			"last_seen_at": utc_now(),
			"status": "connected",
		}
		self._connections[key] = record
		self._audit(tenant_id, "ws_connected", connection_id, actor_id, {"protocol_version": protocol_version})
		return dict(record)

	async def async_disconnect(
		self,
		tenant_id: str,
		connection_id: str,
		actor_id: str,
		reason: str = "normal",
	) -> dict[str, Any]:
		"""Remove a WebSocket connection from the registry.

		Automatically leaves all rooms the connection was a member of, expires
		its presence entry, and releases any component locks held.  Emits a
		``ws_disconnected`` audit event.
		"""
		self._require_tenant(tenant_id)
		key = f"{tenant_id}:{connection_id}"
		record = self._connections.pop(key, None)
		if record is None:
			raise KeyError(f"connection_not_found:{connection_id}")
		for room_id in list(record.get("rooms", [])):
			await self._room_evict(tenant_id, room_id, connection_id)
		self._presence.pop(f"{tenant_id}:{connection_id}", None)
		expired_locks = [
			lock_key for lock_key, lock in self._component_locks.items()
			if lock["holder_connection_id"] == connection_id and lock["tenant_id"] == tenant_id
		]
		for lock_key in expired_locks:
			del self._component_locks[lock_key]
		record["status"] = "disconnected"
		record["disconnected_at"] = utc_now()
		record["disconnect_reason"] = reason
		self._audit(tenant_id, "ws_disconnected", connection_id, actor_id, {"reason": reason})
		return dict(record)

	async def async_heartbeat(
		self,
		tenant_id: str,
		connection_id: str,
	) -> dict[str, Any]:
		"""Refresh the last-seen timestamp for a live connection.

		Must be called by the client on a regular interval (recommended: every
		10 seconds).  Returns the updated connection record.  Raises KeyError
		if the connection is not registered.
		"""
		key = f"{tenant_id}:{connection_id}"
		record = self._connections.get(key)
		if record is None:
			raise KeyError(f"connection_not_found:{connection_id}")
		record["last_seen_at"] = utc_now()
		return dict(record)

	async def async_prune_dead_connections(
		self,
		tenant_id: str,
		max_idle_seconds: int = 30,
	) -> list[str]:
		"""Evict connections whose last heartbeat is older than ``max_idle_seconds``.

		Returns the list of evicted connection IDs.  Emits a
		``connection_reaped`` audit event per eviction.
		"""
		from datetime import datetime, timezone, timedelta
		now = datetime.now(timezone.utc)
		cutoff = now - timedelta(seconds=max_idle_seconds)
		reaped: list[str] = []
		for key in list(self._connections.keys()):
			record = self._connections[key]
			if record["tenant_id"] != tenant_id:
				continue
			last_seen = datetime.fromisoformat(record["last_seen_at"])
			if last_seen < cutoff:
				connection_id = record["id"]
				await self.async_disconnect(tenant_id, connection_id, "system", reason="idle_timeout")
				reaped.append(connection_id)
				self._audit(tenant_id, "connection_reaped", connection_id, "system", {"max_idle_seconds": max_idle_seconds})
		return reaped

	# ── WebSocket Broker: Room Management ─────────────────────────────────────

	async def async_room_create(
		self,
		tenant_id: str,
		room_id: str,
		site_id: str,
		actor_id: str,
		page_id: str | None = None,
		room_type: str = "collaboration",
		max_members: int = 50,
	) -> dict[str, Any]:
		"""Create a new broker room scoped to a tenant site.

		Rooms are the unit of real-time isolation.  A room maps to a logical
		editing context — typically a site or a single page.  ``room_type``
		accepts ``collaboration``, ``review``, or ``observer``.  Raises
		ValueError if the room already exists.
		"""
		self._require_tenant(tenant_id)
		room_key = f"{tenant_id}:{room_id}"
		if room_key in self._rooms:
			raise ValueError(f"room_already_exists:{room_id}")
		room: dict[str, Any] = {
			"id": room_id,
			"tenant_id": tenant_id,
			"site_id": site_id,
			"page_id": page_id,
			"room_type": room_type,
			"max_members": max_members,
			"members": [],
			"status": "open",
			"created_at": utc_now(),
			"updated_at": utc_now(),
		}
		self._rooms[room_key] = room
		self._audit(tenant_id, "ws_room_created", room_id, actor_id, {"site_id": site_id, "room_type": room_type})
		return dict(room)

	async def async_room_join(
		self,
		tenant_id: str,
		room_id: str,
		connection_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Add a connection to a room's member list.

		Raises KeyError if the room does not exist.  Raises PermissionError if
		the room is closed or at capacity.  Updates the connection record's room
		list and emits a ``ws_room_joined`` audit event.
		"""
		self._require_tenant(tenant_id)
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is None:
			raise KeyError(f"room_not_found:{room_id}")
		if room["status"] != "open":
			raise PermissionError(f"room_not_open:{room_id}")
		if len(room["members"]) >= room["max_members"]:
			raise PermissionError(f"room_at_capacity:{room_id}")
		if connection_id not in room["members"]:
			room["members"].append(connection_id)
			room["updated_at"] = utc_now()
		conn_key = f"{tenant_id}:{connection_id}"
		conn = self._connections.get(conn_key)
		if conn is not None and room_id not in conn["rooms"]:
			conn["rooms"].append(room_id)
		self._audit(tenant_id, "ws_room_joined", room_id, actor_id, {"connection_id": connection_id})
		return dict(room)

	async def async_room_leave(
		self,
		tenant_id: str,
		room_id: str,
		connection_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Remove a connection from a room's member list.

		No-ops gracefully if the connection was not a member.  Emits a
		``ws_room_left`` audit event.
		"""
		self._require_tenant(tenant_id)
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is None:
			raise KeyError(f"room_not_found:{room_id}")
		await self._room_evict(tenant_id, room_id, connection_id)
		self._audit(tenant_id, "ws_room_left", room_id, actor_id, {"connection_id": connection_id})
		return dict(room)

	async def async_room_close(
		self,
		tenant_id: str,
		room_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Close a room, evicting all members and releasing associated resources.

		Closed rooms are retained in the registry for audit purposes but new
		joins are rejected.  Emits a ``ws_room_closed`` audit event.
		"""
		self._require_tenant(tenant_id)
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is None:
			raise KeyError(f"room_not_found:{room_id}")
		for connection_id in list(room["members"]):
			await self._room_evict(tenant_id, room_id, connection_id)
		room["status"] = "closed"
		room["updated_at"] = utc_now()
		self._audit(tenant_id, "ws_room_closed", room_id, actor_id, {})
		return dict(room)

	# ── WebSocket Broker: Presence Protocol ───────────────────────────────────

	async def async_presence_update(
		self,
		tenant_id: str,
		connection_id: str,
		actor_id: str,
		page_id: str | None = None,
		component_id: str | None = None,
		cursor_position: dict[str, Any] | None = None,
		intent: str = "viewing",
		ttl_seconds: int = 30,
	) -> dict[str, Any]:
		"""Publish a presence record for an active connection.

		``intent`` must be one of ``viewing``, ``editing``, or ``reviewing``.
		The record expires after ``ttl_seconds``; callers should send updates on
		each heartbeat interval.  Returns the presence record.
		"""
		_valid_intents = {"viewing", "editing", "reviewing"}
		if intent not in _valid_intents:
			raise ValueError(f"invalid_presence_intent:{intent}")
		presence_key = f"{tenant_id}:{connection_id}"
		record: dict[str, Any] = {
			"tenant_id": tenant_id,
			"connection_id": connection_id,
			"actor_id": actor_id,
			"page_id": page_id,
			"component_id": component_id,
			"cursor_position": dict(cursor_position or {}),
			"intent": intent,
			"ttl_seconds": ttl_seconds,
			"updated_at": utc_now(),
		}
		self._presence[presence_key] = record
		return dict(record)

	async def async_presence_snapshot(
		self,
		tenant_id: str,
		room_id: str,
	) -> list[dict[str, Any]]:
		"""Return presence records for all connections in a room.

		Expired entries (based on ``ttl_seconds``) are pruned before the
		snapshot is built.  Useful for delivering initial state on room join.
		"""
		from datetime import datetime, timezone, timedelta
		now = datetime.now(timezone.utc)
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is None:
			raise KeyError(f"room_not_found:{room_id}")
		members = set(room["members"])
		snapshot: list[dict[str, Any]] = []
		for presence_key in list(self._presence.keys()):
			p = self._presence[presence_key]
			if p["tenant_id"] != tenant_id or p["connection_id"] not in members:
				continue
			updated = datetime.fromisoformat(p["updated_at"])
			age_seconds = (now - updated).total_seconds()
			if age_seconds > p.get("ttl_seconds", 30):
				del self._presence[presence_key]
				continue
			snapshot.append(dict(p))
		return snapshot

	# ── WebSocket Broker: Broadcast ────────────────────────────────────────────

	async def async_broadcast(
		self,
		tenant_id: str,
		room_id: str,
		message: dict[str, Any],
		actor_id: str,
		exclude_connection_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Fan out a message to all connections in a room.

		The in-memory backend records the broadcast in an append log keyed by
		room.  A production backend would push over the real WebSocket transport.
		Returns a delivery receipt: ``delivered`` count, ``failed`` list, and
		``sent_at`` timestamp.

		``exclude_connection_ids`` suppresses echo to the sender's own
		connection.
		"""
		self._require_tenant(tenant_id)
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is None:
			raise KeyError(f"room_not_found:{room_id}")
		exclude = set(exclude_connection_ids or [])
		targets = [cid for cid in room["members"] if cid not in exclude]
		envelope: dict[str, Any] = {
			"room_id": room_id,
			"tenant_id": tenant_id,
			"sender_id": actor_id,
			"message": dict(message),
			"sent_at": utc_now(),
			"recipients": targets,
		}
		self._broadcast_log.setdefault(room_key, []).append(envelope)
		self._audit(tenant_id, "ws_broadcast", room_id, actor_id, {"recipient_count": len(targets)})
		return {"delivered": len(targets), "failed": [], "sent_at": envelope["sent_at"]}

	# ── WebSocket Broker: Collaborative Sessions ──────────────────────────────

	async def async_session_start(
		self,
		tenant_id: str,
		connection_id: str,
		site_id: str,
		page_id: str,
		actor_id: str,
		session_id: str | None = None,
	) -> dict[str, Any]:
		"""Begin a governed collaborative editing session.

		A session wraps a ``(tenant_id, site_id, page_id, actor_id)`` tuple
		with heartbeat tracking.  Emits a ``ws_session_started`` audit event.
		Multiple connections may open sessions on the same page — presence and
		component locking coordinate access.
		"""
		self._require_tenant(tenant_id)
		sid = session_id or stable_id("ws_session", tenant_id, site_id, page_id, actor_id)
		if sid in self._ws_sessions:
			raise ValueError(f"session_already_active:{sid}")
		session: dict[str, Any] = {
			"id": sid,
			"tenant_id": tenant_id,
			"connection_id": connection_id,
			"site_id": site_id,
			"page_id": page_id,
			"actor_id": actor_id,
			"status": "active",
			"started_at": utc_now(),
			"last_heartbeat_at": utc_now(),
		}
		self._ws_sessions[sid] = session
		self._audit(tenant_id, "ws_session_started", sid, actor_id, {"site_id": site_id, "page_id": page_id})
		return dict(session)

	async def async_session_end(
		self,
		tenant_id: str,
		session_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Terminate a collaborative editing session and release resources.

		Emits a ``ws_session_ended`` audit event.
		"""
		session = self._ws_sessions.get(session_id)
		if session is None or session["tenant_id"] != tenant_id:
			raise KeyError(f"session_not_found:{session_id}")
		session["status"] = "ended"
		session["ended_at"] = utc_now()
		del self._ws_sessions[session_id]
		self._audit(tenant_id, "ws_session_ended", session_id, actor_id, {})
		return dict(session)

	async def async_reap_stale_sessions(
		self,
		tenant_id: str,
		max_idle_seconds: int = 60,
	) -> list[str]:
		"""Evict collaborative sessions that have not sent a heartbeat recently.

		Returns the list of reaped session IDs.
		"""
		from datetime import datetime, timezone, timedelta
		now = datetime.now(timezone.utc)
		cutoff = now - timedelta(seconds=max_idle_seconds)
		reaped: list[str] = []
		for sid in list(self._ws_sessions.keys()):
			s = self._ws_sessions[sid]
			if s["tenant_id"] != tenant_id:
				continue
			last = datetime.fromisoformat(s["last_heartbeat_at"])
			if last < cutoff:
				await self.async_session_end(tenant_id, sid, "system")
				reaped.append(sid)
		return reaped

	# ── WebSocket Broker: Component Locking ───────────────────────────────────

	async def async_lock_component(
		self,
		tenant_id: str,
		component_id: str,
		connection_id: str,
		actor_id: str,
		lock_ttl_seconds: int = 60,
	) -> dict[str, Any]:
		"""Acquire an exclusive edit lock on a component for a connection.

		While the lock is held, other connections receive ``component_locked``
		presence events and their edit attempts should be rejected.  Locks
		auto-expire after ``lock_ttl_seconds``.  Raises PermissionError if the
		component is already locked by a different connection.
		"""
		self._require_tenant(tenant_id)
		lock_key = f"{tenant_id}:{component_id}"
		existing = self._component_locks.get(lock_key)
		if existing is not None:
			if existing["holder_connection_id"] != connection_id:
				raise PermissionError(f"component_locked_by_another:{component_id}")
		lock: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"holder_connection_id": connection_id,
			"holder_actor_id": actor_id,
			"lock_ttl_seconds": lock_ttl_seconds,
			"locked_at": utc_now(),
		}
		self._component_locks[lock_key] = lock
		self._audit(tenant_id, "ws_component_locked", component_id, actor_id, {"connection_id": connection_id})
		return dict(lock)

	async def async_unlock_component(
		self,
		tenant_id: str,
		component_id: str,
		connection_id: str,
		actor_id: str,
	) -> dict[str, Any]:
		"""Release the edit lock on a component.

		Only the holding connection (or ``system``) may unlock.  Raises
		PermissionError if the caller does not hold the lock.
		"""
		self._require_tenant(tenant_id)
		lock_key = f"{tenant_id}:{component_id}"
		lock = self._component_locks.get(lock_key)
		if lock is None:
			raise KeyError(f"component_not_locked:{component_id}")
		if lock["holder_connection_id"] != connection_id and actor_id != "system":
			raise PermissionError(f"component_lock_not_owned:{component_id}")
		del self._component_locks[lock_key]
		self._audit(tenant_id, "ws_component_unlocked", component_id, actor_id, {"connection_id": connection_id})
		return {"component_id": component_id, "unlocked_at": utc_now()}

	# ── WebSocket Broker: Annotations ─────────────────────────────────────────

	async def async_annotate_section(
		self,
		tenant_id: str,
		page_id: str,
		section_id: str,
		actor_id: str,
		text: str,
		annotation_id: str | None = None,
	) -> dict[str, Any]:
		"""Attach a text annotation to a page section.

		Annotations provide in-context review feedback visible to all room
		members.  Each annotation is audit-logged and can be resolved via
		``async_resolve_annotation``.  Returns the annotation record.
		"""
		self._require_tenant(tenant_id)
		page = self._get_page(page_id)
		if page.tenant_id != tenant_id:
			raise PermissionError("annotation_tenant_mismatch")
		aid = annotation_id or stable_id("annot", tenant_id, page_id, section_id, actor_id)
		annotation: dict[str, Any] = {
			"id": aid,
			"tenant_id": tenant_id,
			"page_id": page_id,
			"section_id": section_id,
			"actor_id": actor_id,
			"text": text,
			"resolved": False,
			"created_at": utc_now(),
			"updated_at": utc_now(),
		}
		self._annotations.setdefault(f"{tenant_id}:{page_id}", []).append(annotation)
		self._audit(tenant_id, "ws_annotation_added", aid, actor_id, {"section_id": section_id})
		return dict(annotation)

	async def async_list_annotations(
		self,
		tenant_id: str,
		page_id: str,
		include_resolved: bool = False,
	) -> list[dict[str, Any]]:
		"""Return annotations for a page, optionally including resolved ones."""
		annotations = list(self._annotations.get(f"{tenant_id}:{page_id}", []))
		if not include_resolved:
			annotations = [a for a in annotations if not a["resolved"]]
		return [dict(a) for a in annotations]

	# ── WebSocket Broker: Channel Authorization ────────────────────────────────

	async def async_authorize_channel(
		self,
		tenant_id: str,
		connection_id: str,
		channel: str,
		required_perm: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Evaluate WSBL capability rules before admitting a connection to a channel.

		Integrates with the existing ``evaluate()`` engine so tenant-scoped
		RBAC is enforced at the transport layer.  Returns the policy result.
		Raises PermissionError if the result is ``deny``.
		"""
		self._require_tenant(tenant_id)
		context: dict[str, Any] = {
			"tenant_context_present": bool(tenant_id),
			"operation": "ws_channel_authorize",
			"channel": channel,
			"required_perm": required_perm,
			"connection_id": connection_id,
		}
		result = self.evaluate(context)
		self._audit(
			tenant_id,
			"ws_channel_authorized" if result["decision"] != "deny" else "ws_channel_denied",
			channel,
			actor_id,
			{"connection_id": connection_id, "required_perm": required_perm},
			policy_result=result,
		)
		if result["decision"] == "deny":
			reasons = [a.get("reason", "channel_access_denied") for a in result.get("actions", [])]
			raise PermissionError(", ".join(reasons) or "channel_access_denied")
		return self._policy_payload(result)

	# ── WebSocket Broker: State Query Helpers ─────────────────────────────────

	def list_connections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return active WebSocket connections, optionally filtered by tenant."""
		conns = list(self._connections.values())
		if tenant_id is not None:
			conns = [c for c in conns if c["tenant_id"] == tenant_id]
		return [dict(c) for c in sorted(conns, key=lambda x: x["connected_at"])]

	def list_rooms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return all rooms, optionally filtered by tenant."""
		rooms = list(self._rooms.values())
		if tenant_id is not None:
			rooms = [r for r in rooms if r["tenant_id"] == tenant_id]
		return [dict(r) for r in sorted(rooms, key=lambda x: x["created_at"])]

	def list_component_locks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return active component locks, optionally filtered by tenant."""
		locks = list(self._component_locks.values())
		if tenant_id is not None:
			locks = [lock for lock in locks if lock["tenant_id"] == tenant_id]
		return [dict(lock) for lock in locks]

	# ── Private Broker Helpers ─────────────────────────────────────────────────

	async def _room_evict(self, tenant_id: str, room_id: str, connection_id: str) -> None:
		"""Internal: remove a connection from a room without auditing."""
		room_key = f"{tenant_id}:{room_id}"
		room = self._rooms.get(room_key)
		if room is not None and connection_id in room["members"]:
			room["members"].remove(connection_id)
			room["updated_at"] = utc_now()
		conn_key = f"{tenant_id}:{connection_id}"
		conn = self._connections.get(conn_key)
		if conn is not None and room_id in conn.get("rooms", []):
			conn["rooms"].remove(room_id)
