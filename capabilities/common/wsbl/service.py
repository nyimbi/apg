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
		component_id = stable_id("component", tenant_id, component_key)
		status = "approved" if custom and reviewed else "review_required" if custom else "available"
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
			metadata=dict(metadata or {}),
		)
		self._components[component_id] = record
		self._audit(tenant_id, "component_created", component_id, reviewed_by or "system", {"custom": custom, "status": status})
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
		if deny_reasons:
			raise PermissionError(", ".join(deny_reasons))
		required_actions = [action.get("required_action", "review_required") for action in result["actions"] if action.get("decision") == "require_review"]
		status = "review_required" if required_actions else "approved"
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
		)
		self._publish_requests[request_id] = record
		self._audit(
			site.tenant_id,
			"publish_request_created",
			request_id,
			requested_by,
			{"status": status, "required_actions": required_actions, "event_stream": self._normalize_token(event_stream)},
		)
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
		return self.evaluate(context)

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
		return self.evaluate(context)

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

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		sites = self.list_sites(tenant_id)
		pages = self.list_pages(tenant_id)
		components = self.list_components(tenant_id)
		requests = self.list_publish_requests(tenant_id)
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
		if result["decision"] == "deny":
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

	def _audit(self, tenant_id: str, action: str, subject_id: str, actor_id: str, details: dict[str, Any] | None = None) -> None:
		event = WebsiteAuditEventRecord(
			id=stable_id("audit", tenant_id, action, subject_id, len(self._audit_events) + 1),
			tenant_id=tenant_id,
			action=action,
			subject_id=subject_id,
			actor_id=actor_id,
			details=dict(details or {}),
		)
		self._audit_events.append(event)

	@staticmethod
	def _filter(records: Any, tenant_id: str | None) -> list[Any]:
		items = list(records)
		if tenant_id is None:
			return items
		return [record for record in items if record.tenant_id == tenant_id]

	@staticmethod
	def _normalize_token(value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
