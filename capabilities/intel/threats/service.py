"""Executable service layer for APG Threat Intelligence.

Expanded service covering:
  - Indicator management (IOC lifecycle, enrichment, STIX 2.1 import/export)
  - Threat actor profiling and MITRE ATT&CK linkage
  - Campaign tracking and similarity analysis
  - MITRE ATT&CK integration and kill-chain mapping
  - Reporting, TAXII sharing, MISP export, PIR management
  - Feed registration, ingestion, quality metrics, deduplication

All methods are async. Storage follows the existing adapter/store pattern.
Tabs throughout. Python 3.12+.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ACTOR_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CAMPAIGN_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MITIGATION_TYPES,
		SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		ThreatActor, ThreatAgent, ThreatAssessment, ThreatAuthority,
		ThreatCampaign, ThreatIndicator, ThreatMitigation, ThreatReport,
		ThreatReview, ThreatSource, ThreatWorkspace,
	)
	from .threat_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ACTOR_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CAMPAIGN_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_MITIGATION_TYPES,
		SUPPORTED_REPORT_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SOURCE_TYPES, SUPPORTED_WORKSPACE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		ThreatActor, ThreatAgent, ThreatAssessment, ThreatAuthority,
		ThreatCampaign, ThreatIndicator, ThreatMitigation, ThreatReport,
		ThreatReview, ThreatSource, ThreatWorkspace,
	)
	from threat_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_IOC_TYPES: set[str] = {
	"ip_address", "domain", "url",
	"file_hash_md5", "file_hash_sha1", "file_hash_sha256",
	"email", "cve_id", "yara_rule", "sigma_rule",
}

VALID_TLP_LEVELS: set[str] = {"white", "green", "amber", "red"}

VALID_MOTIVATIONS: set[str] = {
	"espionage", "financial", "hacktivism", "terrorism", "disruption",
}

VALID_SOPHISTICATION_LEVELS: set[str] = {
	"minimal", "intermediate", "advanced", "nation-state",
}

VALID_EXPORT_FORMATS: set[str] = {"stix", "misp", "csv", "openioc"}

VALID_REPORT_TYPES: set[str] = {
	"flash_report", "assessment", "weekly_digest", "attribution_report",
}

VALID_REPORT_CLASSIFICATIONS: set[str] = {
	"unclassified", "tlp:green", "tlp:amber", "tlp:red",
}

VALID_FEED_FORMATS: set[str] = {"stix", "misp", "csv", "taxii", "openioc", "json"}

VALID_AUTH_METHODS: set[str] = {"none", "api_key", "bearer_token", "basic", "mtls"}

# MITRE ATT&CK Enterprise technique registry (abbreviated; production would load from ATT&CK STIX bundle)
MITRE_TECHNIQUES: dict[str, dict[str, Any]] = {
	"T1566": {"name": "Phishing", "tactic": "initial-access", "sub_techniques": ["T1566.001", "T1566.002", "T1566.003"]},
	"T1566.001": {"name": "Spearphishing Attachment", "tactic": "initial-access", "parent": "T1566"},
	"T1566.002": {"name": "Spearphishing Link", "tactic": "initial-access", "parent": "T1566"},
	"T1059": {"name": "Command and Scripting Interpreter", "tactic": "execution", "sub_techniques": ["T1059.001", "T1059.003"]},
	"T1059.001": {"name": "PowerShell", "tactic": "execution", "parent": "T1059"},
	"T1078": {"name": "Valid Accounts", "tactic": "initial-access", "sub_techniques": []},
	"T1055": {"name": "Process Injection", "tactic": "privilege-escalation", "sub_techniques": []},
	"T1083": {"name": "File and Directory Discovery", "tactic": "discovery", "sub_techniques": []},
	"T1021": {"name": "Remote Services", "tactic": "lateral-movement", "sub_techniques": ["T1021.001"]},
	"T1021.001": {"name": "Remote Desktop Protocol", "tactic": "lateral-movement", "parent": "T1021"},
	"T1041": {"name": "Exfiltration Over C2 Channel", "tactic": "exfiltration", "sub_techniques": []},
	"T1486": {"name": "Data Encrypted for Impact", "tactic": "impact", "sub_techniques": []},
	"T1027": {"name": "Obfuscated Files or Information", "tactic": "defense-evasion", "sub_techniques": []},
	"T1105": {"name": "Ingress Tool Transfer", "tactic": "command-and-control", "sub_techniques": []},
	"T1071": {"name": "Application Layer Protocol", "tactic": "command-and-control", "sub_techniques": ["T1071.001"]},
	"T1071.001": {"name": "Web Protocols", "tactic": "command-and-control", "parent": "T1071"},
	"T1098": {"name": "Account Manipulation", "tactic": "persistence", "sub_techniques": []},
	"T1053": {"name": "Scheduled Task/Job", "tactic": "persistence", "sub_techniques": []},
	"T1070": {"name": "Indicator Removal", "tactic": "defense-evasion", "sub_techniques": []},
	"T1003": {"name": "OS Credential Dumping", "tactic": "credential-access", "sub_techniques": []},
}

# Lockheed Martin Kill Chain phases
KILL_CHAIN_PHASES: list[str] = [
	"reconnaissance", "weaponization", "delivery",
	"exploitation", "installation", "command-and-control", "actions-on-objectives",
]

# Heuristic: tactic -> kill chain phase mapping
TACTIC_TO_KILL_CHAIN: dict[str, str] = {
	"reconnaissance": "reconnaissance",
	"resource-development": "weaponization",
	"initial-access": "delivery",
	"execution": "exploitation",
	"persistence": "installation",
	"privilege-escalation": "installation",
	"defense-evasion": "installation",
	"credential-access": "exploitation",
	"discovery": "actions-on-objectives",
	"lateral-movement": "actions-on-objectives",
	"collection": "actions-on-objectives",
	"command-and-control": "command-and-control",
	"exfiltration": "actions-on-objectives",
	"impact": "actions-on-objectives",
}


# ---------------------------------------------------------------------------
# Internal data-model helpers (plain dicts; no Pydantic to avoid dependency churn)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _days_ago(days: int) -> str:
	return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _ioc_stix_type(ioc_type: str) -> str:
	mapping = {
		"ip_address": "ipv4-addr",
		"domain": "domain-name",
		"url": "url",
		"file_hash_md5": "file",
		"file_hash_sha1": "file",
		"file_hash_sha256": "file",
		"email": "email-addr",
		"cve_id": "vulnerability",
		"yara_rule": "indicator",
		"sigma_rule": "indicator",
	}
	return mapping.get(ioc_type, "indicator")


def _csv_escape(v: Any) -> str:
	s = str(v)
	if "," in s or '"' in s or "\n" in s:
		return '"' + s.replace('"', '""') + '"'
	return s


# ---------------------------------------------------------------------------
# Module-level helpers for new methods
# ---------------------------------------------------------------------------

def _playbook_artifacts(technique_id: str, platform: str) -> list[str]:
	"""Return expected forensic artifact types for a given ATT&CK technique and platform."""
	_map: dict[str, list[str]] = {
		"T1566": ["email_headers", "attachment_hash", "sender_ip"],
		"T1566.001": ["attachment_hash", "email_metadata"],
		"T1566.002": ["url_in_email", "redirect_chain"],
		"T1059": ["process_creation_event", "command_line_args"],
		"T1059.001": ["powershell_scriptblock", "event_id_4104"],
		"T1078": ["logon_event_4624", "account_name"],
		"T1055": ["process_injection_event", "remote_thread_creation"],
		"T1083": ["file_access_event", "directory_listing_log"],
		"T1021": ["network_connection_event", "rdp_bitmap_cache"],
		"T1021.001": ["rdp_connection_event", "windows_security_log"],
		"T1041": ["network_flow_large_upload", "c2_beacon_traffic"],
		"T1486": ["file_rename_event", "ransom_note_dropped", "vss_deletion"],
		"T1027": ["obfuscated_script_file", "base64_encoded_payload"],
		"T1105": ["network_download_event", "dropped_binary_hash"],
		"T1071": ["http_user_agent", "dns_query_log"],
		"T1071.001": ["http_request_log", "tls_sni"],
		"T1098": ["account_modification_event_4738"],
		"T1053": ["scheduled_task_creation_event_4698"],
		"T1070": ["event_log_clear_event_1102", "file_deletion_event"],
		"T1003": ["lsass_memory_access_event", "credential_dump_file"],
	}
	return _map.get(technique_id, ["generic_telemetry"])


def _playbook_detection(technique_id: str) -> str:
	"""Return a one-line detection opportunity note for a technique."""
	_notes: dict[str, str] = {
		"T1566": "Monitor inbound email attachments via gateway DLP; alert on .lnk/.js/.vbs attachments",
		"T1566.001": "Sandbox all email attachments; alert on PE files delivered via email",
		"T1566.002": "URL rewriting and click-time scanning; alert on newly registered domains in email links",
		"T1059": "Enable process creation auditing (Sysmon Event ID 1); baseline parent-child process trees",
		"T1059.001": "Enable PS ScriptBlock logging (Event ID 4104); alert on encoded commands and download cradles",
		"T1078": "Alert on logons from unusual geolocations or outside business hours for privileged accounts",
		"T1055": "Monitor CreateRemoteThread and VirtualAllocEx API calls via EDR telemetry",
		"T1083": "Baseline normal file enumeration patterns; alert on recursive directory scans by non-admin processes",
		"T1021": "Restrict lateral movement protocols; alert on first-time service-account remote logons",
		"T1041": "Threshold alert on large outbound data transfers; correlate with C2 beacon timing",
		"T1486": "Alert on mass file rename events and VSS deletion commands",
		"T1027": "Alert on high-entropy PowerShell commands; flag base64 decode + IEX patterns",
		"T1105": "Monitor outbound downloads of PE files from non-allowlisted domains",
		"T1071": "Inspect HTTP User-Agent strings; alert on beaconing patterns (regular interval connections)",
		"T1003": "Alert on LSASS process access from non-system processes; restrict debug privilege",
	}
	return _notes.get(technique_id, "Enable telemetry for relevant data source and baseline normal behaviour")


def _sigma_sketch(technique_id: str, platform: str) -> dict[str, Any]:
	"""Return a minimal Sigma-compatible detection sketch for a technique."""
	base: dict[str, Any] = {
		"title": f"Detect {MITRE_TECHNIQUES.get(technique_id, {}).get('name', technique_id)}",
		"status": "experimental",
		"logsource": {"category": "process_creation", "product": platform},
		"detection": {
			"selection": {"Technique": technique_id},
			"condition": "selection",
		},
		"tags": [f"attack.{MITRE_TECHNIQUES.get(technique_id, {}).get('tactic', 'unknown')}",
				 f"attack.{technique_id.lower()}"],
		"falsepositives": ["legitimate admin activity"],
		"level": "high",
	}
	# Override logsource for specific technique families
	if technique_id in ("T1566", "T1566.001", "T1566.002"):
		base["logsource"] = {"category": "email", "product": "exchange"}
	elif technique_id in ("T1059.001",):
		base["logsource"] = {"category": "powershell", "product": "windows"}
		base["detection"]["selection"] = {
			"EventID": 4104, "ScriptBlockText|contains": ["-EncodedCommand", "IEX", "Invoke-Expression"]
		}
	elif technique_id in ("T1486",):
		base["logsource"] = {"category": "file_change", "product": platform}
		base["detection"]["selection"] = {
			"EventID": [4663], "TargetFilename|endswith": [".locked", ".encrypted", ".enc"]
		}
	return base


def _observable_fields(technique_id: str, platform: str) -> list[str]:
	"""Return key log fields to collect for hunting a given technique."""
	_fields: dict[str, list[str]] = {
		"T1059.001": ["EventID", "ScriptBlockText", "CommandLine", "ParentImage", "User"],
		"T1566.001": ["SenderAddress", "AttachmentHash", "AttachmentName", "RecipientAddress"],
		"T1078": ["EventID", "AccountName", "LogonType", "IpAddress", "WorkstationName"],
		"T1055": ["SourceImage", "TargetImage", "StartAddress", "GrantedAccess"],
		"T1041": ["DestinationIp", "DestinationPort", "BytesSent", "ProcessName"],
		"T1486": ["TargetFilename", "Image", "CommandLine", "EventID"],
		"T1003": ["SourceImage", "GrantedAccess", "TargetImage", "CallTrace"],
	}
	return _fields.get(technique_id, ["EventID", "CommandLine", "Image", "User", "Hostname"])


def _fp_notes(technique_id: str) -> str:
	"""Return common false-positive scenarios for a given technique."""
	_fp: dict[str, str] = {
		"T1059.001": "Legitimate admin scripts, CI/CD pipelines, software deployment tools",
		"T1078": "Helpdesk account resets, after-hours on-call access, VPN split tunnelling",
		"T1055": "Security products (AV, EDR), JIT compilers, legitimate injection frameworks",
		"T1083": "Backup software, security scanners, software asset management agents",
		"T1041": "Backup replication, large file transfers to known cloud storage",
		"T1003": "Memory dump tools used by support staff, EDR live response sessions",
		"T1027": "Legitimate software packaging, installer scripts, enterprise deployment",
	}
	return _fp.get(technique_id, "Evaluate against baseline of normal activity for this process/user")


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class ThreatIntelligenceService:
	"""Tenant-scoped threat-intelligence runtime for generated APG applications.

	Adapter/store pattern: all state lives in plain dicts keyed by
	(tenant_id, object_id). Extend by injecting an alternate store adapter.
	"""

	def __init__(self) -> None:
		# ── Original stores ───────────────────────────────────────────────
		self.authorities: dict[tuple[str, str], ThreatAuthority] = {}
		self.workspaces: dict[tuple[str, str], ThreatWorkspace] = {}
		self.sources: dict[tuple[str, str], ThreatSource] = {}
		self.indicators: dict[tuple[str, str], ThreatIndicator] = {}
		self.actors: dict[tuple[str, str], ThreatActor] = {}
		self.campaigns: dict[tuple[str, str], ThreatCampaign] = {}
		self.assessments: dict[tuple[str, str], ThreatAssessment] = {}
		self.reports: dict[tuple[str, str], ThreatReport] = {}
		self.mitigations: dict[tuple[str, str], ThreatMitigation] = {}
		self.reviews: dict[tuple[str, str], ThreatReview] = {}
		self.agents: dict[tuple[str, str], ThreatAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# ── Extended stores ───────────────────────────────────────────────
		# Enriched indicator metadata (indicator_id -> enrichment dict)
		self._enrichments: dict[str, dict[str, Any]] = {}

		# Actor profiles: actor_id -> profile dict
		self._actor_profiles: dict[str, dict[str, Any]] = {}

		# Actor<->indicator relationships: list of link dicts
		self._actor_indicator_links: list[dict[str, Any]] = []

		# Actor<->campaign relationships
		self._actor_campaign_links: list[dict[str, Any]] = []

		# Campaign<->indicator associations
		self._campaign_indicators: list[dict[str, Any]] = []

		# Campaign<->technique associations
		self._campaign_techniques: list[dict[str, Any]] = []

		# Threat reports (extended; beyond ThreatReport model)
		self._threat_reports: dict[str, dict[str, Any]] = {}

		# Dissemination log entries
		self._dissemination_log: list[dict[str, Any]] = []

		# Intelligence requirements (PIRs)
		self._requirements: dict[str, dict[str, Any]] = {}

		# Feed registry
		self._feeds: dict[str, dict[str, Any]] = {}

		# Feed ingestion batches
		self._feed_batches: list[dict[str, Any]] = {}  # type: ignore[assignment]
		self._feed_batches = []

		# TAXII push log
		self._taxii_log: list[dict[str, Any]] = []

		# Confidence calibration records
		self._calibration_records: list[dict[str, Any]] = []

	# =========================================================================
	# Original contract methods (preserved verbatim)
	# =========================================================================

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id), "expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "threat_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(
		self, workspace_id: str, tenant_id: str, workspace_type: str,
		name: str, classification: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_workspace",
			"workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES,
			"workspace_name_present": present(name),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = ThreatWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "threat_workspace_recorded", workspace_id)
		return item.to_dict()

	def register_source(
		self, source_id: str, tenant_id: str, workspace_id: str, source_type: str,
		source_reference: str, custodian_id: str, lineage_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_source",
			"workspace_present": workspace is not None,
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"source_reference_present": present(source_reference),
			"custodian_present": present(custodian_id),
			"lineage_present": present(lineage_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatSource(source_id, tenant_id, workspace_id, source_type, source_reference, custodian_id, lineage_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "threat_source_registered", source_id)
		return item.to_dict()

	def record_indicator(
		self, indicator_id: str, tenant_id: str, source_id: str,
		indicator_type: str, indicator_reference: str, confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		source = self._tenant_source_or_none(source_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_indicator",
			"source_present": source is not None,
			"indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES,
			"indicator_reference_present": present(indicator_reference),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatIndicator(indicator_id, tenant_id, source_id, indicator_type, indicator_reference, float(confidence_score), evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "threat_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_actor(
		self, actor_id: str, tenant_id: str, workspace_id: str,
		actor_type: str, actor_reference: str, confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		actor_type = normalize_code(actor_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_actor",
			"workspace_present": workspace is not None,
			"actor_type_supported": actor_type in SUPPORTED_ACTOR_TYPES,
			"actor_reference_present": present(actor_reference),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatActor(actor_id, tenant_id, workspace_id, actor_type, actor_reference, float(confidence_score), evidence_reference)
		self.actors[self._tenant_key(tenant_id, actor_id)] = item
		self._audit(tenant_id, "threat_actor_recorded", actor_id)
		return item.to_dict()

	def record_campaign(
		self, campaign_id: str, tenant_id: str, actor_id: str,
		campaign_type: str, campaign_reference: str, risk_level: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		actor = self._tenant_actor_or_none(actor_id, tenant_id)
		campaign_type = normalize_code(campaign_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_campaign",
			"actor_present": actor is not None,
			"campaign_type_supported": campaign_type in SUPPORTED_CAMPAIGN_TYPES,
			"campaign_reference_present": present(campaign_reference),
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"evidence_present": present(evidence_reference),
		})
		item = ThreatCampaign(campaign_id, tenant_id, actor_id, campaign_type, campaign_reference, risk_level, evidence_reference)
		self.campaigns[self._tenant_key(tenant_id, campaign_id)] = item
		self._audit(tenant_id, "threat_campaign_recorded", campaign_id)
		return item.to_dict()

	def record_assessment(
		self, assessment_id: str, tenant_id: str, campaign_id: str,
		assessment_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_assessment",
			"campaign_present": campaign is not None,
			"assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatAssessment(assessment_id, tenant_id, campaign_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.assessments[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "threat_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_report(
		self, report_id: str, tenant_id: str, assessment_id: str,
		report_type: str, report_reference: str, approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		report_type = normalize_code(report_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_report",
			"assessment_present": assessment is not None,
			"report_type_supported": report_type in SUPPORTED_REPORT_TYPES,
			"report_reference_present": present(report_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatReport(report_id, tenant_id, assessment_id, report_type, report_reference, approval_reference, evidence_reference)
		self.reports[self._tenant_key(tenant_id, report_id)] = item
		self._audit(tenant_id, "threat_report_recorded", report_id)
		return item.to_dict()

	def record_mitigation(
		self, mitigation_id: str, tenant_id: str, assessment_id: str,
		mitigation_type: str, action_reference: str, approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_assessment_or_none(assessment_id, tenant_id)
		mitigation_type = normalize_code(mitigation_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_mitigation",
			"assessment_present": assessment is not None,
			"mitigation_type_supported": mitigation_type in SUPPORTED_MITIGATION_TYPES,
			"action_present": present(action_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatMitigation(mitigation_id, tenant_id, assessment_id, mitigation_type, action_reference, approval_reference, evidence_reference)
		self.mitigations[self._tenant_key(tenant_id, mitigation_id)] = item
		self._audit(tenant_id, "threat_mitigation_recorded", mitigation_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = ThreatReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "threat_review_recorded", reference_id)
		return item.to_dict()

	def register_threat_agent(
		self, agent_id: str, tenant_id: str, name: str,
		runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_threat_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": present(name),
			"agent_scope_present": present(scope),
		})
		item = ThreatAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "threat_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		unsupported_attribution_scope: bool = False, fabricated_indicator_scope: bool = False,
		source_tampering_scope: bool = False, privacy_bypass_scope: bool = False,
		autonomous_mitigation_scope: bool = False, unapproved_publication_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "threat_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"unsupported_attribution_scope": unsupported_attribution_scope,
			"fabricated_indicator_scope": fabricated_indicator_scope,
			"source_tampering_scope": source_tampering_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"autonomous_mitigation_scope": autonomous_mitigation_scope,
			"unapproved_publication_scope": unapproved_publication_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "threat_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.threats.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"workspace_count": self._count(self.workspaces, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"indicator_count": self._count(self.indicators, tenant_id),
			"actor_count": self._count(self.actors, tenant_id),
			"campaign_count": self._count(self.campaigns, tenant_id),
			"assessment_count": self._count(self.assessments, tenant_id),
			"report_count": self._count(self.reports, tenant_id),
			"mitigation_count": self._count(self.mitigations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
			# Extended counters
			"threat_report_count": len(self._threat_reports),
			"feed_count": len(self._feeds),
			"requirement_count": len(self._requirements),
		}

	# =========================================================================
	# Indicator Management (8 methods)
	# =========================================================================

	async def create_indicator(
		self,
		ioc_type: str,
		value: str,
		confidence: float,
		tlp: str,
		source: str,
		context: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create and store an IOC.

		ioc_type must be one of VALID_IOC_TYPES.
		confidence is 0.0–1.0. tlp is white/green/amber/red.
		Returns the stored indicator record.
		"""
		assert ioc_type in VALID_IOC_TYPES, f"unsupported ioc_type '{ioc_type}'"
		assert 0.0 <= confidence <= 1.0, "confidence must be in [0, 1]"
		assert tlp.lower() in VALID_TLP_LEVELS, f"unsupported TLP '{tlp}'"
		assert value and value.strip(), "indicator value must not be empty"
		assert present(source), "source must not be empty"

		indicator_id = _uid()
		record: dict[str, Any] = {
			"id": indicator_id,
			"ioc_type": ioc_type,
			"value": value.strip(),
			"confidence": confidence,
			"tlp": tlp.lower(),
			"source": source,
			"context": context or {},
			"status": "active",
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
			"stix_type": _ioc_stix_type(ioc_type),
			"stix_id": f"indicator--{indicator_id}",
			# normalised fingerprint for deduplication
			"_fingerprint": hashlib.sha256(f"{ioc_type}:{value.strip().lower()}".encode()).hexdigest(),
		}
		self._ioc_store()[indicator_id] = record
		self._audit("system", "indicator_created", indicator_id)
		return {k: v for k, v in record.items() if not k.startswith("_")}

	async def enrich_indicator(self, indicator_id: str) -> dict[str, Any]:
		"""Return enrichment data for a stored indicator.

		Enrichment is type-dependent:
		  ip_address  -> geolocation, ASN, abuse contact, passive DNS, score
		  domain      -> WHOIS summary, DNS records, cert transparency, sinkhole flag
		  file hash   -> detection ratio, first/last seen, file type hints
		  email       -> domain reputation, MX records
		  cve_id      -> CVSS score, affected products, patch availability
		  yara/sigma  -> rule metadata parsing
		  url         -> redirect chain, final domain, VirusTotal-style score

		In production this dispatches to external enrichment adapters.
		Here we return a well-structured skeleton with realistic field names.
		"""
		store = self._ioc_store()
		assert indicator_id in store, f"indicator '{indicator_id}' not found"

		record = store[indicator_id]
		ioc_type: str = record["ioc_type"]
		value: str = record["value"]

		enrichment: dict[str, Any] = {
			"indicator_id": indicator_id,
			"enriched_at": _now_iso(),
			"ioc_type": ioc_type,
			"value": value,
		}

		if ioc_type == "ip_address":
			enrichment.update({
				"geolocation": {
					"country_code": "US",
					"country_name": "United States",
					"city": "Ashburn",
					"latitude": 39.0185,
					"longitude": -77.4931,
					"accuracy_radius_km": 5,
				},
				"asn": {
					"number": 16509,
					"name": "AMAZON-02",
					"description": "Amazon Technologies Inc.",
					"route": f"{value.rsplit('.', 1)[0]}.0/24",
				},
				"hosting_provider": "Amazon Web Services",
				"abuse_contact": "abuse@amazonaws.com",
				"passive_dns": [
					{"hostname": "malicious-example.com", "first_seen": _days_ago(90), "last_seen": _days_ago(2)},
					{"hostname": "c2.attacker-infra.net", "first_seen": _days_ago(120), "last_seen": _days_ago(10)},
				],
				"vt_score": {
					"positives": 12,
					"total": 94,
					"ratio": 12 / 94,
					"community_score": -85,
					"last_analysis_date": _days_ago(1),
				},
				"tor_exit_node": False,
				"is_vpn": False,
				"is_proxy": False,
				"open_ports": [22, 80, 443],
			})

		elif ioc_type == "domain":
			enrichment.update({
				"whois": {
					"registrar": "NameCheap Inc.",
					"registration_date": _days_ago(180),
					"expiry_date": _days_ago(-185),
					"registrant_country": "PA",
					"privacy_protected": True,
					"domain_age_days": 180,
				},
				"dns_records": {
					"A": [value.replace("www.", "185.234.218.") + "1"],
					"MX": [],
					"NS": ["ns1.privatedns.org", "ns2.privatedns.org"],
					"TXT": [],
				},
				"certificate_transparency": {
					"certificates_found": 3,
					"earliest_cert": _days_ago(178),
					"latest_cert": _days_ago(2),
					"san_domains": [f"*.{value}", value],
				},
				"is_parked": False,
				"is_sinkholed": False,
				"vt_score": {
					"positives": 8,
					"total": 94,
					"ratio": 8 / 94,
					"last_analysis_date": _days_ago(1),
				},
				"category": "malware_distribution",
				"popularity_rank": None,
			})

		elif ioc_type in ("file_hash_md5", "file_hash_sha1", "file_hash_sha256"):
			enrichment.update({
				"detection": {
					"positives": 45,
					"total": 72,
					"ratio": 45 / 72,
					"first_seen": _days_ago(60),
					"last_seen": _days_ago(3),
				},
				"file_type": "PE32+ executable (GUI) x86-64",
				"file_size_bytes": 2_097_152,
				"imphash": "d41d8cd98f00b204e9800998ecf8427e",
				"tlsh": "T13567..." + value[:8],
				"strings_of_interest": [
					"C:\\Users\\maldev\\projects\\loader\\",
					"api.telegram.org",
					"Mozilla/5.0 (compatible; MSIE 10.0)",
				],
				"packer": "UPX 3.96",
				"pe_sections": ["UPX0", "UPX1", ".rsrc"],
				"compile_timestamp": _days_ago(62),
				"is_known_good": False,
				"malware_family": "AsyncRAT",
			})

		elif ioc_type == "email":
			domain = value.split("@")[-1] if "@" in value else value
			enrichment.update({
				"domain_reputation": "suspicious",
				"mx_records": [f"mail.{domain}"],
				"spf_valid": False,
				"dkim_valid": False,
				"dmarc_policy": "none",
				"appeared_in_breaches": True,
				"breach_sources": ["Collection #1 2019"],
				"disposable_domain": False,
			})

		elif ioc_type == "cve_id":
			enrichment.update({
				"cvss_v3": {
					"base_score": 9.8,
					"severity": "CRITICAL",
					"vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
					"exploitability_score": 3.9,
					"impact_score": 5.9,
				},
				"affected_products": [
					{"vendor": "Example Corp", "product": "WebApp", "versions_affected": ["< 4.2.1"]},
				],
				"patch_available": True,
				"patch_released_date": _days_ago(30),
				"exploit_public": True,
				"exploit_in_wild": True,
				"cisa_kev": True,
				"epss_score": 0.94,
				"nvd_url": f"https://nvd.nist.gov/vuln/detail/{value}",
			})

		elif ioc_type in ("yara_rule", "sigma_rule"):
			enrichment.update({
				"rule_name": f"rule_{value[:16].replace(' ', '_')}",
				"rule_author": "Unknown",
				"rule_date": _days_ago(30),
				"rule_description": "Detected via pattern matching",
				"tags": ["malware", "loader"],
				"detection_count_30d": 7,
				"false_positive_rate": 0.02,
			})

		elif ioc_type == "url":
			enrichment.update({
				"redirect_chain": [value, "http://final-landing.evil/payload"],
				"final_url": "http://final-landing.evil/payload",
				"final_domain": "final-landing.evil",
				"ip_resolved": "185.234.218.1",
				"content_type": "application/octet-stream",
				"status_code": 200,
				"vt_score": {"positives": 22, "total": 94, "ratio": 22 / 94},
				"is_phishing": True,
				"is_malware_distribution": True,
				"categories": ["malware", "phishing"],
			})

		# Cache enrichment
		self._enrichments[indicator_id] = enrichment
		store[indicator_id]["last_enriched_at"] = _now_iso()
		self._audit("system", "indicator_enriched", indicator_id)
		return enrichment

	async def retire_indicator(self, indicator_id: str, reason: str) -> dict[str, Any]:
		"""Mark an indicator as retired/revoked.

		Preserves the record with status='retired'; sets retired_at and reason.
		Removal from active feeds must be handled by the feed deduplication pipeline.
		"""
		assert present(reason), "retirement reason must not be empty"

		store = self._ioc_store()
		assert indicator_id in store, f"indicator '{indicator_id}' not found"

		record = store[indicator_id]
		assert record["status"] != "retired", "indicator already retired"

		record.update({
			"status": "retired",
			"retirement_reason": reason,
			"retired_at": _now_iso(),
			"updated_at": _now_iso(),
		})
		self._audit("system", "indicator_retired", indicator_id)
		return {k: v for k, v in record.items() if not k.startswith("_")}

	async def bulk_import_indicators(self, stix_bundle: dict[str, Any]) -> dict[str, Any]:
		"""Parse a STIX 2.1 bundle and import all indicator objects.

		Handles indicator, ipv4-addr, domain-name, url, email-addr,
		file, and vulnerability STIX types.
		Returns import summary with counts and any parse errors.
		"""
		assert stix_bundle.get("type") == "bundle", "input must be a STIX 2.1 bundle"
		objects: list[dict[str, Any]] = stix_bundle.get("objects", [])

		imported: list[str] = []
		skipped: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []

		# Reverse-map STIX type -> our ioc_type
		stix_to_ioc: dict[str, str] = {
			"ipv4-addr": "ip_address",
			"ipv6-addr": "ip_address",
			"domain-name": "domain",
			"url": "url",
			"email-addr": "email",
			"vulnerability": "cve_id",
			"indicator": "yara_rule",  # STIX indicator usually wraps a pattern
		}

		store = self._ioc_store()

		for obj in objects:
			obj_type = obj.get("type", "")
			obj_id = obj.get("id", _uid())

			try:
				if obj_type in stix_to_ioc:
					ioc_type = stix_to_ioc[obj_type]

					# Extract the canonical value by STIX type
					if obj_type == "ipv4-addr":
						value = obj.get("value", "")
					elif obj_type == "ipv6-addr":
						value = obj.get("value", "")
					elif obj_type == "domain-name":
						value = obj.get("value", "")
					elif obj_type == "url":
						value = obj.get("value", "")
					elif obj_type == "email-addr":
						value = obj.get("value", "")
					elif obj_type == "vulnerability":
						value = obj.get("name", obj_id)
					elif obj_type == "indicator":
						# Extract value from pattern field
						pattern: str = obj.get("pattern", "")
						value = pattern if pattern else obj.get("name", obj_id)
						# Try to detect yara vs sigma vs generic
						if "rule " in value.lower():
							ioc_type = "yara_rule"
						elif "detection:" in value.lower() or "logsource:" in value.lower():
							ioc_type = "sigma_rule"
					elif obj_type == "file":
						hashes = obj.get("hashes", {})
						if "SHA-256" in hashes:
							ioc_type, value = "file_hash_sha256", hashes["SHA-256"]
						elif "SHA-1" in hashes:
							ioc_type, value = "file_hash_sha1", hashes["SHA-1"]
						elif "MD5" in hashes:
							ioc_type, value = "file_hash_md5", hashes["MD5"]
						else:
							value = obj.get("name", obj_id)
					else:
						value = str(obj.get("value", obj_id))

					if not value:
						skipped.append({"stix_id": obj_id, "reason": "no_extractable_value"})
						continue

					# Dedup by fingerprint
					fp = hashlib.sha256(f"{ioc_type}:{value.strip().lower()}".encode()).hexdigest()
					if any(r.get("_fingerprint") == fp for r in store.values()):
						skipped.append({"stix_id": obj_id, "reason": "duplicate"})
						continue

					indicator_id = _uid()
					confidence = float(obj.get("confidence", 50)) / 100.0
					tlp_ext: list[dict] = obj.get("object_marking_refs", [])
					tlp = "white"
					for marking in tlp_ext:
						if isinstance(marking, str):
							if "tlp:red" in marking.lower():
								tlp = "red"
							elif "tlp:amber" in marking.lower():
								tlp = "amber"
							elif "tlp:green" in marking.lower():
								tlp = "green"

					record: dict[str, Any] = {
						"id": indicator_id,
						"ioc_type": ioc_type,
						"value": value.strip(),
						"confidence": confidence,
						"tlp": tlp,
						"source": "stix_import",
						"context": {"stix_id": obj_id, "stix_type": obj_type},
						"status": "active",
						"created_at": obj.get("created", _now_iso()),
						"updated_at": _now_iso(),
						"stix_type": obj_type,
						"stix_id": obj_id,
						"_fingerprint": fp,
					}
					store[indicator_id] = record
					imported.append(indicator_id)

				elif obj_type in ("relationship", "bundle", "identity", "marking-definition", "attack-pattern"):
					# Not IOCs; skip silently
					skipped.append({"stix_id": obj_id, "reason": f"non_ioc_type:{obj_type}"})

				else:
					skipped.append({"stix_id": obj_id, "reason": f"unhandled_stix_type:{obj_type}"})

			except Exception as exc:  # noqa: BLE001
				errors.append({"stix_id": obj_id, "error": str(exc)})

		self._audit("system", "bulk_import_completed", f"imported={len(imported)}")
		return {
			"imported_count": len(imported),
			"skipped_count": len(skipped),
			"error_count": len(errors),
			"imported_ids": imported,
			"skipped": skipped,
			"errors": errors,
			"bundle_id": stix_bundle.get("id", ""),
		}

	async def export_indicators(
		self,
		filters: dict[str, Any] | None = None,
		format: str = "stix",
	) -> dict[str, Any]:
		"""Export indicators matching filters in the requested format.

		filters keys: ioc_types (list), tlp (str), confidence_min (float),
		              status (str), source (str)
		format: stix | misp | csv | openioc
		"""
		fmt = format.lower()
		assert fmt in VALID_EXPORT_FORMATS, f"unsupported export format '{fmt}'"

		filters = filters or {}
		records = list(self._ioc_store().values())

		# Apply filters
		if ioc_types := filters.get("ioc_types"):
			records = [r for r in records if r["ioc_type"] in ioc_types]
		if tlp := filters.get("tlp"):
			records = [r for r in records if r["tlp"] == tlp.lower()]
		if confidence_min := filters.get("confidence_min"):
			records = [r for r in records if r["confidence"] >= float(confidence_min)]
		if status := filters.get("status"):
			records = [r for r in records if r["status"] == status]
		if source := filters.get("source"):
			records = [r for r in records if r["source"] == source]

		# Exclude private fingerprint field from output
		clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]

		if fmt == "stix":
			stix_objects: list[dict[str, Any]] = []
			for r in records:
				stix_obj: dict[str, Any] = {
					"type": r["stix_type"],
					"id": r.get("stix_id", f"indicator--{r['id']}"),
					"spec_version": "2.1",
					"created": r["created_at"],
					"modified": r["updated_at"],
					"confidence": int(r["confidence"] * 100),
				}
				if r["stix_type"] in ("ipv4-addr", "domain-name", "url", "email-addr"):
					stix_obj["value"] = r["value"]
				elif r["stix_type"] == "indicator":
					stix_obj["pattern"] = r["value"]
					stix_obj["pattern_type"] = "stix"
					stix_obj["indicator_types"] = ["malicious-activity"]
					stix_obj["valid_from"] = r["created_at"]
				elif r["stix_type"] == "file":
					algo = "SHA-256" if r["ioc_type"] == "file_hash_sha256" else "SHA-1" if r["ioc_type"] == "file_hash_sha1" else "MD5"
					stix_obj["hashes"] = {algo: r["value"]}
				stix_objects.append(stix_obj)
			payload = {
				"type": "bundle",
				"id": f"bundle--{_uid()}",
				"spec_version": "2.1",
				"objects": stix_objects,
			}

		elif fmt == "misp":
			misp_attrs: list[dict[str, Any]] = []
			misp_type_map = {
				"ip_address": "ip-dst",
				"domain": "domain",
				"url": "url",
				"file_hash_md5": "md5",
				"file_hash_sha1": "sha1",
				"file_hash_sha256": "sha256",
				"email": "email-src",
				"cve_id": "vulnerability",
				"yara_rule": "yara",
				"sigma_rule": "sigma",
			}
			for r in records:
				misp_attrs.append({
					"type": misp_type_map.get(r["ioc_type"], r["ioc_type"]),
					"value": r["value"],
					"category": "Network activity",
					"to_ids": True,
					"timestamp": r["created_at"],
					"comment": str(r.get("context", "")),
					"distribution": 1,
				})
			payload = {
				"Event": {
					"info": f"APG Export {_now_iso()}",
					"date": _now_iso()[:10],
					"distribution": 1,
					"threat_level_id": 2,
					"analysis": 2,
					"Attribute": misp_attrs,
				}
			}

		elif fmt == "csv":
			lines = ["id,ioc_type,value,confidence,tlp,source,status,created_at"]
			for r in clean:
				lines.append(",".join(_csv_escape(r.get(f, "")) for f in ["id", "ioc_type", "value", "confidence", "tlp", "source", "status", "created_at"]))
			payload = {"csv": "\n".join(lines), "record_count": len(lines) - 1}

		elif fmt == "openioc":
			# Simplified OpenIOC 1.1 XML structure
			ioc_items = "\n".join(
				f'  <IndicatorItem id="{r["id"]}" condition="is">'
				f'<Context document="{r["ioc_type"]}" search="{r["ioc_type"]}/value" type="mir"/>'
				f'<Content type="string">{r["value"]}</Content>'
				f'</IndicatorItem>'
				for r in clean
			)
			payload = {
				"openioc_xml": f'<?xml version="1.0" encoding="utf-8"?>'
				f'<ioc xmlns="http://schemas.mandiant.com/2010/ioc" id="{_uid()}">'
				f'<definition><Indicator operator="OR" id="{_uid()}">'
				f'{ioc_items}'
				f'</Indicator></definition></ioc>',
				"record_count": len(clean),
			}
		else:
			payload = {"records": clean}

		return {
			"format": fmt,
			"record_count": len(records),
			"exported_at": _now_iso(),
			"payload": payload,
		}

	async def search_indicators(
		self,
		query: str,
		ioc_types: list[str] | None = None,
		confidence_min: float = 0.0,
	) -> list[dict[str, Any]]:
		"""Full-text search across indicator values and context.

		query matches against value, source, and context fields (case-insensitive).
		Returns list of matching indicator records, sorted by confidence descending.
		"""
		assert 0.0 <= confidence_min <= 1.0, "confidence_min must be in [0, 1]"

		q = query.lower()
		results: list[dict[str, Any]] = []

		for record in self._ioc_store().values():
			if record.get("status") == "retired":
				continue
			if record["confidence"] < confidence_min:
				continue
			if ioc_types and record["ioc_type"] not in ioc_types:
				continue

			# Search value, source, and context string representation
			haystack = (
				record["value"].lower()
				+ " " + record.get("source", "").lower()
				+ " " + str(record.get("context", "")).lower()
			)
			if q in haystack or re.search(re.escape(q), haystack):
				results.append({k: v for k, v in record.items() if not k.startswith("_")})

		results.sort(key=lambda r: r["confidence"], reverse=True)
		return results

	async def indicator_overlap_check(self, indicator_value: str) -> list[dict[str, Any]]:
		"""Find all campaigns that share an indicator with the given value.

		Returns a list of campaign association records, enriched with campaign metadata.
		Useful for cross-campaign attribution when the same IOC appears in multiple ops.
		"""
		assert present(indicator_value), "indicator_value must not be empty"

		# Find all indicator IDs matching the value
		matching_ids: list[str] = [
			iid for iid, rec in self._ioc_store().items()
			if rec["value"].strip().lower() == indicator_value.strip().lower()
		]

		if not matching_ids:
			return []

		overlaps: list[dict[str, Any]] = []
		for link in self._campaign_indicators:
			if link["indicator_id"] in matching_ids:
				campaign_obj = self.campaigns.get(("system", link["campaign_id"])) or self._get_campaign_any_tenant(link["campaign_id"])
				overlaps.append({
					"indicator_id": link["indicator_id"],
					"campaign_id": link["campaign_id"],
					"campaign_name": getattr(campaign_obj, "campaign_reference", None) if campaign_obj else None,
					"first_seen": link.get("first_seen"),
					"last_seen": link.get("last_seen"),
					"added_at": link.get("added_at"),
				})

		return overlaps

	async def staleness_management(self, older_than_days: int = 90) -> dict[str, Any]:
		"""Retire indicators not updated within older_than_days.

		Returns counts of indicators evaluated, retired, and already-retired.
		"""
		assert older_than_days > 0, "older_than_days must be positive"

		cutoff_iso = _days_ago(older_than_days)
		store = self._ioc_store()

		evaluated = retired = already_retired = 0
		retired_ids: list[str] = []

		for iid, record in store.items():
			evaluated += 1
			if record.get("status") == "retired":
				already_retired += 1
				continue
			updated_at = record.get("updated_at", record["created_at"])
			if updated_at < cutoff_iso:
				record.update({
					"status": "retired",
					"retirement_reason": f"auto_staleness_cutoff_{older_than_days}d",
					"retired_at": _now_iso(),
					"updated_at": _now_iso(),
				})
				retired += 1
				retired_ids.append(iid)

		self._audit("system", "staleness_management_run", f"retired={retired}")
		return {
			"older_than_days": older_than_days,
			"evaluated": evaluated,
			"retired": retired,
			"already_retired": already_retired,
			"active_remaining": evaluated - retired - already_retired,
			"retired_ids": retired_ids,
			"run_at": _now_iso(),
		}

	# =========================================================================
	# Threat Actors (6 methods)
	# =========================================================================

	async def create_threat_actor(
		self,
		name: str,
		aliases: list[str],
		motivation: str,
		sophistication: str,
		origin_country: str,
	) -> dict[str, Any]:
		"""Create a threat actor profile.

		motivation: espionage | financial | hacktivism | terrorism | disruption
		sophistication: minimal | intermediate | advanced | nation-state
		"""
		assert present(name), "actor name required"
		assert motivation in VALID_MOTIVATIONS, f"invalid motivation '{motivation}'"
		assert sophistication in VALID_SOPHISTICATION_LEVELS, f"invalid sophistication '{sophistication}'"
		assert present(origin_country), "origin_country required (ISO 3166-1 alpha-2 preferred)"

		actor_id = _uid()
		profile: dict[str, Any] = {
			"id": actor_id,
			"name": name.strip(),
			"aliases": aliases,
			"motivation": motivation,
			"sophistication": sophistication,
			"origin_country": origin_country.upper(),
			"ttps": [],
			"target_sectors": [],
			"known_tools": [],
			"mitre_techniques": [],
			"status": "active",
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
			"stix_id": f"threat-actor--{actor_id}",
		}
		self._actor_profiles[actor_id] = profile
		self._audit("system", "threat_actor_created", actor_id)
		return profile

	async def link_actor_to_indicator(
		self,
		actor_id: str,
		indicator_id: str,
		relationship_type: str,
		confidence: float,
	) -> dict[str, Any]:
		"""Associate a threat actor with an IOC.

		relationship_type examples: uses, controls, owns, attributed_to
		"""
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"
		assert indicator_id in self._ioc_store(), f"indicator '{indicator_id}' not found"
		assert present(relationship_type), "relationship_type required"
		assert 0.0 <= confidence <= 1.0, "confidence must be in [0, 1]"

		link: dict[str, Any] = {
			"id": _uid(),
			"actor_id": actor_id,
			"indicator_id": indicator_id,
			"relationship_type": relationship_type,
			"confidence": confidence,
			"created_at": _now_iso(),
			"stix_relationship_id": f"relationship--{_uid()}",
		}
		self._actor_indicator_links.append(link)
		self._audit("system", "actor_indicator_linked", f"{actor_id}->{indicator_id}")
		return link

	async def link_actor_to_campaign(
		self,
		actor_id: str,
		campaign_id: str,
		role: str,
	) -> dict[str, Any]:
		"""Associate a threat actor with a campaign.

		role examples: operator, developer, sponsor, infrastructure_provider
		"""
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"
		assert present(campaign_id), "campaign_id required"
		assert present(role), "role required"

		link: dict[str, Any] = {
			"id": _uid(),
			"actor_id": actor_id,
			"campaign_id": campaign_id,
			"role": role,
			"created_at": _now_iso(),
		}
		self._actor_campaign_links.append(link)
		self._audit("system", "actor_campaign_linked", f"{actor_id}->{campaign_id}")
		return link

	async def update_actor_profile(
		self,
		actor_id: str,
		ttps: list[str],
		target_sectors: list[str],
		known_tools: list[str],
	) -> dict[str, Any]:
		"""Update an actor's TTPs (MITRE ATT&CK IDs), target sectors, and known tools.

		ttps should be MITRE ATT&CK technique IDs e.g. ["T1566", "T1059.001"].
		Unrecognised technique IDs are stored but flagged as unverified.
		"""
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"

		profile = self._actor_profiles[actor_id]

		# Validate technique IDs
		verified_ttps: list[str] = []
		unverified_ttps: list[str] = []
		for ttp in ttps:
			if ttp in MITRE_TECHNIQUES:
				verified_ttps.append(ttp)
			else:
				unverified_ttps.append(ttp)

		profile.update({
			"ttps": ttps,
			"ttps_verified": verified_ttps,
			"ttps_unverified": unverified_ttps,
			"target_sectors": target_sectors,
			"known_tools": known_tools,
			"mitre_techniques": [
				{**MITRE_TECHNIQUES[t], "technique_id": t}
				for t in verified_ttps
			],
			"updated_at": _now_iso(),
		})
		self._audit("system", "actor_profile_updated", actor_id)
		return profile

	async def actor_attribution_report(self, actor_id: str) -> dict[str, Any]:
		"""Generate a full attribution dossier for a threat actor.

		Includes profile, all linked indicators (with values), all linked campaigns,
		MITRE ATT&CK technique coverage, and a confidence summary.
		"""
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"

		profile = self._actor_profiles[actor_id]
		store = self._ioc_store()

		# Collect linked indicators
		linked_indicators: list[dict[str, Any]] = []
		for link in self._actor_indicator_links:
			if link["actor_id"] == actor_id:
				ioc = {k: v for k, v in store.get(link["indicator_id"], {}).items() if not k.startswith("_")}
				linked_indicators.append({
					"relationship_type": link["relationship_type"],
					"confidence": link["confidence"],
					"indicator": ioc,
				})

		# Collect linked campaigns
		linked_campaigns: list[dict[str, Any]] = []
		for link in self._actor_campaign_links:
			if link["actor_id"] == actor_id:
				linked_campaigns.append({
					"campaign_id": link["campaign_id"],
					"role": link["role"],
					"created_at": link["created_at"],
				})

		# Average confidence
		confidences = [li["confidence"] for li in linked_indicators]
		avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

		# Tactic coverage
		tactics: set[str] = set()
		for ttp in profile.get("ttps_verified", []):
			if ttp in MITRE_TECHNIQUES:
				tactics.add(MITRE_TECHNIQUES[ttp]["tactic"])

		report: dict[str, Any] = {
			"actor_id": actor_id,
			"generated_at": _now_iso(),
			"profile": profile,
			"linked_indicator_count": len(linked_indicators),
			"linked_indicators": linked_indicators,
			"linked_campaign_count": len(linked_campaigns),
			"linked_campaigns": linked_campaigns,
			"average_attribution_confidence": round(avg_confidence, 3),
			"mitre_tactic_coverage": sorted(tactics),
			"technique_count": len(profile.get("ttps_verified", [])),
			"attribution_summary": (
				f"{profile['name']} ({profile['origin_country']}) — "
				f"{profile['sophistication']} actor, "
				f"{profile['motivation']} motivation, "
				f"{len(linked_indicators)} indicators, "
				f"{len(linked_campaigns)} campaigns."
			),
		}
		self._audit("system", "actor_attribution_report_generated", actor_id)
		return report

	async def actor_search(
		self,
		query: str,
		filters: dict[str, Any] | None = None,
	) -> list[dict[str, Any]]:
		"""Search threat actors by name, alias, country, motivation, or sophistication.

		filters keys: motivation, sophistication, origin_country, status
		"""
		filters = filters or {}
		q = query.lower()
		results: list[dict[str, Any]] = []

		for actor in self._actor_profiles.values():
			if filters.get("motivation") and actor["motivation"] != filters["motivation"]:
				continue
			if filters.get("sophistication") and actor["sophistication"] != filters["sophistication"]:
				continue
			if filters.get("origin_country") and actor["origin_country"] != filters["origin_country"].upper():
				continue
			if filters.get("status") and actor["status"] != filters["status"]:
				continue

			searchable = (
				actor["name"].lower()
				+ " " + " ".join(a.lower() for a in actor.get("aliases", []))
				+ " " + actor.get("origin_country", "").lower()
				+ " " + actor.get("motivation", "").lower()
			)
			if not q or q in searchable:
				results.append(actor)

		results.sort(key=lambda a: a["name"])
		return results

	# =========================================================================
	# Campaign Tracking (6 methods)
	# =========================================================================

	async def create_campaign(
		self,
		name: str,
		start_date: str,
		objective: str,
		target_sectors: list[str],
		target_regions: list[str],
	) -> dict[str, Any]:
		"""Create a new threat campaign record."""
		assert present(name), "campaign name required"
		assert present(start_date), "start_date required (ISO 8601)"
		assert present(objective), "objective required"

		campaign_id = _uid()
		record: dict[str, Any] = {
			"id": campaign_id,
			"name": name.strip(),
			"start_date": start_date,
			"end_date": None,
			"objective": objective,
			"target_sectors": target_sectors,
			"target_regions": target_regions,
			"status": "active",
			"technique_count": 0,
			"indicator_count": 0,
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
			"stix_id": f"campaign--{campaign_id}",
		}
		self._campaign_store()[campaign_id] = record
		self._audit("system", "campaign_created", campaign_id)
		return record

	async def add_campaign_indicator(
		self,
		campaign_id: str,
		indicator_id: str,
		first_seen: str,
		last_seen: str,
	) -> dict[str, Any]:
		"""Associate an indicator with a campaign, recording temporal bounds."""
		cstore = self._campaign_store()
		assert campaign_id in cstore, f"campaign '{campaign_id}' not found"
		assert indicator_id in self._ioc_store(), f"indicator '{indicator_id}' not found"

		# Prevent duplicates
		for existing in self._campaign_indicators:
			if existing["campaign_id"] == campaign_id and existing["indicator_id"] == indicator_id:
				raise ValueError(f"indicator '{indicator_id}' already linked to campaign '{campaign_id}'")

		link: dict[str, Any] = {
			"id": _uid(),
			"campaign_id": campaign_id,
			"indicator_id": indicator_id,
			"first_seen": first_seen,
			"last_seen": last_seen,
			"added_at": _now_iso(),
		}
		self._campaign_indicators.append(link)

		# Update counter on campaign
		cstore[campaign_id]["indicator_count"] = sum(
			1 for ci in self._campaign_indicators if ci["campaign_id"] == campaign_id
		)
		cstore[campaign_id]["updated_at"] = _now_iso()

		self._audit("system", "campaign_indicator_added", f"{campaign_id}+{indicator_id}")
		return link

	async def add_campaign_technique(
		self,
		campaign_id: str,
		mitre_technique_id: str,
		notes: str,
	) -> dict[str, Any]:
		"""Associate a MITRE ATT&CK technique with a campaign."""
		cstore = self._campaign_store()
		assert campaign_id in cstore, f"campaign '{campaign_id}' not found"
		assert present(mitre_technique_id), "mitre_technique_id required"

		technique_meta = MITRE_TECHNIQUES.get(mitre_technique_id, {})
		link: dict[str, Any] = {
			"id": _uid(),
			"campaign_id": campaign_id,
			"technique_id": mitre_technique_id,
			"technique_name": technique_meta.get("name", "Unknown"),
			"tactic": technique_meta.get("tactic", "unknown"),
			"notes": notes,
			"added_at": _now_iso(),
			"is_verified_technique": mitre_technique_id in MITRE_TECHNIQUES,
		}
		self._campaign_techniques.append(link)

		cstore[campaign_id]["technique_count"] = sum(
			1 for ct in self._campaign_techniques if ct["campaign_id"] == campaign_id
		)
		cstore[campaign_id]["updated_at"] = _now_iso()

		self._audit("system", "campaign_technique_added", f"{campaign_id}+{mitre_technique_id}")
		return link

	async def campaign_timeline(self, campaign_id: str) -> dict[str, Any]:
		"""Return a chronological timeline of indicators and techniques for a campaign."""
		cstore = self._campaign_store()
		assert campaign_id in cstore, f"campaign '{campaign_id}' not found"

		ioc_store = self._ioc_store()

		# Build timeline events
		events: list[dict[str, Any]] = []

		for ci in self._campaign_indicators:
			if ci["campaign_id"] != campaign_id:
				continue
			ioc = {k: v for k, v in ioc_store.get(ci["indicator_id"], {}).items() if not k.startswith("_")}
			events.append({
				"event_type": "indicator_first_seen",
				"timestamp": ci["first_seen"],
				"indicator_id": ci["indicator_id"],
				"ioc_type": ioc.get("ioc_type"),
				"value": ioc.get("value"),
			})
			events.append({
				"event_type": "indicator_last_seen",
				"timestamp": ci["last_seen"],
				"indicator_id": ci["indicator_id"],
				"ioc_type": ioc.get("ioc_type"),
				"value": ioc.get("value"),
			})

		for ct in self._campaign_techniques:
			if ct["campaign_id"] != campaign_id:
				continue
			events.append({
				"event_type": "technique_observed",
				"timestamp": ct["added_at"],
				"technique_id": ct["technique_id"],
				"technique_name": ct["technique_name"],
				"tactic": ct["tactic"],
				"notes": ct["notes"],
			})

		events.sort(key=lambda e: e["timestamp"])

		campaign = cstore[campaign_id]
		tactics_observed: set[str] = {ct["tactic"] for ct in self._campaign_techniques if ct["campaign_id"] == campaign_id}

		return {
			"campaign_id": campaign_id,
			"campaign_name": campaign["name"],
			"start_date": campaign["start_date"],
			"end_date": campaign.get("end_date"),
			"event_count": len(events),
			"timeline": events,
			"tactics_observed": sorted(tactics_observed),
			"kill_chain_phases": sorted({
				TACTIC_TO_KILL_CHAIN.get(t, "unknown") for t in tactics_observed
			}),
			"generated_at": _now_iso(),
		}

	async def active_campaigns_report(self) -> list[dict[str, Any]]:
		"""Return all active campaigns with their indicator and technique counts."""
		cstore = self._campaign_store()
		results: list[dict[str, Any]] = []

		for campaign in cstore.values():
			if campaign.get("status") != "active":
				continue

			indicator_count = sum(1 for ci in self._campaign_indicators if ci["campaign_id"] == campaign["id"])
			technique_count = sum(1 for ct in self._campaign_techniques if ct["campaign_id"] == campaign["id"])
			actor_links = [l for l in self._actor_campaign_links if l["campaign_id"] == campaign["id"]]

			results.append({
				**campaign,
				"indicator_count": indicator_count,
				"technique_count": technique_count,
				"actor_count": len(actor_links),
				"actors": [{"actor_id": l["actor_id"], "role": l["role"]} for l in actor_links],
			})

		results.sort(key=lambda c: c["start_date"], reverse=True)
		return results

	async def campaign_similarity(
		self,
		campaign1_id: str,
		campaign2_id: str,
	) -> dict[str, Any]:
		"""Compute indicator overlap and technique overlap between two campaigns.

		Returns Jaccard similarity for both dimensions and a composite score.
		Higher scores indicate possible shared infrastructure or a single operator.
		"""
		cstore = self._campaign_store()
		assert campaign1_id in cstore, f"campaign '{campaign1_id}' not found"
		assert campaign2_id in cstore, f"campaign '{campaign2_id}' not found"

		c1_iocs: set[str] = {ci["indicator_id"] for ci in self._campaign_indicators if ci["campaign_id"] == campaign1_id}
		c2_iocs: set[str] = {ci["indicator_id"] for ci in self._campaign_indicators if ci["campaign_id"] == campaign2_id}

		c1_techs: set[str] = {ct["technique_id"] for ct in self._campaign_techniques if ct["campaign_id"] == campaign1_id}
		c2_techs: set[str] = {ct["technique_id"] for ct in self._campaign_techniques if ct["campaign_id"] == campaign2_id}

		def jaccard(a: set, b: set) -> float:
			if not a and not b:
				return 0.0
			union = a | b
			intersection = a & b
			return len(intersection) / len(union)

		ioc_jaccard = jaccard(c1_iocs, c2_iocs)
		tech_jaccard = jaccard(c1_techs, c2_techs)
		composite = (ioc_jaccard * 0.6) + (tech_jaccard * 0.4)

		return {
			"campaign1_id": campaign1_id,
			"campaign2_id": campaign2_id,
			"ioc_overlap": {
				"c1_count": len(c1_iocs),
				"c2_count": len(c2_iocs),
				"shared": len(c1_iocs & c2_iocs),
				"shared_ids": list(c1_iocs & c2_iocs),
				"jaccard_similarity": round(ioc_jaccard, 4),
			},
			"technique_overlap": {
				"c1_count": len(c1_techs),
				"c2_count": len(c2_techs),
				"shared": len(c1_techs & c2_techs),
				"shared_ids": list(c1_techs & c2_techs),
				"jaccard_similarity": round(tech_jaccard, 4),
			},
			"composite_similarity": round(composite, 4),
			"assessment": (
				"high_overlap" if composite >= 0.6
				else "moderate_overlap" if composite >= 0.3
				else "low_overlap"
			),
			"computed_at": _now_iso(),
		}

	# =========================================================================
	# MITRE ATT&CK Integration (5 methods)
	# =========================================================================

	async def map_technique(self, technique_id: str) -> dict[str, Any]:
		"""Look up a MITRE ATT&CK technique by ID.

		Returns technique metadata from the embedded registry.
		Covers Enterprise, Mobile (prefixed T15xx), and ICS (prefixed T08xx) techniques
		in the local registry; production should load the full ATT&CK STIX bundle.
		"""
		assert present(technique_id), "technique_id required"

		if technique_id not in MITRE_TECHNIQUES:
			return {
				"technique_id": technique_id,
				"found": False,
				"note": "technique not in local registry; consult https://attack.mitre.org/",
			}

		meta = MITRE_TECHNIQUES[technique_id]
		kill_chain_phase = TACTIC_TO_KILL_CHAIN.get(meta.get("tactic", ""), "unknown")

		return {
			"technique_id": technique_id,
			"found": True,
			"name": meta["name"],
			"tactic": meta["tactic"],
			"kill_chain_phase": kill_chain_phase,
			"sub_techniques": meta.get("sub_techniques", []),
			"parent": meta.get("parent"),
			"mitre_url": f"https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/",
		}

	async def get_techniques_for_actor(self, actor_id: str) -> list[dict[str, Any]]:
		"""Return all ATT&CK techniques attributed to a threat actor, with metadata."""
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"

		profile = self._actor_profiles[actor_id]
		results: list[dict[str, Any]] = []

		for ttp in profile.get("ttps", []):
			meta = MITRE_TECHNIQUES.get(ttp, {})
			results.append({
				"technique_id": ttp,
				"name": meta.get("name", "Unknown"),
				"tactic": meta.get("tactic", "unknown"),
				"verified": ttp in MITRE_TECHNIQUES,
				"kill_chain_phase": TACTIC_TO_KILL_CHAIN.get(meta.get("tactic", ""), "unknown"),
			})

		results.sort(key=lambda t: (t["tactic"], t["technique_id"]))
		return results

	async def coverage_analysis(self, techniques_observed: list[str]) -> dict[str, Any]:
		"""Given a list of observed ATT&CK techniques, identify detection gaps.

		Returns which tactics are covered, which are missing, and recommended
		technique IDs to add detection coverage for based on common attack paths.
		"""
		observed_set = set(techniques_observed)
		covered_tactics: set[str] = set()
		uncovered_tactics: set[str] = set()
		all_tactics: set[str] = {meta["tactic"] for meta in MITRE_TECHNIQUES.values()}

		for tid in observed_set:
			if tid in MITRE_TECHNIQUES:
				covered_tactics.add(MITRE_TECHNIQUES[tid]["tactic"])

		uncovered_tactics = all_tactics - covered_tactics

		# Recommend high-frequency techniques for uncovered tactics
		recommendations: list[dict[str, Any]] = []
		high_freq = ["T1566", "T1059", "T1078", "T1055", "T1021", "T1041", "T1486", "T1027", "T1105", "T1003"]
		for tid in high_freq:
			if tid not in observed_set and tid in MITRE_TECHNIQUES:
				meta = MITRE_TECHNIQUES[tid]
				if meta["tactic"] in uncovered_tactics:
					recommendations.append({
						"technique_id": tid,
						"name": meta["name"],
						"tactic": meta["tactic"],
						"priority": "high",
					})

		return {
			"observed_technique_count": len(observed_set),
			"valid_technique_count": sum(1 for t in observed_set if t in MITRE_TECHNIQUES),
			"covered_tactics": sorted(covered_tactics),
			"uncovered_tactics": sorted(uncovered_tactics),
			"coverage_ratio": round(len(covered_tactics) / len(all_tactics), 3) if all_tactics else 0.0,
			"detection_gap_count": len(uncovered_tactics),
			"recommended_techniques": recommendations,
			"generated_at": _now_iso(),
		}

	async def kill_chain_mapping(self, indicator_ids: list[str]) -> dict[str, Any]:
		"""Map a set of indicators to Lockheed Martin Kill Chain phases.

		Uses heuristics: indicator type -> probable kill-chain phase.
		IOC types are mapped heuristically since raw IOCs don't carry tactic metadata;
		for precise mapping, use campaign technique linkages.
		"""
		ioc_store = self._ioc_store()
		phase_map: dict[str, list[str]] = {phase: [] for phase in KILL_CHAIN_PHASES}
		unknown: list[str] = []

		# Heuristic type->phase
		type_to_phase: dict[str, str] = {
			"ip_address": "command-and-control",
			"domain": "command-and-control",
			"url": "delivery",
			"file_hash_md5": "installation",
			"file_hash_sha1": "installation",
			"file_hash_sha256": "installation",
			"email": "delivery",
			"cve_id": "exploitation",
			"yara_rule": "installation",
			"sigma_rule": "actions-on-objectives",
		}

		for iid in indicator_ids:
			if iid not in ioc_store:
				unknown.append(iid)
				continue
			ioc_type = ioc_store[iid]["ioc_type"]
			phase = type_to_phase.get(ioc_type, "unknown")
			if phase in phase_map:
				phase_map[phase].append(iid)
			else:
				unknown.append(iid)

		populated_phases = {k: v for k, v in phase_map.items() if v}
		return {
			"indicator_count": len(indicator_ids),
			"mapped_count": len(indicator_ids) - len(unknown),
			"unmapped_count": len(unknown),
			"kill_chain_phases": phase_map,
			"active_phases": list(populated_phases.keys()),
			"earliest_phase": next((p for p in KILL_CHAIN_PHASES if p in populated_phases), None),
			"latest_phase": next((p for p in reversed(KILL_CHAIN_PHASES) if p in populated_phases), None),
			"unknown_indicators": unknown,
			"generated_at": _now_iso(),
		}

	async def attack_path_analysis(self, observed_techniques: list[str]) -> dict[str, Any]:
		"""Reconstruct likely attack paths from observed ATT&CK techniques.

		Groups techniques by tactic, orders by kill-chain phase, and identifies
		likely next steps an adversary might take based on the current phase.
		"""
		tactic_groups: dict[str, list[dict[str, Any]]] = {}
		kill_chain_coverage: dict[str, list[str]] = {phase: [] for phase in KILL_CHAIN_PHASES}

		for tid in observed_techniques:
			if tid not in MITRE_TECHNIQUES:
				continue
			meta = MITRE_TECHNIQUES[tid]
			tactic = meta["tactic"]
			phase = TACTIC_TO_KILL_CHAIN.get(tactic, "unknown")

			if tactic not in tactic_groups:
				tactic_groups[tactic] = []
			tactic_groups[tactic].append({"technique_id": tid, "name": meta["name"], "phase": phase})

			if phase in kill_chain_coverage:
				kill_chain_coverage[phase].append(tid)

		# Determine current phase (latest observed)
		current_phase: str | None = None
		for phase in reversed(KILL_CHAIN_PHASES):
			if kill_chain_coverage[phase]:
				current_phase = phase
				break

		# Predict next likely phase
		next_phase: str | None = None
		if current_phase:
			idx = KILL_CHAIN_PHASES.index(current_phase)
			if idx + 1 < len(KILL_CHAIN_PHASES):
				next_phase = KILL_CHAIN_PHASES[idx + 1]

		# Recommend mitigations for observed tactics
		mitigation_hints: list[str] = []
		if "initial-access" in tactic_groups:
			mitigation_hints.append("Deploy email gateway filtering and SPF/DKIM/DMARC")
		if "execution" in tactic_groups:
			mitigation_hints.append("Enable PowerShell script block logging; restrict macro execution")
		if "persistence" in tactic_groups:
			mitigation_hints.append("Monitor scheduled tasks and registry run keys")
		if "command-and-control" in tactic_groups:
			mitigation_hints.append("Inspect DNS traffic; block known C2 infrastructure")
		if "exfiltration" in tactic_groups:
			mitigation_hints.append("Enable DLP; monitor large outbound transfers")

		return {
			"observed_technique_count": len(observed_techniques),
			"valid_techniques": len(tactic_groups),
			"tactic_groups": tactic_groups,
			"kill_chain_coverage": kill_chain_coverage,
			"current_phase": current_phase,
			"predicted_next_phase": next_phase,
			"attack_progression": [p for p in KILL_CHAIN_PHASES if kill_chain_coverage[p]],
			"mitigation_hints": mitigation_hints,
			"generated_at": _now_iso(),
		}

	# =========================================================================
	# Reporting & Sharing (6 methods)
	# =========================================================================

	async def generate_threat_report(
		self,
		classification: str,
		report_type: str,
		target_audience: str,
		title: str = "",
		summary: str = "",
		indicator_ids: list[str] | None = None,
		actor_ids: list[str] | None = None,
		campaign_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Generate a structured threat report.

		report_type: flash_report | assessment | weekly_digest | attribution_report
		classification: unclassified | tlp:green | tlp:amber | tlp:red
		"""
		cls_norm = classification.lower()
		assert cls_norm in VALID_REPORT_CLASSIFICATIONS, f"invalid classification '{classification}'"
		assert report_type in VALID_REPORT_TYPES, f"invalid report_type '{report_type}'"
		assert present(target_audience), "target_audience required"

		report_id = _uid()
		indicator_ids = indicator_ids or []
		actor_ids = actor_ids or []
		campaign_ids = campaign_ids or []

		ioc_store = self._ioc_store()
		indicators_section: list[dict] = [
			{k: v for k, v in ioc_store[iid].items() if not k.startswith("_")}
			for iid in indicator_ids if iid in ioc_store
		]
		actors_section: list[dict] = [
			self._actor_profiles[aid]
			for aid in actor_ids if aid in self._actor_profiles
		]
		cstore = self._campaign_store()
		campaigns_section: list[dict] = [
			cstore[cid]
			for cid in campaign_ids if cid in cstore
		]

		report: dict[str, Any] = {
			"id": report_id,
			"report_type": report_type,
			"classification": cls_norm,
			"target_audience": target_audience,
			"title": title or f"{report_type.replace('_', ' ').title()} — {_now_iso()[:10]}",
			"summary": summary,
			"indicator_count": len(indicators_section),
			"indicators": indicators_section,
			"actor_count": len(actors_section),
			"actors": actors_section,
			"campaign_count": len(campaigns_section),
			"campaigns": campaigns_section,
			"status": "draft",
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
			"tlp": cls_norm.replace("tlp:", "") if "tlp:" in cls_norm else "white",
		}
		self._threat_reports[report_id] = report
		self._audit("system", "threat_report_generated", report_id)
		return report

	async def share_via_taxii(
		self,
		report_id: str,
		taxii_server_url: str,
		collection_id: str,
	) -> dict[str, Any]:
		"""Simulate pushing a report to a TAXII 2.1 server collection.

		In production this would POST to <taxii_server_url>/api/v21/collections/<collection_id>/objects/
		using the report's STIX bundle payload. Here we log the intent and return a receipt.
		"""
		assert report_id in self._threat_reports, f"report '{report_id}' not found"
		assert present(taxii_server_url), "taxii_server_url required"
		assert present(collection_id), "collection_id required"

		report = self._threat_reports[report_id]

		# Build minimal STIX bundle for TAXII push
		bundle: dict[str, Any] = {
			"type": "bundle",
			"id": f"bundle--{_uid()}",
			"spec_version": "2.1",
			"objects": [
				{
					"type": "report",
					"id": f"report--{report_id}",
					"spec_version": "2.1",
					"created": report["created_at"],
					"modified": report["updated_at"],
					"name": report["title"],
					"published": _now_iso(),
					"report_types": [report["report_type"]],
					"object_refs": [ind.get("stix_id", f"indicator--{ind['id']}") for ind in report.get("indicators", [])],
				}
			],
		}

		log_entry: dict[str, Any] = {
			"id": _uid(),
			"report_id": report_id,
			"taxii_server_url": taxii_server_url,
			"collection_id": collection_id,
			"bundle_id": bundle["id"],
			"object_count": len(bundle["objects"]),
			"status": "submitted",
			"submitted_at": _now_iso(),
			"http_status": 200,
			"response": {"id": collection_id, "accepted": True},
		}
		self._taxii_log.append(log_entry)
		self._dissemination_log.append({
			**log_entry,
			"channel": "taxii",
		})
		self._audit("system", "taxii_push_submitted", report_id)
		return log_entry

	async def export_misp_event(self, indicator_ids: list[str]) -> dict[str, Any]:
		"""Export a set of indicators as a MISP JSON event.

		Returns a MISP-compatible event dict ready for import via the MISP API.
		"""
		assert indicator_ids, "at least one indicator_id required"

		ioc_store = self._ioc_store()
		misp_type_map = {
			"ip_address": "ip-dst",
			"domain": "domain",
			"url": "url",
			"file_hash_md5": "md5",
			"file_hash_sha1": "sha1",
			"file_hash_sha256": "sha256",
			"email": "email-src",
			"cve_id": "vulnerability",
			"yara_rule": "yara",
			"sigma_rule": "sigma",
		}

		attributes: list[dict[str, Any]] = []
		for iid in indicator_ids:
			if iid not in ioc_store:
				continue
			rec = ioc_store[iid]
			attributes.append({
				"uuid": iid,
				"type": misp_type_map.get(rec["ioc_type"], rec["ioc_type"]),
				"value": rec["value"],
				"category": "Network activity",
				"to_ids": True,
				"timestamp": int(datetime.fromisoformat(rec["created_at"].rstrip("Z")).timestamp()),
				"comment": str(rec.get("context", "")),
				"distribution": 1,
				"object_relation": None,
				"deleted": rec.get("status") == "retired",
			})

		event: dict[str, Any] = {
			"Event": {
				"uuid": _uid(),
				"info": f"APG MISP Export {_now_iso()[:10]}",
				"date": _now_iso()[:10],
				"threat_level_id": "2",
				"analysis": "2",
				"distribution": "1",
				"published": False,
				"timestamp": str(int(datetime.now(timezone.utc).timestamp())),
				"Attribute": attributes,
				"Tag": [{"name": "tlp:white", "colour": "#ffffff"}],
				"Org": {"name": "APG", "uuid": _uid()},
				"Orgc": {"name": "APG", "uuid": _uid()},
			}
		}
		self._audit("system", "misp_event_exported", f"count={len(attributes)}")
		return event

	async def intelligence_requirement(
		self,
		requirement_text: str,
		priority: str,
		requester: str,
	) -> dict[str, Any]:
		"""Register a Priority Intelligence Requirement (PIR).

		priority: critical | high | medium | low
		"""
		valid_priorities = {"critical", "high", "medium", "low"}
		assert requirement_text and requirement_text.strip(), "requirement_text required"
		assert priority.lower() in valid_priorities, f"invalid priority '{priority}'"
		assert present(requester), "requester required"

		req_id = _uid()
		req: dict[str, Any] = {
			"id": req_id,
			"requirement_text": requirement_text.strip(),
			"priority": priority.lower(),
			"requester": requester,
			"status": "open",
			"responses": [],
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
		}
		self._requirements[req_id] = req
		self._audit("system", "intelligence_requirement_created", req_id)
		return req

	async def dissemination_log(self, report_id: str) -> list[dict[str, Any]]:
		"""Return all dissemination events for a given report."""
		assert present(report_id), "report_id required"
		return [entry for entry in self._dissemination_log if entry.get("report_id") == report_id]

	async def confidence_calibration_report(
		self,
		analyst_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a confidence calibration report for an analyst over a period.

		period: ISO 8601 interval string e.g. "2024-01/2024-06" or "2024-Q1"
		Compares stated confidence vs subsequent confirmation rate.
		In production this queries historical verdict data.
		"""
		assert present(analyst_id), "analyst_id required"
		assert present(period), "period required"

		# Collect calibration records for analyst
		records = [r for r in self._calibration_records if r.get("analyst_id") == analyst_id]

		if not records:
			# Return skeleton calibration report
			return {
				"analyst_id": analyst_id,
				"period": period,
				"total_assessments": 0,
				"confirmed_correct": 0,
				"confirmed_incorrect": 0,
				"pending_confirmation": 0,
				"calibration_score": None,
				"brier_score": None,
				"overconfidence_bias": None,
				"underconfidence_bias": None,
				"recommendation": "Insufficient data for calibration analysis.",
				"generated_at": _now_iso(),
			}

		confirmed = [r for r in records if r.get("outcome") == "correct"]
		incorrect = [r for r in records if r.get("outcome") == "incorrect"]
		pending = [r for r in records if r.get("outcome") is None]

		# Brier score: mean((confidence - outcome)^2), outcome=1 for correct, 0 for incorrect
		brier_inputs = [r for r in records if r.get("outcome") in ("correct", "incorrect")]
		brier_score: float | None = None
		if brier_inputs:
			brier_score = sum(
				(r["stated_confidence"] - (1.0 if r["outcome"] == "correct" else 0.0)) ** 2
				for r in brier_inputs
			) / len(brier_inputs)

		avg_confidence = sum(r["stated_confidence"] for r in records) / len(records)
		actual_accuracy = len(confirmed) / len(brier_inputs) if brier_inputs else None

		overconfidence_bias = (avg_confidence - actual_accuracy) if actual_accuracy is not None else None

		return {
			"analyst_id": analyst_id,
			"period": period,
			"total_assessments": len(records),
			"confirmed_correct": len(confirmed),
			"confirmed_incorrect": len(incorrect),
			"pending_confirmation": len(pending),
			"average_stated_confidence": round(avg_confidence, 3),
			"actual_accuracy": round(actual_accuracy, 3) if actual_accuracy is not None else None,
			"calibration_score": round(1.0 - (brier_score or 0.0), 3),
			"brier_score": round(brier_score, 4) if brier_score is not None else None,
			"overconfidence_bias": round(overconfidence_bias, 3) if overconfidence_bias is not None else None,
			"underconfidence_bias": round(-overconfidence_bias, 3) if overconfidence_bias is not None and overconfidence_bias < 0 else None,
			"recommendation": (
				"Well calibrated." if overconfidence_bias is not None and abs(overconfidence_bias) < 0.05
				else "Analyst tends to overstate confidence — apply downward correction."
				if overconfidence_bias is not None and overconfidence_bias > 0.05
				else "Analyst tends to understate confidence — apply upward correction."
			),
			"generated_at": _now_iso(),
		}

	# =========================================================================
	# Feed Management (5 methods)
	# =========================================================================

	async def register_feed(
		self,
		name: str,
		url: str,
		format: str,
		auth_method: str,
		update_frequency: str,
	) -> dict[str, Any]:
		"""Register an external threat intelligence feed.

		format: stix | misp | csv | taxii | openioc | json
		auth_method: none | api_key | bearer_token | basic | mtls
		update_frequency: cron expression or interval string (e.g. "@hourly", "*/6 * * * *")
		"""
		fmt = format.lower()
		auth = auth_method.lower()
		assert present(name), "feed name required"
		assert present(url), "feed url required"
		assert fmt in VALID_FEED_FORMATS, f"unsupported feed format '{fmt}'"
		assert auth in VALID_AUTH_METHODS, f"unsupported auth_method '{auth}'"
		assert present(update_frequency), "update_frequency required"

		feed_id = _uid()
		feed: dict[str, Any] = {
			"id": feed_id,
			"name": name.strip(),
			"url": url.strip(),
			"format": fmt,
			"auth_method": auth,
			"update_frequency": update_frequency,
			"status": "registered",
			"last_ingested_at": None,
			"last_batch_id": None,
			"total_ingested": 0,
			"false_positive_count": 0,
			"created_at": _now_iso(),
			"updated_at": _now_iso(),
		}
		self._feeds[feed_id] = feed
		self._audit("system", "feed_registered", feed_id)
		return feed

	async def ingest_feed(self, feed_id: str) -> dict[str, Any]:
		"""Simulate pulling and parsing the latest indicators from a registered feed.

		In production this dispatches a Bytewax pipeline worker to pull, parse,
		deduplicate, and store indicators. Here we return a realistic batch summary.
		"""
		assert feed_id in self._feeds, f"feed '{feed_id}' not found"

		feed = self._feeds[feed_id]
		batch_id = _uid()

		# Simulate parsed counts
		total_parsed = 142
		duplicates = 17
		imported = total_parsed - duplicates
		errors = 2

		batch: dict[str, Any] = {
			"batch_id": batch_id,
			"feed_id": feed_id,
			"feed_name": feed["name"],
			"started_at": _now_iso(),
			"completed_at": _now_iso(),
			"total_parsed": total_parsed,
			"imported": imported,
			"duplicates_skipped": duplicates,
			"errors": errors,
			"status": "completed",
			"source_url": feed["url"],
			"format": feed["format"],
		}
		self._feed_batches.append(batch)

		# Update feed metadata
		feed.update({
			"last_ingested_at": _now_iso(),
			"last_batch_id": batch_id,
			"total_ingested": feed["total_ingested"] + imported,
			"status": "active",
			"updated_at": _now_iso(),
		})
		self._audit("system", "feed_ingested", feed_id)
		return batch

	async def feed_quality_report(self, feed_id: str) -> dict[str, Any]:
		"""Generate a quality assessment for a registered feed.

		Metrics: false positive rate, staleness ratio, volume trend, dedup rate.
		"""
		assert feed_id in self._feeds, f"feed '{feed_id}' not found"

		feed = self._feeds[feed_id]
		feed_batches = [b for b in self._feed_batches if b["feed_id"] == feed_id]

		total_imported = sum(b["imported"] for b in feed_batches)
		total_duplicates = sum(b["duplicates_skipped"] for b in feed_batches)
		total_errors = sum(b["errors"] for b in feed_batches)
		total_parsed = sum(b["total_parsed"] for b in feed_batches)
		false_positives = feed.get("false_positive_count", 0)

		fp_rate = false_positives / total_imported if total_imported else 0.0
		dedup_rate = total_duplicates / total_parsed if total_parsed else 0.0
		error_rate = total_errors / total_parsed if total_parsed else 0.0

		# Staleness: fraction of imported IOCs from this feed that are now retired
		ioc_store = self._ioc_store()
		feed_iocs = [r for r in ioc_store.values() if r.get("source") == feed["name"]]
		stale_count = sum(1 for r in feed_iocs if r.get("status") == "retired")
		staleness_ratio = stale_count / len(feed_iocs) if feed_iocs else 0.0

		quality_score = max(0.0, 1.0 - fp_rate - error_rate - (staleness_ratio * 0.5))

		return {
			"feed_id": feed_id,
			"feed_name": feed["name"],
			"batch_count": len(feed_batches),
			"total_parsed": total_parsed,
			"total_imported": total_imported,
			"total_duplicates_skipped": total_duplicates,
			"total_errors": total_errors,
			"false_positive_count": false_positives,
			"false_positive_rate": round(fp_rate, 4),
			"deduplication_rate": round(dedup_rate, 4),
			"error_rate": round(error_rate, 4),
			"staleness_ratio": round(staleness_ratio, 4),
			"quality_score": round(quality_score, 3),
			"quality_grade": (
				"A" if quality_score >= 0.9
				else "B" if quality_score >= 0.75
				else "C" if quality_score >= 0.6
				else "D"
			),
			"last_ingested_at": feed.get("last_ingested_at"),
			"generated_at": _now_iso(),
		}

	async def deduplicate_from_feed(self, feed_id: str, batch_id: str) -> dict[str, Any]:
		"""Identify and remove duplicate indicators imported in a specific feed batch.

		Deduplication is fingerprint-based (SHA-256 of ioc_type:value.lower()).
		Returns counts of reviewed, duplicates found, and retired duplicates.
		"""
		assert feed_id in self._feeds, f"feed '{feed_id}' not found"
		assert present(batch_id), "batch_id required"

		ioc_store = self._ioc_store()
		fingerprint_map: dict[str, list[str]] = {}

		for iid, record in ioc_store.items():
			fp = record.get("_fingerprint")
			if fp:
				if fp not in fingerprint_map:
					fingerprint_map[fp] = []
				fingerprint_map[fp].append(iid)

		duplicates_retired: list[str] = []
		duplicate_groups: list[dict[str, Any]] = []

		for fp, ids in fingerprint_map.items():
			if len(ids) <= 1:
				continue
			# Keep earliest, retire the rest
			ids_sorted = sorted(ids, key=lambda i: ioc_store[i].get("created_at", ""))
			canonical = ids_sorted[0]
			duplicates = ids_sorted[1:]

			for dup_id in duplicates:
				rec = ioc_store[dup_id]
				if rec.get("status") != "retired":
					rec.update({
						"status": "retired",
						"retirement_reason": f"dedup_batch_{batch_id}",
						"canonical_id": canonical,
						"retired_at": _now_iso(),
						"updated_at": _now_iso(),
					})
					duplicates_retired.append(dup_id)

			if duplicates:
				duplicate_groups.append({
					"fingerprint": fp,
					"canonical_id": canonical,
					"retired_ids": duplicates,
				})

		self._audit("system", "feed_deduplication_run", f"batch={batch_id} retired={len(duplicates_retired)}")
		return {
			"feed_id": feed_id,
			"batch_id": batch_id,
			"total_indicators_reviewed": len(ioc_store),
			"duplicate_groups_found": len(duplicate_groups),
			"duplicates_retired": len(duplicates_retired),
			"duplicate_groups": duplicate_groups,
			"run_at": _now_iso(),
		}

	async def feeds_dashboard(self) -> dict[str, Any]:
		"""Return a summary dashboard of all registered feeds.

		Includes per-feed status, last ingestion, total indicators, and quality grade.
		"""
		feeds_summary: list[dict[str, Any]] = []
		total_imported = 0

		for feed_id, feed in self._feeds.items():
			feed_batches = [b for b in self._feed_batches if b["feed_id"] == feed_id]
			imported = sum(b["imported"] for b in feed_batches)
			total_imported += imported

			# Minimal quality score inline
			fp_rate = feed.get("false_positive_count", 0) / max(imported, 1)
			quality_score = max(0.0, 1.0 - fp_rate)

			feeds_summary.append({
				"feed_id": feed_id,
				"name": feed["name"],
				"format": feed["format"],
				"status": feed["status"],
				"last_ingested_at": feed.get("last_ingested_at"),
				"batch_count": len(feed_batches),
				"total_imported": imported,
				"quality_grade": (
					"A" if quality_score >= 0.9
					else "B" if quality_score >= 0.75
					else "C" if quality_score >= 0.6
					else "D"
				),
			})

		feeds_summary.sort(key=lambda f: f["total_imported"], reverse=True)

		return {
			"total_feeds": len(self._feeds),
			"active_feeds": sum(1 for f in self._feeds.values() if f["status"] == "active"),
			"total_indicators_from_feeds": total_imported,
			"feeds": feeds_summary,
			"generated_at": _now_iso(),
		}

	# =========================================================================
	# Vulnerability Intelligence (2 methods)
	# =========================================================================

	async def vulnerability_triage(
		self,
		cve_ids: list[str],
	) -> list[dict[str, Any]]:
		"""Score and rank CVEs by exploitation probability for patch prioritisation.

		Combines CVSS base score, EPSS probability estimate, CISA KEV membership,
		public PoC availability, and in-the-wild exploitation status into a composite
		priority score. Returns the list sorted by priority_score descending.

		In production this queries NVD, FIRST EPSS API, and CISA KEV feed.
		Here we compute a deterministic mock score from the CVE-ID string.
		"""
		assert cve_ids, "at least one CVE ID required"

		results: list[dict[str, Any]] = []
		ioc_store = self._ioc_store()

		for cve_id in cve_ids:
			assert cve_id.upper().startswith("CVE-"), f"invalid CVE ID format: '{cve_id}'"

			# Pull enrichment if available; otherwise synthesise realistic fields
			enrichment = self._enrichments.get(cve_id, {})
			cvss = enrichment.get("cvss_v3", {})

			# Deterministic mock EPSS from CVE year + sequence hash
			parts = cve_id.upper().split("-")
			seq_hash = int(hashlib.md5(cve_id.encode()).hexdigest()[:4], 16)
			epss_probability = round(min(1.0, (seq_hash % 1000) / 1000.0), 4)
			cvss_base = cvss.get("base_score", round(5.0 + (seq_hash % 50) / 10.0, 1))
			cvss_severity = cvss.get("severity", "HIGH" if cvss_base >= 7.0 else "MEDIUM")

			exploit_public = enrichment.get("exploit_public", epss_probability > 0.3)
			exploit_wild = enrichment.get("exploit_in_wild", epss_probability > 0.6)
			cisa_kev = enrichment.get("cisa_kev", epss_probability > 0.7)
			patch_available = enrichment.get("patch_available", True)

			# Weighted composite: CVSS(30%) + EPSS(35%) + KEV(20%) + wild(10%) + no_patch(5%)
			priority_score = round(
				(cvss_base / 10.0) * 0.30
				+ epss_probability * 0.35
				+ (0.20 if cisa_kev else 0.0)
				+ (0.10 if exploit_wild else 0.0)
				+ (0.05 if not patch_available else 0.0),
				4,
			)

			priority_label = (
				"CRITICAL" if priority_score >= 0.75
				else "HIGH" if priority_score >= 0.55
				else "MEDIUM" if priority_score >= 0.35
				else "LOW"
			)

			results.append({
				"cve_id": cve_id.upper(),
				"cvss_base_score": cvss_base,
				"cvss_severity": cvss_severity,
				"epss_probability": epss_probability,
				"exploit_public": exploit_public,
				"exploit_in_wild": exploit_wild,
				"cisa_kev": cisa_kev,
				"patch_available": patch_available,
				"priority_score": priority_score,
				"priority_label": priority_label,
				"recommendation": (
					"Patch immediately — active exploitation and KEV-listed."
					if cisa_kev and exploit_wild
					else "Patch within 7 days — public exploit available."
					if exploit_public
					else "Schedule patch in next maintenance window."
					if cvss_base >= 7.0
					else "Monitor; patch at next regular cycle."
				),
				"nvd_url": f"https://nvd.nist.gov/vuln/detail/{cve_id.upper()}",
				"assessed_at": _now_iso(),
			})

		results.sort(key=lambda r: r["priority_score"], reverse=True)
		self._audit("system", "vulnerability_triage_run", f"count={len(results)}")
		return results

	async def ingest_sandbox_report(
		self,
		sandbox_report: dict[str, Any],
		campaign_id: str | None = None,
	) -> dict[str, Any]:
		"""Parse a behavioral sandbox report and extract indicators and technique links.

		Accepts Cuckoo / Any.run / Triage JSON report format (normalized subset).
		Extracts: dropped file hashes, network IOCs (IPs, domains, URLs),
		registry mutations, process trees, and MITRE ATT&CK annotations.
		Stores all extracted IOCs and links them to the given campaign if provided.

		Returns a summary with extracted counts and new indicator IDs.
		"""
		assert isinstance(sandbox_report, dict), "sandbox_report must be a dict"
		report_type = sandbox_report.get("type", "generic")

		extracted_iocs: list[dict[str, Any]] = []
		techniques_linked: list[str] = []
		errors: list[str] = []

		ioc_store = self._ioc_store()

		# --- Network indicators ---
		network = sandbox_report.get("network", {})
		for host in network.get("hosts", []):
			ip = host if isinstance(host, str) else host.get("ip", "")
			if ip:
				fp = hashlib.sha256(f"ip_address:{ip.lower()}".encode()).hexdigest()
				if not any(r.get("_fingerprint") == fp for r in ioc_store.values()):
					iid = _uid()
					ioc_store[iid] = {
						"id": iid, "ioc_type": "ip_address", "value": ip,
						"confidence": 0.75, "tlp": "amber", "source": "sandbox",
						"context": {"report_type": report_type},
						"status": "active", "created_at": _now_iso(), "updated_at": _now_iso(),
						"stix_type": "ipv4-addr", "stix_id": f"ipv4-addr--{iid}",
						"_fingerprint": fp,
					}
					extracted_iocs.append({"ioc_type": "ip_address", "value": ip, "id": iid})

		for domain_entry in network.get("domains", []):
			domain = domain_entry if isinstance(domain_entry, str) else domain_entry.get("domain", "")
			if domain:
				fp = hashlib.sha256(f"domain:{domain.lower()}".encode()).hexdigest()
				if not any(r.get("_fingerprint") == fp for r in ioc_store.values()):
					iid = _uid()
					ioc_store[iid] = {
						"id": iid, "ioc_type": "domain", "value": domain,
						"confidence": 0.70, "tlp": "amber", "source": "sandbox",
						"context": {"report_type": report_type},
						"status": "active", "created_at": _now_iso(), "updated_at": _now_iso(),
						"stix_type": "domain-name", "stix_id": f"domain-name--{iid}",
						"_fingerprint": fp,
					}
					extracted_iocs.append({"ioc_type": "domain", "value": domain, "id": iid})

		# --- Dropped files (hashes) ---
		for dropped in sandbox_report.get("dropped", []):
			for algo, ioc_type in [("sha256", "file_hash_sha256"), ("sha1", "file_hash_sha1"), ("md5", "file_hash_md5")]:
				h = dropped.get(algo, dropped.get(algo.upper(), ""))
				if h:
					fp = hashlib.sha256(f"{ioc_type}:{h.lower()}".encode()).hexdigest()
					if not any(r.get("_fingerprint") == fp for r in ioc_store.values()):
						iid = _uid()
						ioc_store[iid] = {
							"id": iid, "ioc_type": ioc_type, "value": h,
							"confidence": 0.85, "tlp": "amber", "source": "sandbox",
							"context": {"report_type": report_type, "file_name": dropped.get("name", "")},
							"status": "active", "created_at": _now_iso(), "updated_at": _now_iso(),
							"stix_type": "file", "stix_id": f"file--{iid}",
							"_fingerprint": fp,
						}
						extracted_iocs.append({"ioc_type": ioc_type, "value": h, "id": iid})
					break  # one hash per file is enough

		# --- MITRE ATT&CK technique annotations ---
		for sig in sandbox_report.get("signatures", []):
			for tid in sig.get("ttp", sig.get("ttps", [])):
				if tid in MITRE_TECHNIQUES:
					techniques_linked.append(tid)

		# --- Link to campaign if requested ---
		campaign_links: list[dict[str, Any]] = []
		if campaign_id:
			cstore = self._campaign_store()
			if campaign_id not in cstore:
				errors.append(f"campaign '{campaign_id}' not found; IOCs stored but not linked")
			else:
				now = _now_iso()
				for ioc_rec in extracted_iocs:
					# Skip already-linked
					if any(ci["campaign_id"] == campaign_id and ci["indicator_id"] == ioc_rec["id"]
						   for ci in self._campaign_indicators):
						continue
					link: dict[str, Any] = {
						"id": _uid(), "campaign_id": campaign_id, "indicator_id": ioc_rec["id"],
						"first_seen": now, "last_seen": now, "added_at": now,
					}
					self._campaign_indicators.append(link)
					campaign_links.append(link)

				for tid in set(techniques_linked):
					if any(ct["campaign_id"] == campaign_id and ct["technique_id"] == tid
						   for ct in self._campaign_techniques):
						continue
					meta = MITRE_TECHNIQUES[tid]
					ct_link: dict[str, Any] = {
						"id": _uid(), "campaign_id": campaign_id, "technique_id": tid,
						"technique_name": meta["name"], "tactic": meta["tactic"],
						"notes": f"Auto-extracted from sandbox report ({report_type})",
						"added_at": _now_iso(), "is_verified_technique": True,
					}
					self._campaign_techniques.append(ct_link)

		self._audit("system", "sandbox_report_ingested", f"iocs={len(extracted_iocs)}")
		return {
			"report_type": report_type,
			"extracted_ioc_count": len(extracted_iocs),
			"extracted_iocs": extracted_iocs,
			"techniques_linked_count": len(set(techniques_linked)),
			"techniques_linked": list(set(techniques_linked)),
			"campaign_id": campaign_id,
			"campaign_links_created": len(campaign_links),
			"errors": errors,
			"ingested_at": _now_iso(),
		}

	# =========================================================================
	# Attribution Intelligence (2 methods)
	# =========================================================================

	async def reverse_attribution(self, indicator_id: str) -> list[dict[str, Any]]:
		"""Rank candidate threat actors for an unknown indicator.

		Walks indicator -> campaign -> actor link graph, scores each candidate
		actor by sum-of-weighted-link-confidences and TTP profile overlap.
		Returns ranked list of candidate actors with confidence intervals and
		supporting evidence chains.
		"""
		ioc_store = self._ioc_store()
		assert indicator_id in ioc_store, f"indicator '{indicator_id}' not found"

		# Step 1: find campaigns that reference this indicator
		campaign_ids: list[str] = [
			ci["campaign_id"] for ci in self._campaign_indicators
			if ci["indicator_id"] == indicator_id
		]

		# Step 2: collect actor IDs linked to those campaigns
		actor_scores: dict[str, dict[str, Any]] = {}
		for cid in campaign_ids:
			for link in self._actor_campaign_links:
				if link["campaign_id"] != cid:
					continue
				aid = link["actor_id"]
				if aid not in actor_scores:
					actor_scores[aid] = {
						"actor_id": aid, "score": 0.0,
						"evidence_chain": [], "campaign_links": [],
					}
				actor_scores[aid]["score"] += 0.5
				actor_scores[aid]["campaign_links"].append(cid)
				actor_scores[aid]["evidence_chain"].append(
					f"campaign '{cid}' shares this indicator"
				)

		# Step 3: boost via direct actor-indicator links
		for link in self._actor_indicator_links:
			if link["indicator_id"] != indicator_id:
				continue
			aid = link["actor_id"]
			if aid not in actor_scores:
				actor_scores[aid] = {
					"actor_id": aid, "score": 0.0,
					"evidence_chain": [], "campaign_links": [],
				}
			actor_scores[aid]["score"] += link["confidence"]
			actor_scores[aid]["evidence_chain"].append(
				f"direct actor-indicator link (type={link['relationship_type']}, conf={link['confidence']})"
			)

		# Step 4: attach profile data and normalise
		results: list[dict[str, Any]] = []
		max_score = max((v["score"] for v in actor_scores.values()), default=1.0) or 1.0
		for aid, data in actor_scores.items():
			profile = self._actor_profiles.get(aid, {})
			normalised = round(data["score"] / max_score, 4)
			results.append({
				"actor_id": aid,
				"actor_name": profile.get("name", "unknown"),
				"origin_country": profile.get("origin_country", "??"),
				"motivation": profile.get("motivation", "unknown"),
				"sophistication": profile.get("sophistication", "unknown"),
				"attribution_confidence": normalised,
				"confidence_interval": {
					"lower": round(max(0.0, normalised - 0.15), 4),
					"upper": round(min(1.0, normalised + 0.15), 4),
				},
				"evidence_chain": data["evidence_chain"],
				"campaign_count": len(set(data["campaign_links"])),
			})

		results.sort(key=lambda r: r["attribution_confidence"], reverse=True)
		self._audit("system", "reverse_attribution_run", indicator_id)
		return results

	async def generate_simulation_playbook(
		self,
		actor_id: str,
		target_platform: str,
	) -> dict[str, Any]:
		"""Generate an adversary simulation playbook from an actor profile.

		Maps the actor's verified TTPs to ordered attack steps compatible with
		CALDERA / Atomic Red Team. Each step includes technique metadata,
		expected artifact types, and detection opportunity notes.

		target_platform: windows | linux | macos | cloud
		"""
		valid_platforms = {"windows", "linux", "macos", "cloud"}
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"
		assert target_platform.lower() in valid_platforms, \
			f"invalid target_platform '{target_platform}'; choose from {valid_platforms}"

		profile = self._actor_profiles[actor_id]
		ttps = profile.get("ttps_verified", profile.get("ttps", []))

		# Group by kill-chain phase for ordered playbook steps
		phase_buckets: dict[str, list[dict[str, Any]]] = {p: [] for p in KILL_CHAIN_PHASES}

		for tid in ttps:
			if tid not in MITRE_TECHNIQUES:
				continue
			meta = MITRE_TECHNIQUES[tid]
			phase = TACTIC_TO_KILL_CHAIN.get(meta["tactic"], "actions-on-objectives")
			phase_buckets[phase].append({
				"step_type": "attack",
				"technique_id": tid,
				"technique_name": meta["name"],
				"tactic": meta["tactic"],
				"kill_chain_phase": phase,
				"atomic_test_ref": f"https://github.com/redcanaryco/atomic-red-team/blob/master/atomics/{tid}/{tid}.md",
				"caldera_ability_hint": tid.replace(".", "_").lower(),
				"expected_artifacts": _playbook_artifacts(tid, target_platform.lower()),
				"detection_opportunity": _playbook_detection(tid),
			})

		# Flatten into ordered steps
		ordered_steps: list[dict[str, Any]] = []
		step_num = 1
		for phase in KILL_CHAIN_PHASES:
			for step in phase_buckets[phase]:
				step["step_number"] = step_num
				ordered_steps.append(step)
				step_num += 1

		playbook_id = _uid()
		playbook: dict[str, Any] = {
			"id": playbook_id,
			"actor_id": actor_id,
			"actor_name": profile.get("name"),
			"target_platform": target_platform.lower(),
			"step_count": len(ordered_steps),
			"steps": ordered_steps,
			"tactic_coverage": sorted({s["tactic"] for s in ordered_steps}),
			"kill_chain_phases_covered": sorted({s["kill_chain_phase"] for s in ordered_steps}),
			"estimated_duration_minutes": len(ordered_steps) * 15,
			"framework_compatibility": ["CALDERA", "Atomic Red Team", "VECTR"],
			"created_at": _now_iso(),
		}
		self._audit("system", "simulation_playbook_generated", actor_id)
		return playbook

	# =========================================================================
	# Temporal Intelligence (2 methods)
	# =========================================================================

	async def apply_confidence_decay(self, indicator_id: str) -> dict[str, Any]:
		"""Apply time-based confidence decay to a stored indicator.

		Different IOC types age at different rates (half-lives in days):
		  ip_address: 14d | domain: 30d | url: 21d | file_hash_*: 180d
		  email: 60d | cve_id: 365d | yara_rule: 120d | sigma_rule: 120d

		Updates the indicator's confidence score in place and returns the
		decay record with original/new confidence and decay factor.
		"""
		ioc_store = self._ioc_store()
		assert indicator_id in ioc_store, f"indicator '{indicator_id}' not found"

		record = ioc_store[indicator_id]
		if record.get("status") == "retired":
			return {
				"indicator_id": indicator_id,
				"status": "retired",
				"decay_applied": False,
				"note": "retired indicators are not decayed",
			}

		half_lives: dict[str, int] = {
			"ip_address": 14, "domain": 30, "url": 21,
			"file_hash_md5": 180, "file_hash_sha1": 180, "file_hash_sha256": 180,
			"email": 60, "cve_id": 365, "yara_rule": 120, "sigma_rule": 120,
		}
		ioc_type = record["ioc_type"]
		half_life = half_lives.get(ioc_type, 60)

		created_at = record.get("created_at", _now_iso())
		try:
			created_dt = datetime.fromisoformat(created_at.rstrip("Z"))
			if created_dt.tzinfo is None:
				created_dt = created_dt.replace(tzinfo=timezone.utc)
		except ValueError:
			created_dt = datetime.now(timezone.utc)

		age_days = (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400.0
		# Exponential decay: C(t) = C0 * 0.5^(t / half_life)
		decay_factor = math.pow(0.5, age_days / half_life)
		original_confidence = record["confidence"]
		new_confidence = round(original_confidence * decay_factor, 4)

		record["confidence"] = new_confidence
		record["updated_at"] = _now_iso()
		record["decay_applied_at"] = _now_iso()

		self._audit("system", "confidence_decay_applied", indicator_id)
		return {
			"indicator_id": indicator_id,
			"ioc_type": ioc_type,
			"age_days": round(age_days, 2),
			"half_life_days": half_life,
			"original_confidence": original_confidence,
			"decay_factor": round(decay_factor, 6),
			"new_confidence": new_confidence,
			"decay_applied": True,
			"retired_by_decay": new_confidence < 0.1,
			"applied_at": _now_iso(),
		}

	async def longitudinal_trend_analysis(
		self,
		period_days: int = 90,
		bucket: str = "weekly",
	) -> dict[str, Any]:
		"""Bucket indicator and campaign activity into time-series for trend analysis.

		bucket: daily | weekly | monthly
		Returns time-series dicts per bucket with counts of new indicators,
		new campaigns, and actor sightings. Includes simple moving-average
		smoothing and a narrative trend summary.
		"""
		valid_buckets = {"daily", "weekly", "monthly"}
		assert period_days > 0, "period_days must be positive"
		assert bucket in valid_buckets, f"bucket must be one of {valid_buckets}"

		bucket_days = {"daily": 1, "weekly": 7, "monthly": 30}[bucket]
		num_buckets = max(1, period_days // bucket_days)
		now = datetime.now(timezone.utc)

		def _bucket_index(iso_ts: str) -> int | None:
			try:
				dt = datetime.fromisoformat(iso_ts.rstrip("Z")).replace(tzinfo=timezone.utc)
			except ValueError:
				return None
			age = (now - dt).total_seconds() / 86400.0
			if age < 0 or age > period_days:
				return None
			return num_buckets - 1 - int(age // bucket_days)

		# Initialise buckets
		ioc_counts = [0] * num_buckets
		campaign_counts = [0] * num_buckets
		actor_counts = [0] * num_buckets

		for rec in self._ioc_store().values():
			idx = _bucket_index(rec.get("created_at", ""))
			if idx is not None:
				ioc_counts[idx] += 1

		for rec in self._campaign_store().values():
			idx = _bucket_index(rec.get("created_at", ""))
			if idx is not None:
				campaign_counts[idx] += 1

		for rec in self._actor_profiles.values():
			idx = _bucket_index(rec.get("created_at", ""))
			if idx is not None:
				actor_counts[idx] += 1

		# Simple 3-period moving average
		def _sma(series: list[int], window: int = 3) -> list[float]:
			result: list[float] = []
			for i in range(len(series)):
				start = max(0, i - window + 1)
				result.append(round(sum(series[start:i + 1]) / (i - start + 1), 2))
			return result

		# Trend direction: compare last 25% average vs previous 25%
		def _trend_label(series: list[int]) -> str:
			n = len(series)
			if n < 4:
				return "insufficient_data"
			q = max(1, n // 4)
			recent_avg = sum(series[-q:]) / q
			prior_avg = sum(series[-2 * q:-q]) / q
			if prior_avg == 0:
				return "emerging" if recent_avg > 0 else "flat"
			ratio = recent_avg / prior_avg
			if ratio >= 1.25:
				return "accelerating"
			if ratio <= 0.75:
				return "decelerating"
			return "stable"

		# Build bucket labels (ISO week start dates, approximate)
		labels: list[str] = []
		for i in range(num_buckets):
			bucket_start = now - timedelta(days=(num_buckets - 1 - i) * bucket_days)
			labels.append(bucket_start.strftime("%Y-%m-%d"))

		ioc_sma = _sma(ioc_counts)
		campaign_sma = _sma(campaign_counts)

		ioc_trend = _trend_label(ioc_counts)
		campaign_trend = _trend_label(campaign_counts)

		narrative = (
			f"Over the past {period_days} days, indicator volume is {ioc_trend} "
			f"and campaign activity is {campaign_trend}. "
			f"Total: {sum(ioc_counts)} new indicators, {sum(campaign_counts)} new campaigns "
			f"across {num_buckets} {bucket} buckets."
		)

		self._audit("system", "longitudinal_trend_analysis_run", f"period={period_days}d bucket={bucket}")
		return {
			"period_days": period_days,
			"bucket": bucket,
			"bucket_count": num_buckets,
			"labels": labels,
			"indicators": {
				"counts": ioc_counts,
				"sma": ioc_sma,
				"total": sum(ioc_counts),
				"trend": ioc_trend,
			},
			"campaigns": {
				"counts": campaign_counts,
				"sma": campaign_sma,
				"total": sum(campaign_counts),
				"trend": campaign_trend,
			},
			"actors": {
				"counts": actor_counts,
				"total": sum(actor_counts),
			},
			"narrative": narrative,
			"generated_at": _now_iso(),
		}

	# =========================================================================
	# PIR & Hunting (2 methods)
	# =========================================================================

	async def score_pir_satisfaction(self, requirement_id: str) -> dict[str, Any]:
		"""Score how well collected intelligence satisfies a Priority Intelligence Requirement.

		Matches the PIR's requirement_text against indicator values, campaign names,
		actor names, and report titles using keyword overlap. Returns a satisfaction
		score (0–1) and a list of contributing artifact IDs ranked by relevance.
		"""
		assert requirement_id in self._requirements, \
			f"requirement '{requirement_id}' not found"

		req = self._requirements[requirement_id]
		req_text = req["requirement_text"].lower()
		keywords: list[str] = [
			w for w in re.split(r"\W+", req_text) if len(w) > 3
		]
		if not keywords:
			return {
				"requirement_id": requirement_id,
				"satisfaction_score": 0.0,
				"contributing_artifacts": [],
				"note": "no meaningful keywords extracted from requirement text",
			}

		hits: list[dict[str, Any]] = []

		def _kw_score(text: str) -> float:
			t = text.lower()
			matched = sum(1 for kw in keywords if kw in t)
			return matched / len(keywords)

		for iid, rec in self._ioc_store().items():
			score = _kw_score(rec["value"] + " " + str(rec.get("context", "")))
			if score > 0:
				hits.append({"artifact_type": "indicator", "artifact_id": iid,
							 "relevance": round(score, 4)})

		for cid, camp in self._campaign_store().items():
			score = _kw_score(camp.get("name", "") + " " + camp.get("objective", ""))
			if score > 0:
				hits.append({"artifact_type": "campaign", "artifact_id": cid,
							 "relevance": round(score, 4)})

		for aid, actor in self._actor_profiles.items():
			score = _kw_score(actor.get("name", "") + " " + " ".join(actor.get("aliases", [])))
			if score > 0:
				hits.append({"artifact_type": "actor", "artifact_id": aid,
							 "relevance": round(score, 4)})

		for rid, report in self._threat_reports.items():
			score = _kw_score(report.get("title", "") + " " + report.get("summary", ""))
			if score > 0:
				hits.append({"artifact_type": "report", "artifact_id": rid,
							 "relevance": round(score, 4)})

		hits.sort(key=lambda h: h["relevance"], reverse=True)
		top_hits = hits[:20]

		# Satisfaction score: diminishing returns on additional hits
		if not hits:
			satisfaction = 0.0
		else:
			max_relevance = max(h["relevance"] for h in hits)
			satisfaction = round(min(1.0, max_relevance + math.log1p(len(hits)) * 0.05), 4)

		# Update PIR record
		req["satisfaction_score"] = satisfaction
		req["last_scored_at"] = _now_iso()
		req["responses"] = [h["artifact_id"] for h in top_hits[:5]]

		self._audit("system", "pir_satisfaction_scored", requirement_id)
		return {
			"requirement_id": requirement_id,
			"requirement_text": req["requirement_text"],
			"priority": req["priority"],
			"satisfaction_score": satisfaction,
			"satisfaction_label": (
				"satisfied" if satisfaction >= 0.7
				else "partial" if satisfaction >= 0.35
				else "unsatisfied"
			),
			"contributing_artifact_count": len(hits),
			"contributing_artifacts": top_hits,
			"scored_at": _now_iso(),
		}

	async def generate_hunting_hypotheses(
		self,
		actor_id: str,
		target_platform: str,
	) -> list[dict[str, Any]]:
		"""Generate concrete threat-hunting hypotheses from an actor's TTP profile.

		Each hypothesis includes a Sigma-compatible detection logic sketch,
		key observable fields, and a hunt priority based on technique frequency
		and actor sophistication. target_platform: windows | linux | macos | cloud.
		"""
		valid_platforms = {"windows", "linux", "macos", "cloud"}
		assert actor_id in self._actor_profiles, f"actor '{actor_id}' not found"
		assert target_platform.lower() in valid_platforms, \
			f"invalid target_platform '{target_platform}'"

		profile = self._actor_profiles[actor_id]
		platform = target_platform.lower()
		ttps = profile.get("ttps_verified", profile.get("ttps", []))
		sophistication = profile.get("sophistication", "intermediate")

		# Sophistication -> hunt priority multiplier
		soph_multiplier = {
			"minimal": 0.6, "intermediate": 0.8,
			"advanced": 1.0, "nation-state": 1.2,
		}.get(sophistication, 0.8)

		hypotheses: list[dict[str, Any]] = []
		for tid in ttps:
			if tid not in MITRE_TECHNIQUES:
				continue
			meta = MITRE_TECHNIQUES[tid]
			base_priority = 0.7  # most known-actor techniques warrant high attention

			hypothesis: dict[str, Any] = {
				"hypothesis_id": _uid(),
				"actor_id": actor_id,
				"technique_id": tid,
				"technique_name": meta["name"],
				"tactic": meta["tactic"],
				"hunt_priority": round(min(1.0, base_priority * soph_multiplier), 3),
				"platform": platform,
				"sigma_sketch": _sigma_sketch(tid, platform),
				"observable_fields": _observable_fields(tid, platform),
				"false_positive_notes": _fp_notes(tid),
				"mitre_url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/",
				"generated_at": _now_iso(),
			}
			hypotheses.append(hypothesis)

		hypotheses.sort(key=lambda h: h["hunt_priority"], reverse=True)
		self._audit("system", "hunting_hypotheses_generated", actor_id)
		return hypotheses

	# =========================================================================
	# Private helpers (extended)
	# =========================================================================

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> ThreatAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> ThreatWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> ThreatSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_actor_or_none(self, item_id: str, tenant_id: str) -> ThreatActor | None:
		return self.actors.get(self._tenant_key(tenant_id, item_id))

	def _tenant_campaign_or_none(self, item_id: str, tenant_id: str) -> ThreatCampaign | None:
		return self.campaigns.get(self._tenant_key(tenant_id, item_id))

	def _tenant_assessment_or_none(self, item_id: str, tenant_id: str) -> ThreatAssessment | None:
		return self.assessments.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"timestamp": _now_iso(),
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "threat_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "threat_policy_denied")

	def _ioc_store(self) -> dict[str, dict[str, Any]]:
		"""Lazy-init the in-process IOC store."""
		if not hasattr(self, "_ioc_store_data"):
			self._ioc_store_data: dict[str, dict[str, Any]] = {}
		return self._ioc_store_data

	def _campaign_store(self) -> dict[str, dict[str, Any]]:
		"""Lazy-init the in-process extended campaign store."""
		if not hasattr(self, "_campaign_store_data"):
			self._campaign_store_data: dict[str, dict[str, Any]] = {}
		return self._campaign_store_data

	def _get_campaign_any_tenant(self, campaign_id: str) -> ThreatCampaign | None:
		"""Scan across all tenants for a campaign by ID."""
		for (_, cid), campaign in self.campaigns.items():
			if cid == campaign_id:
				return campaign
		return None

	async def ml_threat_score(self, *args, **kwargs):
		"""AI-powered threat intelligence scoring and priority classification. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability  # type: ignore
			ml = MLCapability()
			result = await ml.score(kwargs, task="threat_intelligence_priority")
			return {"threat_priority": round(result.score, 3), "threat_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}


# Alias for backward compatibility
IntelThreatsService = ThreatIntelligenceService
