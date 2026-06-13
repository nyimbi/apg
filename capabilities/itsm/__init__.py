"""
ITSM — IT Service Management Domain
© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

ITIL v4 aligned capability set covering the full IT service lifecycle:
  - cmdb: Configuration Management Database (assets, relationships, discovery, health)
  - inc:  Incident Management (P1–P4 SLA, escalation, major incident, PIR)
  - prb:  Problem Management (RCA, KEDB, workarounds)
  - chg:  Change Management (CAB workflow, schedule, conflict detection, PIR)

Composition graph (NATS subjects):
  itsm_cmdb → itsm_inc  (ci_failure → incident.created)
  itsm_cmdb → itsm_chg  (ci.change_requested → change.created)
  itsm_inc  → itsm_prb  (incident.recurring → problem.created)
  itsm_prb  → itsm_chg  (problem.permanent_fix → change.created)
  intel_alerts → itsm_inc (alert_created → incident.created)
"""

from typing import Any

CAPABILITY_META: dict[str, Any] = {
	"name": "IT Service Management",
	"code": "ITSM",
	"version": "1.0.0",
	"description": "ITIL v4 aligned IT service management platform",
	"industry_focus": "Enterprise IT, Managed Services, Cloud Operations, DevOps",
	"subcapabilities": ["cmdb", "inc", "prb", "chg"],
	"implemented_subcapabilities": ["cmdb", "inc", "prb", "chg"],
	"itil_alignment": "ITIL v4",
	"database_prefix": "it_",
	"menu_category": "IT Service Management",
	"menu_icon": "fa-server",
	"nats_subjects": {
		"published": [
			"itsm.cmdb.ci.registered",
			"itsm.cmdb.ci.updated",
			"itsm.cmdb.ci.decommissioned",
			"itsm.cmdb.ci.change_requested",
			"itsm.inc.incident.created",
			"itsm.inc.incident.resolved",
			"itsm.inc.incident.escalated",
			"itsm.inc.major.declared",
			"itsm.prb.problem.created",
			"itsm.prb.known_error.registered",
			"itsm.chg.change.submitted",
			"itsm.chg.change.implemented",
			"itsm.chg.change.failed",
		],
		"subscribed": [
			"intel.alerts.alert_created",
		],
	},
	"dependencies": ["auth", "audl", "ntfy", "grph", "tmprl"],
}


def get_capability_info() -> dict[str, Any]:
	return CAPABILITY_META


def get_subcapabilities() -> list[str]:
	return CAPABILITY_META["subcapabilities"]


def get_nats_subjects() -> dict[str, list[str]]:
	return CAPABILITY_META["nats_subjects"]


# Lazy imports to avoid circular deps at module load time
def get_cmdb_service() -> Any:
	from .cmdb import CmdbService
	return CmdbService()


def get_inc_service() -> Any:
	from .inc import IncidentManagementService
	return IncidentManagementService()


def get_prb_service() -> Any:
	from .prb import ProblemManagementService
	return ProblemManagementService()


def get_chg_service() -> Any:
	from .chg import ChangeManagementService
	return ChangeManagementService()


__all__ = [
	"CAPABILITY_META",
	"get_capability_info",
	"get_subcapabilities",
	"get_nats_subjects",
	"get_cmdb_service",
	"get_inc_service",
	"get_prb_service",
	"get_chg_service",
]
