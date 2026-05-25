"""Data models for the common APG intelligent-agent capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from uuid_extensions import uuid7str


class AgentType(StrEnum):
	"""Supported APG agent topology types."""

	WORKER = "worker"
	COORDINATOR = "coordinator"
	SPECIALIST = "specialist"
	OBSERVER = "observer"


class AgentRole(StrEnum):
	"""Supported APG agent responsibilities."""

	TASK_MANAGER = "task_manager"
	ORCHESTRATOR = "orchestrator"
	ARCHITECT = "architect"
	DEVELOPER = "developer"
	TESTER = "tester"
	DEVOPS = "devops"


@dataclass
class IntelligentAgent:
	"""Runtime record for a managed intelligent agent."""

	id: str = field(default_factory=uuid7str)
	name: str = ""
	type: AgentType = AgentType.WORKER
	role: AgentRole = AgentRole.TASK_MANAGER
	description: str = ""
	capabilities: list[str] = field(default_factory=list)
	configuration: dict[str, Any] = field(default_factory=dict)
	created_by: str = ""
	tenant_id: str = "default"
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	status: str = "active"


@dataclass
class AgentNetwork:
	"""Group of agents coordinated as one executable network."""

	id: str = field(default_factory=uuid7str)
	name: str = ""
	topology: str = "mesh"
	configuration: dict[str, Any] = field(default_factory=dict)
	created_by: str = ""
	tenant_id: str = "default"
	agent_ids: list[str] = field(default_factory=list)
	created_at: datetime = field(default_factory=datetime.utcnow)
