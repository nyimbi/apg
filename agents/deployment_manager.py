"""Deployment manager for APG agent clusters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from uuid_extensions import uuid7str

from .architect_agent import ArchitectAgent
from .base_agent import AgentRole
from .developer_agent import DeveloperAgent
from .devops_agent import DevOpsAgent
from .orchestrator import AgentOrchestrator
from .tester_agent import TesterAgent


@dataclass
class DeploymentStatus:
	deployment_id: str
	environment: str
	status: str = "provisioning"
	agents_deployed: dict[str, list[str]] = field(default_factory=dict)


class AgentCluster:
	"""A local APG agent cluster used by deployments."""

	def __init__(self, deployment_id: str, environment: str):
		self.deployment_id = deployment_id
		self.environment = environment
		self.orchestrators = {"main": AgentOrchestrator()}
		self.agents: dict[str, Any] = {}

	async def initialize(self, role_counts: dict[AgentRole, int]) -> dict[str, list[str]]:
		deployed: dict[str, list[str]] = {}
		for role, count in role_counts.items():
			deployed[role.value] = []
			for index in range(count):
				agent = self._make_agent(role, f"{role.value}_{index}_{uuid7str()[:8]}")
				self.agents[agent.agent_id] = agent
				deployed[role.value].append(agent.agent_id)
				await self.orchestrators["main"].register_agent(agent)
		return deployed

	def _make_agent(self, role: AgentRole, agent_id: str):
		config = {"learning": {"enabled": True}}
		if role == AgentRole.ARCHITECT:
			return ArchitectAgent(agent_id, config=config)
		if role == AgentRole.TESTER:
			return TesterAgent(agent_id, config=config)
		if role == AgentRole.DEVOPS:
			return DevOpsAgent(agent_id, config=config)
		return DeveloperAgent(agent_id, config=config)

	async def scale_cluster(self, role: AgentRole, target_count: int) -> bool:
		current = [agent for agent in self.agents.values() if agent.role == role]
		while len(current) < target_count:
			agent = self._make_agent(role, f"{role.value}_{len(current)}_{uuid7str()[:8]}")
			self.agents[agent.agent_id] = agent
			await self.orchestrators["main"].register_agent(agent)
			current.append(agent)
		return True

	async def get_cluster_health(self) -> dict[str, Any]:
		agent_status = {
			agent_id: {"status": "healthy", "role": agent.role.value}
			for agent_id, agent in self.agents.items()
		}
		total = len(agent_status)
		return {
			"total_agents": total,
			"healthy_agents": total,
			"health_percentage": 100.0 if total else 0.0,
			"agent_status": agent_status,
		}


class AgentDeploymentManager:
	"""Manage local APG agent deployments."""

	def __init__(self, config_path: str | None = None):
		self.config_path = config_path
		self.deployments: dict[str, DeploymentStatus] = {}
		self.clusters: dict[str, AgentCluster] = {}
		self.environments = {
			"development": {
				AgentRole.ARCHITECT: 1,
				AgentRole.DEVELOPER: 1,
				AgentRole.TESTER: 1,
				AgentRole.DEVOPS: 1,
			},
			"production": {
				AgentRole.ARCHITECT: 2,
				AgentRole.DEVELOPER: 3,
				AgentRole.TESTER: 2,
				AgentRole.DEVOPS: 2,
			},
		}

	async def deploy_environment(self, environment: str = "development") -> str:
		deployment_id = uuid7str()
		role_counts = self.environments.get(environment, self.environments["development"])
		cluster = AgentCluster(deployment_id, environment)
		deployed = await cluster.initialize(role_counts)
		status = DeploymentStatus(
			deployment_id=deployment_id,
			environment=environment,
			status="running",
			agents_deployed=deployed,
		)
		self.deployments[deployment_id] = status
		self.clusters[f"{environment}_{deployment_id[:8]}"] = cluster
		return deployment_id

	def get_deployment_status(self, deployment_id: str) -> DeploymentStatus | None:
		return self.deployments.get(deployment_id)

	async def create_deployment_report(self, deployment_id: str) -> dict[str, Any]:
		status = self.deployments[deployment_id]
		cluster = self._cluster_for(deployment_id)
		health = await cluster.get_cluster_health() if cluster else {}
		return {
			"deployment_info": {
				"deployment_id": deployment_id,
				"environment": status.environment,
				"status": status.status,
			},
			"cluster_health": health,
		}

	async def stop_deployment(self, deployment_id: str) -> bool:
		status = self.deployments.get(deployment_id)
		if status:
			status.status = "stopped"
		return True

	async def get_system_metrics(self) -> dict[str, Any]:
		total_agents = sum(
			sum(len(agent_ids) for agent_ids in status.agents_deployed.values())
			for status in self.deployments.values()
		)
		return {
			"deployments": {"total": len(self.deployments)},
			"agents": {"total": total_agents},
			"clusters": {"total": len(self.clusters)},
		}

	def export_deployment_config(self, deployment_id: str, format: str = "yaml") -> str:
		status = self.deployments[deployment_id]
		payload = {
			"deployment": {
				"id": deployment_id,
				"status": status.status,
				"agents": status.agents_deployed,
			},
			"environment": status.environment,
		}
		if format == "json":
			return json.dumps(payload, indent=2)
		return yaml.safe_dump(payload, sort_keys=False)

	def _cluster_for(self, deployment_id: str) -> AgentCluster | None:
		for name, cluster in self.clusters.items():
			if deployment_id[:8] in name:
				return cluster
		return None


async def deploy_apg_agents(environment: str = "development") -> tuple[str, AgentDeploymentManager]:
	"""Deploy APG agents into a local in-memory cluster."""

	manager = AgentDeploymentManager()
	deployment_id = await manager.deploy_environment(environment)
	return deployment_id, manager
