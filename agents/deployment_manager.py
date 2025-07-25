#!/usr/bin/env python3
"""
Agent Deployment Manager
=======================

Production-ready deployment and orchestration system for APG autonomous agents.
"""

import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from uuid_extensions import uuid7str

from .base_agent import BaseAgent, AgentRole, AgentStatus
from .orchestrator import AgentOrchestrator
from .architect_agent import ArchitectAgent
from .developer_agent import DeveloperAgent
from .tester_agent import TesterAgent
from .devops_agent import DevOpsAgent

@dataclass
class AgentDeploymentConfig:
	"""Agent deployment configuration"""
	role: AgentRole
	instance_count: int = 1
	resource_limits: Dict[str, str] = field(default_factory=dict)
	environment_vars: Dict[str, str] = field(default_factory=dict)
	health_check_path: str = "/health"
	auto_scaling: bool = False
	min_instances: int = 1
	max_instances: int = 5

@dataclass
class DeploymentEnvironment:
	"""Deployment environment configuration"""
	name: str
	orchestrator_config: Dict[str, Any] = field(default_factory=dict)
	agent_configs: Dict[AgentRole, AgentDeploymentConfig] = field(default_factory=dict)
	networking: Dict[str, Any] = field(default_factory=dict)
	storage: Dict[str, Any] = field(default_factory=dict)
	monitoring: Dict[str, Any] = field(default_factory=dict)
	security: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentStatus:
	"""Deployment status tracking"""
	deployment_id: str = field(default_factory=uuid7str)
	environment: str = ""
	status: str = "pending"  # pending, deploying, running, failed, stopped
	agents_deployed: Dict[str, str] = field(default_factory=dict)
	health_status: Dict[str, bool] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_updated: datetime = field(default_factory=datetime.utcnow)
	error_message: Optional[str] = None

class AgentClusterManager:
	"""Manages clusters of agent instances"""
	
	def __init__(self, cluster_name: str, config: Dict[str, Any] = None):
		self.cluster_name = cluster_name
		self.config = config or {}
		self.agents: Dict[str, BaseAgent] = {}
		self.orchestrators: Dict[str, AgentOrchestrator] = {}
		self.load_balancer_config = {}
		self.logger = logging.getLogger(f"cluster.{cluster_name}")
	
	async def deploy_agent_cluster(
		self, 
		role: AgentRole, 
		instance_count: int,
		config: Dict[str, Any] = None
	) -> List[str]:
		"""Deploy a cluster of agents for a specific role"""
		self.logger.info(f"Deploying {instance_count} {role.value} agents")
		
		deployed_agents = []
		agent_class = self._get_agent_class(role)
		
		for i in range(instance_count):
			agent_id = f"{role.value}_{self.cluster_name}_{i:03d}"
			
			try:
				# Create agent instance
				agent = agent_class(agent_id, config=config)
				
				# Start agent
				await self._start_agent(agent)
				
				self.agents[agent_id] = agent
				deployed_agents.append(agent_id)
				
				self.logger.info(f"Deployed agent: {agent_id}")
				
			except Exception as e:
				self.logger.error(f"Failed to deploy agent {agent_id}: {e}")
		
		# Configure load balancing for multiple instances
		if len(deployed_agents) > 1:
			await self._configure_load_balancing(role, deployed_agents)
		
		return deployed_agents
	
	def _get_agent_class(self, role: AgentRole):
		"""Get agent class for role"""
		agent_classes = {
			AgentRole.ARCHITECT: ArchitectAgent,
			AgentRole.DEVELOPER: DeveloperAgent,
			AgentRole.TESTER: TesterAgent,
			AgentRole.DEVOPS: DevOpsAgent
		}
		return agent_classes.get(role, BaseAgent)
	
	async def _start_agent(self, agent: BaseAgent):
		"""Start an individual agent"""
		# Agent initialization is handled in __init__
		# Additional startup logic could go here
		self.logger.debug(f"Started agent: {agent.agent_id}")
	
	async def _configure_load_balancing(self, role: AgentRole, agent_ids: List[str]):
		"""Configure load balancing for multiple agent instances"""
		self.load_balancer_config[role] = {
			'strategy': 'round_robin',
			'agents': agent_ids,
			'health_check_interval': 30,
			'failover_enabled': True
		}
		
		self.logger.info(f"Configured load balancing for {role.value} with {len(agent_ids)} instances")
	
	async def scale_cluster(self, role: AgentRole, target_count: int) -> bool:
		"""Scale agent cluster up or down"""
		current_agents = [aid for aid in self.agents.keys() if role.value in aid]
		current_count = len(current_agents)
		
		if target_count == current_count:
			return True
		
		if target_count > current_count:
			# Scale up
			new_agents = await self.deploy_agent_cluster(
				role, 
				target_count - current_count,
				self.config.get('agent_configs', {}).get(role, {})
			)
			self.logger.info(f"Scaled up {role.value} from {current_count} to {target_count}")
			return len(new_agents) == (target_count - current_count)
		
		else:
			# Scale down
			agents_to_remove = current_agents[target_count:]
			for agent_id in agents_to_remove:
				await self._stop_agent(agent_id)
			self.logger.info(f"Scaled down {role.value} from {current_count} to {target_count}")
			return True
	
	async def _stop_agent(self, agent_id: str):
		"""Stop and remove an agent"""
		if agent_id in self.agents:
			agent = self.agents[agent_id]
			await agent.shutdown()
			del self.agents[agent_id]
			self.logger.info(f"Stopped agent: {agent_id}")
	
	async def get_cluster_health(self) -> Dict[str, Any]:
		"""Get health status of the cluster"""
		health_status = {}
		
		for agent_id, agent in self.agents.items():
			try:
				status_info = agent.get_status_info()
				health_status[agent_id] = {
					'status': status_info['status'],
					'healthy': status_info['status'] not in ['error', 'offline'],
					'queue_size': status_info.get('queue_size', 0),
					'current_task': status_info.get('current_task'),
					'experience_points': status_info.get('experience_points', 0)
				}
			except Exception as e:
				health_status[agent_id] = {
					'status': 'error',
					'healthy': False,
					'error': str(e)
				}
		
		# Calculate overall cluster health
		healthy_agents = sum(1 for status in health_status.values() if status.get('healthy', False))
		total_agents = len(health_status)
		
		return {
			'cluster_name': self.cluster_name,
			'total_agents': total_agents,
			'healthy_agents': healthy_agents,
			'health_percentage': (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
			'agent_status': health_status,
			'load_balancer_config': self.load_balancer_config
		}
	
	async def shutdown_cluster(self):
		"""Shutdown entire cluster"""
		self.logger.info(f"Shutting down cluster: {self.cluster_name}")
		
		for agent_id in list(self.agents.keys()):
			await self._stop_agent(agent_id)
		
		self.agents.clear()
		self.load_balancer_config.clear()

class AgentDeploymentManager:
	"""Main deployment manager for APG agent system"""
	
	def __init__(self, config_path: Optional[str] = None):
		self.config_path = config_path
		self.deployments: Dict[str, DeploymentStatus] = {}
		self.clusters: Dict[str, AgentClusterManager] = {}
		self.environments: Dict[str, DeploymentEnvironment] = {}
		
		# Load configuration
		self.config = self._load_config()
		
		# Initialize logging
		self.logger = logging.getLogger("deployment_manager")
		
		# Health monitoring
		self.health_check_interval = 60  # seconds
		self.auto_scaling_enabled = True
		
		# Load default environments
		self._setup_default_environments()
	
	def _load_config(self) -> Dict[str, Any]:
		"""Load deployment configuration"""
		if self.config_path and Path(self.config_path).exists():
			with open(self.config_path, 'r') as f:
				if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
					return yaml.safe_load(f)
				else:
					return json.load(f)
		
		# Default configuration
		return {
			'default_environment': 'development',
			'health_check_interval': 60,
			'auto_scaling': {
				'enabled': True,
				'scale_up_threshold': 80,
				'scale_down_threshold': 20,
				'cooldown_period': 300
			},
			'resource_limits': {
				'cpu': '1000m',
				'memory': '1Gi'
			},
			'monitoring': {
				'metrics_enabled': True,
				'logging_level': 'INFO'
			}
		}
	
	def _setup_default_environments(self):
		"""Setup default deployment environments"""
		# Development environment
		dev_env = DeploymentEnvironment(
			name='development',
			orchestrator_config={
				'max_concurrent_projects': 2,
				'task_timeout': 1800
			},
			agent_configs={
				AgentRole.ARCHITECT: AgentDeploymentConfig(
					role=AgentRole.ARCHITECT,
					instance_count=1,
					resource_limits={'cpu': '500m', 'memory': '512Mi'}
				),
				AgentRole.DEVELOPER: AgentDeploymentConfig(
					role=AgentRole.DEVELOPER,
					instance_count=1,
					resource_limits={'cpu': '1000m', 'memory': '1Gi'}
				),
				AgentRole.TESTER: AgentDeploymentConfig(
					role=AgentRole.TESTER,
					instance_count=1,
					resource_limits={'cpu': '500m', 'memory': '512Mi'}
				),
				AgentRole.DEVOPS: AgentDeploymentConfig(
					role=AgentRole.DEVOPS,
					instance_count=1,
					resource_limits={'cpu': '500m', 'memory': '512Mi'}
				)
			},
			monitoring={'enabled': True, 'level': 'basic'}
		)
		
		# Production environment
		prod_env = DeploymentEnvironment(
			name='production',
			orchestrator_config={
				'max_concurrent_projects': 10,
				'task_timeout': 3600
			},
			agent_configs={
				AgentRole.ARCHITECT: AgentDeploymentConfig(
					role=AgentRole.ARCHITECT,
					instance_count=2,
					resource_limits={'cpu': '1000m', 'memory': '1Gi'},
					auto_scaling=True,
					max_instances=4
				),
				AgentRole.DEVELOPER: AgentDeploymentConfig(
					role=AgentRole.DEVELOPER,
					instance_count=3,
					resource_limits={'cpu': '2000m', 'memory': '2Gi'},
					auto_scaling=True,
					max_instances=8
				),
				AgentRole.TESTER: AgentDeploymentConfig(
					role=AgentRole.TESTER,
					instance_count=2,
					resource_limits={'cpu': '1000m', 'memory': '1Gi'},
					auto_scaling=True,
					max_instances=5
				),
				AgentRole.DEVOPS: AgentDeploymentConfig(
					role=AgentRole.DEVOPS,
					instance_count=2,
					resource_limits={'cpu': '1000m', 'memory': '1Gi'},
					auto_scaling=True,
					max_instances=4
				)
			},
			monitoring={'enabled': True, 'level': 'comprehensive'},
			security={'ssl_enabled': True, 'authentication_required': True}
		)
		
		self.environments['development'] = dev_env
		self.environments['production'] = prod_env
	
	async def deploy_environment(self, environment_name: str) -> str:
		"""Deploy a complete agent environment"""
		if environment_name not in self.environments:
			raise ValueError(f"Environment '{environment_name}' not configured")
		
		environment = self.environments[environment_name]
		deployment_id = uuid7str()
		
		self.logger.info(f"Starting deployment of environment: {environment_name}")
		
		# Create deployment status
		deployment = DeploymentStatus(
			deployment_id=deployment_id,
			environment=environment_name,
			status="deploying"
		)
		self.deployments[deployment_id] = deployment
		
		try:
			# Create cluster for this environment
			cluster_name = f"{environment_name}_{deployment_id[:8]}"
			cluster = AgentClusterManager(cluster_name, self.config)
			self.clusters[cluster_name] = cluster
			
			# Deploy agents for each role
			for role, agent_config in environment.agent_configs.items():
				self.logger.info(f"Deploying {agent_config.instance_count} {role.value} agents")
				
				agent_ids = await cluster.deploy_agent_cluster(
					role,
					agent_config.instance_count,
					{
						'resource_limits': agent_config.resource_limits,
						'environment_vars': agent_config.environment_vars,
						'learning': {'enabled': True}
					}
				)
				
				deployment.agents_deployed[role.value] = agent_ids
			
			# Create and configure orchestrator
			orchestrator = AgentOrchestrator(environment.orchestrator_config)
			
			# Register all agents with orchestrator
			for role_agents in deployment.agents_deployed.values():
				for agent_id in role_agents:
					agent = cluster.agents[agent_id]
					await orchestrator.register_agent(agent)
			
			cluster.orchestrators['main'] = orchestrator
			
			# Start health monitoring
			asyncio.create_task(self._monitor_deployment_health(deployment_id))
			
			# Start auto-scaling if enabled
			if self.auto_scaling_enabled:
				asyncio.create_task(self._auto_scale_deployment(deployment_id))
			
			deployment.status = "running"
			deployment.last_updated = datetime.utcnow()
			
			self.logger.info(f"Successfully deployed environment: {environment_name}")
			
			return deployment_id
			
		except Exception as e:
			deployment.status = "failed"
			deployment.error_message = str(e)
			deployment.last_updated = datetime.utcnow()
			
			self.logger.error(f"Deployment failed: {e}")
			raise
	
	async def _monitor_deployment_health(self, deployment_id: str):
		"""Monitor deployment health continuously"""
		while deployment_id in self.deployments:
			deployment = self.deployments[deployment_id]
			
			if deployment.status != "running":
				break
			
			try:
				# Get cluster for this deployment
				cluster_name = None
				for name, cluster in self.clusters.items():
					if deployment_id[:8] in name:
						cluster_name = name
						break
				
				if cluster_name:
					cluster = self.clusters[cluster_name]
					health = await cluster.get_cluster_health()
					
					# Update deployment health status
					for agent_id, agent_health in health['agent_status'].items():
						deployment.health_status[agent_id] = agent_health['healthy']
					
					# Check for unhealthy agents
					unhealthy_agents = [
						aid for aid, healthy in deployment.health_status.items() 
						if not healthy
					]
					
					if unhealthy_agents:
						self.logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
						# Could implement auto-healing here
					
					deployment.last_updated = datetime.utcnow()
			
			except Exception as e:
				self.logger.error(f"Health monitoring error for {deployment_id}: {e}")
			
			await asyncio.sleep(self.health_check_interval)
	
	async def _auto_scale_deployment(self, deployment_id: str):
		"""Auto-scale deployment based on load"""
		while deployment_id in self.deployments:
			deployment = self.deployments[deployment_id]
			
			if deployment.status != "running":
				break
			
			try:
				auto_config = self.config.get('auto_scaling', {})
				if not auto_config.get('enabled', True):
					await asyncio.sleep(300)  # Check every 5 minutes
					continue
				
				# Get cluster metrics
				cluster_name = None
				for name, cluster in self.clusters.items():
					if deployment_id[:8] in name:
						cluster_name = name
						break
				
				if cluster_name:
					cluster = self.clusters[cluster_name]
					health = await cluster.get_cluster_health()
					
					# Calculate average load
					total_queue_size = 0
					agent_count = 0
					
					for agent_status in health['agent_status'].values():
						if agent_status.get('healthy', False):
							total_queue_size += agent_status.get('queue_size', 0)
							agent_count += 1
					
					if agent_count > 0:
						avg_queue_size = total_queue_size / agent_count
						
						# Scale up if average queue size is high
						scale_up_threshold = auto_config.get('scale_up_threshold', 5)
						scale_down_threshold = auto_config.get('scale_down_threshold', 1)
						
						environment = self.environments[deployment.environment]
						
						for role, agent_config in environment.agent_configs.items():
							if agent_config.auto_scaling:
								current_count = len(deployment.agents_deployed.get(role.value, []))
								
								if avg_queue_size > scale_up_threshold and current_count < agent_config.max_instances:
									target_count = min(current_count + 1, agent_config.max_instances)
									await cluster.scale_cluster(role, target_count)
									self.logger.info(f"Scaled up {role.value} to {target_count} instances")
								
								elif avg_queue_size < scale_down_threshold and current_count > agent_config.min_instances:
									target_count = max(current_count - 1, agent_config.min_instances)
									await cluster.scale_cluster(role, target_count)
									self.logger.info(f"Scaled down {role.value} to {target_count} instances")
			
			except Exception as e:
				self.logger.error(f"Auto-scaling error for {deployment_id}: {e}")
			
			cooldown = self.config.get('auto_scaling', {}).get('cooldown_period', 300)
			await asyncio.sleep(cooldown)
	
	async def stop_deployment(self, deployment_id: str) -> bool:
		"""Stop a running deployment"""
		if deployment_id not in self.deployments:
			return False
		
		deployment = self.deployments[deployment_id]
		deployment.status = "stopping"
		
		try:
			# Find and shutdown cluster
			cluster_name = None
			for name, cluster in self.clusters.items():
				if deployment_id[:8] in name:
					cluster_name = name
					break
			
			if cluster_name:
				cluster = self.clusters[cluster_name]
				await cluster.shutdown_cluster()
				del self.clusters[cluster_name]
			
			deployment.status = "stopped"
			deployment.last_updated = datetime.utcnow()
			
			self.logger.info(f"Stopped deployment: {deployment_id}")
			return True
			
		except Exception as e:
			deployment.status = "failed"
			deployment.error_message = str(e)
			self.logger.error(f"Failed to stop deployment {deployment_id}: {e}")
			return False
	
	def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
		"""Get status of a specific deployment"""
		return self.deployments.get(deployment_id)
	
	def list_deployments(self) -> List[DeploymentStatus]:
		"""List all deployments"""
		return list(self.deployments.values())
	
	async def get_system_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive system metrics"""
		metrics = {
			'deployments': {
				'total': len(self.deployments),
				'running': len([d for d in self.deployments.values() if d.status == 'running']),
				'failed': len([d for d in self.deployments.values() if d.status == 'failed'])
			},
			'clusters': {
				'total': len(self.clusters),
				'health': {}
			},
			'agents': {
				'total': 0,
				'by_role': {},
				'healthy': 0
			}
		}
		
		# Collect cluster metrics
		for cluster_name, cluster in self.clusters.items():
			health = await cluster.get_cluster_health()
			metrics['clusters']['health'][cluster_name] = health
			
			metrics['agents']['total'] += health['total_agents']
			metrics['agents']['healthy'] += health['healthy_agents']
			
			# Count by role
			for agent_id in health['agent_status'].keys():
				for role in AgentRole:
					if role.value in agent_id:
						metrics['agents']['by_role'][role.value] = metrics['agents']['by_role'].get(role.value, 0) + 1
		
		return metrics
	
	async def create_deployment_report(self, deployment_id: str) -> Dict[str, Any]:
		"""Generate comprehensive deployment report"""
		if deployment_id not in self.deployments:
			return {'error': 'Deployment not found'}
		
		deployment = self.deployments[deployment_id]
		
		# Get cluster health
		cluster_health = {}
		cluster_name = None
		for name, cluster in self.clusters.items():
			if deployment_id[:8] in name:
				cluster_name = name
				cluster_health = await cluster.get_cluster_health()
				break
		
		# Get orchestrator status
		orchestrator_status = {}
		if cluster_name and cluster_name in self.clusters:
			cluster = self.clusters[cluster_name]
			if 'main' in cluster.orchestrators:
				orchestrator_status = cluster.orchestrators['main'].get_system_status()
		
		report = {
			'deployment_info': {
				'id': deployment.deployment_id,
				'environment': deployment.environment,
				'status': deployment.status,
				'created_at': deployment.created_at.isoformat(),
				'last_updated': deployment.last_updated.isoformat(),
				'uptime': str(datetime.utcnow() - deployment.created_at)
			},
			'cluster_health': cluster_health,
			'orchestrator_status': orchestrator_status,
			'agent_deployment': deployment.agents_deployed,
			'health_status': deployment.health_status,
			'recommendations': await self._generate_recommendations(deployment_id)
		}
		
		return report
	
	async def _generate_recommendations(self, deployment_id: str) -> List[str]:
		"""Generate recommendations for deployment optimization"""
		recommendations = []
		
		deployment = self.deployments[deployment_id]
		
		# Check health status
		unhealthy_count = sum(1 for healthy in deployment.health_status.values() if not healthy)
		total_count = len(deployment.health_status)
		
		if unhealthy_count > 0:
			health_percentage = (total_count - unhealthy_count) / total_count * 100
			if health_percentage < 80:
				recommendations.append("Consider scaling up or investigating unhealthy agents")
		
		# Check resource utilization
		if total_count < 2:
			recommendations.append("Consider adding redundancy with multiple agent instances")
		
		# Environment-specific recommendations
		environment = self.environments.get(deployment.environment)
		if environment:
			for role, config in environment.agent_configs.items():
				if config.auto_scaling and config.instance_count == config.min_instances:
					recommendations.append(f"Consider increasing min instances for {role.value} agents")
		
		if not recommendations:
			recommendations.append("Deployment is healthy and well-configured")
		
		return recommendations
	
	def export_deployment_config(self, deployment_id: str, format: str = 'yaml') -> str:
		"""Export deployment configuration"""
		if deployment_id not in self.deployments:
			raise ValueError(f"Deployment {deployment_id} not found")
		
		deployment = self.deployments[deployment_id]
		environment = self.environments[deployment.environment]
		
		config = {
			'deployment': {
				'id': deployment.deployment_id,
				'environment': deployment.environment,
				'created_at': deployment.created_at.isoformat()
			},
			'environment': {
				'name': environment.name,
				'orchestrator_config': environment.orchestrator_config,
				'agent_configs': {
					role.value: {
						'instance_count': config.instance_count,
						'resource_limits': config.resource_limits,
						'auto_scaling': config.auto_scaling,
						'min_instances': config.min_instances,
						'max_instances': config.max_instances
					}
					for role, config in environment.agent_configs.items()
				},
				'monitoring': environment.monitoring,
				'security': environment.security
			}
		}
		
		if format.lower() == 'yaml':
			return yaml.dump(config, default_flow_style=False)
		else:
			return json.dumps(config, indent=2)

# Factory function for easy deployment
async def deploy_apg_agents(
	environment: str = 'development',
	config_path: Optional[str] = None
) -> Tuple[str, AgentDeploymentManager]:
	"""Factory function to easily deploy APG agents"""
	manager = AgentDeploymentManager(config_path)
	deployment_id = await manager.deploy_environment(environment)
	return deployment_id, manager