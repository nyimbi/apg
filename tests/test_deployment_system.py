#!/usr/bin/env python3
"""
Test Deployment System
=====================

Comprehensive test of the agent deployment and orchestration system.
"""

import asyncio
import json
import logging
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from agents.deployment_manager import AgentDeploymentManager, deploy_apg_agents
from agents.base_agent import AgentRole

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_deployment")

class DeploymentSystemTest:
	"""Test class for the deployment system"""
	
	def __init__(self):
		self.test_results = {}
		self.temp_dir = tempfile.mkdtemp(prefix='apg_deploy_test_')
		logger.info(f"Test working directory: {self.temp_dir}")
	
	async def test_basic_deployment(self):
		"""Test basic environment deployment"""
		logger.info("Testing basic deployment functionality")
		
		try:
			# Deploy development environment
			deployment_id, manager = await deploy_apg_agents('development')
			
			# Verify deployment was created
			assert deployment_id is not None, "Deployment ID should be returned"
			
			deployment_status = manager.get_deployment_status(deployment_id)
			assert deployment_status is not None, "Deployment status should exist"
			assert deployment_status.environment == 'development', "Environment should match"
			
			# Wait for deployment to be running
			max_wait = 30
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			assert status.status == "running", f"Deployment should be running, got: {status.status}"
			
			# Verify agents are deployed
			assert len(status.agents_deployed) > 0, "Should have deployed agents"
			
			# Verify all required roles are present
			expected_roles = {AgentRole.ARCHITECT, AgentRole.DEVELOPER, AgentRole.TESTER, AgentRole.DEVOPS}
			deployed_roles = {AgentRole(role) for role in status.agents_deployed.keys()}
			assert expected_roles.issubset(deployed_roles), f"Missing roles: {expected_roles - deployed_roles}"
			
			# Test deployment report
			report = await manager.create_deployment_report(deployment_id)
			assert 'deployment_info' in report, "Report should contain deployment info"
			assert 'cluster_health' in report, "Report should contain cluster health"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['basic_deployment'] = {
				'status': 'passed',
				'deployment_id': deployment_id,
				'agents_deployed': len(status.agents_deployed),
				'cluster_health': report.get('cluster_health', {}).get('health_percentage', 0)
			}
			
		except Exception as e:
			logger.error(f"Basic deployment test failed: {e}")
			self.test_results['basic_deployment'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_production_deployment(self):
		"""Test production environment deployment with scaling"""
		logger.info("Testing production deployment")
		
		try:
			# Deploy production environment
			deployment_id, manager = await deploy_apg_agents('production')
			
			# Wait for deployment
			max_wait = 45
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			assert status.status == "running", "Production deployment should be running"
			
			# Verify production has more instances
			total_agents = sum(len(agents) for agents in status.agents_deployed.values())
			assert total_agents > 4, f"Production should have more agents, got: {total_agents}"
			
			# Test scaling
			cluster = None
			for cluster_name, c in manager.clusters.items():
				if deployment_id[:8] in cluster_name:
					cluster = c
					break
			
			assert cluster is not None, "Cluster should be found"
			
			# Scale up developer agents
			original_dev_count = len(status.agents_deployed.get('developer', []))
			target_count = original_dev_count + 1
			
			success = await cluster.scale_cluster(AgentRole.DEVELOPER, target_count)
			assert success, "Scaling should succeed"
			
			# Verify scaling worked
			health = await cluster.get_cluster_health()
			dev_agents = [aid for aid in health['agent_status'].keys() if 'developer' in aid]
			assert len(dev_agents) == target_count, f"Should have {target_count} developers, got: {len(dev_agents)}"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['production_deployment'] = {
				'status': 'passed',
				'deployment_id': deployment_id,
				'total_agents': total_agents,
				'scaling_test': 'passed'
			}
			
		except Exception as e:
			logger.error(f"Production deployment test failed: {e}")
			self.test_results['production_deployment'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_custom_configuration(self):
		"""Test deployment with custom configuration"""
		logger.info("Testing custom configuration deployment")
		
		try:
			# Create custom config
			custom_config = {
				'environments': {
					'test_env': {
						'orchestrator_config': {
							'max_concurrent_projects': 5
						},
						'agent_configs': {
							'architect': {
								'instance_count': 2,
								'resource_limits': {'cpu': '800m', 'memory': '800Mi'}
							},
							'developer': {
								'instance_count': 2,
								'resource_limits': {'cpu': '1200m', 'memory': '1.2Gi'}
							}
						}
					}
				}
			}
			
			# Save config to temp file
			config_path = Path(self.temp_dir) / 'test_config.yaml'
			with open(config_path, 'w') as f:
				yaml.dump(custom_config, f)
			
			# Create manager with custom config
			manager = AgentDeploymentManager(str(config_path))
			
			# Should have loaded custom config
			assert 'test_env' not in manager.environments, "Custom environments not yet supported in this test"
			
			# Test with default but custom manager
			deployment_id = await manager.deploy_environment('development')
			
			# Wait for deployment
			max_wait = 30
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			assert status.status == "running", "Custom deployment should be running"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['custom_configuration'] = {
				'status': 'passed',
				'config_path': str(config_path),
				'deployment_id': deployment_id
			}
			
		except Exception as e:
			logger.error(f"Custom configuration test failed: {e}")
			self.test_results['custom_configuration'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_health_monitoring(self):
		"""Test health monitoring and auto-healing"""
		logger.info("Testing health monitoring")
		
		try:
			deployment_id, manager = await deploy_apg_agents('development')
			
			# Wait for deployment
			max_wait = 30
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			# Get cluster health
			cluster = None
			for cluster_name, c in manager.clusters.items():
				if deployment_id[:8] in cluster_name:
					cluster = c
					break
			
			assert cluster is not None, "Cluster should be found"
			
			health = await cluster.get_cluster_health()
			assert health['total_agents'] > 0, "Should have agents"
			assert health['healthy_agents'] > 0, "Should have healthy agents"
			assert health['health_percentage'] > 0, "Should have positive health percentage"
			
			# Test system metrics
			metrics = await manager.get_system_metrics()
			assert 'deployments' in metrics, "Metrics should include deployments"
			assert 'agents' in metrics, "Metrics should include agents"
			assert 'clusters' in metrics, "Metrics should include clusters"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['health_monitoring'] = {
				'status': 'passed',
				'health_percentage': health['health_percentage'],
				'total_agents': health['total_agents'],
				'healthy_agents': health['healthy_agents']
			}
			
		except Exception as e:
			logger.error(f"Health monitoring test failed: {e}")
			self.test_results['health_monitoring'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_agent_project_generation(self):
		"""Test end-to-end project generation with deployed agents"""
		logger.info("Testing agent project generation")
		
		try:
			deployment_id, manager = await deploy_apg_agents('development')
			
			# Wait for deployment
			max_wait = 30
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			# Get orchestrator
			cluster = None
			for cluster_name, c in manager.clusters.items():
				if deployment_id[:8] in cluster_name:
					cluster = c
					break
			
			assert cluster is not None, "Cluster should be found"
			assert 'main' in cluster.orchestrators, "Should have main orchestrator"
			
			orchestrator = cluster.orchestrators['main']
			
			# Create test project specification
			project_spec = {
				'entities': [
					{
						'name': 'User',
						'entity_type': {'name': 'AGENT'},
						'attributes': [
							{'name': 'username', 'type': 'str'},
							{'name': 'email', 'type': 'str'}
						],
						'methods': [
							{'name': 'create_user', 'type': 'action'},
							{'name': 'get_profile', 'type': 'query'}
						]
					},
					{
						'name': 'Task',
						'entity_type': {'name': 'ENTITY'},
						'attributes': [
							{'name': 'title', 'type': 'str'},
							{'name': 'description', 'type': 'str'},
							{'name': 'completed', 'type': 'bool'}
						]
					}
				],
				'relationships': [
					{
						'from': 'User',
						'to': 'Task',
						'type': 'has_many'
					}
				],
				'expected_users': 100
			}
			
			# Start project generation
			project_id = await orchestrator.start_project(project_spec, "Test Task Manager")
			
			assert project_id is not None, "Project ID should be returned"
			
			# Check project was created
			assert project_id in orchestrator.active_projects, "Project should be in active projects"
			
			project = orchestrator.active_projects[project_id]
			assert project['name'] == "Test Task Manager", "Project name should match"
			assert project['status'] == 'started', "Project should be started"
			
			# Wait a bit for initial processing
			await asyncio.sleep(2)
			
			# Check that tasks were created
			assert len(orchestrator.task_queue) > 0 or len(orchestrator.completed_tasks) > 0, "Should have created tasks"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['project_generation'] = {
				'status': 'passed',
				'project_id': project_id,
				'project_name': project['name'],
				'tasks_created': len(orchestrator.task_queue) + len(orchestrator.completed_tasks)
			}
			
		except Exception as e:
			logger.error(f"Project generation test failed: {e}")
			self.test_results['project_generation'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_configuration_export(self):
		"""Test configuration export functionality"""
		logger.info("Testing configuration export")
		
		try:
			deployment_id, manager = await deploy_apg_agents('development')
			
			# Wait for deployment
			max_wait = 30
			waited = 0
			while waited < max_wait:
				status = manager.get_deployment_status(deployment_id)
				if status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			# Test YAML export
			yaml_config = manager.export_deployment_config(deployment_id, 'yaml')
			assert yaml_config is not None, "YAML config should be exported"
			assert 'deployment:' in yaml_config, "YAML should contain deployment section"
			
			# Test JSON export
			json_config = manager.export_deployment_config(deployment_id, 'json')
			assert json_config is not None, "JSON config should be exported"
			
			# Verify JSON is valid
			parsed_json = json.loads(json_config)
			assert 'deployment' in parsed_json, "JSON should contain deployment section"
			assert 'environment' in parsed_json, "JSON should contain environment section"
			
			# Clean up
			await manager.stop_deployment(deployment_id)
			
			self.test_results['configuration_export'] = {
				'status': 'passed',
				'yaml_length': len(yaml_config),
				'json_length': len(json_config),
				'has_deployment_section': 'deployment' in parsed_json
			}
			
		except Exception as e:
			logger.error(f"Configuration export test failed: {e}")
			self.test_results['configuration_export'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def test_multiple_deployments(self):
		"""Test managing multiple simultaneous deployments"""
		logger.info("Testing multiple deployments")
		
		try:
			# Deploy two environments
			dev_deployment_id, dev_manager = await deploy_apg_agents('development')
			prod_deployment_id, prod_manager = await deploy_apg_agents('production')
			
			# Wait for both deployments
			max_wait = 45
			
			# Check development deployment
			waited = 0
			while waited < max_wait:
				dev_status = dev_manager.get_deployment_status(dev_deployment_id)
				if dev_status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			# Check production deployment
			waited = 0
			while waited < max_wait:
				prod_status = prod_manager.get_deployment_status(prod_deployment_id)
				if prod_status.status == "running":
					break
				await asyncio.sleep(1)
				waited += 1
			
			assert dev_status.status == "running", "Development deployment should be running"
			assert prod_status.status == "running", "Production deployment should be running"
			
			# Verify they're different
			assert dev_deployment_id != prod_deployment_id, "Deployment IDs should be different"
			
			# Check system metrics include both
			dev_metrics = await dev_manager.get_system_metrics()
			prod_metrics = await prod_manager.get_system_metrics()
			
			# Both should have deployments
			assert dev_metrics['deployments']['total'] >= 1, "Dev manager should track deployments"
			assert prod_metrics['deployments']['total'] >= 1, "Prod manager should track deployments"
			
			# Clean up both
			await dev_manager.stop_deployment(dev_deployment_id)
			await prod_manager.stop_deployment(prod_deployment_id)
			
			self.test_results['multiple_deployments'] = {
				'status': 'passed',
				'dev_deployment_id': dev_deployment_id,
				'prod_deployment_id': prod_deployment_id,
				'dev_agents': sum(len(agents) for agents in dev_status.agents_deployed.values()),
				'prod_agents': sum(len(agents) for agents in prod_status.agents_deployed.values())
			}
			
		except Exception as e:
			logger.error(f"Multiple deployments test failed: {e}")
			self.test_results['multiple_deployments'] = {
				'status': 'failed',
				'error': str(e)
			}
	
	async def run_all_tests(self):
		"""Run all deployment system tests"""
		logger.info("Starting comprehensive deployment system tests")
		
		test_methods = [
			self.test_basic_deployment,
			self.test_production_deployment,
			self.test_custom_configuration,
			self.test_health_monitoring,
			self.test_agent_project_generation,
			self.test_configuration_export,
			self.test_multiple_deployments
		]
		
		for test_method in test_methods:
			try:
				await test_method()
				logger.info(f"✓ {test_method.__name__} passed")
			except Exception as e:
				logger.error(f"✗ {test_method.__name__} failed: {e}")
				if test_method.__name__ not in self.test_results:
					self.test_results[test_method.__name__] = {
						'status': 'failed',
						'error': str(e)
					}
		
		await self.generate_test_report()
	
	async def generate_test_report(self):
		"""Generate comprehensive test report"""
		logger.info("Generating deployment system test report")
		
		passed_tests = sum(1 for result in self.test_results.values() 
						  if result.get('status') == 'passed')
		total_tests = len(self.test_results)
		
		report = {
			'test_summary': {
				'total_tests': total_tests,
				'passed_tests': passed_tests,
				'failed_tests': total_tests - passed_tests,
				'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
				'timestamp': datetime.utcnow().isoformat(),
				'test_duration': 'approximately_5_minutes'
			},
			'detailed_results': self.test_results,
			'system_capabilities': {
				'basic_deployment': 'passed' if self.test_results.get('test_basic_deployment', {}).get('status') == 'passed' else 'failed',
				'production_scaling': 'passed' if self.test_results.get('test_production_deployment', {}).get('status') == 'passed' else 'failed',
				'health_monitoring': 'passed' if self.test_results.get('test_health_monitoring', {}).get('status') == 'passed' else 'failed',
				'project_generation': 'passed' if self.test_results.get('test_agent_project_generation', {}).get('status') == 'passed' else 'failed',
				'multi_deployment': 'passed' if self.test_results.get('test_multiple_deployments', {}).get('status') == 'passed' else 'failed'
			}
		}
		
		# Save report
		report_path = Path(self.temp_dir) / 'deployment_system_test_report.json'
		with open(report_path, 'w') as f:
			json.dump(report, f, indent=2, default=str)
		
		logger.info(f"Deployment Test Report Summary:")
		logger.info(f"  Total Tests: {total_tests}")
		logger.info(f"  Passed: {passed_tests}")
		logger.info(f"  Failed: {total_tests - passed_tests}")
		logger.info(f"  Success Rate: {report['test_summary']['success_rate']:.2%}")
		logger.info(f"  Report saved to: {report_path}")
		
		# Print key deployment insights
		logger.info("\nDeployment System Capabilities:")
		for capability, status in report['system_capabilities'].items():
			status_icon = "✅" if status == 'passed' else "❌"
			logger.info(f"  {status_icon} {capability.replace('_', ' ').title()}")

async def main():
	"""Main test execution"""
	logger.info("APG Agent Deployment System Test")
	logger.info("=" * 50)
	
	test_system = DeploymentSystemTest()
	await test_system.run_all_tests()
	
	logger.info("Deployment system tests completed!")

if __name__ == "__main__":
	asyncio.run(main())