#!/usr/bin/env python3
"""
APG Agents CLI
==============

Command-line interface for managing APG autonomous agent deployments.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional
import click
from tabulate import tabulate

from .deployment_manager import AgentDeploymentManager, deploy_apg_agents
from .base_agent import AgentRole

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("apg_agents_cli")

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
	"""APG Agents - Autonomous Code Generation Agent System"""
	if verbose:
		logging.getLogger().setLevel(logging.DEBUG)
	
	ctx.ensure_object(dict)
	ctx.obj['config'] = config

@cli.command()
@click.option('--environment', '-e', default='development', 
			  help='Environment to deploy (development/production)')
@click.option('--wait', '-w', is_flag=True, help='Wait for deployment to complete')
@click.pass_context
def deploy(ctx, environment, wait):
	"""Deploy APG agent environment"""
	click.echo(f"🚀 Deploying APG agents to {environment} environment...")
	
	async def _deploy():
		try:
			deployment_id, manager = await deploy_apg_agents(environment, ctx.obj['config'])
			
			click.echo(f"✅ Deployment started successfully!")
			click.echo(f"   Deployment ID: {deployment_id}")
			click.echo(f"   Environment: {environment}")
			
			if wait:
				click.echo("⏳ Waiting for deployment to be ready...")
				
				# Wait for deployment to be running
				max_wait = 300  # 5 minutes
				waited = 0
				while waited < max_wait:
					status = manager.get_deployment_status(deployment_id)
					if status and status.status == "running":
						click.echo("✅ Deployment is running!")
						break
					elif status and status.status == "failed":
						click.echo(f"❌ Deployment failed: {status.error_message}")
						sys.exit(1)
					
					await asyncio.sleep(5)
					waited += 5
					
					if waited % 30 == 0:  # Progress update every 30 seconds
						click.echo(f"   Still waiting... ({waited}s)")
				
				if waited >= max_wait:
					click.echo("⚠️  Deployment timeout - check status manually")
			
			# Show deployment info
			await _show_deployment_info(manager, deployment_id)
			
			return deployment_id
			
		except Exception as e:
			click.echo(f"❌ Deployment failed: {e}")
			sys.exit(1)
	
	return asyncio.run(_deploy())

@cli.command()
@click.option('--deployment-id', '-d', help='Specific deployment ID')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), 
			  default='table', help='Output format')
@click.pass_context
def status(ctx, deployment_id, format):
	"""Show deployment status"""
	
	async def _status():
		manager = AgentDeploymentManager(ctx.obj['config'])
		
		if deployment_id:
			# Show specific deployment
			status = manager.get_deployment_status(deployment_id)
			if not status:
				click.echo(f"❌ Deployment {deployment_id} not found")
				sys.exit(1)
			
			await _show_deployment_info(manager, deployment_id, format)
		else:
			# Show all deployments
			deployments = manager.list_deployments()
			
			if not deployments:
				click.echo("No deployments found")
				return
			
			if format == 'json':
				click.echo(json.dumps([d.__dict__ for d in deployments], 
										indent=2, default=str))
			else:
				# Table format
				headers = ['ID', 'Environment', 'Status', 'Created', 'Agents']
				rows = []
				
				for d in deployments:
					agent_count = sum(len(agents) for agents in d.agents_deployed.values())
					rows.append([
						d.deployment_id[:8] + '...',
						d.environment,
						d.status,
						d.created_at.strftime('%Y-%m-%d %H:%M'),
						f"{agent_count} agents"
					])
				
				click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
	
	asyncio.run(_status())

@cli.command()
@click.argument('deployment_id')
@click.pass_context
def stop(ctx, deployment_id):
	"""Stop a running deployment"""
	click.echo(f"🛑 Stopping deployment {deployment_id}...")
	
	async def _stop():
		manager = AgentDeploymentManager(ctx.obj['config'])
		
		success = await manager.stop_deployment(deployment_id)
		
		if success:
			click.echo("✅ Deployment stopped successfully")
		else:
			click.echo("❌ Failed to stop deployment")
			sys.exit(1)
	
	asyncio.run(_stop())

@cli.command()
@click.argument('deployment_id')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def report(ctx, deployment_id, output):
	"""Generate deployment report"""
	
	async def _report():
		manager = AgentDeploymentManager(ctx.obj['config'])
		
		report = await manager.create_deployment_report(deployment_id)
		
		if 'error' in report:
			click.echo(f"❌ {report['error']}")
			sys.exit(1)
		
		report_json = json.dumps(report, indent=2, default=str)
		
		if output:
			with open(output, 'w') as f:
				f.write(report_json)
			click.echo(f"📄 Report saved to {output}")
		else:
			click.echo(report_json)
	
	asyncio.run(_report())

@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), 
			  default='table', help='Output format')
@click.pass_context
def metrics(ctx, format):
	"""Show system metrics"""
	
	async def _metrics():
		manager = AgentDeploymentManager(ctx.obj['config'])
		metrics = await manager.get_system_metrics()
		
		if format == 'json':
			click.echo(json.dumps(metrics, indent=2, default=str))
		else:
			# Table format for key metrics
			click.echo("📊 APG Agent System Metrics")
			click.echo("=" * 40)
			
			# Deployments
			dep_metrics = metrics['deployments']
			click.echo(f"Deployments: {dep_metrics['total']} total, "
					  f"{dep_metrics['running']} running, "
					  f"{dep_metrics['failed']} failed")
			
			# Agents
			agent_metrics = metrics['agents']
			click.echo(f"Agents: {agent_metrics['total']} total, "
					  f"{agent_metrics['healthy']} healthy")
			
			# By role
			if agent_metrics['by_role']:
				click.echo("\nAgents by role:")
				for role, count in agent_metrics['by_role'].items():
					click.echo(f"  {role}: {count}")
			
			# Cluster health
			if metrics['clusters']['health']:
				click.echo("\nCluster Health:")
				for cluster_name, health in metrics['clusters']['health'].items():
					health_pct = health['health_percentage']
					status = "🟢" if health_pct >= 80 else "🟡" if health_pct >= 50 else "🔴"
					click.echo(f"  {status} {cluster_name}: {health_pct:.1f}% healthy")
	
	asyncio.run(_metrics())

@cli.command()
@click.argument('deployment_id')
@click.argument('role', type=click.Choice([r.value for r in AgentRole]))
@click.argument('target_count', type=int)
@click.pass_context
def scale(ctx, deployment_id, role, target_count):
	"""Scale agents for a specific role"""
	click.echo(f"⚖️  Scaling {role} agents to {target_count} instances...")
	
	async def _scale():
		manager = AgentDeploymentManager(ctx.obj['config'])
		
		# Find cluster for deployment
		cluster = None
		for cluster_name, c in manager.clusters.items():
			if deployment_id[:8] in cluster_name:
				cluster = c
				break
		
		if not cluster:
			click.echo(f"❌ Cluster for deployment {deployment_id} not found")
			sys.exit(1)
		
		try:
			agent_role = AgentRole(role)
			success = await cluster.scale_cluster(agent_role, target_count)
			
			if success:
				click.echo(f"✅ Successfully scaled {role} to {target_count} instances")
			else:
				click.echo(f"❌ Failed to scale {role}")
				sys.exit(1)
		
		except Exception as e:
			click.echo(f"❌ Scaling failed: {e}")
			sys.exit(1)
	
	asyncio.run(_scale())

@cli.command()
@click.argument('deployment_id')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), 
			  default='yaml', help='Export format')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def export_config(ctx, deployment_id, format, output):
	"""Export deployment configuration"""
	
	async def _export():
		manager = AgentDeploymentManager(ctx.obj['config'])
		
		try:
			config = manager.export_deployment_config(deployment_id, format)
			
			if output:
				with open(output, 'w') as f:
					f.write(config)
				click.echo(f"📄 Configuration exported to {output}")
			else:
				click.echo(config)
				
		except Exception as e:
			click.echo(f"❌ Export failed: {e}")
			sys.exit(1)
	
	asyncio.run(_export())

@cli.command()
@click.pass_context
def environments(ctx):
	"""List available environments"""
	manager = AgentDeploymentManager(ctx.obj['config'])
	
	click.echo("Available environments:")
	for env_name, env in manager.environments.items():
		click.echo(f"\n📁 {env_name}")
		click.echo(f"   Agent configs:")
		for role, config in env.agent_configs.items():
			scaling = "auto-scaling" if config.auto_scaling else "fixed"
			click.echo(f"     {role.value}: {config.instance_count} instances ({scaling})")

@cli.command()
@click.argument('project_spec_file')
@click.option('--environment', '-e', default='development')
@click.option('--project-name', '-n', help='Project name')
@click.pass_context
def generate(ctx, project_spec_file, environment, project_name):
	"""Generate application using APG agents"""
	click.echo(f"🏗️  Generating application from {project_spec_file}...")
	
	async def _generate():
		# Load project specification
		if not Path(project_spec_file).exists():
			click.echo(f"❌ Project spec file not found: {project_spec_file}")
			sys.exit(1)
		
		with open(project_spec_file, 'r') as f:
			if project_spec_file.endswith('.json'):
				project_spec = json.load(f)
			else:
				import yaml
				project_spec = yaml.safe_load(f)
		
		# Find running deployment for environment
		manager = AgentDeploymentManager(ctx.obj['config'])
		deployments = manager.list_deployments()
		
		running_deployment = None
		for d in deployments:
			if d.environment == environment and d.status == "running":
				running_deployment = d
				break
		
		if not running_deployment:
			click.echo(f"❌ No running deployment found for environment: {environment}")
			click.echo("   Deploy an environment first using 'apg-agents deploy'")
			sys.exit(1)
		
		# Get orchestrator from cluster
		cluster = None
		for cluster_name, c in manager.clusters.items():
			if running_deployment.deployment_id[:8] in cluster_name:
				cluster = c
				break
		
		if not cluster or 'main' not in cluster.orchestrators:
			click.echo("❌ Orchestrator not found")
			sys.exit(1)
		
		orchestrator = cluster.orchestrators['main']
		
		# Start project generation
		click.echo("🚀 Starting project generation...")
		project_id = await orchestrator.start_project(project_spec, project_name)
		
		click.echo(f"✅ Project generation started!")
		click.echo(f"   Project ID: {project_id}")
		click.echo("   Monitor progress with 'apg-agents status'")
		
		return project_id
	
	asyncio.run(_generate())

async def _show_deployment_info(manager: AgentDeploymentManager, deployment_id: str, format: str = 'table'):
	"""Show detailed deployment information"""
	report = await manager.create_deployment_report(deployment_id)
	
	if format == 'json':
		click.echo(json.dumps(report, indent=2, default=str))
		return
	
	# Table format
	click.echo(f"\n📋 Deployment: {deployment_id}")
	click.echo("=" * 50)
	
	info = report['deployment_info']
	click.echo(f"Environment: {info['environment']}")
	click.echo(f"Status: {info['status']}")
	click.echo(f"Created: {info['created_at']}")
	click.echo(f"Uptime: {info['uptime']}")
	
	# Cluster health
	if 'cluster_health' in report:
		health = report['cluster_health']
		click.echo(f"\n🏥 Cluster Health: {health['health_percentage']:.1f}%")
		click.echo(f"   Total agents: {health['total_agents']}")
		click.echo(f"   Healthy agents: {health['healthy_agents']}")
	
	# Agent deployment
	if 'agent_deployment' in report:
		click.echo(f"\n🤖 Agent Deployment:")
		for role, agents in report['agent_deployment'].items():
			click.echo(f"   {role}: {len(agents)} instances")
	
	# Orchestrator status
	if 'orchestrator_status' in report and report['orchestrator_status']:
		orch_status = report['orchestrator_status']
		click.echo(f"\n🎯 Orchestrator:")
		click.echo(f"   Active projects: {orch_status['projects']['active']}")
		click.echo(f"   Queued tasks: {orch_status['tasks']['queued']}")
		click.echo(f"   Completed tasks: {orch_status['tasks']['completed']}")
	
	# Recommendations
	if 'recommendations' in report:
		click.echo(f"\n💡 Recommendations:")
		for rec in report['recommendations']:
			click.echo(f"   • {rec}")

if __name__ == '__main__':
	cli()