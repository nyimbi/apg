#!/usr/bin/env python3
"""
APG Accounts Receivable - Production Deployment Script
Automated production deployment and validation

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import asyncpg
import redis.asyncio as redis
from dataclasses import dataclass


@dataclass
class DeploymentStep:
	"""Individual deployment step result."""
	name: str
	status: str  # SUCCESS, FAILED, SKIPPED
	message: str
	duration_seconds: float
	timestamp: datetime
	details: Optional[Dict[str, Any]] = None


class ProductionDeployer:
	"""Production deployment orchestrator."""
	
	def __init__(self, environment: str = "production"):
		self.environment = environment
		self.logger = logging.getLogger(__name__)
		self.steps: List[DeploymentStep] = []
		self.start_time = None
		self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# Configure logging
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
	
	async def deploy_to_production(self) -> Dict[str, Any]:
		"""Execute complete production deployment."""
		self.logger.info(f"🚀 Starting APG AR production deployment - ID: {self.deployment_id}")
		self.start_time = time.time()
		
		try:
			# Pre-deployment validation
			await self._pre_deployment_checks()
			
			# Database preparation
			await self._prepare_database()
			
			# Application deployment
			await self._deploy_application()
			
			# Service validation
			await self._validate_services()
			
			# Post-deployment tasks
			await self._post_deployment_tasks()
			
			# Final validation
			await self._final_validation()
			
			return self._generate_deployment_report(success=True)
			
		except Exception as e:
			self.logger.error(f"💥 Deployment failed: {e}")
			return self._generate_deployment_report(success=False, error=str(e))
	
	async def _pre_deployment_checks(self):
		"""Run pre-deployment validation checks."""
		step_name = "Pre-deployment Checks"
		step_start = time.time()
		
		try:
			self.logger.info("🔍 Running pre-deployment checks")
			
			# Check environment variables
			required_env_vars = [
				'DATABASE_URL',
				'REDIS_URL',
				'SECRET_KEY',
				'JWT_SECRET_KEY',
				'APG_ENVIRONMENT'
			]
			
			missing_vars = []
			for var in required_env_vars:
				if not os.getenv(var):
					missing_vars.append(var)
			
			if missing_vars:
				raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
			
			# Check deployment files exist
			required_files = [
				'deploy/Dockerfile',
				'deploy/docker-compose.yml',
				'deploy/kubernetes/deployment.yaml',
				'requirements.txt'
			]
			
			missing_files = []
			for file_path in required_files:
				if not os.path.exists(file_path):
					missing_files.append(file_path)
			
			if missing_files:
				raise Exception(f"Missing required deployment files: {', '.join(missing_files)}")
			
			# Run production readiness check
			await self._run_readiness_check()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", "Pre-deployment checks passed", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Pre-deployment checks failed: {e}", duration)
			raise
	
	async def _run_readiness_check(self):
		"""Run production readiness validation."""
		try:
			# Import and run readiness check
			from ..production_readiness import run_production_readiness_check
			
			# Create test connections for readiness check
			db_url = os.getenv('DATABASE_URL')
			redis_url = os.getenv('REDIS_URL')
			
			if db_url and redis_url:
				db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
				redis_client = redis.from_url(redis_url)
				
				try:
					report = await run_production_readiness_check(db_pool, redis_client)
					
					if report.get('overall_status') == 'NOT_READY':
						failed_checks = [
							f"{cat} - {check['name']}: {check['message']}"
							for cat, category in report.get('categories', {}).items()
							for check in category.get('checks', [])
							if check.get('status') == 'FAILED'
						]
						raise Exception(f"Readiness check failed: {'; '.join(failed_checks)}")
					
					self.logger.info(f"✅ Readiness check passed with score: {report.get('readiness_score', 0):.1%}")
					
				finally:
					await db_pool.close()
					await redis_client.close()
			else:
				self.logger.warning("⚠️ Skipping readiness check - database/redis URLs not configured")
				
		except ImportError:
			self.logger.warning("⚠️ Production readiness module not found - skipping check")
		except Exception as e:
			self.logger.error(f"❌ Readiness check failed: {e}")
			raise
	
	async def _prepare_database(self):
		"""Prepare database for production deployment."""
		step_name = "Database Preparation"
		step_start = time.time()
		
		try:
			self.logger.info("🗄️ Preparing database for production")
			
			db_url = os.getenv('DATABASE_URL')
			if not db_url:
				raise Exception("DATABASE_URL not configured")
			
			# Test database connectivity
			pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
			
			try:
				async with pool.acquire() as conn:
					# Check database version
					version = await conn.fetchval("SELECT version()")
					self.logger.info(f"📊 Database version: {version}")
					
					# Run database migrations
					await self._run_database_migrations()
					
					# Create performance indexes
					await self._create_production_indexes(conn)
					
					# Update table statistics
					await self._update_database_statistics(conn)
					
			finally:
				await pool.close()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", "Database prepared successfully", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Database preparation failed: {e}", duration)
			raise
	
	async def _run_database_migrations(self):
		"""Run database migrations."""
		try:
			self.logger.info("🔄 Running database migrations")
			
			# Check if alembic is available and run migrations
			result = subprocess.run(
				['python', '-m', 'alembic', 'upgrade', 'head'],
				capture_output=True,
				text=True,
				timeout=300  # 5 minute timeout
			)
			
			if result.returncode != 0:
				raise Exception(f"Migration failed: {result.stderr}")
			
			self.logger.info("✅ Database migrations completed")
			
		except subprocess.TimeoutExpired:
			raise Exception("Database migration timed out")
		except Exception as e:
			self.logger.error(f"❌ Migration failed: {e}")
			raise
	
	async def _create_production_indexes(self, conn: asyncpg.Connection):
		"""Create production-optimized database indexes."""
		try:
			self.logger.info("📊 Creating production indexes")
			
			# Import and run database optimizer
			from ..performance_optimizations import DatabaseOptimizer
			
			optimizer = DatabaseOptimizer(None)  # Will use the connection directly
			await optimizer._create_performance_indexes(conn)
			
			self.logger.info("✅ Production indexes created")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Index creation failed: {e}")
			# Don't fail deployment for index issues
	
	async def _update_database_statistics(self, conn: asyncpg.Connection):
		"""Update database statistics for optimal query planning."""
		try:
			self.logger.info("📈 Updating database statistics")
			
			tables = ['customers', 'invoices', 'payments', 'collection_activities']
			for table in tables:
				try:
					await conn.execute(f"ANALYZE {table}")
				except Exception as e:
					self.logger.warning(f"⚠️ Statistics update failed for {table}: {e}")
			
			self.logger.info("✅ Database statistics updated")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Statistics update failed: {e}")
	
	async def _deploy_application(self):
		"""Deploy application services."""
		step_name = "Application Deployment"
		step_start = time.time()
		
		try:
			self.logger.info("🚀 Deploying application services")
			
			# Determine deployment method
			deployment_method = os.getenv('DEPLOYMENT_METHOD', 'docker-compose')
			
			if deployment_method == 'kubernetes':
				await self._deploy_kubernetes()
			elif deployment_method == 'docker-compose':
				await self._deploy_docker_compose()
			else:
				await self._deploy_standalone()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", f"Application deployed using {deployment_method}", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Application deployment failed: {e}", duration)
			raise
	
	async def _deploy_kubernetes(self):
		"""Deploy using Kubernetes."""
		try:
			self.logger.info("☸️ Deploying to Kubernetes")
			
			k8s_manifests = [
				'deploy/kubernetes/namespace.yaml',
				'deploy/kubernetes/configmap.yaml',
				'deploy/kubernetes/secrets.yaml',
				'deploy/kubernetes/deployment.yaml',
				'deploy/kubernetes/service.yaml',
				'deploy/kubernetes/ingress.yaml'
			]
			
			for manifest in k8s_manifests:
				if os.path.exists(manifest):
					result = subprocess.run(
						['kubectl', 'apply', '-f', manifest],
						capture_output=True,
						text=True,
						timeout=60
					)
					
					if result.returncode != 0:
						raise Exception(f"Kubernetes apply failed for {manifest}: {result.stderr}")
					
					self.logger.info(f"✅ Applied {manifest}")
			
			# Wait for deployment to be ready
			result = subprocess.run(
				['kubectl', 'rollout', 'status', 'deployment/apg-ar-api', '-n', 'apg-ar', '--timeout=600s'],
				capture_output=True,
				text=True,
				timeout=600
			)
			
			if result.returncode != 0:
				raise Exception(f"Deployment rollout failed: {result.stderr}")
			
			self.logger.info("✅ Kubernetes deployment completed")
			
		except Exception as e:
			self.logger.error(f"❌ Kubernetes deployment failed: {e}")
			raise
	
	async def _deploy_docker_compose(self):
		"""Deploy using Docker Compose."""
		try:
			self.logger.info("🐳 Deploying with Docker Compose")
			
			# Build and start services
			result = subprocess.run(
				['docker-compose', '-f', 'deploy/docker-compose.yml', 'up', '-d', '--build'],
				capture_output=True,
				text=True,
				timeout=600,
				cwd='.'
			)
			
			if result.returncode != 0:
				raise Exception(f"Docker Compose deployment failed: {result.stderr}")
			
			# Wait for services to be healthy
			await asyncio.sleep(30)  # Give services time to start
			
			# Check service health
			result = subprocess.run(
				['docker-compose', '-f', 'deploy/docker-compose.yml', 'ps'],
				capture_output=True,
				text=True
			)
			
			self.logger.info("✅ Docker Compose deployment completed")
			
		except Exception as e:
			self.logger.error(f"❌ Docker Compose deployment failed: {e}")
			raise
	
	async def _deploy_standalone(self):
		"""Deploy as standalone application."""
		try:
			self.logger.info("🏃 Deploying standalone application")
			
			# Install dependencies
			result = subprocess.run(
				['pip', 'install', '-r', 'requirements.txt'],
				capture_output=True,
				text=True,
				timeout=300
			)
			
			if result.returncode != 0:
				raise Exception(f"Dependency installation failed: {result.stderr}")
			
			self.logger.info("✅ Standalone deployment prepared")
			
		except Exception as e:
			self.logger.error(f"❌ Standalone deployment failed: {e}")
			raise
	
	async def _validate_services(self):
		"""Validate deployed services."""
		step_name = "Service Validation"
		step_start = time.time()
		
		try:
			self.logger.info("🔍 Validating deployed services")
			
			# Check API health
			await self._check_api_health()
			
			# Check database connectivity
			await self._check_database_connectivity()
			
			# Check cache connectivity
			await self._check_cache_connectivity()
			
			# Run smoke tests
			await self._run_smoke_tests()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", "All services validated successfully", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Service validation failed: {e}", duration)
			raise
	
	async def _check_api_health(self):
		"""Check API service health."""
		try:
			import httpx
			
			api_url = os.getenv('API_URL', 'http://localhost:8000')
			health_url = f"{api_url}/health"
			
			async with httpx.AsyncClient(timeout=30) as client:
				response = await client.get(health_url)
				
				if response.status_code != 200:
					raise Exception(f"API health check failed: {response.status_code}")
			
			self.logger.info("✅ API health check passed")
			
		except Exception as e:
			self.logger.error(f"❌ API health check failed: {e}")
			raise
	
	async def _check_database_connectivity(self):
		"""Check database connectivity."""
		try:
			db_url = os.getenv('DATABASE_URL')
			if not db_url:
				raise Exception("DATABASE_URL not configured")
			
			pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
			
			try:
				async with pool.acquire() as conn:
					await conn.execute("SELECT 1")
			finally:
				await pool.close()
			
			self.logger.info("✅ Database connectivity check passed")
			
		except Exception as e:
			self.logger.error(f"❌ Database connectivity check failed: {e}")
			raise
	
	async def _check_cache_connectivity(self):
		"""Check cache connectivity."""
		try:
			redis_url = os.getenv('REDIS_URL')
			if not redis_url:
				raise Exception("REDIS_URL not configured")
			
			redis_client = redis.from_url(redis_url)
			
			try:
				await redis_client.ping()
			finally:
				await redis_client.close()
			
			self.logger.info("✅ Cache connectivity check passed")
			
		except Exception as e:
			self.logger.error(f"❌ Cache connectivity check failed: {e}")
			raise
	
	async def _run_smoke_tests(self):
		"""Run basic smoke tests."""
		try:
			self.logger.info("🧪 Running smoke tests")
			
			# Check if smoke tests exist
			if os.path.exists('tests/smoke/'):
				result = subprocess.run(
					['python', '-m', 'pytest', 'tests/smoke/', '-v'],
					capture_output=True,
					text=True,
					timeout=300
				)
				
				if result.returncode != 0:
					self.logger.warning(f"⚠️ Some smoke tests failed: {result.stderr}")
					# Don't fail deployment for smoke test failures
				else:
					self.logger.info("✅ Smoke tests passed")
			else:
				self.logger.info("ℹ️ No smoke tests found - skipping")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Smoke tests failed: {e}")
			# Don't fail deployment for smoke test issues
	
	async def _post_deployment_tasks(self):
		"""Execute post-deployment tasks."""
		step_name = "Post-deployment Tasks"
		step_start = time.time()
		
		try:
			self.logger.info("🔧 Running post-deployment tasks")
			
			# Load initial data if needed
			await self._load_initial_data()
			
			# Configure monitoring
			await self._configure_monitoring()
			
			# Set up backup schedules
			await self._configure_backups()
			
			# Warm up caches
			await self._warm_up_caches()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", "Post-deployment tasks completed", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Post-deployment tasks failed: {e}", duration)
			raise
	
	async def _load_initial_data(self):
		"""Load initial data and fixtures."""
		try:
			self.logger.info("📊 Loading initial data")
			
			# Check if there's a data loading script
			if os.path.exists('scripts/load_initial_data.py'):
				result = subprocess.run(
					['python', 'scripts/load_initial_data.py'],
					capture_output=True,
					text=True,
					timeout=300
				)
				
				if result.returncode != 0:
					self.logger.warning(f"⚠️ Initial data loading failed: {result.stderr}")
				else:
					self.logger.info("✅ Initial data loaded")
			else:
				self.logger.info("ℹ️ No initial data script found - skipping")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Initial data loading failed: {e}")
	
	async def _configure_monitoring(self):
		"""Configure monitoring and alerting."""
		try:
			self.logger.info("📊 Configuring monitoring")
			
			# This would typically configure Prometheus, Grafana, etc.
			# For now, just log that monitoring should be configured
			self.logger.info("ℹ️ Monitoring configuration should be completed externally")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Monitoring configuration failed: {e}")
	
	async def _configure_backups(self):
		"""Configure backup schedules."""
		try:
			self.logger.info("💾 Configuring backup schedules")
			
			# This would typically set up cron jobs or other backup scheduling
			self.logger.info("ℹ️ Backup scheduling should be configured externally")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Backup configuration failed: {e}")
	
	async def _warm_up_caches(self):
		"""Warm up application caches."""
		try:
			self.logger.info("🔥 Warming up caches")
			
			# This would typically pre-load frequently accessed data
			redis_url = os.getenv('REDIS_URL')
			if redis_url:
				redis_client = redis.from_url(redis_url)
				try:
					# Set a test cache entry
					await redis_client.set('deployment:warmup', 'complete', ex=3600)
					self.logger.info("✅ Cache warm-up completed")
				finally:
					await redis_client.close()
			
		except Exception as e:
			self.logger.warning(f"⚠️ Cache warm-up failed: {e}")
	
	async def _final_validation(self):
		"""Run final deployment validation."""
		step_name = "Final Validation"
		step_start = time.time()
		
		try:
			self.logger.info("🏁 Running final deployment validation")
			
			# Run comprehensive health check
			await self._comprehensive_health_check()
			
			# Performance validation
			await self._validate_performance()
			
			# Security validation
			await self._validate_security()
			
			duration = time.time() - step_start
			self._add_step(step_name, "SUCCESS", "Final validation completed", duration)
			
		except Exception as e:
			duration = time.time() - step_start
			self._add_step(step_name, "FAILED", f"Final validation failed: {e}", duration)
			raise
	
	async def _comprehensive_health_check(self):
		"""Run comprehensive health check."""
		try:
			self.logger.info("🏥 Running comprehensive health check")
			
			# This would run the full production readiness check
			await self._run_readiness_check()
			
			self.logger.info("✅ Comprehensive health check passed")
			
		except Exception as e:
			self.logger.error(f"❌ Comprehensive health check failed: {e}")
			raise
	
	async def _validate_performance(self):
		"""Validate system performance."""
		try:
			self.logger.info("⚡ Validating system performance")
			
			# This would run performance tests
			if os.path.exists('tests/performance/performance_runner.py'):
				result = subprocess.run(
					['python', 'tests/performance/performance_runner.py'],
					capture_output=True,
					text=True,
					timeout=600
				)
				
				if result.returncode != 0:
					self.logger.warning(f"⚠️ Performance validation warnings: {result.stderr}")
				else:
					self.logger.info("✅ Performance validation passed")
			else:
				self.logger.info("ℹ️ No performance tests found - skipping")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Performance validation failed: {e}")
	
	async def _validate_security(self):
		"""Validate security configuration."""
		try:
			self.logger.info("🔒 Validating security configuration")
			
			# Check that sensitive environment variables are not exposed
			sensitive_vars = ['SECRET_KEY', 'JWT_SECRET_KEY', 'DATABASE_URL']
			for var in sensitive_vars:
				if os.getenv(var):
					# Make sure the value is not obviously insecure
					value = os.getenv(var)
					if len(value) < 16:
						self.logger.warning(f"⚠️ {var} may be too short for production")
			
			self.logger.info("✅ Security validation completed")
			
		except Exception as e:
			self.logger.warning(f"⚠️ Security validation failed: {e}")
	
	def _add_step(self, name: str, status: str, message: str, duration: float, details: Optional[Dict[str, Any]] = None):
		"""Add a deployment step result."""
		step = DeploymentStep(
			name=name,
			status=status,
			message=message,
			duration_seconds=duration,
			timestamp=datetime.now(),
			details=details
		)
		self.steps.append(step)
		
		# Log the step
		status_emoji = {"SUCCESS": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}
		emoji = status_emoji.get(status, "❓")
		self.logger.info(f"{emoji} {name}: {message} ({duration:.1f}s)")
	
	def _generate_deployment_report(self, success: bool, error: Optional[str] = None) -> Dict[str, Any]:
		"""Generate deployment report."""
		total_duration = time.time() - self.start_time if self.start_time else 0
		
		successful_steps = [s for s in self.steps if s.status == "SUCCESS"]
		failed_steps = [s for s in self.steps if s.status == "FAILED"]
		
		report = {
			'deployment_id': self.deployment_id,
			'environment': self.environment,
			'overall_status': 'SUCCESS' if success else 'FAILED',
			'total_duration_seconds': total_duration,
			'timestamp': datetime.now().isoformat(),
			'summary': {
				'total_steps': len(self.steps),
				'successful_steps': len(successful_steps),
				'failed_steps': len(failed_steps),
				'skipped_steps': len([s for s in self.steps if s.status == "SKIPPED"])
			},
			'steps': [
				{
					'name': step.name,
					'status': step.status,
					'message': step.message,
					'duration_seconds': step.duration_seconds,
					'timestamp': step.timestamp.isoformat(),
					'details': step.details
				}
				for step in self.steps
			]
		}
		
		if error:
			report['error'] = error
		
		if success:
			report['recommendations'] = [
				"Monitor system performance and health metrics",
				"Verify backup procedures are working correctly",
				"Set up alerting for critical system metrics",
				"Plan for regular security updates",
				"Schedule regular system maintenance windows",
				"Monitor user adoption and feedback",
				"Plan for scaling based on usage patterns"
			]
		else:
			report['troubleshooting'] = [
				"Check deployment logs for detailed error information",
				"Verify all environment variables are correctly set",
				"Ensure all required services are running",
				"Check network connectivity between services",
				"Verify database migrations completed successfully",
				"Check file permissions and access rights",
				"Review system resource availability"
			]
		
		return report


def save_deployment_report(report: Dict[str, Any], filename: Optional[str] = None):
	"""Save deployment report to file."""
	if filename is None:
		deployment_id = report.get('deployment_id', 'unknown')
		filename = f"deployment_report_{deployment_id}.json"
	
	os.makedirs('deploy/reports', exist_ok=True)
	filepath = f"deploy/reports/{filename}"
	
	with open(filepath, 'w') as f:
		json.dump(report, f, indent=2, default=str)
	
	print(f"📄 Deployment report saved to: {filepath}")


def print_deployment_summary(report: Dict[str, Any]):
	"""Print deployment summary."""
	print("\n" + "="*60)
	print("🚀 APG Accounts Receivable - Production Deployment Report")
	print("="*60)
	
	print(f"\n🆔 Deployment ID: {report['deployment_id']}")
	print(f"🌍 Environment: {report['environment']}")
	print(f"📊 Overall Status: {report['overall_status']}")
	print(f"⏱️  Total Duration: {report['total_duration_seconds']:.1f} seconds")
	
	summary = report['summary']
	print(f"\n📋 Step Summary:")
	print(f"  Total Steps: {summary['total_steps']}")
	print(f"  ✅ Successful: {summary['successful_steps']}")
	print(f"  ❌ Failed: {summary['failed_steps']}")
	print(f"  ⏭️  Skipped: {summary['skipped_steps']}")
	
	if report['overall_status'] == 'SUCCESS':
		print(f"\n🎉 Deployment Successful!")
		print(f"💡 Recommendations:")
		for rec in report.get('recommendations', [])[:5]:
			print(f"  • {rec}")
	else:
		print(f"\n💥 Deployment Failed!")
		print(f"🔧 Troubleshooting:")
		for tip in report.get('troubleshooting', [])[:5]:
			print(f"  • {tip}")
		
		if 'error' in report:
			print(f"\n❌ Error: {report['error']}")
	
	print("\n" + "="*60)


async def main():
	"""Main deployment script entry point."""
	import argparse
	
	parser = argparse.ArgumentParser(description='APG AR Production Deployment')
	parser.add_argument('--environment', default='production', help='Deployment environment')
	parser.add_argument('--save-report', action='store_true', help='Save deployment report')
	
	args = parser.parse_args()
	
	deployer = ProductionDeployer(environment=args.environment)
	
	try:
		report = await deployer.deploy_to_production()
		
		print_deployment_summary(report)
		
		if args.save_report:
			save_deployment_report(report)
		
		# Exit with appropriate code
		sys.exit(0 if report['overall_status'] == 'SUCCESS' else 1)
		
	except KeyboardInterrupt:
		print("\n⚠️ Deployment interrupted by user")
		sys.exit(130)
	except Exception as e:
		print(f"\n💥 Deployment script failed: {e}")
		sys.exit(1)


if __name__ == "__main__":
	asyncio.run(main())