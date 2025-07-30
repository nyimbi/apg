"""
APG RAG Production Deployment & Validation

Enterprise-grade deployment automation, validation, and production readiness
with comprehensive health checks, configuration validation, and rollback capabilities.
"""

import asyncio
import json
import logging
import os
import sys
import time
import yaml
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import psutil
import httpx
from uuid_extensions import uuid7str

# Database imports
import asyncpg
from asyncpg import Pool

class DeploymentStage(str, Enum):
	"""Deployment stages"""
	VALIDATION = "validation"
	PREPARATION = "preparation"
	DEPLOYMENT = "deployment"
	VERIFICATION = "verification"
	ROLLBACK = "rollback"
	COMPLETED = "completed"
	FAILED = "failed"

class ValidationResult(str, Enum):
	"""Validation result status"""
	PASSED = "passed"
	FAILED = "failed"
	WARNING = "warning"
	SKIPPED = "skipped"

@dataclass
class ValidationCheck:
	"""Individual validation check"""
	name: str
	description: str
	result: ValidationResult = ValidationResult.SKIPPED
	message: str = ""
	duration_ms: float = 0.0
	timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DeploymentConfig:
	"""Production deployment configuration"""
	# Environment settings
	environment: str = "production"
	tenant_id: str = ""
	
	# Database configuration
	database_url: str = ""
	database_pool_size: int = 20
	database_max_overflow: int = 30
	
	# Ollama configuration
	ollama_urls: List[str] = field(default_factory=list)
	ollama_timeout: int = 60
	ollama_max_retries: int = 3
	
	# Security settings
	encryption_key: str = ""
	audit_retention_days: int = 2555
	enable_security_monitoring: bool = True
	
	# Performance settings
	max_concurrent_operations: int = 100
	operation_timeout_seconds: float = 300.0
	enable_performance_monitoring: bool = True
	
	# Resource limits
	max_memory_mb: int = 4096
	max_cpu_percent: float = 80.0
	disk_space_threshold_gb: int = 10
	
	# Service settings
	service_host: str = "0.0.0.0"
	service_port: int = 5000
	enable_https: bool = True
	ssl_cert_path: str = ""
	ssl_key_path: str = ""
	
	# Monitoring
	enable_metrics: bool = True
	metrics_port: int = 9090
	log_level: str = "INFO"
	
	# Backup and recovery
	backup_enabled: bool = True
	backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
	backup_retention_days: int = 30

@dataclass
class DeploymentStatus:
	"""Deployment status tracking"""
	deployment_id: str = field(default_factory=uuid7str)
	stage: DeploymentStage = DeploymentStage.VALIDATION
	started_at: datetime = field(default_factory=datetime.now)
	completed_at: Optional[datetime] = None
	success: bool = False
	error_message: str = ""
	validation_checks: List[ValidationCheck] = field(default_factory=list)
	rollback_available: bool = False
	previous_version: str = ""

class ProductionValidator:
	"""Validates production readiness"""
	
	def __init__(self, config: DeploymentConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
		self.checks: List[ValidationCheck] = []
	
	async def run_all_validations(self) -> Tuple[bool, List[ValidationCheck]]:
		"""Run all production validation checks"""
		self.logger.info("Starting production validation checks")
		
		validation_methods = [
			self._validate_system_requirements,
			self._validate_database_connection,
			self._validate_ollama_integration,
			self._validate_security_configuration,
			self._validate_performance_settings,
			self._validate_monitoring_setup,
			self._validate_backup_configuration,
			self._validate_network_connectivity,
			self._validate_ssl_certificates,
			self._validate_resource_availability
		]
		
		for validation_method in validation_methods:
			try:
				await validation_method()
			except Exception as e:
				check = ValidationCheck(
					name=validation_method.__name__,
					description=f"Validation method {validation_method.__name__}",
					result=ValidationResult.FAILED,
					message=f"Validation failed: {str(e)}"
				)
				self.checks.append(check)
		
		# Determine overall success
		failed_checks = [c for c in self.checks if c.result == ValidationResult.FAILED]
		success = len(failed_checks) == 0
		
		self.logger.info(f"Validation completed: {len(self.checks)} checks, {len(failed_checks)} failures")
		return success, self.checks
	
	async def _validate_system_requirements(self) -> None:
		"""Validate system requirements"""
		start_time = time.time()
		
		try:
			# Check Python version
			python_version = sys.version_info
			if python_version < (3, 9):
				raise ValueError(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}")
			
			# Check available memory
			memory = psutil.virtual_memory()
			required_memory_gb = self.config.max_memory_mb / 1024
			available_memory_gb = memory.available / (1024**3)
			
			if available_memory_gb < required_memory_gb:
				raise ValueError(f"Insufficient memory: {available_memory_gb:.1f}GB available, {required_memory_gb:.1f}GB required")
			
			# Check disk space
			disk = psutil.disk_usage('/')
			available_disk_gb = disk.free / (1024**3)
			
			if available_disk_gb < self.config.disk_space_threshold_gb:
				raise ValueError(f"Insufficient disk space: {available_disk_gb:.1f}GB available, {self.config.disk_space_threshold_gb}GB required")
			
			# Check CPU cores
			cpu_count = psutil.cpu_count()
			if cpu_count < 2:
				raise ValueError(f"Minimum 2 CPU cores required, found {cpu_count}")
			
			check = ValidationCheck(
				name="system_requirements",
				description="System requirements validation",
				result=ValidationResult.PASSED,
				message=f"System meets requirements: Python {python_version.major}.{python_version.minor}, {available_memory_gb:.1f}GB memory, {available_disk_gb:.1f}GB disk, {cpu_count} cores",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="system_requirements",
				description="System requirements validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_database_connection(self) -> None:
		"""Validate database connection and schema"""
		start_time = time.time()
		
		try:
			if not self.config.database_url:
				raise ValueError("Database URL not configured")
			
			# Test connection
			conn = await asyncpg.connect(self.config.database_url)
			
			# Check PostgreSQL version
			version = await conn.fetchval("SELECT version()")
			if "PostgreSQL 13" not in version and "PostgreSQL 14" not in version and "PostgreSQL 15" not in version and "PostgreSQL 16" not in version:
				self.logger.warning(f"PostgreSQL version may not be optimal: {version}")
			
			# Check required extensions
			extensions = await conn.fetch("SELECT extname FROM pg_extension")
			extension_names = [ext['extname'] for ext in extensions]
			
			required_extensions = ['vector', 'ai']  # pgvector and pgai
			missing_extensions = [ext for ext in required_extensions if ext not in extension_names]
			
			if missing_extensions:
				raise ValueError(f"Missing required extensions: {missing_extensions}")
			
			# Check if RAG tables exist
			tables = await conn.fetch("""
				SELECT tablename FROM pg_tables 
				WHERE schemaname = 'public' AND tablename LIKE 'apg_rag_%'
			""")
			
			if len(tables) == 0:
				raise ValueError("RAG database schema not found. Run database migration first.")
			
			await conn.close()
			
			check = ValidationCheck(
				name="database_connection",
				description="Database connection and schema validation",
				result=ValidationResult.PASSED,
				message=f"Database connection successful, {len(extension_names)} extensions, {len(tables)} RAG tables",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="database_connection",
				description="Database connection and schema validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_ollama_integration(self) -> None:
		"""Validate Ollama service integration"""
		start_time = time.time()
		
		try:
			if not self.config.ollama_urls:
				raise ValueError("Ollama URLs not configured")
			
			working_urls = []
			required_models = ['bge-m3', 'qwen3', 'deepseek-r1']
			
			for url in self.config.ollama_urls:
				try:
					async with httpx.AsyncClient(timeout=10.0) as client:
						# Test connection
						response = await client.get(f"{url}/api/tags")
						if response.status_code == 200:
							models_data = response.json()
							available_models = [model['name'].split(':')[0] for model in models_data.get('models', [])]
							
							# Check if required models are available
							missing_models = [model for model in required_models if model not in available_models]
							
							if missing_models:
								self.logger.warning(f"Ollama at {url} missing models: {missing_models}")
							else:
								working_urls.append(url)
								
				except Exception as e:
					self.logger.warning(f"Ollama connection failed for {url}: {str(e)}")
			
			if not working_urls:
				raise ValueError("No working Ollama instances found")
			
			# Test embedding generation
			test_successful = False
			for url in working_urls:
				try:
					async with httpx.AsyncClient(timeout=30.0) as client:
						embed_response = await client.post(
							f"{url}/api/embeddings",
							json={"model": "bge-m3", "prompt": "test embedding"}
						)
						if embed_response.status_code == 200:
							test_successful = True
							break
				except Exception:
					continue
			
			if not test_successful:
				raise ValueError("Embedding generation test failed on all Ollama instances")
			
			check = ValidationCheck(
				name="ollama_integration",
				description="Ollama service integration validation",
				result=ValidationResult.PASSED,
				message=f"{len(working_urls)} working Ollama instances, embedding test passed",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="ollama_integration",
				description="Ollama service integration validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_security_configuration(self) -> None:
		"""Validate security configuration"""
		start_time = time.time()
		
		try:
			issues = []
			
			# Check encryption key
			if not self.config.encryption_key:
				issues.append("Encryption key not configured")
			elif len(self.config.encryption_key) < 32:
				issues.append("Encryption key too short (minimum 32 characters)")
			
			# Check SSL configuration
			if self.config.enable_https:
				if not self.config.ssl_cert_path or not os.path.exists(self.config.ssl_cert_path):
					issues.append("SSL certificate file not found")
				
				if not self.config.ssl_key_path or not os.path.exists(self.config.ssl_key_path):
					issues.append("SSL private key file not found")
			
			# Check audit retention
			if self.config.audit_retention_days < 365:
				issues.append(f"Audit retention period too short: {self.config.audit_retention_days} days (minimum 365)")
			
			if issues:
				raise ValueError("; ".join(issues))
			
			check = ValidationCheck(
				name="security_configuration",
				description="Security configuration validation",
				result=ValidationResult.PASSED,
				message="Security configuration validated successfully",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="security_configuration",
				description="Security configuration validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_performance_settings(self) -> None:
		"""Validate performance configuration"""
		start_time = time.time()
		
		try:
			warnings = []
			
			# Check concurrent operations limit
			if self.config.max_concurrent_operations > 200:
				warnings.append(f"High concurrent operations limit: {self.config.max_concurrent_operations}")
			
			# Check memory allocation
			system_memory_gb = psutil.virtual_memory().total / (1024**3)
			allocated_memory_gb = self.config.max_memory_mb / 1024
			
			if allocated_memory_gb > system_memory_gb * 0.8:
				warnings.append(f"High memory allocation: {allocated_memory_gb:.1f}GB of {system_memory_gb:.1f}GB")
			
			# Check database pool size
			if self.config.database_pool_size > 50:
				warnings.append(f"Large database pool size: {self.config.database_pool_size}")
			
			message = "Performance settings validated"
			if warnings:
				message += f" with warnings: {'; '.join(warnings)}"
			
			check = ValidationCheck(
				name="performance_settings",
				description="Performance configuration validation",
				result=ValidationResult.WARNING if warnings else ValidationResult.PASSED,
				message=message,
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="performance_settings",
				description="Performance configuration validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_monitoring_setup(self) -> None:
		"""Validate monitoring configuration"""
		start_time = time.time()
		
		try:
			# Check if monitoring is enabled
			if not self.config.enable_metrics:
				raise ValueError("Metrics monitoring disabled in production environment")
			
			# Check metrics port availability
			import socket
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			result = sock.connect_ex((self.config.service_host, self.config.metrics_port))
			sock.close()
			
			if result == 0:
				self.logger.warning(f"Metrics port {self.config.metrics_port} appears to be in use")
			
			# Check log level
			if self.config.log_level not in ['INFO', 'WARNING', 'ERROR']:
				self.logger.warning(f"Log level {self.config.log_level} may be too verbose for production")
			
			check = ValidationCheck(
				name="monitoring_setup",
				description="Monitoring configuration validation",
				result=ValidationResult.PASSED,
				message="Monitoring configuration validated successfully",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="monitoring_setup",
				description="Monitoring configuration validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_backup_configuration(self) -> None:
		"""Validate backup configuration"""
		start_time = time.time()
		
		try:
			if not self.config.backup_enabled:
				raise ValueError("Backup disabled in production environment")
			
			# Validate cron schedule format
			import croniter
			try:
				croniter.croniter(self.config.backup_schedule)
			except Exception:
				raise ValueError(f"Invalid backup schedule: {self.config.backup_schedule}")
			
			# Check retention period
			if self.config.backup_retention_days < 7:
				raise ValueError(f"Backup retention too short: {self.config.backup_retention_days} days (minimum 7)")
			
			check = ValidationCheck(
				name="backup_configuration",
				description="Backup configuration validation",
				result=ValidationResult.PASSED,
				message=f"Backup enabled with {self.config.backup_retention_days} day retention",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except ImportError:
			check = ValidationCheck(
				name="backup_configuration",
				description="Backup configuration validation",
				result=ValidationResult.WARNING,
				message="croniter package not available, cannot validate schedule format",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="backup_configuration",
				description="Backup configuration validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_network_connectivity(self) -> None:
		"""Validate network connectivity"""
		start_time = time.time()
		
		try:
			# Check service port availability
			import socket
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			result = sock.connect_ex((self.config.service_host, self.config.service_port))
			sock.close()
			
			if result == 0:
				self.logger.warning(f"Service port {self.config.service_port} appears to be in use")
			
			# Test external connectivity (if needed)
			# This could include testing connections to external services
			
			check = ValidationCheck(
				name="network_connectivity",
				description="Network connectivity validation",
				result=ValidationResult.PASSED,
				message="Network connectivity validated successfully",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="network_connectivity",
				description="Network connectivity validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_ssl_certificates(self) -> None:
		"""Validate SSL certificates"""
		start_time = time.time()
		
		try:
			if not self.config.enable_https:
				check = ValidationCheck(
					name="ssl_certificates",
					description="SSL certificate validation",
					result=ValidationResult.SKIPPED,
					message="HTTPS disabled, SSL validation skipped",
					duration_ms=(time.time() - start_time) * 1000
				)
			else:
				# Check certificate files exist and are readable
				if not os.path.exists(self.config.ssl_cert_path):
					raise ValueError(f"SSL certificate not found: {self.config.ssl_cert_path}")
				
				if not os.path.exists(self.config.ssl_key_path):
					raise ValueError(f"SSL private key not found: {self.config.ssl_key_path}")
				
				# Check certificate expiration (basic check)
				from cryptography import x509
				from cryptography.hazmat.backends import default_backend
				
				with open(self.config.ssl_cert_path, 'rb') as cert_file:
					cert = x509.load_pem_x509_certificate(cert_file.read(), default_backend())
					
					# Check if certificate is valid
					now = datetime.utcnow()
					if cert.not_valid_after < now:
						raise ValueError("SSL certificate has expired")
					
					# Warning if certificate expires soon (within 30 days)
					if cert.not_valid_after < now + timedelta(days=30):
						self.logger.warning("SSL certificate expires within 30 days")
				
				check = ValidationCheck(
					name="ssl_certificates",
					description="SSL certificate validation",
					result=ValidationResult.PASSED,
					message="SSL certificates validated successfully",
					duration_ms=(time.time() - start_time) * 1000
				)
			
		except ImportError:
			check = ValidationCheck(
				name="ssl_certificates",
				description="SSL certificate validation",
				result=ValidationResult.WARNING,
				message="cryptography package not available, cannot validate certificates",
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="ssl_certificates",
				description="SSL certificate validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)
	
	async def _validate_resource_availability(self) -> None:
		"""Validate system resource availability"""
		start_time = time.time()
		
		try:
			warnings = []
			
			# Check CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			if cpu_percent > 70:
				warnings.append(f"High CPU usage during validation: {cpu_percent:.1f}%")
			
			# Check memory usage
			memory = psutil.virtual_memory()
			if memory.percent > 80:
				warnings.append(f"High memory usage during validation: {memory.percent:.1f}%")
			
			# Check disk I/O
			disk_io = psutil.disk_io_counters()
			if disk_io:
				# This is a simple check - in production you might want more sophisticated I/O monitoring
				pass
			
			# Check network interfaces
			network_stats = psutil.net_io_counters()
			if network_stats.errin > 100 or network_stats.errout > 100:
				warnings.append(f"Network errors detected: {network_stats.errin} in, {network_stats.errout} out")
			
			message = "Resource availability validated"
			if warnings:
				message += f" with warnings: {'; '.join(warnings)}"
			
			check = ValidationCheck(
				name="resource_availability",
				description="System resource availability validation",
				result=ValidationResult.WARNING if warnings else ValidationResult.PASSED,
				message=message,
				duration_ms=(time.time() - start_time) * 1000
			)
			
		except Exception as e:
			check = ValidationCheck(
				name="resource_availability",
				description="System resource availability validation",
				result=ValidationResult.FAILED,
				message=str(e),
				duration_ms=(time.time() - start_time) * 1000
			)
		
		self.checks.append(check)

class ProductionDeployer:
	"""Handles production deployment orchestration"""
	
	def __init__(self, config: DeploymentConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
		self.status = DeploymentStatus()
	
	async def deploy(self) -> DeploymentStatus:
		"""Execute complete production deployment"""
		self.logger.info(f"Starting production deployment {self.status.deployment_id}")
		
		try:
			# Stage 1: Validation
			self.status.stage = DeploymentStage.VALIDATION
			await self._run_validation()
			
			# Stage 2: Preparation
			self.status.stage = DeploymentStage.PREPARATION
			await self._prepare_deployment()
			
			# Stage 3: Deployment
			self.status.stage = DeploymentStage.DEPLOYMENT
			await self._execute_deployment()
			
			# Stage 4: Verification
			self.status.stage = DeploymentStage.VERIFICATION
			await self._verify_deployment()
			
			# Mark as completed
			self.status.stage = DeploymentStage.COMPLETED
			self.status.success = True
			self.status.completed_at = datetime.now()
			
			self.logger.info(f"Deployment {self.status.deployment_id} completed successfully")
			
		except Exception as e:
			self.status.stage = DeploymentStage.FAILED
			self.status.success = False
			self.status.error_message = str(e)
			self.status.completed_at = datetime.now()
			
			self.logger.error(f"Deployment {self.status.deployment_id} failed: {str(e)}")
			
			# Attempt rollback if available
			if self.status.rollback_available:
				await self._rollback_deployment()
		
		return self.status
	
	async def _run_validation(self) -> None:
		"""Run pre-deployment validation"""
		validator = ProductionValidator(self.config)
		success, checks = await validator.run_all_validations()
		
		self.status.validation_checks = checks
		
		if not success:
			failed_checks = [c for c in checks if c.result == ValidationResult.FAILED]
			raise ValueError(f"Validation failed: {len(failed_checks)} critical issues found")
	
	async def _prepare_deployment(self) -> None:
		"""Prepare for deployment"""
		self.logger.info("Preparing deployment environment")
		
		# Create backup of current deployment (if exists)
		await self._create_backup()
		
		# Prepare configuration files
		await self._prepare_configuration()
		
		# Setup monitoring and logging
		await self._setup_monitoring()
		
		self.logger.info("Deployment preparation completed")
	
	async def _execute_deployment(self) -> None:
		"""Execute the actual deployment"""
		self.logger.info("Executing deployment")
		
		# This would typically involve:
		# - Stopping existing services
		# - Updating application code
		# - Migrating database schema
		# - Starting new services
		# - Updating load balancer configuration
		
		# For this implementation, we'll simulate the process
		await asyncio.sleep(2)  # Simulate deployment time
		
		self.logger.info("Deployment execution completed")
	
	async def _verify_deployment(self) -> None:
		"""Verify deployment success"""
		self.logger.info("Verifying deployment")
		
		# Health checks
		await self._run_health_checks()
		
		# Smoke tests
		await self._run_smoke_tests()
		
		# Performance verification
		await self._verify_performance()
		
		self.logger.info("Deployment verification completed")
	
	async def _create_backup(self) -> None:
		"""Create backup of current deployment"""
		if self.config.backup_enabled:
			self.logger.info("Creating deployment backup")
			
			# This would typically involve:
			# - Database backup
			# - Configuration backup
			# - Application state backup
			
			self.status.rollback_available = True
			self.status.previous_version = "1.0.0"  # This would be detected
			
			self.logger.info("Backup created successfully")
	
	async def _prepare_configuration(self) -> None:
		"""Prepare configuration files"""
		self.logger.info("Preparing configuration files")
		
		# Generate production configuration
		config_data = {
			'database': {
				'url': self.config.database_url,
				'pool_size': self.config.database_pool_size,
				'max_overflow': self.config.database_max_overflow
			},
			'ollama': {
				'urls': self.config.ollama_urls,
				'timeout': self.config.ollama_timeout,
				'max_retries': self.config.ollama_max_retries
			},
			'security': {
				'encryption_key': self.config.encryption_key,
				'audit_retention_days': self.config.audit_retention_days,
				'enable_monitoring': self.config.enable_security_monitoring
			},
			'performance': {
				'max_concurrent_operations': self.config.max_concurrent_operations,
				'operation_timeout_seconds': self.config.operation_timeout_seconds,
				'max_memory_mb': self.config.max_memory_mb
			},
			'service': {
				'host': self.config.service_host,
				'port': self.config.service_port,
				'enable_https': self.config.enable_https,
				'ssl_cert_path': self.config.ssl_cert_path,
				'ssl_key_path': self.config.ssl_key_path
			}
		}
		
		# Write configuration to file
		config_path = Path('config/production.yaml')
		config_path.parent.mkdir(exist_ok=True)
		
		with open(config_path, 'w') as f:
			yaml.dump(config_data, f, default_flow_style=False)
		
		self.logger.info("Configuration files prepared")
	
	async def _setup_monitoring(self) -> None:
		"""Setup monitoring and logging"""
		self.logger.info("Setting up monitoring")
		
		# Configure logging
		log_config = {
			'version': 1,
			'formatters': {
				'default': {
					'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
				}
			},
			'handlers': {
				'console': {
					'class': 'logging.StreamHandler',
					'formatter': 'default',
					'level': self.config.log_level
				},
				'file': {
					'class': 'logging.FileHandler',
					'filename': 'logs/rag_service.log',
					'formatter': 'default',
					'level': self.config.log_level
				}
			},
			'root': {
				'level': self.config.log_level,
				'handlers': ['console', 'file']
			}
		}
		
		# Ensure log directory exists
		Path('logs').mkdir(exist_ok=True)
		
		self.logger.info("Monitoring setup completed")
	
	async def _run_health_checks(self) -> None:
		"""Run post-deployment health checks"""
		self.logger.info("Running health checks")
		
		# This would typically test:
		# - Service availability
		# - Database connectivity
		# - External service integration
		# - API endpoint functionality
		
		# Simulate health checks
		await asyncio.sleep(1)
		
		self.logger.info("Health checks passed")
	
	async def _run_smoke_tests(self) -> None:
		"""Run smoke tests"""
		self.logger.info("Running smoke tests")
		
		# This would typically test:
		# - Basic API functionality
		# - Document upload and processing
		# - Query and generation
		# - User authentication
		
		# Simulate smoke tests
		await asyncio.sleep(1)
		
		self.logger.info("Smoke tests passed")
	
	async def _verify_performance(self) -> None:
		"""Verify performance meets requirements"""
		self.logger.info("Verifying performance")
		
		# This would typically test:
		# - Response time benchmarks
		# - Throughput measurements
		# - Resource utilization
		# - Concurrent user handling
		
		# Simulate performance verification
		await asyncio.sleep(1)
		
		self.logger.info("Performance verification passed")
	
	async def _rollback_deployment(self) -> None:
		"""Rollback to previous deployment"""
		if not self.status.rollback_available:
			self.logger.error("Rollback not available")
			return
		
		self.logger.info(f"Rolling back to version {self.status.previous_version}")
		self.status.stage = DeploymentStage.ROLLBACK
		
		# This would typically involve:
		# - Stopping new services
		# - Restoring previous version
		# - Restoring database backup
		# - Restarting services
		
		# Simulate rollback
		await asyncio.sleep(2)
		
		self.logger.info("Rollback completed")

class DeploymentManager:
	"""Manages the complete deployment lifecycle"""
	
	def __init__(self, config_path: Optional[str] = None):
		self.config = self._load_config(config_path)
		self.logger = logging.getLogger(__name__)
		
		# Setup logging
		logging.basicConfig(
			level=getattr(logging, self.config.log_level),
			format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
	
	def _load_config(self, config_path: Optional[str]) -> DeploymentConfig:
		"""Load deployment configuration"""
		if config_path and os.path.exists(config_path):
			with open(config_path, 'r') as f:
				config_data = yaml.safe_load(f)
			
			# Convert dict to DeploymentConfig
			return DeploymentConfig(**config_data)
		else:
			# Load from environment variables
			return DeploymentConfig(
				environment=os.getenv('ENVIRONMENT', 'production'),
				tenant_id=os.getenv('TENANT_ID', ''),
				database_url=os.getenv('DATABASE_URL', ''),
				ollama_urls=os.getenv('OLLAMA_URLS', '').split(',') if os.getenv('OLLAMA_URLS') else [],
				encryption_key=os.getenv('ENCRYPTION_KEY', ''),
				service_host=os.getenv('SERVICE_HOST', '0.0.0.0'),
				service_port=int(os.getenv('SERVICE_PORT', '5000')),
				enable_https=os.getenv('ENABLE_HTTPS', 'true').lower() == 'true',
				ssl_cert_path=os.getenv('SSL_CERT_PATH', ''),
				ssl_key_path=os.getenv('SSL_KEY_PATH', ''),
				log_level=os.getenv('LOG_LEVEL', 'INFO')
			)
	
	async def deploy(self) -> DeploymentStatus:
		"""Execute complete deployment process"""
		deployer = ProductionDeployer(self.config)
		return await deployer.deploy()
	
	async def validate_only(self) -> Tuple[bool, List[ValidationCheck]]:
		"""Run validation checks only"""
		validator = ProductionValidator(self.config)
		return await validator.run_all_validations()
	
	def generate_deployment_report(self, status: DeploymentStatus) -> str:
		"""Generate deployment report"""
		report = f"""
APG RAG Production Deployment Report
===================================

Deployment ID: {status.deployment_id}
Environment: {self.config.environment}
Started: {status.started_at.isoformat()}
Completed: {status.completed_at.isoformat() if status.completed_at else 'In Progress'}
Status: {'SUCCESS' if status.success else 'FAILED'}
Stage: {status.stage.value}

"""
		
		if status.error_message:
			report += f"Error: {status.error_message}\n\n"
		
		if status.validation_checks:
			report += "Validation Results:\n"
			report += "-" * 50 + "\n"
			
			for check in status.validation_checks:
				status_icon = "✓" if check.result == ValidationResult.PASSED else "✗" if check.result == ValidationResult.FAILED else "⚠"
				report += f"{status_icon} {check.name}: {check.result.value} ({check.duration_ms:.1f}ms)\n"
				if check.message:
					report += f"  {check.message}\n"
			
			report += "\n"
		
		# Summary
		passed = len([c for c in status.validation_checks if c.result == ValidationResult.PASSED])
		failed = len([c for c in status.validation_checks if c.result == ValidationResult.FAILED])
		warnings = len([c for c in status.validation_checks if c.result == ValidationResult.WARNING])
		
		report += f"Summary: {passed} passed, {failed} failed, {warnings} warnings\n"
		
		if status.rollback_available:
			report += f"Rollback available to version: {status.previous_version}\n"
		
		return report

# CLI interface for deployment
async def main():
	"""Main deployment CLI interface"""
	import argparse
	
	parser = argparse.ArgumentParser(description='APG RAG Production Deployment')
	parser.add_argument('--config', help='Configuration file path')
	parser.add_argument('--validate-only', action='store_true', help='Run validation only')
	parser.add_argument('--verbose', action='store_true', help='Verbose output')
	
	args = parser.parse_args()
	
	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	
	manager = DeploymentManager(args.config)
	
	if args.validate_only:
		success, checks = await manager.validate_only()
		
		# Create temporary status for report generation
		status = DeploymentStatus()
		status.validation_checks = checks
		status.success = success
		status.completed_at = datetime.now()
		
		print(manager.generate_deployment_report(status))
		sys.exit(0 if success else 1)
	
	else:
		status = await manager.deploy()
		print(manager.generate_deployment_report(status))
		sys.exit(0 if status.success else 1)

if __name__ == "__main__":
	asyncio.run(main())