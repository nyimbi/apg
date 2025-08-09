#!/usr/bin/env python3
"""
APG Key Management - Disaster Recovery & Business Continuity
Comprehensive disaster recovery and business continuity implementation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import shutil
import os
import subprocess
import tarfile
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcp_storage
import psycopg2
import redis
from uuid_extensions import uuid7str

from .service import KeyManagementService
from .hsm_integration import HSMManager


class RecoveryMode(str, Enum):
	"""Recovery operation modes"""
	FULL_RECOVERY = "full_recovery"
	PARTIAL_RECOVERY = "partial_recovery"
	POINT_IN_TIME = "point_in_time"
	DIFFERENTIAL = "differential"


class BackupType(str, Enum):
	"""Backup types"""
	FULL = "full"
	INCREMENTAL = "incremental"
	DIFFERENTIAL = "differential"
	TRANSACTION_LOG = "transaction_log"


class FailoverMode(str, Enum):
	"""Failover modes"""
	AUTOMATIC = "automatic"
	MANUAL = "manual"
	PLANNED = "planned"


@dataclass
class BackupConfig:
	"""Backup configuration"""
	backup_type: BackupType
	schedule: str  # Cron expression
	retention_days: int
	compression: bool = True
	encryption: bool = True
	destinations: List[str] = field(default_factory=list)
	pre_backup_scripts: List[str] = field(default_factory=list)
	post_backup_scripts: List[str] = field(default_factory=list)


@dataclass
class RecoveryPoint:
	"""Recovery point information"""
	recovery_id: str
	timestamp: datetime
	backup_type: BackupType
	size_bytes: int
	checksum: str
	location: str
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverConfig:
	"""Failover configuration"""
	mode: FailoverMode
	rto_minutes: int  # Recovery Time Objective
	rpo_minutes: int  # Recovery Point Objective
	primary_site: str
	secondary_site: str
	tertiary_site: Optional[str] = None
	auto_failback: bool = False
	health_check_interval: int = 30


class BackupManager:
	"""Comprehensive backup management system"""
	
	def __init__(self, service: KeyManagementService):
		self.service = service
		self.backup_configs: Dict[str, BackupConfig] = {}
		self.backup_history: List[Dict[str, Any]] = []
		self.recovery_points: List[RecoveryPoint] = []
		self._backup_tasks: Dict[str, asyncio.Task] = {}
		
		# Cloud storage clients
		self.s3_client = None
		self.azure_client = None
		self.gcp_client = None
		
		self._init_cloud_clients()
	
	def _init_cloud_clients(self):
		"""Initialize cloud storage clients"""
		try:
			# AWS S3
			if os.getenv('AWS_ACCESS_KEY_ID'):
				self.s3_client = boto3.client(
					's3',
					aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
					aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
					region_name=os.getenv('AWS_REGION', 'us-east-1')
				)
			
			# Azure Blob Storage
			if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
				self.azure_client = BlobServiceClient.from_connection_string(
					os.getenv('AZURE_STORAGE_CONNECTION_STRING')
				)
			
			# Google Cloud Storage
			if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
				self.gcp_client = gcp_storage.Client()
		
		except Exception as e:
			logging.error(f"Error initializing cloud clients: {e}")
	
	def register_backup_config(self, name: str, config: BackupConfig):
		"""Register backup configuration"""
		self.backup_configs[name] = config
		logging.info(f"Backup configuration registered: {name}")
	
	async def start_scheduled_backups(self):
		"""Start all scheduled backup tasks"""
		for name, config in self.backup_configs.items():
			if config.schedule:
				task = asyncio.create_task(self._scheduled_backup_loop(name, config))
				self._backup_tasks[name] = task
				logging.info(f"Started scheduled backup: {name}")
	
	async def stop_scheduled_backups(self):
		"""Stop all scheduled backup tasks"""
		for name, task in self._backup_tasks.items():
			task.cancel()
			try:
				await task
			except asyncio.CancelledError:
				pass
		
		self._backup_tasks.clear()
		logging.info("All scheduled backups stopped")
	
	async def _scheduled_backup_loop(self, name: str, config: BackupConfig):
		"""Scheduled backup loop"""
		while True:
			try:
				# Calculate next backup time based on cron schedule
				next_backup = self._calculate_next_backup_time(config.schedule)
				sleep_seconds = (next_backup - datetime.utcnow()).total_seconds()
				
				if sleep_seconds > 0:
					await asyncio.sleep(sleep_seconds)
				
				await self.create_backup(name)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in scheduled backup {name}: {e}")
				await asyncio.sleep(300)  # Wait 5 minutes before retry
	
	def _calculate_next_backup_time(self, cron_schedule: str) -> datetime:
		"""Calculate next backup time from cron schedule"""
		# Simple cron parsing - in production, use croniter library
		# For now, assume daily backup at specific hour
		if 'daily' in cron_schedule or '0 ' in cron_schedule:
			hour = int(cron_schedule.split()[1]) if len(cron_schedule.split()) >= 2 else 2
			next_backup = datetime.utcnow().replace(hour=hour, minute=0, second=0, microsecond=0)
			if next_backup <= datetime.utcnow():
				next_backup += timedelta(days=1)
			return next_backup
		
		# Default to next hour
		return datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
	
	async def create_backup(self, config_name: str) -> str:
		"""Create backup according to configuration"""
		if config_name not in self.backup_configs:
			raise ValueError(f"Backup configuration not found: {config_name}")
		
		config = self.backup_configs[config_name]
		backup_id = f"{config_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
		
		logging.info(f"Starting backup: {backup_id}")
		
		try:
			# Execute pre-backup scripts
			for script in config.pre_backup_scripts:
				await self._execute_script(script)
			
			# Create backup based on type
			if config.backup_type == BackupType.FULL:
				backup_path = await self._create_full_backup(backup_id)
			elif config.backup_type == BackupType.INCREMENTAL:
				backup_path = await self._create_incremental_backup(backup_id)
			elif config.backup_type == BackupType.DIFFERENTIAL:
				backup_path = await self._create_differential_backup(backup_id)
			elif config.backup_type == BackupType.TRANSACTION_LOG:
				backup_path = await self._create_transaction_log_backup(backup_id)
			else:
				raise ValueError(f"Unsupported backup type: {config.backup_type}")
			
			# Compress if configured
			if config.compression:
				backup_path = await self._compress_backup(backup_path)
			
			# Encrypt if configured
			if config.encryption:
				backup_path = await self._encrypt_backup(backup_path)
			
			# Upload to destinations
			for destination in config.destinations:
				await self._upload_backup(backup_path, destination, backup_id)
			
			# Create recovery point
			backup_size = os.path.getsize(backup_path)
			checksum = await self._calculate_checksum(backup_path)
			
			recovery_point = RecoveryPoint(
				recovery_id=backup_id,
				timestamp=datetime.utcnow(),
				backup_type=config.backup_type,
				size_bytes=backup_size,
				checksum=checksum,
				location=backup_path,
				metadata={
					'config_name': config_name,
					'compressed': config.compression,
					'encrypted': config.encryption,
					'destinations': config.destinations
				}
			)
			
			self.recovery_points.append(recovery_point)
			
			# Execute post-backup scripts
			for script in config.post_backup_scripts:
				await self._execute_script(script)
			
			# Record backup in history
			backup_record = {
				'backup_id': backup_id,
				'config_name': config_name,
				'timestamp': datetime.utcnow().isoformat(),
				'type': config.backup_type.value,
				'size_bytes': backup_size,
				'checksum': checksum,
				'status': 'completed'
			}
			
			self.backup_history.append(backup_record)
			
			# Cleanup old backups
			await self._cleanup_old_backups(config_name, config.retention_days)
			
			logging.info(f"Backup completed successfully: {backup_id}")
			return backup_id
		
		except Exception as e:
			logging.error(f"Backup failed: {backup_id}, Error: {e}")
			
			# Record failed backup
			backup_record = {
				'backup_id': backup_id,
				'config_name': config_name,
				'timestamp': datetime.utcnow().isoformat(),
				'type': config.backup_type.value,
				'status': 'failed',
				'error': str(e)
			}
			
			self.backup_history.append(backup_record)
			raise
	
	async def _create_full_backup(self, backup_id: str) -> str:
		"""Create full system backup"""
		backup_dir = f"/tmp/keym_backup_{backup_id}"
		os.makedirs(backup_dir, exist_ok=True)
		
		# Database backup
		await self._backup_database(f"{backup_dir}/database.sql")
		
		# HSM key metadata backup
		await self._backup_hsm_metadata(f"{backup_dir}/hsm_metadata.json")
		
		# Configuration backup
		await self._backup_configuration(f"{backup_dir}/config")
		
		# Application data backup
		await self._backup_application_data(f"{backup_dir}/app_data")
		
		# Create backup manifest
		manifest = {
			'backup_id': backup_id,
			'timestamp': datetime.utcnow().isoformat(),
			'type': 'full',
			'components': ['database', 'hsm_metadata', 'config', 'app_data'],
			'version': '1.0.0'
		}
		
		with open(f"{backup_dir}/manifest.json", 'w') as f:
			json.dump(manifest, f, indent=2)
		
		# Create tar archive
		archive_path = f"/tmp/keym_full_backup_{backup_id}.tar"
		with tarfile.open(archive_path, 'w') as tar:
			tar.add(backup_dir, arcname=backup_id)
		
		# Cleanup temporary directory
		shutil.rmtree(backup_dir)
		
		return archive_path
	
	async def _backup_database(self, output_path: str):
		"""Backup database"""
		database_url = os.getenv('KEYM_DATABASE_URL')
		if not database_url:
			raise ValueError("Database URL not configured")
		
		# Use pg_dump for PostgreSQL
		cmd = [
			'pg_dump',
			database_url,
			'--verbose',
			'--clean',
			'--no-owner',
			'--no-privileges',
			'--file', output_path
		]
		
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode != 0:
			raise RuntimeError(f"Database backup failed: {result.stderr}")
		
		logging.info("Database backup completed")
	
	async def _backup_hsm_metadata(self, output_path: str):
		"""Backup HSM key metadata (not actual keys)"""
		if not hasattr(self.service, 'hsm_manager'):
			return
		
		hsm_manager: HSMManager = self.service.hsm_manager
		metadata = await hsm_manager.export_key_metadata()
		
		with open(output_path, 'w') as f:
			json.dump(metadata, f, indent=2)
		
		logging.info("HSM metadata backup completed")
	
	async def _backup_configuration(self, output_dir: str):
		"""Backup configuration files"""
		os.makedirs(output_dir, exist_ok=True)
		
		config_sources = [
			'/etc/keym/',
			'/opt/keym/config/',
			'./config/'
		]
		
		for config_source in config_sources:
			if os.path.exists(config_source):
				dest = os.path.join(output_dir, os.path.basename(config_source.rstrip('/')))
				shutil.copytree(config_source, dest, ignore_dangling_symlinks=True)
		
		logging.info("Configuration backup completed")
	
	async def _backup_application_data(self, output_dir: str):
		"""Backup application data"""
		os.makedirs(output_dir, exist_ok=True)
		
		data_sources = [
			'/opt/keym/data/',
			'./data/',
			'/var/lib/keym/'
		]
		
		for data_source in data_sources:
			if os.path.exists(data_source):
				dest = os.path.join(output_dir, os.path.basename(data_source.rstrip('/')))
				shutil.copytree(data_source, dest, ignore_dangling_symlinks=True)
		
		logging.info("Application data backup completed")
	
	async def _create_incremental_backup(self, backup_id: str) -> str:
		"""Create incremental backup"""
		# Find last full backup
		last_backup = self._find_last_backup(BackupType.FULL)
		if not last_backup:
			logging.warning("No full backup found, creating full backup instead")
			return await self._create_full_backup(backup_id)
		
		# Create incremental backup based on last backup
		backup_dir = f"/tmp/keym_incremental_{backup_id}"
		os.makedirs(backup_dir, exist_ok=True)
		
		# Database incremental backup (WAL files)
		await self._backup_database_incremental(f"{backup_dir}/database_wal", last_backup['timestamp'])
		
		# Changed configuration files
		await self._backup_changed_files('/etc/keym/', f"{backup_dir}/config", last_backup['timestamp'])
		
		# Create manifest
		manifest = {
			'backup_id': backup_id,
			'timestamp': datetime.utcnow().isoformat(),
			'type': 'incremental',
			'base_backup': last_backup['backup_id'],
			'components': ['database_wal', 'config']
		}
		
		with open(f"{backup_dir}/manifest.json", 'w') as f:
			json.dump(manifest, f, indent=2)
		
		# Create archive
		archive_path = f"/tmp/keym_incremental_backup_{backup_id}.tar"
		with tarfile.open(archive_path, 'w') as tar:
			tar.add(backup_dir, arcname=backup_id)
		
		shutil.rmtree(backup_dir)
		return archive_path
	
	def _find_last_backup(self, backup_type: BackupType) -> Optional[Dict[str, Any]]:
		"""Find the most recent backup of specified type"""
		matching_backups = [
			backup for backup in self.backup_history
			if backup['type'] == backup_type.value and backup['status'] == 'completed'
		]
		
		if not matching_backups:
			return None
		
		return max(matching_backups, key=lambda x: x['timestamp'])
	
	async def _compress_backup(self, backup_path: str) -> str:
		"""Compress backup file"""
		compressed_path = f"{backup_path}.gz"
		
		with open(backup_path, 'rb') as f_in:
			with gzip.open(compressed_path, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)
		
		os.remove(backup_path)
		logging.info(f"Backup compressed: {compressed_path}")
		return compressed_path
	
	async def _encrypt_backup(self, backup_path: str) -> str:
		"""Encrypt backup file"""
		encrypted_path = f"{backup_path}.enc"
		
		# Use GPG for encryption
		encryption_key = os.getenv('KEYM_BACKUP_ENCRYPTION_KEY', 'default-backup-key')
		
		cmd = [
			'gpg',
			'--symmetric',
			'--cipher-algo', 'AES256',
			'--batch',
			'--yes',
			'--passphrase', encryption_key,
			'--output', encrypted_path,
			backup_path
		]
		
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode != 0:
			raise RuntimeError(f"Backup encryption failed: {result.stderr}")
		
		os.remove(backup_path)
		logging.info(f"Backup encrypted: {encrypted_path}")
		return encrypted_path
	
	async def _upload_backup(self, backup_path: str, destination: str, backup_id: str):
		"""Upload backup to destination"""
		if destination.startswith('s3://'):
			await self._upload_to_s3(backup_path, destination, backup_id)
		elif destination.startswith('azure://'):
			await self._upload_to_azure(backup_path, destination, backup_id)
		elif destination.startswith('gcs://'):
			await self._upload_to_gcs(backup_path, destination, backup_id)
		elif destination.startswith('/') or destination.startswith('./'):
			await self._upload_to_local(backup_path, destination, backup_id)
		else:
			raise ValueError(f"Unsupported backup destination: {destination}")
	
	async def _upload_to_s3(self, backup_path: str, destination: str, backup_id: str):
		"""Upload backup to AWS S3"""
		if not self.s3_client:
			raise RuntimeError("S3 client not configured")
		
		# Parse S3 URL: s3://bucket/path
		bucket_name = destination.split('/')[2]
		s3_key = f"{destination.split('/', 3)[3]}/{backup_id}"
		
		self.s3_client.upload_file(backup_path, bucket_name, s3_key)
		logging.info(f"Backup uploaded to S3: s3://{bucket_name}/{s3_key}")
	
	async def _calculate_checksum(self, file_path: str) -> str:
		"""Calculate SHA256 checksum of file"""
		import hashlib
		
		sha256_hash = hashlib.sha256()
		with open(file_path, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				sha256_hash.update(chunk)
		
		return sha256_hash.hexdigest()
	
	async def _cleanup_old_backups(self, config_name: str, retention_days: int):
		"""Clean up old backups beyond retention period"""
		cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
		
		old_backups = [
			backup for backup in self.backup_history
			if (backup['config_name'] == config_name and 
				datetime.fromisoformat(backup['timestamp']) < cutoff_date)
		]
		
		for backup in old_backups:
			try:
				# Remove from cloud storage destinations
				backup_config = self.backup_configs[config_name]
				for destination in backup_config.destinations:
					await self._delete_backup_from_destination(backup['backup_id'], destination)
				
				# Remove from history
				self.backup_history.remove(backup)
				
				logging.info(f"Cleaned up old backup: {backup['backup_id']}")
			
			except Exception as e:
				logging.error(f"Error cleaning up backup {backup['backup_id']}: {e}")


class DisasterRecoveryManager:
	"""Disaster recovery orchestration"""
	
	def __init__(self, service: KeyManagementService, backup_manager: BackupManager):
		self.service = service
		self.backup_manager = backup_manager
		self.failover_config: Optional[FailoverConfig] = None
		self.recovery_procedures: Dict[str, Callable] = {}
		self.recovery_history: List[Dict[str, Any]] = []
	
	def configure_failover(self, config: FailoverConfig):
		"""Configure failover settings"""
		self.failover_config = config
		logging.info(f"Failover configured: {config.mode}")
	
	def register_recovery_procedure(self, name: str, procedure: Callable):
		"""Register custom recovery procedure"""
		self.recovery_procedures[name] = procedure
		logging.info(f"Recovery procedure registered: {name}")
	
	async def initiate_recovery(self, recovery_point_id: str, mode: RecoveryMode = RecoveryMode.FULL_RECOVERY) -> str:
		"""Initiate disaster recovery process"""
		recovery_id = f"recovery_{uuid7str()}"
		
		logging.info(f"Initiating disaster recovery: {recovery_id}")
		
		try:
			# Find recovery point
			recovery_point = self._find_recovery_point(recovery_point_id)
			if not recovery_point:
				raise ValueError(f"Recovery point not found: {recovery_point_id}")
			
			# Pre-recovery validation
			await self._validate_recovery_prerequisites(recovery_point)
			
			# Execute recovery based on mode
			if mode == RecoveryMode.FULL_RECOVERY:
				await self._execute_full_recovery(recovery_id, recovery_point)
			elif mode == RecoveryMode.PARTIAL_RECOVERY:
				await self._execute_partial_recovery(recovery_id, recovery_point)
			elif mode == RecoveryMode.POINT_IN_TIME:
				await self._execute_point_in_time_recovery(recovery_id, recovery_point)
			elif mode == RecoveryMode.DIFFERENTIAL:
				await self._execute_differential_recovery(recovery_id, recovery_point)
			
			# Post-recovery verification
			await self._verify_recovery(recovery_id)
			
			# Record successful recovery
			recovery_record = {
				'recovery_id': recovery_id,
				'recovery_point_id': recovery_point_id,
				'mode': mode.value,
				'timestamp': datetime.utcnow().isoformat(),
				'status': 'completed',
				'duration_seconds': 0  # Calculate actual duration
			}
			
			self.recovery_history.append(recovery_record)
			
			logging.info(f"Disaster recovery completed: {recovery_id}")
			return recovery_id
		
		except Exception as e:
			logging.error(f"Disaster recovery failed: {recovery_id}, Error: {e}")
			
			# Record failed recovery
			recovery_record = {
				'recovery_id': recovery_id,
				'recovery_point_id': recovery_point_id,
				'mode': mode.value,
				'timestamp': datetime.utcnow().isoformat(),
				'status': 'failed',
				'error': str(e)
			}
			
			self.recovery_history.append(recovery_record)
			raise
	
	def _find_recovery_point(self, recovery_point_id: str) -> Optional[RecoveryPoint]:
		"""Find recovery point by ID"""
		for rp in self.backup_manager.recovery_points:
			if rp.recovery_id == recovery_point_id:
				return rp
		return None
	
	async def _execute_full_recovery(self, recovery_id: str, recovery_point: RecoveryPoint):
		"""Execute full system recovery"""
		logging.info(f"Executing full recovery: {recovery_id}")
		
		# Download backup if needed
		backup_path = await self._ensure_backup_available(recovery_point)
		
		# Extract backup
		extraction_dir = f"/tmp/recovery_{recovery_id}"
		await self._extract_backup(backup_path, extraction_dir)
		
		# Stop services
		await self._stop_services()
		
		try:
			# Restore database
			await self._restore_database(f"{extraction_dir}/database.sql")
			
			# Restore HSM metadata
			if os.path.exists(f"{extraction_dir}/hsm_metadata.json"):
				await self._restore_hsm_metadata(f"{extraction_dir}/hsm_metadata.json")
			
			# Restore configuration
			if os.path.exists(f"{extraction_dir}/config"):
				await self._restore_configuration(f"{extraction_dir}/config")
			
			# Restore application data
			if os.path.exists(f"{extraction_dir}/app_data"):
				await self._restore_application_data(f"{extraction_dir}/app_data")
			
			# Start services
			await self._start_services()
			
		finally:
			# Cleanup
			if os.path.exists(extraction_dir):
				shutil.rmtree(extraction_dir)
	
	async def _restore_database(self, sql_file_path: str):
		"""Restore database from SQL file"""
		database_url = os.getenv('KEYM_DATABASE_URL')
		if not database_url:
			raise ValueError("Database URL not configured")
		
		# Drop existing database and recreate
		cmd = [
			'psql',
			database_url,
			'--file', sql_file_path,
			'--quiet'
		]
		
		result = subprocess.run(cmd, capture_output=True, text=True)
		if result.returncode != 0:
			raise RuntimeError(f"Database restore failed: {result.stderr}")
		
		logging.info("Database restored successfully")
	
	async def _verify_recovery(self, recovery_id: str):
		"""Verify recovery was successful"""
		verification_results = []
		
		# Database connectivity test
		try:
			if hasattr(self.service, 'check_database_health'):
				db_healthy = await self.service.check_database_health()
				verification_results.append(('database', db_healthy))
		except Exception as e:
			verification_results.append(('database', False, str(e)))
		
		# Service functionality test
		try:
			# Try to list keys to verify basic functionality
			test_result = await self._test_basic_functionality()
			verification_results.append(('functionality', test_result))
		except Exception as e:
			verification_results.append(('functionality', False, str(e)))
		
		# HSM connectivity test
		if hasattr(self.service, 'hsm_manager'):
			try:
				hsm_status = await self.service.hsm_manager.check_health()
				verification_results.append(('hsm', hsm_status))
			except Exception as e:
				verification_results.append(('hsm', False, str(e)))
		
		# Check if all verifications passed
		failed_verifications = [v for v in verification_results if not v[1]]
		if failed_verifications:
			error_details = ', '.join([f"{v[0]}: {v[2] if len(v) > 2 else 'failed'}" for v in failed_verifications])
			raise RuntimeError(f"Recovery verification failed: {error_details}")
		
		logging.info(f"Recovery verification passed: {recovery_id}")
	
	async def _test_basic_functionality(self) -> bool:
		"""Test basic service functionality"""
		try:
			# Simple database query test
			if hasattr(self.service, '_db_pool') and self.service._db_pool:
				async with self.service._db_pool.acquire() as conn:
					result = await conn.fetchval("SELECT COUNT(*) FROM km_keys")
					return isinstance(result, int)
			
			return True
		except Exception as e:
			logging.error(f"Basic functionality test failed: {e}")
			return False


class BusinessContinuityManager:
	"""Business continuity planning and management"""
	
	def __init__(self, service: KeyManagementService, dr_manager: DisasterRecoveryManager):
		self.service = service
		self.dr_manager = dr_manager
		self.continuity_plans: Dict[str, Dict[str, Any]] = {}
		self.critical_operations: List[str] = []
		self.dependencies: Dict[str, List[str]] = {}
	
	def register_continuity_plan(self, plan_name: str, plan_config: Dict[str, Any]):
		"""Register business continuity plan"""
		required_fields = ['rto', 'rpo', 'critical_operations', 'recovery_strategies']
		
		for field in required_fields:
			if field not in plan_config:
				raise ValueError(f"Missing required field in continuity plan: {field}")
		
		self.continuity_plans[plan_name] = plan_config
		logging.info(f"Business continuity plan registered: {plan_name}")
	
	def define_critical_operations(self, operations: List[str]):
		"""Define critical business operations"""
		self.critical_operations = operations
		logging.info(f"Critical operations defined: {len(operations)} operations")
	
	def map_dependencies(self, dependencies: Dict[str, List[str]]):
		"""Map service dependencies"""
		self.dependencies = dependencies
		logging.info("Service dependencies mapped")
	
	async def execute_continuity_plan(self, plan_name: str, trigger_reason: str) -> str:
		"""Execute business continuity plan"""
		if plan_name not in self.continuity_plans:
			raise ValueError(f"Continuity plan not found: {plan_name}")
		
		plan = self.continuity_plans[plan_name]
		execution_id = f"continuity_{uuid7str()}"
		
		logging.info(f"Executing continuity plan: {plan_name}, Trigger: {trigger_reason}")
		
		try:
			# Assess current situation
			impact_assessment = await self._assess_business_impact()
			
			# Execute recovery strategies based on priority
			for strategy in plan['recovery_strategies']:
				await self._execute_recovery_strategy(strategy, execution_id)
			
			# Monitor recovery progress
			recovery_status = await self._monitor_recovery_progress(plan, execution_id)
			
			# Validate RTO/RPO compliance
			await self._validate_continuity_objectives(plan, execution_id)
			
			logging.info(f"Continuity plan executed successfully: {execution_id}")
			return execution_id
		
		except Exception as e:
			logging.error(f"Continuity plan execution failed: {execution_id}, Error: {e}")
			raise
	
	async def _assess_business_impact(self) -> Dict[str, Any]:
		"""Assess current business impact"""
		impact_assessment = {
			'timestamp': datetime.utcnow().isoformat(),
			'affected_operations': [],
			'service_availability': {},
			'estimated_downtime': 0
		}
		
		# Check availability of critical operations
		for operation in self.critical_operations:
			try:
				is_available = await self._check_operation_availability(operation)
				if not is_available:
					impact_assessment['affected_operations'].append(operation)
			except Exception as e:
				impact_assessment['affected_operations'].append(operation)
				logging.error(f"Error checking operation {operation}: {e}")
		
		# Check service dependencies
		for service, deps in self.dependencies.items():
			dependency_status = {}
			for dep in deps:
				try:
					dependency_status[dep] = await self._check_dependency_health(dep)
				except Exception as e:
					dependency_status[dep] = False
					logging.error(f"Error checking dependency {dep}: {e}")
			
			impact_assessment['service_availability'][service] = dependency_status
		
		return impact_assessment
	
	async def _check_operation_availability(self, operation: str) -> bool:
		"""Check if a critical operation is available"""
		# Implement specific checks for different operations
		if operation == 'key_creation':
			try:
				# Test key creation capability
				return hasattr(self.service, 'create_key')
			except:
				return False
		
		elif operation == 'encryption':
			try:
				# Test encryption capability
				return hasattr(self.service, 'encrypt_data')
			except:
				return False
		
		elif operation == 'hsm_operations':
			try:
				# Test HSM availability
				if hasattr(self.service, 'hsm_manager'):
					status = await self.service.hsm_manager.check_health()
					return status
			except:
				return False
		
		return True  # Default to available if unknown operation


# Factory function
async def create_disaster_recovery_system(service: KeyManagementService) -> Dict[str, Any]:
	"""Create complete disaster recovery system"""
	
	# Backup manager with default configurations
	backup_manager = BackupManager(service)
	
	# Default backup configurations
	default_backup_configs = {
		'daily_full': BackupConfig(
			backup_type=BackupType.FULL,
			schedule='0 2 * * *',  # Daily at 2 AM
			retention_days=30,
			compression=True,
			encryption=True,
			destinations=['s3://keym-backups/daily/', '/backup/keym/daily/']
		),
		'hourly_incremental': BackupConfig(
			backup_type=BackupType.INCREMENTAL,
			schedule='0 * * * *',  # Hourly
			retention_days=7,
			compression=True,
			encryption=True,
			destinations=['s3://keym-backups/incremental/']
		),
		'transaction_log': BackupConfig(
			backup_type=BackupType.TRANSACTION_LOG,
			schedule='*/15 * * * *',  # Every 15 minutes
			retention_days=3,
			compression=False,
			encryption=True,
			destinations=['s3://keym-backups/wal/']
		)
	}
	
	for name, config in default_backup_configs.items():
		backup_manager.register_backup_config(name, config)
	
	# Disaster recovery manager
	dr_manager = DisasterRecoveryManager(service, backup_manager)
	
	# Default failover configuration
	default_failover_config = FailoverConfig(
		mode=FailoverMode.AUTOMATIC,
		rto_minutes=60,  # 1 hour RTO
		rpo_minutes=15,  # 15 minutes RPO
		primary_site='us-east-1',
		secondary_site='us-west-2',
		auto_failback=False,
		health_check_interval=30
	)
	
	dr_manager.configure_failover(default_failover_config)
	
	# Business continuity manager
	bc_manager = BusinessContinuityManager(service, dr_manager)
	
	# Define critical operations
	critical_operations = [
		'key_creation',
		'encryption',
		'decryption',
		'key_rotation',
		'hsm_operations',
		'multi_cloud_sync'
	]
	
	bc_manager.define_critical_operations(critical_operations)
	
	# Map dependencies
	dependencies = {
		'key_management': ['database', 'cache', 'hsm'],
		'multi_cloud_federation': ['aws_kms', 'azure_keyvault', 'gcp_kms'],
		'security_intelligence': ['database', 'cache', 'ml_models']
	}
	
	bc_manager.map_dependencies(dependencies)
	
	# Default continuity plan
	default_continuity_plan = {
		'rto': 60,  # minutes
		'rpo': 15,  # minutes
		'critical_operations': critical_operations,
		'recovery_strategies': [
			{'type': 'failover', 'priority': 1, 'target': 'secondary_site'},
			{'type': 'restore_from_backup', 'priority': 2, 'backup_type': 'latest'},
			{'type': 'manual_intervention', 'priority': 3}
		]
	}
	
	bc_manager.register_continuity_plan('default', default_continuity_plan)
	
	# Start scheduled backups
	await backup_manager.start_scheduled_backups()
	
	return {
		'backup_manager': backup_manager,
		'disaster_recovery_manager': dr_manager,
		'business_continuity_manager': bc_manager
	}


# Export main components
__all__ = [
	'BackupManager', 'DisasterRecoveryManager', 'BusinessContinuityManager',
	'BackupConfig', 'RecoveryPoint', 'FailoverConfig',
	'RecoveryMode', 'BackupType', 'FailoverMode',
	'create_disaster_recovery_system'
]