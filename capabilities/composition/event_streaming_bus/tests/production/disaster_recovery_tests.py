"""
APG Event Streaming Bus - Disaster Recovery Tests

Tests for backup, recovery, and business continuity validation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import subprocess
import time
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import asyncpg
import redis
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
import kubernetes
from kubernetes import client, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecoveryTestResult:
	"""Result of a disaster recovery test."""
	test_name: str
	start_time: datetime
	end_time: datetime
	success: bool
	recovery_time_seconds: float
	data_loss_events: int
	error_message: Optional[str] = None
	metrics: Dict[str, Any] = None

@dataclass
class DisasterRecoveryReport:
	"""Complete disaster recovery test report."""
	test_date: datetime
	environment: str
	total_tests: int
	passed_tests: int
	failed_tests: int
	max_recovery_time_seconds: float
	total_data_loss_events: int
	test_results: List[RecoveryTestResult]
	rpo_achieved_seconds: float  # Recovery Point Objective
	rto_achieved_seconds: float  # Recovery Time Objective
	overall_status: str

class DisasterRecoveryTester:
	"""Comprehensive disaster recovery testing."""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.base_url = config.get('api_url', 'http://localhost:8080')
		self.database_url = config.get('database_url')
		self.redis_url = config.get('redis_url', 'redis://localhost:6379/0')
		self.kafka_servers = config.get('kafka_servers', 'localhost:9092')
		self.k8s_namespace = config.get('k8s_namespace', 'apg-event-streaming-bus')
		
		self.session: Optional[aiohttp.ClientSession] = None
		self.test_results: List[RecoveryTestResult] = []
		
		# Test data for validation
		self.test_events: List[Dict[str, Any]] = []
		self.backup_location = config.get('backup_location', '/tmp/dr_backups')
	
	async def setup(self):
		"""Setup disaster recovery test environment."""
		logger.info("Setting up disaster recovery test environment...")
		
		# HTTP session
		timeout = aiohttp.ClientTimeout(total=60)
		self.session = aiohttp.ClientSession(timeout=timeout)
		
		# Kubernetes client
		try:
			config.load_incluster_config()
		except:
			config.load_kube_config()
		
		self.k8s_apps_v1 = client.AppsV1Api()
		self.k8s_core_v1 = client.CoreV1Api()
		
		# Create backup directory
		Path(self.backup_location).mkdir(parents=True, exist_ok=True)
		
		# Generate test data
		await self._generate_test_data()
		
		logger.info("Disaster recovery test environment setup completed")
	
	async def teardown(self):
		"""Cleanup test environment."""
		if self.session:
			await self.session.close()
	
	async def _generate_test_data(self):
		"""Generate test data for recovery validation."""
		logger.info("Generating test data for recovery validation...")
		
		for i in range(1000):
			event = {
				"event_id": f"dr_test_event_{i}",
				"event_type": "disaster_recovery.test",
				"source_capability": "dr_testing",
				"aggregate_id": f"dr_aggregate_{i % 100}",
				"aggregate_type": "DRTest",
				"payload": {
					"test_index": i,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"critical_data": f"important_data_{i}",
					"checksum": f"checksum_{i:08d}"
				},
				"priority": "HIGH",
				"created_at": datetime.now(timezone.utc).isoformat()
			}
			self.test_events.append(event)
		
		# Publish test events to the system
		await self._publish_test_events()
		
		# Wait for events to be processed
		await asyncio.sleep(5)
		
		logger.info(f"Generated and published {len(self.test_events)} test events")
	
	async def _publish_test_events(self):
		"""Publish test events to the system."""
		for event in self.test_events:
			try:
				async with self.session.post(
					f"{self.base_url}/api/v1/events",
					json=event
				) as response:
					if response.status != 200:
						logger.warning(f"Failed to publish test event: {response.status}")
			except Exception as e:
				logger.warning(f"Error publishing test event: {e}")
	
	async def _verify_data_integrity(self) -> Tuple[int, int]:
		"""Verify data integrity after recovery."""
		logger.info("Verifying data integrity...")
		
		recovered_events = 0
		corrupted_events = 0
		
		try:
			async with self.session.get(
				f"{self.base_url}/api/v1/events",
				params={"event_type": "disaster_recovery.test", "limit": 2000}
			) as response:
				if response.status == 200:
					data = await response.json()
					events = data.get('events', [])
					
					for event in events:
						if event.get('event_type') == 'disaster_recovery.test':
							recovered_events += 1
							
							# Verify checksum
							payload = event.get('payload', {})
							expected_checksum = f"checksum_{payload.get('test_index', 0):08d}"
							if payload.get('checksum') != expected_checksum:
								corrupted_events += 1
		
		except Exception as e:
			logger.error(f"Error verifying data integrity: {e}")
		
		logger.info(f"Data integrity check: {recovered_events} recovered, {corrupted_events} corrupted")
		return recovered_events, corrupted_events
	
	async def test_database_backup_recovery(self) -> RecoveryTestResult:
		"""Test database backup and recovery."""
		logger.info("Testing database backup and recovery...")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Create database backup
			backup_file = f"{self.backup_location}/postgres_backup_{int(time.time())}.sql"
			
			# Extract connection details from URL
			import urllib.parse
			parsed_url = urllib.parse.urlparse(self.database_url)
			
			backup_cmd = [
				"pg_dump",
				"-h", parsed_url.hostname,
				"-p", str(parsed_url.port or 5432),
				"-U", parsed_url.username,
				"-d", parsed_url.path[1:],  # Remove leading slash
				"-f", backup_file,
				"--verbose"
			]
			
			# Set password via environment
			env = {"PGPASSWORD": parsed_url.password}
			
			# Execute backup
			backup_process = await asyncio.create_subprocess_exec(
				*backup_cmd,
				env=env,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE
			)
			
			stdout, stderr = await backup_process.communicate()
			
			if backup_process.returncode != 0:
				raise Exception(f"Database backup failed: {stderr.decode()}")
			
			logger.info(f"Database backup created: {backup_file}")
			
			# Simulate data corruption by deleting some test data
			conn = await asyncpg.connect(self.database_url)
			try:
				await conn.execute("DELETE FROM events WHERE event_type = 'disaster_recovery.test' AND event_id LIKE '%_5%'")
				logger.info("Simulated data corruption by deleting events")
			finally:
				await conn.close()
			
			# Restore from backup
			restore_cmd = [
				"psql",
				"-h", parsed_url.hostname,
				"-p", str(parsed_url.port or 5432),
				"-U", parsed_url.username,
				"-d", parsed_url.path[1:],
				"-f", backup_file
			]
			
			restore_process = await asyncio.create_subprocess_exec(
				*restore_cmd,
				env=env,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE
			)
			
			stdout, stderr = await restore_process.communicate()
			
			if restore_process.returncode != 0:
				raise Exception(f"Database restore failed: {stderr.decode()}")
			
			end_time = datetime.now(timezone.utc)
			recovery_time = (end_time - start_time).total_seconds()
			
			# Verify recovery
			recovered_events, corrupted_events = await self._verify_data_integrity()
			
			success = recovered_events > 0 and corrupted_events == 0
			
			return RecoveryTestResult(
				test_name="database_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=success,
				recovery_time_seconds=recovery_time,
				data_loss_events=len(self.test_events) - recovered_events,
				metrics={
					"backup_file_size_mb": Path(backup_file).stat().st_size / (1024 * 1024),
					"recovered_events": recovered_events,
					"corrupted_events": corrupted_events
				}
			)
		
		except Exception as e:
			end_time = datetime.now(timezone.utc)
			return RecoveryTestResult(
				test_name="database_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=False,
				recovery_time_seconds=(end_time - start_time).total_seconds(),
				data_loss_events=len(self.test_events),
				error_message=str(e)
			)
	
	async def test_redis_backup_recovery(self) -> RecoveryTestResult:
		"""Test Redis backup and recovery."""
		logger.info("Testing Redis backup and recovery...")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			redis_client = redis.from_url(self.redis_url)
			
			# Create test data in Redis
			test_keys = {}
			for i in range(100):
				key = f"dr_test:session:{i}"
				value = json.dumps({
					"user_id": f"user_{i}",
					"session_data": f"session_data_{i}",
					"timestamp": datetime.now(timezone.utc).isoformat()
				})
				redis_client.set(key, value, ex=3600)
				test_keys[key] = value
			
			logger.info(f"Created {len(test_keys)} test keys in Redis")
			
			# Create backup
			backup_file = f"{self.backup_location}/redis_backup_{int(time.time())}.rdb"
			
			# BGSAVE to create backup
			redis_client.bgsave()
			
			# Wait for backup to complete
			while redis_client.lastsave() == redis_client.lastsave():
				await asyncio.sleep(1)
			
			# Copy RDB file
			# This would typically be done by copying from Redis data directory
			# For testing, we'll simulate by saving current state
			backup_data = {}
			for key in test_keys.keys():
				backup_data[key] = redis_client.get(key)
			
			with open(backup_file, 'w') as f:
				json.dump({k: v.decode() if isinstance(v, bytes) else v for k, v in backup_data.items()}, f)
			
			# Simulate data loss
			for key in list(test_keys.keys())[::2]:  # Delete every other key
				redis_client.delete(key)
			
			logger.info("Simulated Redis data loss")
			
			# Restore from backup
			with open(backup_file, 'r') as f:
				restore_data = json.load(f)
			
			for key, value in restore_data.items():
				redis_client.set(key, value, ex=3600)
			
			end_time = datetime.now(timezone.utc)
			recovery_time = (end_time - start_time).total_seconds()
			
			# Verify recovery
			recovered_keys = 0
			corrupted_keys = 0
			
			for key, expected_value in test_keys.items():
				actual_value = redis_client.get(key)
				if actual_value:
					recovered_keys += 1
					if actual_value.decode() != expected_value:
						corrupted_keys += 1
			
			success = recovered_keys == len(test_keys) and corrupted_keys == 0
			
			return RecoveryTestResult(
				test_name="redis_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=success,
				recovery_time_seconds=recovery_time,
				data_loss_events=len(test_keys) - recovered_keys,
				metrics={
					"backup_file_size_kb": Path(backup_file).stat().st_size / 1024,
					"recovered_keys": recovered_keys,
					"corrupted_keys": corrupted_keys,
					"total_test_keys": len(test_keys)
				}
			)
		
		except Exception as e:
			end_time = datetime.now(timezone.utc)
			return RecoveryTestResult(
				test_name="redis_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=False,
				recovery_time_seconds=(end_time - start_time).total_seconds(),
				data_loss_events=100,
				error_message=str(e)
			)
	
	async def test_kafka_backup_recovery(self) -> RecoveryTestResult:
		"""Test Kafka topic backup and recovery."""
		logger.info("Testing Kafka backup and recovery...")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Create Kafka admin client
			admin_client = KafkaAdminClient(
				bootstrap_servers=self.kafka_servers.split(','),
				client_id='dr_test_admin'
			)
			
			# Create test topic
			test_topic = "dr-test-topic"
			topic_list = [NewTopic(name=test_topic, num_partitions=3, replication_factor=1)]
			
			try:
				admin_client.create_topics(topic_list)
				await asyncio.sleep(2)  # Wait for topic creation
			except Exception:
				pass  # Topic might already exist
			
			# Produce test messages
			producer = KafkaProducer(
				bootstrap_servers=self.kafka_servers.split(','),
				value_serializer=lambda x: json.dumps(x).encode('utf-8')
			)
			
			test_messages = []
			for i in range(1000):
				message = {
					"message_id": f"dr_msg_{i}",
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"data": f"test_data_{i}"
				}
				producer.send(test_topic, message)
				test_messages.append(message)
			
			producer.flush()
			logger.info(f"Produced {len(test_messages)} test messages to Kafka")
			
			# Backup: Consume all messages to file
			backup_file = f"{self.backup_location}/kafka_backup_{int(time.time())}.json"
			
			consumer = KafkaConsumer(
				test_topic,
				bootstrap_servers=self.kafka_servers.split(','),
				auto_offset_reset='earliest',
				value_deserializer=lambda x: json.loads(x.decode('utf-8')),
				consumer_timeout_ms=10000
			)
			
			backed_up_messages = []
			for message in consumer:
				backed_up_messages.append(message.value)
			
			consumer.close()
			
			with open(backup_file, 'w') as f:
				json.dump(backed_up_messages, f)
			
			logger.info(f"Backed up {len(backed_up_messages)} messages from Kafka")
			
			# Simulate data loss by deleting topic
			admin_client.delete_topics([test_topic])
			await asyncio.sleep(5)  # Wait for deletion
			
			# Recreate topic
			admin_client.create_topics(topic_list)
			await asyncio.sleep(2)
			
			# Restore from backup
			with open(backup_file, 'r') as f:
				restore_messages = json.load(f)
			
			producer = KafkaProducer(
				bootstrap_servers=self.kafka_servers.split(','),
				value_serializer=lambda x: json.dumps(x).encode('utf-8')
			)
			
			for message in restore_messages:
				producer.send(test_topic, message)
			
			producer.flush()
			
			end_time = datetime.now(timezone.utc)
			recovery_time = (end_time - start_time).total_seconds()
			
			# Verify recovery
			consumer = KafkaConsumer(
				test_topic,
				bootstrap_servers=self.kafka_servers.split(','),
				auto_offset_reset='earliest',
				value_deserializer=lambda x: json.loads(x.decode('utf-8')),
				consumer_timeout_ms=10000
			)
			
			recovered_messages = []
			for message in consumer:
				recovered_messages.append(message.value)
			
			consumer.close()
			
			success = len(recovered_messages) == len(test_messages)
			data_loss = len(test_messages) - len(recovered_messages)
			
			return RecoveryTestResult(
				test_name="kafka_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=success,
				recovery_time_seconds=recovery_time,
				data_loss_events=data_loss,
				metrics={
					"backup_file_size_kb": Path(backup_file).stat().st_size / 1024,
					"original_messages": len(test_messages),
					"backed_up_messages": len(backed_up_messages),
					"recovered_messages": len(recovered_messages)
				}
			)
		
		except Exception as e:
			end_time = datetime.now(timezone.utc)
			return RecoveryTestResult(
				test_name="kafka_backup_recovery",
				start_time=start_time,
				end_time=end_time,
				success=False,
				recovery_time_seconds=(end_time - start_time).total_seconds(),
				data_loss_events=1000,
				error_message=str(e)
			)
	
	async def test_application_pod_recovery(self) -> RecoveryTestResult:
		"""Test application pod failure and recovery."""
		logger.info("Testing application pod recovery...")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Get current pods
			pods = self.k8s_core_v1.list_namespaced_pod(
				namespace=self.k8s_namespace,
				label_selector="app.kubernetes.io/name=event-streaming-bus"
			)
			
			if not pods.items:
				raise Exception("No application pods found")
			
			# Delete one pod to simulate failure
			pod_to_delete = pods.items[0]
			pod_name = pod_to_delete.metadata.name
			
			logger.info(f"Deleting pod {pod_name} to simulate failure")
			
			self.k8s_core_v1.delete_namespaced_pod(
				name=pod_name,
				namespace=self.k8s_namespace
			)
			
			# Wait for pod to be deleted
			while True:
				try:
					self.k8s_core_v1.read_namespaced_pod(
						name=pod_name,
						namespace=self.k8s_namespace
					)
					await asyncio.sleep(1)
				except:
					break
			
			logger.info(f"Pod {pod_name} deleted")
			
			# Wait for new pod to be created and become ready
			max_wait_time = 300  # 5 minutes
			wait_time = 0
			new_pod_ready = False
			
			while wait_time < max_wait_time:
				pods = self.k8s_core_v1.list_namespaced_pod(
					namespace=self.k8s_namespace,
					label_selector="app.kubernetes.io/name=event-streaming-bus"
				)
				
				for pod in pods.items:
					if pod.status.phase == "Running":
						# Check if all containers are ready
						if pod.status.container_statuses:
							all_ready = all(
								container.ready for container in pod.status.container_statuses
							)
							if all_ready:
								new_pod_ready = True
								break
				
				if new_pod_ready:
					break
				
				await asyncio.sleep(5)
				wait_time += 5
			
			if not new_pod_ready:
				raise Exception("New pod did not become ready within timeout")
			
			# Test application health
			health_check_passed = False
			for _ in range(12):  # Try for 1 minute
				try:
					async with self.session.get(f"{self.base_url}/health") as response:
						if response.status == 200:
							health_check_passed = True
							break
				except:
					pass
				await asyncio.sleep(5)
			
			end_time = datetime.now(timezone.utc)
			recovery_time = (end_time - start_time).total_seconds()
			
			success = new_pod_ready and health_check_passed
			
			return RecoveryTestResult(
				test_name="application_pod_recovery",
				start_time=start_time,
				end_time=end_time,
				success=success,
				recovery_time_seconds=recovery_time,
				data_loss_events=0,
				metrics={
					"deleted_pod": pod_name,
					"recovery_time_seconds": recovery_time,
					"health_check_passed": health_check_passed
				}
			)
		
		except Exception as e:
			end_time = datetime.now(timezone.utc)
			return RecoveryTestResult(
				test_name="application_pod_recovery",
				start_time=start_time,
				end_time=end_time,
				success=False,
				recovery_time_seconds=(end_time - start_time).total_seconds(),
				data_loss_events=0,
				error_message=str(e)
			)
	
	async def test_complete_system_recovery(self) -> RecoveryTestResult:
		"""Test complete system disaster recovery."""
		logger.info("Testing complete system disaster recovery...")
		
		start_time = datetime.now(timezone.utc)
		
		try:
			# Create comprehensive backup
			backup_dir = f"{self.backup_location}/complete_backup_{int(time.time())}"
			Path(backup_dir).mkdir(parents=True, exist_ok=True)
			
			# Backup all components
			db_result = await self.test_database_backup_recovery()
			redis_result = await self.test_redis_backup_recovery()
			kafka_result = await self.test_kafka_backup_recovery()
			
			# Simulate complete system failure
			logger.info("Simulating complete system failure...")
			
			# Scale down all deployments
			deployments = self.k8s_apps_v1.list_namespaced_deployment(
				namespace=self.k8s_namespace
			)
			
			original_replicas = {}
			for deployment in deployments.items:
				deployment_name = deployment.metadata.name
				original_replicas[deployment_name] = deployment.spec.replicas
				
				# Scale to 0
				deployment.spec.replicas = 0
				self.k8s_apps_v1.patch_namespaced_deployment(
					name=deployment_name,
					namespace=self.k8s_namespace,
					body=deployment
				)
			
			logger.info("All deployments scaled down")
			
			# Wait for pods to terminate
			await asyncio.sleep(30)
			
			# Restore system
			logger.info("Restoring system from backup...")
			
			# Scale deployments back up
			for deployment_name, replicas in original_replicas.items():
				deployment = self.k8s_apps_v1.read_namespaced_deployment(
					name=deployment_name,
					namespace=self.k8s_namespace
				)
				deployment.spec.replicas = replicas
				self.k8s_apps_v1.patch_namespaced_deployment(
					name=deployment_name,
					namespace=self.k8s_namespace,
					body=deployment
				)
			
			# Wait for system to be healthy
			max_wait_time = 600  # 10 minutes
			wait_time = 0
			system_healthy = False
			
			while wait_time < max_wait_time:
				try:
					async with self.session.get(f"{self.base_url}/health") as response:
						if response.status == 200:
							# Verify data integrity
							recovered_events, corrupted_events = await self._verify_data_integrity()
							if recovered_events > 0:
								system_healthy = True
								break
				except:
					pass
				
				await asyncio.sleep(10)
				wait_time += 10
			
			end_time = datetime.now(timezone.utc)
			recovery_time = (end_time - start_time).total_seconds()
			
			# Calculate total data loss
			total_data_loss = (
				db_result.data_loss_events +
				redis_result.data_loss_events +
				kafka_result.data_loss_events
			)
			
			success = (
				system_healthy and
				db_result.success and
				redis_result.success and
				kafka_result.success
			)
			
			return RecoveryTestResult(
				test_name="complete_system_recovery",
				start_time=start_time,
				end_time=end_time,
				success=success,
				recovery_time_seconds=recovery_time,
				data_loss_events=total_data_loss,
				metrics={
					"database_recovery_time": db_result.recovery_time_seconds,
					"redis_recovery_time": redis_result.recovery_time_seconds,
					"kafka_recovery_time": kafka_result.recovery_time_seconds,
					"system_healthy": system_healthy,
					"backup_directory": backup_dir
				}
			)
		
		except Exception as e:
			end_time = datetime.now(timezone.utc)
			return RecoveryTestResult(
				test_name="complete_system_recovery",
				start_time=start_time,
				end_time=end_time,
				success=False,
				recovery_time_seconds=(end_time - start_time).total_seconds(),
				data_loss_events=len(self.test_events),
				error_message=str(e)
			)
	
	async def run_disaster_recovery_tests(self) -> DisasterRecoveryReport:
		"""Run comprehensive disaster recovery tests."""
		logger.info("Starting comprehensive disaster recovery tests...")
		
		test_start = datetime.now(timezone.utc)
		
		try:
			await self.setup()
			
			# Run all recovery tests
			tests = [
				self.test_database_backup_recovery(),
				self.test_redis_backup_recovery(),
				self.test_kafka_backup_recovery(),
				self.test_application_pod_recovery(),
				self.test_complete_system_recovery()
			]
			
			results = await asyncio.gather(*tests, return_exceptions=True)
			
			# Process results
			valid_results = []
			for result in results:
				if isinstance(result, RecoveryTestResult):
					valid_results.append(result)
					self.test_results.append(result)
				else:
					logger.error(f"Test failed with exception: {result}")
		
		finally:
			await self.teardown()
		
		# Calculate metrics
		passed_tests = len([r for r in self.test_results if r.success])
		failed_tests = len(self.test_results) - passed_tests
		max_recovery_time = max([r.recovery_time_seconds for r in self.test_results]) if self.test_results else 0
		total_data_loss = sum([r.data_loss_events for r in self.test_results])
		
		# Calculate RTO/RPO
		rto_achieved = max_recovery_time  # Worst case recovery time
		rpo_achieved = max([r.data_loss_events for r in self.test_results]) if self.test_results else 0
		
		# Determine overall status
		if failed_tests == 0:
			overall_status = "PASSED"
		elif failed_tests <= 1:
			overall_status = "ACCEPTABLE"
		else:
			overall_status = "FAILED"
		
		report = DisasterRecoveryReport(
			test_date=test_start,
			environment=self.config.get('environment', 'test'),
			total_tests=len(self.test_results),
			passed_tests=passed_tests,
			failed_tests=failed_tests,
			max_recovery_time_seconds=max_recovery_time,
			total_data_loss_events=total_data_loss,
			test_results=self.test_results,
			rpo_achieved_seconds=rpo_achieved,
			rto_achieved_seconds=rto_achieved,
			overall_status=overall_status
		)
		
		logger.info("Disaster recovery tests completed")
		return report

async def run_disaster_recovery_tests(config: Dict[str, Any]) -> DisasterRecoveryReport:
	"""Run disaster recovery tests and generate report."""
	tester = DisasterRecoveryTester(config)
	report = await tester.run_disaster_recovery_tests()
	
	# Save report
	report_data = asdict(report)
	report_data['test_results'] = [asdict(r) for r in report.test_results]
	
	with open(f"disaster_recovery_report_{int(datetime.now().timestamp())}.json", "w") as f:
		json.dump(report_data, f, indent=2, default=str)
	
	# Print summary
	print("\n" + "="*60)
	print("DISASTER RECOVERY TEST RESULTS")
	print("="*60)
	print(f"Test Date: {report.test_date}")
	print(f"Environment: {report.environment}")
	print(f"Overall Status: {report.overall_status}")
	print(f"\nTest Summary:")
	print(f"  Total Tests: {report.total_tests}")
	print(f"  Passed: {report.passed_tests}")
	print(f"  Failed: {report.failed_tests}")
	print(f"\nRecovery Metrics:")
	print(f"  Max Recovery Time: {report.max_recovery_time_seconds:.2f} seconds")
	print(f"  Total Data Loss Events: {report.total_data_loss_events}")
	print(f"  RTO Achieved: {report.rto_achieved_seconds:.2f} seconds")
	print(f"  RPO Achieved: {report.rpo_achieved_seconds} events")
	
	print(f"\nDetailed Results:")
	for result in report.test_results:
		status = "✅ PASSED" if result.success else "❌ FAILED"
		print(f"  {result.test_name}: {status}")
		print(f"    Recovery Time: {result.recovery_time_seconds:.2f}s")
		print(f"    Data Loss: {result.data_loss_events} events")
		if result.error_message:
			print(f"    Error: {result.error_message}")
	
	print("="*60)
	
	return report

if __name__ == "__main__":
	import sys
	
	config = {
		'api_url': sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8080',
		'database_url': sys.argv[2] if len(sys.argv) > 2 else 'postgresql://esb_user:esb_password@localhost:5432/apg_esb',
		'redis_url': sys.argv[3] if len(sys.argv) > 3 else 'redis://localhost:6379/0',
		'kafka_servers': sys.argv[4] if len(sys.argv) > 4 else 'localhost:9092',
		'k8s_namespace': sys.argv[5] if len(sys.argv) > 5 else 'apg-event-streaming-bus',
		'environment': 'production'
	}
	
	asyncio.run(run_disaster_recovery_tests(config))