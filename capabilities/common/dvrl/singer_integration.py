#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Singer.io Integration
Enhanced data connectivity using Singer.io taps and targets

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid_extensions import uuid7str

from .models import DataSource, DataSourceType, DataSourceStatus
from .connectors import BaseConnector


class SingerTapConnector(BaseConnector):
	"""Connector that integrates Singer.io taps for data extraction"""
	
	def __init__(self, data_source: DataSource):
		super().__init__(data_source)
		self.tap_name = data_source.connection_config.get('tap_name', '')
		self.tap_config = data_source.connection_config.get('tap_config', {})
		self.catalog_file = None
		self.state_file = None
		self.temp_dir = None
		self.supported_taps = {
			'tap-postgres': 'postgresql database',
			'tap-mysql': 'mysql database', 
			'tap-mongodb': 'mongodb database',
			'tap-salesforce': 'salesforce crm',
			'tap-stripe': 'stripe payments',
			'tap-hubspot': 'hubspot crm',
			'tap-github': 'github repositories',
			'tap-jira': 'atlassian jira',
			'tap-slack': 'slack workspace',
			'tap-zendesk': 'zendesk support',
			'tap-csv': 'csv files',
			'tap-s3-csv': 's3 csv files',
			'tap-google-sheets': 'google sheets',
			'tap-facebook': 'facebook marketing',
			'tap-google-ads': 'google ads',
			'tap-mailchimp': 'mailchimp marketing'
		}
		
	async def initialize(self) -> bool:
		"""Initialize Singer tap connector"""
		try:
			await self._log_info(f"Initializing Singer tap: {self.tap_name}")
			
			# Create temporary directory for Singer files
			self.temp_dir = tempfile.mkdtemp(prefix='dvrl_singer_')
			
			# Validate tap availability
			if not await self._validate_tap_installation():
				raise Exception(f"Singer tap '{self.tap_name}' not installed or not accessible")
			
			# Setup tap configuration
			config_file = await self._create_tap_config()
			if not config_file:
				raise Exception("Failed to create tap configuration")
			
			# Discover schema
			catalog = await self._discover_tap_catalog()
			if not catalog:
				raise Exception("Failed to discover tap catalog")
			
			self.catalog_file = await self._create_catalog_file(catalog)
			self.state_file = await self._create_state_file()
			
			await self._log_info(f"Singer tap initialized successfully: {self.tap_name}")
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to initialize Singer tap {self.tap_name}", e)
			return False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Singer tap connection"""
		try:
			# Measure actual connection time
			start_time = asyncio.get_event_loop().time()
			
			# Run tap discovery to test connection
			discovery_result = await self._discover_tap_catalog()
			
			connection_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
			
			if discovery_result and 'streams' in discovery_result:
				stream_count = len(discovery_result['streams'])
				return {
					'success': True,
					'tap_name': self.tap_name,
					'streams_discovered': stream_count,
					'connection_time_ms': connection_time_ms,
					'tap_version': await self._get_tap_version()
				}
			else:
				return {
					'success': False,
					'error': 'No streams discovered from tap',
					'tap_name': self.tap_name
				}
				
		except Exception as e:
			return {
				'success': False,
				'error': str(e),
				'tap_name': self.tap_name
			}
	
	async def discover_schema(self) -> Any:
		"""Discover schema using Singer tap discovery"""
		try:
			catalog = await self._discover_tap_catalog()
			if not catalog:
				raise Exception("Failed to discover catalog")
			
			# Convert Singer catalog to DVRL schema format
			dvrl_schema = await self._convert_catalog_to_schema(catalog)
			return dvrl_schema
			
		except Exception as e:
			await self._log_error("Schema discovery failed", e)
			raise
	
	async def execute_query(self, query: str, options: Dict[str, Any] = None) -> Any:
		"""Execute query using Singer tap extraction"""
		try:
			await self._log_info(f"Executing Singer tap extraction: {query}")
			
			# For Singer taps, "query" is typically a stream selection
			stream_name = self._extract_stream_from_query(query)
			
			if not stream_name:
				raise Exception("No valid stream found in query")
			
			# Run Singer tap extraction
			extraction_result = await self._run_tap_extraction(stream_name, options or {})
			
			return {
				'tap_name': self.tap_name,
				'stream_name': stream_name,
				'records_extracted': extraction_result.get('record_count', 0),
				'extraction_time_ms': extraction_result.get('duration_ms', 0),
				'data': extraction_result.get('records', [])
			}
			
		except Exception as e:
			await self._log_error(f"Query execution failed: {query}", e)
			raise
	
	async def get_connection_stats(self) -> Dict[str, Any]:
		"""Get Singer tap connection statistics"""
		return {
			'tap_name': self.tap_name,
			'tap_type': 'singer_tap',
			'supported_operations': ['extract', 'discover'],
			'health_status': 'healthy' if await self._validate_tap_installation() else 'unhealthy',
			'capabilities': [
				'schema_discovery',
				'incremental_extraction', 
				'full_table_extraction',
				'catalog_management'
			],
			'streams_available': len((await self._discover_tap_catalog()).get('streams', [])),
			'last_extraction': None,  # Would track in production
			'config_valid': bool(self.tap_config)
		}
	
	# Singer.io Specific Methods
	
	async def _validate_tap_installation(self) -> bool:
		"""Validate that Singer tap is installed and accessible"""
		try:
			# Try to run tap with --version flag
			process = await asyncio.create_subprocess_exec(
				self.tap_name, '--version',
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE
			)
			stdout, stderr = await process.communicate()
			
			return process.returncode == 0
			
		except FileNotFoundError:
			await self._log_error(f"Singer tap not found: {self.tap_name}", None)
			return False
		except Exception as e:
			await self._log_error(f"Tap validation failed: {self.tap_name}", e)
			return False
	
	async def _create_tap_config(self) -> Optional[str]:
		"""Create Singer tap configuration file"""
		try:
			config_path = Path(self.temp_dir) / f"{self.tap_name}_config.json"
			
			with open(config_path, 'w') as f:
				json.dump(self.tap_config, f, indent=2)
			
			return str(config_path)
			
		except Exception as e:
			await self._log_error("Failed to create tap config", e)
			return None
	
	async def _discover_tap_catalog(self) -> Optional[Dict[str, Any]]:
		"""Discover Singer tap catalog (schema)"""
		try:
			config_path = Path(self.temp_dir) / f"{self.tap_name}_config.json"
			
			# Run tap discovery
			process = await asyncio.create_subprocess_exec(
				self.tap_name, '--config', str(config_path), '--discover',
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE
			)
			stdout, stderr = await process.communicate()
			
			if process.returncode != 0:
				await self._log_error(f"Tap discovery failed: {stderr.decode()}", None)
				return None
			
			# Parse catalog JSON
			catalog = json.loads(stdout.decode())
			await self._log_info(f"Discovered {len(catalog.get('streams', []))} streams")
			
			return catalog
			
		except Exception as e:
			await self._log_error("Catalog discovery failed", e)
			return None
	
	async def _create_catalog_file(self, catalog: Dict[str, Any]) -> Optional[str]:
		"""Create Singer catalog file with stream selections"""
		try:
			catalog_path = Path(self.temp_dir) / f"{self.tap_name}_catalog.json"
			
			# Mark all streams as selected by default
			for stream in catalog.get('streams', []):
				stream['metadata'] = stream.get('metadata', [])
				# Add selection metadata
				stream['metadata'].append({
					'breadcrumb': [],
					'metadata': {'selected': True, 'replication-method': 'FULL_TABLE'}
				})
			
			with open(catalog_path, 'w') as f:
				json.dump(catalog, f, indent=2)
			
			return str(catalog_path)
			
		except Exception as e:
			await self._log_error("Failed to create catalog file", e)
			return None
	
	async def _create_state_file(self) -> str:
		"""Create Singer state file for incremental extraction"""
		state_path = Path(self.temp_dir) / f"{self.tap_name}_state.json"
		
		# Initialize empty state
		initial_state = {'bookmarks': {}}
		
		with open(state_path, 'w') as f:
			json.dump(initial_state, f, indent=2)
		
		return str(state_path)
	
	async def _run_tap_extraction(self, stream_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
		"""Run Singer tap extraction for specific stream"""
		try:
			config_path = Path(self.temp_dir) / f"{self.tap_name}_config.json"
			catalog_path = self.catalog_file
			state_path = self.state_file
			output_path = Path(self.temp_dir) / f"extraction_{stream_name}.jsonl"
			
			start_time = datetime.now()
			
			# Run tap extraction
			with open(output_path, 'w') as output_file:
				process = await asyncio.create_subprocess_exec(
					self.tap_name,
					'--config', str(config_path),
					'--catalog', catalog_path,
					'--state', state_path,
					stdout=output_file,
					stderr=asyncio.subprocess.PIPE
				)
				
				_, stderr = await process.communicate()
			
			duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
			
			if process.returncode != 0:
				raise Exception(f"Tap extraction failed: {stderr.decode()}")
			
			# Parse extracted records
			records = await self._parse_extraction_output(output_path, stream_name)
			
			return {
				'record_count': len(records),
				'duration_ms': duration_ms,
				'records': records,
				'stream_name': stream_name
			}
			
		except Exception as e:
			await self._log_error(f"Tap extraction failed for stream {stream_name}", e)
			raise
	
	async def _parse_extraction_output(self, output_path: Path, stream_name: str) -> List[Dict[str, Any]]:
		"""Parse Singer tap output and extract records for specific stream"""
		records = []
		
		try:
			with open(output_path, 'r') as f:
				for line in f:
					if line.strip():
						message = json.loads(line.strip())
						
						# Singer messages have different types: SCHEMA, RECORD, STATE
						if (message.get('type') == 'RECORD' and 
							message.get('stream') == stream_name):
							records.append(message.get('record', {}))
			
			return records
			
		except Exception as e:
			await self._log_error(f"Failed to parse extraction output for {stream_name}", e)
			return []
	
	async def _convert_catalog_to_schema(self, catalog: Dict[str, Any]) -> Any:
		"""Convert Singer catalog to DVRL schema format"""
		from .models import DataSourceSchema
		
		tables = []
		for stream in catalog.get('streams', []):
			table_name = stream.get('stream', stream.get('table_name', 'unknown'))
			schema_props = stream.get('schema', {}).get('properties', {})
			
			columns = []
			for col_name, col_def in schema_props.items():
				columns.append({
					'name': col_name,
					'type': col_def.get('type', ['string']),
					'format': col_def.get('format'),
					'description': col_def.get('description'),
					'nullable': 'null' in col_def.get('type', [])
				})
			
			tables.append({
				'name': table_name,
				'type': 'stream',
				'columns': columns,
				'stream_metadata': stream.get('metadata', [])
			})
		
		return DataSourceSchema(
			schema_name=f"singer_{self.tap_name}",
			data_source_id=self.data_source.id,
			tables=tables,
			discovery_method='singer_tap_discovery',
			discovered_at=datetime.now(timezone.utc),
			confidence_score=0.95,  # High confidence for Singer taps
			tenant_id=self.data_source.tenant_id,
			created_by=self.data_source.created_by
		)
	
	def _extract_stream_from_query(self, query: str) -> Optional[str]:
		"""Extract stream name from query-like string"""
		# Simple parsing - in production would be more sophisticated
		if 'FROM' in query.upper():
			parts = query.upper().split('FROM')
			if len(parts) > 1:
				stream_part = parts[1].strip().split()[0]
				return stream_part.lower()
		
		# Fallback: treat entire query as stream name
		return query.strip().lower()
	
	async def _get_tap_version(self) -> str:
		"""Get Singer tap version"""
		try:
			process = await asyncio.create_subprocess_exec(
				self.tap_name, '--version',
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE
			)
			stdout, stderr = await process.communicate()
			
			if process.returncode == 0:
				return stdout.decode().strip()
			
			return 'unknown'
			
		except Exception:
			return 'unknown'
	
	async def cleanup(self):
		"""Clean up temporary files"""
		if self.temp_dir and Path(self.temp_dir).exists():
			import shutil
			shutil.rmtree(self.temp_dir, ignore_errors=True)


class SingerTapManager:
	"""Manager for Singer.io taps discovery and installation"""
	
	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.available_taps = {}
		self.installed_taps = {}
	
	async def initialize(self) -> bool:
		"""Initialize Singer tap manager"""
		try:
			await self._log_info("Initializing Singer Tap Manager")
			
			# Discover available taps
			await self._discover_available_taps()
			
			# Check installed taps
			await self._check_installed_taps()
			
			await self._log_info(f"Singer Tap Manager initialized - {len(self.installed_taps)} taps available")
			return True
			
		except Exception as e:
			await self._log_error("Failed to initialize Singer Tap Manager", e)
			return False
	
	async def get_available_taps(self) -> Dict[str, Any]:
		"""Get list of available Singer taps"""
		return {
			'available_taps': self.available_taps,
			'installed_taps': self.installed_taps,
			'total_available': len(self.available_taps),
			'total_installed': len(self.installed_taps)
		}
	
	async def install_tap(self, tap_name: str) -> bool:
		"""Install Singer tap using pip"""
		try:
			await self._log_info(f"Installing Singer tap: {tap_name}")
			
			# Real installation using subprocess
			process = await asyncio.create_subprocess_exec(
				'pip', 'install', tap_name,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE
			)
			
			stdout, stderr = await process.communicate()
			
			if process.returncode == 0:
				# Get installed version
				version = await self._get_tap_version_real(tap_name)
				
				self.installed_taps[tap_name] = {
					'installed_at': datetime.now(timezone.utc).isoformat(),
					'version': version,
					'status': 'installed',
					'install_output': stdout.decode('utf-8')
				}
				
				await self._log_info(f"Singer tap installed successfully: {tap_name} v{version}")
				return True
			else:
				error_msg = stderr.decode('utf-8')
				await self._log_error(f"Failed to install {tap_name}: {error_msg}", Exception(error_msg))
				return False
			
		except Exception as e:
			await self._log_error(f"Failed to install Singer tap: {tap_name}", e)
			return False
	
	async def _get_tap_version_real(self, tap_name: str) -> str:
		"""Get real installed version of Singer tap"""
		try:
			process = await asyncio.create_subprocess_exec(
				'pip', 'show', tap_name,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE
			)
			
			stdout, stderr = await process.communicate()
			
			if process.returncode == 0:
				output = stdout.decode('utf-8')
				for line in output.split('\n'):
					if line.startswith('Version:'):
						return line.split('Version:')[1].strip()
			
			return 'unknown'
			
		except Exception:
			return 'unknown'
	
	async def create_tap_connector(self, tap_name: str, tap_config: Dict[str, Any]) -> Optional[SingerTapConnector]:
		"""Create Singer tap connector instance"""
		try:
			if tap_name not in self.installed_taps:
				await self._log_error(f"Tap not installed: {tap_name}", None)
				return None
			
			# Create mock data source for tap
			data_source = DataSource(
				id=uuid7str(),
				name=f"singer_{tap_name}",
				type=DataSourceType.API,  # Most Singer taps are API-based
				connection_config={
					'tap_name': tap_name,
					'tap_config': tap_config
				},
				status=DataSourceStatus.ACTIVE,
				tenant_id=self.tenant_id,
				created_by=self.user_id
			)
			
			connector = SingerTapConnector(data_source)
			
			if await connector.initialize():
				return connector
			else:
				return None
				
		except Exception as e:
			await self._log_error(f"Failed to create tap connector: {tap_name}", e)
			return None
	
	async def _discover_available_taps(self):
		"""Discover available Singer taps from Meltano Hub"""
		try:
			# Real discovery from Meltano Hub API
			import httpx
			
			async with httpx.AsyncClient() as client:
				response = await client.get(
					'https://hub.meltano.com/api/v1/plugins/extractors',
					timeout=30
				)
				
				if response.status_code == 200:
					hub_data = response.json()
					
					# Parse taps from Meltano Hub
					discovered_taps = {}
					for tap_info in hub_data.get('plugins', [])[:50]:  # Limit to first 50
						tap_name = tap_info.get('name', '')
						if tap_name.startswith('tap-'):
							discovered_taps[tap_name] = {
								'description': tap_info.get('description', ''),
								'category': tap_info.get('category', 'unknown'),
								'config_requirements': tap_info.get('settings', []),
								'documentation': tap_info.get('docs', ''),
								'repo_url': tap_info.get('repo', ''),
								'pip_url': tap_info.get('pip_url', tap_name)
							}
					
					self.available_taps.update(discovered_taps)
					await self._log_info(f"Discovered {len(discovered_taps)} Singer taps from Meltano Hub")
					
				else:
					await self._log_error(f"Failed to fetch from Meltano Hub: {response.status_code}")
					
		except Exception as e:
			await self._log_error("Failed to discover taps from Meltano Hub", e)
			
		# Fallback to predefined taps if discovery fails
		if not self.available_taps:
			self.available_taps = {
			'tap-postgres': {
				'description': 'PostgreSQL database tap',
				'category': 'database',
				'config_requirements': ['host', 'port', 'user', 'password', 'dbname'],
				'documentation': 'https://hub.meltano.com/extractors/tap-postgres'
			},
			'tap-mysql': {
				'description': 'MySQL database tap',
				'category': 'database', 
				'config_requirements': ['host', 'port', 'user', 'password', 'database'],
				'documentation': 'https://hub.meltano.com/extractors/tap-mysql'
			},
			'tap-salesforce': {
				'description': 'Salesforce CRM tap',
				'category': 'crm',
				'config_requirements': ['username', 'password', 'security_token'],
				'documentation': 'https://hub.meltano.com/extractors/tap-salesforce'
			},
			'tap-stripe': {
				'description': 'Stripe payments tap',
				'category': 'payments',
				'config_requirements': ['account_id', 'client_secret'],
				'documentation': 'https://hub.meltano.com/extractors/tap-stripe'
			},
			'tap-github': {
				'description': 'GitHub repositories tap',
				'category': 'development',
				'config_requirements': ['access_token', 'repository'],
				'documentation': 'https://hub.meltano.com/extractors/tap-github'
			}
		}
	
	async def _check_installed_taps(self):
		"""Check which Singer taps are actually installed"""
		try:
			# Get list of installed packages using pip list
			process = await asyncio.create_subprocess_exec(
				'pip', 'list', '--format=json',
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE
			)
			
			stdout, stderr = await process.communicate()
			
			if process.returncode == 0:
				import json
				installed_packages = json.loads(stdout.decode('utf-8'))
				
				# Find Singer taps among installed packages
				for package in installed_packages:
					package_name = package.get('name', '').lower()
					if package_name.startswith('tap-'):
						self.installed_taps[package_name] = {
							'installed_at': 'unknown',  # pip list doesn't provide install date
							'version': package.get('version', 'unknown'),
							'status': 'available'
						}
				
				await self._log_info(f"Found {len(self.installed_taps)} installed Singer taps")
				
			else:
				await self._log_error("Failed to check installed packages", Exception(stderr.decode('utf-8')))
				
		except Exception as e:
			await self._log_error("Failed to check installed taps", e)
	
	async def _log_info(self, message: str):
		print(f"[{datetime.now(timezone.utc).isoformat()}] SINGER INFO: {message}")
	
	async def _log_error(self, message: str, error: Exception | None):
		error_str = str(error) if error else 'No error details'
		print(f"[{datetime.now(timezone.utc).isoformat()}] SINGER ERROR: {message} | {error_str}")


# Export Singer integration components
__all__ = [
	"SingerTapConnector",
	"SingerTapManager"
]