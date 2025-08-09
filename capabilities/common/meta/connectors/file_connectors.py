#!/usr/bin/env python3
"""
APG Metadata Management - File Connectors
Connectors for discovering metadata from file-based data sources

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import os
import csv
import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import fastavro
import boto3
from google.cloud import storage
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiofiles
import asyncio
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud.exceptions import NotFound
import logging
import glob
from io import StringIO

from .base_connector import (
	BaseConnector, ConnectorConfig, DiscoveryResult, AssetMetadata,
	ColumnMetadata, ConnectorType, DataType, should_include_asset
)


class CSVConnector(BaseConnector):
	"""CSV file metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "csv"
		self.csv_files = []
		self.base_path = None
	
	async def connect(self) -> bool:
		"""Establish connection to CSV file directory"""
		try:
			# Extract base path from connection string
			self.base_path = self.config.connection_string
			
			if not os.path.exists(self.base_path):
				await self._log_error(f"Path does not exist: {self.base_path}")
				return False
			
			if not os.path.isdir(self.base_path):
				await self._log_error(f"Path is not a directory: {self.base_path}")
				return False
			
			# Discover CSV files
			csv_pattern = os.path.join(self.base_path, "**", "*.csv")
			self.csv_files = glob.glob(csv_pattern, recursive=True)
			
			self.is_connected = True
			await self._log_info(f"Connected to CSV directory: {self.base_path}, found {len(self.csv_files)} files")
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to connect to CSV directory: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection"""
		self.is_connected = False
		self.csv_files = []
		await self._log_info("Disconnected from CSV directory")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to CSV directory"""
		try:
			if await self.connect():
				return {
					"status": "success",
					"message": f"Successfully connected to {self.base_path}",
					"files_found": len(self.csv_files)
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to {self.base_path}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"Connection test failed: {str(e)}"
			}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all CSV files and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish connection")
					result.complete_discovery()
					return result
			
			for csv_file in self.csv_files:
				try:
					# Check if file should be included
					file_name = os.path.basename(csv_file)
					if not should_include_asset(file_name, self.config.include_patterns, self.config.exclude_patterns):
						continue
					
					# Get file stats
					stat = os.stat(csv_file)
					
					# Try to read first few rows to get column info
					async with aiofiles.open(csv_file, 'r', encoding='utf-8') as f:
						content = await f.read(8192)  # Read first 8KB
						
					# Parse CSV header
					sniffer = csv.Sniffer()
					delimiter = sniffer.sniff(content[:1024]).delimiter
					
					reader = csv.reader(StringIO(content), delimiter=delimiter)
					header = next(reader)
					
					# Count approximate rows
					row_count = content.count('\n') - 1  # Subtract header
					if row_count < 0:
						row_count = 0
					
					# Create asset metadata
					asset = AssetMetadata(
						name=file_name,
						asset_type="csv_file",
						source_system=self.source_system,
						full_name=csv_file,
						description=f"CSV file with {len(header)} columns",
						column_count=len(header),
						row_count=row_count,
						size_bytes=stat.st_size,
						created_at=datetime.fromtimestamp(stat.st_ctime),
						modified_at=datetime.fromtimestamp(stat.st_mtime),
						location=csv_file,
						properties={
							"delimiter": delimiter,
							"encoding": "utf-8",
							"file_extension": ".csv"
						}
					)
					
					# Add basic column metadata
					for col_name in header:
						column = ColumnMetadata(
							name=col_name.strip(),
							data_type=DataType.STRING,  # Will be inferred during profiling
							is_nullable=True
						)
						asset.columns.append(column)
					
					# Estimate quality score
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					
					result.add_asset(asset)
					
				except Exception as e:
					result.add_error(f"Failed to process file {csv_file}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific CSV file"""
		try:
			# Find the CSV file
			csv_file = None
			for file_path in self.csv_files:
				if os.path.basename(file_path) == asset_name:
					csv_file = file_path
					break
			
			if not csv_file:
				await self._log_error(f"CSV file not found: {asset_name}")
				return None
			
			# Read file and analyze schema
			df = pd.read_csv(csv_file, nrows=self.config.max_sample_rows)
			
			# Get file stats
			stat = os.stat(csv_file)
			
			# Create asset metadata
			asset = AssetMetadata(
				name=asset_name,
				asset_type="csv_file",
				source_system=self.source_system,
				full_name=csv_file,
				description=f"CSV file with {len(df.columns)} columns and {len(df)} rows",
				column_count=len(df.columns),
				row_count=len(df),
				size_bytes=stat.st_size,
				created_at=datetime.fromtimestamp(stat.st_ctime),
				modified_at=datetime.fromtimestamp(stat.st_mtime),
				location=csv_file
			)
			
			# Profile each column
			for col_name in df.columns:
				column_data = df[col_name].tolist()
				column_metadata = await self.profile_column(asset_name, col_name, column_data)
				asset.columns.append(column_metadata)
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from CSV file"""
		try:
			# Find the CSV file
			csv_file = None
			for file_path in self.csv_files:
				if os.path.basename(file_path) == asset_name:
					csv_file = file_path
					break
			
			if not csv_file:
				await self._log_error(f"CSV file not found: {asset_name}")
				return []
			
			# Read limited rows
			df = pd.read_csv(csv_file, nrows=min(limit, self.config.max_sample_rows))
			
			# Convert to list of dictionaries
			return df.to_dict('records')
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []


class JSONConnector(BaseConnector):
	"""JSON file metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "json"
		self.json_files = []
		self.base_path = None
	
	async def connect(self) -> bool:
		"""Establish connection to JSON file directory"""
		try:
			# Extract base path from connection string
			self.base_path = self.config.connection_string
			
			if not os.path.exists(self.base_path):
				await self._log_error(f"Path does not exist: {self.base_path}")
				return False
			
			if not os.path.isdir(self.base_path):
				await self._log_error(f"Path is not a directory: {self.base_path}")
				return False
			
			# Discover JSON files
			json_pattern = os.path.join(self.base_path, "**", "*.json")
			self.json_files = glob.glob(json_pattern, recursive=True)
			
			self.is_connected = True
			await self._log_info(f"Connected to JSON directory: {self.base_path}, found {len(self.json_files)} files")
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to connect to JSON directory: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection"""
		self.is_connected = False
		self.json_files = []
		await self._log_info("Disconnected from JSON directory")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to JSON directory"""
		try:
			if await self.connect():
				return {
					"status": "success",
					"message": f"Successfully connected to {self.base_path}",
					"files_found": len(self.json_files)
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to {self.base_path}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"Connection test failed: {str(e)}"
			}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all JSON files and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish connection")
					result.complete_discovery()
					return result
			
			for json_file in self.json_files:
				try:
					# Check if file should be included
					file_name = os.path.basename(json_file)
					if not should_include_asset(file_name, self.config.include_patterns, self.config.exclude_patterns):
						continue
					
					# Get file stats
					stat = os.stat(json_file)
					
					# Try to read and parse JSON to get structure info
					async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
						content = await f.read()
					
					data = json.loads(content)
					
					# Analyze JSON structure
					if isinstance(data, list):
						# Array of objects
						if data and isinstance(data[0], dict):
							columns = list(data[0].keys())
							row_count = len(data)
							description = f"JSON array with {len(columns)} fields and {row_count} records"
						else:
							columns = []
							row_count = len(data)
							description = f"JSON array with {row_count} items"
					elif isinstance(data, dict):
						# Single object
						columns = list(data.keys())
						row_count = 1
						description = f"JSON object with {len(columns)} fields"
					else:
						# Primitive value
						columns = []
						row_count = 1
						description = "JSON file with primitive value"
					
					# Create asset metadata
					asset = AssetMetadata(
						name=file_name,
						asset_type="json_file",
						source_system=self.source_system,
						full_name=json_file,
						description=description,
						column_count=len(columns),
						row_count=row_count,
						size_bytes=stat.st_size,
						created_at=datetime.fromtimestamp(stat.st_ctime),
						modified_at=datetime.fromtimestamp(stat.st_mtime),
						location=json_file,
						properties={
							"encoding": "utf-8",
							"file_extension": ".json",
							"json_type": "array" if isinstance(data, list) else "object" if isinstance(data, dict) else "primitive"
						}
					)
					
					# Add basic column metadata
					for col_name in columns:
						column = ColumnMetadata(
							name=col_name,
							data_type=DataType.JSON,  # Will be inferred during profiling
							is_nullable=True
						)
						asset.columns.append(column)
					
					# Estimate quality score
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					
					result.add_asset(asset)
					
				except json.JSONDecodeError as e:
					result.add_error(f"Invalid JSON file {json_file}: {str(e)}")
				except Exception as e:
					result.add_error(f"Failed to process file {json_file}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific JSON file"""
		try:
			# Find the JSON file
			json_file = None
			for file_path in self.json_files:
				if os.path.basename(file_path) == asset_name:
					json_file = file_path
					break
			
			if not json_file:
				await self._log_error(f"JSON file not found: {asset_name}")
				return None
			
			# Read and parse JSON
			async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
				content = await f.read()
			
			data = json.loads(content)
			
			# Get file stats
			stat = os.stat(json_file)
			
			# Create asset metadata
			asset = AssetMetadata(
				name=asset_name,
				asset_type="json_file",
				source_system=self.source_system,
				full_name=json_file,
				size_bytes=stat.st_size,
				created_at=datetime.fromtimestamp(stat.st_ctime),
				modified_at=datetime.fromtimestamp(stat.st_mtime),
				location=json_file
			)
			
			# Analyze structure and create column metadata
			if isinstance(data, list) and data:
				asset.row_count = len(data)
				
				# Get sample for profiling (limit to max_sample_rows)
				sample_data = data[:min(len(data), self.config.max_sample_rows)]
				
				if isinstance(sample_data[0], dict):
					# Extract all possible fields from sample
					all_fields = set()
					for item in sample_data:
						if isinstance(item, dict):
							all_fields.update(item.keys())
					
					asset.description = f"JSON array with {len(all_fields)} fields and {len(data)} records"
					asset.column_count = len(all_fields)
					
					# Profile each field
					for field_name in sorted(all_fields):
						field_values = []
						for item in sample_data:
							if isinstance(item, dict):
								field_values.append(item.get(field_name))
						
						column_metadata = await self.profile_column(asset_name, field_name, field_values)
						asset.columns.append(column_metadata)
				else:
					asset.description = f"JSON array with {len(data)} primitive values"
					asset.column_count = 1
					
					# Profile the values as a single column
					column_metadata = await self.profile_column(asset_name, "value", sample_data)
					asset.columns.append(column_metadata)
			
			elif isinstance(data, dict):
				asset.row_count = 1
				asset.column_count = len(data)
				asset.description = f"JSON object with {len(data)} fields"
				
				# Profile each field
				for field_name, field_value in data.items():
					column_metadata = await self.profile_column(asset_name, field_name, [field_value])
					asset.columns.append(column_metadata)
			
			else:
				# Primitive value
				asset.row_count = 1
				asset.column_count = 1
				asset.description = "JSON file with primitive value"
				
				column_metadata = await self.profile_column(asset_name, "value", [data])
				asset.columns.append(column_metadata)
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except json.JSONDecodeError as e:
			await self._log_error(f"Invalid JSON file {asset_name}: {str(e)}")
			return None
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from JSON file"""
		try:
			# Find the JSON file
			json_file = None
			for file_path in self.json_files:
				if os.path.basename(file_path) == asset_name:
					json_file = file_path
					break
			
			if not json_file:
				await self._log_error(f"JSON file not found: {asset_name}")
				return []
			
			# Read and parse JSON
			async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
				content = await f.read()
			
			data = json.loads(content)
			
			# Convert to list of dictionaries format
			if isinstance(data, list):
				sample_limit = min(limit, len(data), self.config.max_sample_rows)
				if data and isinstance(data[0], dict):
					return data[:sample_limit]
				else:
					# Convert primitive values to dict format
					return [{"value": item} for item in data[:sample_limit]]
			
			elif isinstance(data, dict):
				return [data]
			
			else:
				# Primitive value
				return [{"value": data}]
			
		except json.JSONDecodeError as e:
			await self._log_error(f"Invalid JSON file {asset_name}: {str(e)}")
			return []
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []


class ParquetConnector(BaseConnector):
	"""Parquet file metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "parquet"
		self.parquet_files = []
		self.base_path = None
	
	async def connect(self) -> bool:
		"""Establish connection to Parquet file directory"""
		try:
			# Extract base path from connection string
			self.base_path = self.config.connection_string
			
			if not os.path.exists(self.base_path):
				await self._log_error(f"Path does not exist: {self.base_path}")
				return False
			
			if not os.path.isdir(self.base_path):
				await self._log_error(f"Path is not a directory: {self.base_path}")
				return False
			
			# Discover Parquet files
			parquet_pattern = os.path.join(self.base_path, "**", "*.parquet")
			self.parquet_files = glob.glob(parquet_pattern, recursive=True)
			
			self.is_connected = True
			await self._log_info(f"Connected to Parquet directory: {self.base_path}, found {len(self.parquet_files)} files")
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to connect to Parquet directory: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection"""
		self.is_connected = False
		self.parquet_files = []
		await self._log_info("Disconnected from Parquet directory")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to Parquet directory"""
		try:
			if await self.connect():
				return {
					"status": "success",
					"message": f"Successfully connected to {self.base_path}",
					"files_found": len(self.parquet_files)
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to {self.base_path}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"Connection test failed: {str(e)}"
			}
	
	def _map_arrow_type_to_datatype(self, arrow_type: pa.DataType) -> DataType:
		"""Map PyArrow data type to our DataType enum"""
		if pa.types.is_integer(arrow_type):
			return DataType.INTEGER
		elif pa.types.is_floating(arrow_type):
			return DataType.FLOAT
		elif pa.types.is_boolean(arrow_type):
			return DataType.BOOLEAN
		elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
			return DataType.STRING
		elif pa.types.is_date(arrow_type):
			return DataType.DATE
		elif pa.types.is_timestamp(arrow_type):
			return DataType.TIMESTAMP
		elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
			return DataType.BINARY
		elif pa.types.is_list(arrow_type):
			return DataType.ARRAY
		elif pa.types.is_struct(arrow_type):
			return DataType.OBJECT
		else:
			return DataType.UNKNOWN
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all Parquet files and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish connection")
					result.complete_discovery()
					return result
			
			for parquet_file in self.parquet_files:
				try:
					# Check if file should be included
					file_name = os.path.basename(parquet_file)
					if not should_include_asset(file_name, self.config.include_patterns, self.config.exclude_patterns):
						continue
					
					# Get file stats
					stat = os.stat(parquet_file)
					
					# Read Parquet metadata without loading data
					parquet_table = pq.ParquetFile(parquet_file)
					schema = parquet_table.schema
					metadata = parquet_table.metadata
					
					# Get row count and other metrics
					row_count = metadata.num_rows
					column_count = len(schema)
					
					# Create asset metadata
					asset = AssetMetadata(
						name=file_name,
						asset_type="parquet_file",
						source_system=self.source_system,
						full_name=parquet_file,
						description=f"Parquet file with {column_count} columns and {row_count} rows",
						column_count=column_count,
						row_count=row_count,
						size_bytes=stat.st_size,
						created_at=datetime.fromtimestamp(stat.st_ctime),
						modified_at=datetime.fromtimestamp(stat.st_mtime),
						location=parquet_file,
						properties={
							"file_extension": ".parquet",
							"num_row_groups": metadata.num_row_groups,
							"parquet_version": metadata.version,
							"compression": str(metadata.row_group(0).column(0).compression) if metadata.num_row_groups > 0 else "unknown"
						}
					)
					
					# Add column metadata from schema
					for i, field in enumerate(schema):
						column = ColumnMetadata(
							name=field.name,
							data_type=self._map_arrow_type_to_datatype(field.type),
							is_nullable=field.nullable,
							description=f"Column {i+1} of {column_count}"
						)
						asset.columns.append(column)
					
					# Estimate quality score
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					
					result.add_asset(asset)
					
				except Exception as e:
					result.add_error(f"Failed to process file {parquet_file}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific Parquet file"""
		try:
			# Find the Parquet file
			parquet_file = None
			for file_path in self.parquet_files:
				if os.path.basename(file_path) == asset_name:
					parquet_file = file_path
					break
			
			if not parquet_file:
				await self._log_error(f"Parquet file not found: {asset_name}")
				return None
			
			# Read Parquet file metadata and sample data
			df = pd.read_parquet(parquet_file, engine='pyarrow')
			sample_size = min(len(df), self.config.max_sample_rows)
			sample_df = df.head(sample_size)
			
			# Get file stats
			stat = os.stat(parquet_file)
			
			# Get detailed metadata
			parquet_table = pq.ParquetFile(parquet_file)
			metadata = parquet_table.metadata
			
			# Create asset metadata
			asset = AssetMetadata(
				name=asset_name,
				asset_type="parquet_file",
				source_system=self.source_system,
				full_name=parquet_file,
				description=f"Parquet file with {len(df.columns)} columns and {len(df)} rows",
				column_count=len(df.columns),
				row_count=len(df),
				size_bytes=stat.st_size,
				created_at=datetime.fromtimestamp(stat.st_ctime),
				modified_at=datetime.fromtimestamp(stat.st_mtime),
				location=parquet_file,
				properties={
					"file_extension": ".parquet",
					"num_row_groups": metadata.num_row_groups,
					"parquet_version": metadata.version,
					"compression": str(metadata.row_group(0).column(0).compression) if metadata.num_row_groups > 0 else "unknown"
				}
			)
			
			# Profile each column
			for col_name in df.columns:
				column_data = sample_df[col_name].tolist()
				column_metadata = await self.profile_column(asset_name, col_name, column_data)
				asset.columns.append(column_metadata)
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from Parquet file"""
		try:
			# Find the Parquet file
			parquet_file = None
			for file_path in self.parquet_files:
				if os.path.basename(file_path) == asset_name:
					parquet_file = file_path
					break
			
			if not parquet_file:
				await self._log_error(f"Parquet file not found: {asset_name}")
				return []
			
			# Read limited rows
			df = pd.read_parquet(parquet_file, engine='pyarrow')
			sample_limit = min(limit, len(df), self.config.max_sample_rows)
			sample_df = df.head(sample_limit)
			
			# Convert to list of dictionaries
			return sample_df.to_dict('records')
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []


class AvroConnector(BaseConnector):
	"""Avro file metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "avro"
		self.avro_files = []
		self.base_path = None
	
	async def connect(self) -> bool:
		"""Establish connection to Avro file directory"""
		try:
			# Extract base path from connection string
			self.base_path = self.config.connection_string
			
			if not os.path.exists(self.base_path):
				await self._log_error(f"Path does not exist: {self.base_path}")
				return False
			
			if not os.path.isdir(self.base_path):
				await self._log_error(f"Path is not a directory: {self.base_path}")
				return False
			
			# Discover Avro files
			avro_pattern = os.path.join(self.base_path, "**", "*.avro")
			self.avro_files = glob.glob(avro_pattern, recursive=True)
			
			self.is_connected = True
			await self._log_info(f"Connected to Avro directory: {self.base_path}, found {len(self.avro_files)} files")
			return True
			
		except Exception as e:
			await self._log_error(f"Failed to connect to Avro directory: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close connection"""
		self.is_connected = False
		self.avro_files = []
		await self._log_info("Disconnected from Avro directory")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to Avro directory"""
		try:
			if await self.connect():
				return {
					"status": "success",
					"message": f"Successfully connected to {self.base_path}",
					"files_found": len(self.avro_files)
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to {self.base_path}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"Connection test failed: {str(e)}"
			}
	
	def _map_avro_type_to_datatype(self, avro_type) -> DataType:
		"""Map Avro data type to our DataType enum"""
		if isinstance(avro_type, str):
			type_name = avro_type
		elif isinstance(avro_type, dict):
			type_name = avro_type.get('type', 'unknown')
		else:
			type_name = str(avro_type)
		
		type_mapping = {
			'null': DataType.UNKNOWN,
			'boolean': DataType.BOOLEAN,
			'int': DataType.INTEGER,
			'long': DataType.INTEGER,
			'float': DataType.FLOAT,
			'double': DataType.FLOAT,
			'string': DataType.STRING,
			'bytes': DataType.BINARY,
			'array': DataType.ARRAY,
			'map': DataType.OBJECT,
			'record': DataType.OBJECT,
			'enum': DataType.STRING,
			'fixed': DataType.BINARY,
			'union': DataType.UNKNOWN
		}
		
		return type_mapping.get(type_name, DataType.UNKNOWN)
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all Avro files and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish connection")
					result.complete_discovery()
					return result
			
			for avro_file in self.avro_files:
				try:
					# Check if file should be included
					file_name = os.path.basename(avro_file)
					if not should_include_asset(file_name, self.config.include_patterns, self.config.exclude_patterns):
						continue
					
					# Get file stats
					stat = os.stat(avro_file)
					
					# Read Avro schema
					with open(avro_file, 'rb') as f:
						avro_reader = fastavro.reader(f)
						schema = avro_reader.writer_schema
						
						# Count records by iterating through file
						record_count = 0
						for _ in avro_reader:
							record_count += 1
							if record_count >= 10000:  # Limit counting for large files
								break
					
					# Extract fields from schema
					fields = schema.get('fields', []) if isinstance(schema, dict) else []
					column_count = len(fields)
					
					# Create asset metadata
					asset = AssetMetadata(
						name=file_name,
						asset_type="avro_file",
						source_system=self.source_system,
						full_name=avro_file,
						description=f"Avro file with {column_count} fields and {record_count}+ records",
						column_count=column_count,
						row_count=record_count,
						size_bytes=stat.st_size,
						created_at=datetime.fromtimestamp(stat.st_ctime),
						modified_at=datetime.fromtimestamp(stat.st_mtime),
						location=avro_file,
						properties={
							"file_extension": ".avro",
							"schema_name": schema.get('name', 'unknown') if isinstance(schema, dict) else 'unknown',
							"schema_namespace": schema.get('namespace') if isinstance(schema, dict) else None
						}
					)
					
					# Add column metadata from schema
					for field in fields:
						field_name = field.get('name', 'unknown')
						field_type = field.get('type', 'unknown')
						
						column = ColumnMetadata(
							name=field_name,
							data_type=self._map_avro_type_to_datatype(field_type),
							is_nullable=isinstance(field_type, list) and 'null' in field_type,
							description=field.get('doc', f"Avro field: {field_name}")
						)
						asset.columns.append(column)
					
					# Estimate quality score
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					
					result.add_asset(asset)
					
				except Exception as e:
					result.add_error(f"Failed to process file {avro_file}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"Discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific Avro file"""
		try:
			# Find the Avro file
			avro_file = None
			for file_path in self.avro_files:
				if os.path.basename(file_path) == asset_name:
					avro_file = file_path
					break
			
			if not avro_file:
				await self._log_error(f"Avro file not found: {asset_name}")
				return None
			
			# Read Avro file and extract sample data for profiling
			records = []
			with open(avro_file, 'rb') as f:
				avro_reader = fastavro.reader(f)
				schema = avro_reader.writer_schema
				
				# Collect sample records
				for i, record in enumerate(avro_reader):
					records.append(record)
					if i >= self.config.max_sample_rows - 1:
						break
			
			# Get file stats
			stat = os.stat(avro_file)
			
			# Create asset metadata
			fields = schema.get('fields', []) if isinstance(schema, dict) else []
			asset = AssetMetadata(
				name=asset_name,
				asset_type="avro_file",
				source_system=self.source_system,
				full_name=avro_file,
				description=f"Avro file with {len(fields)} fields and {len(records)} sampled records",
				column_count=len(fields),
				row_count=len(records),
				size_bytes=stat.st_size,
				created_at=datetime.fromtimestamp(stat.st_ctime),
				modified_at=datetime.fromtimestamp(stat.st_mtime),
				location=avro_file,
				properties={
					"file_extension": ".avro",
					"schema_name": schema.get('name', 'unknown') if isinstance(schema, dict) else 'unknown',
					"schema_namespace": schema.get('namespace') if isinstance(schema, dict) else None
				}
			)
			
			# Profile each field
			for field in fields:
				field_name = field.get('name', 'unknown')
				
				# Extract values for this field from sample records
				field_values = [record.get(field_name) for record in records if field_name in record]
				
				column_metadata = await self.profile_column(asset_name, field_name, field_values)
				
				# Override with Avro-specific info
				field_type = field.get('type', 'unknown')
				column_metadata.data_type = self._map_avro_type_to_datatype(field_type)
				column_metadata.is_nullable = isinstance(field_type, list) and 'null' in field_type
				column_metadata.description = field.get('doc', f"Avro field: {field_name}")
				
				asset.columns.append(column_metadata)
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from Avro file"""
		try:
			# Find the Avro file
			avro_file = None
			for file_path in self.avro_files:
				if os.path.basename(file_path) == asset_name:
					avro_file = file_path
					break
			
			if not avro_file:
				await self._log_error(f"Avro file not found: {asset_name}")
				return []
			
			# Read limited records
			records = []
			sample_limit = min(limit, self.config.max_sample_rows)
			
			with open(avro_file, 'rb') as f:
				avro_reader = fastavro.reader(f)
				
				for i, record in enumerate(avro_reader):
					records.append(record)
					if i >= sample_limit - 1:
						break
			
			return records
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from {asset_name}: {str(e)}")
			return []


class S3Connector(BaseConnector):
	"""AWS S3 metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "s3"
		self.s3_client = None
		self.bucket_name = None
		self.prefix = ""
		self.s3_objects = []
	
	async def connect(self) -> bool:
		"""Establish connection to S3"""
		try:
			# Parse connection string (format: s3://bucket-name/prefix or s3://bucket-name)
			connection_parts = self.config.connection_string.replace('s3://', '').split('/', 1)
			self.bucket_name = connection_parts[0]
			self.prefix = connection_parts[1] if len(connection_parts) > 1 else ""
			
			# Initialize S3 client
			self.s3_client = boto3.client(
				's3',
				aws_access_key_id=self.config.username,
				aws_secret_access_key=self.config.password,
				region_name=self.config.additional_params.get('region', 'us-east-1')
			)
			
			# Test connection by listing objects
			response = self.s3_client.list_objects_v2(
				Bucket=self.bucket_name,
				Prefix=self.prefix,
				MaxKeys=1
			)
			
			self.is_connected = True
			await self._log_info(f"Connected to S3 bucket: {self.bucket_name}, prefix: {self.prefix}")
			return True
			
		except NoCredentialsError:
			await self._log_error("AWS credentials not found")
			return False
		except ClientError as e:
			await self._log_error(f"Failed to connect to S3: {str(e)}")
			return False
		except Exception as e:
			await self._log_error(f"Failed to connect to S3: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close S3 connection"""
		self.is_connected = False
		self.s3_client = None
		self.s3_objects = []
		await self._log_info("Disconnected from S3")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to S3"""
		try:
			if await self.connect():
				# Count objects in bucket
				response = self.s3_client.list_objects_v2(
					Bucket=self.bucket_name,
					Prefix=self.prefix
				)
				object_count = len(response.get('Contents', []))
				
				return {
					"status": "success",
					"message": f"Successfully connected to S3 bucket: {self.bucket_name}",
					"bucket": self.bucket_name,
					"prefix": self.prefix,
					"objects_found": object_count
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to S3 bucket: {self.bucket_name}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"S3 connection test failed: {str(e)}"
			}
	
	def _get_file_extension(self, object_key: str) -> str:
		"""Extract file extension from S3 object key"""
		return os.path.splitext(object_key)[1].lower()
	
	def _infer_content_type_from_extension(self, extension: str) -> str:
		"""Infer content type from file extension"""
		extension_mapping = {
			'.csv': 'text/csv',
			'.json': 'application/json',
			'.parquet': 'application/octet-stream',
			'.avro': 'application/octet-stream',
			'.txt': 'text/plain',
			'.xml': 'application/xml',
			'.pdf': 'application/pdf',
			'.zip': 'application/zip',
			'.gz': 'application/gzip'
		}
		return extension_mapping.get(extension, 'application/octet-stream')
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all S3 objects and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish S3 connection")
					result.complete_discovery()
					return result
			
			# List all objects in bucket with prefix
			paginator = self.s3_client.get_paginator('list_objects_v2')
			page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
			
			for page in page_iterator:
				objects = page.get('Contents', [])
				
				for obj in objects:
					try:
						object_key = obj['Key']
						object_name = os.path.basename(object_key)
						
						# Skip directories (keys ending with /)
						if object_key.endswith('/'):
							continue
						
						# Check if object should be included
						if not should_include_asset(object_name, self.config.include_patterns, self.config.exclude_patterns):
							continue
						
						# Get object metadata
						object_size = obj['Size']
						last_modified = obj['LastModified']
						
						# Get additional metadata
						try:
							head_response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
							content_type = head_response.get('ContentType', 'application/octet-stream')
							metadata = head_response.get('Metadata', {})
						except ClientError:
							content_type = self._infer_content_type_from_extension(self._get_file_extension(object_key))
							metadata = {}
						
						# Determine asset type based on file extension
						file_extension = self._get_file_extension(object_key)
						asset_type_mapping = {
							'.csv': 'csv_file',
							'.json': 'json_file',
							'.parquet': 'parquet_file',
							'.avro': 'avro_file',
							'.txt': 'text_file',
							'.xml': 'xml_file'
						}
						asset_type = asset_type_mapping.get(file_extension, 'file')
						
						# Create asset metadata
						asset = AssetMetadata(
							name=object_name,
							asset_type=asset_type,
							source_system=self.source_system,
							full_name=f"s3://{self.bucket_name}/{object_key}",
							description=f"S3 object: {object_key}",
							size_bytes=object_size,
							modified_at=last_modified,
							location=f"s3://{self.bucket_name}/{object_key}",
							properties={
								"bucket": self.bucket_name,
								"key": object_key,
								"content_type": content_type,
								"file_extension": file_extension,
								"s3_metadata": metadata
							}
						)
						
						# For structured files, add basic column info
						if file_extension in ['.csv', '.json', '.parquet', '.avro']:
							# We can't easily determine columns without downloading the file
							# So we'll add a placeholder that gets filled during get_asset_schema
							asset.column_count = 0
							asset.description += " (columns will be determined on detailed inspection)"
						
						# Estimate quality score
						asset.estimated_quality_score = self._estimate_quality_score(asset)
						
						result.add_asset(asset)
						
					except Exception as e:
						result.add_error(f"Failed to process S3 object {obj.get('Key', 'unknown')}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"S3 discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific S3 object"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to S3")
				return None
			
			# Find the S3 object
			object_key = None
			paginator = self.s3_client.get_paginator('list_objects_v2')
			page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
			
			for page in page_iterator:
				for obj in page.get('Contents', []):
					if os.path.basename(obj['Key']) == asset_name:
						object_key = obj['Key']
						break
				if object_key:
					break
			
			if not object_key:
				await self._log_error(f"S3 object not found: {asset_name}")
				return None
			
			# Get object metadata
			head_response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
			object_size = head_response['ContentLength']
			last_modified = head_response['LastModified']
			content_type = head_response.get('ContentType', 'application/octet-stream')
			
			# For structured files, try to download and analyze a sample
			file_extension = self._get_file_extension(object_key)
			
			asset = AssetMetadata(
				name=asset_name,
				asset_type=f"{file_extension[1:]}_file" if file_extension else "file",
				source_system=self.source_system,
				full_name=f"s3://{self.bucket_name}/{object_key}",
				description=f"S3 object: {object_key}",
				size_bytes=object_size,
				modified_at=last_modified,
				location=f"s3://{self.bucket_name}/{object_key}",
				properties={
					"bucket": self.bucket_name,
					"key": object_key,
					"content_type": content_type,
					"file_extension": file_extension
				}
			)
			
			# For CSV files, try to analyze structure
			if file_extension == '.csv' and object_size < 10 * 1024 * 1024:  # Less than 10MB
				try:
					# Download first part of file
					response = self.s3_client.get_object(
						Bucket=self.bucket_name,
						Key=object_key,
						Range='bytes=0-8191'  # First 8KB
					)
					content = response['Body'].read().decode('utf-8')
					
					# Parse CSV header
					sniffer = csv.Sniffer()
					delimiter = sniffer.sniff(content[:1024]).delimiter
					reader = csv.reader(StringIO(content), delimiter=delimiter)
					header = next(reader)
					
					asset.column_count = len(header)
					asset.description = f"S3 CSV file with {len(header)} columns"
					
					# Add basic column metadata
					for col_name in header:
						column = ColumnMetadata(
							name=col_name.strip(),
							data_type=DataType.STRING,
							is_nullable=True
						)
						asset.columns.append(column)
				
				except Exception as e:
					await self._log_error(f"Failed to analyze CSV structure for {asset_name}: {str(e)}")
			
			# For JSON files, try to analyze structure
			elif file_extension == '.json' and object_size < 10 * 1024 * 1024:  # Less than 10MB
				try:
					# Download entire file for JSON (since we need complete structure)
					response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
					content = response['Body'].read().decode('utf-8')
					data = json.loads(content)
					
					if isinstance(data, list) and data and isinstance(data[0], dict):
						columns = list(data[0].keys())
						asset.column_count = len(columns)
						asset.row_count = len(data)
						asset.description = f"S3 JSON array with {len(columns)} fields and {len(data)} records"
						
						# Add basic column metadata
						for col_name in columns:
							column = ColumnMetadata(
								name=col_name,
								data_type=DataType.JSON,
								is_nullable=True
							)
							asset.columns.append(column)
					
					elif isinstance(data, dict):
						columns = list(data.keys())
						asset.column_count = len(columns)
						asset.row_count = 1
						asset.description = f"S3 JSON object with {len(columns)} fields"
						
						# Add basic column metadata
						for col_name in columns:
							column = ColumnMetadata(
								name=col_name,
								data_type=DataType.JSON,
								is_nullable=True
							)
							asset.columns.append(column)
				
				except Exception as e:
					await self._log_error(f"Failed to analyze JSON structure for {asset_name}: {str(e)}")
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for S3 object {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from S3 object"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to S3")
				return []
			
			# Find the S3 object
			object_key = None
			paginator = self.s3_client.get_paginator('list_objects_v2')
			page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
			
			for page in page_iterator:
				for obj in page.get('Contents', []):
					if os.path.basename(obj['Key']) == asset_name:
						object_key = obj['Key']
						break
				if object_key:
					break
			
			if not object_key:
				await self._log_error(f"S3 object not found: {asset_name}")
				return []
			
			# Get object info
			head_response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
			object_size = head_response['ContentLength']
			file_extension = self._get_file_extension(object_key)
			
			# For small files, download and parse
			if object_size < 50 * 1024 * 1024:  # Less than 50MB
				response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
				content = response['Body'].read()
				
				if file_extension == '.csv':
					content_str = content.decode('utf-8')
					df = pd.read_csv(StringIO(content_str), nrows=min(limit, self.config.max_sample_rows))
					return df.to_dict('records')
				
				elif file_extension == '.json':
					content_str = content.decode('utf-8')
					data = json.loads(content_str)
					
					if isinstance(data, list):
						sample_limit = min(limit, len(data), self.config.max_sample_rows)
						return data[:sample_limit] if isinstance(data[0], dict) else [{"value": item} for item in data[:sample_limit]]
					elif isinstance(data, dict):
						return [data]
					else:
						return [{"value": data}]
				
				elif file_extension == '.parquet':
					# For Parquet, we need to save to temp file and read with pandas
					import tempfile
					with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
						tmp_file.write(content)
						tmp_file.flush()
						df = pd.read_parquet(tmp_file.name, engine='pyarrow')
						sample_df = df.head(min(limit, self.config.max_sample_rows))
						return sample_df.to_dict('records')
			
			else:
				# For large files, try to read partial content for CSV/JSON
				if file_extension == '.csv':
					# Read first portion
					response = self.s3_client.get_object(
						Bucket=self.bucket_name,
						Key=object_key,
						Range=f'bytes=0-{min(1024*1024, object_size-1)}'  # First 1MB or entire file
					)
					content = response['Body'].read().decode('utf-8')
					df = pd.read_csv(StringIO(content), nrows=min(limit, self.config.max_sample_rows))
					return df.to_dict('records')
			
			return []
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from S3 object {asset_name}: {str(e)}")
			return []


class GCSConnector(BaseConnector):
	"""Google Cloud Storage metadata discovery connector"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.FILE
		self.source_system = "gcs"
		self.gcs_client = None
		self.bucket_name = None
		self.prefix = ""
		self.bucket = None
	
	async def connect(self) -> bool:
		"""Establish connection to GCS"""
		try:
			# Parse connection string (format: gs://bucket-name/prefix or gs://bucket-name)
			connection_parts = self.config.connection_string.replace('gs://', '').split('/', 1)
			self.bucket_name = connection_parts[0]
			self.prefix = connection_parts[1] if len(connection_parts) > 1 else ""
			
			# Initialize GCS client
			if self.config.additional_params.get('service_account_path'):
				# Use service account key file
				self.gcs_client = storage.Client.from_service_account_json(
					self.config.additional_params['service_account_path']
				)
			elif self.config.additional_params.get('project_id'):
				# Use default credentials with explicit project
				self.gcs_client = storage.Client(project=self.config.additional_params['project_id'])
			else:
				# Use default credentials
				self.gcs_client = storage.Client()
			
			# Test connection by accessing bucket
			self.bucket = self.gcs_client.bucket(self.bucket_name)
			self.bucket.reload()  # This will raise an exception if bucket doesn't exist or no access
			
			self.is_connected = True
			await self._log_info(f"Connected to GCS bucket: {self.bucket_name}, prefix: {self.prefix}")
			return True
			
		except NotFound:
			await self._log_error(f"GCS bucket not found: {self.bucket_name}")
			return False
		except Exception as e:
			await self._log_error(f"Failed to connect to GCS: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Close GCS connection"""
		self.is_connected = False
		self.gcs_client = None
		self.bucket = None
		await self._log_info("Disconnected from GCS")
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test connection to GCS"""
		try:
			if await self.connect():
				# Count objects in bucket
				blob_count = 0
				for blob in self.bucket.list_blobs(prefix=self.prefix, max_results=100):
					blob_count += 1
				
				return {
					"status": "success",
					"message": f"Successfully connected to GCS bucket: {self.bucket_name}",
					"bucket": self.bucket_name,
					"prefix": self.prefix,
					"objects_found": blob_count
				}
			else:
				return {
					"status": "error",
					"message": f"Failed to connect to GCS bucket: {self.bucket_name}"
				}
		except Exception as e:
			return {
				"status": "error",
				"message": f"GCS connection test failed: {str(e)}"
			}
	
	def _get_file_extension(self, blob_name: str) -> str:
		"""Extract file extension from GCS blob name"""
		return os.path.splitext(blob_name)[1].lower()
	
	def _infer_content_type_from_extension(self, extension: str) -> str:
		"""Infer content type from file extension"""
		extension_mapping = {
			'.csv': 'text/csv',
			'.json': 'application/json',
			'.parquet': 'application/octet-stream',
			'.avro': 'application/octet-stream',
			'.txt': 'text/plain',
			'.xml': 'application/xml',
			'.pdf': 'application/pdf',
			'.zip': 'application/zip',
			'.gz': 'application/gzip'
		}
		return extension_mapping.get(extension, 'application/octet-stream')
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover all GCS objects and their metadata"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not self.is_connected:
				if not await self.connect():
					result.add_error("Failed to establish GCS connection")
					result.complete_discovery()
					return result
			
			# List all blobs in bucket with prefix
			for blob in self.bucket.list_blobs(prefix=self.prefix):
				try:
					blob_name = blob.name
					object_name = os.path.basename(blob_name)
					
					# Skip directories (names ending with /)
					if blob_name.endswith('/'):
						continue
					
					# Check if object should be included
					if not should_include_asset(object_name, self.config.include_patterns, self.config.exclude_patterns):
						continue
					
					# Get blob metadata
					blob.reload()  # Ensure we have all metadata
					object_size = blob.size
					last_modified = blob.updated
					content_type = blob.content_type or self._infer_content_type_from_extension(self._get_file_extension(blob_name))
					
					# Determine asset type based on file extension
					file_extension = self._get_file_extension(blob_name)
					asset_type_mapping = {
						'.csv': 'csv_file',
						'.json': 'json_file',
						'.parquet': 'parquet_file',
						'.avro': 'avro_file',
						'.txt': 'text_file',
						'.xml': 'xml_file'
					}
					asset_type = asset_type_mapping.get(file_extension, 'file')
					
					# Create asset metadata
					asset = AssetMetadata(
						name=object_name,
						asset_type=asset_type,
						source_system=self.source_system,
						full_name=f"gs://{self.bucket_name}/{blob_name}",
						description=f"GCS object: {blob_name}",
						size_bytes=object_size,
						created_at=blob.time_created,
						modified_at=last_modified,
						location=f"gs://{self.bucket_name}/{blob_name}",
						properties={
							"bucket": self.bucket_name,
							"blob_name": blob_name,
							"content_type": content_type,
							"file_extension": file_extension,
							"etag": blob.etag,
							"generation": blob.generation,
							"metageneration": blob.metageneration,
							"gcs_metadata": blob.metadata or {}
						}
					)
					
					# For structured files, add basic column info placeholder
					if file_extension in ['.csv', '.json', '.parquet', '.avro']:
						asset.column_count = 0
						asset.description += " (columns will be determined on detailed inspection)"
					
					# Estimate quality score
					asset.estimated_quality_score = self._estimate_quality_score(asset)
					
					result.add_asset(asset)
					
				except Exception as e:
					result.add_error(f"Failed to process GCS object {blob.name}: {str(e)}")
			
			result.complete_discovery()
			await self._log_info(f"Discovery completed: {result.successful_assets} assets found")
			
		except Exception as e:
			result.add_error(f"GCS discovery failed: {str(e)}")
			result.complete_discovery()
		
		return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed schema information for a specific GCS object"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to GCS")
				return None
			
			# Find the GCS blob
			blob = None
			for b in self.bucket.list_blobs(prefix=self.prefix):
				if os.path.basename(b.name) == asset_name:
					blob = b
					break
			
			if not blob:
				await self._log_error(f"GCS object not found: {asset_name}")
				return None
			
			# Get blob metadata
			blob.reload()
			object_size = blob.size
			last_modified = blob.updated
			content_type = blob.content_type or 'application/octet-stream'
			file_extension = self._get_file_extension(blob.name)
			
			asset = AssetMetadata(
				name=asset_name,
				asset_type=f"{file_extension[1:]}_file" if file_extension else "file",
				source_system=self.source_system,
				full_name=f"gs://{self.bucket_name}/{blob.name}",
				description=f"GCS object: {blob.name}",
				size_bytes=object_size,
				created_at=blob.time_created,
				modified_at=last_modified,
				location=f"gs://{self.bucket_name}/{blob.name}",
				properties={
					"bucket": self.bucket_name,
					"blob_name": blob.name,
					"content_type": content_type,
					"file_extension": file_extension,
					"etag": blob.etag
				}
			)
			
			# For CSV files, try to analyze structure
			if file_extension == '.csv' and object_size < 10 * 1024 * 1024:  # Less than 10MB
				try:
					# Download first part of file
					content = blob.download_as_bytes(start=0, end=8191)  # First 8KB
					content_str = content.decode('utf-8')
					
					# Parse CSV header
					sniffer = csv.Sniffer()
					delimiter = sniffer.sniff(content_str[:1024]).delimiter
					reader = csv.reader(StringIO(content_str), delimiter=delimiter)
					header = next(reader)
					
					asset.column_count = len(header)
					asset.description = f"GCS CSV file with {len(header)} columns"
					
					# Add basic column metadata
					for col_name in header:
						column = ColumnMetadata(
							name=col_name.strip(),
							data_type=DataType.STRING,
							is_nullable=True
						)
						asset.columns.append(column)
				
				except Exception as e:
					await self._log_error(f"Failed to analyze CSV structure for {asset_name}: {str(e)}")
			
			# For JSON files, try to analyze structure
			elif file_extension == '.json' and object_size < 10 * 1024 * 1024:  # Less than 10MB
				try:
					# Download entire file for JSON
					content = blob.download_as_bytes()
					content_str = content.decode('utf-8')
					data = json.loads(content_str)
					
					if isinstance(data, list) and data and isinstance(data[0], dict):
						columns = list(data[0].keys())
						asset.column_count = len(columns)
						asset.row_count = len(data)
						asset.description = f"GCS JSON array with {len(columns)} fields and {len(data)} records"
						
						# Add basic column metadata
						for col_name in columns:
							column = ColumnMetadata(
								name=col_name,
								data_type=DataType.JSON,
								is_nullable=True
							)
							asset.columns.append(column)
					
					elif isinstance(data, dict):
						columns = list(data.keys())
						asset.column_count = len(columns)
						asset.row_count = 1
						asset.description = f"GCS JSON object with {len(columns)} fields"
						
						# Add basic column metadata
						for col_name in columns:
							column = ColumnMetadata(
								name=col_name,
								data_type=DataType.JSON,
								is_nullable=True
							)
							asset.columns.append(column)
				
				except Exception as e:
					await self._log_error(f"Failed to analyze JSON structure for {asset_name}: {str(e)}")
			
			# Estimate quality score
			asset.estimated_quality_score = self._estimate_quality_score(asset)
			
			return asset
			
		except Exception as e:
			await self._log_error(f"Failed to get schema for GCS object {asset_name}: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Sample data from GCS object"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to GCS")
				return []
			
			# Find the GCS blob
			blob = None
			for b in self.bucket.list_blobs(prefix=self.prefix):
				if os.path.basename(b.name) == asset_name:
					blob = b
					break
			
			if not blob:
				await self._log_error(f"GCS object not found: {asset_name}")
				return []
			
			# Get blob info
			blob.reload()
			object_size = blob.size
			file_extension = self._get_file_extension(blob.name)
			
			# For small files, download and parse
			if object_size < 50 * 1024 * 1024:  # Less than 50MB
				content = blob.download_as_bytes()
				
				if file_extension == '.csv':
					content_str = content.decode('utf-8')
					df = pd.read_csv(StringIO(content_str), nrows=min(limit, self.config.max_sample_rows))
					return df.to_dict('records')
				
				elif file_extension == '.json':
					content_str = content.decode('utf-8')
					data = json.loads(content_str)
					
					if isinstance(data, list):
						sample_limit = min(limit, len(data), self.config.max_sample_rows)
						return data[:sample_limit] if data and isinstance(data[0], dict) else [{"value": item} for item in data[:sample_limit]]
					elif isinstance(data, dict):
						return [data]
					else:
						return [{"value": data}]
				
				elif file_extension == '.parquet':
					# For Parquet, save to temp file and read with pandas
					import tempfile
					with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
						tmp_file.write(content)
						tmp_file.flush()
						df = pd.read_parquet(tmp_file.name, engine='pyarrow')
						sample_df = df.head(min(limit, self.config.max_sample_rows))
						return sample_df.to_dict('records')
			
			else:
				# For large files, try to read partial content for CSV/JSON
				if file_extension == '.csv':
					# Read first portion
					content = blob.download_as_bytes(start=0, end=min(1024*1024, object_size-1))  # First 1MB
					content_str = content.decode('utf-8')
					df = pd.read_csv(StringIO(content_str), nrows=min(limit, self.config.max_sample_rows))
					return df.to_dict('records')
			
			return []
			
		except Exception as e:
			await self._log_error(f"Failed to sample data from GCS object {asset_name}: {str(e)}")
			return []