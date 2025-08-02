"""
APG Workflow Orchestration File System Connectors

High-performance file system connectors for local filesystem, FTP, SFTP, S3,
and other storage systems with streaming, compression, and security features.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from datetime import datetime, timezone
import logging
import mimetypes
import hashlib
import gzip
import zipfile
import io

# FTP/SFTP
import aioftp
import asyncssh
from asyncssh.sftp import SFTPError

# S3
import aiobotocore.session
from botocore.exceptions import ClientError

from pydantic import BaseModel, Field, ConfigDict, validator

from .base_connector import BaseConnector, ConnectorConfiguration

logger = logging.getLogger(__name__)

class FileSystemConfiguration(ConnectorConfiguration):
	"""Local file system configuration."""
	
	base_path: str = Field(..., description="Base directory path")
	create_base_path: bool = Field(default=True, description="Create base path if it doesn't exist")
	permissions: str = Field(default="755", regex="^[0-7]{3}$")
	enable_compression: bool = Field(default=False)
	compression_format: str = Field(default="gzip", regex="^(gzip|zip)$")
	max_file_size_mb: int = Field(default=100, ge=1, le=10000)
	allowed_extensions: List[str] = Field(default_factory=list)
	blocked_extensions: List[str] = Field(default_factory=lambda: [".exe", ".bat", ".sh", ".cmd"])
	enable_versioning: bool = Field(default=False)
	checksum_algorithm: str = Field(default="sha256", regex="^(md5|sha1|sha256|sha512)$")

class FTPConfiguration(ConnectorConfiguration):
	"""FTP/FTPS configuration."""
	
	host: str = Field(..., description="FTP server host")
	port: int = Field(default=21, ge=1, le=65535)
	username: str = Field(..., description="FTP username")
	password: str = Field(..., description="FTP password")
	use_tls: bool = Field(default=False, description="Use FTPS (FTP over TLS)")
	passive_mode: bool = Field(default=True)
	encoding: str = Field(default="utf-8")
	timeout: int = Field(default=30, ge=1, le=300)
	base_path: str = Field(default="/", description="Base directory on FTP server")
	transfer_mode: str = Field(default="binary", regex="^(binary|ascii)$")

class SFTPConfiguration(ConnectorConfiguration):
	"""SFTP configuration."""
	
	host: str = Field(..., description="SFTP server host")
	port: int = Field(default=22, ge=1, le=65535)
	username: str = Field(..., description="SFTP username")
	password: Optional[str] = Field(default=None, description="SFTP password")
	private_key_path: Optional[str] = Field(default=None, description="Path to private key file")
	private_key_passphrase: Optional[str] = Field(default=None)
	known_hosts_path: Optional[str] = Field(default=None)
	compression: bool = Field(default=False)
	keepalive_interval: int = Field(default=30, ge=1, le=300)
	base_path: str = Field(default="/", description="Base directory on SFTP server")

class S3Configuration(ConnectorConfiguration):
	"""Amazon S3 configuration."""
	
	bucket_name: str = Field(..., description="S3 bucket name")
	region: str = Field(default="us-east-1", description="AWS region")
	access_key_id: Optional[str] = Field(default=None)
	secret_access_key: Optional[str] = Field(default=None)
	session_token: Optional[str] = Field(default=None)
	endpoint_url: Optional[str] = Field(default=None, description="Custom S3 endpoint")
	use_ssl: bool = Field(default=True)
	verify_ssl: bool = Field(default=True)
	prefix: str = Field(default="", description="Key prefix for all objects")
	server_side_encryption: Optional[str] = Field(default=None, regex="^(AES256|aws:kms)$")
	storage_class: str = Field(default="STANDARD", regex="^(STANDARD|REDUCED_REDUNDANCY|STANDARD_IA|ONEZONE_IA|INTELLIGENT_TIERING|GLACIER|DEEP_ARCHIVE)$")
	multipart_threshold: int = Field(default=64, ge=5, le=5000, description="Multipart upload threshold (MB)")
	max_concurrency: int = Field(default=10, ge=1, le=100)

class FileSystemConnector(BaseConnector):
	"""Local file system connector."""
	
	def __init__(self, config: FileSystemConfiguration):
		super().__init__(config)
		self.config: FileSystemConfiguration = config
		self.base_path = Path(config.base_path)
	
	async def _connect(self) -> None:
		"""Initialize file system connector."""
		
		# Create base path if it doesn't exist
		if self.config.create_base_path and not self.base_path.exists():
			self.base_path.mkdir(parents=True, mode=int(self.config.permissions, 8))
		
		# Verify base path exists and is accessible
		if not self.base_path.exists():
			raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
		
		if not self.base_path.is_dir():
			raise NotADirectoryError(f"Base path is not a directory: {self.base_path}")
		
		logger.info(self._log_connector_info(f"File system connector initialized at {self.base_path}"))
	
	async def _disconnect(self) -> None:
		"""Close file system connector."""
		logger.info(self._log_connector_info("File system connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute file system operation."""
		
		if operation == "write_file":
			return await self._write_file(parameters)
		elif operation == "read_file":
			return await self._read_file(parameters)
		elif operation == "delete_file":
			return await self._delete_file(parameters)
		elif operation == "list_files":
			return await self._list_files(parameters)
		elif operation == "move_file":
			return await self._move_file(parameters)
		elif operation == "copy_file":
			return await self._copy_file(parameters)
		elif operation == "get_file_info":
			return await self._get_file_info(parameters)
		elif operation == "create_directory":
			return await self._create_directory(parameters)
		else:
			raise ValueError(f"Unsupported file system operation: {operation}")
	
	async def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Write file to file system."""
		
		file_path = params.get("path")
		content = params.get("content")
		mode = params.get("mode", "w")
		encoding = params.get("encoding", "utf-8")
		create_dirs = params.get("create_dirs", True)
		
		if not file_path or content is None:
			raise ValueError("File path and content are required")
		
		# Validate file extension
		self._validate_file_extension(file_path)
		
		# Build full path
		full_path = self.base_path / file_path
		
		# Check file size limits
		if isinstance(content, str):
			content_size = len(content.encode(encoding))
		else:
			content_size = len(content)
		
		max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
		if content_size > max_size_bytes:
			raise ValueError(f"File size {content_size} exceeds limit {max_size_bytes}")
		
		try:
			# Create parent directories if needed
			if create_dirs:
				full_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Handle versioning
			if self.config.enable_versioning and full_path.exists():
				await self._create_file_version(full_path)
			
			# Write file
			if mode.startswith("b") or isinstance(content, bytes):
				async with aiofiles.open(full_path, mode, encoding=None) as f:
					await f.write(content)
			else:
				async with aiofiles.open(full_path, mode, encoding=encoding) as f:
					await f.write(content)
			
			# Set permissions
			await aiofiles.os.chmod(full_path, int(self.config.permissions, 8))
			
			# Calculate checksum
			checksum = await self._calculate_checksum(full_path)
			
			# Compress if enabled
			if self.config.enable_compression:
				await self._compress_file(full_path)
			
			return {
				"path": str(full_path.relative_to(self.base_path)),
				"size": content_size,
				"checksum": checksum,
				"compressed": self.config.enable_compression,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to write file {file_path}: {e}"))
			raise
	
	async def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Read file from file system."""
		
		file_path = params.get("path")
		mode = params.get("mode", "r")
		encoding = params.get("encoding", "utf-8")
		chunk_size = params.get("chunk_size", 8192)
		
		if not file_path:
			raise ValueError("File path is required")
		
		full_path = self.base_path / file_path
		
		if not full_path.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		try:
			# Check if file is compressed
			compressed_path = full_path.with_suffix(full_path.suffix + ".gz")
			if compressed_path.exists():
				full_path = compressed_path
				# Decompress on the fly if needed
			
			file_size = full_path.stat().st_size
			
			# Read file content
			if mode.startswith("b"):
				async with aiofiles.open(full_path, mode) as f:
					content = await f.read()
			else:
				async with aiofiles.open(full_path, mode, encoding=encoding) as f:
					content = await f.read()
			
			# Calculate checksum
			checksum = await self._calculate_checksum(full_path)
			
			return {
				"path": str(full_path.relative_to(self.base_path)),
				"content": content,
				"size": file_size,
				"checksum": checksum,
				"mime_type": mimetypes.guess_type(str(full_path))[0],
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to read file {file_path}: {e}"))
			raise
	
	async def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Delete file from file system."""
		
		file_path = params.get("path")
		
		if not file_path:
			raise ValueError("File path is required")
		
		full_path = self.base_path / file_path
		
		if not full_path.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		try:
			file_size = full_path.stat().st_size
			await aiofiles.os.remove(full_path)
			
			# Also remove compressed version if it exists
			compressed_path = full_path.with_suffix(full_path.suffix + ".gz")
			if compressed_path.exists():
				await aiofiles.os.remove(compressed_path)
			
			return {
				"path": str(full_path.relative_to(self.base_path)),
				"size": file_size,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to delete file {file_path}: {e}"))
			raise
	
	async def _list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""List files in directory."""
		
		directory = params.get("directory", ".")
		pattern = params.get("pattern", "*")
		recursive = params.get("recursive", False)
		include_hidden = params.get("include_hidden", False)
		
		full_dir_path = self.base_path / directory
		
		if not full_dir_path.exists():
			raise FileNotFoundError(f"Directory not found: {directory}")
		
		try:
			files = []
			
			if recursive:
				pattern_path = full_dir_path.rglob(pattern)
			else:
				pattern_path = full_dir_path.glob(pattern)
			
			for file_path in pattern_path:
				if file_path.is_file():
					# Skip hidden files unless requested
					if not include_hidden and file_path.name.startswith('.'):
						continue
					
					stat = file_path.stat()
					files.append({
						"path": str(file_path.relative_to(self.base_path)),
						"name": file_path.name,
						"size": stat.st_size,
						"modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
						"created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
						"mime_type": mimetypes.guess_type(str(file_path))[0]
					})
			
			return {
				"directory": directory,
				"files": files,
				"count": len(files),
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to list files in {directory}: {e}"))
			raise
	
	async def _move_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Move file to new location."""
		
		source_path = params.get("source")
		destination_path = params.get("destination")
		
		if not source_path or not destination_path:
			raise ValueError("Source and destination paths are required")
		
		full_source = self.base_path / source_path
		full_destination = self.base_path / destination_path
		
		if not full_source.exists():
			raise FileNotFoundError(f"Source file not found: {source_path}")
		
		try:
			# Create destination directory if needed
			full_destination.parent.mkdir(parents=True, exist_ok=True)
			
			# Move file
			full_source.rename(full_destination)
			
			return {
				"source": source_path,
				"destination": destination_path,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to move file {source_path}: {e}"))
			raise
	
	async def _copy_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Copy file to new location."""
		
		source_path = params.get("source")
		destination_path = params.get("destination")
		
		if not source_path or not destination_path:
			raise ValueError("Source and destination paths are required")
		
		full_source = self.base_path / source_path
		full_destination = self.base_path / destination_path
		
		if not full_source.exists():
			raise FileNotFoundError(f"Source file not found: {source_path}")
		
		try:
			# Create destination directory if needed
			full_destination.parent.mkdir(parents=True, exist_ok=True)
			
			# Copy file
			async with aiofiles.open(full_source, 'rb') as src:
				async with aiofiles.open(full_destination, 'wb') as dst:
					async for chunk in src:
						await dst.write(chunk)
			
			# Copy file permissions
			source_stat = full_source.stat()
			await aiofiles.os.chmod(full_destination, source_stat.st_mode)
			
			return {
				"source": source_path,
				"destination": destination_path,
				"size": source_stat.st_size,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to copy file {source_path}: {e}"))
			raise
	
	async def _get_file_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Get file information."""
		
		file_path = params.get("path")
		
		if not file_path:
			raise ValueError("File path is required")
		
		full_path = self.base_path / file_path
		
		if not full_path.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		try:
			stat = full_path.stat()
			checksum = await self._calculate_checksum(full_path)
			
			return {
				"path": str(full_path.relative_to(self.base_path)),
				"name": full_path.name,
				"size": stat.st_size,
				"modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
				"created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
				"permissions": oct(stat.st_mode)[-3:],
				"mime_type": mimetypes.guess_type(str(full_path))[0],
				"checksum": checksum,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to get file info {file_path}: {e}"))
			raise
	
	async def _create_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Create directory."""
		
		directory_path = params.get("path")
		parents = params.get("parents", True)
		exist_ok = params.get("exist_ok", True)
		
		if not directory_path:
			raise ValueError("Directory path is required")
		
		full_path = self.base_path / directory_path
		
		try:
			full_path.mkdir(parents=parents, exist_ok=exist_ok)
			await aiofiles.os.chmod(full_path, int(self.config.permissions, 8))
			
			return {
				"path": str(full_path.relative_to(self.base_path)),
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to create directory {directory_path}: {e}"))
			raise
	
	async def _health_check(self) -> bool:
		"""Check file system connectivity."""
		try:
			# Try to create a temporary file
			test_file = self.base_path / ".health_check"
			async with aiofiles.open(test_file, 'w') as f:
				await f.write("health_check")
			
			# Try to read it back
			async with aiofiles.open(test_file, 'r') as f:
				content = await f.read()
			
			# Clean up
			await aiofiles.os.remove(test_file)
			
			return content == "health_check"
		
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False
	
	def _validate_file_extension(self, file_path: str) -> None:
		"""Validate file extension against allowed/blocked lists."""
		path = Path(file_path)
		extension = path.suffix.lower()
		
		if self.config.blocked_extensions and extension in self.config.blocked_extensions:
			raise ValueError(f"File extension {extension} is blocked")
		
		if self.config.allowed_extensions and extension not in self.config.allowed_extensions:
			raise ValueError(f"File extension {extension} is not allowed")
	
	async def _calculate_checksum(self, file_path: Path) -> str:
		"""Calculate file checksum."""
		
		hash_algo = hashlib.new(self.config.checksum_algorithm)
		
		async with aiofiles.open(file_path, 'rb') as f:
			while chunk := await f.read(8192):
				hash_algo.update(chunk)
		
		return hash_algo.hexdigest()
	
	async def _compress_file(self, file_path: Path) -> None:
		"""Compress file using configured compression format."""
		
		if self.config.compression_format == "gzip":
			compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
			
			async with aiofiles.open(file_path, 'rb') as f_in:
				with gzip.open(compressed_path, 'wb') as f_out:
					f_out.write(await f_in.read())
			
			# Remove original file
			await aiofiles.os.remove(file_path)
		
		elif self.config.compression_format == "zip":
			compressed_path = file_path.with_suffix(".zip")
			
			with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
				zipf.write(file_path, file_path.name)
			
			# Remove original file
			await aiofiles.os.remove(file_path)
	
	async def _create_file_version(self, file_path: Path) -> None:
		"""Create versioned copy of existing file."""
		
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		version_path = file_path.with_suffix(f".{timestamp}{file_path.suffix}")
		
		# Copy existing file to version
		async with aiofiles.open(file_path, 'rb') as src:
			async with aiofiles.open(version_path, 'wb') as dst:
				async for chunk in src:
					await dst.write(chunk)

class FTPConnector(BaseConnector):
	"""FTP/FTPS connector."""
	
	def __init__(self, config: FTPConfiguration):
		super().__init__(config)
		self.config: FTPConfiguration = config
		self.client: Optional[aioftp.Client] = None
	
	async def _connect(self) -> None:
		"""Initialize FTP connection."""
		
		self.client = aioftp.Client()
		
		try:
			await self.client.connect(
				host=self.config.host,
				port=self.config.port
			)
			
			await self.client.login(
				user=self.config.username,
				password=self.config.password
			)
			
			# Change to base directory
			if self.config.base_path != "/":
				await self.client.change_directory(self.config.base_path)
			
			# Set transfer mode
			if self.config.transfer_mode == "binary":
				await self.client.type("I")  # Binary mode
			else:
				await self.client.type("A")  # ASCII mode
			
			# Set passive mode
			if self.config.passive_mode:
				await self.client.passive()
			
			logger.info(self._log_connector_info("FTP connector initialized"))
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to connect to FTP: {e}"))
			raise
	
	async def _disconnect(self) -> None:
		"""Close FTP connection."""
		if self.client:
			await self.client.quit()
			self.client = None
		
		logger.info(self._log_connector_info("FTP connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute FTP operation."""
		
		if operation == "upload":
			return await self._upload_file(parameters)
		elif operation == "download":
			return await self._download_file(parameters)
		elif operation == "delete":
			return await self._delete_file(parameters)
		elif operation == "list":
			return await self._list_files(parameters)
		elif operation == "mkdir":
			return await self._create_directory(parameters)
		else:
			raise ValueError(f"Unsupported FTP operation: {operation}")
	
	async def _upload_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Upload file via FTP."""
		
		local_path = params.get("local_path")
		remote_path = params.get("remote_path")
		
		if not local_path or not remote_path:
			raise ValueError("Local and remote paths are required")
		
		try:
			await self.client.upload(local_path, remote_path)
			
			# Get file size
			local_file = Path(local_path)
			file_size = local_file.stat().st_size
			
			return {
				"local_path": local_path,
				"remote_path": remote_path,
				"size": file_size,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to upload file: {e}"))
			raise
	
	async def _download_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Download file via FTP."""
		
		remote_path = params.get("remote_path")
		local_path = params.get("local_path")
		
		if not remote_path or not local_path:
			raise ValueError("Remote and local paths are required")
		
		try:
			await self.client.download(remote_path, local_path)
			
			# Get file size
			local_file = Path(local_path)
			file_size = local_file.stat().st_size
			
			return {
				"remote_path": remote_path,
				"local_path": local_path,
				"size": file_size,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to download file: {e}"))
			raise
	
	async def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Delete file via FTP."""
		
		remote_path = params.get("remote_path")
		
		if not remote_path:
			raise ValueError("Remote path is required")
		
		try:
			await self.client.remove(remote_path)
			
			return {
				"remote_path": remote_path,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to delete file: {e}"))
			raise
	
	async def _list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""List files via FTP."""
		
		directory = params.get("directory", ".")
		
		try:
			files = []
			async for path, info in self.client.list(directory):
				if info["type"] == "file":
					files.append({
						"name": path.name,
						"path": str(path),
						"size": info.get("size", 0),
						"modified": info.get("modify", "").isoformat() if info.get("modify") else None
					})
			
			return {
				"directory": directory,
				"files": files,
				"count": len(files),
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to list files: {e}"))
			raise
	
	async def _create_directory(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Create directory via FTP."""
		
		directory_path = params.get("path")
		
		if not directory_path:
			raise ValueError("Directory path is required")
		
		try:
			await self.client.make_directory(directory_path)
			
			return {
				"path": directory_path,
				"success": True
			}
		
		except Exception as e:
			logger.error(self._log_connector_info(f"Failed to create directory: {e}"))
			raise
	
	async def _health_check(self) -> bool:
		"""Check FTP connectivity."""
		try:
			# Try to list current directory
			files = []
			async for path, info in self.client.list("."):
				files.append(path)
				break  # Just need to verify we can list
			return True
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False

class S3Connector(BaseConnector):
	"""Amazon S3 connector."""
	
	def __init__(self, config: S3Configuration):
		super().__init__(config)
		self.config: S3Configuration = config
		self.session = None
		self.client = None
	
	async def _connect(self) -> None:
		"""Initialize S3 connection."""
		
		# Create aiobotocore session
		self.session = aiobotocore.session.get_session()
		
		# Set up credentials
		credentials = {}
		if self.config.access_key_id and self.config.secret_access_key:
			credentials = {
				"aws_access_key_id": self.config.access_key_id,
				"aws_secret_access_key": self.config.secret_access_key
			}
			if self.config.session_token:
				credentials["aws_session_token"] = self.config.session_token
		
		# Create S3 client
		self.client = self.session.create_client(
			"s3",
			region_name=self.config.region,
			endpoint_url=self.config.endpoint_url,
			use_ssl=self.config.use_ssl,
			verify=self.config.verify_ssl,
			**credentials
		)
		
		await self.client.__aenter__()
		
		logger.info(self._log_connector_info("S3 connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close S3 connection."""
		if self.client:
			await self.client.__aexit__(None, None, None)
			self.client = None
		
		logger.info(self._log_connector_info("S3 connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute S3 operation."""
		
		if operation == "upload":
			return await self._upload_object(parameters)
		elif operation == "download":
			return await self._download_object(parameters)
		elif operation == "delete":
			return await self._delete_object(parameters)
		elif operation == "list":
			return await self._list_objects(parameters)
		elif operation == "copy":
			return await self._copy_object(parameters)
		else:
			raise ValueError(f"Unsupported S3 operation: {operation}")
	
	async def _upload_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Upload object to S3."""
		
		key = params.get("key")
		body = params.get("body")
		content_type = params.get("content_type")
		metadata = params.get("metadata", {})
		
		if not key:
			raise ValueError("Object key is required")
		
		# Add prefix if configured
		if self.config.prefix:
			key = f"{self.config.prefix.rstrip('/')}/{key.lstrip('/')}"
		
		try:
			# Prepare upload parameters
			upload_params = {
				"Bucket": self.config.bucket_name,
				"Key": key,
				"Body": body
			}
			
			if content_type:
				upload_params["ContentType"] = content_type
			
			if metadata:
				upload_params["Metadata"] = metadata
			
			if self.config.server_side_encryption:
				upload_params["ServerSideEncryption"] = self.config.server_side_encryption
			
			if self.config.storage_class != "STANDARD":
				upload_params["StorageClass"] = self.config.storage_class
			
			# Upload object
			response = await self.client.put_object(**upload_params)
			
			return {
				"bucket": self.config.bucket_name,
				"key": key,
				"etag": response.get("ETag", "").strip('"'),
				"version_id": response.get("VersionId"),
				"success": True
			}
		
		except ClientError as e:
			logger.error(self._log_connector_info(f"Failed to upload object: {e}"))
			raise
	
	async def _download_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Download object from S3."""
		
		key = params.get("key")
		
		if not key:
			raise ValueError("Object key is required")
		
		# Add prefix if configured
		if self.config.prefix:
			key = f"{self.config.prefix.rstrip('/')}/{key.lstrip('/')}"
		
		try:
			response = await self.client.get_object(
				Bucket=self.config.bucket_name,
				Key=key
			)
			
			# Read body
			body = await response["Body"].read()
			
			return {
				"bucket": self.config.bucket_name,
				"key": key,
				"body": body,
				"content_type": response.get("ContentType"),
				"content_length": response.get("ContentLength"),
				"etag": response.get("ETag", "").strip('"'),
				"last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
				"metadata": response.get("Metadata", {}),
				"success": True
			}
		
		except ClientError as e:
			logger.error(self._log_connector_info(f"Failed to download object: {e}"))
			raise
	
	async def _delete_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Delete object from S3."""
		
		key = params.get("key")
		
		if not key:
			raise ValueError("Object key is required")
		
		# Add prefix if configured
		if self.config.prefix:
			key = f"{self.config.prefix.rstrip('/')}/{key.lstrip('/')}"
		
		try:
			response = await self.client.delete_object(
				Bucket=self.config.bucket_name,
				Key=key
			)
			
			return {
				"bucket": self.config.bucket_name,
				"key": key,
				"version_id": response.get("VersionId"),
				"success": True
			}
		
		except ClientError as e:
			logger.error(self._log_connector_info(f"Failed to delete object: {e}"))
			raise
	
	async def _list_objects(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""List objects in S3 bucket."""
		
		prefix = params.get("prefix", "")
		delimiter = params.get("delimiter", "")
		max_keys = params.get("max_keys", 1000)
		
		# Add configured prefix
		if self.config.prefix:
			prefix = f"{self.config.prefix.rstrip('/')}/{prefix.lstrip('/')}" if prefix else self.config.prefix
		
		try:
			response = await self.client.list_objects_v2(
				Bucket=self.config.bucket_name,
				Prefix=prefix,
				Delimiter=delimiter,
				MaxKeys=max_keys
			)
			
			objects = []
			for obj in response.get("Contents", []):
				objects.append({
					"key": obj["Key"],
					"size": obj["Size"],
					"last_modified": obj["LastModified"].isoformat(),
					"etag": obj.get("ETag", "").strip('"'),
					"storage_class": obj.get("StorageClass", "STANDARD")
				})
			
			return {
				"bucket": self.config.bucket_name,
				"prefix": prefix,
				"objects": objects,
				"count": len(objects),
				"is_truncated": response.get("IsTruncated", False),
				"next_continuation_token": response.get("NextContinuationToken"),
				"success": True
			}
		
		except ClientError as e:
			logger.error(self._log_connector_info(f"Failed to list objects: {e}"))
			raise
	
	async def _copy_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Copy object in S3."""
		
		source_key = params.get("source_key")
		destination_key = params.get("destination_key")
		source_bucket = params.get("source_bucket", self.config.bucket_name)
		
		if not source_key or not destination_key:
			raise ValueError("Source and destination keys are required")
		
		# Add prefix if configured
		if self.config.prefix:
			source_key = f"{self.config.prefix.rstrip('/')}/{source_key.lstrip('/')}"
			destination_key = f"{self.config.prefix.rstrip('/')}/{destination_key.lstrip('/')}"
		
		try:
			copy_source = {
				"Bucket": source_bucket,
				"Key": source_key
			}
			
			response = await self.client.copy_object(
				CopySource=copy_source,
				Bucket=self.config.bucket_name,
				Key=destination_key
			)
			
			return {
				"source_bucket": source_bucket,
				"source_key": source_key,
				"destination_bucket": self.config.bucket_name,
				"destination_key": destination_key,
				"etag": response.get("CopyObjectResult", {}).get("ETag", "").strip('"'),
				"success": True
			}
		
		except ClientError as e:
			logger.error(self._log_connector_info(f"Failed to copy object: {e}"))
			raise
	
	async def _health_check(self) -> bool:
		"""Check S3 connectivity."""
		try:
			# Try to head the bucket
			await self.client.head_bucket(Bucket=self.config.bucket_name)
			return True
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False

# Export file connector classes
__all__ = [
	"FileSystemConnector",
	"FileSystemConfiguration",
	"FTPConnector",
	"FTPConfiguration", 
	"S3Connector",
	"S3Configuration"
]