"""
File Service for APG Workflow Mobile

Handles file operations, attachments, and media management.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
import os
import mimetypes
import hashlib
from typing import Optional, Dict, Any, List, Union, BinaryIO
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass
import json

from ..models.api_response import APIResponse
from ..utils.constants import MAX_FILE_SIZE, SUPPORTED_FILE_TYPES, UPLOAD_CHUNK_SIZE
from ..utils.exceptions import FileException, APIException
from ..utils.security import generate_random_string, secure_hash
from ..utils.validators import validate_file_size, validate_file_extension


@dataclass
class FileMetadata:
	"""File metadata information"""
	id: str
	filename: str
	original_filename: str
	file_size: int
	mime_type: str
	checksum: str
	upload_date: datetime
	last_accessed: Optional[datetime] = None
	tags: Optional[List[str]] = None
	description: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'id': self.id,
			'filename': self.filename,
			'original_filename': self.original_filename,
			'file_size': self.file_size,
			'mime_type': self.mime_type,
			'checksum': self.checksum,
			'upload_date': self.upload_date.isoformat(),
			'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
			'tags': self.tags or [],
			'description': self.description
		}
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
		return cls(
			id=data['id'],
			filename=data['filename'],
			original_filename=data['original_filename'],
			file_size=data['file_size'],
			mime_type=data['mime_type'],
			checksum=data['checksum'],
			upload_date=datetime.fromisoformat(data['upload_date']),
			last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
			tags=data.get('tags', []),
			description=data.get('description')
		)


@dataclass
class UploadProgress:
	"""Upload progress tracking"""
	file_id: str
	filename: str
	total_bytes: int
	uploaded_bytes: int
	percentage: float
	status: str  # 'uploading', 'completed', 'failed', 'cancelled'
	error_message: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'file_id': self.file_id,
			'filename': self.filename,
			'total_bytes': self.total_bytes,
			'uploaded_bytes': self.uploaded_bytes,
			'percentage': self.percentage,
			'status': self.status,
			'error_message': self.error_message
		}


class FileService:
	"""Service for file operations and management"""
	
	def __init__(self, app=None):
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# File storage paths
		self.temp_dir = Path.home() / ".apg_mobile" / "temp"
		self.cache_dir = Path.home() / ".apg_mobile" / "files"
		self.upload_dir = Path.home() / ".apg_mobile" / "uploads"
		
		# Ensure directories exist
		for directory in [self.temp_dir, self.cache_dir, self.upload_dir]:
			directory.mkdir(parents=True, exist_ok=True)
		
		# Upload tracking
		self.active_uploads: Dict[str, UploadProgress] = {}
		self.upload_callbacks: Dict[str, callable] = {}
		
		self.logger.info("File Service initialized")
	
	async def upload_file(
		self,
		file_path: Union[str, Path],
		entity_type: str,
		entity_id: str,
		description: Optional[str] = None,
		tags: Optional[List[str]] = None,
		progress_callback: Optional[callable] = None
	) -> APIResponse:
		"""Upload file to server"""
		try:
			file_path = Path(file_path)
			
			if not file_path.exists():
				return APIResponse(
					success=False,
					message="File not found",
					status_code=404
				)
			
			# Validate file
			validation_result = await self._validate_file(file_path)
			if not validation_result.success:
				return validation_result
			
			# Generate file metadata
			file_metadata = await self._generate_file_metadata(
				file_path, description, tags
			)
			
			# Track upload progress
			if progress_callback:
				self.upload_callbacks[file_metadata.id] = progress_callback
			
			# Perform chunked upload
			upload_response = await self._chunked_upload(
				file_path, file_metadata, entity_type, entity_id
			)
			
			return upload_response
			
		except Exception as e:
			self.logger.error(f"File upload failed: {e}")
			return APIResponse(
				success=False,
				message=f"Upload failed: {e}",
				status_code=500
			)
	
	async def _validate_file(self, file_path: Path) -> APIResponse:
		"""Validate file before upload"""
		try:
			# Check file size
			file_size = file_path.stat().st_size
			is_valid_size, size_error = validate_file_size(file_size, MAX_FILE_SIZE // (1024 * 1024))
			
			if not is_valid_size:
				return APIResponse(
					success=False,
					message=size_error,
					status_code=400
				)
			
			# Check file extension
			is_valid_ext, ext_error = validate_file_extension(file_path.name, SUPPORTED_FILE_TYPES)
			
			if not is_valid_ext:
				return APIResponse(
					success=False,
					message=ext_error,
					status_code=400
				)
			
			return APIResponse(success=True, message="File validation passed")
			
		except Exception as e:
			return APIResponse(
				success=False,
				message=f"File validation failed: {e}",
				status_code=500
			)
	
	async def _generate_file_metadata(
		self,
		file_path: Path,
		description: Optional[str] = None,
		tags: Optional[List[str]] = None
	) -> FileMetadata:
		"""Generate metadata for file"""
		
		# Generate unique file ID
		file_id = str(uuid.uuid4())
		
		# Get file stats
		file_stat = file_path.stat()
		file_size = file_stat.st_size
		
		# Detect MIME type
		mime_type, _ = mimetypes.guess_type(str(file_path))
		if not mime_type:
			mime_type = "application/octet-stream"
		
		# Generate checksum
		checksum = await self._calculate_file_checksum(file_path)
		
		# Generate safe filename
		safe_filename = f"{file_id}_{file_path.name}"
		
		return FileMetadata(
			id=file_id,
			filename=safe_filename,
			original_filename=file_path.name,
			file_size=file_size,
			mime_type=mime_type,
			checksum=checksum,
			upload_date=datetime.utcnow(),
			tags=tags or [],
			description=description
		)
	
	async def _calculate_file_checksum(self, file_path: Path) -> str:
		"""Calculate SHA-256 checksum of file"""
		hash_sha256 = hashlib.sha256()
		
		with open(file_path, 'rb') as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash_sha256.update(chunk)
		
		return hash_sha256.hexdigest()
	
	async def _chunked_upload(
		self,
		file_path: Path,
		metadata: FileMetadata,
		entity_type: str,
		entity_id: str
	) -> APIResponse:
		"""Upload file in chunks"""
		try:
			# Initialize upload progress
			progress = UploadProgress(
				file_id=metadata.id,
				filename=metadata.original_filename,
				total_bytes=metadata.file_size,
				uploaded_bytes=0,
				percentage=0.0,
				status='uploading'
			)
			self.active_uploads[metadata.id] = progress
			
			# Get API service
			if not self.app or not hasattr(self.app, 'api_service'):
				raise FileException("API service not available")
			
			api_service = self.app.api_service
			
			# Initialize upload session
			init_response = await api_service.post('/files/upload/init', {
				'metadata': metadata.to_dict(),
				'entity_type': entity_type,
				'entity_id': entity_id
			})
			
			if not init_response.success:
				progress.status = 'failed'
				progress.error_message = init_response.message
				return init_response
			
			upload_id = init_response.data.get('upload_id')
			
			# Upload file in chunks
			with open(file_path, 'rb') as file:
				chunk_number = 0
				
				while True:
					chunk = file.read(UPLOAD_CHUNK_SIZE)
					if not chunk:
						break
					
					# Upload chunk
					chunk_response = await self._upload_chunk(
						api_service, upload_id, chunk_number, chunk, metadata
					)
					
					if not chunk_response.success:
						progress.status = 'failed'
						progress.error_message = chunk_response.message
						return chunk_response
					
					# Update progress
					progress.uploaded_bytes += len(chunk)
					progress.percentage = (progress.uploaded_bytes / progress.total_bytes) * 100
					
					# Call progress callback
					if metadata.id in self.upload_callbacks:
						await self._call_progress_callback(metadata.id, progress)
					
					chunk_number += 1
			
			# Finalize upload
			finalize_response = await api_service.post(f'/files/upload/{upload_id}/complete', {
				'checksum': metadata.checksum,
				'total_chunks': chunk_number
			})
			
			if finalize_response.success:
				progress.status = 'completed'
				progress.percentage = 100.0
				
				# Store file locally for caching
				await self._cache_file_locally(file_path, metadata)
			else:
				progress.status = 'failed'
				progress.error_message = finalize_response.message
			
			# Final progress callback
			if metadata.id in self.upload_callbacks:
				await self._call_progress_callback(metadata.id, progress)
			
			return finalize_response
			
		except Exception as e:
			progress.status = 'failed'
			progress.error_message = str(e)
			
			self.logger.error(f"Chunked upload failed: {e}")
			return APIResponse(
				success=False,
				message=f"Upload failed: {e}",
				status_code=500
			)
		
		finally:
			# Cleanup
			if metadata.id in self.active_uploads:
				del self.active_uploads[metadata.id]
			if metadata.id in self.upload_callbacks:
				del self.upload_callbacks[metadata.id]
	
	async def _upload_chunk(
		self,
		api_service,
		upload_id: str,
		chunk_number: int,
		chunk: bytes,
		metadata: FileMetadata
	) -> APIResponse:
		"""Upload individual chunk"""
		
		# Create form data for chunk upload
		form_data = {
			'upload_id': upload_id,
			'chunk_number': chunk_number,
			'chunk_size': len(chunk),
			'file_id': metadata.id
		}
		
		# Upload chunk with retry logic
		max_retries = 3
		for attempt in range(max_retries):
			try:
				response = await api_service.post(
					f'/files/upload/{upload_id}/chunk',
					form_data,
					files={'chunk': chunk}
				)
				
				if response.success:
					return response
				
				# Retry on server errors
				if response.status_code >= 500 and attempt < max_retries - 1:
					await asyncio.sleep(2 ** attempt)  # Exponential backoff
					continue
				
				return response
				
			except Exception as e:
				if attempt < max_retries - 1:
					await asyncio.sleep(2 ** attempt)
					continue
				
				return APIResponse(
					success=False,
					message=f"Chunk upload failed: {e}",
					status_code=500
				)
		
		return APIResponse(
			success=False,
			message="Chunk upload failed after retries",
			status_code=500
		)
	
	async def _call_progress_callback(self, file_id: str, progress: UploadProgress):
		"""Call progress callback if available"""
		try:
			if file_id in self.upload_callbacks:
				callback = self.upload_callbacks[file_id]
				if asyncio.iscoroutinefunction(callback):
					await callback(progress)
				else:
					callback(progress)
		except Exception as e:
			self.logger.warning(f"Progress callback failed: {e}")
	
	async def _cache_file_locally(self, file_path: Path, metadata: FileMetadata):
		"""Cache uploaded file locally"""
		try:
			cache_path = self.cache_dir / metadata.filename
			
			# Copy file to cache
			import shutil
			shutil.copy2(file_path, cache_path)
			
			# Store metadata
			metadata_path = self.cache_dir / f"{metadata.id}.json"
			with open(metadata_path, 'w') as f:
				json.dump(metadata.to_dict(), f, indent=2)
			
			self.logger.debug(f"File cached locally: {cache_path}")
			
		except Exception as e:
			self.logger.warning(f"Failed to cache file locally: {e}")
	
	async def download_file(self, file_id: str, save_path: Optional[Path] = None) -> APIResponse:
		"""Download file from server"""
		try:
			# Check local cache first
			cache_path = await self._get_cached_file_path(file_id)
			if cache_path and cache_path.exists():
				if save_path:
					import shutil
					shutil.copy2(cache_path, save_path)
				
				return APIResponse(
					success=True,
					message="File retrieved from cache",
					data={'file_path': str(save_path or cache_path)}
				)
			
			# Download from server
			if not self.app or not hasattr(self.app, 'api_service'):
				raise FileException("API service not available")
			
			api_service = self.app.api_service
			
			# Get file metadata
			metadata_response = await api_service.get(f'/files/{file_id}/metadata')
			if not metadata_response.success:
				return metadata_response
			
			metadata = FileMetadata.from_dict(metadata_response.data)
			
			# Download file content
			download_response = await api_service.get(f'/files/{file_id}/download', stream=True)
			if not download_response.success:
				return download_response
			
			# Save file
			if not save_path:
				save_path = self.cache_dir / metadata.filename
			
			with open(save_path, 'wb') as f:
				# Stream download in chunks
				async for chunk in download_response.stream():
					f.write(chunk)
			
			# Verify checksum
			downloaded_checksum = await self._calculate_file_checksum(save_path)
			if downloaded_checksum != metadata.checksum:
				save_path.unlink()  # Delete corrupted file
				return APIResponse(
					success=False,
					message="File integrity check failed",
					status_code=500
				)
			
			# Cache metadata
			await self._cache_file_locally(save_path, metadata)
			
			return APIResponse(
				success=True,
				message="File downloaded successfully",
				data={'file_path': str(save_path), 'metadata': metadata.to_dict()}
			)
			
		except Exception as e:
			self.logger.error(f"File download failed: {e}")
			return APIResponse(
				success=False,
				message=f"Download failed: {e}",
				status_code=500
			)
	
	async def _get_cached_file_path(self, file_id: str) -> Optional[Path]:
		"""Get cached file path if available"""
		try:
			metadata_path = self.cache_dir / f"{file_id}.json"
			if not metadata_path.exists():
				return None
			
			with open(metadata_path, 'r') as f:
				metadata_dict = json.load(f)
			
			metadata = FileMetadata.from_dict(metadata_dict)
			cache_path = self.cache_dir / metadata.filename
			
			return cache_path if cache_path.exists() else None
			
		except Exception:
			return None
	
	async def delete_file(self, file_id: str) -> APIResponse:
		"""Delete file from server and local cache"""
		try:
			# Delete from server
			if self.app and hasattr(self.app, 'api_service'):
				api_service = self.app.api_service
				server_response = await api_service.delete(f'/files/{file_id}')
				
				if not server_response.success and server_response.status_code != 404:
					return server_response
			
			# Delete from local cache
			await self._delete_cached_file(file_id)
			
			return APIResponse(
				success=True,
				message="File deleted successfully"
			)
			
		except Exception as e:
			self.logger.error(f"File deletion failed: {e}")
			return APIResponse(
				success=False,
				message=f"Deletion failed: {e}",
				status_code=500
			)
	
	async def _delete_cached_file(self, file_id: str):
		"""Delete file from local cache"""
		try:
			# Delete metadata file
			metadata_path = self.cache_dir / f"{file_id}.json"
			if metadata_path.exists():
				with open(metadata_path, 'r') as f:
					metadata_dict = json.load(f)
				
				metadata = FileMetadata.from_dict(metadata_dict)
				
				# Delete actual file
				cache_path = self.cache_dir / metadata.filename
				if cache_path.exists():
					cache_path.unlink()
				
				# Delete metadata
				metadata_path.unlink()
			
		except Exception as e:
			self.logger.warning(f"Failed to delete cached file: {e}")
	
	async def get_file_metadata(self, file_id: str) -> APIResponse:
		"""Get file metadata"""
		try:
			# Check local cache first
			metadata_path = self.cache_dir / f"{file_id}.json"
			if metadata_path.exists():
				with open(metadata_path, 'r') as f:
					metadata_dict = json.load(f)
				
				return APIResponse(
					success=True,
					message="Metadata retrieved from cache",
					data=metadata_dict
				)
			
			# Get from server
			if not self.app or not hasattr(self.app, 'api_service'):
				raise FileException("API service not available")
			
			api_service = self.app.api_service
			return await api_service.get(f'/files/{file_id}/metadata')
			
		except Exception as e:
			self.logger.error(f"Failed to get file metadata: {e}")
			return APIResponse(
				success=False,
				message=f"Failed to get metadata: {e}",
				status_code=500
			)
	
	async def list_files(
		self,
		entity_type: Optional[str] = None,
		entity_id: Optional[str] = None,
		file_type: Optional[str] = None,
		limit: int = 50,
		offset: int = 0
	) -> APIResponse:
		"""List files with optional filtering"""
		try:
			if not self.app or not hasattr(self.app, 'api_service'):
				raise FileException("API service not available")
			
			api_service = self.app.api_service
			
			params = {
				'limit': limit,
				'offset': offset
			}
			
			if entity_type:
				params['entity_type'] = entity_type
			if entity_id:
				params['entity_id'] = entity_id
			if file_type:
				params['file_type'] = file_type
			
			return await api_service.get('/files', params=params)
			
		except Exception as e:
			self.logger.error(f"Failed to list files: {e}")
			return APIResponse(
				success=False,
				message=f"Failed to list files: {e}",
				status_code=500
			)
	
	async def get_upload_progress(self, file_id: str) -> Optional[UploadProgress]:
		"""Get upload progress for file"""
		return self.active_uploads.get(file_id)
	
	async def cancel_upload(self, file_id: str) -> bool:
		"""Cancel ongoing upload"""
		try:
			if file_id in self.active_uploads:
				progress = self.active_uploads[file_id]
				progress.status = 'cancelled'
				
				# Notify callback
				if file_id in self.upload_callbacks:
					await self._call_progress_callback(file_id, progress)
				
				return True
			
			return False
			
		except Exception as e:
			self.logger.error(f"Failed to cancel upload: {e}")
			return False
	
	async def clear_cache(self, older_than_days: int = 30) -> Dict[str, Any]:
		"""Clear old cached files"""
		try:
			cutoff_time = datetime.utcnow() - timedelta(days=older_than_days)
			deleted_files = 0
			freed_space = 0
			
			for file_path in self.cache_dir.iterdir():
				if file_path.is_file():
					file_stat = file_path.stat()
					file_time = datetime.fromtimestamp(file_stat.st_mtime)
					
					if file_time < cutoff_time:
						file_size = file_stat.st_size
						file_path.unlink()
						deleted_files += 1
						freed_space += file_size
			
			result = {
				'deleted_files': deleted_files,
				'freed_space_bytes': freed_space,
				'cutoff_date': cutoff_time.isoformat()
			}
			
			self.logger.info(f"Cache cleared: {result}")
			return result
			
		except Exception as e:
			self.logger.error(f"Failed to clear cache: {e}")
			return {'error': str(e)}
	
	def get_cache_statistics(self) -> Dict[str, Any]:
		"""Get cache statistics"""
		try:
			total_files = 0
			total_size = 0
			
			for file_path in self.cache_dir.iterdir():
				if file_path.is_file() and not file_path.name.endswith('.json'):
					total_files += 1
					total_size += file_path.stat().st_size
			
			return {
				'total_files': total_files,
				'total_size_bytes': total_size,
				'cache_directory': str(self.cache_dir),
				'active_uploads': len(self.active_uploads)
			}
			
		except Exception as e:
			self.logger.error(f"Failed to get cache statistics: {e}")
			return {'error': str(e)}