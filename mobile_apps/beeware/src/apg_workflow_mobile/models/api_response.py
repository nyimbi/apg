"""
API Response models and utilities

© 2025 Datacraft. All rights reserved.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Generic, TypeVar
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import uuid

T = TypeVar('T')


@dataclass
class PaginationInfo:
	"""Pagination information for API responses"""
	page: int = 1
	limit: int = 20
	total: int = 0
	total_pages: int = 0
	has_next: bool = False
	has_previous: bool = False
	
	def __post_init__(self):
		"""Calculate derived fields"""
		if self.total > 0 and self.limit > 0:
			self.total_pages = ((self.total - 1) // self.limit) + 1
			self.has_next = self.page < self.total_pages
			self.has_previous = self.page > 1
		else:
			self.total_pages = 0
			self.has_next = False
			self.has_previous = False
	
	@property
	def offset(self) -> int:
		"""Get offset for current page"""
		return (self.page - 1) * self.limit
	
	@property
	def items_on_page(self) -> int:
		"""Get number of items on current page"""
		if self.page < self.total_pages:
			return self.limit
		elif self.page == self.total_pages:
			return self.total - self.offset
		else:
			return 0
	
	@property
	def progress_percentage(self) -> float:
		"""Get progress through total items as percentage"""
		if self.total == 0:
			return 0.0
		return min(100.0, (self.offset + self.items_on_page) / self.total * 100)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			"page": self.page,
			"limit": self.limit,
			"total": self.total,
			"total_pages": self.total_pages,
			"has_next": self.has_next,
			"has_previous": self.has_previous,
			"offset": self.offset,
			"items_on_page": self.items_on_page,
			"progress_percentage": self.progress_percentage,
		}


class APIError:
	"""API error information"""
	def __init__(self, code: str = "", message: str = "", field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
		self.code = code
		self.message = message
		self.field = field
		self.details = details or {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			"code": self.code,
			"message": self.message,
			"field": self.field,
			"details": self.details,
		}


class APIMetadata:
	"""API response metadata"""
	def __init__(self, request_id: Optional[str] = None, timestamp: Optional[datetime] = None, 
				 processing_time: Optional[float] = None, api_version: Optional[str] = None,
				 rate_limit_remaining: Optional[int] = None, rate_limit_reset: Optional[datetime] = None,
				 server_info: Optional[Dict[str, Any]] = None):
		self.request_id = request_id
		self.timestamp = timestamp
		self.processing_time = processing_time
		self.api_version = api_version
		self.rate_limit_remaining = rate_limit_remaining
		self.rate_limit_reset = rate_limit_reset
		self.server_info = server_info or {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			"request_id": self.request_id,
			"timestamp": self.timestamp.isoformat() if self.timestamp else None,
			"processing_time": self.processing_time,
			"api_version": self.api_version,
			"rate_limit_remaining": self.rate_limit_remaining,
			"rate_limit_reset": self.rate_limit_reset.isoformat() if self.rate_limit_reset else None,
			"server_info": self.server_info,
		}


class APIResponse(Generic[T]):
	"""Generic API response wrapper"""
	
	def __init__(
		self,
		success: bool = True,
		status_code: int = 200,
		message: Optional[str] = None,
		data: Optional[T] = None,
		errors: Optional[List[APIError]] = None,
		pagination: Optional[PaginationInfo] = None,
		metadata: Optional[APIMetadata] = None,
		headers: Optional[Dict[str, str]] = None,
		raw_response: Optional[str] = None
	):
		self.success = success
		self.status_code = status_code
		self.message = message
		self.data = data
		self.errors = errors or []
		self.pagination = pagination
		self.metadata = metadata or APIMetadata()
		self.headers = headers or {}
		self.raw_response = raw_response
		self.created_at = datetime.utcnow()
	
	@property
	def is_success(self) -> bool:
		"""Check if response is successful"""
		return self.success and 200 <= self.status_code < 300
	
	@property
	def is_client_error(self) -> bool:
		"""Check if response is client error (4xx)"""
		return 400 <= self.status_code < 500
	
	@property
	def is_server_error(self) -> bool:
		"""Check if response is server error (5xx)"""
		return 500 <= self.status_code < 600
	
	@property
	def has_errors(self) -> bool:
		"""Check if response has errors"""
		return len(self.errors) > 0
	
	@property
	def has_data(self) -> bool:
		"""Check if response has data"""
		return self.data is not None
	
	@property
	def has_pagination(self) -> bool:
		"""Check if response has pagination"""
		return self.pagination is not None
	
	@property
	def error_messages(self) -> List[str]:
		"""Get list of error messages"""
		return [error.message for error in self.errors]
	
	@property
	def first_error(self) -> Optional[APIError]:
		"""Get first error if any"""
		return self.errors[0] if self.errors else None
	
	@property
	def first_error_message(self) -> Optional[str]:
		"""Get first error message if any"""
		first = self.first_error
		return first.message if first else None
	
	def add_error(self, code: str, message: str, field: Optional[str] = None, 
				  details: Optional[Dict[str, Any]] = None):
		"""Add error to response"""
		error = APIError(
			code=code,
			message=message,
			field=field,
			details=details or {}
		)
		self.errors.append(error)
		self.success = False
	
	def get_errors_by_field(self, field: str) -> List[APIError]:
		"""Get errors for specific field"""
		return [error for error in self.errors if error.field == field]
	
	def get_field_error_messages(self, field: str) -> List[str]:
		"""Get error messages for specific field"""
		return [error.message for error in self.get_errors_by_field(field)]
	
	def has_field_errors(self, field: str) -> bool:
		"""Check if field has errors"""
		return len(self.get_errors_by_field(field)) > 0
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert response to dictionary"""
		result = {
			"success": self.success,
			"status_code": self.status_code,
			"message": self.message,
			"is_success": self.is_success,
			"is_client_error": self.is_client_error,
			"is_server_error": self.is_server_error,
			"has_errors": self.has_errors,
			"has_data": self.has_data,
			"has_pagination": self.has_pagination,
			"created_at": self.created_at.isoformat(),
		}
		
		# Add data if present
		if self.data is not None:
			if hasattr(self.data, 'to_dict'):
				result["data"] = self.data.to_dict()
			elif hasattr(self.data, '__dict__'):
				result["data"] = self.data.__dict__
			else:
				result["data"] = self.data
		
		# Add errors if present
		if self.errors:
			result["errors"] = [error.to_dict() for error in self.errors]
			result["error_messages"] = self.error_messages
		
		# Add pagination if present
		if self.pagination:
			result["pagination"] = self.pagination.to_dict()
		
		# Add metadata if present
		if self.metadata:
			result["metadata"] = self.metadata.to_dict()
		
		# Add headers
		if self.headers:
			result["headers"] = self.headers
		
		return result
	
	@classmethod
	def success_response(
		cls,
		data: Optional[T] = None,
		message: Optional[str] = None,
		pagination: Optional[PaginationInfo] = None,
		metadata: Optional[APIMetadata] = None
	) -> "APIResponse[T]":
		"""Create successful response"""
		return cls(
			success=True,
			status_code=200,
			message=message,
			data=data,
			pagination=pagination,
			metadata=metadata
		)
	
	@classmethod
	def error_response(
		cls,
		message: str,
		status_code: int = 400,
		errors: Optional[List[APIError]] = None,
		metadata: Optional[APIMetadata] = None
	) -> "APIResponse[T]":
		"""Create error response"""
		return cls(
			success=False,
			status_code=status_code,
			message=message,
			errors=errors,
			metadata=metadata
		)
	
	@classmethod
	def validation_error_response(
		cls,
		message: str = "Validation failed",
		field_errors: Optional[Dict[str, List[str]]] = None
	) -> "APIResponse[T]":
		"""Create validation error response"""
		errors = []
		if field_errors:
			for field, messages in field_errors.items():
				for msg in messages:
					errors.append(APIError(
						code="VALIDATION_ERROR",
						message=msg,
						field=field
					))
		
		return cls(
			success=False,
			status_code=422,
			message=message,
			errors=errors
		)
	
	@classmethod
	def unauthorized_response(
		cls,
		message: str = "Authentication required"
	) -> "APIResponse[T]":
		"""Create unauthorized response"""
		return cls(
			success=False,
			status_code=401,
			message=message,
			errors=[APIError(code="UNAUTHORIZED", message=message)]
		)
	
	@classmethod
	def forbidden_response(
		cls,
		message: str = "Access forbidden"
	) -> "APIResponse[T]":
		"""Create forbidden response"""
		return cls(
			success=False,
			status_code=403,
			message=message,
			errors=[APIError(code="FORBIDDEN", message=message)]
		)
	
	@classmethod
	def not_found_response(
		cls,
		message: str = "Resource not found"
	) -> "APIResponse[T]":
		"""Create not found response"""
		return cls(
			success=False,
			status_code=404,
			message=message,
			errors=[APIError(code="NOT_FOUND", message=message)]
		)
	
	@classmethod
	def server_error_response(
		cls,
		message: str = "Internal server error",
		details: Optional[Dict[str, Any]] = None
	) -> "APIResponse[T]":
		"""Create server error response"""
		return cls(
			success=False,
			status_code=500,
			message=message,
			errors=[APIError(
				code="INTERNAL_SERVER_ERROR",
				message=message,
				details=details or {}
			)]
		)


# Type aliases for common response types
WorkflowResponse = APIResponse[Dict[str, Any]]
TaskResponse = APIResponse[Dict[str, Any]]
UserResponse = APIResponse[Dict[str, Any]]
NotificationResponse = APIResponse[Dict[str, Any]]
ListResponse = APIResponse[List[Dict[str, Any]]]
BoolResponse = APIResponse[bool]
StringResponse = APIResponse[str]
EmptyResponse = APIResponse[None]


class BatchResponse:
	"""Response for batch operations"""
	def __init__(self, total_items: int = 0, successful_items: int = 0, failed_items: int = 0,
				 responses: Optional[List['APIResponse']] = None, errors: Optional[List[APIError]] = None):
		self.total_items = total_items
		self.successful_items = successful_items
		self.failed_items = failed_items
		self.responses = responses or []
		self.errors = errors or []
	
	@property
	def success_rate(self) -> float:
		"""Get success rate as percentage"""
		if self.total_items == 0:
			return 0.0
		return (self.successful_items / self.total_items) * 100
	
	@property
	def is_partial_success(self) -> bool:
		"""Check if batch had partial success"""
		return 0 < self.successful_items < self.total_items
	
	@property
	def is_complete_success(self) -> bool:
		"""Check if batch was completely successful"""
		return self.successful_items == self.total_items and self.total_items > 0
	
	@property
	def is_complete_failure(self) -> bool:
		"""Check if batch completely failed"""
		return self.successful_items == 0 and self.total_items > 0
	
	def add_response(self, response: APIResponse):
		"""Add individual response to batch"""
		self.responses.append(response)
		self.total_items += 1
		
		if response.is_success:
			self.successful_items += 1
		else:
			self.failed_items += 1
			self.errors.extend(response.errors)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			"total_items": self.total_items,
			"successful_items": self.successful_items,
			"failed_items": self.failed_items,
			"success_rate": self.success_rate,
			"is_partial_success": self.is_partial_success,
			"is_complete_success": self.is_complete_success,
			"is_complete_failure": self.is_complete_failure,
			"responses": [resp.to_dict() for resp in self.responses],
			"errors": [error.to_dict() for error in self.errors],
		}


class ResponseCache:
	"""Simple response caching mechanism"""
	
	def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
		self.max_size = max_size
		self.ttl_seconds = ttl_seconds
		self._cache: Dict[str, tuple[APIResponse, datetime]] = {}
	
	def get(self, key: str) -> Optional[APIResponse]:
		"""Get cached response"""
		if key in self._cache:
			response, cached_at = self._cache[key]
			
			# Check if expired
			if datetime.utcnow() - cached_at > timedelta(seconds=self.ttl_seconds):
				del self._cache[key]
				return None
			
			return response
		
		return None
	
	def set(self, key: str, response: APIResponse):
		"""Cache response"""
		# Evict oldest if at capacity
		if len(self._cache) >= self.max_size:
			oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
			del self._cache[oldest_key]
		
		self._cache[key] = (response, datetime.utcnow())
	
	def clear(self):
		"""Clear all cached responses"""
		self._cache.clear()
	
	def remove(self, key: str):
		"""Remove specific cached response"""
		self._cache.pop(key, None)
	
	@property
	def size(self) -> int:
		"""Get current cache size"""
		return len(self._cache)
	
	def cleanup_expired(self):
		"""Remove expired cache entries"""
		now = datetime.utcnow()
		expired_keys = [
			key for key, (_, cached_at) in self._cache.items()
			if now - cached_at > timedelta(seconds=self.ttl_seconds)
		]
		
		for key in expired_keys:
			del self._cache[key]