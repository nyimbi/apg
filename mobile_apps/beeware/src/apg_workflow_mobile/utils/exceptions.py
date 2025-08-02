"""
Custom exceptions for APG Workflow Mobile

© 2025 Datacraft. All rights reserved.
"""

from typing import Optional, Dict, Any


class APGException(Exception):
	"""Base exception for APG Workflow Mobile"""
	
	def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
		super().__init__(message)
		self.message = message
		self.details = details or {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		return {
			"type": self.__class__.__name__,
			"message": self.message,
			"details": self.details
		}


class APIException(APGException):
	"""Exception for API-related errors"""
	
	def __init__(self, message: str, status_code: Optional[int] = None, 
				 response_data: Optional[Dict[str, Any]] = None):
		super().__init__(message)
		self.status_code = status_code
		self.response_data = response_data or {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"status_code": self.status_code,
			"response_data": self.response_data
		})
		return result


class AuthenticationException(APGException):
	"""Exception for authentication-related errors"""
	
	def __init__(self, message: str, auth_required: bool = True, 
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.auth_required = auth_required
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result["auth_required"] = self.auth_required
		return result


class ValidationException(APGException):
	"""Exception for data validation errors"""
	
	def __init__(self, message: str, field_errors: Optional[Dict[str, List[str]]] = None,
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.field_errors = field_errors or {}
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result["field_errors"] = self.field_errors
		return result


class NetworkException(APGException):
	"""Exception for network-related errors"""
	
	def __init__(self, message: str, is_connectivity_issue: bool = True,
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.is_connectivity_issue = is_connectivity_issue
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result["is_connectivity_issue"] = self.is_connectivity_issue
		return result


class OfflineException(APGException):
	"""Exception for offline mode errors"""
	
	def __init__(self, message: str, operation: Optional[str] = None,
				 can_queue: bool = False, details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.operation = operation
		self.can_queue = can_queue
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"operation": self.operation,
			"can_queue": self.can_queue
		})
		return result


class BiometricException(APGException):
	"""Exception for biometric authentication errors"""
	
	def __init__(self, message: str, error_code: Optional[str] = None,
				 is_recoverable: bool = True, details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.error_code = error_code
		self.is_recoverable = is_recoverable
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"error_code": self.error_code,
			"is_recoverable": self.is_recoverable
		})
		return result


class FileException(APGException):
	"""Exception for file operation errors"""
	
	def __init__(self, message: str, file_path: Optional[str] = None,
				 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.file_path = file_path
		self.operation = operation
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"file_path": self.file_path,
			"operation": self.operation
		})
		return result


class SyncException(APGException):
	"""Exception for synchronization errors"""
	
	def __init__(self, message: str, sync_type: Optional[str] = None,
				 failed_items: Optional[List[str]] = None, 
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.sync_type = sync_type
		self.failed_items = failed_items or []
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"sync_type": self.sync_type,
			"failed_items": self.failed_items
		})
		return result


class WorkflowException(APGException):
	"""Exception for workflow execution errors"""
	
	def __init__(self, message: str, workflow_id: Optional[str] = None,
				 instance_id: Optional[str] = None, task_id: Optional[str] = None,
				 is_recoverable: bool = True, details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.workflow_id = workflow_id
		self.instance_id = instance_id
		self.task_id = task_id
		self.is_recoverable = is_recoverable
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"workflow_id": self.workflow_id,
			"instance_id": self.instance_id,
			"task_id": self.task_id,
			"is_recoverable": self.is_recoverable
		})
		return result


class ConfigurationException(APGException):
	"""Exception for configuration errors"""
	
	def __init__(self, message: str, config_key: Optional[str] = None,
				 config_section: Optional[str] = None, 
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.config_key = config_key
		self.config_section = config_section
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"config_key": self.config_key,
			"config_section": self.config_section
		})
		return result


class PermissionException(APGException):
	"""Exception for permission-related errors"""
	
	def __init__(self, message: str, required_permission: Optional[str] = None,
				 resource_type: Optional[str] = None, resource_id: Optional[str] = None,
				 details: Optional[Dict[str, Any]] = None):
		super().__init__(message, details)
		self.required_permission = required_permission
		self.resource_type = resource_type
		self.resource_id = resource_id
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert exception to dictionary"""
		result = super().to_dict()
		result.update({
			"required_permission": self.required_permission,
			"resource_type": self.resource_type,
			"resource_id": self.resource_id
		})
		return result


# Error code mappings for different exception types
ERROR_CODES = {
	# API Errors (1000-1999)
	"API_GENERAL": 1000,
	"API_TIMEOUT": 1001,
	"API_CONNECTION": 1002,
	"API_INVALID_RESPONSE": 1003,
	"API_RATE_LIMITED": 1004,
	
	# Authentication Errors (2000-2999)
	"AUTH_INVALID_CREDENTIALS": 2000,
	"AUTH_TOKEN_EXPIRED": 2001,
	"AUTH_TOKEN_INVALID": 2002,
	"AUTH_INSUFFICIENT_PERMISSIONS": 2003,
	"AUTH_ACCOUNT_LOCKED": 2004,
	"AUTH_ACCOUNT_DISABLED": 2005,
	
	# Validation Errors (3000-3999)
	"VALIDATION_REQUIRED_FIELD": 3000,
	"VALIDATION_INVALID_FORMAT": 3001,
	"VALIDATION_VALUE_TOO_LONG": 3002,
	"VALIDATION_VALUE_TOO_SHORT": 3003,
	"VALIDATION_INVALID_RANGE": 3004,
	
	# Network Errors (4000-4999)
	"NETWORK_UNREACHABLE": 4000,
	"NETWORK_TIMEOUT": 4001,
	"NETWORK_DNS_FAILED": 4002,
	"NETWORK_SSL_ERROR": 4003,
	
	# Offline Errors (5000-5999)
	"OFFLINE_MODE_ACTIVE": 5000,
	"OFFLINE_DATA_UNAVAILABLE": 5001,
	"OFFLINE_SYNC_FAILED": 5002,
	"OFFLINE_STORAGE_FULL": 5003,
	
	# Biometric Errors (6000-6999)
	"BIOMETRIC_NOT_AVAILABLE": 6000,
	"BIOMETRIC_NOT_ENROLLED": 6001,
	"BIOMETRIC_AUTHENTICATION_FAILED": 6002,
	"BIOMETRIC_SENSOR_ERROR": 6003,
	"BIOMETRIC_LOCKOUT": 6004,
	
	# File Errors (7000-7999)
	"FILE_NOT_FOUND": 7000,
	"FILE_ACCESS_DENIED": 7001,
	"FILE_TOO_LARGE": 7002,
	"FILE_INVALID_FORMAT": 7003,
	"FILE_UPLOAD_FAILED": 7004,
	"FILE_DOWNLOAD_FAILED": 7005,
	
	# Workflow Errors (8000-8999)
	"WORKFLOW_NOT_FOUND": 8000,
	"WORKFLOW_EXECUTION_FAILED": 8001,
	"WORKFLOW_INVALID_STATE": 8002,
	"WORKFLOW_PERMISSION_DENIED": 8003,
	"WORKFLOW_TIMEOUT": 8004,
	
	# Configuration Errors (9000-9999)
	"CONFIG_MISSING": 9000,
	"CONFIG_INVALID": 9001,
	"CONFIG_ACCESS_DENIED": 9002,
}


def get_error_code(exception_type: str, specific_error: str = "GENERAL") -> int:
	"""Get error code for exception type and specific error"""
	error_key = f"{exception_type}_{specific_error}"
	return ERROR_CODES.get(error_key, ERROR_CODES.get(f"{exception_type}_GENERAL", 0))