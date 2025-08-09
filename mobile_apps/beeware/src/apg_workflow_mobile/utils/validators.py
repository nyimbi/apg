"""
Data validation utilities

© 2025 Datacraft. All rights reserved.
"""

import re
import datetime
from typing import Optional, List, Dict, Any, Union
from email_validator import validate_email as _validate_email, EmailNotValidError
from urllib.parse import urlparse

from .exceptions import ValidationException


def validate_email(email: str) -> tuple[bool, Optional[str]]:
	"""Validate email address format"""
	try:
		# Use email-validator library for comprehensive validation
		valid_email = _validate_email(email)
		return True, valid_email.email
	except EmailNotValidError as e:
		return False, str(e)


def validate_username(username: str) -> tuple[bool, Optional[str]]:
	"""Validate username format and constraints"""
	if not username:
		return False, "Username is required"
	
	if len(username) < 3:
		return False, "Username must be at least 3 characters long"
	
	if len(username) > 50:
		return False, "Username must be less than 50 characters"
	
	# Check for valid characters (alphanumeric, underscore, hyphen)
	if not re.match(r'^[a-zA-Z0-9_-]+$', username):
		return False, "Username can only contain letters, numbers, underscores, and hyphens"
	
	# Must start with a letter or number
	if not re.match(r'^[a-zA-Z0-9]', username):
		return False, "Username must start with a letter or number"
	
	# Cannot end with underscore or hyphen
	if username.endswith(('_', '-')):
		return False, "Username cannot end with underscore or hyphen"
	
	return True, None


def validate_password(password: str) -> tuple[bool, List[str]]:
	"""Validate password strength"""
	errors = []
	
	if not password:
		errors.append("Password is required")
		return False, errors
	
	if len(password) < 8:
		errors.append("Password must be at least 8 characters long")
	
	if len(password) > 128:
		errors.append("Password must be less than 128 characters")
	
	# Check for at least one lowercase letter
	if not re.search(r'[a-z]', password):
		errors.append("Password must contain at least one lowercase letter")
	
	# Check for at least one uppercase letter
	if not re.search(r'[A-Z]', password):
		errors.append("Password must contain at least one uppercase letter")
	
	# Check for at least one digit
	if not re.search(r'\d', password):
		errors.append("Password must contain at least one number")
	
	# Check for at least one special character
	if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
		errors.append("Password must contain at least one special character")
	
	# Check for common weak passwords
	weak_passwords = [
		'password', '12345678', 'qwerty', 'abc123', 'password123',
		'admin', 'letmein', 'welcome', 'monkey', '123456789'
	]
	
	if password.lower() in weak_passwords:
		errors.append("Password is too common, please choose a stronger password")
	
	return len(errors) == 0, errors


def validate_phone_number(phone: str, country_code: Optional[str] = None) -> tuple[bool, Optional[str]]:
	"""Validate phone number format"""
	if not phone:
		return False, "Phone number is required"
	
	# Remove all non-digit characters except +
	cleaned = re.sub(r'[^\d+]', '', phone)
	
	# Basic validation
	if not cleaned:
		return False, "Invalid phone number format"
	
	# Check if it starts with +
	if cleaned.startswith('+'):
		if len(cleaned) < 8 or len(cleaned) > 16:
			return False, "International phone number must be 8-15 digits after country code"
	else:
		# Assume local number, add validation based on country code
		if country_code == "US" or country_code == "CA":
			# North American numbering plan
			if len(cleaned) != 10:
				return False, "US/Canada phone number must be 10 digits"
			if not re.match(r'^[2-9]\d{2}[2-9]\d{6}$', cleaned):
				return False, "Invalid US/Canada phone number format"
		else:
			# Generic validation
			if len(cleaned) < 7 or len(cleaned) > 15:
				return False, "Phone number must be 7-15 digits"
	
	return True, None


def validate_url(url: str, require_https: bool = False) -> tuple[bool, Optional[str]]:
	"""Validate URL format"""
	if not url:
		return False, "URL is required"
	
	try:
		parsed = urlparse(url)
		
		if not parsed.scheme:
			return False, "URL must include protocol (http:// or https://)"
		
		if not parsed.netloc:
			return False, "URL must include domain name"
		
		if require_https and parsed.scheme != 'https':
			return False, "URL must use HTTPS protocol"
		
		if parsed.scheme not in ['http', 'https', 'ftp', 'ftps']:
			return False, "URL protocol must be http, https, ftp, or ftps"
		
		return True, None
		
	except Exception as e:
		return False, f"Invalid URL format: {e}"


def validate_workflow_name(name: str) -> tuple[bool, Optional[str]]:
	"""Validate workflow name"""
	if not name:
		return False, "Workflow name is required"
	
	name = name.strip()
	
	if len(name) < 3:
		return False, "Workflow name must be at least 3 characters long"
	
	if len(name) > 200:
		return False, "Workflow name must be less than 200 characters"
	
	# Check for valid characters (allow letters, numbers, spaces, and common punctuation)
	if not re.match(r'^[a-zA-Z0-9\s\-_.,()]+$', name):
		return False, "Workflow name contains invalid characters"
	
	# Must start and end with alphanumeric character
	if not re.match(r'^[a-zA-Z0-9].*[a-zA-Z0-9]$', name) and len(name) > 1:
		return False, "Workflow name must start and end with a letter or number"
	
	return True, None


def validate_task_name(name: str) -> tuple[bool, Optional[str]]:
	"""Validate task name"""
	if not name:
		return False, "Task name is required"
	
	name = name.strip()
	
	if len(name) < 3:
		return False, "Task name must be at least 3 characters long"
	
	if len(name) > 200:
		return False, "Task name must be less than 200 characters"
	
	# Check for valid characters
	if not re.match(r'^[a-zA-Z0-9\s\-_.,()]+$', name):
		return False, "Task name contains invalid characters"
	
	return True, None


def validate_date_string(date_string: str, date_format: str = "%Y-%m-%d") -> tuple[bool, Optional[str]]:
	"""Validate date string format"""
	if not date_string:
		return False, "Date is required"
	
	try:
		datetime.datetime.strptime(date_string, date_format)
		return True, None
	except ValueError:
		return False, f"Invalid date format. Expected format: {date_format}"


def validate_datetime_string(datetime_string: str) -> tuple[bool, Optional[str]]:
	"""Validate ISO datetime string format"""
	if not datetime_string:
		return False, "DateTime is required"
	
	try:
		datetime.datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
		return True, None
	except ValueError:
		return False, "Invalid datetime format. Expected ISO format (YYYY-MM-DDTHH:MM:SS)"


def validate_json(json_string: str) -> tuple[bool, Optional[str]]:
	"""Validate JSON string format"""
	if not json_string:
		return False, "JSON is required"
	
	try:
		import json
		json.loads(json_string)
		return True, None
	except json.JSONDecodeError as e:
		return False, f"Invalid JSON format: {e}"


def validate_file_size(file_size: int, max_size_mb: int = 100) -> tuple[bool, Optional[str]]:
	"""Validate file size"""
	if file_size <= 0:
		return False, "File size must be greater than 0"
	
	max_size_bytes = max_size_mb * 1024 * 1024
	
	if file_size > max_size_bytes:
		return False, f"File size must be less than {max_size_mb}MB"
	
	return True, None


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> tuple[bool, Optional[str]]:
	"""Validate file extension"""
	if not filename:
		return False, "Filename is required"
	
	# Extract extension
	if '.' not in filename:
		return False, "File must have an extension"
	
	extension = '.' + filename.split('.')[-1].lower()
	
	if extension not in [ext.lower() for ext in allowed_extensions]:
		return False, f"File extension must be one of: {', '.join(allowed_extensions)}"
	
	return True, None


def validate_tenant_id(tenant_id: str) -> tuple[bool, Optional[str]]:
	"""Validate tenant ID format"""
	if not tenant_id:
		return False, "Tenant ID is required"
	
	tenant_id = tenant_id.strip()
	
	if len(tenant_id) < 3:
		return False, "Tenant ID must be at least 3 characters long"
	
	if len(tenant_id) > 50:
		return False, "Tenant ID must be less than 50 characters"
	
	# Check for valid characters (alphanumeric, underscore, hyphen)
	if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
		return False, "Tenant ID can only contain letters, numbers, underscores, and hyphens"
	
	return True, None


def validate_uuid(uuid_string: str) -> tuple[bool, Optional[str]]:
	"""Validate UUID format"""
	if not uuid_string:
		return False, "UUID is required"
	
	# UUID pattern: 8-4-4-4-12 hexadecimal digits
	uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
	
	if not re.match(uuid_pattern, uuid_string.lower()):
		return False, "Invalid UUID format"
	
	return True, None


def validate_port_number(port: Union[int, str]) -> tuple[bool, Optional[str]]:
	"""Validate port number"""
	try:
		port_num = int(port)
		
		if port_num < 1 or port_num > 65535:
			return False, "Port number must be between 1 and 65535"
		
		return True, None
		
	except (ValueError, TypeError):
		return False, "Port number must be a valid integer"


def validate_ip_address(ip: str) -> tuple[bool, Optional[str]]:
	"""Validate IP address (IPv4 or IPv6)"""
	if not ip:
		return False, "IP address is required"
	
	import ipaddress
	
	try:
		ipaddress.ip_address(ip)
		return True, None
	except ValueError:
		return False, "Invalid IP address format"


def validate_mac_address(mac: str) -> tuple[bool, Optional[str]]:
	"""Validate MAC address format"""
	if not mac:
		return False, "MAC address is required"
	
	# Remove common separators and convert to lowercase
	mac_clean = mac.replace(':', '').replace('-', '').replace('.', '').lower()
	
	# Check if it's exactly 12 hexadecimal characters
	if len(mac_clean) != 12 or not re.match(r'^[0-9a-f]{12}$', mac_clean):
		return False, "Invalid MAC address format"
	
	return True, None


def validate_cron_expression(cron: str) -> tuple[bool, Optional[str]]:
	"""Validate cron expression format"""
	if not cron:
		return False, "Cron expression is required"
	
	parts = cron.split()
	
	# Standard cron has 5 fields: minute hour day month weekday
	# Extended cron has 6 fields: second minute hour day month weekday
	if len(parts) not in [5, 6]:
		return False, "Cron expression must have 5 or 6 fields"
	
	# Basic validation for each field
	field_ranges = [
		(0, 59),  # second (if 6 fields)
		(0, 59),  # minute
		(0, 23),  # hour
		(1, 31),  # day
		(1, 12),  # month
		(0, 7),   # weekday (0 and 7 are Sunday)
	]
	
	start_idx = 0 if len(parts) == 6 else 1
	
	for i, part in enumerate(parts):
		field_range = field_ranges[start_idx + i]
		
		# Skip validation for special characters
		if part in ['*', '?']:
			continue
		
		# Handle ranges (e.g., 1-5)
		if '-' in part:
			try:
				start, end = map(int, part.split('-'))
				if start < field_range[0] or end > field_range[1] or start > end:
					return False, f"Invalid range in field {i + 1}: {part}"
			except ValueError:
				return False, f"Invalid range format in field {i + 1}: {part}"
			continue
		
		# Handle lists (e.g., 1,3,5)
		if ',' in part:
			try:
				values = [int(x) for x in part.split(',')]
				for value in values:
					if value < field_range[0] or value > field_range[1]:
						return False, f"Invalid value in field {i + 1}: {value}"
			except ValueError:
				return False, f"Invalid list format in field {i + 1}: {part}"
			continue
		
		# Handle step values (e.g., */5)
		if '/' in part:
			base, step = part.split('/')
			try:
				step_val = int(step)
				if step_val <= 0:
					return False, f"Invalid step value in field {i + 1}: {step}"
			except ValueError:
				return False, f"Invalid step format in field {i + 1}: {part}"
			continue
		
		# Handle single values
		try:
			value = int(part)
			if value < field_range[0] or value > field_range[1]:
				return False, f"Value out of range in field {i + 1}: {value}"
		except ValueError:
			return False, f"Invalid value in field {i + 1}: {part}"
	
	return True, None


def validate_workflow_definition(definition: Dict[str, Any]) -> tuple[bool, List[str]]:
	"""Validate workflow definition structure"""
	errors = []
	
	# Required fields
	required_fields = ['version', 'tasks']
	for field in required_fields:
		if field not in definition:
			errors.append(f"Missing required field: {field}")
	
	# Validate version
	if 'version' in definition:
		if not isinstance(definition['version'], str):
			errors.append("Version must be a string")
	
	# Validate tasks
	if 'tasks' in definition:
		if not isinstance(definition['tasks'], list):
			errors.append("Tasks must be a list")
		else:
			for i, task in enumerate(definition['tasks']):
				if not isinstance(task, dict):
					errors.append(f"Task {i} must be an object")
					continue
				
				# Required task fields
				task_required = ['id', 'name', 'type']
				for field in task_required:
					if field not in task:
						errors.append(f"Task {i} missing required field: {field}")
	
	# Validate triggers if present
	if 'triggers' in definition:
		if not isinstance(definition['triggers'], list):
			errors.append("Triggers must be a list")
	
	# Validate variables if present
	if 'variables' in definition:
		if not isinstance(definition['variables'], dict):
			errors.append("Variables must be an object")
	
	return len(errors) == 0, errors


class ValidationError(ValidationException):
	"""Exception raised when validation fails"""
	
	def __init__(self, message: str, field: Optional[str] = None, errors: Optional[List[str]] = None):
		super().__init__(message)
		self.field = field
		self.errors = errors or []


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
	"""Validate that all required fields are present"""
	missing_fields = []
	
	for field in required_fields:
		if field not in data or data[field] is None:
			missing_fields.append(field)
	
	if missing_fields:
		raise ValidationError(
			f"Missing required fields: {', '.join(missing_fields)}",
			errors=missing_fields
		)


def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> None:
	"""Validate field types"""
	type_errors = []
	
	for field, expected_type in field_types.items():
		if field in data and data[field] is not None:
			if not isinstance(data[field], expected_type):
				type_errors.append(f"{field} must be of type {expected_type.__name__}")
	
	if type_errors:
		raise ValidationError(
			f"Type validation failed: {'; '.join(type_errors)}",
			errors=type_errors
		)