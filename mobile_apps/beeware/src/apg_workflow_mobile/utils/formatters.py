"""
Data formatting utilities

© 2025 Datacraft. All rights reserved.
"""

import datetime
from typing import Optional, Union, Any, Dict, List
import json
import re
from decimal import Decimal


def format_datetime(
	dt: Union[datetime.datetime, str, float, int],
	format_string: Optional[str] = None,
	timezone: Optional[str] = None,
	relative: bool = False
) -> str:
	"""Format datetime for display"""
	
	# Convert input to datetime object
	if isinstance(dt, str):
		try:
			# Try ISO format first
			dt_obj = datetime.datetime.fromisoformat(dt.replace('Z', '+00:00'))
		except ValueError:
			# Try other common formats
			for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
				try:
					dt_obj = datetime.datetime.strptime(dt, fmt)
					break
				except ValueError:
					continue
			else:
				return dt  # Return original string if parsing fails
	elif isinstance(dt, (int, float)):
		dt_obj = datetime.datetime.fromtimestamp(dt)
	elif isinstance(dt, datetime.datetime):
		dt_obj = dt
	else:
		return str(dt)
	
	# Format relative time if requested
	if relative:
		return format_relative_time(dt_obj)
	
	# Apply timezone conversion if specified
	if timezone:
		# Note: In a real implementation, you'd use pytz or zoneinfo
		# For now, we'll use the datetime as-is
		pass
	
	# Use custom format string or default
	if format_string:
		return dt_obj.strftime(format_string)
	else:
		# Default format based on time difference
		now = datetime.datetime.utcnow()
		diff = abs((dt_obj - now).days)
		
		if diff == 0:
			return dt_obj.strftime("%H:%M")  # Same day: show time
		elif diff < 7:
			return dt_obj.strftime("%a %H:%M")  # This week: show day and time
		elif dt_obj.year == now.year:
			return dt_obj.strftime("%b %d")  # Same year: show month and day
		else:
			return dt_obj.strftime("%b %d, %Y")  # Different year: show full date


def format_relative_time(dt: datetime.datetime) -> str:
	"""Format datetime as relative time (e.g., '2 hours ago')"""
	now = datetime.datetime.utcnow()
	
	# Ensure we're working with timezone-naive datetimes for comparison
	if dt.tzinfo is not None:
		dt = dt.replace(tzinfo=None)
	if now.tzinfo is not None:
		now = now.replace(tzinfo=None)
	
	diff = now - dt
	
	# Future times
	if diff.total_seconds() < 0:
		diff = dt - now
		suffix = "from now"
	else:
		suffix = "ago"
	
	seconds = abs(diff.total_seconds())
	
	if seconds < 60:
		return "just now" if suffix == "ago" else "in a moment"
	elif seconds < 3600:  # Less than 1 hour
		minutes = int(seconds / 60)
		return f"{minutes} minute{'s' if minutes != 1 else ''} {suffix}"
	elif seconds < 86400:  # Less than 1 day
		hours = int(seconds / 3600)
		return f"{hours} hour{'s' if hours != 1 else ''} {suffix}"
	elif seconds < 2592000:  # Less than 30 days
		days = int(seconds / 86400)
		return f"{days} day{'s' if days != 1 else ''} {suffix}"
	elif seconds < 31536000:  # Less than 1 year
		months = int(seconds / 2592000)
		return f"{months} month{'s' if months != 1 else ''} {suffix}"
	else:
		years = int(seconds / 31536000)
		return f"{years} year{'s' if years != 1 else ''} {suffix}"


def format_duration(
	seconds: Union[int, float],
	precision: str = "auto",
	max_units: int = 2
) -> str:
	"""Format duration in seconds to human readable format"""
	
	if seconds < 0:
		return "0 seconds"
	
	# Time units in seconds
	units = [
		("year", 31536000),
		("month", 2592000),
		("week", 604800),
		("day", 86400),
		("hour", 3600),
		("minute", 60),
		("second", 1),
	]
	
	# Auto precision based on duration
	if precision == "auto":
		if seconds < 60:
			precision = "second"
		elif seconds < 3600:
			precision = "minute"
		elif seconds < 86400:
			precision = "hour"
		else:
			precision = "day"
	
	# Find starting unit based on precision
	start_unit = 0
	for i, (unit_name, _) in enumerate(units):
		if unit_name == precision:
			start_unit = i
			break
	
	# Build duration string
	parts = []
	remaining = int(seconds)
	
	for i in range(start_unit, len(units)):
		unit_name, unit_seconds = units[i]
		
		if remaining >= unit_seconds and len(parts) < max_units:
			count = remaining // unit_seconds
			remaining = remaining % unit_seconds
			
			unit_label = unit_name if count == 1 else f"{unit_name}s"
			parts.append(f"{count} {unit_label}")
	
	if not parts:
		return "0 seconds"
	
	return " ".join(parts)


def format_file_size(size_bytes: Union[int, float], binary: bool = True) -> str:
	"""Format file size in bytes to human readable format"""
	
	if size_bytes < 0:
		return "0 B"
	
	# Use binary (1024) or decimal (1000) units
	base = 1024 if binary else 1000
	
	if binary:
		units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
	else:
		units = ["B", "KB", "MB", "GB", "TB", "PB"]
	
	size = float(size_bytes)
	unit_index = 0
	
	while size >= base and unit_index < len(units) - 1:
		size /= base
		unit_index += 1
	
	# Format with appropriate precision
	if unit_index == 0:
		return f"{int(size)} {units[unit_index]}"
	elif size >= 100:
		return f"{size:.0f} {units[unit_index]}"
	elif size >= 10:
		return f"{size:.1f} {units[unit_index]}"
	else:
		return f"{size:.2f} {units[unit_index]}"


def format_percentage(
	value: Union[int, float, Decimal],
	total: Optional[Union[int, float, Decimal]] = None,
	decimal_places: int = 1
) -> str:
	"""Format percentage value"""
	
	if total is not None:
		if total == 0:
			percentage = 0
		else:
			percentage = (float(value) / float(total)) * 100
	else:
		percentage = float(value)
	
	return f"{percentage:.{decimal_places}f}%"


def format_currency(
	amount: Union[int, float, Decimal],
	currency: str = "USD",
	locale: Optional[str] = None
) -> str:
	"""Format currency amount"""
	
	# Simple currency formatting without locale library
	currency_symbols = {
		"USD": "$",
		"EUR": "€",
		"GBP": "£",
		"JPY": "¥",
		"CAD": "C$",
		"AUD": "A$",
	}
	
	symbol = currency_symbols.get(currency, currency)
	
	# Format with appropriate decimal places
	if currency == "JPY":
		# Japanese Yen typically doesn't use decimal places
		return f"{symbol}{int(amount):,}"
	else:
		return f"{symbol}{float(amount):,.2f}"


def format_phone_number(phone: str, country_code: str = "US") -> str:
	"""Format phone number for display"""
	
	# Remove all non-digit characters
	digits = re.sub(r'\D', '', phone)
	
	if country_code == "US" and len(digits) == 10:
		# US format: (XXX) XXX-XXXX
		return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
	elif country_code == "US" and len(digits) == 11 and digits[0] == '1':
		# US format with country code: +1 (XXX) XXX-XXXX
		return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
	elif digits.startswith('1') and len(digits) == 11:
		# North American format
		return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
	else:
		# Generic international format
		if len(digits) > 10:
			return f"+{digits[:len(digits)-10]} {digits[-10:-7]} {digits[-7:-4]} {digits[-4:]}"
		else:
			return phone


def format_json(data: Any, indent: int = 2, sort_keys: bool = True) -> str:
	"""Format data as pretty-printed JSON"""
	
	def json_serializer(obj):
		"""Custom JSON serializer for special types"""
		if isinstance(obj, datetime.datetime):
			return obj.isoformat()
		elif isinstance(obj, datetime.date):
			return obj.isoformat()
		elif isinstance(obj, Decimal):
			return float(obj)
		elif hasattr(obj, 'to_dict'):
			return obj.to_dict()
		elif hasattr(obj, '__dict__'):
			return obj.__dict__
		else:
			return str(obj)
	
	try:
		return json.dumps(
			data,
			indent=indent,
			sort_keys=sort_keys,
			default=json_serializer,
			ensure_ascii=False
		)
	except (TypeError, ValueError) as e:
		return f"Error formatting JSON: {e}"


def format_list(
	items: List[Any],
	conjunction: str = "and",
	max_items: Optional[int] = None
) -> str:
	"""Format list of items as readable string"""
	
	if not items:
		return ""
	
	# Convert items to strings
	str_items = [str(item) for item in items]
	
	# Limit number of items if specified
	if max_items and len(str_items) > max_items:
		displayed_items = str_items[:max_items]
		remaining = len(str_items) - max_items
		displayed_items.append(f"and {remaining} more")
		str_items = displayed_items
	
	if len(str_items) == 1:
		return str_items[0]
	elif len(str_items) == 2:
		return f"{str_items[0]} {conjunction} {str_items[1]}"
	else:
		return f"{', '.join(str_items[:-1])}, {conjunction} {str_items[-1]}"


def format_address(address_dict: Dict[str, str]) -> str:
	"""Format address dictionary as readable string"""
	
	parts = []
	
	# Street address
	if "street" in address_dict:
		parts.append(address_dict["street"])
	
	# City, state, zip
	city_state_zip = []
	if "city" in address_dict:
		city_state_zip.append(address_dict["city"])
	if "state" in address_dict:
		city_state_zip.append(address_dict["state"])
	if "zip" in address_dict or "postal_code" in address_dict:
		zip_code = address_dict.get("zip") or address_dict.get("postal_code")
		city_state_zip.append(zip_code)
	
	if city_state_zip:
		parts.append(", ".join(city_state_zip))
	
	# Country
	if "country" in address_dict:
		parts.append(address_dict["country"])
	
	return "\n".join(parts)


def format_name(
	first_name: Optional[str] = None,
	last_name: Optional[str] = None,
	middle_name: Optional[str] = None,
	title: Optional[str] = None,
	format_style: str = "full"
) -> str:
	"""Format person's name"""
	
	parts = []
	
	if format_style == "full":
		if title:
			parts.append(title)
		if first_name:
			parts.append(first_name)
		if middle_name:
			parts.append(middle_name)
		if last_name:
			parts.append(last_name)
	elif format_style == "first_last":
		if first_name:
			parts.append(first_name)
		if last_name:
			parts.append(last_name)
	elif format_style == "last_first":
		if last_name and first_name:
			parts.append(f"{last_name}, {first_name}")
		elif last_name:
			parts.append(last_name)
		elif first_name:
			parts.append(first_name)
	elif format_style == "initials":
		if first_name:
			parts.append(f"{first_name[0]}.")
		if middle_name:
			parts.append(f"{middle_name[0]}.")
		if last_name:
			parts.append(f"{last_name[0]}.")
	
	return " ".join(parts)


def format_truncate(
	text: str,
	max_length: int,
	ellipsis: str = "...",
	word_boundary: bool = True
) -> str:
	"""Truncate text to specified length"""
	
	if len(text) <= max_length:
		return text
	
	if word_boundary:
		# Find last space before max_length
		truncated = text[:max_length - len(ellipsis)]
		last_space = truncated.rfind(' ')
		
		if last_space > 0:
			truncated = truncated[:last_space]
		
		return truncated + ellipsis
	else:
		return text[:max_length - len(ellipsis)] + ellipsis


def format_mask_sensitive(
	text: str,
	mask_char: str = "*",
	visible_start: int = 2,
	visible_end: int = 2
) -> str:
	"""Mask sensitive information like credit card numbers"""
	
	if len(text) <= visible_start + visible_end:
		return mask_char * len(text)
	
	start = text[:visible_start]
	end = text[-visible_end:] if visible_end > 0 else ""
	middle_length = len(text) - visible_start - visible_end
	middle = mask_char * middle_length
	
	return start + middle + end


def format_camel_case_to_title(camel_case: str) -> str:
	"""Convert camelCase to Title Case"""
	
	# Insert space before uppercase letters
	spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', camel_case)
	
	# Capitalize first letter of each word
	return spaced.title()


def format_snake_case_to_title(snake_case: str) -> str:
	"""Convert snake_case to Title Case"""
	
	# Replace underscores with spaces and title case
	return snake_case.replace('_', ' ').title()


def format_bytes_to_hex(data: bytes, separator: str = "", uppercase: bool = False) -> str:
	"""Format bytes as hexadecimal string"""
	
	hex_str = data.hex()
	
	if uppercase:
		hex_str = hex_str.upper()
	
	if separator:
		# Insert separator every 2 characters
		return separator.join([hex_str[i:i+2] for i in range(0, len(hex_str), 2)])
	
	return hex_str


def format_smart_quotes(text: str) -> str:
	"""Convert straight quotes to smart quotes"""
	
	# Replace straight quotes with smart quotes
	text = re.sub(r'"([^"]*)"', r'"\1"', text)  # Double quotes
	text = re.sub(r"'([^']*)'", r"'\1'", text)  # Single quotes
	
	return text