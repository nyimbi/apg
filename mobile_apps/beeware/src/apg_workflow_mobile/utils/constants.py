"""
Application constants

© 2025 Datacraft. All rights reserved.
"""

import os
from pathlib import Path

# Application Information
APP_NAME = "APG Workflow Manager"
APP_VERSION = "1.0.0"
APP_ID = "co.ke.datacraft.apg-workflow-mobile"
COMPANY_NAME = "Datacraft"
COMPANY_URL = "https://www.datacraft.co.ke"

# API Configuration
API_BASE_URL = os.getenv(
	"APG_API_BASE_URL",
	"https://api.datacraft.co.ke/apg"  # Production URL
)

# Development override
if os.getenv("APG_DEV_MODE", "false").lower() == "true":
	API_BASE_URL = "http://localhost:5000"

API_TIMEOUT = float(os.getenv("APG_API_TIMEOUT", "30.0"))
API_RETRY_ATTEMPTS = int(os.getenv("APG_API_RETRY_ATTEMPTS", "3"))
API_RETRY_DELAY = float(os.getenv("APG_API_RETRY_DELAY", "1.0"))

# WebSocket Configuration
WS_RECONNECT_ATTEMPTS = 5
WS_RECONNECT_DELAY = 2.0
WS_HEARTBEAT_INTERVAL = 30.0

# Security Configuration  
KEYRING_SERVICE_NAME = f"{APP_ID}.keyring"
ENCRYPTION_KEY_LENGTH = 32
TOKEN_REFRESH_THRESHOLD = 300  # 5 minutes before expiry

# File Storage
APP_DATA_DIR = Path.home() / ".apg_workflow_mobile"
CACHE_DIR = APP_DATA_DIR / "cache"
OFFLINE_DATA_DIR = APP_DATA_DIR / "offline"
LOGS_DIR = APP_DATA_DIR / "logs"
TEMP_DIR = APP_DATA_DIR / "temp"

# Create directories if they don't exist
for directory in [APP_DATA_DIR, CACHE_DIR, OFFLINE_DATA_DIR, LOGS_DIR, TEMP_DIR]:
	directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
OFFLINE_DB_PATH = OFFLINE_DATA_DIR / "offline.db"
CACHE_DB_PATH = CACHE_DIR / "cache.db"

# Logging Configuration
LOG_LEVEL = os.getenv("APG_LOG_LEVEL", "INFO")
LOG_FILE_PATH = LOGS_DIR / "app.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# UI Configuration
THEME_MODE = "auto"  # light, dark, auto
PRIMARY_COLOR = "#1976D2"  # Material Blue
SECONDARY_COLOR = "#FFC107"  # Material Amber
SUCCESS_COLOR = "#4CAF50"  # Material Green
WARNING_COLOR = "#FF9800"  # Material Orange
ERROR_COLOR = "#F44336"  # Material Red

# Animation and Timing
ANIMATION_DURATION = 300  # milliseconds
TOAST_DURATION = 3000  # milliseconds
SPLASH_DURATION = 2000  # milliseconds

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Synchronization
SYNC_INTERVAL = 300  # 5 minutes
SYNC_BATCH_SIZE = 50
SYNC_TIMEOUT = 60  # 1 minute

# File Upload/Download
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_FILE_EXTENSIONS = [
	".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx", 
	".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3",
	".zip", ".rar", ".csv", ".json", ".xml"
]
UPLOAD_TIMEOUT = 300  # 5 minutes
DOWNLOAD_TIMEOUT = 600  # 10 minutes

# Notification Configuration
MAX_NOTIFICATIONS = 100
NOTIFICATION_CLEANUP_INTERVAL = 3600  # 1 hour

# Biometric Configuration
BIOMETRIC_TIMEOUT = 30  # seconds
BIOMETRIC_MAX_ATTEMPTS = 3

# Network Configuration
NETWORK_CHECK_INTERVAL = 10  # seconds
NETWORK_TIMEOUT = 5  # seconds
OFFLINE_MODE_THRESHOLD = 30  # seconds

# Cache Configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000  # items
IMAGE_CACHE_SIZE = 100  # images

# Workflow Configuration
MAX_WORKFLOW_STEPS = 100
MAX_PARALLEL_EXECUTIONS = 10
WORKFLOW_TIMEOUT = 3600  # 1 hour

# Task Configuration
TASK_UPDATE_INTERVAL = 30  # seconds
TASK_REMINDER_THRESHOLD = 3600  # 1 hour before due

# Error Reporting
ENABLE_ERROR_REPORTING = True
ERROR_REPORT_ENDPOINT = f"{API_BASE_URL}/errors"

# Feature Flags
FEATURES = {
	"biometric_auth": True,
	"offline_mode": True,
	"push_notifications": True,
	"file_upload": True,
	"voice_commands": True,
	"real_time_sync": True,
	"analytics": True,
	"crash_reporting": True,
}

# Development Settings
DEBUG_MODE = os.getenv("APG_DEBUG", "false").lower() == "true"
MOCK_API_RESPONSES = os.getenv("APG_MOCK_API", "false").lower() == "true"
VERBOSE_LOGGING = os.getenv("APG_VERBOSE", "false").lower() == "true"

# Platform-specific settings
import platform
PLATFORM = platform.system().lower()
IS_MOBILE = PLATFORM in ["android", "ios"]
IS_DESKTOP = PLATFORM in ["windows", "darwin", "linux"]

# URL Patterns
URL_PATTERNS = {
	"auth": {
		"login": "/auth/login",
		"logout": "/auth/logout", 
		"refresh": "/auth/refresh",
		"biometric_login": "/auth/biometric-login",
		"setup_biometric": "/auth/setup-biometric",
		"disable_biometric": "/auth/disable-biometric",
	},
	"users": {
		"profile": "/users/{user_id}",
		"update_profile": "/users/{user_id}",
		"avatar": "/users/{user_id}/avatar",
	},
	"workflows": {
		"list": "/workflows",
		"detail": "/workflows/{workflow_id}",
		"create": "/workflows",
		"update": "/workflows/{workflow_id}",
		"delete": "/workflows/{workflow_id}",
		"execute": "/workflows/{workflow_id}/execute",
		"instances": "/workflows/{workflow_id}/instances",
		"instance_detail": "/workflow-instances/{instance_id}",
	},
	"tasks": {
		"list": "/tasks",
		"detail": "/tasks/{task_id}",
		"create": "/tasks",
		"update": "/tasks/{task_id}",
		"delete": "/tasks/{task_id}",
		"assign": "/tasks/{task_id}/assign",
		"complete": "/tasks/{task_id}/complete",
		"comments": "/tasks/{task_id}/comments",
		"attachments": "/tasks/{task_id}/attachments",
	},
	"notifications": {
		"list": "/notifications",
		"mark_read": "/notifications/{notification_id}/read",
		"mark_all_read": "/notifications/read-all",
		"settings": "/notifications/settings",
	},
	"files": {
		"upload": "/files/upload",
		"download": "/files/{file_id}",
		"delete": "/files/{file_id}",
	},
	"system": {
		"health": "/health",
		"version": "/version",
		"config": "/config",
	},
	"websocket": {
		"notifications": "/ws/notifications",
		"workflow_updates": "/ws/workflows",
		"task_updates": "/ws/tasks",
		"chat": "/ws/chat",
	},
}

# HTTP Status Codes
HTTP_STATUS = {
	"OK": 200,
	"CREATED": 201,
	"NO_CONTENT": 204,
	"BAD_REQUEST": 400,
	"UNAUTHORIZED": 401,
	"FORBIDDEN": 403,
	"NOT_FOUND": 404,
	"CONFLICT": 409,
	"UNPROCESSABLE_ENTITY": 422,
	"INTERNAL_SERVER_ERROR": 500,
	"BAD_GATEWAY": 502,
	"SERVICE_UNAVAILABLE": 503,
}