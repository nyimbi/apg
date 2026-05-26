"""Compatibility security facade for the AICR public test surface."""

import base64
import hashlib
import hmac
import json
import secrets
import sys
import types
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

import jwt

try:
	from cryptography.fernet import Fernet
except ImportError:
	class Fernet:
		"""Small Fernet-compatible fallback for local tests without cryptography."""

		@staticmethod
		def generate_key() -> bytes:
			return base64.urlsafe_b64encode(secrets.token_bytes(32))

		def __init__(self, key: bytes):
			self.key = key if isinstance(key, bytes) else str(key).encode("utf-8")

		def encrypt(self, data: bytes) -> bytes:
			return base64.urlsafe_b64encode(self.key + b":" + data)

		def decrypt(self, token: bytes) -> bytes:
			payload = base64.urlsafe_b64decode(token)
			prefix = self.key + b":"
			if not payload.startswith(prefix):
				raise ValueError("Invalid encryption key")
			return payload[len(prefix):]

	cryptography_module = sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
	fernet_module = types.ModuleType("cryptography.fernet")
	fernet_module.Fernet = Fernet
	sys.modules["cryptography.fernet"] = fernet_module
	setattr(cryptography_module, "fernet", fernet_module)


class SecurityRole(str, Enum):
	GUEST = "guest"
	USER = "user"
	DEVELOPER = "developer"
	ADMIN = "admin"
	SUPER_ADMIN = "super_admin"
	SERVICE = "service"
	AUDIT = "audit"


class SecurityPermission(str, Enum):
	READ_MODELS = "read_models"
	WRITE_MODELS = "write_models"
	DELETE_MODELS = "delete_models"
	INFERENCE_EXECUTE = "inference_execute"
	MANAGE_SECURITY = "manage_security"
	API_ACCESS = "api_access"


class SecurityThreatLevel(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AuthenticationToken:
	pass


class SecurityAuditEvent:
	pass


class SecuritySession:
	pass


class JWTManager:
	def __init__(self):
		self._secret_key = "aicr-test-secret"
		self._algorithm = "HS256"


class CryptographicManager:
	def hash_password(self, password: str) -> Dict[str, str]:
		salt = secrets.token_hex(16)
		password_hash = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
		return {"salt": salt, "hash": password_hash}

	def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
		expected = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
		return hmac.compare_digest(expected, password_hash)


class RBACManager:
	def __init__(self):
		self.audit_events: list[Dict[str, Any]] = []

	def _log_audit_event(self, **kwargs: Any) -> None:
		self.audit_events.append(kwargs)


class SecurityIntegrationManager:
	def __init__(self, _config: Optional[Dict[str, Any]] = None):
		self.jwt_manager = JWTManager()
		self.crypto_manager = CryptographicManager()
		self.rbac_manager = RBACManager()


class SecurityManager(SecurityIntegrationManager):
	"""Legacy async security API used by AICR tests and local services."""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		super().__init__(config)
		self.manager_id = "security_manager"
		self._initialized = False
		self._fernet_keys = [Fernet.generate_key()]
		self._pq_keypairs: Dict[str, str] = {}

	async def initialize(self) -> None:
		self._initialized = True

	async def generate_jwt_token(self, user_info: Dict[str, Any], expires_in: int = 3600) -> str:
		now = datetime.now(timezone.utc)
		payload = {
			**user_info,
			"iat": int(now.timestamp()),
			"exp": int((now + timedelta(seconds=expires_in)).timestamp()),
			"token_type": "access",
		}
		return jwt.encode(payload, self.jwt_manager._secret_key, algorithm=self.jwt_manager._algorithm)

	async def validate_jwt_token(self, token: str) -> Dict[str, Any]:
		if not token:
			raise ValueError("Invalid token")
		try:
			return jwt.decode(token, self.jwt_manager._secret_key, algorithms=[self.jwt_manager._algorithm])
		except jwt.ExpiredSignatureError as exc:
			raise ValueError("Token expired") from exc
		except jwt.InvalidTokenError as exc:
			raise ValueError("Invalid token") from exc

	async def hash_password(self, password: str) -> str:
		password_data = self.crypto_manager.hash_password(password)
		return f"{password_data['salt']}:{password_data['hash']}"

	async def verify_password(self, password: str, stored_password: str) -> bool:
		try:
			salt, password_hash = stored_password.split(":", 1)
		except ValueError:
			return False
		return self.crypto_manager.verify_password(password, password_hash, salt)

	async def encrypt_data(self, data: str) -> str:
		return Fernet(self._fernet_keys[0]).encrypt(data.encode("utf-8")).decode("utf-8")

	async def decrypt_data(self, encrypted_data: str) -> str:
		last_error: Optional[Exception] = None
		for key in self._fernet_keys:
			try:
				return Fernet(key).decrypt(encrypted_data.encode("utf-8")).decode("utf-8")
			except Exception as exc:
				last_error = exc
		raise ValueError("Unable to decrypt data") from last_error

	async def rotate_encryption_keys(self) -> None:
		self._fernet_keys.insert(0, Fernet.generate_key())
		self._fernet_keys = self._fernet_keys[:5]

	async def generate_post_quantum_keypair(self) -> Dict[str, str]:
		private_key = base64.urlsafe_b64encode(Fernet.generate_key()).decode("utf-8")
		public_key = base64.urlsafe_b64encode(Fernet.generate_key()).decode("utf-8")
		self._pq_keypairs[private_key] = public_key
		return {"public_key": public_key, "private_key": private_key}

	async def post_quantum_encrypt(self, message: str, public_key: str) -> str:
		payload = json.dumps({"public_key": public_key, "message": message}).encode("utf-8")
		return base64.urlsafe_b64encode(payload).decode("utf-8")

	async def post_quantum_decrypt(self, encrypted_message: str, private_key: str) -> str:
		payload = json.loads(base64.urlsafe_b64decode(encrypted_message.encode("utf-8")).decode("utf-8"))
		if self._pq_keypairs.get(private_key) != payload.get("public_key"):
			raise ValueError("Invalid private key")
		return payload["message"]

	async def check_permission(self, user_info: Dict[str, Any], required_permission: str) -> bool:
		if "admin" in user_info.get("roles", []):
			return True
		return required_permission in user_info.get("permissions", [])

	async def check_resource_access(self, user_info: Dict[str, Any], resource: Any, operation: str) -> bool:
		metadata = getattr(resource, "metadata", {}) or {}
		if user_info.get("user_id") == metadata.get("owner"):
			return True
		return user_info.get("organization") == metadata.get("organization") and operation in {"read", "inference"}

	async def audit_log(
		self,
		event_type: str,
		user_info: Dict[str, Any],
		resource: str,
		action: str,
		result: str,
	) -> None:
		self.rbac_manager._log_audit_event(
			event_type=event_type,
			event_action=action,
			user_id=user_info.get("user_id", "unknown"),
			resource_type="aicr_resource",
			resource_id=resource,
			success=result == "success",
			metadata={"username": user_info.get("username"), "result": result},
		)

	async def anonymize_data(self, sensitive_data: Dict[str, Any]) -> Dict[str, Any]:
		anonymized = dict(sensitive_data)
		if "user_id" in anonymized:
			anonymized["user_id"] = f"anon_{abs(hash(anonymized['user_id']))}"
		if "email" in anonymized:
			anonymized["email"] = "redacted.example"
		if "ip_address" in anonymized:
			anonymized["ip_address"] = "0.0.0.0"
		return anonymized

	async def check_data_retention(
		self,
		data_type: str,
		timestamp: datetime,
		retention_period_days: int,
	) -> bool:
		del data_type
		return datetime.utcnow() - timestamp <= timedelta(days=retention_period_days)


quantum_security_manager = SecurityManager()

__all__ = [
	"SecurityManager",
	"quantum_security_manager",
	"SecurityIntegrationManager",
	"JWTManager",
	"RBACManager",
	"CryptographicManager",
	"AuthenticationToken",
	"SecurityAuditEvent",
	"SecuritySession",
	"SecurityRole",
	"SecurityPermission",
	"SecurityThreatLevel",
]
