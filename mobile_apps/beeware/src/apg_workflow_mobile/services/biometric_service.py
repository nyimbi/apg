"""
Biometric Service for APG Workflow Mobile

Handles biometric authentication across platforms.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
import platform
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import hashlib
import base64

from ..utils.exceptions import BiometricException
from ..utils.security import generate_random_string, secure_hash


@dataclass
class BiometricResult:
	"""Result of biometric authentication operation"""
	success: bool
	error: Optional[str] = None
	signature: Optional[str] = None
	template: Optional[str] = None
	method: Optional[str] = None
	timestamp: Optional[datetime] = None
	
	def __post_init__(self):
		if self.timestamp is None:
			self.timestamp = datetime.utcnow()


@dataclass
class BiometricCapabilities:
	"""Platform biometric capabilities"""
	fingerprint_available: bool = False
	face_available: bool = False
	voice_available: bool = False
	iris_available: bool = False
	device_supports_biometrics: bool = False
	secure_hardware: bool = False
	platform: str = ""
	
	@property
	def available_methods(self) -> List[str]:
		"""Get list of available biometric methods"""
		methods = []
		if self.fingerprint_available:
			methods.append("fingerprint")
		if self.face_available:
			methods.append("face")
		if self.voice_available:
			methods.append("voice")
		if self.iris_available:
			methods.append("iris")
		return methods
	
	@property
	def has_any_biometric(self) -> bool:
		"""Check if any biometric method is available"""
		return len(self.available_methods) > 0


class BiometricService:
	"""Cross-platform biometric authentication service"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self.platform = platform.system().lower()
		self.capabilities: Optional[BiometricCapabilities] = None
		self._initialized = False
		
		self.logger.info(f"Biometric Service initialized for platform: {self.platform}")
	
	async def initialize(self) -> BiometricCapabilities:
		"""Initialize biometric service and detect capabilities"""
		try:
			self.capabilities = await self._detect_capabilities()
			self._initialized = True
			
			self.logger.info(f"Biometric capabilities detected: {self.capabilities.available_methods}")
			return self.capabilities
			
		except Exception as e:
			self.logger.error(f"Failed to initialize biometric service: {e}")
			self.capabilities = BiometricCapabilities(platform=self.platform)
			raise BiometricException(f"Failed to initialize biometric service: {e}")
	
	async def _detect_capabilities(self) -> BiometricCapabilities:
		"""Detect platform-specific biometric capabilities"""
		capabilities = BiometricCapabilities(platform=self.platform)
		
		try:
			if self.platform == "darwin":  # macOS
				capabilities = await self._detect_macos_capabilities()
			elif self.platform == "windows":
				capabilities = await self._detect_windows_capabilities()
			elif self.platform == "linux":
				capabilities = await self._detect_linux_capabilities()
			elif self.platform == "android":
				capabilities = await self._detect_android_capabilities()
			elif self.platform == "ios":
				capabilities = await self._detect_ios_capabilities()
			else:
				self.logger.warning(f"Unsupported platform for biometrics: {self.platform}")
			
			return capabilities
			
		except Exception as e:
			self.logger.error(f"Error detecting biometric capabilities: {e}")
			return capabilities
	
	async def _detect_macos_capabilities(self) -> BiometricCapabilities:
		"""Detect macOS biometric capabilities"""
		capabilities = BiometricCapabilities(platform="darwin")
		
		try:
			# Try to detect Touch ID availability
			# In a real implementation, this would use macOS Security framework
			import subprocess
			
			# Check for biometric hardware
			result = subprocess.run(
				["system_profiler", "SPHardwareDataType"],
				capture_output=True,
				text=True,
				timeout=5
			)
			
			if "Touch ID" in result.stdout:
				capabilities.fingerprint_available = True
				capabilities.device_supports_biometrics = True
				capabilities.secure_hardware = True
			
			# Check for Face ID (newer Macs)
			if "Face ID" in result.stdout:
				capabilities.face_available = True
				capabilities.device_supports_biometrics = True
				capabilities.secure_hardware = True
			
		except Exception as e:
			self.logger.warning(f"Could not detect macOS biometric capabilities: {e}")
		
		return capabilities
	
	async def _detect_windows_capabilities(self) -> BiometricCapabilities:
		"""Detect Windows biometric capabilities"""
		capabilities = BiometricCapabilities(platform="windows")
		
		try:
			# Check Windows Hello availability
			import subprocess
			
			# Use PowerShell to check biometric devices
			ps_command = """
			Get-WmiObject -Namespace root/cimv2/security/microsofttpm -Class Win32_Tpm | Select-Object IsEnabled_InitialValue
			"""
			
			result = subprocess.run(
				["powershell", "-Command", ps_command],
				capture_output=True,
				text=True,
				timeout=10
			)
			
			if result.returncode == 0:
				capabilities.device_supports_biometrics = True
				capabilities.secure_hardware = True
				
				# Windows Hello typically supports fingerprint and face
				capabilities.fingerprint_available = True
				capabilities.face_available = True
			
		except Exception as e:
			self.logger.warning(f"Could not detect Windows biometric capabilities: {e}")
		
		return capabilities
	
	async def _detect_linux_capabilities(self) -> BiometricCapabilities:
		"""Detect Linux biometric capabilities"""
		capabilities = BiometricCapabilities(platform="linux")
		
		try:
			# Check for fingerprint readers using libfprint
			import subprocess
			
			# Try to find fingerprint devices
			result = subprocess.run(
				["lsusb"],
				capture_output=True,
				text=True,
				timeout=5
			)
			
			# Look for common fingerprint reader vendors
			fingerprint_vendors = [
				"Validity Sensors", "AuthenTec", "Upek", "STMicroelectronics",
				"Synaptics", "Elan", "FocalTech", "Goodix"
			]
			
			for vendor in fingerprint_vendors:
				if vendor.lower() in result.stdout.lower():
					capabilities.fingerprint_available = True
					capabilities.device_supports_biometrics = True
					break
			
		except Exception as e:
			self.logger.warning(f"Could not detect Linux biometric capabilities: {e}")
		
		return capabilities
	
	async def _detect_android_capabilities(self) -> BiometricCapabilities:
		"""Detect Android biometric capabilities"""
		capabilities = BiometricCapabilities(platform="android")
		
		try:
			# In a real Android implementation, this would use Android Biometric API
			# For now, assume modern Android devices have biometric capabilities
			capabilities.fingerprint_available = True
			capabilities.face_available = True
			capabilities.device_supports_biometrics = True
			capabilities.secure_hardware = True
			
		except Exception as e:
			self.logger.warning(f"Could not detect Android biometric capabilities: {e}")
		
		return capabilities
	
	async def _detect_ios_capabilities(self) -> BiometricCapabilities:
		"""Detect iOS biometric capabilities"""
		capabilities = BiometricCapabilities(platform="ios")
		
		try:
			# In a real iOS implementation, this would use LocalAuthentication framework
			# For now, assume modern iOS devices have biometric capabilities
			capabilities.fingerprint_available = True  # Touch ID
			capabilities.face_available = True  # Face ID
			capabilities.device_supports_biometrics = True
			capabilities.secure_hardware = True
			
		except Exception as e:
			self.logger.warning(f"Could not detect iOS biometric capabilities: {e}")
		
		return capabilities
	
	async def get_available_methods(self) -> List[str]:
		"""Get list of available biometric methods"""
		if not self._initialized:
			await self.initialize()
		
		return self.capabilities.available_methods if self.capabilities else []
	
	async def is_biometric_available(self) -> bool:
		"""Check if any biometric authentication is available"""
		if not self._initialized:
			await self.initialize()
		
		return self.capabilities.has_any_biometric if self.capabilities else False
	
	async def authenticate(self, method: Optional[str] = None, 
						   prompt: str = "Authenticate to continue") -> BiometricResult:
		"""Perform biometric authentication"""
		try:
			if not self._initialized:
				await self.initialize()
			
			if not self.capabilities or not self.capabilities.has_any_biometric:
				return BiometricResult(
					success=False,
					error="No biometric authentication methods available"
				)
			
			# Use specified method or first available
			auth_method = method
			if not auth_method:
				available_methods = self.capabilities.available_methods
				if not available_methods:
					return BiometricResult(
						success=False,
						error="No biometric methods available"
					)
				auth_method = available_methods[0]
			
			# Perform platform-specific authentication
			if self.platform == "darwin":
				return await self._authenticate_macos(auth_method, prompt)
			elif self.platform == "windows":
				return await self._authenticate_windows(auth_method, prompt)
			elif self.platform == "linux":
				return await self._authenticate_linux(auth_method, prompt)
			elif self.platform == "android":
				return await self._authenticate_android(auth_method, prompt)
			elif self.platform == "ios":
				return await self._authenticate_ios(auth_method, prompt)
			else:
				return BiometricResult(
					success=False,
					error=f"Biometric authentication not supported on {self.platform}"
				)
			
		except Exception as e:
			self.logger.error(f"Biometric authentication failed: {e}")
			return BiometricResult(
				success=False,
				error=f"Authentication failed: {e}"
			)
	
	async def _authenticate_macos(self, method: str, prompt: str) -> BiometricResult:
		"""Perform macOS biometric authentication"""
		try:
			# In a real implementation, this would use LocalAuthentication framework
			# For demo purposes, we'll simulate the authentication
			
			self.logger.info(f"Simulating macOS {method} authentication")
			
			# Simulate authentication delay
			await asyncio.sleep(1)
			
			# Generate mock signature
			signature = self._generate_mock_signature(method)
			
			return BiometricResult(
				success=True,
				signature=signature,
				method=method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			return BiometricResult(
				success=False,
				error=f"macOS authentication failed: {e}"
			)
	
	async def _authenticate_windows(self, method: str, prompt: str) -> BiometricResult:
		"""Perform Windows biometric authentication"""
		try:
			# In a real implementation, this would use Windows Hello API
			self.logger.info(f"Simulating Windows {method} authentication")
			
			await asyncio.sleep(1)
			signature = self._generate_mock_signature(method)
			
			return BiometricResult(
				success=True,
				signature=signature,
				method=method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			return BiometricResult(
				success=False,
				error=f"Windows authentication failed: {e}"
			)
	
	async def _authenticate_linux(self, method: str, prompt: str) -> BiometricResult:
		"""Perform Linux biometric authentication"""
		try:
			# In a real implementation, this would use libfprint or similar
			self.logger.info(f"Simulating Linux {method} authentication")
			
			await asyncio.sleep(1)
			signature = self._generate_mock_signature(method)
			
			return BiometricResult(
				success=True,
				signature=signature,
				method=method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			return BiometricResult(
				success=False,
				error=f"Linux authentication failed: {e}"
			)
	
	async def _authenticate_android(self, method: str, prompt: str) -> BiometricResult:
		"""Perform Android biometric authentication"""
		try:
			# In a real implementation, this would use BiometricPrompt API
			self.logger.info(f"Simulating Android {method} authentication")
			
			await asyncio.sleep(1)
			signature = self._generate_mock_signature(method)
			
			return BiometricResult(
				success=True,
				signature=signature,
				method=method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			return BiometricResult(
				success=False,
				error=f"Android authentication failed: {e}"
			)
	
	async def _authenticate_ios(self, method: str, prompt: str) -> BiometricResult:
		"""Perform iOS biometric authentication"""
		try:
			# In a real implementation, this would use LocalAuthentication framework
			self.logger.info(f"Simulating iOS {method} authentication")
			
			await asyncio.sleep(1)
			signature = self._generate_mock_signature(method)
			
			return BiometricResult(
				success=True,
				signature=signature,
				method=method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			return BiometricResult(
				success=False,
				error=f"iOS authentication failed: {e}"
			)
	
	async def enroll_user(self, user_id: str, method: Optional[str] = None) -> BiometricResult:
		"""Enroll user for biometric authentication"""
		try:
			if not self._initialized:
				await self.initialize()
			
			if not self.capabilities or not self.capabilities.has_any_biometric:
				return BiometricResult(
					success=False,
					error="No biometric authentication methods available"
				)
			
			# Use specified method or first available
			enroll_method = method
			if not enroll_method:
				available_methods = self.capabilities.available_methods
				if not available_methods:
					return BiometricResult(
						success=False,
						error="No biometric methods available"
					)
				enroll_method = available_methods[0]
			
			self.logger.info(f"Enrolling user {user_id} for {enroll_method} authentication")
			
			# Simulate enrollment process
			await asyncio.sleep(2)
			
			# Generate biometric template
			template = self._generate_biometric_template(user_id, enroll_method)
			
			return BiometricResult(
				success=True,
				template=template,
				method=enroll_method,
				timestamp=datetime.utcnow()
			)
			
		except Exception as e:
			self.logger.error(f"Biometric enrollment failed: {e}")
			return BiometricResult(
				success=False,
				error=f"Enrollment failed: {e}"
			)
	
	def _generate_mock_signature(self, method: str) -> str:
		"""Generate mock biometric signature for testing"""
		# Create a signature based on method and timestamp
		timestamp = datetime.utcnow().isoformat()
		data = f"{method}:{timestamp}:{generate_random_string(16)}"
		signature_hash = secure_hash(data, "sha256")
		
		return base64.urlsafe_b64encode(signature_hash.encode()).decode()
	
	def _generate_biometric_template(self, user_id: str, method: str) -> str:
		"""Generate mock biometric template for testing"""
		# Create a template based on user ID and method
		template_data = f"{user_id}:{method}:{generate_random_string(32)}"
		template_hash = secure_hash(template_data, "sha256")
		
		return base64.urlsafe_b64encode(template_hash.encode()).decode()
	
	async def verify_template(self, template: str, signature: str) -> bool:
		"""Verify biometric signature against template"""
		try:
			# In a real implementation, this would perform actual template matching
			# For now, we'll do a simple validation
			
			if not template or not signature:
				return False
			
			# Both template and signature should be valid base64 encoded strings
			try:
				base64.urlsafe_b64decode(template.encode())
				base64.urlsafe_b64decode(signature.encode())
				return True
			except Exception:
				return False
			
		except Exception as e:
			self.logger.error(f"Template verification failed: {e}")
			return False
	
	async def delete_biometric_data(self, user_id: str) -> bool:
		"""Delete stored biometric data for user"""
		try:
			self.logger.info(f"Deleting biometric data for user: {user_id}")
			
			# In a real implementation, this would delete stored biometric templates
			# For now, we'll just simulate the operation
			await asyncio.sleep(0.5)
			
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to delete biometric data: {e}")
			return False
	
	def get_capabilities(self) -> Optional[BiometricCapabilities]:
		"""Get biometric capabilities"""
		return self.capabilities
	
	def is_secure_hardware_available(self) -> bool:
		"""Check if secure hardware is available for biometrics"""
		return (
			self.capabilities and 
			self.capabilities.secure_hardware 
			if self.capabilities else False
		)
	
	async def check_biometric_changed(self) -> bool:
		"""Check if biometric enrollment has changed"""
		try:
			# In a real implementation, this would check if new biometrics were enrolled
			# or existing ones were removed since last check
			
			# For now, always return False (no changes)
			return False
			
		except Exception as e:
			self.logger.error(f"Failed to check biometric changes: {e}")
			return False