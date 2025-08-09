"""
APG Context Manager

Manages APG capability integration and provides unified access
to APG services following composition patterns.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .config import APGDocumentConfig

logger = logging.getLogger(__name__)


class APGCapabilityProtocol(Protocol):
	"""Protocol for APG capability services"""
	
	async def initialize(self) -> None:
		"""Initialize the capability service"""
		...
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		...


class APGAuthService(APGCapabilityProtocol):
	"""APG Authentication and RBAC Service Interface"""
	
	async def authorize_action(self, user: Any, action: str, resource_type: str) -> bool:
		"""Authorize user action on resource type"""
		...
	
	async def get_user_permissions(self, user_id: str) -> list[str]:
		"""Get user permissions"""
		...
	
	async def evaluate_access(self, subject: str, resource: str, action: str, context: Dict[str, Any]) -> Any:
		"""Evaluate access decision"""
		...


class APGAuditService(APGCapabilityProtocol):
	"""APG Audit and Compliance Service Interface"""
	
	async def log_event(self, event_type: str, resource_id: str, user_id: str, metadata: Dict[str, Any]) -> None:
		"""Log audit event"""
		...
	
	async def log_access_attempt(self, user_id: str, resource_type: str, resource_id: str, 
								action: str, result: bool, context: Dict[str, Any]) -> None:
		"""Log access attempt"""
		...


class APGVisionService(APGCapabilityProtocol):
	"""APG Computer Vision Service Interface"""
	
	async def extract_text(self, image_path: str) -> Any:
		"""Extract text from image/document"""
		...
	
	async def analyze_layout(self, image_path: str) -> Any:
		"""Analyze document layout"""
		...
	
	async def assess_image_quality(self, image_path: str) -> float:
		"""Assess image quality score"""
		...


class APGNLPService(APGCapabilityProtocol):
	"""APG Natural Language Processing Service Interface"""
	
	async def extract_entities(self, content: str) -> list[Dict[str, Any]]:
		"""Extract named entities from content"""
		...
	
	async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
		"""Analyze content sentiment"""
		...
	
	async def generate_summary(self, content: str) -> str:
		"""Generate content summary"""
		...
	
	async def identify_topics(self, content: str) -> list[Dict[str, Any]]:
		"""Identify content topics"""
		...


class APGOrchestrationService(APGCapabilityProtocol):
	"""APG AI Orchestration Service Interface"""
	
	async def create_workflow(self, name: str, document_id: str, steps: list[str]) -> Any:
		"""Create processing workflow"""
		...
	
	async def execute_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Any:
		"""Execute workflow"""
		...


@dataclass
class APGServiceRegistry:
	"""Registry of available APG services"""
	
	auth_rbac: Optional[APGAuthService] = None
	audit_compliance: Optional[APGAuditService] = None
	computer_vision: Optional[APGVisionService] = None
	nlp: Optional[APGNLPService] = None
	ai_orchestration: Optional[APGOrchestrationService] = None
	real_time_collaboration: Optional[APGCapabilityProtocol] = None
	notification: Optional[APGCapabilityProtocol] = None
	
	def is_service_available(self, service_name: str) -> bool:
		"""Check if service is available"""
		return getattr(self, service_name, None) is not None


class APGContext:
	"""
	APG Context Manager
	
	Provides unified access to APG capabilities with proper initialization,
	error handling, and service discovery following APG composition patterns.
	"""
	
	def __init__(self, config: APGDocumentConfig, tenant_id: str):
		"""Initialize APG context"""
		assert tenant_id, "tenant_id is required"
		assert isinstance(config, APGDocumentConfig), "config must be APGDocumentConfig instance"
		
		self.config = config
		self.tenant_id = tenant_id
		self.services = APGServiceRegistry()
		self._initialized = False
		self._health_status: Dict[str, bool] = {}
		
		self._log_context_created()
	
	def _log_context_created(self) -> None:
		"""Log APG context creation"""
		logger.info(f"APG Context created for tenant: {self.tenant_id}")
		logger.info(f"APG base URL: {self.config.apg_base_url}")
		logger.info(f"Tenant mode: {self.config.tenant_mode}")
	
	async def initialize(self) -> None:
		"""Initialize APG context and discover services"""
		assert not self._initialized, "APG context already initialized"
		
		self._log_initialization_start()
		
		try:
			# Initialize core required services
			await self._initialize_auth_service()
			await self._initialize_audit_service()
			await self._initialize_vision_service()
			await self._initialize_nlp_service()
			await self._initialize_orchestration_service()
			
			# Initialize optional services
			await self._initialize_optional_services()
			
			# Validate required services
			await self._validate_required_services()
			
			self._initialized = True
			self._log_initialization_complete()
			
		except Exception as e:
			logger.error(f"APG context initialization failed: {e}")
			raise
	
	async def _initialize_auth_service(self) -> None:
		"""Initialize authentication and RBAC service"""
		try:
			# In production, this would discover and connect to the actual APG auth service
			# For now, we'll use a mock implementation
			from .mocks import MockAPGAuthService
			self.services.auth_rbac = MockAPGAuthService(self.tenant_id)
			await self.services.auth_rbac.initialize()
			self._health_status["auth_rbac"] = True
			logger.info("APG auth_rbac service initialized")
		except Exception as e:
			logger.error(f"Failed to initialize auth service: {e}")
			raise
	
	async def _initialize_audit_service(self) -> None:
		"""Initialize audit and compliance service"""
		try:
			from .mocks import MockAPGAuditService
			self.services.audit_compliance = MockAPGAuditService(self.tenant_id)
			await self.services.audit_compliance.initialize()
			self._health_status["audit_compliance"] = True
			logger.info("APG audit_compliance service initialized")
		except Exception as e:
			logger.error(f"Failed to initialize audit service: {e}")
			raise
	
	async def _initialize_vision_service(self) -> None:
		"""Initialize computer vision service"""
		try:
			from .mocks import MockAPGVisionService
			self.services.computer_vision = MockAPGVisionService(self.config.ollama_base_url)
			await self.services.computer_vision.initialize()
			self._health_status["computer_vision"] = True
			logger.info("APG computer_vision service initialized")
		except Exception as e:
			logger.error(f"Failed to initialize vision service: {e}")
			raise
	
	async def _initialize_nlp_service(self) -> None:
		"""Initialize NLP service"""
		try:
			from .mocks import MockAPGNLPService
			self.services.nlp = MockAPGNLPService(self.config.ollama_base_url)
			await self.services.nlp.initialize()
			self._health_status["nlp"] = True
			logger.info("APG nlp service initialized")
		except Exception as e:
			logger.error(f"Failed to initialize NLP service: {e}")
			raise
	
	async def _initialize_orchestration_service(self) -> None:
		"""Initialize AI orchestration service"""
		try:
			from .mocks import MockAPGOrchestrationService
			self.services.ai_orchestration = MockAPGOrchestrationService(self.tenant_id)
			await self.services.ai_orchestration.initialize()
			self._health_status["ai_orchestration"] = True
			logger.info("APG ai_orchestration service initialized")
		except Exception as e:
			logger.error(f"Failed to initialize orchestration service: {e}")
			raise
	
	async def _initialize_optional_services(self) -> None:
		"""Initialize optional APG services"""
		# These services are optional and failure won't prevent startup
		optional_services = [
			("real_time_collaboration", self._init_collaboration_service),
			("notification", self._init_notification_service)
		]
		
		for service_name, init_func in optional_services:
			try:
				await init_func()
				self._health_status[service_name] = True
				logger.info(f"APG {service_name} service initialized")
			except Exception as e:
				logger.warning(f"Optional service {service_name} unavailable: {e}")
				self._health_status[service_name] = False
	
	async def _init_collaboration_service(self) -> None:
		"""Initialize real-time collaboration service"""
		if self.config.real_time_enabled:
			from .mocks import MockAPGCollaborationService
			self.services.real_time_collaboration = MockAPGCollaborationService()
			await self.services.real_time_collaboration.initialize()
	
	async def _init_notification_service(self) -> None:
		"""Initialize notification service"""
		if self.config.notifications_enabled:
			from .mocks import MockAPGNotificationService
			self.services.notification = MockAPGNotificationService()
			await self.services.notification.initialize()
	
	async def _validate_required_services(self) -> None:
		"""Validate that all required services are available"""
		required_services = ["auth_rbac", "audit_compliance", "computer_vision", "nlp", "ai_orchestration"]
		
		for service_name in required_services:
			if not self._health_status.get(service_name, False):
				raise RuntimeError(f"Required APG service {service_name} is not available")
		
		logger.info("All required APG services validated")
	
	def get_capability(self, capability_name: str) -> Optional[APGCapabilityProtocol]:
		"""Get APG capability service by name"""
		assert self._initialized, "APG context must be initialized first"
		
		service = getattr(self.services, capability_name.replace(".", "_"), None)
		if not service:
			logger.warning(f"APG capability {capability_name} not available")
		return service
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check health of all APG services"""
		if not self._initialized:
			return {"status": "not_initialized", "services": {}}
		
		health_results = {}
		overall_healthy = True
		
		for service_name in self._health_status:
			service = getattr(self.services, service_name)
			if service:
				try:
					service_health = await service.health_check()
					health_results[service_name] = service_health
					if not service_health.get("healthy", False):
						overall_healthy = False
				except Exception as e:
					health_results[service_name] = {"healthy": False, "error": str(e)}
					overall_healthy = False
			else:
				health_results[service_name] = {"healthy": False, "error": "Service not available"}
				overall_healthy = False
		
		return {
			"status": "healthy" if overall_healthy else "degraded",
			"tenant_id": self.tenant_id,
			"services": health_results,
			"timestamp": asyncio.get_event_loop().time()
		}
	
	def _log_initialization_start(self) -> None:
		"""Log initialization start"""
		logger.info(f"Initializing APG context for tenant: {self.tenant_id}")
	
	def _log_initialization_complete(self) -> None:
		"""Log initialization completion"""
		healthy_services = sum(1 for status in self._health_status.values() if status)
		total_services = len(self._health_status)
		logger.info(f"APG context initialization complete: {healthy_services}/{total_services} services healthy")
	
	async def close(self) -> None:
		"""Clean up APG context and close service connections"""
		if not self._initialized:
			return
		
		logger.info("Closing APG context")
		
		# Close all services
		for service_name in self._health_status:
			service = getattr(self.services, service_name)
			if service and hasattr(service, 'close'):
				try:
					await service.close()
				except Exception as e:
					logger.warning(f"Error closing service {service_name}: {e}")
		
		self._initialized = False
		logger.info("APG context closed")


async def create_apg_context(config: APGDocumentConfig, tenant_id: str) -> APGContext:
	"""Create and initialize APG context"""
	context = APGContext(config, tenant_id)
	await context.initialize()
	return context