"""
APG Service Mocks

Mock implementations of APG services for development and testing.
These mocks simulate the behavior of actual APG capabilities.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid_extensions import uuid7str
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MockAccessDecision:
	"""Mock access decision result"""
	allowed: bool
	reason: str = ""
	context: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.context is None:
			self.context = {}


@dataclass
class MockOCRResult:
	"""Mock OCR result"""
	text: str
	confidence: float
	regions: List[Dict[str, Any]]
	language: str = "en"


@dataclass
class MockLayoutResult:
	"""Mock layout analysis result"""
	regions: List[Dict[str, Any]]
	elements: List[Dict[str, Any]]
	structure: Dict[str, Any]


class MockAPGAuthService:
	"""Mock APG Authentication and RBAC Service"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self._user_permissions: Dict[str, List[str]] = {}
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock auth service"""
		self._initialized = True
		logger.info(f"Mock APG Auth Service initialized for tenant: {self.tenant_id}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "auth_rbac",
			"tenant_id": self.tenant_id,
			"features": ["authentication", "authorization", "rbac"]
		}
	
	async def authorize_action(self, user: Any, action: str, resource_type: str) -> bool:
		"""Mock authorization - always allow for development"""
		await asyncio.sleep(0.01)  # Simulate network call
		logger.debug(f"Mock auth: User {getattr(user, 'user_id', 'unknown')} authorized for {action} on {resource_type}")
		return True
	
	async def get_user_permissions(self, user_id: str) -> List[str]:
		"""Mock user permissions retrieval"""
		await asyncio.sleep(0.01)
		
		# Return default permissions for development
		default_permissions = [
			"document:read",
			"document:write", 
			"document:delete",
			"document:share",
			"document:admin"
		]
		
		return self._user_permissions.get(user_id, default_permissions)
	
	async def evaluate_access(self, subject: str, resource: str, action: str, context: Dict[str, Any]) -> MockAccessDecision:
		"""Mock access evaluation"""
		await asyncio.sleep(0.01)
		
		# For development, allow most actions
		allowed = True
		reason = "Mock authorization - allowed"
		
		# Simulate some access restrictions for testing
		if context.get("document_classification") == "top_secret" and "admin" not in action:
			allowed = False
			reason = "Insufficient clearance for top secret documents"
		
		return MockAccessDecision(
			allowed=allowed,
			reason=reason,
			context=context
		)
	
	def set_user_permissions(self, user_id: str, permissions: List[str]) -> None:
		"""Set user permissions for testing"""
		self._user_permissions[user_id] = permissions


class MockAPGAuditService:
	"""Mock APG Audit and Compliance Service"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self._audit_log: List[Dict[str, Any]] = []
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock audit service"""
		self._initialized = True
		logger.info(f"Mock APG Audit Service initialized for tenant: {self.tenant_id}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "audit_compliance",
			"tenant_id": self.tenant_id,
			"events_logged": len(self._audit_log)
		}
	
	async def log_event(self, event_type: str, resource_id: str, user_id: str, metadata: Dict[str, Any]) -> None:
		"""Mock event logging"""
		event = {
			"event_id": uuid7str(),
			"event_type": event_type,
			"resource_id": resource_id,
			"user_id": user_id,
			"tenant_id": self.tenant_id,
			"timestamp": datetime.utcnow().isoformat(),
			"metadata": metadata
		}
		
		self._audit_log.append(event)
		logger.debug(f"Mock audit: Logged {event_type} event for resource {resource_id}")
	
	async def log_access_attempt(self, user_id: str, resource_type: str, resource_id: str, 
								action: str, result: bool, context: Dict[str, Any]) -> None:
		"""Mock access attempt logging"""
		event = {
			"event_id": uuid7str(),
			"event_type": "access_attempt",
			"user_id": user_id,
			"resource_type": resource_type,
			"resource_id": resource_id,
			"action": action,
			"result": "success" if result else "denied",
			"tenant_id": self.tenant_id,
			"timestamp": datetime.utcnow().isoformat(),
			"context": context
		}
		
		self._audit_log.append(event)
		logger.debug(f"Mock audit: Logged access attempt by {user_id} to {resource_type}:{resource_id}")
	
	def get_audit_log(self) -> List[Dict[str, Any]]:
		"""Get audit log for testing"""
		return self._audit_log.copy()
	
	def clear_audit_log(self) -> None:
		"""Clear audit log for testing"""
		self._audit_log.clear()


class MockAPGVisionService:
	"""Mock APG Computer Vision Service"""
	
	def __init__(self, ollama_url: str):
		self.ollama_url = ollama_url
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock vision service"""
		self._initialized = True
		logger.info(f"Mock APG Vision Service initialized with Ollama URL: {self.ollama_url}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "computer_vision",
			"ollama_url": self.ollama_url,
			"models": ["qwen2.5-vl:latest"]
		}
	
	async def extract_text(self, image_path: str) -> MockOCRResult:
		"""Mock OCR text extraction"""
		await asyncio.sleep(0.5)  # Simulate processing time
		
		# Simulate different results based on file type
		if image_path.endswith('.pdf'):
			text = "This is extracted text from a PDF document. It contains multiple paragraphs and formatted content."
			confidence = 0.95
		elif any(image_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff']):
			text = "Text extracted from image using OCR. Quality may vary based on image resolution and clarity."
			confidence = 0.87
		else:
			text = "Generic extracted text content"
			confidence = 0.80
		
		regions = [
			{"x": 0, "y": 0, "width": 100, "height": 20, "text": text[:50]},
			{"x": 0, "y": 25, "width": 100, "height": 20, "text": text[50:100] if len(text) > 50 else ""}
		]
		
		logger.debug(f"Mock vision: Extracted {len(text)} characters from {image_path}")
		return MockOCRResult(text=text, confidence=confidence, regions=regions)
	
	async def analyze_layout(self, image_path: str) -> MockLayoutResult:
		"""Mock layout analysis"""
		await asyncio.sleep(0.3)
		
		regions = [
			{"type": "header", "x": 0, "y": 0, "width": 100, "height": 10, "confidence": 0.92},
			{"type": "paragraph", "x": 0, "y": 15, "width": 100, "height": 60, "confidence": 0.88},
			{"type": "footer", "x": 0, "y": 85, "width": 100, "height": 10, "confidence": 0.85}
		]
		
		elements = [
			{"type": "text_block", "content": "Header text", "position": {"x": 0, "y": 0}},
			{"type": "image", "description": "Inline image", "position": {"x": 50, "y": 30}},
			{"type": "table", "rows": 3, "cols": 2, "position": {"x": 0, "y": 50}}
		]
		
		structure = {
			"document_type": "article",
			"page_count": 1,
			"text_density": 0.75,
			"image_count": 1,
			"table_count": 1
		}
		
		return MockLayoutResult(regions=regions, elements=elements, structure=structure)
	
	async def assess_image_quality(self, image_path: str) -> float:
		"""Mock image quality assessment"""
		await asyncio.sleep(0.1)
		
		# Simulate quality assessment based on file type
		if image_path.endswith('.png'):
			quality = 0.95  # PNG typically high quality
		elif image_path.endswith('.jpg'):
			quality = 0.85  # JPEG variable quality
		elif image_path.endswith('.tiff'):
			quality = 0.98  # TIFF highest quality
		else:
			quality = 0.75  # Default quality
		
		logger.debug(f"Mock vision: Assessed quality {quality} for {image_path}")
		return quality


class MockAPGNLPService:
	"""Mock APG Natural Language Processing Service"""
	
	def __init__(self, ollama_url: str):
		self.ollama_url = ollama_url
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock NLP service"""
		self._initialized = True
		logger.info(f"Mock APG NLP Service initialized with Ollama URL: {self.ollama_url}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "nlp",
			"ollama_url": self.ollama_url,
			"models": ["gemma2:latest"]
		}
	
	async def extract_entities(self, content: str) -> List[Dict[str, Any]]:
		"""Mock entity extraction"""
		await asyncio.sleep(0.2)
		
		# Simple mock entity extraction based on common patterns
		entities = []
		
		# Mock person names (capitalized words)
		import re
		person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
		persons = re.findall(person_pattern, content)
		for person in persons:
			entities.append({
				"text": person,
				"label": "PERSON",
				"confidence": 0.87,
				"start": content.find(person),
				"end": content.find(person) + len(person)
			})
		
		# Mock dates
		date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
		dates = re.findall(date_pattern, content)
		for date in dates:
			entities.append({
				"text": date,
				"label": "DATE",
				"confidence": 0.95,
				"start": content.find(date),
				"end": content.find(date) + len(date)
			})
		
		logger.debug(f"Mock NLP: Extracted {len(entities)} entities from {len(content)} characters")
		return entities
	
	async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
		"""Mock sentiment analysis"""
		await asyncio.sleep(0.2)
		
		# Simple sentiment analysis based on word count
		positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
		negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
		
		content_lower = content.lower()
		pos_count = sum(1 for word in positive_words if word in content_lower)
		neg_count = sum(1 for word in negative_words if word in content_lower)
		
		if pos_count > neg_count:
			sentiment = "positive"
			score = 0.6 + (pos_count - neg_count) * 0.1
		elif neg_count > pos_count:
			sentiment = "negative"  
			score = 0.4 - (neg_count - pos_count) * 0.1
		else:
			sentiment = "neutral"
			score = 0.5
		
		score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
		
		return {
			"sentiment": sentiment,
			"score": score,
			"confidence": 0.82,
			"positive_indicators": pos_count,
			"negative_indicators": neg_count
		}
	
	async def generate_summary(self, content: str, max_sentences: int = 3) -> str:
		"""Mock text summarization"""
		await asyncio.sleep(0.3)
		
		sentences = content.split('. ')
		if len(sentences) <= max_sentences:
			summary = content
		else:
			# Take first sentence, middle sentence, and last sentence
			indices = [0, len(sentences)//2, len(sentences)-1]
			summary_sentences = [sentences[i] for i in indices if i < len(sentences)]
			summary = '. '.join(summary_sentences)
			if not summary.endswith('.'):
				summary += '.'
		
		logger.debug(f"Mock NLP: Generated summary of {len(summary)} characters from {len(content)} characters")
		return summary
	
	async def identify_topics(self, content: str) -> List[Dict[str, Any]]:
		"""Mock topic identification"""
		await asyncio.sleep(0.2)
		
		# Simple topic identification based on keywords
		topics = []
		content_lower = content.lower()
		
		topic_keywords = {
			"business": ["business", "company", "market", "sales", "revenue", "profit"],
			"technology": ["technology", "software", "computer", "digital", "system", "data"],
			"legal": ["legal", "contract", "law", "agreement", "compliance", "regulation"],
			"financial": ["financial", "money", "cost", "budget", "investment", "payment"],
			"general": ["document", "information", "content", "text"]
		}
		
		for topic, keywords in topic_keywords.items():
			matches = sum(1 for keyword in keywords if keyword in content_lower)
			if matches > 0:
				confidence = min(0.9, 0.3 + matches * 0.1)
				topics.append({
					"topic": topic,
					"confidence": confidence,
					"keywords_found": matches,
					"relevance_score": matches / len(keywords)
				})
		
		# Sort by confidence
		topics.sort(key=lambda x: x["confidence"], reverse=True)
		
		logger.debug(f"Mock NLP: Identified {len(topics)} topics")
		return topics


class MockAPGOrchestrationService:
	"""Mock APG AI Orchestration Service"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self._workflows: Dict[str, Dict[str, Any]] = {}
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock orchestration service"""
		self._initialized = True
		logger.info(f"Mock APG Orchestration Service initialized for tenant: {self.tenant_id}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "ai_orchestration", 
			"tenant_id": self.tenant_id,
			"active_workflows": len(self._workflows)
		}
	
	async def create_workflow(self, name: str, document_id: str, steps: List[str]) -> 'MockWorkflow':
		"""Mock workflow creation"""
		workflow_id = uuid7str()
		
		workflow_data = {
			"workflow_id": workflow_id,
			"name": name,
			"document_id": document_id,
			"steps": steps,
			"status": "created",
			"tenant_id": self.tenant_id,
			"created_at": datetime.utcnow().isoformat(),
			"progress": 0,
			"results": {}
		}
		
		self._workflows[workflow_id] = workflow_data
		
		logger.debug(f"Mock orchestration: Created workflow {workflow_id} with {len(steps)} steps")
		return MockWorkflow(workflow_data, self)
	
	async def execute_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Mock workflow execution"""
		if workflow_id not in self._workflows:
			raise ValueError(f"Workflow {workflow_id} not found")
		
		workflow = self._workflows[workflow_id]
		workflow["status"] = "running"
		workflow["context"] = context
		
		# Simulate execution time
		await asyncio.sleep(0.1)
		
		# Mock successful execution
		workflow["status"] = "completed"
		workflow["progress"] = 100
		workflow["results"] = {
			"success": True,
			"steps_completed": len(workflow["steps"]),
			"execution_time": 0.1,
			"output": "Workflow completed successfully"
		}
		
		return workflow["results"]
	
	def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
		"""Get workflow for testing"""
		return self._workflows.get(workflow_id)


class MockWorkflow:
	"""Mock workflow object"""
	
	def __init__(self, workflow_data: Dict[str, Any], orchestration_service: MockAPGOrchestrationService):
		self.data = workflow_data
		self.service = orchestration_service
	
	@property
	def workflow_id(self) -> str:
		return self.data["workflow_id"]
	
	@property
	def status(self) -> str:
		return self.data["status"]
	
	async def complete(self, results: Dict[str, Any]) -> None:
		"""Mark workflow as completed"""
		self.data["status"] = "completed"
		self.data["results"] = results
		self.data["completed_at"] = datetime.utcnow().isoformat()
		logger.debug(f"Mock workflow {self.workflow_id} completed")


class MockAPGCollaborationService:
	"""Mock APG Real-time Collaboration Service"""
	
	def __init__(self):
		self._sessions: Dict[str, Dict[str, Any]] = {}
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock collaboration service"""
		self._initialized = True
		logger.info("Mock APG Collaboration Service initialized")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "real_time_collaboration",
			"active_sessions": len(self._sessions)
		}


class MockAPGNotificationService:
	"""Mock APG Notification Service"""
	
	def __init__(self):
		self._notifications: List[Dict[str, Any]] = []
		self._initialized = False
	
	async def initialize(self) -> None:
		"""Initialize mock notification service"""
		self._initialized = True
		logger.info("Mock APG Notification Service initialized")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health"""
		return {
			"healthy": self._initialized,
			"service": "notification",
			"notifications_sent": len(self._notifications)
		}
	
	async def send_notification(self, user_id: str, message: str, priority: str = "normal") -> None:
		"""Mock notification sending"""
		notification = {
			"notification_id": uuid7str(),
			"user_id": user_id,
			"message": message,
			"priority": priority,
			"timestamp": datetime.utcnow().isoformat(),
			"status": "sent"
		}
		
		self._notifications.append(notification)
		logger.debug(f"Mock notification: Sent to {user_id}: {message}")
	
	def get_notifications(self) -> List[Dict[str, Any]]:
		"""Get notifications for testing"""
		return self._notifications.copy()