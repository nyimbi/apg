"""
APG Integration Stubs for Development and Testing

These stubs simulate APG services for local development when the actual APG
services are not available.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class APGAuthStub:
	"""Mock APG authentication service"""
	
	async def validate_token(self, token: str) -> Dict[str, Any]:
		"""Mock token validation"""
		return {
			"valid": True,
			"user_id": "dev_user_123",
			"tenant_id": "dev_tenant",
			"permissions": ["rtc:read", "rtc:write", "rtc:admin"],
			"expires_at": "2025-12-31T23:59:59Z"
		}
	
	async def get_user_info(self, user_id: str) -> Dict[str, Any]:
		"""Mock user information"""
		return {
			"user_id": user_id,
			"username": f"dev_user_{user_id[-3:]}",
			"display_name": f"Development User {user_id[-3:]}",
			"email": f"dev.user.{user_id[-3:]}@datacraft.co.ke",
			"tenant_id": "dev_tenant",
			"role": "admin"
		}

class APGAIStub:
	"""Mock APG AI orchestration service"""
	
	async def suggest_participants(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Mock participant suggestions"""
		await asyncio.sleep(0.1)  # Simulate processing time
		
		return [
			{
				"user_id": "ai_suggested_001",
				"display_name": "AI Expert Alice",
				"expertise": ["collaboration", "project_management"],
				"confidence": 0.95,
				"reason": "High expertise in collaborative workflows"
			},
			{
				"user_id": "ai_suggested_002", 
				"display_name": "AI Expert Bob",
				"expertise": ["technical_support", "troubleshooting"],
				"confidence": 0.87,
				"reason": "Strong technical problem-solving skills"
			}
		]
	
	async def route_assistance_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
		"""Mock assistance request routing"""
		await asyncio.sleep(0.2)  # Simulate AI processing
		
		return {
			"routed_to": "ai_expert_001",
			"priority": "high",
			"estimated_response_time": "2-5 minutes",
			"routing_reason": "Best match for request type and context",
			"suggested_actions": [
				"Review the form field in question",
				"Check business rules for validation",
				"Provide contextual guidance"
			]
		}
	
	async def analyze_collaboration_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
		"""Mock collaboration context analysis"""
		return {
			"page_type": "form_entry",
			"business_process": "user_management",
			"complexity_score": 0.7,
			"suggested_collaboration_mode": "guided",
			"key_fields": ["email", "role", "permissions"],
			"potential_issues": ["email_validation", "role_conflicts"]
		}

class APGNotificationStub:
	"""Mock APG notification engine"""
	
	async def send_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
		"""Mock notification sending"""
		logger.info(f"Mock notification sent: {notification.get('title', 'No title')}")
		
		return {
			"notification_id": f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
			"status": "sent",
			"delivery_method": "mock",
			"sent_at": datetime.now().isoformat()
		}
	
	async def send_assistance_alert(self, user_id: str, request: Dict[str, Any]) -> bool:
		"""Mock assistance alert"""
		logger.info(f"Mock assistance alert sent to {user_id}")
		return True
	
	async def send_delegation_notification(self, delegatee_id: str, delegation: Dict[str, Any]) -> bool:
		"""Mock delegation notification"""
		logger.info(f"Mock delegation notification sent to {delegatee_id}")
		return True

# Global stub instances
auth_service = APGAuthStub()
ai_service = APGAIStub()
notification_service = APGNotificationStub()

# Utility functions for easy access
async def get_user_permissions(user_id: str) -> List[str]:
	"""Get user permissions (mocked)"""
	user_info = await auth_service.get_user_info(user_id)
	return ["rtc:read", "rtc:write", "rtc:admin"]  # Mock permissions

async def validate_user_token(token: str) -> bool:
	"""Validate user token (mocked)"""
	result = await auth_service.validate_token(token)
	return result.get("valid", False)

async def get_ai_participant_suggestions(page_url: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""Get AI-powered participant suggestions (mocked)"""
	return await ai_service.suggest_participants({
		"page_url": page_url,
		"context": context
	})

def is_development_mode() -> bool:
	"""Check if running in development mode"""
	import os
	return os.getenv('ENVIRONMENT', 'development') == 'development'

def get_mock_tenant_id() -> str:
	"""Get mock tenant ID for development"""
	return "dev_tenant_123"

def get_mock_user_id() -> str:
	"""Get mock user ID for development"""
	return "dev_user_123"

class APGCapabilityStub:
	"""Mock APG capability integration"""
	
	def __init__(self, capability_name: str):
		self.capability_name = capability_name
		self.logger = logging.getLogger(f"apg.{capability_name}")
	
	async def register_capability(self) -> bool:
		"""Mock capability registration"""
		self.logger.info(f"Mock registration of {self.capability_name} capability")
		return True
	
	async def get_capability_status(self) -> Dict[str, Any]:
		"""Mock capability status"""
		return {
			"capability": self.capability_name,
			"status": "active",
			"version": "1.0.0-dev",
			"health": "healthy",
			"last_check": datetime.now().isoformat()
		}

# Create RTC capability stub
rtc_capability = APGCapabilityStub("real_time_collaboration")

if __name__ == "__main__":
	# Test the stubs
	async def test_stubs():
		print("Testing APG integration stubs...")
		
		# Test auth
		token_result = await auth_service.validate_token("test_token")
		print(f"Auth validation: {token_result}")
		
		# Test AI
		suggestions = await ai_service.suggest_participants({"page_url": "/admin/users"})
		print(f"AI suggestions: {len(suggestions)} participants")
		
		# Test notifications
		notif_result = await notification_service.send_notification({
			"title": "Test Notification",
			"message": "This is a test"
		})
		print(f"Notification result: {notif_result}")
		
		print("✅ All APG stubs working correctly")
	
	asyncio.run(test_stubs())