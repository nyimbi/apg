"""
APG Threat Detection & Monitoring - REST API

FastAPI-based REST API for threat detection and security monitoring
with comprehensive endpoint coverage and enterprise security features.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	SecurityEvent, ThreatIndicator, SecurityIncident, BehavioralProfile,
	ThreatIntelligence, SecurityRule, IncidentResponse, ThreatAnalysis,
	SecurityMetrics, ForensicEvidence, ThreatSeverity, ThreatStatus
)
from .service import ThreatDetectionService


class ThreatDetectionAPI:
	"""Threat Detection REST API endpoints"""
	
	def __init__(self):
		self.router = APIRouter(prefix="/api/v1/threat-detection", tags=["Threat Detection"])
		self._setup_routes()
	
	def _setup_routes(self):
		"""Setup API routes"""
		
		@self.router.post("/events", response_model=Dict[str, Any])
		async def submit_security_event(
			event_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Submit security event for analysis"""
			try:
				event = await service.process_security_event(event_data)
				return {
					"event_id": event.id,
					"status": "processed",
					"risk_score": float(event.risk_score),
					"confidence": float(event.confidence),
					"timestamp": event.timestamp.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error processing security event: {str(e)}"
				)
		
		@self.router.get("/events/{event_id}", response_model=Dict[str, Any])
		async def get_security_event(
			event_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get security event details"""
			try:
				event = await service.get_security_event(event_id)
				if not event:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Security event not found"
					)
				return event.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving security event: {str(e)}"
				)
		
		@self.router.get("/events", response_model=Dict[str, Any])
		async def list_security_events(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			event_type: Optional[str] = Query(None),
			severity: Optional[ThreatSeverity] = Query(None),
			start_time: Optional[datetime] = Query(None),
			end_time: Optional[datetime] = Query(None),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""List security events with filtering"""
			try:
				events, total_count = await service.list_security_events(
					limit=limit,
					offset=offset,
					event_type=event_type,
					severity=severity,
					start_time=start_time,
					end_time=end_time
				)
				
				return {
					"events": [event.dict() for event in events],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing security events: {str(e)}"
				)
		
		@self.router.post("/incidents", response_model=Dict[str, Any])
		async def create_security_incident(
			incident_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Create security incident"""
			try:
				incident = await service.create_security_incident(incident_data)
				return {
					"incident_id": incident.id,
					"status": "created",
					"severity": incident.severity.value,
					"created_at": incident.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating security incident: {str(e)}"
				)
		
		@self.router.get("/incidents/{incident_id}", response_model=Dict[str, Any])
		async def get_security_incident(
			incident_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get security incident details"""
			try:
				incident = await service.get_security_incident(incident_id)
				if not incident:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Security incident not found"
					)
				return incident.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving security incident: {str(e)}"
				)
		
		@self.router.put("/incidents/{incident_id}", response_model=Dict[str, Any])
		async def update_security_incident(
			incident_id: str,
			incident_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Update security incident"""
			try:
				incident = await service.update_security_incident(incident_id, incident_data)
				return {
					"incident_id": incident.id,
					"status": "updated",
					"updated_at": incident.updated_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error updating security incident: {str(e)}"
				)
		
		@self.router.get("/incidents", response_model=Dict[str, Any])
		async def list_security_incidents(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			severity: Optional[ThreatSeverity] = Query(None),
			status: Optional[ThreatStatus] = Query(None),
			assigned_to: Optional[str] = Query(None),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""List security incidents with filtering"""
			try:
				incidents, total_count = await service.list_security_incidents(
					limit=limit,
					offset=offset,
					severity=severity,
					status=status,
					assigned_to=assigned_to
				)
				
				return {
					"incidents": [incident.dict() for incident in incidents],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing security incidents: {str(e)}"
				)
		
		@self.router.post("/behavioral-analysis", response_model=Dict[str, Any])
		async def analyze_behavioral_patterns(
			user_id: str,
			period_days: int = Query(30, ge=1, le=365),
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Analyze user behavioral patterns"""
			try:
				profile = await service.analyze_behavioral_patterns(user_id, period_days)
				return {
					"profile_id": profile.id,
					"user_id": user_id,
					"baseline_established": profile.baseline_established,
					"baseline_confidence": float(profile.baseline_confidence),
					"anomaly_score": float(profile.anomaly_score),
					"risk_score": float(profile.risk_score),
					"analysis_period": {
						"start": profile.profile_period_start.isoformat(),
						"end": profile.profile_period_end.isoformat()
					}
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error analyzing behavioral patterns: {str(e)}"
				)
		
		@self.router.get("/behavioral-profiles/{user_id}", response_model=Dict[str, Any])
		async def get_behavioral_profile(
			user_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get user behavioral profile"""
			try:
				profile = await service.get_behavioral_profile(user_id)
				if not profile:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Behavioral profile not found"
					)
				return profile.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving behavioral profile: {str(e)}"
				)
		
		@self.router.post("/threat-hunting", response_model=Dict[str, Any])
		async def execute_threat_hunt(
			hypothesis: str,
			query_params: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Execute threat hunting query"""
			try:
				hunt_results = await service.hunt_threats(hypothesis, query_params)
				return hunt_results
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error executing threat hunt: {str(e)}"
				)
		
		@self.router.get("/threat-intelligence", response_model=Dict[str, Any])
		async def get_threat_intelligence(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			intelligence_type: Optional[str] = Query(None),
			severity: Optional[ThreatSeverity] = Query(None),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get threat intelligence data"""
			try:
				intelligence, total_count = await service.get_threat_intelligence(
					limit=limit,
					offset=offset,
					intelligence_type=intelligence_type,
					severity=severity
				)
				
				return {
					"intelligence": [intel.dict() for intel in intelligence],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving threat intelligence: {str(e)}"
				)
		
		@self.router.post("/indicators", response_model=Dict[str, Any])
		async def create_threat_indicator(
			indicator_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Create threat indicator (IOC)"""
			try:
				indicator = await service.create_threat_indicator(indicator_data)
				return {
					"indicator_id": indicator.id,
					"status": "created",
					"indicator_type": indicator.indicator_type,
					"indicator_value": indicator.indicator_value,
					"severity": indicator.severity.value
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating threat indicator: {str(e)}"
				)
		
		@self.router.get("/indicators", response_model=Dict[str, Any])
		async def list_threat_indicators(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			indicator_type: Optional[str] = Query(None),
			severity: Optional[ThreatSeverity] = Query(None),
			is_active: Optional[bool] = Query(None),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""List threat indicators with filtering"""
			try:
				indicators, total_count = await service.list_threat_indicators(
					limit=limit,
					offset=offset,
					indicator_type=indicator_type,
					severity=severity,
					is_active=is_active
				)
				
				return {
					"indicators": [indicator.dict() for indicator in indicators],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing threat indicators: {str(e)}"
				)
		
		@self.router.post("/rules", response_model=Dict[str, Any])
		async def create_security_rule(
			rule_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Create security detection rule"""
			try:
				rule = await service.create_security_rule(rule_data)
				return {
					"rule_id": rule.id,
					"status": "created",
					"rule_name": rule.name,
					"is_active": rule.is_active
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating security rule: {str(e)}"
				)
		
		@self.router.get("/rules", response_model=Dict[str, Any])
		async def list_security_rules(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			rule_type: Optional[str] = Query(None),
			is_active: Optional[bool] = Query(None),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""List security detection rules"""
			try:
				rules, total_count = await service.list_security_rules(
					limit=limit,
					offset=offset,
					rule_type=rule_type,
					is_active=is_active
				)
				
				return {
					"rules": [rule.dict() for rule in rules],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing security rules: {str(e)}"
				)
		
		@self.router.put("/rules/{rule_id}", response_model=Dict[str, Any])
		async def update_security_rule(
			rule_id: str,
			rule_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Update security detection rule"""
			try:
				rule = await service.update_security_rule(rule_id, rule_data)
				return {
					"rule_id": rule.id,
					"status": "updated",
					"updated_at": rule.updated_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error updating security rule: {str(e)}"
				)
		
		@self.router.post("/response/execute", response_model=Dict[str, Any])
		async def execute_incident_response(
			incident_id: str,
			playbook_name: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Execute automated incident response"""
			try:
				response = await service.execute_incident_response(incident_id, playbook_name)
				return {
					"response_id": response.id,
					"status": response.status,
					"playbook_name": response.playbook_name,
					"start_time": response.start_time.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error executing incident response: {str(e)}"
				)
		
		@self.router.get("/response/{response_id}", response_model=Dict[str, Any])
		async def get_incident_response(
			response_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get incident response details"""
			try:
				response = await service.get_incident_response(response_id)
				if not response:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Incident response not found"
					)
				return response.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving incident response: {str(e)}"
				)
		
		@self.router.get("/metrics", response_model=Dict[str, Any])
		async def get_security_metrics(
			tenant_id: str = Query(..., description="Tenant identifier"),
			period_days: int = Query(7, ge=1, le=365),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get security operations metrics"""
			try:
				metrics = await service.get_security_metrics(period_days)
				return metrics.dict()
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving security metrics: {str(e)}"
				)
		
		@self.router.get("/dashboard", response_model=Dict[str, Any])
		async def get_security_dashboard(
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get security operations dashboard data"""
			try:
				dashboard_data = await service.get_security_dashboard()
				return dashboard_data
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving security dashboard: {str(e)}"
				)
		
		@self.router.post("/forensics/evidence", response_model=Dict[str, Any])
		async def collect_forensic_evidence(
			incident_id: str,
			evidence_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Collect forensic evidence for incident"""
			try:
				evidence = await service.collect_forensic_evidence(incident_id, evidence_data)
				return {
					"evidence_id": evidence.id,
					"status": "collected",
					"evidence_type": evidence.evidence_type,
					"integrity_verified": evidence.integrity_verified
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error collecting forensic evidence: {str(e)}"
				)
		
		@self.router.get("/forensics/evidence/{incident_id}", response_model=Dict[str, Any])
		async def get_forensic_evidence(
			incident_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get forensic evidence for incident"""
			try:
				evidence_list = await service.get_forensic_evidence(incident_id)
				return {
					"incident_id": incident_id,
					"evidence": [evidence.dict() for evidence in evidence_list],
					"evidence_count": len(evidence_list)
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving forensic evidence: {str(e)}"
				)
		
		@self.router.post("/analysis/threat", response_model=Dict[str, Any])
		async def analyze_threat(
			analysis_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Perform advanced threat analysis"""
			try:
				analysis = await service.analyze_threat(analysis_data)
				return {
					"analysis_id": analysis.id,
					"threat_score": float(analysis.threat_score),
					"confidence_score": float(analysis.confidence_score),
					"analysis_engine": analysis.analysis_engine.value,
					"findings_count": len(analysis.findings)
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error performing threat analysis: {str(e)}"
				)
		
		@self.router.get("/analysis/{analysis_id}", response_model=Dict[str, Any])
		async def get_threat_analysis(
			analysis_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatDetectionService = Depends(self._get_service)
		):
			"""Get threat analysis results"""
			try:
				analysis = await service.get_threat_analysis(analysis_id)
				if not analysis:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Threat analysis not found"
					)
				return analysis.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving threat analysis: {str(e)}"
				)
	
	def _get_service(self, db: AsyncSession, tenant_id: str) -> ThreatDetectionService:
		"""Dependency injection for ThreatDetectionService"""
		return ThreatDetectionService(db, tenant_id)
	
	def get_router(self) -> APIRouter:
		"""Get the configured router"""
		return self.router