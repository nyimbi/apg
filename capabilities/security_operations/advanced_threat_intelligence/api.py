"""
APG Advanced Threat Intelligence - REST API

FastAPI-based REST API for threat intelligence orchestration with
comprehensive endpoint coverage and enterprise intelligence features.

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
	IntelligenceFeed, ThreatActor, AttackCampaign, ThreatIndicator,
	IntelligenceEnrichment, AttributionAnalysis, IntelligenceAlert,
	IntelligenceReport, IntelligenceMetrics, FeedType, ThreatSeverity
)
from .service import ThreatIntelligenceService


class ThreatIntelligenceAPI:
	"""Threat Intelligence REST API endpoints"""
	
	def __init__(self):
		self.router = APIRouter(prefix="/api/v1/threat-intelligence", tags=["Threat Intelligence"])
		self._setup_routes()
	
	def _setup_routes(self):
		"""Setup API routes"""
		
		@self.router.post("/feeds", response_model=Dict[str, Any])
		async def create_intelligence_feed(
			feed_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Create new intelligence feed"""
			try:
				feed = await service.create_intelligence_feed(feed_data)
				return {
					"feed_id": feed.id,
					"status": "created",
					"name": feed.name,
					"feed_type": feed.feed_type.value,
					"quality_score": float(feed.quality_score),
					"created_at": feed.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating intelligence feed: {str(e)}"
				)
		
		@self.router.get("/feeds", response_model=Dict[str, Any])
		async def list_intelligence_feeds(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			feed_type: Optional[FeedType] = Query(None),
			is_active: Optional[bool] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""List intelligence feeds with filtering"""
			try:
				feeds, total_count = await service.list_intelligence_feeds(
					limit=limit,
					offset=offset,
					feed_type=feed_type,
					is_active=is_active
				)
				
				return {
					"feeds": [feed.dict() for feed in feeds],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing intelligence feeds: {str(e)}"
				)
		
		@self.router.post("/feeds/{feed_id}/ingest", response_model=Dict[str, Any])
		async def ingest_threat_intelligence(
			feed_id: str,
			intelligence_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Ingest threat intelligence data"""
			try:
				indicators = await service.ingest_threat_intelligence(feed_id, intelligence_data)
				return {
					"feed_id": feed_id,
					"status": "processed",
					"indicators_processed": len(indicators),
					"processing_time": datetime.utcnow().isoformat(),
					"indicators": [
						{
							"id": ind.id,
							"type": ind.indicator_type,
							"value": ind.indicator_value,
							"severity": ind.severity.value,
							"confidence": ind.confidence.value
						} for ind in indicators
					]
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error ingesting threat intelligence: {str(e)}"
				)
		
		@self.router.get("/indicators", response_model=Dict[str, Any])
		async def list_threat_indicators(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			indicator_type: Optional[str] = Query(None),
			severity: Optional[ThreatSeverity] = Query(None),
			is_active: Optional[bool] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
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
		
		@self.router.get("/indicators/{indicator_id}", response_model=Dict[str, Any])
		async def get_threat_indicator(
			indicator_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Get threat indicator details"""
			try:
				indicator = await service.get_threat_indicator(indicator_id)
				if not indicator:
					raise HTTPException(
						status_code=status.HTTP_404_NOT_FOUND,
						detail="Threat indicator not found"
					)
				return indicator.dict()
			except HTTPException:
				raise
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving threat indicator: {str(e)}"
				)
		
		@self.router.post("/indicators/{indicator_id}/enrich", response_model=Dict[str, Any])
		async def enrich_threat_indicator(
			indicator_id: str,
			enrichment_config: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Enrich threat indicator with additional context"""
			try:
				enrichment = await service.enrich_threat_indicator(indicator_id, enrichment_config)
				return {
					"indicator_id": indicator_id,
					"status": "enriched",
					"enrichment_id": enrichment.id,
					"confidence_score": float(enrichment.confidence_score),
					"relevance_score": float(enrichment.relevance_score),
					"enrichment_timestamp": enrichment.enrichment_timestamp.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error enriching threat indicator: {str(e)}"
				)
		
		@self.router.post("/actors", response_model=Dict[str, Any])
		async def create_threat_actor(
			actor_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Create threat actor profile"""
			try:
				actor = await service.create_threat_actor_profile(actor_data)
				return {
					"actor_id": actor.id,
					"status": "created",
					"name": actor.name,
					"actor_type": actor.actor_type.value,
					"threat_score": float(actor.threat_score),
					"confidence_score": float(actor.confidence_score),
					"created_at": actor.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating threat actor: {str(e)}"
				)
		
		@self.router.get("/actors", response_model=Dict[str, Any])
		async def list_threat_actors(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			actor_type: Optional[str] = Query(None),
			is_active: Optional[bool] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""List threat actors with filtering"""
			try:
				actors, total_count = await service.list_threat_actors(
					limit=limit,
					offset=offset,
					actor_type=actor_type,
					is_active=is_active
				)
				
				return {
					"actors": [actor.dict() for actor in actors],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing threat actors: {str(e)}"
				)
		
		@self.router.post("/campaigns", response_model=Dict[str, Any])
		async def create_attack_campaign(
			campaign_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Create attack campaign tracking"""
			try:
				campaign = await service.track_attack_campaign(campaign_data)
				return {
					"campaign_id": campaign.id,
					"status": "created",
					"name": campaign.name,
					"severity": campaign.severity.value,
					"attributed_actors": campaign.attributed_actors,
					"start_date": campaign.start_date.isoformat(),
					"created_at": campaign.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating attack campaign: {str(e)}"
				)
		
		@self.router.get("/campaigns", response_model=Dict[str, Any])
		async def list_attack_campaigns(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			is_active: Optional[bool] = Query(None),
			severity: Optional[ThreatSeverity] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""List attack campaigns with filtering"""
			try:
				campaigns, total_count = await service.list_attack_campaigns(
					limit=limit,
					offset=offset,
					is_active=is_active,
					severity=severity
				)
				
				return {
					"campaigns": [campaign.dict() for campaign in campaigns],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing attack campaigns: {str(e)}"
				)
		
		@self.router.post("/attribution/analyze", response_model=Dict[str, Any])
		async def analyze_attribution(
			analysis_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Perform threat attribution analysis"""
			try:
				attribution = await service.analyze_threat_attribution(analysis_data)
				return {
					"analysis_id": attribution.id,
					"attribution_confidence": float(attribution.attribution_confidence),
					"attributed_actors": attribution.attributed_actors,
					"attribution_methods": [method.value for method in attribution.attribution_methods],
					"analysis_timestamp": attribution.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error performing attribution analysis: {str(e)}"
				)
		
		@self.router.post("/alerts", response_model=Dict[str, Any])
		async def create_intelligence_alert(
			alert_data: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Create intelligence-based alert"""
			try:
				alert = await service.generate_intelligence_alert(alert_data)
				return {
					"alert_id": alert.id,
					"status": "created",
					"alert_type": alert.alert_type,
					"severity": alert.severity.value,
					"confidence": alert.confidence.value,
					"recommended_actions": alert.recommended_actions,
					"alert_timestamp": alert.alert_timestamp.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error creating intelligence alert: {str(e)}"
				)
		
		@self.router.get("/alerts", response_model=Dict[str, Any])
		async def list_intelligence_alerts(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			severity: Optional[ThreatSeverity] = Query(None),
			is_acknowledged: Optional[bool] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""List intelligence alerts with filtering"""
			try:
				alerts, total_count = await service.list_intelligence_alerts(
					limit=limit,
					offset=offset,
					severity=severity,
					is_acknowledged=is_acknowledged
				)
				
				return {
					"alerts": [alert.dict() for alert in alerts],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing intelligence alerts: {str(e)}"
				)
		
		@self.router.post("/hunting", response_model=Dict[str, Any])
		async def intelligence_driven_hunting(
			hunt_query: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Execute intelligence-driven threat hunting"""
			try:
				hunt_results = await service.hunt_with_intelligence(hunt_query)
				return hunt_results
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error executing intelligence hunting: {str(e)}"
				)
		
		@self.router.post("/reports/generate", response_model=Dict[str, Any])
		async def generate_intelligence_report(
			report_config: Dict[str, Any],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Generate comprehensive intelligence report"""
			try:
				report = await service.generate_intelligence_report(report_config)
				return {
					"report_id": report.id,
					"status": "generated",
					"report_type": report.report_type,
					"title": report.title,
					"period_start": report.report_period_start.isoformat(),
					"period_end": report.report_period_end.isoformat(),
					"key_findings_count": len(report.key_findings),
					"generated_at": report.created_at.isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error generating intelligence report: {str(e)}"
				)
		
		@self.router.get("/reports", response_model=Dict[str, Any])
		async def list_intelligence_reports(
			tenant_id: str = Query(..., description="Tenant identifier"),
			limit: int = Query(100, ge=1, le=1000),
			offset: int = Query(0, ge=0),
			report_type: Optional[str] = Query(None),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""List intelligence reports"""
			try:
				reports, total_count = await service.list_intelligence_reports(
					limit=limit,
					offset=offset,
					report_type=report_type
				)
				
				return {
					"reports": [report.dict() for report in reports],
					"total_count": total_count,
					"limit": limit,
					"offset": offset
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error listing intelligence reports: {str(e)}"
				)
		
		@self.router.get("/metrics", response_model=Dict[str, Any])
		async def get_intelligence_metrics(
			tenant_id: str = Query(..., description="Tenant identifier"),
			period_days: int = Query(30, ge=1, le=365),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Get intelligence operations metrics"""
			try:
				metrics = await service.get_intelligence_metrics(period_days)
				return metrics.dict()
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving intelligence metrics: {str(e)}"
				)
		
		@self.router.get("/dashboard", response_model=Dict[str, Any])
		async def get_intelligence_dashboard(
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Get intelligence operations dashboard data"""
			try:
				dashboard_data = await service.get_intelligence_dashboard()
				return dashboard_data
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error retrieving intelligence dashboard: {str(e)}"
				)
		
		@self.router.post("/feeds/{feed_id}/test", response_model=Dict[str, Any])
		async def test_intelligence_feed(
			feed_id: str,
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Test intelligence feed connectivity"""
			try:
				test_result = await service.test_feed_connectivity(feed_id)
				return {
					"feed_id": feed_id,
					"connectivity": test_result.get("status", "unknown"),
					"response_time": test_result.get("response_time", 0),
					"last_update": test_result.get("last_update"),
					"test_timestamp": datetime.utcnow().isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error testing intelligence feed: {str(e)}"
				)
		
		@self.router.post("/indicators/bulk-import", response_model=Dict[str, Any])
		async def bulk_import_indicators(
			indicators_data: List[Dict[str, Any]],
			tenant_id: str = Query(..., description="Tenant identifier"),
			service: ThreatIntelligenceService = Depends(self._get_service)
		):
			"""Bulk import threat indicators"""
			try:
				import_result = await service.bulk_import_indicators(indicators_data)
				return {
					"status": "completed",
					"total_indicators": len(indicators_data),
					"successful_imports": import_result.get("successful", 0),
					"failed_imports": import_result.get("failed", 0),
					"duplicates_skipped": import_result.get("duplicates", 0),
					"import_timestamp": datetime.utcnow().isoformat()
				}
			except Exception as e:
				raise HTTPException(
					status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
					detail=f"Error bulk importing indicators: {str(e)}"
				)
	
	def _get_service(self, db: AsyncSession, tenant_id: str) -> ThreatIntelligenceService:
		"""Dependency injection for ThreatIntelligenceService"""
		return ThreatIntelligenceService(db, tenant_id)
	
	def get_router(self) -> APIRouter:
		"""Get the configured router"""
		return self.router