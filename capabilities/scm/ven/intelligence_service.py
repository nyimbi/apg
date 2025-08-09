"""
APG Vendor Management - Intelligence Service
Advanced AI-powered vendor intelligence and analytics service

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from uuid_extensions import uuid7str

from .models import (
	VMVendor, VMPerformance, VMRisk, VMIntelligence, VMBenchmark,
	VendorIntelligenceSummary, VendorAIDecision, VendorOptimizationPlan,
	RiskSeverity, VendorStatus
)
from .service import VMDatabaseContext


# ============================================================================
# VENDOR INTELLIGENCE ENGINE
# ============================================================================

class VendorIntelligenceEngine:
	"""
	Advanced AI-powered vendor intelligence engine
	Provides predictive analytics, behavior analysis, and optimization recommendations
	"""
	
	def __init__(self, tenant_id: UUID, db_context: VMDatabaseContext):
		self.tenant_id = tenant_id
		self.db_context = db_context
		self._ai_models_cache: Dict[str, Any] = {}
		self._current_user_id: Optional[UUID] = None
	
	def set_current_user(self, user_id: UUID) -> None:
		"""Set current user context"""
		self._current_user_id = user_id
	
	def _log_pretty_path(self, vendor_id: str) -> str:
		"""Format vendor path for logging"""
		return f"intelligence/vendor/{vendor_id[:8]}..."
	
	# ========================================================================
	# BEHAVIORAL PATTERN ANALYSIS
	# ========================================================================
	
	async def analyze_vendor_behavior_patterns(
		self, 
		vendor_id: str,
		analysis_period_days: int = 180
	) -> List[Dict[str, Any]]:
		"""Analyze vendor behavior patterns using AI-powered pattern recognition"""
		
		# Collect behavioral data from multiple sources
		behavior_data = await self._collect_behavioral_data(vendor_id, analysis_period_days)
		
		# Run AI pattern analysis
		patterns = []
		
		# Communication behavior analysis
		comm_pattern = await self._analyze_communication_patterns(vendor_id, behavior_data)
		if comm_pattern:
			patterns.append(comm_pattern)
		
		# Performance consistency analysis
		perf_pattern = await self._analyze_performance_consistency(vendor_id, behavior_data)
		if perf_pattern:
			patterns.append(perf_pattern)
		
		# Risk behavior analysis
		risk_pattern = await self._analyze_risk_behavior(vendor_id, behavior_data)
		if risk_pattern:
			patterns.append(risk_pattern)
		
		# Innovation behavior analysis
		innovation_pattern = await self._analyze_innovation_behavior(vendor_id, behavior_data)
		if innovation_pattern:
			patterns.append(innovation_pattern)
		
		# Compliance behavior analysis
		compliance_pattern = await self._analyze_compliance_behavior(vendor_id, behavior_data)
		if compliance_pattern:
			patterns.append(compliance_pattern)
		
		return patterns
	
	async def _collect_behavioral_data(
		self, 
		vendor_id: str, 
		period_days: int
	) -> Dict[str, Any]:
		"""Collect comprehensive behavioral data for analysis"""
		
		connection = await self.db_context.get_connection()
		try:
			cutoff_date = datetime.utcnow() - timedelta(days=period_days)
			
			# Performance data
			perf_query = """
				SELECT * FROM vm_performance 
				WHERE vendor_id = $1 AND tenant_id = $2 AND start_date >= $3
				ORDER BY start_date DESC
			"""
			performance_data = await connection.fetch(
				perf_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			# Communication data
			comm_query = """
				SELECT * FROM vm_communication 
				WHERE vendor_id = $1 AND tenant_id = $2 AND communication_date >= $3
				ORDER BY communication_date DESC
			"""
			communication_data = await connection.fetch(
				comm_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			# Risk data
			risk_query = """
				SELECT * FROM vm_risk 
				WHERE vendor_id = $1 AND tenant_id = $2 AND identified_date >= $3
				ORDER BY identified_date DESC
			"""
			risk_data = await connection.fetch(
				risk_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			# Compliance data
			compliance_query = """
				SELECT * FROM vm_compliance 
				WHERE vendor_id = $1 AND tenant_id = $2 AND created_at >= $3
				ORDER BY created_at DESC
			"""
			compliance_data = await connection.fetch(
				compliance_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			return {
				'performance': [dict(row) for row in performance_data],
				'communication': [dict(row) for row in communication_data],
				'risk': [dict(row) for row in risk_data],
				'compliance': [dict(row) for row in compliance_data],
				'analysis_period': period_days,
				'data_points': {
					'performance_records': len(performance_data),
					'communications': len(communication_data),
					'risk_events': len(risk_data),
					'compliance_checks': len(compliance_data)
				}
			}
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def _analyze_communication_patterns(
		self, 
		vendor_id: str, 
		behavior_data: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Analyze communication behavior patterns"""
		
		communications = behavior_data.get('communication', [])
		if len(communications) < 5:  # Need minimum data for pattern analysis
			return None
		
		# Calculate response time patterns
		response_times = []
		sentiment_scores = []
		
		for comm in communications:
			if comm.get('direction') == 'inbound' and comm.get('sentiment_score'):
				sentiment_scores.append(float(comm['sentiment_score']))
		
		if not sentiment_scores:
			return None
		
		avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
		sentiment_variance = np.var(sentiment_scores) if len(sentiment_scores) > 1 else 0
		
		# Determine communication pattern
		if avg_sentiment > 0.3:
			pattern_type = "positive_communicator"
			description = "Vendor consistently demonstrates positive communication tone"
			confidence = min(0.95, 0.7 + (avg_sentiment * 0.25))
		elif avg_sentiment < -0.3:
			pattern_type = "challenging_communicator"
			description = "Vendor shows patterns of challenging or negative communication"
			confidence = min(0.95, 0.7 + (abs(avg_sentiment) * 0.25))
		else:
			pattern_type = "neutral_communicator"
			description = "Vendor maintains neutral communication patterns"
			confidence = 0.8
		
		# Analyze trend
		if len(sentiment_scores) >= 10:
			recent_sentiment = sum(sentiment_scores[:5]) / 5
			older_sentiment = sum(sentiment_scores[-5:]) / 5
			
			if recent_sentiment > older_sentiment + 0.1:
				trend = "improving"
			elif recent_sentiment < older_sentiment - 0.1:
				trend = "declining"
			else:
				trend = "stable"
		else:
			trend = "stable"
		
		return {
			"pattern_type": pattern_type,
			"category": "communication_behavior",
			"confidence": confidence,
			"description": description,
			"trend": trend,
			"metrics": {
				"average_sentiment": avg_sentiment,
				"sentiment_variance": sentiment_variance,
				"communication_frequency": len(communications) / behavior_data['analysis_period'] * 30,
				"data_quality": min(1.0, len(communications) / 20)
			},
			"impact": "positive" if avg_sentiment > 0 else "negative" if avg_sentiment < 0 else "neutral"
		}
	
	async def _analyze_performance_consistency(
		self, 
		vendor_id: str, 
		behavior_data: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Analyze performance consistency patterns"""
		
		performance_records = behavior_data.get('performance', [])
		if len(performance_records) < 3:
			return None
		
		# Extract performance scores over time
		overall_scores = [float(p['overall_score']) for p in performance_records]
		quality_scores = [float(p['quality_score']) for p in performance_records]
		delivery_scores = [float(p['delivery_score']) for p in performance_records]
		
		# Calculate consistency metrics
		overall_variance = np.var(overall_scores)
		quality_variance = np.var(quality_scores)
		delivery_variance = np.var(delivery_scores)
		
		avg_overall = sum(overall_scores) / len(overall_scores)
		
		# Determine consistency pattern
		if overall_variance < 25:  # Low variance indicates consistency
			if avg_overall >= 85:
				pattern_type = "consistently_excellent"
				description = "Vendor demonstrates consistently excellent performance"
				confidence = 0.92
				impact = "highly_positive"
			elif avg_overall >= 70:
				pattern_type = "consistently_good"
				description = "Vendor maintains consistently good performance levels"
				confidence = 0.88
				impact = "positive"
			else:
				pattern_type = "consistently_underperforming"
				description = "Vendor shows consistent underperformance"
				confidence = 0.85
				impact = "negative"
		else:
			pattern_type = "variable_performance"
			description = "Vendor shows variable performance with significant fluctuations"
			confidence = 0.82
			impact = "concerning"
		
		# Analyze trend
		if len(overall_scores) >= 5:
			recent_avg = sum(overall_scores[:3]) / 3
			older_avg = sum(overall_scores[-3:]) / 3
			
			if recent_avg > older_avg + 5:
				trend = "improving"
			elif recent_avg < older_avg - 5:
				trend = "declining"
			else:
				trend = "stable"
		else:
			trend = "stable"
		
		return {
			"pattern_type": pattern_type,
			"category": "performance_consistency",
			"confidence": confidence,
			"description": description,
			"trend": trend,
			"metrics": {
				"performance_variance": overall_variance,
				"quality_variance": quality_variance,
				"delivery_variance": delivery_variance,
				"average_performance": avg_overall,
				"performance_stability_index": max(0, 100 - overall_variance)
			},
			"impact": impact
		}
	
	async def _analyze_risk_behavior(
		self, 
		vendor_id: str, 
		behavior_data: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Analyze risk-related behavior patterns"""
		
		risk_events = behavior_data.get('risk', [])
		if len(risk_events) < 2:
			return None
		
		# Analyze risk frequency and severity
		high_risks = [r for r in risk_events if r.get('severity') == 'high']
		medium_risks = [r for r in risk_events if r.get('severity') == 'medium']
		
		risk_frequency = len(risk_events) / behavior_data['analysis_period'] * 30  # per month
		
		# Determine risk behavior pattern
		if len(high_risks) > 2:
			pattern_type = "high_risk_prone"
			description = "Vendor frequently encounters high-severity risks"
			confidence = 0.88
			impact = "negative"
		elif risk_frequency > 0.5:  # More than 0.5 risks per month
			pattern_type = "risk_prone"
			description = "Vendor shows higher than average risk occurrence"
			confidence = 0.82
			impact = "concerning"
		elif risk_frequency < 0.1:  # Less than 0.1 risks per month
			pattern_type = "low_risk"
			description = "Vendor demonstrates excellent risk management"
			confidence = 0.90
			impact = "positive"
		else:
			pattern_type = "moderate_risk"
			description = "Vendor shows moderate risk profile"
			confidence = 0.75
			impact = "neutral"
		
		# Analyze resolution patterns
		resolved_risks = [r for r in risk_events if r.get('status') == 'resolved']
		resolution_rate = len(resolved_risks) / len(risk_events) if risk_events else 0
		
		return {
			"pattern_type": pattern_type,
			"category": "risk_behavior",
			"confidence": confidence,
			"description": description,
			"trend": "stable",  # Would need more sophisticated analysis for trend
			"metrics": {
				"risk_frequency_per_month": risk_frequency,
				"high_risk_count": len(high_risks),
				"medium_risk_count": len(medium_risks),
				"resolution_rate": resolution_rate,
				"average_risk_score": sum(float(r.get('overall_risk_score', 0)) for r in risk_events) / len(risk_events)
			},
			"impact": impact
		}
	
	async def _analyze_innovation_behavior(
		self, 
		vendor_id: str, 
		behavior_data: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Analyze innovation-related behavior patterns"""
		
		performance_records = behavior_data.get('performance', [])
		if not performance_records:
			return None
		
		# Extract innovation scores
		innovation_scores = [
			float(p.get('innovation_score', 0)) 
			for p in performance_records 
			if p.get('innovation_score', 0) > 0
		]
		
		if len(innovation_scores) < 2:
			return None
		
		avg_innovation = sum(innovation_scores) / len(innovation_scores)
		
		# Determine innovation pattern
		if avg_innovation >= 80:
			pattern_type = "innovation_leader"
			description = "Vendor consistently demonstrates strong innovation capabilities"
			confidence = 0.85
			impact = "strategic_advantage"
		elif avg_innovation >= 60:
			pattern_type = "innovative"
			description = "Vendor shows good innovation potential"
			confidence = 0.78
			impact = "positive"
		elif avg_innovation >= 40:
			pattern_type = "moderately_innovative"
			description = "Vendor demonstrates moderate innovation efforts"
			confidence = 0.72
			impact = "neutral"
		else:
			pattern_type = "low_innovation"
			description = "Vendor shows limited innovation capabilities"
			confidence = 0.80
			impact = "concerning"
		
		# Analyze trend
		if len(innovation_scores) >= 3:
			recent_avg = sum(innovation_scores[:2]) / 2
			older_avg = sum(innovation_scores[-2:]) / 2
			
			if recent_avg > older_avg + 10:
				trend = "accelerating"
			elif recent_avg < older_avg - 10:
				trend = "declining"
			else:
				trend = "stable"
		else:
			trend = "stable"
		
		return {
			"pattern_type": pattern_type,
			"category": "innovation_behavior",
			"confidence": confidence,
			"description": description,
			"trend": trend,
			"metrics": {
				"average_innovation_score": avg_innovation,
				"innovation_consistency": max(0, 100 - np.var(innovation_scores)),
				"innovation_data_points": len(innovation_scores)
			},
			"impact": impact
		}
	
	async def _analyze_compliance_behavior(
		self, 
		vendor_id: str, 
		behavior_data: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Analyze compliance behavior patterns"""
		
		compliance_records = behavior_data.get('compliance', [])
		if len(compliance_records) < 2:
			return None
		
		# Calculate compliance metrics
		compliant_records = [c for c in compliance_records if c.get('status') == 'compliant']
		non_compliant_records = [c for c in compliance_records if c.get('status') == 'non_compliant']
		
		compliance_rate = len(compliant_records) / len(compliance_records)
		avg_compliance_score = sum(float(c.get('compliance_score', 100)) for c in compliance_records) / len(compliance_records)
		
		# Determine compliance pattern
		if compliance_rate >= 0.95 and avg_compliance_score >= 95:
			pattern_type = "exemplary_compliance"
			description = "Vendor demonstrates exemplary compliance standards"
			confidence = 0.92
			impact = "highly_positive"
		elif compliance_rate >= 0.85 and avg_compliance_score >= 85:
			pattern_type = "good_compliance"
			description = "Vendor maintains good compliance standards"
			confidence = 0.88
			impact = "positive"
		elif compliance_rate >= 0.70:
			pattern_type = "acceptable_compliance"
			description = "Vendor meets acceptable compliance levels"
			confidence = 0.82
			impact = "neutral"
		else:
			pattern_type = "compliance_issues"
			description = "Vendor shows concerning compliance patterns"
			confidence = 0.90
			impact = "negative"
		
		return {
			"pattern_type": pattern_type,
			"category": "compliance_behavior",
			"confidence": confidence,
			"description": description,
			"trend": "stable",
			"metrics": {
				"compliance_rate": compliance_rate,
				"average_compliance_score": avg_compliance_score,
				"violations_count": len(non_compliant_records),
				"total_compliance_checks": len(compliance_records)
			},
			"impact": impact
		}
	
	# ========================================================================
	# PREDICTIVE ANALYTICS
	# ========================================================================
	
	async def generate_predictive_insights(
		self, 
		vendor_id: str,
		prediction_horizon_months: int = 6
	) -> List[Dict[str, Any]]:
		"""Generate predictive insights using advanced ML models"""
		
		# Collect historical data for prediction
		historical_data = await self._collect_prediction_data(vendor_id)
		
		insights = []
		
		# Performance prediction
		perf_insight = await self._predict_performance_trajectory(
			vendor_id, historical_data, prediction_horizon_months
		)
		if perf_insight:
			insights.append(perf_insight)
		
		# Risk prediction
		risk_insight = await self._predict_risk_emergence(
			vendor_id, historical_data, prediction_horizon_months
		)
		if risk_insight:
			insights.append(risk_insight)
		
		# Contract renewal prediction
		contract_insight = await self._predict_contract_renewal_success(
			vendor_id, historical_data, prediction_horizon_months
		)
		if contract_insight:
			insights.append(contract_insight)
		
		# Relationship health prediction
		relationship_insight = await self._predict_relationship_trajectory(
			vendor_id, historical_data, prediction_horizon_months
		)
		if relationship_insight:
			insights.append(relationship_insight)
		
		return insights
	
	async def _collect_prediction_data(self, vendor_id: str) -> Dict[str, Any]:
		"""Collect comprehensive data for predictive modeling"""
		
		connection = await self.db_context.get_connection()
		try:
			# Get vendor base data
			vendor_query = "SELECT * FROM vm_vendor WHERE id = $1 AND tenant_id = $2"
			vendor_data = await connection.fetchrow(vendor_query, vendor_id, self.tenant_id)
			
			# Get performance history (last 18 months)
			cutoff_date = datetime.utcnow() - timedelta(days=550)
			perf_query = """
				SELECT * FROM vm_performance 
				WHERE vendor_id = $1 AND tenant_id = $2 AND start_date >= $3
				ORDER BY start_date DESC
			"""
			performance_history = await connection.fetch(
				perf_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			# Get risk history
			risk_query = """
				SELECT * FROM vm_risk 
				WHERE vendor_id = $1 AND tenant_id = $2 AND identified_date >= $3
				ORDER BY identified_date DESC
			"""
			risk_history = await connection.fetch(
				risk_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			# Get contract data
			contract_query = """
				SELECT * FROM vm_contract 
				WHERE vendor_id = $1 AND tenant_id = $2
				ORDER BY effective_date DESC
			"""
			contract_data = await connection.fetch(
				contract_query, vendor_id, self.tenant_id
			)
			
			# Get communication patterns
			comm_query = """
				SELECT * FROM vm_communication 
				WHERE vendor_id = $1 AND tenant_id = $2 AND communication_date >= $3
				ORDER BY communication_date DESC
			"""
			communication_data = await connection.fetch(
				comm_query, vendor_id, self.tenant_id, cutoff_date
			)
			
			return {
				'vendor': dict(vendor_data) if vendor_data else {},
				'performance_history': [dict(row) for row in performance_history],
				'risk_history': [dict(row) for row in risk_history],
				'contracts': [dict(row) for row in contract_data],
				'communications': [dict(row) for row in communication_data]
			}
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def _predict_performance_trajectory(
		self, 
		vendor_id: str, 
		historical_data: Dict[str, Any],
		horizon_months: int
	) -> Optional[Dict[str, Any]]:
		"""Predict vendor performance trajectory"""
		
		performance_history = historical_data.get('performance_history', [])
		if len(performance_history) < 3:
			return None
		
		# Extract performance trends
		scores = [float(p['overall_score']) for p in performance_history]
		dates = [p['start_date'] for p in performance_history]
		
		# Simple linear trend analysis (in production, would use sophisticated ML models)
		if len(scores) >= 6:
			recent_avg = sum(scores[:3]) / 3
			older_avg = sum(scores[-3:]) / 3
			trend_slope = (recent_avg - older_avg) / len(scores)
		else:
			trend_slope = 0
			recent_avg = sum(scores) / len(scores)
		
		# Predict future performance
		predicted_score = recent_avg + (trend_slope * horizon_months)
		predicted_score = max(0, min(100, predicted_score))  # Bound between 0-100
		
		# Determine confidence based on data consistency
		score_variance = np.var(scores)
		confidence = max(0.5, min(0.95, 1.0 - (score_variance / 1000)))
		
		# Generate insight
		if trend_slope > 2:
			prediction = f"Performance score likely to improve by {abs(trend_slope * horizon_months):.1f}% over {horizon_months} months"
			trajectory = "improving"
		elif trend_slope < -2:
			prediction = f"Performance score may decline by {abs(trend_slope * horizon_months):.1f}% over {horizon_months} months"
			trajectory = "declining"
		else:
			prediction = f"Performance score expected to remain stable around {recent_avg:.1f}% over {horizon_months} months"
			trajectory = "stable"
		
		return {
			"insight_type": "performance_trajectory",
			"category": "performance_prediction",
			"timeframe": f"{horizon_months}_months",
			"confidence": confidence,
			"prediction": prediction,
			"trajectory": trajectory,
			"predicted_score": predicted_score,
			"current_score": recent_avg,
			"factors": self._identify_performance_factors(historical_data),
			"recommendations": self._generate_performance_recommendations(trajectory, historical_data)
		}
	
	async def _predict_risk_emergence(
		self, 
		vendor_id: str, 
		historical_data: Dict[str, Any],
		horizon_months: int
	) -> Optional[Dict[str, Any]]:
		"""Predict potential risk emergence"""
		
		risk_history = historical_data.get('risk_history', [])
		vendor_data = historical_data.get('vendor', {})
		
		# Calculate risk frequency
		if risk_history:
			risk_frequency = len(risk_history) / 18  # risks per month over 18 months
			high_risk_frequency = len([r for r in risk_history if r.get('severity') == 'high']) / 18
		else:
			risk_frequency = 0
			high_risk_frequency = 0
		
		# Predict future risk probability
		base_risk_probability = min(0.8, risk_frequency * horizon_months)
		
		# Adjust based on current vendor metrics
		current_risk_score = float(vendor_data.get('risk_score', 25))
		if current_risk_score > 60:
			base_risk_probability *= 1.5
		elif current_risk_score < 30:
			base_risk_probability *= 0.7
		
		# Performance degradation increases risk probability
		performance_history = historical_data.get('performance_history', [])
		if len(performance_history) >= 2:
			recent_perf = float(performance_history[0]['overall_score'])
			older_perf = float(performance_history[-1]['overall_score'])
			if recent_perf < older_perf - 10:  # 10+ point decline
				base_risk_probability *= 1.3
		
		predicted_probability = min(0.9, base_risk_probability)
		confidence = 0.75 if len(risk_history) >= 3 else 0.60
		
		# Determine risk level
		if predicted_probability > 0.6:
			risk_level = "high"
			prediction = f"High probability ({predicted_probability:.1%}) of new risks emerging within {horizon_months} months"
		elif predicted_probability > 0.3:
			risk_level = "moderate"
			prediction = f"Moderate probability ({predicted_probability:.1%}) of new risks emerging within {horizon_months} months"
		else:
			risk_level = "low"
			prediction = f"Low probability ({predicted_probability:.1%}) of new risks emerging within {horizon_months} months"
		
		return {
			"insight_type": "risk_emergence",
			"category": "risk_prediction",
			"timeframe": f"{horizon_months}_months",
			"confidence": confidence,
			"prediction": prediction,
			"risk_level": risk_level,
			"probability": predicted_probability,
			"risk_factors": self._identify_risk_factors(historical_data),
			"mitigation_strategies": self._suggest_risk_mitigation(risk_level, historical_data)
		}
	
	async def _predict_contract_renewal_success(
		self, 
		vendor_id: str, 
		historical_data: Dict[str, Any],
		horizon_months: int
	) -> Optional[Dict[str, Any]]:
		"""Predict contract renewal success probability"""
		
		contracts = historical_data.get('contracts', [])
		vendor_data = historical_data.get('vendor', {})
		
		if not contracts:
			return None
		
		# Get current active contract
		active_contracts = [c for c in contracts if c.get('status') == 'active']
		if not active_contracts:
			return None
		
		current_contract = active_contracts[0]
		expiration_date = current_contract.get('expiration_date')
		
		if not expiration_date or (expiration_date - datetime.utcnow()).days > horizon_months * 30:
			return None  # Contract not expiring within horizon
		
		# Calculate renewal success probability based on multiple factors
		success_factors = []
		
		# Performance factor (40% weight)
		current_performance = float(vendor_data.get('performance_score', 85))
		if current_performance >= 85:
			perf_factor = 0.9
			success_factors.append("Excellent performance history")
		elif current_performance >= 70:
			perf_factor = 0.7
			success_factors.append("Good performance history")
		else:
			perf_factor = 0.4
			success_factors.append("Performance concerns")
		
		# Risk factor (30% weight)
		current_risk = float(vendor_data.get('risk_score', 25))
		if current_risk < 30:
			risk_factor = 0.9
			success_factors.append("Low risk profile")
		elif current_risk < 60:
			risk_factor = 0.7
		else:
			risk_factor = 0.3
			success_factors.append("High risk concerns")
		
		# Relationship factor (20% weight)
		relationship_score = float(vendor_data.get('relationship_score', 75))
		if relationship_score >= 80:
			rel_factor = 0.9
			success_factors.append("Strong business relationship")
		elif relationship_score >= 60:
			rel_factor = 0.7
		else:
			rel_factor = 0.4
			success_factors.append("Relationship challenges")
		
		# Strategic importance factor (10% weight)
		is_strategic = vendor_data.get('strategic_partner', False)
		strategic_factor = 0.9 if is_strategic else 0.6
		if is_strategic:
			success_factors.append("Strategic partnership status")
		
		# Calculate weighted probability
		renewal_probability = (
			perf_factor * 0.4 + 
			risk_factor * 0.3 + 
			rel_factor * 0.2 + 
			strategic_factor * 0.1
		)
		
		confidence = 0.85 if len(contracts) > 1 else 0.75
		
		# Determine renewal outlook
		if renewal_probability >= 0.8:
			outlook = "highly_likely"
			prediction = f"Contract renewal highly likely ({renewal_probability:.1%} probability)"
		elif renewal_probability >= 0.6:
			outlook = "likely"
			prediction = f"Contract renewal likely ({renewal_probability:.1%} probability)"
		elif renewal_probability >= 0.4:
			outlook = "uncertain"
			prediction = f"Contract renewal uncertain ({renewal_probability:.1%} probability)"
		else:
			outlook = "unlikely"
			prediction = f"Contract renewal unlikely ({renewal_probability:.1%} probability)"
		
		return {
			"insight_type": "contract_renewal",
			"category": "contract_prediction",
			"timeframe": f"{horizon_months}_months",
			"confidence": confidence,
			"prediction": prediction,
			"renewal_outlook": outlook,
			"probability": renewal_probability,
			"success_factors": success_factors,
			"contract_expiration": expiration_date,
			"recommendations": self._generate_renewal_recommendations(outlook, success_factors)
		}
	
	async def _predict_relationship_trajectory(
		self, 
		vendor_id: str, 
		historical_data: Dict[str, Any],
		horizon_months: int
	) -> Optional[Dict[str, Any]]:
		"""Predict vendor relationship trajectory"""
		
		communications = historical_data.get('communications', [])
		vendor_data = historical_data.get('vendor', {})
		
		if len(communications) < 5:
			return None
		
		# Analyze communication sentiment trends
		recent_comms = communications[:10]  # Last 10 communications
		older_comms = communications[-10:] if len(communications) > 10 else []
		
		recent_sentiment = [
			float(c.get('sentiment_score', 0)) 
			for c in recent_comms 
			if c.get('sentiment_score') is not None
		]
		
		older_sentiment = [
			float(c.get('sentiment_score', 0)) 
			for c in older_comms 
			if c.get('sentiment_score') is not None
		] if older_comms else []
		
		if not recent_sentiment:
			return None
		
		avg_recent_sentiment = sum(recent_sentiment) / len(recent_sentiment)
		avg_older_sentiment = sum(older_sentiment) / len(older_sentiment) if older_sentiment else avg_recent_sentiment
		
		sentiment_trend = avg_recent_sentiment - avg_older_sentiment
		current_relationship_score = float(vendor_data.get('relationship_score', 75))
		
		# Predict relationship trajectory
		predicted_change = sentiment_trend * 20  # Amplify sentiment impact
		predicted_score = current_relationship_score + (predicted_change * horizon_months / 6)
		predicted_score = max(0, min(100, predicted_score))
		
		confidence = min(0.9, len(recent_sentiment) / 10)
		
		# Determine trajectory
		if sentiment_trend > 0.1:
			trajectory = "strengthening"
			prediction = f"Vendor relationship likely to strengthen over {horizon_months} months"
		elif sentiment_trend < -0.1:
			trajectory = "weakening"
			prediction = f"Vendor relationship may weaken over {horizon_months} months"
		else:
			trajectory = "stable"
			prediction = f"Vendor relationship expected to remain stable over {horizon_months} months"
		
		return {
			"insight_type": "relationship_trajectory",
			"category": "relationship_prediction",
			"timeframe": f"{horizon_months}_months",
			"confidence": confidence,
			"prediction": prediction,
			"trajectory": trajectory,
			"current_score": current_relationship_score,
			"predicted_score": predicted_score,
			"sentiment_trend": sentiment_trend,
			"factors": self._identify_relationship_factors(communications, vendor_data),
			"recommendations": self._generate_relationship_recommendations(trajectory, communications)
		}
	
	# ========================================================================
	# OPTIMIZATION RECOMMENDATIONS
	# ========================================================================
	
	async def generate_optimization_plan(
		self, 
		vendor_id: str,
		optimization_objectives: List[str]
	) -> VendorOptimizationPlan:
		"""Generate comprehensive vendor optimization plan"""
		
		# Collect current state data
		current_baseline = await self._collect_optimization_baseline(vendor_id)
		
		# Generate specific optimization actions
		optimization_actions = []
		
		for objective in optimization_objectives:
			if objective == "performance_improvement":
				actions = await self._generate_performance_optimization_actions(vendor_id, current_baseline)
				optimization_actions.extend(actions)
			elif objective == "cost_reduction":
				actions = await self._generate_cost_optimization_actions(vendor_id, current_baseline)
				optimization_actions.extend(actions)
			elif objective == "risk_mitigation":
				actions = await self._generate_risk_optimization_actions(vendor_id, current_baseline)
				optimization_actions.extend(actions)
			elif objective == "relationship_enhancement":
				actions = await self._generate_relationship_optimization_actions(vendor_id, current_baseline)
				optimization_actions.extend(actions)
		
		# Predict optimization outcomes
		predicted_outcomes = await self._simulate_optimization_outcomes(
			vendor_id, current_baseline, optimization_actions
		)
		
		# Create implementation plan
		implementation_plan = await self._create_optimization_implementation_plan(
			optimization_actions
		)
		
		# Define success metrics
		success_metrics = await self._define_optimization_success_metrics(
			optimization_objectives, current_baseline
		)
		
		# Create monitoring schedule
		monitoring_schedule = await self._create_optimization_monitoring_schedule(
			optimization_actions
		)
		
		return VendorOptimizationPlan(
			vendor_id=vendor_id,
			optimization_objectives=optimization_objectives,
			current_baseline=current_baseline,
			recommended_actions=optimization_actions,
			predicted_outcomes=predicted_outcomes,
			implementation_plan=implementation_plan,
			success_metrics=success_metrics,
			monitoring_schedule=monitoring_schedule,
			expires_at=datetime.utcnow() + timedelta(days=90)
		)
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	def _identify_performance_factors(self, historical_data: Dict[str, Any]) -> List[str]:
		"""Identify key factors affecting performance"""
		factors = []
		
		performance_history = historical_data.get('performance_history', [])
		if performance_history:
			latest = performance_history[0]
			if float(latest.get('quality_score', 0)) > 85:
				factors.append("strong_quality_management")
			if float(latest.get('delivery_score', 0)) > 85:
				factors.append("reliable_delivery")
			if float(latest.get('innovation_score', 0)) > 70:
				factors.append("innovation_capability")
		
		return factors
	
	def _generate_performance_recommendations(
		self, 
		trajectory: str, 
		historical_data: Dict[str, Any]
	) -> List[str]:
		"""Generate performance improvement recommendations"""
		recommendations = []
		
		if trajectory == "declining":
			recommendations.extend([
				"Conduct performance review meeting",
				"Implement performance improvement plan",
				"Increase monitoring frequency"
			])
		elif trajectory == "stable":
			recommendations.extend([
				"Explore opportunities for performance enhancement",
				"Consider performance incentives"
			])
		else:  # improving
			recommendations.extend([
				"Recognize and reinforce positive trends",
				"Explore expansion opportunities"
			])
		
		return recommendations
	
	def _identify_risk_factors(self, historical_data: Dict[str, Any]) -> List[str]:
		"""Identify key risk factors"""
		factors = []
		
		risk_history = historical_data.get('risk_history', [])
		for risk in risk_history:
			if risk.get('risk_category') not in [f["factor"] for f in factors]:
				factors.append(risk.get('risk_category', 'unknown'))
		
		return factors[:5]  # Top 5 risk factors
	
	def _suggest_risk_mitigation(
		self, 
		risk_level: str, 
		historical_data: Dict[str, Any]
	) -> List[str]:
		"""Suggest risk mitigation strategies"""
		strategies = []
		
		if risk_level == "high":
			strategies.extend([
				"Implement enhanced monitoring protocols",
				"Develop contingency plans",
				"Consider alternative vendor sourcing"
			])
		elif risk_level == "moderate":
			strategies.extend([
				"Increase communication frequency",
				"Review contract terms",
				"Monitor key risk indicators"
			])
		else:
			strategies.extend([
				"Maintain current monitoring",
				"Continue regular reviews"
			])
		
		return strategies
	
	def _generate_renewal_recommendations(
		self, 
		outlook: str, 
		success_factors: List[str]
	) -> List[str]:
		"""Generate contract renewal recommendations"""
		recommendations = []
		
		if outlook == "unlikely":
			recommendations.extend([
				"Address performance concerns immediately",
				"Initiate relationship improvement program",
				"Consider contract restructuring",
				"Develop alternative vendor options"
			])
		elif outlook == "uncertain":
			recommendations.extend([
				"Schedule renewal discussions early",
				"Address identified concerns",
				"Strengthen business relationship"
			])
		else:
			recommendations.extend([
				"Prepare renewal documentation",
				"Negotiate favorable terms",
				"Explore expansion opportunities"
			])
		
		return recommendations
	
	def _identify_relationship_factors(
		self, 
		communications: List[Dict[str, Any]], 
		vendor_data: Dict[str, Any]
	) -> List[str]:
		"""Identify factors affecting vendor relationship"""
		factors = []
		
		if communications:
			avg_sentiment = sum(
				float(c.get('sentiment_score', 0)) 
				for c in communications[:10] 
				if c.get('sentiment_score') is not None
			) / min(10, len(communications))
			
			if avg_sentiment > 0.3:
				factors.append("positive_communication_tone")
			elif avg_sentiment < -0.3:
				factors.append("communication_challenges")
		
		if vendor_data.get('strategic_partner'):
			factors.append("strategic_partnership")
		
		if float(vendor_data.get('performance_score', 0)) > 85:
			factors.append("strong_performance")
		
		return factors
	
	def _generate_relationship_recommendations(
		self, 
		trajectory: str, 
		communications: List[Dict[str, Any]]
	) -> List[str]:
		"""Generate relationship improvement recommendations"""
		recommendations = []
		
		if trajectory == "weakening":
			recommendations.extend([
				"Schedule relationship review meeting",
				"Address communication issues",
				"Implement relationship improvement plan"
			])
		elif trajectory == "stable":
			recommendations.extend([
				"Maintain regular communication",
				"Explore collaboration opportunities"
			])
		else:  # strengthening
			recommendations.extend([
				"Leverage positive momentum",
				"Explore strategic partnership opportunities"
			])
		
		return recommendations
	
	async def _collect_optimization_baseline(self, vendor_id: str) -> Dict[str, Any]:
		"""Collect current baseline for optimization"""
		connection = await self.db_context.get_connection()
		try:
			# Get current vendor state
			vendor_query = "SELECT * FROM vm_vendor WHERE id = $1 AND tenant_id = $2"
			vendor_data = await connection.fetchrow(vendor_query, vendor_id, self.tenant_id)
			
			if not vendor_data:
				return {}
			
			return {
				"current_performance_score": float(vendor_data['performance_score']),
				"current_risk_score": float(vendor_data['risk_score']),
				"current_intelligence_score": float(vendor_data['intelligence_score']),
				"current_relationship_score": float(vendor_data['relationship_score']),
				"vendor_status": vendor_data['status'],
				"strategic_importance": vendor_data['strategic_importance'],
				"baseline_timestamp": datetime.utcnow()
			}
			
		finally:
			await self.db_context.release_connection(connection)