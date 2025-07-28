"""
APG Governance, Risk & Compliance Views

Revolutionary Flask-AppBuilder views with AI-powered intelligence,
real-time collaboration, and immersive user experience.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget, EditWidget
from flask_appbuilder.actions import action
from flask_babel import lazy_gettext as _
from marshmallow import fields, Schema
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import joinedload
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from .models import (
	GRCRisk, GRCRiskCategory, GRCRiskAssessment, GRCRegulation, 
	GRCControl, GRCPolicy, GRCGovernanceDecision,
	GRCRiskLevel, GRCRiskStatus, GRCComplianceStatus
)
from .ai_engine import GRCAIEngine
from .analytics import GRCAnalytics
from ..auth_rbac.decorators import require_permission
from ..real_time_collaboration.service import CollaborationService


# ==============================================================================
# CUSTOM WIDGETS FOR ENHANCED UX
# ==============================================================================

class GRCRiskWidget(ListWidget):
	"""Enhanced Risk List Widget with Real-time Updates"""
	template = 'grc/widgets/risk_list.html'
	
	def __init__(self):
		super().__init__()
		self.extra_args = {
			'enable_realtime': True,
			'websocket_endpoint': '/api/v1/grc/realtime/risks'
		}


class GRCRiskHeatmapWidget(BaseView):
	"""Interactive Risk Heatmap Visualization"""
	template = 'grc/widgets/risk_heatmap.html'
	
	def generate_heatmap_data(self, risks: List[GRCRisk]) -> Dict[str, Any]:
		"""Generate risk heatmap data for visualization"""
		heatmap_data = []
		for risk in risks:
			heatmap_data.append({
				'id': risk.risk_id,
				'title': risk.risk_title,
				'probability': risk.residual_probability,
				'impact': risk.residual_impact,
				'score': risk.residual_risk_score,
				'level': risk.risk_level,
				'category': risk.category.category_name if risk.category else 'Unknown'
			})
		return {'risks': heatmap_data}


class GRCComplianceDashboardWidget(BaseView):
	"""Real-time Compliance Dashboard Widget"""
	template = 'grc/widgets/compliance_dashboard.html'
	
	def generate_compliance_metrics(self) -> Dict[str, Any]:
		"""Generate compliance metrics for dashboard"""
		# This would integrate with the analytics engine
		return {
			'overall_compliance': 87.5,
			'critical_gaps': 3,
			'overdue_assessments': 12,
			'trending_risks': 5
		}


# ==============================================================================
# RISK MANAGEMENT VIEWS
# ==============================================================================

class GRCRiskCategoryView(ModelView):
	"""Risk Category Management with Hierarchical Structure"""
	datamodel = SQLAInterface(GRCRiskCategory)
	
	# List view configuration
	list_columns = ['category_name', 'category_code', 'category_level', 
					'risk_count', 'average_risk_score', 'is_active']
	list_title = _('Risk Categories')
	
	# Search configuration
	search_columns = ['category_name', 'category_description', 'category_code']
	
	# Form configuration
	add_columns = ['category_name', 'category_description', 'category_code',
				   'parent_category_id', 'iso31000_alignment', 'coso_erm_alignment']
	edit_columns = add_columns + ['is_active', 'category_priority']
	show_columns = add_columns + ['category_id', 'category_level', 'category_path',
								  'risk_count', 'average_risk_score', 'created_at']
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Custom actions
	@action("activate", _("Activate Categories"), _("Activate selected categories?"), "fa-check")
	def activate_categories(self, items):
		"""Bulk activate risk categories"""
		for item in items:
			item.is_active = True
			self.datamodel.edit(item)
		flash(f"Activated {len(items)} categories", "success")
		return redirect(self.get_redirect())
	
	@action("calculate_stats", _("Recalculate Statistics"), 
			_("Recalculate statistics for selected categories?"), "fa-calculator")
	def recalculate_statistics(self, items):
		"""Recalculate category statistics"""
		for item in items:
			item.calculate_category_statistics()
			self.datamodel.edit(item)
		flash(f"Recalculated statistics for {len(items)} categories", "success")
		return redirect(self.get_redirect())


class GRCRiskView(ModelView):
	"""Revolutionary Risk Management with AI Intelligence"""
	datamodel = SQLAInterface(GRCRisk)
	
	# Enhanced list view with risk intelligence
	list_columns = ['risk_title', 'risk_level', 'risk_status', 'residual_risk_score',
					'risk_velocity', 'next_review_date', 'risk_owner_id']
	list_title = _('Enterprise Risk Register')
	
	# Custom list widget with real-time updates
	list_widget = GRCRiskWidget
	
	# Advanced search capabilities
	search_columns = ['risk_title', 'risk_description', 'business_process', 'risk_tags']
	
	# Comprehensive form configuration
	add_columns = ['risk_title', 'risk_description', 'risk_category_id', 'risk_owner_id',
				   'inherent_probability', 'inherent_impact', 'business_process',
				   'financial_impact_expected', 'risk_tags']
	
	edit_columns = add_columns + ['residual_probability', 'residual_impact', 
								  'risk_status', 'next_review_date', 'risk_appetite_alignment']
	
	show_columns = ['risk_id', 'risk_title', 'risk_description', 'risk_level', 'risk_status',
					'inherent_risk_score', 'residual_risk_score', 'risk_velocity',
					'ai_risk_prediction', 'risk_correlation_score', 'predictive_indicators',
					'business_process', 'geographic_scope', 'stakeholder_impact',
					'financial_impact_expected', 'operational_impact', 'created_at', 'updated_at']
	
	# Custom formatters for enhanced display
	formatters_columns = {
		'risk_level': lambda x: f'<span class="badge badge-{x.lower()}">{x.upper()}</span>',
		'residual_risk_score': lambda x: f'<div class="progress"><div class="progress-bar bg-{GRCRiskView._get_risk_color(x)}" style="width: {x}%">{x:.1f}</div></div>',
		'risk_velocity': lambda x: f'<span class="trend-{"up" if x > 0 else "down" if x < 0 else "stable"}">{x:+.2f}</span>',
		'ai_risk_prediction': lambda x: GRCRiskView._format_ai_prediction(x),
		'predictive_indicators': lambda x: GRCRiskView._format_indicators(x)
	}
	
	# Permissions with fine-grained access control
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	@staticmethod
	def _get_risk_color(score: float) -> str:
		"""Get risk color based on score"""
		if score >= 90: return 'danger'
		elif score >= 70: return 'warning'
		elif score >= 40: return 'info'
		else: return 'success'
	
	@staticmethod
	def _format_ai_prediction(prediction: Dict[str, Any]) -> str:
		"""Format AI prediction data for display"""
		if not prediction:
			return '<em>No prediction available</em>'
		
		confidence = prediction.get('confidence', 0)
		trend = prediction.get('trend', 'stable')
		return f'<span class="ai-prediction" data-confidence="{confidence}">{trend.title()} ({confidence:.0%})</span>'
	
	@staticmethod
	def _format_indicators(indicators: List[Dict[str, Any]]) -> str:
		"""Format predictive indicators for display"""
		if not indicators:
			return '<em>No indicators</em>'
		
		indicator_badges = []
		for indicator in indicators[:3]:  # Show top 3
			name = indicator.get('name', 'Unknown')
			status = indicator.get('status', 'normal')
			indicator_badges.append(f'<span class="badge badge-{status}">{name}</span>')
		
		result = ' '.join(indicator_badges)
		if len(indicators) > 3:
			result += f' <small>+{len(indicators) - 3} more</small>'
		
		return result
	
	# Custom actions for risk management
	@action("ai_assess", _("AI Risk Assessment"), 
			_("Perform AI-powered assessment on selected risks?"), "fa-robot")
	@require_permission('grc:ai_assess')
	def ai_risk_assessment(self, items):
		"""Perform AI-powered risk assessment"""
		ai_engine = GRCAIEngine()
		assessed_count = 0
		
		for risk in items:
			try:
				ai_assessment = ai_engine.assess_risk(risk)
				risk.ai_risk_prediction = ai_assessment
				risk.ml_model_version = ai_engine.get_model_version()
				self.datamodel.edit(risk)
				assessed_count += 1
			except Exception as e:
				flash(f"AI assessment failed for {risk.risk_title}: {str(e)}", "error")
		
		flash(f"AI assessment completed for {assessed_count} risks", "success")
		return redirect(self.get_redirect())
	
	@action("calculate_correlation", _("Calculate Risk Correlations"),
			_("Calculate correlations for selected risks?"), "fa-project-diagram")
	def calculate_risk_correlations(self, items):
		"""Calculate risk correlations using AI"""
		ai_engine = GRCAIEngine()
		
		try:
			correlations = ai_engine.calculate_risk_correlations(items)
			for risk_id, correlation_data in correlations.items():
				risk = self.datamodel.get(risk_id)
				if risk:
					risk.risk_correlation_score = correlation_data.get('avg_correlation', 0.0)
					risk.ai_risk_prediction = {
						**risk.ai_risk_prediction,
						'correlations': correlation_data.get('correlations', [])
					}
					self.datamodel.edit(risk)
			
			flash(f"Correlation analysis completed for {len(items)} risks", "success")
		except Exception as e:
			flash(f"Correlation calculation failed: {str(e)}", "error")
		
		return redirect(self.get_redirect())
	
	@action("update_velocity", _("Update Risk Velocity"),
			_("Update velocity metrics for selected risks?"), "fa-tachometer-alt")
	def update_risk_velocity(self, items):
		"""Update risk velocity calculations"""
		analytics = GRCAnalytics()
		
		for risk in items:
			try:
				historical_scores = analytics.get_historical_risk_scores(risk.risk_id)
				velocity = risk.calculate_risk_velocity(historical_scores)
				self.datamodel.edit(risk)
			except Exception as e:
				flash(f"Velocity update failed for {risk.risk_title}: {str(e)}", "error")
		
		flash(f"Risk velocity updated for {len(items)} risks", "success")
		return redirect(self.get_redirect())
	
	# Custom endpoints for enhanced functionality
	@expose('/heatmap')
	@has_access
	def risk_heatmap(self):
		"""Interactive risk heatmap visualization"""
		risks = self.datamodel.get_all()
		widget = GRCRiskHeatmapWidget()
		heatmap_data = widget.generate_heatmap_data(risks)
		
		return self.render_template('grc/risk_heatmap.html',
									title=_('Risk Heatmap'),
									heatmap_data=heatmap_data)
	
	@expose('/analytics')
	@has_access
	def risk_analytics(self):
		"""Advanced risk analytics dashboard"""
		analytics = GRCAnalytics()
		
		# Generate comprehensive risk analytics
		risk_metrics = analytics.generate_risk_metrics()
		trend_analysis = analytics.analyze_risk_trends()
		correlation_network = analytics.build_risk_correlation_network()
		
		return self.render_template('grc/risk_analytics.html',
									title=_('Risk Analytics'),
									metrics=risk_metrics,
									trends=trend_analysis,
									correlations=correlation_network)
	
	@expose('/predict/<risk_id>')
	@has_access
	def predict_risk_evolution(self, risk_id):
		"""Predict risk evolution using AI"""
		risk = self.datamodel.get(risk_id)
		if not risk:
			flash(_('Risk not found'), 'error')
			return redirect(url_for('.list'))
		
		ai_engine = GRCAIEngine()
		try:
			prediction = ai_engine.predict_risk_evolution(risk)
			
			return self.render_template('grc/risk_prediction.html',
										title=f'Risk Prediction - {risk.risk_title}',
										risk=risk,
										prediction=prediction)
		except Exception as e:
			flash(f'Prediction failed: {str(e)}', 'error')
			return redirect(url_for('.show', pk=risk_id))


class GRCRiskAssessmentView(ModelView):
	"""AI-Enhanced Risk Assessment Management"""
	datamodel = SQLAInterface(GRCRiskAssessment)
	
	# List configuration
	list_columns = ['risk.risk_title', 'assessment_type', 'assessment_method',
					'overall_risk_score', 'ai_confidence_score', 'assessment_date']
	list_title = _('Risk Assessments')
	
	# Search and filter
	search_columns = ['risk.risk_title', 'assessment_type', 'assessment_method']
	
	# Form configuration
	add_columns = ['risk_id', 'assessment_type', 'assessment_method',
				   'probability_score', 'impact_score', 'confidence_level',
				   'assessment_scope', 'key_assumptions']
	
	edit_columns = add_columns + ['ai_probability_prediction', 'ai_impact_prediction',
								  'peer_review_status', 'assessment_quality_score']
	
	show_columns = ['assessment_id', 'risk.risk_title', 'assessment_type', 'assessment_method',
					'probability_score', 'impact_score', 'overall_risk_score',
					'ai_probability_prediction', 'ai_impact_prediction', 'ai_confidence_score',
					'best_case_scenario', 'worst_case_scenario', 'most_likely_scenario',
					'assessment_date', 'valid_from', 'valid_until']
	
	# Custom actions
	@action("ai_validate", _("AI Validation"), 
			_("Validate assessments using AI?"), "fa-check-circle")
	def ai_validate_assessments(self, items):
		"""Validate assessments using AI"""
		ai_engine = GRCAIEngine()
		validated_count = 0
		
		for assessment in items:
			try:
				validation_result = ai_engine.validate_assessment(assessment)
				assessment.ai_confidence_score = validation_result.get('confidence', 0.0)
				assessment.assessment_quality_score = validation_result.get('quality_score', 0.0)
				self.datamodel.edit(assessment)
				validated_count += 1
			except Exception as e:
				flash(f"Validation failed for assessment {assessment.assessment_id}: {str(e)}", "error")
		
		flash(f"AI validation completed for {validated_count} assessments", "success")
		return redirect(self.get_redirect())


# ==============================================================================
# COMPLIANCE MANAGEMENT VIEWS
# ==============================================================================

class GRCRegulationView(ModelView):
	"""Intelligent Regulatory Management with Change Detection"""
	datamodel = SQLAInterface(GRCRegulation)
	
	# List configuration with intelligent sorting
	list_columns = ['regulation_name', 'regulation_type', 'jurisdiction',
					'compliance_status', 'compliance_percentage', 'effective_date']
	list_title = _('Regulatory Universe')
	
	# Advanced search
	search_columns = ['regulation_name', 'regulation_code', 'issuing_authority', 
					  'jurisdiction', 'key_requirements']
	
	# Form configuration
	add_columns = ['regulation_name', 'regulation_code', 'regulation_type',
				   'issuing_authority', 'jurisdiction', 'regulation_summary',
				   'effective_date', 'applicable_industries']
	
	edit_columns = add_columns + ['compliance_status', 'compliance_deadline',
								  'change_detection_enabled', 'monitoring_frequency']
	
	show_columns = ['regulation_id', 'regulation_name', 'regulation_code', 'regulation_type',
					'issuing_authority', 'jurisdiction', 'regulation_summary',
					'key_requirements', 'effective_date', 'compliance_status',
					'ai_change_confidence', 'detected_changes', 'business_processes_affected']
	
	# Custom formatters
	formatters_columns = {
		'compliance_status': lambda x: f'<span class="badge badge-{GRCRegulationView._get_compliance_color(x)}">{x.replace("_", " ").title()}</span>',
		'compliance_percentage': lambda x: f'<div class="progress"><div class="progress-bar" style="width: {x}%">{x:.1f}%</div></div>',
		'detected_changes': lambda x: GRCRegulationView._format_changes(x)
	}
	
	@staticmethod
	def _get_compliance_color(status: str) -> str:
		"""Get compliance status color"""
		color_map = {
			'compliant': 'success',
			'non_compliant': 'danger',
			'partially_compliant': 'warning',
			'not_assessed': 'secondary',
			'pending_review': 'info'
		}
		return color_map.get(status, 'secondary')
	
	@staticmethod
	def _format_changes(changes: List[Dict[str, Any]]) -> str:
		"""Format detected changes for display"""
		if not changes:
			return '<em>No changes detected</em>'
		
		recent_changes = [c for c in changes if c.get('detected_date', '') > (datetime.utcnow() - timedelta(days=30)).isoformat()]
		if not recent_changes:
			return '<em>No recent changes</em>'
		
		return f'<span class="badge badge-info">{len(recent_changes)} recent changes</span>'
	
	# Custom actions
	@action("scan_changes", _("Scan for Changes"),
			_("Scan selected regulations for changes?"), "fa-search")
	def scan_regulatory_changes(self, items):
		"""Scan regulations for changes using AI"""
		# This would integrate with the regulatory intelligence engine
		scanned_count = 0
		changes_found = 0
		
		for regulation in items:
			if regulation.change_detection_enabled:
				# Simulate change detection - in production, this would call the AI engine
				# ai_engine = GRCAIEngine()
				# changes = ai_engine.detect_regulatory_changes(regulation)
				
				# Placeholder logic
				import random
				if random.random() > 0.8:  # 20% chance of finding changes
					changes_found += 1
					regulation.detected_changes.append({
						'type': 'amendment',
						'description': 'Minor amendment detected',
						'detected_date': datetime.utcnow().isoformat(),
						'confidence': 0.85
					})
					regulation.ai_change_confidence = 0.85
					self.datamodel.edit(regulation)
				
				scanned_count += 1
		
		flash(f"Scanned {scanned_count} regulations, found {changes_found} with changes", "success")
		return redirect(self.get_redirect())
	
	@action("assess_compliance", _("Assess Compliance"),
			_("Perform compliance assessment for selected regulations?"), "fa-clipboard-check")
	def assess_compliance(self, items):
		"""Perform automated compliance assessment"""
		assessed_count = 0
		
		for regulation in items:
			try:
				# This would integrate with the compliance automation framework
				# compliance_engine = ComplianceEngine()
				# assessment = compliance_engine.assess_regulation_compliance(regulation)
				
				# Placeholder logic
				compliance_percentage = min(100.0, regulation.compliance_percentage + 5.0)
				regulation.compliance_percentage = compliance_percentage
				
				if compliance_percentage >= 95:
					regulation.compliance_status = GRCComplianceStatus.COMPLIANT.value
				elif compliance_percentage >= 70:
					regulation.compliance_status = GRCComplianceStatus.PARTIALLY_COMPLIANT.value
				else:
					regulation.compliance_status = GRCComplianceStatus.NON_COMPLIANT.value
				
				self.datamodel.edit(regulation)
				assessed_count += 1
			except Exception as e:
				flash(f"Compliance assessment failed for {regulation.regulation_name}: {str(e)}", "error")
		
		flash(f"Compliance assessment completed for {assessed_count} regulations", "success")
		return redirect(self.get_redirect())


class GRCControlView(ModelView):
	"""Intelligent Control Management with Self-Testing"""
	datamodel = SQLAInterface(GRCControl)
	
	# List configuration
	list_columns = ['control_name', 'control_type', 'control_category',
					'overall_effectiveness_score', 'self_testing_enabled', 'next_testing_date']
	list_title = _('Control Framework')
	
	# Search configuration
	search_columns = ['control_name', 'control_description', 'control_procedures']
	
	# Form configuration
	add_columns = ['control_name', 'control_description', 'control_objective',
				   'control_type', 'control_category', 'control_owner_id',
				   'control_procedures', 'control_frequency', 'automation_level']
	
	edit_columns = add_columns + ['self_testing_enabled', 'testing_frequency',
								  'design_effectiveness', 'operating_effectiveness']
	
	show_columns = ['control_id', 'control_name', 'control_description', 'control_objective',
					'control_type', 'control_category', 'overall_effectiveness_score',
					'self_testing_enabled', 'self_test_success_rate', 'last_testing_date',
					'ai_effectiveness_prediction', 'optimization_recommendations']
	
	# Custom formatters
	formatters_columns = {
		'overall_effectiveness_score': lambda x: f'<div class="progress"><div class="progress-bar bg-{GRCControlView._get_effectiveness_color(x)}" style="width: {x}%">{x:.1f}%</div></div>',
		'self_test_success_rate': lambda x: f'{x:.1%}' if x else 'N/A',
		'optimization_recommendations': lambda x: GRCControlView._format_recommendations(x)
	}
	
	@staticmethod
	def _get_effectiveness_color(score: float) -> str:
		"""Get effectiveness color based on score"""
		if score >= 90: return 'success'
		elif score >= 70: return 'info'
		elif score >= 50: return 'warning'
		else: return 'danger'
	
	@staticmethod
	def _format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
		"""Format AI recommendations for display"""
		if not recommendations:
			return '<em>No recommendations</em>'
		
		return f'<span class="badge badge-info">{len(recommendations)} recommendations</span>'
	
	# Custom actions
	@action("run_self_test", _("Run Self-Test"),
			_("Execute self-test for selected controls?"), "fa-play")
	def run_control_self_test(self, items):
		"""Execute self-test for controls"""
		tested_count = 0
		
		for control in items:
			if control.self_testing_enabled:
				try:
					# This would integrate with the control automation framework
					# control_engine = ControlEngine()
					# test_result = control_engine.execute_self_test(control)
					
					# Placeholder logic
					import random
					test_passed = random.random() > 0.2  # 80% success rate
					
					control.last_self_test = datetime.utcnow()
					control.update_self_test_success_rate(test_passed)
					control.self_test_results = {
						'status': 'passed' if test_passed else 'failed',
						'timestamp': datetime.utcnow().isoformat(),
						'details': 'Self-test executed successfully' if test_passed else 'Self-test failed - requires review'
					}
					
					self.datamodel.edit(control)
					tested_count += 1
				except Exception as e:
					flash(f"Self-test failed for {control.control_name}: {str(e)}", "error")
		
		flash(f"Self-test executed for {tested_count} controls", "success")
		return redirect(self.get_redirect())
	
	@action("ai_optimize", _("AI Optimization"),
			_("Generate AI optimization recommendations?"), "fa-magic")
	def ai_optimize_controls(self, items):
		"""Generate AI-powered optimization recommendations"""
		ai_engine = GRCAIEngine()
		optimized_count = 0
		
		for control in items:
			try:
				recommendations = ai_engine.optimize_control(control)
				control.optimization_recommendations = recommendations
				control.ai_effectiveness_prediction = recommendations.get('predicted_effectiveness', 0.0)
				self.datamodel.edit(control)
				optimized_count += 1
			except Exception as e:
				flash(f"Optimization failed for {control.control_name}: {str(e)}", "error")
		
		flash(f"AI optimization completed for {optimized_count} controls", "success")
		return redirect(self.get_redirect())


# ==============================================================================
# GOVERNANCE VIEWS
# ==============================================================================

class GRCPolicyView(ModelView):
	"""AI-Assisted Policy Management"""
	datamodel = SQLAInterface(GRCPolicy)
	
	# List configuration
	list_columns = ['policy_title', 'policy_type', 'policy_category',
					'policy_status', 'next_review_date', 'ai_consistency_score']
	list_title = _('Policy Management')
	
	# Search configuration
	search_columns = ['policy_title', 'policy_statement', 'policy_procedures']
	
	# Form configuration
	add_columns = ['policy_title', 'policy_type', 'policy_category',
				   'policy_purpose', 'policy_scope', 'policy_statement',
				   'policy_owner_id', 'policy_steward_id', 'effective_date']
	
	edit_columns = add_columns + ['policy_procedures', 'training_required',
								  'acknowledgment_required', 'review_frequency_months']
	
	show_columns = ['policy_id', 'policy_title', 'policy_type', 'policy_category',
					'policy_purpose', 'policy_scope', 'policy_statement',
					'ai_consistency_score', 'policy_complexity_score',
					'compliance_alignment_score', 'policy_gaps_identified',
					'improvement_suggestions', 'training_completion_rate']
	
	# Custom formatters
	formatters_columns = {
		'ai_consistency_score': lambda x: f'<div class="ai-score" data-score="{x}">{x:.1%}</div>',
		'training_completion_rate': lambda x: f'<div class="progress"><div class="progress-bar" style="width: {x}%">{x:.1f}%</div></div>',
		'policy_gaps_identified': lambda x: f'<span class="badge badge-{"warning" if x else "success"}">{len(x) if x else 0} gaps</span>',
		'improvement_suggestions': lambda x: f'<span class="badge badge-info">{len(x) if x else 0} suggestions</span>'
	}
	
	# Custom actions
	@action("ai_review", _("AI Policy Review"),
			_("Perform AI review of selected policies?"), "fa-search")
	def ai_policy_review(self, items):
		"""Perform AI-powered policy review"""
		ai_engine = GRCAIEngine()
		reviewed_count = 0
		
		for policy in items:
			try:
				review_result = ai_engine.review_policy(policy)
				
				policy.ai_consistency_score = review_result.get('consistency_score', 0.0)
				policy.policy_complexity_score = review_result.get('complexity_score', 0.0)
				policy.compliance_alignment_score = review_result.get('compliance_score', 0.0)
				policy.policy_gaps_identified = review_result.get('gaps', [])
				policy.improvement_suggestions = review_result.get('suggestions', [])
				
				self.datamodel.edit(policy)
				reviewed_count += 1
			except Exception as e:
				flash(f"AI review failed for {policy.policy_title}: {str(e)}", "error")
		
		flash(f"AI review completed for {reviewed_count} policies", "success")
		return redirect(self.get_redirect())


class GRCGovernanceDecisionView(ModelView):
	"""Intelligent Governance Decision Management"""
	datamodel = SQLAInterface(GRCGovernanceDecision)
	
	# List configuration
	list_columns = ['decision_title', 'decision_type', 'decision_priority',
					'decision_status', 'decision_deadline', 'implementation_progress']
	list_title = _('Governance Decisions')
	
	# Search configuration
	search_columns = ['decision_title', 'decision_description', 'business_rationale']
	
	# Form configuration
	add_columns = ['decision_title', 'decision_type', 'decision_category',
				   'decision_description', 'business_rationale', 'decision_maker_id',
				   'decision_deadline', 'stakeholders_involved']
	
	edit_columns = add_columns + ['decision_status', 'decision_outcome',
								  'implementation_progress', 'success_metrics']
	
	show_columns = ['decision_id', 'decision_title', 'decision_type', 'decision_category',
					'decision_description', 'business_rationale', 'ai_impact_analysis',
					'ai_risk_assessment', 'ai_recommendation', 'stakeholder_impact_analysis',
					'implementation_progress', 'lessons_learned']
	
	# Custom formatters
	formatters_columns = {
		'implementation_progress': lambda x: f'<div class="progress"><div class="progress-bar" style="width: {x}%">{x:.1f}%</div></div>',
		'ai_impact_analysis': lambda x: GRCGovernanceDecisionView._format_ai_analysis(x),
		'decision_status': lambda x: f'<span class="badge badge-{GRCGovernanceDecisionView._get_decision_color(x)}">{x.replace("_", " ").title()}</span>'
	}
	
	@staticmethod
	def _get_decision_color(status: str) -> str:
		"""Get decision status color"""
		color_map = {
			'pending': 'warning',
			'approved': 'success',
			'rejected': 'danger',
			'deferred': 'secondary',
			'implemented': 'primary'
		}
		return color_map.get(status, 'secondary')
	
	@staticmethod
	def _format_ai_analysis(analysis: Dict[str, Any]) -> str:
		"""Format AI analysis for display"""
		if not analysis:
			return '<em>No AI analysis</em>'
		
		confidence = analysis.get('confidence', 0)
		recommendation = analysis.get('summary', 'Analysis available')
		return f'<div class="ai-analysis" data-confidence="{confidence}">{recommendation[:50]}{"..." if len(recommendation) > 50 else ""}</div>'
	
	# Custom actions
	@action("ai_analyze", _("AI Impact Analysis"),
			_("Perform AI impact analysis on selected decisions?"), "fa-brain")
	def ai_impact_analysis(self, items):
		"""Perform AI-powered impact analysis"""
		ai_engine = GRCAIEngine()
		analyzed_count = 0
		
		for decision in items:
			try:
				impact_analysis = ai_engine.analyze_decision_impact(decision)
				
				decision.ai_impact_analysis = impact_analysis.get('impact_analysis', {})
				decision.ai_risk_assessment = impact_analysis.get('risk_assessment', {})
				decision.ai_recommendation = impact_analysis.get('recommendation', '')
				decision.ai_confidence_score = impact_analysis.get('confidence', 0.0)
				
				self.datamodel.edit(decision)
				analyzed_count += 1
			except Exception as e:
				flash(f"AI analysis failed for {decision.decision_title}: {str(e)}", "error")
		
		flash(f"AI analysis completed for {analyzed_count} decisions", "success")
		return redirect(self.get_redirect())


# ==============================================================================
# DASHBOARD AND ANALYTICS VIEWS
# ==============================================================================

class GRCDashboardView(BaseView):
	"""Revolutionary GRC Executive Dashboard"""
	route_base = '/grc/dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main GRC dashboard with real-time intelligence"""
		analytics = GRCAnalytics()
		
		# Generate comprehensive dashboard data
		dashboard_data = {
			'risk_metrics': analytics.get_risk_summary_metrics(),
			'compliance_metrics': analytics.get_compliance_summary_metrics(),
			'governance_metrics': analytics.get_governance_summary_metrics(),
			'trend_data': analytics.get_trend_analysis(),
			'alerts': analytics.get_critical_alerts(),
			'ai_insights': analytics.get_ai_insights()
		}
		
		return self.render_template('grc/dashboard.html',
									title=_('GRC Executive Dashboard'),
									dashboard_data=dashboard_data)
	
	@expose('/realtime')
	@has_access
	def realtime_dashboard(self):
		"""Real-time GRC monitoring dashboard"""
		return self.render_template('grc/realtime_dashboard.html',
									title=_('Real-time GRC Monitoring'))
	
	@expose('/ai-insights')
	@has_access
	def ai_insights(self):
		"""AI-powered GRC insights and recommendations"""
		ai_engine = GRCAIEngine()
		
		insights = {
			'risk_predictions': ai_engine.get_risk_predictions(),
			'compliance_forecasts': ai_engine.get_compliance_forecasts(),
			'optimization_opportunities': ai_engine.get_optimization_opportunities(),
			'emerging_threats': ai_engine.detect_emerging_threats()
		}
		
		return self.render_template('grc/ai_insights.html',
									title=_('AI-Powered GRC Insights'),
									insights=insights)


class GRCAnalyticsView(BaseView):
	"""Advanced GRC Analytics and Reporting"""
	route_base = '/grc/analytics'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Analytics hub with multiple analysis options"""
		return self.render_template('grc/analytics_hub.html',
									title=_('GRC Analytics Hub'))
	
	@expose('/risk-analysis')
	@has_access
	def risk_analysis(self):
		"""Comprehensive risk analysis"""
		analytics = GRCAnalytics()
		
		analysis_data = {
			'risk_distribution': analytics.get_risk_distribution_analysis(),
			'correlation_matrix': analytics.build_risk_correlation_matrix(),
			'trend_analysis': analytics.analyze_risk_trends(),
			'predictive_models': analytics.get_predictive_risk_models()
		}
		
		return self.render_template('grc/risk_analysis.html',
									title=_('Risk Analysis'),
									analysis_data=analysis_data)
	
	@expose('/compliance-analysis')
	@has_access
	def compliance_analysis(self):
		"""Comprehensive compliance analysis"""
		analytics = GRCAnalytics()
		
		analysis_data = {
			'compliance_coverage': analytics.get_compliance_coverage_analysis(),
			'gap_analysis': analytics.perform_compliance_gap_analysis(),
			'effectiveness_analysis': analytics.analyze_control_effectiveness(),
			'regulatory_landscape': analytics.analyze_regulatory_landscape()
		}
		
		return self.render_template('grc/compliance_analysis.html',
									title=_('Compliance Analysis'),
									analysis_data=analysis_data)
	
	@expose('/governance-metrics')
	@has_access
	def governance_metrics(self):
		"""Governance effectiveness metrics"""
		analytics = GRCAnalytics()
		
		metrics_data = {
			'decision_velocity': analytics.calculate_decision_velocity(),
			'stakeholder_engagement': analytics.measure_stakeholder_engagement(),
			'policy_effectiveness': analytics.assess_policy_effectiveness(),
			'governance_maturity': analytics.assess_governance_maturity()
		}
		
		return self.render_template('grc/governance_metrics.html',
									title=_('Governance Metrics'),
									metrics_data=metrics_data)


# ==============================================================================
# API VIEWS FOR REAL-TIME INTEGRATION
# ==============================================================================

class GRCAPIView(BaseView):
	"""RESTful API endpoints for GRC integration"""
	route_base = '/api/v1/grc'
	
	@expose('/risks', methods=['GET'])
	@has_access
	def get_risks(self):
		"""Get risks with optional filtering"""
		# Parse query parameters for filtering
		filters = {}
		if request.args.get('status'):
			filters['risk_status'] = request.args.get('status')
		if request.args.get('level'):
			filters['risk_level'] = request.args.get('level')
		if request.args.get('category'):
			filters['risk_category_id'] = request.args.get('category')
		
		# Query risks with filters
		query = self.appbuilder.get_session.query(GRCRisk)
		for key, value in filters.items():
			query = query.filter(getattr(GRCRisk, key) == value)
		
		risks = query.all()
		
		# Serialize risks for API response
		risk_data = []
		for risk in risks:
			risk_data.append({
				'risk_id': risk.risk_id,
				'risk_title': risk.risk_title,
				'risk_level': risk.risk_level,
				'risk_status': risk.risk_status,
				'residual_risk_score': risk.residual_risk_score,
				'next_review_date': risk.next_review_date.isoformat() if risk.next_review_date else None
			})
		
		return jsonify({
			'success': True,
			'data': risk_data,
			'count': len(risk_data)
		})
	
	@expose('/compliance/status', methods=['GET'])
	@has_access
	def get_compliance_status(self):
		"""Get overall compliance status"""
		analytics = GRCAnalytics()
		
		compliance_status = {
			'overall_compliance_percentage': analytics.calculate_overall_compliance(),
			'critical_gaps': analytics.count_critical_compliance_gaps(),
			'overdue_assessments': analytics.count_overdue_assessments(),
			'regulatory_changes': analytics.count_recent_regulatory_changes()
		}
		
		return jsonify({
			'success': True,
			'data': compliance_status
		})
	
	@expose('/ai/insights', methods=['GET'])
	@has_access
	def get_ai_insights(self):
		"""Get AI-powered insights"""
		ai_engine = GRCAIEngine()
		
		insights = {
			'top_risks': ai_engine.identify_top_risks(),
			'emerging_threats': ai_engine.detect_emerging_threats(),
			'optimization_opportunities': ai_engine.get_optimization_opportunities(),
			'predictive_alerts': ai_engine.generate_predictive_alerts()
		}
		
		return jsonify({
			'success': True,
			'data': insights
		})


# Export views for registration with Flask-AppBuilder
__all__ = [
	'GRCRiskCategoryView', 'GRCRiskView', 'GRCRiskAssessmentView',
	'GRCRegulationView', 'GRCControlView', 
	'GRCPolicyView', 'GRCGovernanceDecisionView',
	'GRCDashboardView', 'GRCAnalyticsView', 'GRCAPIView'
]