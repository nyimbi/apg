"""
Financial Reporting Views

Flask-AppBuilder views for Financial Reporting functionality including
report templates, financial statements, consolidations, and analytical reports.
"""

from flask import request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, TextAreaField, SelectField, DecimalField, BooleanField, DateField
from wtforms.validators import DataRequired, Length, NumberRange
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import json

from .models import (
	CFRFReportTemplate, CFRFReportDefinition, CFRFReportLine, CFRFReportPeriod,
	CFRFReportGeneration, CFRFFinancialStatement, CFRFConsolidation, CFRFNotes,
	CFRFDisclosure, CFRFAnalyticalReport, CFRFReportDistribution
)
from .service import FinancialReportingService
from ...auth_rbac.models import db


class CFRFReportTemplateModelView(ModelView):
	"""Report Template management view"""
	datamodel = SQLAInterface(CFRFReportTemplate)
	
	list_title = "Financial Report Templates"
	show_title = "Report Template Details"
	add_title = "Create Report Template"
	edit_title = "Edit Report Template"
	
	list_columns = [
		'template_code', 'template_name', 'statement_type', 'category',
		'format_type', 'is_active', 'version', 'created_at'
	]
	
	show_columns = [
		'template_code', 'template_name', 'description', 'statement_type',
		'category', 'format_type', 'is_system', 'is_active', 'version',
		'currency_type', 'show_percentages', 'show_variances', 'decimal_places',
		'page_orientation', 'font_size', 'auto_generate', 'generation_frequency',
		'configuration', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'template_code', 'template_name', 'description', 'statement_type',
		'category', 'format_type', 'currency_type', 'show_percentages',
		'show_variances', 'decimal_places', 'page_orientation', 'font_size',
		'auto_generate', 'generation_frequency'
	]
	
	edit_columns = add_columns + ['is_active', 'version']
	
	search_columns = ['template_code', 'template_name', 'statement_type', 'category']
	
	label_columns = {
		'template_code': 'Template Code',
		'template_name': 'Template Name',
		'statement_type': 'Statement Type',
		'format_type': 'Format Type',
		'currency_type': 'Currency Type',
		'show_percentages': 'Show Percentages',
		'show_variances': 'Show Variances',
		'decimal_places': 'Decimal Places',
		'page_orientation': 'Page Orientation',
		'auto_generate': 'Auto Generate',
		'generation_frequency': 'Generation Frequency',
		'is_system': 'System Template',
		'is_active': 'Active'
	}
	
	validators_columns = {
		'template_code': [DataRequired(), Length(max=50)],
		'template_name': [DataRequired(), Length(max=200)],
		'statement_type': [DataRequired()],
		'decimal_places': [NumberRange(min=0, max=6)]
	}
	
	@action("clone_template", "Clone Template", confirmation="Clone this template?", icon="fa-copy")
	def clone_template(self, items):
		"""Clone selected templates"""
		if not items:
			flash("No templates selected", "warning")
			return redirect(request.referrer)
		
		service = FinancialReportingService(self.get_current_tenant())
		cloned_count = 0
		
		for template in items:
			try:
				# Create cloned template data
				template_data = {
					'template_code': f"{template.template_code}_COPY",
					'template_name': f"{template.template_name} (Copy)",
					'description': template.description,
					'statement_type': template.statement_type,
					'category': template.category,
					'format_type': template.format_type,
					'currency_type': template.currency_type,
					'show_percentages': template.show_percentages,
					'show_variances': template.show_variances,
					'decimal_places': template.decimal_places,
					'configuration': template.configuration
				}
				
				service.create_report_template(template_data)
				cloned_count += 1
				
			except Exception as e:
				flash(f"Error cloning template {template.template_name}: {str(e)}", "error")
		
		if cloned_count > 0:
			flash(f"Successfully cloned {cloned_count} template(s)", "success")
		
		return redirect(request.referrer)
	
	@action("activate_templates", "Activate Templates", confirmation="Activate selected templates?", icon="fa-check")
	def activate_templates(self, items):
		"""Activate selected templates"""
		if not items:
			flash("No templates selected", "warning")
			return redirect(request.referrer)
		
		activated_count = 0
		for template in items:
			if not template.is_active:
				template.is_active = True
				activated_count += 1
		
		if activated_count > 0:
			db.session.commit()
			flash(f"Successfully activated {activated_count} template(s)", "success")
		else:
			flash("No templates were activated", "info")
		
		return redirect(request.referrer)
	
	def get_current_tenant(self) -> str:
		"""Get current tenant ID - to be implemented based on auth system"""
		return "default_tenant"


class CFRFFinancialStatementModelView(ModelView):
	"""Financial Statement management view"""
	datamodel = SQLAInterface(CFRFFinancialStatement)
	
	list_title = "Financial Statements"
	show_title = "Financial Statement Details"
	
	list_columns = [
		'statement_name', 'statement_type', 'as_of_date', 'currency_code',
		'is_final', 'is_published', 'version', 'created_at'
	]
	
	show_columns = [
		'statement_name', 'statement_type', 'description', 'as_of_date',
		'currency_code', 'reporting_entity', 'consolidation_level',
		'is_final', 'is_published', 'version', 'total_assets', 'total_liabilities',
		'total_equity', 'total_revenue', 'net_income', 'balance_difference',
		'variance_percentage', 'data_completeness', 'created_at', 'updated_at'
	]
	
	search_columns = ['statement_name', 'statement_type', 'reporting_entity']
	
	label_columns = {
		'statement_name': 'Statement Name',
		'statement_type': 'Statement Type',
		'as_of_date': 'As of Date',
		'currency_code': 'Currency',
		'reporting_entity': 'Reporting Entity',
		'consolidation_level': 'Consolidation Level',
		'is_final': 'Final',
		'is_published': 'Published',
		'total_assets': 'Total Assets',
		'total_liabilities': 'Total Liabilities',
		'total_equity': 'Total Equity',
		'total_revenue': 'Total Revenue',
		'net_income': 'Net Income',
		'balance_difference': 'Balance Difference',
		'variance_percentage': 'Variance %',
		'data_completeness': 'Data Completeness %'
	}
	
	@action("publish_statements", "Publish Statements", confirmation="Publish selected statements?", icon="fa-share")
	def publish_statements(self, items):
		"""Publish selected financial statements"""
		if not items:
			flash("No statements selected", "warning")
			return redirect(request.referrer)
		
		published_count = 0
		for statement in items:
			if statement.is_final and not statement.is_published:
				statement.is_published = True
				published_count += 1
		
		if published_count > 0:
			db.session.commit()
			flash(f"Successfully published {published_count} statement(s)", "success")
		else:
			flash("No statements were published (must be final and not already published)", "info")
		
		return redirect(request.referrer)
	
	@action("finalize_statements", "Finalize Statements", confirmation="Finalize selected statements?", icon="fa-lock")
	def finalize_statements(self, items):
		"""Finalize selected financial statements"""
		if not items:
			flash("No statements selected", "warning")
			return redirect(request.referrer)
		
		finalized_count = 0
		for statement in items:
			if not statement.is_final:
				statement.is_final = True
				finalized_count += 1
		
		if finalized_count > 0:
			db.session.commit()
			flash(f"Successfully finalized {finalized_count} statement(s)", "success")
		else:
			flash("No statements were finalized", "info")
		
		return redirect(request.referrer)


class CFRFConsolidationModelView(ModelView):
	"""Consolidation management view"""
	datamodel = SQLAInterface(CFRFConsolidation)
	
	list_title = "Consolidation Rules"
	show_title = "Consolidation Details"
	add_title = "Create Consolidation Rule"
	edit_title = "Edit Consolidation Rule"
	
	list_columns = [
		'consolidation_code', 'consolidation_name', 'parent_entity',
		'subsidiary_entity', 'consolidation_method', 'ownership_percentage',
		'effective_from', 'is_active'
	]
	
	show_columns = [
		'consolidation_code', 'consolidation_name', 'description',
		'parent_entity', 'subsidiary_entity', 'consolidation_method',
		'ownership_percentage', 'voting_percentage', 'acquisition_date',
		'disposal_date', 'eliminate_intercompany', 'currency_translation_method',
		'functional_currency', 'reporting_currency', 'effective_from',
		'effective_to', 'is_active', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'consolidation_code', 'consolidation_name', 'description',
		'parent_entity', 'subsidiary_entity', 'consolidation_method',
		'ownership_percentage', 'voting_percentage', 'acquisition_date',
		'eliminate_intercompany', 'currency_translation_method',
		'functional_currency', 'reporting_currency', 'effective_from', 'effective_to'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['consolidation_code', 'consolidation_name', 'parent_entity', 'subsidiary_entity']
	
	label_columns = {
		'consolidation_code': 'Consolidation Code',
		'consolidation_name': 'Consolidation Name',
		'parent_entity': 'Parent Entity',
		'subsidiary_entity': 'Subsidiary Entity',
		'consolidation_method': 'Consolidation Method',
		'ownership_percentage': 'Ownership %',
		'voting_percentage': 'Voting %',
		'acquisition_date': 'Acquisition Date',
		'disposal_date': 'Disposal Date',
		'eliminate_intercompany': 'Eliminate Intercompany',
		'currency_translation_method': 'Currency Translation Method',
		'functional_currency': 'Functional Currency',
		'reporting_currency': 'Reporting Currency',
		'effective_from': 'Effective From',
		'effective_to': 'Effective To',
		'is_active': 'Active'
	}
	
	validators_columns = {
		'consolidation_code': [DataRequired(), Length(max=50)],
		'consolidation_name': [DataRequired(), Length(max=200)],
		'parent_entity': [DataRequired(), Length(max=100)],
		'subsidiary_entity': [DataRequired(), Length(max=100)],
		'consolidation_method': [DataRequired()],
		'ownership_percentage': [DataRequired(), NumberRange(min=0, max=100)],
		'effective_from': [DataRequired()]
	}


class CFRFAnalyticalReportModelView(ModelView):
	"""Analytical Report management view"""
	datamodel = SQLAInterface(CFRFAnalyticalReport)
	
	list_title = "Analytical Reports"
	show_title = "Analytical Report Details"
	add_title = "Create Analytical Report"
	edit_title = "Edit Analytical Report"
	
	list_columns = [
		'report_code', 'report_name', 'report_type', 'report_category',
		'analysis_type', 'is_scheduled', 'last_generated', 'is_active'
	]
	
	show_columns = [
		'report_code', 'report_name', 'description', 'report_type',
		'report_category', 'analysis_type', 'analysis_periods',
		'comparison_basis', 'is_scheduled', 'schedule_frequency',
		'last_generated', 'next_generation', 'default_format',
		'is_public', 'restricted_access', 'is_active', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'report_code', 'report_name', 'description', 'report_type',
		'report_category', 'analysis_type', 'analysis_periods',
		'comparison_basis', 'is_scheduled', 'schedule_frequency',
		'default_format', 'is_public', 'restricted_access'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['report_code', 'report_name', 'report_type', 'report_category']
	
	label_columns = {
		'report_code': 'Report Code',
		'report_name': 'Report Name',
		'report_type': 'Report Type',
		'report_category': 'Report Category',
		'analysis_type': 'Analysis Type',
		'analysis_periods': 'Analysis Periods',
		'comparison_basis': 'Comparison Basis',
		'is_scheduled': 'Scheduled',
		'schedule_frequency': 'Schedule Frequency',
		'last_generated': 'Last Generated',
		'next_generation': 'Next Generation',
		'default_format': 'Default Format',
		'is_public': 'Public',
		'restricted_access': 'Restricted Access',
		'is_active': 'Active'
	}
	
	validators_columns = {
		'report_code': [DataRequired(), Length(max=50)],
		'report_name': [DataRequired(), Length(max=200)],
		'report_type': [DataRequired()],
		'report_category': [DataRequired()],
		'analysis_type': [DataRequired()],
		'analysis_periods': [NumberRange(min=1, max=60)]
	}
	
	@action("generate_reports", "Generate Reports", confirmation="Generate selected reports?", icon="fa-play")
	def generate_reports(self, items):
		"""Generate selected analytical reports"""
		if not items:
			flash("No reports selected", "warning")
			return redirect(request.referrer)
		
		service = FinancialReportingService(self.get_current_tenant())
		generated_count = 0
		
		for report in items:
			try:
				service.generate_analytical_report(report.report_id, {})
				generated_count += 1
			except Exception as e:
				flash(f"Error generating report {report.report_name}: {str(e)}", "error")
		
		if generated_count > 0:
			flash(f"Successfully generated {generated_count} report(s)", "success")
		
		return redirect(request.referrer)
	
	def get_current_tenant(self) -> str:
		"""Get current tenant ID - to be implemented based on auth system"""
		return "default_tenant"


class CFRFReportGenerationView(BaseView):
	"""Report Generation management view"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Show report generation dashboard"""
		service = FinancialReportingService(self.get_current_tenant())
		
		# Get recent generations
		recent_generations = db.session.query(CFRFReportGeneration).filter(
			CFRFReportGeneration.tenant_id == self.get_current_tenant()
		).order_by(CFRFReportGeneration.created_at.desc()).limit(10).all()
		
		# Get templates for generation
		templates = db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.tenant_id == self.get_current_tenant(),
				CFRFReportTemplate.is_active == True
			)
		).order_by(CFRFReportTemplate.template_name).all()
		
		# Get available periods
		periods = db.session.query(CFRFReportPeriod).filter(
			CFRFReportPeriod.tenant_id == self.get_current_tenant()
		).order_by(CFRFReportPeriod.start_date.desc()).limit(12).all()
		
		return self.render_template(
			'financial_reporting/generation_dashboard.html',
			recent_generations=recent_generations,
			templates=templates,
			periods=periods
		)
	
	@expose('/generate', methods=['GET', 'POST'])
	@has_access
	def generate(self):
		"""Generate a financial statement"""
		if request.method == 'POST':
			try:
				service = FinancialReportingService(self.get_current_tenant())
				
				generation_data = {
					'template_id': request.form.get('template_id'),
					'period_id': request.form.get('period_id'),
					'as_of_date': datetime.strptime(request.form.get('as_of_date'), '%Y-%m-%d').date(),
					'generation_name': request.form.get('generation_name'),
					'description': request.form.get('description'),
					'include_adjustments': request.form.get('include_adjustments') == 'on',
					'consolidation_level': request.form.get('consolidation_level'),
					'currency_code': request.form.get('currency_code', 'USD'),
					'output_format': request.form.get('output_format', 'pdf')
				}
				
				generation = service.generate_financial_statement(generation_data)
				
				flash(f"Financial statement generation started: {generation.generation_name}", "success")
				return redirect(url_for('CFRFReportGenerationView.status', generation_id=generation.generation_id))
				
			except Exception as e:
				flash(f"Error starting generation: {str(e)}", "error")
		
		# GET request - show generation form
		templates = db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.tenant_id == self.get_current_tenant(),
				CFRFReportTemplate.is_active == True
			)
		).order_by(CFRFReportTemplate.template_name).all()
		
		periods = db.session.query(CFRFReportPeriod).filter(
			CFRFReportPeriod.tenant_id == self.get_current_tenant()
		).order_by(CFRFReportPeriod.start_date.desc()).limit(12).all()
		
		return self.render_template(
			'financial_reporting/generate_form.html',
			templates=templates,
			periods=periods
		)
	
	@expose('/status/<generation_id>')
	@has_access
	def status(self, generation_id):
		"""Show generation status"""
		generation = db.session.query(CFRFReportGeneration).filter(
			and_(
				CFRFReportGeneration.generation_id == generation_id,
				CFRFReportGeneration.tenant_id == self.get_current_tenant()
			)
		).first()
		
		if not generation:
			flash("Generation not found", "error")
			return redirect(url_for('CFRFReportGenerationView.index'))
		
		return self.render_template(
			'financial_reporting/generation_status.html',
			generation=generation
		)
	
	@expose('/api/status/<generation_id>')
	@has_access
	def api_status(self, generation_id):
		"""API endpoint for generation status"""
		generation = db.session.query(CFRFReportGeneration).filter(
			and_(
				CFRFReportGeneration.generation_id == generation_id,
				CFRFReportGeneration.tenant_id == self.get_current_tenant()
			)
		).first()
		
		if not generation:
			return jsonify({'error': 'Generation not found'}), 404
		
		return jsonify({
			'generation_id': generation.generation_id,
			'status': generation.status,
			'progress_percentage': generation.progress_percentage,
			'start_time': generation.start_time.isoformat() if generation.start_time else None,
			'end_time': generation.end_time.isoformat() if generation.end_time else None,
			'duration_seconds': generation.duration_seconds,
			'error_count': generation.error_count,
			'warning_count': generation.warning_count
		})
	
	def get_current_tenant(self) -> str:
		"""Get current tenant ID - to be implemented based on auth system"""
		return "default_tenant"


class CFRFFinancialDashboardView(BaseView):
	"""Financial Dashboard view"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Show financial dashboard"""
		service = FinancialReportingService(self.get_current_tenant())
		
		# Get current period summary
		current_period = service.get_current_period()
		summary = None
		if current_period:
			summary = service.get_financial_summary(current_period.period_id)
		
		# Get recent statements
		recent_statements = db.session.query(CFRFFinancialStatement).filter(
			CFRFFinancialStatement.tenant_id == self.get_current_tenant()
		).order_by(CFRFFinancialStatement.created_at.desc()).limit(5).all()
		
		# Get consolidations
		consolidations = db.session.query(CFRFConsolidation).filter(
			and_(
				CFRFConsolidation.tenant_id == self.get_current_tenant(),
				CFRFConsolidation.is_active == True
			)
		).count()
		
		# Get active analytical reports
		analytical_reports = db.session.query(CFRFAnalyticalReport).filter(
			and_(
				CFRFAnalyticalReport.tenant_id == self.get_current_tenant(),
				CFRFAnalyticalReport.is_active == True
			)
		).count()
		
		dashboard_data = {
			'current_period': current_period,
			'financial_summary': summary,
			'recent_statements': recent_statements,
			'consolidation_count': consolidations,
			'analytical_report_count': analytical_reports,
			'statement_counts': {
				'total': len(recent_statements),
				'published': sum(1 for s in recent_statements if s.is_published),
				'draft': sum(1 for s in recent_statements if not s.is_final)
			}
		}
		
		return self.render_template(
			'financial_reporting/dashboard.html',
			dashboard_data=dashboard_data
		)
	
	@expose('/api/metrics')
	@has_access
	def api_metrics(self):
		"""API endpoint for dashboard metrics"""
		service = FinancialReportingService(self.get_current_tenant())
		
		# Get key metrics
		current_period = service.get_current_period()
		if current_period:
			summary = service.get_financial_summary(current_period.period_id)
		else:
			summary = {
				'total_assets': 0,
				'total_liabilities': 0,
				'total_equity': 0,
				'total_revenue': 0,
				'net_income': 0
			}
		
		return jsonify({
			'financial_metrics': summary,
			'generation_stats': {
				'pending': db.session.query(CFRFReportGeneration).filter(
					and_(
						CFRFReportGeneration.tenant_id == self.get_current_tenant(),
						CFRFReportGeneration.status == 'pending'
					)
				).count(),
				'running': db.session.query(CFRFReportGeneration).filter(
					and_(
						CFRFReportGeneration.tenant_id == self.get_current_tenant(),
						CFRFReportGeneration.status == 'running'
					)
				).count(),
				'completed': db.session.query(CFRFReportGeneration).filter(
					and_(
						CFRFReportGeneration.tenant_id == self.get_current_tenant(),
						CFRFReportGeneration.status == 'completed'
					)
				).count()
			}
		})
	
	def get_current_tenant(self) -> str:
		"""Get current tenant ID - to be implemented based on auth system"""
		return "default_tenant"


# Additional view classes for other models

class CFRFReportPeriodModelView(ModelView):
	"""Report Period management view"""
	datamodel = SQLAInterface(CFRFReportPeriod)
	
	list_title = "Reporting Periods"
	show_title = "Period Details"
	add_title = "Create Reporting Period"
	edit_title = "Edit Reporting Period"
	
	list_columns = [
		'period_code', 'period_name', 'period_type', 'fiscal_year',
		'start_date', 'end_date', 'is_current', 'is_closed'
	]
	
	show_columns = [
		'period_code', 'period_name', 'description', 'period_type',
		'fiscal_year', 'period_number', 'start_date', 'end_date',
		'days_in_period', 'is_current', 'is_closed', 'is_adjusting',
		'created_at', 'updated_at'
	]
	
	add_columns = [
		'period_code', 'period_name', 'description', 'period_type',
		'fiscal_year', 'period_number', 'start_date', 'end_date',
		'is_current', 'is_adjusting'
	]
	
	edit_columns = add_columns + ['is_closed']
	
	search_columns = ['period_code', 'period_name', 'period_type']
	
	label_columns = {
		'period_code': 'Period Code',
		'period_name': 'Period Name',
		'period_type': 'Period Type',
		'fiscal_year': 'Fiscal Year',
		'period_number': 'Period Number',
		'start_date': 'Start Date',
		'end_date': 'End Date',
		'days_in_period': 'Days in Period',
		'is_current': 'Current Period',
		'is_closed': 'Closed',
		'is_adjusting': 'Adjusting Period'
	}


class CFRFNotesModelView(ModelView):
	"""Financial Statement Notes management view"""
	datamodel = SQLAInterface(CFRFNotes)
	
	list_title = "Financial Statement Notes"
	show_title = "Note Details"
	add_title = "Create Note"
	edit_title = "Edit Note"
	
	list_columns = [
		'note_number', 'note_title', 'note_category', 'is_required',
		'approval_status', 'sort_order'
	]
	
	show_columns = [
		'note_number', 'note_title', 'note_category', 'note_text',
		'note_format', 'is_required', 'is_standard', 'sort_order',
		'approval_status', 'approved_by', 'approved_date',
		'created_at', 'updated_at'
	]
	
	add_columns = [
		'note_number', 'note_title', 'note_category', 'note_text',
		'note_format', 'is_required', 'is_standard', 'sort_order'
	]
	
	edit_columns = add_columns + ['approval_status']
	
	search_columns = ['note_number', 'note_title', 'note_category']
	
	label_columns = {
		'note_number': 'Note Number',
		'note_title': 'Note Title',
		'note_category': 'Category',
		'note_text': 'Note Text',
		'note_format': 'Format',
		'is_required': 'Required',
		'is_standard': 'Standard Note',
		'sort_order': 'Sort Order',
		'approval_status': 'Approval Status',
		'approved_by': 'Approved By',
		'approved_date': 'Approved Date'
	}


class CFRFDisclosureModelView(ModelView):
	"""Regulatory Disclosure management view"""
	datamodel = SQLAInterface(CFRFDisclosure)
	
	list_title = "Regulatory Disclosures"
	show_title = "Disclosure Details"
	add_title = "Create Disclosure"
	edit_title = "Edit Disclosure"
	
	list_columns = [
		'disclosure_code', 'disclosure_title', 'disclosure_type',
		'regulation_framework', 'compliance_level', 'disclosure_status'
	]
	
	show_columns = [
		'disclosure_code', 'disclosure_title', 'disclosure_type',
		'regulation_framework', 'regulation_section', 'compliance_level',
		'disclosure_text', 'risk_category', 'risk_level',
		'effective_from', 'effective_to', 'disclosure_status',
		'review_frequency', 'next_review_date', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'disclosure_code', 'disclosure_title', 'disclosure_type',
		'regulation_framework', 'regulation_section', 'compliance_level',
		'disclosure_text', 'risk_category', 'risk_level',
		'effective_from', 'effective_to', 'review_frequency'
	]
	
	edit_columns = add_columns + ['disclosure_status', 'next_review_date']
	
	search_columns = ['disclosure_code', 'disclosure_title', 'disclosure_type', 'regulation_framework']
	
	label_columns = {
		'disclosure_code': 'Disclosure Code',
		'disclosure_title': 'Disclosure Title',
		'disclosure_type': 'Disclosure Type',
		'regulation_framework': 'Regulation Framework',
		'regulation_section': 'Regulation Section',
		'compliance_level': 'Compliance Level',
		'disclosure_text': 'Disclosure Text',
		'risk_category': 'Risk Category',
		'risk_level': 'Risk Level',
		'effective_from': 'Effective From',
		'effective_to': 'Effective To',
		'disclosure_status': 'Status',
		'review_frequency': 'Review Frequency',
		'next_review_date': 'Next Review Date'
	}


class CFRFReportDistributionModelView(ModelView):
	"""Report Distribution management view"""
	datamodel = SQLAInterface(CFRFReportDistribution)
	
	list_title = "Report Distribution Lists"
	show_title = "Distribution Details"
	add_title = "Create Distribution List"
	edit_title = "Edit Distribution List"
	
	list_columns = [
		'distribution_name', 'distribution_type', 'delivery_method',
		'delivery_format', 'last_distribution', 'success_count',
		'failure_count', 'is_active'
	]
	
	show_columns = [
		'distribution_name', 'description', 'distribution_type',
		'delivery_method', 'delivery_format', 'delivery_schedule',
		'include_attachments', 'encryption_required', 'password_protection',
		'last_distribution', 'next_distribution', 'distribution_count',
		'success_count', 'failure_count', 'is_active', 'created_at', 'updated_at'
	]
	
	add_columns = [
		'distribution_name', 'description', 'distribution_type',
		'delivery_method', 'delivery_format', 'delivery_schedule',
		'include_attachments', 'encryption_required', 'password_protection',
		'requires_approval'
	]
	
	edit_columns = add_columns + ['is_active']
	
	search_columns = ['distribution_name', 'distribution_type', 'delivery_method']
	
	label_columns = {
		'distribution_name': 'Distribution Name',
		'distribution_type': 'Distribution Type',
		'delivery_method': 'Delivery Method',
		'delivery_format': 'Delivery Format',
		'delivery_schedule': 'Delivery Schedule',
		'include_attachments': 'Include Attachments',
		'encryption_required': 'Encryption Required',
		'password_protection': 'Password Protection',
		'last_distribution': 'Last Distribution',
		'next_distribution': 'Next Distribution',
		'distribution_count': 'Distribution Count',
		'success_count': 'Success Count',
		'failure_count': 'Failure Count',
		'requires_approval': 'Requires Approval',
		'is_active': 'Active'
	}