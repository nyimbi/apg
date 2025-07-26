"""
Financial Reporting Service

Business logic for Financial Reporting operations including statement generation,
consolidation, notes management, and analytical reporting functionality.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
import json
import pandas as pd
from pathlib import Path

from .models import (
	CFRFReportTemplate, CFRFReportDefinition, CFRFReportLine, CFRFReportPeriod,
	CFRFReportGeneration, CFRFFinancialStatement, CFRFConsolidation, CFRFNotes,
	CFRFDisclosure, CFRFAnalyticalReport, CFRFReportDistribution
)
from ..general_ledger.models import CFGLAccount, CFGLJournalEntry, CFGLPosting
from ...auth_rbac.models import db


class FinancialReportingService:
	"""Service class for Financial Reporting operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Report Template Management
	
	def create_report_template(self, template_data: Dict[str, Any]) -> CFRFReportTemplate:
		"""Create a new financial report template"""
		assert 'template_code' in template_data, "Template code is required"
		assert 'template_name' in template_data, "Template name is required"
		assert 'statement_type' in template_data, "Statement type is required"
		
		template = CFRFReportTemplate(
			tenant_id=self.tenant_id,
			template_code=template_data['template_code'],
			template_name=template_data['template_name'],
			description=template_data.get('description'),
			statement_type=template_data['statement_type'],
			category=template_data.get('category', 'standard'),
			format_type=template_data.get('format_type', 'comparative'),
			is_system=template_data.get('is_system', False),
			is_active=template_data.get('is_active', True),
			version=template_data.get('version', '1.0'),
			currency_type=template_data.get('currency_type', 'single'),
			show_percentages=template_data.get('show_percentages', False),
			show_variances=template_data.get('show_variances', False),
			decimal_places=template_data.get('decimal_places', 2),
			configuration=template_data.get('configuration')
		)
		
		db.session.add(template)
		db.session.commit()
		
		self._log_template_creation(template)
		return template
	
	def get_report_template(self, template_id: str) -> Optional[CFRFReportTemplate]:
		"""Get a report template by ID"""
		return db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.template_id == template_id,
				CFRFReportTemplate.tenant_id == self.tenant_id
			)
		).first()
	
	def get_templates_by_type(self, statement_type: str) -> List[CFRFReportTemplate]:
		"""Get all templates for a specific statement type"""
		return db.session.query(CFRFReportTemplate).filter(
			and_(
				CFRFReportTemplate.tenant_id == self.tenant_id,
				CFRFReportTemplate.statement_type == statement_type,
				CFRFReportTemplate.is_active == True
			)
		).order_by(CFRFReportTemplate.template_name).all()
	
	# Report Definition Management
	
	def create_report_definition(self, definition_data: Dict[str, Any]) -> CFRFReportDefinition:
		"""Create a report definition for a template"""
		assert 'template_id' in definition_data, "Template ID is required"
		assert 'definition_name' in definition_data, "Definition name is required"
		
		definition = CFRFReportDefinition(
			tenant_id=self.tenant_id,
			template_id=definition_data['template_id'],
			definition_name=definition_data['definition_name'],
			description=definition_data.get('description'),
			version=definition_data.get('version', '1.0'),
			calculation_method=definition_data.get('calculation_method', 'standard'),
			balance_check=definition_data.get('balance_check', True),
			zero_suppression=definition_data.get('zero_suppression', False),
			period_type=definition_data.get('period_type', 'monthly'),
			periods_to_show=definition_data.get('periods_to_show', 1),
			comparative_periods=definition_data.get('comparative_periods', 0),
			consolidation_required=definition_data.get('consolidation_required', False),
			requires_approval=definition_data.get('requires_approval', True),
			calculation_rules=definition_data.get('calculation_rules'),
			formatting_rules=definition_data.get('formatting_rules')
		)
		
		db.session.add(definition)
		db.session.commit()
		
		return definition
	
	def add_report_lines(self, definition_id: str, lines_data: List[Dict[str, Any]]) -> List[CFRFReportLine]:
		"""Add report lines to a definition"""
		lines = []
		
		for line_data in lines_data:
			line = CFRFReportLine(
				tenant_id=self.tenant_id,
				definition_id=definition_id,
				line_code=line_data['line_code'],
				line_name=line_data['line_name'],
				description=line_data.get('description'),
				parent_line_id=line_data.get('parent_line_id'),
				level=line_data.get('level', 0),
				sort_order=line_data.get('sort_order', 0),
				section_name=line_data.get('section_name'),
				line_type=line_data.get('line_type', 'detail'),
				data_source=line_data.get('data_source', 'accounts'),
				calculation_method=line_data.get('calculation_method'),
				account_filter=line_data.get('account_filter'),
				account_type_filter=line_data.get('account_type_filter'),
				include_children=line_data.get('include_children', True),
				formula=line_data.get('formula'),
				sign_reversal=line_data.get('sign_reversal', False),
				absolute_value=line_data.get('absolute_value', False),
				indent_level=line_data.get('indent_level', 0),
				bold=line_data.get('bold', False),
				show_line=line_data.get('show_line', True),
				note_reference=line_data.get('note_reference'),
				line_attributes=line_data.get('line_attributes')
			)
			lines.append(line)
			db.session.add(line)
		
		db.session.commit()
		
		# Update line count in definition
		definition = self.get_report_definition(definition_id)
		if definition:
			definition.line_count = len(lines)
			db.session.commit()
		
		return lines
	
	def get_report_definition(self, definition_id: str) -> Optional[CFRFReportDefinition]:
		"""Get a report definition by ID"""
		return db.session.query(CFRFReportDefinition).filter(
			and_(
				CFRFReportDefinition.definition_id == definition_id,
				CFRFReportDefinition.tenant_id == self.tenant_id
			)
		).first()
	
	# Report Period Management
	
	def create_report_period(self, period_data: Dict[str, Any]) -> CFRFReportPeriod:
		"""Create a new reporting period"""
		assert 'period_code' in period_data, "Period code is required"
		assert 'period_name' in period_data, "Period name is required"
		assert 'start_date' in period_data, "Start date is required"
		assert 'end_date' in period_data, "End date is required"
		
		start_date = period_data['start_date']
		end_date = period_data['end_date']
		days_in_period = (end_date - start_date).days + 1
		
		period = CFRFReportPeriod(
			tenant_id=self.tenant_id,
			period_code=period_data['period_code'],
			period_name=period_data['period_name'],
			description=period_data.get('description'),
			period_type=period_data.get('period_type', 'month'),
			fiscal_year=period_data.get('fiscal_year', start_date.year),
			period_number=period_data.get('period_number'),
			start_date=start_date,
			end_date=end_date,
			days_in_period=days_in_period,
			is_current=period_data.get('is_current', False),
			is_closed=period_data.get('is_closed', False),
			parent_period_id=period_data.get('parent_period_id'),
			period_attributes=period_data.get('period_attributes')
		)
		
		db.session.add(period)
		db.session.commit()
		
		return period
	
	def get_current_period(self, period_type: str = 'month') -> Optional[CFRFReportPeriod]:
		"""Get the current reporting period"""
		return db.session.query(CFRFReportPeriod).filter(
			and_(
				CFRFReportPeriod.tenant_id == self.tenant_id,
				CFRFReportPeriod.period_type == period_type,
				CFRFReportPeriod.is_current == True
			)
		).first()
	
	def get_periods_in_range(self, start_date: date, end_date: date) -> List[CFRFReportPeriod]:
		"""Get all periods within a date range"""
		return db.session.query(CFRFReportPeriod).filter(
			and_(
				CFRFReportPeriod.tenant_id == self.tenant_id,
				CFRFReportPeriod.start_date >= start_date,
				CFRFReportPeriod.end_date <= end_date
			)
		).order_by(CFRFReportPeriod.start_date).all()
	
	# Financial Statement Generation
	
	def generate_financial_statement(self, generation_data: Dict[str, Any]) -> CFRFReportGeneration:
		"""Generate a financial statement from a template"""
		assert 'template_id' in generation_data, "Template ID is required"
		assert 'period_id' in generation_data, "Period ID is required"
		assert 'as_of_date' in generation_data, "As of date is required"
		
		# Create generation record
		generation = CFRFReportGeneration(
			tenant_id=self.tenant_id,
			template_id=generation_data['template_id'],
			period_id=generation_data['period_id'],
			generation_name=generation_data.get('generation_name', 'Financial Statement Generation'),
			description=generation_data.get('description'),
			generation_type=generation_data.get('generation_type', 'standard'),
			as_of_date=generation_data['as_of_date'],
			include_adjustments=generation_data.get('include_adjustments', True),
			consolidation_level=generation_data.get('consolidation_level'),
			currency_code=generation_data.get('currency_code', 'USD'),
			output_format=generation_data.get('output_format', 'pdf'),
			parameters=generation_data.get('parameters')
		)
		
		db.session.add(generation)
		db.session.flush()  # Get the ID
		
		try:
			# Update status to running
			generation.status = 'running'
			generation.start_time = datetime.now()
			generation.progress_percentage = 0
			db.session.commit()
			
			# Get template and definition
			template = self.get_report_template(generation_data['template_id'])
			if not template:
				raise ValueError("Template not found")
			
			definition = template.report_definitions[0] if template.report_definitions else None
			if not definition:
				raise ValueError("No report definition found for template")
			
			# Generate statement data
			statement_data = self._generate_statement_data(generation, template, definition)
			
			# Create financial statement record
			statement = CFRFFinancialStatement(
				tenant_id=self.tenant_id,
				generation_id=generation.generation_id,
				period_id=generation_data['period_id'],
				statement_name=f"{template.template_name} - {generation_data['as_of_date']}",
				statement_type=template.statement_type,
				as_of_date=generation_data['as_of_date'],
				currency_code=generation.currency_code,
				reporting_entity=generation_data.get('reporting_entity'),
				consolidation_level=generation.consolidation_level,
				statement_data=statement_data,
				calculation_details=self._get_calculation_details(statement_data)
			)
			
			# Extract key financial metrics
			self._extract_financial_metrics(statement, statement_data, template.statement_type)
			
			db.session.add(statement)
			db.session.flush()
			
			# Verify balance if balance sheet
			if template.statement_type == 'balance_sheet':
				balance_diff = self._verify_balance_sheet(statement_data)
				statement.balance_difference = balance_diff
				generation.balance_verified = abs(balance_diff or 0) < 0.01
			
			# Update generation status
			generation.status = 'completed'
			generation.end_time = datetime.now()
			generation.duration_seconds = int((generation.end_time - generation.start_time).total_seconds())
			generation.progress_percentage = 100
			
			db.session.commit()
			
			self._log_statement_generation(generation, statement)
			
		except Exception as e:
			generation.status = 'failed'
			generation.end_time = datetime.now()
			generation.error_count = 1
			generation.generation_log = {'error': str(e)}
			db.session.commit()
			raise
		
		return generation
	
	def _generate_statement_data(self, generation: CFRFReportGeneration, 
								template: CFRFReportTemplate, 
								definition: CFRFReportDefinition) -> Dict[str, Any]:
		"""Generate the actual statement data from GL accounts"""
		statement_data = {
			'lines': [],
			'metadata': {
				'generation_date': datetime.now().isoformat(),
				'template_code': template.template_code,
				'definition_version': definition.version,
				'currency': generation.currency_code
			}
		}
		
		# Get report lines ordered by sort order
		report_lines = db.session.query(CFRFReportLine).filter(
			CFRFReportLine.definition_id == definition.definition_id
		).order_by(CFRFReportLine.sort_order).all()
		
		for line in report_lines:
			line_data = self._calculate_line_value(line, generation)
			statement_data['lines'].append(line_data)
			
			# Update progress
			progress = min(95, int((len(statement_data['lines']) / len(report_lines)) * 90))
			generation.progress_percentage = progress
			db.session.commit()
		
		return statement_data
	
	def _calculate_line_value(self, line: CFRFReportLine, generation: CFRFReportGeneration) -> Dict[str, Any]:
		"""Calculate the value for a specific report line"""
		line_data = {
			'line_code': line.line_code,
			'line_name': line.line_name,
			'line_type': line.line_type,
			'level': line.level,
			'indent_level': line.indent_level,
			'formatting': {
				'bold': line.bold,
				'italic': line.italic,
				'underline': line.underline
			},
			'show_line': line.show_line,
			'current_value': Decimal('0.00'),
			'prior_value': None,
			'variance': None,
			'percentage': None
		}
		
		if line.data_source == 'accounts' and line.account_filter:
			# Calculate from GL accounts
			value = self._calculate_account_balance(
				line.account_filter,
				generation.as_of_date,
				line.account_type_filter,
				line.include_children
			)
			
			if line.sign_reversal:
				value = -value
			if line.absolute_value:
				value = abs(value)
				
			line_data['current_value'] = value
			
		elif line.data_source == 'calculation' and line.formula:
			# Calculate from formula
			line_data['current_value'] = self._evaluate_formula(line.formula, generation)
			
		elif line.data_source == 'manual':
			# Manual entry - would come from user input or configuration
			line_data['current_value'] = Decimal('0.00')
		
		return line_data
	
	def _calculate_account_balance(self, account_filter: str, as_of_date: date, 
								  account_type_filter: Optional[str] = None,
								  include_children: bool = True) -> Decimal:
		"""Calculate balance from GL accounts based on filter criteria"""
		# Build account query
		query = db.session.query(func.sum(CFGLPosting.amount)).join(CFGLAccount)
		
		# Apply tenant filter
		query = query.filter(CFGLAccount.tenant_id == self.tenant_id)
		
		# Apply date filter
		query = query.filter(CFGLPosting.posting_date <= as_of_date)
		
		# Apply account filter (supports wildcards)
		if '*' in account_filter:
			pattern = account_filter.replace('*', '%')
			query = query.filter(CFGLAccount.account_code.like(pattern))
		else:
			# Exact match or range
			if '-' in account_filter:
				start, end = account_filter.split('-')
				query = query.filter(
					and_(
						CFGLAccount.account_code >= start.strip(),
						CFGLAccount.account_code <= end.strip()
					)
				)
			else:
				query = query.filter(CFGLAccount.account_code == account_filter)
		
		# Apply account type filter if specified
		if account_type_filter:
			query = query.join(CFGLAccount.account_type).filter(
				CFGLAccount.account_type.has(type_code=account_type_filter)
			)
		
		result = query.scalar()
		return result or Decimal('0.00')
	
	def _evaluate_formula(self, formula: str, generation: CFRFReportGeneration) -> Decimal:
		"""Evaluate a calculation formula"""
		# This is a simplified formula evaluator
		# In production, you'd want a more robust formula parser
		
		# Replace line references with actual values
		# Formula might be like: "LINE001 + LINE002 - LINE003"
		
		# For now, return zero - this would need proper implementation
		return Decimal('0.00')
	
	def _extract_financial_metrics(self, statement: CFRFFinancialStatement, 
								   statement_data: Dict[str, Any], 
								   statement_type: str):
		"""Extract key financial metrics from statement data"""
		lines = statement_data.get('lines', [])
		
		if statement_type == 'balance_sheet':
			# Extract balance sheet metrics
			for line in lines:
				line_code = line.get('line_code', '').upper()
				value = line.get('current_value', 0)
				
				if 'TOTAL_ASSETS' in line_code:
					statement.total_assets = value
				elif 'TOTAL_LIABILITIES' in line_code:
					statement.total_liabilities = value
				elif 'TOTAL_EQUITY' in line_code:
					statement.total_equity = value
					
		elif statement_type == 'income_statement':
			# Extract income statement metrics
			for line in lines:
				line_code = line.get('line_code', '').upper()
				value = line.get('current_value', 0)
				
				if 'TOTAL_REV' in line_code or 'TOTAL_SALES' in line_code:
					statement.total_revenue = value
				elif 'NET_INCOME' in line_code:
					statement.net_income = value
	
	def _verify_balance_sheet(self, statement_data: Dict[str, Any]) -> Decimal:
		"""Verify that balance sheet balances (Assets = Liabilities + Equity)"""
		lines = statement_data.get('lines', [])
		
		total_assets = Decimal('0.00')
		total_liabilities = Decimal('0.00')
		total_equity = Decimal('0.00')
		
		for line in lines:
			line_code = line.get('line_code', '').upper()
			value = line.get('current_value', 0)
			
			if 'TOTAL_ASSETS' in line_code:
				total_assets = Decimal(str(value))
			elif 'TOTAL_LIABILITIES' in line_code:
				total_liabilities = Decimal(str(value))
			elif 'TOTAL_EQUITY' in line_code:
				total_equity = Decimal(str(value))
		
		return total_assets - (total_liabilities + total_equity)
	
	def _get_calculation_details(self, statement_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get detailed calculation breakdown"""
		return {
			'calculation_method': 'standard',
			'total_lines': len(statement_data.get('lines', [])),
			'calculation_date': datetime.now().isoformat()
		}
	
	# Consolidation Management
	
	def create_consolidation(self, consolidation_data: Dict[str, Any]) -> CFRFConsolidation:
		"""Create a new consolidation rule"""
		assert 'consolidation_code' in consolidation_data, "Consolidation code is required"
		assert 'parent_entity' in consolidation_data, "Parent entity is required"
		assert 'subsidiary_entity' in consolidation_data, "Subsidiary entity is required"
		
		consolidation = CFRFConsolidation(
			tenant_id=self.tenant_id,
			consolidation_code=consolidation_data['consolidation_code'],
			consolidation_name=consolidation_data.get('consolidation_name', 
													 f"{consolidation_data['parent_entity']} - {consolidation_data['subsidiary_entity']}"),
			description=consolidation_data.get('description'),
			parent_entity=consolidation_data['parent_entity'],
			subsidiary_entity=consolidation_data['subsidiary_entity'],
			consolidation_method=consolidation_data.get('consolidation_method', 'full'),
			ownership_percentage=Decimal(str(consolidation_data.get('ownership_percentage', 100.0))),
			voting_percentage=consolidation_data.get('voting_percentage'),
			acquisition_date=consolidation_data.get('acquisition_date'),
			eliminate_intercompany=consolidation_data.get('eliminate_intercompany', True),
			currency_translation_method=consolidation_data.get('currency_translation_method'),
			functional_currency=consolidation_data.get('functional_currency'),
			reporting_currency=consolidation_data.get('reporting_currency', 'USD'),
			effective_from=consolidation_data.get('effective_from', date.today()),
			effective_to=consolidation_data.get('effective_to'),
			consolidation_rules=consolidation_data.get('consolidation_rules')
		)
		
		db.session.add(consolidation)
		db.session.commit()
		
		return consolidation
	
	def perform_consolidation(self, consolidation_id: str, period_id: str) -> Dict[str, Any]:
		"""Perform consolidation for a specific period"""
		consolidation = db.session.query(CFRFConsolidation).filter(
			and_(
				CFRFConsolidation.consolidation_id == consolidation_id,
				CFRFConsolidation.tenant_id == self.tenant_id
			)
		).first()
		
		if not consolidation:
			raise ValueError("Consolidation not found")
		
		# This is a simplified consolidation process
		# In production, this would involve complex consolidation logic
		
		consolidation_result = {
			'consolidation_id': consolidation_id,
			'period_id': period_id,
			'method': consolidation.consolidation_method,
			'ownership_percentage': float(consolidation.ownership_percentage),
			'entities_consolidated': [consolidation.parent_entity, consolidation.subsidiary_entity],
			'elimination_entries': [],
			'consolidated_balances': {},
			'consolidation_date': datetime.now().isoformat()
		}
		
		return consolidation_result
	
	# Notes and Disclosures Management
	
	def create_statement_note(self, note_data: Dict[str, Any]) -> CFRFNotes:
		"""Create a financial statement note"""
		assert 'statement_id' in note_data, "Statement ID is required"
		assert 'note_number' in note_data, "Note number is required"
		assert 'note_title' in note_data, "Note title is required"
		
		note = CFRFNotes(
			tenant_id=self.tenant_id,
			statement_id=note_data['statement_id'],
			note_number=note_data['note_number'],
			note_title=note_data['note_title'],
			note_category=note_data.get('note_category', 'disclosure'),
			note_text=note_data.get('note_text', ''),
			note_format=note_data.get('note_format', 'text'),
			is_required=note_data.get('is_required', False),
			is_standard=note_data.get('is_standard', False),
			sort_order=note_data.get('sort_order', 0),
			referenced_accounts=note_data.get('referenced_accounts'),
			referenced_lines=note_data.get('referenced_lines'),
			note_attributes=note_data.get('note_attributes')
		)
		
		db.session.add(note)
		db.session.commit()
		
		return note
	
	def create_disclosure(self, disclosure_data: Dict[str, Any]) -> CFRFDisclosure:
		"""Create a regulatory disclosure"""
		assert 'statement_id' in disclosure_data, "Statement ID is required"
		assert 'disclosure_code' in disclosure_data, "Disclosure code is required"
		assert 'disclosure_title' in disclosure_data, "Disclosure title is required"
		
		disclosure = CFRFDisclosure(
			tenant_id=self.tenant_id,
			statement_id=disclosure_data['statement_id'],
			disclosure_code=disclosure_data['disclosure_code'],
			disclosure_title=disclosure_data['disclosure_title'],
			disclosure_type=disclosure_data.get('disclosure_type', 'regulatory'),
			regulation_framework=disclosure_data.get('regulation_framework'),
			regulation_section=disclosure_data.get('regulation_section'),
			compliance_level=disclosure_data.get('compliance_level'),
			disclosure_text=disclosure_data.get('disclosure_text', ''),
			disclosure_format=disclosure_data.get('disclosure_format', 'text'),
			supporting_data=disclosure_data.get('supporting_data'),
			risk_category=disclosure_data.get('risk_category'),
			risk_level=disclosure_data.get('risk_level'),
			mitigation_measures=disclosure_data.get('mitigation_measures'),
			effective_from=disclosure_data.get('effective_from', date.today()),
			effective_to=disclosure_data.get('effective_to'),
			review_frequency=disclosure_data.get('review_frequency'),
			disclosure_attributes=disclosure_data.get('disclosure_attributes')
		)
		
		db.session.add(disclosure)
		db.session.commit()
		
		return disclosure
	
	# Analytical Reporting
	
	def create_analytical_report(self, report_data: Dict[str, Any]) -> CFRFAnalyticalReport:
		"""Create a custom analytical report"""
		assert 'report_code' in report_data, "Report code is required"
		assert 'report_name' in report_data, "Report name is required"
		assert 'report_type' in report_data, "Report type is required"
		
		report = CFRFAnalyticalReport(
			tenant_id=self.tenant_id,
			consolidation_id=report_data.get('consolidation_id'),
			report_code=report_data['report_code'],
			report_name=report_data['report_name'],
			description=report_data.get('description'),
			report_type=report_data['report_type'],
			report_category=report_data.get('report_category', 'management'),
			analysis_type=report_data.get('analysis_type', 'period_comparison'),
			analysis_periods=report_data.get('analysis_periods', 12),
			comparison_basis=report_data.get('comparison_basis'),
			account_selection=report_data.get('account_selection'),
			entity_selection=report_data.get('entity_selection'),
			dimension_filters=report_data.get('dimension_filters'),
			chart_types=report_data.get('chart_types'),
			key_metrics=report_data.get('key_metrics'),
			threshold_values=report_data.get('threshold_values'),
			is_scheduled=report_data.get('is_scheduled', False),
			schedule_frequency=report_data.get('schedule_frequency'),
			output_formats=report_data.get('output_formats'),
			default_format=report_data.get('default_format', 'pdf'),
			is_public=report_data.get('is_public', False),
			restricted_access=report_data.get('restricted_access', False),
			access_groups=report_data.get('access_groups'),
			report_configuration=report_data.get('report_configuration'),
			calculation_logic=report_data.get('calculation_logic')
		)
		
		db.session.add(report)
		db.session.commit()
		
		return report
	
	def generate_analytical_report(self, report_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate an analytical report"""
		report = db.session.query(CFRFAnalyticalReport).filter(
			and_(
				CFRFAnalyticalReport.report_id == report_id,
				CFRFAnalyticalReport.tenant_id == self.tenant_id
			)
		).first()
		
		if not report:
			raise ValueError("Analytical report not found")
		
		# Generate report based on type and configuration
		report_data = {
			'report_id': report_id,
			'report_name': report.report_name,
			'report_type': report.report_type,
			'analysis_type': report.analysis_type,
			'generation_date': datetime.now().isoformat(),
			'parameters': parameters,
			'data': self._generate_analytical_data(report, parameters),
			'charts': self._generate_charts(report, parameters),
			'key_metrics': self._calculate_key_metrics(report, parameters)
		}
		
		# Update last generated timestamp
		report.last_generated = datetime.now()
		db.session.commit()
		
		return report_data
	
	def _generate_analytical_data(self, report: CFRFAnalyticalReport, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate the analytical data for a report"""
		# This would contain the actual analytical calculations
		# based on the report configuration and parameters
		
		return {
			'summary': {},
			'details': [],
			'trends': [],
			'variances': []
		}
	
	def _generate_charts(self, report: CFRFAnalyticalReport, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate chart data for analytical report"""
		charts = []
		
		if report.chart_types:
			for chart_type in report.chart_types:
				chart_data = {
					'type': chart_type,
					'title': f"{report.report_name} - {chart_type}",
					'data': [],
					'options': {}
				}
				charts.append(chart_data)
		
		return charts
	
	def _calculate_key_metrics(self, report: CFRFAnalyticalReport, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Calculate key performance indicators"""
		metrics = {}
		
		if report.key_metrics:
			for metric in report.key_metrics:
				metrics[metric] = {
					'value': 0,
					'trend': 'stable',
					'variance': 0,
					'threshold_status': 'normal'
				}
		
		return metrics
	
	# Report Distribution
	
	def create_distribution_list(self, distribution_data: Dict[str, Any]) -> CFRFReportDistribution:
		"""Create a report distribution list"""
		assert 'distribution_name' in distribution_data, "Distribution name is required"
		assert 'distribution_type' in distribution_data, "Distribution type is required"
		
		distribution = CFRFReportDistribution(
			tenant_id=self.tenant_id,
			analytical_report_id=distribution_data.get('analytical_report_id'),
			distribution_name=distribution_data['distribution_name'],
			description=distribution_data.get('description'),
			distribution_type=distribution_data['distribution_type'],
			email_recipients=distribution_data.get('email_recipients'),
			distribution_groups=distribution_data.get('distribution_groups'),
			external_recipients=distribution_data.get('external_recipients'),
			delivery_method=distribution_data.get('delivery_method', 'email'),
			delivery_format=distribution_data.get('delivery_format', 'pdf'),
			delivery_schedule=distribution_data.get('delivery_schedule'),
			email_subject_template=distribution_data.get('email_subject_template'),
			email_body_template=distribution_data.get('email_body_template'),
			include_attachments=distribution_data.get('include_attachments', True),
			file_path=distribution_data.get('file_path'),
			file_naming_pattern=distribution_data.get('file_naming_pattern'),
			encryption_required=distribution_data.get('encryption_required', False),
			password_protection=distribution_data.get('password_protection', False),
			requires_approval=distribution_data.get('requires_approval', False),
			configuration=distribution_data.get('configuration')
		)
		
		db.session.add(distribution)
		db.session.commit()
		
		return distribution
	
	def distribute_report(self, distribution_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Distribute a report using a distribution list"""
		distribution = db.session.query(CFRFReportDistribution).filter(
			and_(
				CFRFReportDistribution.distribution_id == distribution_id,
				CFRFReportDistribution.tenant_id == self.tenant_id
			)
		).first()
		
		if not distribution:
			raise ValueError("Distribution list not found")
		
		distribution_result = {
			'distribution_id': distribution_id,
			'distribution_name': distribution.distribution_name,
			'delivery_method': distribution.delivery_method,
			'delivery_format': distribution.delivery_format,
			'distribution_date': datetime.now().isoformat(),
			'recipients': [],
			'success_count': 0,
			'failure_count': 0,
			'status': 'completed'
		}
		
		# Update distribution statistics
		distribution.last_distribution = datetime.now()
		distribution.distribution_count += 1
		
		# In a real implementation, this would handle actual distribution
		# via email, file upload, API calls, etc.
		
		db.session.commit()
		
		return distribution_result
	
	# Utility and Logging Methods
	
	def _log_template_creation(self, template: CFRFReportTemplate):
		"""Log template creation"""
		print(f"Created report template: {template.template_name} ({template.template_code})")
	
	def _log_statement_generation(self, generation: CFRFReportGeneration, statement: CFRFFinancialStatement):
		"""Log statement generation"""
		print(f"Generated financial statement: {statement.statement_name} in {generation.duration_seconds}s")
	
	def get_financial_summary(self, period_id: str) -> Dict[str, Any]:
		"""Get financial summary for a period"""
		statements = db.session.query(CFRFFinancialStatement).filter(
			and_(
				CFRFFinancialStatement.tenant_id == self.tenant_id,
				CFRFFinancialStatement.period_id == period_id,
				CFRFFinancialStatement.is_final == True
			)
		).all()
		
		summary = {
			'period_id': period_id,
			'statement_count': len(statements),
			'total_assets': sum(s.total_assets or 0 for s in statements if s.total_assets),
			'total_liabilities': sum(s.total_liabilities or 0 for s in statements if s.total_liabilities),
			'total_equity': sum(s.total_equity or 0 for s in statements if s.total_equity),
			'total_revenue': sum(s.total_revenue or 0 for s in statements if s.total_revenue),
			'net_income': sum(s.net_income or 0 for s in statements if s.net_income),
			'statements': [
				{
					'statement_id': s.statement_id,
					'statement_name': s.statement_name,
					'statement_type': s.statement_type,
					'as_of_date': s.as_of_date.isoformat() if s.as_of_date else None
				}
				for s in statements
			]
		}
		
		return summary