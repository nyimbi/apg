"""
APG Threat Hunting Platform - Core Service

Enterprise threat hunting service with hypothesis-driven investigations,
collaborative hunting workflows, and advanced query orchestration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

import pandas as pd
import numpy as np
from elasticsearch import AsyncElasticsearch
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	ThreatHunt, HuntQuery, HuntFinding, HuntEvidence, HuntWorkflow,
	HuntMetrics, HuntTemplate, HuntType, HuntStatus, HuntPriority,
	QueryLanguage, HuntOutcome, EvidenceType
)


class ThreatHuntingService:
	"""Core threat hunting platform service"""
	
	def __init__(self, db_session: AsyncSession, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(__name__)
		
		self._active_hunts = {}
		self._hunt_queries = {}
		self._data_connectors = {}
		self._hunt_templates = {}
		self._evidence_store = {}
		
		asyncio.create_task(self._initialize_service())
	
	async def _initialize_service(self):
		"""Initialize threat hunting service"""
		try:
			await self._load_active_hunts()
			await self._initialize_data_connectors()
			await self._load_hunt_templates()
			await self._setup_query_engines()
			await self._initialize_evidence_store()
			
			self.logger.info(f"Threat hunting service initialized for tenant {self.tenant_id}")
		except Exception as e:
			self.logger.error(f"Failed to initialize threat hunting service: {str(e)}")
			raise
	
	async def create_threat_hunt(self, hunt_data: Dict[str, Any]) -> ThreatHunt:
		"""Create comprehensive threat hunting campaign"""
		try:
			hunt = ThreatHunt(
				tenant_id=self.tenant_id,
				**hunt_data
			)
			
			# Validate hunt parameters
			validation_result = await self._validate_hunt_parameters(hunt)
			if not validation_result['valid']:
				raise ValueError(f"Invalid hunt parameters: {validation_result['errors']}")
			
			# Set up hunt phases
			if not hunt.hunt_phases:
				hunt.hunt_phases = await self._generate_default_hunt_phases(hunt.hunt_type)
			
			# Initialize hunt scope
			if not hunt.hunt_scope:
				hunt.hunt_scope = await self._determine_hunt_scope(hunt)
			
			# Validate data source availability
			available_sources = await self._validate_data_sources(hunt.data_sources)
			if not available_sources:
				raise ValueError("No available data sources for hunt")
			
			hunt.data_sources = available_sources
			
			# Set up hunt workflow if not provided
			if hunt_data.get('create_workflow', True):
				workflow = await self._create_hunt_workflow(hunt)
				hunt.hunt_phases.append({
					'name': 'workflow_execution',
					'workflow_id': workflow.id,
					'automated': True
				})
			
			await self._store_threat_hunt(hunt)
			
			# Cache the hunt
			self._active_hunts[hunt.id] = hunt
			
			# Start hunt if requested
			if hunt_data.get('start_immediately', False):
				await self.start_hunt_campaign(hunt.id)
			
			return hunt
			
		except Exception as e:
			self.logger.error(f"Error creating threat hunt: {str(e)}")
			raise
	
	async def start_hunt_campaign(self, hunt_id: str) -> ThreatHunt:
		"""Start active threat hunting campaign"""
		try:
			hunt = await self._get_threat_hunt(hunt_id)
			if not hunt:
				raise ValueError(f"Hunt {hunt_id} not found")
			
			if hunt.status != HuntStatus.PLANNING:
				raise ValueError(f"Hunt {hunt_id} is not in planning status")
			
			# Check approval requirements
			if hunt.requires_approval and not hunt.approved_by:
				raise ValueError(f"Hunt {hunt_id} requires approval before starting")
			
			# Transition to active status
			hunt.status = HuntStatus.ACTIVE
			hunt.current_phase = hunt.hunt_phases[0]['name'] if hunt.hunt_phases else "initial_reconnaissance"
			
			await self._update_threat_hunt(hunt)
			
			# Start hunt execution asynchronously
			asyncio.create_task(self._execute_hunt_campaign(hunt))
			
			return hunt
			
		except Exception as e:
			self.logger.error(f"Error starting hunt campaign: {str(e)}")
			raise
	
	async def _execute_hunt_campaign(self, hunt: ThreatHunt):
		"""Execute hunt campaign with phase management"""
		try:
			self.logger.info(f"Starting hunt campaign execution: {hunt.id}")
			
			# Execute each hunt phase
			for phase in hunt.hunt_phases:
				hunt.current_phase = phase['name']
				await self._update_threat_hunt(hunt)
				
				phase_result = await self._execute_hunt_phase(hunt, phase)
				
				if not phase_result['success']:
					self.logger.error(f"Hunt phase {phase['name']} failed: {phase_result.get('error')}")
					if phase.get('critical', False):
						hunt.status = HuntStatus.SUSPENDED
						await self._update_threat_hunt(hunt)
						return
				
				# Update progress
				phase_index = hunt.hunt_phases.index(phase)
				progress = ((phase_index + 1) / len(hunt.hunt_phases)) * 100
				hunt.progress_percentage = Decimal(str(progress))
				await self._update_threat_hunt(hunt)
			
			# Mark hunt as investigating if findings found
			if hunt.findings_count > 0:
				hunt.status = HuntStatus.INVESTIGATING
			else:
				hunt.status = HuntStatus.CONCLUDED
			
			hunt.progress_percentage = Decimal('100.0')
			await self._update_threat_hunt(hunt)
			
			self.logger.info(f"Hunt campaign completed: {hunt.id}")
			
		except Exception as e:
			hunt.status = HuntStatus.SUSPENDED
			await self._update_threat_hunt(hunt)
			self.logger.error(f"Hunt campaign execution failed: {str(e)}")
	
	async def _execute_hunt_phase(self, hunt: ThreatHunt, phase: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute individual hunt phase"""
		try:
			phase_name = phase['name']
			self.logger.info(f"Executing hunt phase: {phase_name}")
			
			if phase_name == "initial_reconnaissance":
				return await self._execute_reconnaissance_phase(hunt, phase)
			elif phase_name == "hypothesis_validation":
				return await self._execute_hypothesis_validation_phase(hunt, phase)
			elif phase_name == "deep_investigation":
				return await self._execute_deep_investigation_phase(hunt, phase)
			elif phase_name == "evidence_collection":
				return await self._execute_evidence_collection_phase(hunt, phase)
			elif phase_name == "analysis_correlation":
				return await self._execute_analysis_correlation_phase(hunt, phase)
			elif phase_name == "workflow_execution":
				return await self._execute_workflow_phase(hunt, phase)
			else:
				return await self._execute_custom_phase(hunt, phase)
			
		except Exception as e:
			return {'success': False, 'error': str(e)}
	
	async def create_hunt_query(self, hunt_id: str, query_data: Dict[str, Any]) -> HuntQuery:
		"""Create and validate hunt query"""
		try:
			hunt = await self._get_threat_hunt(hunt_id)
			if not hunt:
				raise ValueError(f"Hunt {hunt_id} not found")
			
			query = HuntQuery(
				tenant_id=self.tenant_id,
				hunt_id=hunt_id,
				**query_data
			)
			
			# Validate query syntax
			validation_result = await self._validate_query_syntax(query)
			if not validation_result['valid']:
				raise ValueError(f"Invalid query syntax: {validation_result['errors']}")
			
			query.is_validated = True
			query.validation_results = validation_result
			
			# Generate query plan and optimization suggestions
			query.query_plan = await self._generate_query_plan(query)
			query.optimization_suggestions = await self._analyze_query_optimization(query)
			
			await self._store_hunt_query(query)
			
			# Cache the query
			self._hunt_queries[query.id] = query
			
			return query
			
		except Exception as e:
			self.logger.error(f"Error creating hunt query: {str(e)}")
			raise
	
	async def execute_hunt_query(self, query_id: str, execution_params: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute hunt query and return results"""
		try:
			query = await self._get_hunt_query(query_id)
			if not query:
				raise ValueError(f"Query {query_id} not found")
			
			if not query.is_validated:
				raise ValueError(f"Query {query_id} is not validated")
			
			start_time = datetime.utcnow()
			
			# Execute query based on language
			if query.query_language == QueryLanguage.KQL:
				results = await self._execute_kql_query(query, execution_params)
			elif query.query_language == QueryLanguage.SPL:
				results = await self._execute_spl_query(query, execution_params)
			elif query.query_language == QueryLanguage.SQL:
				results = await self._execute_sql_query(query, execution_params)
			elif query.query_language == QueryLanguage.ELASTIC:
				results = await self._execute_elastic_query(query, execution_params)
			else:
				raise ValueError(f"Unsupported query language: {query.query_language}")
			
			end_time = datetime.utcnow()
			execution_time = (end_time - start_time).total_seconds()
			
			# Update query execution statistics
			query.execution_count += 1
			query.last_executed = end_time
			
			if query.average_execution_time:
				query.average_execution_time = (query.average_execution_time + Decimal(str(execution_time))) / 2
			else:
				query.average_execution_time = Decimal(str(execution_time))
			
			# Update results metadata
			query.total_results = len(results.get('data', []))
			query.unique_results = len(set(str(r) for r in results.get('data', [])))
			query.result_hash = hashlib.md5(str(results).encode()).hexdigest()
			
			await self._update_hunt_query(query)
			
			# Analyze results for potential findings
			findings = await self._analyze_query_results_for_findings(query, results)
			
			return {
				'query_id': query_id,
				'execution_time': execution_time,
				'total_results': query.total_results,
				'unique_results': query.unique_results,
				'data': results.get('data', []),
				'metadata': results.get('metadata', {}),
				'potential_findings': len(findings),
				'findings': findings
			}
			
		except Exception as e:
			self.logger.error(f"Error executing hunt query: {str(e)}")
			raise
	
	async def create_hunt_finding(self, hunt_id: str, finding_data: Dict[str, Any]) -> HuntFinding:
		"""Create and track hunt finding"""
		try:
			hunt = await self._get_threat_hunt(hunt_id)
			if not hunt:
				raise ValueError(f"Hunt {hunt_id} not found")
			
			finding = HuntFinding(
				tenant_id=self.tenant_id,
				hunt_id=hunt_id,
				**finding_data
			)
			
			# Analyze finding for threat indicators
			threat_analysis = await self._analyze_finding_for_threats(finding)
			finding.related_threats = threat_analysis.get('threats', [])
			finding.attack_techniques = threat_analysis.get('techniques', [])
			finding.confidence_score = threat_analysis.get('confidence', Decimal('50.0'))
			
			# Auto-assign severity based on indicators
			finding.severity = await self._assess_finding_severity(finding)
			
			# Determine investigation priority
			finding.investigation_priority = await self._determine_investigation_priority(finding)
			
			await self._store_hunt_finding(finding)
			
			# Update hunt statistics
			hunt.findings_count += 1
			await self._update_threat_hunt(hunt)
			
			# Auto-assign investigator if configured
			if finding_data.get('auto_assign', True):
				investigator = await self._assign_investigator(finding)
				if investigator:
					finding.assigned_investigator = investigator
					await self._update_hunt_finding(finding)
			
			# Create investigation workflow if needed
			if finding.investigation_priority in [HuntPriority.CRITICAL, HuntPriority.HIGH]:
				workflow = await self._create_investigation_workflow(finding)
				finding.analysis_notes.append({
					'timestamp': datetime.utcnow().isoformat(),
					'type': 'workflow_created',
					'workflow_id': workflow.id,
					'note': f"Investigation workflow created for {finding.severity.value} priority finding"
				})
				await self._update_hunt_finding(finding)
			
			return finding
			
		except Exception as e:
			self.logger.error(f"Error creating hunt finding: {str(e)}")
			raise
	
	async def collect_hunt_evidence(self, finding_id: str, evidence_data: Dict[str, Any]) -> HuntEvidence:
		"""Collect and preserve digital evidence"""
		try:
			finding = await self._get_hunt_finding(finding_id)
			if not finding:
				raise ValueError(f"Finding {finding_id} not found")
			
			evidence = HuntEvidence(
				tenant_id=self.tenant_id,
				hunt_id=finding.hunt_id,
				finding_id=finding_id,
				**evidence_data
			)
			
			# Calculate file hashes if applicable
			if evidence.evidence_type in [EvidenceType.FILE_ARTIFACT, EvidenceType.MEMORY_DUMP, EvidenceType.DISK_IMAGE]:
				evidence.integrity_hashes = await self._calculate_evidence_hashes(evidence)
			
			# Initialize chain of custody
			evidence.custody_chain.append({
				'timestamp': datetime.utcnow().isoformat(),
				'action': 'collected',
				'person': evidence.collected_by,
				'location': evidence.source_location,
				'notes': f"Evidence collected from {evidence.source_system}"
			})
			
			# Store evidence securely
			storage_result = await self._store_evidence_securely(evidence)
			evidence.storage_location = storage_result['primary_location']
			evidence.backup_locations = storage_result.get('backup_locations', [])
			
			# Perform initial analysis
			if evidence_data.get('analyze_immediately', True):
				analysis_result = await self._perform_initial_evidence_analysis(evidence)
				evidence.analysis_performed.extend(analysis_result.get('analyses', []))
				evidence.analysis_results = analysis_result.get('results', {})
			
			await self._store_hunt_evidence(evidence)
			
			# Update finding with evidence reference
			finding.evidence_collected.append(evidence.id)
			await self._update_hunt_finding(finding)
			
			return evidence
			
		except Exception as e:
			self.logger.error(f"Error collecting hunt evidence: {str(e)}")
			raise
	
	async def conclude_hunt_investigation(self, finding_id: str, conclusion_data: Dict[str, Any]) -> HuntFinding:
		"""Conclude hunt investigation with final disposition"""
		try:
			finding = await self._get_hunt_finding(finding_id)
			if not finding:
				raise ValueError(f"Finding {finding_id} not found")
			
			# Set final outcome
			finding.outcome = HuntOutcome(conclusion_data['outcome'])
			finding.outcome_reason = conclusion_data.get('outcome_reason', '')
			finding.outcome_confidence = Decimal(str(conclusion_data.get('outcome_confidence', 85.0)))
			
			# Record response actions taken
			finding.response_actions = conclusion_data.get('response_actions', [])
			finding.containment_actions = conclusion_data.get('containment_actions', [])
			finding.remediation_actions = conclusion_data.get('remediation_actions', [])
			
			# Update investigation status
			finding.investigation_status = "concluded"
			
			# Add conclusion notes
			finding.analysis_notes.append({
				'timestamp': datetime.utcnow().isoformat(),
				'type': 'conclusion',
				'investigator': conclusion_data.get('concluded_by', finding.assigned_investigator),
				'outcome': finding.outcome.value,
				'confidence': float(finding.outcome_confidence),
				'notes': conclusion_data.get('conclusion_notes', '')
			})
			
			await self._update_hunt_finding(finding)
			
			# Update hunt statistics
			hunt = await self._get_threat_hunt(finding.hunt_id)
			if finding.outcome == HuntOutcome.TRUE_POSITIVE:
				hunt.true_positives += 1
			elif finding.outcome == HuntOutcome.FALSE_POSITIVE:
				hunt.false_positives += 1
			
			await self._update_threat_hunt(hunt)
			
			# Generate lessons learned if true positive
			if finding.outcome == HuntOutcome.TRUE_POSITIVE:
				lessons = await self._generate_lessons_learned(finding)
				hunt.lessons_learned.extend(lessons)
				await self._update_threat_hunt(hunt)
			
			return finding
			
		except Exception as e:
			self.logger.error(f"Error concluding hunt investigation: {str(e)}")
			raise
	
	async def generate_hunt_metrics(self, period_days: int = 30) -> HuntMetrics:
		"""Generate comprehensive hunt metrics"""
		try:
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(days=period_days)
			
			metrics = HuntMetrics(
				tenant_id=self.tenant_id,
				metric_period_start=start_time,
				metric_period_end=end_time
			)
			
			# Hunt campaign metrics
			all_hunts = await self._get_hunts_in_period(start_time, end_time)
			metrics.total_hunts = len(all_hunts)
			metrics.active_hunts = len([h for h in all_hunts if h.status == HuntStatus.ACTIVE])
			metrics.completed_hunts = len([h for h in all_hunts if h.status == HuntStatus.CONCLUDED])
			metrics.successful_hunts = len([h for h in all_hunts if h.true_positives > 0])
			
			# Hunt effectiveness
			all_findings = []
			for hunt in all_hunts:
				hunt_findings = await self._get_findings_for_hunt(hunt.id)
				all_findings.extend(hunt_findings)
			
			metrics.total_findings = len(all_findings)
			metrics.true_positives = len([f for f in all_findings if f.outcome == HuntOutcome.TRUE_POSITIVE])
			metrics.false_positives = len([f for f in all_findings if f.outcome == HuntOutcome.FALSE_POSITIVE])
			
			if metrics.total_findings > 0:
				metrics.true_positive_rate = Decimal(str((metrics.true_positives / metrics.total_findings) * 100))
				metrics.false_positive_rate = Decimal(str((metrics.false_positives / metrics.total_findings) * 100))
			
			# Query performance
			all_queries = await self._get_queries_in_period(start_time, end_time)
			metrics.total_queries_executed = sum(q.execution_count for q in all_queries)
			
			if all_queries:
				avg_execution_times = [float(q.average_execution_time) for q in all_queries if q.average_execution_time]
				if avg_execution_times:
					metrics.average_query_execution_time = Decimal(str(sum(avg_execution_times) / len(avg_execution_times)))
			
			# Data processing metrics
			metrics.total_data_processed = sum(h.data_volume_processed for h in all_hunts)
			if metrics.total_hunts > 0:
				metrics.average_data_volume_per_hunt = metrics.total_data_processed / metrics.total_hunts
			
			# Hunter productivity
			unique_hunters = set()
			for hunt in all_hunts:
				unique_hunters.add(hunt.lead_hunter)
				unique_hunters.update(hunt.hunt_team)
			
			metrics.total_hunters = len(unique_hunters)
			metrics.active_hunters = len(set(h.lead_hunter for h in all_hunts if h.status == HuntStatus.ACTIVE))
			
			if metrics.total_hunters > 0:
				metrics.average_hunts_per_hunter = Decimal(str(metrics.total_hunts / metrics.total_hunters))
			
			# Time metrics
			hunt_durations = [h.hunt_duration for h in all_hunts if h.hunt_duration]
			if hunt_durations:
				metrics.average_hunt_duration = sum(hunt_durations, timedelta()) / len(hunt_durations)
			
			# Threat coverage
			all_techniques = set()
			all_actors = set()
			for hunt in all_hunts:
				all_techniques.update(hunt.attack_techniques)
				all_actors.update(hunt.threat_actors)
			
			metrics.techniques_hunted = list(all_techniques)
			metrics.threat_actors_tracked = list(all_actors)
			
			# Business impact
			metrics.threats_detected = metrics.true_positives
			metrics.incidents_prevented = len([f for f in all_findings 
											 if f.outcome == HuntOutcome.TRUE_POSITIVE and 
											 'prevention' in f.response_actions])
			
			await self._store_hunt_metrics(metrics)
			
			return metrics
			
		except Exception as e:
			self.logger.error(f"Error generating hunt metrics: {str(e)}")
			raise
	
	async def create_hunt_template(self, template_data: Dict[str, Any]) -> HuntTemplate:
		"""Create reusable hunt template"""
		try:
			template = HuntTemplate(
				tenant_id=self.tenant_id,
				**template_data
			)
			
			# Validate template components
			validation_result = await self._validate_hunt_template(template)
			if not validation_result['valid']:
				raise ValueError(f"Invalid hunt template: {validation_result['errors']}")
			
			template.template_validation = validation_result
			
			# Generate template parameters schema
			if not template.parameter_validation:
				template.parameter_validation = await self._generate_parameter_validation_schema(template)
			
			await self._store_hunt_template(template)
			
			# Cache the template
			self._hunt_templates[template.id] = template
			
			return template
			
		except Exception as e:
			self.logger.error(f"Error creating hunt template: {str(e)}")
			raise
	
	# Helper methods for implementation
	async def _load_active_hunts(self):
		"""Load active hunts into cache"""
		pass
	
	async def _initialize_data_connectors(self):
		"""Initialize data source connectors"""
		pass
	
	async def _load_hunt_templates(self):
		"""Load hunt templates into cache"""
		pass
	
	async def _setup_query_engines(self):
		"""Setup query execution engines"""
		pass
	
	async def _initialize_evidence_store(self):
		"""Initialize evidence storage system"""
		pass
	
	async def _validate_hunt_parameters(self, hunt: ThreatHunt) -> Dict[str, Any]:
		"""Validate hunt parameters"""
		return {'valid': True, 'errors': []}
	
	async def _generate_default_hunt_phases(self, hunt_type: HuntType) -> List[Dict[str, Any]]:
		"""Generate default hunt phases based on type"""
		if hunt_type == HuntType.HYPOTHESIS_DRIVEN:
			return [
				{'name': 'initial_reconnaissance', 'automated': True},
				{'name': 'hypothesis_validation', 'automated': False},
				{'name': 'deep_investigation', 'automated': False},
				{'name': 'evidence_collection', 'automated': True},
				{'name': 'analysis_correlation', 'automated': False}
			]
		else:
			return [
				{'name': 'initial_reconnaissance', 'automated': True},
				{'name': 'deep_investigation', 'automated': False},
				{'name': 'evidence_collection', 'automated': True}
			]
	
	async def _execute_kql_query(self, query: HuntQuery, params: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute KQL query"""
		# Placeholder implementation
		return {'data': [], 'metadata': {'execution_time': 0.1}}
	
	async def _execute_spl_query(self, query: HuntQuery, params: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute Splunk SPL query"""
		# Placeholder implementation
		return {'data': [], 'metadata': {'execution_time': 0.1}}
	
	async def _execute_sql_query(self, query: HuntQuery, params: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute SQL query"""
		# Placeholder implementation
		return {'data': [], 'metadata': {'execution_time': 0.1}}
	
	async def _execute_elastic_query(self, query: HuntQuery, params: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Execute Elasticsearch query"""
		# Placeholder implementation
		return {'data': [], 'metadata': {'execution_time': 0.1}}
	
	# Placeholder implementations for database operations
	async def _store_threat_hunt(self, hunt: ThreatHunt):
		"""Store threat hunt to database"""
		pass
	
	async def _store_hunt_query(self, query: HuntQuery):
		"""Store hunt query to database"""
		pass
	
	async def _store_hunt_finding(self, finding: HuntFinding):
		"""Store hunt finding to database"""
		pass
	
	async def _store_hunt_evidence(self, evidence: HuntEvidence):
		"""Store hunt evidence to database"""
		pass
	
	async def _store_hunt_metrics(self, metrics: HuntMetrics):
		"""Store hunt metrics to database"""
		pass
	
	async def _store_hunt_template(self, template: HuntTemplate):
		"""Store hunt template to database"""
		pass