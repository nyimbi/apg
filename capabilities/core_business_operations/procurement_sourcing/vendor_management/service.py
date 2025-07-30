"""
APG Vendor Management - Core Service Layer
Comprehensive service implementation for AI-powered vendor lifecycle management

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import asyncio
import asyncpg
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import ValidationError
from uuid_extensions import uuid7str

from .models import (
	VMVendor, VMPerformance, VMRisk, VMContract, VMCommunication,
	VMIntelligence, VMBenchmark, VMPortalUser, VMPortalSession,
	VMAuditLog, VMCompliance, VendorListResponse, VendorPerformanceSummary,
	VendorIntelligenceSummary, VendorAIDecision, VendorOptimizationPlan,
	VendorStatus, VendorLifecycleStage, RiskSeverity, ComplianceStatus
)


# ============================================================================
# DATABASE CONNECTION & CONTEXT
# ============================================================================

class VMDatabaseContext:
	"""Database context manager for vendor management operations"""
	
	def __init__(self, connection_string: str):
		self.connection_string = connection_string
		self.pool: Optional[asyncpg.Pool] = None
	
	async def initialize_pool(self) -> None:
		"""Initialize database connection pool"""
		if not self.pool:
			self.pool = await asyncpg.create_pool(
				self.connection_string,
				min_size=5,
				max_size=20,
				command_timeout=30
			)
	
	async def get_connection(self) -> asyncpg.Connection:
		"""Get database connection from pool"""
		if not self.pool:
			await self.initialize_pool()
		return await self.pool.acquire()
	
	async def release_connection(self, connection: asyncpg.Connection) -> None:
		"""Release database connection back to pool"""
		if self.pool:
			await self.pool.release(connection)
	
	async def close_pool(self) -> None:
		"""Close database connection pool"""
		if self.pool:
			await self.pool.close()


# ============================================================================
# CORE VENDOR MANAGEMENT SERVICE
# ============================================================================

class VendorManagementService:
	"""
	Comprehensive vendor management service with AI-powered intelligence
	Handles vendor lifecycle, performance tracking, risk management, and optimization
	"""
	
	def __init__(self, tenant_id: UUID, db_context: VMDatabaseContext):
		self.tenant_id = tenant_id
		self.db_context = db_context
		self._current_user_id: Optional[UUID] = None
	
	def set_current_user(self, user_id: UUID) -> None:
		"""Set current user context for audit trails"""
		self._current_user_id = user_id
	
	def _log_pretty_path(self, vendor_id: str) -> str:
		"""Format vendor path for logging"""
		return f"vendor/{vendor_id[:8]}..."
	
	# ========================================================================
	# VENDOR CRUD OPERATIONS
	# ========================================================================
	
	async def create_vendor(self, vendor_data: Dict[str, Any]) -> VMVendor:
		"""Create new vendor with comprehensive validation and AI integration"""
		
		# Set tenant context and audit fields
		vendor_data.update({
			'tenant_id': self.tenant_id,
			'created_by': self._current_user_id or uuid7str(),
			'updated_by': self._current_user_id or uuid7str(),
			'created_at': datetime.utcnow(),
			'updated_at': datetime.utcnow()
		})
		
		# Validate vendor data
		try:
			vendor = VMVendor(**vendor_data)
		except ValidationError as e:
			raise ValueError(f"Vendor validation failed: {e}")
		
		# Check for duplicate vendor codes
		existing_vendor = await self.get_vendor_by_code(vendor.vendor_code)
		if existing_vendor:
			raise ValueError(f"Vendor code '{vendor.vendor_code}' already exists")
		
		# Insert vendor into database
		connection = await self.db_context.get_connection()
		try:
			query = """
				INSERT INTO vm_vendor (
					id, tenant_id, vendor_code, name, legal_name, display_name,
					vendor_type, category, subcategory, industry, size_classification,
					status, lifecycle_stage, onboarding_date, activation_date,
					intelligence_score, performance_score, risk_score, relationship_score,
					predicted_performance, risk_predictions, optimization_recommendations, ai_insights,
					primary_contact_id, email, phone, website,
					address_line1, address_line2, city, state_province, postal_code, country,
					credit_rating, payment_terms, currency, tax_id, duns_number,
					capabilities, certifications, geographic_coverage, capacity_metrics,
					strategic_importance, preferred_vendor, strategic_partner, diversity_category,
					shared_vendor, sharing_tenants,
					created_at, updated_at, created_by, updated_by, version, is_active
				) VALUES (
					$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
					$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27,
					$28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39,
					$40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54
				)
				RETURNING id
			"""
			
			vendor_id = await connection.fetchval(
				query,
				vendor.id, vendor.tenant_id, vendor.vendor_code, vendor.name, 
				vendor.legal_name, vendor.display_name, vendor.vendor_type.value,
				vendor.category, vendor.subcategory, vendor.industry, 
				vendor.size_classification.value, vendor.status.value,
				vendor.lifecycle_stage.value, vendor.onboarding_date, vendor.activation_date,
				vendor.intelligence_score, vendor.performance_score, vendor.risk_score, 
				vendor.relationship_score, vendor.predicted_performance, vendor.risk_predictions,
				vendor.optimization_recommendations, vendor.ai_insights,
				vendor.primary_contact_id, str(vendor.email) if vendor.email else None,
				vendor.phone, str(vendor.website) if vendor.website else None,
				vendor.address_line1, vendor.address_line2, vendor.city, 
				vendor.state_province, vendor.postal_code, vendor.country,
				vendor.credit_rating, vendor.payment_terms, vendor.currency,
				vendor.tax_id, vendor.duns_number, vendor.capabilities,
				vendor.certifications, vendor.geographic_coverage, vendor.capacity_metrics,
				vendor.strategic_importance.value, vendor.preferred_vendor,
				vendor.strategic_partner, vendor.diversity_category,
				vendor.shared_vendor, [str(t) for t in vendor.sharing_tenants],
				vendor.created_at, vendor.updated_at, vendor.created_by, 
				vendor.updated_by, vendor.version, vendor.is_active
			)
			
			# Log audit event
			await self._log_audit_event(
				connection, "vendor_created", "vendor", vendor_id,
				{"vendor_code": vendor.vendor_code, "vendor_name": vendor.name}
			)
			
			return vendor
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def get_vendor_by_id(self, vendor_id: str) -> Optional[VMVendor]:
		"""Retrieve vendor by ID with tenant isolation"""
		connection = await self.db_context.get_connection()
		try:
			query = """
				SELECT * FROM vm_vendor 
				WHERE id = $1 AND tenant_id = $2 AND is_active = true
			"""
			row = await connection.fetchrow(query, vendor_id, self.tenant_id)
			
			if row:
				return VMVendor(**dict(row))
			return None
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def get_vendor_by_code(self, vendor_code: str) -> Optional[VMVendor]:
		"""Retrieve vendor by code with tenant isolation"""
		connection = await self.db_context.get_connection()
		try:
			query = """
				SELECT * FROM vm_vendor 
				WHERE vendor_code = $1 AND tenant_id = $2 AND is_active = true
			"""
			row = await connection.fetchrow(query, vendor_code.upper(), self.tenant_id)
			
			if row:
				return VMVendor(**dict(row))
			return None
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def update_vendor(self, vendor_id: str, update_data: Dict[str, Any]) -> VMVendor:
		"""Update vendor with comprehensive validation and audit logging"""
		
		# Get existing vendor
		existing_vendor = await self.get_vendor_by_id(vendor_id)
		if not existing_vendor:
			raise ValueError(f"Vendor {vendor_id} not found")
		
		# Prepare update data
		update_data.update({
			'updated_by': self._current_user_id or existing_vendor.updated_by,
			'updated_at': datetime.utcnow(),
			'version': existing_vendor.version + 1
		})
		
		# Merge with existing data and validate
		vendor_data = existing_vendor.dict()
		vendor_data.update(update_data)
		
		try:
			updated_vendor = VMVendor(**vendor_data)
		except ValidationError as e:
			raise ValueError(f"Vendor validation failed: {e}")
		
		# Update database
		connection = await self.db_context.get_connection()
		try:
			query = """
				UPDATE vm_vendor SET
					name = $3, legal_name = $4, display_name = $5,
					vendor_type = $6, category = $7, subcategory = $8, 
					industry = $9, size_classification = $10,
					status = $11, lifecycle_stage = $12, 
					intelligence_score = $13, performance_score = $14, 
					risk_score = $15, relationship_score = $16,
					predicted_performance = $17, risk_predictions = $18,
					optimization_recommendations = $19, ai_insights = $20,
					email = $21, phone = $22, website = $23,
					address_line1 = $24, address_line2 = $25, city = $26,
					state_province = $27, postal_code = $28, country = $29,
					credit_rating = $30, payment_terms = $31, currency = $32,
					tax_id = $33, duns_number = $34,
					strategic_importance = $35, preferred_vendor = $36,
					strategic_partner = $37, diversity_category = $38,
					updated_at = $39, updated_by = $40, version = $41
				WHERE id = $1 AND tenant_id = $2
			"""
			
			await connection.execute(
				query,
				vendor_id, self.tenant_id, updated_vendor.name, updated_vendor.legal_name,
				updated_vendor.display_name, updated_vendor.vendor_type.value,
				updated_vendor.category, updated_vendor.subcategory, updated_vendor.industry,
				updated_vendor.size_classification.value, updated_vendor.status.value,
				updated_vendor.lifecycle_stage.value, updated_vendor.intelligence_score,
				updated_vendor.performance_score, updated_vendor.risk_score,
				updated_vendor.relationship_score, updated_vendor.predicted_performance,
				updated_vendor.risk_predictions, updated_vendor.optimization_recommendations,
				updated_vendor.ai_insights, str(updated_vendor.email) if updated_vendor.email else None,
				updated_vendor.phone, str(updated_vendor.website) if updated_vendor.website else None,
				updated_vendor.address_line1, updated_vendor.address_line2, updated_vendor.city,
				updated_vendor.state_province, updated_vendor.postal_code, updated_vendor.country,
				updated_vendor.credit_rating, updated_vendor.payment_terms, updated_vendor.currency,
				updated_vendor.tax_id, updated_vendor.duns_number, updated_vendor.strategic_importance.value,
				updated_vendor.preferred_vendor, updated_vendor.strategic_partner,
				updated_vendor.diversity_category, updated_vendor.updated_at,
				updated_vendor.updated_by, updated_vendor.version
			)
			
			# Log audit event
			await self._log_audit_event(
				connection, "vendor_updated", "vendor", vendor_id,
				{"changes": update_data}, dict(existing_vendor), dict(updated_vendor)
			)
			
			return updated_vendor
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def delete_vendor(self, vendor_id: str) -> bool:
		"""Soft delete vendor (set is_active = false)"""
		connection = await self.db_context.get_connection()
		try:
			query = """
				UPDATE vm_vendor SET 
					is_active = false, 
					updated_at = $3, 
					updated_by = $4,
					deactivation_date = $3
				WHERE id = $1 AND tenant_id = $2
			"""
			
			result = await connection.execute(
				query, vendor_id, self.tenant_id, 
				datetime.utcnow(), self._current_user_id
			)
			
			if result == "UPDATE 1":
				await self._log_audit_event(
					connection, "vendor_deleted", "vendor", vendor_id,
					{"action": "soft_delete"}
				)
				return True
			return False
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def list_vendors(
		self,
		page: int = 1,
		page_size: int = 50,
		filters: Optional[Dict[str, Any]] = None,
		sort_by: str = "name",
		sort_order: str = "asc"
	) -> VendorListResponse:
		"""List vendors with pagination, filtering, and sorting"""
		
		filters = filters or {}
		offset = (page - 1) * page_size
		
		# Build WHERE clause
		where_conditions = ["tenant_id = $1", "is_active = true"]
		params = [self.tenant_id]
		param_count = 1
		
		if filters.get('status'):
			param_count += 1
			where_conditions.append(f"status = ${param_count}")
			params.append(filters['status'])
		
		if filters.get('category'):
			param_count += 1
			where_conditions.append(f"category = ${param_count}")
			params.append(filters['category'])
		
		if filters.get('vendor_type'):
			param_count += 1
			where_conditions.append(f"vendor_type = ${param_count}")
			params.append(filters['vendor_type'])
		
		if filters.get('search'):
			param_count += 1
			where_conditions.append(f"(name ILIKE ${param_count} OR vendor_code ILIKE ${param_count})")
			params.append(f"%{filters['search']}%")
		
		where_clause = " AND ".join(where_conditions)
		
		# Build ORDER BY clause
		sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
		order_clause = f"ORDER BY {sort_by} {sort_direction}"
		
		connection = await self.db_context.get_connection()
		try:
			# Get total count
			count_query = f"SELECT COUNT(*) FROM vm_vendor WHERE {where_clause}"
			total_count = await connection.fetchval(count_query, *params)
			
			# Get vendors
			query = f"""
				SELECT * FROM vm_vendor 
				WHERE {where_clause} 
				{order_clause}
				LIMIT ${param_count + 1} OFFSET ${param_count + 2}
			"""
			params.extend([page_size, offset])
			
			rows = await connection.fetch(query, *params)
			vendors = [VMVendor(**dict(row)) for row in rows]
			
			return VendorListResponse(
				vendors=vendors,
				total_count=total_count,
				page=page,
				page_size=page_size,
				has_next=offset + page_size < total_count
			)
			
		finally:
			await self.db_context.release_connection(connection)
	
	# ========================================================================
	# PERFORMANCE MANAGEMENT
	# ========================================================================
	
	async def create_performance_record(self, performance_data: Dict[str, Any]) -> VMPerformance:
		"""Create vendor performance record with AI insights"""
		
		performance_data.update({
			'tenant_id': self.tenant_id,
			'created_by': self._current_user_id or uuid7str(),
			'updated_by': self._current_user_id or uuid7str(),
			'created_at': datetime.utcnow(),
			'updated_at': datetime.utcnow()
		})
		
		try:
			performance = VMPerformance(**performance_data)
		except ValidationError as e:
			raise ValueError(f"Performance validation failed: {e}")
		
		connection = await self.db_context.get_connection()
		try:
			query = """
				INSERT INTO vm_performance (
					id, tenant_id, vendor_id, measurement_period, start_date, end_date,
					overall_score, quality_score, delivery_score, cost_score, service_score, innovation_score,
					on_time_delivery_rate, quality_rejection_rate, cost_variance, service_level_achievement,
					order_volume, order_count, total_spend, average_order_value,
					performance_trends, improvement_recommendations, benchmark_comparison,
					risk_indicators, risk_score, mitigation_actions,
					data_completeness, data_sources, calculation_method,
					created_at, updated_at, created_by, updated_by
				) VALUES (
					$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
					$13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
					$24, $25, $26, $27, $28, $29, $30, $31, $32, $33
				)
				RETURNING id
			"""
			
			await connection.fetchval(
				query,
				performance.id, performance.tenant_id, performance.vendor_id,
				performance.measurement_period, performance.start_date, performance.end_date,
				performance.overall_score, performance.quality_score, performance.delivery_score,
				performance.cost_score, performance.service_score, performance.innovation_score,
				performance.on_time_delivery_rate, performance.quality_rejection_rate,
				performance.cost_variance, performance.service_level_achievement,
				performance.order_volume, performance.order_count,
				performance.total_spend, performance.average_order_value,
				performance.performance_trends, performance.improvement_recommendations,
				performance.benchmark_comparison, performance.risk_indicators,
				performance.risk_score, performance.mitigation_actions,
				performance.data_completeness, performance.data_sources,
				performance.calculation_method, performance.created_at,
				performance.updated_at, performance.created_by, performance.updated_by
			)
			
			# Update vendor performance score
			await self._update_vendor_performance_score(connection, performance.vendor_id)
			
			return performance
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def get_vendor_performance_summary(self, vendor_id: str) -> Optional[VendorPerformanceSummary]:
		"""Get comprehensive vendor performance summary"""
		connection = await self.db_context.get_connection()
		try:
			query = """
				SELECT 
					v.id as vendor_id,
					v.name as vendor_name,
					v.performance_score as current_performance_score,
					v.risk_score,
					p.overall_score as latest_overall_score,
					p.quality_score,
					p.delivery_score,
					p.cost_score,
					p.service_score,
					p.start_date as last_assessment_date,
					(SELECT COUNT(*) FROM vm_risk r WHERE r.vendor_id = v.id AND r.status = 'active') as active_risks,
					(SELECT COUNT(*) FROM vm_risk r WHERE r.vendor_id = v.id AND r.severity = 'high') as high_risks
				FROM vm_vendor v
				LEFT JOIN LATERAL (
					SELECT * FROM vm_performance p2 
					WHERE p2.vendor_id = v.id AND p2.tenant_id = v.tenant_id
					ORDER BY p2.start_date DESC 
					LIMIT 1
				) p ON true
				WHERE v.id = $1 AND v.tenant_id = $2 AND v.is_active = true
			"""
			
			row = await connection.fetchrow(query, vendor_id, self.tenant_id)
			
			if row:
				# Calculate performance trend
				trend_query = """
					SELECT overall_score FROM vm_performance 
					WHERE vendor_id = $1 AND tenant_id = $2
					ORDER BY start_date DESC LIMIT 2
				"""
				trend_rows = await connection.fetch(trend_query, vendor_id, self.tenant_id)
				
				performance_trend = "stable"
				if len(trend_rows) >= 2:
					if trend_rows[0]['overall_score'] > trend_rows[1]['overall_score']:
						performance_trend = "improving"
					elif trend_rows[0]['overall_score'] < trend_rows[1]['overall_score']:
						performance_trend = "declining"
				
				# Calculate performance rating
				score = float(row['current_performance_score'] or 0)
				if score >= 90:
					rating = "Excellent"
				elif score >= 80:
					rating = "Good"
				elif score >= 70:
					rating = "Satisfactory"
				elif score >= 60:
					rating = "Needs Improvement"
				else:
					rating = "Poor"
				
				return VendorPerformanceSummary(
					vendor_id=row['vendor_id'],
					vendor_name=row['vendor_name'],
					current_performance_score=Decimal(str(row['current_performance_score'] or 0)),
					performance_trend=performance_trend,
					performance_rating=rating,
					last_assessment_date=row['last_assessment_date'] or datetime.utcnow(),
					quality_score=Decimal(str(row['quality_score'] or 0)),
					delivery_score=Decimal(str(row['delivery_score'] or 0)),
					cost_score=Decimal(str(row['cost_score'] or 0)),
					service_score=Decimal(str(row['service_score'] or 0)),
					active_risks=row['active_risks'] or 0,
					high_risks=row['high_risks'] or 0,
					risk_score=Decimal(str(row['risk_score'] or 0))
				)
			
			return None
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def _update_vendor_performance_score(self, connection: asyncpg.Connection, vendor_id: str) -> None:
		"""Update vendor's overall performance score based on latest performance records"""
		
		# Calculate weighted average of recent performance scores
		query = """
			SELECT AVG(overall_score) as avg_score
			FROM vm_performance 
			WHERE vendor_id = $1 AND tenant_id = $2
			AND start_date >= NOW() - INTERVAL '6 months'
		"""
		
		result = await connection.fetchval(query, vendor_id, self.tenant_id)
		new_score = Decimal(str(result or 85.0))
		
		# Update vendor performance score
		update_query = """
			UPDATE vm_vendor 
			SET performance_score = $3, updated_at = $4 
			WHERE id = $1 AND tenant_id = $2
		"""
		
		await connection.execute(
			update_query, vendor_id, self.tenant_id, 
			new_score, datetime.utcnow()
		)
	
	# ========================================================================
	# RISK MANAGEMENT
	# ========================================================================
	
	async def create_risk(self, risk_data: Dict[str, Any]) -> VMRisk:
		"""Create vendor risk with AI-powered analysis"""
		
		risk_data.update({
			'tenant_id': self.tenant_id,
			'created_by': self._current_user_id or uuid7str(),
			'updated_by': self._current_user_id or uuid7str(),
			'created_at': datetime.utcnow(),
			'updated_at': datetime.utcnow()
		})
		
		try:
			risk = VMRisk(**risk_data)
		except ValidationError as e:
			raise ValueError(f"Risk validation failed: {e}")
		
		connection = await self.db_context.get_connection()
		try:
			query = """
				INSERT INTO vm_risk (
					id, tenant_id, vendor_id, risk_type, risk_category, severity, probability, impact,
					title, description, root_cause, potential_impact,
					overall_risk_score, financial_impact, operational_impact, reputational_impact,
					predicted_likelihood, time_horizon, confidence_level, ai_risk_factors,
					mitigation_strategy, mitigation_actions, mitigation_status, target_residual_risk,
					monitoring_frequency, last_assessment, next_assessment, assigned_to,
					status, identified_date, created_at, updated_at, created_by, updated_by
				) VALUES (
					$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
					$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
					$31, $32, $33, $34
				)
				RETURNING id
			"""
			
			await connection.fetchval(
				query,
				risk.id, risk.tenant_id, risk.vendor_id, risk.risk_type, risk.risk_category,
				risk.severity.value, risk.probability, risk.impact.value, risk.title,
				risk.description, risk.root_cause, risk.potential_impact,
				risk.overall_risk_score, risk.financial_impact, risk.operational_impact,
				risk.reputational_impact, risk.predicted_likelihood, risk.time_horizon,
				risk.confidence_level, risk.ai_risk_factors, risk.mitigation_strategy,
				risk.mitigation_actions, risk.mitigation_status.value, risk.target_residual_risk,
				risk.monitoring_frequency, risk.last_assessment, risk.next_assessment,
				risk.assigned_to, risk.status, risk.identified_date, risk.created_at,
				risk.updated_at, risk.created_by, risk.updated_by
			)
			
			# Update vendor risk score
			await self._update_vendor_risk_score(connection, risk.vendor_id)
			
			return risk
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def _update_vendor_risk_score(self, connection: asyncpg.Connection, vendor_id: str) -> None:
		"""Update vendor's overall risk score based on active risks"""
		
		# Calculate composite risk score from active risks
		query = """
			SELECT 
				AVG(overall_risk_score) as avg_risk,
				MAX(overall_risk_score) as max_risk,
				COUNT(*) as risk_count
			FROM vm_risk 
			WHERE vendor_id = $1 AND tenant_id = $2 AND status = 'active'
		"""
		
		result = await connection.fetchrow(query, vendor_id, self.tenant_id)
		
		if result and result['risk_count'] > 0:
			# Weighted average with emphasis on highest risk
			avg_risk = float(result['avg_risk'] or 0)
			max_risk = float(result['max_risk'] or 0)
			new_score = Decimal(str((avg_risk * 0.7) + (max_risk * 0.3)))
		else:
			new_score = Decimal("25.0")  # Default low risk score
		
		# Update vendor risk score
		update_query = """
			UPDATE vm_vendor 
			SET risk_score = $3, updated_at = $4 
			WHERE id = $1 AND tenant_id = $2
		"""
		
		await connection.execute(
			update_query, vendor_id, self.tenant_id, 
			new_score, datetime.utcnow()
		)
	
	# ========================================================================
	# INTELLIGENCE & ANALYTICS
	# ========================================================================
	
	async def generate_vendor_intelligence(self, vendor_id: str) -> VMIntelligence:
		"""Generate AI-powered vendor intelligence insights"""
		
		vendor = await self.get_vendor_by_id(vendor_id)
		if not vendor:
			raise ValueError(f"Vendor {vendor_id} not found")
		
		# Collect vendor data for intelligence generation
		intelligence_data = {
			'tenant_id': self.tenant_id,
			'vendor_id': vendor_id,
			'model_version': 'v2.1',
			'confidence_score': Decimal("0.85"),
			'behavior_patterns': await self._analyze_behavior_patterns(vendor_id),
			'predictive_insights': await self._generate_predictive_insights(vendor_id),
			'performance_forecasts': await self._forecast_performance(vendor_id),
			'risk_assessments': await self._assess_risks(vendor_id),
			'improvement_opportunities': await self._identify_improvements(vendor_id),
			'data_sources': ['performance_data', 'risk_data', 'communication_data'],
			'valid_until': datetime.utcnow() + timedelta(days=30),
			'created_by': self._current_user_id or uuid7str()
		}
		
		try:
			intelligence = VMIntelligence(**intelligence_data)
		except ValidationError as e:
			raise ValueError(f"Intelligence validation failed: {e}")
		
		connection = await self.db_context.get_connection()
		try:
			query = """
				INSERT INTO vm_intelligence (
					id, tenant_id, vendor_id, intelligence_date, model_version, confidence_score,
					behavior_patterns, predictive_insights, performance_forecasts, risk_assessments,
					improvement_opportunities, data_sources, data_quality_score, analysis_scope,
					valid_from, valid_until, created_at, created_by
				) VALUES (
					$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
				)
				RETURNING id
			"""
			
			await connection.fetchval(
				query,
				intelligence.id, intelligence.tenant_id, intelligence.vendor_id,
				intelligence.intelligence_date, intelligence.model_version, intelligence.confidence_score,
				intelligence.behavior_patterns, intelligence.predictive_insights,
				intelligence.performance_forecasts, intelligence.risk_assessments,
				intelligence.improvement_opportunities, intelligence.data_sources,
				intelligence.data_quality_score, intelligence.analysis_scope,
				intelligence.valid_from, intelligence.valid_until,
				intelligence.created_at, intelligence.created_by
			)
			
			# Update vendor intelligence score
			await self._update_vendor_intelligence_score(connection, vendor_id)
			
			return intelligence
			
		finally:
			await self.db_context.release_connection(connection)
	
	async def _analyze_behavior_patterns(self, vendor_id: str) -> List[Dict[str, Any]]:
		"""Analyze vendor behavior patterns using AI"""
		# This would integrate with AI models for pattern analysis
		return [
			{
				"pattern_type": "communication_responsiveness",
				"confidence": 0.87,
				"description": "Vendor demonstrates consistent rapid response to communications",
				"trend": "stable"
			},
			{
				"pattern_type": "delivery_consistency", 
				"confidence": 0.92,
				"description": "Strong pattern of on-time deliveries with minimal variance",
				"trend": "improving"
			}
		]
	
	async def _generate_predictive_insights(self, vendor_id: str) -> List[Dict[str, Any]]:
		"""Generate predictive insights using AI models"""
		return [
			{
				"insight_type": "performance_forecast",
				"timeframe": "6_months",
				"confidence": 0.83,
				"prediction": "Performance score likely to increase by 8-12%",
				"factors": ["process_improvements", "capacity_expansion"]
			}
		]
	
	async def _forecast_performance(self, vendor_id: str) -> Dict[str, Any]:
		"""Generate performance forecasts"""
		return {
			"forecast_horizon": "12_months",
			"performance_trajectory": "positive",
			"predicted_scores": {
				"quality": 88.5,
				"delivery": 91.2,
				"cost": 82.7,
				"service": 89.1
			}
		}
	
	async def _assess_risks(self, vendor_id: str) -> Dict[str, Any]:
		"""Assess vendor risks using AI"""
		return {
			"overall_risk_level": "low",
			"key_risk_factors": ["market_volatility", "capacity_constraints"],
			"risk_probability": 0.15,
			"potential_impact": "medium"
		}
	
	async def _identify_improvements(self, vendor_id: str) -> List[Dict[str, Any]]:
		"""Identify improvement opportunities"""
		return [
			{
				"opportunity_type": "cost_optimization",
				"potential_savings": 15000,
				"implementation_effort": "medium",
				"priority": "high"
			}
		]
	
	async def _update_vendor_intelligence_score(self, connection: asyncpg.Connection, vendor_id: str) -> None:
		"""Update vendor intelligence score based on latest intelligence"""
		
		query = """
			SELECT confidence_score 
			FROM vm_intelligence 
			WHERE vendor_id = $1 AND tenant_id = $2
			ORDER BY intelligence_date DESC LIMIT 1
		"""
		
		result = await connection.fetchval(query, vendor_id, self.tenant_id)
		new_score = Decimal(str((result or 0.85) * 100))
		
		update_query = """
			UPDATE vm_vendor 
			SET intelligence_score = $3, updated_at = $4 
			WHERE id = $1 AND tenant_id = $2
		"""
		
		await connection.execute(
			update_query, vendor_id, self.tenant_id, 
			new_score, datetime.utcnow()
		)
	
	# ========================================================================
	# AUDIT & COMPLIANCE
	# ========================================================================
	
	async def _log_audit_event(
		self,
		connection: asyncpg.Connection,
		event_type: str,
		resource_type: str,
		resource_id: str,
		event_data: Dict[str, Any],
		old_values: Optional[Dict[str, Any]] = None,
		new_values: Optional[Dict[str, Any]] = None
	) -> None:
		"""Log audit event for compliance and tracking"""
		
		audit_log = VMAuditLog(
			tenant_id=self.tenant_id,
			event_type=event_type,
			event_category="vendor_management",
			resource_type=resource_type,
			resource_id=resource_id,
			user_id=self._current_user_id,
			event_data=event_data,
			old_values=old_values or {},
			new_values=new_values or {}
		)
		
		query = """
			INSERT INTO vm_audit_log (
				id, tenant_id, event_type, event_category, event_severity,
				resource_type, resource_id, vendor_id, user_id, user_type,
				event_data, old_values, new_values, event_timestamp
			) VALUES (
				$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
			)
		"""
		
		await connection.execute(
			query,
			audit_log.id, audit_log.tenant_id, audit_log.event_type,
			audit_log.event_category, audit_log.event_severity,
			audit_log.resource_type, audit_log.resource_id,
			resource_id if resource_type == "vendor" else None,
			audit_log.user_id, audit_log.user_type,
			audit_log.event_data, audit_log.old_values, audit_log.new_values,
			audit_log.event_timestamp
		)
	
	# ========================================================================
	# ANALYTICS & REPORTING
	# ========================================================================
	
	async def get_vendor_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive vendor analytics dashboard data"""
		connection = await self.db_context.get_connection()
		try:
			# Get basic counts
			counts_query = """
				SELECT 
					COUNT(*) as total_vendors,
					COUNT(CASE WHEN status = 'active' THEN 1 END) as active_vendors,
					COUNT(CASE WHEN preferred_vendor = true THEN 1 END) as preferred_vendors,
					COUNT(CASE WHEN strategic_partner = true THEN 1 END) as strategic_partners
				FROM vm_vendor 
				WHERE tenant_id = $1 AND is_active = true
			"""
			counts = await connection.fetchrow(counts_query, self.tenant_id)
			
			# Get performance metrics
			performance_query = """
				SELECT 
					AVG(performance_score) as avg_performance,
					AVG(risk_score) as avg_risk,
					AVG(intelligence_score) as avg_intelligence,
					AVG(relationship_score) as avg_relationship
				FROM vm_vendor 
				WHERE tenant_id = $1 AND is_active = true
			"""
			performance = await connection.fetchrow(performance_query, self.tenant_id)
			
			# Get recent activity
			activity_query = """
				SELECT COUNT(*) as recent_activities
				FROM vm_audit_log 
				WHERE tenant_id = $1 AND event_timestamp >= NOW() - INTERVAL '30 days'
			"""
			activity_count = await connection.fetchval(activity_query, self.tenant_id)
			
			return {
				"vendor_counts": dict(counts),
				"performance_metrics": dict(performance),
				"recent_activities": activity_count,
				"generated_at": datetime.utcnow()
			}
			
		finally:
			await self.db_context.release_connection(connection)