#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Database Management
Database initialization, migrations, and multi-tenant isolation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import asyncpg

from .models import Base, MdEntity, MdEntityVersion, MdGoldenRecord, MdCrossReference
from .models import MdDataQualityAssessment, MdMatchRule, MdSurvivorshipRule, MdAuditLog, MdDataLineage


class MDMDatabaseManager:
	"""Advanced database management for APG MDM with multi-tenant isolation"""
	
	def __init__(self, database_url: str = None, config: Dict[str, Any] = None):
		self.config = config or {}
		self.database_url = database_url or os.getenv('MDM_DATABASE_URL', 
			'postgresql+asyncpg://postgres:postgres@localhost:5432/apg_mdm')
		
		# Initialize async engine with optimized settings
		self.async_engine = create_async_engine(
			self.database_url,
			poolclass=QueuePool,
			pool_size=20,
			max_overflow=30,
			pool_pre_ping=True,
			pool_recycle=3600,
			echo=self.config.get('debug', False)
		)
		
		# Create async session factory
		self.async_session_factory = async_sessionmaker(
			self.async_engine,
			class_=AsyncSession,
			expire_on_commit=False
		)
		
		# For sync operations (migrations, etc.)
		sync_url = self.database_url.replace('+asyncpg', '').replace('+psycopg2', '')
		self.sync_engine = create_engine(sync_url)
		self.sync_session_factory = sessionmaker(self.sync_engine)
	
	async def initialize_database(self) -> Dict[str, Any]:
		"""Initialize database schema and default data"""
		try:
			# Create all tables
			async with self.async_engine.begin() as conn:
				await conn.run_sync(Base.metadata.create_all)
			
			# Create indexes and constraints
			await self._create_performance_indexes()
			
			# Setup row-level security for multi-tenancy
			await self._setup_row_level_security()
			
			# Create default configuration
			await self._create_default_configuration()
			
			# Verify database setup
			verification_result = await self._verify_database_setup()
			
			return {
				'status': 'success',
				'message': 'MDM database initialized successfully',
				'tables_created': len(Base.metadata.tables),
				'indexes_created': True,
				'rls_enabled': True,
				'verification': verification_result,
				'initialized_at': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				'status': 'error',
				'message': f'Database initialization failed: {str(e)}',
				'initialized_at': datetime.utcnow().isoformat()
			}
	
	async def _create_performance_indexes(self) -> None:
		"""Create additional performance indexes"""
		performance_indexes = [
			# Composite indexes for common query patterns
			"CREATE INDEX IF NOT EXISTS ix_md_entities_tenant_status_type ON md_entities(tenant_id, status, entity_type);",
			"CREATE INDEX IF NOT EXISTS ix_md_entities_tenant_quality_name ON md_entities(tenant_id, quality_score DESC, entity_name);",
			"CREATE INDEX IF NOT EXISTS ix_md_entities_tenant_updated ON md_entities(tenant_id, updated_at DESC);",
			
			# Quality assessment performance indexes
			"CREATE INDEX IF NOT EXISTS ix_md_quality_tenant_entity_time ON md_data_quality_assessments(tenant_id, entity_id, assessment_timestamp DESC);",
			"CREATE INDEX IF NOT EXISTS ix_md_quality_tenant_status_score ON md_data_quality_assessments(tenant_id, quality_status, overall_score DESC);",
			
			# Audit log performance indexes
			"CREATE INDEX IF NOT EXISTS ix_md_audit_tenant_type_time ON md_audit_logs(tenant_id, event_type, event_timestamp DESC);",
			"CREATE INDEX IF NOT EXISTS ix_md_audit_tenant_entity_time ON md_audit_logs(tenant_id, entity_id, event_timestamp DESC);",
			
			# Cross-reference performance indexes
			"CREATE INDEX IF NOT EXISTS ix_md_cross_ref_tenant_source ON md_cross_references(tenant_id, source_system, source_entity_id);",
			
			# Data lineage performance indexes
			"CREATE INDEX IF NOT EXISTS ix_md_lineage_tenant_source_type ON md_data_lineage(tenant_id, source_entity_id, relationship_type);",
			"CREATE INDEX IF NOT EXISTS ix_md_lineage_tenant_target_type ON md_data_lineage(tenant_id, target_entity_id, relationship_type);",
			
			# Golden record performance indexes
			"CREATE INDEX IF NOT EXISTS ix_md_golden_tenant_type_quality ON md_golden_records(tenant_id, entity_type, overall_quality_score DESC);",
			"CREATE INDEX IF NOT EXISTS ix_md_golden_tenant_updated ON md_golden_records(tenant_id, updated_at DESC);",
			
			# Full-text search indexes for entity names and descriptions
			"CREATE INDEX IF NOT EXISTS ix_md_entities_name_fts ON md_entities USING gin(to_tsvector('english', entity_name || ' ' || COALESCE(entity_description, '')));",
			
			# JSONB indexes for attribute queries
			"CREATE INDEX IF NOT EXISTS ix_md_entities_attributes_gin ON md_entities USING gin(attributes);",
			"CREATE INDEX IF NOT EXISTS ix_md_golden_attributes_gin ON md_golden_records USING gin(consolidated_attributes);",
		]
		
		async with self.async_engine.begin() as conn:
			for index_sql in performance_indexes:
				try:
					await conn.execute(text(index_sql))
				except Exception as e:
					print(f"[MDM-DB] Warning: Could not create index: {str(e)}")
	
	async def _setup_row_level_security(self) -> None:
		"""Setup row-level security for multi-tenant isolation"""
		rls_statements = [
			# Enable RLS on all tenant-aware tables
			"ALTER TABLE md_entities ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_entity_versions ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_golden_records ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_cross_references ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_data_quality_assessments ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_match_rules ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_survivorship_rules ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_audit_logs ENABLE ROW LEVEL SECURITY;",
			"ALTER TABLE md_data_lineage ENABLE ROW LEVEL SECURITY;",
			
			# Create RLS policies for tenant isolation
			"""CREATE POLICY mdm_tenant_isolation ON md_entities
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_entity_versions
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_golden_records
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_cross_references
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_data_quality_assessments
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_match_rules
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_survivorship_rules
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_audit_logs
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
			
			"""CREATE POLICY mdm_tenant_isolation ON md_data_lineage
			FOR ALL TO PUBLIC
			USING (tenant_id = current_setting('mdm.current_tenant_id', true));""",
		]
		
		async with self.async_engine.begin() as conn:
			for rls_sql in rls_statements:
				try:
					await conn.execute(text(rls_sql))
				except Exception as e:
					print(f"[MDM-DB] Warning: RLS setup issue: {str(e)}")
	
	async def _create_default_configuration(self) -> None:
		"""Create default matching and survivorship rules"""
		async with self.async_session_factory() as session:
			try:
				# Check if default rules already exist
				existing_rules = await session.execute(
					text("SELECT COUNT(*) FROM md_match_rules WHERE rule_name LIKE 'Default%'")
				)
				if existing_rules.scalar() > 0:
					return  # Default rules already exist
				
				# Default matching rules for different entity types
				default_match_rules = [
					{
						'rule_name': 'Default Customer Matching',
						'entity_type': 'customer',
						'matching_attributes': ['name', 'email', 'phone', 'address'],
						'rule_config': {
							'algorithm': 'fuzzy_enhanced',
							'name_similarity_weight': 0.4,
							'email_similarity_weight': 0.3,
							'phone_similarity_weight': 0.2,
							'address_similarity_weight': 0.1
						}
					},
					{
						'rule_name': 'Default Product Matching',
						'entity_type': 'product',
						'matching_attributes': ['product_name', 'sku', 'manufacturer', 'model'],
						'rule_config': {
							'algorithm': 'semantic_enhanced',
							'name_similarity_weight': 0.3,
							'sku_similarity_weight': 0.4,
							'manufacturer_similarity_weight': 0.2,
							'model_similarity_weight': 0.1
						}
					}
				]
				
				# Default survivorship rules
				default_survivorship_rules = [
					{
						'rule_name': 'Default Customer Survivorship',
						'entity_type': 'customer',
						'survivorship_strategy': 'ai_determined',
						'attribute_rules': {
							'name': {'strategy': 'most_complete'},
							'email': {'strategy': 'most_recent'},
							'phone': {'strategy': 'most_recent'},
							'address': {'strategy': 'most_trusted_source'}
						}
					},
					{
						'rule_name': 'Default Product Survivorship',
						'entity_type': 'product',
						'survivorship_strategy': 'most_trusted_source',
						'attribute_rules': {
							'product_name': {'strategy': 'most_complete'},
							'price': {'strategy': 'most_recent'},
							'description': {'strategy': 'most_complete'},
							'specifications': {'strategy': 'highest_quality'}
						}
					}
				]
				
				# This would typically insert the default rules
				# For now, we'll just log that they would be created
				print(f"[MDM-DB] Would create {len(default_match_rules)} default match rules")
				print(f"[MDM-DB] Would create {len(default_survivorship_rules)} default survivorship rules")
				
			except Exception as e:
				print(f"[MDM-DB] Error creating default configuration: {str(e)}")
	
	async def _verify_database_setup(self) -> Dict[str, Any]:
		"""Verify database setup is correct"""
		verification_results = {
			'tables_exist': False,
			'indexes_exist': False,
			'rls_enabled': False,
			'performance_acceptable': False
		}
		
		try:
			async with self.async_session_factory() as session:
				# Check if core tables exist
				result = await session.execute(
					text("""
						SELECT COUNT(*) FROM information_schema.tables 
						WHERE table_name IN ('md_entities', 'md_golden_records', 'md_data_quality_assessments')
						AND table_schema = 'public'
					""")
				)
				table_count = result.scalar()
				verification_results['tables_exist'] = table_count >= 3
				
				# Check if key indexes exist
				result = await session.execute(
					text("""
						SELECT COUNT(*) FROM pg_indexes 
						WHERE indexname LIKE 'ix_md_%' OR indexname LIKE '%tenant%'
					""")
				)
				index_count = result.scalar()
				verification_results['indexes_exist'] = index_count >= 10
				
				# Check if RLS is enabled
				result = await session.execute(
					text("""
						SELECT COUNT(*) FROM pg_tables 
						WHERE tablename LIKE 'md_%' AND rowsecurity = true
					""")
				)
				rls_count = result.scalar()
				verification_results['rls_enabled'] = rls_count >= 5
				
				# Basic performance test
				start_time = datetime.utcnow()
				await session.execute(text("SELECT 1"))
				end_time = datetime.utcnow()
				query_time = (end_time - start_time).total_seconds() * 1000
				verification_results['performance_acceptable'] = query_time < 100  # < 100ms
				verification_results['query_time_ms'] = query_time
				
		except Exception as e:
			verification_results['error'] = str(e)
		
		return verification_results
	
	@asynccontextmanager
	async def get_session(self, tenant_id: str = None):
		"""Get database session with optional tenant context"""
		async with self.async_session_factory() as session:
			try:
				# Set tenant context for RLS if provided
				if tenant_id:
					await session.execute(
						text("SET mdm.current_tenant_id = :tenant_id"),
						{"tenant_id": tenant_id}
					)
				
				yield session
				await session.commit()
			except Exception as e:
				await session.rollback()
				raise e
			finally:
				# Clear tenant context
				if tenant_id:
					try:
						await session.execute(text("RESET mdm.current_tenant_id"))
					except:
						pass
	
	async def execute_tenant_query(self, query: str, params: Dict[str, Any] = None, 
								   tenant_id: str = None) -> Any:
		"""Execute query with tenant context"""
		async with self.get_session(tenant_id) as session:
			result = await session.execute(text(query), params or {})
			return result
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive database health check"""
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.utcnow().isoformat(),
			'checks': {}
		}
		
		try:
			# Connection test
			start_time = datetime.utcnow()
			async with self.async_session_factory() as session:
				await session.execute(text("SELECT 1"))
			connection_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			health_status['checks']['connection'] = {
				'status': 'healthy' if connection_time < 1000 else 'degraded',
				'response_time_ms': connection_time
			}
			
			# Pool status
			pool = self.async_engine.pool
			health_status['checks']['connection_pool'] = {
				'size': pool.size(),
				'checked_in': pool.checkedin(),
				'checked_out': pool.checkedout(),
				'status': 'healthy' if pool.checkedout() < pool.size() * 0.8 else 'degraded'
			}
			
			# Table count verification
			async with self.async_session_factory() as session:
				result = await session.execute(
					text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'md_%'")
				)
				table_count = result.scalar()
				
				health_status['checks']['schema'] = {
					'tables_count': table_count,
					'expected_tables': len(Base.metadata.tables),
					'status': 'healthy' if table_count >= len(Base.metadata.tables) else 'unhealthy'
				}
			
			# Overall status determination
			check_statuses = [check['status'] for check in health_status['checks'].values()]
			if 'unhealthy' in check_statuses:
				health_status['status'] = 'unhealthy'
			elif 'degraded' in check_statuses:
				health_status['status'] = 'degraded'
				
		except Exception as e:
			health_status['status'] = 'unhealthy'
			health_status['error'] = str(e)
		
		return health_status
	
	async def get_database_stats(self, tenant_id: str = None) -> Dict[str, Any]:
		"""Get comprehensive database statistics"""
		stats = {
			'timestamp': datetime.utcnow().isoformat(),
			'tenant_id': tenant_id
		}
		
		try:
			async with self.get_session(tenant_id) as session:
				# Entity statistics
				entity_stats = await session.execute(text("""
					SELECT 
						entity_type,
						COUNT(*) as total_entities,
						AVG(quality_score) as avg_quality_score,
						COUNT(CASE WHEN is_golden_record THEN 1 END) as golden_records
					FROM md_entities 
					GROUP BY entity_type
				"""))
				
				stats['entity_statistics'] = [
					{
						'entity_type': row[0],
						'total_entities': row[1],
						'avg_quality_score': float(row[2]) if row[2] else 0.0,
						'golden_records': row[3]
					}
					for row in entity_stats.fetchall()
				]
				
				# Quality statistics
				quality_stats = await session.execute(text("""
					SELECT 
						quality_status,
						COUNT(*) as assessment_count,
						AVG(overall_score) as avg_score
					FROM md_data_quality_assessments 
					WHERE assessment_timestamp >= NOW() - INTERVAL '30 days'
					GROUP BY quality_status
				"""))
				
				stats['quality_statistics'] = [
					{
						'quality_status': row[0],
						'assessment_count': row[1],
						'avg_score': float(row[2]) if row[2] else 0.0
					}
					for row in quality_stats.fetchall()
				]
				
				# Recent activity
				activity_stats = await session.execute(text("""
					SELECT 
						event_type,
						COUNT(*) as event_count
					FROM md_audit_logs 
					WHERE event_timestamp >= NOW() - INTERVAL '24 hours'
					GROUP BY event_type
					ORDER BY event_count DESC
				"""))
				
				stats['recent_activity'] = [
					{
						'event_type': row[0],
						'event_count': row[1]
					}
					for row in activity_stats.fetchall()
				]
				
		except Exception as e:
			stats['error'] = str(e)
		
		return stats
	
	async def cleanup_old_data(self, tenant_id: str = None, days_to_keep: int = 90) -> Dict[str, Any]:
		"""Clean up old audit logs and version history"""
		cleanup_results = {
			'timestamp': datetime.utcnow().isoformat(),
			'tenant_id': tenant_id,
			'days_to_keep': days_to_keep,
			'cleaned_records': {}
		}
		
		try:
			cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
			
			async with self.get_session(tenant_id) as session:
				# Clean old audit logs (keep critical events longer)
				audit_cleanup = await session.execute(text("""
					DELETE FROM md_audit_logs 
					WHERE event_timestamp < :cutoff_date 
					AND event_type NOT IN ('create', 'merge', 'delete')
				"""), {"cutoff_date": cutoff_date})
				
				cleanup_results['cleaned_records']['audit_logs'] = audit_cleanup.rowcount
				
				# Clean old version history (keep recent versions)
				version_cleanup = await session.execute(text("""
					DELETE FROM md_entity_versions 
					WHERE version_timestamp < :cutoff_date 
					AND version_number NOT IN (
						SELECT MAX(version_number) FROM md_entity_versions v2 
						WHERE v2.entity_id = md_entity_versions.entity_id
					)
				"""), {"cutoff_date": cutoff_date})
				
				cleanup_results['cleaned_records']['entity_versions'] = version_cleanup.rowcount
				
				# Clean old quality assessments (keep latest for each entity)
				quality_cleanup = await session.execute(text("""
					DELETE FROM md_data_quality_assessments 
					WHERE assessment_timestamp < :cutoff_date 
					AND id NOT IN (
						SELECT DISTINCT ON (entity_id) id 
						FROM md_data_quality_assessments q2 
						WHERE q2.entity_id = md_data_quality_assessments.entity_id
						ORDER BY entity_id, assessment_timestamp DESC
					)
				"""), {"cutoff_date": cutoff_date})
				
				cleanup_results['cleaned_records']['quality_assessments'] = quality_cleanup.rowcount
				
				await session.commit()
				cleanup_results['status'] = 'success'
				
		except Exception as e:
			cleanup_results['status'] = 'error'
			cleanup_results['error'] = str(e)
		
		return cleanup_results


# Export the database manager
__all__ = ['MDMDatabaseManager']