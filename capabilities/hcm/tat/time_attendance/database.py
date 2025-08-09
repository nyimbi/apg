"""
Time & Attendance Capability Database Configuration

PostgreSQL database setup with multi-tenant isolation, comprehensive indexing,
and APG ecosystem integration for the revolutionary Time & Attendance capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from alembic import command
from alembic.config import Config
import asyncpg

from .config import get_config

logger = logging.getLogger(__name__)

# Database Base
Base = declarative_base()

class DatabaseManager:
	"""
	Multi-tenant PostgreSQL database manager for Time & Attendance capability
	
	Provides tenant isolation, connection pooling, and APG ecosystem integration.
	"""
	
	def __init__(self, config=None):
		self.config = config or get_config()
		self.engines: Dict[str, Any] = {}
		self.sessions: Dict[str, Any] = {}
		self.metadata = MetaData()
		
	async def initialize(self) -> None:
		"""Initialize database connections and schemas"""
		logger.info("Initializing Time & Attendance database manager")
		
		try:
			# Create main engine
			await self._create_main_engine()
			
			# Setup tenant isolation
			await self._setup_tenant_isolation()
			
			# Create schemas if needed
			await self._create_schemas()
			
			# Setup monitoring
			await self._setup_monitoring()
			
			logger.info("Database manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Error initializing database manager: {str(e)}")
			raise
	
	async def _create_main_engine(self) -> None:
		"""Create main database engine with connection pooling"""
		
		# Async engine for main operations
		async_database_url = (
			f"postgresql+asyncpg://{self.config.database.username}:"
			f"{self.config.database.password}@{self.config.database.host}:"
			f"{self.config.database.port}/{self.config.database.database_name}"
		)
		
		self.engines['async'] = create_async_engine(
			async_database_url,
			pool_size=self.config.database.pool_size,
			max_overflow=self.config.database.max_overflow,
			pool_timeout=self.config.database.pool_timeout,
			pool_recycle=3600,  # Recycle connections every hour
			echo=self.config.database.echo_sql,
			future=True
		)
		
		# Sync engine for migrations
		sync_database_url = (
			f"postgresql://{self.config.database.username}:"
			f"{self.config.database.password}@{self.config.database.host}:"
			f"{self.config.database.port}/{self.config.database.database_name}"
		)
		
		self.engines['sync'] = create_engine(
			sync_database_url,
			pool_size=self.config.database.pool_size,
			max_overflow=self.config.database.max_overflow,
			echo=self.config.database.echo_sql
		)
		
		# Create session factories
		self.sessions['async'] = async_sessionmaker(
			bind=self.engines['async'],
			class_=AsyncSession,
			expire_on_commit=False
		)
		
		self.sessions['sync'] = sessionmaker(
			bind=self.engines['sync'],
			expire_on_commit=False
		)
	
	async def _setup_tenant_isolation(self) -> None:
		"""Setup tenant isolation using schema-based approach"""
		logger.info("Setting up tenant isolation")
		
		# Create tenant management schema
		async with self.engines['async'].begin() as conn:
			await conn.execute(text("CREATE SCHEMA IF NOT EXISTS ta_tenant_management"))
			
			# Create tenant registry table
			await conn.execute(text("""
				CREATE TABLE IF NOT EXISTS ta_tenant_management.tenants (
					id VARCHAR(36) PRIMARY KEY,
					name VARCHAR(255) NOT NULL,
					schema_name VARCHAR(63) NOT NULL UNIQUE,
					status VARCHAR(20) DEFAULT 'active',
					created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
					updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
					configuration JSONB DEFAULT '{}'::jsonb
				)
			"""))
			
			# Create indexes
			await conn.execute(text("""
				CREATE INDEX IF NOT EXISTS idx_tenants_status 
				ON ta_tenant_management.tenants(status)
			"""))
			
			await conn.execute(text("""
				CREATE INDEX IF NOT EXISTS idx_tenants_schema_name 
				ON ta_tenant_management.tenants(schema_name)
			"""))
	
	async def _create_schemas(self) -> None:
		"""Create database schemas for all tables"""
		logger.info("Creating database schemas")
		
		schema_sql = """
		-- Core Time & Attendance Schema
		CREATE SCHEMA IF NOT EXISTS ta_core;
		
		-- Extensions
		CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
		CREATE EXTENSION IF NOT EXISTS "pg_trgm";
		CREATE EXTENSION IF NOT EXISTS "btree_gin";
		
		-- Employees table
		CREATE TABLE IF NOT EXISTS ta_core.ta_employees (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_number VARCHAR(50) UNIQUE NOT NULL,
			first_name VARCHAR(100) NOT NULL,
			last_name VARCHAR(100) NOT NULL,
			email VARCHAR(255) UNIQUE NOT NULL,
			department_id VARCHAR(36),
			manager_id VARCHAR(36),
			hire_date DATE NOT NULL,
			employment_status VARCHAR(20) DEFAULT 'active',
			workforce_type VARCHAR(20) DEFAULT 'human',
			work_schedule JSONB DEFAULT '{}'::jsonb,
			biometric_templates JSONB DEFAULT '{}'::jsonb,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_employees_manager FOREIGN KEY (manager_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- Time Entries table
		CREATE TABLE IF NOT EXISTS ta_core.ta_time_entries (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_id VARCHAR(36) NOT NULL,
			entry_date DATE NOT NULL,
			clock_in TIMESTAMP WITH TIME ZONE,
			clock_out TIMESTAMP WITH TIME ZONE,
			total_hours DECIMAL(8,2),
			regular_hours DECIMAL(8,2),
			overtime_hours DECIMAL(8,2),
			break_minutes INTEGER DEFAULT 0,
			entry_type VARCHAR(20) DEFAULT 'regular',
			status VARCHAR(20) DEFAULT 'draft',
			clock_in_location JSONB,
			clock_out_location JSONB,
			device_info JSONB DEFAULT '{}'::jsonb,
			biometric_verification JSONB DEFAULT '{}'::jsonb,
			verification_confidence DECIMAL(3,2),
			fraud_indicators JSONB DEFAULT '[]'::jsonb,
			anomaly_score DECIMAL(3,2) DEFAULT 0.00,
			validation_results JSONB DEFAULT '{}'::jsonb,
			requires_approval BOOLEAN DEFAULT false,
			approval_status VARCHAR(20),
			approved_by VARCHAR(36),
			approved_at TIMESTAMP WITH TIME ZONE,
			project_assignments JSONB DEFAULT '[]'::jsonb,
			notes TEXT,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_time_entries_employee FOREIGN KEY (employee_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- Schedules table
		CREATE TABLE IF NOT EXISTS ta_core.ta_schedules (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_id VARCHAR(36) NOT NULL,
			schedule_name VARCHAR(200) NOT NULL,
			schedule_type VARCHAR(30) DEFAULT 'fixed',
			effective_date DATE NOT NULL,
			end_date DATE,
			weekly_hours DECIMAL(4,2),
			daily_schedule JSONB NOT NULL,
			break_schedule JSONB DEFAULT '{}'::jsonb,
			overtime_rules JSONB DEFAULT '{}'::jsonb,
			location_constraints JSONB DEFAULT '{}'::jsonb,
			is_active BOOLEAN DEFAULT true,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_schedules_employee FOREIGN KEY (employee_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- Leave Requests table
		CREATE TABLE IF NOT EXISTS ta_core.ta_leave_requests (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_id VARCHAR(36) NOT NULL,
			leave_type VARCHAR(30) NOT NULL,
			start_date DATE NOT NULL,
			end_date DATE NOT NULL,
			days_requested INTEGER NOT NULL,
			reason TEXT,
			status VARCHAR(20) DEFAULT 'pending',
			approval_workflow JSONB DEFAULT '[]'::jsonb,
			supporting_documents JSONB DEFAULT '[]'::jsonb,
			is_emergency BOOLEAN DEFAULT false,
			auto_approved BOOLEAN DEFAULT false,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_leave_requests_employee FOREIGN KEY (employee_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- Fraud Detection table
		CREATE TABLE IF NOT EXISTS ta_core.ta_fraud_detection (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			time_entry_id VARCHAR(36) NOT NULL,
			fraud_type VARCHAR(30) NOT NULL,
			severity VARCHAR(20) NOT NULL,
			confidence_score DECIMAL(3,2) NOT NULL,
			indicators JSONB NOT NULL,
			location_analysis JSONB DEFAULT '{}'::jsonb,
			device_analysis JSONB DEFAULT '{}'::jsonb,
			pattern_analysis JSONB DEFAULT '{}'::jsonb,
			remediation_status VARCHAR(20) DEFAULT 'pending',
			investigated_by VARCHAR(36),
			investigation_notes TEXT,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			CONSTRAINT fk_ta_fraud_detection_time_entry FOREIGN KEY (time_entry_id) REFERENCES ta_core.ta_time_entries(id)
		);
		
		-- Biometric Authentication table
		CREATE TABLE IF NOT EXISTS ta_core.ta_biometric_authentication (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_id VARCHAR(36) NOT NULL,
			biometric_type VARCHAR(20) NOT NULL,
			template_data TEXT NOT NULL,
			enrollment_date DATE NOT NULL,
			last_verification TIMESTAMP WITH TIME ZONE,
			verification_count INTEGER DEFAULT 0,
			quality_score DECIMAL(3,2),
			is_active BOOLEAN DEFAULT true,
			device_info JSONB DEFAULT '{}'::jsonb,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_biometric_auth_employee FOREIGN KEY (employee_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- Predictive Analytics table
		CREATE TABLE IF NOT EXISTS ta_core.ta_predictive_analytics (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			analysis_name VARCHAR(200) NOT NULL,
			analysis_type VARCHAR(50) NOT NULL,
			date_range JSONB NOT NULL,
			models_used JSONB DEFAULT '[]'::jsonb,
			model_confidence DECIMAL(3,2) NOT NULL,
			staffing_predictions JSONB DEFAULT '{}'::jsonb,
			absence_predictions JSONB DEFAULT '{}'::jsonb,
			overtime_predictions JSONB DEFAULT '{}'::jsonb,
			productivity_trends JSONB DEFAULT '{}'::jsonb,
			efficiency_opportunities JSONB DEFAULT '[]'::jsonb,
			cost_optimization JSONB DEFAULT '{}'::jsonb,
			compliance_risks JSONB DEFAULT '[]'::jsonb,
			operational_risks JSONB DEFAULT '[]'::jsonb,
			actionable_insights JSONB DEFAULT '[]'::jsonb,
			projected_savings DECIMAL(12,2),
			roi_estimates JSONB DEFAULT '{}'::jsonb,
			strategic_recommendations JSONB DEFAULT '[]'::jsonb,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL
		);
		
		-- Compliance Rules table
		CREATE TABLE IF NOT EXISTS ta_core.ta_compliance_rules (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			rule_name VARCHAR(200) NOT NULL,
			rule_category VARCHAR(50) NOT NULL,
			jurisdiction VARCHAR(50) NOT NULL,
			rule_definition JSONB NOT NULL,
			violation_thresholds JSONB DEFAULT '{}'::jsonb,
			enforcement_actions JSONB DEFAULT '[]'::jsonb,
			is_active BOOLEAN DEFAULT true,
			effective_date DATE NOT NULL,
			expiration_date DATE,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL
		);
		
		-- Remote Workers table
		CREATE TABLE IF NOT EXISTS ta_core.ta_remote_workers (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			employee_id VARCHAR(36) NOT NULL,
			workspace_id VARCHAR(100) NOT NULL,
			work_mode VARCHAR(20) NOT NULL,
			home_office_setup JSONB DEFAULT '{}'::jsonb,
			timezone VARCHAR(50) DEFAULT 'UTC',
			preferred_work_hours JSONB DEFAULT '{}'::jsonb,
			current_activity VARCHAR(30) DEFAULT 'offline',
			productivity_metrics JSONB DEFAULT '[]'::jsonb,
			overall_productivity_score DECIMAL(3,2) DEFAULT 0.00,
			work_life_balance_score DECIMAL(3,2) DEFAULT 0.80,
			collaboration_platform_integrations JSONB DEFAULT '[]'::jsonb,
			environmental_data JSONB DEFAULT '{}'::jsonb,
			wellbeing_metrics JSONB DEFAULT '{}'::jsonb,
			burnout_risk_indicators JSONB DEFAULT '[]'::jsonb,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL,
			CONSTRAINT fk_ta_remote_workers_employee FOREIGN KEY (employee_id) REFERENCES ta_core.ta_employees(id)
		);
		
		-- AI Agents table
		CREATE TABLE IF NOT EXISTS ta_core.ta_ai_agents (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			agent_name VARCHAR(100) NOT NULL,
			agent_type VARCHAR(30) NOT NULL,
			agent_version VARCHAR(50) DEFAULT '1.0.0',
			capabilities JSONB NOT NULL,
			configuration JSONB DEFAULT '{}'::jsonb,
			deployment_environment VARCHAR(100) DEFAULT 'production',
			api_endpoints JSONB DEFAULT '{}'::jsonb,
			health_status VARCHAR(20) DEFAULT 'healthy',
			last_health_check TIMESTAMP WITH TIME ZONE,
			operational_cost_per_hour DECIMAL(10,4) DEFAULT 0.0000,
			total_operational_cost DECIMAL(12,2) DEFAULT 0.00,
			tasks_completed INTEGER DEFAULT 0,
			average_task_duration_seconds DECIMAL(10,2),
			cost_per_task DECIMAL(10,4),
			accuracy_score DECIMAL(3,2) DEFAULT 1.00,
			error_rate DECIMAL(3,2) DEFAULT 0.00,
			overall_performance_score DECIMAL(3,2) DEFAULT 0.00,
			cost_efficiency_score DECIMAL(3,2),
			uptime_percentage DECIMAL(5,2) DEFAULT 99.90,
			cpu_hours DECIMAL(12,4) DEFAULT 0.0000,
			gpu_hours DECIMAL(12,4) DEFAULT 0.0000,
			memory_usage_gb_hours DECIMAL(12,4) DEFAULT 0.0000,
			api_calls_count INTEGER DEFAULT 0,
			storage_used_gb DECIMAL(10,2) DEFAULT 0.00,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL
		);
		
		-- Hybrid Collaboration table
		CREATE TABLE IF NOT EXISTS ta_core.ta_hybrid_collaboration (
			id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(36) NOT NULL,
			session_name VARCHAR(200) NOT NULL,
			project_id VARCHAR(36) NOT NULL,
			session_type VARCHAR(50) DEFAULT 'collaborative_work',
			human_participants JSONB NOT NULL,
			ai_participants JSONB NOT NULL,
			session_lead VARCHAR(36) NOT NULL,
			start_time TIMESTAMP WITH TIME ZONE NOT NULL,
			end_time TIMESTAMP WITH TIME ZONE,
			planned_duration_minutes INTEGER DEFAULT 60,
			actual_duration_minutes INTEGER,
			work_allocation JSONB DEFAULT '{}'::jsonb,
			collaboration_effectiveness DECIMAL(3,2),
			human_ai_interaction_quality DECIMAL(3,2),
			task_distribution JSONB DEFAULT '{}'::jsonb,
			communication_patterns JSONB DEFAULT '{}'::jsonb,
			efficiency_metrics JSONB DEFAULT '{}'::jsonb,
			outcomes JSONB DEFAULT '{}'::jsonb,
			lessons_learned JSONB DEFAULT '[]'::jsonb,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			created_by VARCHAR(36) NOT NULL
		);
		"""
		
		async with self.engines['async'].begin() as conn:
			# Split and execute schema creation statements
			statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
			for statement in statements:
				if statement:
					await conn.execute(text(statement))
		
		# Create indexes for performance
		await self._create_indexes()
	
	async def _create_indexes(self) -> None:
		"""Create performance indexes"""
		logger.info("Creating database indexes")
		
		index_sql = """
		-- Employee indexes
		CREATE INDEX IF NOT EXISTS idx_ta_employees_tenant_id ON ta_core.ta_employees(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_employees_employee_number ON ta_core.ta_employees(employee_number);
		CREATE INDEX IF NOT EXISTS idx_ta_employees_email ON ta_core.ta_employees(email);
		CREATE INDEX IF NOT EXISTS idx_ta_employees_department_id ON ta_core.ta_employees(department_id);
		CREATE INDEX IF NOT EXISTS idx_ta_employees_status ON ta_core.ta_employees(employment_status);
		CREATE INDEX IF NOT EXISTS idx_ta_employees_workforce_type ON ta_core.ta_employees(workforce_type);
		
		-- Time entry indexes
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_tenant_id ON ta_core.ta_time_entries(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_employee_id ON ta_core.ta_time_entries(employee_id);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_entry_date ON ta_core.ta_time_entries(entry_date);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_status ON ta_core.ta_time_entries(status);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_approval_status ON ta_core.ta_time_entries(approval_status);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_employee_date ON ta_core.ta_time_entries(employee_id, entry_date);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_clock_in ON ta_core.ta_time_entries(clock_in);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_anomaly_score ON ta_core.ta_time_entries(anomaly_score);
		
		-- Schedule indexes
		CREATE INDEX IF NOT EXISTS idx_ta_schedules_tenant_id ON ta_core.ta_schedules(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_schedules_employee_id ON ta_core.ta_schedules(employee_id);
		CREATE INDEX IF NOT EXISTS idx_ta_schedules_effective_date ON ta_core.ta_schedules(effective_date);
		CREATE INDEX IF NOT EXISTS idx_ta_schedules_is_active ON ta_core.ta_schedules(is_active);
		
		-- Leave request indexes
		CREATE INDEX IF NOT EXISTS idx_ta_leave_requests_tenant_id ON ta_core.ta_leave_requests(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_leave_requests_employee_id ON ta_core.ta_leave_requests(employee_id);
		CREATE INDEX IF NOT EXISTS idx_ta_leave_requests_start_date ON ta_core.ta_leave_requests(start_date);
		CREATE INDEX IF NOT EXISTS idx_ta_leave_requests_status ON ta_core.ta_leave_requests(status);
		CREATE INDEX IF NOT EXISTS idx_ta_leave_requests_leave_type ON ta_core.ta_leave_requests(leave_type);
		
		-- Fraud detection indexes
		CREATE INDEX IF NOT EXISTS idx_ta_fraud_detection_tenant_id ON ta_core.ta_fraud_detection(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_fraud_detection_time_entry_id ON ta_core.ta_fraud_detection(time_entry_id);
		CREATE INDEX IF NOT EXISTS idx_ta_fraud_detection_fraud_type ON ta_core.ta_fraud_detection(fraud_type);
		CREATE INDEX IF NOT EXISTS idx_ta_fraud_detection_severity ON ta_core.ta_fraud_detection(severity);
		CREATE INDEX IF NOT EXISTS idx_ta_fraud_detection_confidence ON ta_core.ta_fraud_detection(confidence_score);
		
		-- Biometric authentication indexes
		CREATE INDEX IF NOT EXISTS idx_ta_biometric_auth_tenant_id ON ta_core.ta_biometric_authentication(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_biometric_auth_employee_id ON ta_core.ta_biometric_authentication(employee_id);
		CREATE INDEX IF NOT EXISTS idx_ta_biometric_auth_type ON ta_core.ta_biometric_authentication(biometric_type);
		CREATE INDEX IF NOT EXISTS idx_ta_biometric_auth_is_active ON ta_core.ta_biometric_authentication(is_active);
		
		-- Predictive analytics indexes
		CREATE INDEX IF NOT EXISTS idx_ta_predictive_analytics_tenant_id ON ta_core.ta_predictive_analytics(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_predictive_analytics_type ON ta_core.ta_predictive_analytics(analysis_type);
		CREATE INDEX IF NOT EXISTS idx_ta_predictive_analytics_confidence ON ta_core.ta_predictive_analytics(model_confidence);
		
		-- Compliance rules indexes
		CREATE INDEX IF NOT EXISTS idx_ta_compliance_rules_tenant_id ON ta_core.ta_compliance_rules(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_compliance_rules_category ON ta_core.ta_compliance_rules(rule_category);
		CREATE INDEX IF NOT EXISTS idx_ta_compliance_rules_jurisdiction ON ta_core.ta_compliance_rules(jurisdiction);
		CREATE INDEX IF NOT EXISTS idx_ta_compliance_rules_is_active ON ta_core.ta_compliance_rules(is_active);
		
		-- Remote workers indexes
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_tenant_id ON ta_core.ta_remote_workers(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_employee_id ON ta_core.ta_remote_workers(employee_id);
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_work_mode ON ta_core.ta_remote_workers(work_mode);
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_current_activity ON ta_core.ta_remote_workers(current_activity);
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_productivity_score ON ta_core.ta_remote_workers(overall_productivity_score);
		
		-- AI agents indexes
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_tenant_id ON ta_core.ta_ai_agents(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_agent_type ON ta_core.ta_ai_agents(agent_type);
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_health_status ON ta_core.ta_ai_agents(health_status);
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_performance_score ON ta_core.ta_ai_agents(overall_performance_score);
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_cost_efficiency ON ta_core.ta_ai_agents(cost_efficiency_score);
		
		-- Hybrid collaboration indexes
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_tenant_id ON ta_core.ta_hybrid_collaboration(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_project_id ON ta_core.ta_hybrid_collaboration(project_id);
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_session_type ON ta_core.ta_hybrid_collaboration(session_type);
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_start_time ON ta_core.ta_hybrid_collaboration(start_time);
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_effectiveness ON ta_core.ta_hybrid_collaboration(collaboration_effectiveness);
		
		-- JSONB GIN indexes for efficient JSON queries
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_device_info_gin ON ta_core.ta_time_entries USING gin(device_info);
		CREATE INDEX IF NOT EXISTS idx_ta_time_entries_fraud_indicators_gin ON ta_core.ta_time_entries USING gin(fraud_indicators);
		CREATE INDEX IF NOT EXISTS idx_ta_remote_workers_productivity_metrics_gin ON ta_core.ta_remote_workers USING gin(productivity_metrics);
		CREATE INDEX IF NOT EXISTS idx_ta_ai_agents_capabilities_gin ON ta_core.ta_ai_agents USING gin(capabilities);
		CREATE INDEX IF NOT EXISTS idx_ta_hybrid_collaboration_work_allocation_gin ON ta_core.ta_hybrid_collaboration USING gin(work_allocation);
		"""
		
		async with self.engines['async'].begin() as conn:
			statements = [stmt.strip() for stmt in index_sql.split(';') if stmt.strip()]
			for statement in statements:
				if statement:
					await conn.execute(text(statement))
	
	async def _setup_monitoring(self) -> None:
		"""Setup database monitoring and performance tracking"""
		logger.info("Setting up database monitoring")
		
		monitoring_sql = """
		-- Performance monitoring views
		CREATE OR REPLACE VIEW ta_core.v_time_entry_performance AS
		SELECT 
			DATE_TRUNC('hour', created_at) as hour_bucket,
			COUNT(*) as entries_per_hour,
			AVG(total_hours) as avg_hours,
			AVG(anomaly_score) as avg_anomaly_score,
			COUNT(CASE WHEN requires_approval THEN 1 END) as requiring_approval
		FROM ta_core.ta_time_entries
		WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
		GROUP BY DATE_TRUNC('hour', created_at)
		ORDER BY hour_bucket DESC;
		
		-- Fraud detection summary
		CREATE OR REPLACE VIEW ta_core.v_fraud_detection_summary AS
		SELECT 
			DATE_TRUNC('day', created_at) as detection_date,
			fraud_type,
			severity,
			COUNT(*) as detection_count,
			AVG(confidence_score) as avg_confidence
		FROM ta_core.ta_fraud_detection
		WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
		GROUP BY DATE_TRUNC('day', created_at), fraud_type, severity
		ORDER BY detection_date DESC, detection_count DESC;
		
		-- Remote worker productivity summary
		CREATE OR REPLACE VIEW ta_core.v_remote_worker_productivity AS
		SELECT 
			rw.work_mode,
			COUNT(*) as worker_count,
			AVG(rw.overall_productivity_score) as avg_productivity,
			AVG(rw.work_life_balance_score) as avg_work_life_balance,
			COUNT(CASE WHEN jsonb_array_length(rw.burnout_risk_indicators) > 0 THEN 1 END) as burnout_risk_count
		FROM ta_core.ta_remote_workers rw
		WHERE rw.current_activity != 'offline'
		GROUP BY rw.work_mode;
		
		-- AI agent performance summary
		CREATE OR REPLACE VIEW ta_core.v_ai_agent_performance AS
		SELECT 
			agent_type,
			COUNT(*) as agent_count,
			AVG(overall_performance_score) as avg_performance,
			AVG(cost_efficiency_score) as avg_cost_efficiency,
			SUM(total_operational_cost) as total_cost,
			SUM(tasks_completed) as total_tasks_completed
		FROM ta_core.ta_ai_agents
		WHERE health_status = 'healthy'
		GROUP BY agent_type;
		"""
		
		async with self.engines['async'].begin() as conn:
			statements = [stmt.strip() for stmt in monitoring_sql.split(';') if stmt.strip()]
			for statement in statements:
				if statement:
					await conn.execute(text(statement))
	
	async def create_tenant_schema(self, tenant_id: str, tenant_name: str) -> str:
		"""Create isolated schema for a new tenant"""
		schema_name = f"ta_tenant_{tenant_id.replace('-', '_')}"
		
		logger.info(f"Creating tenant schema: {schema_name} for tenant: {tenant_name}")
		
		async with self.engines['async'].begin() as conn:
			# Create schema
			await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
			
			# Register tenant
			await conn.execute(text("""
				INSERT INTO ta_tenant_management.tenants (id, name, schema_name)
				VALUES (:tenant_id, :tenant_name, :schema_name)
				ON CONFLICT (id) DO UPDATE SET
					name = EXCLUDED.name,
					schema_name = EXCLUDED.schema_name,
					updated_at = CURRENT_TIMESTAMP
			"""), {
				"tenant_id": tenant_id,
				"tenant_name": tenant_name,
				"schema_name": schema_name
			})
		
		return schema_name
	
	async def get_session(self, tenant_id: str = None) -> AsyncSession:
		"""Get database session, optionally scoped to tenant"""
		session = self.sessions['async']()
		
		if tenant_id:
			# Set search path to tenant schema
			schema_name = await self.get_tenant_schema(tenant_id)
			if schema_name:
				await session.execute(text(f"SET search_path TO {schema_name}, ta_core, public"))
		
		return session
	
	async def get_tenant_schema(self, tenant_id: str) -> Optional[str]:
		"""Get schema name for a tenant"""
		async with self.engines['async'].begin() as conn:
			result = await conn.execute(text("""
				SELECT schema_name FROM ta_tenant_management.tenants 
				WHERE id = :tenant_id AND status = 'active'
			"""), {"tenant_id": tenant_id})
			
			row = result.fetchone()
			return row[0] if row else None
	
	async def cleanup(self) -> None:
		"""Cleanup database connections"""
		logger.info("Cleaning up database connections")
		
		for engine in self.engines.values():
			if hasattr(engine, 'dispose'):
				await engine.dispose()
		
		self.engines.clear()
		self.sessions.clear()


# Global database manager instance
db_manager = DatabaseManager()

async def get_db_session(tenant_id: str = None) -> AsyncSession:
	"""Get database session for dependency injection"""
	return await db_manager.get_session(tenant_id)

async def init_database():
	"""Initialize database for the application"""
	await db_manager.initialize()

async def cleanup_database():
	"""Cleanup database connections"""
	await db_manager.cleanup()

# Export public interface
__all__ = [
	"DatabaseManager",
	"db_manager", 
	"get_db_session",
	"init_database",
	"cleanup_database",
	"Base"
]