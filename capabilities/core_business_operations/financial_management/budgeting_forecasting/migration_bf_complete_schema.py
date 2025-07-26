"""
APG Budgeting & Forecasting - Complete Database Migration

Creates the complete multi-tenant database schema for APG Budgeting & Forecasting capability.
Implements PostgreSQL with schema-based tenant isolation and row-level security.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>

Migration: bf_001_complete_schema
Revision ID: bf001_complete_2025
Revises: None
Create Date: 2025-01-26 15:30:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateSequence, DropSequence
from datetime import datetime


# Revision identifiers
revision = 'bf001_complete_2025'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
	"""
	Create the complete APG Budgeting & Forecasting schema.
	
	Implements:
	- Multi-tenant schema structure
	- Row-level security policies
	- Comprehensive indexing strategy
	- APG integration tables
	- Complete data model
	"""
	
	# =============================================================================
	# 1. Create Shared Schema and Common Tables
	# =============================================================================
	
	# Create the shared schema for common resources
	op.execute('CREATE SCHEMA IF NOT EXISTS bf_shared')
	
	# Create tenant configuration table
	op.create_table(
		'tenant_config',
		sa.Column('tenant_id', sa.String(36), primary_key=True),
		sa.Column('tenant_name', sa.String(255), nullable=False),
		sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		sa.Column('updated_at', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		sa.Column('is_active', sa.Boolean, default=True),
		
		# Feature configuration
		sa.Column('features_enabled', postgresql.JSONB, default=lambda: []),
		sa.Column('budget_features', postgresql.JSONB, default=lambda: {}),
		sa.Column('forecast_features', postgresql.JSONB, default=lambda: {}),
		sa.Column('analytics_features', postgresql.JSONB, default=lambda: {}),
		
		# Integration configuration
		sa.Column('apg_integrations', postgresql.JSONB, default=lambda: {}),
		sa.Column('external_systems', postgresql.JSONB, default=lambda: {}),
		
		# Business configuration
		sa.Column('fiscal_year_start', sa.Date, default='2025-01-01'),
		sa.Column('default_currency', sa.String(3), default='USD'),
		sa.Column('time_zone', sa.String(50), default='UTC'),
		
		# Organization structure
		sa.Column('department_hierarchy', postgresql.JSONB, default=lambda: {}),
		sa.Column('cost_centers', postgresql.JSONB, default=lambda: []),
		sa.Column('approval_workflows', postgresql.JSONB, default=lambda: {}),
		
		# Security and compliance
		sa.Column('data_retention_days', sa.Integer, default=2555),  # 7 years
		sa.Column('encryption_enabled', sa.Boolean, default=True),
		sa.Column('audit_level', sa.String(20), default='detailed'),
		
		# Performance configuration
		sa.Column('max_budget_lines', sa.Integer, default=100000),
		sa.Column('forecast_horizon_limit', sa.Integer, default=60),  # months
		sa.Column('concurrent_users_limit', sa.Integer, default=100),
		
		# Billing and limits
		sa.Column('subscription_tier', sa.String(20), default='standard'),
		sa.Column('usage_limits', postgresql.JSONB, default=lambda: {}),
		sa.Column('billing_contact', sa.String(255), nullable=True),
		
		schema='bf_shared'
	)
	
	# Create budget templates table
	op.create_table(
		'budget_templates',
		sa.Column('template_id', sa.String(36), primary_key=True),
		sa.Column('owner_tenant_id', sa.String(36), nullable=False),
		sa.Column('template_name', sa.String(255), nullable=False),
		sa.Column('template_description', sa.Text, nullable=True),
		sa.Column('template_category', sa.String(100), nullable=True),
		sa.Column('is_public', sa.Boolean, default=False),
		sa.Column('is_system', sa.Boolean, default=False),
		sa.Column('usage_count', sa.Integer, default=0),
		sa.Column('shared_with_tenants', postgresql.ARRAY(sa.String(36)), default=lambda: []),
		sa.Column('template_data', postgresql.JSONB, nullable=False),
		sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		sa.Column('updated_at', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		sa.Column('created_by', sa.String(36), nullable=False),
		sa.Column('is_deleted', sa.Boolean, default=False),
		
		schema='bf_shared'
	)
	
	# Create account categories reference table
	op.create_table(
		'account_categories',
		sa.Column('category_id', sa.String(36), primary_key=True),
		sa.Column('category_code', sa.String(20), nullable=False),
		sa.Column('category_name', sa.String(100), nullable=False),
		sa.Column('category_type', sa.String(20), nullable=False),  # revenue, expense, asset, liability
		sa.Column('parent_category_id', sa.String(36), nullable=True),
		sa.Column('gl_account_range', sa.String(50), nullable=True),
		sa.Column('is_system', sa.Boolean, default=True),
		sa.Column('sort_order', sa.Integer, default=0),
		
		schema='bf_shared'
	)
	
	# Create currency rates table
	op.create_table(
		'currency_rates',
		sa.Column('rate_id', sa.String(36), primary_key=True),
		sa.Column('from_currency', sa.String(3), nullable=False),
		sa.Column('to_currency', sa.String(3), nullable=False),
		sa.Column('rate_date', sa.Date, nullable=False),
		sa.Column('exchange_rate', sa.DECIMAL(12, 6), nullable=False),
		sa.Column('rate_source', sa.String(50), default='manual'),
		sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		
		schema='bf_shared'
	)
	
	# Create industry benchmarks table
	op.create_table(
		'industry_benchmarks',
		sa.Column('benchmark_id', sa.String(36), primary_key=True),
		sa.Column('industry_code', sa.String(20), nullable=False),
		sa.Column('metric_name', sa.String(100), nullable=False),
		sa.Column('metric_category', sa.String(50), nullable=False),
		sa.Column('percentile_25', sa.DECIMAL(15, 4), nullable=True),
		sa.Column('percentile_50', sa.DECIMAL(15, 4), nullable=True),
		sa.Column('percentile_75', sa.DECIMAL(15, 4), nullable=True),
		sa.Column('percentile_90', sa.DECIMAL(15, 4), nullable=True),
		sa.Column('participant_count', sa.Integer, default=0),
		sa.Column('anonymized', sa.Boolean, default=True),
		sa.Column('last_updated', sa.TIMESTAMP(timezone=True), default=datetime.utcnow),
		
		schema='bf_shared'
	)
	
	# =============================================================================
	# 2. Create Tenant Management Functions
	# =============================================================================
	
	# Function to create a new tenant schema
	tenant_setup_function = """
	CREATE OR REPLACE FUNCTION bf_shared.setup_tenant(
		p_tenant_id VARCHAR(36),
		p_tenant_name VARCHAR(255),
		p_created_by VARCHAR(36)
	) RETURNS BOOLEAN AS $$
	DECLARE
		schema_name VARCHAR(50);
		sql_stmt TEXT;
	BEGIN
		-- Create schema name
		schema_name := 'bf_' || p_tenant_id;
		
		-- Create the tenant schema
		sql_stmt := 'CREATE SCHEMA IF NOT EXISTS ' || quote_ident(schema_name);
		EXECUTE sql_stmt;
		
		-- Set search path to the new schema
		sql_stmt := 'SET search_path = ' || quote_ident(schema_name) || ', bf_shared, public';
		EXECUTE sql_stmt;
		
		-- Create budgets table
		sql_stmt := format('
			CREATE TABLE %I.budgets (
				id VARCHAR(36) PRIMARY KEY,
				tenant_id VARCHAR(36) NOT NULL,
				budget_name VARCHAR(255) NOT NULL,
				budget_code VARCHAR(50),
				budget_type VARCHAR(20) NOT NULL,
				fiscal_year INTEGER NOT NULL,
				budget_period_start DATE NOT NULL,
				budget_period_end DATE NOT NULL,
				
				-- Status and workflow
				status VARCHAR(20) DEFAULT ''draft'',
				workflow_state VARCHAR(100),
				approval_level INTEGER DEFAULT 0,
				requires_approval BOOLEAN DEFAULT TRUE,
				
				-- Financial configuration
				base_currency VARCHAR(3) DEFAULT ''USD'',
				budget_method VARCHAR(50) DEFAULT ''zero_based'',
				planning_horizon_months INTEGER DEFAULT 12,
				
				-- Template and inheritance
				template_id VARCHAR(36),
				parent_budget_id VARCHAR(36),
				is_template BOOLEAN DEFAULT FALSE,
				template_usage_count INTEGER DEFAULT 0,
				
				-- Organizational hierarchy
				department_code VARCHAR(50),
				cost_center_code VARCHAR(50),
				business_unit VARCHAR(100),
				region_code VARCHAR(50),
				
				-- Performance tracking
				total_budget_amount DECIMAL(15, 2) DEFAULT 0.00,
				total_committed_amount DECIMAL(15, 2) DEFAULT 0.00,
				total_actual_amount DECIMAL(15, 2) DEFAULT 0.00,
				variance_amount DECIMAL(15, 2) DEFAULT 0.00,
				variance_percent DECIMAL(10, 4) DEFAULT 0.0000,
				
				-- AI/ML insights
				ai_confidence_score DECIMAL(5, 3) DEFAULT 0.000,
				risk_assessment_score DECIMAL(5, 3) DEFAULT 0.000,
				ai_recommendations JSONB DEFAULT ''[]'',
				forecast_accuracy_score DECIMAL(5, 3),
				
				-- Collaboration and communication
				collaboration_enabled BOOLEAN DEFAULT TRUE,
				notification_settings JSONB DEFAULT ''{}'',
				last_activity_date DATE,
				active_contributors JSONB DEFAULT ''[]'',
				
				-- APG Integration fields
				document_folder_id VARCHAR(36),
				workflow_instance_id VARCHAR(36),
				ai_job_id VARCHAR(36),
				
				-- Audit fields
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				created_by VARCHAR(36) NOT NULL,
				updated_by VARCHAR(36) NOT NULL,
				version INTEGER DEFAULT 1,
				is_deleted BOOLEAN DEFAULT FALSE,
				deleted_at TIMESTAMP WITH TIME ZONE,
				deleted_by VARCHAR(36),
				
				CONSTRAINT chk_budget_period CHECK (budget_period_end > budget_period_start),
				CONSTRAINT chk_fiscal_year CHECK (fiscal_year BETWEEN 2015 AND 2035),
				CONSTRAINT chk_planning_horizon CHECK (planning_horizon_months BETWEEN 1 AND 60)
			)
		', schema_name);
		EXECUTE sql_stmt;
		
		-- Create budget_lines table
		sql_stmt := format('
			CREATE TABLE %I.budget_lines (
				id VARCHAR(36) PRIMARY KEY,
				budget_id VARCHAR(36) NOT NULL,
				line_number INTEGER NOT NULL,
				line_description VARCHAR(500) NOT NULL,
				line_category VARCHAR(100) NOT NULL,
				line_type VARCHAR(20) NOT NULL,
				
				-- Account mapping
				account_code VARCHAR(50) NOT NULL,
				account_category_id VARCHAR(36),
				gl_account VARCHAR(50),
				
				-- Organizational allocation
				department_code VARCHAR(50),
				cost_center_code VARCHAR(50),
				project_code VARCHAR(50),
				activity_code VARCHAR(50),
				location_code VARCHAR(50),
				
				-- Time period allocation
				period_start DATE NOT NULL,
				period_end DATE NOT NULL,
				allocation_method VARCHAR(50) DEFAULT ''equal'',
				
				-- Budget amounts
				budgeted_amount DECIMAL(15, 2) NOT NULL,
				forecasted_amount DECIMAL(15, 2),
				committed_amount DECIMAL(15, 2) DEFAULT 0.00,
				actual_amount DECIMAL(15, 2) DEFAULT 0.00,
				variance_amount DECIMAL(15, 2) DEFAULT 0.00,
				variance_percent DECIMAL(10, 4) DEFAULT 0.0000,
				
				-- Monthly breakdown
				month_01_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_02_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_03_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_04_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_05_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_06_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_07_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_08_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_09_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_10_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_11_amount DECIMAL(15, 2) DEFAULT 0.00,
				month_12_amount DECIMAL(15, 2) DEFAULT 0.00,
				
				-- Quarterly breakdown
				q1_amount DECIMAL(15, 2) DEFAULT 0.00,
				q2_amount DECIMAL(15, 2) DEFAULT 0.00,
				q3_amount DECIMAL(15, 2) DEFAULT 0.00,
				q4_amount DECIMAL(15, 2) DEFAULT 0.00,
				
				-- Currency and exchange
				currency_code VARCHAR(3) DEFAULT ''USD'',
				exchange_rate DECIMAL(12, 6) DEFAULT 1.000000,
				base_currency_amount DECIMAL(15, 2),
				
				-- Driver-based budgeting
				quantity_driver VARCHAR(100),
				unit_quantity DECIMAL(15, 4),
				unit_price DECIMAL(15, 4),
				price_escalation_percent DECIMAL(8, 4) DEFAULT 0.0000,
				
				-- Approval and workflow
				approval_status VARCHAR(20) DEFAULT ''pending'',
				approval_level INTEGER DEFAULT 0,
				approved_by VARCHAR(36),
				approved_date DATE,
				rejection_reason TEXT,
				
				-- AI/ML insights
				ai_confidence_score DECIMAL(5, 3) DEFAULT 0.000,
				seasonality_factor DECIMAL(8, 4) DEFAULT 1.0000,
				trend_factor DECIMAL(8, 4) DEFAULT 1.0000,
				ai_adjustments JSONB DEFAULT ''{}'',
				
				-- Comments and notes
				line_notes TEXT,
				business_justification TEXT,
				assumptions TEXT,
				
				FOREIGN KEY (budget_id) REFERENCES %I.budgets(id),
				CONSTRAINT chk_budget_line_period CHECK (period_end > period_start),
				CONSTRAINT chk_exchange_rate CHECK (exchange_rate > 0)
			)
		', schema_name, schema_name);
		EXECUTE sql_stmt;
		
		-- Create forecasts table
		sql_stmt := format('
			CREATE TABLE %I.forecasts (
				id VARCHAR(36) PRIMARY KEY,
				tenant_id VARCHAR(36) NOT NULL,
				forecast_name VARCHAR(255) NOT NULL,
				forecast_code VARCHAR(50),
				forecast_type VARCHAR(20) NOT NULL,
				forecast_method VARCHAR(20) NOT NULL,
				
				-- Time horizon and frequency
				forecast_horizon_months INTEGER NOT NULL,
				forecast_frequency VARCHAR(20) DEFAULT ''monthly'',
				base_period_start DATE NOT NULL,
				base_period_end DATE NOT NULL,
				forecast_period_start DATE NOT NULL,
				forecast_period_end DATE NOT NULL,
				
				-- Model and algorithm configuration
				algorithm_type VARCHAR(50),
				model_version VARCHAR(50),
				model_parameters JSONB DEFAULT ''{}'',
				feature_selection JSONB DEFAULT ''{}'',
				
				-- Data sources and inputs
				data_sources JSONB DEFAULT ''[]'',
				input_variables JSONB DEFAULT ''[]'',
				external_factors JSONB DEFAULT ''[]'',
				historical_months_used INTEGER DEFAULT 24,
				
				-- Accuracy and confidence
				accuracy_score DECIMAL(5, 3),
				confidence_level DECIMAL(5, 3) DEFAULT 0.950,
				confidence_interval_lower DECIMAL(15, 2),
				confidence_interval_upper DECIMAL(15, 2),
				mae_score DECIMAL(15, 4),
				mape_score DECIMAL(15, 4),
				rmse_score DECIMAL(15, 4),
				
				-- Scenario analysis
				scenario_type VARCHAR(20) DEFAULT ''base'',
				probability_weight DECIMAL(5, 3) DEFAULT 1.000,
				scenario_assumptions TEXT,
				sensitivity_analysis JSONB DEFAULT ''{}'',
				
				-- Business context
				department_code VARCHAR(50),
				business_unit VARCHAR(100),
				product_category VARCHAR(100),
				market_segment VARCHAR(100),
				geographic_region VARCHAR(100),
				
				-- Status and lifecycle
				status VARCHAR(20) DEFAULT ''draft'',
				generation_status VARCHAR(50),
				last_generation_date TIMESTAMP WITH TIME ZONE,
				next_generation_date TIMESTAMP WITH TIME ZONE,
				auto_generation_enabled BOOLEAN DEFAULT FALSE,
				
				-- Performance tracking
				forecast_value DECIMAL(15, 2),
				actual_value DECIMAL(15, 2),
				variance_amount DECIMAL(15, 2),
				variance_percent DECIMAL(10, 4),
				accuracy_trend VARCHAR(20),
				
				-- APG Integration fields
				ai_job_id VARCHAR(36),
				time_series_job_id VARCHAR(36),
				federated_learning_session_id VARCHAR(36),
				
				-- Approval and review
				reviewed_by VARCHAR(36),
				review_date DATE,
				review_notes TEXT,
				approved_for_planning BOOLEAN DEFAULT FALSE,
				
				-- Audit fields
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				created_by VARCHAR(36) NOT NULL,
				updated_by VARCHAR(36) NOT NULL,
				version INTEGER DEFAULT 1,
				is_deleted BOOLEAN DEFAULT FALSE,
				deleted_at TIMESTAMP WITH TIME ZONE,
				deleted_by VARCHAR(36),
				
				CONSTRAINT chk_forecast_horizon CHECK (forecast_horizon_months BETWEEN 1 AND 60),
				CONSTRAINT chk_forecast_periods CHECK (forecast_period_start > base_period_end),
				CONSTRAINT chk_historical_months CHECK (historical_months_used BETWEEN 3 AND 120)
			)
		', schema_name);
		EXECUTE sql_stmt;
		
		-- Create forecast_data_points table
		sql_stmt := format('
			CREATE TABLE %I.forecast_data_points (
				id VARCHAR(36) PRIMARY KEY,
				forecast_id VARCHAR(36) NOT NULL,
				period_date DATE NOT NULL,
				period_type VARCHAR(20) NOT NULL,
				period_sequence INTEGER NOT NULL,
				fiscal_year INTEGER,
				fiscal_quarter INTEGER,
				fiscal_month INTEGER,
				
				-- Forecast values
				forecasted_value DECIMAL(15, 2) NOT NULL,
				confidence_lower DECIMAL(15, 2),
				confidence_upper DECIMAL(15, 2),
				actual_value DECIMAL(15, 2),
				variance_amount DECIMAL(15, 2),
				variance_percent DECIMAL(10, 4),
				
				-- Decomposition components
				trend_component DECIMAL(15, 2),
				seasonal_component DECIMAL(15, 2),
				cyclical_component DECIMAL(15, 2),
				irregular_component DECIMAL(15, 2),
				external_factor_impact DECIMAL(15, 2),
				
				-- Model insights
				prediction_strength DECIMAL(5, 3),
				volatility_score DECIMAL(5, 3),
				anomaly_score DECIMAL(5, 3),
				feature_importance JSONB DEFAULT ''{}'',
				
				-- Business drivers
				volume_driver DECIMAL(15, 4),
				price_driver DECIMAL(15, 4),
				mix_driver DECIMAL(15, 4),
				external_drivers JSONB DEFAULT ''{}'',
				
				-- Currency handling
				currency_code VARCHAR(3) DEFAULT ''USD'',
				exchange_rate DECIMAL(12, 6) DEFAULT 1.000000,
				base_currency_value DECIMAL(15, 2),
				
				-- Notes and assumptions
				period_notes TEXT,
				assumptions TEXT,
				risk_factors TEXT,
				
				FOREIGN KEY (forecast_id) REFERENCES %I.forecasts(id),
				CONSTRAINT chk_fiscal_quarter CHECK (fiscal_quarter BETWEEN 1 AND 4),
				CONSTRAINT chk_fiscal_month CHECK (fiscal_month BETWEEN 1 AND 12)
			)
		', schema_name, schema_name);
		EXECUTE sql_stmt;
		
		-- Create variance_analysis table
		sql_stmt := format('
			CREATE TABLE %I.variance_analysis (
				id VARCHAR(36) PRIMARY KEY,
				tenant_id VARCHAR(36) NOT NULL,
				analysis_name VARCHAR(255) NOT NULL,
				analysis_type VARCHAR(50) NOT NULL,
				analysis_period_start DATE NOT NULL,
				analysis_period_end DATE NOT NULL,
				comparison_period_start DATE,
				comparison_period_end DATE,
				
				-- Subject of analysis
				budget_id VARCHAR(36),
				forecast_id VARCHAR(36),
				department_code VARCHAR(50),
				account_category VARCHAR(100),
				analysis_scope VARCHAR(50) DEFAULT ''detailed'',
				
				-- Variance calculations
				baseline_amount DECIMAL(15, 2) NOT NULL,
				actual_amount DECIMAL(15, 2) NOT NULL,
				variance_amount DECIMAL(15, 2) NOT NULL,
				variance_percent DECIMAL(10, 4) NOT NULL,
				absolute_variance DECIMAL(15, 2) NOT NULL,
				
				-- Variance classification
				variance_type VARCHAR(20) NOT NULL,
				significance_level VARCHAR(20) NOT NULL,
				variance_threshold_exceeded BOOLEAN DEFAULT FALSE,
				requires_investigation BOOLEAN DEFAULT FALSE,
				
				-- Root cause analysis
				primary_cause VARCHAR(100),
				contributing_factors JSONB DEFAULT ''[]'',
				root_cause_category VARCHAR(50),
				impact_assessment TEXT,
				
				-- AI-powered insights
				ai_explanation TEXT,
				ai_confidence_score DECIMAL(5, 3),
				anomaly_detected BOOLEAN DEFAULT FALSE,
				pattern_analysis JSONB DEFAULT ''{}'',
				correlation_factors JSONB DEFAULT ''{}'',
				
				-- Corrective actions
				recommended_actions JSONB DEFAULT ''[]'',
				action_priority VARCHAR(20) DEFAULT ''medium'',
				estimated_impact DECIMAL(15, 2),
				action_timeline VARCHAR(50),
				responsible_party VARCHAR(100),
				
				-- Investigation tracking
				investigation_status VARCHAR(50) DEFAULT ''pending'',
				investigated_by VARCHAR(36),
				investigation_date DATE,
				investigation_notes TEXT,
				resolution_status VARCHAR(50) DEFAULT ''open'',
				
				-- Performance metrics
				analysis_accuracy DECIMAL(5, 3),
				prediction_quality DECIMAL(5, 3),
				time_to_detection_days INTEGER,
				resolution_time_days INTEGER,
				
				-- APG Integration fields
				ai_analysis_job_id VARCHAR(36),
				notification_sent BOOLEAN DEFAULT FALSE,
				workflow_triggered BOOLEAN DEFAULT FALSE,
				
				-- Audit fields
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				created_by VARCHAR(36) NOT NULL,
				updated_by VARCHAR(36) NOT NULL,
				version INTEGER DEFAULT 1,
				is_deleted BOOLEAN DEFAULT FALSE,
				deleted_at TIMESTAMP WITH TIME ZONE,
				deleted_by VARCHAR(36),
				
				FOREIGN KEY (budget_id) REFERENCES %I.budgets(id),
				FOREIGN KEY (forecast_id) REFERENCES %I.forecasts(id),
				CONSTRAINT chk_variance_analysis_period CHECK (analysis_period_end > analysis_period_start)
			)
		', schema_name, schema_name, schema_name);
		EXECUTE sql_stmt;
		
		-- Create scenarios table
		sql_stmt := format('
			CREATE TABLE %I.scenarios (
				id VARCHAR(36) PRIMARY KEY,
				tenant_id VARCHAR(36) NOT NULL,
				scenario_name VARCHAR(255) NOT NULL,
				scenario_description TEXT,
				scenario_type VARCHAR(20) NOT NULL,
				scenario_category VARCHAR(50),
				
				-- Scenario parameters
				probability_weight DECIMAL(5, 3) DEFAULT 0.333,
				time_horizon_months INTEGER NOT NULL,
				scenario_start_date DATE NOT NULL,
				scenario_end_date DATE NOT NULL,
				
				-- Base scenario reference
				base_budget_id VARCHAR(36),
				base_forecast_id VARCHAR(36),
				comparison_baseline VARCHAR(50) DEFAULT ''current_budget'',
				
				-- Scenario assumptions
				key_assumptions JSONB DEFAULT ''[]'',
				variable_changes JSONB DEFAULT ''{}'',
				external_factors JSONB DEFAULT ''{}'',
				market_conditions TEXT,
				
				-- Financial impact
				total_revenue_impact DECIMAL(15, 2) DEFAULT 0.00,
				total_expense_impact DECIMAL(15, 2) DEFAULT 0.00,
				net_income_impact DECIMAL(15, 2) DEFAULT 0.00,
				cash_flow_impact DECIMAL(15, 2) DEFAULT 0.00,
				
				-- Risk assessment
				risk_level VARCHAR(20) DEFAULT ''medium'',
				downside_risk DECIMAL(15, 2),
				upside_potential DECIMAL(15, 2),
				value_at_risk DECIMAL(15, 2),
				
				-- Sensitivity analysis
				sensitivity_variables JSONB DEFAULT ''[]'',
				elasticity_factors JSONB DEFAULT ''{}'',
				break_even_points JSONB DEFAULT ''{}'',
				
				-- Monte Carlo simulation
				simulation_enabled BOOLEAN DEFAULT FALSE,
				simulation_iterations INTEGER DEFAULT 1000,
				confidence_intervals JSONB DEFAULT ''{}'',
				distribution_parameters JSONB DEFAULT ''{}'',
				
				-- Decision support
				strategic_implications TEXT,
				recommended_decisions JSONB DEFAULT ''[]'',
				contingency_plans JSONB DEFAULT ''[]'',
				monitoring_indicators JSONB DEFAULT ''[]'',
				
				-- Modeling and calculation
				calculation_method VARCHAR(50) DEFAULT ''analytical'',
				model_complexity VARCHAR(20) DEFAULT ''medium'',
				last_calculation_date TIMESTAMP WITH TIME ZONE,
				calculation_duration_seconds INTEGER,
				
				-- Collaboration and review
				scenario_owner VARCHAR(36),
				review_participants JSONB DEFAULT ''[]'',
				last_review_date DATE,
				review_status VARCHAR(50) DEFAULT ''draft'',
				approval_required BOOLEAN DEFAULT FALSE,
				approved_by VARCHAR(36),
				approval_date DATE,
				
				-- APG Integration fields
				simulation_job_id VARCHAR(36),
				ai_modeling_job_id VARCHAR(36),
				document_folder_id VARCHAR(36),
				
				-- Audit fields
				created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
				created_by VARCHAR(36) NOT NULL,
				updated_by VARCHAR(36) NOT NULL,
				version INTEGER DEFAULT 1,
				is_deleted BOOLEAN DEFAULT FALSE,
				deleted_at TIMESTAMP WITH TIME ZONE,
				deleted_by VARCHAR(36),
				
				FOREIGN KEY (base_budget_id) REFERENCES %I.budgets(id),
				FOREIGN KEY (base_forecast_id) REFERENCES %I.forecasts(id),
				CONSTRAINT chk_scenario_horizon CHECK (time_horizon_months BETWEEN 1 AND 60),
				CONSTRAINT chk_scenario_period CHECK (scenario_end_date > scenario_start_date),
				CONSTRAINT chk_simulation_iterations CHECK (simulation_iterations BETWEEN 100 AND 100000)
			)
		', schema_name, schema_name, schema_name);
		EXECUTE sql_stmt;
		
		-- Insert tenant configuration
		INSERT INTO bf_shared.tenant_config (tenant_id, tenant_name)
		VALUES (p_tenant_id, p_tenant_name);
		
		RETURN TRUE;
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'Error creating tenant schema: %', SQLERRM;
			RETURN FALSE;
	END;
	$$ LANGUAGE plpgsql;
	"""
	
	op.execute(tenant_setup_function)
	
	# =============================================================================
	# 3. Create Indexes for Shared Tables
	# =============================================================================
	
	# Tenant config indexes
	op.create_index(
		'idx_tenant_config_active',
		'tenant_config',
		['is_active'],
		schema='bf_shared'
	)
	
	# Budget templates indexes
	op.create_index(
		'idx_budget_templates_owner',
		'budget_templates',
		['owner_tenant_id', 'is_deleted'],
		schema='bf_shared'
	)
	
	op.create_index(
		'idx_budget_templates_public',
		'budget_templates',
		['is_public', 'is_deleted'],
		schema='bf_shared'
	)
	
	# Account categories indexes
	op.create_index(
		'idx_account_categories_type',
		'account_categories',
		['category_type', 'parent_category_id'],
		schema='bf_shared'
	)
	
	# Currency rates indexes
	op.create_index(
		'idx_currency_rates_date',
		'currency_rates',
		['from_currency', 'to_currency', 'rate_date'],
		schema='bf_shared'
	)
	
	# Industry benchmarks indexes
	op.create_index(
		'idx_industry_benchmarks_industry',
		'industry_benchmarks',
		['industry_code', 'metric_category'],
		schema='bf_shared'
	)
	
	# =============================================================================
	# 4. Create Row-Level Security Policies Template Function
	# =============================================================================
	
	rls_function = """
	CREATE OR REPLACE FUNCTION bf_shared.create_tenant_rls_policies(p_tenant_id VARCHAR(36))
	RETURNS BOOLEAN AS $$
	DECLARE
		schema_name VARCHAR(50);
		sql_stmt TEXT;
		table_names TEXT[] := ARRAY['budgets', 'budget_lines', 'forecasts', 'forecast_data_points', 'variance_analysis', 'scenarios'];
		table_name TEXT;
	BEGIN
		schema_name := 'bf_' || p_tenant_id;
		
		-- Enable RLS on all tenant tables
		FOREACH table_name IN ARRAY table_names LOOP
			sql_stmt := format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', schema_name, table_name);
			EXECUTE sql_stmt;
			
			-- Create tenant isolation policy
			sql_stmt := format('
				CREATE POLICY tenant_isolation_%s ON %I.%I
				FOR ALL TO app_role
				USING (tenant_id = current_setting(''app.current_tenant''))
			', table_name, schema_name, table_name);
			EXECUTE sql_stmt;
			
			-- Create audit policy
			sql_stmt := format('
				CREATE POLICY audit_access_%s ON %I.%I
				FOR SELECT TO audit_role
				USING (true)
			', table_name, schema_name, table_name);
			EXECUTE sql_stmt;
		END LOOP;
		
		RETURN TRUE;
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'Error creating RLS policies: %', SQLERRM;
			RETURN FALSE;
	END;
	$$ LANGUAGE plpgsql;
	"""
	
	op.execute(rls_function)
	
	# =============================================================================
	# 5. Create Tenant Index Template Function
	# =============================================================================
	
	index_function = """
	CREATE OR REPLACE FUNCTION bf_shared.create_tenant_indexes(p_tenant_id VARCHAR(36))
	RETURNS BOOLEAN AS $$
	DECLARE
		schema_name VARCHAR(50);
		sql_stmt TEXT;
	BEGIN
		schema_name := 'bf_' || p_tenant_id;
		
		-- Budget indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budgets_tenant_fiscal ON %I.budgets(tenant_id, fiscal_year)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budgets_status ON %I.budgets(status, is_deleted)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budgets_department ON %I.budgets(department_code, cost_center_code)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		-- Budget lines indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budget_lines_budget ON %I.budget_lines(budget_id, line_number)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budget_lines_account ON %I.budget_lines(account_code, account_category_id)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_budget_lines_period ON %I.budget_lines(period_start, period_end)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		-- Forecast indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_forecasts_tenant_type ON %I.forecasts(tenant_id, forecast_type, status)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_forecasts_method ON %I.forecasts(forecast_method, algorithm_type)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		-- Forecast data points indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_forecast_data_forecast ON %I.forecast_data_points(forecast_id, period_date)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_forecast_data_fiscal ON %I.forecast_data_points(fiscal_year, fiscal_quarter)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		-- Variance analysis indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_variance_period ON %I.variance_analysis(analysis_period_start, analysis_period_end)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_variance_significance ON %I.variance_analysis(significance_level, requires_investigation)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		-- Scenario indexes
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_scenarios_type ON %I.scenarios(scenario_type, scenario_category)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		sql_stmt := format('CREATE INDEX CONCURRENTLY idx_%s_scenarios_base ON %I.scenarios(base_budget_id, base_forecast_id)', p_tenant_id, schema_name);
		EXECUTE sql_stmt;
		
		RETURN TRUE;
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'Error creating tenant indexes: %', SQLERRM;
			RETURN FALSE;
	END;
	$$ LANGUAGE plpgsql;
	"""
	
	op.execute(index_function)
	
	# =============================================================================
	# 6. Create Default Data Insert Function
	# =============================================================================
	
	default_data_function = """
	CREATE OR REPLACE FUNCTION bf_shared.insert_default_data()
	RETURNS BOOLEAN AS $$
	BEGIN
		-- Insert default account categories
		INSERT INTO bf_shared.account_categories (category_id, category_code, category_name, category_type, sort_order) VALUES
		('cat_revenue_001', 'REV', 'Revenue', 'revenue', 1),
		('cat_revenue_002', 'REV_SALES', 'Sales Revenue', 'revenue', 2),
		('cat_revenue_003', 'REV_SERVICE', 'Service Revenue', 'revenue', 3),
		('cat_expense_001', 'EXP', 'Expenses', 'expense', 10),
		('cat_expense_002', 'EXP_SALARY', 'Salary Expenses', 'expense', 11),
		('cat_expense_003', 'EXP_TRAVEL', 'Travel Expenses', 'expense', 12),
		('cat_expense_004', 'EXP_MARKETING', 'Marketing Expenses', 'expense', 13),
		('cat_asset_001', 'ASSET', 'Assets', 'asset', 20),
		('cat_liability_001', 'LIABILITY', 'Liabilities', 'liability', 30);
		
		-- Insert default budget template
		INSERT INTO bf_shared.budget_templates (
			template_id, owner_tenant_id, template_name, template_description, 
			template_category, is_public, is_system, template_data, created_by
		) VALUES (
			'template_system_001', 'system', 'Standard Annual Budget Template', 
			'Comprehensive annual budget template with standard account categories',
			'Annual', TRUE, TRUE, 
			'{"categories": ["revenue", "expense"], "structure": "hierarchical", "periods": 12}',
			'system'
		);
		
		-- Insert major currency rates (placeholders - would be updated by external service)
		INSERT INTO bf_shared.currency_rates (rate_id, from_currency, to_currency, rate_date, exchange_rate, rate_source) VALUES
		('rate_usd_eur_001', 'USD', 'EUR', CURRENT_DATE, 0.85, 'system'),
		('rate_usd_gbp_001', 'USD', 'GBP', CURRENT_DATE, 0.75, 'system'),
		('rate_usd_jpy_001', 'USD', 'JPY', CURRENT_DATE, 110.0, 'system'),
		('rate_eur_usd_001', 'EUR', 'USD', CURRENT_DATE, 1.18, 'system'),
		('rate_gbp_usd_001', 'GBP', 'USD', CURRENT_DATE, 1.33, 'system');
		
		RETURN TRUE;
	EXCEPTION
		WHEN OTHERS THEN
			RAISE NOTICE 'Error inserting default data: %', SQLERRM;
			RETURN FALSE;
	END;
	$$ LANGUAGE plpgsql;
	"""
	
	op.execute(default_data_function)
	
	# Insert the default data
	op.execute('SELECT bf_shared.insert_default_data()')
	
	# =============================================================================
	# 7. Create Application Roles
	# =============================================================================
	
	# Create application roles for row-level security
	op.execute("CREATE ROLE IF NOT EXISTS app_role")
	op.execute("CREATE ROLE IF NOT EXISTS audit_role")
	
	# Grant permissions to roles
	op.execute("GRANT USAGE ON SCHEMA bf_shared TO app_role")
	op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bf_shared TO app_role")
	op.execute("GRANT USAGE ON SCHEMA bf_shared TO audit_role")
	op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA bf_shared TO audit_role")


def downgrade():
	"""
	Drop the complete APG Budgeting & Forecasting schema.
	
	WARNING: This will destroy all budgeting and forecasting data.
	"""
	
	# Drop all tenant-specific schemas (this would be generated dynamically in real implementation)
	# For now, we'll drop the common setup
	
	# Drop functions
	op.execute('DROP FUNCTION IF EXISTS bf_shared.setup_tenant(VARCHAR, VARCHAR, VARCHAR)')
	op.execute('DROP FUNCTION IF EXISTS bf_shared.create_tenant_rls_policies(VARCHAR)')
	op.execute('DROP FUNCTION IF EXISTS bf_shared.create_tenant_indexes(VARCHAR)')
	op.execute('DROP FUNCTION IF EXISTS bf_shared.insert_default_data()')
	
	# Drop roles
	op.execute('DROP ROLE IF EXISTS app_role')
	op.execute('DROP ROLE IF EXISTS audit_role')
	
	# Drop shared tables
	op.drop_table('industry_benchmarks', schema='bf_shared')
	op.drop_table('currency_rates', schema='bf_shared')
	op.drop_table('account_categories', schema='bf_shared')
	op.drop_table('budget_templates', schema='bf_shared')
	op.drop_table('tenant_config', schema='bf_shared')
	
	# Drop shared schema
	op.execute('DROP SCHEMA IF EXISTS bf_shared CASCADE')
	
	print("WARNING: APG Budgeting & Forecasting schema and all data have been destroyed!")