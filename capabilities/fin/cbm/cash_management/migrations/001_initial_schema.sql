-- ============================================================================
-- APG Cash Management - Initial Database Schema
-- 
-- PostgreSQL schema with APG multi-tenant partitioning patterns.
-- Optimized for enterprise-scale cash management operations.
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
-- ============================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============================================================================
-- APG Multi-Tenant Base Schema
-- ============================================================================

-- Create schema for cash management
CREATE SCHEMA IF NOT EXISTS cm_cash_management;

-- Set search path
SET search_path TO cm_cash_management, public;

-- ============================================================================
-- Core Entity Tables
-- ============================================================================

-- Banks table with APG multi-tenant pattern
CREATE TABLE cm_banks (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Bank identification
	bank_code VARCHAR(20) NOT NULL,
	bank_name VARCHAR(200) NOT NULL,
	swift_code VARCHAR(11) NOT NULL,
	
	-- Bank details
	country_code VARCHAR(3) NOT NULL,
	city VARCHAR(100) NOT NULL,
	address_line1 VARCHAR(255) NOT NULL,
	address_line2 VARCHAR(255),
	postal_code VARCHAR(20) NOT NULL,
	
	-- Contact information (JSONB for flexibility)
	contacts JSONB DEFAULT '[]'::jsonb,
	
	-- Relationship details
	status VARCHAR(20) DEFAULT 'active' NOT NULL,
	relationship_manager VARCHAR(100),
	credit_rating VARCHAR(10),
	
	-- API integration
	api_enabled BOOLEAN DEFAULT false,
	api_endpoint VARCHAR(500),
	api_credentials_encrypted TEXT,
	last_api_sync TIMESTAMPTZ,
	
	-- Fees and terms (JSONB for flexibility)
	standard_fees JSONB DEFAULT '{}'::jsonb,
	fee_structure TEXT,
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_banks_status CHECK (status IN ('active', 'inactive', 'under_review', 'restricted', 'terminated')),
	CONSTRAINT ck_cm_banks_swift_code_length CHECK (LENGTH(swift_code) IN (8, 11)),
	CONSTRAINT ck_cm_banks_country_code_length CHECK (LENGTH(country_code) IN (2, 3))
) PARTITION BY HASH (tenant_id);

-- Create partitions for banks (16 partitions for horizontal scaling)
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_banks_%s PARTITION OF cm_banks FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- Cash accounts table with APG multi-tenant pattern
CREATE TABLE cm_cash_accounts (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Account identification
	account_number VARCHAR(50) NOT NULL,
	account_name VARCHAR(200) NOT NULL,
	iban VARCHAR(34),
	
	-- Account details
	bank_id VARCHAR(26) NOT NULL,
	account_type VARCHAR(20) NOT NULL,
	currency_code VARCHAR(3) NOT NULL,
	
	-- Entity and organization
	entity_id VARCHAR(26) NOT NULL,
	cost_center VARCHAR(50),
	department VARCHAR(100),
	purpose VARCHAR(500),
	
	-- Balance information
	current_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	available_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	pending_credits DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	pending_debits DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	last_balance_update TIMESTAMPTZ,
	
	-- Account status and limits
	status VARCHAR(20) DEFAULT 'active' NOT NULL,
	overdraft_limit DECIMAL(18,2),
	minimum_balance DECIMAL(18,2),
	maximum_balance DECIMAL(18,2),
	
	-- Interest and fees (JSONB for flexibility)
	interest_rate DECIMAL(10,6),
	fee_schedule JSONB DEFAULT '{}'::jsonb,
	
	-- Automation settings
	auto_sweep_enabled BOOLEAN DEFAULT false,
	sweep_target_account VARCHAR(26),
	sweep_threshold DECIMAL(18,2),
	
	-- Reconciliation
	last_reconciled_date DATE,
	reconciliation_status VARCHAR(20) DEFAULT 'pending' NOT NULL,
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_cash_accounts_type CHECK (account_type IN ('checking', 'savings', 'money_market', 'certificate_deposit', 'investment', 'credit_line', 'petty_cash', 'escrow')),
	CONSTRAINT ck_cm_cash_accounts_status CHECK (status IN ('active', 'inactive', 'closed', 'frozen', 'restricted', 'pending_closure')),
	CONSTRAINT ck_cm_cash_accounts_currency_length CHECK (LENGTH(currency_code) = 3),
	CONSTRAINT ck_cm_cash_accounts_reconciliation_status CHECK (reconciliation_status IN ('pending', 'reconciled', 'variance'))
) PARTITION BY HASH (tenant_id);

-- Create partitions for cash accounts
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_cash_accounts_%s PARTITION OF cm_cash_accounts FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- ============================================================================
-- Time-Series Tables for Cash Flow History
-- ============================================================================

-- Cash positions table (time-series data)
CREATE TABLE cm_cash_positions (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Position identification
	position_date DATE NOT NULL,
	entity_id VARCHAR(26) NOT NULL,
	currency_code VARCHAR(3) NOT NULL,
	
	-- Balance aggregation
	total_cash DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	available_cash DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	restricted_cash DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	invested_cash DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	
	-- Account breakdown
	checking_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	savings_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	money_market_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	investment_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	
	-- Projected flows (next 30 days)
	projected_inflows DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	projected_outflows DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	net_projected_flow DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	
	-- Key ratios and metrics
	liquidity_ratio DECIMAL(8,4),
	concentration_risk DECIMAL(8,4),
	yield_rate DECIMAL(8,4),
	
	-- Risk indicators
	days_cash_on_hand INTEGER,
	stress_test_coverage DECIMAL(8,4),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_cash_positions_currency_length CHECK (LENGTH(currency_code) = 3)
) PARTITION BY RANGE (position_date);

-- Create monthly partitions for cash positions (current year + 2 years)
DO $$ 
DECLARE
	start_date DATE := DATE_TRUNC('year', CURRENT_DATE);
	end_date DATE;
	partition_name TEXT;
BEGIN
	FOR year_offset IN 0..2 LOOP
		FOR month_num IN 1..12 LOOP
			start_date := DATE_TRUNC('month', start_date + INTERVAL '1 year' * year_offset + INTERVAL '1 month' * (month_num - 1));
			end_date := start_date + INTERVAL '1 month';
			partition_name := format('cm_cash_positions_%s_%s', 
				EXTRACT(YEAR FROM start_date), 
				LPAD(EXTRACT(MONTH FROM start_date)::text, 2, '0')
			);
			
			EXECUTE format('CREATE TABLE %s PARTITION OF cm_cash_positions FOR VALUES FROM (''%s'') TO (''%s'')', 
				partition_name, start_date, end_date);
		END LOOP;
	END LOOP;
END $$;

-- Cash flows table (transaction-level time-series)
CREATE TABLE cm_cash_flows (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Transaction identification
	flow_date DATE NOT NULL,
	transaction_id VARCHAR(50),
	description VARCHAR(500) NOT NULL,
	
	-- Flow details
	account_id VARCHAR(26) NOT NULL,
	transaction_type VARCHAR(20) NOT NULL,
	amount DECIMAL(18,2) NOT NULL,
	currency_code VARCHAR(3) NOT NULL,
	
	-- Categorization
	category VARCHAR(100) NOT NULL,
	subcategory VARCHAR(100),
	business_unit VARCHAR(100),
	cost_center VARCHAR(50),
	
	-- Source integration
	source_module VARCHAR(50),
	source_document VARCHAR(100),
	counterparty VARCHAR(200),
	
	-- Forecasting attributes
	is_recurring BOOLEAN DEFAULT false,
	recurrence_pattern VARCHAR(50),
	forecast_confidence DECIMAL(4,3),
	
	-- Timing
	planned_date DATE,
	actual_date DATE,
	value_date DATE,
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_cash_flows_transaction_type CHECK (transaction_type IN ('deposit', 'withdrawal', 'transfer', 'investment', 'redemption', 'interest_earned', 'fees', 'fx_conversion', 'sweep')),
	CONSTRAINT ck_cm_cash_flows_currency_length CHECK (LENGTH(currency_code) = 3)
) PARTITION BY RANGE (flow_date);

-- Create monthly partitions for cash flows
DO $$ 
DECLARE
	start_date DATE := DATE_TRUNC('year', CURRENT_DATE);
	end_date DATE;
	partition_name TEXT;
BEGIN
	FOR year_offset IN 0..2 LOOP
		FOR month_num IN 1..12 LOOP
			start_date := DATE_TRUNC('month', start_date + INTERVAL '1 year' * year_offset + INTERVAL '1 month' * (month_num - 1));
			end_date := start_date + INTERVAL '1 month';
			partition_name := format('cm_cash_flows_%s_%s', 
				EXTRACT(YEAR FROM start_date), 
				LPAD(EXTRACT(MONTH FROM start_date)::text, 2, '0')
			);
			
			EXECUTE format('CREATE TABLE %s PARTITION OF cm_cash_flows FOR VALUES FROM (''%s'') TO (''%s'')', 
				partition_name, start_date, end_date);
		END LOOP;
	END LOOP;
END $$;

-- ============================================================================
-- Forecasting and Analytics Tables
-- ============================================================================

-- Cash forecasts table
CREATE TABLE cm_cash_forecasts (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Forecast identification
	forecast_id VARCHAR(26) NOT NULL DEFAULT uuid_generate_v7()::text,
	forecast_date DATE NOT NULL,
	forecast_type VARCHAR(20) NOT NULL,
	scenario VARCHAR(20) DEFAULT 'base_case' NOT NULL,
	
	-- Forecast scope
	entity_id VARCHAR(26) NOT NULL,
	currency_code VARCHAR(3) NOT NULL,
	horizon_days INTEGER NOT NULL CHECK (horizon_days >= 1 AND horizon_days <= 365),
	
	-- Opening position
	opening_balance DECIMAL(18,2) NOT NULL,
	opening_date DATE NOT NULL,
	
	-- Forecast components
	projected_inflows DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	projected_outflows DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	net_flow DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	closing_balance DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	
	-- Statistical measures
	confidence_level DECIMAL(4,3) NOT NULL CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
	confidence_interval_lower DECIMAL(18,2) NOT NULL,
	confidence_interval_upper DECIMAL(18,2) NOT NULL,
	standard_deviation DECIMAL(18,2),
	
	-- Model information
	model_used VARCHAR(100) NOT NULL,
	model_version VARCHAR(50) NOT NULL,
	training_data_period VARCHAR(100),
	feature_importance JSONB DEFAULT '{}'::jsonb,
	
	-- Forecast accuracy (for backtesting)
	actual_outcome DECIMAL(18,2),
	forecast_error DECIMAL(18,2),
	accuracy_percentage DECIMAL(8,4),
	
	-- Risk assessment
	shortfall_probability DECIMAL(4,3),
	stress_test_result DECIMAL(18,2),
	value_at_risk DECIMAL(18,2),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_cash_forecasts_type CHECK (forecast_type IN ('daily', 'weekly', 'monthly', 'quarterly', 'rolling', 'scenario')),
	CONSTRAINT ck_cm_cash_forecasts_scenario CHECK (scenario IN ('base_case', 'optimistic', 'pessimistic', 'stress_test', 'custom')),
	CONSTRAINT ck_cm_cash_forecasts_currency_length CHECK (LENGTH(currency_code) = 3)
) PARTITION BY HASH (tenant_id);

-- Create partitions for forecasts
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_cash_forecasts_%s PARTITION OF cm_cash_forecasts FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- Forecast assumptions table
CREATE TABLE cm_forecast_assumptions (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Assumption identification
	forecast_id VARCHAR(26) NOT NULL,
	assumption_name VARCHAR(200) NOT NULL,
	category VARCHAR(100) NOT NULL,
	
	-- Assumption details
	base_value DECIMAL(18,6) NOT NULL,
	optimistic_value DECIMAL(18,6),
	pessimistic_value DECIMAL(18,6),
	
	-- Statistical properties
	probability_distribution VARCHAR(50),
	mean_value DECIMAL(18,6),
	standard_deviation DECIMAL(18,6),
	minimum_value DECIMAL(18,6),
	maximum_value DECIMAL(18,6),
	
	-- Sensitivity analysis
	sensitivity_coefficient DECIMAL(10,6),
	correlation_factors JSONB DEFAULT '{}'::jsonb,
	
	-- Documentation
	description VARCHAR(1000),
	data_source VARCHAR(200),
	last_reviewed DATE,
	confidence_level DECIMAL(4,3) CHECK (confidence_level IS NULL OR (confidence_level >= 0.0 AND confidence_level <= 1.0)),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100)
) PARTITION BY HASH (tenant_id);

-- Create partitions for forecast assumptions
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_forecast_assumptions_%s PARTITION OF cm_forecast_assumptions FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- ============================================================================
-- Investment Management Tables
-- ============================================================================

-- Investments table
CREATE TABLE cm_investments (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Investment identification
	investment_number VARCHAR(50) NOT NULL,
	external_reference VARCHAR(100),
	
	-- Investment details
	investment_type VARCHAR(30) NOT NULL,
	issuer VARCHAR(200) NOT NULL,
	issuer_rating VARCHAR(10),
	
	-- Financial terms
	principal_amount DECIMAL(18,2) NOT NULL CHECK (principal_amount > 0),
	currency_code VARCHAR(3) NOT NULL,
	interest_rate DECIMAL(10,6) NOT NULL,
	compounding_frequency VARCHAR(20) DEFAULT 'daily' NOT NULL,
	
	-- Dates and timing
	trade_date DATE NOT NULL,
	value_date DATE NOT NULL,
	maturity_date DATE NOT NULL,
	early_redemption_date DATE,
	
	-- Status and lifecycle
	status VARCHAR(20) DEFAULT 'pending' NOT NULL,
	booking_account_id VARCHAR(26) NOT NULL,
	
	-- Performance tracking
	expected_return DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	accrued_interest DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	current_value DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	realized_return DECIMAL(18,2),
	
	-- Risk management
	risk_rating VARCHAR(15) DEFAULT 'low' NOT NULL,
	credit_limit_impact DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	liquidity_rating VARCHAR(20),
	regulatory_treatment VARCHAR(100),
	
	-- Optimization metadata
	optimization_score DECIMAL(5,2) CHECK (optimization_score IS NULL OR (optimization_score >= 0.0 AND optimization_score <= 100.0)),
	selection_reason VARCHAR(500),
	alternative_considered VARCHAR(200),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_investments_type CHECK (investment_type IN ('money_market_fund', 'treasury_bill', 'certificate_deposit', 'commercial_paper', 'repurchase_agreement', 'term_deposit', 'government_bond', 'corporate_bond')),
	CONSTRAINT ck_cm_investments_status CHECK (status IN ('pending', 'active', 'matured', 'redeemed', 'cancelled', 'defaulted')),
	CONSTRAINT ck_cm_investments_risk_rating CHECK (risk_rating IN ('very_low', 'low', 'medium', 'high', 'very_high')),
	CONSTRAINT ck_cm_investments_currency_length CHECK (LENGTH(currency_code) = 3),
	CONSTRAINT ck_cm_investments_dates CHECK (maturity_date >= value_date AND value_date >= trade_date)
) PARTITION BY HASH (tenant_id);

-- Create partitions for investments
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_investments_%s PARTITION OF cm_investments FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- Investment opportunities table
CREATE TABLE cm_investment_opportunities (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Opportunity identification
	opportunity_id VARCHAR(26) NOT NULL DEFAULT uuid_generate_v7()::text,
	source VARCHAR(100) NOT NULL,
	provider VARCHAR(200) NOT NULL,
	
	-- Investment details
	investment_type VARCHAR(30) NOT NULL,
	minimum_amount DECIMAL(18,2) NOT NULL CHECK (minimum_amount > 0),
	maximum_amount DECIMAL(18,2) CHECK (maximum_amount IS NULL OR maximum_amount >= minimum_amount),
	currency_code VARCHAR(3) NOT NULL,
	
	-- Terms and conditions
	interest_rate DECIMAL(10,6) NOT NULL,
	term_days INTEGER NOT NULL CHECK (term_days >= 1),
	early_redemption_penalty DECIMAL(18,2),
	fees DECIMAL(18,2) DEFAULT 0.00 NOT NULL,
	
	-- Counterparty information
	counterparty_name VARCHAR(200) NOT NULL,
	counterparty_rating VARCHAR(10),
	risk_rating VARCHAR(15) NOT NULL,
	
	-- Opportunity window
	available_from TIMESTAMPTZ NOT NULL,
	available_until TIMESTAMPTZ NOT NULL,
	last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	
	-- AI scoring and optimization
	ai_score DECIMAL(5,2) NOT NULL CHECK (ai_score >= 0.0 AND ai_score <= 100.0),
	yield_score DECIMAL(5,2) NOT NULL CHECK (yield_score >= 0.0 AND yield_score <= 100.0),
	risk_score DECIMAL(5,2) NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 100.0),
	liquidity_score DECIMAL(5,2) NOT NULL CHECK (liquidity_score >= 0.0 AND liquidity_score <= 100.0),
	
	-- Recommendation
	recommended_amount DECIMAL(18,2) CHECK (recommended_amount IS NULL OR recommended_amount > 0),
	recommendation_reason VARCHAR(500),
	fit_score DECIMAL(5,2) CHECK (fit_score IS NULL OR (fit_score >= 0.0 AND fit_score <= 100.0)),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_investment_opportunities_type CHECK (investment_type IN ('money_market_fund', 'treasury_bill', 'certificate_deposit', 'commercial_paper', 'repurchase_agreement', 'term_deposit', 'government_bond', 'corporate_bond')),
	CONSTRAINT ck_cm_investment_opportunities_risk_rating CHECK (risk_rating IN ('very_low', 'low', 'medium', 'high', 'very_high')),
	CONSTRAINT ck_cm_investment_opportunities_currency_length CHECK (LENGTH(currency_code) = 3),
	CONSTRAINT ck_cm_investment_opportunities_window CHECK (available_until > available_from)
) PARTITION BY HASH (tenant_id);

-- Create partitions for investment opportunities
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_investment_opportunities_%s PARTITION OF cm_investment_opportunities FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- ============================================================================
-- Alert and Notification Tables
-- ============================================================================

-- Cash alerts table
CREATE TABLE cm_cash_alerts (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Alert identification
	alert_type VARCHAR(30) NOT NULL,
	severity VARCHAR(20) NOT NULL,
	title VARCHAR(200) NOT NULL,
	description VARCHAR(1000) NOT NULL,
	
	-- Alert context
	entity_id VARCHAR(26) NOT NULL,
	account_id VARCHAR(26),
	currency_code VARCHAR(3),
	
	-- Alert data
	current_value DECIMAL(18,2),
	threshold_value DECIMAL(18,2),
	variance_amount DECIMAL(18,2),
	variance_percentage DECIMAL(8,4),
	
	-- Alert timing
	triggered_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	escalated_at TIMESTAMPTZ,
	resolved_at TIMESTAMPTZ,
	acknowledged_at TIMESTAMPTZ,
	acknowledged_by VARCHAR(100),
	
	-- Alert status
	status VARCHAR(20) DEFAULT 'active' NOT NULL,
	resolution_notes VARCHAR(1000),
	auto_resolved BOOLEAN DEFAULT false,
	
	-- Notification tracking
	notifications_sent TEXT[] DEFAULT '{}',
	escalation_level INTEGER DEFAULT 1 NOT NULL,
	max_escalations INTEGER DEFAULT 3 NOT NULL,
	
	-- Related data
	related_forecast_id VARCHAR(26),
	related_investment_id VARCHAR(26),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_cash_alerts_type CHECK (alert_type IN ('balance_low', 'balance_high', 'forecast_shortfall', 'investment_maturity', 'rate_change', 'risk_threshold', 'bank_connection', 'compliance_violation')),
	CONSTRAINT ck_cm_cash_alerts_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
	CONSTRAINT ck_cm_cash_alerts_status CHECK (status IN ('active', 'acknowledged', 'resolved', 'dismissed')),
	CONSTRAINT ck_cm_cash_alerts_currency_length CHECK (currency_code IS NULL OR LENGTH(currency_code) = 3)
) PARTITION BY RANGE (triggered_at);

-- Create monthly partitions for alerts
DO $$ 
DECLARE
	start_date TIMESTAMPTZ := DATE_TRUNC('month', CURRENT_TIMESTAMP);
	end_date TIMESTAMPTZ;
	partition_name TEXT;
BEGIN
	FOR month_offset IN 0..11 LOOP
		start_date := DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '1 month' * month_offset);
		end_date := start_date + INTERVAL '1 month';
		partition_name := format('cm_cash_alerts_%s_%s', 
			EXTRACT(YEAR FROM start_date), 
			LPAD(EXTRACT(MONTH FROM start_date)::text, 2, '0')
		);
		
		EXECUTE format('CREATE TABLE %s PARTITION OF cm_cash_alerts FOR VALUES FROM (''%s'') TO (''%s'')', 
			partition_name, start_date, end_date);
	END LOOP;
END $$;

-- Optimization rules table
CREATE TABLE cm_optimization_rules (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Rule identification
	rule_name VARCHAR(200) NOT NULL,
	rule_code VARCHAR(50) NOT NULL,
	category VARCHAR(100) NOT NULL,
	
	-- Rule scope
	entity_ids TEXT[] DEFAULT '{}',
	currency_codes VARCHAR(3)[] DEFAULT '{}',
	account_types VARCHAR(20)[] DEFAULT '{}',
	
	-- Rule logic
	optimization_goal VARCHAR(30) NOT NULL,
	priority INTEGER DEFAULT 50 CHECK (priority >= 1 AND priority <= 100),
	
	-- Conditions
	minimum_amount DECIMAL(18,2) CHECK (minimum_amount IS NULL OR minimum_amount > 0),
	maximum_amount DECIMAL(18,2) CHECK (maximum_amount IS NULL OR maximum_amount > 0),
	time_constraints VARCHAR(500),
	market_conditions VARCHAR(500),
	
	-- Investment parameters
	maximum_maturity_days INTEGER CHECK (maximum_maturity_days IS NULL OR maximum_maturity_days > 0),
	minimum_yield_rate DECIMAL(10,6),
	maximum_risk_rating VARCHAR(15),
	diversification_limits JSONB DEFAULT '{}'::jsonb,
	
	-- Risk controls
	single_counterparty_limit DECIMAL(18,2),
	concentration_limit DECIMAL(5,2) CHECK (concentration_limit IS NULL OR (concentration_limit >= 0.0 AND concentration_limit <= 100.0)),
	stress_test_threshold DECIMAL(18,2),
	
	-- Execution settings
	auto_execute BOOLEAN DEFAULT false,
	approval_required_above DECIMAL(18,2) CHECK (approval_required_above IS NULL OR approval_required_above > 0),
	notification_recipients TEXT[] DEFAULT '{}',
	
	-- Rule status
	is_active BOOLEAN DEFAULT true,
	last_executed TIMESTAMPTZ,
	execution_count INTEGER DEFAULT 0,
	success_rate DECIMAL(4,3) CHECK (success_rate IS NULL OR (success_rate >= 0.0 AND success_rate <= 1.0)),
	
	-- AI enhancement
	ai_enhanced BOOLEAN DEFAULT true,
	learning_enabled BOOLEAN DEFAULT true,
	model_version VARCHAR(50),
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL,
	updated_by VARCHAR(100),
	
	-- Versioning for optimistic locking
	version INTEGER DEFAULT 1 NOT NULL,
	
	-- Soft delete support
	is_deleted BOOLEAN DEFAULT false,
	deleted_at TIMESTAMPTZ,
	deleted_by VARCHAR(100),
	
	-- Constraints
	CONSTRAINT ck_cm_optimization_rules_goal CHECK (optimization_goal IN ('maximize_yield', 'minimize_risk', 'maximize_liquidity', 'balance_yield_risk', 'minimize_fees')),
	CONSTRAINT ck_cm_optimization_rules_max_risk CHECK (maximum_risk_rating IS NULL OR maximum_risk_rating IN ('very_low', 'low', 'medium', 'high', 'very_high')),
	CONSTRAINT uq_cm_optimization_rules_code_tenant UNIQUE (rule_code, tenant_id)
) PARTITION BY HASH (tenant_id);

-- Create partitions for optimization rules
DO $$ 
BEGIN
	FOR i IN 0..15 LOOP
		EXECUTE format('CREATE TABLE cm_optimization_rules_%s PARTITION OF cm_optimization_rules FOR VALUES WITH (modulus 16, remainder %s)', i, i);
	END LOOP;
END $$;

-- ============================================================================
-- APG Audit Compliance Tables
-- ============================================================================

-- Audit trail table for comprehensive activity logging
CREATE TABLE cm_audit_trail (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Audit event identification
	event_type VARCHAR(50) NOT NULL,
	entity_type VARCHAR(50) NOT NULL,
	entity_id VARCHAR(26) NOT NULL,
	
	-- User and session information
	user_id VARCHAR(100) NOT NULL,
	session_id VARCHAR(100),
	ip_address INET,
	user_agent TEXT,
	
	-- Action details
	action VARCHAR(50) NOT NULL,
	description VARCHAR(500),
	old_values JSONB,
	new_values JSONB,
	
	-- Regulatory compliance
	compliance_category VARCHAR(50),
	regulatory_impact BOOLEAN DEFAULT false,
	retention_period_years INTEGER DEFAULT 7,
	
	-- Timing
	event_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	
	-- Additional metadata
	metadata JSONB DEFAULT '{}'::jsonb,
	
	-- Constraints
	CONSTRAINT ck_cm_audit_trail_action CHECK (action IN ('create', 'update', 'delete', 'view', 'export', 'approve', 'reject', 'execute'))
) PARTITION BY RANGE (event_timestamp);

-- Create monthly partitions for audit trail
DO $$ 
DECLARE
	start_date TIMESTAMPTZ := DATE_TRUNC('month', CURRENT_TIMESTAMP);
	end_date TIMESTAMPTZ;
	partition_name TEXT;
BEGIN
	FOR month_offset IN 0..11 LOOP
		start_date := DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '1 month' * month_offset);
		end_date := start_date + INTERVAL '1 month';
		partition_name := format('cm_audit_trail_%s_%s', 
			EXTRACT(YEAR FROM start_date), 
			LPAD(EXTRACT(MONTH FROM start_date)::text, 2, '0')
		);
		
		EXECUTE format('CREATE TABLE %s PARTITION OF cm_audit_trail FOR VALUES FROM (''%s'') TO (''%s'')', 
			partition_name, start_date, end_date);
	END LOOP;
END $$;

-- ============================================================================
-- Performance Monitoring Tables
-- ============================================================================

-- Performance metrics table
CREATE TABLE cm_performance_metrics (
	id VARCHAR(26) PRIMARY KEY DEFAULT uuid_generate_v7()::text,
	tenant_id VARCHAR(26) NOT NULL,
	
	-- Metric identification
	metric_name VARCHAR(100) NOT NULL,
	metric_category VARCHAR(50) NOT NULL,
	entity_type VARCHAR(50),
	entity_id VARCHAR(26),
	
	-- Metric values
	metric_value DECIMAL(18,6) NOT NULL,
	metric_unit VARCHAR(20) NOT NULL,
	baseline_value DECIMAL(18,6),
	target_value DECIMAL(18,6),
	
	-- Time period
	measurement_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	period_start TIMESTAMPTZ,
	period_end TIMESTAMPTZ,
	
	-- Additional dimensions
	dimensions JSONB DEFAULT '{}'::jsonb,
	
	-- APG audit compliance fields
	created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
	created_by VARCHAR(100) NOT NULL
) PARTITION BY RANGE (measurement_timestamp);

-- Create monthly partitions for performance metrics
DO $$ 
DECLARE
	start_date TIMESTAMPTZ := DATE_TRUNC('month', CURRENT_TIMESTAMP);
	end_date TIMESTAMPTZ;
	partition_name TEXT;
BEGIN
	FOR month_offset IN 0..11 LOOP
		start_date := DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '1 month' * month_offset);
		end_date := start_date + INTERVAL '1 month';
		partition_name := format('cm_performance_metrics_%s_%s', 
			EXTRACT(YEAR FROM start_date), 
			LPAD(EXTRACT(MONTH FROM start_date)::text, 2, '0')
		);
		
		EXECUTE format('CREATE TABLE %s PARTITION OF cm_performance_metrics FOR VALUES FROM (''%s'') TO (''%s'')', 
			partition_name, start_date, end_date);
	END LOOP;
END $$;

-- ============================================================================
-- Complete Schema - Migration 001 Successfully Applied
-- ============================================================================