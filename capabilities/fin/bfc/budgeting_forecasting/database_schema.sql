-- =============================================================================
-- APG Budgeting & Forecasting - Database Schema
-- Multi-tenant PostgreSQL schema with APG integration patterns
-- 
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =============================================================================
-- APG Multi-Tenant Schema Structure
-- =============================================================================

-- Create tenant-specific schemas for data isolation
-- Example: bf_tenant_abc, bf_tenant_xyz
-- Note: These will be created dynamically for each tenant

-- Main application schema for shared resources
CREATE SCHEMA IF NOT EXISTS bf_shared;

-- =============================================================================
-- APG Base Tables (Shared Across Tenants)
-- =============================================================================

-- Tenant configuration and metadata
CREATE TABLE bf_shared.tenant_config (
	tenant_id VARCHAR(50) PRIMARY KEY,
	tenant_name VARCHAR(255) NOT NULL,
	status VARCHAR(20) NOT NULL DEFAULT 'active',
	features_enabled TEXT[] DEFAULT '{}',
	custom_settings JSONB DEFAULT '{}',
	integration_config JSONB DEFAULT '{}',
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by VARCHAR(50) NOT NULL,
	updated_by VARCHAR(50) NOT NULL,
	version INTEGER DEFAULT 1,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE,
	deleted_by VARCHAR(50)
);

-- Budget template library (shared across tenants with permissions)
CREATE TABLE bf_shared.budget_templates (
	id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
	template_name VARCHAR(255) NOT NULL,
	template_description TEXT,
	template_type VARCHAR(50) NOT NULL, -- annual, quarterly, monthly, project
	category VARCHAR(100) NOT NULL,
	industry_code VARCHAR(20),
	
	-- Template structure and configuration
	template_structure JSONB NOT NULL,
	default_settings JSONB DEFAULT '{}',
	approval_workflow JSONB DEFAULT '{}',
	
	-- Sharing and permissions
	owner_tenant_id VARCHAR(50),
	is_public BOOLEAN DEFAULT FALSE,
	shared_with_tenants TEXT[] DEFAULT '{}',
	usage_count INTEGER DEFAULT 0,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by VARCHAR(50) NOT NULL,
	updated_by VARCHAR(50) NOT NULL,
	version INTEGER DEFAULT 1,
	is_deleted BOOLEAN DEFAULT FALSE,
	deleted_at TIMESTAMP WITH TIME ZONE,
	deleted_by VARCHAR(50),
	
	FOREIGN KEY (owner_tenant_id) REFERENCES bf_shared.tenant_config(tenant_id)
);

-- Chart of accounts mapping (shared reference data)
CREATE TABLE bf_shared.account_categories (
	id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
	category_code VARCHAR(50) NOT NULL UNIQUE,
	category_name VARCHAR(255) NOT NULL,
	category_type VARCHAR(50) NOT NULL, -- revenue, expense, asset, liability
	parent_category_id VARCHAR(50),
	level_number INTEGER NOT NULL DEFAULT 1,
	is_active BOOLEAN DEFAULT TRUE,
	sort_order INTEGER DEFAULT 0,
	
	-- Industry and standard mapping
	gaap_mapping VARCHAR(100),
	ifrs_mapping VARCHAR(100),
	industry_specific JSONB DEFAULT '{}',
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by VARCHAR(50) NOT NULL,
	updated_by VARCHAR(50) NOT NULL,
	version INTEGER DEFAULT 1,
	
	FOREIGN KEY (parent_category_id) REFERENCES bf_shared.account_categories(id)
);

-- =============================================================================
-- Core Budgeting & Forecasting Tables (Tenant-Specific)
-- =============================================================================

-- Function to create tenant-specific schema and tables
CREATE OR REPLACE FUNCTION bf_shared.create_tenant_schema(tenant_id VARCHAR(50))
RETURNS VOID AS $$
DECLARE
    schema_name VARCHAR(100);
BEGIN
    schema_name := 'bf_' || tenant_id;
    
    -- Create tenant schema
    EXECUTE 'CREATE SCHEMA IF NOT EXISTS ' || quote_ident(schema_name);
    
    -- Set search path for table creation
    EXECUTE 'SET search_path TO ' || quote_ident(schema_name) || ', bf_shared, public';
    
    -- Create all tenant-specific tables
    PERFORM bf_shared.create_tenant_tables(schema_name, tenant_id);
    
END;
$$ LANGUAGE plpgsql;

-- Function to create all tables within a tenant schema
CREATE OR REPLACE FUNCTION bf_shared.create_tenant_tables(schema_name VARCHAR(100), tenant_id VARCHAR(50))
RETURNS VOID AS $$
BEGIN
    -- Set search path
    EXECUTE 'SET search_path TO ' || quote_ident(schema_name) || ', bf_shared, public';
    
    -- Budget master table
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.budgets (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Budget identification
        budget_name VARCHAR(255) NOT NULL,
        budget_code VARCHAR(50) UNIQUE,
        budget_type VARCHAR(50) NOT NULL, -- ANNUAL, QUARTERLY, MONTHLY, ROLLING
        fiscal_year INTEGER NOT NULL,
        budget_period_start DATE NOT NULL,
        budget_period_end DATE NOT NULL,
        
        -- Status and workflow
        status VARCHAR(50) NOT NULL DEFAULT ''DRAFT'', -- DRAFT, SUBMITTED, APPROVED, ACTIVE, CLOSED
        workflow_state VARCHAR(100),
        approval_level INTEGER DEFAULT 0,
        requires_approval BOOLEAN DEFAULT TRUE,
        
        -- Financial configuration
        base_currency VARCHAR(3) NOT NULL DEFAULT ''USD'',
        budget_method VARCHAR(50) DEFAULT ''zero_based'', -- zero_based, incremental, activity_based
        planning_horizon_months INTEGER DEFAULT 12,
        
        -- Template and inheritance
        template_id VARCHAR(50),
        parent_budget_id VARCHAR(50),
        is_template BOOLEAN DEFAULT FALSE,
        template_usage_count INTEGER DEFAULT 0,
        
        -- Organizational hierarchy
        department_code VARCHAR(50),
        cost_center_code VARCHAR(50),
        business_unit VARCHAR(100),
        region_code VARCHAR(50),
        
        -- Performance tracking
        total_budget_amount DECIMAL(18,2) DEFAULT 0.00,
        total_committed_amount DECIMAL(18,2) DEFAULT 0.00,
        total_actual_amount DECIMAL(18,2) DEFAULT 0.00,
        variance_amount DECIMAL(18,2) DEFAULT 0.00,
        variance_percent DECIMAL(8,4) DEFAULT 0.00,
        
        -- AI/ML insights
        ai_confidence_score DECIMAL(4,3) DEFAULT 0.000,
        risk_assessment_score DECIMAL(4,3) DEFAULT 0.000,
        ai_recommendations JSONB DEFAULT ''[]'',
        forecast_accuracy_score DECIMAL(4,3),
        
        -- Collaboration and communication
        collaboration_enabled BOOLEAN DEFAULT TRUE,
        notification_settings JSONB DEFAULT ''{}'',
        last_activity_date DATE,
        active_contributors TEXT[] DEFAULT ''{}'',
        
        -- APG Integration fields
        document_folder_id VARCHAR(50),
        workflow_instance_id VARCHAR(50),
        ai_job_id VARCHAR(50),
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        deleted_at TIMESTAMP WITH TIME ZONE,
        deleted_by VARCHAR(50),
        
        FOREIGN KEY (template_id) REFERENCES bf_shared.budget_templates(id),
        FOREIGN KEY (parent_budget_id) REFERENCES ' || quote_ident(schema_name) || '.budgets(id),
        
        CONSTRAINT check_budget_dates CHECK (budget_period_end > budget_period_start),
        CONSTRAINT check_planning_horizon CHECK (planning_horizon_months > 0),
        CONSTRAINT check_ai_scores CHECK (
            ai_confidence_score BETWEEN 0 AND 1 AND
            risk_assessment_score BETWEEN 0 AND 1 AND
            (forecast_accuracy_score IS NULL OR forecast_accuracy_score BETWEEN 0 AND 1)
        )
    )';
    
    -- Budget line items (detailed budget allocations)
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.budget_lines (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        budget_id VARCHAR(50) NOT NULL,
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Line item identification
        line_number INTEGER NOT NULL,
        line_description VARCHAR(500) NOT NULL,
        line_category VARCHAR(100) NOT NULL,
        line_type VARCHAR(50) NOT NULL, -- revenue, expense, capital, transfer
        
        -- Account mapping
        account_code VARCHAR(50) NOT NULL,
        account_category_id VARCHAR(50),
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
        allocation_method VARCHAR(50) DEFAULT ''equal'', -- equal, weighted, seasonal, custom
        
        -- Budget amounts
        budgeted_amount DECIMAL(18,2) NOT NULL DEFAULT 0.00,
        forecasted_amount DECIMAL(18,2),
        committed_amount DECIMAL(18,2) DEFAULT 0.00,
        actual_amount DECIMAL(18,2) DEFAULT 0.00,
        variance_amount DECIMAL(18,2) DEFAULT 0.00,
        variance_percent DECIMAL(8,4) DEFAULT 0.00,
        
        -- Monthly breakdown (for detailed planning)
        month_01_amount DECIMAL(18,2) DEFAULT 0.00,
        month_02_amount DECIMAL(18,2) DEFAULT 0.00,
        month_03_amount DECIMAL(18,2) DEFAULT 0.00,
        month_04_amount DECIMAL(18,2) DEFAULT 0.00,
        month_05_amount DECIMAL(18,2) DEFAULT 0.00,
        month_06_amount DECIMAL(18,2) DEFAULT 0.00,
        month_07_amount DECIMAL(18,2) DEFAULT 0.00,
        month_08_amount DECIMAL(18,2) DEFAULT 0.00,
        month_09_amount DECIMAL(18,2) DEFAULT 0.00,
        month_10_amount DECIMAL(18,2) DEFAULT 0.00,
        month_11_amount DECIMAL(18,2) DEFAULT 0.00,
        month_12_amount DECIMAL(18,2) DEFAULT 0.00,
        
        -- Quarterly breakdown
        q1_amount DECIMAL(18,2) DEFAULT 0.00,
        q2_amount DECIMAL(18,2) DEFAULT 0.00,
        q3_amount DECIMAL(18,2) DEFAULT 0.00,
        q4_amount DECIMAL(18,2) DEFAULT 0.00,
        
        -- Currency and exchange
        currency_code VARCHAR(3) NOT NULL DEFAULT ''USD'',
        exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
        base_currency_amount DECIMAL(18,2),
        
        -- Driver-based budgeting
        quantity_driver VARCHAR(100),
        unit_quantity DECIMAL(12,4),
        unit_price DECIMAL(12,4),
        price_escalation_percent DECIMAL(6,4) DEFAULT 0.00,
        
        -- Approval and workflow
        approval_status VARCHAR(50) DEFAULT ''pending'',
        approval_level INTEGER DEFAULT 0,
        approved_by VARCHAR(50),
        approved_date DATE,
        rejection_reason TEXT,
        
        -- AI/ML insights
        ai_confidence_score DECIMAL(4,3) DEFAULT 0.000,
        seasonality_factor DECIMAL(6,4) DEFAULT 1.0000,
        trend_factor DECIMAL(6,4) DEFAULT 1.0000,
        ai_adjustments JSONB DEFAULT ''{}'',
        
        -- Comments and notes
        line_notes TEXT,
        business_justification TEXT,
        assumptions TEXT,
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        deleted_at TIMESTAMP WITH TIME ZONE,
        deleted_by VARCHAR(50),
        
        FOREIGN KEY (budget_id) REFERENCES ' || quote_ident(schema_name) || '.budgets(id) ON DELETE CASCADE,
        FOREIGN KEY (account_category_id) REFERENCES bf_shared.account_categories(id),
        
        CONSTRAINT check_line_dates CHECK (period_end >= period_start),
        CONSTRAINT check_line_amounts CHECK (budgeted_amount >= 0),
        CONSTRAINT check_unit_values CHECK (
            (unit_quantity IS NULL AND unit_price IS NULL) OR
            (unit_quantity IS NOT NULL AND unit_price IS NOT NULL AND unit_quantity >= 0 AND unit_price >= 0)
        )
    )';
    
    -- Forecast master table
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.forecasts (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Forecast identification
        forecast_name VARCHAR(255) NOT NULL,
        forecast_code VARCHAR(50) UNIQUE,
        forecast_type VARCHAR(50) NOT NULL, -- REVENUE, EXPENSE, CASH_FLOW, DEMAND, INTEGRATED
        forecast_method VARCHAR(50) NOT NULL, -- STATISTICAL, ML, HYBRID, JUDGMENTAL
        
        -- Time horizon and frequency
        forecast_horizon_months INTEGER NOT NULL,
        forecast_frequency VARCHAR(20) DEFAULT ''monthly'', -- daily, weekly, monthly, quarterly
        base_period_start DATE NOT NULL,
        base_period_end DATE NOT NULL,
        forecast_period_start DATE NOT NULL,
        forecast_period_end DATE NOT NULL,
        
        -- Model and algorithm configuration
        algorithm_type VARCHAR(50), -- arima, exponential_smoothing, neural_network, ensemble
        model_version VARCHAR(50),
        model_parameters JSONB DEFAULT ''{}'',
        feature_selection JSONB DEFAULT ''{}'',
        
        -- Data sources and inputs
        data_sources TEXT[] DEFAULT ''{}'',
        input_variables TEXT[] DEFAULT ''{}'',
        external_factors TEXT[] DEFAULT ''{}'',
        historical_months_used INTEGER DEFAULT 24,
        
        -- Accuracy and confidence
        accuracy_score DECIMAL(4,3),
        confidence_level DECIMAL(4,3) DEFAULT 0.950,
        confidence_interval_lower DECIMAL(18,2),
        confidence_interval_upper DECIMAL(18,2),
        mae_score DECIMAL(12,4), -- Mean Absolute Error
        mape_score DECIMAL(6,4), -- Mean Absolute Percentage Error
        rmse_score DECIMAL(12,4), -- Root Mean Square Error
        
        -- Scenario analysis
        scenario_type VARCHAR(50) DEFAULT ''base'', -- base, optimistic, pessimistic, stress
        probability_weight DECIMAL(4,3) DEFAULT 1.000,
        scenario_assumptions TEXT,
        sensitivity_analysis JSONB DEFAULT ''{}'',
        
        -- Business context
        department_code VARCHAR(50),
        business_unit VARCHAR(100),
        product_category VARCHAR(100),
        market_segment VARCHAR(100),
        geographic_region VARCHAR(100),
        
        -- Status and lifecycle
        status VARCHAR(50) NOT NULL DEFAULT ''DRAFT'', -- DRAFT, GENERATING, COMPLETED, PUBLISHED, ARCHIVED
        generation_status VARCHAR(50), -- pending, running, completed, failed
        last_generation_date TIMESTAMP WITH TIME ZONE,
        next_generation_date TIMESTAMP WITH TIME ZONE,
        auto_generation_enabled BOOLEAN DEFAULT FALSE,
        
        -- Performance tracking
        forecast_value DECIMAL(18,2),
        actual_value DECIMAL(18,2),
        variance_amount DECIMAL(18,2),
        variance_percent DECIMAL(8,4),
        accuracy_trend VARCHAR(20), -- improving, stable, declining
        
        -- APG Integration fields
        ai_job_id VARCHAR(50),
        time_series_job_id VARCHAR(50),
        federated_learning_session_id VARCHAR(50),
        
        -- Approval and review
        reviewed_by VARCHAR(50),
        review_date DATE,
        review_notes TEXT,
        approved_for_planning BOOLEAN DEFAULT FALSE,
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        deleted_at TIMESTAMP WITH TIME ZONE,
        deleted_by VARCHAR(50),
        
        CONSTRAINT check_forecast_dates CHECK (
            forecast_period_end > forecast_period_start AND
            base_period_end > base_period_start AND
            forecast_period_start >= base_period_end
        ),
        CONSTRAINT check_forecast_horizon CHECK (forecast_horizon_months > 0),
        CONSTRAINT check_accuracy_scores CHECK (
            (accuracy_score IS NULL OR accuracy_score BETWEEN 0 AND 1) AND
            confidence_level BETWEEN 0 AND 1 AND
            probability_weight BETWEEN 0 AND 1
        )
    )';
    
    -- Forecast data points (detailed forecast values)
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.forecast_data (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        forecast_id VARCHAR(50) NOT NULL,
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Time period
        period_date DATE NOT NULL,
        period_type VARCHAR(20) NOT NULL, -- daily, weekly, monthly, quarterly, annual
        period_sequence INTEGER NOT NULL,
        fiscal_year INTEGER,
        fiscal_quarter INTEGER,
        fiscal_month INTEGER,
        
        -- Forecast values
        forecasted_value DECIMAL(18,2) NOT NULL,
        confidence_lower DECIMAL(18,2),
        confidence_upper DECIMAL(18,2),
        actual_value DECIMAL(18,2),
        variance_amount DECIMAL(18,2),
        variance_percent DECIMAL(8,4),
        
        -- Decomposition components
        trend_component DECIMAL(18,2),
        seasonal_component DECIMAL(18,2),
        cyclical_component DECIMAL(18,2),
        irregular_component DECIMAL(18,2),
        external_factor_impact DECIMAL(18,2),
        
        -- Model insights
        prediction_strength DECIMAL(4,3),
        volatility_score DECIMAL(4,3),
        anomaly_score DECIMAL(4,3),
        feature_importance JSONB DEFAULT ''{}'',
        
        -- Business drivers
        volume_driver DECIMAL(12,4),
        price_driver DECIMAL(12,4),
        mix_driver DECIMAL(12,4),
        external_drivers JSONB DEFAULT ''{}'',
        
        -- Currency handling
        currency_code VARCHAR(3) NOT NULL DEFAULT ''USD'',
        exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
        base_currency_value DECIMAL(18,2),
        
        -- Notes and assumptions
        period_notes TEXT,
        assumptions TEXT,
        risk_factors TEXT,
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        
        FOREIGN KEY (forecast_id) REFERENCES ' || quote_ident(schema_name) || '.forecasts(id) ON DELETE CASCADE,
        
        CONSTRAINT check_forecast_values CHECK (
            (confidence_lower IS NULL OR confidence_upper IS NULL OR confidence_lower <= confidence_upper) AND
            (prediction_strength IS NULL OR prediction_strength BETWEEN 0 AND 1) AND
            (volatility_score IS NULL OR volatility_score BETWEEN 0 AND 1) AND
            (anomaly_score IS NULL OR anomaly_score BETWEEN 0 AND 1)
        )
    )';
    
    -- Variance analysis table
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.variance_analysis (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Analysis identification
        analysis_name VARCHAR(255) NOT NULL,
        analysis_type VARCHAR(50) NOT NULL, -- budget_vs_actual, forecast_vs_actual, period_comparison
        analysis_period_start DATE NOT NULL,
        analysis_period_end DATE NOT NULL,
        comparison_period_start DATE,
        comparison_period_end DATE,
        
        -- Subject of analysis
        budget_id VARCHAR(50),
        forecast_id VARCHAR(50),
        department_code VARCHAR(50),
        account_category VARCHAR(100),
        analysis_scope VARCHAR(50) DEFAULT ''detailed'', -- summary, detailed, line_item
        
        -- Variance calculations
        baseline_amount DECIMAL(18,2) NOT NULL,
        actual_amount DECIMAL(18,2) NOT NULL,
        variance_amount DECIMAL(18,2) NOT NULL,
        variance_percent DECIMAL(8,4) NOT NULL,
        absolute_variance DECIMAL(18,2) NOT NULL,
        
        -- Variance classification
        variance_type VARCHAR(50), -- favorable, unfavorable, neutral
        significance_level VARCHAR(20), -- critical, high, medium, low, minimal
        variance_threshold_exceeded BOOLEAN DEFAULT FALSE,
        requires_investigation BOOLEAN DEFAULT FALSE,
        
        -- Root cause analysis
        primary_cause VARCHAR(100),
        contributing_factors TEXT[] DEFAULT ''{}'',
        root_cause_category VARCHAR(50), -- volume, price, mix, timing, operational, external
        impact_assessment TEXT,
        
        -- AI-powered insights
        ai_explanation TEXT,
        ai_confidence_score DECIMAL(4,3),
        anomaly_detected BOOLEAN DEFAULT FALSE,
        pattern_analysis JSONB DEFAULT ''{}'',
        correlation_factors JSONB DEFAULT ''{}'',
        
        -- Corrective actions
        recommended_actions TEXT[] DEFAULT ''{}'',
        action_priority VARCHAR(20) DEFAULT ''medium'',
        estimated_impact DECIMAL(18,2),
        action_timeline VARCHAR(50),
        responsible_party VARCHAR(100),
        
        -- Investigation tracking
        investigation_status VARCHAR(50) DEFAULT ''pending'',
        investigated_by VARCHAR(50),
        investigation_date DATE,
        investigation_notes TEXT,
        resolution_status VARCHAR(50) DEFAULT ''open'',
        
        -- Performance metrics
        analysis_accuracy DECIMAL(4,3),
        prediction_quality DECIMAL(4,3),
        time_to_detection_days INTEGER,
        resolution_time_days INTEGER,
        
        -- APG Integration fields
        ai_analysis_job_id VARCHAR(50),
        notification_sent BOOLEAN DEFAULT FALSE,
        workflow_triggered BOOLEAN DEFAULT FALSE,
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        deleted_at TIMESTAMP WITH TIME ZONE,
        deleted_by VARCHAR(50),
        
        FOREIGN KEY (budget_id) REFERENCES ' || quote_ident(schema_name) || '.budgets(id),
        FOREIGN KEY (forecast_id) REFERENCES ' || quote_ident(schema_name) || '.forecasts(id),
        
        CONSTRAINT check_variance_dates CHECK (analysis_period_end >= analysis_period_start),
        CONSTRAINT check_variance_amounts CHECK (absolute_variance >= 0),
        CONSTRAINT check_ai_scores CHECK (
            (ai_confidence_score IS NULL OR ai_confidence_score BETWEEN 0 AND 1) AND
            (analysis_accuracy IS NULL OR analysis_accuracy BETWEEN 0 AND 1) AND
            (prediction_quality IS NULL OR prediction_quality BETWEEN 0 AND 1)
        )
    )';
    
    -- Scenario planning table
    EXECUTE '
    CREATE TABLE ' || quote_ident(schema_name) || '.scenarios (
        id VARCHAR(50) PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id VARCHAR(50) NOT NULL DEFAULT ''' || tenant_id || ''',
        
        -- Scenario identification
        scenario_name VARCHAR(255) NOT NULL,
        scenario_description TEXT,
        scenario_type VARCHAR(50) NOT NULL, -- base, optimistic, pessimistic, stress, what_if
        scenario_category VARCHAR(50), -- market, operational, financial, strategic
        
        -- Scenario parameters
        probability_weight DECIMAL(4,3) DEFAULT 0.333,
        time_horizon_months INTEGER NOT NULL,
        scenario_start_date DATE NOT NULL,
        scenario_end_date DATE NOT NULL,
        
        -- Base scenario reference
        base_budget_id VARCHAR(50),
        base_forecast_id VARCHAR(50),
        comparison_baseline VARCHAR(50) DEFAULT ''current_budget'',
        
        -- Scenario assumptions
        key_assumptions TEXT[] DEFAULT ''{}'',
        variable_changes JSONB DEFAULT ''{}'', -- {"revenue_growth": 0.15, "cost_inflation": 0.08}
        external_factors JSONB DEFAULT ''{}'',
        market_conditions TEXT,
        
        -- Financial impact
        total_revenue_impact DECIMAL(18,2) DEFAULT 0.00,
        total_expense_impact DECIMAL(18,2) DEFAULT 0.00,
        net_income_impact DECIMAL(18,2) DEFAULT 0.00,
        cash_flow_impact DECIMAL(18,2) DEFAULT 0.00,
        
        -- Risk assessment
        risk_level VARCHAR(20) DEFAULT ''medium'', -- low, medium, high, extreme
        downside_risk DECIMAL(18,2),
        upside_potential DECIMAL(18,2),
        value_at_risk DECIMAL(18,2),
        
        -- Sensitivity analysis
        sensitivity_variables TEXT[] DEFAULT ''{}'',
        elasticity_factors JSONB DEFAULT ''{}'',
        break_even_points JSONB DEFAULT ''{}'',
        
        -- Monte Carlo simulation
        simulation_enabled BOOLEAN DEFAULT FALSE,
        simulation_iterations INTEGER DEFAULT 1000,
        confidence_intervals JSONB DEFAULT ''{}'',
        distribution_parameters JSONB DEFAULT ''{}'',
        
        -- Decision support
        strategic_implications TEXT,
        recommended_decisions TEXT[] DEFAULT ''{}'',
        contingency_plans TEXT[] DEFAULT ''{}'',
        monitoring_indicators TEXT[] DEFAULT ''{}'',
        
        -- Modeling and calculation
        calculation_method VARCHAR(50) DEFAULT ''analytical'', -- analytical, simulation, hybrid
        model_complexity VARCHAR(20) DEFAULT ''medium'',
        last_calculation_date TIMESTAMP WITH TIME ZONE,
        calculation_duration_seconds INTEGER,
        
        -- Collaboration and review
        scenario_owner VARCHAR(50),
        review_participants TEXT[] DEFAULT ''{}'',
        last_review_date DATE,
        review_status VARCHAR(50) DEFAULT ''draft'',
        approval_required BOOLEAN DEFAULT FALSE,
        approved_by VARCHAR(50),
        approval_date DATE,
        
        -- APG Integration fields
        simulation_job_id VARCHAR(50),
        ai_modeling_job_id VARCHAR(50),
        document_folder_id VARCHAR(50),
        
        -- APG audit fields
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        is_deleted BOOLEAN DEFAULT FALSE,
        deleted_at TIMESTAMP WITH TIME ZONE,
        deleted_by VARCHAR(50),
        
        FOREIGN KEY (base_budget_id) REFERENCES ' || quote_ident(schema_name) || '.budgets(id),
        FOREIGN KEY (base_forecast_id) REFERENCES ' || quote_ident(schema_name) || '.forecasts(id),
        
        CONSTRAINT check_scenario_dates CHECK (scenario_end_date > scenario_start_date),
        CONSTRAINT check_scenario_horizon CHECK (time_horizon_months > 0),
        CONSTRAINT check_probability CHECK (probability_weight BETWEEN 0 AND 1),
        CONSTRAINT check_simulation_params CHECK (
            (simulation_enabled = FALSE) OR 
            (simulation_enabled = TRUE AND simulation_iterations >= 100)
        )
    )';

END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Performance Optimization Indexes
-- =============================================================================

-- Function to create performance indexes for a tenant schema
CREATE OR REPLACE FUNCTION bf_shared.create_performance_indexes(schema_name VARCHAR(100))
RETURNS VOID AS $$
BEGIN
    -- Budget table indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budgets_tenant_fiscal 
        ON ' || quote_ident(schema_name) || '.budgets(tenant_id, fiscal_year)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budgets_status_type 
        ON ' || quote_ident(schema_name) || '.budgets(status, budget_type) WHERE is_deleted = FALSE';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budgets_department 
        ON ' || quote_ident(schema_name) || '.budgets(department_code, cost_center_code) WHERE is_deleted = FALSE';
    
    -- Budget lines indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budget_lines_budget_period 
        ON ' || quote_ident(schema_name) || '.budget_lines(budget_id, period_start, period_end)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budget_lines_account 
        ON ' || quote_ident(schema_name) || '.budget_lines(account_code, account_category_id)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_budget_lines_org 
        ON ' || quote_ident(schema_name) || '.budget_lines(department_code, cost_center_code, project_code)';
    
    -- Forecast table indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_forecasts_tenant_type 
        ON ' || quote_ident(schema_name) || '.forecasts(tenant_id, forecast_type, forecast_method)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_forecasts_horizon_dates 
        ON ' || quote_ident(schema_name) || '.forecasts(forecast_horizon_months, forecast_period_start, forecast_period_end)';
    
    -- Forecast data indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_forecast_data_forecast_period 
        ON ' || quote_ident(schema_name) || '.forecast_data(forecast_id, period_date, period_type)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_forecast_data_fiscal 
        ON ' || quote_ident(schema_name) || '.forecast_data(fiscal_year, fiscal_quarter, fiscal_month)';
    
    -- Variance analysis indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_variance_analysis_period 
        ON ' || quote_ident(schema_name) || '.variance_analysis(analysis_period_start, analysis_period_end)';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_variance_analysis_significance 
        ON ' || quote_ident(schema_name) || '.variance_analysis(significance_level, requires_investigation) WHERE is_deleted = FALSE';
    
    -- Scenario planning indexes
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_scenarios_type_probability 
        ON ' || quote_ident(schema_name) || '.scenarios(scenario_type, probability_weight) WHERE is_deleted = FALSE';
    
    EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_' || replace(schema_name, 'bf_', '') || '_scenarios_dates 
        ON ' || quote_ident(schema_name) || '.scenarios(scenario_start_date, scenario_end_date)';

END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Row-Level Security (RLS) Policies
-- =============================================================================

-- Function to create RLS policies for tenant isolation
CREATE OR REPLACE FUNCTION bf_shared.create_rls_policies(schema_name VARCHAR(100), tenant_id VARCHAR(50))
RETURNS VOID AS $$
BEGIN
    -- Enable RLS on all tables
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.budgets ENABLE ROW LEVEL SECURITY';
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.budget_lines ENABLE ROW LEVEL SECURITY';
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.forecasts ENABLE ROW LEVEL SECURITY';
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.forecast_data ENABLE ROW LEVEL SECURITY';
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.variance_analysis ENABLE ROW LEVEL SECURITY';
    EXECUTE 'ALTER TABLE ' || quote_ident(schema_name) || '.scenarios ENABLE ROW LEVEL SECURITY';
    
    -- Create policies for tenant isolation
    EXECUTE 'CREATE POLICY tenant_isolation_budgets ON ' || quote_ident(schema_name) || '.budgets 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';
    
    EXECUTE 'CREATE POLICY tenant_isolation_budget_lines ON ' || quote_ident(schema_name) || '.budget_lines 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';
    
    EXECUTE 'CREATE POLICY tenant_isolation_forecasts ON ' || quote_ident(schema_name) || '.forecasts 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';
    
    EXECUTE 'CREATE POLICY tenant_isolation_forecast_data ON ' || quote_ident(schema_name) || '.forecast_data 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';
    
    EXECUTE 'CREATE POLICY tenant_isolation_variance_analysis ON ' || quote_ident(schema_name) || '.variance_analysis 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';
    
    EXECUTE 'CREATE POLICY tenant_isolation_scenarios ON ' || quote_ident(schema_name) || '.scenarios 
        FOR ALL TO app_role 
        USING (tenant_id = current_setting(''app.current_tenant''))';

END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Data Partitioning for Large Datasets
-- =============================================================================

-- Function to create partitioned tables for high-volume data
CREATE OR REPLACE FUNCTION bf_shared.create_partitioned_tables(schema_name VARCHAR(100))
RETURNS VOID AS $$
DECLARE
    current_year INTEGER := EXTRACT(YEAR FROM CURRENT_DATE);
    partition_year INTEGER;
BEGIN
    -- Create partitioned forecast_data table by fiscal year
    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(schema_name) || '.forecast_data CASCADE';
    
    EXECUTE 'CREATE TABLE ' || quote_ident(schema_name) || '.forecast_data (
        id VARCHAR(50) DEFAULT gen_random_uuid(),
        forecast_id VARCHAR(50) NOT NULL,
        tenant_id VARCHAR(50) NOT NULL,
        period_date DATE NOT NULL,
        period_type VARCHAR(20) NOT NULL,
        period_sequence INTEGER NOT NULL,
        fiscal_year INTEGER,
        fiscal_quarter INTEGER,
        fiscal_month INTEGER,
        forecasted_value DECIMAL(18,2) NOT NULL,
        confidence_lower DECIMAL(18,2),
        confidence_upper DECIMAL(18,2),
        actual_value DECIMAL(18,2),
        variance_amount DECIMAL(18,2),
        variance_percent DECIMAL(8,4),
        trend_component DECIMAL(18,2),
        seasonal_component DECIMAL(18,2),
        cyclical_component DECIMAL(18,2),
        irregular_component DECIMAL(18,2),
        external_factor_impact DECIMAL(18,2),
        prediction_strength DECIMAL(4,3),
        volatility_score DECIMAL(4,3),
        anomaly_score DECIMAL(4,3),
        feature_importance JSONB DEFAULT ''{}'',
        volume_driver DECIMAL(12,4),
        price_driver DECIMAL(12,4),
        mix_driver DECIMAL(12,4),
        external_drivers JSONB DEFAULT ''{}'',
        currency_code VARCHAR(3) NOT NULL DEFAULT ''USD'',
        exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
        base_currency_value DECIMAL(18,2),
        period_notes TEXT,
        assumptions TEXT,
        risk_factors TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_by VARCHAR(50) NOT NULL,
        updated_by VARCHAR(50) NOT NULL,
        version INTEGER DEFAULT 1,
        PRIMARY KEY (id, fiscal_year)
    ) PARTITION BY RANGE (fiscal_year)';
    
    -- Create partitions for current, previous, and next 2 years
    FOR partition_year IN (current_year - 1)..(current_year + 2) LOOP
        EXECUTE 'CREATE TABLE ' || quote_ident(schema_name) || '.forecast_data_' || partition_year || 
                ' PARTITION OF ' || quote_ident(schema_name) || '.forecast_data 
                FOR VALUES FROM (' || partition_year || ') TO (' || (partition_year + 1) || ')';
    END LOOP;

END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Database Functions and Triggers
-- =============================================================================

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION bf_shared.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to create triggers for all tables in a schema
CREATE OR REPLACE FUNCTION bf_shared.create_update_triggers(schema_name VARCHAR(100))
RETURNS VOID AS $$
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN 
        SELECT tablename FROM pg_tables WHERE schemaname = schema_name
    LOOP
        EXECUTE 'CREATE TRIGGER trigger_update_' || table_name || '_updated_at 
                 BEFORE UPDATE ON ' || quote_ident(schema_name) || '.' || quote_ident(table_name) || '
                 FOR EACH ROW EXECUTE FUNCTION bf_shared.update_updated_at_column()';
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate budget line totals
CREATE OR REPLACE FUNCTION bf_shared.calculate_budget_totals(budget_id VARCHAR(50), schema_name VARCHAR(100))
RETURNS TABLE(
    total_budgeted DECIMAL(18,2),
    total_forecasted DECIMAL(18,2),
    total_actual DECIMAL(18,2),
    total_variance DECIMAL(18,2)
) AS $$
BEGIN
    RETURN QUERY EXECUTE '
        SELECT 
            COALESCE(SUM(budgeted_amount), 0.00) as total_budgeted,
            COALESCE(SUM(forecasted_amount), 0.00) as total_forecasted,
            COALESCE(SUM(actual_amount), 0.00) as total_actual,
            COALESCE(SUM(variance_amount), 0.00) as total_variance
        FROM ' || quote_ident(schema_name) || '.budget_lines 
        WHERE budget_id = $1 AND is_deleted = FALSE'
    USING budget_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Complete Tenant Setup Function
-- =============================================================================

-- Master function to set up a complete tenant environment
CREATE OR REPLACE FUNCTION bf_shared.setup_tenant(tenant_id VARCHAR(50), tenant_name VARCHAR(255), created_by VARCHAR(50))
RETURNS VOID AS $$
DECLARE
    schema_name VARCHAR(100);
BEGIN
    schema_name := 'bf_' || tenant_id;
    
    -- 1. Insert tenant configuration
    INSERT INTO bf_shared.tenant_config (tenant_id, tenant_name, created_by, updated_by)
    VALUES (tenant_id, tenant_name, created_by, created_by)
    ON CONFLICT (tenant_id) DO NOTHING;
    
    -- 2. Create tenant schema and tables
    PERFORM bf_shared.create_tenant_schema(tenant_id);
    
    -- 3. Create performance indexes
    PERFORM bf_shared.create_performance_indexes(schema_name);
    
    -- 4. Set up row-level security
    PERFORM bf_shared.create_rls_policies(schema_name, tenant_id);
    
    -- 5. Create update triggers
    PERFORM bf_shared.create_update_triggers(schema_name);
    
    -- 6. Set up partitioned tables for high-volume data
    PERFORM bf_shared.create_partitioned_tables(schema_name);
    
    RAISE NOTICE 'Tenant % successfully set up with schema %', tenant_id, schema_name;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Initial Data and Configuration
-- =============================================================================

-- Insert default account categories
INSERT INTO bf_shared.account_categories (category_code, category_name, category_type, level_number, created_by, updated_by) VALUES
('REV', 'Revenue', 'revenue', 1, 'system', 'system'),
('REV_SALES', 'Sales Revenue', 'revenue', 2, 'system', 'system'),
('REV_SERVICES', 'Service Revenue', 'revenue', 2, 'system', 'system'),
('REV_OTHER', 'Other Revenue', 'revenue', 2, 'system', 'system'),

('EXP', 'Expenses', 'expense', 1, 'system', 'system'),
('EXP_COGS', 'Cost of Goods Sold', 'expense', 2, 'system', 'system'),
('EXP_OPEX', 'Operating Expenses', 'expense', 2, 'system', 'system'),
('EXP_ADMIN', 'Administrative Expenses', 'expense', 2, 'system', 'system'),
('EXP_SALES', 'Sales & Marketing', 'expense', 2, 'system', 'system'),

('CAP', 'Capital Expenditures', 'asset', 1, 'system', 'system'),
('CAP_EQUIPMENT', 'Equipment', 'asset', 2, 'system', 'system'),
('CAP_SOFTWARE', 'Software', 'asset', 2, 'system', 'system'),
('CAP_FACILITIES', 'Facilities', 'asset', 2, 'system', 'system');

-- Update parent relationships
UPDATE bf_shared.account_categories SET parent_category_id = (SELECT id FROM bf_shared.account_categories WHERE category_code = 'REV') WHERE category_code IN ('REV_SALES', 'REV_SERVICES', 'REV_OTHER');
UPDATE bf_shared.account_categories SET parent_category_id = (SELECT id FROM bf_shared.account_categories WHERE category_code = 'EXP') WHERE category_code IN ('EXP_COGS', 'EXP_OPEX', 'EXP_ADMIN', 'EXP_SALES');
UPDATE bf_shared.account_categories SET parent_category_id = (SELECT id FROM bf_shared.account_categories WHERE category_code = 'CAP') WHERE category_code IN ('CAP_EQUIPMENT', 'CAP_SOFTWARE', 'CAP_FACILITIES');

-- Insert default budget templates
INSERT INTO bf_shared.budget_templates (template_name, template_description, template_type, category, template_structure, created_by, updated_by) VALUES
('Annual Operating Budget', 'Standard annual operating budget template', 'annual', 'operating', 
 '{"sections": ["revenue", "operating_expenses", "capital_expenditures"], "periods": 12, "currency": "USD"}', 
 'system', 'system'),

('Quarterly Rolling Forecast', 'Rolling quarterly forecast template', 'quarterly', 'forecast',
 '{"sections": ["revenue_forecast", "expense_forecast", "cash_flow"], "periods": 4, "rolling": true}',
 'system', 'system'),

('Project Budget Template', 'Template for project-specific budgeting', 'project', 'project',
 '{"sections": ["project_revenue", "project_costs", "resource_allocation"], "milestone_based": true}',
 'system', 'system');

-- =============================================================================
-- Schema Documentation and Comments
-- =============================================================================

COMMENT ON SCHEMA bf_shared IS 'APG Budgeting & Forecasting - Shared resources and configuration';

COMMENT ON TABLE bf_shared.tenant_config IS 'Tenant configuration and feature management';
COMMENT ON TABLE bf_shared.budget_templates IS 'Shared budget templates and structures';
COMMENT ON TABLE bf_shared.account_categories IS 'Chart of accounts categories and mapping';

COMMENT ON FUNCTION bf_shared.setup_tenant(VARCHAR, VARCHAR, VARCHAR) IS 'Complete tenant environment setup with schema, tables, indexes, and security';
COMMENT ON FUNCTION bf_shared.create_tenant_schema(VARCHAR) IS 'Create tenant-specific schema and base structure';
COMMENT ON FUNCTION bf_shared.create_performance_indexes(VARCHAR) IS 'Create optimized indexes for query performance';

-- =============================================================================
-- Performance Monitoring Views
-- =============================================================================

-- Create view for monitoring tenant usage and performance
CREATE OR REPLACE VIEW bf_shared.tenant_usage_summary AS
SELECT 
    tc.tenant_id,
    tc.tenant_name,
    tc.status,
    tc.features_enabled,
    tc.created_at as tenant_created,
    
    -- Table counts would be calculated by querying information_schema
    -- This is a placeholder for actual implementation
    0 as budget_count,
    0 as forecast_count,
    0 as variance_analysis_count,
    
    tc.updated_at as last_activity
FROM bf_shared.tenant_config tc
WHERE tc.is_deleted = FALSE;

COMMENT ON VIEW bf_shared.tenant_usage_summary IS 'Summary view of tenant usage and activity metrics';

-- =============================================================================
-- Security and Access Control
-- =============================================================================

-- Create application role for database access
CREATE ROLE IF NOT EXISTS app_role;

-- Grant appropriate permissions to application role
GRANT USAGE ON SCHEMA bf_shared TO app_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA bf_shared TO app_role;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA bf_shared TO app_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bf_shared TO app_role;

-- Create read-only role for reporting
CREATE ROLE IF NOT EXISTS bf_readonly;
GRANT USAGE ON SCHEMA bf_shared TO bf_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA bf_shared TO bf_readonly;

COMMENT ON ROLE app_role IS 'Application role for APG Budgeting & Forecasting operations';
COMMENT ON ROLE bf_readonly IS 'Read-only role for reporting and analytics';

-- =============================================================================
-- End of Schema Definition
-- =============================================================================

-- Example usage to set up a tenant:
-- SELECT bf_shared.setup_tenant('demo_corp', 'Demo Corporation', 'admin_user');