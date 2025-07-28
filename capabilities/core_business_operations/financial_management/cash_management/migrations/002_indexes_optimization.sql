-- ============================================================================
-- APG Cash Management - Performance Optimization Indexes
-- 
-- Comprehensive indexing strategy for enterprise-scale performance.
-- Optimized for APG multi-tenant architecture and query patterns.
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
-- ============================================================================

-- Set search path
SET search_path TO cm_cash_management, public;

-- ============================================================================
-- Primary Entity Indexes - Banks
-- ============================================================================

-- Unique indexes for business constraints
CREATE UNIQUE INDEX CONCURRENTLY idx_cm_banks_tenant_bank_code 
ON cm_banks (tenant_id, bank_code) 
WHERE NOT is_deleted;

CREATE UNIQUE INDEX CONCURRENTLY idx_cm_banks_tenant_swift_code 
ON cm_banks (tenant_id, swift_code) 
WHERE NOT is_deleted;

-- Performance indexes for common queries
CREATE INDEX CONCURRENTLY idx_cm_banks_status_active 
ON cm_banks (tenant_id, status, created_at DESC) 
WHERE status = 'active' AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_banks_country_status 
ON cm_banks (tenant_id, country_code, status) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_banks_api_enabled 
ON cm_banks (tenant_id, api_enabled, last_api_sync DESC) 
WHERE api_enabled = true AND NOT is_deleted;

-- GIN index for contact information searches
CREATE INDEX CONCURRENTLY idx_cm_banks_contacts_gin 
ON cm_banks USING GIN (contacts);

-- Text search index for bank names
CREATE INDEX CONCURRENTLY idx_cm_banks_name_trgm 
ON cm_banks USING GIN (bank_name gin_trgm_ops) 
WHERE NOT is_deleted;

-- ============================================================================
-- Primary Entity Indexes - Cash Accounts
-- ============================================================================

-- Unique indexes for business constraints
CREATE UNIQUE INDEX CONCURRENTLY idx_cm_cash_accounts_tenant_account_number 
ON cm_cash_accounts (tenant_id, bank_id, account_number) 
WHERE NOT is_deleted;

-- Performance indexes for common queries
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_entity_currency 
ON cm_cash_accounts (tenant_id, entity_id, currency_code, status) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_bank_type 
ON cm_cash_accounts (tenant_id, bank_id, account_type, status) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_balance_update 
ON cm_cash_accounts (tenant_id, last_balance_update DESC) 
WHERE status = 'active' AND NOT is_deleted;

-- Indexes for cash positioning queries
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_current_balance 
ON cm_cash_accounts (tenant_id, currency_code, current_balance DESC) 
WHERE status = 'active' AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_available_balance 
ON cm_cash_accounts (tenant_id, currency_code, available_balance DESC) 
WHERE status = 'active' AND NOT is_deleted;

-- Sweep optimization indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_sweep_enabled 
ON cm_cash_accounts (tenant_id, auto_sweep_enabled, sweep_threshold) 
WHERE auto_sweep_enabled = true AND status = 'active' AND NOT is_deleted;

-- Reconciliation status indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_reconciliation 
ON cm_cash_accounts (tenant_id, reconciliation_status, last_reconciled_date DESC) 
WHERE NOT is_deleted;

-- GIN index for fee schedule searches
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_fees_gin 
ON cm_cash_accounts USING GIN (fee_schedule);

-- ============================================================================
-- Time-Series Indexes - Cash Positions
-- ============================================================================

-- Primary business query indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_positions_entity_date 
ON cm_cash_positions (tenant_id, entity_id, position_date DESC, currency_code) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_positions_currency_date 
ON cm_cash_positions (tenant_id, currency_code, position_date DESC) 
WHERE NOT is_deleted;

-- Performance indexes for dashboard queries
CREATE INDEX CONCURRENTLY idx_cm_cash_positions_total_cash 
ON cm_cash_positions (tenant_id, position_date DESC, total_cash DESC) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_positions_available_cash 
ON cm_cash_positions (tenant_id, position_date DESC, available_cash DESC) 
WHERE NOT is_deleted;

-- Risk monitoring indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_positions_concentration_risk 
ON cm_cash_positions (tenant_id, concentration_risk DESC, position_date DESC) 
WHERE concentration_risk IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_positions_liquidity_ratio 
ON cm_cash_positions (tenant_id, liquidity_ratio ASC, position_date DESC) 
WHERE liquidity_ratio IS NOT NULL AND NOT is_deleted;

-- Forecasting and analytics indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_positions_projected_flows 
ON cm_cash_positions (tenant_id, position_date, net_projected_flow) 
WHERE NOT is_deleted;

-- Composite index for global cash position rollups
CREATE INDEX CONCURRENTLY idx_cm_cash_positions_global_rollup 
ON cm_cash_positions (tenant_id, position_date, currency_code, total_cash, available_cash) 
WHERE NOT is_deleted;

-- ============================================================================
-- Time-Series Indexes - Cash Flows
-- ============================================================================

-- Primary business query indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_account_date 
ON cm_cash_flows (tenant_id, account_id, flow_date DESC) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_flows_entity_date 
ON cm_cash_flows (tenant_id, flow_date DESC, transaction_type) 
WHERE NOT is_deleted;

-- Transaction type and category analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_type_category 
ON cm_cash_flows (tenant_id, transaction_type, category, flow_date DESC) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_flows_business_unit 
ON cm_cash_flows (tenant_id, business_unit, flow_date DESC, amount) 
WHERE business_unit IS NOT NULL AND NOT is_deleted;

-- Amount-based queries for analytics
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_large_amounts 
ON cm_cash_flows (tenant_id, flow_date DESC, ABS(amount) DESC) 
WHERE ABS(amount) > 10000 AND NOT is_deleted;

-- Forecasting and pattern analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_recurring 
ON cm_cash_flows (tenant_id, is_recurring, recurrence_pattern, flow_date DESC) 
WHERE is_recurring = true AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_flows_confidence 
ON cm_cash_flows (tenant_id, forecast_confidence DESC, flow_date DESC) 
WHERE forecast_confidence IS NOT NULL AND NOT is_deleted;

-- Source system integration
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_source_module 
ON cm_cash_flows (tenant_id, source_module, flow_date DESC) 
WHERE source_module IS NOT NULL AND NOT is_deleted;

-- Transaction timing analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_timing_variance 
ON cm_cash_flows (tenant_id, planned_date, actual_date, flow_date) 
WHERE planned_date IS NOT NULL AND actual_date IS NOT NULL AND NOT is_deleted;

-- Text search for descriptions and counterparties
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_description_trgm 
ON cm_cash_flows USING GIN (description gin_trgm_ops) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_flows_counterparty_trgm 
ON cm_cash_flows USING GIN (counterparty gin_trgm_ops) 
WHERE counterparty IS NOT NULL AND NOT is_deleted;

-- ============================================================================
-- Forecasting Indexes
-- ============================================================================

-- Primary forecast queries
CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_entity_date 
ON cm_cash_forecasts (tenant_id, entity_id, forecast_date DESC, forecast_type) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_scenario_horizon 
ON cm_cash_forecasts (tenant_id, scenario, horizon_days, forecast_date DESC) 
WHERE NOT is_deleted;

-- Model performance tracking
CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_accuracy 
ON cm_cash_forecasts (tenant_id, model_used, accuracy_percentage DESC, forecast_date DESC) 
WHERE accuracy_percentage IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_confidence 
ON cm_cash_forecasts (tenant_id, confidence_level DESC, forecast_date DESC) 
WHERE NOT is_deleted;

-- Risk analysis indexes
CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_shortfall_risk 
ON cm_cash_forecasts (tenant_id, shortfall_probability DESC, forecast_date DESC) 
WHERE shortfall_probability IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_stress_test 
ON cm_cash_forecasts (tenant_id, stress_test_result, forecast_date DESC) 
WHERE stress_test_result IS NOT NULL AND NOT is_deleted;

-- GIN index for feature importance analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_forecasts_features_gin 
ON cm_cash_forecasts USING GIN (feature_importance);

-- Forecast assumptions indexes
CREATE INDEX CONCURRENTLY idx_cm_forecast_assumptions_forecast_category 
ON cm_forecast_assumptions (tenant_id, forecast_id, category) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_forecast_assumptions_sensitivity 
ON cm_forecast_assumptions (tenant_id, sensitivity_coefficient DESC) 
WHERE sensitivity_coefficient IS NOT NULL AND NOT is_deleted;

-- GIN index for correlation factors
CREATE INDEX CONCURRENTLY idx_cm_forecast_assumptions_correlations_gin 
ON cm_forecast_assumptions USING GIN (correlation_factors);

-- ============================================================================
-- Investment Management Indexes
-- ============================================================================

-- Primary investment queries
CREATE INDEX CONCURRENTLY idx_cm_investments_status_maturity 
ON cm_investments (tenant_id, status, maturity_date) 
WHERE NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_investments_type_rating 
ON cm_investments (tenant_id, investment_type, risk_rating, maturity_date) 
WHERE NOT is_deleted;

-- Performance tracking indexes
CREATE INDEX CONCURRENTLY idx_cm_investments_return_performance 
ON cm_investments (tenant_id, realized_return DESC, maturity_date DESC) 
WHERE realized_return IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_investments_yield_analysis 
ON cm_investments (tenant_id, interest_rate DESC, principal_amount DESC) 
WHERE status = 'active' AND NOT is_deleted;

-- Risk management indexes
CREATE INDEX CONCURRENTLY idx_cm_investments_issuer_concentration 
ON cm_investments (tenant_id, issuer, principal_amount DESC) 
WHERE status IN ('active', 'pending') AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_investments_currency_exposure 
ON cm_investments (tenant_id, currency_code, principal_amount DESC) 
WHERE status IN ('active', 'pending') AND NOT is_deleted;

-- Maturity monitoring indexes
CREATE INDEX CONCURRENTLY idx_cm_investments_upcoming_maturity 
ON cm_investments (tenant_id, maturity_date) 
WHERE status = 'active' AND maturity_date >= CURRENT_DATE AND NOT is_deleted;

-- Optimization indexes
CREATE INDEX CONCURRENTLY idx_cm_investments_optimization_score 
ON cm_investments (tenant_id, optimization_score DESC, trade_date DESC) 
WHERE optimization_score IS NOT NULL AND NOT is_deleted;

-- Investment opportunities indexes
CREATE INDEX CONCURRENTLY idx_cm_investment_opportunities_available 
ON cm_investment_opportunities (tenant_id, available_until, ai_score DESC) 
WHERE available_until > CURRENT_TIMESTAMP AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_investment_opportunities_yield_risk 
ON cm_investment_opportunities (tenant_id, yield_score DESC, risk_score ASC) 
WHERE available_until > CURRENT_TIMESTAMP AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_investment_opportunities_term_amount 
ON cm_investment_opportunities (tenant_id, term_days, minimum_amount) 
WHERE available_until > CURRENT_TIMESTAMP AND NOT is_deleted;

-- ============================================================================
-- Alert and Notification Indexes
-- ============================================================================

-- Active alerts monitoring
CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_active 
ON cm_cash_alerts (tenant_id, status, severity, triggered_at DESC) 
WHERE status = 'active' AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_entity_type 
ON cm_cash_alerts (tenant_id, entity_id, alert_type, triggered_at DESC) 
WHERE NOT is_deleted;

-- Escalation monitoring
CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_escalation 
ON cm_cash_alerts (tenant_id, escalation_level, triggered_at) 
WHERE status = 'active' AND escalation_level > 1 AND NOT is_deleted;

-- Performance analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_resolution_time 
ON cm_cash_alerts (tenant_id, triggered_at, resolved_at) 
WHERE resolved_at IS NOT NULL AND NOT is_deleted;

-- Alert type analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_type_frequency 
ON cm_cash_alerts (tenant_id, alert_type, triggered_at DESC) 
WHERE NOT is_deleted;

-- Optimization rules indexes
CREATE INDEX CONCURRENTLY idx_cm_optimization_rules_active 
ON cm_optimization_rules (tenant_id, is_active, priority DESC) 
WHERE is_active = true AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_optimization_rules_goal_category 
ON cm_optimization_rules (tenant_id, optimization_goal, category) 
WHERE is_active = true AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_optimization_rules_execution 
ON cm_optimization_rules (tenant_id, last_executed DESC, success_rate DESC) 
WHERE is_active = true AND NOT is_deleted;

-- ============================================================================
-- APG Audit and Compliance Indexes
-- ============================================================================

-- Audit trail queries
CREATE INDEX CONCURRENTLY idx_cm_audit_trail_entity 
ON cm_audit_trail (tenant_id, entity_type, entity_id, event_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_cm_audit_trail_user_action 
ON cm_audit_trail (tenant_id, user_id, action, event_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_cm_audit_trail_regulatory 
ON cm_audit_trail (tenant_id, regulatory_impact, compliance_category, event_timestamp DESC) 
WHERE regulatory_impact = true;

-- User activity monitoring
CREATE INDEX CONCURRENTLY idx_cm_audit_trail_session 
ON cm_audit_trail (tenant_id, session_id, event_timestamp DESC) 
WHERE session_id IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_cm_audit_trail_ip_address 
ON cm_audit_trail (tenant_id, ip_address, event_timestamp DESC) 
WHERE ip_address IS NOT NULL;

-- GIN index for metadata searches
CREATE INDEX CONCURRENTLY idx_cm_audit_trail_metadata_gin 
ON cm_audit_trail USING GIN (metadata);

-- ============================================================================
-- Performance Monitoring Indexes
-- ============================================================================

-- Performance metrics queries
CREATE INDEX CONCURRENTLY idx_cm_performance_metrics_category_time 
ON cm_performance_metrics (tenant_id, metric_category, measurement_timestamp DESC);

CREATE INDEX CONCURRENTLY idx_cm_performance_metrics_entity 
ON cm_performance_metrics (tenant_id, entity_type, entity_id, measurement_timestamp DESC) 
WHERE entity_id IS NOT NULL;

-- Metric analysis indexes
CREATE INDEX CONCURRENTLY idx_cm_performance_metrics_value_trends 
ON cm_performance_metrics (tenant_id, metric_name, measurement_timestamp DESC, metric_value);

CREATE INDEX CONCURRENTLY idx_cm_performance_metrics_targets 
ON cm_performance_metrics (tenant_id, metric_name, target_value, metric_value) 
WHERE target_value IS NOT NULL;

-- GIN index for dimensions analysis
CREATE INDEX CONCURRENTLY idx_cm_performance_metrics_dimensions_gin 
ON cm_performance_metrics USING GIN (dimensions);

-- ============================================================================
-- Cross-Table Foreign Key Indexes
-- ============================================================================

-- Cash accounts to banks
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_bank_id 
ON cm_cash_accounts (bank_id) 
WHERE NOT is_deleted;

-- Cash flows to accounts
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_account_id 
ON cm_cash_flows (account_id) 
WHERE NOT is_deleted;

-- Investments to accounts
CREATE INDEX CONCURRENTLY idx_cm_investments_booking_account_id 
ON cm_investments (booking_account_id) 
WHERE NOT is_deleted;

-- Forecast assumptions to forecasts
CREATE INDEX CONCURRENTLY idx_cm_forecast_assumptions_forecast_id 
ON cm_forecast_assumptions (forecast_id) 
WHERE NOT is_deleted;

-- Alerts to related entities
CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_account_id 
ON cm_cash_alerts (account_id) 
WHERE account_id IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_forecast_id 
ON cm_cash_alerts (related_forecast_id) 
WHERE related_forecast_id IS NOT NULL AND NOT is_deleted;

CREATE INDEX CONCURRENTLY idx_cm_cash_alerts_investment_id 
ON cm_cash_alerts (related_investment_id) 
WHERE related_investment_id IS NOT NULL AND NOT is_deleted;

-- ============================================================================
-- Composite Indexes for Complex Queries
-- ============================================================================

-- Cash position summary queries
CREATE INDEX CONCURRENTLY idx_cm_cash_position_summary 
ON cm_cash_positions (tenant_id, entity_id, currency_code, position_date DESC, total_cash, available_cash) 
WHERE NOT is_deleted;

-- Investment portfolio analysis
CREATE INDEX CONCURRENTLY idx_cm_investment_portfolio 
ON cm_investments (tenant_id, currency_code, investment_type, status, principal_amount DESC, maturity_date) 
WHERE NOT is_deleted;

-- Cash flow pattern analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_flow_patterns 
ON cm_cash_flows (tenant_id, transaction_type, category, flow_date, amount) 
WHERE NOT is_deleted;

-- Risk monitoring dashboard
CREATE INDEX CONCURRENTLY idx_cm_risk_monitoring 
ON cm_cash_positions (tenant_id, position_date DESC, concentration_risk DESC, liquidity_ratio ASC, stress_test_coverage ASC) 
WHERE NOT is_deleted;

-- ============================================================================
-- Functional Indexes for Calculated Fields
-- ============================================================================

-- Effective balance calculation for accounts
CREATE INDEX CONCURRENTLY idx_cm_cash_accounts_effective_balance 
ON cm_cash_accounts (tenant_id, (current_balance + pending_credits - pending_debits) DESC) 
WHERE status = 'active' AND NOT is_deleted;

-- Investment yield calculations
CREATE INDEX CONCURRENTLY idx_cm_investments_annualized_yield 
ON cm_investments (tenant_id, (interest_rate * 365.0 / (maturity_date - value_date)) DESC) 
WHERE status = 'active' AND maturity_date > value_date AND NOT is_deleted;

-- Days to maturity for investments
CREATE INDEX CONCURRENTLY idx_cm_investments_days_to_maturity 
ON cm_investments (tenant_id, (maturity_date - CURRENT_DATE)) 
WHERE status = 'active' AND maturity_date >= CURRENT_DATE AND NOT is_deleted;

-- Cash flow variance analysis
CREATE INDEX CONCURRENTLY idx_cm_cash_flows_timing_variance 
ON cm_cash_flows (tenant_id, ABS(EXTRACT(days FROM (actual_date - planned_date)))) 
WHERE planned_date IS NOT NULL AND actual_date IS NOT NULL AND NOT is_deleted;

-- ============================================================================
-- Statistics and Maintenance
-- ============================================================================

-- Update statistics for all tables
ANALYZE cm_banks;
ANALYZE cm_cash_accounts;
ANALYZE cm_cash_positions;
ANALYZE cm_cash_flows;
ANALYZE cm_cash_forecasts;
ANALYZE cm_forecast_assumptions;
ANALYZE cm_investments;
ANALYZE cm_investment_opportunities;
ANALYZE cm_cash_alerts;
ANALYZE cm_optimization_rules;
ANALYZE cm_audit_trail;
ANALYZE cm_performance_metrics;

-- ============================================================================
-- Index Monitoring Views
-- ============================================================================

-- Create view for index usage monitoring
CREATE OR REPLACE VIEW cm_index_usage_stats AS
SELECT 
	schemaname,
	tablename,
	indexname,
	idx_scan,
	idx_tup_read,
	idx_tup_fetch,
	CASE 
		WHEN idx_scan = 0 THEN 'Never Used'
		WHEN idx_scan < 100 THEN 'Low Usage'
		WHEN idx_scan < 1000 THEN 'Medium Usage'
		ELSE 'High Usage'
	END AS usage_category
FROM pg_stat_user_indexes 
WHERE schemaname = 'cm_cash_management'
ORDER BY idx_scan DESC;

-- Create view for table size monitoring
CREATE OR REPLACE VIEW cm_table_sizes AS
SELECT 
	schemaname,
	tablename,
	pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
	pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
	pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS index_size
FROM pg_tables 
WHERE schemaname = 'cm_cash_management'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- ============================================================================
-- Performance Optimization - Migration 002 Successfully Applied
-- ============================================================================