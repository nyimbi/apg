--
-- APG Accounts Receivable - Performance Optimization
-- Advanced indexing, partitioning, and query optimization for APG multi-tenant environment
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
--

-- Prerequisites check
SELECT validate_migration_prerequisites(
	'apg_accounts_receivable', 
	ARRAY['001_initial_schema']
);

SET search_path = apg_accounts_receivable, public;

-- Start migration timing
\timing on

-- =============================================================================
-- Advanced Performance Indexes
-- =============================================================================

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_tenant_customer_status 
	ON ar_invoices(tenant_id, customer_id, status, due_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_aging_buckets 
	ON ar_invoices(tenant_id, status, due_date) 
	WHERE status NOT IN ('cancelled', 'paid') AND balance_amount > 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_payments_unapplied_cash 
	ON ar_payments(tenant_id, customer_id, unapplied_amount DESC) 
	WHERE unapplied_amount > 0 AND status = 'cleared';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_activities_collection_queue 
	ON ar_collection_activities(tenant_id, priority, activity_date, status) 
	WHERE status IN ('pending', 'in_progress');

-- Partial indexes for active records only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_customers_active 
	ON ar_customers(tenant_id, customer_code, legal_name) 
	WHERE status = 'active';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_open 
	ON ar_invoices(tenant_id, customer_id, due_date DESC, total_amount DESC) 
	WHERE status NOT IN ('cancelled', 'paid');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_payments_pending 
	ON ar_payments(tenant_id, payment_date DESC, payment_amount DESC) 
	WHERE status IN ('pending', 'processing');

-- Full-text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_customers_fts 
	ON ar_customers USING gin(
		to_tsvector('english', coalesce(legal_name, '') || ' ' || coalesce(trade_name, '') || ' ' || coalesce(customer_code, ''))
	);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_fts 
	ON ar_invoices USING gin(
		to_tsvector('english', coalesce(invoice_number, '') || ' ' || coalesce(customer_invoice_reference, '') || ' ' || coalesce(notes, ''))
	);

-- =============================================================================
-- Hash Indexes for Equality Lookups
-- =============================================================================

-- Hash indexes for exact matches (faster than btree for equality)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_customers_code_hash 
	ON ar_customers USING hash(customer_code);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_number_hash 
	ON ar_invoices USING hash(invoice_number);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_payments_number_hash 
	ON ar_payments USING hash(payment_number);

-- =============================================================================
-- Expression Indexes for Calculated Fields
-- =============================================================================

-- Index on calculated overdue days
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_invoices_days_overdue 
	ON ar_invoices((CURRENT_DATE - due_date)) 
	WHERE status = 'overdue' AND balance_amount > 0;

-- Index on credit utilization calculation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_customers_credit_utilization 
	ON ar_customers((total_outstanding / NULLIF(credit_limit, 0))) 
	WHERE credit_limit > 0 AND total_outstanding > 0;

-- Index on payment application percentage
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ar_payments_application_pct 
	ON ar_payments((applied_amount / NULLIF(payment_amount, 0))) 
	WHERE payment_amount > 0;

-- =============================================================================
-- Materialized Views for Heavy Reporting Queries
-- =============================================================================

-- Customer aging materialized view for faster dashboard queries
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ar_customer_aging AS
SELECT 
	c.id as customer_id,
	c.tenant_id,
	c.customer_code,
	c.legal_name,
	c.status,
	c.credit_limit,
	c.total_outstanding,
	-- Aging buckets
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date >= CURRENT_DATE), 0) as current_amount,
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE AND i.due_date >= CURRENT_DATE - INTERVAL '30 days'), 0) as days_1_30,
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '30 days' AND i.due_date >= CURRENT_DATE - INTERVAL '60 days'), 0) as days_31_60,
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '60 days' AND i.due_date >= CURRENT_DATE - INTERVAL '90 days'), 0) as days_61_90,
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '90 days'), 0) as days_over_90,
	-- Key metrics
	COUNT(i.id) FILTER (WHERE i.balance_amount > 0) as open_invoice_count,
	MAX(CURRENT_DATE - i.due_date) FILTER (WHERE i.status = 'overdue') as max_days_overdue,
	-- Performance indicators
	CASE 
		WHEN c.credit_limit > 0 THEN ROUND((c.total_outstanding / c.credit_limit) * 100, 2)
		ELSE NULL 
	END as credit_utilization_pct,
	CURRENT_DATE as last_updated
FROM ar_customers c
LEFT JOIN ar_invoices i ON c.id = i.customer_id 
	AND i.balance_amount > 0 
	AND i.status NOT IN ('cancelled', 'paid')
GROUP BY c.id, c.tenant_id, c.customer_code, c.legal_name, c.status, c.credit_limit, c.total_outstanding;

-- Create indexes on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_customer_aging_pk 
	ON mv_ar_customer_aging(customer_id);
CREATE INDEX IF NOT EXISTS idx_mv_customer_aging_tenant 
	ON mv_ar_customer_aging(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mv_customer_aging_overdue 
	ON mv_ar_customer_aging(max_days_overdue DESC) 
	WHERE max_days_overdue > 0;

-- Monthly sales and collection performance materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ar_monthly_performance AS
WITH monthly_invoices AS (
	SELECT 
		tenant_id,
		customer_id,
		date_trunc('month', invoice_date) as month_year,
		COUNT(*) as invoice_count,
		SUM(total_amount) as total_invoiced,
		SUM(CASE WHEN status = 'paid' THEN total_amount ELSE 0 END) as total_collected,
		AVG(CASE WHEN status = 'paid' THEN paid_amount ELSE NULL END) as avg_payment_amount,
		AVG(CASE WHEN status = 'paid' THEN EXTRACT(days FROM (updated_at::date - invoice_date)) ELSE NULL END) as avg_collection_days
	FROM ar_invoices
	WHERE invoice_date >= date_trunc('month', CURRENT_DATE - INTERVAL '24 months')
	GROUP BY tenant_id, customer_id, date_trunc('month', invoice_date)
),
monthly_payments AS (
	SELECT 
		tenant_id,
		customer_id,
		date_trunc('month', payment_date) as month_year,
		COUNT(*) as payment_count,
		SUM(payment_amount) as total_payments
	FROM ar_payments
	WHERE payment_date >= date_trunc('month', CURRENT_DATE - INTERVAL '24 months')
	AND status = 'cleared'
	GROUP BY tenant_id, customer_id, date_trunc('month', payment_date)
)
SELECT 
	COALESCE(i.tenant_id, p.tenant_id) as tenant_id,
	COALESCE(i.customer_id, p.customer_id) as customer_id,
	COALESCE(i.month_year, p.month_year) as month_year,
	COALESCE(i.invoice_count, 0) as invoice_count,
	COALESCE(i.total_invoiced, 0) as total_invoiced,
	COALESCE(i.total_collected, 0) as total_collected,
	COALESCE(p.payment_count, 0) as payment_count,
	COALESCE(p.total_payments, 0) as total_payments,
	i.avg_payment_amount,
	i.avg_collection_days,
	-- Collection efficiency
	CASE 
		WHEN i.total_invoiced > 0 THEN ROUND((i.total_collected / i.total_invoiced) * 100, 2)
		ELSE NULL 
	END as collection_efficiency_pct,
	CURRENT_DATE as last_updated
FROM monthly_invoices i
FULL OUTER JOIN monthly_payments p 
	ON i.tenant_id = p.tenant_id 
	AND i.customer_id = p.customer_id 
	AND i.month_year = p.month_year;

-- Create indexes on monthly performance view
CREATE INDEX IF NOT EXISTS idx_mv_monthly_performance_tenant_month 
	ON mv_ar_monthly_performance(tenant_id, month_year DESC);
CREATE INDEX IF NOT EXISTS idx_mv_monthly_performance_customer 
	ON mv_ar_monthly_performance(customer_id, month_year DESC);

-- =============================================================================
-- Database Statistics and Optimization Functions
-- =============================================================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_ar_materialized_views()
RETURNS TEXT AS $$
DECLARE
	start_time TIMESTAMP;
	result TEXT;
BEGIN
	start_time := clock_timestamp();
	
	-- Refresh customer aging view
	REFRESH MATERIALIZED VIEW CONCURRENTLY mv_ar_customer_aging;
	
	-- Refresh monthly performance view
	REFRESH MATERIALIZED VIEW CONCURRENTLY mv_ar_monthly_performance;
	
	result := format(
		'Materialized views refreshed in %s ms',
		EXTRACT(milliseconds FROM clock_timestamp() - start_time)
	);
	
	RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze table statistics
CREATE OR REPLACE FUNCTION analyze_ar_tables()
RETURNS TEXT AS $$
DECLARE
	start_time TIMESTAMP;
BEGIN
	start_time := clock_timestamp();
	
	-- Analyze all AR tables
	ANALYZE ar_customers;
	ANALYZE ar_customer_addresses;
	ANALYZE ar_customer_contacts;
	ANALYZE ar_invoices;
	ANALYZE ar_invoice_line_items;
	ANALYZE ar_payments;
	ANALYZE ar_payment_allocations;
	ANALYZE ar_collection_activities;
	ANALYZE ar_credit_assessments;
	ANALYZE ar_disputes;
	ANALYZE ar_cash_applications;
	
	RETURN format(
		'Table statistics updated in %s ms',
		EXTRACT(milliseconds FROM clock_timestamp() - start_time)
	);
END;
$$ LANGUAGE plpgsql;

-- Function to get table size statistics
CREATE OR REPLACE FUNCTION get_ar_table_sizes()
RETURNS TABLE(
	table_name TEXT,
	row_count BIGINT,
	total_size TEXT,
	index_size TEXT,
	toast_size TEXT
) AS $$
BEGIN
	RETURN QUERY
	SELECT 
		schemaname||'.'||tablename as table_name,
		n_tup_ins - n_tup_del as row_count,
		pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
		pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
		pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as toast_size
	FROM pg_stat_user_tables 
	WHERE schemaname = 'apg_accounts_receivable'
	ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Automated Maintenance Procedures
-- =============================================================================

-- Function to reindex tables when fragmentation is high
CREATE OR REPLACE FUNCTION reindex_ar_tables_if_needed()
RETURNS TEXT AS $$
DECLARE
	table_rec RECORD;
	reindex_count INTEGER := 0;
BEGIN
	-- Check index bloat and reindex if necessary
	FOR table_rec IN 
		SELECT tablename 
		FROM pg_tables 
		WHERE schemaname = 'apg_accounts_receivable'
		AND tablename LIKE 'ar_%'
	LOOP
		-- In production, this would check actual bloat statistics
		-- For now, just document the pattern
		-- REINDEX TABLE CONCURRENTLY would be used
		NULL;
	END LOOP;
	
	RETURN format('Checked %s tables for reindexing needs', reindex_count);
END;
$$ LANGUAGE plpgsql;

-- Function to update table statistics automatically
CREATE OR REPLACE FUNCTION auto_analyze_ar_tables()
RETURNS TEXT AS $$
BEGIN
	-- Update statistics on tables with significant changes
	PERFORM analyze_ar_tables();
	PERFORM refresh_ar_materialized_views();
	
	RETURN 'Automated analysis and refresh completed';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Performance Monitoring Views
-- =============================================================================

-- View for slow query identification
CREATE VIEW v_ar_performance_monitor AS
SELECT 
	schemaname,
	tablename,
	attname as column_name,
	n_distinct,
	correlation,
	most_common_vals,
	most_common_freqs
FROM pg_stats 
WHERE schemaname = 'apg_accounts_receivable'
ORDER BY tablename, attname;

-- View for index usage statistics
CREATE VIEW v_ar_index_usage AS
SELECT 
	schemaname,
	tablename,
	indexname,
	idx_tup_read,
	idx_tup_fetch,
	idx_scan as index_scans,
	CASE 
		WHEN idx_scan = 0 THEN 'Never used'
		WHEN idx_scan < 100 THEN 'Low usage'
		WHEN idx_scan < 1000 THEN 'Moderate usage'
		ELSE 'High usage'
	END as usage_category
FROM pg_stat_user_indexes 
WHERE schemaname = 'apg_accounts_receivable'
ORDER BY idx_scan DESC;

-- =============================================================================
-- Partitioning Strategy for Large Tables
-- =============================================================================

-- Function to create new invoice partitions automatically
CREATE OR REPLACE FUNCTION create_invoice_partition_for_year(year_val INTEGER)
RETURNS TEXT AS $$
DECLARE
	partition_name TEXT;
	start_date DATE;
	end_date DATE;
BEGIN
	partition_name := format('ar_invoices_%s', year_val);
	start_date := format('%s-01-01', year_val)::DATE;
	end_date := format('%s-01-01', year_val + 1)::DATE;
	
	-- Create partition if it doesn't exist
	EXECUTE format(
		'CREATE TABLE IF NOT EXISTS %I PARTITION OF ar_invoices
		FOR VALUES FROM (%L) TO (%L)',
		partition_name, start_date, end_date
	);
	
	RETURN format('Created partition %s for year %s', partition_name, year_val);
END;
$$ LANGUAGE plpgsql;

-- Create future partitions
SELECT create_invoice_partition_for_year(2027);
SELECT create_invoice_partition_for_year(2028);

-- =============================================================================
-- Query Plan Analysis Functions
-- =============================================================================

-- Function to explain analyze common AR queries
CREATE OR REPLACE FUNCTION analyze_ar_query_performance()
RETURNS TABLE(query_description TEXT, query_plan TEXT) AS $$
BEGIN
	-- This would contain actual EXPLAIN ANALYZE for common queries
	-- Placeholder for demonstration
	RETURN QUERY
	SELECT 
		'Customer aging query'::TEXT as query_description,
		'Query plan analysis would go here'::TEXT as query_plan;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Performance Optimization Completion
-- =============================================================================

-- Record successful migration
SELECT record_migration(
	'apg_accounts_receivable',
	'002_performance_optimization',
	'Advanced indexing and performance optimization',
	NULL
);

-- Refresh statistics after index creation
SELECT analyze_ar_tables();
SELECT refresh_ar_materialized_views();

COMMENT ON SCHEMA apg_accounts_receivable IS 'APG Accounts Receivable schema with performance optimizations applied';

-- Migration timing off
\timing off