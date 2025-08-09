--
-- APG Accounts Receivable - Initial Database Schema
-- Multi-tenant PostgreSQL schema with APG integration
--
-- © 2025 Datacraft. All rights reserved.
-- Author: Nyimbi Odero <nyimbi@gmail.com>
--

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "hstore";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema for accounts receivable capability
CREATE SCHEMA IF NOT EXISTS apg_accounts_receivable;

-- Set search path for this migration
SET search_path = apg_accounts_receivable, public;

-- =============================================================================
-- Core Customer Management Tables
-- =============================================================================

-- AR Customers table with multi-tenant support
CREATE TABLE ar_customers (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_code VARCHAR(50) NOT NULL,
	legal_name VARCHAR(255) NOT NULL,
	trade_name VARCHAR(255),
	customer_type VARCHAR(50) NOT NULL DEFAULT 'individual',
	status VARCHAR(50) NOT NULL DEFAULT 'active',
	
	-- Credit information
	credit_limit DECIMAL(15,2) DEFAULT 0.00,
	credit_rating VARCHAR(20),
	payment_terms_days INTEGER DEFAULT 30,
	
	-- Contact information
	primary_contact_name VARCHAR(255),
	primary_contact_email VARCHAR(255),
	primary_contact_phone VARCHAR(50),
	
	-- Financial metrics
	total_outstanding DECIMAL(15,2) DEFAULT 0.00,
	overdue_amount DECIMAL(15,2) DEFAULT 0.00,
	
	-- Collection information
	collection_priority VARCHAR(20) DEFAULT 'normal',
	collection_notes TEXT,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	version INTEGER DEFAULT 1,
	
	-- Constraints
	CONSTRAINT ar_customers_tenant_code_unique UNIQUE (tenant_id, customer_code),
	CONSTRAINT ar_customers_type_check CHECK (customer_type IN ('individual', 'corporation', 'partnership', 'government', 'non_profit')),
	CONSTRAINT ar_customers_status_check CHECK (status IN ('active', 'inactive', 'suspended', 'credit_hold')),
	CONSTRAINT ar_customers_rating_check CHECK (credit_rating IN ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D', 'NR')),
	CONSTRAINT ar_customers_priority_check CHECK (collection_priority IN ('low', 'normal', 'high', 'critical'))
);

-- Customer addresses with multi-tenant support
CREATE TABLE ar_customer_addresses (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	address_type VARCHAR(50) NOT NULL DEFAULT 'billing',
	line1 VARCHAR(255) NOT NULL,
	line2 VARCHAR(255),
	city VARCHAR(100) NOT NULL,
	state_province VARCHAR(100),
	postal_code VARCHAR(20),
	country_code CHAR(2) NOT NULL DEFAULT 'US',
	is_primary BOOLEAN DEFAULT false,
	is_active BOOLEAN DEFAULT true,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id) ON DELETE CASCADE,
	
	-- Constraints
	CONSTRAINT ar_addresses_type_check CHECK (address_type IN ('billing', 'shipping', 'correspondence', 'legal'))
);

-- Customer contacts with multi-tenant support
CREATE TABLE ar_customer_contacts (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	contact_type VARCHAR(50) NOT NULL DEFAULT 'primary',
	name VARCHAR(255) NOT NULL,
	title VARCHAR(100),
	email VARCHAR(255),
	phone VARCHAR(50),
	mobile VARCHAR(50),
	is_primary BOOLEAN DEFAULT false,
	is_active BOOLEAN DEFAULT true,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id) ON DELETE CASCADE,
	
	-- Constraints
	CONSTRAINT ar_contacts_type_check CHECK (contact_type IN ('primary', 'billing', 'collections', 'technical', 'executive'))
);

-- =============================================================================
-- Invoice Management Tables
-- =============================================================================

-- AR Invoices table with multi-tenant support and partitioning
CREATE TABLE ar_invoices (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	invoice_number VARCHAR(100) NOT NULL,
	customer_invoice_reference VARCHAR(100),
	
	-- Invoice dates
	invoice_date DATE NOT NULL,
	due_date DATE NOT NULL,
	service_date_from DATE,
	service_date_to DATE,
	
	-- Financial information
	currency_code CHAR(3) NOT NULL DEFAULT 'USD',
	exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
	subtotal_amount DECIMAL(15,2) NOT NULL DEFAULT 0.00,
	tax_amount DECIMAL(15,2) NOT NULL DEFAULT 0.00,
	discount_amount DECIMAL(15,2) NOT NULL DEFAULT 0.00,
	total_amount DECIMAL(15,2) NOT NULL,
	paid_amount DECIMAL(15,2) DEFAULT 0.00,
	balance_amount DECIMAL(15,2) GENERATED ALWAYS AS (total_amount - paid_amount) STORED,
	
	-- Status and workflow
	status VARCHAR(50) NOT NULL DEFAULT 'draft',
	payment_status VARCHAR(50) NOT NULL DEFAULT 'unpaid',
	
	-- Document management integration
	document_id UUID,
	pdf_url VARCHAR(500),
	
	-- Business metadata
	project_reference VARCHAR(100),
	purchase_order_number VARCHAR(100),
	terms_and_conditions TEXT,
	notes TEXT,
	
	-- APG AI/ML fields
	ai_extraction_confidence DECIMAL(5,4),
	ai_matching_confidence DECIMAL(5,4),
	requires_manual_review BOOLEAN DEFAULT false,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	version INTEGER DEFAULT 1,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	
	-- Constraints
	CONSTRAINT ar_invoices_tenant_number_unique UNIQUE (tenant_id, invoice_number),
	CONSTRAINT ar_invoices_status_check CHECK (status IN ('draft', 'pending', 'sent', 'partially_paid', 'paid', 'overdue', 'cancelled', 'disputed')),
	CONSTRAINT ar_invoices_payment_status_check CHECK (payment_status IN ('unpaid', 'partially_paid', 'paid', 'overdue', 'writeoff')),
	CONSTRAINT ar_invoices_amounts_check CHECK (total_amount >= 0 AND subtotal_amount >= 0 AND tax_amount >= 0),
	CONSTRAINT ar_invoices_balance_check CHECK (paid_amount <= total_amount)
) PARTITION BY RANGE (invoice_date);

-- Create invoice partitions for better performance
CREATE TABLE ar_invoices_2024 PARTITION OF ar_invoices
	FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE ar_invoices_2025 PARTITION OF ar_invoices
	FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE ar_invoices_2026 PARTITION OF ar_invoices
	FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Invoice line items with multi-tenant support
CREATE TABLE ar_invoice_line_items (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	invoice_id UUID NOT NULL,
	line_number INTEGER NOT NULL,
	
	-- Product/service information
	product_code VARCHAR(100),
	description TEXT NOT NULL,
	quantity DECIMAL(15,4) NOT NULL DEFAULT 1.0000,
	unit_price DECIMAL(15,4) NOT NULL,
	line_amount DECIMAL(15,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
	
	-- GL and cost center information
	gl_account_code VARCHAR(50),
	cost_center VARCHAR(50),
	department VARCHAR(50),
	project_code VARCHAR(50),
	
	-- Tax information
	tax_code VARCHAR(20),
	tax_rate DECIMAL(7,4) DEFAULT 0.0000,
	tax_amount DECIMAL(15,2) DEFAULT 0.00,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (invoice_id) REFERENCES ar_invoices(id) ON DELETE CASCADE,
	
	-- Constraints
	CONSTRAINT ar_line_items_invoice_line_unique UNIQUE (invoice_id, line_number),
	CONSTRAINT ar_line_items_quantity_check CHECK (quantity > 0),
	CONSTRAINT ar_line_items_amounts_check CHECK (line_amount >= 0 AND tax_amount >= 0)
);

-- =============================================================================
-- Payment Management Tables
-- =============================================================================

-- AR Payments table with multi-tenant support
CREATE TABLE ar_payments (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	payment_number VARCHAR(100) NOT NULL,
	
	-- Payment information
	payment_date DATE NOT NULL,
	payment_amount DECIMAL(15,2) NOT NULL,
	payment_method VARCHAR(50) NOT NULL DEFAULT 'check',
	currency_code CHAR(3) NOT NULL DEFAULT 'USD',
	exchange_rate DECIMAL(10,6) DEFAULT 1.000000,
	
	-- Bank information
	bank_reference VARCHAR(100),
	check_number VARCHAR(50),
	wire_reference VARCHAR(100),
	ach_trace_number VARCHAR(50),
	
	-- Payment processing
	status VARCHAR(50) NOT NULL DEFAULT 'pending',
	processing_date DATE,
	settlement_date DATE,
	
	-- APG AI/ML fraud detection
	fraud_score DECIMAL(5,4) DEFAULT 0.0000,
	fraud_flags JSONB,
	risk_level VARCHAR(20) DEFAULT 'low',
	
	-- Unapplied cash tracking
	applied_amount DECIMAL(15,2) DEFAULT 0.00,
	unapplied_amount DECIMAL(15,2) GENERATED ALWAYS AS (payment_amount - applied_amount) STORED,
	
	-- Business metadata
	payment_memo TEXT,
	notes TEXT,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	version INTEGER DEFAULT 1,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	
	-- Constraints
	CONSTRAINT ar_payments_tenant_number_unique UNIQUE (tenant_id, payment_number),
	CONSTRAINT ar_payments_method_check CHECK (payment_method IN ('cash', 'check', 'ach', 'wire', 'credit_card', 'online', 'other')),
	CONSTRAINT ar_payments_status_check CHECK (status IN ('pending', 'processing', 'cleared', 'bounced', 'cancelled', 'reversed')),
	CONSTRAINT ar_payments_risk_check CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
	CONSTRAINT ar_payments_amounts_check CHECK (payment_amount > 0 AND applied_amount >= 0 AND applied_amount <= payment_amount)
);

-- Payment allocations for cash application
CREATE TABLE ar_payment_allocations (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	payment_id UUID NOT NULL,
	invoice_id UUID NOT NULL,
	
	-- Allocation information
	allocation_amount DECIMAL(15,2) NOT NULL,
	discount_taken DECIMAL(15,2) DEFAULT 0.00,
	writeoff_amount DECIMAL(15,2) DEFAULT 0.00,
	
	-- APG AI matching information
	matching_method VARCHAR(50) DEFAULT 'manual',
	ai_confidence_score DECIMAL(5,4),
	auto_applied BOOLEAN DEFAULT false,
	
	-- Business metadata
	allocation_memo TEXT,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (payment_id) REFERENCES ar_payments(id) ON DELETE CASCADE,
	FOREIGN KEY (invoice_id) REFERENCES ar_invoices(id),
	
	-- Constraints
	CONSTRAINT ar_allocations_payment_invoice_unique UNIQUE (payment_id, invoice_id),
	CONSTRAINT ar_allocations_method_check CHECK (matching_method IN ('manual', 'auto_exact', 'auto_fuzzy', 'ai_ml')),
	CONSTRAINT ar_allocations_amounts_check CHECK (allocation_amount > 0 AND discount_taken >= 0 AND writeoff_amount >= 0)
);

-- =============================================================================
-- Collections Management Tables
-- =============================================================================

-- Collection activities with multi-tenant support
CREATE TABLE ar_collection_activities (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	invoice_id UUID,
	
	-- Activity information
	activity_type VARCHAR(50) NOT NULL,
	activity_date DATE NOT NULL DEFAULT CURRENT_DATE,
	due_date DATE,
	priority VARCHAR(20) NOT NULL DEFAULT 'normal',
	
	-- Communication details
	contact_method VARCHAR(50),
	contact_person VARCHAR(255),
	subject VARCHAR(500),
	notes TEXT,
	outcome VARCHAR(50),
	
	-- Follow-up information
	follow_up_date DATE,
	follow_up_assigned_to UUID,
	
	-- APG integration fields
	notification_id UUID,
	document_id UUID,
	workflow_id UUID,
	
	-- Status tracking
	status VARCHAR(50) NOT NULL DEFAULT 'pending',
	completed_date DATE,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	FOREIGN KEY (invoice_id) REFERENCES ar_invoices(id),
	
	-- Constraints
	CONSTRAINT ar_activities_type_check CHECK (activity_type IN ('phone_call', 'email', 'letter', 'sms', 'visit', 'legal_notice', 'settlement_offer')),
	CONSTRAINT ar_activities_method_check CHECK (contact_method IN ('phone', 'email', 'sms', 'mail', 'in_person', 'automated')),
	CONSTRAINT ar_activities_priority_check CHECK (priority IN ('low', 'normal', 'high', 'critical')),
	CONSTRAINT ar_activities_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled', 'escalated')),
	CONSTRAINT ar_activities_outcome_check CHECK (outcome IN ('successful', 'no_response', 'busy', 'promise_to_pay', 'dispute', 'unable_to_pay', 'refused'))
);

-- Credit assessments with APG AI integration
CREATE TABLE ar_credit_assessments (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	
	-- Assessment information
	assessment_date DATE NOT NULL DEFAULT CURRENT_DATE,
	assessment_type VARCHAR(50) NOT NULL DEFAULT 'periodic',
	credit_score INTEGER,
	risk_rating VARCHAR(20),
	recommended_credit_limit DECIMAL(15,2),
	
	-- APG AI/ML integration
	ai_model_version VARCHAR(50),
	ai_confidence_score DECIMAL(5,4),
	risk_factors JSONB,
	
	-- Financial metrics used in assessment
	annual_revenue DECIMAL(15,2),
	debt_to_income_ratio DECIMAL(7,4),
	payment_history_score INTEGER,
	industry_risk_factor DECIMAL(5,4),
	
	-- Assessment outcome
	approval_status VARCHAR(50) NOT NULL DEFAULT 'pending',
	approved_credit_limit DECIMAL(15,2),
	assessment_notes TEXT,
	
	-- Review information
	reviewed_by UUID,
	reviewed_date DATE,
	next_review_date DATE,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	
	-- Constraints
	CONSTRAINT ar_assessments_type_check CHECK (assessment_type IN ('initial', 'periodic', 'triggered', 'manual')),
	CONSTRAINT ar_assessments_rating_check CHECK (risk_rating IN ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D', 'NR')),
	CONSTRAINT ar_assessments_status_check CHECK (approval_status IN ('pending', 'approved', 'rejected', 'under_review')),
	CONSTRAINT ar_assessments_score_check CHECK (credit_score >= 300 AND credit_score <= 850)
);

-- =============================================================================
-- Dispute Management Tables
-- =============================================================================

-- AR Disputes with multi-tenant support
CREATE TABLE ar_disputes (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	invoice_id UUID,
	
	-- Dispute information
	dispute_number VARCHAR(100) NOT NULL,
	dispute_date DATE NOT NULL DEFAULT CURRENT_DATE,
	dispute_type VARCHAR(50) NOT NULL,
	dispute_reason VARCHAR(100) NOT NULL,
	disputed_amount DECIMAL(15,2) NOT NULL,
	
	-- Status and resolution
	status VARCHAR(50) NOT NULL DEFAULT 'open',
	priority VARCHAR(20) NOT NULL DEFAULT 'normal',
	assigned_to UUID,
	
	-- Resolution information
	resolution_date DATE,
	resolution_type VARCHAR(50),
	resolved_amount DECIMAL(15,2),
	resolution_notes TEXT,
	
	-- Communication tracking
	customer_contact_info TEXT,
	internal_notes TEXT,
	
	-- APG document management integration
	supporting_documents JSONB,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	version INTEGER DEFAULT 1,
	
	-- Foreign keys
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	FOREIGN KEY (invoice_id) REFERENCES ar_invoices(id),
	
	-- Constraints
	CONSTRAINT ar_disputes_tenant_number_unique UNIQUE (tenant_id, dispute_number),
	CONSTRAINT ar_disputes_type_check CHECK (dispute_type IN ('billing_error', 'product_quality', 'service_issue', 'pricing_dispute', 'credit_memo_request', 'other')),
	CONSTRAINT ar_disputes_status_check CHECK (status IN ('open', 'under_investigation', 'resolved', 'escalated', 'closed')),
	CONSTRAINT ar_disputes_priority_check CHECK (priority IN ('low', 'normal', 'high', 'critical')),
	CONSTRAINT ar_disputes_resolution_check CHECK (resolution_type IN ('full_credit', 'partial_credit', 'no_adjustment', 'replacement', 'refund')),
	CONSTRAINT ar_disputes_amounts_check CHECK (disputed_amount > 0 AND (resolved_amount IS NULL OR resolved_amount >= 0))
);

-- =============================================================================
-- Cash Application Tables
-- =============================================================================

-- Cash application records with APG AI integration
CREATE TABLE ar_cash_applications (
	id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
	tenant_id UUID NOT NULL,
	payment_id UUID NOT NULL,
	customer_id UUID NOT NULL,
	
	-- Application information
	application_date DATE NOT NULL DEFAULT CURRENT_DATE,
	application_amount DECIMAL(15,2) NOT NULL,
	
	-- APG AI matching information
	matching_method VARCHAR(50) NOT NULL DEFAULT 'manual',
	ai_matching_score DECIMAL(5,4),
	model_version VARCHAR(50),
	auto_applied BOOLEAN DEFAULT false,
	
	-- Matching rules applied
	matching_rules_applied JSONB,
	confidence_factors JSONB,
	
	-- Review information
	requires_review BOOLEAN DEFAULT false,
	reviewed_by UUID,
	reviewed_date DATE,
	review_notes TEXT,
	
	-- Exception handling
	exception_reason VARCHAR(100),
	exception_resolved BOOLEAN DEFAULT false,
	
	-- APG audit fields
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by UUID NOT NULL,
	updated_by UUID NOT NULL,
	
	-- Foreign keys
	FOREIGN KEY (payment_id) REFERENCES ar_payments(id),
	FOREIGN KEY (customer_id) REFERENCES ar_customers(id),
	
	-- Constraints
	CONSTRAINT ar_cash_apps_method_check CHECK (matching_method IN ('manual', 'exact_match', 'fuzzy_match', 'ai_ml', 'rule_based')),
	CONSTRAINT ar_cash_apps_amounts_check CHECK (application_amount > 0)
);

-- =============================================================================
-- Performance Optimization Indexes
-- =============================================================================

-- Customer indexes for performance
CREATE INDEX CONCURRENTLY idx_ar_customers_tenant_id ON ar_customers(tenant_id);
CREATE INDEX CONCURRENTLY idx_ar_customers_status ON ar_customers(status) WHERE status != 'inactive';
CREATE INDEX CONCURRENTLY idx_ar_customers_credit_rating ON ar_customers(credit_rating);
CREATE INDEX CONCURRENTLY idx_ar_customers_code_search ON ar_customers USING gin(customer_code gin_trgm_ops);
CREATE INDEX CONCURRENTLY idx_ar_customers_name_search ON ar_customers USING gin(legal_name gin_trgm_ops);

-- Invoice indexes for performance
CREATE INDEX CONCURRENTLY idx_ar_invoices_tenant_id ON ar_invoices(tenant_id);
CREATE INDEX CONCURRENTLY idx_ar_invoices_customer_id ON ar_invoices(customer_id);
CREATE INDEX CONCURRENTLY idx_ar_invoices_status ON ar_invoices(status);
CREATE INDEX CONCURRENTLY idx_ar_invoices_payment_status ON ar_invoices(payment_status);
CREATE INDEX CONCURRENTLY idx_ar_invoices_date_range ON ar_invoices(invoice_date, due_date);
CREATE INDEX CONCURRENTLY idx_ar_invoices_overdue ON ar_invoices(due_date, status) WHERE status IN ('sent', 'overdue');
CREATE INDEX CONCURRENTLY idx_ar_invoices_number_search ON ar_invoices USING gin(invoice_number gin_trgm_ops);

-- Payment indexes for performance
CREATE INDEX CONCURRENTLY idx_ar_payments_tenant_id ON ar_payments(tenant_id);
CREATE INDEX CONCURRENTLY idx_ar_payments_customer_id ON ar_payments(customer_id);
CREATE INDEX CONCURRENTLY idx_ar_payments_status ON ar_payments(status);
CREATE INDEX CONCURRENTLY idx_ar_payments_date ON ar_payments(payment_date DESC);
CREATE INDEX CONCURRENTLY idx_ar_payments_unapplied ON ar_payments(unapplied_amount) WHERE unapplied_amount > 0;

-- Collection activity indexes
CREATE INDEX CONCURRENTLY idx_ar_activities_tenant_id ON ar_collection_activities(tenant_id);
CREATE INDEX CONCURRENTLY idx_ar_activities_customer_id ON ar_collection_activities(customer_id);
CREATE INDEX CONCURRENTLY idx_ar_activities_date ON ar_collection_activities(activity_date DESC);
CREATE INDEX CONCURRENTLY idx_ar_activities_follow_up ON ar_collection_activities(follow_up_date) WHERE follow_up_date IS NOT NULL;
CREATE INDEX CONCURRENTLY idx_ar_activities_status ON ar_collection_activities(status);

-- Multi-column indexes for common queries
CREATE INDEX CONCURRENTLY idx_ar_invoices_customer_status ON ar_invoices(customer_id, status, due_date);
CREATE INDEX CONCURRENTLY idx_ar_payments_customer_date ON ar_payments(customer_id, payment_date DESC);
CREATE INDEX CONCURRENTLY idx_ar_activities_customer_type ON ar_collection_activities(customer_id, activity_type, activity_date DESC);

-- =============================================================================
-- Multi-Tenant Security Policies (Row Level Security)
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE ar_customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_customer_addresses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_customer_contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_invoices ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_invoice_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_payment_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_collection_activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_credit_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_disputes ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_cash_applications ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
-- Note: In production, these would use APG's authentication context

-- Customer table policies
CREATE POLICY ar_customers_tenant_isolation ON ar_customers
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_customer_addresses_tenant_isolation ON ar_customer_addresses
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_customer_contacts_tenant_isolation ON ar_customer_contacts
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

-- Invoice table policies
CREATE POLICY ar_invoices_tenant_isolation ON ar_invoices
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_invoice_line_items_tenant_isolation ON ar_invoice_line_items
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

-- Payment table policies
CREATE POLICY ar_payments_tenant_isolation ON ar_payments
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_payment_allocations_tenant_isolation ON ar_payment_allocations
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

-- Collection and dispute policies
CREATE POLICY ar_collection_activities_tenant_isolation ON ar_collection_activities
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_credit_assessments_tenant_isolation ON ar_credit_assessments
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_disputes_tenant_isolation ON ar_disputes
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

CREATE POLICY ar_cash_applications_tenant_isolation ON ar_cash_applications
	FOR ALL
	USING (tenant_id = current_setting('apg.current_tenant_id')::UUID);

-- =============================================================================
-- Triggers for Audit Trail and Business Logic
-- =============================================================================

-- Function to update timestamp and version
CREATE OR REPLACE FUNCTION update_modified_columns()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = NOW();
	IF TG_TABLE_NAME IN ('ar_customers', 'ar_invoices', 'ar_payments', 'ar_disputes', 'ar_credit_assessments') THEN
		NEW.version = OLD.version + 1;
	END IF;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to main tables
CREATE TRIGGER tr_ar_customers_update BEFORE UPDATE ON ar_customers
	FOR EACH ROW EXECUTE FUNCTION update_modified_columns();

CREATE TRIGGER tr_ar_invoices_update BEFORE UPDATE ON ar_invoices
	FOR EACH ROW EXECUTE FUNCTION update_modified_columns();

CREATE TRIGGER tr_ar_payments_update BEFORE UPDATE ON ar_payments
	FOR EACH ROW EXECUTE FUNCTION update_modified_columns();

CREATE TRIGGER tr_ar_disputes_update BEFORE UPDATE ON ar_disputes
	FOR EACH ROW EXECUTE FUNCTION update_modified_columns();

CREATE TRIGGER tr_ar_credit_assessments_update BEFORE UPDATE ON ar_credit_assessments
	FOR EACH ROW EXECUTE FUNCTION update_modified_columns();

-- Function to maintain customer outstanding amounts
CREATE OR REPLACE FUNCTION update_customer_outstanding()
RETURNS TRIGGER AS $$
BEGIN
	-- Update customer outstanding balance when invoice amounts change
	IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
		UPDATE ar_customers 
		SET total_outstanding = COALESCE((
			SELECT SUM(balance_amount) 
			FROM ar_invoices 
			WHERE customer_id = NEW.customer_id 
			AND status NOT IN ('cancelled', 'paid')
		), 0),
		overdue_amount = COALESCE((
			SELECT SUM(balance_amount) 
			FROM ar_invoices 
			WHERE customer_id = NEW.customer_id 
			AND status = 'overdue'
		), 0)
		WHERE id = NEW.customer_id;
	END IF;
	
	IF TG_OP = 'DELETE' THEN
		UPDATE ar_customers 
		SET total_outstanding = COALESCE((
			SELECT SUM(balance_amount) 
			FROM ar_invoices 
			WHERE customer_id = OLD.customer_id 
			AND status NOT IN ('cancelled', 'paid')
		), 0),
		overdue_amount = COALESCE((
			SELECT SUM(balance_amount) 
			FROM ar_invoices 
			WHERE customer_id = OLD.customer_id 
			AND status = 'overdue'
		), 0)
		WHERE id = OLD.customer_id;
	END IF;
	
	RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Apply customer outstanding triggers
CREATE TRIGGER tr_ar_invoices_customer_outstanding 
	AFTER INSERT OR UPDATE OR DELETE ON ar_invoices
	FOR EACH ROW EXECUTE FUNCTION update_customer_outstanding();

-- Function to update payment applied amounts
CREATE OR REPLACE FUNCTION update_payment_applied()
RETURNS TRIGGER AS $$
BEGIN
	-- Update payment applied amount when allocations change
	IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
		UPDATE ar_payments 
		SET applied_amount = COALESCE((
			SELECT SUM(allocation_amount) 
			FROM ar_payment_allocations 
			WHERE payment_id = NEW.payment_id
		), 0)
		WHERE id = NEW.payment_id;
	END IF;
	
	IF TG_OP = 'DELETE' THEN
		UPDATE ar_payments 
		SET applied_amount = COALESCE((
			SELECT SUM(allocation_amount) 
			FROM ar_payment_allocations 
			WHERE payment_id = OLD.payment_id
		), 0)
		WHERE id = OLD.payment_id;
	END IF;
	
	RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Apply payment allocation triggers
CREATE TRIGGER tr_ar_allocations_payment_applied 
	AFTER INSERT OR UPDATE OR DELETE ON ar_payment_allocations
	FOR EACH ROW EXECUTE FUNCTION update_payment_applied();

-- =============================================================================
-- Views for Common Queries and Reporting
-- =============================================================================

-- Customer summary view with key metrics
CREATE VIEW v_ar_customer_summary AS
SELECT 
	c.id,
	c.tenant_id,
	c.customer_code,
	c.legal_name,
	c.status,
	c.credit_limit,
	c.credit_rating,
	c.total_outstanding,
	c.overdue_amount,
	c.collection_priority,
	-- Invoice counts
	COALESCE(inv_stats.total_invoices, 0) as total_invoices,
	COALESCE(inv_stats.open_invoices, 0) as open_invoices,
	COALESCE(inv_stats.overdue_invoices, 0) as overdue_invoices,
	-- Payment statistics
	COALESCE(pay_stats.total_payments, 0) as total_payments,
	COALESCE(pay_stats.ytd_payments, 0) as ytd_payments,
	pay_stats.last_payment_date,
	-- Days sales outstanding
	CASE 
		WHEN COALESCE(pay_stats.ytd_payments, 0) > 0 THEN
			ROUND(c.total_outstanding * 365.0 / pay_stats.ytd_payments, 0)
		ELSE NULL 
	END as days_sales_outstanding,
	-- Credit utilization
	CASE 
		WHEN c.credit_limit > 0 THEN
			ROUND((c.total_outstanding / c.credit_limit) * 100, 2)
		ELSE NULL 
	END as credit_utilization_pct
FROM ar_customers c
LEFT JOIN (
	SELECT 
		customer_id,
		COUNT(*) as total_invoices,
		COUNT(*) FILTER (WHERE status NOT IN ('paid', 'cancelled')) as open_invoices,
		COUNT(*) FILTER (WHERE status = 'overdue') as overdue_invoices
	FROM ar_invoices
	GROUP BY customer_id
) inv_stats ON c.id = inv_stats.customer_id
LEFT JOIN (
	SELECT 
		customer_id,
		COUNT(*) as total_payments,
		SUM(payment_amount) FILTER (WHERE EXTRACT(YEAR FROM payment_date) = EXTRACT(YEAR FROM CURRENT_DATE)) as ytd_payments,
		MAX(payment_date) as last_payment_date
	FROM ar_payments 
	WHERE status = 'cleared'
	GROUP BY customer_id
) pay_stats ON c.id = pay_stats.customer_id;

-- Aging report view
CREATE VIEW v_ar_aging_summary AS
SELECT 
	c.id as customer_id,
	c.tenant_id,
	c.customer_code,
	c.legal_name,
	-- Current (not yet due)
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date >= CURRENT_DATE), 0) as current_amount,
	-- 1-30 days past due
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE AND i.due_date >= CURRENT_DATE - INTERVAL '30 days'), 0) as days_1_30,
	-- 31-60 days past due
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '30 days' AND i.due_date >= CURRENT_DATE - INTERVAL '60 days'), 0) as days_31_60,
	-- 61-90 days past due
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '60 days' AND i.due_date >= CURRENT_DATE - INTERVAL '90 days'), 0) as days_61_90,
	-- Over 90 days past due
	COALESCE(SUM(i.balance_amount) FILTER (WHERE i.due_date < CURRENT_DATE - INTERVAL '90 days'), 0) as days_over_90,
	-- Total outstanding
	COALESCE(SUM(i.balance_amount), 0) as total_outstanding
FROM ar_customers c
LEFT JOIN ar_invoices i ON c.id = i.customer_id 
	AND i.balance_amount > 0 
	AND i.status NOT IN ('cancelled', 'paid')
GROUP BY c.id, c.tenant_id, c.customer_code, c.legal_name;

-- Collection workbench view
CREATE VIEW v_ar_collection_workbench AS
SELECT 
	c.id as customer_id,
	c.tenant_id,
	c.customer_code,
	c.legal_name,
	c.collection_priority,
	c.overdue_amount,
	-- Days past due
	CASE 
		WHEN c.overdue_amount > 0 THEN
			(SELECT MAX(CURRENT_DATE - due_date) FROM ar_invoices WHERE customer_id = c.id AND status = 'overdue')
		ELSE 0 
	END as max_days_past_due,
	-- Last collection activity
	last_activity.activity_date as last_activity_date,
	last_activity.activity_type as last_activity_type,
	last_activity.outcome as last_activity_outcome,
	-- Next scheduled activity
	next_activity.activity_date as next_activity_date,
	next_activity.activity_type as next_activity_type,
	-- Contact information
	cc.name as primary_contact_name,
	cc.email as primary_contact_email,
	cc.phone as primary_contact_phone
FROM ar_customers c
LEFT JOIN ar_customer_contacts cc ON c.id = cc.customer_id AND cc.is_primary = true
LEFT JOIN LATERAL (
	SELECT activity_date, activity_type, outcome
	FROM ar_collection_activities
	WHERE customer_id = c.id AND status = 'completed'
	ORDER BY activity_date DESC
	LIMIT 1
) last_activity ON true
LEFT JOIN LATERAL (
	SELECT activity_date, activity_type
	FROM ar_collection_activities
	WHERE customer_id = c.id AND status IN ('pending', 'in_progress')
	ORDER BY activity_date ASC
	LIMIT 1
) next_activity ON true
WHERE c.overdue_amount > 0 OR c.collection_priority IN ('high', 'critical');

-- =============================================================================
-- Initial Reference Data
-- =============================================================================

-- Note: This would be populated by separate data migration scripts
-- Including sample data structure for reference

COMMENT ON SCHEMA apg_accounts_receivable IS 'APG Accounts Receivable capability schema with multi-tenant support and APG platform integration';

-- Migration completion marker
INSERT INTO apg_schema_migrations (schema_name, migration_version, applied_at) 
VALUES ('apg_accounts_receivable', '001_initial_schema', NOW())
ON CONFLICT DO NOTHING;