-- APG Billing Database Schema
-- Comprehensive PostgreSQL schema for billing capability
-- 
-- © 2025 Datacraft - www.datacraft.co.ke
-- Author: Nyimbi Odero <nyimbi@gmail.com>

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create billing schema
CREATE SCHEMA IF NOT EXISTS billing;
SET search_path TO billing, public;

-- Create enum types
CREATE TYPE billing_currency AS ENUM (
    'USD', 'EUR', 'GBP', 'KES', 'JPY', 'CAD', 'AUD'
);

CREATE TYPE subscription_status AS ENUM (
    'trial', 'active', 'past_due', 'cancelled', 'unpaid', 'paused', 'expired'
);

CREATE TYPE invoice_status AS ENUM (
    'draft', 'pending', 'paid', 'void', 'uncollectible', 'overdue'
);

CREATE TYPE payment_status AS ENUM (
    'pending', 'processing', 'succeeded', 'failed', 'cancelled', 'refunded', 'disputed'
);

CREATE TYPE billing_period AS ENUM (
    'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'usage_based'
);

CREATE TYPE pricing_model AS ENUM (
    'flat_rate', 'tiered', 'volume', 'usage_based', 'freemium', 'hybrid'
);

CREATE TYPE usage_aggregation AS ENUM (
    'sum', 'max', 'last_value', 'unique_count', 'average'
);

CREATE TYPE tax_type AS ENUM (
    'vat', 'gst', 'sales_tax', 'excise', 'custom'
);

-- Customers table
CREATE TABLE bl_customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    external_id VARCHAR(255),
    
    -- Customer Information
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    phone VARCHAR(50),
    
    -- Billing Details
    currency billing_currency DEFAULT 'USD',
    billing_address JSONB DEFAULT '{}',
    tax_info JSONB DEFAULT '{}',
    payment_terms INTEGER,
    
    -- Status
    active BOOLEAN DEFAULT TRUE,
    credit_limit DECIMAL(15,2),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_customers_email_tenant UNIQUE (email, tenant_id),
    CONSTRAINT uk_bl_customers_external_id UNIQUE (external_id) WHERE external_id IS NOT NULL,
    CONSTRAINT chk_bl_customers_credit_limit CHECK (credit_limit >= 0)
);

-- Plans table
CREATE TABLE bl_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    external_id VARCHAR(255),
    
    -- Plan Details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Pricing
    pricing_model pricing_model DEFAULT 'flat_rate',
    base_price DECIMAL(15,2) DEFAULT 0,
    currency billing_currency DEFAULT 'USD',
    billing_period billing_period DEFAULT 'monthly',
    
    -- Usage-based pricing
    usage_charges JSONB DEFAULT '[]',
    included_usage JSONB DEFAULT '{}',
    
    -- Features and Limits
    features JSONB DEFAULT '[]',
    limits JSONB DEFAULT '{}',
    
    -- Trial Settings
    trial_period_days INTEGER,
    trial_requires_payment BOOLEAN DEFAULT FALSE,
    
    -- Plan Management
    active BOOLEAN DEFAULT TRUE,
    version INTEGER DEFAULT 1,
    archived BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_plans_name_tenant UNIQUE (name, tenant_id),
    CONSTRAINT uk_bl_plans_external_id UNIQUE (external_id) WHERE external_id IS NOT NULL,
    CONSTRAINT chk_bl_plans_base_price CHECK (base_price >= 0),
    CONSTRAINT chk_bl_plans_trial_period CHECK (trial_period_days >= 0)
);

-- Subscriptions table
CREATE TABLE bl_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    customer_id UUID NOT NULL REFERENCES bl_customers(id),
    plan_id UUID NOT NULL REFERENCES bl_plans(id),
    external_id VARCHAR(255),
    
    -- Subscription Details
    status subscription_status DEFAULT 'trial',
    
    -- Billing Cycle
    current_period_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    current_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_cycle_anchor TIMESTAMP WITH TIME ZONE,
    
    -- Trial Management
    trial_start TIMESTAMP WITH TIME ZONE,
    trial_end TIMESTAMP WITH TIME ZONE,
    
    -- Pricing Overrides
    price_override DECIMAL(15,2),
    currency_override billing_currency,
    
    -- Usage Tracking
    usage_reset_date TIMESTAMP WITH TIME ZONE,
    included_usage_override JSONB DEFAULT '{}',
    
    -- Discounts and Promotions
    applied_discounts JSONB DEFAULT '[]',
    discount_amount DECIMAL(15,2) DEFAULT 0,
    
    -- Lifecycle Management
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    cancellation_reason TEXT,
    
    -- Pause/Resume
    paused_at TIMESTAMP WITH TIME ZONE,
    pause_reason TEXT,
    
    -- Payment
    default_payment_method VARCHAR(255),
    collection_method VARCHAR(50) DEFAULT 'charge_automatically',
    
    -- Analytics
    churn_score DECIMAL(5,4),
    lifetime_value DECIMAL(15,2),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_subscriptions_external_id UNIQUE (external_id) WHERE external_id IS NOT NULL,
    CONSTRAINT chk_bl_subscriptions_price_override CHECK (price_override >= 0),
    CONSTRAINT chk_bl_subscriptions_discount_amount CHECK (discount_amount >= 0),
    CONSTRAINT chk_bl_subscriptions_churn_score CHECK (churn_score >= 0 AND churn_score <= 1),
    CONSTRAINT chk_bl_subscriptions_lifetime_value CHECK (lifetime_value >= 0)
);

-- Usage table
CREATE TABLE bl_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id UUID NOT NULL REFERENCES bl_subscriptions(id),
    customer_id UUID NOT NULL REFERENCES bl_customers(id),
    
    -- Usage Details
    metric_name VARCHAR(255) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Billing Period
    billing_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    billing_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Pricing Context
    unit_price DECIMAL(15,8),
    total_amount DECIMAL(15,2),
    currency billing_currency DEFAULT 'USD',
    
    -- Aggregation
    aggregation_method usage_aggregation DEFAULT 'sum',
    aggregation_key VARCHAR(255),
    
    -- Context Information
    source_system VARCHAR(255) NOT NULL,
    source_id VARCHAR(255),
    resource_id VARCHAR(255),
    
    -- Processing Status
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    invoice_id UUID,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_bl_usage_quantity CHECK (quantity >= 0),
    CONSTRAINT chk_bl_usage_unit_price CHECK (unit_price >= 0),
    CONSTRAINT chk_bl_usage_total_amount CHECK (total_amount >= 0)
);

-- Invoices table
CREATE TABLE bl_invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    customer_id UUID NOT NULL REFERENCES bl_customers(id),
    subscription_id UUID REFERENCES bl_subscriptions(id),
    external_id VARCHAR(255),
    
    -- Invoice Details
    invoice_number VARCHAR(255) NOT NULL,
    status invoice_status DEFAULT 'draft',
    
    -- Amounts
    subtotal DECIMAL(15,2) DEFAULT 0,
    tax_amount DECIMAL(15,2) DEFAULT 0,
    discount_amount DECIMAL(15,2) DEFAULT 0,
    total DECIMAL(15,2) NOT NULL,
    amount_paid DECIMAL(15,2) DEFAULT 0,
    amount_due DECIMAL(15,2) DEFAULT 0,
    currency billing_currency DEFAULT 'USD',
    
    -- Billing Period
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Invoice Dates
    invoice_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    due_date TIMESTAMP WITH TIME ZONE NOT NULL,
    paid_at TIMESTAMP WITH TIME ZONE,
    
    -- Line Items
    line_items JSONB DEFAULT '[]',
    
    -- Discounts and Taxes
    applied_discounts JSONB DEFAULT '[]',
    tax_details JSONB DEFAULT '[]',
    
    -- Payment Information
    payment_method VARCHAR(255),
    payment_intent_id VARCHAR(255),
    
    -- Document Information
    pdf_url TEXT,
    hosted_url TEXT,
    
    -- Collection
    collection_method VARCHAR(50) DEFAULT 'charge_automatically',
    attempted_collections INTEGER DEFAULT 0,
    next_payment_attempt TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_invoices_number_tenant UNIQUE (invoice_number, tenant_id),
    CONSTRAINT uk_bl_invoices_external_id UNIQUE (external_id) WHERE external_id IS NOT NULL,
    CONSTRAINT chk_bl_invoices_amounts CHECK (
        subtotal >= 0 AND 
        tax_amount >= 0 AND 
        discount_amount >= 0 AND 
        total >= 0 AND 
        amount_paid >= 0 AND 
        amount_due >= 0
    ),
    CONSTRAINT chk_bl_invoices_attempted_collections CHECK (attempted_collections >= 0)
);

-- Payments table
CREATE TABLE bl_payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    customer_id UUID NOT NULL REFERENCES bl_customers(id),
    invoice_id UUID REFERENCES bl_invoices(id),
    external_id VARCHAR(255),
    
    -- Payment Details
    payment_intent_id VARCHAR(255),
    status payment_status DEFAULT 'pending',
    
    -- Amount Information
    amount DECIMAL(15,2) NOT NULL,
    currency billing_currency DEFAULT 'USD',
    fee_amount DECIMAL(15,2),
    net_amount DECIMAL(15,2),
    
    -- Payment Method
    payment_method_type VARCHAR(100) NOT NULL,
    payment_method_id VARCHAR(255),
    payment_processor VARCHAR(100) DEFAULT 'stripe',
    
    -- Processing Information
    processed_at TIMESTAMP WITH TIME ZONE,
    settled_at TIMESTAMP WITH TIME ZONE,
    failure_reason TEXT,
    failure_code VARCHAR(100),
    
    -- Refund Information
    refunded BOOLEAN DEFAULT FALSE,
    refund_amount DECIMAL(15,2) DEFAULT 0,
    refunded_at TIMESTAMP WITH TIME ZONE,
    
    -- Dispute Information
    disputed BOOLEAN DEFAULT FALSE,
    dispute_reason TEXT,
    disputed_at TIMESTAMP WITH TIME ZONE,
    
    -- Risk Assessment
    risk_score DECIMAL(5,4),
    risk_level VARCHAR(20),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_payments_external_id UNIQUE (external_id) WHERE external_id IS NOT NULL,
    CONSTRAINT chk_bl_payments_amount CHECK (amount > 0),
    CONSTRAINT chk_bl_payments_fee_amount CHECK (fee_amount >= 0),
    CONSTRAINT chk_bl_payments_net_amount CHECK (net_amount >= 0),
    CONSTRAINT chk_bl_payments_refund_amount CHECK (refund_amount >= 0),
    CONSTRAINT chk_bl_payments_risk_score CHECK (risk_score >= 0 AND risk_score <= 1)
);

-- Pricing Rules table
CREATE TABLE bl_pricing_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Rule Details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    active BOOLEAN DEFAULT TRUE,
    
    -- Pricing Configuration
    metric_name VARCHAR(255) NOT NULL,
    pricing_tiers JSONB DEFAULT '[]',
    
    -- Conditions
    conditions JSONB DEFAULT '{}',
    customer_segments JSONB DEFAULT '[]',
    
    -- Time-based Rules
    effective_date TIMESTAMP WITH TIME ZONE,
    expiry_date TIMESTAMP WITH TIME ZONE,
    time_based_conditions JSONB DEFAULT '{}',
    
    -- Priority and Conflicts
    priority INTEGER DEFAULT 100,
    conflict_resolution VARCHAR(50) DEFAULT 'highest_priority',
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_pricing_rules_name_tenant UNIQUE (name, tenant_id),
    CONSTRAINT chk_bl_pricing_rules_priority CHECK (priority >= 0)
);

-- Tax table
CREATE TABLE bl_tax (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    invoice_id UUID NOT NULL REFERENCES bl_invoices(id),
    
    -- Tax Details
    tax_type tax_type NOT NULL,
    tax_name VARCHAR(255) NOT NULL,
    tax_rate DECIMAL(8,6) NOT NULL,
    tax_amount DECIMAL(15,2) NOT NULL,
    
    -- Tax Jurisdiction
    country VARCHAR(2) NOT NULL,
    state_province VARCHAR(100),
    city VARCHAR(100),
    postal_code VARCHAR(20),
    
    -- Tax Base
    taxable_amount DECIMAL(15,2) NOT NULL,
    exempt_amount DECIMAL(15,2) DEFAULT 0,
    
    -- Compliance
    tax_id VARCHAR(255),
    reverse_charge BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_bl_tax_rate CHECK (tax_rate >= 0 AND tax_rate <= 1),
    CONSTRAINT chk_bl_tax_amount CHECK (tax_amount >= 0),
    CONSTRAINT chk_bl_tax_taxable_amount CHECK (taxable_amount >= 0),
    CONSTRAINT chk_bl_tax_exempt_amount CHECK (exempt_amount >= 0)
);

-- Discounts table
CREATE TABLE bl_discounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    
    -- Discount Details
    name VARCHAR(255) NOT NULL,
    code VARCHAR(100),
    description TEXT,
    
    -- Discount Type
    discount_type VARCHAR(50) NOT NULL,
    discount_value DECIMAL(15,4) NOT NULL,
    currency billing_currency,
    
    -- Validity
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE,
    active BOOLEAN DEFAULT TRUE,
    
    -- Usage Limits
    max_uses INTEGER,
    max_uses_per_customer INTEGER,
    current_uses INTEGER DEFAULT 0,
    
    -- Applicability
    applicable_plans JSONB DEFAULT '[]',
    customer_segments JSONB DEFAULT '[]',
    minimum_amount DECIMAL(15,2),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uk_bl_discounts_code UNIQUE (code) WHERE code IS NOT NULL,
    CONSTRAINT chk_bl_discounts_value CHECK (discount_value >= 0),
    CONSTRAINT chk_bl_discounts_uses CHECK (
        current_uses >= 0 AND 
        (max_uses IS NULL OR current_uses <= max_uses)
    ),
    CONSTRAINT chk_bl_discounts_minimum_amount CHECK (minimum_amount >= 0)
);

-- Revenue table
CREATE TABLE bl_revenue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    tenant_id VARCHAR(255) NOT NULL,
    subscription_id UUID REFERENCES bl_subscriptions(id),
    invoice_id UUID REFERENCES bl_invoices(id),
    
    -- Revenue Details
    revenue_amount DECIMAL(15,2) NOT NULL,
    currency billing_currency DEFAULT 'USD',
    recognition_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Revenue Category
    revenue_type VARCHAR(100) NOT NULL,
    product_category VARCHAR(100),
    
    -- Recognition Schedule
    deferred_revenue DECIMAL(15,2) DEFAULT 0,
    recognized_revenue DECIMAL(15,2) DEFAULT 0,
    recognition_schedule JSONB DEFAULT '[]',
    
    -- Performance Obligations
    performance_obligations JSONB DEFAULT '[]',
    
    -- Compliance
    accounting_period VARCHAR(20) NOT NULL,
    revenue_stream VARCHAR(255) NOT NULL,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_bl_revenue_amounts CHECK (
        revenue_amount >= 0 AND 
        deferred_revenue >= 0 AND 
        recognized_revenue >= 0
    )
);

-- Create indexes for performance
CREATE INDEX idx_bl_customers_tenant_id ON bl_customers(tenant_id);
CREATE INDEX idx_bl_customers_email ON bl_customers(email);
CREATE INDEX idx_bl_customers_active ON bl_customers(active);

CREATE INDEX idx_bl_plans_tenant_id ON bl_plans(tenant_id);
CREATE INDEX idx_bl_plans_active ON bl_plans(active);
CREATE INDEX idx_bl_plans_pricing_model ON bl_plans(pricing_model);

CREATE INDEX idx_bl_subscriptions_tenant_id ON bl_subscriptions(tenant_id);
CREATE INDEX idx_bl_subscriptions_customer_id ON bl_subscriptions(customer_id);
CREATE INDEX idx_bl_subscriptions_plan_id ON bl_subscriptions(plan_id);
CREATE INDEX idx_bl_subscriptions_status ON bl_subscriptions(status);
CREATE INDEX idx_bl_subscriptions_period_end ON bl_subscriptions(current_period_end);

CREATE INDEX idx_bl_usage_tenant_id ON bl_usage(tenant_id);
CREATE INDEX idx_bl_usage_subscription_id ON bl_usage(subscription_id);
CREATE INDEX idx_bl_usage_customer_id ON bl_usage(customer_id);
CREATE INDEX idx_bl_usage_metric_name ON bl_usage(metric_name);
CREATE INDEX idx_bl_usage_timestamp ON bl_usage(timestamp);
CREATE INDEX idx_bl_usage_processed ON bl_usage(processed);

CREATE INDEX idx_bl_invoices_tenant_id ON bl_invoices(tenant_id);
CREATE INDEX idx_bl_invoices_customer_id ON bl_invoices(customer_id);
CREATE INDEX idx_bl_invoices_subscription_id ON bl_invoices(subscription_id);
CREATE INDEX idx_bl_invoices_status ON bl_invoices(status);
CREATE INDEX idx_bl_invoices_due_date ON bl_invoices(due_date);
CREATE INDEX idx_bl_invoices_invoice_date ON bl_invoices(invoice_date);

CREATE INDEX idx_bl_payments_tenant_id ON bl_payments(tenant_id);
CREATE INDEX idx_bl_payments_customer_id ON bl_payments(customer_id);
CREATE INDEX idx_bl_payments_invoice_id ON bl_payments(invoice_id);
CREATE INDEX idx_bl_payments_status ON bl_payments(status);
CREATE INDEX idx_bl_payments_processed_at ON bl_payments(processed_at);

CREATE INDEX idx_bl_pricing_rules_tenant_id ON bl_pricing_rules(tenant_id);
CREATE INDEX idx_bl_pricing_rules_metric_name ON bl_pricing_rules(metric_name);
CREATE INDEX idx_bl_pricing_rules_active ON bl_pricing_rules(active);
CREATE INDEX idx_bl_pricing_rules_priority ON bl_pricing_rules(priority);

CREATE INDEX idx_bl_tax_invoice_id ON bl_tax(invoice_id);
CREATE INDEX idx_bl_tax_country ON bl_tax(country);

CREATE INDEX idx_bl_discounts_tenant_id ON bl_discounts(tenant_id);
CREATE INDEX idx_bl_discounts_code ON bl_discounts(code);
CREATE INDEX idx_bl_discounts_active ON bl_discounts(active);

CREATE INDEX idx_bl_revenue_tenant_id ON bl_revenue(tenant_id);
CREATE INDEX idx_bl_revenue_subscription_id ON bl_revenue(subscription_id);
CREATE INDEX idx_bl_revenue_invoice_id ON bl_revenue(invoice_id);
CREATE INDEX idx_bl_revenue_recognition_date ON bl_revenue(recognition_date);
CREATE INDEX idx_bl_revenue_accounting_period ON bl_revenue(accounting_period);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_bl_customers_updated_at BEFORE UPDATE ON bl_customers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_plans_updated_at BEFORE UPDATE ON bl_plans FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_subscriptions_updated_at BEFORE UPDATE ON bl_subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_invoices_updated_at BEFORE UPDATE ON bl_invoices FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_payments_updated_at BEFORE UPDATE ON bl_payments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_pricing_rules_updated_at BEFORE UPDATE ON bl_pricing_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_discounts_updated_at BEFORE UPDATE ON bl_discounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bl_revenue_updated_at BEFORE UPDATE ON bl_revenue FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for reporting
CREATE VIEW vw_bl_active_subscriptions AS
SELECT 
    s.*,
    c.name as customer_name,
    c.email as customer_email,
    p.name as plan_name,
    p.base_price,
    p.billing_period
FROM bl_subscriptions s
JOIN bl_customers c ON s.customer_id = c.id
JOIN bl_plans p ON s.plan_id = p.id
WHERE s.status IN ('active', 'trial');

CREATE VIEW vw_bl_revenue_summary AS
SELECT 
    tenant_id,
    DATE_TRUNC('month', recognition_date) as month,
    currency,
    SUM(revenue_amount) as total_revenue,
    SUM(recognized_revenue) as recognized_revenue,
    SUM(deferred_revenue) as deferred_revenue,
    COUNT(*) as revenue_records
FROM bl_revenue
GROUP BY tenant_id, DATE_TRUNC('month', recognition_date), currency;

CREATE VIEW vw_bl_customer_analytics AS
SELECT 
    c.id,
    c.tenant_id,
    c.name,
    c.email,
    COUNT(s.id) as total_subscriptions,
    COUNT(CASE WHEN s.status = 'active' THEN 1 END) as active_subscriptions,
    COALESCE(SUM(i.total), 0) as total_invoiced,
    COALESCE(SUM(i.amount_paid), 0) as total_paid,
    COALESCE(SUM(i.amount_due), 0) as outstanding_amount,
    MAX(i.invoice_date) as last_invoice_date,
    MAX(p.processed_at) as last_payment_date
FROM bl_customers c
LEFT JOIN bl_subscriptions s ON c.id = s.customer_id
LEFT JOIN bl_invoices i ON c.id = i.customer_id
LEFT JOIN bl_payments p ON c.id = p.customer_id AND p.status = 'succeeded'
GROUP BY c.id, c.tenant_id, c.name, c.email;

-- Insert default data
INSERT INTO bl_plans (tenant_id, name, description, pricing_model, base_price, currency, billing_period, features, trial_period_days) VALUES
('default', 'Starter', 'Perfect for small teams getting started', 'flat_rate', 29.99, 'USD', 'monthly', '["Basic features", "Email support", "5 users"]', 14),
('default', 'Professional', 'For growing businesses with advanced needs', 'flat_rate', 99.99, 'USD', 'monthly', '["Advanced features", "Priority support", "25 users", "Analytics"]', 14),
('default', 'Enterprise', 'For large organizations with custom requirements', 'hybrid', 299.99, 'USD', 'monthly', '["All features", "24/7 support", "Unlimited users", "Custom integrations"]', 30);

-- Grant permissions
GRANT USAGE ON SCHEMA billing TO billing_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA billing TO billing_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA billing TO billing_user;

COMMENT ON SCHEMA billing IS 'APG Billing capability database schema';
COMMENT ON TABLE bl_customers IS 'Billing customers with comprehensive customer information';
COMMENT ON TABLE bl_plans IS 'Billing plans with flexible pricing models';
COMMENT ON TABLE bl_subscriptions IS 'Customer subscriptions with lifecycle management';
COMMENT ON TABLE bl_usage IS 'Usage tracking for billing calculations';
COMMENT ON TABLE bl_invoices IS 'Invoice generation and management';
COMMENT ON TABLE bl_payments IS 'Payment processing and tracking';
COMMENT ON TABLE bl_pricing_rules IS 'Dynamic pricing rules engine';
COMMENT ON TABLE bl_tax IS 'Tax calculation and compliance';
COMMENT ON TABLE bl_discounts IS 'Discount and promotion management';
COMMENT ON TABLE bl_revenue IS 'Revenue recognition and accounting';