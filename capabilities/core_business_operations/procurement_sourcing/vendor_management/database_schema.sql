-- APG Vendor Management Database Schema
-- Comprehensive schema supporting AI-powered vendor lifecycle management
-- with multi-tenant architecture, audit trails, and performance optimization

-- ============================================================================
-- CORE VENDOR MANAGEMENT TABLES
-- ============================================================================

-- Vendor Master Table
CREATE TABLE vm_vendor (
	-- Primary Keys & Identification
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_code			VARCHAR(50) NOT NULL,
	
	-- Basic Information  
	name				VARCHAR(200) NOT NULL,
	legal_name			VARCHAR(250),
	display_name		VARCHAR(200),
	
	-- Classification
	vendor_type			VARCHAR(50) NOT NULL DEFAULT 'supplier',
	category			VARCHAR(100) NOT NULL,
	subcategory			VARCHAR(100),
	industry			VARCHAR(100),
	size_classification	VARCHAR(50) DEFAULT 'medium',
	
	-- Status & Lifecycle
	status				VARCHAR(50) NOT NULL DEFAULT 'active',
	lifecycle_stage		VARCHAR(50) NOT NULL DEFAULT 'qualified',
	onboarding_date		TIMESTAMP WITH TIME ZONE,
	activation_date		TIMESTAMP WITH TIME ZONE,
	deactivation_date	TIMESTAMP WITH TIME ZONE,
	
	-- AI-Powered Intelligence Scores
	intelligence_score	DECIMAL(5,2) DEFAULT 85.00 CHECK (intelligence_score >= 0 AND intelligence_score <= 100),
	performance_score	DECIMAL(5,2) DEFAULT 85.00 CHECK (performance_score >= 0 AND performance_score <= 100),
	risk_score			DECIMAL(5,2) DEFAULT 25.00 CHECK (risk_score >= 0 AND risk_score <= 100),
	relationship_score	DECIMAL(5,2) DEFAULT 75.00 CHECK (relationship_score >= 0 AND relationship_score <= 100),
	
	-- Predictive Analytics (JSONB for flexibility)
	predicted_performance	JSONB DEFAULT '{}',
	risk_predictions		JSONB DEFAULT '{}',
	optimization_recommendations JSONB DEFAULT '[]',
	ai_insights				JSONB DEFAULT '{}',
	
	-- Contact Information
	primary_contact_id	UUID,
	email				VARCHAR(255),
	phone				VARCHAR(50),
	website				VARCHAR(255),
	
	-- Address Information  
	address_line1		VARCHAR(255),
	address_line2		VARCHAR(255),
	city				VARCHAR(100),
	state_province		VARCHAR(100),
	postal_code			VARCHAR(20),
	country				VARCHAR(100),
	
	-- Financial Information
	credit_rating		VARCHAR(10),
	payment_terms		VARCHAR(50) DEFAULT 'Net 30',
	currency			VARCHAR(3) DEFAULT 'USD',
	tax_id				VARCHAR(50),
	duns_number			VARCHAR(15),
	
	-- Operational Details (JSONB arrays for flexibility)
	capabilities		JSONB DEFAULT '[]',
	certifications		JSONB DEFAULT '[]',
	geographic_coverage	JSONB DEFAULT '[]',
	capacity_metrics	JSONB DEFAULT '{}',
	
	-- Strategic Information
	strategic_importance VARCHAR(50) DEFAULT 'standard',
	preferred_vendor	BOOLEAN DEFAULT FALSE,
	strategic_partner	BOOLEAN DEFAULT FALSE,
	diversity_category	VARCHAR(100),
	
	-- Multi-tenant & Sharing
	shared_vendor		BOOLEAN DEFAULT FALSE,
	sharing_tenants		UUID[] DEFAULT '{}',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL,
	version				INTEGER DEFAULT 1,
	is_active			BOOLEAN DEFAULT TRUE
);

-- Performance Tracking Table
CREATE TABLE vm_performance (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Performance Period
	measurement_period	VARCHAR(50) NOT NULL, -- 'monthly', 'quarterly', 'annual'
	start_date			TIMESTAMP WITH TIME ZONE NOT NULL,
	end_date			TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Core Performance Metrics
	overall_score		DECIMAL(5,2) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 100),
	quality_score		DECIMAL(5,2) NOT NULL CHECK (quality_score >= 0 AND quality_score <= 100),
	delivery_score		DECIMAL(5,2) NOT NULL CHECK (delivery_score >= 0 AND delivery_score <= 100),
	cost_score			DECIMAL(5,2) NOT NULL CHECK (cost_score >= 0 AND cost_score <= 100),
	service_score		DECIMAL(5,2) NOT NULL CHECK (service_score >= 0 AND service_score <= 100),
	innovation_score	DECIMAL(5,2) DEFAULT 0 CHECK (innovation_score >= 0 AND innovation_score <= 100),
	
	-- Detailed Performance Metrics
	on_time_delivery_rate		DECIMAL(5,2) DEFAULT 0 CHECK (on_time_delivery_rate >= 0 AND on_time_delivery_rate <= 100),
	quality_rejection_rate		DECIMAL(5,2) DEFAULT 0 CHECK (quality_rejection_rate >= 0 AND quality_rejection_rate <= 100),
	cost_variance				DECIMAL(10,2) DEFAULT 0,
	service_level_achievement	DECIMAL(5,2) DEFAULT 0 CHECK (service_level_achievement >= 0 AND service_level_achievement <= 100),
	
	-- Volume & Financial Metrics
	order_volume		DECIMAL(15,2) DEFAULT 0,
	order_count			INTEGER DEFAULT 0,
	total_spend			DECIMAL(15,2) DEFAULT 0,
	average_order_value	DECIMAL(15,2) DEFAULT 0,
	
	-- AI Insights & Analytics
	performance_trends			JSONB DEFAULT '{}',
	improvement_recommendations	JSONB DEFAULT '[]',
	benchmark_comparison		JSONB DEFAULT '{}',
	
	-- Risk Indicators
	risk_indicators		JSONB DEFAULT '[]',
	risk_score			DECIMAL(5,2) DEFAULT 0 CHECK (risk_score >= 0 AND risk_score <= 100),
	mitigation_actions	JSONB DEFAULT '[]',
	
	-- Data Quality & Validation
	data_completeness	DECIMAL(5,2) DEFAULT 100,
	data_sources		JSONB DEFAULT '[]',
	calculation_method	VARCHAR(100) DEFAULT 'weighted_average',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL,
	
	-- Ensure unique performance records per period
	UNIQUE(tenant_id, vendor_id, measurement_period, start_date)
);

-- Risk Management Table
CREATE TABLE vm_risk (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Risk Classification
	risk_type			VARCHAR(50) NOT NULL,
	risk_category		VARCHAR(100) NOT NULL,
	severity			VARCHAR(20) NOT NULL DEFAULT 'medium',
	probability			DECIMAL(3,2) NOT NULL CHECK (probability >= 0 AND probability <= 1),
	impact				VARCHAR(20) NOT NULL DEFAULT 'medium',
	
	-- Risk Details
	title				VARCHAR(200) NOT NULL,
	description			TEXT NOT NULL,
	root_cause			TEXT,
	potential_impact	TEXT,
	
	-- Risk Scoring
	overall_risk_score	DECIMAL(5,2) NOT NULL CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100),
	financial_impact	DECIMAL(15,2) DEFAULT 0,
	operational_impact	INTEGER DEFAULT 5 CHECK (operational_impact >= 1 AND operational_impact <= 10),
	reputational_impact	INTEGER DEFAULT 5 CHECK (reputational_impact >= 1 AND reputational_impact <= 10),
	
	-- AI Predictions
	predicted_likelihood	DECIMAL(3,2) CHECK (predicted_likelihood >= 0 AND predicted_likelihood <= 1),
	time_horizon			INTEGER, -- days
	confidence_level		DECIMAL(3,2) CHECK (confidence_level >= 0 AND confidence_level <= 1),
	ai_risk_factors			JSONB DEFAULT '[]',
	
	-- Mitigation & Response
	mitigation_strategy		TEXT,
	mitigation_actions		JSONB DEFAULT '[]',
	mitigation_status		VARCHAR(50) DEFAULT 'identified',
	target_residual_risk	DECIMAL(5,2) CHECK (target_residual_risk >= 0 AND target_residual_risk <= 100),
	
	-- Monitoring & Review
	monitoring_frequency	VARCHAR(50) DEFAULT 'monthly',
	last_assessment			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	next_assessment			TIMESTAMP WITH TIME ZONE,
	assigned_to				UUID,
	
	-- Status & Lifecycle
	status				VARCHAR(50) DEFAULT 'active',
	identified_date		TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	resolved_date		TIMESTAMP WITH TIME ZONE,
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL
);

-- Contract Management Integration Table
CREATE TABLE vm_contract (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Contract Identification
	contract_number		VARCHAR(100) NOT NULL,
	contract_name		VARCHAR(200) NOT NULL,
	contract_type		VARCHAR(50) NOT NULL,
	
	-- Contract Dates
	effective_date		DATE NOT NULL,
	expiration_date		DATE NOT NULL,
	renewal_date		DATE,
	notice_period_days	INTEGER DEFAULT 30,
	
	-- Financial Terms
	contract_value		DECIMAL(15,2) NOT NULL,
	currency			VARCHAR(3) DEFAULT 'USD',
	payment_terms		VARCHAR(100),
	pricing_model		VARCHAR(50),
	
	-- Contract Status
	status				VARCHAR(50) NOT NULL DEFAULT 'active',
	auto_renewal		BOOLEAN DEFAULT FALSE,
	
	-- Document Management
	document_id			UUID, -- Link to document management system
	contract_terms		JSONB DEFAULT '{}', -- AI-extracted terms
	key_clauses			JSONB DEFAULT '[]',
	
	-- Performance & Compliance
	performance_requirements	JSONB DEFAULT '{}',
	compliance_requirements		JSONB DEFAULT '[]',
	sla_requirements			JSONB DEFAULT '{}',
	
	-- AI Analysis
	risk_analysis		JSONB DEFAULT '{}',
	optimization_opportunities JSONB DEFAULT '[]',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL,
	
	-- Ensure unique contract numbers per tenant
	UNIQUE(tenant_id, contract_number)
);

-- Communication & Collaboration Table
CREATE TABLE vm_communication (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Communication Details
	communication_type	VARCHAR(50) NOT NULL, -- 'email', 'meeting', 'call', 'message'
	subject				VARCHAR(500),
	content				TEXT,
	communication_date	TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- Participants
	internal_participants	UUID[] DEFAULT '{}',
	vendor_participants		JSONB DEFAULT '[]',
	
	-- Communication Metadata
	direction			VARCHAR(20) NOT NULL, -- 'inbound', 'outbound', 'internal'
	priority			VARCHAR(20) DEFAULT 'normal',
	status				VARCHAR(50) DEFAULT 'sent',
	
	-- Related Records
	related_project_id	UUID,
	related_contract_id	UUID,
	related_issue_id	UUID,
	
	-- Attachments & References
	attachments			JSONB DEFAULT '[]',
	references			JSONB DEFAULT '[]',
	
	-- AI Analysis
	sentiment_score		DECIMAL(3,2), -- -1 to 1 scale
	topic_categories	JSONB DEFAULT '[]',
	action_items		JSONB DEFAULT '[]',
	ai_summary			TEXT,
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL
);

-- ============================================================================
-- VENDOR INTELLIGENCE & ANALYTICS TABLES
-- ============================================================================

-- Vendor Intelligence Table
CREATE TABLE vm_intelligence (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Intelligence Generation
	intelligence_date	TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	model_version		VARCHAR(50) NOT NULL,
	confidence_score	DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
	
	-- Intelligence Insights
	behavior_patterns	JSONB NOT NULL DEFAULT '[]',
	predictive_insights	JSONB NOT NULL DEFAULT '[]',
	performance_forecasts JSONB NOT NULL DEFAULT '{}',
	risk_assessments	JSONB NOT NULL DEFAULT '{}',
	
	-- Market Intelligence
	market_position		JSONB DEFAULT '{}',
	competitive_analysis JSONB DEFAULT '{}',
	pricing_intelligence JSONB DEFAULT '{}',
	
	-- Optimization Recommendations
	improvement_opportunities JSONB DEFAULT '[]',
	cost_optimization		JSONB DEFAULT '[]',
	relationship_optimization JSONB DEFAULT '[]',
	
	-- Data Sources & Quality
	data_sources		JSONB NOT NULL DEFAULT '[]',
	data_quality_score	DECIMAL(3,2) DEFAULT 1.0,
	analysis_scope		JSONB DEFAULT '{}',
	
	-- Intelligence Validity
	valid_from			TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	valid_until			TIMESTAMP WITH TIME ZONE NOT NULL,
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL
);

-- Vendor Benchmarking Table
CREATE TABLE vm_benchmark (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Benchmark Configuration
	benchmark_type		VARCHAR(50) NOT NULL, -- 'industry', 'peer', 'internal'
	benchmark_category	VARCHAR(100) NOT NULL,
	measurement_period	VARCHAR(50) NOT NULL,
	
	-- Benchmark Data
	vendor_value		DECIMAL(15,4) NOT NULL,
	benchmark_value		DECIMAL(15,4) NOT NULL,
	percentile_rank		INTEGER CHECK (percentile_rank >= 1 AND percentile_rank <= 100),
	
	-- Benchmark Context
	peer_group_size		INTEGER,
	data_points			INTEGER,
	measurement_unit	VARCHAR(50),
	
	-- Performance Analysis
	performance_gap		DECIMAL(15,4),
	improvement_potential DECIMAL(15,4),
	recommendations		JSONB DEFAULT '[]',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	
	-- Ensure unique benchmarks per vendor/type/period
	UNIQUE(tenant_id, vendor_id, benchmark_type, benchmark_category, measurement_period, created_at)
);

-- ============================================================================
-- VENDOR PORTAL & EXTERNAL ACCESS TABLES
-- ============================================================================

-- Vendor Portal Users Table
CREATE TABLE vm_portal_user (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- User Information
	email				VARCHAR(255) NOT NULL,
	first_name			VARCHAR(100) NOT NULL,
	last_name			VARCHAR(100) NOT NULL,
	job_title			VARCHAR(150),
	phone				VARCHAR(50),
	
	-- Authentication
	password_hash		VARCHAR(255),
	mfa_enabled			BOOLEAN DEFAULT TRUE,
	mfa_secret			VARCHAR(255),
	
	-- Account Status
	status				VARCHAR(50) NOT NULL DEFAULT 'pending_verification',
	email_verified		BOOLEAN DEFAULT FALSE,
	last_login			TIMESTAMP WITH TIME ZONE,
	failed_login_attempts INTEGER DEFAULT 0,
	account_locked_until TIMESTAMP WITH TIME ZONE,
	
	-- Permissions & Access
	role				VARCHAR(50) DEFAULT 'vendor_portal_user',
	permissions			JSONB DEFAULT '[]',
	access_restrictions	JSONB DEFAULT '{}',
	
	-- Security Profile
	allowed_ip_ranges	JSONB DEFAULT '[]',
	require_device_registration BOOLEAN DEFAULT TRUE,
	session_timeout_minutes INTEGER DEFAULT 30,
	
	-- Portal Preferences
	language			VARCHAR(10) DEFAULT 'en',
	timezone			VARCHAR(100) DEFAULT 'UTC',
	notification_preferences JSONB DEFAULT '{}',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL,
	
	-- Ensure unique emails per tenant
	UNIQUE(tenant_id, email)
);

-- Vendor Portal Sessions Table
CREATE TABLE vm_portal_session (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	user_id				UUID NOT NULL REFERENCES vm_portal_user(id) ON DELETE CASCADE,
	vendor_id			UUID NOT NULL,
	
	-- Session Details
	session_token		VARCHAR(255) NOT NULL,
	csrf_token			VARCHAR(255) NOT NULL,
	device_fingerprint	VARCHAR(255),
	
	-- Session Timing
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	expires_at			TIMESTAMP WITH TIME ZONE NOT NULL,
	last_activity		TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	
	-- Request Context
	ip_address			INET,
	user_agent			TEXT,
	
	-- Security Context
	security_context	JSONB DEFAULT '{}',
	
	-- Ensure unique session tokens
	UNIQUE(session_token)
);

-- ============================================================================
-- AUDIT & COMPLIANCE TABLES
-- ============================================================================

-- Vendor Activity Audit Table
CREATE TABLE vm_audit_log (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	
	-- Event Details
	event_type			VARCHAR(100) NOT NULL,
	event_category		VARCHAR(50) NOT NULL,
	event_severity		VARCHAR(20) DEFAULT 'info',
	
	-- Resource Information
	resource_type		VARCHAR(50) NOT NULL,
	resource_id			UUID NOT NULL,
	vendor_id			UUID,
	
	-- User & Session Context
	user_id				UUID,
	session_id			UUID,
	user_type			VARCHAR(50) DEFAULT 'internal',
	
	-- Event Data
	event_data			JSONB NOT NULL DEFAULT '{}',
	old_values			JSONB DEFAULT '{}',
	new_values			JSONB DEFAULT '{}',
	
	-- Request Context
	ip_address			INET,
	user_agent			TEXT,
	request_method		VARCHAR(10),
	request_path		TEXT,
	
	-- Compliance & Audit
	compliance_tags		TEXT[] DEFAULT '{}',
	business_impact		JSONB DEFAULT '{}',
	
	-- Timing
	event_timestamp		TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	
	-- Data Retention (for GDPR compliance)
	retention_until		TIMESTAMP WITH TIME ZONE
);

-- Vendor Compliance Tracking Table
CREATE TABLE vm_compliance (
	-- Primary Keys
	id					UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	tenant_id			UUID NOT NULL,
	vendor_id			UUID NOT NULL REFERENCES vm_vendor(id) ON DELETE CASCADE,
	
	-- Compliance Framework
	framework			VARCHAR(100) NOT NULL, -- 'SOX', 'GDPR', 'PROCUREMENT_REGULATIONS'
	requirement			VARCHAR(200) NOT NULL,
	requirement_type	VARCHAR(50) NOT NULL,
	
	-- Compliance Status
	status				VARCHAR(50) NOT NULL DEFAULT 'compliant',
	compliance_score	DECIMAL(5,2) DEFAULT 100 CHECK (compliance_score >= 0 AND compliance_score <= 100),
	
	-- Evidence & Documentation
	evidence_documents	JSONB DEFAULT '[]',
	compliance_notes	TEXT,
	
	-- Review & Monitoring
	last_review_date	TIMESTAMP WITH TIME ZONE,
	next_review_date	TIMESTAMP WITH TIME ZONE NOT NULL,
	review_frequency	VARCHAR(50) DEFAULT 'annual',
	assigned_reviewer	UUID,
	
	-- Violations & Issues
	violations			JSONB DEFAULT '[]',
	remediation_actions	JSONB DEFAULT '[]',
	
	-- APG Integration & Audit
	created_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	updated_at			TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	created_by			UUID NOT NULL,
	updated_by			UUID NOT NULL,
	
	-- Ensure unique compliance records
	UNIQUE(tenant_id, vendor_id, framework, requirement)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Core Vendor Indexes
CREATE INDEX idx_vm_vendor_tenant_id ON vm_vendor(tenant_id);
CREATE INDEX idx_vm_vendor_vendor_code ON vm_vendor(tenant_id, vendor_code);
CREATE INDEX idx_vm_vendor_category ON vm_vendor(tenant_id, category);
CREATE INDEX idx_vm_vendor_status ON vm_vendor(tenant_id, status);
CREATE INDEX idx_vm_vendor_performance_score ON vm_vendor(tenant_id, performance_score DESC);
CREATE INDEX idx_vm_vendor_risk_score ON vm_vendor(tenant_id, risk_score DESC);
CREATE INDEX idx_vm_vendor_created_at ON vm_vendor(tenant_id, created_at DESC);

-- Performance Indexes
CREATE INDEX idx_vm_performance_vendor_id ON vm_performance(tenant_id, vendor_id);
CREATE INDEX idx_vm_performance_period ON vm_performance(tenant_id, measurement_period, start_date DESC);
CREATE INDEX idx_vm_performance_overall_score ON vm_performance(tenant_id, overall_score DESC);

-- Risk Indexes
CREATE INDEX idx_vm_risk_vendor_id ON vm_risk(tenant_id, vendor_id);
CREATE INDEX idx_vm_risk_severity ON vm_risk(tenant_id, severity, overall_risk_score DESC);
CREATE INDEX idx_vm_risk_category ON vm_risk(tenant_id, risk_category);
CREATE INDEX idx_vm_risk_status ON vm_risk(tenant_id, status);

-- Communication Indexes
CREATE INDEX idx_vm_communication_vendor_id ON vm_communication(tenant_id, vendor_id);
CREATE INDEX idx_vm_communication_date ON vm_communication(tenant_id, communication_date DESC);
CREATE INDEX idx_vm_communication_type ON vm_communication(tenant_id, communication_type);

-- Intelligence Indexes
CREATE INDEX idx_vm_intelligence_vendor_id ON vm_intelligence(tenant_id, vendor_id);
CREATE INDEX idx_vm_intelligence_date ON vm_intelligence(tenant_id, intelligence_date DESC);
CREATE INDEX idx_vm_intelligence_confidence ON vm_intelligence(tenant_id, confidence_score DESC);

-- Portal User Indexes
CREATE INDEX idx_vm_portal_user_vendor_id ON vm_portal_user(tenant_id, vendor_id);
CREATE INDEX idx_vm_portal_user_email ON vm_portal_user(tenant_id, email);
CREATE INDEX idx_vm_portal_user_status ON vm_portal_user(tenant_id, status);

-- Audit Indexes
CREATE INDEX idx_vm_audit_log_tenant_id ON vm_audit_log(tenant_id);
CREATE INDEX idx_vm_audit_log_resource ON vm_audit_log(tenant_id, resource_type, resource_id);
CREATE INDEX idx_vm_audit_log_vendor_id ON vm_audit_log(tenant_id, vendor_id);
CREATE INDEX idx_vm_audit_log_event_timestamp ON vm_audit_log(tenant_id, event_timestamp DESC);
CREATE INDEX idx_vm_audit_log_user_id ON vm_audit_log(tenant_id, user_id);

-- Compliance Indexes  
CREATE INDEX idx_vm_compliance_vendor_id ON vm_compliance(tenant_id, vendor_id);
CREATE INDEX idx_vm_compliance_framework ON vm_compliance(tenant_id, framework);
CREATE INDEX idx_vm_compliance_status ON vm_compliance(tenant_id, status);
CREATE INDEX idx_vm_compliance_next_review ON vm_compliance(tenant_id, next_review_date);

-- ============================================================================
-- ROW LEVEL SECURITY POLICIES
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE vm_vendor ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_risk ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_contract ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_communication ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_benchmark ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_portal_user ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_portal_session ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE vm_compliance ENABLE ROW LEVEL SECURITY;

-- Multi-tenant isolation policies
CREATE POLICY vm_vendor_tenant_isolation ON vm_vendor
	FOR ALL
	USING (
		tenant_id = current_setting('app.current_tenant_id')::uuid
		OR 
		(
			shared_vendor = true 
			AND current_setting('app.current_tenant_id')::uuid = ANY(sharing_tenants)
		)
	);

CREATE POLICY vm_performance_tenant_isolation ON vm_performance
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_risk_tenant_isolation ON vm_risk
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_contract_tenant_isolation ON vm_contract
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_communication_tenant_isolation ON vm_communication
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_intelligence_tenant_isolation ON vm_intelligence
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_benchmark_tenant_isolation ON vm_benchmark
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_portal_user_tenant_isolation ON vm_portal_user
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_audit_log_tenant_isolation ON vm_audit_log
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY vm_compliance_tenant_isolation ON vm_compliance
	FOR ALL
	USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

-- ============================================================================
-- TRIGGERS FOR AUTOMATED TASKS
-- ============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.updated_at = NOW();
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to relevant tables
CREATE TRIGGER update_vm_vendor_updated_at BEFORE UPDATE ON vm_vendor
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_performance_updated_at BEFORE UPDATE ON vm_performance
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_risk_updated_at BEFORE UPDATE ON vm_risk
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_contract_updated_at BEFORE UPDATE ON vm_contract
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_communication_updated_at BEFORE UPDATE ON vm_communication
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_portal_user_updated_at BEFORE UPDATE ON vm_portal_user
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vm_compliance_updated_at BEFORE UPDATE ON vm_compliance
	FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to increment version numbers
CREATE OR REPLACE FUNCTION increment_version_column()
RETURNS TRIGGER AS $$
BEGIN
	NEW.version = OLD.version + 1;
	RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply version increment triggers
CREATE TRIGGER increment_vm_vendor_version BEFORE UPDATE ON vm_vendor
	FOR EACH ROW EXECUTE FUNCTION increment_version_column();

-- ============================================================================
-- PERFORMANCE OPTIMIZATION VIEWS
-- ============================================================================

-- Vendor Performance Summary View
CREATE VIEW vm_vendor_performance_summary AS
SELECT 
	v.tenant_id,
	v.id as vendor_id,
	v.name as vendor_name,
	v.category,
	v.status,
	v.performance_score,
	v.risk_score,
	v.relationship_score,
	p.overall_score as current_performance,
	p.quality_score,
	p.delivery_score,
	p.cost_score,
	p.service_score,
	p.measurement_period,
	p.start_date as performance_period_start,
	p.end_date as performance_period_end,
	CASE 
		WHEN v.performance_score >= 90 THEN 'Excellent'
		WHEN v.performance_score >= 80 THEN 'Good'
		WHEN v.performance_score >= 70 THEN 'Satisfactory'
		WHEN v.performance_score >= 60 THEN 'Needs Improvement'
		ELSE 'Poor'
	END as performance_rating,
	v.updated_at
FROM vm_vendor v
LEFT JOIN LATERAL (
	SELECT * FROM vm_performance p2 
	WHERE p2.vendor_id = v.id AND p2.tenant_id = v.tenant_id
	ORDER BY p2.start_date DESC 
	LIMIT 1
) p ON true
WHERE v.is_active = true;

-- Vendor Risk Dashboard View
CREATE VIEW vm_vendor_risk_dashboard AS
SELECT 
	v.tenant_id,
	v.id as vendor_id,
	v.name as vendor_name,
	v.category,
	v.risk_score,
	COUNT(r.id) as active_risks,
	COUNT(CASE WHEN r.severity = 'high' THEN 1 END) as high_risks,
	COUNT(CASE WHEN r.severity = 'medium' THEN 1 END) as medium_risks,
	COUNT(CASE WHEN r.severity = 'low' THEN 1 END) as low_risks,
	MAX(r.overall_risk_score) as highest_risk_score,
	COUNT(CASE WHEN r.mitigation_status = 'pending' THEN 1 END) as risks_pending_mitigation,
	v.updated_at
FROM vm_vendor v
LEFT JOIN vm_risk r ON r.vendor_id = v.id AND r.tenant_id = v.tenant_id AND r.status = 'active'
WHERE v.is_active = true
GROUP BY v.tenant_id, v.id, v.name, v.category, v.risk_score, v.updated_at;

-- Vendor Intelligence Summary View  
CREATE VIEW vm_vendor_intelligence_summary AS
SELECT 
	v.tenant_id,
	v.id as vendor_id,
	v.name as vendor_name,
	v.intelligence_score,
	i.confidence_score as latest_intelligence_confidence,
	i.intelligence_date as latest_intelligence_date,
	i.behavior_patterns,
	i.predictive_insights,
	i.improvement_opportunities,
	v.optimization_recommendations,
	v.predicted_performance,
	v.updated_at
FROM vm_vendor v
LEFT JOIN LATERAL (
	SELECT * FROM vm_intelligence i2
	WHERE i2.vendor_id = v.id AND i2.tenant_id = v.tenant_id
	ORDER BY i2.intelligence_date DESC
	LIMIT 1
) i ON true
WHERE v.is_active = true;

-- ============================================================================
-- DATABASE COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE vm_vendor IS 'Core vendor master data with AI-powered intelligence scores and multi-tenant support';
COMMENT ON TABLE vm_performance IS 'Vendor performance tracking with comprehensive metrics and AI insights';
COMMENT ON TABLE vm_risk IS 'Vendor risk management with predictive analytics and mitigation tracking';
COMMENT ON TABLE vm_contract IS 'Contract integration table linking vendors to contract management system';
COMMENT ON TABLE vm_communication IS 'Vendor communication history with AI analysis and sentiment tracking';
COMMENT ON TABLE vm_intelligence IS 'AI-generated vendor intelligence insights and recommendations';
COMMENT ON TABLE vm_benchmark IS 'Vendor benchmarking data for performance comparison and analysis';
COMMENT ON TABLE vm_portal_user IS 'External vendor portal users with enhanced security features';
COMMENT ON TABLE vm_portal_session IS 'Secure session management for vendor portal access';
COMMENT ON TABLE vm_audit_log IS 'Comprehensive audit trail for all vendor management activities';
COMMENT ON TABLE vm_compliance IS 'Vendor compliance tracking across multiple regulatory frameworks';

-- Column comments for key fields
COMMENT ON COLUMN vm_vendor.intelligence_score IS 'AI-calculated overall vendor intelligence score (0-100)';
COMMENT ON COLUMN vm_vendor.predicted_performance IS 'JSONB containing AI predictions for future performance';
COMMENT ON COLUMN vm_vendor.risk_predictions IS 'JSONB containing AI risk predictions and scenarios';
COMMENT ON COLUMN vm_vendor.optimization_recommendations IS 'JSONB array of AI-generated optimization recommendations';
COMMENT ON COLUMN vm_vendor.shared_vendor IS 'Boolean indicating if vendor can be shared across tenants';
COMMENT ON COLUMN vm_vendor.sharing_tenants IS 'Array of tenant UUIDs that can access this shared vendor';

-- ============================================================================
-- INITIAL DATA SETUP
-- ============================================================================

-- Insert default vendor categories
INSERT INTO vm_vendor (tenant_id, vendor_code, name, category, vendor_type, created_by, updated_by) VALUES
('00000000-0000-0000-0000-000000000000', 'DEFAULT', 'System Default Vendor', 'system', 'internal', '00000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- SCHEMA VALIDATION QUERIES
-- ============================================================================

-- Query to validate schema setup
SELECT 
	'Schema Setup Complete' as status,
	COUNT(*) as table_count
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE 'vm_%';

-- Query to validate indexes
SELECT 
	'Indexes Created' as status,
	COUNT(*) as index_count
FROM pg_indexes 
WHERE tablename LIKE 'vm_%';

-- Query to validate RLS policies
SELECT 
	'RLS Policies Created' as status,
	COUNT(*) as policy_count
FROM pg_policies 
WHERE tablename LIKE 'vm_%';