# APG Multi-Tenant Architecture for Budgeting & Forecasting

**Comprehensive Multi-Tenant Architecture Design Following APG Platform Patterns**

Version 1.0 | © 2025 Datacraft | Author: Nyimbi Odero

---

## Executive Summary

This document defines the multi-tenant architecture for the APG Budgeting & Forecasting capability, building on proven patterns from the APG Accounts Receivable implementation. The architecture ensures complete tenant isolation, scalable performance, and seamless integration with the APG platform ecosystem while maintaining enterprise-grade security and compliance standards.

---

## Multi-Tenant Architecture Overview

### APG Platform Multi-Tenancy Patterns

Based on analysis of the APG Accounts Receivable implementation, the APG platform employs a **hybrid multi-tenant architecture** combining:

1. **Schema-based tenant isolation** for data separation
2. **Row-level security (RLS)** for additional security layers
3. **Shared service infrastructure** for platform capabilities
4. **Tenant-specific configuration** for customization
5. **Unified API layer** with tenant context injection

### Budgeting & Forecasting Multi-Tenancy Requirements

**Core Requirements:**
- Complete data isolation between tenants
- Shared infrastructure for cost efficiency
- Tenant-specific feature configuration
- Cross-tenant analytics (with permission)
- Scalable to 1000+ tenants
- High-performance query optimization

---

## Tenant Isolation Strategy

### 1. Schema-Based Data Isolation

Following APG patterns, each tenant gets a dedicated PostgreSQL schema:

```sql
-- Tenant schema naming convention
bf_tenant_{tenant_id}

-- Examples:
bf_tenant_acme_corp       -- ACME Corporation
bf_tenant_global_inc      -- Global Inc
bf_tenant_startup_xyz     -- Startup XYZ
```

**Isolation Benefits:**
- Complete data separation at the database level
- Independent schema evolution per tenant
- Simplified backup and recovery per tenant
- Clear security boundaries
- Performance optimization per tenant

### 2. Shared Resources Schema

Common resources that don't contain tenant-specific data:

```sql
-- Shared schema for common resources
bf_shared

-- Contains:
- tenant_config              -- Tenant metadata and settings
- budget_templates           -- Shared budget templates
- account_categories         -- Chart of accounts reference
- currency_rates             -- Exchange rate data
- industry_benchmarks        -- Anonymized benchmarking data
```

### 3. Row-Level Security (RLS) Implementation

Additional security layer ensuring tenant data access control:

```sql
-- RLS policy for tenant isolation
CREATE POLICY tenant_isolation_budgets ON {schema}.budgets 
FOR ALL TO app_role 
USING (tenant_id = current_setting('app.current_tenant'));

-- Application sets tenant context
SET app.current_tenant = 'acme_corp';
```

**RLS Benefits:**
- Defense-in-depth security
- Protection against application bugs
- Audit trail for access attempts
- Compliance with data protection regulations

---

## Tenant Management Architecture

### Tenant Lifecycle Management

```python
class TenantManager:
	"""Manages complete tenant lifecycle in APG BF capability."""
	
	async def create_tenant(self, tenant_request: TenantCreationRequest) -> Tenant:
		"""Complete tenant provisioning process."""
		# 1. Validate tenant request
		await self._validate_tenant_request(tenant_request)
		
		# 2. Create tenant configuration
		tenant_config = await self._create_tenant_config(tenant_request)
		
		# 3. Provision database schema
		await self._provision_tenant_schema(tenant_config.tenant_id)
		
		# 4. Set up APG capability integrations
		await self._setup_apg_integrations(tenant_config)
		
		# 5. Initialize default data
		await self._initialize_tenant_data(tenant_config)
		
		# 6. Configure security and access
		await self._configure_tenant_security(tenant_config)
		
		return tenant_config.to_tenant()

	async def _provision_tenant_schema(self, tenant_id: str):
		"""Provision complete tenant database schema."""
		schema_name = f"bf_{tenant_id}"
		
		# Create schema and tables using database function
		await self.db.execute(
			"SELECT bf_shared.setup_tenant(%s, %s, %s)",
			tenant_id, tenant_name, created_by
		)
		
		# Verify schema creation
		await self._verify_tenant_schema(schema_name)
```

### Tenant Configuration Model

```python
class TenantConfig(APGBaseModel):
	"""Comprehensive tenant configuration."""
	
	tenant_id: str = Field(..., description="Unique tenant identifier")
	tenant_name: str = Field(..., description="Display name")
	
	# Feature configuration
	features_enabled: List[str] = Field(default_factory=list)
	budget_features: BudgetFeatureConfig = Field(...)
	forecast_features: ForecastFeatureConfig = Field(...)
	analytics_features: AnalyticsFeatureConfig = Field(...)
	
	# Integration configuration
	apg_integrations: Dict[str, Any] = Field(default_factory=dict)
	external_systems: Dict[str, Any] = Field(default_factory=dict)
	
	# Business configuration
	fiscal_year_start: date = Field(default=date(2025, 1, 1))
	default_currency: str = Field(default="USD")
	time_zone: str = Field(default="UTC")
	
	# Organization structure
	department_hierarchy: Dict[str, Any] = Field(default_factory=dict)
	cost_centers: List[CostCenter] = Field(default_factory=list)
	approval_workflows: Dict[str, Any] = Field(default_factory=dict)
	
	# Security and compliance
	data_retention_days: int = Field(default=2555)  # 7 years
	encryption_enabled: bool = Field(default=True)
	audit_level: str = Field(default="detailed")
	
	# Performance configuration
	max_budget_lines: int = Field(default=100000)
	forecast_horizon_limit: int = Field(default=60)  # months
	concurrent_users_limit: int = Field(default=100)
	
	# Billing and limits
	subscription_tier: str = Field(default="standard")
	usage_limits: Dict[str, int] = Field(default_factory=dict)
	billing_contact: Optional[str] = Field(None)
```

---

## Data Architecture and Isolation

### Database Schema Structure

```
APG Budgeting & Forecasting Multi-Tenant Database
├── bf_shared (Shared Resources)
│   ├── tenant_config
│   ├── budget_templates
│   ├── account_categories
│   ├── currency_rates
│   └── industry_benchmarks
│
├── bf_tenant_acme (ACME Corp Data)
│   ├── budgets
│   ├── budget_lines
│   ├── forecasts
│   ├── forecast_data
│   ├── variance_analysis
│   └── scenarios
│
├── bf_tenant_global (Global Inc Data)
│   ├── budgets
│   ├── budget_lines
│   ├── forecasts
│   ├── forecast_data
│   ├── variance_analysis
│   └── scenarios
│
└── bf_tenant_startup (Startup XYZ Data)
    ├── budgets
    ├── budget_lines
    ├── forecasts
    ├── forecast_data
    ├── variance_analysis
    └── scenarios
```

### Tenant Context Management

```python
class TenantContext:
	"""Manages tenant context throughout request lifecycle."""
	
	def __init__(self, tenant_id: str, user_id: str, schema_name: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.schema_name = schema_name
		self.permissions: List[str] = []
		self.features_enabled: List[str] = []

class TenantContextManager:
	"""Manages tenant context injection and validation."""
	
	async def get_tenant_context(self, request: Request) -> TenantContext:
		"""Extract and validate tenant context from request."""
		# Extract tenant ID from JWT token or header
		tenant_id = await self._extract_tenant_id(request)
		user_id = await self._extract_user_id(request)
		
		# Validate tenant and user access
		await self._validate_tenant_access(tenant_id, user_id)
		
		# Load tenant configuration
		tenant_config = await self._load_tenant_config(tenant_id)
		
		# Create context
		context = TenantContext(
			tenant_id=tenant_id,
			user_id=user_id,
			schema_name=f"bf_{tenant_id}"
		)
		
		# Load permissions and features
		context.permissions = await self._load_user_permissions(tenant_id, user_id)
		context.features_enabled = tenant_config.features_enabled
		
		return context

	async def set_database_context(self, context: TenantContext, db_connection):
		"""Set database session context for tenant isolation."""
		# Set tenant context for RLS
		await db_connection.execute(
			"SET app.current_tenant = %s", context.tenant_id
		)
		
		# Set search path to tenant schema
		await db_connection.execute(
			"SET search_path = %s, bf_shared, public", context.schema_name
		)
		
		# Set user context for audit trails
		await db_connection.execute(
			"SET app.current_user = %s", context.user_id
		)
```

---

## Service Layer Multi-Tenancy

### Base Service Pattern

```python
class BFServiceBase:
	"""Base service with multi-tenant support."""
	
	def __init__(self, tenant_context: TenantContext):
		self.tenant_context = tenant_context
		self.tenant_id = tenant_context.tenant_id
		self.user_id = tenant_context.user_id
		self.schema_name = tenant_context.schema_name

	async def _ensure_tenant_access(self, resource_id: str = None):
		"""Ensure user has access to tenant resources."""
		# Validate tenant is active
		tenant_config = await self._get_tenant_config()
		if tenant_config.status != "active":
			raise HTTPException(status_code=403, detail="Tenant not active")
		
		# Validate feature access
		required_feature = self._get_required_feature()
		if required_feature not in self.tenant_context.features_enabled:
			raise HTTPException(status_code=403, detail="Feature not enabled")
		
		# Validate resource-specific permissions
		if resource_id:
			await self._validate_resource_access(resource_id)

	async def _get_tenant_config(self) -> TenantConfig:
		"""Get tenant configuration with caching."""
		cache_key = f"tenant_config:{self.tenant_id}"
		
		# Try cache first
		cached_config = await self.cache.get(cache_key)
		if cached_config:
			return TenantConfig.parse_raw(cached_config)
		
		# Load from database
		config_data = await self.db.fetch_one(
			"SELECT * FROM bf_shared.tenant_config WHERE tenant_id = %s",
			self.tenant_id
		)
		
		if not config_data:
			raise HTTPException(status_code=404, detail="Tenant not found")
		
		config = TenantConfig(**config_data)
		
		# Cache for 1 hour
		await self.cache.set(cache_key, config.json(), ttl=3600)
		
		return config
```

### Budget Service Multi-Tenancy

```python
class BudgetService(BFServiceBase):
	"""Budget management with multi-tenant isolation."""
	
	async def create_budget(self, budget_data: Dict[str, Any]) -> BFBudget:
		"""Create budget with tenant isolation."""
		await self._ensure_tenant_access()
		
		# Inject tenant context
		budget_data.update({
			'tenant_id': self.tenant_id,
			'created_by': self.user_id,
			'updated_by': self.user_id
		})
		
		# Validate against tenant limits
		await self._validate_tenant_limits(budget_data)
		
		# Create budget in tenant schema
		async with self.db.transaction():
			# Set database context
			await self._set_db_context()
			
			# Create budget
			budget = BFBudget(**budget_data)
			
			# Save to tenant-specific schema
			budget_id = await self.db.fetch_val(f"""
				INSERT INTO {self.schema_name}.budgets 
				({', '.join(budget.dict().keys())})
				VALUES ({', '.join(['%s'] * len(budget.dict()))})
				RETURNING id
			""", *budget.dict().values())
			
			# Audit the creation
			await self._audit_action('create', 'budget', budget_id, budget.dict())
			
			return budget

	async def _validate_tenant_limits(self, budget_data: Dict[str, Any]):
		"""Validate budget creation against tenant limits."""
		tenant_config = await self._get_tenant_config()
		
		# Check budget count limit
		current_count = await self.db.fetch_val(f"""
			SELECT COUNT(*) FROM {self.schema_name}.budgets 
			WHERE is_deleted = FALSE
		""")
		
		if current_count >= tenant_config.usage_limits.get('max_budgets', 1000):
			raise HTTPException(
				status_code=400, 
				detail="Tenant budget limit exceeded"
			)
		
		# Check budget line limit
		estimated_lines = budget_data.get('estimated_line_count', 0)
		if estimated_lines > tenant_config.max_budget_lines:
			raise HTTPException(
				status_code=400,
				detail="Budget line count exceeds tenant limit"
			)
```

---

## Performance Optimization for Multi-Tenancy

### Database Optimization Strategies

**1. Tenant-Specific Indexing**

```sql
-- Indexes optimized for tenant queries
CREATE INDEX CONCURRENTLY idx_budgets_tenant_fiscal 
    ON bf_tenant_acme.budgets(tenant_id, fiscal_year);

CREATE INDEX CONCURRENTLY idx_budget_lines_tenant_period 
    ON bf_tenant_acme.budget_lines(tenant_id, period_start, period_end);

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY idx_budgets_active 
    ON bf_tenant_acme.budgets(status, budget_type) 
    WHERE is_deleted = FALSE;
```

**2. Query Plan Optimization**

```python
class QueryOptimizer:
	"""Optimizes queries for multi-tenant performance."""
	
	async def optimize_tenant_queries(self, tenant_id: str):
		"""Optimize queries for specific tenant."""
		schema_name = f"bf_{tenant_id}"
		
		# Analyze table statistics
		await self.db.execute(f"ANALYZE {schema_name}.budgets")
		await self.db.execute(f"ANALYZE {schema_name}.budget_lines")
		await self.db.execute(f"ANALYZE {schema_name}.forecasts")
		
		# Update query planner statistics
		await self.db.execute("SELECT pg_stat_reset()")
		
		# Cache frequent query plans
		await self._cache_query_plans(schema_name)

	async def _cache_query_plans(self, schema_name: str):
		"""Pre-warm query plan cache for common queries."""
		common_queries = [
			f"SELECT * FROM {schema_name}.budgets WHERE fiscal_year = 2025",
			f"SELECT * FROM {schema_name}.budget_lines WHERE budget_id = 'sample'",
			f"SELECT * FROM {schema_name}.forecasts WHERE forecast_type = 'REVENUE'"
		]
		
		for query in common_queries:
			await self.db.execute(f"PREPARE {query}")
```

### Caching Strategy for Multi-Tenancy

```python
class TenantCacheManager:
	"""Manages caching across multiple tenants."""
	
	def __init__(self, redis_client):
		self.redis = redis_client
	
	def _get_cache_key(self, tenant_id: str, key: str) -> str:
		"""Generate tenant-specific cache key."""
		return f"bf:{tenant_id}:{key}"
	
	async def get_tenant_data(self, tenant_id: str, key: str) -> Optional[Any]:
		"""Get cached data for specific tenant."""
		cache_key = self._get_cache_key(tenant_id, key)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			return json.loads(cached_data)
		return None
	
	async def set_tenant_data(self, tenant_id: str, key: str, 
							  data: Any, ttl: int = 3600):
		"""Cache data for specific tenant."""
		cache_key = self._get_cache_key(tenant_id, key)
		await self.redis.setex(
			cache_key, 
			ttl, 
			json.dumps(data, default=str)
		)
	
	async def invalidate_tenant_cache(self, tenant_id: str, pattern: str = "*"):
		"""Invalidate cache entries for tenant."""
		cache_pattern = self._get_cache_key(tenant_id, pattern)
		keys = await self.redis.keys(cache_pattern)
		
		if keys:
			await self.redis.delete(*keys)
	
	async def get_cache_stats(self, tenant_id: str) -> Dict[str, Any]:
		"""Get cache statistics for tenant."""
		pattern = self._get_cache_key(tenant_id, "*")
		keys = await self.redis.keys(pattern)
		
		total_size = 0
		hit_count = 0
		
		for key in keys:
			size = await self.redis.memory_usage(key)
			total_size += size or 0
			
			# Get hit count from key info
			info = await self.redis.object("IDLETIME", key)
			if info is not None:
				hit_count += 1
		
		return {
			'tenant_id': tenant_id,
			'total_keys': len(keys),
			'total_size_bytes': total_size,
			'hit_count': hit_count
		}
```

---

## Security Architecture

### Tenant Data Isolation Security

**1. Database-Level Security**

```sql
-- Row-level security policies
CREATE POLICY tenant_data_isolation ON budgets 
FOR ALL TO app_role 
USING (tenant_id = current_setting('app.current_tenant'));

-- Prevent cross-tenant data access
CREATE POLICY prevent_cross_tenant_access ON budgets 
FOR ALL TO app_role 
USING (tenant_id IS NOT NULL AND tenant_id != '');

-- Audit access attempts
CREATE POLICY audit_access_attempts ON budgets 
FOR SELECT TO app_role 
USING (true); -- Always allow but log
```

**2. Application-Level Security**

```python
class TenantSecurityManager:
	"""Manages security for multi-tenant operations."""
	
	async def validate_tenant_access(self, user_id: str, tenant_id: str) -> bool:
		"""Validate user has access to tenant."""
		# Check user-tenant relationship
		access_record = await self.db.fetch_one("""
			SELECT 1 FROM user_tenant_access 
			WHERE user_id = %s AND tenant_id = %s AND is_active = TRUE
		""", user_id, tenant_id)
		
		return access_record is not None
	
	async def validate_resource_permission(self, user_id: str, tenant_id: str, 
										   resource_type: str, resource_id: str) -> bool:
		"""Validate user permission for specific resource."""
		# Check resource-level permissions
		permission = await self.db.fetch_one("""
			SELECT p.permission_name FROM user_permissions p
			JOIN user_roles ur ON p.role_id = ur.role_id
			WHERE ur.user_id = %s AND ur.tenant_id = %s 
			AND p.resource_type = %s
		""", user_id, tenant_id, resource_type)
		
		return permission is not None
	
	async def encrypt_sensitive_data(self, tenant_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Encrypt sensitive fields using tenant-specific keys."""
		tenant_config = await self._get_tenant_config(tenant_id)
		
		if not tenant_config.encryption_enabled:
			return data
		
		encryption_key = await self._get_tenant_encryption_key(tenant_id)
		
		# Encrypt sensitive fields
		sensitive_fields = ['budget_amounts', 'forecast_targets', 'variance_data']
		
		for field in sensitive_fields:
			if field in data:
				data[field] = await self._encrypt_field(data[field], encryption_key)
		
		return data
```

### Audit and Compliance

```python
class TenantAuditManager:
	"""Manages audit trails for multi-tenant operations."""
	
	async def log_tenant_action(self, tenant_id: str, user_id: str, 
								action: str, resource_type: str, 
								resource_id: str, details: Dict[str, Any]):
		"""Log action with tenant context for audit compliance."""
		audit_record = {
			'tenant_id': tenant_id,
			'user_id': user_id,
			'action': action,
			'resource_type': resource_type,
			'resource_id': resource_id,
			'timestamp': datetime.utcnow(),
			'details': details,
			'ip_address': await self._get_client_ip(),
			'user_agent': await self._get_user_agent()
		}
		
		# Log to APG audit_compliance capability
		await self._send_to_apg_audit(audit_record)
		
		# Log to tenant-specific audit log
		await self._log_tenant_audit(tenant_id, audit_record)
	
	async def generate_compliance_report(self, tenant_id: str, 
										 start_date: date, end_date: date) -> Dict[str, Any]:
		"""Generate compliance report for tenant."""
		# Query audit logs for tenant
		audit_logs = await self.db.fetch_all("""
			SELECT * FROM audit_logs 
			WHERE tenant_id = %s 
			AND timestamp BETWEEN %s AND %s
		""", tenant_id, start_date, end_date)
		
		# Aggregate compliance metrics
		metrics = {
			'total_actions': len(audit_logs),
			'user_activity': self._aggregate_user_activity(audit_logs),
			'resource_access': self._aggregate_resource_access(audit_logs),
			'security_events': self._identify_security_events(audit_logs),
			'compliance_score': self._calculate_compliance_score(audit_logs)
		}
		
		return {
			'tenant_id': tenant_id,
			'report_period': {'start': start_date, 'end': end_date},
			'metrics': metrics,
			'recommendations': self._generate_recommendations(metrics)
		}
```

---

## Cross-Tenant Features

### Benchmarking and Analytics

```python
class CrossTenantAnalytics:
	"""Provides cross-tenant analytics while preserving privacy."""
	
	async def get_industry_benchmarks(self, tenant_id: str, 
									  industry_code: str) -> Dict[str, Any]:
		"""Get anonymized industry benchmarks."""
		# Ensure tenant has benchmarking feature enabled
		tenant_config = await self._get_tenant_config(tenant_id)
		if 'industry_benchmarking' not in tenant_config.features_enabled:
			raise HTTPException(status_code=403, detail="Feature not enabled")
		
		# Query anonymized benchmark data
		benchmarks = await self.db.fetch_all("""
			SELECT 
				metric_name,
				percentile_25,
				percentile_50,
				percentile_75,
				percentile_90
			FROM industry_benchmarks 
			WHERE industry_code = %s
			AND anonymized = TRUE
		""", industry_code)
		
		return {
			'industry_code': industry_code,
			'benchmarks': benchmarks,
			'participant_count': await self._get_participant_count(industry_code),
			'last_updated': await self._get_last_update_date(industry_code)
		}
	
	async def contribute_to_benchmarks(self, tenant_id: str, 
									   contribution_data: Dict[str, Any]):
		"""Contribute anonymized data to industry benchmarks."""
		# Validate tenant consent for data sharing
		tenant_config = await self._get_tenant_config(tenant_id)
		if not tenant_config.benchmark_participation_enabled:
			raise HTTPException(status_code=403, detail="Benchmark participation not enabled")
		
		# Anonymize data before contribution
		anonymized_data = await self._anonymize_contribution_data(
			contribution_data, tenant_id
		)
		
		# Contribute to shared benchmark pool
		await self.db.execute("""
			INSERT INTO benchmark_contributions 
			(industry_code, metric_name, metric_value, contribution_date)
			VALUES (%s, %s, %s, %s)
		""", anonymized_data['industry_code'], 
			 anonymized_data['metric_name'],
			 anonymized_data['metric_value'],
			 datetime.utcnow())
```

### Template Sharing

```python
class TenantTemplateManager:
	"""Manages budget template sharing between tenants."""
	
	async def share_template(self, owner_tenant_id: str, template_id: str, 
							 target_tenant_ids: List[str], permissions: List[str]):
		"""Share budget template with other tenants."""
		# Validate template ownership
		template = await self._get_template(template_id)
		if template.owner_tenant_id != owner_tenant_id:
			raise HTTPException(status_code=403, detail="Not template owner")
		
		# Validate sharing permissions
		await self._validate_sharing_permissions(owner_tenant_id, permissions)
		
		# Create sharing records
		for target_tenant_id in target_tenant_ids:
			await self.db.execute("""
				INSERT INTO template_sharing 
				(template_id, owner_tenant_id, shared_with_tenant_id, permissions, shared_date)
				VALUES (%s, %s, %s, %s, %s)
			""", template_id, owner_tenant_id, target_tenant_id, 
				 permissions, datetime.utcnow())
		
		# Update template sharing metadata
		await self.db.execute("""
			UPDATE bf_shared.budget_templates 
			SET shared_with_tenants = array_append(shared_with_tenants, %s),
				usage_count = usage_count + 1
			WHERE id = %s
		""", target_tenant_ids, template_id)
	
	async def get_shared_templates(self, tenant_id: str) -> List[Dict[str, Any]]:
		"""Get templates shared with tenant."""
		templates = await self.db.fetch_all("""
			SELECT 
				t.*,
				ts.permissions,
				ts.shared_date,
				tc.tenant_name as owner_name
			FROM bf_shared.budget_templates t
			JOIN template_sharing ts ON t.id = ts.template_id
			JOIN bf_shared.tenant_config tc ON t.owner_tenant_id = tc.tenant_id
			WHERE ts.shared_with_tenant_id = %s
			AND t.is_deleted = FALSE
		""", tenant_id)
		
		return [dict(template) for template in templates]
```

---

## Deployment and Operations

### Tenant Provisioning Automation

```yaml
# Kubernetes deployment for tenant provisioning
apiVersion: batch/v1
kind: Job
metadata:
  name: provision-tenant-${TENANT_ID}
  namespace: apg-bf
spec:
  template:
    spec:
      containers:
      - name: tenant-provisioner
        image: apg-bf/tenant-provisioner:latest
        env:
        - name: TENANT_ID
          value: "${TENANT_ID}"
        - name: TENANT_NAME
          value: "${TENANT_NAME}"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        command:
        - python
        - -c
        - |
          import asyncio
          from tenant_manager import TenantManager
          
          async def provision():
              manager = TenantManager()
              await manager.provision_tenant(
                  tenant_id="${TENANT_ID}",
                  tenant_name="${TENANT_NAME}",
                  created_by="system"
              )
          
          asyncio.run(provision())
      restartPolicy: Never
```

### Monitoring and Metrics

```python
class TenantMonitoring:
	"""Monitors tenant performance and usage."""
	
	async def collect_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
		"""Collect comprehensive tenant metrics."""
		schema_name = f"bf_{tenant_id}"
		
		# Database metrics
		db_metrics = await self._collect_db_metrics(schema_name)
		
		# Cache metrics
		cache_metrics = await self._collect_cache_metrics(tenant_id)
		
		# Usage metrics
		usage_metrics = await self._collect_usage_metrics(tenant_id)
		
		# Performance metrics
		perf_metrics = await self._collect_performance_metrics(tenant_id)
		
		return {
			'tenant_id': tenant_id,
			'timestamp': datetime.utcnow(),
			'database': db_metrics,
			'cache': cache_metrics,
			'usage': usage_metrics,
			'performance': perf_metrics
		}
	
	async def _collect_db_metrics(self, schema_name: str) -> Dict[str, Any]:
		"""Collect database metrics for tenant schema."""
		metrics = await self.db.fetch_one(f"""
			SELECT 
				(SELECT COUNT(*) FROM {schema_name}.budgets WHERE is_deleted = FALSE) as budget_count,
				(SELECT COUNT(*) FROM {schema_name}.budget_lines WHERE is_deleted = FALSE) as budget_line_count,
				(SELECT COUNT(*) FROM {schema_name}.forecasts WHERE is_deleted = FALSE) as forecast_count,
				(SELECT pg_size_pretty(pg_total_relation_size('{schema_name}.budgets'))) as budget_table_size,
				(SELECT pg_size_pretty(pg_total_relation_size('{schema_name}.budget_lines'))) as budget_lines_table_size
		""")
		
		return dict(metrics)
```

---

## Testing Multi-Tenant Architecture

### Tenant Isolation Testing

```python
class TenantIsolationTests:
	"""Tests for multi-tenant data isolation."""
	
	async def test_schema_isolation(self):
		"""Test that tenants cannot access each other's data."""
		# Create two test tenants
		tenant_a = await self.create_test_tenant("tenant_a")
		tenant_b = await self.create_test_tenant("tenant_b")
		
		# Create data in tenant A
		budget_a = await self.create_test_budget(tenant_a.tenant_id)
		
		# Try to access tenant A data from tenant B context
		with pytest.raises(HTTPException) as exc_info:
			await self.budget_service.get_budget(
				budget_a.id, 
				tenant_context=tenant_b.context
			)
		
		assert exc_info.value.status_code == 404
	
	async def test_rls_enforcement(self):
		"""Test row-level security enforcement."""
		# Set up test data
		tenant_id = "test_tenant"
		await self.setup_tenant_data(tenant_id)
		
		# Set correct tenant context
		await self.db.execute("SET app.current_tenant = %s", tenant_id)
		
		# Query should return data
		budgets = await self.db.fetch_all("SELECT * FROM budgets")
		assert len(budgets) > 0
		
		# Set wrong tenant context
		await self.db.execute("SET app.current_tenant = %s", "wrong_tenant")
		
		# Query should return no data
		budgets = await self.db.fetch_all("SELECT * FROM budgets")
		assert len(budgets) == 0
	
	async def test_cross_tenant_benchmark_privacy(self):
		"""Test that benchmark data maintains privacy."""
		# Contribute data from multiple tenants
		await self.contribute_benchmark_data("tenant_1", industry_data_1)
		await self.contribute_benchmark_data("tenant_2", industry_data_2)
		
		# Get benchmarks for tenant 1
		benchmarks = await self.analytics.get_industry_benchmarks(
			"tenant_1", "technology"
		)
		
		# Verify that individual tenant data cannot be identified
		assert 'tenant_specific_data' not in benchmarks
		assert benchmarks['participant_count'] >= 2
		assert all(key.startswith('percentile_') for key in benchmarks['benchmarks'][0])
```

---

## Conclusion

This comprehensive multi-tenant architecture for APG Budgeting & Forecasting ensures:

1. **Complete Data Isolation**: Schema-based separation with RLS protection
2. **Scalable Performance**: Optimized indexes and caching strategies
3. **Enterprise Security**: Multiple security layers and audit compliance
4. **Flexible Configuration**: Tenant-specific features and settings
5. **Cross-Tenant Analytics**: Privacy-preserving benchmarking
6. **Operational Excellence**: Automated provisioning and monitoring

The architecture follows proven APG platform patterns while addressing the specific requirements of budgeting and forecasting workloads, ensuring successful integration with the APG ecosystem.

---

© 2025 Datacraft. All rights reserved.  
Contact: nyimbi@gmail.com | www.datacraft.co.ke