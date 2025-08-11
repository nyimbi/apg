# APG Data Virtualization (DVRL) Developer Guide

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [APG Integration Patterns](#apg-integration-patterns)
4. [Core Components](#core-components)
5. [Development Workflows](#development-workflows)
6. [Testing Strategies](#testing-strategies)
7. [Performance Optimization](#performance-optimization)
8. [Security Implementation](#security-implementation)
9. [Extension Points](#extension-points)
10. [Best Practices](#best-practices)
11. [Code Examples](#code-examples)

## Development Environment Setup

### Prerequisites

```bash
# Required tools
python --version  # 3.11+
docker --version  # 24.0+
kubectl version   # 1.28+
helm version      # 3.12+

# APG CLI tools
pip install apg-cli
apg auth login
```

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/apg-platform/dvrl-capability.git
cd dvrl-capability

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Set up local APG environment
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
uv run alembic upgrade head
```

### Environment Configuration

```bash
# .env.development
export TENANT_ID="dev-tenant"
export APG_BASE_URL="http://localhost:8080"
export DATABASE_URL="postgresql://dvrl_dev:password@localhost:5432/dvrl_dev"
export REDIS_URL="redis://localhost:6379/0"
export LOG_LEVEL="DEBUG"
export DVRL_DEBUG="true"
```

### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

**PyCharm Configuration**:
- Enable async/await syntax highlighting
- Configure pytest as the test runner
- Set up Docker integration for testing

## Architecture Overview

### High-Level Architecture

```mermaid
graph TB
    UI[Web Interface] --> API[REST API Layer]
    API --> Auth[APG Auth Service]
    API --> Service[DVRL Service Layer]
    Service --> Parser[SQL Parser]
    Service --> Optimizer[Query Optimizer] 
    Service --> Executor[Federation Executor]
    Service --> Cache[APG Cache Service]
    Service --> Meta[APG Meta Service]
    Executor --> Connectors[Data Source Connectors]
    Connectors --> DS1[PostgreSQL]
    Connectors --> DS2[MySQL]
    Connectors --> DS3[MongoDB]
```

### Component Layers

1. **API Layer** (`api.py`): REST endpoints, request/response handling
2. **Service Layer** (`service.py`): Business logic, orchestration
3. **Data Layer** (`models.py`): Pydantic models, data validation
4. **Connector Layer** (`connectors.py`): Data source abstractions
5. **Integration Layer** (`apg_integrations.py`): APG platform services

### Key Design Patterns

- **Repository Pattern**: Data access abstraction
- **Strategy Pattern**: Pluggable query execution strategies
- **Observer Pattern**: Event-driven metadata updates
- **Factory Pattern**: Data source connector creation
- **Decorator Pattern**: Authentication, caching, error handling

## APG Integration Patterns

### Authentication Integration

```python
from apg_platform import AuthClient
from functools import wraps

class APGAuthMixin:
    def __init__(self):
        self.auth_client = AuthClient(base_url=os.environ['APG_BASE_URL'])
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate APG token and return user context"""
        try:
            user_info = await self.auth_client.validate_token(token)
            return User(
                user_id=user_info['sub'],
                tenant_id=user_info['tenant_id'],
                roles=user_info.get('roles', []),
                permissions=user_info.get('permissions', [])
            )
        except Exception as e:
            self._log_error(f"Token validation failed: {e}")
            return None

def require_permission(permission: str):
    """Decorator for permission-based access control"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'current_user'):
                raise PermissionError("Authentication required")
            
            if permission not in self.current_user.permissions:
                raise PermissionError(f"Permission required: {permission}")
            
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator
```

### Metadata Service Integration

```python
from apg_platform import MetaClient

class MetadataIntegration:
    def __init__(self, tenant_id: str):
        self.meta_client = MetaClient(
            base_url=os.environ['APG_BASE_URL'],
            tenant_id=tenant_id
        )
    
    async def register_schema(self, schema: DataSourceSchema) -> None:
        """Register discovered schema with APG metadata service"""
        metadata = {
            'schema_name': schema.schema_name,
            'data_source_id': schema.data_source_id,
            'tables': [
                {
                    'name': table.name,
                    'columns': [
                        {
                            'name': col.name,
                            'type': col.data_type,
                            'nullable': col.is_nullable,
                            'primary_key': col.is_primary_key
                        } for col in table.columns
                    ],
                    'row_count': table.estimated_row_count
                } for table in schema.tables
            ],
            'discovered_at': datetime.utcnow().isoformat()
        }
        
        await self.meta_client.register_schema(schema.data_source_id, metadata)
    
    async def track_query_lineage(self, query: FederatedQuery) -> None:
        """Track data lineage for federated queries"""
        lineage = {
            'query_id': query.id,
            'sql': query.sql,
            'source_tables': [
                f"{ds}.{table}" for ds, table in query.tables_accessed
            ],
            'output_schema': query.result_schema,
            'executed_at': query.executed_at.isoformat(),
            'execution_time_ms': query.execution_time_ms
        }
        
        await self.meta_client.track_lineage(lineage)
```

### Cache Service Integration

```python
from apg_platform import CacheClient

class IntelligentCaching:
    def __init__(self, tenant_id: str):
        self.cache_client = CacheClient(
            base_url=os.environ['APG_BASE_URL'],
            tenant_id=tenant_id
        )
        self.local_cache = {}
    
    async def get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result with multi-level lookup"""
        # Level 1: Local memory cache
        if query_hash in self.local_cache:
            entry = self.local_cache[query_hash]
            if entry['expires_at'] > datetime.utcnow():
                return entry['result']
        
        # Level 2: APG distributed cache
        result = await self.cache_client.get(f"dvrl:query:{query_hash}")
        if result:
            # Populate local cache
            self.local_cache[query_hash] = {
                'result': result['data'],
                'expires_at': datetime.fromisoformat(result['expires_at'])
            }
            return result['data']
        
        return None
    
    async def cache_result(self, query_hash: str, result: Dict[str, Any], 
                          ttl_seconds: int = 3600) -> None:
        """Cache query result with intelligent TTL"""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        # Cache locally
        self.local_cache[query_hash] = {
            'result': result,
            'expires_at': expires_at
        }
        
        # Cache in APG distributed cache
        await self.cache_client.set(
            f"dvrl:query:{query_hash}",
            {
                'data': result,
                'expires_at': expires_at.isoformat(),
                'query_complexity': self._calculate_complexity(result)
            },
            ttl=ttl_seconds
        )
    
    def _calculate_complexity(self, result: Dict[str, Any]) -> str:
        """Calculate query complexity for intelligent caching"""
        row_count = len(result.get('rows', []))
        if row_count > 10000:
            return 'high'
        elif row_count > 1000:
            return 'medium'
        else:
            return 'low'
```

## Core Components

### SQL Parser and Analyzer

```python
class SQLParser:
    """Advanced SQL parser with federation support"""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        
    async def parse_query(self, sql: str) -> ParsedQuery:
        """Parse SQL with comprehensive analysis"""
        # Normalize SQL
        normalized_sql = self._normalize_sql(sql)
        
        # Extract components
        query_type = self._extract_query_type(normalized_sql)
        tables = await self._extract_tables(normalized_sql)
        columns = await self._extract_columns(normalized_sql)
        joins = await self._extract_joins(normalized_sql)
        conditions = await self._extract_conditions(normalized_sql)
        
        # Analyze complexity
        complexity = await self._analyze_complexity(normalized_sql)
        
        return ParsedQuery(
            original_sql=sql,
            normalized_sql=normalized_sql,
            query_type=query_type,
            tables=tables,
            columns=columns,
            joins=joins,
            conditions=conditions,
            complexity=complexity,
            estimated_cost=self._estimate_cost(complexity)
        )
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for consistent parsing"""
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Standardize keywords
        keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY']
        for keyword in keywords:
            sql = re.sub(rf'\b{keyword.lower()}\b', keyword, sql, flags=re.IGNORECASE)
        
        return sql
```

### Query Optimizer

```python
class QueryOptimizer:
    """ML-powered query optimizer for federated queries"""
    
    def __init__(self):
        self.cost_model = self._load_cost_model()
        self.optimization_rules = self._load_rules()
    
    async def optimize(self, parsed_query: ParsedQuery, 
                      data_sources: Dict[str, DataSource]) -> OptimizedQuery:
        """Optimize query using cost-based optimization"""
        
        # Generate multiple execution plans
        plans = await self._generate_plans(parsed_query, data_sources)
        
        # Cost each plan
        costed_plans = []
        for plan in plans:
            cost = await self._estimate_plan_cost(plan, data_sources)
            costed_plans.append((plan, cost))
        
        # Select best plan
        best_plan, best_cost = min(costed_plans, key=lambda x: x[1])
        
        return OptimizedQuery(
            original_query=parsed_query,
            execution_plan=best_plan,
            estimated_cost=best_cost,
            optimization_applied=best_plan.optimizations,
            alternative_plans=[(p, c) for p, c in costed_plans if p != best_plan]
        )
    
    async def _generate_plans(self, query: ParsedQuery, 
                            data_sources: Dict[str, DataSource]) -> List[ExecutionPlan]:
        """Generate alternative execution plans"""
        plans = []
        
        # Plan 1: Push-down optimization
        if self._can_push_down(query, data_sources):
            plans.append(await self._create_pushdown_plan(query, data_sources))
        
        # Plan 2: Broadcast join
        if self._has_small_table(query, data_sources):
            plans.append(await self._create_broadcast_plan(query, data_sources))
        
        # Plan 3: Hash join
        plans.append(await self._create_hash_join_plan(query, data_sources))
        
        return plans
```

### Federation Executor

```python
class FederationExecutor:
    """Execute federated queries across data sources"""
    
    def __init__(self, connector_manager: ConnectorManager):
        self.connector_manager = connector_manager
        self.execution_context = {}
    
    async def execute_plan(self, plan: ExecutionPlan, 
                          data_sources: Dict[str, DataSource]) -> FederationResult:
        """Execute optimized federation plan"""
        
        execution_id = str(uuid.uuid4())
        self.execution_context[execution_id] = {
            'start_time': datetime.utcnow(),
            'plan': plan,
            'intermediate_results': {}
        }
        
        try:
            # Execute plan steps
            if plan.strategy == 'parallel':
                result = await self._execute_parallel(execution_id, plan, data_sources)
            elif plan.strategy == 'sequential':
                result = await self._execute_sequential(execution_id, plan, data_sources)
            else:
                result = await self._execute_adaptive(execution_id, plan, data_sources)
            
            return FederationResult(
                execution_id=execution_id,
                result_data=result,
                execution_time=datetime.utcnow() - self.execution_context[execution_id]['start_time'],
                plan_used=plan
            )
            
        finally:
            # Cleanup execution context
            del self.execution_context[execution_id]
    
    async def _execute_parallel(self, execution_id: str, plan: ExecutionPlan,
                               data_sources: Dict[str, DataSource]) -> Dict[str, Any]:
        """Execute plan steps in parallel where possible"""
        
        # Group independent steps
        step_groups = self._group_independent_steps(plan.steps)
        
        final_result = None
        for group in step_groups:
            # Execute group in parallel
            tasks = [
                self._execute_step(execution_id, step, data_sources)
                for step in group
            ]
            
            group_results = await asyncio.gather(*tasks)
            
            # Merge results for next group
            for i, result in enumerate(group_results):
                self.execution_context[execution_id]['intermediate_results'][group[i].step_id] = result
            
            # If this is the final group, get the result
            if group == step_groups[-1]:
                final_result = group_results[0]  # Assuming single final step
        
        return final_result
```

## Development Workflows

### Feature Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/query-streaming

# 2. Implement feature with tests
# - Add functionality to service.py
# - Create comprehensive tests
# - Update documentation

# 3. Run tests
uv run pytest tests/ -v
uv run pytest tests/integration/ -v

# 4. Check code quality
ruff check .
black --check .
mypy src/

# 5. Test APG integration
python -m pytest tests/apg_integration/ -v

# 6. Create pull request
git push origin feature/query-streaming
```

### Testing Development

```python
# tests/test_query_execution.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from dvrl.service import DVRLService
from dvrl.models import DataSource, DataSourceType

class TestQueryExecution:
    """Test query execution functionality"""
    
    @pytest.fixture
    async def dvrl_service(self):
        """Create DVRL service for testing"""
        service = DVRLService(
            tenant_id="test-tenant",
            user_id="test-user"
        )
        await service.initialize()
        return service
    
    @pytest.fixture
    def sample_data_source(self):
        """Create sample data source for testing"""
        return DataSource(
            id="test-ds-1",
            name="Test PostgreSQL",
            type=DataSourceType.POSTGRESQL,
            connection_config={
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass"
            },
            tenant_id="test-tenant",
            created_by="test-user"
        )
    
    async def test_simple_query_execution(self, dvrl_service, sample_data_source):
        """Test simple query execution"""
        # Register data source
        await dvrl_service.register_data_source(sample_data_source.dict())
        
        # Execute query
        result = await dvrl_service.execute_federated_query(
            "SELECT COUNT(*) FROM users"
        )
        
        assert result.status == "completed"
        assert result.rows_returned > 0
        assert result.execution_time_ms > 0
    
    @patch('dvrl.connectors.postgresql.PostgreSQLConnector.execute_query')
    async def test_query_with_mock(self, mock_execute, dvrl_service):
        """Test query execution with mocked data source"""
        # Mock response
        mock_execute.return_value = {
            'columns': [{'name': 'count', 'type': 'bigint'}],
            'rows': [[42]],
            'row_count': 1
        }
        
        # Execute query
        result = await dvrl_service.execute_federated_query(
            "SELECT COUNT(*) FROM products"
        )
        
        assert result.results['rows'][0][0] == 42
        mock_execute.assert_called_once()
```

### Performance Testing

```python
# tests/performance/test_query_performance.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestQueryPerformance:
    """Performance benchmarks for query execution"""
    
    @pytest.mark.benchmark
    async def test_concurrent_query_execution(self, dvrl_service, benchmark):
        """Test concurrent query performance"""
        
        async def execute_query():
            return await dvrl_service.execute_federated_query(
                "SELECT id, name FROM users LIMIT 100"
            )
        
        # Benchmark concurrent execution
        def run_concurrent_queries():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            tasks = [execute_query() for _ in range(10)]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()
            return results
        
        results = benchmark(run_concurrent_queries)
        assert len(results) == 10
        assert all(r.status == "completed" for r in results)
    
    async def test_large_result_set_performance(self, dvrl_service):
        """Test performance with large result sets"""
        start_time = time.time()
        
        result = await dvrl_service.execute_federated_query(
            "SELECT * FROM large_table LIMIT 50000"
        )
        
        execution_time = time.time() - start_time
        
        assert result.rows_returned == 50000
        assert execution_time < 30  # Should complete within 30 seconds
        assert result.performance_metrics['memory_usage_mb'] < 1024
```

## Testing Strategies

### Unit Testing

```python
# Focus on individual components
# Use pytest fixtures for setup
# Mock external dependencies
# Test edge cases and error conditions

@pytest.fixture
async def mock_connector():
    connector = Mock()
    connector.execute_query.return_value = {'rows': [], 'columns': []}
    return connector

async def test_query_parser_error_handling():
    parser = SQLParser()
    
    with pytest.raises(SQLParseError):
        await parser.parse_query("INVALID SQL SYNTAX")
```

### Integration Testing

```python
# Test component interactions
# Use real database connections for critical paths
# Test APG service integrations
# Verify data flow end-to-end

async def test_apg_auth_integration():
    auth_service = APGAuthService()
    
    # Test valid token
    user = await auth_service.validate_token(valid_token)
    assert user.tenant_id == "expected-tenant"
    
    # Test invalid token
    user = await auth_service.validate_token(invalid_token)
    assert user is None
```

### Load Testing

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between

class DVRLUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get auth token"""
        self.token = self.get_auth_token()
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def execute_simple_query(self):
        """Execute simple analytical query"""
        self.client.post(
            "/api/v1/queries/sql",
            json={
                "sql": "SELECT COUNT(*) FROM orders WHERE date >= '2024-01-01'",
                "options": {"cache_strategy": "conservative"}
            },
            headers=self.headers
        )
    
    @task(1)
    def execute_complex_query(self):
        """Execute complex federated query"""
        self.client.post(
            "/api/v1/queries/sql", 
            json={
                "sql": """
                    SELECT c.region, SUM(o.total_amount)
                    FROM customers c
                    JOIN orders o ON c.id = o.customer_id
                    WHERE o.created_at >= '2024-01-01'
                    GROUP BY c.region
                """,
                "options": {"federation_strategy": "optimal"}
            },
            headers=self.headers
        )
```

## Performance Optimization

### Query Optimization Techniques

```python
class QueryOptimizationTechniques:
    """Advanced query optimization patterns"""
    
    async def predicate_pushdown(self, query: ParsedQuery, 
                                data_sources: Dict[str, DataSource]) -> str:
        """Push WHERE conditions to data sources"""
        optimized_fragments = []
        
        for table in query.tables:
            # Find conditions that can be pushed down
            pushdown_conditions = []
            for condition in query.conditions:
                if self._can_pushdown_condition(condition, table):
                    pushdown_conditions.append(condition)
            
            # Rewrite query fragment
            if pushdown_conditions:
                fragment_sql = f"""
                    SELECT * FROM {table.full_name}
                    WHERE {' AND '.join(pushdown_conditions)}
                """
                optimized_fragments.append(fragment_sql)
        
        return self._combine_fragments(optimized_fragments, query)
    
    async def join_optimization(self, query: ParsedQuery) -> str:
        """Optimize join order and algorithms"""
        # Estimate table sizes
        table_sizes = {}
        for table in query.tables:
            size = await self._estimate_table_size(table)
            table_sizes[table.name] = size
        
        # Reorder joins by size (smallest first)
        sorted_tables = sorted(query.tables, 
                             key=lambda t: table_sizes.get(t.name, 0))
        
        # Generate optimized join sequence
        optimized_sql = self._generate_join_sequence(sorted_tables, query.joins)
        return optimized_sql
    
    async def partition_elimination(self, query: ParsedQuery) -> str:
        """Eliminate unnecessary partitions"""
        # Analyze date/time conditions
        date_conditions = [
            c for c in query.conditions 
            if self._is_date_condition(c)
        ]
        
        if date_conditions:
            # Determine relevant partitions
            relevant_partitions = await self._find_relevant_partitions(
                date_conditions, query.tables
            )
            
            # Rewrite with partition pruning
            return self._rewrite_with_partitions(query.sql, relevant_partitions)
        
        return query.sql
```

### Caching Strategies

```python
class IntelligentCacheManager:
    """ML-powered cache management"""
    
    def __init__(self):
        self.cache_predictor = self._load_cache_model()
        self.access_patterns = {}
    
    async def should_cache_query(self, query_hash: str, 
                                query_info: Dict[str, Any]) -> bool:
        """Predict if query should be cached"""
        features = self._extract_cache_features(query_info)
        prediction = self.cache_predictor.predict([features])
        
        return prediction[0] > 0.7  # Cache if >70% confidence
    
    def _extract_cache_features(self, query_info: Dict[str, Any]) -> List[float]:
        """Extract features for cache prediction"""
        return [
            query_info.get('complexity_score', 0),
            len(query_info.get('tables', [])),
            len(query_info.get('joins', [])),
            query_info.get('estimated_rows', 0),
            self._get_historical_access_frequency(query_info['query_hash'])
        ]
    
    async def intelligent_eviction(self) -> None:
        """Evict cache entries using ML predictions"""
        cache_entries = await self._get_all_cache_entries()
        
        # Predict future access probability
        predictions = []
        for entry in cache_entries:
            features = self._extract_access_features(entry)
            prob = self.access_predictor.predict_proba([features])[0][1]
            predictions.append((entry, prob))
        
        # Evict entries with lowest predicted access probability
        to_evict = sorted(predictions, key=lambda x: x[1])[:len(predictions)//4]
        
        for entry, _ in to_evict:
            await self._evict_cache_entry(entry['key'])
```

## Security Implementation

### Authentication and Authorization

```python
class DVRLSecurityManager:
    """Comprehensive security management"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.rbac_client = RBACClient()
        self.audit_logger = AuditLogger()
    
    async def check_data_source_access(self, user: User, data_source_id: str, 
                                     operation: str) -> bool:
        """Check if user has access to data source"""
        # Check explicit permissions
        has_permission = await self.rbac_client.check_permission(
            user.user_id,
            f"dvrl:data_source:{data_source_id}:{operation}"
        )
        
        if has_permission:
            await self.audit_logger.log_access(
                user_id=user.user_id,
                resource=f"data_source:{data_source_id}",
                operation=operation,
                result="allowed"
            )
            return True
        
        # Check role-based access
        user_roles = await self.rbac_client.get_user_roles(user.user_id)
        for role in user_roles:
            if await self._role_has_data_source_access(role, data_source_id, operation):
                await self.audit_logger.log_access(
                    user_id=user.user_id,
                    resource=f"data_source:{data_source_id}",
                    operation=operation,
                    result="allowed",
                    via_role=role
                )
                return True
        
        # Access denied
        await self.audit_logger.log_access(
            user_id=user.user_id,
            resource=f"data_source:{data_source_id}",
            operation=operation,
            result="denied"
        )
        return False
    
    async def apply_row_level_security(self, user: User, sql: str) -> str:
        """Apply row-level security policies"""
        # Parse SQL to identify tables
        parsed = await SQLParser().parse_query(sql)
        
        # Apply security policies per table
        secured_sql = sql
        for table in parsed.tables:
            policy = await self._get_row_level_policy(user, table.name)
            if policy:
                secured_sql = self._apply_policy_to_sql(secured_sql, table, policy)
        
        return secured_sql
    
    async def mask_sensitive_data(self, user: User, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic data masking"""
        masked_results = results.copy()
        
        for i, column in enumerate(results['columns']):
            if await self._is_sensitive_column(column['name']):
                masking_rule = await self._get_masking_rule(user, column['name'])
                
                if masking_rule:
                    # Apply masking to all rows
                    for row in masked_results['rows']:
                        row[i] = self._apply_masking_rule(row[i], masking_rule)
        
        return masked_results
```

### Data Encryption

```python
class DataEncryption:
    """Handle data encryption at rest and in transit"""
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.encryption_algo = "AES-256-GCM"
    
    async def encrypt_connection_string(self, connection_string: str, 
                                      data_source_id: str) -> str:
        """Encrypt connection strings for storage"""
        key = await self.key_manager.get_encryption_key(
            f"data_source:{data_source_id}"
        )
        
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(connection_string.encode())
        
        # Return base64 encoded encrypted data
        encrypted_data = {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'tag': base64.b64encode(tag).decode(),
            'nonce': base64.b64encode(cipher.nonce).decode()
        }
        
        return json.dumps(encrypted_data)
    
    async def decrypt_connection_string(self, encrypted_data: str, 
                                      data_source_id: str) -> str:
        """Decrypt connection strings for use"""
        key = await self.key_manager.get_encryption_key(
            f"data_source:{data_source_id}"
        )
        
        data = json.loads(encrypted_data)
        ciphertext = base64.b64decode(data['ciphertext'])
        tag = base64.b64decode(data['tag'])
        nonce = base64.b64decode(data['nonce'])
        
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        connection_string = cipher.decrypt_and_verify(ciphertext, tag)
        
        return connection_string.decode()
```

## Extension Points

### Custom Data Source Connectors

```python
from dvrl.connectors.base import BaseConnector

class CustomDatabaseConnector(BaseConnector):
    """Template for custom data source connectors"""
    
    def __init__(self, connection_config: Dict[str, Any]):
        super().__init__(connection_config)
        self.connection_pool = None
    
    async def connect(self) -> None:
        """Establish connection to custom database"""
        self.connection_pool = await create_custom_pool(
            host=self.connection_config['host'],
            port=self.connection_config['port'],
            database=self.connection_config['database'],
            min_size=5,
            max_size=20
        )
    
    async def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute query against custom database"""
        async with self.connection_pool.acquire() as conn:
            # Translate SQL if needed
            native_sql = self._translate_sql(sql)
            
            # Execute query
            result = await conn.fetch(native_sql)
            
            # Convert to standard format
            return {
                'columns': self._extract_columns(result),
                'rows': [list(row.values()) for row in result],
                'row_count': len(result)
            }
    
    async def discover_schema(self) -> DataSourceSchema:
        """Discover schema from custom database"""
        async with self.connection_pool.acquire() as conn:
            # Custom schema discovery logic
            tables_query = "SHOW TABLES"  # Adapt to your database
            tables = await conn.fetch(tables_query)
            
            schema_tables = []
            for table_row in tables:
                table_name = table_row['table_name']
                columns = await self._discover_table_columns(conn, table_name)
                
                schema_tables.append(TableSchema(
                    name=table_name,
                    columns=columns,
                    table_type='table'
                ))
            
            return DataSourceSchema(
                schema_name=self.connection_config['database'],
                data_source_id=self.data_source_id,
                tables=schema_tables
            )
    
    def _translate_sql(self, sql: str) -> str:
        """Translate standard SQL to database-specific dialect"""
        # Implement SQL dialect translation
        return sql
```

### Custom Query Optimizers

```python
from dvrl.optimization.base import BaseOptimizer

class MLQueryOptimizer(BaseOptimizer):
    """Machine learning powered query optimizer"""
    
    def __init__(self):
        super().__init__()
        self.model = self._load_ml_model()
    
    async def optimize_query(self, parsed_query: ParsedQuery,
                            data_sources: Dict[str, DataSource]) -> OptimizedQuery:
        """Optimize using ML predictions"""
        
        # Extract features for ML model
        features = self._extract_query_features(parsed_query, data_sources)
        
        # Predict optimal execution strategy
        strategy_prediction = self.model.predict_strategy(features)
        
        # Generate execution plan based on prediction
        if strategy_prediction == 'pushdown':
            plan = await self._create_pushdown_plan(parsed_query, data_sources)
        elif strategy_prediction == 'broadcast':
            plan = await self._create_broadcast_plan(parsed_query, data_sources)
        else:
            plan = await self._create_hash_join_plan(parsed_query, data_sources)
        
        return OptimizedQuery(
            original_query=parsed_query,
            execution_plan=plan,
            optimizer_used='ml_optimizer',
            confidence_score=strategy_prediction.confidence
        )
    
    def _extract_query_features(self, query: ParsedQuery, 
                               data_sources: Dict[str, DataSource]) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            len(query.tables),
            len(query.joins),
            len(query.conditions),
            query.complexity.score,
            sum(ds.estimated_size for ds in data_sources.values()),
            len([j for j in query.joins if j.type == 'INNER']),
            len([j for j in query.joins if j.type in ['LEFT', 'RIGHT']]),
        ]
        return np.array(features)
```

## Best Practices

### Code Organization

```python
# Follow APG coding standards
# Use async/await throughout
# Implement proper error handling
# Add comprehensive logging
# Use type hints consistently

from typing import Optional, Dict, List, Any
import logging
from datetime import datetime

class DVRLBestPractices:
    """Demonstrate DVRL development best practices"""
    
    def __init__(self, tenant_id: str, user_id: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.logger = logging.getLogger(f"dvrl.{self.__class__.__name__}")
    
    async def example_method(self, required_param: str, 
                           optional_param: Optional[int] = None) -> Dict[str, Any]:
        """
        Example method demonstrating best practices.
        
        Args:
            required_param: Required string parameter
            optional_param: Optional integer parameter
            
        Returns:
            Dictionary containing operation result
            
        Raises:
            ValueError: If required_param is invalid
            ConnectionError: If unable to connect to required services
        """
        # Input validation
        if not required_param or not required_param.strip():
            raise ValueError("required_param cannot be empty")
        
        # Logging with context
        self.logger.info(
            "Starting example operation",
            extra={
                'tenant_id': self.tenant_id,
                'user_id': self.user_id,
                'required_param': required_param,
                'optional_param': optional_param
            }
        )
        
        try:
            # Main logic with proper error handling
            result = await self._perform_operation(required_param, optional_param)
            
            self.logger.info(
                "Example operation completed successfully",
                extra={
                    'tenant_id': self.tenant_id,
                    'result_size': len(result),
                    'execution_time_ms': result.get('execution_time_ms')
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Example operation failed",
                extra={
                    'tenant_id': self.tenant_id,
                    'user_id': self.user_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
    
    async def _perform_operation(self, param1: str, 
                                param2: Optional[int]) -> Dict[str, Any]:
        """Private helper method with proper naming and typing"""
        # Implementation details
        return {
            'status': 'success',
            'data': f"Processed {param1}",
            'execution_time_ms': 150
        }
```

### Error Handling Patterns

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def database_transaction(data_source: DataSource) -> AsyncGenerator[Any, None]:
    """Context manager for database transactions"""
    connector = await get_connector(data_source)
    transaction = await connector.begin_transaction()
    
    try:
        yield connector
        await transaction.commit()
    except Exception as e:
        await transaction.rollback()
        logger.error(f"Transaction rolled back due to error: {e}")
        raise
    finally:
        await connector.close()

# Usage:
async def update_with_transaction(data_source: DataSource, updates: List[str]):
    async with database_transaction(data_source) as conn:
        for update_sql in updates:
            await conn.execute(update_sql)
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(operation_name: str):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                execution_time = time.time() - start_time
                metrics.record_operation_time(operation_name, execution_time)
                metrics.increment_counter(f"{operation_name}.success")
                
                return result
                
            except Exception as e:
                # Record error metrics
                execution_time = time.time() - start_time
                metrics.record_operation_time(f"{operation_name}.error", execution_time)
                metrics.increment_counter(f"{operation_name}.error")
                raise
                
        return wrapper
    return decorator

# Usage:
class DVRLService:
    @monitor_performance("federated_query_execution")
    async def execute_federated_query(self, sql: str) -> FederatedQuery:
        # Implementation
        pass
```

---

This developer guide provides comprehensive patterns and practices for extending and maintaining the APG DVRL capability. Follow these patterns to ensure consistent, maintainable, and performant code.

**Document Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team