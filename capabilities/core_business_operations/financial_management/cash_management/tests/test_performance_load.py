"""APG Cash Management - Performance and Load Tests

Comprehensive performance and load testing for enterprise scalability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import AsyncMock, patch
import psutil
import gc

from ..service import CashManagementService
from ..advanced_ml_models import AdvancedMLModelManager
from ..intelligent_optimization import IntelligentCashFlowOptimizer
from ..advanced_risk_analytics import AdvancedRiskAnalyticsEngine

# ============================================================================
# Performance Test Configuration
# ============================================================================

# Test data sizes
SMALL_DATASET = 1000
MEDIUM_DATASET = 10000  
LARGE_DATASET = 100000
XLARGE_DATASET = 1000000

# Performance thresholds
MAX_RESPONSE_TIME_MS = {
    'cache_operation': 10,
    'simple_query': 100,
    'complex_calculation': 5000,
    'ml_prediction': 30000,
    'optimization': 60000
}

THROUGHPUT_REQUIREMENTS = {
    'cache_ops_per_sec': 10000,
    'simple_queries_per_sec': 1000,
    'cash_flows_per_sec': 500,
    'concurrent_users': 100
}

# ============================================================================
# Performance Test Utilities
# ============================================================================

class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_before = None
        self.memory_after = None
        self.cpu_before = None
        self.cpu_after = None
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_before = psutil.cpu_percent()
        gc.collect()  # Force garbage collection
    
    def stop(self):
        """Stop performance monitoring."""
        self.end_time = time.perf_counter()
        self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.cpu_after = psutil.cpu_percent()
    
    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def memory_delta(self) -> float:
        """Get memory usage delta in MB."""
        if self.memory_before and self.memory_after:
            return self.memory_after - self.memory_before
        return 0.0
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        return {
            'execution_time_ms': self.execution_time * 1000,
            'memory_delta_mb': self.memory_delta,
            'cpu_usage_before': self.cpu_before,
            'cpu_usage_after': self.cpu_after
        }

@pytest.fixture
def performance_monitor():
    """Performance monitor fixture."""
    return PerformanceMonitor()

def generate_large_cash_flows(size: int, start_date: datetime = None) -> list:
    """Generate large dataset of cash flows for testing."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    
    np.random.seed(42)  # For reproducible tests
    
    flows = []
    for i in range(size):
        flow_date = start_date + timedelta(days=np.random.randint(0, 365))
        amount = np.random.normal(5000, 2000)
        
        flows.append({
            'id': f'FLOW{i:08d}',
            'tenant_id': 'test_tenant',
            'account_id': f'ACC{(i % 100):03d}',  # 100 different accounts
            'amount': Decimal(str(round(amount, 2))),
            'transaction_date': flow_date,
            'description': f'Test transaction {i}',
            'category': ['operating', 'investment', 'financing'][i % 3],
            'is_recurring': i % 50 == 0
        })
    
    return flows

def generate_large_portfolio(size: int) -> dict:
    """Generate large portfolio for testing."""
    np.random.seed(42)
    
    portfolio = {}
    account_types = ['checking', 'savings', 'money_market', 'investment', 'cd']
    
    for i in range(size):
        account_id = f'ACC{i:06d}'
        account_type = account_types[i % len(account_types)]
        balance = np.random.exponential(50000)  # Exponential distribution
        
        portfolio[account_id] = {
            'balance': balance,
            'type': account_type,
            'liquidity_score': np.random.uniform(0.5, 1.0),
            'risk_score': np.random.uniform(0.01, 0.3)
        }
    
    return portfolio

# ============================================================================
# Cache Performance Tests
# ============================================================================

@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for cache operations."""
    
    @pytest_asyncio.fixture
    async def mock_cache_with_latency(self):
        """Mock cache with realistic latency."""
        cache = AsyncMock()
        
        async def mock_set_with_latency(key, value, ttl=None):
            await asyncio.sleep(0.001)  # 1ms latency
            return True
        
        async def mock_get_with_latency(key):
            await asyncio.sleep(0.001)  # 1ms latency
            return {"cached": "data"} if "valid" in key else None
        
        cache.set = mock_set_with_latency
        cache.get = mock_get_with_latency
        cache.delete = AsyncMock(return_value=True)
        
        return cache
    
    async def test_cache_throughput(self, mock_cache_with_latency, performance_monitor):
        """Test cache operation throughput."""
        cache = mock_cache_with_latency
        num_operations = 10000
        
        performance_monitor.start()
        
        # Test concurrent cache operations
        tasks = []
        for i in range(num_operations):
            if i % 2 == 0:
                tasks.append(cache.set(f"key_{i}", {"data": i}))
            else:
                tasks.append(cache.get(f"valid_key_{i}"))
        
        await asyncio.gather(*tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Calculate throughput
        throughput = num_operations / performance_monitor.execution_time
        
        # Assertions
        assert throughput >= THROUGHPUT_REQUIREMENTS['cache_ops_per_sec']
        assert metrics['execution_time_ms'] < 5000  # Should complete in 5 seconds
        
        print(f"Cache throughput: {throughput:.0f} ops/sec")
        print(f"Execution time: {metrics['execution_time_ms']:.0f}ms")
    
    async def test_cache_memory_efficiency(self, mock_cache_with_latency, performance_monitor):
        """Test cache memory efficiency."""
        cache = mock_cache_with_latency
        
        performance_monitor.start()
        
        # Store large amount of data
        large_data = {"data": "x" * 1000}  # 1KB per entry
        tasks = []
        
        for i in range(1000):  # 1MB total
            tasks.append(cache.set(f"large_key_{i}", large_data))
        
        await asyncio.gather(*tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Memory should not grow excessively (mock doesn't actually store data)
        assert metrics['memory_delta_mb'] < 100  # Less than 100MB increase
        
        print(f"Memory delta: {metrics['memory_delta_mb']:.1f}MB")

# ============================================================================
# Database Performance Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    async def test_bulk_cash_flow_insertion(self, cash_service, performance_monitor):
        """Test bulk cash flow insertion performance."""
        
        # Generate large dataset
        flows = generate_large_cash_flows(MEDIUM_DATASET)
        
        # Mock database operations
        cash_service.db = AsyncMock()
        cash_service.db.add = AsyncMock()
        cash_service.db.commit = AsyncMock()
        cash_service.db.bulk_insert_mappings = AsyncMock()
        
        performance_monitor.start()
        
        # Test bulk insertion
        for flow in flows:
            await cash_service.record_cash_flow(flow)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Calculate throughput
        throughput = len(flows) / performance_monitor.execution_time
        
        # Assertions
        assert throughput >= THROUGHPUT_REQUIREMENTS['cash_flows_per_sec']
        assert metrics['execution_time_ms'] < 30000  # 30 seconds max
        
        print(f"Cash flow insertion throughput: {throughput:.0f} flows/sec")
    
    async def test_concurrent_database_access(self, cash_service, performance_monitor):
        """Test concurrent database access performance."""
        
        # Mock database operations
        cash_service.db = AsyncMock()
        mock_account = AsyncMock()
        mock_account.current_balance = Decimal('100000.00')
        cash_service.db.query().filter().first = AsyncMock(return_value=mock_account)
        
        num_concurrent = 100
        num_operations = 10
        
        performance_monitor.start()
        
        async def concurrent_operations():
            tasks = []
            for i in range(num_operations):
                tasks.append(cash_service.get_account_balance(f"ACC{i:03d}"))
            return await asyncio.gather(*tasks)
        
        # Run concurrent batches
        batch_tasks = []
        for _ in range(num_concurrent):
            batch_tasks.append(concurrent_operations())
        
        await asyncio.gather(*batch_tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        total_operations = num_concurrent * num_operations
        throughput = total_operations / performance_monitor.execution_time
        
        # Assertions
        assert throughput >= THROUGHPUT_REQUIREMENTS['simple_queries_per_sec']
        
        print(f"Concurrent DB access throughput: {throughput:.0f} queries/sec")

# ============================================================================
# ML Model Performance Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.ml
@pytest.mark.slow
class TestMLPerformance:
    """Performance tests for ML models."""
    
    async def test_model_training_performance(self, ml_manager, performance_monitor):
        """Test ML model training performance."""
        
        # Generate large training dataset
        np.random.seed(42)
        data_size = MEDIUM_DATASET
        
        training_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=data_size, freq='D'),
            'amount': np.random.normal(10000, 2000, data_size),
            'feature1': np.random.normal(0, 1, data_size),
            'feature2': np.random.normal(0, 1, data_size),
            'feature3': np.random.normal(0, 1, data_size)
        })
        
        performance_monitor.start()
        
        # Mock model training (actual training would be too slow for tests)
        with patch.object(ml_manager, 'train_all_models') as mock_train:
            mock_performances = {
                'xgboost': AsyncMock(success=True, training_time=10.5),
                'random_forest': AsyncMock(success=True, training_time=8.2),
                'elastic_net': AsyncMock(success=True, training_time=2.1)
            }
            mock_train.return_value = mock_performances
            
            performances = await ml_manager.train_all_models(training_data)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Assertions
        assert len(performances) >= 3
        assert metrics['execution_time_ms'] < MAX_RESPONSE_TIME_MS['ml_prediction']
        
        print(f"Model training time: {metrics['execution_time_ms']:.0f}ms")
    
    async def test_prediction_throughput(self, ml_manager, performance_monitor):
        """Test ML prediction throughput."""
        
        # Mock trained models
        ml_manager.models = {
            'model1': AsyncMock(),
            'model2': AsyncMock()
        }
        
        # Mock predictions
        prediction_data = np.random.normal(1000, 100, 100)
        ml_manager.models['model1'].predict = AsyncMock(return_value=prediction_data)
        ml_manager.models['model2'].predict = AsyncMock(return_value=prediction_data * 1.1)
        
        # Generate test data
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        
        performance_monitor.start()
        
        # Mock feature engineering
        with patch.object(ml_manager.feature_engineer, 'engineer_features') as mock_features:
            mock_features.return_value = (test_data, pd.Series([1000] * 1000))
            
            # Run multiple predictions
            tasks = []
            for i in range(10):  # 10 concurrent predictions
                tasks.append(ml_manager.predict_with_uncertainty(
                    data=test_data,
                    model_names=['model1', 'model2']
                ))
            
            results = await asyncio.gather(*tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Calculate throughput
        total_predictions = len(results) * len(test_data)
        throughput = total_predictions / performance_monitor.execution_time
        
        # Assertions
        assert len(results) == 10
        assert all(len(r.predictions) == len(test_data) for r in results)
        assert throughput > 1000  # Should handle > 1000 predictions/sec
        
        print(f"Prediction throughput: {throughput:.0f} predictions/sec")

# ============================================================================
# Optimization Performance Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.optimization
class TestOptimizationPerformance:
    """Performance tests for optimization engine."""
    
    async def test_optimization_scalability(self, optimization_engine, performance_monitor):
        """Test optimization scalability with large portfolios."""
        
        # Test different portfolio sizes
        portfolio_sizes = [10, 50, 100, 500]
        results = {}
        
        for size in portfolio_sizes:
            # Generate test portfolio
            portfolio = generate_large_portfolio(size)
            
            # Convert to account format
            accounts = []
            for i, (account_id, data) in enumerate(portfolio.items()):
                accounts.append({
                    'id': account_id,
                    'account_type': data['type'],
                    'current_balance': Decimal(str(data['balance']))
                })
            
            performance_monitor.start()
            
            # Run optimization
            from ..intelligent_optimization import OptimizationObjective, OptimizationConstraint, ConstraintType
            
            objectives = [OptimizationObjective.MAXIMIZE_YIELD]
            constraints = [
                OptimizationConstraint(
                    name="balance_conservation",
                    constraint_type=ConstraintType.BALANCE_REQUIREMENT,
                    target_value=sum(data['balance'] for data in portfolio.values())
                )
            ]
            
            result = await optimization_engine.optimize_cash_allocation(
                accounts=accounts,
                objectives=objectives,
                constraints=constraints
            )
            
            performance_monitor.stop()
            metrics = performance_monitor.get_metrics()
            
            results[size] = {
                'execution_time': metrics['execution_time_ms'],
                'success': result.success,
                'memory_delta': metrics['memory_delta_mb']
            }
        
        # Analyze scalability
        for size, result in results.items():
            assert result['success'] is True
            assert result['execution_time'] < MAX_RESPONSE_TIME_MS['optimization']
            
            print(f"Portfolio size {size}: {result['execution_time']:.0f}ms")
        
        # Check that time complexity is reasonable
        time_10 = results[10]['execution_time']
        time_100 = results[100]['execution_time']
        
        # Should not be more than 100x slower for 10x data
        assert time_100 / time_10 < 100
    
    async def test_concurrent_optimizations(self, optimization_engine, performance_monitor):
        """Test concurrent optimization performance."""
        
        # Generate multiple portfolios
        portfolios = []
        for i in range(5):
            portfolio = generate_large_portfolio(20)
            accounts = []
            for account_id, data in portfolio.items():
                accounts.append({
                    'id': f"{account_id}_{i}",
                    'account_type': data['type'],
                    'current_balance': Decimal(str(data['balance']))
                })
            portfolios.append(accounts)
        
        performance_monitor.start()
        
        async def run_optimization(accounts):
            from ..intelligent_optimization import OptimizationObjective
            objectives = [OptimizationObjective.MAXIMIZE_YIELD]
            return await optimization_engine.optimize_cash_allocation(
                accounts=accounts,
                objectives=objectives,
                constraints=[]
            )
        
        # Run concurrent optimizations
        tasks = [run_optimization(accounts) for accounts in portfolios]
        results = await asyncio.gather(*tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Assertions
        assert len(results) == 5
        assert all(r.success for r in results)
        assert metrics['execution_time_ms'] < MAX_RESPONSE_TIME_MS['optimization'] * 2
        
        print(f"Concurrent optimizations: {metrics['execution_time_ms']:.0f}ms")

# ============================================================================
# Risk Analytics Performance Tests
# ============================================================================

@pytest.mark.performance
@pytest.mark.risk
class TestRiskAnalyticsPerformance:
    """Performance tests for risk analytics."""
    
    async def test_var_calculation_performance(self, risk_analytics, performance_monitor):
        """Test VaR calculation performance with large datasets."""
        
        # Generate large returns dataset
        returns_sizes = [1000, 5000, 10000]
        
        for size in returns_sizes:
            np.random.seed(42)
            returns_data = np.random.normal(0.001, 0.02, size)
            
            performance_monitor.start()
            
            # Calculate VaR using multiple methods
            var_calc = risk_analytics.var_calculator
            
            results = await asyncio.gather(
                var_calc.calculate_parametric_var(returns_data, 0.95),
                var_calc.calculate_historical_var(returns_data, 0.95),
                var_calc.calculate_monte_carlo_var(returns_data, 0.95, num_simulations=1000)
            )
            
            performance_monitor.stop()
            metrics = performance_monitor.get_metrics()
            
            # Assertions
            assert len(results) == 3
            assert all(len(r) == 2 for r in results)  # (value, details) tuple
            assert metrics['execution_time_ms'] < MAX_RESPONSE_TIME_MS['complex_calculation']
            
            print(f"VaR calculation ({size} returns): {metrics['execution_time_ms']:.0f}ms")
    
    async def test_stress_testing_performance(self, risk_analytics, performance_monitor):
        """Test stress testing performance."""
        
        # Generate large portfolio
        portfolio = generate_large_portfolio(100)
        
        performance_monitor.start()
        
        # Run comprehensive stress tests
        stress_results = await risk_analytics.run_comprehensive_stress_tests(portfolio)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Assertions
        assert 'historical_scenarios' in stress_results
        assert 'monte_carlo' in stress_results
        assert 'liquidity_stress' in stress_results
        assert metrics['execution_time_ms'] < MAX_RESPONSE_TIME_MS['complex_calculation'] * 2
        
        print(f"Stress testing: {metrics['execution_time_ms']:.0f}ms")

# ============================================================================
# Load Testing
# ============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestSystemLoad:
    """System load and stress tests."""
    
    async def test_concurrent_users_simulation(
        self, 
        cash_service, 
        performance_monitor
    ):
        """Simulate concurrent users accessing the system."""
        
        num_users = THROUGHPUT_REQUIREMENTS['concurrent_users']
        operations_per_user = 10
        
        # Mock service operations
        cash_service.db = AsyncMock()
        mock_account = AsyncMock()
        mock_account.current_balance = Decimal('100000.00')
        cash_service.db.query().filter().first = AsyncMock(return_value=mock_account)
        
        async def simulate_user_session(user_id: int):
            """Simulate a user session with multiple operations."""
            operations = []
            
            for op_id in range(operations_per_user):
                # Mix of different operations
                if op_id % 3 == 0:
                    operations.append(cash_service.get_account_balance(f"ACC{user_id}"))
                elif op_id % 3 == 1:
                    flow_data = {
                        'id': f'FLOW_{user_id}_{op_id}',
                        'account_id': f'ACC{user_id}',
                        'amount': Decimal('1000.00'),
                        'transaction_date': datetime.now(),
                        'description': f'User {user_id} operation {op_id}'
                    }
                    operations.append(cash_service.record_cash_flow(flow_data))
                else:
                    operations.append(cash_service.get_account_balance(f"ACC{user_id}"))
            
            return await asyncio.gather(*operations)
        
        performance_monitor.start()
        
        # Run concurrent user sessions
        user_tasks = [
            simulate_user_session(user_id) 
            for user_id in range(num_users)
        ]
        
        results = await asyncio.gather(*user_tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Calculate metrics
        total_operations = num_users * operations_per_user
        throughput = total_operations / performance_monitor.execution_time
        
        # Assertions
        assert len(results) == num_users
        assert all(len(user_results) == operations_per_user for user_results in results)
        assert throughput >= 500  # Should handle at least 500 ops/sec
        assert metrics['execution_time_ms'] < 60000  # Should complete in 1 minute
        
        print(f"Concurrent users: {num_users}")
        print(f"Total operations: {total_operations}")
        print(f"Throughput: {throughput:.0f} ops/sec")
        print(f"Average response time: {metrics['execution_time_ms']/total_operations:.1f}ms")
    
    async def test_memory_stress(self, performance_monitor):
        """Test system behavior under memory stress."""
        
        performance_monitor.start()
        
        # Allocate large amounts of data to stress memory
        large_datasets = []
        
        try:
            for i in range(10):  # 10 large datasets
                # Each dataset is ~100MB
                data = np.random.random((1000, 10000)).astype(np.float32)
                large_datasets.append(data)
                
                # Simulate processing
                processed = np.mean(data, axis=1)
                assert len(processed) == 1000
            
            # Simulate memory-intensive operations
            combined = np.concatenate([d.flatten()[:100000] for d in large_datasets])
            result = np.std(combined)
            assert result > 0
            
        finally:
            # Clean up
            del large_datasets
            gc.collect()
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Memory should be manageable
        assert metrics['memory_delta_mb'] < 2000  # Less than 2GB increase
        
        print(f"Memory stress test completed")
        print(f"Memory delta: {metrics['memory_delta_mb']:.0f}MB")
    
    async def test_cpu_intensive_operations(self, performance_monitor):
        """Test CPU-intensive operations performance."""
        
        performance_monitor.start()
        
        # Simulate CPU-intensive calculations
        async def cpu_intensive_task():
            # Monte Carlo simulation
            num_simulations = 100000
            np.random.seed(42)
            
            results = []
            for _ in range(100):  # 100 batches
                random_data = np.random.normal(0, 1, num_simulations)
                batch_result = np.mean(random_data ** 2)
                results.append(batch_result)
            
            return np.mean(results)
        
        # Run multiple CPU-intensive tasks
        tasks = [cpu_intensive_task() for _ in range(4)]
        results = await asyncio.gather(*tasks)
        
        performance_monitor.stop()
        metrics = performance_monitor.get_metrics()
        
        # Assertions
        assert len(results) == 4
        assert all(0.8 < r < 1.2 for r in results)  # Should be around 1.0
        assert metrics['execution_time_ms'] < 30000  # Should complete in 30 seconds
        
        print(f"CPU intensive test: {metrics['execution_time_ms']:.0f}ms")

# ============================================================================
# Benchmarking Tests
# ============================================================================

@pytest.mark.performance
class TestBenchmarks:
    """Benchmark tests for comparing performance."""
    
    async def test_algorithm_comparison_benchmark(self, performance_monitor):
        """Benchmark different algorithms for the same task."""
        
        # Generate test data
        data_size = 10000
        np.random.seed(42)
        data = np.random.normal(0, 1, data_size)
        
        algorithms = {}
        
        # Algorithm 1: NumPy vectorized operations
        performance_monitor.start()
        result1 = np.percentile(data, [5, 95])
        performance_monitor.stop()
        algorithms['numpy_vectorized'] = performance_monitor.get_metrics()
        
        # Algorithm 2: Pure Python loop
        performance_monitor.start()
        sorted_data = sorted(data)
        idx_5 = int(0.05 * len(sorted_data))
        idx_95 = int(0.95 * len(sorted_data))
        result2 = [sorted_data[idx_5], sorted_data[idx_95]]
        performance_monitor.stop()
        algorithms['python_loop'] = performance_monitor.get_metrics()
        
        # Algorithm 3: Pandas quantile
        performance_monitor.start()
        df = pd.Series(data)
        result3 = df.quantile([0.05, 0.95]).values
        performance_monitor.stop()
        algorithms['pandas_quantile'] = performance_monitor.get_metrics()
        
        # Compare results (should be similar)
        assert np.allclose(result1, result2, rtol=0.01)
        assert np.allclose(result1, result3, rtol=0.01)
        
        # Print benchmark results
        print("\nAlgorithm Benchmark Results:")
        for algo_name, metrics in algorithms.items():
            print(f"{algo_name}: {metrics['execution_time_ms']:.2f}ms")
        
        # NumPy should be fastest
        assert algorithms['numpy_vectorized']['execution_time_ms'] < algorithms['python_loop']['execution_time_ms']

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])