# APG Accounts Receivable - Performance Testing Suite

Comprehensive performance testing and validation for the APG Accounts Receivable capability.

## Overview

This performance testing suite validates that the AR capability meets enterprise-grade performance requirements under various load conditions, memory constraints, and scalability scenarios.

## Performance Requirements

Based on the APG capability specification, the AR system must meet these performance targets:

### Response Time Targets
- **API Response Time**: < 200ms for standard operations
- **AI Operations**: < 1000ms for credit scoring and collections optimization
- **Cash Flow Forecasting**: < 2000ms for 30-60 day forecasts
- **Bulk Operations**: Process 1000+ invoices in < 5 seconds

### Throughput Targets
- **Concurrent Users**: Support 1000+ concurrent users
- **Invoice Processing**: 50,000+ invoices per hour
- **Payment Processing**: 10,000+ payments per hour
- **AI Assessments**: 100+ credit assessments per minute

### Scalability Targets
- **Linear Scalability**: 70%+ efficiency up to 10x load
- **Memory Efficiency**: < 1KB per customer record
- **Database Performance**: Query response < 100ms for 100K+ records

## Test Files

### 1. Load Performance Tests (`test_load_performance.py`)

Tests system performance under various load conditions:

- **Concurrent Customer Creation**: 50+ concurrent customer creation operations
- **Customer Query Performance**: 100+ concurrent query operations  
- **Bulk Invoice Creation**: 1000+ invoice batch processing
- **AI Service Performance**: Batch credit assessments and collections optimization
- **End-to-End Workflow**: Complete AR workflow under load

**Key Metrics:**
- Success Rate (target: >99%)
- Average Response Time
- 95th/99th Percentile Response Times
- Requests Per Second (RPS)

### 2. Memory Performance Tests (`test_memory_performance.py`)

Validates memory usage patterns and optimization:

- **Large Dataset Efficiency**: Memory usage with 10,000+ customer records
- **Concurrent Operations Stability**: Memory stability under concurrent operations
- **Long-Running Process Stability**: Memory usage over extended operations
- **Object Lifecycle Optimization**: Memory cleanup and garbage collection efficiency

**Key Metrics:**
- Peak Memory Usage
- Memory Per Object
- Cleanup Efficiency
- Memory Growth Rate

### 3. Scalability Tests (`test_scalability.py`)

Tests system scalability characteristics:

- **Linear Scalability**: Performance scaling with increasing load
- **Mixed Operations Scalability**: Scalability with varied operation types
- **AI Service Scalability**: AI operations scaling with batch sizes
- **Data Volume Scalability**: Query performance with increasing dataset sizes

**Key Metrics:**
- Scalability Efficiency (target: >70%)
- Performance Degradation Factor
- Throughput Scaling
- Response Time Scaling

## Running Performance Tests

### Prerequisites

Install required dependencies:
```bash
pip install pytest pytest-asyncio psutil
```

### Run All Performance Tests

```bash
# Using the performance runner (recommended)
python tests/performance/performance_runner.py

# Using pytest directly
pytest tests/performance/ -v -s -m performance
```

### Run Specific Test Categories

```bash
# Load performance only
python tests/performance/performance_runner.py test_load_performance.py

# Memory performance only  
python tests/performance/performance_runner.py test_memory_performance.py

# Scalability tests only
python tests/performance/performance_runner.py test_scalability.py

# Multiple specific tests
pytest tests/performance/test_load_performance.py tests/performance/test_scalability.py -v -s -m performance
```

### Run Individual Test Classes

```bash
# Test specific service performance
pytest tests/performance/test_load_performance.py::TestCustomerServicePerformance -v -s

# Test AI service scalability
pytest tests/performance/test_scalability.py::TestAIScalability -v -s

# Test memory optimization
pytest tests/performance/test_memory_performance.py::TestMemoryUsageOptimization -v -s
```

## Performance Test Configuration

### Load Test Parameters

```python
# Concurrent operation levels
LOAD_LEVELS = [10, 25, 50, 100, 200]

# Batch processing sizes
BATCH_SIZES = [100, 500, 1000, 2000]

# Dataset sizes for scalability testing
DATASET_SIZES = [1000, 5000, 10000, 20000]
```

### Performance Thresholds

```python
# Response time thresholds (milliseconds)
API_RESPONSE_TIME_THRESHOLD = 200
AI_RESPONSE_TIME_THRESHOLD = 1000
BULK_OPERATION_TIME_THRESHOLD = 5000

# Success rate thresholds (percentage)
MIN_SUCCESS_RATE = 99.0
MIN_AI_SUCCESS_RATE = 95.0

# Scalability thresholds
MIN_SCALABILITY_EFFICIENCY = 0.7
MAX_PERFORMANCE_DEGRADATION = 2.0

# Memory thresholds
MAX_MEMORY_PER_CUSTOMER = 1024  # bytes
MAX_PEAK_MEMORY_INCREASE = 500  # MB
MIN_CLEANUP_EFFICIENCY = 0.9    # 90%
```

## Performance Test Reports

The performance runner generates detailed JSON reports with:

### Summary Metrics
- Overall test status (PASSED/FAILED)
- Total test duration
- Test counts (passed/failed/skipped)
- Success rates

### Performance Insights
- Load performance metrics (response times, throughput)
- Memory performance metrics (peak usage, efficiency)
- Scalability metrics (efficiency, degradation factors)

### Recommendations
- Performance optimization suggestions
- Monitoring recommendations
- CI/CD integration guidance

### Sample Report Structure

```json
{
  "summary": {
    "overall_status": "PASSED",
    "total_duration": 125.3,
    "total_tests": 15,
    "passed": 15,
    "failed": 0,
    "success_rate": 100.0
  },
  "performance_insights": {
    "load_performance": {
      "success_rate": 99.2,
      "avg_response_time_ms": 145.3,
      "throughput_rps": 187.5
    },
    "memory_performance": {
      "peak_memory_increase_mb": 89.2,
      "memory_per_customer_bytes": 736,
      "cleanup_efficiency_percent": 94.1
    },
    "scalability_performance": {
      "scalability_efficiency_percent": 73.2,
      "performance_degradation_factor": 1.8
    }
  }
}
```

## Continuous Integration

### CI Pipeline Integration

Add performance testing to CI pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest psutil
      - name: Run Performance Tests
        run: |
          python tests/performance/performance_runner.py
      - name: Upload Performance Report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: tests/performance/performance_report_*.json
```

### Performance Monitoring

Set up alerts for performance degradation:

```python
# Performance monitoring thresholds
ALERT_THRESHOLDS = {
    'response_time_p95': 300,  # ms
    'success_rate': 98.0,      # %
    'memory_usage': 512,       # MB
    'throughput': 150          # RPS
}
```

## Troubleshooting

### Common Performance Issues

1. **High Response Times**
   - Check database query optimization
   - Verify async operation efficiency
   - Review AI service integration latency

2. **Memory Leaks**
   - Check object lifecycle management
   - Verify proper cleanup in exception handling
   - Review large dataset processing patterns

3. **Scalability Bottlenecks**
   - Analyze concurrent operation patterns
   - Check database connection pooling
   - Review APG service integration efficiency

### Performance Debugging

Enable detailed performance profiling:

```python
# Add to test files for detailed profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... test code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## Performance Optimization Guidelines

### Code Optimization
- Use async/await patterns consistently
- Implement proper database connection pooling
- Optimize AI service batch operations
- Cache frequently accessed data

### Database Optimization  
- Index optimization for query patterns
- Proper pagination for large result sets
- Connection pooling configuration
- Query optimization for multi-tenant patterns

### Memory Optimization
- Implement proper object lifecycle management
- Use generators for large data processing
- Optimize Pydantic model memory usage
- Implement periodic garbage collection

### AI Service Optimization
- Batch AI operations when possible
- Implement proper timeout handling
- Cache AI model results when appropriate
- Optimize feature extraction pipelines

## Related Documentation

- [APG Capability Specification](../cap_spec.md)
- [Testing Strategy](../tests/README.md)
- [API Documentation](../api_endpoints.py)
- [Service Architecture](../service.py)