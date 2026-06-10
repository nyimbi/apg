#!/usr/bin/env python3
"""
APG System Health Management (HLTH) - Performance Tests
Comprehensive performance and load testing

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import time
import psutil
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from models import HealthMetric, SystemComponent, HealthDimension, ComponentType


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    async def test_metric_processing_throughput(self, health_service):
        """Benchmark metric processing throughput"""
        tenant_id = 'throughput-test'
        metric_count = 1000
        batch_size = 100
        
        # Create test tenant
        tenant_config = {
            'tenant_id': tenant_id,
            'tier': 'enterprise_plus'  # No limits
        }
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create test components
        components = []
        for i in range(10):
            component = SystemComponent(
                component_id=f'throughput-component-{i:02d}',
                tenant_id=tenant_id,
                name=f'Throughput Component {i}',
                component_type=ComponentType.SERVICE
            )
            components.append(component)
            await health_service.register_system_component(component)
        
        # Prepare metrics
        metrics = []
        metric_names = ['cpu_utilization', 'memory_utilization', 'response_time', 'error_rate']
        
        for i in range(metric_count):
            metric = HealthMetric(
                tenant_id=tenant_id,
                component_id=components[i % len(components)].component_id,
                name=metric_names[i % len(metric_names)],
                value=50.0 + (i % 50),
                dimension=HealthDimension.PERFORMANCE
            )
            metrics.append(metric)
        
        # Measure processing time
        start_time = time.time()
        
        # Process in batches for better performance
        for i in range(0, len(metrics), batch_size):
            batch = metrics[i:i + batch_size]
            tasks = [
                health_service.process_health_metric(metric)
                for metric in batch
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        
        end_time = time.time()
        duration = end_time - start_time
        throughput = metric_count / duration
        
        print(f"\nThroughput Test Results:")
        print(f"Processed {metric_count} metrics in {duration:.2f} seconds")
        print(f"Throughput: {throughput:.2f} metrics/second")
        
        # Performance assertions
        assert throughput > 100, f"Throughput {throughput:.2f} below expected 100 metrics/second"
        assert duration < 30, f"Processing took {duration:.2f}s, expected < 30s"
    
    async def test_concurrent_tenant_processing(self, health_service):
        """Test concurrent processing across multiple tenants"""
        tenant_count = 5
        metrics_per_tenant = 100
        
        # Create tenants
        tenant_ids = []
        for i in range(tenant_count):
            tenant_config = {
                'tenant_id': f'concurrent-tenant-{i:02d}',
                'tier': 'professional'
            }
            result = await health_service.create_enterprise_tenant(tenant_config)
            assert result['status'] == 'success'
            tenant_ids.append(tenant_config['tenant_id'])
        
        # Create components for each tenant
        for tenant_id in tenant_ids:
            for j in range(3):  # 3 components per tenant
                component = SystemComponent(
                    component_id=f'{tenant_id}-component-{j}',
                    tenant_id=tenant_id,
                    name=f'Concurrent Component {j}',
                    component_type=ComponentType.SERVICE
                )
                await health_service.register_system_component(component)
        
        # Prepare concurrent tasks
        all_tasks = []
        start_time = time.time()
        
        for tenant_id in tenant_ids:
            for k in range(metrics_per_tenant):
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=f'{tenant_id}-component-{k % 3}',
                    name='concurrent_metric',
                    value=60.0 + (k % 40),
                    dimension=HealthDimension.PERFORMANCE
                )
                
                task = health_service.process_health_metric_with_enterprise_features(
                    metric, tenant_id
                )
                all_tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        error_count = len(results) - len(successful_results)
        
        total_operations = tenant_count * metrics_per_tenant
        throughput = total_operations / duration
        error_rate = error_count / total_operations
        
        print(f"\nConcurrent Processing Results:")
        print(f"Tenants: {tenant_count}, Metrics per tenant: {metrics_per_tenant}")
        print(f"Total operations: {total_operations}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Throughput: {throughput:.2f} ops/second")
        print(f"Error rate: {error_rate:.2%}")
        
        # Performance assertions
        assert error_rate < 0.05, f"Error rate {error_rate:.2%} too high"
        assert throughput > 50, f"Concurrent throughput {throughput:.2f} below expected 50 ops/sec"
    
    async def test_memory_usage_under_load(self, health_service):
        """Test memory usage stability under sustained load"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        tenant_id = 'memory-load-test'
        
        # Create tenant
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise'}
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create components
        components = []
        for i in range(20):
            component = SystemComponent(
                component_id=f'memory-component-{i:02d}',
                tenant_id=tenant_id,
                name=f'Memory Component {i}',
                component_type=ComponentType.SERVICE
            )
            components.append(component)
            await health_service.register_system_component(component)
        
        memory_samples = [initial_memory]
        
        # Process load in phases
        for phase in range(5):
            print(f"Memory test phase {phase + 1}/5")
            
            # Process batch of metrics
            tasks = []
            for i in range(200):  # 200 metrics per phase
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=components[i % len(components)].component_id,
                    name=f'memory_metric_{phase}',
                    value=50.0 + (i % 50),
                    dimension=HealthDimension.PERFORMANCE
                )
                tasks.append(health_service.process_health_metric(metric))
            
            await asyncio.gather(*tasks, return_exceptions=True)

            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Short delay between phases
            await asyncio.sleep(0.5)
        
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory Usage Results:")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Peak memory: {max_memory:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        
        # Memory assertions
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB, expected < 50MB"
        assert max_memory < initial_memory + 100, f"Peak memory {max_memory:.2f}MB too high"
    
    async def test_database_operation_performance(self, health_service):
        """Test database operation performance"""
        tenant_id = 'db-performance-test'
        operation_count = 500
        
        # Create tenant
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise'}
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Time component registrations
        start_time = time.time()
        
        registration_tasks = []
        for i in range(operation_count):
            component = SystemComponent(
                component_id=f'db-perf-component-{i:03d}',
                tenant_id=tenant_id,
                name=f'DB Performance Component {i}',
                component_type=ComponentType.SERVICE
            )
            registration_tasks.append(
                health_service.register_system_component(component)
            )
        
        await asyncio.gather(*registration_tasks, return_exceptions=True)

        
        registration_time = time.time() - start_time
        registration_throughput = operation_count / registration_time
        
        # Time health assessments
        start_time = time.time()
        
        assessment_tasks = []
        for i in range(operation_count):
            task = health_service.assess_component_health(
                f'db-perf-component-{i:03d}', tenant_id
            )
            assessment_tasks.append(task)
        
        await asyncio.gather(*assessment_tasks, return_exceptions=True)

        
        assessment_time = time.time() - start_time
        assessment_throughput = operation_count / assessment_time
        
        print(f"\nDatabase Performance Results:")
        print(f"Component registrations: {registration_throughput:.2f} ops/sec")
        print(f"Health assessments: {assessment_throughput:.2f} ops/sec")
        
        # Performance assertions
        assert registration_throughput > 50, f"Registration throughput {registration_throughput:.2f} too low"
        assert assessment_throughput > 30, f"Assessment throughput {assessment_throughput:.2f} too low"
    
    async def test_ml_prediction_performance(self, health_service):
        """Test ML prediction performance under load"""
        tenant_id = 'ml-performance-test'
        prediction_count = 50
        
        # Create tenant
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise'}
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create components and generate training data
        components = []
        for i in range(10):
            component = SystemComponent(
                component_id=f'ml-perf-component-{i:02d}',
                tenant_id=tenant_id,
                name=f'ML Performance Component {i}',
                component_type=ComponentType.SERVICE
            )
            components.append(component)
            await health_service.register_system_component(component)
            
            # Generate some historical data
            for j in range(10):
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=component.component_id,
                    name='performance_data',
                    value=50.0 + (j * 5),
                    dimension=HealthDimension.PERFORMANCE
                )
                await health_service.process_health_metric(metric)
        
        # Time ML predictions
        start_time = time.time()
        
        prediction_tasks = []
        for i in range(prediction_count):
            component = components[i % len(components)]
            task = health_service.predict_component_health_advanced(
                component.component_id, tenant_id, 24
            )
            prediction_tasks.append(task)
        
        predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
        
        prediction_time = time.time() - start_time
        
        # Filter successful predictions
        successful_predictions = [
            p for p in predictions 
            if not isinstance(p, Exception) and 'error' not in p
        ]
        
        prediction_throughput = len(successful_predictions) / prediction_time
        success_rate = len(successful_predictions) / prediction_count
        
        print(f"\nML Prediction Performance Results:")
        print(f"Predictions attempted: {prediction_count}")
        print(f"Successful predictions: {len(successful_predictions)}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Prediction throughput: {prediction_throughput:.2f} predictions/sec")
        print(f"Average prediction time: {prediction_time / prediction_count:.3f} seconds")
        
        # Performance assertions
        assert success_rate > 0.8, f"Prediction success rate {success_rate:.2%} too low"
        assert prediction_throughput > 5, f"Prediction throughput {prediction_throughput:.2f} too low"
    
    async def test_dashboard_generation_performance(self, health_service):
        """Test dashboard generation performance"""
        tenant_id = 'dashboard-performance-test'
        dashboard_count = 20
        
        # Create tenant with data
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise'}
        await health_service.create_enterprise_tenant(tenant_config)
        
        # Create components and metrics for realistic dashboard data
        for i in range(10):
            component = SystemComponent(
                component_id=f'dashboard-component-{i:02d}',
                tenant_id=tenant_id,
                name=f'Dashboard Component {i}',
                component_type=ComponentType.SERVICE
            )
            await health_service.register_system_component(component)
            
            # Add various metrics
            metrics = [
                ('cpu_utilization', 60 + (i * 2)),
                ('memory_utilization', 50 + (i * 3)),
                ('response_time', 100 + (i * 10)),
                ('error_rate', 0.01 + (i * 0.005))
            ]
            
            for metric_name, value in metrics:
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=component.component_id,
                    name=metric_name,
                    value=value,
                    dimension=HealthDimension.PERFORMANCE
                )
                await health_service.process_health_metric(metric)
        
        # Time dashboard generations
        dashboard_types = ['executive', 'operational', 'predictive']
        
        start_time = time.time()
        
        dashboard_tasks = []
        for i in range(dashboard_count):
            dashboard_type = dashboard_types[i % len(dashboard_types)]
            task = health_service.create_tenant_health_dashboard(
                tenant_id, dashboard_type
            )
            dashboard_tasks.append(task)
        
        dashboards = await asyncio.gather(*dashboard_tasks, return_exceptions=True)
        
        dashboard_time = time.time() - start_time
        
        # Analyze results
        successful_dashboards = [
            d for d in dashboards 
            if not isinstance(d, Exception) and 'error' not in d
        ]
        
        dashboard_throughput = len(successful_dashboards) / dashboard_time
        success_rate = len(successful_dashboards) / dashboard_count
        
        print(f"\nDashboard Performance Results:")
        print(f"Dashboards generated: {dashboard_count}")
        print(f"Successful dashboards: {len(successful_dashboards)}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Dashboard throughput: {dashboard_throughput:.2f} dashboards/sec")
        print(f"Average generation time: {dashboard_time / dashboard_count:.3f} seconds")
        
        # Performance assertions
        assert success_rate > 0.9, f"Dashboard success rate {success_rate:.2%} too low"
        assert dashboard_throughput > 2, f"Dashboard throughput {dashboard_throughput:.2f} too low"
    
    async def test_end_to_end_performance(self, health_service):
        """Test end-to-end performance of complete workflows"""
        tenant_count = 3
        components_per_tenant = 5
        metrics_per_component = 10
        
        workflow_start = time.time()
        
        # Phase 1: Tenant creation
        phase1_start = time.time()
        
        tenant_ids = []
        for i in range(tenant_count):
            tenant_config = {
                'tenant_id': f'e2e-perf-tenant-{i}',
                'tier': 'professional'
            }
            result = await health_service.create_enterprise_tenant(tenant_config)
            assert result['status'] == 'success'
            tenant_ids.append(tenant_config['tenant_id'])
        
        phase1_time = time.time() - phase1_start
        
        # Phase 2: Component registration
        phase2_start = time.time()
        
        all_components = []
        registration_tasks = []
        
        for tenant_id in tenant_ids:
            for j in range(components_per_tenant):
                component = SystemComponent(
                    component_id=f'{tenant_id}-e2e-component-{j}',
                    tenant_id=tenant_id,
                    name=f'E2E Component {j}',
                    component_type=ComponentType.SERVICE
                )
                all_components.append(component)
                registration_tasks.append(
                    health_service.register_system_component(component)
                )
        
        await asyncio.gather(*registration_tasks, return_exceptions=True)

        phase2_time = time.time() - phase2_start
        
        # Phase 3: Metric processing
        phase3_start = time.time()
        
        metric_tasks = []
        for component in all_components:
            for k in range(metrics_per_component):
                metric = HealthMetric(
                    tenant_id=component.tenant_id,
                    component_id=component.component_id,
                    name='e2e_performance_metric',
                    value=50.0 + (k * 5),
                    dimension=HealthDimension.PERFORMANCE
                )
                metric_tasks.append(
                    health_service.process_health_metric_with_enterprise_features(
                        metric, component.tenant_id
                    )
                )
        
        await asyncio.gather(*metric_tasks, return_exceptions=True)

        phase3_time = time.time() - phase3_start
        
        # Phase 4: Health assessments
        phase4_start = time.time()
        
        assessment_tasks = []
        for component in all_components:
            assessment_tasks.append(
                health_service.assess_component_health(
                    component.component_id, component.tenant_id
                )
            )
        
        await asyncio.gather(*assessment_tasks, return_exceptions=True)

        phase4_time = time.time() - phase4_start
        
        # Phase 5: Report generation
        phase5_start = time.time()
        
        report_tasks = []
        for tenant_id in tenant_ids:
            report_tasks.append(
                health_service.generate_health_report(
                    tenant_id=tenant_id,
                    report_type='comprehensive'
                )
            )
        
        await asyncio.gather(*report_tasks, return_exceptions=True)

        phase5_time = time.time() - phase5_start
        
        total_time = time.time() - workflow_start
        
        # Calculate metrics
        total_operations = (
            len(tenant_ids) +                    # Tenant creation
            len(all_components) +                # Component registration
            len(metric_tasks) +                  # Metric processing
            len(assessment_tasks) +              # Health assessments
            len(report_tasks)                    # Report generation
        )
        
        overall_throughput = total_operations / total_time
        
        print(f"\nEnd-to-End Performance Results:")
        print(f"Phase 1 (Tenant Creation): {phase1_time:.2f}s")
        print(f"Phase 2 (Component Registration): {phase2_time:.2f}s")
        print(f"Phase 3 (Metric Processing): {phase3_time:.2f}s")
        print(f"Phase 4 (Health Assessments): {phase4_time:.2f}s")
        print(f"Phase 5 (Report Generation): {phase5_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Operations: {total_operations}")
        print(f"Overall Throughput: {overall_throughput:.2f} ops/sec")
        
        # Performance assertions
        assert total_time < 60, f"E2E workflow took {total_time:.2f}s, expected < 60s"
        assert overall_throughput > 10, f"E2E throughput {overall_throughput:.2f} too low"
    
    async def test_stress_test_sustained_load(self, health_service):
        """Stress test with sustained load over time"""
        tenant_id = 'stress-test'
        duration_seconds = 30
        ops_per_second = 20
        
        # Create tenant and components
        tenant_config = {'tenant_id': tenant_id, 'tier': 'enterprise_plus'}
        await health_service.create_enterprise_tenant(tenant_config)
        
        components = []
        for i in range(5):
            component = SystemComponent(
                component_id=f'stress-component-{i}',
                tenant_id=tenant_id,
                name=f'Stress Component {i}',
                component_type=ComponentType.SERVICE
            )
            components.append(component)
            await health_service.register_system_component(component)
        
        # Run sustained load
        start_time = time.time()
        operation_count = 0
        error_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            interval_start = time.time()
            
            # Process operations for this interval
            tasks = []
            for _ in range(ops_per_second):
                component = components[operation_count % len(components)]
                metric = HealthMetric(
                    tenant_id=tenant_id,
                    component_id=component.component_id,
                    name='stress_metric',
                    value=50.0 + (operation_count % 50),
                    dimension=HealthDimension.PERFORMANCE
                )
                
                tasks.append(
                    health_service.process_health_metric(metric)
                )
                operation_count += 1
            
            # Execute interval operations
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, Exception)]
                error_count += len(errors)
            except Exception as e:
                error_count += len(tasks)
            
            # Wait for next interval
            interval_time = time.time() - interval_start
            sleep_time = max(0, 1.0 - interval_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        actual_duration = time.time() - start_time
        actual_throughput = operation_count / actual_duration
        error_rate = error_count / operation_count if operation_count > 0 else 0
        
        print(f"\nStress Test Results:")
        print(f"Target duration: {duration_seconds}s")
        print(f"Actual duration: {actual_duration:.2f}s")
        print(f"Target throughput: {ops_per_second} ops/sec")
        print(f"Actual throughput: {actual_throughput:.2f} ops/sec")
        print(f"Total operations: {operation_count}")
        print(f"Errors: {error_count}")
        print(f"Error rate: {error_rate:.2%}")
        
        # Stress test assertions
        assert error_rate < 0.1, f"Error rate {error_rate:.2%} too high under stress"
        assert actual_throughput > ops_per_second * 0.8, f"Throughput {actual_throughput:.2f} below 80% of target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])