#!/usr/bin/env python3
"""
Google News Crawler Performance Benchmark Suite
===============================================

Comprehensive performance testing and benchmarking for the Google News Crawler.
Tests response times, throughput, memory usage, and scalability characteristics.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import time
import sys
import os
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import concurrent.futures

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

@dataclass
class BenchmarkResult:
    """Stores benchmark test results."""
    test_name: str
    duration_seconds: float
    memory_usage_mb: float
    operations_per_second: float
    peak_memory_mb: float
    cpu_usage_percent: float
    additional_metrics: Dict[str, Any]

class PerformanceBenchmark:
    """Performance benchmark suite for Google News Crawler."""
    
    def __init__(self):
        self.results = []
        self.start_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    async def benchmark_test(self, test_name: str, test_func, iterations: int = 1):
        """Run a benchmark test and record results."""
        print(f"🔥 Benchmarking: {test_name}")
        
        # Cleanup before test
        gc.collect()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        
        # Run test
        start_time = time.perf_counter()
        cpu_start = time.process_time()
        
        for i in range(iterations):
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            
            # Track peak memory
            current_memory = self._get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        
        end_time = time.perf_counter()
        cpu_end = time.process_time()
        
        # Calculate metrics
        duration = end_time - start_time
        memory_used = self._get_memory_usage() - start_memory
        ops_per_second = iterations / duration if duration > 0 else 0
        cpu_usage = ((cpu_end - cpu_start) / duration) * 100 if duration > 0 else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            duration_seconds=duration,
            memory_usage_mb=memory_used,
            operations_per_second=ops_per_second,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=cpu_usage,
            additional_metrics={}
        )
        
        self.results.append(result)
        
        print(f"   Duration: {duration:.3f}s")
        print(f"   Memory: {memory_used:.2f}MB (Peak: {peak_memory:.2f}MB)")
        print(f"   Ops/sec: {ops_per_second:.1f}")
        print(f"   CPU: {cpu_usage:.1f}%")
        
        return result

    async def test_configuration_loading_performance(self):
        """Benchmark configuration loading performance."""
        from cli.utils import save_cli_config, load_cli_config
        import tempfile
        
        # Create test configuration
        config = {
            "database": {"url": "postgresql://test:test@localhost/test"},
            "crawlee": {
                "max_requests": 100,
                "max_concurrent": 5,
                "target_countries": ["ET", "SO", "KE"] * 10  # Larger config
            },
            "sources": {f"source_{i}": {"name": f"Source {i}", "url": f"https://source{i}.com"} for i in range(50)}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        # Save once
        save_cli_config(config_path, config)
        
        async def load_config():
            return load_cli_config(config_path)
        
        await self.benchmark_test("Configuration Loading", load_config, iterations=100)
        
        # Cleanup
        config_path.unlink()

    async def test_mock_database_performance(self):
        """Benchmark mock database operations."""
        from cli.utils import create_mock_db_manager
        
        db_manager = create_mock_db_manager()
        
        # Create test articles
        test_articles = [
            {
                'title': f'Test Article {i}',
                'url': f'https://example.com/article-{i}',
                'content': f'Content for article {i} with some longer text to simulate real articles',
                'published_date': datetime.now(),
                'source': 'Test Source'
            }
            for i in range(100)
        ]
        
        async def store_articles():
            return await db_manager.store_articles(test_articles)
        
        await self.benchmark_test("Mock Database Storage", store_articles, iterations=10)

    async def test_country_validation_performance(self):
        """Benchmark country validation performance."""
        from cli.utils import validate_countries
        
        # Test with various input sizes
        test_countries = ['ET', 'SO', 'KE', 'INVALID1', 'UG', 'TZ', 'INVALID2', 'SD', 'SS'] * 10
        
        def validate_test():
            return validate_countries(test_countries)
        
        await self.benchmark_test("Country Validation", validate_test, iterations=1000)

    async def test_date_parsing_performance(self):
        """Benchmark date parsing performance."""
        from cli.utils import parse_date_input
        
        test_dates = ['7d', '24h', '2w', '2025-06-28', '2025-12-31', '1m', '3d'] * 20
        
        def parse_dates():
            results = []
            for date_str in test_dates:
                result = parse_date_input(date_str)
                results.append(result)
            return results
        
        await self.benchmark_test("Date Parsing", parse_dates, iterations=100)

    async def test_query_enhancement_performance(self):
        """Benchmark search query enhancement."""
        def enhance_query(query: str) -> str:
            conflict_terms = ['conflict', 'violence', 'security', 'crisis', 'peace']
            if not any(term in query.lower() for term in conflict_terms):
                return f"{query} OR (conflict OR security OR crisis)"
            return query
        
        test_queries = [
            'Ethiopia news',
            'Somalia update',
            'Kenya economic development',
            'Sudan political situation',
            'Horn of Africa weather'
        ] * 20
        
        def enhance_queries():
            return [enhance_query(q) for q in test_queries]
        
        await self.benchmark_test("Query Enhancement", enhance_queries, iterations=100)

    async def test_content_quality_scoring_performance(self):
        """Benchmark content quality scoring."""
        def calculate_quality_score(content: str, title: str, metadata: Dict[str, Any]) -> float:
            score = 0.0
            word_count = len(content.split()) if content else 0
            
            if word_count >= 300:
                score += 0.4
            elif word_count >= 100:
                score += 0.2
            
            if title and len(title) > 10:
                score += 0.2
            
            if metadata.get('authors'):
                score += 0.2
            if metadata.get('published_date'):
                score += 0.2
            
            return min(score, 1.0)
        
        # Test content
        test_content = " ".join(["word"] * 500)  # 500 words
        test_title = "This is a comprehensive test article title"
        test_metadata = {'authors': ['Author'], 'published_date': datetime.now()}
        
        def score_content():
            return calculate_quality_score(test_content, test_title, test_metadata)
        
        await self.benchmark_test("Content Quality Scoring", score_content, iterations=1000)

    async def test_concurrent_operations(self):
        """Benchmark concurrent operations."""
        from cli.utils import validate_countries, parse_date_input
        
        async def concurrent_task():
            # Simulate concurrent validation and parsing
            countries = validate_countries(['ET', 'SO', 'KE', 'UG'])
            date_result = parse_date_input('7d')
            return len(countries), date_result is not None
        
        async def run_concurrent():
            tasks = [concurrent_task() for _ in range(50)]
            results = await asyncio.gather(*tasks)
            return len(results)
        
        await self.benchmark_test("Concurrent Operations", run_concurrent, iterations=10)

    async def test_large_data_processing(self):
        """Benchmark processing of large datasets."""
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            article = {
                'id': i,
                'title': f'Article {i} with comprehensive title information',
                'content': f'This is article {i} with substantial content. ' * 20,  # ~20 words * 20 = 400 words
                'metadata': {
                    'authors': [f'Author {i}', f'Co-author {i}'],
                    'tags': [f'tag{j}' for j in range(5)],
                    'published_date': datetime.now(),
                    'source': f'Source {i % 10}'
                }
            }
            large_dataset.append(article)
        
        def process_large_dataset():
            # Simulate processing operations
            processed = 0
            for article in large_dataset:
                # Simulate some processing
                word_count = len(article['content'].split())
                has_authors = bool(article['metadata']['authors'])
                is_recent = article['metadata']['published_date'] is not None
                
                if word_count > 100 and has_authors and is_recent:
                    processed += 1
            
            return processed
        
        await self.benchmark_test("Large Dataset Processing", process_large_dataset, iterations=5)

    async def test_memory_efficiency(self):
        """Test memory efficiency with varying data sizes."""
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            # Create data of specific size
            data = [{
                'id': i,
                'content': 'x' * 1000,  # 1KB per item
                'metadata': {'tags': [f'tag{j}' for j in range(10)]}
            } for i in range(size)]
            
            start_memory = self._get_memory_usage()
            
            # Process data
            processed_count = 0
            for item in data:
                if len(item['content']) > 500:
                    processed_count += 1
            
            end_memory = self._get_memory_usage()
            memory_per_item = (end_memory - start_memory) / size if size > 0 else 0
            
            result = BenchmarkResult(
                test_name=f"Memory Efficiency ({size} items)",
                duration_seconds=0,
                memory_usage_mb=end_memory - start_memory,
                operations_per_second=0,
                peak_memory_mb=end_memory,
                cpu_usage_percent=0,
                additional_metrics={
                    'items_processed': processed_count,
                    'memory_per_item_kb': memory_per_item * 1024,
                    'data_size': size
                }
            )
            
            self.results.append(result)
            print(f"📊 Memory test ({size} items): {memory_per_item*1024:.2f} KB/item")
            
            # Cleanup
            del data
            gc.collect()

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("="*80)
        report.append("Google News Crawler - Performance Benchmark Report")
        report.append("="*80)
        report.append(f"Benchmark Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Python Version: {sys.version.split()[0]}")
        report.append(f"Platform: {sys.platform}")
        
        # System info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        report.append(f"CPU Cores: {cpu_count}")
        report.append(f"Total Memory: {memory_gb:.1f} GB")
        report.append("")
        
        # Performance summary
        if self.results:
            avg_duration = sum(r.duration_seconds for r in self.results) / len(self.results)
            max_ops_per_sec = max((r.operations_per_second for r in self.results), default=0)
            total_memory = sum(r.memory_usage_mb for r in self.results)
            
            report.append("Performance Summary:")
            report.append("-" * 30)
            report.append(f"Average Test Duration: {avg_duration:.3f}s")
            report.append(f"Peak Operations/Second: {max_ops_per_sec:.1f}")
            report.append(f"Total Memory Used: {total_memory:.2f}MB")
            report.append("")
        
        # Detailed results
        report.append("Detailed Benchmark Results:")
        report.append("-" * 50)
        
        for result in self.results:
            report.append(f"\n🔥 {result.test_name}")
            report.append(f"   Duration: {result.duration_seconds:.3f}s")
            report.append(f"   Memory: {result.memory_usage_mb:.2f}MB (Peak: {result.peak_memory_mb:.2f}MB)")
            report.append(f"   Operations/sec: {result.operations_per_second:.1f}")
            report.append(f"   CPU Usage: {result.cpu_usage_percent:.1f}%")
            
            if result.additional_metrics:
                report.append("   Additional Metrics:")
                for key, value in result.additional_metrics.items():
                    report.append(f"     {key}: {value}")
        
        # Performance grades
        report.append("\n" + "="*50)
        report.append("PERFORMANCE GRADES")
        report.append("="*50)
        
        # Grade configuration loading
        config_results = [r for r in self.results if "Configuration" in r.test_name]
        if config_results:
            avg_config_ops = sum(r.operations_per_second for r in config_results) / len(config_results)
            config_grade = "A" if avg_config_ops > 1000 else "B" if avg_config_ops > 500 else "C"
            report.append(f"Configuration Loading: {config_grade} ({avg_config_ops:.1f} ops/sec)")
        
        # Grade database operations
        db_results = [r for r in self.results if "Database" in r.test_name]
        if db_results:
            avg_db_ops = sum(r.operations_per_second for r in db_results) / len(db_results)
            db_grade = "A" if avg_db_ops > 100 else "B" if avg_db_ops > 50 else "C"
            report.append(f"Database Operations: {db_grade} ({avg_db_ops:.1f} ops/sec)")
        
        # Grade validation operations
        validation_results = [r for r in self.results if "Validation" in r.test_name or "Parsing" in r.test_name]
        if validation_results:
            avg_val_ops = sum(r.operations_per_second for r in validation_results) / len(validation_results)
            val_grade = "A" if avg_val_ops > 5000 else "B" if avg_val_ops > 1000 else "C"
            report.append(f"Validation & Parsing: {val_grade} ({avg_val_ops:.1f} ops/sec)")
        
        # Overall memory efficiency
        memory_results = [r for r in self.results if r.memory_usage_mb > 0]
        if memory_results:
            avg_memory = sum(r.memory_usage_mb for r in memory_results) / len(memory_results)
            memory_grade = "A" if avg_memory < 50 else "B" if avg_memory < 100 else "C"
            report.append(f"Memory Efficiency: {memory_grade} ({avg_memory:.1f}MB avg)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

async def run_performance_benchmarks():
    """Run all performance benchmarks."""
    print("🚀 Google News Crawler Performance Benchmark Suite")
    print("="*60)
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmark tests
    await benchmark.test_configuration_loading_performance()
    await benchmark.test_mock_database_performance()
    await benchmark.test_country_validation_performance()
    await benchmark.test_date_parsing_performance()
    await benchmark.test_query_enhancement_performance()
    await benchmark.test_content_quality_scoring_performance()
    await benchmark.test_concurrent_operations()
    await benchmark.test_large_data_processing()
    await benchmark.test_memory_efficiency()
    
    # Generate and save report
    report = benchmark.generate_performance_report()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BENCHMARK COMPLETED")
    print("="*60)
    
    return report, benchmark.results

if __name__ == "__main__":
    # Run benchmarks
    report, results = asyncio.run(run_performance_benchmarks())
    
    # Print full report
    print(report)
    
    # Save to file
    report_file = Path(__file__).parent / "performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save results as JSON
    results_file = Path(__file__).parent / "benchmark_results.json"
    results_data = []
    for result in results:
        results_data.append({
            'test_name': result.test_name,
            'duration_seconds': result.duration_seconds,
            'memory_usage_mb': result.memory_usage_mb,
            'operations_per_second': result.operations_per_second,
            'peak_memory_mb': result.peak_memory_mb,
            'cpu_usage_percent': result.cpu_usage_percent,
            'additional_metrics': result.additional_metrics
        })
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    print(f"📊 Benchmark data saved to: {results_file}")
    
    # Performance validation
    total_tests = len(results)
    fast_tests = sum(1 for r in results if r.operations_per_second > 100)
    efficiency_rate = (fast_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 Performance Summary: {fast_tests}/{total_tests} tests achieved >100 ops/sec ({efficiency_rate:.1f}%)")
    
    exit_code = 0 if efficiency_rate >= 70 else 1
    sys.exit(exit_code)