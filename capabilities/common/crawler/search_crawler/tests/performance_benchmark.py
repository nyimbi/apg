#!/usr/bin/env python3
"""
Search Crawler Performance Benchmark Suite
===========================================

Comprehensive performance testing and benchmarking for the Search Crawler package.
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
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures
import statistics

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
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SearchCrawlerBenchmark:
    """Performance benchmark suite for Search Crawler package."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_memory = self._get_memory_usage()
        
        # Benchmark configuration
        self.config = {
            'iterations': 10,
            'max_results': 5,  # Small for benchmarking
            'timeout': 30,
            'enable_network_tests': True,
            'enable_crawlee_tests': False  # Disable for pure performance testing
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    async def benchmark_test(self, test_name: str, test_func, iterations: int = None):
        """Run a benchmark test and record results."""
        iterations = iterations or self.config['iterations']
        print(f"🔥 Benchmarking: {test_name} ({iterations} iterations)")
        
        # Cleanup before test
        gc.collect()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        
        # Run test iterations
        durations = []
        start_time = time.perf_counter()
        cpu_start = time.process_time()
        
        for i in range(iterations):
            iter_start = time.perf_counter()
            
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            
            iter_duration = time.perf_counter() - iter_start
            durations.append(iter_duration)
            
            # Track peak memory
            current_memory = self._get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        
        end_time = time.perf_counter()
        cpu_end = time.process_time()
        
        # Calculate metrics
        total_duration = end_time - start_time
        avg_duration = statistics.mean(durations)
        memory_used = self._get_memory_usage() - start_memory
        ops_per_second = iterations / total_duration if total_duration > 0 else 0
        cpu_usage = ((cpu_end - cpu_start) / total_duration) * 100 if total_duration > 0 else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            duration_seconds=total_duration,
            memory_usage_mb=memory_used,
            operations_per_second=ops_per_second,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=cpu_usage,
            additional_metrics={
                'iterations': iterations,
                'avg_iteration_time': avg_duration,
                'min_iteration_time': min(durations),
                'max_iteration_time': max(durations),
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
            }
        )
        
        self.results.append(result)
        
        print(f"   Duration: {total_duration:.3f}s (avg: {avg_duration:.3f}s)")
        print(f"   Memory: {memory_used:.2f}MB (Peak: {peak_memory:.2f}MB)")
        print(f"   Ops/sec: {ops_per_second:.1f}")
        print(f"   CPU: {cpu_usage:.1f}%")
        
        return result
    
    async def benchmark_configuration_loading(self):
        """Benchmark configuration loading performance."""
        from core.search_crawler import SearchCrawlerConfig
        from core.conflict_search_crawler import ConflictSearchConfig
        
        def load_basic_config():
            return SearchCrawlerConfig(
                engines=['google', 'bing', 'duckduckgo'],
                max_results_per_engine=10,
                total_max_results=30
            )
        
        def load_conflict_config():
            return ConflictSearchConfig(
                engines=['google', 'bing', 'duckduckgo', 'yandex'],
                conflict_regions=['horn_of_africa'],
                enable_alerts=True,
                escalation_threshold=0.8,
                trusted_sources=['reuters.com', 'bbc.com']
            )
        
        await self.benchmark_test("Basic Config Loading", load_basic_config, iterations=100)
        await self.benchmark_test("Conflict Config Loading", load_conflict_config, iterations=100)
        
        # Test Crawlee config if available
        try:
            from core.crawlee_enhanced_search_crawler import create_crawlee_search_config
            
            def load_crawlee_config():
                return create_crawlee_search_config(
                    engines=['google', 'bing'],
                    max_results=20,
                    enable_content_extraction=True,
                    target_countries=['ET', 'SO', 'KE']
                )
            
            await self.benchmark_test("Crawlee Config Loading", load_crawlee_config, iterations=100)
        except ImportError:
            print("⚠️ Crawlee config benchmarking skipped - not available")
    
    async def benchmark_search_engine_initialization(self):
        """Benchmark search engine initialization."""
        from engines import SEARCH_ENGINES, create_engine
        
        def init_single_engine():
            engine = create_engine('duckduckgo')
            return engine
        
        def init_multiple_engines():
            engines = {}
            for engine_name in ['google', 'bing', 'duckduckgo']:
                if engine_name in SEARCH_ENGINES:
                    engines[engine_name] = create_engine(engine_name)
            return engines
        
        await self.benchmark_test("Single Engine Init", init_single_engine, iterations=50)
        await self.benchmark_test("Multiple Engine Init", init_multiple_engines, iterations=20)
    
    async def benchmark_keyword_processing(self):
        """Benchmark keyword processing performance."""
        from keywords.conflict_keywords import ConflictKeywordManager
        from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
        from keywords.keyword_analyzer import KeywordAnalyzer
        
        # Initialize once
        conflict_manager = ConflictKeywordManager()
        hoa_keywords = HornOfAfricaKeywords()
        analyzer = KeywordAnalyzer()
        
        def get_conflict_keywords():
            high_priority = conflict_manager.get_high_priority_keywords()
            weighted = conflict_manager.get_weighted_keywords()
            violence = conflict_manager.get_keywords_by_category('violence')
            return len(high_priority) + len(weighted) + len(violence)
        
        def get_geographic_keywords():
            ethiopia = hoa_keywords.get_country_keywords('ethiopia')
            somalia = hoa_keywords.get_country_keywords('somalia')
            all_locations = hoa_keywords.get_all_location_keywords()
            return len(ethiopia) + len(somalia) + len(all_locations)
        
        def analyze_text():
            test_text = "Violence has erupted in Ethiopia and Somalia. The conflict involves armed groups and has led to displacement."
            keywords = ['violence', 'conflict', 'Ethiopia', 'Somalia', 'displacement']
            analysis = analyzer.analyze_text(test_text, keywords)
            return len(analysis) if analysis else 0
        
        await self.benchmark_test("Conflict Keyword Retrieval", get_conflict_keywords, iterations=100)
        await self.benchmark_test("Geographic Keyword Retrieval", get_geographic_keywords, iterations=100)
        await self.benchmark_test("Text Analysis", analyze_text, iterations=50)
    
    async def benchmark_search_operations(self):
        """Benchmark basic search operations."""
        if not self.config['enable_network_tests']:
            print("⚠️ Network benchmarks skipped - disabled")
            return
        
        from core.search_crawler import SearchCrawler, SearchCrawlerConfig
        
        # Initialize crawler once
        config = SearchCrawlerConfig(
            engines=['duckduckgo'],  # Single engine for consistent testing
            max_results_per_engine=self.config['max_results'],
            total_max_results=self.config['max_results'],
            download_content=False,  # Disable for pure search benchmarking
            timeout=self.config['timeout']
        )
        
        crawler = SearchCrawler(config)
        
        async def perform_search():
            results = await crawler.search("test query", max_results=self.config['max_results'])
            return len(results)
        
        # Benchmark search performance
        await self.benchmark_test("Basic Search Operation", perform_search, iterations=5)
        
        await crawler.close()
    
    async def benchmark_conflict_search_operations(self):
        """Benchmark conflict-specific search operations."""
        if not self.config['enable_network_tests']:
            print("⚠️ Conflict search benchmarks skipped - disabled")
            return
        
        from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
        
        config = ConflictSearchConfig(
            engines=['duckduckgo'],
            max_results_per_engine=self.config['max_results'],
            timeout=self.config['timeout']
        )
        
        conflict_crawler = ConflictSearchCrawler(config)
        
        async def perform_conflict_search():
            results = await conflict_crawler.search_conflicts(
                region='horn_of_africa',
                keywords=['Ethiopia', 'conflict'],
                max_results=self.config['max_results']
            )
            return len(results)
        
        await self.benchmark_test("Conflict Search Operation", perform_conflict_search, iterations=3)
        
        await conflict_crawler.close()
    
    async def benchmark_content_analysis(self):
        """Benchmark content analysis operations."""
        try:
            from core.crawlee_enhanced_search_crawler import CrawleeEnhancedSearchCrawler, create_crawlee_search_config
        except ImportError:
            print("⚠️ Content analysis benchmarks skipped - Crawlee not available")
            return
        
        config = create_crawlee_search_config()
        crawler = CrawleeEnhancedSearchCrawler(config)
        
        # Test content for analysis
        test_content = """
        Violence has erupted in multiple regions of Ethiopia and Somalia, affecting thousands of civilians.
        The conflict involves various armed groups and has led to significant displacement of populations.
        Humanitarian organizations are struggling to provide aid in the affected areas of the Horn of Africa.
        Kenya and Uganda are monitoring the situation closely and have increased security along their borders.
        The international community has called for immediate ceasefire and peaceful resolution.
        """
        
        def extract_geographic_entities():
            return crawler._extract_geographic_entities(test_content)
        
        def extract_conflict_indicators():
            return crawler._extract_conflict_indicators(test_content)
        
        def calculate_relevance_score():
            geo_entities = crawler._extract_geographic_entities(test_content)
            conflict_indicators = crawler._extract_conflict_indicators(test_content)
            return crawler._calculate_relevance_score(test_content, geo_entities, conflict_indicators)
        
        def score_extraction_result():
            result = {
                'content': test_content,
                'title': 'Test Article',
                'authors': ['Test Author'],
                'published_date': '2025-06-28',
                'keywords': ['conflict', 'Ethiopia', 'Somalia']
            }
            return crawler._score_extraction_result(result)
        
        await self.benchmark_test("Geographic Entity Extraction", extract_geographic_entities, iterations=100)
        await self.benchmark_test("Conflict Indicator Extraction", extract_conflict_indicators, iterations=100)
        await self.benchmark_test("Relevance Score Calculation", calculate_relevance_score, iterations=100)
        await self.benchmark_test("Extraction Result Scoring", score_extraction_result, iterations=100)
    
    async def benchmark_concurrent_operations(self):
        """Benchmark concurrent operations performance."""
        from core.search_crawler import SearchCrawler, SearchCrawlerConfig
        
        async def concurrent_task():
            config = SearchCrawlerConfig(
                engines=['duckduckgo'],
                max_results_per_engine=2,
                timeout=15
            )
            crawler = SearchCrawler(config)
            
            try:
                results = await crawler.search("test", max_results=2)
                return len(results)
            finally:
                await crawler.close()
        
        async def run_concurrent_searches():
            if not self.config['enable_network_tests']:
                return 0
            
            # Run 3 concurrent searches
            tasks = [concurrent_task() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful results
            successful = sum(1 for r in results if isinstance(r, int))
            return successful
        
        await self.benchmark_test("Concurrent Search Operations", run_concurrent_searches, iterations=2)
    
    async def benchmark_memory_efficiency(self):
        """Test memory efficiency with varying data sizes."""
        from core.search_crawler import SearchCrawler, SearchCrawlerConfig
        
        sizes = [10, 50, 100]
        
        for size in sizes:
            start_memory = self._get_memory_usage()
            
            # Create multiple crawlers to test memory usage
            crawlers = []
            for i in range(size):
                config = SearchCrawlerConfig(
                    engines=['duckduckgo'],
                    max_results_per_engine=1
                )
                crawler = SearchCrawler(config)
                crawlers.append(crawler)
            
            end_memory = self._get_memory_usage()
            memory_per_crawler = (end_memory - start_memory) / size if size > 0 else 0
            
            # Clean up
            for crawler in crawlers:
                await crawler.close()
            
            result = BenchmarkResult(
                test_name=f"Memory Efficiency ({size} crawlers)",
                duration_seconds=0,
                memory_usage_mb=end_memory - start_memory,
                operations_per_second=0,
                peak_memory_mb=end_memory,
                cpu_usage_percent=0,
                additional_metrics={
                    'crawlers_created': size,
                    'memory_per_crawler_mb': memory_per_crawler,
                    'total_memory_mb': end_memory - start_memory
                }
            )
            
            self.results.append(result)
            print(f"📊 Memory test ({size} crawlers): {memory_per_crawler:.2f} MB/crawler")
            
            # Cleanup
            del crawlers
            gc.collect()
    
    async def benchmark_ranking_algorithms(self):
        """Benchmark ranking algorithm performance."""
        try:
            from ranking.result_ranker import ResultRanker
        except ImportError:
            print("⚠️ Ranking benchmarks skipped - not available")
            return
        
        # Create mock results for ranking
        from core.search_crawler import EnhancedSearchResult
        from datetime import datetime
        
        def create_mock_results(count: int):
            results = []
            for i in range(count):
                result = EnhancedSearchResult(
                    title=f"Test Result {i}",
                    url=f"https://example.com/result-{i}",
                    snippet=f"This is test snippet {i}",
                    engine="test",
                    rank=i+1,
                    timestamp=datetime.now(),
                    relevance_score=0.8 - (i * 0.1),
                    engines_found=["test"],
                    engine_ranks={"test": i+1}
                )
                results.append(result)
            return results
        
        # Test different ranking strategies
        strategies = ['relevance', 'freshness', 'authority', 'hybrid']
        result_counts = [10, 50, 100]
        
        for strategy in strategies:
            ranker = ResultRanker(strategy=strategy)
            
            for count in result_counts:
                mock_results = create_mock_results(count)
                
                def rank_results():
                    return ranker.rank(mock_results, query="test query")
                
                test_name = f"Ranking {strategy} ({count} results)"
                await self.benchmark_test(test_name, rank_results, iterations=20)
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("="*80)
        report.append("SEARCH CRAWLER - PERFORMANCE BENCHMARK REPORT")
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
        total_tests = len(self.results)
        avg_duration = sum(r.duration_seconds for r in self.results) / total_tests
        max_ops_per_sec = max((r.operations_per_second for r in self.results), default=0)
        total_memory = sum(r.memory_usage_mb for r in self.results)
        
        report.append("Performance Summary:")
        report.append("-" * 40)
        report.append(f"Total Benchmark Tests: {total_tests}")
        report.append(f"Average Test Duration: {avg_duration:.3f}s")
        report.append(f"Peak Operations/Second: {max_ops_per_sec:.1f}")
        report.append(f"Total Memory Used: {total_memory:.2f}MB")
        report.append("")
        
        # Detailed results
        report.append("Detailed Benchmark Results:")
        report.append("-" * 60)
        
        for result in self.results:
            report.append(f"\n🔥 {result.test_name}")
            report.append(f"   Duration: {result.duration_seconds:.3f}s")
            
            if result.operations_per_second > 0:
                report.append(f"   Operations/sec: {result.operations_per_second:.1f}")
            
            report.append(f"   Memory: {result.memory_usage_mb:.2f}MB (Peak: {result.peak_memory_mb:.2f}MB)")
            report.append(f"   CPU Usage: {result.cpu_usage_percent:.1f}%")
            
            if result.additional_metrics:
                for key, value in result.additional_metrics.items():
                    if isinstance(value, float):
                        report.append(f"   {key}: {value:.3f}")
                    else:
                        report.append(f"   {key}: {value}")
        
        # Performance grades
        report.append("\n" + "="*60)
        report.append("PERFORMANCE GRADES")
        report.append("="*60)
        
        # Grade categories
        categories = {
            "Configuration": [r for r in self.results if "Config" in r.test_name],
            "Keyword Processing": [r for r in self.results if "Keyword" in r.test_name or "Text Analysis" in r.test_name],
            "Search Operations": [r for r in self.results if "Search" in r.test_name],
            "Content Analysis": [r for r in self.results if "Entity" in r.test_name or "Indicator" in r.test_name or "Relevance" in r.test_name],
            "Memory Efficiency": [r for r in self.results if "Memory" in r.test_name]
        }
        
        for category, results in categories.items():
            if results:
                if any(r.operations_per_second > 0 for r in results):
                    avg_ops = sum(r.operations_per_second for r in results if r.operations_per_second > 0) / len([r for r in results if r.operations_per_second > 0])
                    
                    if avg_ops > 1000:
                        grade = "A"
                    elif avg_ops > 100:
                        grade = "B"
                    elif avg_ops > 10:
                        grade = "C"
                    else:
                        grade = "D"
                    
                    report.append(f"{category}: {grade} ({avg_ops:.1f} ops/sec)")
                else:
                    avg_duration = sum(r.duration_seconds for r in results) / len(results)
                    
                    if avg_duration < 0.1:
                        grade = "A"
                    elif avg_duration < 1.0:
                        grade = "B"
                    elif avg_duration < 5.0:
                        grade = "C"
                    else:
                        grade = "D"
                    
                    report.append(f"{category}: {grade} ({avg_duration:.3f}s avg)")
        
        # Overall performance assessment
        report.append("\n" + "="*80)
        report.append("OVERALL PERFORMANCE ASSESSMENT")
        report.append("="*80)
        
        # Calculate overall score based on various factors
        fast_tests = sum(1 for r in self.results if r.operations_per_second > 100 or r.duration_seconds < 1.0)
        efficiency_rate = (fast_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if efficiency_rate >= 80:
            report.append("🚀 EXCELLENT: High-performance system ready for production")
        elif efficiency_rate >= 70:
            report.append("✅ GOOD: Solid performance suitable for most use cases")
        elif efficiency_rate >= 60:
            report.append("👍 ACCEPTABLE: Adequate performance for light usage")
        else:
            report.append("⚠️ NEEDS OPTIMIZATION: Performance may need improvement")
        
        report.append(f"\nPerformance Efficiency: {efficiency_rate:.1f}% ({fast_tests}/{total_tests} tests)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


async def main():
    """Run performance benchmarks."""
    print("🚀 Search Crawler Performance Benchmark Suite")
    print("="*60)
    
    benchmark = SearchCrawlerBenchmark()
    
    try:
        # Run all benchmark categories
        await benchmark.benchmark_configuration_loading()
        await benchmark.benchmark_search_engine_initialization()
        await benchmark.benchmark_keyword_processing()
        await benchmark.benchmark_search_operations()
        await benchmark.benchmark_conflict_search_operations()
        await benchmark.benchmark_content_analysis()
        await benchmark.benchmark_concurrent_operations()
        await benchmark.benchmark_memory_efficiency()
        await benchmark.benchmark_ranking_algorithms()
        
        # Generate and save report
        report = benchmark.generate_performance_report()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 BENCHMARK COMPLETED")
        print("="*60)
        print(report)
        
        # Save report
        report_file = Path(__file__).parent / "performance_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = Path(__file__).parent / "benchmark_results.json"
        results_data = []
        for result in benchmark.results:
            results_data.append({
                'test_name': result.test_name,
                'duration_seconds': result.duration_seconds,
                'memory_usage_mb': result.memory_usage_mb,
                'operations_per_second': result.operations_per_second,
                'peak_memory_mb': result.peak_memory_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'additional_metrics': result.additional_metrics,
                'timestamp': result.timestamp.isoformat()
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📄 Performance report saved to: {report_file}")
        print(f"📊 Benchmark data saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)