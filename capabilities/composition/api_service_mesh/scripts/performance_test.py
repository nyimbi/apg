#!/usr/bin/env python3
"""
APG API Service Mesh - Performance Testing Script

Comprehensive performance testing framework for validating service mesh
performance, scalability, and resource utilization under various load conditions.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import sys
import signal

import aiohttp
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel


# =============================================================================
# Configuration and Data Models
# =============================================================================

@dataclass
class TestConfig:
    """Performance test configuration."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 50
    test_duration: int = 60
    ramp_up_time: int = 10
    endpoints: List[str] = None
    request_rate: float = 10.0  # requests per second per user
    timeout: int = 30
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = [
                "/api/health",
                "/api/services",
                "/api/metrics/query",
                "/api/topology"
            ]

@dataclass
class PerformanceMetrics:
    """Performance test results."""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_io: Dict[str, int]
    network_io: Dict[str, int]


# =============================================================================
# Performance Test Framework
# =============================================================================

class PerformanceTestRunner:
    """Main performance test runner."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.console = Console()
        self.results: List[PerformanceMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.response_times: List[float] = []
        self.status_codes: List[int] = []
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.console.print("\n[yellow]Stopping performance tests gracefully...[/yellow]")
        self.running = False
    
    async def run_all_tests(self) -> List[PerformanceMetrics]:
        """Run comprehensive performance test suite."""
        self.console.print("[bold blue]APG API Service Mesh - Performance Testing[/bold blue]")
        self.console.print(f"Base URL: {self.config.base_url}")
        self.console.print(f"Duration: {self.config.test_duration}s")
        self.console.print(f"Concurrent Users: {self.config.concurrent_users}")
        self.console.print()
        
        # Test scenarios
        test_scenarios = [
            ("Baseline Load Test", self._run_baseline_test),
            ("Stress Test", self._run_stress_test),
            ("Spike Test", self._run_spike_test),
            ("Endurance Test", self._run_endurance_test),
            ("Mixed Workload Test", self._run_mixed_workload_test)
        ]
        
        for test_name, test_func in test_scenarios:
            if not self.running:
                break
                
            self.console.print(f"[bold green]Running: {test_name}[/bold green]")
            
            try:
                metrics = await test_func()
                metrics.test_name = test_name
                self.results.append(metrics)
                
                # Display results
                self._display_test_results(metrics)
                
                # Cool down between tests
                if self.running:
                    await asyncio.sleep(5)
                    
            except Exception as e:
                self.console.print(f"[red]Test failed: {e}[/red]")
                continue
        
        # Generate comprehensive report
        await self._generate_report()
        
        return self.results
    
    async def _run_baseline_test(self) -> PerformanceMetrics:
        """Run baseline performance test with normal load."""
        return await self._run_test(
            concurrent_users=self.config.concurrent_users,
            duration=self.config.test_duration,
            request_rate=self.config.request_rate
        )
    
    async def _run_stress_test(self) -> PerformanceMetrics:
        """Run stress test with high load."""
        return await self._run_test(
            concurrent_users=self.config.concurrent_users * 2,
            duration=self.config.test_duration,
            request_rate=self.config.request_rate * 2
        )
    
    async def _run_spike_test(self) -> PerformanceMetrics:
        """Run spike test with sudden load increase."""
        return await self._run_test(
            concurrent_users=self.config.concurrent_users * 5,
            duration=30,  # Shorter duration for spike
            request_rate=self.config.request_rate * 3
        )
    
    async def _run_endurance_test(self) -> PerformanceMetrics:
        """Run endurance test for extended period."""
        return await self._run_test(
            concurrent_users=self.config.concurrent_users,
            duration=self.config.test_duration * 3,  # 3x longer
            request_rate=self.config.request_rate
        )
    
    async def _run_mixed_workload_test(self) -> PerformanceMetrics:
        """Run mixed workload test with various endpoints."""
        return await self._run_test(
            concurrent_users=self.config.concurrent_users,
            duration=self.config.test_duration,
            request_rate=self.config.request_rate,
            mixed_workload=True
        )
    
    async def _run_test(
        self,
        concurrent_users: int,
        duration: int,
        request_rate: float,
        mixed_workload: bool = False
    ) -> PerformanceMetrics:
        """Run individual performance test."""
        start_time = datetime.now(timezone.utc)
        
        # Reset metrics
        self.response_times = []
        self.status_codes = []
        self.system_metrics = []
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources())
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Create user tasks
            tasks = []
            for user_id in range(concurrent_users):
                if mixed_workload:
                    task = asyncio.create_task(
                        self._simulate_mixed_user(session, user_id, duration, request_rate)
                    )
                else:
                    task = asyncio.create_task(
                        self._simulate_user(session, user_id, duration, request_rate)
                    )
                tasks.append(task)
            
            # Wait for all users to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop monitoring
        monitor_task.cancel()
        
        end_time = datetime.now(timezone.utc)
        
        # Calculate metrics
        return self._calculate_metrics(start_time, end_time)
    
    async def _simulate_user(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        duration: int,
        request_rate: float
    ):
        """Simulate a single user making requests."""
        end_time = time.time() + duration
        request_interval = 1.0 / request_rate if request_rate > 0 else 1.0
        
        while time.time() < end_time and self.running:
            # Select endpoint (round-robin)
            endpoint = self.config.endpoints[user_id % len(self.config.endpoints)]
            url = f"{self.config.base_url}{endpoint}"
            
            # Make request
            await self._make_request(session, url)
            
            # Wait before next request
            await asyncio.sleep(request_interval)
    
    async def _simulate_mixed_user(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        duration: int,
        request_rate: float
    ):
        """Simulate user with mixed workload patterns."""
        end_time = time.time() + duration
        request_interval = 1.0 / request_rate if request_rate > 0 else 1.0
        
        # Define workload patterns
        patterns = [
            # Read-heavy workload
            {"GET": 0.8, "POST": 0.1, "PUT": 0.05, "DELETE": 0.05},
            # Write-heavy workload
            {"GET": 0.4, "POST": 0.3, "PUT": 0.2, "DELETE": 0.1},
            # Balanced workload
            {"GET": 0.6, "POST": 0.2, "PUT": 0.15, "DELETE": 0.05}
        ]
        
        pattern = patterns[user_id % len(patterns)]
        
        while time.time() < end_time and self.running:
            # Select endpoint and method based on pattern
            endpoint = self.config.endpoints[user_id % len(self.config.endpoints)]
            url = f"{self.config.base_url}{endpoint}"
            
            # Determine HTTP method based on pattern
            import random
            rand = random.random()
            cumulative = 0
            method = "GET"
            
            for http_method, probability in pattern.items():
                cumulative += probability
                if rand <= cumulative:
                    method = http_method
                    break
            
            # Make request
            await self._make_request(session, url, method)
            
            # Variable wait time for more realistic patterns
            jitter = random.uniform(0.5, 1.5)
            await asyncio.sleep(request_interval * jitter)
    
    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        method: str = "GET",
        data: Optional[Dict] = None
    ):
        """Make HTTP request and record metrics."""
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    await response.text()
                    status_code = response.status
            elif method == "POST":
                request_data = data or {"test": "data"}
                async with session.post(url, json=request_data) as response:
                    await response.text()
                    status_code = response.status
            elif method == "PUT":
                request_data = data or {"test": "update"}
                async with session.put(url, json=request_data) as response:
                    await response.text()
                    status_code = response.status
            elif method == "DELETE":
                async with session.delete(url) as response:
                    await response.text()
                    status_code = response.status
            else:
                status_code = 405  # Method not allowed
                
        except Exception as e:
            status_code = 0  # Connection error
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Record metrics
        self.response_times.append(response_time)
        self.status_codes.append(status_code)
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage during test."""
        while self.running:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_metrics = {
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes
                } if disk_io else {"read_bytes": 0, "write_bytes": 0}
                
                # Network I/O
                network_io = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv
                } if network_io else {"bytes_sent": 0, "bytes_recv": 0}
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(timezone.utc),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available=memory.available,
                    disk_io=disk_metrics,
                    network_io=network_metrics
                )
                
                self.system_metrics.append(metrics)
                
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to collect system metrics: {e}[/yellow]")
            
            await asyncio.sleep(1)
    
    def _calculate_metrics(self, start_time: datetime, end_time: datetime) -> PerformanceMetrics:
        """Calculate performance metrics from collected data."""
        total_requests = len(self.response_times)
        successful_requests = sum(1 for code in self.status_codes if 200 <= code < 400)
        failed_requests = total_requests - successful_requests
        
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            
            # Calculate percentiles
            sorted_times = sorted(self.response_times)
            p50_response_time = statistics.median(sorted_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        # Calculate throughput
        duration_seconds = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / duration_seconds if duration_seconds > 0 else 0
        
        # Calculate error rate
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # System resource averages
        if self.system_metrics:
            avg_cpu = statistics.mean(m.cpu_percent for m in self.system_metrics)
            avg_memory = statistics.mean(m.memory_percent for m in self.system_metrics)
            
            # Network I/O delta
            if len(self.system_metrics) > 1:
                first_network = self.system_metrics[0].network_io
                last_network = self.system_metrics[-1].network_io
                network_delta = {
                    "bytes_sent": last_network["bytes_sent"] - first_network["bytes_sent"],
                    "bytes_recv": last_network["bytes_recv"] - first_network["bytes_recv"]
                }
            else:
                network_delta = {"bytes_sent": 0, "bytes_recv": 0}
        else:
            avg_cpu = avg_memory = 0
            network_delta = {"bytes_sent": 0, "bytes_recv": 0}
        
        return PerformanceMetrics(
            test_name="",  # Will be set by caller
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            network_io=network_delta
        )
    
    def _display_test_results(self, metrics: PerformanceMetrics):
        """Display test results in a formatted table."""
        table = Table(title=f"Performance Test Results: {metrics.test_name}")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Unit", style="green")
        
        table.add_row("Total Requests", str(metrics.total_requests), "requests")
        table.add_row("Successful Requests", str(metrics.successful_requests), "requests")
        table.add_row("Failed Requests", str(metrics.failed_requests), "requests")
        table.add_row("Success Rate", f"{100 - metrics.error_rate:.2f}", "%")
        table.add_row("Requests/Second", f"{metrics.requests_per_second:.2f}", "req/s")
        table.add_row("Avg Response Time", f"{metrics.avg_response_time:.2f}", "ms")
        table.add_row("P50 Response Time", f"{metrics.p50_response_time:.2f}", "ms")
        table.add_row("P95 Response Time", f"{metrics.p95_response_time:.2f}", "ms")
        table.add_row("P99 Response Time", f"{metrics.p99_response_time:.2f}", "ms")
        table.add_row("CPU Usage", f"{metrics.cpu_usage:.2f}", "%")
        table.add_row("Memory Usage", f"{metrics.memory_usage:.2f}", "%")
        
        self.console.print(table)
        self.console.print()
    
    async def _generate_report(self):
        """Generate comprehensive performance test report."""
        if not self.results:
            self.console.print("[yellow]No test results to report[/yellow]")
            return
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"performance_results_{timestamp}.json"
        
        results_data = {
            "test_config": {
                "base_url": self.config.base_url,
                "concurrent_users": self.config.concurrent_users,
                "test_duration": self.config.test_duration,
                "endpoints": self.config.endpoints
            },
            "results": [result.to_dict() for result in self.results],
            "summary": self._generate_summary()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.console.print(f"[green]Results saved to: {results_file}[/green]")
        
        # Generate charts
        await self._generate_charts(timestamp)
        
        # Display summary
        self._display_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        if not self.results:
            return {}
        
        total_requests = sum(r.total_requests for r in self.results)
        total_failures = sum(r.failed_requests for r in self.results)
        avg_response_times = [r.avg_response_time for r in self.results]
        rps_values = [r.requests_per_second for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_success_rate": ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": statistics.mean(avg_response_times) if avg_response_times else 0,
            "max_throughput": max(rps_values) if rps_values else 0,
            "min_throughput": min(rps_values) if rps_values else 0
        }
    
    async def _generate_charts(self, timestamp: str):
        """Generate performance charts."""
        if not self.results:
            return
        
        # Response time comparison chart
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Response Times
        plt.subplot(2, 2, 1)
        test_names = [r.test_name for r in self.results]
        avg_times = [r.avg_response_time for r in self.results]
        p95_times = [r.p95_response_time for r in self.results]
        
        x = range(len(test_names))
        plt.bar([i - 0.2 for i in x], avg_times, 0.4, label='Average', alpha=0.7)
        plt.bar([i + 0.2 for i in x], p95_times, 0.4, label='P95', alpha=0.7)
        plt.xlabel('Test')
        plt.ylabel('Response Time (ms)')
        plt.title('Response Time Comparison')
        plt.xticks(x, test_names, rotation=45)
        plt.legend()
        
        # Subplot 2: Throughput
        plt.subplot(2, 2, 2)
        rps_values = [r.requests_per_second for r in self.results]
        plt.bar(test_names, rps_values, alpha=0.7, color='green')
        plt.xlabel('Test')
        plt.ylabel('Requests/Second')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45)
        
        # Subplot 3: Error Rates
        plt.subplot(2, 2, 3)
        error_rates = [r.error_rate for r in self.results]
        plt.bar(test_names, error_rates, alpha=0.7, color='red')
        plt.xlabel('Test')
        plt.ylabel('Error Rate (%)')
        plt.title('Error Rate Comparison')
        plt.xticks(rotation=45)
        
        # Subplot 4: Resource Usage
        plt.subplot(2, 2, 4)
        cpu_usage = [r.cpu_usage for r in self.results]
        memory_usage = [r.memory_usage for r in self.results]
        
        x = range(len(test_names))
        plt.bar([i - 0.2 for i in x], cpu_usage, 0.4, label='CPU %', alpha=0.7)
        plt.bar([i + 0.2 for i in x], memory_usage, 0.4, label='Memory %', alpha=0.7)
        plt.xlabel('Test')
        plt.ylabel('Usage (%)')
        plt.title('Resource Usage')
        plt.xticks(x, test_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        chart_file = f"performance_charts_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green]Charts saved to: {chart_file}[/green]")
    
    def _display_summary(self):
        """Display final test summary."""
        summary = self._generate_summary()
        
        panel_content = f"""
[bold]Performance Test Summary[/bold]

Total Tests: {summary.get('total_tests', 0)}
Total Requests: {summary.get('total_requests', 0):,}
Total Failures: {summary.get('total_failures', 0):,}
Overall Success Rate: {summary.get('overall_success_rate', 0):.2f}%

Average Response Time: {summary.get('avg_response_time', 0):.2f} ms
Max Throughput: {summary.get('max_throughput', 0):.2f} req/s
Min Throughput: {summary.get('min_throughput', 0):.2f} req/s
        """
        
        panel = Panel(panel_content.strip(), title="Test Summary", expand=False)
        self.console.print(panel)


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main entry point for performance testing."""
    parser = argparse.ArgumentParser(description="APG API Service Mesh Performance Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rate", type=float, default=10.0, help="Requests per second per user")
    parser.add_argument("--endpoints", nargs="+", help="Specific endpoints to test")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Create test configuration
    config = TestConfig(
        base_url=args.url,
        concurrent_users=args.users,
        test_duration=args.duration,
        request_rate=args.rate,
        timeout=args.timeout
    )
    
    if args.endpoints:
        config.endpoints = args.endpoints
    
    # Run performance tests
    runner = PerformanceTestRunner(config)
    
    try:
        results = await runner.run_all_tests()
        
        if results:
            console = Console()
            console.print("\n[bold green]Performance testing completed successfully![/bold green]")
            
            # Performance validation
            passed_tests = 0
            for result in results:
                if (result.error_rate < 5.0 and  # Less than 5% error rate
                    result.avg_response_time < 2000 and  # Less than 2s average response
                    result.p95_response_time < 5000):  # Less than 5s P95 response
                    passed_tests += 1
            
            console.print(f"Tests passed performance criteria: {passed_tests}/{len(results)}")
            
            if passed_tests < len(results):
                console.print("[yellow]Some tests failed performance criteria. Review results for optimization opportunities.[/yellow]")
                sys.exit(1)
        else:
            console.print("[red]No tests completed successfully[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Performance testing interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console = Console()
        console.print(f"[red]Performance testing failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())