#!/usr/bin/env python3
"""
APG API Service Mesh - Health Check Script

Comprehensive health checking and monitoring script for validating
service mesh health, dependencies, and overall system status.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import socket
import ssl
import subprocess

import aiohttp
import asyncpg
import redis.asyncio as redis
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.status import Status


# =============================================================================
# Configuration and Data Models
# =============================================================================

@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    api_url: str = "http://localhost:8000"
    database_url: str = "postgresql://asm_user:asm_password@localhost:5432/api_service_mesh_dev"
    redis_url: str = "redis://localhost:6379/0"
    timeout: int = 30
    retries: int = 3
    retry_delay: int = 5
    detailed: bool = False

@dataclass
class HealthStatus:
    """Health check result for a component."""
    component: str
    status: str  # "healthy", "unhealthy", "degraded", "unknown"
    response_time: float
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str
    components: List[HealthStatus]
    overall_response_time: float
    healthy_components: int
    total_components: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "components": [comp.to_dict() for comp in self.components],
            "overall_response_time": self.overall_response_time,
            "healthy_components": self.healthy_components,
            "total_components": self.total_components,
            "timestamp": self.timestamp.isoformat(),
            "health_percentage": (self.healthy_components / self.total_components * 100) if self.total_components > 0 else 0
        }


# =============================================================================
# Health Check Framework
# =============================================================================

class HealthChecker:
    """Comprehensive health checking framework."""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.console = Console()
    
    async def check_all_components(self) -> SystemHealth:
        """Check health of all system components."""
        start_time = time.time()
        
        self.console.print("[bold blue]APG API Service Mesh - Health Check[/bold blue]")
        
        # Define health checks
        health_checks = [
            ("API Service", self._check_api_health),
            ("Database", self._check_database_health),
            ("Redis Cache", self._check_redis_health),
            ("Network Connectivity", self._check_network_health),
            ("Service Dependencies", self._check_service_dependencies),
            ("Resource Usage", self._check_resource_health),
            ("Security", self._check_security_health)
        ]
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Running health checks...", total=len(health_checks))
            
            for component_name, check_func in health_checks:
                progress.update(task, description=f"Checking {component_name}...")
                
                try:
                    status = await self._run_with_retry(check_func)
                    status.component = component_name
                    results.append(status)
                except Exception as e:
                    error_status = HealthStatus(
                        component=component_name,
                        status="unknown",
                        response_time=0,
                        details={},
                        error=str(e)
                    )
                    results.append(error_status)
                
                progress.advance(task)
        
        # Calculate overall health
        overall_time = time.time() - start_time
        healthy_count = sum(1 for r in results if r.status == "healthy")
        total_count = len(results)
        
        # Determine overall status
        if healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count >= total_count * 0.8:  # 80% threshold
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        system_health = SystemHealth(
            status=overall_status,
            components=results,
            overall_response_time=overall_time,
            healthy_components=healthy_count,
            total_components=total_count,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Display results
        self._display_health_results(system_health)
        
        return system_health
    
    async def _run_with_retry(self, check_func) -> HealthStatus:
        """Run health check with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retries):
            try:
                return await check_func()
            except Exception as e:
                last_error = e
                if attempt < self.config.retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue
        
        # All retries failed
        return HealthStatus(
            component="",
            status="unhealthy",
            response_time=0,
            details={},
            error=str(last_error)
        )
    
    async def _check_api_health(self) -> HealthStatus:
        """Check API service health."""
        start_time = time.time()
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            # Check main health endpoint
            health_url = f"{self.config.api_url}/api/health"
            async with session.get(health_url) as response:
                response_data = await response.json()
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200 and response_data.get("success"):
                    status = "healthy"
                    details = {
                        "status_code": response.status,
                        "api_data": response_data.get("data", {}),
                        "endpoints_tested": ["/api/health"]
                    }
                    
                    # Test additional endpoints if detailed check
                    if self.config.detailed:
                        additional_endpoints = [
                            "/api/info",
                            "/api/services",
                            "/api/metrics/query"
                        ]
                        
                        endpoint_results = {}
                        for endpoint in additional_endpoints:
                            try:
                                endpoint_start = time.time()
                                async with session.get(f"{self.config.api_url}{endpoint}") as ep_response:
                                    endpoint_time = (time.time() - endpoint_start) * 1000
                                    endpoint_results[endpoint] = {
                                        "status_code": ep_response.status,
                                        "response_time": endpoint_time,
                                        "healthy": 200 <= ep_response.status < 400
                                    }
                            except Exception as e:
                                endpoint_results[endpoint] = {
                                    "error": str(e),
                                    "healthy": False
                                }
                        
                        details["additional_endpoints"] = endpoint_results
                        details["endpoints_tested"].extend(additional_endpoints)
                        
                        # Check if any additional endpoints failed
                        failed_endpoints = [ep for ep, result in endpoint_results.items() if not result.get("healthy")]
                        if failed_endpoints:
                            status = "degraded"
                            details["failed_endpoints"] = failed_endpoints
                    
                else:
                    status = "unhealthy"
                    details = {
                        "status_code": response.status,
                        "response_data": response_data
                    }
                    
                return HealthStatus(
                    component="",
                    status=status,
                    response_time=response_time,
                    details=details
                )
    
    async def _check_database_health(self) -> HealthStatus:
        """Check PostgreSQL database health."""
        start_time = time.time()
        
        try:
            # Connect to database
            conn = await asyncpg.connect(self.config.database_url)
            
            # Test basic connectivity
            result = await conn.fetchval("SELECT 1")
            
            details = {
                "connectivity": "success",
                "test_query_result": result
            }
            
            if self.config.detailed:
                # Get database statistics
                db_stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections
                """)
                
                # Check if service mesh tables exist
                table_check = await conn.fetchval("""
                    SELECT count(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'sm_%'
                """)
                
                details.update({
                    "database_size": db_stats["db_size"],
                    "active_connections": db_stats["active_connections"],
                    "max_connections": int(db_stats["max_connections"]),
                    "service_mesh_tables": table_check
                })
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="",
                status="healthy",
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="",
                status="unhealthy",
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    async def _check_redis_health(self) -> HealthStatus:
        """Check Redis cache health."""
        start_time = time.time()
        
        try:
            # Connect to Redis
            redis_client = redis.from_url(self.config.redis_url)
            
            # Test basic connectivity
            pong = await redis_client.ping()
            
            details = {
                "connectivity": "success",
                "ping_response": pong
            }
            
            if self.config.detailed:
                # Get Redis info
                info = await redis_client.info()
                
                # Test set/get operations
                test_key = "apg:health_check:test"
                test_value = f"test_{int(time.time())}"
                
                await redis_client.set(test_key, test_value, ex=60)
                retrieved_value = await redis_client.get(test_key)
                await redis_client.delete(test_key)
                
                details.update({
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "test_operation": "success" if retrieved_value.decode() == test_value else "failed"
                })
            
            await redis_client.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="",
                status="healthy",
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="",
                status="unhealthy",
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    async def _check_network_health(self) -> HealthStatus:
        """Check network connectivity and DNS resolution."""
        start_time = time.time()
        
        details = {}
        all_healthy = True
        
        try:
            # Parse API URL for connectivity test
            from urllib.parse import urlparse
            parsed_url = urlparse(self.config.api_url)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
            
            # Test socket connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            details["socket_connectivity"] = "success" if result == 0 else f"failed (error {result})"
            if result != 0:
                all_healthy = False
            
            # Test DNS resolution
            import socket
            try:
                ip_address = socket.gethostbyname(host)
                details["dns_resolution"] = {
                    "hostname": host,
                    "ip_address": ip_address,
                    "status": "success"
                }
            except socket.gaierror as e:
                details["dns_resolution"] = {
                    "hostname": host,
                    "error": str(e),
                    "status": "failed"
                }
                all_healthy = False
            
            if self.config.detailed:
                # Test external connectivity
                external_hosts = [
                    ("google.com", 80),
                    ("github.com", 443),
                    ("8.8.8.8", 53)  # Google DNS
                ]
                
                external_results = {}
                for ext_host, ext_port in external_hosts:
                    try:
                        ext_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        ext_sock.settimeout(5)  # Shorter timeout for external
                        ext_result = ext_sock.connect_ex((ext_host, ext_port))
                        ext_sock.close()
                        external_results[f"{ext_host}:{ext_port}"] = "success" if ext_result == 0 else "failed"
                    except Exception as e:
                        external_results[f"{ext_host}:{ext_port}"] = f"error: {e}"
                
                details["external_connectivity"] = external_results
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="",
                status="healthy" if all_healthy else "degraded",
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="",
                status="unhealthy",
                response_time=response_time,
                details=details,
                error=str(e)
            )
    
    async def _check_service_dependencies(self) -> HealthStatus:
        """Check external service dependencies."""
        start_time = time.time()
        
        details = {}
        all_healthy = True
        
        # List of service dependencies to check
        dependencies = []
        
        # Parse configuration for APG services
        if hasattr(self.config, 'apg_registry_url'):
            dependencies.append(("APG Capability Registry", self.config.apg_registry_url))
        
        if not dependencies:
            # If no external dependencies configured, mark as healthy
            return HealthStatus(
                component="",
                status="healthy",
                response_time=(time.time() - start_time) * 1000,
                details={"message": "No external dependencies configured"}
            )
        
        timeout = aiohttp.ClientTimeout(total=10)  # Shorter timeout for dependencies
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            for dep_name, dep_url in dependencies:
                try:
                    health_endpoint = f"{dep_url}/health" if not dep_url.endswith('/health') else dep_url
                    async with session.get(health_endpoint) as response:
                        if response.status == 200:
                            details[dep_name] = "healthy"
                        else:
                            details[dep_name] = f"unhealthy (status: {response.status})"
                            all_healthy = False
                except Exception as e:
                    details[dep_name] = f"error: {str(e)}"
                    all_healthy = False
        
        response_time = (time.time() - start_time) * 1000
        
        return HealthStatus(
            component="",
            status="healthy" if all_healthy else "degraded",
            response_time=response_time,
            details=details
        )
    
    async def _check_resource_health(self) -> HealthStatus:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": round(disk.used / disk.total * 100, 2),
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            
            # Determine health based on thresholds
            status = "healthy"
            warnings = []
            
            if cpu_percent > 90:
                status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            elif cpu_percent > 95:
                status = "unhealthy"
            
            if memory.percent > 85:
                status = "degraded"
                warnings.append(f"High memory usage: {memory.percent}%")
            elif memory.percent > 95:
                status = "unhealthy"
            
            if disk.used / disk.total > 0.9:
                status = "degraded"
                warnings.append(f"High disk usage: {round(disk.used / disk.total * 100, 2)}%")
            elif disk.used / disk.total > 0.95:
                status = "unhealthy"
            
            if warnings:
                details["warnings"] = warnings
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="",
                status=status,
                response_time=response_time,
                details=details
            )
            
        except ImportError:
            return HealthStatus(
                component="",
                status="unknown",
                response_time=(time.time() - start_time) * 1000,
                details={"message": "psutil not available for resource monitoring"}
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="",
                status="unknown",
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    async def _check_security_health(self) -> HealthStatus:
        """Check security-related configurations."""
        start_time = time.time()
        
        details = {}
        warnings = []
        
        try:
            # Check SSL/TLS configuration
            if self.config.api_url.startswith("https://"):
                from urllib.parse import urlparse
                parsed_url = urlparse(self.config.api_url)
                host = parsed_url.hostname
                port = parsed_url.port or 443
                
                try:
                    # Check SSL certificate
                    context = ssl.create_default_context()
                    with socket.create_connection((host, port), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=host) as ssock:
                            cert = ssock.getpeercert()
                            details["ssl_certificate"] = {
                                "subject": dict(x[0] for x in cert.get("subject", [])),
                                "issuer": dict(x[0] for x in cert.get("issuer", [])),
                                "expiry": cert.get("notAfter"),
                                "valid": True
                            }
                except Exception as e:
                    details["ssl_certificate"] = {
                        "error": str(e),
                        "valid": False
                    }
                    warnings.append("SSL certificate validation failed")
            else:
                warnings.append("API endpoint not using HTTPS")
                details["ssl_certificate"] = {"message": "HTTP endpoint, no SSL"}
            
            # Check for secure headers (if detailed)
            if self.config.detailed:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.config.api_url}/api/health") as response:
                        headers = dict(response.headers)
                        
                        security_headers = {
                            "Strict-Transport-Security": headers.get("Strict-Transport-Security"),
                            "X-Content-Type-Options": headers.get("X-Content-Type-Options"),
                            "X-Frame-Options": headers.get("X-Frame-Options"),
                            "X-XSS-Protection": headers.get("X-XSS-Protection"),
                            "Content-Security-Policy": headers.get("Content-Security-Policy")
                        }
                        
                        details["security_headers"] = security_headers
                        
                        missing_headers = [name for name, value in security_headers.items() if value is None]
                        if missing_headers:
                            warnings.append(f"Missing security headers: {', '.join(missing_headers)}")
            
            # Determine overall security status
            if len(warnings) == 0:
                status = "healthy"
            elif len(warnings) <= 2:
                status = "degraded"
            else:
                status = "unhealthy"
            
            if warnings:
                details["warnings"] = warnings
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                component="",
                status=status,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                component="",
                status="unknown",
                response_time=response_time,
                details={},
                error=str(e)
            )
    
    def _display_health_results(self, system_health: SystemHealth):
        """Display health check results in formatted output."""
        # Overall status panel
        status_color = {
            "healthy": "green",
            "degraded": "yellow", 
            "unhealthy": "red",
            "unknown": "blue"
        }.get(system_health.status, "white")
        
        status_text = f"[bold {status_color}]{system_health.status.upper()}[/bold {status_color}]"
        
        panel_content = f"""
System Status: {status_text}
Health Score: {system_health.healthy_components}/{system_health.total_components} ({system_health.healthy_components/system_health.total_components*100:.1f}%)
Overall Response Time: {system_health.overall_response_time:.2f}s
Check Time: {system_health.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        panel = Panel(panel_content.strip(), title="System Health Summary", expand=False)
        self.console.print(panel)
        self.console.print()
        
        # Component details table
        table = Table(title="Component Health Details")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Response Time", justify="right", style="magenta")
        table.add_column("Details", style="dim")
        
        for component in system_health.components:
            # Status with color
            status_style = {
                "healthy": "green",
                "degraded": "yellow",
                "unhealthy": "red",
                "unknown": "blue"
            }.get(component.status, "white")
            
            status_text = f"[{status_style}]{component.status}[/{status_style}]"
            
            # Format details
            if component.error:
                details_text = f"Error: {component.error}"
            elif component.details:
                if self.config.detailed:
                    # Show detailed information
                    details_parts = []
                    for key, value in component.details.items():
                        if isinstance(value, dict):
                            details_parts.append(f"{key}: {json.dumps(value, default=str)}")
                        else:
                            details_parts.append(f"{key}: {value}")
                    details_text = "; ".join(details_parts[:3])  # Limit for readability
                else:
                    # Show summary information
                    key_details = []
                    for key in ["connectivity", "status_code", "test_query_result", "ping_response"]:
                        if key in component.details:
                            key_details.append(f"{key}: {component.details[key]}")
                    details_text = "; ".join(key_details) if key_details else "OK"
            else:
                details_text = "N/A"
            
            # Truncate long details
            if len(details_text) > 50:
                details_text = details_text[:47] + "..."
            
            table.add_row(
                component.component,
                status_text,
                f"{component.response_time:.1f}ms",
                details_text
            )
        
        self.console.print(table)
        self.console.print()


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """Main entry point for health checking."""
    parser = argparse.ArgumentParser(description="APG API Service Mesh Health Check")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--database-url", help="Database connection URL")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed checks")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries in seconds")
    parser.add_argument("--detailed", action="store_true", help="Run detailed health checks")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output", help="Save results to file")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code if unhealthy")
    
    args = parser.parse_args()
    
    # Create configuration
    config = HealthCheckConfig(
        api_url=args.api_url,
        redis_url=args.redis_url,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        detailed=args.detailed
    )
    
    if args.database_url:
        config.database_url = args.database_url
    
    # Run health checks
    checker = HealthChecker(config)
    
    try:
        system_health = await checker.check_all_components()
        
        # Output results
        if args.json:
            output_data = system_health.to_dict()
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(f"Results saved to: {args.output}")
            else:
                print(json.dumps(output_data, indent=2, default=str))
        else:
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(f"Health Check Results - {system_health.timestamp}\n")
                    f.write(f"Overall Status: {system_health.status}\n")
                    f.write(f"Healthy Components: {system_health.healthy_components}/{system_health.total_components}\n\n")
                    
                    for component in system_health.components:
                        f.write(f"{component.component}: {component.status}\n")
                        if component.error:
                            f.write(f"  Error: {component.error}\n")
                        f.write(f"  Response Time: {component.response_time:.1f}ms\n\n")
                
                print(f"Results saved to: {args.output}")
        
        # Exit code handling
        if args.exit_code:
            if system_health.status in ["unhealthy", "unknown"]:
                sys.exit(1)
            elif system_health.status == "degraded":
                sys.exit(2)
            else:
                sys.exit(0)
        
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Health check interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console = Console()
        console.print(f"[red]Health check failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())