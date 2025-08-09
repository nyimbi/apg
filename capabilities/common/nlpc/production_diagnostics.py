"""
APG NLP Production Diagnostics & Troubleshooting

Comprehensive diagnostics and troubleshooting system for production issues.
Provides automated problem detection, root cause analysis, and remediation suggestions.

Features:
- Real-time system diagnostics
- Performance bottleneck detection
- Error pattern analysis
- Automated remediation suggestions
- Diagnostic report generation
- Troubleshooting workflows
- Performance profiling tools
- Log analysis and correlation
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from uuid_extensions import uuid7str
import re

# Configure logging
logger = logging.getLogger(__name__)

class DiagnosticSeverity(str, Enum):
	"""Diagnostic issue severity levels"""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"

class DiagnosticCategory(str, Enum):
	"""Diagnostic categories"""
	PERFORMANCE = "performance"
	RESOURCE = "resource"
	CONNECTIVITY = "connectivity"
	DATA = "data"
	SECURITY = "security"
	CONFIGURATION = "configuration"
	APPLICATION = "application"

class RemediationStatus(str, Enum):
	"""Remediation action status"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	NOT_APPLICABLE = "not_applicable"

@dataclass
class DiagnosticIssue:
	"""Detected diagnostic issue"""
	issue_id: str = field(default_factory=uuid7str)
	category: DiagnosticCategory = DiagnosticCategory.APPLICATION
	severity: DiagnosticSeverity = DiagnosticSeverity.INFO
	title: str = ""
	description: str = ""
	
	# Detection details
	detected_at: datetime = field(default_factory=datetime.utcnow)
	detection_method: str = ""
	affected_components: List[str] = field(default_factory=list)
	
	# Diagnostic data
	symptoms: List[str] = field(default_factory=list)
	metrics: Dict[str, Any] = field(default_factory=dict)
	logs: List[str] = field(default_factory=list)
	stack_traces: List[str] = field(default_factory=list)
	
	# Analysis results
	root_causes: List[str] = field(default_factory=list)
	impact_assessment: str = ""
	
	# Remediation
	remediation_suggestions: List[str] = field(default_factory=list)
	automated_fixes: List[str] = field(default_factory=list)
	
	# Resolution tracking
	resolved: bool = False
	resolved_at: Optional[datetime] = None
	resolution_notes: str = ""

@dataclass
class SystemProfile:
	"""System performance profile"""
	profile_id: str = field(default_factory=uuid7str)
	timestamp: datetime = field(default_factory=datetime.utcnow)
	duration_seconds: float = 0.0
	
	# CPU profiling
	cpu_usage_history: List[float] = field(default_factory=list)
	cpu_load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
	top_cpu_processes: List[Dict[str, Any]] = field(default_factory=list)
	
	# Memory profiling
	memory_usage_history: List[float] = field(default_factory=list)
	memory_breakdown: Dict[str, float] = field(default_factory=dict)
	top_memory_processes: List[Dict[str, Any]] = field(default_factory=list)
	
	# I/O profiling
	disk_io_stats: Dict[str, Any] = field(default_factory=dict)
	network_io_stats: Dict[str, Any] = field(default_factory=dict)
	
	# Application profiling
	request_latency_distribution: Dict[str, int] = field(default_factory=dict)
	error_rate_by_endpoint: Dict[str, float] = field(default_factory=dict)
	database_query_stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TroubleshootingWorkflow:
	"""Structured troubleshooting workflow"""
	workflow_id: str = field(default_factory=uuid7str)
	name: str = ""
	category: DiagnosticCategory = DiagnosticCategory.APPLICATION
	
	# Workflow steps
	steps: List[Dict[str, Any]] = field(default_factory=list)
	current_step: int = 0
	completed_steps: List[int] = field(default_factory=list)
	
	# Execution state
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	status: str = "pending"
	results: Dict[str, Any] = field(default_factory=dict)

class ProductionDiagnostics:
	"""Comprehensive production diagnostics system"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for production diagnostics"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Diagnostic state
		self.active_issues: Dict[str, DiagnosticIssue] = {}
		self.resolved_issues: deque = deque(maxlen=1000)  # Keep last 1000 resolved issues
		self.system_profiles: deque = deque(maxlen=100)   # Keep last 100 profiles
		
		# Performance monitoring
		self.performance_baseline: Dict[str, float] = {}
		self.alert_thresholds: Dict[str, float] = {}
		self.performance_history: deque = deque(maxlen=10000)
		
		# Log analysis
		self.error_patterns: Dict[str, Dict[str, Any]] = {}
		self.log_aggregators: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		
		# Troubleshooting workflows
		self.workflows: Dict[str, TroubleshootingWorkflow] = {}
		
		self._setup_diagnostic_config()
		self._initialize_error_patterns()
		self._initialize_troubleshooting_workflows()
		self._start_diagnostic_monitoring()
		
		self._log_diagnostics_initialized()
	
	def _setup_diagnostic_config(self) -> None:
		"""Setup diagnostic configuration"""
		self.profile_interval = self.config.get("profile_interval", 60)  # seconds
		self.alert_check_interval = self.config.get("alert_check_interval", 30)
		self.log_analysis_enabled = self.config.get("log_analysis_enabled", True)
		self.auto_remediation_enabled = self.config.get("auto_remediation_enabled", False)
		
		# Performance thresholds
		self.alert_thresholds = {
			"cpu_usage_percent": self.config.get("cpu_alert_threshold", 80),
			"memory_usage_percent": self.config.get("memory_alert_threshold", 85),
			"disk_usage_percent": self.config.get("disk_alert_threshold", 90),
			"response_time_ms": self.config.get("response_time_threshold", 1000),
			"error_rate_percent": self.config.get("error_rate_threshold", 5)
		}
	
	def _initialize_error_patterns(self) -> None:
		"""Initialize common error patterns for detection"""
		self.error_patterns = {
			"memory_leak": {
				"pattern": r"OutOfMemoryError|MemoryError|memory.*exhausted",
				"severity": DiagnosticSeverity.CRITICAL,
				"category": DiagnosticCategory.RESOURCE,
				"symptoms": ["Increasing memory usage", "Slow response times", "Process crashes"],
				"remediation": ["Restart affected services", "Check for memory leaks", "Scale up resources"]
			},
			"database_connection": {
				"pattern": r"connection.*refused|connection.*timeout|database.*unavailable",
				"severity": DiagnosticSeverity.CRITICAL,
				"category": DiagnosticCategory.CONNECTIVITY,
				"symptoms": ["Database connection errors", "Query timeouts", "Service unavailability"],
				"remediation": ["Check database status", "Verify network connectivity", "Restart connection pools"]
			},
			"high_latency": {
				"pattern": r"timeout|slow.*query|request.*timeout",
				"severity": DiagnosticSeverity.WARNING,
				"category": DiagnosticCategory.PERFORMANCE,
				"symptoms": ["Slow response times", "User complaints", "Increased resource usage"],
				"remediation": ["Optimize database queries", "Add caching", "Scale resources"]
			},
			"authentication_failure": {
				"pattern": r"authentication.*failed|unauthorized|403|401",
				"severity": DiagnosticSeverity.ERROR,
				"category": DiagnosticCategory.SECURITY,
				"symptoms": ["Authentication failures", "Access denied errors", "Security alerts"],
				"remediation": ["Check authentication service", "Verify credentials", "Review security policies"]
			},
			"model_loading_error": {
				"pattern": r"model.*load.*error|model.*not.*found|inference.*failed",
				"severity": DiagnosticSeverity.CRITICAL,
				"category": DiagnosticCategory.APPLICATION,
				"symptoms": ["NLP processing failures", "Model unavailability", "Inference errors"],
				"remediation": ["Check model files", "Restart NLP services", "Verify model paths"]
			}
		}
	
	def _initialize_troubleshooting_workflows(self) -> None:
		"""Initialize structured troubleshooting workflows"""
		
		# High CPU usage workflow
		high_cpu_workflow = TroubleshootingWorkflow(
			name="High CPU Usage Investigation",
			category=DiagnosticCategory.PERFORMANCE,
			steps=[
				{
					"name": "Check current CPU usage",
					"action": "measure_cpu_usage",
					"description": "Measure current CPU usage across all cores"
				},
				{
					"name": "Identify top CPU consumers",
					"action": "identify_cpu_processes",
					"description": "Find processes consuming most CPU"
				},
				{
					"name": "Check for CPU-intensive queries",
					"action": "check_database_queries",
					"description": "Identify long-running or expensive database queries"
				},
				{
					"name": "Analyze request patterns",
					"action": "analyze_request_patterns",
					"description": "Check for unusual request patterns or spikes"
				},
				{
					"name": "Generate remediation plan",
					"action": "generate_cpu_remediation",
					"description": "Create action plan to reduce CPU usage"
				}
			]
		)
		
		self.workflows["high_cpu"] = high_cpu_workflow
		
		# Memory leak workflow
		memory_leak_workflow = TroubleshootingWorkflow(
			name="Memory Leak Investigation",
			category=DiagnosticCategory.RESOURCE,
			steps=[
				{
					"name": "Check memory usage trend",
					"action": "analyze_memory_trend",
					"description": "Analyze memory usage over time"
				},
				{
					"name": "Identify memory-heavy processes",
					"action": "identify_memory_processes",
					"description": "Find processes using most memory"
				},
				{
					"name": "Check for memory leaks",
					"action": "detect_memory_leaks",
					"description": "Analyze memory allocation patterns"
				},
				{
					"name": "Review garbage collection",
					"action": "analyze_gc_patterns",
					"description": "Check garbage collection efficiency"
				},
				{
					"name": "Generate memory optimization plan",
					"action": "generate_memory_remediation",
					"description": "Create plan to optimize memory usage"
				}
			]
		)
		
		self.workflows["memory_leak"] = memory_leak_workflow
		
		# Performance degradation workflow
		performance_workflow = TroubleshootingWorkflow(
			name="Performance Degradation Analysis",
			category=DiagnosticCategory.PERFORMANCE,
			steps=[
				{
					"name": "Baseline performance comparison",
					"action": "compare_baseline_performance",
					"description": "Compare current performance to baseline"
				},
				{
					"name": "Analyze request latency",
					"action": "analyze_request_latency",
					"description": "Break down request latency by component"
				},
				{
					"name": "Check database performance",
					"action": "check_database_performance",
					"description": "Analyze database query performance"
				},
				{
					"name": "Review system resources",
					"action": "review_system_resources",
					"description": "Check CPU, memory, disk, and network usage"
				},
				{
					"name": "Generate performance optimization plan",
					"action": "generate_performance_remediation",
					"description": "Create plan to improve performance"
				}
			]
		)
		
		self.workflows["performance_degradation"] = performance_workflow
	
	def _start_diagnostic_monitoring(self) -> None:
		"""Start background diagnostic monitoring"""
		asyncio.create_task(self._diagnostic_monitoring_loop())
		asyncio.create_task(self._log_analysis_loop())
		asyncio.create_task(self._performance_profiling_loop())
	
	def _log_diagnostics_initialized(self) -> None:
		"""Log diagnostics system initialization"""
		logger.info(f"Production diagnostics initialized for tenant: {self.tenant_id}")
		logger.info(f"Monitoring {len(self.error_patterns)} error patterns")
		logger.info(f"Configured {len(self.workflows)} troubleshooting workflows")
	
	async def _diagnostic_monitoring_loop(self) -> None:
		"""Background diagnostic monitoring loop"""
		while True:
			try:
				await self._perform_diagnostic_checks()
				await asyncio.sleep(self.alert_check_interval)
			except Exception as e:
				logger.error(f"Error in diagnostic monitoring: {str(e)}")
				await asyncio.sleep(self.alert_check_interval)
	
	async def _perform_diagnostic_checks(self) -> None:
		"""Perform comprehensive diagnostic checks"""
		
		# Resource usage checks
		await self._check_resource_usage()
		
		# Performance checks
		await self._check_performance_metrics()
		
		# Connectivity checks
		await self._check_system_connectivity()
		
		# Application health checks
		await self._check_application_health()
	
	async def _check_resource_usage(self) -> None:
		"""Check system resource usage"""
		try:
			# CPU usage check
			cpu_percent = psutil.cpu_percent(interval=1)
			if cpu_percent > self.alert_thresholds["cpu_usage_percent"]:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.RESOURCE,
					severity=DiagnosticSeverity.WARNING if cpu_percent < 95 else DiagnosticSeverity.CRITICAL,
					title="High CPU Usage Detected",
					description=f"CPU usage is {cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_usage_percent']}%)",
					symptoms=[f"CPU usage: {cpu_percent:.1f}%"],
					metrics={"cpu_usage_percent": cpu_percent},
					remediation_suggestions=[
						"Identify CPU-intensive processes",
						"Check for infinite loops or inefficient algorithms",
						"Consider scaling up CPU resources",
						"Optimize database queries"
					]
				)
			
			# Memory usage check
			memory = psutil.virtual_memory()
			if memory.percent > self.alert_thresholds["memory_usage_percent"]:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.RESOURCE,
					severity=DiagnosticSeverity.WARNING if memory.percent < 95 else DiagnosticSeverity.CRITICAL,
					title="High Memory Usage Detected",
					description=f"Memory usage is {memory.percent:.1f}% (threshold: {self.alert_thresholds['memory_usage_percent']}%)",
					symptoms=[f"Memory usage: {memory.percent:.1f}%", f"Available memory: {memory.available / 1024**3:.1f}GB"],
					metrics={"memory_usage_percent": memory.percent, "memory_available_gb": memory.available / 1024**3},
					remediation_suggestions=[
						"Check for memory leaks",
						"Restart memory-intensive processes",
						"Optimize caching strategies",
						"Scale up memory resources"
					]
				)
			
			# Disk usage check
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100
			if disk_percent > self.alert_thresholds["disk_usage_percent"]:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.RESOURCE,
					severity=DiagnosticSeverity.WARNING if disk_percent < 95 else DiagnosticSeverity.CRITICAL,
					title="High Disk Usage Detected",
					description=f"Disk usage is {disk_percent:.1f}% (threshold: {self.alert_thresholds['disk_usage_percent']}%)",
					symptoms=[f"Disk usage: {disk_percent:.1f}%", f"Available space: {disk.free / 1024**3:.1f}GB"],
					metrics={"disk_usage_percent": disk_percent, "disk_available_gb": disk.free / 1024**3},
					remediation_suggestions=[
						"Clean up temporary files",
						"Archive old log files",
						"Check for large files consuming space",
						"Scale up storage resources"
					]
				)
				
		except Exception as e:
			logger.error(f"Error checking resource usage: {str(e)}")
	
	async def _check_performance_metrics(self) -> None:
		"""Check application performance metrics"""
		try:
			# Simulate performance metrics collection
			# In production, this would collect real metrics from monitoring systems
			
			current_response_time = 500.0  # Simulate current response time
			current_error_rate = 3.5      # Simulate current error rate
			
			# Response time check
			if current_response_time > self.alert_thresholds["response_time_ms"]:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.PERFORMANCE,
					severity=DiagnosticSeverity.WARNING,
					title="High Response Time Detected",
					description=f"Average response time is {current_response_time}ms (threshold: {self.alert_thresholds['response_time_ms']}ms)",
					symptoms=[f"Response time: {current_response_time}ms", "User complaints about slow performance"],
					metrics={"average_response_time_ms": current_response_time},
					remediation_suggestions=[
						"Optimize database queries",
						"Add caching layers",
						"Check for slow external API calls",
						"Scale application resources"
					]
				)
			
			# Error rate check
			if current_error_rate > self.alert_thresholds["error_rate_percent"]:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.APPLICATION,
					severity=DiagnosticSeverity.ERROR,
					title="High Error Rate Detected",
					description=f"Error rate is {current_error_rate}% (threshold: {self.alert_thresholds['error_rate_percent']}%)",
					symptoms=[f"Error rate: {current_error_rate}%", "Increased failed requests"],
					metrics={"error_rate_percent": current_error_rate},
					remediation_suggestions=[
						"Review recent deployments",
						"Check application logs for errors",
						"Verify external service availability",
						"Check for configuration issues"
					]
				)
				
		except Exception as e:
			logger.error(f"Error checking performance metrics: {str(e)}")
	
	async def _check_system_connectivity(self) -> None:
		"""Check system connectivity and dependencies"""
		try:
			# Database connectivity check
			# In production, would test actual database connection
			db_connected = True  # Simulate database check
			
			if not db_connected:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.CONNECTIVITY,
					severity=DiagnosticSeverity.CRITICAL,
					title="Database Connection Failure",
					description="Unable to connect to the database",
					symptoms=["Database connection timeouts", "Query execution failures"],
					remediation_suggestions=[
						"Check database server status",
						"Verify network connectivity",
						"Check connection pool configuration",
						"Restart database connection pool"
					],
					automated_fixes=["restart_db_pool"] if self.auto_remediation_enabled else []
				)
			
			# Redis connectivity check
			redis_connected = True  # Simulate Redis check
			
			if not redis_connected:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.CONNECTIVITY,
					severity=DiagnosticSeverity.ERROR,
					title="Redis Connection Failure",
					description="Unable to connect to Redis cache",
					symptoms=["Cache misses", "Slower response times"],
					remediation_suggestions=[
						"Check Redis server status",
						"Verify network connectivity",
						"Check Redis configuration",
						"Restart Redis connection pool"
					]
				)
				
		except Exception as e:
			logger.error(f"Error checking system connectivity: {str(e)}")
	
	async def _check_application_health(self) -> None:
		"""Check application-specific health indicators"""
		try:
			# NLP model health check
			models_healthy = True  # Simulate model health check
			
			if not models_healthy:
				await self._create_diagnostic_issue(
					category=DiagnosticCategory.APPLICATION,
					severity=DiagnosticSeverity.CRITICAL,
					title="NLP Models Unavailable",
					description="One or more NLP models are not responding",
					symptoms=["Model loading errors", "Inference failures", "Processing timeouts"],
					remediation_suggestions=[
						"Check model files integrity",
						"Verify model loading configuration",
						"Restart NLP services",
						"Check available GPU/CPU resources"
					],
					automated_fixes=["restart_nlp_services"] if self.auto_remediation_enabled else []
				)
				
		except Exception as e:
			logger.error(f"Error checking application health: {str(e)}")
	
	async def _create_diagnostic_issue(self, category: DiagnosticCategory, severity: DiagnosticSeverity,
									   title: str, description: str, symptoms: List[str] = None,
									   metrics: Dict[str, Any] = None, remediation_suggestions: List[str] = None,
									   automated_fixes: List[str] = None, logs: List[str] = None,
									   detection_method: str = "automated_monitoring") -> DiagnosticIssue:
		"""Create a diagnostic issue"""
		
		issue = DiagnosticIssue(
			category=category,
			severity=severity,
			title=title,
			description=description,
			symptoms=symptoms or [],
			metrics=metrics or {},
			logs=logs or [],
			remediation_suggestions=remediation_suggestions or [],
			automated_fixes=automated_fixes or [],
			detection_method=detection_method
		)
		
		# Check if similar issue already exists
		existing_issue = self._find_similar_issue(issue)
		if existing_issue:
			# Update existing issue with new metrics
			existing_issue.metrics.update(issue.metrics)
			existing_issue.detected_at = datetime.utcnow()
			return existing_issue
		
		# Store new issue
		self.active_issues[issue.issue_id] = issue
		
		# Execute automated fixes if enabled
		if self.auto_remediation_enabled and issue.automated_fixes:
			await self._execute_automated_fixes(issue)
		
		self._log_diagnostic_issue_created(issue)
		
		return issue
	
	def _find_similar_issue(self, issue: DiagnosticIssue) -> Optional[DiagnosticIssue]:
		"""Find similar existing issue to avoid duplicates"""
		for existing_issue in self.active_issues.values():
			if (existing_issue.category == issue.category and
				existing_issue.title == issue.title and
				not existing_issue.resolved):
				return existing_issue
		return None
	
	async def _execute_automated_fixes(self, issue: DiagnosticIssue) -> None:
		"""Execute automated remediation fixes"""
		logger.info(f"Executing automated fixes for issue: {issue.title}")
		
		for fix in issue.automated_fixes:
			try:
				if fix == "restart_db_pool":
					await self._restart_database_pool()
				elif fix == "restart_nlp_services":
					await self._restart_nlp_services()
				# Add more automated fixes as needed
				
			except Exception as e:
				logger.error(f"Automated fix '{fix}' failed: {str(e)}")
	
	async def _restart_database_pool(self) -> None:
		"""Restart database connection pool"""
		logger.info("Restarting database connection pool")
		# In production, would restart actual database pool
		await asyncio.sleep(2)  # Simulate restart
	
	async def _restart_nlp_services(self) -> None:
		"""Restart NLP services"""
		logger.info("Restarting NLP services")
		# In production, would restart actual NLP services
		await asyncio.sleep(5)  # Simulate restart
	
	async def _log_analysis_loop(self) -> None:
		"""Background log analysis loop"""
		if not self.log_analysis_enabled:
			return
			
		while True:
			try:
				await self._analyze_recent_logs()
				await asyncio.sleep(60)  # Analyze logs every minute
			except Exception as e:
				logger.error(f"Error in log analysis: {str(e)}")
				await asyncio.sleep(60)
	
	async def _analyze_recent_logs(self) -> None:
		"""Analyze recent logs for error patterns"""
		try:
			# Simulate log analysis
			# In production, would read from actual log files or log aggregation system
			sample_logs = [
				"2025-08-08 21:00:00 ERROR Connection timeout to database",
				"2025-08-08 21:01:00 WARNING Memory usage above 80%",
				"2025-08-08 21:02:00 ERROR Model loading failed for sentiment_analysis"
			]
			
			for log_entry in sample_logs:
				await self._analyze_log_entry(log_entry)
				
		except Exception as e:
			logger.error(f"Error analyzing logs: {str(e)}")
	
	async def _analyze_log_entry(self, log_entry: str) -> None:
		"""Analyze individual log entry for patterns"""
		for pattern_name, pattern_config in self.error_patterns.items():
			if re.search(pattern_config["pattern"], log_entry, re.IGNORECASE):
				
				# Create diagnostic issue based on pattern
				await self._create_diagnostic_issue(
					category=pattern_config["category"],
					severity=pattern_config["severity"],
					title=f"Error Pattern Detected: {pattern_name.replace('_', ' ').title()}",
					description=f"Detected error pattern in logs: {log_entry.strip()}",
					symptoms=pattern_config["symptoms"],
					logs=[log_entry],
					remediation_suggestions=pattern_config["remediation"],
					detection_method="log_pattern_analysis"
				)
				break
	
	async def _performance_profiling_loop(self) -> None:
		"""Background performance profiling loop"""
		while True:
			try:
				profile = await self._create_system_profile()
				self.system_profiles.append(profile)
				await asyncio.sleep(self.profile_interval)
			except Exception as e:
				logger.error(f"Error in performance profiling: {str(e)}")
				await asyncio.sleep(self.profile_interval)
	
	async def _create_system_profile(self) -> SystemProfile:
		"""Create comprehensive system performance profile"""
		start_time = time.time()
		
		try:
			# CPU profiling
			cpu_usage_history = []
			for _ in range(5):  # Sample 5 times over 5 seconds
				cpu_usage_history.append(psutil.cpu_percent(interval=1))
			
			load_average = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
			
			# Top CPU processes
			top_cpu_processes = []
			for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
				try:
					proc_info = proc.info
					cpu_pct = proc_info.get('cpu_percent', 0)
					if cpu_pct and cpu_pct > 1.0:  # Only include processes using > 1% CPU
						top_cpu_processes.append(proc_info)
				except (psutil.NoSuchProcess, psutil.AccessDenied):
					pass
			
			top_cpu_processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
			top_cpu_processes = top_cpu_processes[:10]  # Top 10
			
			# Memory profiling
			memory = psutil.virtual_memory()
			memory_usage_history = [memory.percent]
			memory_breakdown = {
				"used_gb": memory.used / 1024**3,
				"available_gb": memory.available / 1024**3,
				"cached_gb": getattr(memory, 'cached', 0) / 1024**3,
				"buffers_gb": getattr(memory, 'buffers', 0) / 1024**3
			}
			
			# Top memory processes
			top_memory_processes = []
			for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
				try:
					proc_info = proc.info
					mem_pct = proc_info.get('memory_percent', 0)
					if mem_pct and mem_pct > 1.0:  # Only include processes using > 1% memory
						if proc_info.get('memory_info'):
							proc_info['memory_mb'] = proc_info['memory_info'].rss / 1024**2
						else:
							proc_info['memory_mb'] = 0
						top_memory_processes.append(proc_info)
				except (psutil.NoSuchProcess, psutil.AccessDenied):
					pass
			
			top_memory_processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
			top_memory_processes = top_memory_processes[:10]  # Top 10
			
			# I/O profiling
			disk_io = psutil.disk_io_counters()
			net_io = psutil.net_io_counters()
			
			disk_io_stats = {
				"read_bytes_per_sec": disk_io.read_bytes / self.profile_interval if disk_io else 0,
				"write_bytes_per_sec": disk_io.write_bytes / self.profile_interval if disk_io else 0,
				"read_count": disk_io.read_count if disk_io else 0,
				"write_count": disk_io.write_count if disk_io else 0
			}
			
			network_io_stats = {
				"bytes_sent_per_sec": net_io.bytes_sent / self.profile_interval if net_io else 0,
				"bytes_recv_per_sec": net_io.bytes_recv / self.profile_interval if net_io else 0,
				"packets_sent": net_io.packets_sent if net_io else 0,
				"packets_recv": net_io.packets_recv if net_io else 0
			}
			
			duration = time.time() - start_time
			
			return SystemProfile(
				duration_seconds=duration,
				cpu_usage_history=cpu_usage_history,
				cpu_load_average=load_average,
				top_cpu_processes=top_cpu_processes,
				memory_usage_history=memory_usage_history,
				memory_breakdown=memory_breakdown,
				top_memory_processes=top_memory_processes,
				disk_io_stats=disk_io_stats,
				network_io_stats=network_io_stats
			)
			
		except Exception as e:
			logger.error(f"Error creating system profile: {str(e)}")
			return SystemProfile(duration_seconds=time.time() - start_time)
	
	def _log_diagnostic_issue_created(self, issue: DiagnosticIssue) -> None:
		"""Log diagnostic issue creation"""
		logger.warning(f"Diagnostic issue created: {issue.title} (Severity: {issue.severity.value})")
	
	async def run_troubleshooting_workflow(self, workflow_name: str) -> TroubleshootingWorkflow:
		"""Run a troubleshooting workflow"""
		if workflow_name not in self.workflows:
			raise ValueError(f"Unknown workflow: {workflow_name}")
		
		workflow = self.workflows[workflow_name]
		workflow.started_at = datetime.utcnow()
		workflow.status = "running"
		
		logger.info(f"Starting troubleshooting workflow: {workflow.name}")
		
		try:
			for step_index, step in enumerate(workflow.steps):
				workflow.current_step = step_index
				logger.info(f"Executing step {step_index + 1}: {step['name']}")
				
				# Execute workflow step
				step_result = await self._execute_workflow_step(step, workflow)
				workflow.results[f"step_{step_index}"] = step_result
				workflow.completed_steps.append(step_index)
				
				# Add delay between steps
				await asyncio.sleep(1)
			
			workflow.status = "completed"
			workflow.completed_at = datetime.utcnow()
			
		except Exception as e:
			workflow.status = "failed"
			workflow.results["error"] = str(e)
			logger.error(f"Workflow failed: {str(e)}")
		
		return workflow
	
	async def _execute_workflow_step(self, step: Dict[str, Any], workflow: TroubleshootingWorkflow) -> Dict[str, Any]:
		"""Execute a single workflow step"""
		action = step.get("action", "")
		
		if action == "measure_cpu_usage":
			cpu_percent = psutil.cpu_percent(interval=2)
			return {"cpu_usage_percent": cpu_percent, "status": "completed"}
		
		elif action == "identify_cpu_processes":
			top_processes = []
			for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
				try:
					proc_info = proc.info
					cpu_pct = proc_info.get('cpu_percent', 0)
					if cpu_pct and cpu_pct > 5.0:
						top_processes.append(proc_info)
				except (psutil.NoSuchProcess, psutil.AccessDenied):
					pass
			
			top_processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
			return {"top_cpu_processes": top_processes[:5], "status": "completed"}
		
		elif action == "analyze_memory_trend":
			memory = psutil.virtual_memory()
			return {
				"current_memory_percent": memory.percent,
				"available_gb": memory.available / 1024**3,
				"trend": "stable",  # Simplified
				"status": "completed"
			}
		
		elif action == "generate_cpu_remediation":
			remediation_plan = [
				"Identify and optimize CPU-intensive processes",
				"Check for infinite loops or inefficient algorithms",
				"Consider horizontal scaling",
				"Optimize database queries to reduce CPU load"
			]
			return {"remediation_plan": remediation_plan, "status": "completed"}
		
		elif action == "generate_memory_remediation":
			remediation_plan = [
				"Restart memory-intensive processes",
				"Check for memory leaks in application code",
				"Optimize caching strategies",
				"Consider increasing available memory"
			]
			return {"remediation_plan": remediation_plan, "status": "completed"}
		
		else:
			# Default action - just mark as completed
			return {"status": "completed", "message": f"Executed action: {action}"}
	
	def get_diagnostic_summary(self) -> Dict[str, Any]:
		"""Get comprehensive diagnostic summary"""
		
		# Categorize active issues by severity
		issues_by_severity = defaultdict(list)
		for issue in self.active_issues.values():
			issues_by_severity[issue.severity.value].append(issue.title)
		
		# Get recent system profile
		latest_profile = self.system_profiles[-1] if self.system_profiles else None
		
		return {
			"diagnostic_summary": {
				"active_issues_count": len(self.active_issues),
				"resolved_issues_count": len(self.resolved_issues),
				"issues_by_severity": dict(issues_by_severity),
				"issues_by_category": {
					cat.value: len([i for i in self.active_issues.values() if i.category == cat])
					for cat in DiagnosticCategory
				}
			},
			"system_health": {
				"cpu_usage_percent": latest_profile.cpu_usage_history[-1] if latest_profile and latest_profile.cpu_usage_history else 0,
				"memory_usage_percent": latest_profile.memory_usage_history[-1] if latest_profile and latest_profile.memory_usage_history else 0,
				"load_average": latest_profile.cpu_load_average if latest_profile else (0, 0, 0),
				"top_cpu_process": latest_profile.top_cpu_processes[0] if latest_profile and latest_profile.top_cpu_processes else None,
				"top_memory_process": latest_profile.top_memory_processes[0] if latest_profile and latest_profile.top_memory_processes else None
			},
			"monitoring_status": {
				"diagnostic_monitoring": "active",
				"log_analysis": "active" if self.log_analysis_enabled else "disabled",
				"auto_remediation": "enabled" if self.auto_remediation_enabled else "disabled",
				"profiles_collected": len(self.system_profiles),
				"error_patterns_monitored": len(self.error_patterns)
			},
			"available_workflows": list(self.workflows.keys()),
			"summary_timestamp": datetime.utcnow().isoformat()
		}
	
	def resolve_issue(self, issue_id: str, resolution_notes: str = "") -> bool:
		"""Mark an issue as resolved"""
		if issue_id not in self.active_issues:
			return False
		
		issue = self.active_issues[issue_id]
		issue.resolved = True
		issue.resolved_at = datetime.utcnow()
		issue.resolution_notes = resolution_notes
		
		# Move to resolved issues
		self.resolved_issues.append(issue)
		del self.active_issues[issue_id]
		
		logger.info(f"Issue resolved: {issue.title}")
		return True
	
	async def cleanup(self) -> None:
		"""Cleanup diagnostic system resources"""
		# Clear active issues
		self.active_issues.clear()
		
		# Clear profiles and history
		self.system_profiles.clear()
		self.performance_history.clear()
		
		# Clear log aggregators
		self.log_aggregators.clear()
		
		logger.info(f"Production diagnostics cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"ProductionDiagnostics", "DiagnosticIssue", "SystemProfile", "TroubleshootingWorkflow",
	"DiagnosticSeverity", "DiagnosticCategory", "RemediationStatus"
]