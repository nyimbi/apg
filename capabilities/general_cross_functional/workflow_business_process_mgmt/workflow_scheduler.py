"""
APG Workflow & Business Process Management - Workflow Scheduler

Comprehensive workflow scheduling, timing controls, and automation system with
deep instrumentation for time limits, alerts, and periodic execution.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import cron_descriptor
from uuid_extensions import uuid7str

from models import (
	APGTenantContext, WBPMProcessDefinition, WBPMProcessInstance, 
	WBPMServiceResponse, ProcessStatus, TaskStatus, APGBaseModel
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Scheduling and Timing Core Classes
# =============================================================================

class ScheduleType(str, Enum):
	"""Types of workflow schedules."""
	ONE_TIME = "one_time"
	RECURRING = "recurring"
	CRON = "cron"
	EVENT_DRIVEN = "event_driven"
	CONDITIONAL = "conditional"


class TimerType(str, Enum):
	"""Types of process timers."""
	DURATION = "duration"
	CYCLE = "cycle"
	DATE = "date"
	ESCALATION = "escalation"
	SLA = "sla"


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertType(str, Enum):
	"""Types of timing alerts."""
	PROCESS_TIMEOUT = "process_timeout"
	ACTIVITY_TIMEOUT = "activity_timeout"
	SLA_BREACH = "sla_breach"
	DEADLINE_APPROACHING = "deadline_approaching"
	ESCALATION_TRIGGER = "escalation_trigger"
	SCHEDULE_FAILURE = "schedule_failure"


@dataclass
class WorkflowSchedule(APGBaseModel):
	"""Workflow execution schedule configuration."""
	schedule_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	process_definition_id: str = ""
	schedule_type: ScheduleType = ScheduleType.ONE_TIME
	
	# Timing Configuration
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	cron_expression: Optional[str] = None
	interval_minutes: Optional[int] = None
	max_executions: Optional[int] = None
	
	# Execution Context
	input_variables: Dict[str, Any] = field(default_factory=dict)
	execution_timeout_minutes: int = 60
	retry_attempts: int = 3
	retry_delay_minutes: int = 5
	
	# State Management
	is_active: bool = True
	execution_count: int = 0
	last_execution: Optional[datetime] = None
	next_execution: Optional[datetime] = None
	
	# APG Integration
	assigned_to: Optional[str] = None
	notification_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessTimer(APGBaseModel):
	"""Process and activity timing configuration."""
	timer_id: str = field(default_factory=uuid7str)
	process_instance_id: str = ""
	activity_id: Optional[str] = None
	timer_type: TimerType = TimerType.DURATION
	
	# Timer Configuration
	duration_minutes: Optional[int] = None
	target_date: Optional[datetime] = None
	cycle_definition: Optional[str] = None
	
	# Alert Configuration
	warning_threshold_percent: int = 80  # Warn at 80% of time limit
	escalation_threshold_percent: int = 100  # Escalate at 100% of time limit
	alert_recipients: List[str] = field(default_factory=list)
	
	# State Management
	is_active: bool = True
	start_time: Optional[datetime] = None
	warning_sent: bool = False
	escalation_triggered: bool = False
	
	# SLA Configuration
	sla_target_minutes: Optional[int] = None
	sla_critical_minutes: Optional[int] = None


@dataclass
class TimingAlert(APGBaseModel):
	"""Timing-related alert notification."""
	alert_id: str = field(default_factory=uuid7str)
	alert_type: AlertType = AlertType.PROCESS_TIMEOUT
	severity: AlertSeverity = AlertSeverity.MEDIUM
	
	# Context Information
	process_instance_id: Optional[str] = None
	activity_id: Optional[str] = None
	timer_id: Optional[str] = None
	schedule_id: Optional[str] = None
	
	# Alert Details
	title: str = ""
	message: str = ""
	details: Dict[str, Any] = field(default_factory=dict)
	
	# Recipients and Actions
	recipients: List[str] = field(default_factory=list)
	notification_channels: List[str] = field(default_factory=list)
	escalation_actions: List[str] = field(default_factory=list)
	
	# State Management
	is_resolved: bool = False
	acknowledged_by: Optional[str] = None
	acknowledged_at: Optional[datetime] = None
	resolution_notes: Optional[str] = None


# =============================================================================
# Workflow Scheduler Engine
# =============================================================================

class WorkflowScheduler:
	"""Comprehensive workflow scheduling and timing engine."""
	
	def __init__(self, tenant_context: APGTenantContext):
		self.tenant_context = tenant_context
		self.active_schedules: Dict[str, WorkflowSchedule] = {}
		self.active_timers: Dict[str, ProcessTimer] = {}
		self.alert_handlers: Dict[AlertType, List[Callable]] = {}
		self.is_running = False
		
		# Scheduler configuration
		self.scheduler_interval = 30  # Check every 30 seconds
		self.max_concurrent_executions = 100
		self.default_execution_timeout = 3600  # 1 hour
		
		# Alert configuration
		self.alert_retention_days = 30
		self.max_alerts_per_minute = 100
		
		# Performance tracking
		self.execution_stats = {
			'total_scheduled_executions': 0,
			'successful_executions': 0,
			'failed_executions': 0,
			'timeout_executions': 0,
			'alerts_generated': 0
		}
	
	
	async def start(self) -> None:
		"""Start the workflow scheduler."""
		if self.is_running:
			logger.warning("Scheduler already running")
			return
		
		self.is_running = True
		logger.info(f"Starting workflow scheduler for tenant {self.tenant_context.tenant_id}")
		
		# Start scheduler loop
		asyncio.create_task(self._scheduler_loop())
		asyncio.create_task(self._timer_monitor_loop())
		asyncio.create_task(self._alert_cleanup_loop())
	
	
	async def stop(self) -> None:
		"""Stop the workflow scheduler."""
		self.is_running = False
		logger.info(f"Stopping workflow scheduler for tenant {self.tenant_context.tenant_id}")
	
	
	# =============================================================================
	# Schedule Management
	# =============================================================================
	
	async def create_schedule(self, schedule: WorkflowSchedule) -> WBPMServiceResponse:
		"""Create a new workflow schedule."""
		try:
			# Validate schedule configuration
			validation_result = await self._validate_schedule(schedule)
			if not validation_result.success:
				return validation_result
			
			# Calculate next execution time
			next_execution = await self._calculate_next_execution(schedule)
			if next_execution:
				schedule.next_execution = next_execution
			
			# Store schedule
			self.active_schedules[schedule.schedule_id] = schedule
			
			logger.info(f"Created schedule {schedule.schedule_id}: {schedule.name}")
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule created successfully",
				data={"schedule_id": schedule.schedule_id, "next_execution": next_execution}
			)
			
		except Exception as e:
			logger.error(f"Error creating schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create schedule: {str(e)}"
			)
	
	
	async def update_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> WBPMServiceResponse:
		"""Update an existing schedule."""
		try:
			if schedule_id not in self.active_schedules:
				return WBPMServiceResponse(
					success=False,
					message="Schedule not found"
				)
			
			schedule = self.active_schedules[schedule_id]
			
			# Update schedule properties
			for key, value in updates.items():
				if hasattr(schedule, key):
					setattr(schedule, key, value)
			
			# Recalculate next execution if timing changed
			if any(key in updates for key in ['cron_expression', 'start_time', 'interval_minutes']):
				next_execution = await self._calculate_next_execution(schedule)
				schedule.next_execution = next_execution
			
			logger.info(f"Updated schedule {schedule_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule updated successfully",
				data={"schedule_id": schedule_id, "next_execution": schedule.next_execution}
			)
			
		except Exception as e:
			logger.error(f"Error updating schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update schedule: {str(e)}"
			)
	
	
	async def delete_schedule(self, schedule_id: str) -> WBPMServiceResponse:
		"""Delete a workflow schedule."""
		try:
			if schedule_id not in self.active_schedules:
				return WBPMServiceResponse(
					success=False,
					message="Schedule not found"
				)
			
			schedule = self.active_schedules.pop(schedule_id)
			logger.info(f"Deleted schedule {schedule_id}: {schedule.name}")
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule deleted successfully"
			)
			
		except Exception as e:
			logger.error(f"Error deleting schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to delete schedule: {str(e)}"
			)
	
	
	async def pause_schedule(self, schedule_id: str) -> WBPMServiceResponse:
		"""Pause a workflow schedule."""
		try:
			if schedule_id not in self.active_schedules:
				return WBPMServiceResponse(
					success=False,
					message="Schedule not found"
				)
			
			self.active_schedules[schedule_id].is_active = False
			logger.info(f"Paused schedule {schedule_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule paused successfully"
			)
			
		except Exception as e:
			logger.error(f"Error pausing schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to pause schedule: {str(e)}"
			)
	
	
	async def resume_schedule(self, schedule_id: str) -> WBPMServiceResponse:
		"""Resume a paused workflow schedule."""
		try:
			if schedule_id not in self.active_schedules:
				return WBPMServiceResponse(
					success=False,
					message="Schedule not found"
				)
			
			schedule = self.active_schedules[schedule_id]
			schedule.is_active = True
			
			# Recalculate next execution
			next_execution = await self._calculate_next_execution(schedule)
			schedule.next_execution = next_execution
			
			logger.info(f"Resumed schedule {schedule_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule resumed successfully",
				data={"next_execution": next_execution}
			)
			
		except Exception as e:
			logger.error(f"Error resuming schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to resume schedule: {str(e)}"
			)
	
	
	# =============================================================================
	# Timer Management
	# =============================================================================
	
	async def create_process_timer(self, timer: ProcessTimer) -> WBPMServiceResponse:
		"""Create a process or activity timer."""
		try:
			# Set timer start time
			timer.start_time = datetime.now(timezone.utc)
			
			# Store timer
			self.active_timers[timer.timer_id] = timer
			
			logger.info(f"Created timer {timer.timer_id} for process {timer.process_instance_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Timer created successfully",
				data={"timer_id": timer.timer_id}
			)
			
		except Exception as e:
			logger.error(f"Error creating timer: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create timer: {str(e)}"
			)
	
	
	async def update_timer(self, timer_id: str, updates: Dict[str, Any]) -> WBPMServiceResponse:
		"""Update a process timer."""
		try:
			if timer_id not in self.active_timers:
				return WBPMServiceResponse(
					success=False,
					message="Timer not found"
				)
			
			timer = self.active_timers[timer_id]
			
			# Update timer properties
			for key, value in updates.items():
				if hasattr(timer, key):
					setattr(timer, key, value)
			
			logger.info(f"Updated timer {timer_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Timer updated successfully"
			)
			
		except Exception as e:
			logger.error(f"Error updating timer: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to update timer: {str(e)}"
			)
	
	
	async def cancel_timer(self, timer_id: str) -> WBPMServiceResponse:
		"""Cancel a process timer."""
		try:
			if timer_id not in self.active_timers:
				return WBPMServiceResponse(
					success=False,
					message="Timer not found"
				)
			
			timer = self.active_timers.pop(timer_id)
			logger.info(f"Cancelled timer {timer_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Timer cancelled successfully"
			)
			
		except Exception as e:
			logger.error(f"Error cancelling timer: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to cancel timer: {str(e)}"
			)
	
	
	# =============================================================================
	# Alert Management
	# =============================================================================
	
	async def create_alert(self, alert: TimingAlert) -> WBPMServiceResponse:
		"""Create a timing alert."""
		try:
			# Generate alert
			await self._send_alert_notification(alert)
			
			# Update stats
			self.execution_stats['alerts_generated'] += 1
			
			logger.info(f"Created alert {alert.alert_id}: {alert.title}")
			
			return WBPMServiceResponse(
				success=True,
				message="Alert created successfully",
				data={"alert_id": alert.alert_id}
			)
			
		except Exception as e:
			logger.error(f"Error creating alert: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create alert: {str(e)}"
			)
	
	
	async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> WBPMServiceResponse:
		"""Acknowledge a timing alert."""
		try:
			# In production, would update alert in database
			logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
			
			return WBPMServiceResponse(
				success=True,
				message="Alert acknowledged successfully"
			)
			
		except Exception as e:
			logger.error(f"Error acknowledging alert: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to acknowledge alert: {str(e)}"
			)
	
	
	async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str) -> WBPMServiceResponse:
		"""Resolve a timing alert."""
		try:
			# In production, would update alert in database
			logger.info(f"Alert {alert_id} resolved by {resolved_by}: {resolution_notes}")
			
			return WBPMServiceResponse(
				success=True,
				message="Alert resolved successfully"
			)
			
		except Exception as e:
			logger.error(f"Error resolving alert: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to resolve alert: {str(e)}"
			)
	
	
	# =============================================================================
	# Scheduler Implementation
	# =============================================================================
	
	async def _scheduler_loop(self) -> None:
		"""Main scheduler loop for executing scheduled workflows."""
		while self.is_running:
			try:
				current_time = datetime.now(timezone.utc)
				
				# Check for scheduled executions
				for schedule_id, schedule in list(self.active_schedules.items()):
					if not schedule.is_active:
						continue
					
					if schedule.next_execution and schedule.next_execution <= current_time:
						await self._execute_scheduled_workflow(schedule)
				
				# Wait before next check
				await asyncio.sleep(self.scheduler_interval)
				
			except Exception as e:
				logger.error(f"Error in scheduler loop: {e}")
				await asyncio.sleep(self.scheduler_interval)
	
	
	async def _timer_monitor_loop(self) -> None:
		"""Monitor process timers for warnings and escalations."""
		while self.is_running:
			try:
				current_time = datetime.now(timezone.utc)
				
				# Check active timers
				for timer_id, timer in list(self.active_timers.items()):
					if not timer.is_active or not timer.start_time:
						continue
					
					elapsed_time = current_time - timer.start_time
					
					# Check for timer expiration and alerts
					await self._check_timer_thresholds(timer, elapsed_time)
				
				# Wait before next check
				await asyncio.sleep(60)  # Check every minute
				
			except Exception as e:
				logger.error(f"Error in timer monitor loop: {e}")
				await asyncio.sleep(60)
	
	
	async def _alert_cleanup_loop(self) -> None:
		"""Clean up old resolved alerts."""
		while self.is_running:
			try:
				# In production, would clean up old alerts from database
				logger.debug("Alert cleanup completed")
				
				# Wait 24 hours before next cleanup
				await asyncio.sleep(86400)
				
			except Exception as e:
				logger.error(f"Error in alert cleanup loop: {e}")
				await asyncio.sleep(86400)
	
	
	async def _execute_scheduled_workflow(self, schedule: WorkflowSchedule) -> None:
		"""Execute a scheduled workflow."""
		try:
			logger.info(f"Executing scheduled workflow: {schedule.name}")
			
			# Update execution tracking
			schedule.execution_count += 1
			schedule.last_execution = datetime.now(timezone.utc)
			self.execution_stats['total_scheduled_executions'] += 1
			
			# In production, would start actual process instance
			# For now, simulate execution
			await asyncio.sleep(0.1)
			
			self.execution_stats['successful_executions'] += 1
			
			# Calculate next execution
			next_execution = await self._calculate_next_execution(schedule)
			schedule.next_execution = next_execution
			
			# Check if schedule should be deactivated
			if schedule.max_executions and schedule.execution_count >= schedule.max_executions:
				schedule.is_active = False
				logger.info(f"Schedule {schedule.schedule_id} deactivated after {schedule.execution_count} executions")
			
		except Exception as e:
			logger.error(f"Error executing scheduled workflow: {e}")
			self.execution_stats['failed_executions'] += 1
			
			# Create failure alert
			alert = TimingAlert(
				alert_type=AlertType.SCHEDULE_FAILURE,
				severity=AlertSeverity.HIGH,
				schedule_id=schedule.schedule_id,
				title=f"Scheduled workflow execution failed",
				message=f"Failed to execute scheduled workflow '{schedule.name}': {str(e)}",
				recipients=schedule.notification_settings.get('failure_recipients', [])
			)
			await self.create_alert(alert)
	
	
	async def _check_timer_thresholds(self, timer: ProcessTimer, elapsed_time: timedelta) -> None:
		"""Check timer thresholds and generate alerts."""
		try:
			if not timer.duration_minutes:
				return
			
			total_duration = timedelta(minutes=timer.duration_minutes)
			elapsed_percent = (elapsed_time.total_seconds() / total_duration.total_seconds()) * 100
			
			# Check warning threshold
			if not timer.warning_sent and elapsed_percent >= timer.warning_threshold_percent:
				await self._send_timer_warning(timer, elapsed_time, elapsed_percent)
				timer.warning_sent = True
			
			# Check escalation threshold
			if not timer.escalation_triggered and elapsed_percent >= timer.escalation_threshold_percent:
				await self._send_timer_escalation(timer, elapsed_time, elapsed_percent)
				timer.escalation_triggered = True
			
			# Check SLA thresholds
			if timer.sla_target_minutes:
				sla_elapsed_percent = (elapsed_time.total_seconds() / (timer.sla_target_minutes * 60)) * 100
				if sla_elapsed_percent >= 100:
					await self._send_sla_breach_alert(timer, elapsed_time)
		
		except Exception as e:
			logger.error(f"Error checking timer thresholds: {e}")
	
	
	async def _send_timer_warning(self, timer: ProcessTimer, elapsed_time: timedelta, elapsed_percent: float) -> None:
		"""Send timer warning alert."""
		alert = TimingAlert(
			alert_type=AlertType.DEADLINE_APPROACHING,
			severity=AlertSeverity.MEDIUM,
			process_instance_id=timer.process_instance_id,
			activity_id=timer.activity_id,
			timer_id=timer.timer_id,
			title=f"Timer threshold warning",
			message=f"Timer has reached {elapsed_percent:.1f}% of allocated time ({elapsed_time})",
			recipients=timer.alert_recipients,
			details={
				"elapsed_time_minutes": elapsed_time.total_seconds() / 60,
				"total_duration_minutes": timer.duration_minutes,
				"elapsed_percent": elapsed_percent,
				"threshold_percent": timer.warning_threshold_percent
			}
		)
		await self.create_alert(alert)
	
	
	async def _send_timer_escalation(self, timer: ProcessTimer, elapsed_time: timedelta, elapsed_percent: float) -> None:
		"""Send timer escalation alert."""
		alert_type = AlertType.PROCESS_TIMEOUT if not timer.activity_id else AlertType.ACTIVITY_TIMEOUT
		
		alert = TimingAlert(
			alert_type=alert_type,
			severity=AlertSeverity.HIGH,
			process_instance_id=timer.process_instance_id,
			activity_id=timer.activity_id,
			timer_id=timer.timer_id,
			title=f"Timer escalation triggered",
			message=f"Timer has exceeded allocated time: {elapsed_time} (limit: {timer.duration_minutes} minutes)",
			recipients=timer.alert_recipients,
			escalation_actions=["notify_manager", "reassign_task", "process_intervention"],
			details={
				"elapsed_time_minutes": elapsed_time.total_seconds() / 60,
				"total_duration_minutes": timer.duration_minutes,
				"elapsed_percent": elapsed_percent,
				"escalation_threshold_percent": timer.escalation_threshold_percent
			}
		)
		await self.create_alert(alert)
	
	
	async def _send_sla_breach_alert(self, timer: ProcessTimer, elapsed_time: timedelta) -> None:
		"""Send SLA breach alert."""
		alert = TimingAlert(
			alert_type=AlertType.SLA_BREACH,
			severity=AlertSeverity.CRITICAL,
			process_instance_id=timer.process_instance_id,
			activity_id=timer.activity_id,
			timer_id=timer.timer_id,
			title=f"SLA breach detected",
			message=f"Process has breached SLA target: {elapsed_time} (target: {timer.sla_target_minutes} minutes)",
			recipients=timer.alert_recipients,
			escalation_actions=["escalate_to_management", "initiate_recovery_procedure"],
			details={
				"elapsed_time_minutes": elapsed_time.total_seconds() / 60,
				"sla_target_minutes": timer.sla_target_minutes,
				"sla_critical_minutes": timer.sla_critical_minutes,
				"breach_severity": "critical" if elapsed_time.total_seconds() / 60 > timer.sla_critical_minutes else "standard"
			}
		)
		await self.create_alert(alert)
	
	
	async def _send_alert_notification(self, alert: TimingAlert) -> None:
		"""Send alert notification through configured channels."""
		try:
			# In production, would integrate with APG notification service
			logger.info(f"Alert notification: {alert.title} - {alert.message}")
			
			# Log alert details for monitoring
			logger.info(f"Alert details: {json.dumps(alert.details, default=str, indent=2)}")
			
		except Exception as e:
			logger.error(f"Error sending alert notification: {e}")
	
	
	async def _calculate_next_execution(self, schedule: WorkflowSchedule) -> Optional[datetime]:
		"""Calculate the next execution time for a schedule."""
		try:
			current_time = datetime.now(timezone.utc)
			
			if schedule.schedule_type == ScheduleType.ONE_TIME:
				return schedule.start_time if schedule.start_time and schedule.start_time > current_time else None
			
			elif schedule.schedule_type == ScheduleType.RECURRING and schedule.interval_minutes:
				if schedule.last_execution:
					return schedule.last_execution + timedelta(minutes=schedule.interval_minutes)
				else:
					return schedule.start_time or current_time
			
			elif schedule.schedule_type == ScheduleType.CRON and schedule.cron_expression:
				# In production, would use proper cron parser library
				# For now, return simple interval
				return current_time + timedelta(hours=1)
			
			return None
			
		except Exception as e:
			logger.error(f"Error calculating next execution: {e}")
			return None
	
	
	async def _validate_schedule(self, schedule: WorkflowSchedule) -> WBPMServiceResponse:
		"""Validate schedule configuration."""
		try:
			if not schedule.name:
				return WBPMServiceResponse(
					success=False,
					message="Schedule name is required"
				)
			
			if not schedule.process_definition_id:
				return WBPMServiceResponse(
					success=False,
					message="Process definition ID is required"
				)
			
			if schedule.schedule_type == ScheduleType.CRON and not schedule.cron_expression:
				return WBPMServiceResponse(
					success=False,
					message="CRON expression is required for CRON schedule type"
				)
			
			if schedule.schedule_type == ScheduleType.RECURRING and not schedule.interval_minutes:
				return WBPMServiceResponse(
					success=False,
					message="Interval is required for recurring schedule type"
				)
			
			if schedule.start_time and schedule.end_time and schedule.start_time >= schedule.end_time:
				return WBPMServiceResponse(
					success=False,
					message="Start time must be before end time"
				)
			
			return WBPMServiceResponse(
				success=True,
				message="Schedule validation passed"
			)
			
		except Exception as e:
			logger.error(f"Error validating schedule: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Schedule validation failed: {str(e)}"
			)
	
	
	# =============================================================================
	# Reporting and Monitoring
	# =============================================================================
	
	async def get_schedule_status(self, schedule_id: Optional[str] = None) -> WBPMServiceResponse:
		"""Get status of schedules."""
		try:
			if schedule_id:
				if schedule_id not in self.active_schedules:
					return WBPMServiceResponse(
						success=False,
						message="Schedule not found"
					)
				
				schedule = self.active_schedules[schedule_id]
				return WBPMServiceResponse(
					success=True,
					data={
						"schedule": {
							"schedule_id": schedule.schedule_id,
							"name": schedule.name,
							"is_active": schedule.is_active,
							"execution_count": schedule.execution_count,
							"last_execution": schedule.last_execution,
							"next_execution": schedule.next_execution
						}
					}
				)
			else:
				schedules = []
				for schedule in self.active_schedules.values():
					schedules.append({
						"schedule_id": schedule.schedule_id,
						"name": schedule.name,
						"is_active": schedule.is_active,
						"execution_count": schedule.execution_count,
						"last_execution": schedule.last_execution,
						"next_execution": schedule.next_execution
					})
				
				return WBPMServiceResponse(
					success=True,
					data={"schedules": schedules, "total_count": len(schedules)}
				)
			
		except Exception as e:
			logger.error(f"Error getting schedule status: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get schedule status: {str(e)}"
			)
	
	
	async def get_timer_status(self, timer_id: Optional[str] = None) -> WBPMServiceResponse:
		"""Get status of timers."""
		try:
			if timer_id:
				if timer_id not in self.active_timers:
					return WBPMServiceResponse(
						success=False,
						message="Timer not found"
					)
				
				timer = self.active_timers[timer_id]
				current_time = datetime.now(timezone.utc)
				elapsed_time = current_time - timer.start_time if timer.start_time else timedelta(0)
				
				return WBPMServiceResponse(
					success=True,
					data={
						"timer": {
							"timer_id": timer.timer_id,
							"process_instance_id": timer.process_instance_id,
							"activity_id": timer.activity_id,
							"timer_type": timer.timer_type,
							"is_active": timer.is_active,
							"start_time": timer.start_time,
							"elapsed_minutes": elapsed_time.total_seconds() / 60,
							"duration_minutes": timer.duration_minutes,
							"warning_sent": timer.warning_sent,
							"escalation_triggered": timer.escalation_triggered
						}
					}
				)
			else:
				timers = []
				current_time = datetime.now(timezone.utc)
				
				for timer in self.active_timers.values():
					elapsed_time = current_time - timer.start_time if timer.start_time else timedelta(0)
					timers.append({
						"timer_id": timer.timer_id,
						"process_instance_id": timer.process_instance_id,
						"activity_id": timer.activity_id,
						"timer_type": timer.timer_type,
						"is_active": timer.is_active,
						"elapsed_minutes": elapsed_time.total_seconds() / 60,
						"duration_minutes": timer.duration_minutes,
						"warning_sent": timer.warning_sent,
						"escalation_triggered": timer.escalation_triggered
					})
				
				return WBPMServiceResponse(
					success=True,
					data={"timers": timers, "total_count": len(timers)}
				)
			
		except Exception as e:
			logger.error(f"Error getting timer status: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get timer status: {str(e)}"
			)
	
	
	async def get_execution_stats(self) -> WBPMServiceResponse:
		"""Get scheduler execution statistics."""
		try:
			stats = self.execution_stats.copy()
			stats.update({
				"active_schedules": len(self.active_schedules),
				"active_timers": len(self.active_timers),
				"scheduler_running": self.is_running,
				"scheduler_interval_seconds": self.scheduler_interval
			})
			
			return WBPMServiceResponse(
				success=True,
				data={"execution_stats": stats}
			)
			
		except Exception as e:
			logger.error(f"Error getting execution stats: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get execution stats: {str(e)}"
			)


# =============================================================================
# Scheduler Factory and Helpers
# =============================================================================

class SchedulerFactory:
	"""Factory for creating workflow schedulers."""
	
	_instances: Dict[str, WorkflowScheduler] = {}
	
	@classmethod
	async def get_scheduler(cls, tenant_context: APGTenantContext) -> WorkflowScheduler:
		"""Get or create scheduler instance for tenant."""
		tenant_id = tenant_context.tenant_id
		
		if tenant_id not in cls._instances:
			scheduler = WorkflowScheduler(tenant_context)
			await scheduler.start()
			cls._instances[tenant_id] = scheduler
		
		return cls._instances[tenant_id]
	
	@classmethod
	async def shutdown_all(cls) -> None:
		"""Shutdown all scheduler instances."""
		for scheduler in cls._instances.values():
			await scheduler.stop()
		cls._instances.clear()


def create_cron_schedule(
	name: str,
	process_definition_id: str,
	cron_expression: str,
	tenant_context: APGTenantContext,
	input_variables: Optional[Dict[str, Any]] = None
) -> WorkflowSchedule:
	"""Create a CRON-based workflow schedule."""
	return WorkflowSchedule(
		tenant_id=tenant_context.tenant_id,
		created_by=tenant_context.user_id,
		updated_by=tenant_context.user_id,
		name=name,
		process_definition_id=process_definition_id,
		schedule_type=ScheduleType.CRON,
		cron_expression=cron_expression,
		input_variables=input_variables or {}
	)


def create_recurring_schedule(
	name: str,
	process_definition_id: str,
	interval_minutes: int,
	tenant_context: APGTenantContext,
	start_time: Optional[datetime] = None,
	input_variables: Optional[Dict[str, Any]] = None
) -> WorkflowSchedule:
	"""Create a recurring workflow schedule."""
	return WorkflowSchedule(
		tenant_id=tenant_context.tenant_id,
		created_by=tenant_context.user_id,
		updated_by=tenant_context.user_id,
		name=name,
		process_definition_id=process_definition_id,
		schedule_type=ScheduleType.RECURRING,
		interval_minutes=interval_minutes,
		start_time=start_time,
		input_variables=input_variables or {}
	)


def create_process_timer(
	process_instance_id: str,
	duration_minutes: int,
	tenant_context: APGTenantContext,
	activity_id: Optional[str] = None,
	alert_recipients: Optional[List[str]] = None
) -> ProcessTimer:
	"""Create a process execution timer."""
	return ProcessTimer(
		tenant_id=tenant_context.tenant_id,
		created_by=tenant_context.user_id,
		updated_by=tenant_context.user_id,
		process_instance_id=process_instance_id,
		activity_id=activity_id,
		timer_type=TimerType.DURATION,
		duration_minutes=duration_minutes,
		alert_recipients=alert_recipients or []
	)


def create_sla_timer(
	process_instance_id: str,
	sla_target_minutes: int,
	sla_critical_minutes: int,
	tenant_context: APGTenantContext,
	activity_id: Optional[str] = None,
	alert_recipients: Optional[List[str]] = None
) -> ProcessTimer:
	"""Create an SLA monitoring timer."""
	return ProcessTimer(
		tenant_id=tenant_context.tenant_id,
		created_by=tenant_context.user_id,
		updated_by=tenant_context.user_id,
		process_instance_id=process_instance_id,
		activity_id=activity_id,
		timer_type=TimerType.SLA,
		sla_target_minutes=sla_target_minutes,
		sla_critical_minutes=sla_critical_minutes,
		duration_minutes=sla_critical_minutes,  # Use critical as duration
		alert_recipients=alert_recipients or []
	)


# =============================================================================
# Example Usage and Testing
# =============================================================================

async def example_usage():
	"""Example usage of the workflow scheduler."""
	
	# Create tenant context
	tenant_context = APGTenantContext(
		tenant_id="example_tenant",
		user_id="admin@example.com",
		user_roles=["admin"],
		permissions=["workflow_manage", "workflow_schedule"]
	)
	
	# Get scheduler instance
	scheduler = await SchedulerFactory.get_scheduler(tenant_context)
	
	# Create a recurring schedule
	recurring_schedule = create_recurring_schedule(
		name="Daily Report Generation",
		process_definition_id="report_process_v1",
		interval_minutes=1440,  # Daily
		tenant_context=tenant_context,
		input_variables={"report_type": "daily_summary"}
	)
	
	result = await scheduler.create_schedule(recurring_schedule)
	print(f"Schedule created: {result.success}")
	
	# Create a CRON schedule
	cron_schedule = create_cron_schedule(
		name="Weekly Cleanup Process",
		process_definition_id="cleanup_process_v1",
		cron_expression="0 2 * * 0",  # Every Sunday at 2 AM
		tenant_context=tenant_context
	)
	
	result = await scheduler.create_schedule(cron_schedule)
	print(f"CRON schedule created: {result.success}")
	
	# Create a process timer
	process_timer = create_process_timer(
		process_instance_id="process_123",
		duration_minutes=120,  # 2 hours
		tenant_context=tenant_context,
		alert_recipients=["manager@example.com"]
	)
	
	result = await scheduler.create_process_timer(process_timer)
	print(f"Process timer created: {result.success}")
	
	# Create an SLA timer
	sla_timer = create_sla_timer(
		process_instance_id="process_123",
		sla_target_minutes=60,  # 1 hour target
		sla_critical_minutes=120,  # 2 hours critical
		tenant_context=tenant_context,
		activity_id="approval_task",
		alert_recipients=["sla_manager@example.com"]
	)
	
	result = await scheduler.create_process_timer(sla_timer)
	print(f"SLA timer created: {result.success}")
	
	# Get execution stats
	stats_result = await scheduler.get_execution_stats()
	if stats_result.success:
		print(f"Execution stats: {stats_result.data}")


if __name__ == "__main__":
	asyncio.run(example_usage())