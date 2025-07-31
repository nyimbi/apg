"""
GDELT Alert System for Critical Event Detection
===============================================

Advanced alerting system for GDELT data monitoring with conflict detection,
severity assessment, and real-time notification capabilities.

Key Features:
- **Conflict Detection**: Automated detection of conflict-related events
- **Severity Assessment**: Multi-level severity classification
- **Real-time Alerts**: Immediate notifications for critical events
- **Customizable Thresholds**: Configurable alert triggers and conditions
- **Multi-channel Notifications**: Email, Slack, webhook support
- **Alert Aggregation**: Intelligent grouping of related alerts

Alert Categories:
- **Critical Conflicts**: High-severity conflict events
- **System Failures**: ETL and processing failures
- **Data Quality**: Data anomalies and quality issues
- **Performance**: System performance degradation

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import smtplib
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories."""
    CONFLICT = "conflict"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"


@dataclass
class AlertConfig:
    """Configuration for GDELT alert system."""
    
    # Alert thresholds
    conflict_fatality_threshold: int = 10
    conflict_severity_threshold: float = -5.0
    system_failure_threshold: int = 3
    processing_delay_threshold: int = 3600  # seconds
    
    # Notification settings
    enable_email: bool = False
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    
    enable_slack: bool = False
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#alerts"
    
    enable_webhook: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    
    # Alert management
    alert_cooldown_minutes: int = 30
    max_alerts_per_hour: int = 10
    enable_alert_aggregation: bool = True


@dataclass
class Alert:
    """Represents an alert."""
    alert_id: str
    level: AlertLevel
    category: AlertCategory
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False


class GDELTAlertSystem:
    """
    Advanced alert system for GDELT monitoring and critical event detection.
    
    Provides real-time alerting for conflicts, system issues, and data quality
    problems with multi-channel notification support.
    """
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_counts: Dict[str, int] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Setup default handlers
        if config.enable_email:
            self.notification_handlers.append(self._send_email_alert)
        if config.enable_slack:
            self.notification_handlers.append(self._send_slack_alert)
        if config.enable_webhook:
            self.notification_handlers.append(self._send_webhook_alert)
    
    async def check_daily_events(self, date: datetime, processed_counts: Dict[str, int]):
        """Check daily events for alert conditions."""
        try:
            # Check processing success
            total_processed = sum(processed_counts.values())
            if total_processed == 0:
                await self.create_alert(
                    level=AlertLevel.ERROR,
                    category=AlertCategory.SYSTEM,
                    title="Daily Processing Failed",
                    message=f"No events processed for {date.date()}",
                    data={"date": date.isoformat(), "counts": processed_counts}
                )
            
            # Check for low processing counts (potential data quality issue)
            expected_minimum = 1000  # Adjust based on historical data
            if total_processed < expected_minimum:
                await self.create_alert(
                    level=AlertLevel.WARNING,
                    category=AlertCategory.DATA_QUALITY,
                    title="Low Processing Count",
                    message=f"Only {total_processed} events processed for {date.date()}, expected minimum {expected_minimum}",
                    data={"date": date.isoformat(), "counts": processed_counts, "total": total_processed}
                )
                
        except Exception as e:
            logger.error(f"Failed to check daily events: {e}")
    
    async def check_conflict_events(self, events: List[Dict[str, Any]]):
        """Check events for conflict-related alerts."""
        try:
            high_severity_events = []
            high_fatality_events = []
            
            for event in events:
                # Check fatality threshold
                fatalities = event.get('fatalities_count', 0) or 0
                if fatalities >= self.config.conflict_fatality_threshold:
                    high_fatality_events.append(event)
                
                # Check severity threshold
                goldstein = event.get('goldstein_scale')
                if goldstein and goldstein <= self.config.conflict_severity_threshold:
                    high_severity_events.append(event)
            
            # Create alerts for high-fatality events
            if high_fatality_events:
                await self.create_alert(
                    level=AlertLevel.CRITICAL,
                    category=AlertCategory.CONFLICT,
                    title=f"High-Fatality Events Detected",
                    message=f"Detected {len(high_fatality_events)} events with fatalities >= {self.config.conflict_fatality_threshold}",
                    data={
                        "event_count": len(high_fatality_events),
                        "threshold": self.config.conflict_fatality_threshold,
                        "events": [{"id": e.get("external_id"), "fatalities": e.get("fatalities_count"), "location": e.get("location_name")} for e in high_fatality_events[:5]]
                    }
                )
            
            # Create alerts for high-severity events
            if high_severity_events:
                await self.create_alert(
                    level=AlertLevel.WARNING,
                    category=AlertCategory.CONFLICT,
                    title=f"High-Severity Events Detected",
                    message=f"Detected {len(high_severity_events)} events with severity <= {self.config.conflict_severity_threshold}",
                    data={
                        "event_count": len(high_severity_events),
                        "threshold": self.config.conflict_severity_threshold,
                        "events": [{"id": e.get("external_id"), "severity": e.get("goldstein_scale"), "location": e.get("location_name")} for e in high_severity_events[:5]]
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to check conflict events: {e}")
    
    async def check_system_health(self, health_data: Dict[str, Any]):
        """Check system health for alert conditions."""
        try:
            overall_status = health_data.get('overall_status', 'unknown')
            
            if overall_status == 'error':
                await self.create_alert(
                    level=AlertLevel.CRITICAL,
                    category=AlertCategory.SYSTEM,
                    title="System Health Critical",
                    message="Overall system status is in error state",
                    data=health_data
                )
            elif overall_status == 'degraded':
                await self.create_alert(
                    level=AlertLevel.WARNING,
                    category=AlertCategory.SYSTEM,
                    title="System Health Degraded",
                    message="Overall system status is degraded",
                    data=health_data
                )
            
            # Check individual components
            components = health_data.get('components', {})
            for component, status in components.items():
                if isinstance(status, dict) and status.get('status') == 'error':
                    await self.create_alert(
                        level=AlertLevel.ERROR,
                        category=AlertCategory.SYSTEM,
                        title=f"Component Failure: {component}",
                        message=f"Component {component} is in error state: {status.get('error', 'Unknown error')}",
                        data={"component": component, "status": status}
                    )
                    
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    async def create_alert(
        self,
        level: AlertLevel,
        category: AlertCategory,
        title: str,
        message: str,
        data: Dict[str, Any] = None
    ) -> str:
        """Create a new alert."""
        
        # Check rate limiting
        alert_key = f"{category.value}_{title}"
        now = datetime.now(timezone.utc)
        
        # Check cooldown period
        if alert_key in self.last_alert_times:
            time_since_last = (now - self.last_alert_times[alert_key]).total_seconds()
            if time_since_last < (self.config.alert_cooldown_minutes * 60):
                logger.debug(f"Alert {alert_key} in cooldown period, skipping")
                return ""
        
        # Check hourly rate limit
        hour_key = f"{alert_key}_{now.strftime('%Y%m%d%H')}"
        if hour_key in self.alert_counts:
            if self.alert_counts[hour_key] >= self.config.max_alerts_per_hour:
                logger.warning(f"Alert rate limit exceeded for {alert_key}")
                return ""
        
        # Create alert
        alert_id = f"{category.value}_{int(now.timestamp())}"
        alert = Alert(
            alert_id=alert_id,
            level=level,
            category=category,
            title=title,
            message=message,
            data=data or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = now
        self.alert_counts[hour_key] = self.alert_counts.get(hour_key, 0) + 1
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.info(f"Created alert: {level.value} - {title}")
        return alert_id
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email notification."""
        try:
            if not self.config.email_recipients:
                return
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username or "gdelt-alerts@system.local"
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = f"[GDELT Alert - {alert.level.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
GDELT Alert Notification

Level: {alert.level.value.upper()}
Category: {alert.category.value}
Time: {alert.timestamp.isoformat()}

Title: {alert.title}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2, default=str)}

Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.email_smtp_host, self.config.email_smtp_port) as server:
                if self.config.email_username and self.config.email_password:
                    server.starttls()
                    server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack notification."""
        try:
            if not self.config.slack_webhook_url:
                return
            
            # Determine color based on alert level
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9500", 
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#8B0000"
            }
            
            # Create Slack message
            slack_message = {
                "channel": self.config.slack_channel,
                "username": "GDELT Alert System",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.level, "#808080"),
                    "title": f"{alert.level.value.upper()}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": True}
                    ],
                    "footer": "GDELT Alert System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook_url, json=slack_message) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.alert_id}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification."""
        try:
            if not self.config.webhook_urls:
                return
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "category": alert.category.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data
            }
            
            # Send to all webhook URLs
            async with aiohttp.ClientSession() as session:
                for webhook_url in self.config.webhook_urls:
                    try:
                        async with session.post(webhook_url, json=payload) as response:
                            if response.status == 200:
                                logger.info(f"Webhook alert sent to {webhook_url} for {alert.alert_id}")
                            else:
                                logger.error(f"Failed to send webhook alert to {webhook_url}: {response.status}")
                    except Exception as e:
                        logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alerts: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        
        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(self.get_active_alerts()),
            "alerts_last_24h": len(recent_alerts),
            "alerts_by_level": {
                level.value: len([a for a in recent_alerts if a.level == level])
                for level in AlertLevel
            },
            "alerts_by_category": {
                category.value: len([a for a in recent_alerts if a.category == category])
                for category in AlertCategory
            }
        }


# Factory function
def create_alert_system(
    conflict_threshold: int = 10,
    enable_email: bool = False,
    email_recipients: List[str] = None,
    enable_slack: bool = False,
    slack_webhook: str = None,
    **kwargs
) -> GDELTAlertSystem:
    """
    Create a GDELT alert system with configuration.
    
    Args:
        conflict_threshold: Fatality threshold for conflict alerts
        enable_email: Enable email notifications
        email_recipients: List of email recipients
        enable_slack: Enable Slack notifications
        slack_webhook: Slack webhook URL
        **kwargs: Additional configuration options
        
    Returns:
        Configured GDELTAlertSystem instance
    """
    config = AlertConfig(
        conflict_fatality_threshold=conflict_threshold,
        enable_email=enable_email,
        email_recipients=email_recipients or [],
        enable_slack=enable_slack,
        slack_webhook_url=slack_webhook,
        **kwargs
    )
    return GDELTAlertSystem(config)


# Export components
__all__ = [
    'GDELTAlertSystem',
    'AlertConfig',
    'Alert',
    'AlertLevel',
    'AlertCategory',
    'create_alert_system'
]