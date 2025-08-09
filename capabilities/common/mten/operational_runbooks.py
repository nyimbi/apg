#!/usr/bin/env python3
"""
Operational Runbooks and Troubleshooting Guides for MTen Multi-Tenant Management

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive operational procedures, troubleshooting guides, incident response,
and maintenance procedures for production MTen environments.
"""

import asyncio
import json
import time
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunbookCategory(str, Enum):
    """Runbook categories"""
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BACKUP_RECOVERY = "backup_recovery"
    TROUBLESHOOTING = "troubleshooting"
    MAINTENANCE = "maintenance"
    INCIDENT_RESPONSE = "incident_response"


@dataclass
class Runbook:
    """Operational runbook definition"""
    id: str
    title: str
    category: RunbookCategory
    severity: IncidentSeverity
    description: str
    prerequisites: List[str]
    steps: List[Dict[str, Any]]
    validation: List[str]
    rollback: List[str]
    estimated_duration: int  # minutes
    required_permissions: List[str]
    related_runbooks: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"


@dataclass
class IncidentReport:
    """Incident tracking and reporting"""
    incident_id: str
    title: str
    severity: IncidentSeverity
    status: str
    description: str
    affected_services: List[str]
    start_time: datetime
    detection_time: datetime
    response_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    assignee: Optional[str] = None
    runbooks_used: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


class RunbookEngine:
    """Operational runbook management and execution engine"""
    
    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self.incident_history: List[IncidentReport] = []
        self._initialize_runbooks()
    
    def _initialize_runbooks(self):
        """Initialize standard operational runbooks"""
        self.runbooks.update(self._create_deployment_runbooks())
        self.runbooks.update(self._create_monitoring_runbooks())
        self.runbooks.update(self._create_performance_runbooks())
        self.runbooks.update(self._create_security_runbooks())
        self.runbooks.update(self._create_backup_recovery_runbooks())
        self.runbooks.update(self._create_troubleshooting_runbooks())
        self.runbooks.update(self._create_maintenance_runbooks())
        self.runbooks.update(self._create_incident_response_runbooks())
    
    def _create_deployment_runbooks(self) -> Dict[str, Runbook]:
        """Create deployment-related runbooks"""
        runbooks = {}
        
        # Production Deployment Runbook
        runbooks["deploy-production"] = Runbook(
            id="deploy-production",
            title="Production Deployment Procedure",
            category=RunbookCategory.DEPLOYMENT,
            severity=IncidentSeverity.HIGH,
            description="Step-by-step procedure for deploying MTen to production environment",
            prerequisites=[
                "Staging deployment successful and validated",
                "All tests passing in CI/CD pipeline",
                "Change management approval obtained",
                "Backup of current production state completed",
                "Rollback plan prepared and tested"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Pre-deployment Validation",
                    "description": "Validate all prerequisites and system readiness",
                    "commands": [
                        "python ci_cd_pipeline.py --validate",
                        "kubectl get nodes --show-labels",
                        "terraform plan -var-file=production.tfvars"
                    ],
                    "expected_outcome": "All validations pass, resources available"
                },
                {
                    "step": 2,
                    "title": "Create Production Backup",
                    "description": "Create full backup of production environment",
                    "commands": [
                        "./backup/backup.sh production",
                        "kubectl create backup production-$(date +%Y%m%d-%H%M%S)"
                    ],
                    "expected_outcome": "Backup completed successfully"
                },
                {
                    "step": 3,
                    "title": "Deploy Infrastructure Changes",
                    "description": "Apply infrastructure changes using Terraform",
                    "commands": [
                        "terraform apply -var-file=production.tfvars -auto-approve",
                        "terraform output > deployment-outputs.json"
                    ],
                    "expected_outcome": "Infrastructure updated without errors"
                },
                {
                    "step": 4,
                    "title": "Deploy Application",
                    "description": "Deploy application using blue-green strategy",
                    "commands": [
                        "kubectl apply -f kubernetes/production/",
                        "kubectl rollout status deployment/mten -n mten-production --timeout=600s"
                    ],
                    "expected_outcome": "Application deployed and healthy"
                },
                {
                    "step": 5,
                    "title": "Run Smoke Tests",
                    "description": "Execute smoke tests against production deployment",
                    "commands": [
                        "python smoke_tests.py --environment=production",
                        "curl -f https://mten.example.com/health"
                    ],
                    "expected_outcome": "All smoke tests pass"
                },
                {
                    "step": 6,
                    "title": "Update DNS and Traffic Routing",
                    "description": "Route production traffic to new deployment",
                    "commands": [
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"version\":\"new\"}}}'",
                        "sleep 300 # Allow DNS propagation"
                    ],
                    "expected_outcome": "Traffic successfully routed to new version"
                },
                {
                    "step": 7,
                    "title": "Monitor and Validate",
                    "description": "Monitor system health and performance",
                    "commands": [
                        "python performance_monitor.py --check-health",
                        "kubectl get events -n mten-production --sort-by=.metadata.creationTimestamp"
                    ],
                    "expected_outcome": "System healthy with normal performance metrics"
                }
            ],
            validation=[
                "Application responds to health checks",
                "All replicas are running and ready",
                "Database connectivity confirmed",
                "Cache connectivity confirmed",
                "Monitoring alerts are not firing",
                "Performance metrics within acceptable ranges"
            ],
            rollback=[
                "kubectl rollout undo deployment/mten -n mten-production",
                "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"version\":\"previous\"}}}'",
                "terraform apply -var-file=production-previous.tfvars -auto-approve",
                "Restore from backup if necessary: ./backup/recovery.sh [backup_timestamp]"
            ],
            estimated_duration=60,
            required_permissions=["cluster-admin", "terraform-apply", "dns-update"]
        )
        
        # Rollback Runbook
        runbooks["rollback-production"] = Runbook(
            id="rollback-production",
            title="Production Rollback Procedure",
            category=RunbookCategory.DEPLOYMENT,
            severity=IncidentSeverity.CRITICAL,
            description="Emergency rollback procedure for production deployments",
            prerequisites=[
                "Incident confirmed requiring rollback",
                "Previous working version identified",
                "Backup availability confirmed"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Assess Rollback Scope",
                    "description": "Determine what needs to be rolled back",
                    "commands": [
                        "kubectl rollout history deployment/mten -n mten-production",
                        "terraform show | grep 'last_updated'"
                    ],
                    "expected_outcome": "Rollback scope and target version identified"
                },
                {
                    "step": 2,
                    "title": "Stop Traffic to Current Version",
                    "description": "Immediately stop traffic to problematic deployment",
                    "commands": [
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"version\":\"maintenance\"}}}'",
                        "kubectl scale deployment/mten --replicas=0 -n mten-production"
                    ],
                    "expected_outcome": "Traffic stopped, users see maintenance page"
                },
                {
                    "step": 3,
                    "title": "Rollback Application",
                    "description": "Rollback to previous working version",
                    "commands": [
                        "kubectl rollout undo deployment/mten -n mten-production",
                        "kubectl rollout status deployment/mten -n mten-production --timeout=300s"
                    ],
                    "expected_outcome": "Previous version deployed successfully"
                },
                {
                    "step": 4,
                    "title": "Restore Database if Needed",
                    "description": "Restore database from backup if schema changes occurred",
                    "commands": [
                        "# Only if database rollback needed:",
                        "./backup/recovery.sh [backup_timestamp]"
                    ],
                    "expected_outcome": "Database restored to working state"
                },
                {
                    "step": 5,
                    "title": "Resume Traffic",
                    "description": "Route traffic back to working version",
                    "commands": [
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"app\":\"mten\"}}}'",
                        "curl -f https://mten.example.com/health"
                    ],
                    "expected_outcome": "Service fully operational"
                }
            ],
            validation=[
                "Service responds normally",
                "Error rates returned to baseline",
                "All critical features working",
                "Database integrity verified"
            ],
            rollback=[],  # This IS the rollback procedure
            estimated_duration=15,
            required_permissions=["cluster-admin", "database-admin"]
        )
        
        return runbooks
    
    def _create_monitoring_runbooks(self) -> Dict[str, Runbook]:
        """Create monitoring-related runbooks"""
        runbooks = {}
        
        # High Response Time Investigation
        runbooks["investigate-high-response-time"] = Runbook(
            id="investigate-high-response-time",
            title="Investigate High Response Time",
            category=RunbookCategory.MONITORING,
            severity=IncidentSeverity.MEDIUM,
            description="Procedure to investigate and resolve high API response times",
            prerequisites=[
                "Response time alert triggered",
                "Monitoring dashboard access available"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Confirm the Issue",
                    "description": "Verify response time metrics and scope",
                    "commands": [
                        "curl -w '%{time_total}' https://mten.example.com/health",
                        "python performance_monitor.py --check-response-times"
                    ],
                    "expected_outcome": "Response times confirmed elevated"
                },
                {
                    "step": 2,
                    "title": "Check System Resources",
                    "description": "Examine CPU, memory, and disk usage",
                    "commands": [
                        "kubectl top nodes",
                        "kubectl top pods -n mten-production",
                        "df -h # Check disk space"
                    ],
                    "expected_outcome": "Resource usage patterns identified"
                },
                {
                    "step": 3,
                    "title": "Analyze Database Performance",
                    "description": "Check database connection pool and query performance",
                    "commands": [
                        "psql -c 'SELECT * FROM pg_stat_activity;'",
                        "psql -c 'SELECT * FROM pg_stat_database;'"
                    ],
                    "expected_outcome": "Database performance status determined"
                },
                {
                    "step": 4,
                    "title": "Check Cache Performance",
                    "description": "Verify Redis cache hit rates and connection status",
                    "commands": [
                        "redis-cli info stats | grep hit_rate",
                        "redis-cli info clients"
                    ],
                    "expected_outcome": "Cache performance metrics analyzed"
                },
                {
                    "step": 5,
                    "title": "Apply Immediate Fixes",
                    "description": "Apply quick fixes based on findings",
                    "commands": [
                        "# Scale up if resource constrained:",
                        "kubectl scale deployment/mten --replicas=8 -n mten-production",
                        "# Clear cache if hit rate low:",
                        "redis-cli flushall",
                        "# Restart if needed:",
                        "kubectl rollout restart deployment/mten -n mten-production"
                    ],
                    "expected_outcome": "Response times improved"
                }
            ],
            validation=[
                "Response times below 100ms",
                "Error rates normal",
                "Resource usage sustainable"
            ],
            rollback=[
                "kubectl scale deployment/mten --replicas=5 -n mten-production"
            ],
            estimated_duration=20,
            required_permissions=["monitoring-read", "cluster-operator"]
        )
        
        return runbooks
    
    def _create_performance_runbooks(self) -> Dict[str, Runbook]:
        """Create performance-related runbooks"""
        runbooks = {}
        
        # Memory Leak Investigation
        runbooks["investigate-memory-leak"] = Runbook(
            id="investigate-memory-leak",
            title="Investigate Memory Leak",
            category=RunbookCategory.PERFORMANCE,
            severity=IncidentSeverity.HIGH,
            description="Procedure to identify and resolve memory leaks",
            prerequisites=[
                "Memory usage consistently increasing",
                "Application performance degrading"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Collect Memory Metrics",
                    "description": "Gather detailed memory usage data",
                    "commands": [
                        "kubectl top pods -n mten-production --containers",
                        "python performance_monitor.py --memory-analysis"
                    ],
                    "expected_outcome": "Memory usage patterns documented"
                },
                {
                    "step": 2,
                    "title": "Generate Memory Dump",
                    "description": "Create memory dump for analysis",
                    "commands": [
                        "kubectl exec -it deployment/mten -n mten-production -- python -c \"import gc; gc.collect(); print('Memory freed')\"",
                        "# Generate heap dump if using Java/similar"
                    ],
                    "expected_outcome": "Memory dump created for analysis"
                },
                {
                    "step": 3,
                    "title": "Restart Affected Pods",
                    "description": "Restart pods with high memory usage",
                    "commands": [
                        "kubectl delete pod -l app=mten -n mten-production",
                        "kubectl rollout status deployment/mten -n mten-production"
                    ],
                    "expected_outcome": "Pods restarted with normal memory usage"
                },
                {
                    "step": 4,
                    "title": "Monitor Recovery",
                    "description": "Monitor memory usage after restart",
                    "commands": [
                        "watch kubectl top pods -n mten-production"
                    ],
                    "expected_outcome": "Memory usage remains stable"
                }
            ],
            validation=[
                "Memory usage stable over 1 hour",
                "No memory-related errors in logs",
                "Application performance normal"
            ],
            rollback=[
                "Restart can be reversed by scaling down and up if issues persist"
            ],
            estimated_duration=30,
            required_permissions=["cluster-operator", "exec-pods"]
        )
        
        return runbooks
    
    def _create_security_runbooks(self) -> Dict[str, Runbook]:
        """Create security-related runbooks"""
        runbooks = {}
        
        # Security Incident Response
        runbooks["security-incident-response"] = Runbook(
            id="security-incident-response",
            title="Security Incident Response",
            category=RunbookCategory.SECURITY,
            severity=IncidentSeverity.CRITICAL,
            description="Immediate response procedure for security incidents",
            prerequisites=[
                "Security incident detected or reported",
                "Incident response team notified"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Assess and Contain",
                    "description": "Quickly assess incident scope and contain the threat",
                    "commands": [
                        "# Check for suspicious activity:",
                        "kubectl logs -l app=mten -n mten-production --since=1h | grep -i 'error\\|fail\\|attack'",
                        "# Block suspicious IPs if identified:",
                        "kubectl apply -f network-policies/block-suspicious-ips.yaml"
                    ],
                    "expected_outcome": "Threat contained and impact assessed"
                },
                {
                    "step": 2,
                    "title": "Collect Evidence",
                    "description": "Preserve logs and evidence for investigation",
                    "commands": [
                        "kubectl logs -l app=mten -n mten-production --since=24h > security-incident-logs.txt",
                        "kubectl get events -n mten-production --sort-by=.metadata.creationTimestamp > security-incident-events.txt"
                    ],
                    "expected_outcome": "Evidence collected and preserved"
                },
                {
                    "step": 3,
                    "title": "Rotate Credentials",
                    "description": "Rotate all potentially compromised credentials",
                    "commands": [
                        "# Generate new API keys:",
                        "python generate_api_keys.py --rotate-all",
                        "# Update database passwords:",
                        "kubectl create secret generic mten-secrets-new --from-env-file=.env.new -n mten-production",
                        "kubectl patch deployment/mten -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"mten\",\"envFrom\":[{\"secretRef\":{\"name\":\"mten-secrets-new\"}}]}]}}}}' -n mten-production"
                    ],
                    "expected_outcome": "All credentials rotated successfully"
                },
                {
                    "step": 4,
                    "title": "Patch Vulnerabilities",
                    "description": "Apply security patches if vulnerabilities identified",
                    "commands": [
                        "# Update container images:",
                        "kubectl set image deployment/mten mten=mten:security-patch -n mten-production",
                        "# Update system packages:",
                        "kubectl apply -f patches/security-updates.yaml"
                    ],
                    "expected_outcome": "Security patches applied"
                },
                {
                    "step": 5,
                    "title": "Enhanced Monitoring",
                    "description": "Enable enhanced security monitoring",
                    "commands": [
                        "kubectl apply -f monitoring/security-monitoring.yaml",
                        "python security_monitor.py --enable-enhanced-monitoring"
                    ],
                    "expected_outcome": "Enhanced monitoring active"
                }
            ],
            validation=[
                "No ongoing security alerts",
                "All systems responding normally",
                "Enhanced monitoring functional",
                "Audit trail complete"
            ],
            rollback=[
                "Credential rotation cannot be rolled back",
                "Security patches should not be rolled back",
                "Enhanced monitoring can be disabled if causing issues"
            ],
            estimated_duration=90,
            required_permissions=["security-admin", "cluster-admin", "secret-management"]
        )
        
        return runbooks
    
    def _create_backup_recovery_runbooks(self) -> Dict[str, Runbook]:
        """Create backup and recovery runbooks"""
        runbooks = {}
        
        # Database Recovery
        runbooks["database-recovery"] = Runbook(
            id="database-recovery",
            title="Database Recovery Procedure",
            category=RunbookCategory.BACKUP_RECOVERY,
            severity=IncidentSeverity.CRITICAL,
            description="Complete database recovery from backup",
            prerequisites=[
                "Database failure confirmed",
                "Valid backup identified",
                "Maintenance window scheduled"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Stop Application Traffic",
                    "description": "Prevent new database connections",
                    "commands": [
                        "kubectl scale deployment/mten --replicas=0 -n mten-production",
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"version\":\"maintenance\"}}}'",
                    ],
                    "expected_outcome": "All application traffic stopped"
                },
                {
                    "step": 2,
                    "title": "Identify Recovery Point",
                    "description": "Select appropriate backup for recovery",
                    "commands": [
                        "ls -la /backups/production/",
                        "# Select most recent consistent backup before incident"
                    ],
                    "expected_outcome": "Recovery point selected"
                },
                {
                    "step": 3,
                    "title": "Restore Database",
                    "description": "Execute database restoration",
                    "commands": [
                        "./backup/recovery.sh [backup_timestamp]",
                        "psql -c 'SELECT version();' # Verify connection"
                    ],
                    "expected_outcome": "Database restored successfully"
                },
                {
                    "step": 4,
                    "title": "Verify Data Integrity",
                    "description": "Validate restored data integrity",
                    "commands": [
                        "psql -c 'SELECT COUNT(*) FROM tenants;'",
                        "python data_integrity_check.py"
                    ],
                    "expected_outcome": "Data integrity confirmed"
                },
                {
                    "step": 5,
                    "title": "Restart Services",
                    "description": "Bring application services back online",
                    "commands": [
                        "kubectl scale deployment/mten --replicas=5 -n mten-production",
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"app\":\"mten\"}}}'",
                        "kubectl rollout status deployment/mten -n mten-production"
                    ],
                    "expected_outcome": "All services operational"
                }
            ],
            validation=[
                "Database accessible and responsive",
                "Application health checks passing",
                "Data integrity verified",
                "Performance metrics normal"
            ],
            rollback=[
                "If recovery fails, can attempt with earlier backup",
                "Document any data loss between backup and incident time"
            ],
            estimated_duration=45,
            required_permissions=["database-admin", "cluster-admin", "backup-access"]
        )
        
        return runbooks
    
    def _create_troubleshooting_runbooks(self) -> Dict[str, Runbook]:
        """Create troubleshooting runbooks"""
        runbooks = {}
        
        # General Troubleshooting
        runbooks["general-troubleshooting"] = Runbook(
            id="general-troubleshooting",
            title="General Troubleshooting Guide",
            category=RunbookCategory.TROUBLESHOOTING,
            severity=IncidentSeverity.MEDIUM,
            description="General troubleshooting steps for common issues",
            prerequisites=[
                "Issue reported or detected",
                "Basic system access available"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Gather Initial Information",
                    "description": "Collect basic system status information",
                    "commands": [
                        "kubectl get pods -n mten-production",
                        "kubectl get events -n mten-production --sort-by=.metadata.creationTimestamp | tail -20",
                        "curl -s https://mten.example.com/health | jq ."
                    ],
                    "expected_outcome": "System overview obtained"
                },
                {
                    "step": 2,
                    "title": "Check Application Logs",
                    "description": "Examine recent application logs for errors",
                    "commands": [
                        "kubectl logs -l app=mten -n mten-production --tail=100",
                        "kubectl logs -l app=mten -n mten-production --since=1h | grep -i error"
                    ],
                    "expected_outcome": "Error patterns identified"
                },
                {
                    "step": 3,
                    "title": "Verify Dependencies",
                    "description": "Check database, cache, and external service connectivity",
                    "commands": [
                        "kubectl exec deployment/mten -n mten-production -- pg_isready -h $DATABASE_HOST",
                        "kubectl exec deployment/mten -n mten-production -- redis-cli -h $REDIS_HOST ping"
                    ],
                    "expected_outcome": "Dependency health confirmed"
                },
                {
                    "step": 4,
                    "title": "Check Resource Constraints",
                    "description": "Verify system has adequate resources",
                    "commands": [
                        "kubectl describe nodes | grep -A5 'Allocated resources'",
                        "kubectl top pods -n mten-production"
                    ],
                    "expected_outcome": "Resource usage within limits"
                }
            ],
            validation=[
                "Root cause identified or narrowed down",
                "System stability confirmed",
                "Dependencies healthy"
            ],
            rollback=[],
            estimated_duration=15,
            required_permissions=["monitoring-read", "logs-read"]
        )
        
        return runbooks
    
    def _create_maintenance_runbooks(self) -> Dict[str, Runbook]:
        """Create maintenance runbooks"""
        runbooks = {}
        
        # Scheduled Maintenance
        runbooks["scheduled-maintenance"] = Runbook(
            id="scheduled-maintenance",
            title="Scheduled Maintenance Procedure",
            category=RunbookCategory.MAINTENANCE,
            severity=IncidentSeverity.MEDIUM,
            description="Procedure for scheduled system maintenance",
            prerequisites=[
                "Maintenance window scheduled and announced",
                "Backup completed",
                "Change management approval obtained"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Pre-maintenance Backup",
                    "description": "Create full system backup before maintenance",
                    "commands": [
                        "./backup/backup.sh production",
                        "kubectl create backup maintenance-$(date +%Y%m%d-%H%M%S)"
                    ],
                    "expected_outcome": "Complete backup created"
                },
                {
                    "step": 2,
                    "title": "Enable Maintenance Mode",
                    "description": "Route traffic to maintenance page",
                    "commands": [
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"version\":\"maintenance\"}}}' -n mten-production",
                        "kubectl apply -f maintenance/maintenance-page.yaml"
                    ],
                    "expected_outcome": "Maintenance page active"
                },
                {
                    "step": 3,
                    "title": "Perform Maintenance Tasks",
                    "description": "Execute planned maintenance activities",
                    "commands": [
                        "# Update system packages:",
                        "kubectl apply -f maintenance/system-updates.yaml",
                        "# Database maintenance:",
                        "psql -c 'REINDEX DATABASE mten_production;'",
                        "psql -c 'VACUUM ANALYZE;'"
                    ],
                    "expected_outcome": "All maintenance tasks completed"
                },
                {
                    "step": 4,
                    "title": "Validation Testing",
                    "description": "Verify system functionality after maintenance",
                    "commands": [
                        "python smoke_tests.py --environment=production",
                        "kubectl get pods -n mten-production"
                    ],
                    "expected_outcome": "All systems functional"
                },
                {
                    "step": 5,
                    "title": "Resume Normal Operations",
                    "description": "Return system to normal operation",
                    "commands": [
                        "kubectl patch service mten -p '{\"spec\":{\"selector\":{\"app\":\"mten\"}}}' -n mten-production",
                        "kubectl delete -f maintenance/maintenance-page.yaml"
                    ],
                    "expected_outcome": "Normal operations resumed"
                }
            ],
            validation=[
                "All services responding normally",
                "Performance metrics normal",
                "No errors in application logs"
            ],
            rollback=[
                "Restore from pre-maintenance backup if issues arise",
                "./backup/recovery.sh [maintenance_backup_timestamp]"
            ],
            estimated_duration=120,
            required_permissions=["maintenance-admin", "cluster-admin"]
        )
        
        return runbooks
    
    def _create_incident_response_runbooks(self) -> Dict[str, Runbook]:
        """Create incident response runbooks"""
        runbooks = {}
        
        # Critical Service Down
        runbooks["service-down-critical"] = Runbook(
            id="service-down-critical",
            title="Critical Service Down Response",
            category=RunbookCategory.INCIDENT_RESPONSE,
            severity=IncidentSeverity.CRITICAL,
            description="Immediate response for critical service outage",
            prerequisites=[
                "Service outage confirmed",
                "Incident response team activated"
            ],
            steps=[
                {
                    "step": 1,
                    "title": "Immediate Assessment",
                    "description": "Quickly assess outage scope and impact",
                    "commands": [
                        "curl -I https://mten.example.com/health",
                        "kubectl get pods -n mten-production",
                        "kubectl get events -n mten-production --sort-by=.metadata.creationTimestamp | tail -10"
                    ],
                    "expected_outcome": "Outage scope and cause identified"
                },
                {
                    "step": 2,
                    "title": "Immediate Recovery Actions",
                    "description": "Apply quick fixes to restore service",
                    "commands": [
                        "# Restart failing pods:",
                        "kubectl delete pods -l app=mten -n mten-production",
                        "# Scale up if needed:",
                        "kubectl scale deployment/mten --replicas=10 -n mten-production",
                        "# Switch to backup systems if available:",
                        "kubectl apply -f disaster-recovery/backup-deployment.yaml"
                    ],
                    "expected_outcome": "Service restoration attempted"
                },
                {
                    "step": 3,
                    "title": "Monitor Recovery",
                    "description": "Monitor service recovery progress",
                    "commands": [
                        "watch kubectl get pods -n mten-production",
                        "curl -f https://mten.example.com/health"
                    ],
                    "expected_outcome": "Service health restored"
                },
                {
                    "step": 4,
                    "title": "Validate Full Functionality",
                    "description": "Ensure all critical functions are working",
                    "commands": [
                        "python critical_function_tests.py",
                        "python performance_monitor.py --validate-sla"
                    ],
                    "expected_outcome": "All critical functions operational"
                }
            ],
            validation=[
                "Service responding to health checks",
                "All critical endpoints functional",
                "Performance within SLA",
                "Error rates normal"
            ],
            rollback=[
                "If recovery actions cause issues, rollback deployment",
                "kubectl rollout undo deployment/mten -n mten-production"
            ],
            estimated_duration=10,
            required_permissions=["incident-commander", "cluster-admin"]
        )
        
        return runbooks
    
    async def execute_runbook(self, runbook_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a runbook with guided steps"""
        if runbook_id not in self.runbooks:
            raise ValueError(f"Runbook {runbook_id} not found")
        
        runbook = self.runbooks[runbook_id]
        
        logger.info(f"🔧 Executing runbook: {runbook.title}")
        logger.info(f"   Category: {runbook.category.value}")
        logger.info(f"   Severity: {runbook.severity.value}")
        logger.info(f"   Estimated Duration: {runbook.estimated_duration} minutes")
        
        execution_log = {
            "runbook_id": runbook_id,
            "start_time": datetime.now(UTC),
            "steps_completed": [],
            "steps_failed": [],
            "context": context or {},
            "status": "in_progress"
        }
        
        try:
            # Validate prerequisites
            logger.info(f"\n📋 Prerequisites ({len(runbook.prerequisites)}):")
            for i, prereq in enumerate(runbook.prerequisites, 1):
                logger.info(f"   {i}. {prereq}")
            
            input("Press Enter to confirm prerequisites are met...")
            
            # Execute steps
            for step in runbook.steps:
                step_num = step["step"]
                step_title = step["title"]
                step_desc = step["description"]
                
                logger.info(f"\n🔄 Step {step_num}: {step_title}")
                logger.info(f"   Description: {step_desc}")
                
                if "commands" in step:
                    logger.info("   Commands to execute:")
                    for cmd in step["commands"]:
                        logger.info(f"     $ {cmd}")
                
                if "expected_outcome" in step:
                    logger.info(f"   Expected outcome: {step['expected_outcome']}")
                
                # Wait for user confirmation
                result = input(f"\nExecute step {step_num}? (y/n/s for skip): ").lower()
                
                if result == 'n':
                    execution_log["steps_failed"].append(step_num)
                    execution_log["status"] = "failed"
                    logger.error(f"❌ Step {step_num} failed or cancelled")
                    break
                elif result == 's':
                    logger.info(f"⏭️ Step {step_num} skipped")
                    continue
                else:
                    execution_log["steps_completed"].append(step_num)
                    logger.info(f"✅ Step {step_num} completed")
            
            # Validation
            if execution_log["status"] != "failed":
                logger.info(f"\n✅ Validation Checklist ({len(runbook.validation)}):")
                for i, check in enumerate(runbook.validation, 1):
                    logger.info(f"   {i}. {check}")
                
                validation_result = input("\nAll validations passed? (y/n): ").lower()
                if validation_result == 'y':
                    execution_log["status"] = "completed"
                    logger.info("🎉 Runbook execution completed successfully!")
                else:
                    execution_log["status"] = "validation_failed"
                    logger.error("❌ Validation failed")
        
        except KeyboardInterrupt:
            execution_log["status"] = "cancelled"
            logger.info("\n⏹️ Runbook execution cancelled")
        
        except Exception as e:
            execution_log["status"] = "error"
            execution_log["error"] = str(e)
            logger.error(f"❌ Runbook execution error: {e}")
        
        finally:
            execution_log["end_time"] = datetime.now(UTC)
            execution_log["duration"] = (execution_log["end_time"] - execution_log["start_time"]).total_seconds()
        
        return execution_log
    
    def get_runbook_by_symptoms(self, symptoms: List[str]) -> List[Runbook]:
        """Get recommended runbooks based on symptoms"""
        recommendations = []
        
        # Simple keyword matching - in production would use more sophisticated matching
        symptom_keywords = [s.lower() for s in symptoms]
        
        for runbook in self.runbooks.values():
            runbook_text = f"{runbook.title} {runbook.description}".lower()
            
            matches = sum(1 for keyword in symptom_keywords if keyword in runbook_text)
            if matches > 0:
                recommendations.append((runbook, matches))
        
        # Sort by relevance (number of matches)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [r[0] for r in recommendations[:5]]  # Return top 5 recommendations
    
    def create_incident_report(self, 
                             title: str, 
                             severity: IncidentSeverity, 
                             description: str,
                             affected_services: List[str]) -> IncidentReport:
        """Create a new incident report"""
        incident_id = f"INC-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        
        report = IncidentReport(
            incident_id=incident_id,
            title=title,
            severity=severity,
            status="open",
            description=description,
            affected_services=affected_services,
            start_time=datetime.now(UTC),
            detection_time=datetime.now(UTC)
        )
        
        self.incident_history.append(report)
        
        logger.info(f"🚨 Incident created: {incident_id}")
        logger.info(f"   Title: {title}")
        logger.info(f"   Severity: {severity.value}")
        logger.info(f"   Affected services: {', '.join(affected_services)}")
        
        return report
    
    def get_runbook_recommendations(self, incident_id: str) -> List[Runbook]:
        """Get runbook recommendations for an incident"""
        incident = next((i for i in self.incident_history if i.incident_id == incident_id), None)
        if not incident:
            return []
        
        # Create symptoms from incident data
        symptoms = [incident.title, incident.description] + incident.affected_services
        
        return self.get_runbook_by_symptoms(symptoms)
    
    def update_incident_status(self, incident_id: str, status: str, **kwargs):
        """Update incident status and metadata"""
        incident = next((i for i in self.incident_history if i.incident_id == incident_id), None)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident.status = status
        
        if status == "resolved":
            incident.resolution_time = datetime.now(UTC)
        
        # Update other fields
        for key, value in kwargs.items():
            if hasattr(incident, key):
                setattr(incident, key, value)
        
        logger.info(f"📝 Incident {incident_id} updated: {status}")


class OperationalDashboard:
    """Operational dashboard for runbook management"""
    
    def __init__(self, runbook_engine: RunbookEngine):
        self.engine = runbook_engine
    
    async def display_runbook_menu(self):
        """Display interactive runbook menu"""
        while True:
            print("\n" + "=" * 60)
            print("🔧 MTen Operational Runbooks Dashboard")
            print("=" * 60)
            
            print("\n📚 Available Options:")
            print("1. List all runbooks")
            print("2. Search runbooks by category")
            print("3. Execute runbook")
            print("4. Create incident report")
            print("5. Get runbook recommendations")
            print("6. View incident history")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                await self._list_runbooks()
            elif choice == "2":
                await self._search_runbooks_by_category()
            elif choice == "3":
                await self._execute_runbook()
            elif choice == "4":
                await self._create_incident()
            elif choice == "5":
                await self._get_recommendations()
            elif choice == "6":
                await self._view_incident_history()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    async def _list_runbooks(self):
        """List all available runbooks"""
        print("\n📋 Available Runbooks:")
        print("-" * 80)
        
        for category in RunbookCategory:
            category_runbooks = [r for r in self.engine.runbooks.values() if r.category == category]
            if category_runbooks:
                print(f"\n📁 {category.value.replace('_', ' ').title()}:")
                for runbook in category_runbooks:
                    severity_emoji = {
                        IncidentSeverity.LOW: "🟢",
                        IncidentSeverity.MEDIUM: "🟡", 
                        IncidentSeverity.HIGH: "🟠",
                        IncidentSeverity.CRITICAL: "🔴"
                    }
                    print(f"   {severity_emoji[runbook.severity]} {runbook.id}: {runbook.title}")
                    print(f"      Duration: ~{runbook.estimated_duration}min")
    
    async def _search_runbooks_by_category(self):
        """Search runbooks by category"""
        print("\n📁 Categories:")
        for i, category in enumerate(RunbookCategory, 1):
            print(f"   {i}. {category.value.replace('_', ' ').title()}")
        
        try:
            choice = int(input("\nSelect category (1-8): ")) - 1
            if 0 <= choice < len(RunbookCategory):
                selected_category = list(RunbookCategory)[choice]
                category_runbooks = [r for r in self.engine.runbooks.values() if r.category == selected_category]
                
                print(f"\n📋 {selected_category.value.replace('_', ' ').title()} Runbooks:")
                for runbook in category_runbooks:
                    print(f"   • {runbook.id}: {runbook.title}")
                    print(f"     {runbook.description}")
                    print(f"     Estimated duration: {runbook.estimated_duration} minutes\n")
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    async def _execute_runbook(self):
        """Execute a runbook interactively"""
        runbook_id = input("\n🔧 Enter runbook ID to execute: ").strip()
        
        if runbook_id in self.engine.runbooks:
            context = {}
            context_input = input("Enter context (JSON format, or press Enter for none): ").strip()
            if context_input:
                try:
                    context = json.loads(context_input)
                except json.JSONDecodeError:
                    print("⚠️ Invalid JSON format, proceeding without context")
            
            execution_log = await self.engine.execute_runbook(runbook_id, context)
            
            print(f"\n📊 Execution Summary:")
            print(f"   Status: {execution_log['status']}")
            print(f"   Duration: {execution_log['duration']:.2f} seconds")
            print(f"   Steps completed: {len(execution_log['steps_completed'])}")
            if execution_log['steps_failed']:
                print(f"   Steps failed: {execution_log['steps_failed']}")
        else:
            print("❌ Runbook not found")
    
    async def _create_incident(self):
        """Create a new incident report"""
        print("\n🚨 Create Incident Report:")
        
        title = input("Incident title: ").strip()
        description = input("Description: ").strip()
        
        print("\n📊 Severity levels:")
        for i, severity in enumerate(IncidentSeverity, 1):
            print(f"   {i}. {severity.value}")
        
        try:
            severity_choice = int(input("Select severity (1-4): ")) - 1
            severity = list(IncidentSeverity)[severity_choice]
        except (ValueError, IndexError):
            severity = IncidentSeverity.MEDIUM
            print("⚠️ Invalid selection, defaulting to MEDIUM")
        
        services = input("Affected services (comma-separated): ").strip().split(",")
        services = [s.strip() for s in services if s.strip()]
        
        incident = self.engine.create_incident_report(title, severity, description, services)
        
        # Get recommendations
        recommendations = self.engine.get_runbook_recommendations(incident.incident_id)
        if recommendations:
            print(f"\n💡 Recommended runbooks:")
            for i, runbook in enumerate(recommendations, 1):
                print(f"   {i}. {runbook.id}: {runbook.title}")
    
    async def _get_recommendations(self):
        """Get runbook recommendations based on symptoms"""
        symptoms = input("\n🔍 Enter symptoms (comma-separated): ").strip().split(",")
        symptoms = [s.strip() for s in symptoms if s.strip()]
        
        if symptoms:
            recommendations = self.engine.get_runbook_by_symptoms(symptoms)
            
            if recommendations:
                print(f"\n💡 Recommended runbooks for symptoms: {', '.join(symptoms)}")
                for i, runbook in enumerate(recommendations, 1):
                    print(f"   {i}. {runbook.id}: {runbook.title}")
                    print(f"      Category: {runbook.category.value}")
                    print(f"      Severity: {runbook.severity.value}")
                    print(f"      Duration: ~{runbook.estimated_duration}min\n")
            else:
                print("❌ No matching runbooks found")
        else:
            print("❌ No symptoms provided")
    
    async def _view_incident_history(self):
        """View incident history"""
        if not self.engine.incident_history:
            print("\n📋 No incidents recorded")
            return
        
        print("\n📋 Incident History:")
        print("-" * 80)
        
        for incident in reversed(self.engine.incident_history[-10:]):  # Show last 10
            duration = ""
            if incident.resolution_time:
                duration = f" (Duration: {(incident.resolution_time - incident.start_time).total_seconds():.0f}s)"
            
            print(f"🚨 {incident.incident_id}: {incident.title}")
            print(f"   Status: {incident.status} | Severity: {incident.severity.value}")
            print(f"   Started: {incident.start_time.strftime('%Y-%m-%d %H:%M:%S')}{duration}")
            if incident.affected_services:
                print(f"   Affected: {', '.join(incident.affected_services)}")
            print()


async def main():
    """Main entry point for operational runbooks system"""
    print("🔧 MTen Operational Runbooks & Troubleshooting System")
    print("=" * 70)
    
    # Initialize runbook engine
    engine = RunbookEngine()
    dashboard = OperationalDashboard(engine)
    
    # Display available runbooks
    print(f"\n📚 Loaded {len(engine.runbooks)} operational runbooks:")
    
    categories_count = {}
    for runbook in engine.runbooks.values():
        category = runbook.category.value.replace('_', ' ').title()
        categories_count[category] = categories_count.get(category, 0) + 1
    
    for category, count in categories_count.items():
        print(f"   • {category}: {count} runbooks")
    
    # Start interactive dashboard
    print(f"\n🎯 Ready for operational support!")
    await dashboard.display_runbook_menu()


if __name__ == "__main__":
    asyncio.run(main())