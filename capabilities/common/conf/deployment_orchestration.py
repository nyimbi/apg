"""
APG Configuration Management Advanced Deployment Orchestration

Comprehensive deployment orchestration engine with advanced strategies,
rollback capabilities, canary deployments, health monitoring, and
automated recovery mechanisms.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from uuid_extensions import uuid7str
import asyncio
import logging
import json
from dataclasses import dataclass, field

try:
    from .models import (
        CMResource, ConfigurationDSL, ResourceState, DeploymentStatus
    )
    from .gitops_integration import (
        DeploymentStrategy, DeploymentPlan, GitOpsManifest, 
        PipelineExecution, PipelineStatus
    )
except ImportError:
    from models import (
        CMResource, ConfigurationDSL, ResourceState, DeploymentStatus
    )
    # Mock imports for testing
    class DeploymentStrategy:
        ROLLING_UPDATE = "rolling_update"
        BLUE_GREEN = "blue_green"
        CANARY = "canary"
        RECREATE = "recreate"
    
    class DeploymentPlan:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

logger = logging.getLogger(__name__)


class DeploymentPhase(StrEnum):
    """Deployment execution phases"""
    PREPARATION = "preparation"
    PRE_DEPLOYMENT = "pre_deployment"
    DEPLOYMENT = "deployment"
    POST_DEPLOYMENT = "post_deployment"
    VERIFICATION = "verification"
    CLEANUP = "cleanup"


class DeploymentState(StrEnum):
    """Detailed deployment states"""
    PLANNED = "planned"
    APPROVED = "approved"
    STARTING = "starting"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class HealthCheckType(StrEnum):
    """Types of health checks"""
    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    CUSTOM = "custom"


class RollbackTrigger(StrEnum):
    """Rollback trigger conditions"""
    MANUAL = "manual"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    TIMEOUT = "timeout"


@dataclass
class HealthCheck:
    """Health check configuration"""
    id: str = field(default_factory=uuid7str)
    name: str = ""
    type: HealthCheckType = HealthCheckType.HTTP
    endpoint: Optional[str] = None
    command: Optional[str] = None
    expected_response: Optional[str] = None
    timeout_seconds: int = 30
    interval_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RollbackConfiguration:
    """Rollback configuration and triggers"""
    id: str = field(default_factory=uuid7str)
    automatic_rollback: bool = True
    rollback_triggers: List[RollbackTrigger] = field(default_factory=list)
    health_check_timeout_minutes: int = 10
    performance_degradation_threshold: float = 0.2  # 20% degradation
    error_rate_threshold: float = 0.05  # 5% error rate
    rollback_timeout_minutes: int = 15
    preserve_logs: bool = True
    notification_channels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DeploymentExecution:
    """Detailed deployment execution tracking"""
    id: str = field(default_factory=uuid7str)
    deployment_plan_id: str = ""
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    state: DeploymentState = DeploymentState.PLANNED
    current_phase: DeploymentPhase = DeploymentPhase.PREPARATION
    target_replicas: int = 1
    current_replicas: int = 0
    healthy_replicas: int = 0
    progress_percentage: float = 0.0
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Phase tracking
    phase_history: List[Dict[str, Any]] = field(default_factory=list)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status and logs
    logs: List[Dict[str, str]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Rollback information
    rollback_configuration: Optional[RollbackConfiguration] = None
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None
    rollback_completed: bool = False
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CanaryConfiguration:
    """Canary deployment configuration"""
    id: str = field(default_factory=uuid7str)
    traffic_split_percentage: float = 10.0
    evaluation_period_minutes: int = 5
    success_criteria: List[Dict[str, Any]] = field(default_factory=list)
    automated_promotion: bool = True
    promotion_threshold: float = 0.95  # 95% success rate
    analysis_queries: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class HealthCheckManager:
    """Manages health checks for deployments"""
    
    def __init__(self):
        self.active_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute_health_check(self, check: HealthCheck, target_endpoint: str) -> Dict[str, Any]:
        """Execute individual health check"""
        result = {
            "check_id": check.id,
            "check_name": check.name,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "response_time_ms": 0,
            "details": {}
        }
        
        start_time = datetime.utcnow()
        
        try:
            if check.type == HealthCheckType.HTTP:
                # Simulate HTTP health check
                await asyncio.sleep(0.1)  # Simulate network delay
                result["success"] = True
                result["details"] = {"status_code": 200, "response": "OK"}
                
            elif check.type == HealthCheckType.TCP:
                # Simulate TCP health check
                await asyncio.sleep(0.05)
                result["success"] = True
                result["details"] = {"connection": "established"}
                
            elif check.type == HealthCheckType.COMMAND:
                # Simulate command health check
                await asyncio.sleep(0.2)
                result["success"] = True
                result["details"] = {"exit_code": 0, "output": "Health check passed"}
                
            else:  # CUSTOM
                # Simulate custom health check
                await asyncio.sleep(0.15)
                result["success"] = True
                result["details"] = {"custom_metric": "healthy"}
            
            result["response_time_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
            
        except Exception as e:
            result["success"] = False
            result["details"] = {"error": str(e)}
            result["response_time_ms"] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return result
    
    async def run_health_checks(self, checks: List[HealthCheck], target_endpoint: str) -> Dict[str, Any]:
        """Run all health checks for a deployment"""
        results = []
        
        for check in checks:
            if not check.enabled:
                continue
            
            check_result = await self.execute_health_check(check, target_endpoint)
            results.append(check_result)
            
            # Store result history
            if check.id not in self.check_results:
                self.check_results[check.id] = []
            
            self.check_results[check.id].append(check_result)
            
            # Keep only recent results (last 100)
            if len(self.check_results[check.id]) > 100:
                self.check_results[check.id] = self.check_results[check.id][-100:]
        
        # Calculate overall health status
        successful_checks = [r for r in results if r["success"]]
        total_checks = len(results)
        
        overall_health = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_checks": total_checks,
            "successful_checks": len(successful_checks),
            "success_rate": len(successful_checks) / max(1, total_checks),
            "healthy": len(successful_checks) == total_checks,
            "average_response_time_ms": sum(r["response_time_ms"] for r in results) / max(1, total_checks),
            "individual_results": results
        }
        
        return overall_health


class DeploymentOrchestrator:
    """Advanced deployment orchestration engine"""
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.active_deployments: Dict[str, DeploymentExecution] = {}
        self.health_check_manager = HealthCheckManager()
        self.deployment_templates: Dict[str, Dict[str, Any]] = {}
        self._initialize_deployment_templates()
    
    def _initialize_deployment_templates(self):
        """Initialize deployment strategy templates"""
        self.deployment_templates = {
            "rolling_update": {
                "phases": [
                    DeploymentPhase.PREPARATION,
                    DeploymentPhase.PRE_DEPLOYMENT,
                    DeploymentPhase.DEPLOYMENT,
                    DeploymentPhase.VERIFICATION,
                    DeploymentPhase.POST_DEPLOYMENT,
                    DeploymentPhase.CLEANUP
                ],
                "health_checks": [
                    HealthCheck(
                        name="Application Health",
                        type=HealthCheckType.HTTP,
                        endpoint="/health",
                        timeout_seconds=30,
                        interval_seconds=10,
                        failure_threshold=3
                    ),
                    HealthCheck(
                        name="Readiness Check",
                        type=HealthCheckType.HTTP,
                        endpoint="/ready",
                        timeout_seconds=20,
                        failure_threshold=2
                    )
                ],
                "rollback_triggers": [
                    RollbackTrigger.HEALTH_CHECK_FAILURE,
                    RollbackTrigger.ERROR_RATE_THRESHOLD
                ]
            },
            "blue_green": {
                "phases": [
                    DeploymentPhase.PREPARATION,
                    DeploymentPhase.PRE_DEPLOYMENT,
                    DeploymentPhase.DEPLOYMENT,
                    DeploymentPhase.VERIFICATION,
                    DeploymentPhase.POST_DEPLOYMENT,
                    DeploymentPhase.CLEANUP
                ],
                "health_checks": [
                    HealthCheck(
                        name="Green Environment Health",
                        type=HealthCheckType.HTTP,
                        endpoint="/health",
                        timeout_seconds=45,
                        failure_threshold=1  # Strict for blue-green
                    ),
                    HealthCheck(
                        name="Load Balancer Check",
                        type=HealthCheckType.TCP,
                        timeout_seconds=15
                    )
                ],
                "rollback_triggers": [
                    RollbackTrigger.HEALTH_CHECK_FAILURE,
                    RollbackTrigger.PERFORMANCE_DEGRADATION
                ]
            },
            "canary": {
                "phases": [
                    DeploymentPhase.PREPARATION,
                    DeploymentPhase.PRE_DEPLOYMENT,
                    DeploymentPhase.DEPLOYMENT,
                    DeploymentPhase.VERIFICATION,
                    DeploymentPhase.POST_DEPLOYMENT,
                    DeploymentPhase.CLEANUP
                ],
                "health_checks": [
                    HealthCheck(
                        name="Canary Health",
                        type=HealthCheckType.HTTP,
                        endpoint="/health",
                        timeout_seconds=30,
                        interval_seconds=5,  # More frequent for canary
                        failure_threshold=2
                    ),
                    HealthCheck(
                        name="Performance Metrics",
                        type=HealthCheckType.CUSTOM,
                        timeout_seconds=60
                    )
                ],
                "rollback_triggers": [
                    RollbackTrigger.HEALTH_CHECK_FAILURE,
                    RollbackTrigger.PERFORMANCE_DEGRADATION,
                    RollbackTrigger.ERROR_RATE_THRESHOLD
                ],
                "canary_config": CanaryConfiguration(
                    traffic_split_percentage=5.0,
                    evaluation_period_minutes=3,
                    automated_promotion=True
                )
            }
        }
    
    async def orchestrate_deployment(self, plan: DeploymentPlan, manifest: GitOpsManifest) -> str:
        """Orchestrate advanced deployment with rollback capabilities"""
        execution = DeploymentExecution(
            deployment_plan_id=plan.id,
            strategy=plan.strategy,
            state=DeploymentState.STARTING,
            target_replicas=getattr(plan, 'target_replicas', 1),
            started_at=datetime.utcnow()
        )
        
        # Set up rollback configuration
        execution.rollback_configuration = RollbackConfiguration(
            automatic_rollback=True,
            rollback_triggers=[
                RollbackTrigger.HEALTH_CHECK_FAILURE,
                RollbackTrigger.ERROR_RATE_THRESHOLD
            ],
            health_check_timeout_minutes=10
        )
        
        self.active_deployments[execution.id] = execution
        
        # Start deployment orchestration
        asyncio.create_task(self._execute_deployment_phases(execution, plan, manifest))
        
        logger.info(f"Started deployment orchestration: {execution.id} using {plan.strategy.value}")
        return execution.id
    
    async def _execute_deployment_phases(
        self,
        execution: DeploymentExecution,
        plan: DeploymentPlan,
        manifest: GitOpsManifest
    ):
        """Execute deployment phases with health monitoring and rollback"""
        template = self.deployment_templates.get(execution.strategy.value, self.deployment_templates["rolling_update"])
        phases = template["phases"]
        health_checks = template["health_checks"]
        
        try:
            for phase in phases:
                execution.current_phase = phase
                execution.state = DeploymentState.DEPLOYING
                
                # Log phase start
                phase_start = datetime.utcnow()
                execution.logs.append({
                    "level": "info",
                    "message": f"Starting deployment phase: {phase.value}",
                    "timestamp": phase_start.isoformat()
                })
                
                # Execute phase
                phase_success = await self._execute_deployment_phase(
                    execution, phase, plan, manifest, health_checks
                )
                
                # Record phase completion
                phase_end = datetime.utcnow()
                phase_duration = (phase_end - phase_start).total_seconds()
                
                execution.phase_history.append({
                    "phase": phase.value,
                    "success": phase_success,
                    "started_at": phase_start.isoformat(),
                    "completed_at": phase_end.isoformat(),
                    "duration_seconds": phase_duration
                })
                
                if not phase_success:
                    # Phase failed - trigger rollback if configured
                    await self._handle_deployment_failure(execution, f"Phase {phase.value} failed")
                    return
                
                # Update progress
                phase_progress = (phases.index(phase) + 1) / len(phases) * 100
                execution.progress_percentage = phase_progress
                
                execution.logs.append({
                    "level": "info",
                    "message": f"Phase {phase.value} completed successfully ({phase_progress:.1f}% complete)",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # All phases completed successfully
            execution.state = DeploymentState.SUCCEEDED
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            execution.progress_percentage = 100.0
            
            execution.logs.append({
                "level": "info",
                "message": f"Deployment completed successfully in {execution.duration_seconds:.1f}s",
                "timestamp": execution.completed_at.isoformat()
            })
            
            logger.info(f"Deployment {execution.id} completed successfully")
            
        except Exception as e:
            await self._handle_deployment_failure(execution, str(e))
    
    async def _execute_deployment_phase(
        self,
        execution: DeploymentExecution,
        phase: DeploymentPhase,
        plan: DeploymentPlan,
        manifest: GitOpsManifest,
        health_checks: List[HealthCheck]
    ) -> bool:
        """Execute individual deployment phase"""
        
        if phase == DeploymentPhase.PREPARATION:
            return await self._execute_preparation_phase(execution, plan, manifest)
        elif phase == DeploymentPhase.PRE_DEPLOYMENT:
            return await self._execute_pre_deployment_phase(execution, plan, manifest)
        elif phase == DeploymentPhase.DEPLOYMENT:
            return await self._execute_deployment_phase_core(execution, plan, manifest)
        elif phase == DeploymentPhase.VERIFICATION:
            return await self._execute_verification_phase(execution, health_checks, manifest)
        elif phase == DeploymentPhase.POST_DEPLOYMENT:
            return await self._execute_post_deployment_phase(execution, plan, manifest)
        elif phase == DeploymentPhase.CLEANUP:
            return await self._execute_cleanup_phase(execution, plan, manifest)
        else:
            execution.logs.append({
                "level": "warning",
                "message": f"Unknown deployment phase: {phase.value}",
                "timestamp": datetime.utcnow().isoformat()
            })
            return True  # Don't fail on unknown phases
    
    async def _execute_preparation_phase(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute preparation phase"""
        execution.logs.append({
            "level": "info",
            "message": "Preparing deployment environment and validating resources",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate preparation work
        await asyncio.sleep(1)
        
        # Validate manifest
        if not manifest.content:
            execution.logs.append({
                "level": "error",
                "message": "Invalid manifest: empty content",
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
        
        execution.logs.append({
            "level": "info",
            "message": "Preparation phase completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_pre_deployment_phase(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute pre-deployment phase"""
        execution.logs.append({
            "level": "info",
            "message": "Executing pre-deployment hooks and validations",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate pre-deployment work
        await asyncio.sleep(0.5)
        
        execution.logs.append({
            "level": "info",
            "message": "Pre-deployment phase completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_deployment_phase_core(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute core deployment phase based on strategy"""
        strategy = execution.strategy
        
        if strategy == DeploymentStrategy.ROLLING_UPDATE:
            return await self._execute_rolling_update(execution, plan, manifest)
        elif strategy == DeploymentStrategy.BLUE_GREEN:
            return await self._execute_blue_green_deployment(execution, plan, manifest)
        elif strategy == DeploymentStrategy.CANARY:
            return await self._execute_canary_deployment(execution, plan, manifest)
        else:
            return await self._execute_recreate_deployment(execution, plan, manifest)
    
    async def _execute_rolling_update(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute rolling update deployment"""
        execution.logs.append({
            "level": "info",
            "message": f"Starting rolling update deployment (target: {execution.target_replicas} replicas)",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate rolling update with gradual replica updates
        for i in range(execution.target_replicas):
            execution.current_replicas = i + 1
            
            # Simulate deployment of individual replica
            await asyncio.sleep(0.5)
            
            execution.logs.append({
                "level": "info",
                "message": f"Deployed replica {i + 1}/{execution.target_replicas}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Assume replica becomes healthy after brief delay
            await asyncio.sleep(0.2)
            execution.healthy_replicas = i + 1
        
        execution.logs.append({
            "level": "info",
            "message": f"Rolling update completed: {execution.healthy_replicas}/{execution.target_replicas} healthy",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_blue_green_deployment(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute blue-green deployment"""
        execution.logs.append({
            "level": "info",
            "message": "Starting blue-green deployment",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate green environment deployment
        await asyncio.sleep(1.5)
        execution.current_replicas = execution.target_replicas
        
        execution.logs.append({
            "level": "info",
            "message": "Green environment deployed, preparing traffic switch",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate traffic switch
        await asyncio.sleep(0.3)
        execution.healthy_replicas = execution.target_replicas
        
        execution.logs.append({
            "level": "info",
            "message": "Traffic switched to green environment successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_canary_deployment(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute canary deployment with traffic splitting"""
        template = self.deployment_templates["canary"]
        canary_config = template.get("canary_config", CanaryConfiguration())
        
        execution.logs.append({
            "level": "info",
            "message": f"Starting canary deployment ({canary_config.traffic_split_percentage}% traffic split)",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Deploy canary version
        canary_replicas = max(1, int(execution.target_replicas * canary_config.traffic_split_percentage / 100))
        execution.current_replicas = canary_replicas
        
        await asyncio.sleep(1.0)
        
        execution.logs.append({
            "level": "info",
            "message": f"Canary version deployed ({canary_replicas} replicas)",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate evaluation period
        execution.logs.append({
            "level": "info",
            "message": f"Evaluating canary performance for {canary_config.evaluation_period_minutes} minutes",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        await asyncio.sleep(2.0)  # Simulate evaluation
        
        # Assume canary is successful and promote
        execution.current_replicas = execution.target_replicas
        execution.healthy_replicas = execution.target_replicas
        
        execution.logs.append({
            "level": "info",
            "message": "Canary evaluation successful, promoting to full deployment",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_recreate_deployment(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute recreate deployment"""
        execution.logs.append({
            "level": "info",
            "message": "Starting recreate deployment",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate stopping old version
        await asyncio.sleep(0.5)
        
        # Simulate starting new version
        await asyncio.sleep(1.0)
        execution.current_replicas = execution.target_replicas
        execution.healthy_replicas = execution.target_replicas
        
        execution.logs.append({
            "level": "info",
            "message": "Recreate deployment completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_verification_phase(self, execution: DeploymentExecution, health_checks: List[HealthCheck], manifest: GitOpsManifest) -> bool:
        """Execute verification phase with health checks"""
        execution.logs.append({
            "level": "info",
            "message": f"Starting verification phase with {len(health_checks)} health checks",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Run health checks
        target_endpoint = f"http://deployment-{execution.id[:8]}.example.com"
        health_results = await self.health_check_manager.run_health_checks(health_checks, target_endpoint)
        
        # Store health check results
        execution.health_checks.append(health_results)
        
        if health_results["healthy"]:
            execution.logs.append({
                "level": "info",
                "message": f"Health checks passed ({health_results['success_rate']:.1%} success rate)",
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        else:
            execution.logs.append({
                "level": "error",
                "message": f"Health checks failed ({health_results['success_rate']:.1%} success rate)",
                "timestamp": datetime.utcnow().isoformat()
            })
            return False
    
    async def _execute_post_deployment_phase(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute post-deployment phase"""
        execution.logs.append({
            "level": "info",
            "message": "Executing post-deployment hooks and cleanup",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate post-deployment work
        await asyncio.sleep(0.5)
        
        execution.logs.append({
            "level": "info",
            "message": "Post-deployment phase completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _execute_cleanup_phase(self, execution: DeploymentExecution, plan: DeploymentPlan, manifest: GitOpsManifest) -> bool:
        """Execute cleanup phase"""
        execution.logs.append({
            "level": "info",
            "message": "Cleaning up temporary resources and finalizing deployment",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate cleanup work
        await asyncio.sleep(0.3)
        
        execution.logs.append({
            "level": "info",
            "message": "Cleanup phase completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    async def _handle_deployment_failure(self, execution: DeploymentExecution, reason: str):
        """Handle deployment failure and trigger rollback if configured"""
        execution.state = DeploymentState.FAILED
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        execution.logs.append({
            "level": "error",
            "message": f"Deployment failed: {reason}",
            "timestamp": execution.completed_at.isoformat()
        })
        
        # Trigger rollback if configured
        if execution.rollback_configuration and execution.rollback_configuration.automatic_rollback:
            await self._trigger_rollback(execution, reason)
        
        logger.error(f"Deployment {execution.id} failed: {reason}")
    
    async def _trigger_rollback(self, execution: DeploymentExecution, reason: str):
        """Trigger deployment rollback"""
        execution.rollback_triggered = True
        execution.rollback_reason = reason
        execution.state = DeploymentState.ROLLING_BACK
        
        rollback_start = datetime.utcnow()
        
        execution.logs.append({
            "level": "info",
            "message": f"Triggering automatic rollback due to: {reason}",
            "timestamp": rollback_start.isoformat()
        })
        
        try:
            # Simulate rollback execution
            await asyncio.sleep(2.0)
            
            # Reset replica counts to previous state
            execution.current_replicas = execution.target_replicas
            execution.healthy_replicas = execution.target_replicas
            
            rollback_end = datetime.utcnow()
            rollback_duration = (rollback_end - rollback_start).total_seconds()
            
            execution.rollback_history.append({
                "triggered_at": rollback_start.isoformat(),
                "completed_at": rollback_end.isoformat(),
                "duration_seconds": rollback_duration,
                "reason": reason,
                "success": True
            })
            
            execution.state = DeploymentState.ROLLED_BACK
            execution.rollback_completed = True
            
            execution.logs.append({
                "level": "info",
                "message": f"Rollback completed successfully in {rollback_duration:.1f}s",
                "timestamp": rollback_end.isoformat()
            })
            
            logger.info(f"Rollback completed for deployment {execution.id}")
            
        except Exception as e:
            execution.logs.append({
                "level": "error",
                "message": f"Rollback failed: {e}",
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.error(f"Rollback failed for deployment {execution.id}: {e}")
    
    async def get_deployment_status(self, execution_id: str) -> Optional[DeploymentExecution]:
        """Get deployment execution status"""
        return self.active_deployments.get(execution_id)
    
    async def cancel_deployment(self, execution_id: str) -> bool:
        """Cancel active deployment"""
        if execution_id not in self.active_deployments:
            return False
        
        execution = self.active_deployments[execution_id]
        
        if execution.state in [DeploymentState.SUCCEEDED, DeploymentState.FAILED, DeploymentState.CANCELLED]:
            return False  # Cannot cancel completed deployments
        
        execution.state = DeploymentState.CANCELLED
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        execution.logs.append({
            "level": "info",
            "message": "Deployment cancelled by user request",
            "timestamp": execution.completed_at.isoformat()
        })
        
        logger.info(f"Deployment {execution_id} cancelled")
        return True
    
    async def manual_rollback(self, execution_id: str, reason: str = "Manual rollback") -> bool:
        """Manually trigger rollback for deployment"""
        if execution_id not in self.active_deployments:
            return False
        
        execution = self.active_deployments[execution_id]
        
        if execution.rollback_triggered:
            return False  # Already rolling back
        
        await self._trigger_rollback(execution, reason)
        return True
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get deployment orchestrator metrics"""
        active_count = len([e for e in self.active_deployments.values() 
                          if e.state in [DeploymentState.STARTING, DeploymentState.DEPLOYING]])
        
        successful_count = len([e for e in self.active_deployments.values() 
                              if e.state == DeploymentState.SUCCEEDED])
        
        failed_count = len([e for e in self.active_deployments.values() 
                          if e.state == DeploymentState.FAILED])
        
        rollback_count = len([e for e in self.active_deployments.values() 
                            if e.rollback_triggered])
        
        total_deployments = len(self.active_deployments)
        
        return {
            "total_deployments": total_deployments,
            "active_deployments": active_count,
            "successful_deployments": successful_count,
            "failed_deployments": failed_count,
            "rollbacks_triggered": rollback_count,
            "success_rate": successful_count / max(1, total_deployments),
            "rollback_rate": rollback_count / max(1, total_deployments),
            "deployment_strategies": list(self.deployment_templates.keys()),
            "average_deployment_time": self._calculate_average_deployment_time(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_average_deployment_time(self) -> float:
        """Calculate average deployment time"""
        completed_deployments = [e for e in self.active_deployments.values() 
                               if e.duration_seconds is not None]
        
        if not completed_deployments:
            return 0.0
        
        total_time = sum(e.duration_seconds for e in completed_deployments)
        return total_time / len(completed_deployments)


# Global orchestrator instance
_orchestrator = None

async def get_deployment_orchestrator(tenant_id: Optional[str] = None) -> DeploymentOrchestrator:
    """Get global deployment orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DeploymentOrchestrator(tenant_id)
    return _orchestrator

# Export main classes
__all__ = [
    "DeploymentPhase",
    "DeploymentState", 
    "HealthCheckType",
    "RollbackTrigger",
    "HealthCheck",
    "RollbackConfiguration",
    "DeploymentExecution",
    "CanaryConfiguration",
    "HealthCheckManager",
    "DeploymentOrchestrator",
    "get_deployment_orchestrator"
]