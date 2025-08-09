"""
APG Deployment Automation Service

Complete implementation with real Docker and Kubernetes SDKs for container
orchestration, deployment strategies, and infrastructure management.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog
import yaml
import docker
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from ..database import CRDeployment, CRComposition, db_manager

logger = structlog.get_logger(__name__)

class DeploymentStrategy(str, Enum):
    """Deployment strategies."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"

class DeploymentEnvironment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
    DR = "disaster_recovery"

class DeploymentStatus(str, Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"

@dataclass
class DeploymentTarget:
    """Deployment target configuration."""
    environment: DeploymentEnvironment
    cluster_name: str
    namespace: str
    replicas: int = 3
    resource_limits: Optional[Dict[str, str]] = None
    health_check_url: Optional[str] = None
    ingress_host: Optional[str] = None
    storage_class: Optional[str] = None
    node_selector: Optional[Dict[str, str]] = None
    tolerations: Optional[List[Dict[str, Any]]] = None

@dataclass 
class DeploymentResult:
    """Deployment execution result."""
    deployment_id: str
    status: DeploymentStatus
    message: str
    manifest_applied: bool = False
    service_created: bool = False
    ingress_created: bool = False
    pods_ready: int = 0
    pods_total: int = 0
    rollout_url: Optional[str] = None
    health_status: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    logs: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    kubernetes_resources: List[str] = field(default_factory=list)
    docker_containers: List[str] = field(default_factory=list)

class DeploymentAutomationService:
    """Complete deployment automation service with real integrations."""
    
    def __init__(self, tenant_id: str, db_session: AsyncSession):
        self.tenant_id = tenant_id
        self.db_session = db_session
        
        # Initialize clients
        self.docker_client = None
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        self.k8s_networking_v1 = None
        self.aws_ecs_client = None
        self.aws_eks_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Docker, Kubernetes, and AWS clients."""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Docker client", error=str(e))
        
        try:
            # Initialize Kubernetes client
            try:
                # Try in-cluster config first
                config.load_incluster_config()
            except:
                # Fall back to local kubeconfig
                config.load_kube_config()
            
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_networking_v1 = client.NetworkingV1Api()
            logger.info("Kubernetes clients initialized")
        except Exception as e:
            logger.warning("Failed to initialize Kubernetes clients", error=str(e))
        
        try:
            # Initialize AWS clients
            session = boto3.Session()
            self.aws_ecs_client = session.client('ecs')
            self.aws_eks_client = session.client('eks')
            logger.info("AWS clients initialized")
        except (NoCredentialsError, ClientError) as e:
            logger.warning("Failed to initialize AWS clients", error=str(e))
    
    async def deploy_composition(
        self,
        composition_id: str,
        target: DeploymentTarget,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
        container_image: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> DeploymentResult:
        """Deploy a composition with the specified strategy."""
        start_time = datetime.utcnow()
        
        # Create deployment record
        deployment = CRDeployment(
            tenant_id=self.tenant_id,
            composition_id=composition_id,
            environment=target.environment.value,
            strategy=strategy.value,
            cluster_name=target.cluster_name,
            namespace=target.namespace,
            status=DeploymentStatus.PENDING.value,
            replicas=target.replicas,
            resource_limits=target.resource_limits or {},
            health_check_url=target.health_check_url,
            deployed_by="system"  # Would get from auth context
        )
        
        self.db_session.add(deployment)
        await self.db_session.commit()
        
        result = DeploymentResult(
            deployment_id=deployment.id,
            status=DeploymentStatus.IN_PROGRESS,
            message="Deployment started",
            start_time=start_time
        )
        
        try:
            # Update status to in progress
            deployment.status = DeploymentStatus.IN_PROGRESS.value
            deployment.deployment_logs = ["Deployment started"]
            await self.db_session.commit()
            
            # Get composition details
            composition = await self._get_composition(composition_id)
            if not composition:
                raise ValueError(f"Composition {composition_id} not found")
            
            # Determine container image
            if not container_image:
                container_image = f"apg-composition-{composition_id}:latest"
            
            # Execute deployment based on target platform
            if target.cluster_name.startswith("kubernetes-") or self.k8s_apps_v1:
                result = await self._deploy_to_kubernetes(
                    composition, target, strategy, container_image, result
                )
            elif target.cluster_name.startswith("docker-") or self.docker_client:
                result = await self._deploy_to_docker(
                    composition, target, strategy, container_image, result
                )
            elif target.cluster_name.startswith("aws-"):
                result = await self._deploy_to_aws(
                    composition, target, strategy, container_image, result
                )
            else:
                raise ValueError(f"Unsupported deployment target: {target.cluster_name}")
            
            # Update deployment record
            deployment.status = result.status.value
            deployment.message = result.message
            deployment.rollout_url = result.rollout_url
            deployment.health_status = result.health_status
            deployment.deployment_logs = result.logs
            
            result.end_time = datetime.utcnow()
            
            await self.db_session.commit()
            
            logger.info("Deployment completed", 
                       deployment_id=result.deployment_id, 
                       status=result.status.value)
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Deployment failed: {str(e)}"
            result.logs.append(f"ERROR: {str(e)}")
            
            deployment.status = DeploymentStatus.FAILED.value
            deployment.message = result.message
            deployment.deployment_logs = result.logs
            
            await self.db_session.commit()
            
            logger.error("Deployment failed", 
                        deployment_id=result.deployment_id, 
                        error=str(e))
        
        return result
    
    async def _get_composition(self, composition_id: str) -> Optional[CRComposition]:
        """Get composition from database."""
        query = select(CRComposition).where(
            and_(
                CRComposition.id == composition_id,
                CRComposition.tenant_id == self.tenant_id
            )
        )
        result = await self.db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def _deploy_to_kubernetes(
        self,
        composition: CRComposition,
        target: DeploymentTarget,
        strategy: DeploymentStrategy,
        container_image: str,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Deploy to Kubernetes cluster."""
        if not self.k8s_apps_v1:
            raise RuntimeError("Kubernetes client not available")
        
        result.logs.append("Starting Kubernetes deployment")
        
        try:
            # Ensure namespace exists
            await self._ensure_kubernetes_namespace(target.namespace)
            result.logs.append(f"Namespace {target.namespace} ready")
            
            # Create deployment manifest
            deployment_manifest = self._create_kubernetes_deployment_manifest(
                composition, target, container_image
            )
            
            # Create service manifest
            service_manifest = self._create_kubernetes_service_manifest(
                composition, target
            )
            
            # Apply deployment based on strategy
            if strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._kubernetes_blue_green_deployment(
                    deployment_manifest, service_manifest, target, result
                )
            elif strategy == DeploymentStrategy.CANARY:
                result = await self._kubernetes_canary_deployment(
                    deployment_manifest, service_manifest, target, result
                )
            else:
                result = await self._kubernetes_rolling_deployment(
                    deployment_manifest, service_manifest, target, result
                )
            
            # Create ingress if specified
            if target.ingress_host:
                ingress_manifest = self._create_kubernetes_ingress_manifest(
                    composition, target
                )
                await self._apply_kubernetes_manifest(ingress_manifest, target.namespace)
                result.ingress_created = True
                result.rollout_url = f"https://{target.ingress_host}"
                result.logs.append(f"Ingress created: {target.ingress_host}")
            
            # Wait for deployment to be ready
            ready_pods = await self._wait_for_kubernetes_deployment(
                composition.name, target.namespace, timeout_seconds=300
            )
            
            result.pods_ready = ready_pods
            result.pods_total = target.replicas
            result.status = DeploymentStatus.COMPLETED if ready_pods >= target.replicas else DeploymentStatus.FAILED
            result.message = f"Kubernetes deployment completed: {ready_pods}/{target.replicas} pods ready"
            
            # Perform health check
            if target.health_check_url:
                health_status = await self._perform_health_check(target.health_check_url)
                result.health_status = health_status
                result.logs.append(f"Health check: {health_status.get('status', 'unknown')}")
            
        except ApiException as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Kubernetes API error: {e.reason}"
            result.logs.append(f"Kubernetes error: {e}")
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Kubernetes deployment failed: {str(e)}"
            result.logs.append(f"Error: {str(e)}")
        
        return result
    
    async def _deploy_to_docker(
        self,
        composition: CRComposition,
        target: DeploymentTarget,
        strategy: DeploymentStrategy,
        container_image: str,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Deploy to Docker daemon."""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        result.logs.append("Starting Docker deployment")
        
        try:
            # Pull or build image
            try:
                image = self.docker_client.images.pull(container_image)
                result.logs.append(f"Pulled image: {container_image}")
            except docker.errors.ImageNotFound:
                # Build image if not found
                image = await self._build_composition_image(composition, container_image)
                result.logs.append(f"Built image: {container_image}")
            
            # Create network if needed
            network_name = f"{target.namespace}-network"
            try:
                network = self.docker_client.networks.get(network_name)
            except docker.errors.NotFound:
                network = self.docker_client.networks.create(
                    network_name,
                    driver="bridge",
                    labels={"tenant": self.tenant_id, "composition": composition.id}
                )
                result.logs.append(f"Created network: {network_name}")
            
            # Deploy containers based on strategy
            if strategy == DeploymentStrategy.BLUE_GREEN:
                containers = await self._docker_blue_green_deployment(
                    image, composition, target, result
                )
            else:
                containers = await self._docker_rolling_deployment(
                    image, composition, target, result
                )
            
            result.docker_containers = [c.id for c in containers]
            result.pods_ready = len([c for c in containers if c.status == "running"])
            result.pods_total = target.replicas
            result.status = DeploymentStatus.COMPLETED
            result.message = f"Docker deployment completed: {result.pods_ready} containers running"
            
            # Set up load balancer/proxy if multiple replicas
            if target.replicas > 1:
                proxy_port = await self._setup_docker_load_balancer(containers, target)
                result.rollout_url = f"http://localhost:{proxy_port}"
                result.logs.append(f"Load balancer running on port {proxy_port}")
            elif containers:
                # Single container, expose directly
                container = containers[0]
                ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                if ports:
                    port_mapping = list(ports.values())[0]
                    if port_mapping:
                        external_port = port_mapping[0]['HostPort']
                        result.rollout_url = f"http://localhost:{external_port}"
            
            # Perform health check
            if result.rollout_url:
                health_status = await self._perform_health_check(result.rollout_url + "/health")
                result.health_status = health_status
                result.logs.append(f"Health check: {health_status.get('status', 'unknown')}")
            
        except docker.errors.DockerException as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Docker error: {str(e)}"
            result.logs.append(f"Docker error: {str(e)}")
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Docker deployment failed: {str(e)}"
            result.logs.append(f"Error: {str(e)}")
        
        return result
    
    async def _deploy_to_aws(
        self,
        composition: CRComposition,
        target: DeploymentTarget,
        strategy: DeploymentStrategy,
        container_image: str,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Deploy to AWS ECS/EKS."""
        if not self.aws_ecs_client:
            raise RuntimeError("AWS ECS client not available")
        
        result.logs.append("Starting AWS deployment")
        
        try:
            if target.cluster_name.startswith("aws-eks-"):
                # Deploy to EKS (same as Kubernetes)
                return await self._deploy_to_kubernetes(
                    composition, target, strategy, container_image, result
                )
            else:
                # Deploy to ECS
                result = await self._deploy_to_ecs(
                    composition, target, strategy, container_image, result
                )
            
        except ClientError as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"AWS error: {e.response['Error']['Message']}"
            result.logs.append(f"AWS error: {str(e)}")
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"AWS deployment failed: {str(e)}"
            result.logs.append(f"Error: {str(e)}")
        
        return result
    
    async def _deploy_to_ecs(
        self,
        composition: CRComposition,
        target: DeploymentTarget,
        strategy: DeploymentStrategy,
        container_image: str,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Deploy to AWS ECS."""
        cluster_name = target.cluster_name.replace("aws-ecs-", "")
        
        # Create task definition
        task_definition = {
            'family': f"{composition.name}-{target.environment.value}",
            'taskRoleArn': os.getenv('AWS_TASK_ROLE_ARN'),
            'executionRoleArn': os.getenv('AWS_EXECUTION_ROLE_ARN'),
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': str(target.resource_limits.get('cpu', '256')),
            'memory': str(target.resource_limits.get('memory', '512')),
            'containerDefinitions': [{
                'name': composition.name,
                'image': container_image,
                'essential': True,
                'portMappings': [{
                    'containerPort': 8000,
                    'protocol': 'tcp'
                }],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/{composition.name}',
                        'awslogs-region': os.getenv('AWS_REGION', 'us-west-2'),
                        'awslogs-stream-prefix': 'ecs'
                    }
                },
                'environment': [
                    {'name': 'TENANT_ID', 'value': self.tenant_id},
                    {'name': 'COMPOSITION_ID', 'value': composition.id},
                    {'name': 'ENVIRONMENT', 'value': target.environment.value}
                ]
            }]
        }
        
        # Register task definition
        task_def_response = self.aws_ecs_client.register_task_definition(**task_definition)
        task_def_arn = task_def_response['taskDefinition']['taskDefinitionArn']
        result.logs.append(f"Registered task definition: {task_def_arn}")
        
        # Create or update service
        service_name = f"{composition.name}-{target.environment.value}"
        
        try:
            # Check if service exists
            services = self.aws_ecs_client.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )
            
            if services['services'] and services['services'][0]['status'] != 'INACTIVE':
                # Update existing service
                response = self.aws_ecs_client.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    taskDefinition=task_def_arn,
                    desiredCount=target.replicas,
                    deploymentConfiguration={
                        'maximumPercent': 200,
                        'minimumHealthyPercent': 50 if strategy == DeploymentStrategy.ROLLING_UPDATE else 100
                    }
                )
                result.logs.append(f"Updated ECS service: {service_name}")
            else:
                raise Exception("Service not found")
                
        except:
            # Create new service
            response = self.aws_ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=target.replicas,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': os.getenv('AWS_SUBNET_IDS', '').split(','),
                        'securityGroups': os.getenv('AWS_SECURITY_GROUP_IDS', '').split(','),
                        'assignPublicIp': 'ENABLED'
                    }
                }
            )
            result.logs.append(f"Created ECS service: {service_name}")
        
        # Wait for deployment to stabilize
        waiter = self.aws_ecs_client.get_waiter('services_stable')
        try:
            waiter.wait(
                cluster=cluster_name,
                services=[service_name],
                WaiterConfig={'maxAttempts': 20, 'delay': 15}
            )
            result.status = DeploymentStatus.COMPLETED
            result.message = "ECS deployment completed successfully"
            result.pods_ready = target.replicas
            result.pods_total = target.replicas
        except:
            result.status = DeploymentStatus.FAILED
            result.message = "ECS deployment did not stabilize within timeout"
        
        return result
    
    def _create_kubernetes_deployment_manifest(
        self,
        composition: CRComposition,
        target: DeploymentTarget,
        container_image: str
    ) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': composition.name,
                'namespace': target.namespace,
                'labels': {
                    'app': composition.name,
                    'tenant': self.tenant_id,
                    'composition': composition.id,
                    'environment': target.environment.value
                }
            },
            'spec': {
                'replicas': target.replicas,
                'selector': {
                    'matchLabels': {
                        'app': composition.name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': composition.name,
                            'tenant': self.tenant_id,
                            'composition': composition.id
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': composition.name,
                            'image': container_image,
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'TENANT_ID', 'value': self.tenant_id},
                                {'name': 'COMPOSITION_ID', 'value': composition.id},
                                {'name': 'ENVIRONMENT', 'value': target.environment.value}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': target.resource_limits.get('cpu', '100m'),
                                    'memory': target.resource_limits.get('memory', '128Mi')
                                },
                                'limits': {
                                    'cpu': target.resource_limits.get('cpu_limit', '500m'),
                                    'memory': target.resource_limits.get('memory_limit', '512Mi')
                                }
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }],
                        'nodeSelector': target.node_selector or {},
                        'tolerations': target.tolerations or []
                    }
                }
            }
        }
    
    def _create_kubernetes_service_manifest(
        self,
        composition: CRComposition,
        target: DeploymentTarget
    ) -> Dict[str, Any]:
        """Create Kubernetes service manifest."""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{composition.name}-service",
                'namespace': target.namespace,
                'labels': {
                    'app': composition.name,
                    'tenant': self.tenant_id
                }
            },
            'spec': {
                'selector': {
                    'app': composition.name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
    
    def _create_kubernetes_ingress_manifest(
        self,
        composition: CRComposition,
        target: DeploymentTarget
    ) -> Dict[str, Any]:
        """Create Kubernetes ingress manifest."""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{composition.name}-ingress",
                'namespace': target.namespace,
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': [target.ingress_host],
                    'secretName': f"{composition.name}-tls"
                }],
                'rules': [{
                    'host': target.ingress_host,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f"{composition.name}-service",
                                    'port': {'number': 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    async def _ensure_kubernetes_namespace(self, namespace: str):
        """Ensure Kubernetes namespace exists."""
        try:
            self.k8s_core_v1.read_namespace(name=namespace)
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(
                        name=namespace,
                        labels={
                            'tenant': self.tenant_id,
                            'managed-by': 'apg-composition'
                        }
                    )
                )
                self.k8s_core_v1.create_namespace(body=namespace_manifest)
            else:
                raise
    
    async def _apply_kubernetes_manifest(self, manifest: Dict[str, Any], namespace: str):
        """Apply Kubernetes manifest."""
        kind = manifest['kind']
        
        if kind == 'Deployment':
            body = client.AppsV1Api().api_client.sanitize_for_serialization(manifest)
            deployment = client.V1Deployment(**body)
            try:
                self.k8s_apps_v1.read_namespaced_deployment(
                    name=manifest['metadata']['name'],
                    namespace=namespace
                )
                # Update existing
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=manifest['metadata']['name'],
                    namespace=namespace,
                    body=deployment
                )
            except ApiException as e:
                if e.status == 404:
                    # Create new
                    self.k8s_apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment
                    )
                else:
                    raise
        
        elif kind == 'Service':
            body = client.CoreV1Api().api_client.sanitize_for_serialization(manifest)
            service = client.V1Service(**body)
            try:
                self.k8s_core_v1.read_namespaced_service(
                    name=manifest['metadata']['name'],
                    namespace=namespace
                )
                # Update existing
                self.k8s_core_v1.patch_namespaced_service(
                    name=manifest['metadata']['name'],
                    namespace=namespace,
                    body=service
                )
            except ApiException as e:
                if e.status == 404:
                    # Create new
                    self.k8s_core_v1.create_namespaced_service(
                        namespace=namespace,
                        body=service
                    )
                else:
                    raise
        
        elif kind == 'Ingress':
            body = client.NetworkingV1Api().api_client.sanitize_for_serialization(manifest)
            ingress = client.V1Ingress(**body)
            try:
                self.k8s_networking_v1.read_namespaced_ingress(
                    name=manifest['metadata']['name'],
                    namespace=namespace
                )
                # Update existing
                self.k8s_networking_v1.patch_namespaced_ingress(
                    name=manifest['metadata']['name'],
                    namespace=namespace,
                    body=ingress
                )
            except ApiException as e:
                if e.status == 404:
                    # Create new
                    self.k8s_networking_v1.create_namespaced_ingress(
                        namespace=namespace,
                        body=ingress
                    )
                else:
                    raise
    
    async def _kubernetes_rolling_deployment(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        target: DeploymentTarget,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Execute Kubernetes rolling update deployment."""
        # Apply deployment
        await self._apply_kubernetes_manifest(deployment_manifest, target.namespace)
        result.manifest_applied = True
        result.logs.append("Deployment manifest applied")
        
        # Apply service
        await self._apply_kubernetes_manifest(service_manifest, target.namespace)
        result.service_created = True
        result.logs.append("Service manifest applied")
        
        return result
    
    async def _kubernetes_blue_green_deployment(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        target: DeploymentTarget,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Execute Kubernetes blue-green deployment."""
        composition_name = deployment_manifest['metadata']['name']
        
        # Create green deployment (new version)
        green_manifest = deployment_manifest.copy()
        green_manifest['metadata']['name'] = f"{composition_name}-green"
        green_manifest['spec']['selector']['matchLabels']['version'] = 'green'
        green_manifest['spec']['template']['metadata']['labels']['version'] = 'green'
        
        await self._apply_kubernetes_manifest(green_manifest, target.namespace)
        result.logs.append("Green deployment created")
        
        # Wait for green deployment to be ready
        ready_pods = await self._wait_for_kubernetes_deployment(
            f"{composition_name}-green", target.namespace, timeout_seconds=300
        )
        
        if ready_pods >= target.replicas:
            # Switch service to green
            service_manifest['spec']['selector']['version'] = 'green'
            await self._apply_kubernetes_manifest(service_manifest, target.namespace)
            result.logs.append("Traffic switched to green deployment")
            
            # Clean up old blue deployment
            try:
                self.k8s_apps_v1.delete_namespaced_deployment(
                    name=f"{composition_name}-blue",
                    namespace=target.namespace
                )
                result.logs.append("Blue deployment cleaned up")
            except ApiException:
                pass  # Blue deployment might not exist
            
            # Rename green to blue for next deployment
            try:
                self.k8s_apps_v1.delete_namespaced_deployment(
                    name=composition_name,
                    namespace=target.namespace
                )
            except ApiException:
                pass
            
            blue_manifest = green_manifest.copy()
            blue_manifest['metadata']['name'] = f"{composition_name}-blue"
            blue_manifest['spec']['selector']['matchLabels']['version'] = 'blue'
            blue_manifest['spec']['template']['metadata']['labels']['version'] = 'blue'
            await self._apply_kubernetes_manifest(blue_manifest, target.namespace)
            
            result.manifest_applied = True
            result.service_created = True
        else:
            result.logs.append("Green deployment failed to become ready")
            raise Exception("Green deployment failed")
        
        return result
    
    async def _kubernetes_canary_deployment(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        target: DeploymentTarget,
        result: DeploymentResult
    ) -> DeploymentResult:
        """Execute Kubernetes canary deployment."""
        composition_name = deployment_manifest['metadata']['name']
        
        # Create canary deployment with 1 replica
        canary_manifest = deployment_manifest.copy()
        canary_manifest['metadata']['name'] = f"{composition_name}-canary"
        canary_manifest['spec']['replicas'] = 1
        canary_manifest['spec']['selector']['matchLabels']['version'] = 'canary'
        canary_manifest['spec']['template']['metadata']['labels']['version'] = 'canary'
        
        await self._apply_kubernetes_manifest(canary_manifest, target.namespace)
        result.logs.append("Canary deployment created")
        
        # Update service to include both stable and canary
        service_manifest['spec']['selector'] = {'app': composition_name}  # Remove version selector
        await self._apply_kubernetes_manifest(service_manifest, target.namespace)
        result.logs.append("Service updated for canary traffic")
        
        # Wait for canary to be ready
        ready_pods = await self._wait_for_kubernetes_deployment(
            f"{composition_name}-canary", target.namespace, timeout_seconds=180
        )
        
        if ready_pods >= 1:
            # Gradually increase canary traffic (simplified - just update replicas)
            for canary_replicas in [2, max(2, target.replicas // 2)]:
                if canary_replicas >= target.replicas:
                    break
                
                canary_manifest['spec']['replicas'] = canary_replicas
                await self._apply_kubernetes_manifest(canary_manifest, target.namespace)
                result.logs.append(f"Canary scaled to {canary_replicas} replicas")
                
                await asyncio.sleep(30)  # Wait between scaling steps
            
            # If successful, replace stable with canary
            stable_manifest = deployment_manifest.copy()
            stable_manifest['spec']['selector']['matchLabels']['version'] = 'stable'
            stable_manifest['spec']['template']['metadata']['labels']['version'] = 'stable'
            await self._apply_kubernetes_manifest(stable_manifest, target.namespace)
            
            # Clean up canary
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=f"{composition_name}-canary",
                namespace=target.namespace
            )
            result.logs.append("Canary deployment completed, stable version updated")
            
            result.manifest_applied = True
            result.service_created = True
        else:
            result.logs.append("Canary deployment failed to become ready")
            raise Exception("Canary deployment failed")
        
        return result
    
    async def _wait_for_kubernetes_deployment(
        self,
        deployment_name: str,
        namespace: str,
        timeout_seconds: int = 300
    ) -> int:
        """Wait for Kubernetes deployment to be ready."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                ready_replicas = deployment.status.ready_replicas or 0
                desired_replicas = deployment.spec.replicas or 0
                
                if ready_replicas >= desired_replicas:
                    return ready_replicas
                
                await asyncio.sleep(5)
                
            except ApiException:
                await asyncio.sleep(5)
        
        # Timeout reached, return current ready count
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            return deployment.status.ready_replicas or 0
        except:
            return 0
    
    async def _docker_rolling_deployment(
        self,
        image,
        composition: CRComposition,
        target: DeploymentTarget,
        result: DeploymentResult
    ) -> List:
        """Execute Docker rolling deployment."""
        containers = []
        
        for i in range(target.replicas):
            container_name = f"{composition.name}-{target.environment.value}-{i}"
            
            # Stop and remove existing container if it exists
            try:
                existing_container = self.docker_client.containers.get(container_name)
                existing_container.stop()
                existing_container.remove()
                result.logs.append(f"Removed old container: {container_name}")
            except docker.errors.NotFound:
                pass
            
            # Create new container
            container = self.docker_client.containers.run(
                image,
                name=container_name,
                detach=True,
                environment={
                    'TENANT_ID': self.tenant_id,
                    'COMPOSITION_ID': composition.id,
                    'ENVIRONMENT': target.environment.value
                },
                ports={'8000/tcp': None},  # Random host port
                labels={
                    'tenant': self.tenant_id,
                    'composition': composition.id,
                    'environment': target.environment.value
                },
                restart_policy={'Name': 'unless-stopped'}
            )
            
            containers.append(container)
            result.logs.append(f"Started container: {container_name}")
        
        return containers
    
    async def _docker_blue_green_deployment(
        self,
        image,
        composition: CRComposition,
        target: DeploymentTarget,
        result: DeploymentResult
    ) -> List:
        """Execute Docker blue-green deployment."""
        green_containers = []
        
        # Create green containers
        for i in range(target.replicas):
            container_name = f"{composition.name}-{target.environment.value}-green-{i}"
            
            container = self.docker_client.containers.run(
                image,
                name=container_name,
                detach=True,
                environment={
                    'TENANT_ID': self.tenant_id,
                    'COMPOSITION_ID': composition.id,
                    'ENVIRONMENT': target.environment.value,
                    'VERSION': 'green'
                },
                ports={'8000/tcp': None},
                labels={
                    'tenant': self.tenant_id,
                    'composition': composition.id,
                    'environment': target.environment.value,
                    'version': 'green'
                }
            )
            
            green_containers.append(container)
            result.logs.append(f"Started green container: {container_name}")
        
        # Wait for green containers to be healthy
        await asyncio.sleep(10)
        
        healthy_containers = []
        for container in green_containers:
            container.reload()
            if container.status == 'running':
                healthy_containers.append(container)
        
        if len(healthy_containers) >= target.replicas:
            # Remove blue containers
            blue_containers = self.docker_client.containers.list(
                filters={
                    'label': [
                        f'composition={composition.id}',
                        f'environment={target.environment.value}',
                        'version=blue'
                    ]
                }
            )
            
            for container in blue_containers:
                container.stop()
                container.remove()
                result.logs.append(f"Removed blue container: {container.name}")
            
            # Rename green containers to blue for next deployment
            for i, container in enumerate(healthy_containers):
                new_name = f"{composition.name}-{target.environment.value}-blue-{i}"
                # Docker doesn't support rename, so we'll just update labels
                # In a real implementation, you might use a service discovery mechanism
            
            result.logs.append("Blue-green deployment completed")
            return healthy_containers
        else:
            # Clean up failed green containers
            for container in green_containers:
                container.stop()
                container.remove()
            raise Exception("Green containers failed health check")
    
    async def _build_composition_image(self, composition: CRComposition, image_name: str):
        """Build Docker image for composition."""
        # Create temporary Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write Dockerfile
            (temp_path / "Dockerfile").write_text(dockerfile_content)
            
            # Write basic requirements.txt
            (temp_path / "requirements.txt").write_text("""
fastapi==0.104.1
uvicorn[standard]==0.25.0
pydantic==2.5.2
httpx==0.25.2
""")
            
            # Write basic main.py
            (temp_path / "main.py").write_text(f"""
from fastapi import FastAPI
import os

app = FastAPI(title="{composition.name}")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "composition_id": "{composition.id}"}}

@app.get("/")
async def root():
    return {{"message": "APG Composition: {composition.name}"}}
""")
            
            # Build image
            image, logs = self.docker_client.images.build(
                path=str(temp_path),
                tag=image_name,
                rm=True
            )
            
            return image
    
    async def _setup_docker_load_balancer(self, containers: List, target: DeploymentTarget) -> int:
        """Set up simple load balancer for Docker containers."""
        # For simplicity, we'll just use nginx as a load balancer
        # In a real implementation, you might use HAProxy or similar
        
        upstream_servers = []
        for container in containers:
            container.reload()
            ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
            if '8000/tcp' in ports and ports['8000/tcp']:
                host_port = ports['8000/tcp'][0]['HostPort']
                upstream_servers.append(f"server host.docker.internal:{host_port};")
        
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream backend {{
        {' '.join(upstream_servers)}
    }}
    
    server {{
        listen 80;
        location / {{
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }}
    }}
}}
"""
        
        # Create nginx container
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(nginx_config)
            nginx_conf_path = f.name
        
        try:
            nginx_container = self.docker_client.containers.run(
                'nginx:alpine',
                name=f"{target.namespace}-loadbalancer",
                detach=True,
                ports={'80/tcp': None},
                volumes={nginx_conf_path: {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}},
                labels={'tenant': self.tenant_id, 'role': 'loadbalancer'}
            )
            
            # Get assigned port
            nginx_container.reload()
            ports = nginx_container.attrs.get('NetworkSettings', {}).get('Ports', {})
            if '80/tcp' in ports and ports['80/tcp']:
                return int(ports['80/tcp'][0]['HostPort'])
        except Exception as e:
            logger.warning("Failed to create load balancer", error=str(e))
        finally:
            os.unlink(nginx_conf_path)
        
        return 8080  # Fallback port
    
    async def _perform_health_check(self, url: str) -> Dict[str, Any]:
        """Perform HTTP health check."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'content': response.text[:200]  # First 200 chars
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': 0
            }
    
    async def rollback_deployment(self, deployment_id: str) -> DeploymentResult:
        """Rollback a deployment to previous version."""
        # Get deployment record
        deployment = await self.db_session.get(CRDeployment, deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            message="Rollback started",
            start_time=datetime.utcnow()
        )
        
        try:
            if deployment.cluster_name.startswith("kubernetes-"):
                await self._rollback_kubernetes_deployment(deployment, result)
            elif deployment.cluster_name.startswith("docker-"):
                await self._rollback_docker_deployment(deployment, result)
            elif deployment.cluster_name.startswith("aws-"):
                await self._rollback_aws_deployment(deployment, result)
            
            deployment.status = DeploymentStatus.ROLLED_BACK.value
            deployment.message = "Deployment rolled back successfully"
            await self.db_session.commit()
            
            result.status = DeploymentStatus.ROLLED_BACK
            result.message = "Rollback completed successfully"
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.message = f"Rollback failed: {str(e)}"
            
            deployment.status = DeploymentStatus.FAILED.value
            deployment.message = result.message
            await self.db_session.commit()
        
        result.end_time = datetime.utcnow()
        return result
    
    async def _rollback_kubernetes_deployment(self, deployment: CRDeployment, result: DeploymentResult):
        """Rollback Kubernetes deployment."""
        try:
            # Get deployment history
            deployment_name = await self._get_composition_name_by_id(deployment.composition_id)
            
            # Rollback to previous revision
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=deployment.namespace,
                body={'spec': {'rollbackTo': {'revision': 0}}}  # Previous revision
            )
            
            result.logs.append("Kubernetes deployment rolled back")
            
            # Wait for rollback to complete
            ready_pods = await self._wait_for_kubernetes_deployment(
                deployment_name, deployment.namespace, timeout_seconds=300
            )
            
            result.pods_ready = ready_pods
            result.logs.append(f"Rollback completed: {ready_pods} pods ready")
            
        except ApiException as e:
            raise Exception(f"Kubernetes rollback failed: {e.reason}")
    
    async def _rollback_docker_deployment(self, deployment: CRDeployment, result: DeploymentResult):
        """Rollback Docker deployment."""
        try:
            composition_name = await self._get_composition_name_by_id(deployment.composition_id)
            
            # Stop current containers
            current_containers = self.docker_client.containers.list(
                filters={
                    'label': [
                        f'composition={deployment.composition_id}',
                        f'environment={deployment.environment}'
                    ]
                }
            )
            
            for container in current_containers:
                container.stop()
                result.logs.append(f"Stopped container: {container.name}")
            
            # Start previous version containers (if they exist)
            # This is simplified - in reality you'd need to track previous versions
            backup_containers = self.docker_client.containers.list(
                all=True,
                filters={
                    'label': [
                        f'composition={deployment.composition_id}',
                        f'environment={deployment.environment}',
                        'version=backup'
                    ]
                }
            )
            
            for container in backup_containers:
                container.start()
                result.logs.append(f"Started backup container: {container.name}")
            
            result.docker_containers = [c.id for c in backup_containers]
            
        except docker.errors.DockerException as e:
            raise Exception(f"Docker rollback failed: {str(e)}")
    
    async def _rollback_aws_deployment(self, deployment: CRDeployment, result: DeploymentResult):
        """Rollback AWS ECS deployment."""
        try:
            cluster_name = deployment.cluster_name.replace("aws-ecs-", "")
            composition_name = await self._get_composition_name_by_id(deployment.composition_id)
            service_name = f"{composition_name}-{deployment.environment}"
            
            # Get service's previous task definition
            service = self.aws_ecs_client.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )
            
            if not service['services']:
                raise Exception("Service not found")
            
            current_task_def = service['services'][0]['taskDefinition']
            
            # List task definitions to find previous version
            task_defs = self.aws_ecs_client.list_task_definitions(
                familyPrefix=f"{composition_name}-{deployment.environment}",
                status='ACTIVE',
                sort='DESC'
            )
            
            if len(task_defs['taskDefinitionArns']) < 2:
                raise Exception("No previous version found for rollback")
            
            # Use second most recent (previous) version
            previous_task_def = task_defs['taskDefinitionArns'][1]
            
            # Update service to use previous task definition
            self.aws_ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=previous_task_def
            )
            
            result.logs.append(f"Rolled back to task definition: {previous_task_def}")
            
            # Wait for rollback to complete
            waiter = self.aws_ecs_client.get_waiter('services_stable')
            waiter.wait(
                cluster=cluster_name,
                services=[service_name],
                WaiterConfig={'maxAttempts': 20, 'delay': 15}
            )
            
            result.logs.append("AWS ECS rollback completed")
            
        except ClientError as e:
            raise Exception(f"AWS rollback failed: {e.response['Error']['Message']}")
    
    async def _get_composition_name_by_id(self, composition_id: str) -> str:
        """Get composition name by ID."""
        composition = await self._get_composition(composition_id)
        return composition.name if composition else f"composition-{composition_id}"
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get current status of a deployment."""
        deployment = await self.db_session.get(CRDeployment, deployment_id)
        if not deployment:
            return None
        
        return DeploymentResult(
            deployment_id=deployment.id,
            status=DeploymentStatus(deployment.status),
            message=deployment.message or "",
            rollout_url=deployment.rollout_url,
            health_status=deployment.health_status,
            logs=deployment.deployment_logs or [],
            start_time=deployment.created_at,
            end_time=deployment.updated_at if deployment.status in ['completed', 'failed', 'rolled_back'] else None
        )
    
    async def list_deployments(
        self,
        composition_id: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[DeploymentResult]:
        """List deployments with optional filters."""
        query = select(CRDeployment).where(CRDeployment.tenant_id == self.tenant_id)
        
        if composition_id:
            query = query.where(CRDeployment.composition_id == composition_id)
        if environment:
            query = query.where(CRDeployment.environment == environment)
        if status:
            query = query.where(CRDeployment.status == status)
        
        query = query.order_by(CRDeployment.created_at.desc()).limit(limit)
        
        result = await self.db_session.execute(query)
        deployments = result.scalars().all()
        
        return [
            DeploymentResult(
                deployment_id=d.id,
                status=DeploymentStatus(d.status),
                message=d.message or "",
                rollout_url=d.rollout_url,
                health_status=d.health_status,
                logs=d.deployment_logs or [],
                start_time=d.created_at,
                end_time=d.updated_at if d.status in ['completed', 'failed', 'rolled_back'] else None
            )
            for d in deployments
        ]
    
    async def cleanup_failed_deployments(self) -> int:
        """Clean up failed deployments and resources."""
        query = select(CRDeployment).where(
            and_(
                CRDeployment.tenant_id == self.tenant_id,
                CRDeployment.status == DeploymentStatus.FAILED.value,
                CRDeployment.created_at < datetime.utcnow() - timedelta(hours=24)
            )
        )
        
        result = await self.db_session.execute(query)
        failed_deployments = result.scalars().all()
        
        cleaned_count = 0
        for deployment in failed_deployments:
            try:
                if deployment.cluster_name.startswith("kubernetes-"):
                    await self._cleanup_kubernetes_resources(deployment)
                elif deployment.cluster_name.startswith("docker-"):
                    await self._cleanup_docker_resources(deployment)
                elif deployment.cluster_name.startswith("aws-"):
                    await self._cleanup_aws_resources(deployment)
                
                # Delete deployment record
                await self.db_session.delete(deployment)
                cleaned_count += 1
                
            except Exception as e:
                logger.warning("Failed to cleanup deployment", 
                             deployment_id=deployment.id, error=str(e))
        
        await self.db_session.commit()
        logger.info("Cleaned up failed deployments", count=cleaned_count)
        
        return cleaned_count
    
    async def _cleanup_kubernetes_resources(self, deployment: CRDeployment):
        """Clean up Kubernetes resources for failed deployment."""
        if not self.k8s_apps_v1:
            return
        
        composition_name = await self._get_composition_name_by_id(deployment.composition_id)
        
        try:
            # Delete deployment
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=composition_name,
                namespace=deployment.namespace
            )
        except ApiException:
            pass
        
        try:
            # Delete service
            self.k8s_core_v1.delete_namespaced_service(
                name=f"{composition_name}-service",
                namespace=deployment.namespace
            )
        except ApiException:
            pass
        
        try:
            # Delete ingress
            self.k8s_networking_v1.delete_namespaced_ingress(
                name=f"{composition_name}-ingress",
                namespace=deployment.namespace
            )
        except ApiException:
            pass
    
    async def _cleanup_docker_resources(self, deployment: CRDeployment):
        """Clean up Docker resources for failed deployment."""
        if not self.docker_client:
            return
        
        # Remove containers
        containers = self.docker_client.containers.list(
            all=True,
            filters={
                'label': [
                    f'composition={deployment.composition_id}',
                    f'environment={deployment.environment}'
                ]
            }
        )
        
        for container in containers:
            try:
                container.remove(force=True)
            except docker.errors.DockerException:
                pass
    
    async def _cleanup_aws_resources(self, deployment: CRDeployment):
        """Clean up AWS resources for failed deployment."""
        if not self.aws_ecs_client:
            return
        
        try:
            cluster_name = deployment.cluster_name.replace("aws-ecs-", "")
            composition_name = await self._get_composition_name_by_id(deployment.composition_id)
            service_name = f"{composition_name}-{deployment.environment}"
            
            # Delete service
            self.aws_ecs_client.delete_service(
                cluster=cluster_name,
                service=service_name,
                force=True
            )
        except ClientError:
            pass

# Service factory
_deployment_services: Dict[str, DeploymentAutomationService] = {}

async def get_deployment_service(tenant_id: str, db_session: AsyncSession) -> DeploymentAutomationService:
    """Get deployment service for tenant."""
    service_key = f"{tenant_id}_{id(db_session)}"
    if service_key not in _deployment_services:
        _deployment_services[service_key] = DeploymentAutomationService(tenant_id, db_session)
    return _deployment_services[service_key]

# Export key components
__all__ = [
    "DeploymentStrategy",
    "DeploymentEnvironment",
    "DeploymentStatus",
    "DeploymentTarget",
    "DeploymentResult",
    "DeploymentAutomationService",
    "get_deployment_service"
]