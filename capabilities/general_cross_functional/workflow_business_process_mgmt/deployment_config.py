"""
APG Workflow & Business Process Management - Deployment Configuration

Comprehensive deployment configuration management for Kubernetes, Docker,
and infrastructure as code with environment-specific settings.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import yaml
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path
import base64

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Deployment Configuration Core Classes
# =============================================================================

class EnvironmentType(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ServiceType(str, Enum):
    """Service component types."""
    API = "api"
    WORKER = "worker"
    SCHEDULER = "scheduler"
    MONITOR = "monitor"
    MIGRATION = "migration"


class DatabaseType(str, Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    MONGODB = "mongodb"


@dataclass
class ResourceLimits:
    """Container resource limits."""
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    storage_request: str = "1Gi"
    storage_limit: str = "10Gi"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    database: str = "wbpm"
    username: str = "wbpm_user"
    password_secret: str = "wbpm-db-password"
    ssl_mode: str = "require"
    connection_pool_size: int = 20
    max_overflow: int = 30
    connection_timeout: int = 30
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM


@dataclass
class ServiceConfig:
    """Individual service configuration."""
    name: str = ""
    service_type: ServiceType = ServiceType.API
    image: str = ""
    tag: str = "latest"
    port: int = 8000
    replicas: int = 1
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    startup_probe_delay: int = 30
    readiness_probe_delay: int = 10
    liveness_probe_delay: int = 30
    auto_scaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70


@dataclass
class NetworkConfig:
    """Network configuration."""
    ingress_enabled: bool = True
    ingress_class: str = "nginx"
    tls_enabled: bool = True
    tls_secret_name: str = "wbpm-tls"
    domain: str = "wbpm.datacraft.co.ke"
    cors_origins: List[str] = field(default_factory=lambda: ["https://app.datacraft.co.ke"])
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 1000


@dataclass
class SecurityConfig:
    """Security configuration."""
    pod_security_context_enabled: bool = True
    run_as_non_root: bool = True
    run_as_user: int = 1000
    run_as_group: int = 1000
    fs_group: int = 1000
    network_policies_enabled: bool = True
    rbac_enabled: bool = True
    service_account_name: str = "wbpm-service-account"
    image_pull_policy: str = "Always"
    security_context_capabilities: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    jaeger_enabled: bool = True
    log_level: str = "INFO"
    structured_logging: bool = True
    metrics_path: str = "/metrics"
    health_check_interval: int = 30
    alert_manager_enabled: bool = True


@dataclass
class BackupConfig:
    """Backup and disaster recovery configuration."""
    enabled: bool = True
    storage_class: str = "standard"
    retention_days: int = 30
    backup_schedule: str = "0 1 * * *"  # Daily at 1 AM
    backup_location: str = "s3://wbpm-backups"
    encryption_enabled: bool = True
    compression_enabled: bool = True


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    namespace: str = "wbpm-dev"
    cluster_name: str = "datacraft-dev"
    region: str = "us-west-2"
    services: List[ServiceConfig] = field(default_factory=list)
    databases: List[DatabaseConfig] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    secrets: Dict[str, str] = field(default_factory=dict)
    config_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)


# =============================================================================
# Kubernetes Manifest Generator
# =============================================================================

class KubernetesManifestGenerator:
    """Generate Kubernetes deployment manifests."""
    
    def __init__(self, environment_config: EnvironmentConfig):
        self.config = environment_config
        
    async def generate_all_manifests(self, output_dir: Path) -> Dict[str, List[str]]:
        """Generate all Kubernetes manifests."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            generated_files = {
                "deployments": [],
                "services": [],
                "configmaps": [],
                "secrets": [],
                "ingress": [],
                "hpa": [],
                "rbac": [],
                "monitoring": []
            }
            
            # Generate namespace
            namespace_file = await self._generate_namespace(output_dir)
            generated_files["deployments"].append(namespace_file)
            
            # Generate secrets
            secrets_file = await self._generate_secrets(output_dir)
            generated_files["secrets"].append(secrets_file)
            
            # Generate config maps
            configmaps_file = await self._generate_configmaps(output_dir)
            generated_files["configmaps"].append(configmaps_file)
            
            # Generate RBAC
            rbac_files = await self._generate_rbac(output_dir)
            generated_files["rbac"].extend(rbac_files)
            
            # Generate database manifests
            db_files = await self._generate_database_manifests(output_dir)
            generated_files["deployments"].extend(db_files)
            
            # Generate service manifests
            for service in self.config.services:
                service_files = await self._generate_service_manifests(service, output_dir)
                generated_files["deployments"].append(service_files["deployment"])
                generated_files["services"].append(service_files["service"])
                
                if service.auto_scaling_enabled:
                    generated_files["hpa"].append(service_files["hpa"])
            
            # Generate ingress
            if self.config.network.ingress_enabled:
                ingress_file = await self._generate_ingress(output_dir)
                generated_files["ingress"].append(ingress_file)
            
            # Generate monitoring manifests
            monitoring_files = await self._generate_monitoring_manifests(output_dir)
            generated_files["monitoring"].extend(monitoring_files)
            
            # Generate network policies
            if self.config.security.network_policies_enabled:
                netpol_file = await self._generate_network_policies(output_dir)
                generated_files["rbac"].append(netpol_file)
            
            logger.info(f"Generated Kubernetes manifests for {self.config.environment.value}")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating Kubernetes manifests: {e}")
            raise
    
    async def _generate_namespace(self, output_dir: Path) -> str:
        """Generate namespace manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "wbpm",
                    "app.kubernetes.io/part-of": "apg-platform",
                    "environment": self.config.environment.value
                }
            }
        }
        
        file_path = output_dir / "01-namespace.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_secrets(self, output_dir: Path) -> str:
        """Generate secrets manifest."""
        secrets = []
        
        # Database secrets
        for db in self.config.databases:
            secret = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": db.password_secret,
                    "namespace": self.config.namespace
                },
                "type": "Opaque",
                "stringData": {
                    "password": "REPLACE_WITH_ACTUAL_PASSWORD",
                    "username": db.username,
                    "database": db.database
                }
            }
            secrets.append(secret)
        
        # Application secrets
        app_secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "wbpm-app-secrets",
                "namespace": self.config.namespace
            },
            "type": "Opaque",
            "stringData": {
                "jwt_secret": "REPLACE_WITH_JWT_SECRET",
                "api_key": "REPLACE_WITH_API_KEY",
                "webhook_secret": "REPLACE_WITH_WEBHOOK_SECRET"
            }
        }
        secrets.append(app_secret)
        
        # TLS secrets (placeholder)
        if self.config.network.tls_enabled:
            tls_secret = {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": self.config.network.tls_secret_name,
                    "namespace": self.config.namespace
                },
                "type": "kubernetes.io/tls",
                "stringData": {
                    "tls.crt": "REPLACE_WITH_TLS_CERTIFICATE",
                    "tls.key": "REPLACE_WITH_TLS_PRIVATE_KEY"
                }
            }
            secrets.append(tls_secret)
        
        file_path = output_dir / "02-secrets.yaml"
        with open(file_path, 'w') as f:
            yaml.dump_all(secrets, f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_configmaps(self, output_dir: Path) -> str:
        """Generate config maps manifest."""
        configmaps = []
        
        # Application configuration
        app_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "wbpm-app-config",
                "namespace": self.config.namespace
            },
            "data": {
                "environment": self.config.environment.value,
                "log_level": self.config.monitoring.log_level,
                "metrics_enabled": str(self.config.monitoring.prometheus_enabled).lower(),
                "cors_origins": ",".join(self.config.network.cors_origins),
                "rate_limit_rpm": str(self.config.network.rate_limit_requests_per_minute)
            }
        }
        configmaps.append(app_config)
        
        # Database configuration
        db_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "wbpm-db-config",
                "namespace": self.config.namespace
            },
            "data": {}
        }
        
        for db in self.config.databases:
            prefix = db.db_type.value.upper()
            db_config["data"].update({
                f"{prefix}_HOST": db.host,
                f"{prefix}_PORT": str(db.port),
                f"{prefix}_DATABASE": db.database,
                f"{prefix}_SSL_MODE": db.ssl_mode,
                f"{prefix}_POOL_SIZE": str(db.connection_pool_size),
                f"{prefix}_MAX_OVERFLOW": str(db.max_overflow)
            })
        
        configmaps.append(db_config)
        
        file_path = output_dir / "03-configmaps.yaml"
        with open(file_path, 'w') as f:
            yaml.dump_all(configmaps, f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_service_manifests(self, service: ServiceConfig, output_dir: Path) -> Dict[str, str]:
        """Generate deployment and service manifests for a service."""
        # Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"wbpm-{service.name}",
                "namespace": self.config.namespace,
                "labels": {
                    "app.kubernetes.io/name": f"wbpm-{service.name}",
                    "app.kubernetes.io/component": service.service_type.value,
                    "app.kubernetes.io/part-of": "wbpm",
                    "environment": self.config.environment.value
                }
            },
            "spec": {
                "replicas": service.replicas,
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": f"wbpm-{service.name}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": f"wbpm-{service.name}",
                            "app.kubernetes.io/component": service.service_type.value
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": str(service.port),
                            "prometheus.io/path": self.config.monitoring.metrics_path
                        }
                    },
                    "spec": {
                        "serviceAccountName": self.config.security.service_account_name,
                        "securityContext": {
                            "runAsNonRoot": self.config.security.run_as_non_root,
                            "runAsUser": self.config.security.run_as_user,
                            "runAsGroup": self.config.security.run_as_group,
                            "fsGroup": self.config.security.fs_group
                        },
                        "containers": [{
                            "name": service.name,
                            "image": f"{service.image}:{service.tag}",
                            "imagePullPolicy": self.config.security.image_pull_policy,
                            "ports": [{
                                "name": "http",
                                "containerPort": service.port,
                                "protocol": "TCP"
                            }],
                            "env": self._generate_environment_variables(service),
                            "resources": {
                                "requests": {
                                    "cpu": service.resources.cpu_request,
                                    "memory": service.resources.memory_request
                                },
                                "limits": {
                                    "cpu": service.resources.cpu_limit,
                                    "memory": service.resources.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": service.health_check_path,
                                    "port": "http"
                                },
                                "initialDelaySeconds": service.liveness_probe_delay,
                                "periodSeconds": 30,
                                "timeoutSeconds": 10,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": service.readiness_check_path,
                                    "port": "http"
                                },
                                "initialDelaySeconds": service.readiness_probe_delay,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "startupProbe": {
                                "httpGet": {
                                    "path": service.health_check_path,
                                    "port": "http"
                                },
                                "initialDelaySeconds": service.startup_probe_delay,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Service
        k8s_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"wbpm-{service.name}",
                "namespace": self.config.namespace,
                "labels": {
                    "app.kubernetes.io/name": f"wbpm-{service.name}",
                    "app.kubernetes.io/component": service.service_type.value
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{
                    "name": "http",
                    "port": 80,
                    "targetPort": "http",
                    "protocol": "TCP"
                }],
                "selector": {
                    "app.kubernetes.io/name": f"wbpm-{service.name}"
                }
            }
        }
        
        # HPA (if auto-scaling enabled)
        hpa = None
        if service.auto_scaling_enabled:
            hpa = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"wbpm-{service.name}",
                    "namespace": self.config.namespace
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": f"wbpm-{service.name}"
                    },
                    "minReplicas": service.min_replicas,
                    "maxReplicas": service.max_replicas,
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": service.target_cpu_utilization
                            }
                        }
                    }]
                }
            }
        
        # Save files
        deployment_file = output_dir / f"10-deployment-{service.name}.yaml"
        with open(deployment_file, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        service_file = output_dir / f"11-service-{service.name}.yaml"
        with open(service_file, 'w') as f:
            yaml.dump(k8s_service, f, default_flow_style=False)
        
        files = {
            "deployment": str(deployment_file),
            "service": str(service_file)
        }
        
        if hpa:
            hpa_file = output_dir / f"12-hpa-{service.name}.yaml"
            with open(hpa_file, 'w') as f:
                yaml.dump(hpa, f, default_flow_style=False)
            files["hpa"] = str(hpa_file)
        
        return files
    
    def _generate_environment_variables(self, service: ServiceConfig) -> List[Dict[str, Any]]:
        """Generate environment variables for service."""
        env_vars = []
        
        # Service-specific environment variables
        for key, value in service.environment_vars.items():
            env_vars.append({
                "name": key,
                "value": value
            })
        
        # Common environment variables from config maps
        env_vars.extend([
            {
                "name": "ENVIRONMENT",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": "wbpm-app-config",
                        "key": "environment"
                    }
                }
            },
            {
                "name": "LOG_LEVEL",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": "wbpm-app-config",
                        "key": "log_level"
                    }
                }
            }
        ])
        
        # Database environment variables
        for db in self.config.databases:
            prefix = db.db_type.value.upper()
            env_vars.extend([
                {
                    "name": f"{prefix}_HOST",
                    "valueFrom": {
                        "configMapKeyRef": {
                            "name": "wbpm-db-config",
                            "key": f"{prefix}_HOST"
                        }
                    }
                },
                {
                    "name": f"{prefix}_PASSWORD",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": db.password_secret,
                            "key": "password"
                        }
                    }
                }
            ])
        
        # Secret environment variables
        for secret_name in service.secrets:
            env_vars.append({
                "name": secret_name.upper(),
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "wbpm-app-secrets",
                        "key": secret_name
                    }
                }
            })
        
        return env_vars
    
    async def _generate_ingress(self, output_dir: Path) -> str:
        """Generate ingress manifest."""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "wbpm-ingress",
                "namespace": self.config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": self.config.network.ingress_class,
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "nginx.ingress.kubernetes.io/cors-allow-origin": ",".join(self.config.network.cors_origins),
                    "nginx.ingress.kubernetes.io/rate-limit": str(self.config.network.rate_limit_requests_per_minute)
                }
            },
            "spec": {
                "rules": [{
                    "host": self.config.network.domain,
                    "http": {
                        "paths": [{
                            "path": "/api",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "wbpm-api",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if self.config.network.tls_enabled:
            ingress["spec"]["tls"] = [{
                "hosts": [self.config.network.domain],
                "secretName": self.config.network.tls_secret_name
            }]
        
        file_path = output_dir / "20-ingress.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(ingress, f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_rbac(self, output_dir: Path) -> List[str]:
        """Generate RBAC manifests."""
        files = []
        
        # Service Account
        service_account = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": self.config.security.service_account_name,
                "namespace": self.config.namespace
            }
        }
        
        # Role
        role = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": "wbpm-role",
                "namespace": self.config.namespace
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps", "secrets"],
                    "verbs": ["get", "list", "watch"]
                },
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments"],
                    "verbs": ["get", "list", "watch"]
                }
            ]
        }
        
        # Role Binding
        role_binding = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": "wbpm-role-binding",
                "namespace": self.config.namespace
            },
            "subjects": [{
                "kind": "ServiceAccount",
                "name": self.config.security.service_account_name,
                "namespace": self.config.namespace
            }],
            "roleRef": {
                "kind": "Role",
                "name": "wbpm-role",
                "apiGroup": "rbac.authorization.k8s.io"
            }
        }
        
        rbac_file = output_dir / "04-rbac.yaml"
        with open(rbac_file, 'w') as f:
            yaml.dump_all([service_account, role, role_binding], f, default_flow_style=False)
        
        files.append(str(rbac_file))
        return files
    
    async def _generate_database_manifests(self, output_dir: Path) -> List[str]:
        """Generate database deployment manifests."""
        files = []
        
        for db in self.config.databases:
            if db.db_type == DatabaseType.POSTGRESQL:
                postgres_file = await self._generate_postgresql_manifest(db, output_dir)
                files.append(postgres_file)
            elif db.db_type == DatabaseType.REDIS:
                redis_file = await self._generate_redis_manifest(db, output_dir)
                files.append(redis_file)
        
        return files
    
    async def _generate_postgresql_manifest(self, db: DatabaseConfig, output_dir: Path) -> str:
        """Generate PostgreSQL deployment manifest."""
        postgres_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "postgresql",
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "postgresql"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "postgresql"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "postgresql",
                            "image": "postgres:15-alpine",
                            "env": [
                                {
                                    "name": "POSTGRES_DB",
                                    "value": db.database
                                },
                                {
                                    "name": "POSTGRES_USER",
                                    "value": db.username
                                },
                                {
                                    "name": "POSTGRES_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": db.password_secret,
                                            "key": "password"
                                        }
                                    }
                                }
                            ],
                            "ports": [{
                                "containerPort": 5432
                            }],
                            "volumeMounts": [{
                                "name": "postgres-storage",
                                "mountPath": "/var/lib/postgresql/data"
                            }]
                        }],
                        "volumes": [{
                            "name": "postgres-storage",
                            "persistentVolumeClaim": {
                                "claimName": "postgres-pvc"
                            }
                        }]
                    }
                }
            }
        }
        
        postgres_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "postgresql",
                "namespace": self.config.namespace
            },
            "spec": {
                "ports": [{
                    "port": 5432,
                    "targetPort": 5432
                }],
                "selector": {
                    "app": "postgresql"
                }
            }
        }
        
        postgres_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "postgres-pvc",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {
                    "requests": {
                        "storage": "10Gi"
                    }
                }
            }
        }
        
        file_path = output_dir / "05-postgresql.yaml"
        with open(file_path, 'w') as f:
            yaml.dump_all([postgres_pvc, postgres_deployment, postgres_service], f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_redis_manifest(self, db: DatabaseConfig, output_dir: Path) -> str:
        """Generate Redis deployment manifest."""
        redis_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "redis",
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "redis"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "redis"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "redis",
                            "image": "redis:7-alpine",
                            "ports": [{
                                "containerPort": 6379
                            }],
                            "command": ["redis-server", "--appendonly", "yes"]
                        }]
                    }
                }
            }
        }
        
        redis_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "redis",
                "namespace": self.config.namespace
            },
            "spec": {
                "ports": [{
                    "port": 6379,
                    "targetPort": 6379
                }],
                "selector": {
                    "app": "redis"
                }
            }
        }
        
        file_path = output_dir / "06-redis.yaml"
        with open(file_path, 'w') as f:
            yaml.dump_all([redis_deployment, redis_service], f, default_flow_style=False)
        
        return str(file_path)
    
    async def _generate_monitoring_manifests(self, output_dir: Path) -> List[str]:
        """Generate monitoring manifests."""
        files = []
        
        if self.config.monitoring.prometheus_enabled:
            # ServiceMonitor for Prometheus
            service_monitor = {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "ServiceMonitor",
                "metadata": {
                    "name": "wbpm-metrics",
                    "namespace": self.config.namespace,
                    "labels": {
                        "app.kubernetes.io/name": "wbpm"
                    }
                },
                "spec": {
                    "selector": {
                        "matchLabels": {
                            "app.kubernetes.io/part-of": "wbpm"
                        }
                    },
                    "endpoints": [{
                        "port": "http",
                        "path": self.config.monitoring.metrics_path,
                        "interval": "30s"
                    }]
                }
            }
            
            monitor_file = output_dir / "30-monitoring.yaml"
            with open(monitor_file, 'w') as f:
                yaml.dump(service_monitor, f, default_flow_style=False)
            
            files.append(str(monitor_file))
        
        return files
    
    async def _generate_network_policies(self, output_dir: Path) -> str:
        """Generate network policies."""
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "wbpm-network-policy",
                "namespace": self.config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/part-of": "wbpm"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "ingress-nginx"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 8000
                            }
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "app": "postgresql"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 5432
                            }
                        ]
                    },
                    {
                        "to": [
                            {
                                "podSelector": {
                                    "matchLabels": {
                                        "app": "redis"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {
                                "protocol": "TCP",
                                "port": 6379
                            }
                        ]
                    }
                ]
            }
        }
        
        file_path = output_dir / "40-network-policy.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(network_policy, f, default_flow_style=False)
        
        return str(file_path)


# =============================================================================
# Docker Configuration Generator
# =============================================================================

class DockerConfigGenerator:
    """Generate Docker configurations."""
    
    def __init__(self, environment_config: EnvironmentConfig):
        self.config = environment_config
    
    async def generate_dockerfile(self, service: ServiceConfig, output_dir: Path) -> str:
        """Generate Dockerfile for service."""
        dockerfile_content = f"""# Multi-stage Dockerfile for WBPM {service.name}
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser && \\
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-{self.config.environment.value}.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt && \\
    pip install -r requirements-{self.config.environment.value}.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/appuser/.local/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser && \\
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \\
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{service.port}{service.health_check_path} || exit 1

# Expose port
EXPOSE {service.port}

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{service.port}"]
"""
        
        dockerfile_path = output_dir / f"Dockerfile.{service.name}"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return str(dockerfile_path)
    
    async def generate_docker_compose(self, output_dir: Path) -> str:
        """Generate docker-compose.yml for local development."""
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "wbpm-network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {}
            }
        }
        
        # Add database services
        for db in self.config.databases:
            if db.db_type == DatabaseType.POSTGRESQL:
                compose_config["services"]["postgres"] = {
                    "image": "postgres:15-alpine",
                    "environment": {
                        "POSTGRES_DB": db.database,
                        "POSTGRES_USER": db.username,
                        "POSTGRES_PASSWORD": "dev_password"
                    },
                    "ports": [f"{db.port}:5432"],
                    "volumes": ["postgres_data:/var/lib/postgresql/data"],
                    "networks": ["wbpm-network"],
                    "healthcheck": {
                        "test": ["CMD-SHELL", f"pg_isready -U {db.username} -d {db.database}"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            elif db.db_type == DatabaseType.REDIS:
                compose_config["services"]["redis"] = {
                    "image": "redis:7-alpine",
                    "ports": [f"{db.port}:6379"],
                    "volumes": ["redis_data:/data"],
                    "networks": ["wbpm-network"],
                    "command": ["redis-server", "--appendonly", "yes"],
                    "healthcheck": {
                        "test": ["CMD", "redis-cli", "ping"],
                        "interval": "10s",
                        "timeout": "3s",
                        "retries": 3
                    }
                }
        
        # Add application services
        for service in self.config.services:
            service_config = {
                "build": {
                    "context": ".",
                    "dockerfile": f"Dockerfile.{service.name}"
                },
                "ports": [f"{service.port}:{service.port}"],
                "environment": {
                    "ENVIRONMENT": self.config.environment.value,
                    "LOG_LEVEL": self.config.monitoring.log_level,
                    **service.environment_vars
                },
                "networks": ["wbpm-network"],
                "depends_on": []
            }
            
            # Add database dependencies
            for db in self.config.databases:
                if db.db_type == DatabaseType.POSTGRESQL:
                    service_config["depends_on"].append("postgres")
                    service_config["environment"]["POSTGRESQL_HOST"] = "postgres"
                    service_config["environment"]["POSTGRESQL_PASSWORD"] = "dev_password"
                elif db.db_type == DatabaseType.REDIS:
                    service_config["depends_on"].append("redis")
                    service_config["environment"]["REDIS_HOST"] = "redis"
            
            # Add health check
            service_config["healthcheck"] = {
                "test": [
                    "CMD-SHELL",
                    f"curl -f http://localhost:{service.port}{service.health_check_path} || exit 1"
                ],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "40s"
            }
            
            compose_config["services"][f"wbpm-{service.name}"] = service_config
        
        compose_path = output_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        return str(compose_path)
    
    async def generate_dockerignore(self, output_dir: Path) -> str:
        """Generate .dockerignore file."""
        dockerignore_content = """# Git
.git
.gitignore

# Documentation
README.md
docs/
*.md

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Testing
.coverage
.pytest_cache/
.tox/
htmlcov/

# Local development
.env
.env.local
docker-compose.override.yml

# Kubernetes
k8s/
manifests/

# Deployment
deployment/
"""
        
        dockerignore_path = output_dir / ".dockerignore"
        with open(dockerignore_path, 'w') as f:
            f.write(dockerignore_content)
        
        return str(dockerignore_path)


# =============================================================================
# Environment Configuration Factory
# =============================================================================

class EnvironmentConfigFactory:
    """Factory for creating environment-specific configurations."""
    
    @staticmethod
    def create_development_config() -> EnvironmentConfig:
        """Create development environment configuration."""
        # Development services with minimal resources
        services = [
            ServiceConfig(
                name="api",
                service_type=ServiceType.API,
                image="wbpm-api",
                tag="dev",
                port=8000,
                replicas=1,
                resources=ResourceLimits(
                    cpu_request="100m",
                    cpu_limit="500m",
                    memory_request="256Mi",
                    memory_limit="512Mi"
                ),
                environment_vars={
                    "DEBUG": "true",
                    "RELOAD": "true"
                },
                auto_scaling_enabled=False
            ),
            ServiceConfig(
                name="worker",
                service_type=ServiceType.WORKER,
                image="wbpm-worker",
                tag="dev",
                port=8001,
                replicas=1,
                resources=ResourceLimits(
                    cpu_request="50m",
                    cpu_limit="200m",
                    memory_request="128Mi",
                    memory_limit="256Mi"
                ),
                health_check_path="/health",
                auto_scaling_enabled=False
            )
        ]
        
        # Development databases
        databases = [
            DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                host="postgres",
                port=5432,
                database="wbpm_dev",
                username="wbpm_dev",
                password_secret="wbpm-db-password-dev",
                connection_pool_size=5,
                backup_enabled=False
            ),
            DatabaseConfig(
                db_type=DatabaseType.REDIS,
                host="redis",
                port=6379,
                database="0",
                username="",
                password_secret="",
                backup_enabled=False
            )
        ]
        
        return EnvironmentConfig(
            environment=EnvironmentType.DEVELOPMENT,
            namespace="wbpm-dev",
            cluster_name="dev-cluster",
            region="us-west-2",
            services=services,
            databases=databases,
            network=NetworkConfig(
                domain="wbpm-dev.datacraft.co.ke",
                tls_enabled=False,
                cors_origins=["http://localhost:3000", "http://localhost:8080"],
                rate_limit_requests_per_minute=10000
            ),
            security=SecurityConfig(
                pod_security_context_enabled=False,
                network_policies_enabled=False,
                image_pull_policy="IfNotPresent"
            ),
            monitoring=MonitoringConfig(
                log_level="DEBUG",
                prometheus_enabled=True,
                grafana_enabled=False,
                jaeger_enabled=False
            ),
            backup=BackupConfig(
                enabled=False
            )
        )
    
    @staticmethod
    def create_staging_config() -> EnvironmentConfig:
        """Create staging environment configuration."""
        # Staging services with moderate resources
        services = [
            ServiceConfig(
                name="api",
                service_type=ServiceType.API,
                image="wbpm-api",
                tag="staging",
                port=8000,
                replicas=2,
                resources=ResourceLimits(
                    cpu_request="200m",
                    cpu_limit="1000m",
                    memory_request="512Mi",
                    memory_limit="1Gi"
                ),
                auto_scaling_enabled=True,
                min_replicas=2,
                max_replicas=5,
                target_cpu_utilization=70
            ),
            ServiceConfig(
                name="worker",
                service_type=ServiceType.WORKER,
                image="wbpm-worker",
                tag="staging",
                port=8001,
                replicas=2,
                resources=ResourceLimits(
                    cpu_request="100m",
                    cpu_limit="500m",
                    memory_request="256Mi",
                    memory_limit="512Mi"
                ),
                auto_scaling_enabled=True,
                min_replicas=1,
                max_replicas=3
            ),
            ServiceConfig(
                name="scheduler",
                service_type=ServiceType.SCHEDULER,
                image="wbpm-scheduler",
                tag="staging",
                port=8002,
                replicas=1,
                resources=ResourceLimits(
                    cpu_request="50m",
                    cpu_limit="200m",
                    memory_request="128Mi",
                    memory_limit="256Mi"
                )
            )
        ]
        
        # Staging databases
        databases = [
            DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                host="postgres",
                port=5432,
                database="wbpm_staging",
                username="wbpm_staging",
                password_secret="wbpm-db-password-staging",
                connection_pool_size=10,
                backup_enabled=True,
                backup_schedule="0 3 * * *"
            ),
            DatabaseConfig(
                db_type=DatabaseType.REDIS,
                host="redis",
                port=6379,
                database="0",
                username="",
                password_secret="",
                backup_enabled=False
            )
        ]
        
        return EnvironmentConfig(
            environment=EnvironmentType.STAGING,
            namespace="wbpm-staging",
            cluster_name="staging-cluster",
            region="us-west-2",
            services=services,
            databases=databases,
            network=NetworkConfig(
                domain="wbpm-staging.datacraft.co.ke",
                tls_enabled=True,
                cors_origins=["https://app-staging.datacraft.co.ke"],
                rate_limit_requests_per_minute=5000
            ),
            security=SecurityConfig(
                pod_security_context_enabled=True,
                network_policies_enabled=True,
                image_pull_policy="Always"
            ),
            monitoring=MonitoringConfig(
                log_level="INFO",
                prometheus_enabled=True,
                grafana_enabled=True,
                jaeger_enabled=True
            ),
            backup=BackupConfig(
                enabled=True,
                retention_days=14
            )
        )
    
    @staticmethod
    def create_production_config() -> EnvironmentConfig:
        """Create production environment configuration."""
        # Production services with high availability
        services = [
            ServiceConfig(
                name="api",
                service_type=ServiceType.API,
                image="wbpm-api",
                tag="latest",
                port=8000,
                replicas=3,
                resources=ResourceLimits(
                    cpu_request="500m",
                    cpu_limit="2000m",
                    memory_request="1Gi",
                    memory_limit="2Gi"
                ),
                auto_scaling_enabled=True,
                min_replicas=3,
                max_replicas=10,
                target_cpu_utilization=60
            ),
            ServiceConfig(
                name="worker",
                service_type=ServiceType.WORKER,
                image="wbpm-worker",
                tag="latest",
                port=8001,
                replicas=5,
                resources=ResourceLimits(
                    cpu_request="300m",
                    cpu_limit="1000m",
                    memory_request="512Mi",
                    memory_limit="1Gi"
                ),
                auto_scaling_enabled=True,
                min_replicas=3,
                max_replicas=15,
                target_cpu_utilization=70
            ),
            ServiceConfig(
                name="scheduler",
                service_type=ServiceType.SCHEDULER,
                image="wbpm-scheduler",
                tag="latest",
                port=8002,
                replicas=2,
                resources=ResourceLimits(
                    cpu_request="200m",
                    cpu_limit="500m",
                    memory_request="256Mi",
                    memory_limit="512Mi"
                ),
                auto_scaling_enabled=False
            ),
            ServiceConfig(
                name="monitor",
                service_type=ServiceType.MONITOR,
                image="wbpm-monitor",
                tag="latest",
                port=8003,
                replicas=2,
                resources=ResourceLimits(
                    cpu_request="100m",
                    cpu_limit="300m",
                    memory_request="256Mi",
                    memory_limit="512Mi"
                )
            )
        ]
        
        # Production databases with high availability
        databases = [
            DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                host="postgres-primary",
                port=5432,
                database="wbpm_prod",
                username="wbpm_prod",
                password_secret="wbpm-db-password-prod",
                connection_pool_size=50,
                max_overflow=100,
                backup_enabled=True,
                backup_schedule="0 1 * * *"
            ),
            DatabaseConfig(
                db_type=DatabaseType.REDIS,
                host="redis-cluster",
                port=6379,
                database="0",
                username="",
                password_secret="redis-password-prod",
                backup_enabled=True
            )
        ]
        
        return EnvironmentConfig(
            environment=EnvironmentType.PRODUCTION,
            namespace="wbpm-prod",
            cluster_name="prod-cluster",
            region="us-west-2",
            services=services,
            databases=databases,
            network=NetworkConfig(
                domain="wbpm.datacraft.co.ke",
                tls_enabled=True,
                cors_origins=["https://app.datacraft.co.ke"],
                rate_limit_requests_per_minute=1000
            ),
            security=SecurityConfig(
                pod_security_context_enabled=True,
                network_policies_enabled=True,
                image_pull_policy="Always"
            ),
            monitoring=MonitoringConfig(
                log_level="WARNING",
                prometheus_enabled=True,
                grafana_enabled=True,
                jaeger_enabled=True,
                alert_manager_enabled=True
            ),
            backup=BackupConfig(
                enabled=True,
                retention_days=90,
                backup_location="s3://wbpm-backups-prod",
                encryption_enabled=True
            )
        )


# =============================================================================
# Deployment Manager
# =============================================================================

class DeploymentManager:
    """Manage deployment configurations and generation."""
    
    def __init__(self):
        self.config_factory = EnvironmentConfigFactory()
        
    async def generate_deployment_artifacts(
        self,
        environment: EnvironmentType,
        output_dir: str = "./deployment"
    ) -> Dict[str, List[str]]:
        """Generate all deployment artifacts for environment."""
        try:
            # Get environment configuration
            if environment == EnvironmentType.DEVELOPMENT:
                config = self.config_factory.create_development_config()
            elif environment == EnvironmentType.STAGING:
                config = self.config_factory.create_staging_config()
            elif environment == EnvironmentType.PRODUCTION:
                config = self.config_factory.create_production_config()
            else:
                raise ValueError(f"Unsupported environment: {environment}")
            
            output_path = Path(output_dir) / environment.value
            output_path.mkdir(parents=True, exist_ok=True)
            
            generated_files = {
                "kubernetes": [],
                "docker": [],
                "scripts": []
            }
            
            # Generate Kubernetes manifests
            k8s_generator = KubernetesManifestGenerator(config)
            k8s_dir = output_path / "kubernetes"
            k8s_files = await k8s_generator.generate_all_manifests(k8s_dir)
            generated_files["kubernetes"] = k8s_files
            
            # Generate Docker configurations
            docker_generator = DockerConfigGenerator(config)
            docker_dir = output_path / "docker"
            docker_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate Dockerfiles for each service
            for service in config.services:
                dockerfile = await docker_generator.generate_dockerfile(service, docker_dir)
                generated_files["docker"].append(dockerfile)
            
            # Generate docker-compose
            compose_file = await docker_generator.generate_docker_compose(docker_dir)
            generated_files["docker"].append(compose_file)
            
            # Generate .dockerignore
            dockerignore_file = await docker_generator.generate_dockerignore(docker_dir)
            generated_files["docker"].append(dockerignore_file)
            
            # Generate deployment scripts
            scripts_dir = output_path / "scripts"
            script_files = await self._generate_deployment_scripts(config, scripts_dir)
            generated_files["scripts"] = script_files
            
            # Generate configuration file
            config_file = output_path / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
            
            logger.info(f"Generated deployment artifacts for {environment.value}")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating deployment artifacts: {e}")
            raise
    
    async def _generate_deployment_scripts(self, config: EnvironmentConfig, output_dir: Path) -> List[str]:
        """Generate deployment and utility scripts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        scripts = []
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
set -e

# WBPM Deployment Script for {config.environment.value}
echo "Deploying WBPM to {config.environment.value} environment..."

# Set variables
NAMESPACE="{config.namespace}"
ENVIRONMENT="{config.environment.value}"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests in order
echo "Applying Kubernetes manifests..."
kubectl apply -f ../kubernetes/01-namespace.yaml
kubectl apply -f ../kubernetes/02-secrets.yaml
kubectl apply -f ../kubernetes/03-configmaps.yaml
kubectl apply -f ../kubernetes/04-rbac.yaml
kubectl apply -f ../kubernetes/05-postgresql.yaml
kubectl apply -f ../kubernetes/06-redis.yaml

# Wait for databases to be ready
echo "Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=postgresql -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# Apply application manifests
echo "Deploying application services..."
kubectl apply -f ../kubernetes/10-deployment-api.yaml
kubectl apply -f ../kubernetes/11-service-api.yaml
kubectl apply -f ../kubernetes/10-deployment-worker.yaml
kubectl apply -f ../kubernetes/11-service-worker.yaml

# Apply autoscaling if enabled
if [ "$ENVIRONMENT" != "development" ]; then
    kubectl apply -f ../kubernetes/12-hpa-api.yaml || true
    kubectl apply -f ../kubernetes/12-hpa-worker.yaml || true
fi

# Apply ingress
kubectl apply -f ../kubernetes/20-ingress.yaml

# Apply monitoring
kubectl apply -f ../kubernetes/30-monitoring.yaml || true

# Apply network policies
if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
    kubectl apply -f ../kubernetes/40-network-policy.yaml
fi

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=wbpm -n $NAMESPACE --timeout=600s

# Run database migrations
echo "Running database migrations..."
kubectl run wbpm-migration --image=wbmp-api:{config.services[0].tag} --restart=Never -n $NAMESPACE -- python -m alembic upgrade head
kubectl wait --for=condition=complete job/wbpm-migration -n $NAMESPACE --timeout=300s
kubectl delete job wbpm-migration -n $NAMESPACE

echo "Deployment completed successfully!"
echo "Application URL: https://{config.network.domain}"

# Show status
kubectl get all -n $NAMESPACE
"""
        
        deploy_file = output_dir / "deploy.sh"
        with open(deploy_file, 'w') as f:
            f.write(deploy_script)
        deploy_file.chmod(0o755)
        scripts.append(str(deploy_file))
        
        # Status script
        status_script = f"""#!/bin/bash

# WBPM Status Check Script
NAMESPACE="{config.namespace}"

echo "=== WBPM Status for {config.environment.value} ==="
echo

echo "Namespace: $NAMESPACE"
echo "Environment: {config.environment.value}"
echo

echo "=== Pods ==="
kubectl get pods -n $NAMESPACE

echo
echo "=== Services ==="
kubectl get services -n $NAMESPACE

echo
echo "=== Ingress ==="
kubectl get ingress -n $NAMESPACE

echo
echo "=== HPA (if enabled) ==="
kubectl get hpa -n $NAMESPACE 2>/dev/null || echo "No HPA configured"

echo
echo "=== Recent Events ==="
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -10

echo
echo "=== Resource Usage ==="
kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Metrics server not available"
"""
        
        status_file = output_dir / "status.sh"
        with open(status_file, 'w') as f:
            f.write(status_script)
        status_file.chmod(0o755)
        scripts.append(str(status_file))
        
        # Cleanup script
        cleanup_script = f"""#!/bin/bash

# WBPM Cleanup Script
NAMESPACE="{config.namespace}"

echo "WARNING: This will delete all WBPM resources in $NAMESPACE"
echo "Press Ctrl+C to cancel or Enter to continue..."
read

echo "Deleting WBPM resources..."

# Delete application resources
kubectl delete deployment,service,ingress,hpa,configmap,secret -l app.kubernetes.io/part-of=wbpm -n $NAMESPACE || true

# Delete database resources (be careful in production!)
if [ "{config.environment.value}" = "development" ]; then
    kubectl delete deployment,service,pvc -l app=postgresql -n $NAMESPACE || true
    kubectl delete deployment,service -l app=redis -n $NAMESPACE || true
fi

# Delete namespace (optional)
echo "Delete namespace $NAMESPACE? (y/N)"
read DELETE_NS
if [ "$DELETE_NS" = "y" ] || [ "$DELETE_NS" = "Y" ]; then
    kubectl delete namespace $NAMESPACE
    echo "Namespace deleted"
else
    echo "Namespace preserved"
fi

echo "Cleanup completed"
"""
        
        cleanup_file = output_dir / "cleanup.sh"
        with open(cleanup_file, 'w') as f:
            f.write(cleanup_script)
        cleanup_file.chmod(0o755)
        scripts.append(str(cleanup_file))
        
        # Build script
        build_script = f"""#!/bin/bash
set -e

# WBPM Build Script
echo "Building WBPM images for {config.environment.value}..."

# Build images for each service
"""
        
        for service in config.services:
            build_script += f"""
echo "Building {service.name} image..."
docker build -f ../docker/Dockerfile.{service.name} -t {service.image}:{service.tag} .
"""
        
        build_script += f"""
echo "Build completed successfully!"

# Tag for registry if not development
if [ "{config.environment.value}" != "development" ]; then
    echo "Tagging images for registry..."
"""
        
        for service in config.services:
            build_script += f"""    docker tag {service.image}:{service.tag} registry.datacraft.co.ke/{service.image}:{service.tag}
"""
        
        build_script += """fi

echo "All images built successfully!"
"""
        
        build_file = output_dir / "build.sh"
        with open(build_file, 'w') as f:
            f.write(build_script)
        build_file.chmod(0o755)
        scripts.append(str(build_file))
        
        return scripts


# =============================================================================
# Service Factory
# =============================================================================

async def generate_deployment_configurations(
    environments: List[EnvironmentType] = None,
    output_dir: str = "./deployment"
) -> Dict[str, Dict[str, List[str]]]:
    """Generate deployment configurations for specified environments."""
    try:
        if environments is None:
            environments = [EnvironmentType.DEVELOPMENT, EnvironmentType.STAGING, EnvironmentType.PRODUCTION]
        
        deployment_manager = DeploymentManager()
        all_generated_files = {}
        
        for environment in environments:
            generated_files = await deployment_manager.generate_deployment_artifacts(
                environment, output_dir
            )
            all_generated_files[environment.value] = generated_files
        
        logger.info("Deployment configurations generated successfully")
        return all_generated_files
        
    except Exception as e:
        logger.error(f"Error generating deployment configurations: {e}")
        raise


# Export main classes
__all__ = [
    'DeploymentManager',
    'EnvironmentConfigFactory',
    'KubernetesManifestGenerator',
    'DockerConfigGenerator',
    'EnvironmentConfig',
    'ServiceConfig',
    'DatabaseConfig',
    'NetworkConfig',
    'SecurityConfig',
    'MonitoringConfig',
    'BackupConfig',
    'EnvironmentType',
    'ServiceType',
    'DatabaseType',
    'generate_deployment_configurations'
]