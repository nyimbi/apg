"""
Audio Processing Deployment Configuration

Production deployment configurations, container orchestration, 
infrastructure as code, and deployment automation.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from uuid_extensions import uuid7str


@dataclass
class DatabaseConfig:
	"""Database configuration"""
	host: str = "localhost"
	port: int = 5432
	name: str = "apg_audio_processing"
	user: str = "apg_user"
	password: str = ""
	pool_size: int = 20
	max_overflow: int = 10
	pool_timeout: int = 30
	ssl_mode: str = "require"


@dataclass
class RedisConfig:
	"""Redis configuration"""
	host: str = "localhost"
	port: int = 6379
	db: int = 0
	password: str = ""
	ssl: bool = False
	pool_size: int = 10
	timeout: int = 5


@dataclass
class ResourceLimits:
	"""Resource limits configuration"""
	cpu_request: str = "500m"
	cpu_limit: str = "2000m"
	memory_request: str = "1Gi"
	memory_limit: str = "4Gi"
	storage_request: str = "10Gi"
	storage_limit: str = "100Gi"


@dataclass
class AutoScalingConfig:
	"""Auto-scaling configuration"""
	min_replicas: int = 2
	max_replicas: int = 10
	target_cpu_utilization: int = 70
	target_memory_utilization: int = 80
	scale_down_delay: int = 300  # seconds
	scale_up_delay: int = 60     # seconds


@dataclass
class HealthCheckConfig:
	"""Health check configuration"""
	enabled: bool = True
	path: str = "/audio_processing/health"
	initial_delay: int = 30
	period_seconds: int = 10
	timeout_seconds: int = 5
	failure_threshold: int = 3
	success_threshold: int = 1


@dataclass
class SecurityConfig:
	"""Security configuration"""
	enable_rbac: bool = True
	enable_network_policies: bool = True
	enable_pod_security_policy: bool = True
	enable_tls: bool = True
	tls_cert_path: str = "/etc/certs/tls.crt"
	tls_key_path: str = "/etc/certs/tls.key"
	jwt_secret_key: str = ""
	allowed_origins: List[str] = None


@dataclass
class MonitoringConfig:
	"""Monitoring configuration"""
	prometheus_enabled: bool = True
	grafana_enabled: bool = True
	jaeger_enabled: bool = True
	log_level: str = "INFO"
	metrics_port: int = 8080
	traces_endpoint: str = "http://jaeger:14268/api/traces"


@dataclass
class DeploymentConfig:
	"""Complete deployment configuration"""
	environment: str = "production"
	namespace: str = "apg-audio-processing"
	app_name: str = "audio-processing"
	version: str = "1.0.0"
	image: str = "apg/audio-processing:1.0.0"
	replicas: int = 3
	
	database: DatabaseConfig = None
	redis: RedisConfig = None
	resources: ResourceLimits = None
	autoscaling: AutoScalingConfig = None
	health_check: HealthCheckConfig = None
	security: SecurityConfig = None
	monitoring: MonitoringConfig = None
	
	def __post_init__(self):
		if self.database is None:
			self.database = DatabaseConfig()
		if self.redis is None:
			self.redis = RedisConfig()
		if self.resources is None:
			self.resources = ResourceLimits()
		if self.autoscaling is None:
			self.autoscaling = AutoScalingConfig()
		if self.health_check is None:
			self.health_check = HealthCheckConfig()
		if self.security is None:
			self.security = SecurityConfig()
		if self.monitoring is None:
			self.monitoring = MonitoringConfig()


class KubernetesManifestGenerator:
	"""Generate Kubernetes deployment manifests"""
	
	def __init__(self, config: DeploymentConfig):
		self.config = config
	
	def generate_namespace(self) -> Dict[str, Any]:
		"""Generate namespace manifest"""
		return {
			"apiVersion": "v1",
			"kind": "Namespace",
			"metadata": {
				"name": self.config.namespace,
				"labels": {
					"app": self.config.app_name,
					"version": self.config.version,
					"environment": self.config.environment
				}
			}
		}
	
	def generate_configmap(self) -> Dict[str, Any]:
		"""Generate ConfigMap manifest"""
		config_data = {
			"DATABASE_HOST": self.config.database.host,
			"DATABASE_PORT": str(self.config.database.port),
			"DATABASE_NAME": self.config.database.name,
			"DATABASE_USER": self.config.database.user,
			"DATABASE_POOL_SIZE": str(self.config.database.pool_size),
			"REDIS_HOST": self.config.redis.host,
			"REDIS_PORT": str(self.config.redis.port),
			"REDIS_DB": str(self.config.redis.db),
			"LOG_LEVEL": self.config.monitoring.log_level,
			"METRICS_PORT": str(self.config.monitoring.metrics_port),
			"ENVIRONMENT": self.config.environment
		}
		
		return {
			"apiVersion": "v1",
			"kind": "ConfigMap",
			"metadata": {
				"name": f"{self.config.app_name}-config",
				"namespace": self.config.namespace
			},
			"data": config_data
		}
	
	def generate_secret(self) -> Dict[str, Any]:
		"""Generate Secret manifest"""
		return {
			"apiVersion": "v1",
			"kind": "Secret",
			"metadata": {
				"name": f"{self.config.app_name}-secrets",
				"namespace": self.config.namespace
			},
			"type": "Opaque",
			"data": {
				"DATABASE_PASSWORD": "",  # Base64 encoded
				"REDIS_PASSWORD": "",     # Base64 encoded
				"JWT_SECRET_KEY": ""      # Base64 encoded
			}
		}
	
	def generate_deployment(self) -> Dict[str, Any]:
		"""Generate Deployment manifest"""
		container_spec = {
			"name": self.config.app_name,
			"image": self.config.image,
			"imagePullPolicy": "IfNotPresent",
			"ports": [
				{"containerPort": 8000, "name": "http"},
				{"containerPort": self.config.monitoring.metrics_port, "name": "metrics"}
			],
			"envFrom": [
				{"configMapRef": {"name": f"{self.config.app_name}-config"}},
				{"secretRef": {"name": f"{self.config.app_name}-secrets"}}
			],
			"resources": {
				"requests": {
					"cpu": self.config.resources.cpu_request,
					"memory": self.config.resources.memory_request
				},
				"limits": {
					"cpu": self.config.resources.cpu_limit,
					"memory": self.config.resources.memory_limit
				}
			}
		}
		
		# Add health checks if enabled
		if self.config.health_check.enabled:
			container_spec["livenessProbe"] = {
				"httpGet": {
					"path": self.config.health_check.path,
					"port": "http"
				},
				"initialDelaySeconds": self.config.health_check.initial_delay,
				"periodSeconds": self.config.health_check.period_seconds,
				"timeoutSeconds": self.config.health_check.timeout_seconds,
				"failureThreshold": self.config.health_check.failure_threshold
			}
			
			container_spec["readinessProbe"] = {
				"httpGet": {
					"path": self.config.health_check.path,
					"port": "http"
				},
				"initialDelaySeconds": 10,
				"periodSeconds": 5,
				"timeoutSeconds": self.config.health_check.timeout_seconds,
				"successThreshold": self.config.health_check.success_threshold
			}
		
		return {
			"apiVersion": "apps/v1",
			"kind": "Deployment",
			"metadata": {
				"name": self.config.app_name,
				"namespace": self.config.namespace,
				"labels": {
					"app": self.config.app_name,
					"version": self.config.version
				}
			},
			"spec": {
				"replicas": self.config.replicas,
				"selector": {
					"matchLabels": {
						"app": self.config.app_name
					}
				},
				"template": {
					"metadata": {
						"labels": {
							"app": self.config.app_name,
							"version": self.config.version
						}
					},
					"spec": {
						"containers": [container_spec],
						"securityContext": {
							"runAsNonRoot": True,
							"runAsUser": 1000,
							"fsGroup": 2000
						} if self.config.security.enable_pod_security_policy else {}
					}
				}
			}
		}
	
	def generate_service(self) -> Dict[str, Any]:
		"""Generate Service manifest"""
		return {
			"apiVersion": "v1",
			"kind": "Service",
			"metadata": {
				"name": self.config.app_name,
				"namespace": self.config.namespace,
				"labels": {
					"app": self.config.app_name
				}
			},
			"spec": {
				"selector": {
					"app": self.config.app_name
				},
				"ports": [
					{
						"name": "http",
						"port": 80,
						"targetPort": "http",
						"protocol": "TCP"
					},
					{
						"name": "metrics",
						"port": self.config.monitoring.metrics_port,
						"targetPort": "metrics",
						"protocol": "TCP"
					}
				],
				"type": "ClusterIP"
			}
		}
	
	def generate_hpa(self) -> Dict[str, Any]:
		"""Generate HorizontalPodAutoscaler manifest"""
		return {
			"apiVersion": "autoscaling/v2",
			"kind": "HorizontalPodAutoscaler",
			"metadata": {
				"name": f"{self.config.app_name}-hpa",
				"namespace": self.config.namespace
			},
			"spec": {
				"scaleTargetRef": {
					"apiVersion": "apps/v1",
					"kind": "Deployment",
					"name": self.config.app_name
				},
				"minReplicas": self.config.autoscaling.min_replicas,
				"maxReplicas": self.config.autoscaling.max_replicas,
				"metrics": [
					{
						"type": "Resource",
						"resource": {
							"name": "cpu",
							"target": {
								"type": "Utilization",
								"averageUtilization": self.config.autoscaling.target_cpu_utilization
							}
						}
					},
					{
						"type": "Resource",
						"resource": {
							"name": "memory",
							"target": {
								"type": "Utilization",
								"averageUtilization": self.config.autoscaling.target_memory_utilization
							}
						}
					}
				],
				"behavior": {
					"scaleDown": {
						"stabilizationWindowSeconds": self.config.autoscaling.scale_down_delay
					},
					"scaleUp": {
						"stabilizationWindowSeconds": self.config.autoscaling.scale_up_delay
					}
				}
			}
		}
	
	def generate_ingress(self, host: str, tls_enabled: bool = True) -> Dict[str, Any]:
		"""Generate Ingress manifest"""
		ingress_spec = {
			"rules": [
				{
					"host": host,
					"http": {
						"paths": [
							{
								"path": "/audio_processing",
								"pathType": "Prefix",
								"backend": {
									"service": {
										"name": self.config.app_name,
										"port": {"number": 80}
									}
								}
							}
						]
					}
				}
			]
		}
		
		if tls_enabled and self.config.security.enable_tls:
			ingress_spec["tls"] = [
				{
					"hosts": [host],
					"secretName": f"{self.config.app_name}-tls"
				}
			]
		
		return {
			"apiVersion": "networking.k8s.io/v1",
			"kind": "Ingress",
			"metadata": {
				"name": f"{self.config.app_name}-ingress",
				"namespace": self.config.namespace,
				"annotations": {
					"nginx.ingress.kubernetes.io/rewrite-target": "/",
					"cert-manager.io/cluster-issuer": "letsencrypt-prod"
				}
			},
			"spec": ingress_spec
		}
	
	def generate_all_manifests(self, host: str = "audio.apg.local") -> List[Dict[str, Any]]:
		"""Generate all Kubernetes manifests"""
		manifests = [
			self.generate_namespace(),
			self.generate_configmap(),
			self.generate_secret(),
			self.generate_deployment(),
			self.generate_service(),
			self.generate_hpa(),
			self.generate_ingress(host)
		]
		
		return manifests


class DockerComposeGenerator:
	"""Generate Docker Compose configuration"""
	
	def __init__(self, config: DeploymentConfig):
		self.config = config
	
	def generate_compose(self) -> Dict[str, Any]:
		"""Generate docker-compose.yml configuration"""
		compose_config = {
			"version": "3.8",
			"services": {
				"audio-processing": {
					"image": self.config.image,
					"ports": [
						"8000:8000",
						f"{self.config.monitoring.metrics_port}:{self.config.monitoring.metrics_port}"
					],
					"environment": {
						"DATABASE_HOST": self.config.database.host,
						"DATABASE_PORT": self.config.database.port,
						"DATABASE_NAME": self.config.database.name,
						"DATABASE_USER": self.config.database.user,
						"REDIS_HOST": self.config.redis.host,
						"REDIS_PORT": self.config.redis.port,
						"REDIS_DB": self.config.redis.db,
						"LOG_LEVEL": self.config.monitoring.log_level,
						"ENVIRONMENT": self.config.environment
					},
					"depends_on": ["postgres", "redis"],
					"volumes": [
						"./data:/app/data",
						"./logs:/app/logs"
					],
					"restart": "unless-stopped",
					"healthcheck": {
						"test": f"curl -f http://localhost:8000{self.config.health_check.path} || exit 1",
						"interval": f"{self.config.health_check.period_seconds}s",
						"timeout": f"{self.config.health_check.timeout_seconds}s",
						"retries": self.config.health_check.failure_threshold,
						"start_period": f"{self.config.health_check.initial_delay}s"
					}
				},
				"postgres": {
					"image": "postgres:15-alpine",
					"environment": {
						"POSTGRES_DB": self.config.database.name,
						"POSTGRES_USER": self.config.database.user,
						"POSTGRES_PASSWORD": "${DATABASE_PASSWORD}",
						"POSTGRES_INITDB_ARGS": "--auth-host=scram-sha-256"
					},
					"ports": [f"{self.config.database.port}:5432"],
					"volumes": [
						"postgres_data:/var/lib/postgresql/data",
						"./sql/init.sql:/docker-entrypoint-initdb.d/init.sql"
					],
					"restart": "unless-stopped"
				},
				"redis": {
					"image": "redis:7-alpine",
					"ports": [f"{self.config.redis.port}:6379"],
					"volumes": ["redis_data:/data"],
					"restart": "unless-stopped",
					"command": "redis-server --appendonly yes"
				}
			},
			"volumes": {
				"postgres_data": {},
				"redis_data": {}
			},
			"networks": {
				"default": {
					"name": "apg-audio-processing"
				}
			}
		}
		
		# Add monitoring services if enabled
		if self.config.monitoring.prometheus_enabled:
			compose_config["services"]["prometheus"] = {
				"image": "prom/prometheus:latest",
				"ports": ["9090:9090"],
				"volumes": [
					"./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
					"prometheus_data:/prometheus"
				],
				"command": [
					"--config.file=/etc/prometheus/prometheus.yml",
					"--storage.tsdb.path=/prometheus",
					"--web.console.libraries=/etc/prometheus/console_libraries",
					"--web.console.templates=/etc/prometheus/consoles",
					"--web.enable-lifecycle"
				],
				"restart": "unless-stopped"
			}
			compose_config["volumes"]["prometheus_data"] = {}
		
		if self.config.monitoring.grafana_enabled:
			compose_config["services"]["grafana"] = {
				"image": "grafana/grafana:latest",
				"ports": ["3000:3000"],
				"environment": {
					"GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD}"
				},
				"volumes": [
					"grafana_data:/var/lib/grafana",
					"./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards",
					"./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources"
				],
				"restart": "unless-stopped"
			}
			compose_config["volumes"]["grafana_data"] = {}
		
		return compose_config


class TerraformGenerator:
	"""Generate Terraform infrastructure configuration"""
	
	def __init__(self, config: DeploymentConfig):
		self.config = config
	
	def generate_main_tf(self) -> str:
		"""Generate main Terraform configuration"""
		terraform_config = f"""
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.16"
    }}
    helm = {{
      source  = "hashicorp/helm"
      version = "~> 2.8"
    }}
  }}
}}

provider "kubernetes" {{
  config_path = "~/.kube/config"
}}

provider "helm" {{
  kubernetes {{
    config_path = "~/.kube/config"
  }}
}}

# Namespace
resource "kubernetes_namespace" "audio_processing" {{
  metadata {{
    name = "{self.config.namespace}"
    labels = {{
      app         = "{self.config.app_name}"
      version     = "{self.config.version}"
      environment = "{self.config.environment}"
    }}
  }}
}}

# ConfigMap
resource "kubernetes_config_map" "audio_processing_config" {{
  metadata {{
    name      = "{self.config.app_name}-config"
    namespace = kubernetes_namespace.audio_processing.metadata[0].name
  }}
  
  data = {{
    DATABASE_HOST     = var.database_host
    DATABASE_PORT     = var.database_port
    DATABASE_NAME     = var.database_name
    DATABASE_USER     = var.database_user
    REDIS_HOST        = var.redis_host
    REDIS_PORT        = var.redis_port
    REDIS_DB          = var.redis_db
    LOG_LEVEL         = "{self.config.monitoring.log_level}"
    METRICS_PORT      = "{self.config.monitoring.metrics_port}"
    ENVIRONMENT       = "{self.config.environment}"
  }}
}}

# Secret
resource "kubernetes_secret" "audio_processing_secrets" {{
  metadata {{
    name      = "{self.config.app_name}-secrets"
    namespace = kubernetes_namespace.audio_processing.metadata[0].name
  }}
  
  data = {{
    DATABASE_PASSWORD = var.database_password
    REDIS_PASSWORD    = var.redis_password
    JWT_SECRET_KEY    = var.jwt_secret_key
  }}
  
  type = "Opaque"
}}

# Deployment
resource "kubernetes_deployment" "audio_processing" {{
  metadata {{
    name      = "{self.config.app_name}"
    namespace = kubernetes_namespace.audio_processing.metadata[0].name
    labels = {{
      app     = "{self.config.app_name}"
      version = "{self.config.version}"
    }}
  }}
  
  spec {{
    replicas = {self.config.replicas}
    
    selector {{
      match_labels = {{
        app = "{self.config.app_name}"
      }}
    }}
    
    template {{
      metadata {{
        labels = {{
          app     = "{self.config.app_name}"
          version = "{self.config.version}"
        }}
      }}
      
      spec {{
        container {{
          name  = "{self.config.app_name}"
          image = "{self.config.image}"
          
          port {{
            container_port = 8000
            name           = "http"
          }}
          
          port {{
            container_port = {self.config.monitoring.metrics_port}
            name           = "metrics"
          }}
          
          env_from {{
            config_map_ref {{
              name = kubernetes_config_map.audio_processing_config.metadata[0].name
            }}
          }}
          
          env_from {{
            secret_ref {{
              name = kubernetes_secret.audio_processing_secrets.metadata[0].name
            }}
          }}
          
          resources {{
            requests = {{
              cpu    = "{self.config.resources.cpu_request}"
              memory = "{self.config.resources.memory_request}"
            }}
            limits = {{
              cpu    = "{self.config.resources.cpu_limit}"
              memory = "{self.config.resources.memory_limit}"
            }}
          }}
          
          liveness_probe {{
            http_get {{
              path = "{self.config.health_check.path}"
              port = "http"
            }}
            initial_delay_seconds = {self.config.health_check.initial_delay}
            period_seconds        = {self.config.health_check.period_seconds}
            timeout_seconds       = {self.config.health_check.timeout_seconds}
            failure_threshold     = {self.config.health_check.failure_threshold}
          }}
          
          readiness_probe {{
            http_get {{
              path = "{self.config.health_check.path}"
              port = "http"
            }}
            initial_delay_seconds = 10
            period_seconds        = 5
            timeout_seconds       = {self.config.health_check.timeout_seconds}
            success_threshold     = {self.config.health_check.success_threshold}
          }}
        }}
        
        security_context {{
          run_as_non_root = true
          run_as_user     = 1000
          fs_group        = 2000
        }}
      }}
    }}
  }}
}}

# Service
resource "kubernetes_service" "audio_processing" {{
  metadata {{
    name      = "{self.config.app_name}"
    namespace = kubernetes_namespace.audio_processing.metadata[0].name
    labels = {{
      app = "{self.config.app_name}"
    }}
  }}
  
  spec {{
    selector = {{
      app = "{self.config.app_name}"
    }}
    
    port {{
      name        = "http"
      port        = 80
      target_port = "http"
      protocol    = "TCP"
    }}
    
    port {{
      name        = "metrics"
      port        = {self.config.monitoring.metrics_port}
      target_port = "metrics"
      protocol    = "TCP"
    }}
    
    type = "ClusterIP"
  }}
}}

# HPA
resource "kubernetes_horizontal_pod_autoscaler_v2" "audio_processing_hpa" {{
  metadata {{
    name      = "{self.config.app_name}-hpa"
    namespace = kubernetes_namespace.audio_processing.metadata[0].name
  }}
  
  spec {{
    scale_target_ref {{
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.audio_processing.metadata[0].name
    }}
    
    min_replicas = {self.config.autoscaling.min_replicas}
    max_replicas = {self.config.autoscaling.max_replicas}
    
    metric {{
      type = "Resource"
      resource {{
        name = "cpu"
        target {{
          type                = "Utilization"
          average_utilization = {self.config.autoscaling.target_cpu_utilization}
        }}
      }}
    }}
    
    metric {{
      type = "Resource"
      resource {{
        name = "memory"
        target {{
          type                = "Utilization"
          average_utilization = {self.config.autoscaling.target_memory_utilization}
        }}
      }}
    }}
  }}
}}
"""
		return terraform_config
	
	def generate_variables_tf(self) -> str:
		"""Generate Terraform variables configuration"""
		return """
variable "database_host" {
  description = "Database host"
  type        = string
  default     = "postgres"
}

variable "database_port" {
  description = "Database port"
  type        = string
  default     = "5432"
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "apg_audio_processing"
}

variable "database_user" {
  description = "Database user"
  type        = string
  default     = "apg_user"
}

variable "database_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_host" {
  description = "Redis host"
  type        = string
  default     = "redis"
}

variable "redis_port" {
  description = "Redis port"
  type        = string
  default     = "6379"
}

variable "redis_db" {
  description = "Redis database"
  type        = string
  default     = "0"
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "jwt_secret_key" {
  description = "JWT secret key"
  type        = string
  sensitive   = true
}
"""
	
	def generate_outputs_tf(self) -> str:
		"""Generate Terraform outputs configuration"""
		return f"""
output "namespace" {{
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.audio_processing.metadata[0].name
}}

output "service_name" {{
  description = "Kubernetes service name"
  value       = kubernetes_service.audio_processing.metadata[0].name
}}

output "deployment_name" {{
  description = "Kubernetes deployment name"
  value       = kubernetes_deployment.audio_processing.metadata[0].name
}}
"""


def create_deployment_config(environment: str = "production") -> DeploymentConfig:
	"""Create deployment configuration with environment-specific defaults"""
	if environment == "development":
		return DeploymentConfig(
			environment="development",
			replicas=1,
			resources=ResourceLimits(
				cpu_request="100m",
				cpu_limit="500m",
				memory_request="256Mi",
				memory_limit="1Gi"
			),
			autoscaling=AutoScalingConfig(
				min_replicas=1,
				max_replicas=3
			),
			security=SecurityConfig(
				enable_rbac=False,
				enable_network_policies=False,
				enable_pod_security_policy=False,
				enable_tls=False
			)
		)
	elif environment == "staging":
		return DeploymentConfig(
			environment="staging",
			replicas=2,
			autoscaling=AutoScalingConfig(
				min_replicas=2,
				max_replicas=6
			)
		)
	else:  # production
		return DeploymentConfig(
			environment="production",
			replicas=3,
			autoscaling=AutoScalingConfig(
				min_replicas=3,
				max_replicas=10
			)
		)


def save_manifests_to_files(manifests: List[Dict[str, Any]], output_dir: str = "./manifests") -> None:
	"""Save Kubernetes manifests to YAML files"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)
	
	for i, manifest in enumerate(manifests):
		kind = manifest.get("kind", "Unknown").lower()
		name = manifest.get("metadata", {}).get("name", f"resource-{i}")
		filename = f"{kind}-{name}.yaml"
		
		with open(output_path / filename, 'w') as f:
			yaml.dump(manifest, f, default_flow_style=False)


def save_docker_compose(compose_config: Dict[str, Any], output_file: str = "./docker-compose.yml") -> None:
	"""Save Docker Compose configuration to file"""
	with open(output_file, 'w') as f:
		yaml.dump(compose_config, f, default_flow_style=False)


def save_terraform_files(main_tf: str, variables_tf: str, outputs_tf: str, output_dir: str = "./terraform") -> None:
	"""Save Terraform configuration files"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)
	
	with open(output_path / "main.tf", 'w') as f:
		f.write(main_tf)
	
	with open(output_path / "variables.tf", 'w') as f:
		f.write(variables_tf)
	
	with open(output_path / "outputs.tf", 'w') as f:
		f.write(outputs_tf)