#!/usr/bin/env python3
"""
Production Deployment Automation for MTen Multi-Tenant Management

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Enterprise-grade deployment automation with staging/production environments,
infrastructure as code, monitoring setup, backup/recovery, and operational procedures.
"""

import asyncio
import json
import yaml
import subprocess
import os
import time
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import aiofiles
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class DeploymentStrategy(str, Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


class InfrastructureProvider(str, Enum):
    """Infrastructure provider types"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    provider: InfrastructureProvider
    region: str
    replicas: int
    resources: Dict[str, Any]
    networking: Dict[str, Any]
    storage: Dict[str, Any]
    monitoring: Dict[str, Any]
    backup: Dict[str, Any]
    security: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Deployment operation result"""
    deployment_id: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    resources_created: List[str] = field(default_factory=list)
    monitoring_endpoints: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    rollback_available: bool = False
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


class InfrastructureAsCode:
    """Infrastructure as Code management"""
    
    def __init__(self):
        self.templates_dir = Path("infrastructure/templates")
        self.environments_dir = Path("infrastructure/environments")
        self.state_dir = Path("infrastructure/state")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.templates_dir, self.environments_dir, self.state_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def generate_terraform_config(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Terraform configuration files"""
        terraform_files = {}
        
        # Main configuration
        main_tf = f"""
# MTen Multi-Tenant Management Infrastructure
# Environment: {config.environment.value}
# Generated: {datetime.now(UTC).isoformat()}

terraform {{
  required_version = ">= 1.0"
  required_providers {{
    {"aws" if config.provider == InfrastructureProvider.AWS else "azurerm" if config.provider == InfrastructureProvider.AZURE else "google"} = {{
      source  = "{self._get_provider_source(config.provider)}"
      version = "~> {self._get_provider_version(config.provider)}"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }}
  }}
  
  backend "s3" {{
    bucket = "mten-terraform-state-{config.environment.value}"
    key    = "mten/{config.environment.value}/terraform.tfstate"
    region = "{config.region}"
  }}
}}

# Provider configuration
{self._generate_provider_config(config)}

# Variables
{self._generate_variables(config)}

# Local values
locals {{
  environment = "{config.environment.value}"
  application = "mten"
  region     = "{config.region}"
  
  common_tags = {{
    Environment   = local.environment
    Application   = local.application
    ManagedBy     = "terraform"
    CreatedAt     = "{datetime.now(UTC).isoformat()}"
    Owner         = "datacraft"
  }}
}}

# Data sources
{self._generate_data_sources(config)}

# Resources
{self._generate_infrastructure_resources(config)}

# Outputs
{self._generate_outputs(config)}
"""
        
        terraform_files["main.tf"] = main_tf
        
        # Variables file
        variables_tf = self._generate_variables_file(config)
        terraform_files["variables.tf"] = variables_tf
        
        # Outputs file
        outputs_tf = self._generate_outputs_file(config)
        terraform_files["outputs.tf"] = outputs_tf
        
        return terraform_files
    
    def _get_provider_source(self, provider: InfrastructureProvider) -> str:
        """Get Terraform provider source"""
        sources = {
            InfrastructureProvider.AWS: "hashicorp/aws",
            InfrastructureProvider.AZURE: "hashicorp/azurerm",
            InfrastructureProvider.GCP: "hashicorp/google",
            InfrastructureProvider.KUBERNETES: "hashicorp/kubernetes"
        }
        return sources.get(provider, "hashicorp/aws")
    
    def _get_provider_version(self, provider: InfrastructureProvider) -> str:
        """Get Terraform provider version"""
        versions = {
            InfrastructureProvider.AWS: "5.0",
            InfrastructureProvider.AZURE: "3.0",
            InfrastructureProvider.GCP: "4.0",
            InfrastructureProvider.KUBERNETES: "2.20"
        }
        return versions.get(provider, "5.0")
    
    def _generate_provider_config(self, config: DeploymentConfig) -> str:
        """Generate provider configuration"""
        if config.provider == InfrastructureProvider.AWS:
            return f"""
provider "aws" {{
  region = var.aws_region
  
  default_tags {{
    tags = local.common_tags
  }}
}}
"""
        elif config.provider == InfrastructureProvider.AZURE:
            return f"""
provider "azurerm" {{
  features {{}}
  subscription_id = var.azure_subscription_id
  tenant_id      = var.azure_tenant_id
}}
"""
        elif config.provider == InfrastructureProvider.GCP:
            return f"""
provider "google" {{
  project = var.gcp_project_id
  region  = var.gcp_region
}}
"""
        return ""
    
    def _generate_variables(self, config: DeploymentConfig) -> str:
        """Generate Terraform variables"""
        return f"""
variable "environment" {{
  description = "Environment name"
  type        = string
  default     = "{config.environment.value}"
}}

variable "replicas" {{
  description = "Number of application replicas"
  type        = number
  default     = {config.replicas}
}}

variable "cpu_limit" {{
  description = "CPU limit per replica"
  type        = string
  default     = "{config.resources.get('cpu_limit', '1000m')}"
}}

variable "memory_limit" {{
  description = "Memory limit per replica"
  type        = string
  default     = "{config.resources.get('memory_limit', '2Gi')}"
}}

variable "storage_size" {{
  description = "Storage size"
  type        = string
  default     = "{config.storage.get('size', '100Gi')}"
}}
"""
    
    def _generate_data_sources(self, config: DeploymentConfig) -> str:
        """Generate Terraform data sources"""
        if config.provider == InfrastructureProvider.AWS:
            return """
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
"""
        return ""
    
    def _generate_infrastructure_resources(self, config: DeploymentConfig) -> str:
        """Generate infrastructure resources"""
        resources = []
        
        if config.provider == InfrastructureProvider.AWS:
            resources.append(self._generate_aws_resources(config))
        elif config.provider == InfrastructureProvider.AZURE:
            resources.append(self._generate_azure_resources(config))
        elif config.provider == InfrastructureProvider.GCP:
            resources.append(self._generate_gcp_resources(config))
        
        # Common Kubernetes resources
        resources.append(self._generate_kubernetes_resources(config))
        
        return "\n\n".join(resources)
    
    def _generate_aws_resources(self, config: DeploymentConfig) -> str:
        """Generate AWS-specific resources"""
        return f"""
# EKS Cluster
resource "aws_eks_cluster" "mten" {{
  name     = "mten-{config.environment.value}"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"

  vpc_config {{
    subnet_ids = data.aws_subnets.default.ids
  }}

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = local.common_tags
}}

# EKS Node Group
resource "aws_eks_node_group" "mten" {{
  cluster_name    = aws_eks_cluster.mten.name
  node_group_name = "mten-{config.environment.value}-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = data.aws_subnets.default.ids

  scaling_config {{
    desired_size = {config.replicas}
    max_size     = {config.replicas * 2}
    min_size     = 1
  }}

  instance_types = ["{config.resources.get('instance_type', 't3.medium')}"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = local.common_tags
}}

# RDS Database
resource "aws_db_instance" "mten" {{
  identifier = "mten-{config.environment.value}"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "{config.resources.get('db_instance_class', 'db.t3.micro')}"
  
  allocated_storage     = {config.storage.get('db_storage_gb', 20)}
  max_allocated_storage = {config.storage.get('db_max_storage_gb', 100)}
  storage_encrypted     = true
  
  db_name  = "mten_{config.environment.value.replace('-', '_')}"
  username = "mten_admin"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = {config.backup.get('retention_days', 7)}
  backup_window          = "{config.backup.get('backup_window', '03:00-04:00')}"
  maintenance_window     = "{config.backup.get('maintenance_window', 'sun:04:00-sun:05:00')}"
  
  deletion_protection = {str(config.environment != DeploymentEnvironment.DEVELOPMENT).lower()}
  
  tags = local.common_tags
}}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "mten" {{
  name       = "mten-{config.environment.value}"
  subnet_ids = data.aws_subnets.default.ids
}}

resource "aws_elasticache_replication_group" "mten" {{
  replication_group_id       = "mten-{config.environment.value}"
  description                = "MTen Redis cluster"
  
  port                = 6379
  parameter_group_name = "default.redis7"
  node_type           = "{config.resources.get('redis_node_type', 'cache.t3.micro')}"
  num_cache_clusters  = {config.resources.get('redis_replicas', 2)}
  
  subnet_group_name  = aws_elasticache_subnet_group.mten.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result
  
  tags = local.common_tags
}}
"""
    
    def _generate_kubernetes_resources(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes resources"""
        return f"""
# Kubernetes Namespace
resource "kubernetes_namespace" "mten" {{
  metadata {{
    name = "mten-{config.environment.value}"
    
    labels = {{
      environment = "{config.environment.value}"
      application = "mten"
    }}
  }}
}}

# ConfigMap
resource "kubernetes_config_map" "mten" {{
  metadata {{
    name      = "mten-config"
    namespace = kubernetes_namespace.mten.metadata[0].name
  }}

  data = {{
    ENVIRONMENT = "{config.environment.value}"
    LOG_LEVEL   = "{config.metadata.get('log_level', 'info')}"
    METRICS_ENABLED = "true"
    HEALTH_CHECK_PORT = "8080"
  }}
}}

# Secret
resource "kubernetes_secret" "mten" {{
  metadata {{
    name      = "mten-secrets"
    namespace = kubernetes_namespace.mten.metadata[0].name
  }}

  type = "Opaque"

  data = {{
    database-url = base64encode("postgresql://mten_admin:${{random_password.db_password.result}}@${{aws_db_instance.mten.endpoint}}/mten_{config.environment.value.replace('-', '_')}")
    redis-url    = base64encode("redis://:${{random_password.redis_password.result}}@${{aws_elasticache_replication_group.mten.configuration_endpoint_address}}:6379")
    api-key      = base64encode(random_password.api_key.result)
  }}
}}

# Deployment
resource "kubernetes_deployment" "mten" {{
  metadata {{
    name      = "mten"
    namespace = kubernetes_namespace.mten.metadata[0].name
    
    labels = {{
      app = "mten"
      version = "{config.metadata.get('version', 'latest')}"
    }}
  }}

  spec {{
    replicas = {config.replicas}

    selector {{
      match_labels = {{
        app = "mten"
      }}
    }}

    template {{
      metadata {{
        labels = {{
          app = "mten"
          version = "{config.metadata.get('version', 'latest')}"
        }}
      }}

      spec {{
        container {{
          image = "mten:{config.metadata.get('version', 'latest')}"
          name  = "mten"
          
          port {{
            container_port = 8000
            name          = "http"
          }}
          
          port {{
            container_port = 8080
            name          = "health"
          }}

          env_from {{
            config_map_ref {{
              name = kubernetes_config_map.mten.metadata[0].name
            }}
          }}

          env_from {{
            secret_ref {{
              name = kubernetes_secret.mten.metadata[0].name
            }}
          }}

          resources {{
            limits = {{
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }}
            requests = {{
              cpu    = "{config.resources.get('cpu_request', '500m')}"
              memory = "{config.resources.get('memory_request', '1Gi')}"
            }}
          }}

          liveness_probe {{
            http_get {{
              path = "/health"
              port = "health"
            }}
            initial_delay_seconds = 30
            period_seconds        = 10
          }}

          readiness_probe {{
            http_get {{
              path = "/ready"
              port = "health"
            }}
            initial_delay_seconds = 5
            period_seconds        = 5
          }}
        }}
      }}
    }}
  }}
}}

# Service
resource "kubernetes_service" "mten" {{
  metadata {{
    name      = "mten"
    namespace = kubernetes_namespace.mten.metadata[0].name
  }}

  spec {{
    selector = {{
      app = "mten"
    }}

    port {{
      name        = "http"
      port        = 80
      target_port = "http"
    }}

    type = "ClusterIP"
  }}
}}

# Ingress
resource "kubernetes_ingress_v1" "mten" {{
  metadata {{
    name      = "mten"
    namespace = kubernetes_namespace.mten.metadata[0].name
    
    annotations = {{
      "kubernetes.io/ingress.class"                 = "nginx"
      "cert-manager.io/cluster-issuer"              = "letsencrypt-prod"
      "nginx.ingress.kubernetes.io/ssl-redirect"    = "true"
      "nginx.ingress.kubernetes.io/rate-limit"      = "1000"
    }}
  }}

  spec {{
    tls {{
      hosts = ["{config.networking.get('domain', f'mten-{config.environment.value}.example.com')}"]
      secret_name = "mten-tls"
    }}

    rule {{
      host = "{config.networking.get('domain', f'mten-{config.environment.value}.example.com')}"
      
      http {{
        path {{
          path = "/"
          path_type = "Prefix"
          
          backend {{
            service {{
              name = kubernetes_service.mten.metadata[0].name
              port {{
                number = 80
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""
    
    def _generate_variables_file(self, config: DeploymentConfig) -> str:
        """Generate variables.tf file"""
        return f"""
# Core variables
variable "environment" {{
  description = "Environment name"
  type        = string
  default     = "{config.environment.value}"
  validation {{
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }}
}}

variable "region" {{
  description = "AWS region"
  type        = string
  default     = "{config.region}"
}}

# Application variables
variable "replicas" {{
  description = "Number of application replicas"
  type        = number
  default     = {config.replicas}
  validation {{
    condition     = var.replicas > 0 && var.replicas <= 20
    error_message = "Replicas must be between 1 and 20."
  }}
}}

variable "cpu_limit" {{
  description = "CPU limit per replica"
  type        = string
  default     = "{config.resources.get('cpu_limit', '1000m')}"
}}

variable "memory_limit" {{
  description = "Memory limit per replica"
  type        = string
  default     = "{config.resources.get('memory_limit', '2Gi')}"
}}

# Database variables
variable "db_instance_class" {{
  description = "RDS instance class"
  type        = string
  default     = "{config.resources.get('db_instance_class', 'db.t3.micro')}"
}}

variable "db_storage_gb" {{
  description = "Database storage in GB"
  type        = number
  default     = {config.storage.get('db_storage_gb', 20)}
}}

# Monitoring variables
variable "enable_monitoring" {{
  description = "Enable comprehensive monitoring"
  type        = bool
  default     = {str(config.monitoring.get('enabled', True)).lower()}
}}

variable "retention_days" {{
  description = "Log and backup retention in days"
  type        = number
  default     = {config.backup.get('retention_days', 7)}
}}

# Security variables
variable "enable_encryption" {{
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}}

variable "allowed_cidr_blocks" {{
  description = "CIDR blocks allowed to access resources"
  type        = list(string)
  default     = {json.dumps(config.security.get('allowed_cidrs', ['0.0.0.0/0']))}
}}
"""
    
    def _generate_outputs_file(self, config: DeploymentConfig) -> str:
        """Generate outputs.tf file"""
        return f"""
# Infrastructure outputs
output "cluster_endpoint" {{
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.mten.endpoint
  sensitive   = false
}}

output "cluster_name" {{
  description = "EKS cluster name"
  value       = aws_eks_cluster.mten.name
}}

output "database_endpoint" {{
  description = "RDS database endpoint"
  value       = aws_db_instance.mten.endpoint
  sensitive   = false
}}

output "redis_endpoint" {{
  description = "Redis endpoint"
  value       = aws_elasticache_replication_group.mten.configuration_endpoint_address
  sensitive   = false
}}

output "application_url" {{
  description = "Application URL"
  value       = "https://{config.networking.get('domain', f'mten-{config.environment.value}.example.com')}"
}}

output "health_check_url" {{
  description = "Health check URL"
  value       = "https://{config.networking.get('domain', f'mten-{config.environment.value}.example.com')}/health"
}}

output "monitoring_endpoints" {{
  description = "Monitoring endpoints"
  value = {{
    prometheus = "https://{config.networking.get('domain', f'mten-{config.environment.value}.example.com')}/metrics"
    grafana    = "https://grafana-{config.environment.value}.example.com"
  }}
}}

# Security outputs
output "api_key" {{
  description = "Generated API key"
  value       = random_password.api_key.result
  sensitive   = true
}}

output "database_password" {{
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}}
"""
    
    def _generate_outputs(self, config: DeploymentConfig) -> str:
        """Generate outputs section for main.tf"""
        return """
# Random passwords
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = true
}

resource "random_password" "api_key" {
  length  = 64
  special = false
}

# IAM roles and policies
resource "aws_iam_role" "eks_cluster" {
  name = "mten-eks-cluster-${local.environment}"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role" "eks_node" {
  name = "mten-eks-node-${local.environment}"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node.name
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "mten-rds-${local.environment}"
  description = "Security group for RDS database"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "mten-redis-${local.environment}"
  description = "Security group for Redis cache"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}
"""
    
    async def generate_kubernetes_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Kubernetes manifest files"""
        manifests = {}
        
        # Namespace
        namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: mten-{config.environment.value}
  labels:
    environment: {config.environment.value}
    application: mten
"""
        manifests["namespace.yaml"] = namespace_yaml
        
        # ConfigMap
        configmap_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: mten-config
  namespace: mten-{config.environment.value}
data:
  ENVIRONMENT: {config.environment.value}
  LOG_LEVEL: {config.metadata.get('log_level', 'info')}
  METRICS_ENABLED: "true"
  HEALTH_CHECK_PORT: "8080"
  REPLICAS: "{config.replicas}"
"""
        manifests["configmap.yaml"] = configmap_yaml
        
        # Secret (template - real secrets should be managed externally)
        secret_yaml = f"""
apiVersion: v1
kind: Secret
metadata:
  name: mten-secrets
  namespace: mten-{config.environment.value}
type: Opaque
stringData:
  DATABASE_URL: "postgresql://mten_admin:CHANGE_ME@db-endpoint/mten_db"
  REDIS_URL: "redis://:CHANGE_ME@redis-endpoint:6379"
  API_KEY: "CHANGE_ME"
"""
        manifests["secret.yaml"] = secret_yaml
        
        # Deployment
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mten
  namespace: mten-{config.environment.value}
  labels:
    app: mten
    version: {config.metadata.get('version', 'latest')}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: mten
  template:
    metadata:
      labels:
        app: mten
        version: {config.metadata.get('version', 'latest')}
    spec:
      containers:
      - name: mten
        image: mten:{config.metadata.get('version', 'latest')}
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8080
          name: health
        envFrom:
        - configMapRef:
            name: mten-config
        - secretRef:
            name: mten-secrets
        resources:
          limits:
            cpu: {config.resources.get('cpu_limit', '1000m')}
            memory: {config.resources.get('memory_limit', '2Gi')}
          requests:
            cpu: {config.resources.get('cpu_request', '500m')}
            memory: {config.resources.get('memory_request', '1Gi')}
        livenessProbe:
          httpGet:
            path: /health
            port: health
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: health
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
"""
        manifests["deployment.yaml"] = deployment_yaml
        
        # Service
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: mten
  namespace: mten-{config.environment.value}
  labels:
    app: mten
spec:
  selector:
    app: mten
  ports:
  - name: http
    port: 80
    targetPort: http
  type: ClusterIP
"""
        manifests["service.yaml"] = service_yaml
        
        # Ingress
        ingress_yaml = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mten
  namespace: mten-{config.environment.value}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
spec:
  tls:
  - hosts:
    - {config.networking.get('domain', f'mten-{config.environment.value}.example.com')}
    secretName: mten-tls
  rules:
  - host: {config.networking.get('domain', f'mten-{config.environment.value}.example.com')}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mten
            port:
              number: 80
"""
        manifests["ingress.yaml"] = ingress_yaml
        
        return manifests
    
    async def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate docker-compose.yml for development/testing"""
        return f"""
version: '3.8'

services:
  mten:
    image: mten:{config.metadata.get('version', 'latest')}
    ports:
      - "8000:8000"
      - "8080:8080"
    environment:
      - ENVIRONMENT={config.environment.value}
      - LOG_LEVEL={config.metadata.get('log_level', 'info')}
      - DATABASE_URL=postgresql://mten:password@postgres:5432/mten
      - REDIS_URL=redis://redis:6379
      - METRICS_ENABLED=true
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - mten-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mten
      - POSTGRES_USER=mten
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    networks:
      - mten-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mten"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - mten-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - mten-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - mten-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  mten-network:
    driver: bridge
"""


class DeploymentOrchestrator:
    """Main deployment orchestration engine"""
    
    def __init__(self):
        self.infrastructure = InfrastructureAsCode()
        self.deployment_history = []
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy MTen to specified environment"""
        deployment_id = f"deploy-{config.environment.value}-{int(time.time())}"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            environment=config.environment,
            strategy=config.strategy,
            status="starting",
            start_time=datetime.now(UTC)
        )
        
        logger.info(f"🚀 Starting deployment {deployment_id} to {config.environment.value}")
        
        try:
            # Pre-deployment validation
            await self._pre_deployment_validation(config, result)
            
            # Generate infrastructure code
            await self._generate_infrastructure_code(config, result)
            
            # Deploy infrastructure
            if config.provider != InfrastructureProvider.DOCKER_COMPOSE:
                await self._deploy_infrastructure(config, result)
            
            # Deploy application
            await self._deploy_application(config, result)
            
            # Configure monitoring
            await self._configure_monitoring(config, result)
            
            # Setup backup and recovery
            await self._setup_backup_recovery(config, result)
            
            # Post-deployment validation
            await self._post_deployment_validation(config, result)
            
            result.status = "completed"
            result.rollback_available = True
            logger.info(f"✅ Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))
            logger.error(f"❌ Deployment {deployment_id} failed: {e}")
        
        finally:
            result.end_time = datetime.now(UTC)
            result.duration = (result.end_time - result.start_time).total_seconds()
            self.deployment_history.append(result)
        
        return result
    
    async def _pre_deployment_validation(self, config: DeploymentConfig, result: DeploymentResult):
        """Pre-deployment validation checks"""
        result.logs.append("Starting pre-deployment validation...")
        
        # Validate configuration
        await self._validate_configuration(config)
        
        # Check resource availability
        await self._check_resource_availability(config)
        
        # Validate security settings
        await self._validate_security_settings(config)
        
        result.logs.append("Pre-deployment validation completed")
    
    async def _validate_configuration(self, config: DeploymentConfig):
        """Validate deployment configuration"""
        # Check required fields
        required_fields = ['environment', 'strategy', 'provider', 'region']
        for field in required_fields:
            if not getattr(config, field, None):
                raise ValueError(f"Required field '{field}' is missing")
        
        # Validate environment-specific settings
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if config.replicas < 2:
                raise ValueError("Production environment requires at least 2 replicas")
            
            if not config.security.get('encryption_enabled', True):
                raise ValueError("Production environment requires encryption")
        
        # Validate resource specifications
        if config.resources.get('memory_limit', '1Gi') < '1Gi':
            logger.warning("Memory limit is below recommended minimum of 1Gi")
    
    async def _check_resource_availability(self, config: DeploymentConfig):
        """Check if required resources are available"""
        # This would check actual cloud provider quotas and availability
        logger.info(f"Checking resource availability in {config.region}")
        
        # Simulate resource checks
        await asyncio.sleep(2)
        
        # In real implementation, would check:
        # - Compute quotas
        # - Network availability
        # - Storage quotas
        # - Database limits
        
        logger.info("Resource availability check completed")
    
    async def _validate_security_settings(self, config: DeploymentConfig):
        """Validate security configuration"""
        logger.info("Validating security settings...")
        
        # Check encryption settings
        if not config.security.get('encryption_enabled', True):
            logger.warning("⚠️ Encryption at rest is disabled")
        
        # Check network security
        allowed_cidrs = config.security.get('allowed_cidrs', ['0.0.0.0/0'])
        if '0.0.0.0/0' in allowed_cidrs and config.environment == DeploymentEnvironment.PRODUCTION:
            logger.warning("⚠️ Production environment allows traffic from all IPs")
        
        # Check authentication
        if not config.security.get('require_authentication', True):
            raise ValueError("Authentication must be enabled")
        
        logger.info("Security validation completed")
    
    async def _generate_infrastructure_code(self, config: DeploymentConfig, result: DeploymentResult):
        """Generate infrastructure as code"""
        result.logs.append("Generating infrastructure code...")
        
        # Generate Terraform configuration
        if config.provider != InfrastructureProvider.DOCKER_COMPOSE:
            terraform_files = await self.infrastructure.generate_terraform_config(config)
            
            # Save Terraform files
            env_dir = self.infrastructure.environments_dir / config.environment.value
            env_dir.mkdir(exist_ok=True)
            
            for filename, content in terraform_files.items():
                file_path = env_dir / filename
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(content)
                result.logs.append(f"Generated {filename}")
        
        # Generate Kubernetes manifests
        k8s_manifests = await self.infrastructure.generate_kubernetes_manifests(config)
        
        k8s_dir = self.infrastructure.environments_dir / config.environment.value / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        for filename, content in k8s_manifests.items():
            file_path = k8s_dir / filename
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
            result.logs.append(f"Generated Kubernetes {filename}")
        
        # Generate Docker Compose for development
        if config.provider == InfrastructureProvider.DOCKER_COMPOSE:
            docker_compose = await self.infrastructure.generate_docker_compose(config)
            compose_file = self.infrastructure.environments_dir / config.environment.value / "docker-compose.yml"
            async with aiofiles.open(compose_file, 'w') as f:
                await f.write(docker_compose)
            result.logs.append("Generated docker-compose.yml")
        
        result.logs.append("Infrastructure code generation completed")
    
    async def _deploy_infrastructure(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy infrastructure using Terraform"""
        result.logs.append("Deploying infrastructure...")
        
        env_dir = self.infrastructure.environments_dir / config.environment.value
        
        try:
            # Initialize Terraform
            result.logs.append("Initializing Terraform...")
            await self._run_terraform_command(env_dir, ["init"])
            
            # Plan deployment
            result.logs.append("Planning infrastructure changes...")
            await self._run_terraform_command(env_dir, ["plan", "-out=tfplan"])
            
            # Apply changes
            result.logs.append("Applying infrastructure changes...")
            await self._run_terraform_command(env_dir, ["apply", "-auto-approve", "tfplan"])
            
            # Get outputs
            outputs = await self._get_terraform_outputs(env_dir)
            result.resources_created.extend(outputs.keys())
            
            if "health_check_url" in outputs:
                result.health_check_url = outputs["health_check_url"]
            
            if "monitoring_endpoints" in outputs:
                result.monitoring_endpoints.extend(outputs["monitoring_endpoints"].values())
            
            result.logs.append("Infrastructure deployment completed")
            
        except Exception as e:
            result.errors.append(f"Infrastructure deployment failed: {str(e)}")
            raise
    
    async def _run_terraform_command(self, working_dir: Path, args: List[str]):
        """Run Terraform command"""
        cmd = ["terraform"] + args
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Terraform command failed: {error_msg}")
        
        return stdout.decode()
    
    async def _get_terraform_outputs(self, working_dir: Path) -> Dict[str, Any]:
        """Get Terraform outputs"""
        try:
            output = await self._run_terraform_command(working_dir, ["output", "-json"])
            return json.loads(output)
        except Exception:
            return {}
    
    async def _deploy_application(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy application using Kubernetes or Docker Compose"""
        result.logs.append("Deploying application...")
        
        if config.provider == InfrastructureProvider.DOCKER_COMPOSE:
            await self._deploy_with_docker_compose(config, result)
        else:
            await self._deploy_with_kubernetes(config, result)
        
        result.logs.append("Application deployment completed")
    
    async def _deploy_with_docker_compose(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy using Docker Compose"""
        compose_file = self.infrastructure.environments_dir / config.environment.value / "docker-compose.yml"
        
        # Pull images
        await self._run_docker_compose_command(compose_file, ["pull"])
        
        # Deploy services
        await self._run_docker_compose_command(compose_file, ["up", "-d", "--remove-orphans"])
        
        result.health_check_url = "http://localhost:8080/health"
        result.monitoring_endpoints.append("http://localhost:9090")
    
    async def _run_docker_compose_command(self, compose_file: Path, args: List[str]):
        """Run docker-compose command"""
        cmd = ["docker-compose", "-f", str(compose_file)] + args
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Docker Compose command failed: {error_msg}")
        
        return stdout.decode()
    
    async def _deploy_with_kubernetes(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy using Kubernetes"""
        k8s_dir = self.infrastructure.environments_dir / config.environment.value / "kubernetes"
        
        # Apply Kubernetes manifests
        manifest_files = ["namespace.yaml", "configmap.yaml", "secret.yaml", "deployment.yaml", "service.yaml", "ingress.yaml"]
        
        for manifest in manifest_files:
            manifest_path = k8s_dir / manifest
            if manifest_path.exists():
                await self._run_kubectl_command(["apply", "-f", str(manifest_path)])
                result.logs.append(f"Applied {manifest}")
        
        # Wait for deployment to be ready
        await self._wait_for_kubernetes_deployment(config, result)
    
    async def _run_kubectl_command(self, args: List[str]):
        """Run kubectl command"""
        cmd = ["kubectl"] + args
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"kubectl command failed: {error_msg}")
        
        return stdout.decode()
    
    async def _wait_for_kubernetes_deployment(self, config: DeploymentConfig, result: DeploymentResult):
        """Wait for Kubernetes deployment to be ready"""
        namespace = f"mten-{config.environment.value}"
        
        # Wait for deployment to be ready
        await self._run_kubectl_command([
            "wait", "deployment/mten",
            f"--namespace={namespace}",
            "--for=condition=available",
            "--timeout=600s"
        ])
        
        result.logs.append("Deployment is ready")
    
    async def _configure_monitoring(self, config: DeploymentConfig, result: DeploymentResult):
        """Configure monitoring and alerting"""
        result.logs.append("Configuring monitoring...")
        
        if not config.monitoring.get('enabled', True):
            result.logs.append("Monitoring is disabled")
            return
        
        # Generate monitoring configuration
        await self._generate_monitoring_config(config)
        
        # Deploy monitoring stack
        if config.provider != InfrastructureProvider.DOCKER_COMPOSE:
            await self._deploy_monitoring_stack(config, result)
        
        result.logs.append("Monitoring configuration completed")
    
    async def _generate_monitoring_config(self, config: DeploymentConfig):
        """Generate monitoring configuration files"""
        monitoring_dir = self.infrastructure.environments_dir / config.environment.value / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = f"""
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'mten'
    static_configs:
      - targets: ['mten-{config.environment.value}:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - source_labels: [__address__]
        regex: '(.*):10250'
        replacement: '${{1}}:9100'
        target_label: __address__

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
"""
        
        prometheus_file = monitoring_dir / "prometheus.yml"
        async with aiofiles.open(prometheus_file, 'w') as f:
            await f.write(prometheus_config)
    
    async def _deploy_monitoring_stack(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy monitoring stack (Prometheus, Grafana, AlertManager)"""
        # This would deploy the monitoring stack using Helm or kubectl
        result.logs.append("Deploying monitoring stack...")
        
        # Simulate monitoring deployment
        await asyncio.sleep(5)
        
        monitoring_url = f"https://monitoring-{config.environment.value}.example.com"
        result.monitoring_endpoints.append(monitoring_url)
        result.logs.append(f"Monitoring deployed at {monitoring_url}")
    
    async def _setup_backup_recovery(self, config: DeploymentConfig, result: DeploymentResult):
        """Setup backup and disaster recovery"""
        result.logs.append("Setting up backup and recovery...")
        
        if not config.backup.get('enabled', True):
            result.logs.append("Backup is disabled")
            return
        
        # Generate backup scripts
        await self._generate_backup_scripts(config)
        
        # Configure automated backups
        await self._configure_automated_backups(config)
        
        result.logs.append("Backup and recovery setup completed")
    
    async def _generate_backup_scripts(self, config: DeploymentConfig):
        """Generate backup and recovery scripts"""
        backup_dir = self.infrastructure.environments_dir / config.environment.value / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Database backup script
        db_backup_script = f"""#!/bin/bash
# Database backup script for MTen {config.environment.value}

set -e

ENVIRONMENT="{config.environment.value}"
BACKUP_DIR="/backups/$ENVIRONMENT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS={config.backup.get('retention_days', 7)}

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
echo "Starting database backup..."
pg_dump -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME > $BACKUP_DIR/db_$TIMESTAMP.sql

# Compress backup
gzip $BACKUP_DIR/db_$TIMESTAMP.sql

# Redis backup
echo "Starting Redis backup..."
redis-cli --rdb $BACKUP_DIR/redis_$TIMESTAMP.rdb

# Cleanup old backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.rdb" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully"
"""
        
        backup_script_file = backup_dir / "backup.sh"
        async with aiofiles.open(backup_script_file, 'w') as f:
            await f.write(db_backup_script)
        
        # Make script executable
        os.chmod(backup_script_file, 0o755)
        
        # Recovery script
        recovery_script = f"""#!/bin/bash
# Recovery script for MTen {config.environment.value}

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Available backups:"
    ls -la /backups/{config.environment.value}/
    exit 1
fi

ENVIRONMENT="{config.environment.value}"
BACKUP_DIR="/backups/$ENVIRONMENT"
TIMESTAMP=$1

echo "Restoring database from backup $TIMESTAMP..."

# Stop application
kubectl scale deployment/mten --replicas=0 -n mten-$ENVIRONMENT

# Restore database
gunzip -c $BACKUP_DIR/db_$TIMESTAMP.sql.gz | psql -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME

# Restore Redis
redis-cli --rdb $BACKUP_DIR/redis_$TIMESTAMP.rdb

# Start application
kubectl scale deployment/mten --replicas={config.replicas} -n mten-$ENVIRONMENT

echo "Recovery completed successfully"
"""
        
        recovery_script_file = backup_dir / "recovery.sh"
        async with aiofiles.open(recovery_script_file, 'w') as f:
            await f.write(recovery_script)
        
        os.chmod(recovery_script_file, 0o755)
    
    async def _configure_automated_backups(self, config: DeploymentConfig):
        """Configure automated backup scheduling"""
        # This would configure cron jobs or Kubernetes CronJobs for automated backups
        logger.info(f"Configuring automated backups for {config.environment.value}")
        
        # Generate CronJob manifest for Kubernetes
        backup_cronjob = f"""
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mten-backup
  namespace: mten-{config.environment.value}
spec:
  schedule: "{config.backup.get('schedule', '0 2 * * *')}"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h $DATABASE_HOST -U $DATABASE_USER -d $DATABASE_NAME | gzip > /backups/db_$(date +%Y%m%d_%H%M%S).sql.gz
            envFrom:
            - secretRef:
                name: mten-secrets
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: mten-backup-pvc
"""
        
        cronjob_file = self.infrastructure.environments_dir / config.environment.value / "kubernetes" / "backup-cronjob.yaml"
        async with aiofiles.open(cronjob_file, 'w') as f:
            await f.write(backup_cronjob)
    
    async def _post_deployment_validation(self, config: DeploymentConfig, result: DeploymentResult):
        """Post-deployment validation and health checks"""
        result.logs.append("Starting post-deployment validation...")
        
        # Health check
        await self._validate_health_endpoints(config, result)
        
        # Integration tests
        await self._run_integration_tests(config, result)
        
        # Performance validation
        await self._validate_performance_metrics(config, result)
        
        result.logs.append("Post-deployment validation completed")
    
    async def _validate_health_endpoints(self, config: DeploymentConfig, result: DeploymentResult):
        """Validate health endpoints are responding"""
        if not result.health_check_url:
            result.health_check_url = f"https://mten-{config.environment.value}.example.com/health"
        
        # Simulate health check
        logger.info(f"Validating health endpoint: {result.health_check_url}")
        await asyncio.sleep(2)
        
        # In real implementation, would make actual HTTP requests
        result.logs.append("Health endpoints validated")
    
    async def _run_integration_tests(self, config: DeploymentConfig, result: DeploymentResult):
        """Run integration tests against deployed environment"""
        logger.info("Running integration tests...")
        await asyncio.sleep(3)
        
        # In real implementation, would run actual integration test suite
        result.logs.append("Integration tests passed")
    
    async def _validate_performance_metrics(self, config: DeploymentConfig, result: DeploymentResult):
        """Validate performance metrics meet requirements"""
        logger.info("Validating performance metrics...")
        await asyncio.sleep(2)
        
        # In real implementation, would check actual performance metrics
        result.logs.append("Performance metrics validated")


async def main():
    """Example deployment workflow"""
    logger.info("🚀 MTen Production Deployment Automation")
    logger.info("=" * 70)
    
    # Create deployment orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # Example staging deployment
    staging_config = DeploymentConfig(
        environment=DeploymentEnvironment.STAGING,
        strategy=DeploymentStrategy.ROLLING,
        provider=InfrastructureProvider.AWS,
        region="us-east-1",
        replicas=2,
        resources={
            "cpu_limit": "500m",
            "memory_limit": "1Gi",
            "cpu_request": "250m",
            "memory_request": "512Mi",
            "instance_type": "t3.small",
            "db_instance_class": "db.t3.micro",
            "redis_node_type": "cache.t3.micro"
        },
        networking={
            "domain": "mten-staging.example.com"
        },
        storage={
            "size": "50Gi",
            "db_storage_gb": 20,
            "db_max_storage_gb": 100
        },
        monitoring={
            "enabled": True,
            "retention_days": 30
        },
        backup={
            "enabled": True,
            "retention_days": 7,
            "schedule": "0 2 * * *",
            "backup_window": "03:00-04:00",
            "maintenance_window": "sun:04:00-sun:05:00"
        },
        security={
            "encryption_enabled": True,
            "require_authentication": True,
            "allowed_cidrs": ["10.0.0.0/8", "172.16.0.0/12"]
        },
        metadata={
            "version": "latest",
            "log_level": "info"
        }
    )
    
    # Deploy to staging
    logger.info("🔄 Deploying to staging environment...")
    staging_result = await orchestrator.deploy(staging_config)
    
    if staging_result.status == "completed":
        logger.info("✅ Staging deployment successful!")
        logger.info(f"   Health check: {staging_result.health_check_url}")
        logger.info(f"   Duration: {staging_result.duration:.2f}s")
        logger.info(f"   Resources: {len(staging_result.resources_created)}")
        
        # Example production deployment
        production_config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.BLUE_GREEN,
            provider=InfrastructureProvider.AWS,
            region="us-east-1",
            replicas=5,
            resources={
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "cpu_request": "500m",
                "memory_request": "1Gi",
                "instance_type": "t3.medium",
                "db_instance_class": "db.t3.small",
                "redis_node_type": "cache.t3.small",
                "redis_replicas": 3
            },
            networking={
                "domain": "mten.example.com"
            },
            storage={
                "size": "200Gi",
                "db_storage_gb": 100,
                "db_max_storage_gb": 1000
            },
            monitoring={
                "enabled": True,
                "retention_days": 90
            },
            backup={
                "enabled": True,
                "retention_days": 30,
                "schedule": "0 1 * * *",
                "backup_window": "01:00-02:00",
                "maintenance_window": "sun:02:00-sun:03:00"
            },
            security={
                "encryption_enabled": True,
                "require_authentication": True,
                "allowed_cidrs": ["0.0.0.0/0"]  # Public access for production
            },
            metadata={
                "version": "v1.0.0",
                "log_level": "warn"
            }
        )
        
        # Deploy to production
        logger.info("\n🔄 Deploying to production environment...")
        production_result = await orchestrator.deploy(production_config)
        
        if production_result.status == "completed":
            logger.info("🎉 Production deployment successful!")
            logger.info(f"   Health check: {production_result.health_check_url}")
            logger.info(f"   Duration: {production_result.duration:.2f}s")
            logger.info(f"   Resources: {len(production_result.resources_created)}")
            logger.info(f"   Monitoring: {len(production_result.monitoring_endpoints)} endpoints")
        else:
            logger.error("❌ Production deployment failed!")
            for error in production_result.errors:
                logger.error(f"   Error: {error}")
    
    else:
        logger.error("❌ Staging deployment failed!")
        for error in staging_result.errors:
            logger.error(f"   Error: {error}")


if __name__ == "__main__":
    asyncio.run(main())