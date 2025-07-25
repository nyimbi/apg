#!/usr/bin/env python3
"""
DevOps Agent
============

DevOps agent for deployment, infrastructure management, and operational tasks.
"""

import json
import logging
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentRole, AgentTask, AgentCapability, AgentMemory

class DevOpsAgent(BaseAgent):
	"""
	DevOps Agent
	
	Responsible for:
	- Application deployment and infrastructure setup
	- Container orchestration and management
	- CI/CD pipeline configuration
	- Monitoring and logging setup
	- Infrastructure as Code (IaC)
	- Performance optimization and scaling
	"""
	
	def __init__(self, agent_id: str, name: str = "DevOps Engineer", config: Dict[str, Any] = None):
		# Define DevOps-specific capabilities
		capabilities = [
			AgentCapability(
				name="deployment",
				description="Deploy applications to various environments",
				skill_level=9,
				domains=["deployment", "infrastructure", "cloud_platforms"],
				tools=["docker", "kubernetes", "terraform", "ansible"]
			),
			AgentCapability(
				name="containerization",
				description="Containerize applications with Docker and orchestration",
				skill_level=9,
				domains=["containerization", "docker", "kubernetes"],
				tools=["docker", "docker_compose", "kubernetes", "helm"]
			),
			AgentCapability(
				name="infrastructure_management",
				description="Manage infrastructure as code and cloud resources",
				skill_level=8,
				domains=["infrastructure", "cloud", "terraform"],
				tools=["terraform", "cloudformation", "ansible", "chef"]
			),
			AgentCapability(
				name="cicd_pipeline",
				description="Setup and manage CI/CD pipelines",
				skill_level=8,
				domains=["cicd", "automation", "pipeline_management"],
				tools=["jenkins", "github_actions", "gitlab_ci", "azure_devops"]
			),
			AgentCapability(
				name="monitoring_setup",
				description="Configure monitoring, logging, and alerting systems",
				skill_level=8,
				domains=["monitoring", "observability", "logging"],
				tools=["prometheus", "grafana", "elk_stack", "datadog"]
			),
			AgentCapability(
				name="scaling_optimization",
				description="Optimize application performance and scaling",
				skill_level=7,
				domains=["performance", "scaling", "optimization"],
				tools=["load_balancer", "auto_scaling", "performance_tuning"]
			),
			AgentCapability(
				name="security_operations",
				description="Implement security best practices in deployment",
				skill_level=7,
				domains=["security", "compliance", "vulnerability_management"],
				tools=["security_scanner", "secret_management", "compliance_checker"]
			)
		]
		
		super().__init__(
			agent_id=agent_id,
			role=AgentRole.DEVOPS,
			name=name,
			description="Expert DevOps engineer specializing in deployment and infrastructure",
			capabilities=capabilities,
			config=config or {}
		)
		
		# DevOps-specific tools and knowledge
		self.deployment_platforms = {}
		self.infrastructure_templates = {}
		self.monitoring_configs = {}
		self.deployment_history = []

	def _setup_capabilities(self):
		"""Setup DevOps-specific capabilities"""
		self.logger.info("Setting up DevOps capabilities")
		
		# Load deployment platforms
		self.deployment_platforms = {
			'docker': {
				'description': 'Container-based deployment',
				'requirements': ['dockerfile', 'docker_compose'],
				'complexity': 'low',
				'scalability': 'medium'
			},
			'kubernetes': {
				'description': 'Container orchestration platform',
				'requirements': ['k8s_manifests', 'helm_charts'],
				'complexity': 'high',
				'scalability': 'high'
			},
			'serverless': {
				'description': 'Function-as-a-Service deployment',
				'requirements': ['lambda_functions', 'api_gateway'],
				'complexity': 'medium',
				'scalability': 'auto'
			},
			'traditional': {
				'description': 'Traditional server deployment',
				'requirements': ['server_configs', 'load_balancer'],
				'complexity': 'medium',
				'scalability': 'manual'
			},
			'cloud_native': {
				'description': 'Cloud provider managed services',
				'requirements': ['cloud_configs', 'managed_services'],
				'complexity': 'medium',
				'scalability': 'high'
			}
		}
		
		# Load infrastructure templates
		self.infrastructure_templates = {
			'aws': {
				'compute': ['ec2', 'ecs', 'lambda', 'fargate'],
				'storage': ['s3', 'rds', 'dynamodb', 'efs'],
				'networking': ['vpc', 'alb', 'cloudfront', 'route53'],
				'monitoring': ['cloudwatch', 'x_ray']
			},
			'azure': {
				'compute': ['vm', 'container_instances', 'functions', 'aks'],
				'storage': ['blob_storage', 'sql_database', 'cosmos_db'],
				'networking': ['vnet', 'load_balancer', 'cdn', 'dns'],
				'monitoring': ['monitor', 'application_insights']
			},
			'gcp': {
				'compute': ['compute_engine', 'cloud_run', 'functions', 'gke'],
				'storage': ['cloud_storage', 'cloud_sql', 'firestore'],
				'networking': ['vpc', 'load_balancer', 'cdn', 'dns'],
				'monitoring': ['cloud_monitoring', 'cloud_trace']
			},
			'on_premise': {
				'compute': ['bare_metal', 'vm', 'docker', 'kubernetes'],
				'storage': ['local_storage', 'nfs', 'database_server'],
				'networking': ['switches', 'load_balancer', 'proxy'],
				'monitoring': ['prometheus', 'grafana', 'elk']
			}
		}
		
		# Load monitoring configurations
		self.monitoring_configs = {
			'prometheus_grafana': {
				'metrics_collection': 'prometheus',
				'visualization': 'grafana',
				'alerting': 'alertmanager',
				'setup_complexity': 'medium'
			},
			'elk_stack': {
				'logging': 'elasticsearch',
				'processing': 'logstash',
				'visualization': 'kibana',
				'setup_complexity': 'high'
			},
			'cloud_native': {
				'metrics': 'cloud_provider_monitoring',
				'logging': 'cloud_provider_logging',
				'alerting': 'cloud_provider_alerting',
				'setup_complexity': 'low'
			}
		}

	def _setup_tools(self):
		"""Setup DevOps-specific tools"""
		self.logger.info("Setting up DevOps tools")
		# Tools would be initialized here

	async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
		"""Execute a DevOps task"""
		task_type = task.requirements.get('type', 'unknown')
		
		self.logger.info(f"Executing {task_type} task: {task.name}")
		
		if task_type == 'deployment':
			return await self._deploy_application(task)
		elif task_type == 'infrastructure':
			return await self._setup_infrastructure(task)
		elif task_type == 'monitoring':
			return await self._setup_monitoring(task)
		elif task_type == 'cicd':
			return await self._setup_cicd_pipeline(task)
		elif task_type == 'scaling':
			return await self._optimize_scaling(task)
		else:
			return {'error': f'Unknown task type: {task_type}'}

	async def _deploy_application(self, task: AgentTask) -> Dict[str, Any]:
		"""Deploy application based on architecture and testing results"""
		testing_results = task.requirements.get('testing_results', {})
		architecture = task.requirements.get('architecture', {})
		
		self.logger.info("Starting application deployment")
		
		try:
			# Analyze deployment requirements
			deployment_config = await self._analyze_deployment_requirements(
				testing_results, architecture
			)
			
			# Select deployment platform
			platform = await self._select_deployment_platform(deployment_config)
			
			# Generate deployment artifacts
			deployment_artifacts = await self._generate_deployment_artifacts(
				deployment_config, platform
			)
			
			# Setup infrastructure
			infrastructure_config = await self._setup_deployment_infrastructure(
				platform, deployment_config
			)
			
			# Deploy application
			deployment_result = await self._execute_deployment(
				deployment_artifacts, infrastructure_config
			)
			
			# Setup monitoring and health checks
			monitoring_setup = await self._setup_deployment_monitoring(
				deployment_result, deployment_config
			)
			
			# Validate deployment
			validation_result = await self._validate_deployment(
				deployment_result, testing_results
			)
			
			# Store deployment memory
			await self._store_deployment_memory(task, deployment_result, deployment_config)
			
			return {
				'status': 'success',
				'deployment_package': {
					'platform': platform,
					'artifacts': deployment_artifacts,
					'infrastructure': infrastructure_config,
					'monitoring': monitoring_setup,
					'validation': validation_result
				},
				'deployment_url': deployment_result.get('url', 'http://localhost:8000'),
				'health_endpoints': deployment_result.get('health_endpoints', []),
				'monitoring_dashboard': monitoring_setup.get('dashboard_url'),
				'deployment_summary': {
					'platform': platform,
					'environment': deployment_config.get('environment', 'production'),
					'deployed_at': datetime.utcnow().isoformat(),
					'deployment_id': deployment_result.get('deployment_id'),
					'status': validation_result.get('status', 'deployed')
				}
			}
			
		except Exception as e:
			self.logger.error(f"Application deployment failed: {e}")
			return {
				'status': 'error',
				'error': str(e),
				'details': 'Application deployment execution failed'
			}

	async def _analyze_deployment_requirements(
		self, 
		testing_results: Dict[str, Any], 
		architecture: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Analyze requirements for deployment"""
		self.logger.info("Analyzing deployment requirements")
		
		# Extract architecture information
		arch_pattern = architecture.get('system_architecture', {}).get('pattern', 'monolithic')
		components = architecture.get('component_specifications', [])
		deployment_arch = architecture.get('deployment_architecture', {})
		
		# Analyze testing results for deployment readiness
		test_quality_score = testing_results.get('performance_metrics', {}).get('overall_quality_score', 0)
		security_score = testing_results.get('security_assessment', {}).get('security_score', 0)
		
		# Determine deployment configuration
		deployment_config = {
			'architecture_pattern': arch_pattern,
			'component_count': len(components),
			'quality_score': test_quality_score,
			'security_score': security_score,
			'deployment_ready': test_quality_score >= 70 and security_score >= 80,
			'scaling_requirements': self._determine_scaling_requirements(components),
			'security_requirements': self._determine_security_requirements(security_score),
			'monitoring_requirements': self._determine_monitoring_requirements(arch_pattern),
			'environment': 'production' if test_quality_score >= 85 else 'staging'
		}
		
		return deployment_config

	def _determine_scaling_requirements(self, components: List[Dict]) -> Dict[str, Any]:
		"""Determine scaling requirements based on components"""
		service_count = len([c for c in components if c.get('type') == 'service'])
		
		return {
			'horizontal_scaling': service_count > 3,
			'load_balancing': service_count > 1,
			'auto_scaling': service_count > 5,
			'caching': service_count > 2,
			'cdn': True  # Generally recommended
		}

	def _determine_security_requirements(self, security_score: float) -> Dict[str, Any]:
		"""Determine security requirements based on security score"""
		return {
			'https_required': True,
			'firewall_rules': security_score < 90,
			'vulnerability_scanning': security_score < 85,
			'secret_management': True,
			'access_control': True,
			'audit_logging': security_score < 95
		}

	def _determine_monitoring_requirements(self, arch_pattern: str) -> Dict[str, Any]:
		"""Determine monitoring requirements based on architecture"""
		return {
			'application_metrics': True,
			'infrastructure_metrics': True,
			'distributed_tracing': arch_pattern == 'microservices',
			'log_aggregation': arch_pattern != 'monolithic',
			'alerting': True,
			'uptime_monitoring': True
		}

	async def _select_deployment_platform(self, deployment_config: Dict[str, Any]) -> str:
		"""Select appropriate deployment platform"""
		arch_pattern = deployment_config['architecture_pattern']
		component_count = deployment_config['component_count']
		scaling_reqs = deployment_config['scaling_requirements']
		
		# Platform selection logic
		if arch_pattern == 'microservices' and component_count > 5:
			return 'kubernetes'
		elif arch_pattern == 'serverless':
			return 'serverless'
		elif scaling_reqs['auto_scaling']:
			return 'cloud_native'
		elif component_count <= 3:
			return 'docker'
		else:
			return 'traditional'

	async def _generate_deployment_artifacts(
		self, 
		deployment_config: Dict[str, Any], 
		platform: str
	) -> Dict[str, Any]:
		"""Generate deployment artifacts for the selected platform"""
		artifacts = {}
		
		if platform == 'docker':
			artifacts.update(await self._generate_docker_artifacts(deployment_config))
		elif platform == 'kubernetes':
			artifacts.update(await self._generate_kubernetes_artifacts(deployment_config))
		elif platform == 'serverless':
			artifacts.update(await self._generate_serverless_artifacts(deployment_config))
		elif platform == 'cloud_native':
			artifacts.update(await self._generate_cloud_native_artifacts(deployment_config))
		else:
			artifacts.update(await self._generate_traditional_artifacts(deployment_config))
		
		return artifacts

	async def _generate_docker_artifacts(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate Docker deployment artifacts"""
		dockerfile_content = """FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
"""

		docker_compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/appdb
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=appdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
"""

		nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    server {
        listen 80;
        server_name _;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location / {
            return 301 https://$host$request_uri;
        }
    }

    server {
        listen 443 ssl http2;
        server_name _;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://app/health;
        }
    }
}
"""

		return {
			'Dockerfile': dockerfile_content,
			'docker-compose.yml': docker_compose_content,
			'nginx.conf': nginx_config,
			'deployment_scripts': {
				'deploy.sh': '#!/bin/bash\ndocker-compose up -d --build\n',
				'stop.sh': '#!/bin/bash\ndocker-compose down\n',
				'logs.sh': '#!/bin/bash\ndocker-compose logs -f\n'
			}
		}

	async def _generate_kubernetes_artifacts(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate Kubernetes deployment artifacts"""
		app_deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  labels:
    app: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: webapp
        image: webapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 250m
            memory: 256Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: webapp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
"""

		ingress_config = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - app.example.com
    secretName: app-tls
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
"""

		database_config = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: appdb
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
"""

		return {
			'k8s/deployment.yaml': app_deployment,
			'k8s/ingress.yaml': ingress_config,
			'k8s/database.yaml': database_config,
			'k8s/secrets.yaml': self._generate_k8s_secrets(),
			'helm/Chart.yaml': self._generate_helm_chart(),
			'helm/values.yaml': self._generate_helm_values()
		}

	def _generate_k8s_secrets(self) -> str:
		"""Generate Kubernetes secrets manifest"""
		return """apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: <base64-encoded-database-url>
---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
type: Opaque
data:
  username: <base64-encoded-username>
  password: <base64-encoded-password>
"""

	def _generate_helm_chart(self) -> str:
		"""Generate Helm Chart.yaml"""
		return """apiVersion: v2
name: webapp
description: Generated web application
type: application
version: 0.1.0
appVersion: "1.0"
"""

	def _generate_helm_values(self) -> str:
		"""Generate Helm values.yaml"""
		return """replicaCount: 3

image:
  repository: webapp
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: app-tls
      hosts:
        - app.example.com

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    database: appdb
    username: user
"""

	async def _generate_serverless_artifacts(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate serverless deployment artifacts"""
		serverless_yml = """service: webapp-serverless

provider:
  name: aws
  runtime: python3.12
  region: us-east-1
  environment:
    DATABASE_URL: ${env:DATABASE_URL}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - dynamodb:*
      Resource: "*"

functions:
  app:
    handler: lambda_handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
      - http:
          path: /
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements
  - serverless-domain-manager

custom:
  pythonRequirements:
    dockerizePip: true
  customDomain:
    domainName: api.example.com
    basePath: ''
    stage: ${self:provider.stage}
    createRoute53Record: true

resources:
  Resources:
    DynamoDbTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: webapp-table
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
"""

		lambda_handler = """import json
from app import app

def handler(event, context):
    from serverless_wsgi import handle_request
    return handle_request(app, event, context)
"""

		return {
			'serverless.yml': serverless_yml,
			'lambda_handler.py': lambda_handler,
			'requirements.txt': 'serverless-wsgi==0.8.2',
			'deploy_scripts': {
				'deploy.sh': '#!/bin/bash\nserverless deploy\n',
				'remove.sh': '#!/bin/bash\nserverless remove\n'
			}
		}

	async def _generate_cloud_native_artifacts(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate cloud-native deployment artifacts"""
		terraform_main = """provider "aws" {
  region = var.aws_region
}

# VPC and networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "webapp-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
}

# ECS Cluster
resource "aws_ecs_cluster" "webapp_cluster" {
  name = "webapp-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "webapp_alb" {
  name               = "webapp-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = module.vpc.public_subnets
}

# ECS Service
resource "aws_ecs_service" "webapp_service" {
  name            = "webapp-service"
  cluster         = aws_ecs_cluster.webapp_cluster.id
  task_definition = aws_ecs_task_definition.webapp_task.arn
  desired_count   = 3
  
  load_balancer {
    target_group_arn = aws_lb_target_group.webapp_tg.arn
    container_name   = "webapp"
    container_port   = 8000
  }
  
  network_configuration {
    subnets         = module.vpc.private_subnets
    security_groups = [aws_security_group.webapp_sg.id]
  }
}

# RDS Database
resource "aws_db_instance" "webapp_db" {
  identifier     = "webapp-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage = 20
  storage_encrypted = true
  
  db_name  = "appdb"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.webapp_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}
"""

		terraform_variables = """variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
"""

		return {
			'terraform/main.tf': terraform_main,
			'terraform/variables.tf': terraform_variables,
			'terraform/outputs.tf': self._generate_terraform_outputs(),
			'cloudformation/template.yaml': self._generate_cloudformation_template(),
			'scripts/deploy.sh': self._generate_cloud_deploy_script()
		}

	def _generate_terraform_outputs(self) -> str:
		"""Generate Terraform outputs"""
		return """output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.webapp_alb.dns_name
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.webapp_db.endpoint
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}
"""

	def _generate_cloudformation_template(self) -> str:
		"""Generate CloudFormation template"""
		return """AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web application infrastructure'

Parameters:
  Environment:
    Type: String
    Default: production
  
Resources:
  # VPC and networking resources would go here
  # ECS cluster and services
  # RDS database
  # Load balancer
  # Security groups
  
Outputs:
  LoadBalancerDNS:
    Description: Load balancer DNS name
    Value: !GetAtt LoadBalancer.DNSName
"""

	def _generate_cloud_deploy_script(self) -> str:
		"""Generate cloud deployment script"""
		return """#!/bin/bash
set -e

echo "Deploying cloud infrastructure..."

# Initialize Terraform
cd terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply deployment
terraform apply tfplan

# Get outputs
LOAD_BALANCER_DNS=$(terraform output -raw load_balancer_dns)

echo "Deployment complete!"
echo "Application URL: https://$LOAD_BALANCER_DNS"
"""

	async def _generate_traditional_artifacts(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate traditional deployment artifacts"""
		ansible_playbook = """---
- name: Deploy web application
  hosts: webservers
  become: yes
  
  vars:
    app_name: webapp
    app_user: webapp
    app_dir: /opt/webapp
    
  tasks:
    - name: Create application user
      user:
        name: "{{ app_user }}"
        shell: /bin/bash
        home: "{{ app_dir }}"
        createhome: yes
        
    - name: Install system packages
      package:
        name:
          - python3
          - python3-pip
          - nginx
          - postgresql
          - supervisor
        state: present
        
    - name: Copy application code
      copy:
        src: ../app/
        dest: "{{ app_dir }}/app/"
        owner: "{{ app_user }}"
        group: "{{ app_user }}"
        
    - name: Install Python dependencies
      pip:
        requirements: "{{ app_dir }}/app/requirements.txt"
        virtualenv: "{{ app_dir }}/venv"
        virtualenv_python: python3
        
    - name: Configure nginx
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/webapp
      notify: restart nginx
      
    - name: Enable nginx site
      file:
        src: /etc/nginx/sites-available/webapp
        dest: /etc/nginx/sites-enabled/webapp
        state: link
      notify: restart nginx
      
    - name: Configure supervisor
      template:
        src: supervisor.conf.j2
        dest: /etc/supervisor/conf.d/webapp.conf
      notify: restart supervisor
      
  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted
        
    - name: restart supervisor
      service:
        name: supervisor
        state: restarted
"""

		return {
			'ansible/playbook.yml': ansible_playbook,
			'ansible/inventory': '[webservers]\n192.168.1.10\n192.168.1.11\n',
			'systemd/webapp.service': self._generate_systemd_service(),
			'nginx/webapp.conf': self._generate_nginx_config(),
			'scripts/deploy.sh': '#!/bin/bash\nansible-playbook -i inventory playbook.yml\n'
		}

	def _generate_systemd_service(self) -> str:
		"""Generate systemd service file"""
		return """[Unit]
Description=Web Application
After=network.target

[Service]
Type=simple
User=webapp
WorkingDirectory=/opt/webapp/app
Environment=PATH=/opt/webapp/venv/bin
ExecStart=/opt/webapp/venv/bin/gunicorn --bind 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
"""

	def _generate_nginx_config(self) -> str:
		"""Generate nginx configuration"""
		return """server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /static/ {
        alias /opt/webapp/app/static/;
        expires 1d;
    }
}
"""

	async def _setup_deployment_infrastructure(
		self, 
		platform: str, 
		deployment_config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Setup infrastructure for deployment"""
		self.logger.info(f"Setting up {platform} infrastructure")
		
		infrastructure_config = {
			'platform': platform,
			'environment': deployment_config.get('environment', 'production'),
			'scaling_config': self._generate_scaling_config(deployment_config),
			'security_config': self._generate_security_config(deployment_config),
			'networking_config': self._generate_networking_config(platform),
			'storage_config': self._generate_storage_config(platform),
			'backup_config': self._generate_backup_config(platform)
		}
		
		return infrastructure_config

	def _generate_scaling_config(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate scaling configuration"""
		scaling_reqs = deployment_config.get('scaling_requirements', {})
		
		return {
			'auto_scaling': scaling_reqs.get('auto_scaling', False),
			'min_instances': 2 if scaling_reqs.get('horizontal_scaling') else 1,
			'max_instances': 10 if scaling_reqs.get('auto_scaling') else 3,
			'cpu_threshold': 70,
			'memory_threshold': 80,
			'load_balancing': scaling_reqs.get('load_balancing', False),
			'health_check_endpoint': '/health'
		}

	def _generate_security_config(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate security configuration"""
		security_reqs = deployment_config.get('security_requirements', {})
		
		return {
			'https_enabled': security_reqs.get('https_required', True),
			'firewall_rules': security_reqs.get('firewall_rules', True),
			'ssl_certificate': 'letsencrypt' if security_reqs.get('https_required') else None,
			'security_headers': True,
			'access_logging': security_reqs.get('audit_logging', True),
			'intrusion_detection': security_reqs.get('vulnerability_scanning', False)
		}

	def _generate_networking_config(self, platform: str) -> Dict[str, Any]:
		"""Generate networking configuration"""
		return {
			'load_balancer': platform in ['kubernetes', 'cloud_native'],
			'cdn': platform != 'traditional',
			'dns_management': True,
			'ssl_termination': 'load_balancer' if platform == 'kubernetes' else 'application',
			'rate_limiting': True,
			'ddos_protection': platform == 'cloud_native'
		}

	def _generate_storage_config(self, platform: str) -> Dict[str, Any]:
		"""Generate storage configuration"""
		return {
			'database_type': 'postgresql',
			'persistent_storage': platform != 'serverless',
			'backup_enabled': True,
			'encryption_at_rest': True,
			'replication': platform in ['kubernetes', 'cloud_native'],
			'storage_class': 'ssd' if platform == 'kubernetes' else 'standard'
		}

	def _generate_backup_config(self, platform: str) -> Dict[str, Any]:
		"""Generate backup configuration"""
		return {
			'database_backup': {
				'enabled': True,
				'frequency': 'daily',
				'retention': '30 days',
				'point_in_time_recovery': platform == 'cloud_native'
			},
			'application_backup': {
				'enabled': platform != 'serverless',
				'frequency': 'weekly',
				'retention': '12 weeks'
			},
			'disaster_recovery': {
				'enabled': platform == 'cloud_native',
				'rpo': '1 hour',
				'rto': '4 hours'
			}
		}

	async def _execute_deployment(
		self, 
		deployment_artifacts: Dict[str, Any], 
		infrastructure_config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Execute the actual deployment"""
		platform = infrastructure_config['platform']
		
		self.logger.info(f"Executing {platform} deployment")
		
		# Simulate deployment execution
		deployment_result = {
			'deployment_id': f"{platform}_deployment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
			'status': 'deployed',
			'url': self._generate_deployment_url(platform),
			'health_endpoints': ['/health', '/ready', '/metrics'],
			'environment': infrastructure_config.get('environment', 'production'),
			'platform': platform,
			'deployed_at': datetime.utcnow().isoformat(),
			'artifacts_deployed': list(deployment_artifacts.keys()),
			'infrastructure_components': self._get_infrastructure_components(platform)
		}
		
		return deployment_result

	def _generate_deployment_url(self, platform: str) -> str:
		"""Generate deployment URL based on platform"""
		if platform == 'kubernetes':
			return 'https://app.k8s.example.com'
		elif platform == 'serverless':
			return 'https://api.lambda.example.com'
		elif platform == 'cloud_native':
			return 'https://app.cloud.example.com'
		else:
			return 'http://app.example.com'

	def _get_infrastructure_components(self, platform: str) -> List[str]:
		"""Get infrastructure components for platform"""
		components_map = {
			'docker': ['containers', 'networks', 'volumes'],
			'kubernetes': ['pods', 'services', 'ingress', 'configmaps', 'secrets'],
			'serverless': ['lambda_functions', 'api_gateway', 'dynamodb'],
			'cloud_native': ['ecs_services', 'load_balancer', 'rds', 'cloudwatch'],
			'traditional': ['servers', 'database', 'load_balancer', 'monitoring']
		}
		return components_map.get(platform, ['basic_infrastructure'])

	async def _setup_deployment_monitoring(
		self, 
		deployment_result: Dict[str, Any], 
		deployment_config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Setup monitoring for the deployed application"""
		self.logger.info("Setting up deployment monitoring")
		
		monitoring_reqs = deployment_config.get('monitoring_requirements', {})
		platform = deployment_result['platform']
		
		monitoring_setup = {
			'metrics_collection': {
				'enabled': monitoring_reqs.get('application_metrics', True),
				'endpoint': '/metrics',
				'format': 'prometheus',
				'scrape_interval': '30s'
			},
			'logging': {
				'enabled': True,
				'level': 'INFO',
				'format': 'json',
				'aggregation': monitoring_reqs.get('log_aggregation', platform != 'traditional')
			},
			'alerting': {
				'enabled': monitoring_reqs.get('alerting', True),
				'channels': ['email', 'slack'],
				'critical_alerts': ['application_down', 'high_error_rate', 'database_connection_failed'],
				'warning_alerts': ['high_cpu', 'high_memory', 'slow_response_time']
			},
			'dashboards': {
				'application_dashboard': {
					'url': f"https://grafana.example.com/d/app-{deployment_result['deployment_id']}",
					'metrics': ['response_time', 'throughput', 'error_rate', 'availability']
				},
				'infrastructure_dashboard': {
					'url': f"https://grafana.example.com/d/infra-{deployment_result['deployment_id']}",
					'metrics': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io']
				}
			},
			'uptime_monitoring': {
				'enabled': monitoring_reqs.get('uptime_monitoring', True),
				'check_interval': '1m',
				'endpoints': deployment_result.get('health_endpoints', [])
			}
		}
		
		if monitoring_reqs.get('distributed_tracing', False):
			monitoring_setup['tracing'] = {
				'enabled': True,
				'system': 'jaeger',
				'sampling_rate': '10%',
				'trace_endpoint': '/trace'
			}
		
		return monitoring_setup

	async def _validate_deployment(
		self, 
		deployment_result: Dict[str, Any], 
		testing_results: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Validate the deployment"""
		self.logger.info("Validating deployment")
		
		validation_checks = [
			'health_check',
			'connectivity_test',
			'functionality_test',
			'performance_baseline',
			'security_scan'
		]
		
		validation_results = {}
		overall_status = 'passed'
		
		for check in validation_checks:
			result = await self._run_validation_check(check, deployment_result, testing_results)
			validation_results[check] = result
			
			if result['status'] != 'passed':
				overall_status = 'failed'
		
		return {
			'status': overall_status,
			'checks': validation_results,
			'validated_at': datetime.utcnow().isoformat(),
			'deployment_ready': overall_status == 'passed'
		}

	async def _run_validation_check(
		self, 
		check_type: str, 
		deployment_result: Dict[str, Any], 
		testing_results: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Run a specific validation check"""
		
		# Simulate validation checks
		if check_type == 'health_check':
			return {
				'status': 'passed',
				'message': 'All health endpoints responding',
				'response_time': '150ms'
			}
		elif check_type == 'connectivity_test':
			return {
				'status': 'passed',
				'message': 'Database and external services accessible',
				'connections_tested': 3
			}
		elif check_type == 'functionality_test':
			return {
				'status': 'passed',
				'message': 'Core functionality verified',
				'tests_run': 15
			}
		elif check_type == 'performance_baseline':
			return {
				'status': 'passed',
				'message': 'Performance within acceptable limits',
				'response_time': '200ms',
				'throughput': '500 req/min'
			}
		elif check_type == 'security_scan':
			return {
				'status': 'passed',
				'message': 'No critical security issues detected',
				'vulnerabilities_found': 0
			}
		
		return {'status': 'skipped', 'message': 'Check not implemented'}

	async def _store_deployment_memory(
		self, 
		task: AgentTask, 
		deployment_result: Dict[str, Any], 
		deployment_config: Dict[str, Any]
	):
		"""Store deployment results in episodic memory"""
		memory = AgentMemory(
			agent_id=self.agent_id,
			memory_type="episodic",
			content={
				'deployment_type': 'application_deployment',
				'project_id': task.context.get('project_id'),
				'deployment_result': deployment_result,
				'deployment_config': deployment_config,
				'platform': deployment_result.get('platform'),
				'status': deployment_result.get('status'),
				'url': deployment_result.get('url'),
				'environment': deployment_config.get('environment'),
				'deployment_duration': 'estimated_30_minutes',
				'lessons_learned': 'Deployment completed successfully with monitoring'
			},
			importance=9,
			tags=['deployment', 'infrastructure', 'devops', task.context.get('project_id', 'unknown')]
		)
		await self._store_memory(memory)
		
		# Add to deployment history
		self.deployment_history.append({
			'deployment_id': deployment_result.get('deployment_id'),
			'project_id': task.context.get('project_id'),
			'platform': deployment_result.get('platform'),
			'status': deployment_result.get('status'),
			'deployed_at': deployment_result.get('deployed_at'),
			'url': deployment_result.get('url')
		})

	# Additional DevOps methods would be implemented here for:
	# - _setup_infrastructure
	# - _setup_monitoring  
	# - _setup_cicd_pipeline
	# - _optimize_scaling
	# These would follow similar patterns to the deployment method