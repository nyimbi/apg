# APG Composition Infrastructure as Code
# Terraform configuration for production deployment of the composition capability

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "apg-terraform-state"
    key    = "composition/production/terraform.tfstate"
    region = "us-west-2"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "APG-Composition"
      ManagedBy   = "Terraform"
      Owner       = "APG-Platform-Team"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Variables
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "apg-composition-prod"
}

variable "node_group_size" {
  description = "EKS node group configuration"
  type = object({
    desired = number
    min     = number
    max     = number
  })
  default = {
    desired = 3
    min     = 2
    max     = 10
  }
}

variable "instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "database_config" {
  description = "RDS configuration"
  type = object({
    instance_class      = string
    allocated_storage   = number
    max_allocated_storage = number
    backup_retention   = number
    multi_az          = bool
  })
  default = {
    instance_class        = "db.r6g.large"
    allocated_storage     = 100
    max_allocated_storage = 1000
    backup_retention     = 30
    multi_az             = true
  }
}

# Local values
locals {
  name_prefix = "${var.environment}-composition"
  
  common_tags = {
    Environment = var.environment
    Service     = "composition"
    Component   = "apg-platform"
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]

  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  create_database_subnet_group = true
  create_database_internet_gateway_route = false

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.27"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  eks_managed_node_groups = {
    composition_nodes = {
      name = "${local.name_prefix}-nodes"

      instance_types = var.instance_types
      capacity_type  = "ON_DEMAND"

      min_size     = var.node_group_size.min
      max_size     = var.node_group_size.max
      desired_size = var.node_group_size.desired

      ami_type = "AL2_x86_64"
      
      disk_size = 100
      disk_type = "gp3"
      disk_throughput = 150
      disk_iops = 3000

      labels = {
        Environment = var.environment
        NodeGroup   = "composition"
      }

      taints = {
        composition = {
          key    = "composition"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      update_config = {
        max_unavailable_percentage = 25
      }
    }
  }

  tags = local.common_tags
}

# RDS PostgreSQL Database
resource "aws_db_subnet_group" "composition" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name_prefix}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.composition_app.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

resource "aws_db_instance" "composition" {
  identifier = "${local.name_prefix}-db"

  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.database_config.instance_class

  allocated_storage     = var.database_config.allocated_storage
  max_allocated_storage = var.database_config.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "composition_db"
  username = "composition_user"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.composition.name

  backup_retention_period = var.database_config.backup_retention
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  multi_az               = var.database_config.multi_az
  publicly_accessible    = false
  deletion_protection    = true

  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-database"
  })
}

# RDS Read Replica
resource "aws_db_instance" "composition_replica" {
  identifier = "${local.name_prefix}-db-replica"

  replicate_source_db = aws_db_instance.composition.identifier

  instance_class = var.database_config.instance_class
  publicly_accessible = false

  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-database-replica"
  })
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "composition" {
  name       = "${local.name_prefix}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "elasticache" {
  name_prefix = "${local.name_prefix}-cache-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.composition_app.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cache-sg"
  })
}

resource "aws_elasticache_replication_group" "composition" {
  replication_group_id         = "${local.name_prefix}-cache"
  description                  = "Redis cluster for APG Composition"

  port                         = 6379
  parameter_group_name         = "default.redis7"
  node_type                    = "cache.r6g.large"
  
  num_cache_clusters           = 3
  automatic_failover_enabled   = true
  multi_az_enabled            = true

  subnet_group_name           = aws_elasticache_subnet_group.composition.name
  security_group_ids          = [aws_security_group.elasticache.id]

  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  auth_token                  = random_password.redis_password.result

  tags = local.common_tags
}

# Application Security Group
resource "aws_security_group" "composition_app" {
  name_prefix = "${local.name_prefix}-app-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }

  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-app-sg"
  })
}

# Secrets Manager
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "aws_secretsmanager_secret" "composition_secrets" {
  name = "${local.name_prefix}-secrets"
  description = "Secrets for APG Composition service"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "composition_secrets" {
  secret_id = aws_secretsmanager_secret.composition_secrets.id
  secret_string = jsonencode({
    database_url = "postgresql://${aws_db_instance.composition.username}:${random_password.db_password.result}@${aws_db_instance.composition.endpoint}/${aws_db_instance.composition.db_name}?sslmode=require"
    database_replica_url = "postgresql://${aws_db_instance.composition.username}:${random_password.db_password.result}@${aws_db_instance.composition_replica.endpoint}/${aws_db_instance.composition.db_name}?sslmode=require"
    redis_url = "rediss://:${random_password.redis_password.result}@${aws_elasticache_replication_group.composition.primary_endpoint_address}:6379"
    jwt_secret = random_password.jwt_secret.result
    secret_key = random_password.jwt_secret.result
  })
}

# IAM Roles
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.name_prefix}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "composition_backups" {
  bucket = "${local.name_prefix}-backups-${random_string.bucket_suffix.result}"

  tags = local.common_tags
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "composition_backups" {
  bucket = aws_s3_bucket.composition_backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "composition_backups" {
  bucket = aws_s3_bucket.composition_backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "composition_backups" {
  bucket = aws_s3_bucket.composition_backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "composition" {
  name              = "/aws/eks/${var.cluster_name}/composition"
  retention_in_days = 30

  tags = local.common_tags
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.composition.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.composition.primary_endpoint_address
  sensitive   = true
}

output "secrets_manager_arn" {
  description = "Secrets Manager ARN"
  value       = aws_secretsmanager_secret.composition_secrets.arn
}

output "backup_bucket" {
  description = "S3 backup bucket name"
  value       = aws_s3_bucket.composition_backups.bucket
}