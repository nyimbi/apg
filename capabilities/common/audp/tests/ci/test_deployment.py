"""
Audio Processing Deployment Tests

Unit tests for deployment configuration, Kubernetes manifests,
Docker Compose, and Terraform infrastructure generation.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import yaml
import json
from typing import Dict, Any, List
from pathlib import Path
from unittest.mock import patch, mock_open

from ...deployment import (
	DatabaseConfig, RedisConfig, ResourceLimits, AutoScalingConfig,
	HealthCheckConfig, SecurityConfig, MonitoringConfig, DeploymentConfig,
	KubernetesManifestGenerator, DockerComposeGenerator, TerraformGenerator,
	create_deployment_config, save_manifests_to_files, save_docker_compose,
	save_terraform_files
)


class TestConfigurationDataClasses:
	"""Test configuration data classes"""
	
	def test_database_config_defaults(self):
		"""Test DatabaseConfig default values"""
		config = DatabaseConfig()
		
		assert config.host == "localhost"
		assert config.port == 5432
		assert config.name == "apg_audio_processing"
		assert config.user == "apg_user"
		assert config.password == ""
		assert config.pool_size == 20
		assert config.max_overflow == 10
		assert config.pool_timeout == 30
		assert config.ssl_mode == "require"
	
	def test_database_config_custom(self):
		"""Test DatabaseConfig with custom values"""
		config = DatabaseConfig(
			host="db.example.com",
			port=5433,
			name="custom_db",
			user="custom_user",
			password="secret123",
			pool_size=30,
			ssl_mode="disable"
		)
		
		assert config.host == "db.example.com"
		assert config.port == 5433
		assert config.name == "custom_db"
		assert config.user == "custom_user"
		assert config.password == "secret123"
		assert config.pool_size == 30
		assert config.ssl_mode == "disable"
	
	def test_redis_config_defaults(self):
		"""Test RedisConfig default values"""
		config = RedisConfig()
		
		assert config.host == "localhost"
		assert config.port == 6379
		assert config.db == 0
		assert config.password == ""
		assert config.ssl is False
		assert config.pool_size == 10
		assert config.timeout == 5
	
	def test_resource_limits_defaults(self):
		"""Test ResourceLimits default values"""
		config = ResourceLimits()
		
		assert config.cpu_request == "500m"
		assert config.cpu_limit == "2000m"
		assert config.memory_request == "1Gi"
		assert config.memory_limit == "4Gi"
		assert config.storage_request == "10Gi"
		assert config.storage_limit == "100Gi"
	
	def test_autoscaling_config_defaults(self):
		"""Test AutoScalingConfig default values"""
		config = AutoScalingConfig()
		
		assert config.min_replicas == 2
		assert config.max_replicas == 10
		assert config.target_cpu_utilization == 70
		assert config.target_memory_utilization == 80
		assert config.scale_down_delay == 300
		assert config.scale_up_delay == 60
	
	def test_health_check_config_defaults(self):
		"""Test HealthCheckConfig default values"""
		config = HealthCheckConfig()
		
		assert config.enabled is True
		assert config.path == "/audio_processing/health"
		assert config.initial_delay == 30
		assert config.period_seconds == 10
		assert config.timeout_seconds == 5
		assert config.failure_threshold == 3
		assert config.success_threshold == 1
	
	def test_security_config_defaults(self):
		"""Test SecurityConfig default values"""
		config = SecurityConfig()
		
		assert config.enable_rbac is True
		assert config.enable_network_policies is True
		assert config.enable_pod_security_policy is True
		assert config.enable_tls is True
		assert config.tls_cert_path == "/etc/certs/tls.crt"
		assert config.tls_key_path == "/etc/certs/tls.key"
		assert config.jwt_secret_key == ""
		assert config.allowed_origins is None
	
	def test_monitoring_config_defaults(self):
		"""Test MonitoringConfig default values"""
		config = MonitoringConfig()
		
		assert config.prometheus_enabled is True
		assert config.grafana_enabled is True
		assert config.jaeger_enabled is True
		assert config.log_level == "INFO"
		assert config.metrics_port == 8080
		assert config.traces_endpoint == "http://jaeger:14268/api/traces"
	
	def test_deployment_config_initialization(self):
		"""Test DeploymentConfig initialization"""
		config = DeploymentConfig()
		
		assert config.environment == "production"
		assert config.namespace == "apg-audio-processing"
		assert config.app_name == "audio-processing"
		assert config.version == "1.0.0"
		assert config.image == "apg/audio-processing:1.0.0"
		assert config.replicas == 3
		
		# Check that nested configs are initialized
		assert config.database is not None
		assert config.redis is not None
		assert config.resources is not None
		assert config.autoscaling is not None
		assert config.health_check is not None
		assert config.security is not None
		assert config.monitoring is not None
	
	def test_deployment_config_custom_nested_configs(self):
		"""Test DeploymentConfig with custom nested configs"""
		custom_db = DatabaseConfig(host="custom-db.com", port=5433)
		custom_resources = ResourceLimits(cpu_limit="4000m", memory_limit="8Gi")
		
		config = DeploymentConfig(
			environment="staging",
			database=custom_db,
			resources=custom_resources
		)
		
		assert config.environment == "staging"
		assert config.database.host == "custom-db.com"
		assert config.database.port == 5433
		assert config.resources.cpu_limit == "4000m"
		assert config.resources.memory_limit == "8Gi"


class TestKubernetesManifestGenerator:
	"""Test Kubernetes manifest generation"""
	
	@pytest.fixture
	def deployment_config(self):
		"""Create deployment config for testing"""
		return DeploymentConfig(
			environment="test",
			namespace="test-audio",
			app_name="test-audio-app",
			version="1.0.0-test"
		)
	
	@pytest.fixture
	def manifest_generator(self, deployment_config):
		"""Create manifest generator instance"""
		return KubernetesManifestGenerator(deployment_config)
	
	def test_generate_namespace(self, manifest_generator):
		"""Test namespace manifest generation"""
		namespace = manifest_generator.generate_namespace()
		
		assert namespace["apiVersion"] == "v1"
		assert namespace["kind"] == "Namespace"
		assert namespace["metadata"]["name"] == "test-audio"
		assert namespace["metadata"]["labels"]["app"] == "test-audio-app"
		assert namespace["metadata"]["labels"]["version"] == "1.0.0-test"
		assert namespace["metadata"]["labels"]["environment"] == "test"
	
	def test_generate_configmap(self, manifest_generator):
		"""Test ConfigMap manifest generation"""
		configmap = manifest_generator.generate_configmap()
		
		assert configmap["apiVersion"] == "v1"
		assert configmap["kind"] == "ConfigMap"
		assert configmap["metadata"]["name"] == "test-audio-app-config"
		assert configmap["metadata"]["namespace"] == "test-audio"
		
		data = configmap["data"]
		assert data["DATABASE_HOST"] == "localhost"
		assert data["DATABASE_PORT"] == "5432"
		assert data["REDIS_HOST"] == "localhost"
		assert data["REDIS_PORT"] == "6379"
		assert data["LOG_LEVEL"] == "INFO"
		assert data["ENVIRONMENT"] == "test"
	
	def test_generate_secret(self, manifest_generator):
		"""Test Secret manifest generation"""
		secret = manifest_generator.generate_secret()
		
		assert secret["apiVersion"] == "v1"
		assert secret["kind"] == "Secret"
		assert secret["metadata"]["name"] == "test-audio-app-secrets"
		assert secret["metadata"]["namespace"] == "test-audio"
		assert secret["type"] == "Opaque"
		
		data = secret["data"]
		assert "DATABASE_PASSWORD" in data
		assert "REDIS_PASSWORD" in data
		assert "JWT_SECRET_KEY" in data
	
	def test_generate_deployment(self, manifest_generator):
		"""Test Deployment manifest generation"""
		deployment = manifest_generator.generate_deployment()
		
		assert deployment["apiVersion"] == "apps/v1"
		assert deployment["kind"] == "Deployment"
		assert deployment["metadata"]["name"] == "test-audio-app"
		assert deployment["metadata"]["namespace"] == "test-audio"
		
		spec = deployment["spec"]
		assert spec["replicas"] == 3
		
		container = spec["template"]["spec"]["containers"][0]
		assert container["name"] == "test-audio-app"
		assert container["image"] == "apg/audio-processing:1.0.0"
		
		# Check ports
		ports = container["ports"]
		port_names = [p["name"] for p in ports]
		assert "http" in port_names
		assert "metrics" in port_names
		
		# Check resource limits
		resources = container["resources"]
		assert resources["requests"]["cpu"] == "500m"
		assert resources["requests"]["memory"] == "1Gi"
		assert resources["limits"]["cpu"] == "2000m"
		assert resources["limits"]["memory"] == "4Gi"
		
		# Check health checks
		assert "livenessProbe" in container
		assert "readinessProbe" in container
		assert container["livenessProbe"]["httpGet"]["path"] == "/audio_processing/health"
	
	def test_generate_service(self, manifest_generator):
		"""Test Service manifest generation"""
		service = manifest_generator.generate_service()
		
		assert service["apiVersion"] == "v1"
		assert service["kind"] == "Service"
		assert service["metadata"]["name"] == "test-audio-app"
		assert service["metadata"]["namespace"] == "test-audio"
		
		spec = service["spec"]
		assert spec["selector"]["app"] == "test-audio-app"
		assert spec["type"] == "ClusterIP"
		
		ports = spec["ports"]
		assert len(ports) == 2
		
		port_names = [p["name"] for p in ports]
		assert "http" in port_names
		assert "metrics" in port_names
	
	def test_generate_hpa(self, manifest_generator):
		"""Test HorizontalPodAutoscaler manifest generation"""
		hpa = manifest_generator.generate_hpa()
		
		assert hpa["apiVersion"] == "autoscaling/v2"
		assert hpa["kind"] == "HorizontalPodAutoscaler"
		assert hpa["metadata"]["name"] == "test-audio-app-hpa"
		assert hpa["metadata"]["namespace"] == "test-audio"
		
		spec = hpa["spec"]
		assert spec["minReplicas"] == 2
		assert spec["maxReplicas"] == 10
		
		# Check metrics
		metrics = spec["metrics"]
		assert len(metrics) == 2
		
		metric_names = [m["resource"]["name"] for m in metrics]
		assert "cpu" in metric_names
		assert "memory" in metric_names
		
		# Check behavior
		behavior = spec["behavior"]
		assert "scaleDown" in behavior
		assert "scaleUp" in behavior
	
	def test_generate_ingress(self, manifest_generator):
		"""Test Ingress manifest generation"""
		host = "audio.test.local"
		ingress = manifest_generator.generate_ingress(host, tls_enabled=True)
		
		assert ingress["apiVersion"] == "networking.k8s.io/v1"
		assert ingress["kind"] == "Ingress"
		assert ingress["metadata"]["name"] == "test-audio-app-ingress"
		assert ingress["metadata"]["namespace"] == "test-audio"
		
		spec = ingress["spec"]
		rules = spec["rules"]
		assert len(rules) == 1
		assert rules[0]["host"] == host
		
		paths = rules[0]["http"]["paths"]
		assert len(paths) == 1
		assert paths[0]["path"] == "/audio_processing"
		assert paths[0]["pathType"] == "Prefix"
		
		# Check TLS configuration
		tls = spec["tls"]
		assert len(tls) == 1
		assert host in tls[0]["hosts"]
		assert tls[0]["secretName"] == "test-audio-app-tls"
	
	def test_generate_all_manifests(self, manifest_generator):
		"""Test generating all manifests"""
		manifests = manifest_generator.generate_all_manifests("audio.test.local")
		
		assert len(manifests) == 7
		
		kinds = [m["kind"] for m in manifests]
		expected_kinds = [
			"Namespace", "ConfigMap", "Secret", "Deployment", 
			"Service", "HorizontalPodAutoscaler", "Ingress"
		]
		
		for expected_kind in expected_kinds:
			assert expected_kind in kinds


class TestDockerComposeGenerator:
	"""Test Docker Compose generation"""
	
	@pytest.fixture
	def deployment_config(self):
		"""Create deployment config for testing"""
		return DeploymentConfig(
			environment="development",
			monitoring=MonitoringConfig(
				prometheus_enabled=True,
				grafana_enabled=True
			)
		)
	
	@pytest.fixture
	def compose_generator(self, deployment_config):
		"""Create Docker Compose generator instance"""
		return DockerComposeGenerator(deployment_config)
	
	def test_generate_compose_basic_services(self, compose_generator):
		"""Test Docker Compose generation with basic services"""
		compose_config = compose_generator.generate_compose()
		
		assert compose_config["version"] == "3.8"
		
		services = compose_config["services"]
		assert "audio-processing" in services
		assert "postgres" in services
		assert "redis" in services
		
		# Check audio processing service
		audio_service = services["audio-processing"]
		assert audio_service["image"] == "apg/audio-processing:1.0.0"
		assert "8000:8000" in audio_service["ports"]
		assert "postgres" in audio_service["depends_on"]
		assert "redis" in audio_service["depends_on"]
		assert audio_service["restart"] == "unless-stopped"
		
		# Check environment variables
		env = audio_service["environment"]
		assert env["DATABASE_HOST"] == "localhost"
		assert env["REDIS_HOST"] == "localhost"
		assert env["ENVIRONMENT"] == "production"
		
		# Check health check
		healthcheck = audio_service["healthcheck"]
		assert "/audio_processing/health" in healthcheck["test"]
		assert healthcheck["interval"] == "10s"
		assert healthcheck["retries"] == 3
	
	def test_generate_compose_postgres_service(self, compose_generator):
		"""Test PostgreSQL service configuration"""
		compose_config = compose_generator.generate_compose()
		
		postgres_service = compose_config["services"]["postgres"]
		assert postgres_service["image"] == "postgres:15-alpine"
		assert postgres_service["restart"] == "unless-stopped"
		
		env = postgres_service["environment"]
		assert env["POSTGRES_DB"] == "apg_audio_processing"
		assert env["POSTGRES_USER"] == "apg_user"
		assert env["POSTGRES_PASSWORD"] == "${DATABASE_PASSWORD}"
		
		volumes = postgres_service["volumes"]
		assert "postgres_data:/var/lib/postgresql/data" in volumes
		assert "./sql/init.sql:/docker-entrypoint-initdb.d/init.sql" in volumes
	
	def test_generate_compose_redis_service(self, compose_generator):
		"""Test Redis service configuration"""
		compose_config = compose_generator.generate_compose()
		
		redis_service = compose_config["services"]["redis"]
		assert redis_service["image"] == "redis:7-alpine"
		assert redis_service["restart"] == "unless-stopped"
		assert redis_service["command"] == "redis-server --appendonly yes"
		
		volumes = redis_service["volumes"]
		assert "redis_data:/data" in volumes
	
	def test_generate_compose_monitoring_services(self, compose_generator):
		"""Test monitoring services when enabled"""
		compose_config = compose_generator.generate_compose()
		
		services = compose_config["services"]
		
		# Check Prometheus service
		assert "prometheus" in services
		prometheus_service = services["prometheus"]
		assert prometheus_service["image"] == "prom/prometheus:latest"
		assert "9090:9090" in prometheus_service["ports"]
		assert "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" in prometheus_service["volumes"]
		
		# Check Grafana service
		assert "grafana" in services
		grafana_service = services["grafana"]
		assert grafana_service["image"] == "grafana/grafana:latest"
		assert "3000:3000" in grafana_service["ports"]
		assert grafana_service["environment"]["GF_SECURITY_ADMIN_PASSWORD"] == "${GRAFANA_PASSWORD}"
	
	def test_generate_compose_volumes_and_networks(self, compose_generator):
		"""Test volumes and networks configuration"""
		compose_config = compose_generator.generate_compose()
		
		volumes = compose_config["volumes"]
		expected_volumes = ["postgres_data", "redis_data", "prometheus_data", "grafana_data"]
		for volume in expected_volumes:
			assert volume in volumes
		
		networks = compose_config["networks"]
		assert "default" in networks
		assert networks["default"]["name"] == "apg-audio-processing"


class TestTerraformGenerator:
	"""Test Terraform configuration generation"""
	
	@pytest.fixture
	def deployment_config(self):
		"""Create deployment config for testing"""
		return DeploymentConfig(
			environment="production",
			namespace="prod-audio",
			app_name="prod-audio-app"
		)
	
	@pytest.fixture
	def terraform_generator(self, deployment_config):
		"""Create Terraform generator instance"""
		return TerraformGenerator(deployment_config)
	
	def test_generate_main_tf(self, terraform_generator):
		"""Test main Terraform configuration generation"""
		main_tf = terraform_generator.generate_main_tf()
		
		assert "terraform {" in main_tf
		assert "required_version = \">= 1.0\"" in main_tf
		assert "hashicorp/kubernetes" in main_tf
		assert "hashicorp/helm" in main_tf
		
		# Check namespace resource
		assert "resource \"kubernetes_namespace\" \"audio_processing\"" in main_tf
		assert "name = \"prod-audio\"" in main_tf
		
		# Check ConfigMap resource
		assert "resource \"kubernetes_config_map\" \"audio_processing_config\"" in main_tf
		assert "name      = \"prod-audio-app-config\"" in main_tf
		
		# Check Secret resource
		assert "resource \"kubernetes_secret\" \"audio_processing_secrets\"" in main_tf
		assert "name      = \"prod-audio-app-secrets\"" in main_tf
		
		# Check Deployment resource
		assert "resource \"kubernetes_deployment\" \"audio_processing\"" in main_tf
		assert "name      = \"prod-audio-app\"" in main_tf
		assert "replicas = 3" in main_tf
		
		# Check Service resource
		assert "resource \"kubernetes_service\" \"audio_processing\"" in main_tf
		
		# Check HPA resource
		assert "resource \"kubernetes_horizontal_pod_autoscaler_v2\" \"audio_processing_hpa\"" in main_tf
		assert "min_replicas = 2" in main_tf
		assert "max_replicas = 10" in main_tf
	
	def test_generate_variables_tf(self, terraform_generator):
		"""Test Terraform variables configuration"""
		variables_tf = terraform_generator.generate_variables_tf()
		
		# Check variable declarations
		assert "variable \"database_host\"" in variables_tf
		assert "variable \"database_port\"" in variables_tf
		assert "variable \"database_name\"" in variables_tf
		assert "variable \"database_user\"" in variables_tf
		assert "variable \"database_password\"" in variables_tf
		assert "variable \"redis_host\"" in variables_tf
		assert "variable \"redis_port\"" in variables_tf
		assert "variable \"redis_password\"" in variables_tf
		assert "variable \"jwt_secret_key\"" in variables_tf
		
		# Check sensitive variables
		assert "sensitive   = true" in variables_tf
		
		# Check default values
		assert "default     = \"postgres\"" in variables_tf
		assert "default     = \"redis\"" in variables_tf
		assert "default     = \"5432\"" in variables_tf
		assert "default     = \"6379\"" in variables_tf
	
	def test_generate_outputs_tf(self, terraform_generator):
		"""Test Terraform outputs configuration"""
		outputs_tf = terraform_generator.generate_outputs_tf()
		
		assert "output \"namespace\"" in outputs_tf
		assert "output \"service_name\"" in outputs_tf
		assert "output \"deployment_name\"" in outputs_tf
		
		assert "kubernetes_namespace.audio_processing.metadata[0].name" in outputs_tf
		assert "kubernetes_service.audio_processing.metadata[0].name" in outputs_tf
		assert "kubernetes_deployment.audio_processing.metadata[0].name" in outputs_tf


class TestDeploymentFactories:
	"""Test deployment factory functions"""
	
	def test_create_deployment_config_development(self):
		"""Test creating development deployment config"""
		config = create_deployment_config("development")
		
		assert config.environment == "development"
		assert config.replicas == 1
		assert config.resources.cpu_request == "100m"
		assert config.resources.cpu_limit == "500m"
		assert config.resources.memory_request == "256Mi"
		assert config.resources.memory_limit == "1Gi"
		assert config.autoscaling.min_replicas == 1
		assert config.autoscaling.max_replicas == 3
		assert config.security.enable_rbac is False
		assert config.security.enable_tls is False
	
	def test_create_deployment_config_staging(self):
		"""Test creating staging deployment config"""
		config = create_deployment_config("staging")
		
		assert config.environment == "staging"
		assert config.replicas == 2
		assert config.autoscaling.min_replicas == 2
		assert config.autoscaling.max_replicas == 6
		# Should use production defaults for other settings
		assert config.security.enable_rbac is True
		assert config.security.enable_tls is True
	
	def test_create_deployment_config_production(self):
		"""Test creating production deployment config"""
		config = create_deployment_config("production")
		
		assert config.environment == "production"
		assert config.replicas == 3
		assert config.autoscaling.min_replicas == 3
		assert config.autoscaling.max_replicas == 10
		assert config.security.enable_rbac is True
		assert config.security.enable_tls is True
		assert config.resources.cpu_request == "500m"
		assert config.resources.memory_request == "1Gi"
	
	def test_create_deployment_config_default(self):
		"""Test creating deployment config with default environment"""
		config = create_deployment_config()
		
		# Should default to production
		assert config.environment == "production"
		assert config.replicas == 3


class TestDeploymentUtilityFunctions:
	"""Test deployment utility functions"""
	
	def test_save_manifests_to_files(self, tmp_path):
		"""Test saving Kubernetes manifests to files"""
		manifests = [
			{
				"apiVersion": "v1",
				"kind": "Namespace",
				"metadata": {"name": "test-namespace"}
			},
			{
				"apiVersion": "apps/v1",
				"kind": "Deployment",
				"metadata": {"name": "test-deployment"}
			}
		]
		
		output_dir = str(tmp_path / "manifests")
		
		with patch('pathlib.Path.mkdir') as mock_mkdir, \
			 patch('builtins.open', mock_open()) as mock_file, \
			 patch('yaml.dump') as mock_yaml_dump:
			
			save_manifests_to_files(manifests, output_dir)
			
			# Check that directory was created
			mock_mkdir.assert_called_once_with(exist_ok=True)
			
			# Check that files were opened for writing
			assert mock_file.call_count == 2
			
			# Check that YAML was dumped for each manifest
			assert mock_yaml_dump.call_count == 2
	
	def test_save_docker_compose(self, tmp_path):
		"""Test saving Docker Compose configuration"""
		compose_config = {
			"version": "3.8",
			"services": {
				"test-service": {
					"image": "test:latest"
				}
			}
		}
		
		output_file = str(tmp_path / "docker-compose.yml")
		
		with patch('builtins.open', mock_open()) as mock_file, \
			 patch('yaml.dump') as mock_yaml_dump:
			
			save_docker_compose(compose_config, output_file)
			
			# Check that file was opened for writing
			mock_file.assert_called_once_with(output_file, 'w')
			
			# Check that YAML was dumped
			mock_yaml_dump.assert_called_once_with(compose_config, mock_file(), default_flow_style=False)
	
	def test_save_terraform_files(self, tmp_path):
		"""Test saving Terraform configuration files"""
		main_tf = "# Main Terraform configuration"
		variables_tf = "# Variables configuration"
		outputs_tf = "# Outputs configuration"
		output_dir = str(tmp_path / "terraform")
		
		with patch('pathlib.Path.mkdir') as mock_mkdir, \
			 patch('builtins.open', mock_open()) as mock_file:
			
			save_terraform_files(main_tf, variables_tf, outputs_tf, output_dir)
			
			# Check that directory was created
			mock_mkdir.assert_called_once_with(exist_ok=True)
			
			# Check that files were opened for writing
			assert mock_file.call_count == 3
			
			# Check that content was written
			handle = mock_file()
			assert handle.write.call_count == 3


class TestDeploymentIntegration:
	"""Test deployment component integration"""
	
	def test_complete_kubernetes_deployment_generation(self):
		"""Test complete Kubernetes deployment generation"""
		config = create_deployment_config("production")
		generator = KubernetesManifestGenerator(config)
		
		manifests = generator.generate_all_manifests("audio.example.com")
		
		# Verify all manifests are generated
		assert len(manifests) == 7
		
		# Verify manifest consistency
		namespace_name = None
		app_name = None
		
		for manifest in manifests:
			if manifest["kind"] == "Namespace":
				namespace_name = manifest["metadata"]["name"]
			elif "metadata" in manifest and "namespace" in manifest["metadata"]:
				assert manifest["metadata"]["namespace"] == namespace_name
			
			if "metadata" in manifest and "labels" in manifest["metadata"]:
				labels = manifest["metadata"]["labels"]
				if "app" in labels:
					if app_name is None:
						app_name = labels["app"]
					else:
						assert labels["app"] == app_name
	
	def test_complete_docker_compose_generation(self):
		"""Test complete Docker Compose generation"""
		config = create_deployment_config("development")
		config.monitoring.prometheus_enabled = True
		config.monitoring.grafana_enabled = True
		
		generator = DockerComposeGenerator(config)
		compose_config = generator.generate_compose()
		
		# Verify service dependencies
		audio_service = compose_config["services"]["audio-processing"]
		assert "postgres" in audio_service["depends_on"]
		assert "redis" in audio_service["depends_on"]
		
		# Verify volumes are referenced
		postgres_service = compose_config["services"]["postgres"]
		redis_service = compose_config["services"]["redis"]
		
		postgres_volumes = [v for v in postgres_service["volumes"] if "postgres_data:" in v]
		redis_volumes = [v for v in redis_service["volumes"] if "redis_data:" in v]
		
		assert len(postgres_volumes) > 0
		assert len(redis_volumes) > 0
		
		# Verify monitoring services
		assert "prometheus" in compose_config["services"]
		assert "grafana" in compose_config["services"]
	
	def test_complete_terraform_generation(self):
		"""Test complete Terraform generation"""
		config = create_deployment_config("production")
		generator = TerraformGenerator(config)
		
		main_tf = generator.generate_main_tf()
		variables_tf = generator.generate_variables_tf()
		outputs_tf = generator.generate_outputs_tf()
		
		# Verify Terraform configuration consistency
		assert config.namespace in main_tf
		assert config.app_name in main_tf
		assert str(config.replicas) in main_tf
		
		# Verify variables are used in main configuration
		assert "var.database_host" in main_tf
		assert "var.database_password" in main_tf
		assert "var.jwt_secret_key" in main_tf
		
		# Verify outputs reference correct resources
		assert "kubernetes_namespace.audio_processing" in outputs_tf
		assert "kubernetes_service.audio_processing" in outputs_tf
		assert "kubernetes_deployment.audio_processing" in outputs_tf


class TestDeploymentEdgeCases:
	"""Test deployment edge cases and configuration validation"""
	
	def test_deployment_config_with_disabled_features(self):
		"""Test deployment config with disabled features"""
		config = DeploymentConfig(
			health_check=HealthCheckConfig(enabled=False),
			security=SecurityConfig(enable_tls=False),
			monitoring=MonitoringConfig(
				prometheus_enabled=False,
				grafana_enabled=False
			)
		)
		
		generator = KubernetesManifestGenerator(config)
		deployment = generator.generate_deployment()
		
		# Health checks should not be present
		container = deployment["spec"]["template"]["spec"]["containers"][0]
		assert "livenessProbe" not in container
		assert "readinessProbe" not in container
		
		# Ingress should not have TLS
		ingress = generator.generate_ingress("test.local", tls_enabled=False)
		assert "tls" not in ingress["spec"]
	
	def test_docker_compose_without_monitoring(self):
		"""Test Docker Compose generation without monitoring services"""
		config = DeploymentConfig(
			monitoring=MonitoringConfig(
				prometheus_enabled=False,
				grafana_enabled=False
			)
		)
		
		generator = DockerComposeGenerator(config)
		compose_config = generator.generate_compose()
		
		services = compose_config["services"]
		assert "prometheus" not in services
		assert "grafana" not in services
		
		volumes = compose_config["volumes"]
		assert "prometheus_data" not in volumes
		assert "grafana_data" not in volumes
	
	def test_minimal_deployment_config(self):
		"""Test deployment with minimal configuration"""
		config = DeploymentConfig(
			replicas=1,
			resources=ResourceLimits(
				cpu_request="100m",
				cpu_limit="200m",
				memory_request="128Mi",
				memory_limit="256Mi"
			),
			autoscaling=AutoScalingConfig(
				min_replicas=1,
				max_replicas=1  # No scaling
			)
		)
		
		generator = KubernetesManifestGenerator(config)
		deployment = generator.generate_deployment()
		hpa = generator.generate_hpa()
		
		assert deployment["spec"]["replicas"] == 1
		
		container = deployment["spec"]["template"]["spec"]["containers"][0]
		resources = container["resources"]
		assert resources["requests"]["cpu"] == "100m"
		assert resources["limits"]["memory"] == "256Mi"
		
		assert hpa["spec"]["minReplicas"] == 1
		assert hpa["spec"]["maxReplicas"] == 1