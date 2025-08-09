#!/usr/bin/env python3
"""
APG Metadata Management - ML Platform Connectors
Connectors for discovering metadata from machine learning platforms

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .base_connector import (
	BaseConnector, ConnectorConfig, DiscoveryResult, AssetMetadata,
	ColumnMetadata, ConnectorType, DataType, should_include_asset
)


class MLflowConnector(BaseConnector):
	"""MLflow metadata discovery connector - placeholder"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.ML_PLATFORM
		self.source_system = "mlflow"
	
	async def connect(self) -> bool:
		"""Connect to MLflow tracking server"""
		try:
			# Test connection to MLflow tracking server
			tracking_uri = self.config.connection_string or "http://localhost:5000"
			response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=10)
			self.is_connected = response.status_code == 200
			return self.is_connected
		except Exception as e:
			await self._log_error(f"MLflow connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from MLflow"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test MLflow connection"""
		if await self.connect():
			return {"status": "success", "message": "MLflow connection successful"}
		else:
			return {"status": "error", "message": "Failed to connect to MLflow"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover MLflow experiments and models"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not await self.connect():
				result.add_error("Failed to connect to MLflow")
				return result
			
			tracking_uri = self.config.connection_string or "http://localhost:5000"
			
			# Discover experiments
			exp_response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search")
			if exp_response.status_code == 200:
				experiments = exp_response.json().get("experiments", [])
				for exp in experiments:
					experiment_asset = AssetMetadata(
						name=f"experiment_{exp['name']}",
						display_name=exp["name"],
						asset_type="ml_experiment",
						source_system=self.source_system,
						description=f"MLflow experiment: {exp['name']}",
						custom_attributes={
							"experiment_id": exp["experiment_id"],
							"artifact_location": exp.get("artifact_location"),
							"lifecycle_stage": exp.get("lifecycle_stage")
						}
					)
					result.add_asset(experiment_asset)
					
			# Discover registered models
			model_response = requests.get(f"{tracking_uri}/api/2.0/mlflow/registered-models/search")
			if model_response.status_code == 200:
				models = model_response.json().get("registered_models", [])
				for model in models:
					model_asset = AssetMetadata(
						name=f"model_{model['name']}",
						display_name=model["name"],
						asset_type="ml_model",
						source_system=self.source_system,
						description=model.get("description", f"MLflow model: {model['name']}"),
						custom_attributes={
							"creation_timestamp": model.get("creation_timestamp"),
							"last_updated_timestamp": model.get("last_updated_timestamp"),
							"latest_versions": [v.get("version") for v in model.get("latest_versions", [])]
						}
					)
					result.add_asset(model_asset)
			
			result.complete_discovery()
			return result
			
		except Exception as e:
			result.add_error(f"MLflow discovery failed: {str(e)}")
			result.complete_discovery()
			return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed MLflow asset schema"""
		try:
			if not self.is_connected:
				if not await self.connect():
					return None
			
			tracking_uri = self.config.connection_string or "http://localhost:5000"
			
			if asset_name.startswith("experiment_"):
				exp_name = asset_name.replace("experiment_", "")
				response = requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/get-by-name?experiment_name={exp_name}")
				if response.status_code == 200:
					exp_data = response.json().get("experiment")
					return AssetMetadata(
						name=asset_name,
						display_name=exp_data["name"],
						asset_type="ml_experiment",
						source_system=self.source_system,
						description=f"MLflow experiment: {exp_data['name']}",
						custom_attributes=exp_data
					)
			
			elif asset_name.startswith("model_"):
				model_name = asset_name.replace("model_", "")
				response = requests.get(f"{tracking_uri}/api/2.0/mlflow/registered-models/get?name={model_name}")
				if response.status_code == 200:
					model_data = response.json().get("registered_model")
					return AssetMetadata(
						name=asset_name,
						display_name=model_data["name"],
						asset_type="ml_model",
						source_system=self.source_system,
						description=model_data.get("description", f"MLflow model: {model_data['name']}"),
						custom_attributes=model_data
					)
			
			return None
			
		except Exception as e:
			await self._log_error(f"Failed to get MLflow asset schema: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get sample data from MLflow experiment or run"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to MLflow")
				return []
			
			# Parse asset name to determine if it's an experiment or run
			if asset_name.startswith("experiment:"):
				experiment_name = asset_name[11:]  # Remove "experiment:" prefix
				return await self._sample_experiment_data(experiment_name, limit)
			elif asset_name.startswith("run:"):
				run_id = asset_name[4:]  # Remove "run:" prefix
				return await self._sample_run_data(run_id, limit)
			else:
				await self._log_error(f"Unknown MLflow asset type: {asset_name}")
				return []
				
		except Exception as e:
			await self._log_error(f"Failed to sample MLflow data: {str(e)}")
			return []
	
	async def _sample_experiment_data(self, experiment_name: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from an MLflow experiment"""
		try:
			from mlflow.tracking import MlflowClient
			client = MlflowClient(self.tracking_uri)
			
			# Get experiment
			experiment = client.get_experiment_by_name(experiment_name)
			if not experiment:
				return []
			
			# Get runs from the experiment
			runs = client.search_runs(
				experiment_ids=[experiment.experiment_id],
				max_results=min(limit, 100)
			)
			
			# Convert runs to sample data format
			sample_data = []
			for run in runs:
				run_data = {
					"run_id": run.info.run_id,
					"status": run.info.status,
					"start_time": run.info.start_time,
					"end_time": run.info.end_time,
					"artifact_uri": run.info.artifact_uri,
					"metrics": dict(run.data.metrics),
					"params": dict(run.data.params),
					"tags": dict(run.data.tags)
				}
				sample_data.append(run_data)
			
			return sample_data
			
		except Exception as e:
			await self._log_error(f"Failed to sample experiment data: {str(e)}")
			return []
	
	async def _sample_run_data(self, run_id: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from a specific MLflow run"""
		try:
			from mlflow.tracking import MlflowClient
			client = MlflowClient(self.tracking_uri)
			
			# Get run details
			run = client.get_run(run_id)
			
			# Get metrics history for sampling
			metrics_data = []
			for metric_name in run.data.metrics.keys():
				metric_history = client.get_metric_history(run_id, metric_name)
				for metric_point in metric_history[:limit]:
					metrics_data.append({
						"metric_name": metric_name,
						"value": metric_point.value,
						"timestamp": metric_point.timestamp,
						"step": metric_point.step
					})
			
			return metrics_data[:limit]
			
		except Exception as e:
			await self._log_error(f"Failed to sample run data: {str(e)}")
			return []


class KubeflowConnector(BaseConnector):
	"""Kubeflow metadata discovery connector - placeholder"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.ML_PLATFORM
		self.source_system = "kubeflow"
	
	async def connect(self) -> bool:
		"""Connect to Kubeflow Metadata API"""
		try:
			# Connect to Kubeflow using kubernetes client
			namespace = self.config.custom_attributes.get("namespace", "kubeflow")
			api_server = self.config.connection_string or "http://localhost:8080"
			
			# Test connection to Kubeflow API server
			response = requests.get(f"{api_server}/api/v1/namespaces/{namespace}/pods", timeout=10)
			self.is_connected = response.status_code in [200, 401]  # 401 means auth needed but server responding
			return self.is_connected
		except Exception as e:
			await self._log_error(f"Kubeflow connection failed: {str(e)}")
			self.is_connected = False
			return False
	
	async def disconnect(self):
		"""Disconnect from Kubeflow"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Kubeflow connection"""
		if await self.connect():
			return {"status": "success", "message": "Kubeflow connection successful"}
		else:
			return {"status": "error", "message": "Failed to connect to Kubeflow"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover Kubeflow pipelines and models"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not await self.connect():
				result.add_error("Failed to connect to Kubeflow")
				return result
			
			# Simulate discovering Kubeflow artifacts
			sample_pipelines = [
				{"name": "training-pipeline", "version": "v1.0", "type": "training"},
				{"name": "inference-pipeline", "version": "v2.1", "type": "inference"}
			]
			
			for pipeline in sample_pipelines:
				pipeline_asset = AssetMetadata(
					name=f"pipeline_{pipeline['name']}",
					display_name=pipeline["name"],
					asset_type="ml_pipeline",
					source_system=self.source_system,
					description=f"Kubeflow pipeline: {pipeline['name']}",
					custom_attributes={
						"version": pipeline["version"],
						"pipeline_type": pipeline["type"],
						"namespace": self.config.custom_attributes.get("namespace", "kubeflow")
					}
				)
				result.add_asset(pipeline_asset)
			
			result.complete_discovery()
			return result
			
		except Exception as e:
			result.add_error(f"Kubeflow discovery failed: {str(e)}")
			result.complete_discovery()
			return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed Kubeflow asset schema"""
		try:
			if not self.is_connected:
				if not await self.connect():
					return None
			
			if asset_name.startswith("pipeline_"):
				pipeline_name = asset_name.replace("pipeline_", "")
				return AssetMetadata(
					name=asset_name,
					display_name=pipeline_name,
					asset_type="ml_pipeline",
					source_system=self.source_system,
					description=f"Kubeflow pipeline: {pipeline_name}",
					custom_attributes={
						"pipeline_definition": f"Pipeline definition for {pipeline_name}",
						"components": ["data-prep", "training", "evaluation"],
						"namespace": self.config.custom_attributes.get("namespace", "kubeflow")
					}
				)
			
			return None
			
		except Exception as e:
			await self._log_error(f"Failed to get Kubeflow asset schema: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get sample data from Kubeflow pipeline or model"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to Kubeflow")
				return []
			
			# Parse asset name to determine type
			if asset_name.startswith("pipeline:"):
				pipeline_name = asset_name[9:]  # Remove "pipeline:" prefix
				return await self._sample_pipeline_data(pipeline_name, limit)
			elif asset_name.startswith("model:"):
				model_name = asset_name[6:]  # Remove "model:" prefix
				return await self._sample_model_data(model_name, limit)
			else:
				await self._log_error(f"Unknown Kubeflow asset type: {asset_name}")
				return []
				
		except Exception as e:
			await self._log_error(f"Failed to sample Kubeflow data: {str(e)}")
			return []
	
	async def _sample_pipeline_data(self, pipeline_name: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from a Kubeflow pipeline"""
		try:
			# Simulate pipeline run data
			sample_runs = []
			for i in range(min(limit, 10)):
				run_data = {
					"run_id": f"run-{pipeline_name}-{i:03d}",
					"pipeline_name": pipeline_name,
					"status": "Succeeded" if i % 3 != 0 else "Failed",
					"start_time": f"2025-01-{9-i:02d}T10:00:00Z",
					"end_time": f"2025-01-{9-i:02d}T11:30:00Z",
					"parameters": {
						"learning_rate": 0.001 * (i + 1),
						"batch_size": 32 * (i + 1),
						"epochs": 10 + i
					},
					"metrics": {
						"accuracy": 0.85 + (i * 0.02),
						"loss": 0.3 - (i * 0.01)
					}
				}
				sample_runs.append(run_data)
			
			return sample_runs
			
		except Exception as e:
			await self._log_error(f"Failed to sample pipeline data: {str(e)}")
			return []
	
	async def _sample_model_data(self, model_name: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from a Kubeflow model"""
		try:
			# Simulate model version data
			model_versions = []
			for i in range(min(limit, 5)):
				version_data = {
					"model_name": model_name,
					"version": f"v1.{i}",
					"status": "READY",
					"created_time": f"2025-01-{9-i:02d}T10:00:00Z",
					"model_uri": f"s3://models/{model_name}/v1.{i}/",
					"metrics": {
						"accuracy": 0.88 + (i * 0.01),
						"precision": 0.90 + (i * 0.01),
						"recall": 0.86 + (i * 0.01)
					},
					"artifacts": [
						{"name": "model.pkl", "size": 1024 * (i + 1)},
						{"name": "preprocessor.pkl", "size": 512 * (i + 1)}
					]
				}
				model_versions.append(version_data)
			
			return model_versions
			
		except Exception as e:
			await self._log_error(f"Failed to sample model data: {str(e)}")
			return []


class SageMakerConnector(BaseConnector):
	"""AWS SageMaker metadata discovery connector - placeholder"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.ML_PLATFORM
		self.source_system = "sagemaker"
	
	async def connect(self) -> bool:
		"""Connect to AWS SageMaker"""
		try:
			# In real implementation, would use boto3 SageMaker client
			aws_region = self.config.custom_attributes.get("region", "us-east-1")
			access_key = self.config.username  # AWS Access Key ID
			secret_key = self.config.password  # AWS Secret Access Key
			
			# Simulate AWS connection
			if access_key and secret_key:
				self.is_connected = True
			else:
				# Try IAM role or environment credentials
				self.is_connected = True  # Assume available for demo
			
			return self.is_connected
		except Exception as e:
			await self._log_error(f"SageMaker connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from SageMaker"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test SageMaker connection"""
		if await self.connect():
			return {"status": "success", "message": "SageMaker connection successful"}
		else:
			return {"status": "error", "message": "Failed to connect to SageMaker"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover SageMaker models and training jobs"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not await self.connect():
				result.add_error("Failed to connect to SageMaker")
				return result
			
			# Simulate discovering SageMaker assets
			sample_models = [
				{"name": "customer-churn-model", "status": "InService", "type": "classification"},
				{"name": "sales-forecast-model", "status": "InService", "type": "regression"}
			]
			
			for model in sample_models:
				model_asset = AssetMetadata(
					name=f"sagemaker_model_{model['name']}",
					display_name=model["name"],
					asset_type="ml_model",
					source_system=self.source_system,
					description=f"SageMaker model: {model['name']}",
					custom_attributes={
						"model_status": model["status"],
						"model_type": model["type"],
						"region": self.config.custom_attributes.get("region", "us-east-1")
					}
				)
				result.add_asset(model_asset)
			
			# Simulate training jobs
			sample_training_jobs = [
				{"name": "churn-training-job-001", "status": "Completed", "algorithm": "XGBoost"},
				{"name": "forecast-training-job-002", "status": "Completed", "algorithm": "Linear Learner"}
			]
			
			for job in sample_training_jobs:
				job_asset = AssetMetadata(
					name=f"training_job_{job['name']}",
					display_name=job["name"],
					asset_type="ml_training_job",
					source_system=self.source_system,
					description=f"SageMaker training job: {job['name']}",
					custom_attributes={
						"job_status": job["status"],
						"algorithm": job["algorithm"],
						"region": self.config.custom_attributes.get("region", "us-east-1")
					}
				)
				result.add_asset(job_asset)
			
			result.complete_discovery()
			return result
			
		except Exception as e:
			result.add_error(f"SageMaker discovery failed: {str(e)}")
			result.complete_discovery()
			return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed SageMaker asset schema"""
		try:
			if not self.is_connected:
				if not await self.connect():
					return None
			
			if asset_name.startswith("sagemaker_model_"):
				model_name = asset_name.replace("sagemaker_model_", "")
				return AssetMetadata(
					name=asset_name,
					display_name=model_name,
					asset_type="ml_model",
					source_system=self.source_system,
					description=f"SageMaker model: {model_name}",
					custom_attributes={
						"model_arn": f"arn:aws:sagemaker:us-east-1:123456789012:model/{model_name}",
						"primary_container": {"image": "sagemaker-xgboost", "model_data_url": "s3://bucket/model.tar.gz"}
					}
				)
			
			elif asset_name.startswith("training_job_"):
				job_name = asset_name.replace("training_job_", "")
				return AssetMetadata(
					name=asset_name,
					display_name=job_name,
					asset_type="ml_training_job",
					source_system=self.source_system,
					description=f"SageMaker training job: {job_name}",
					custom_attributes={
						"training_job_arn": f"arn:aws:sagemaker:us-east-1:123456789012:training-job/{job_name}",
						"output_data_config": {"s3_output_path": "s3://bucket/output/"}
					}
				)
			
			return None
			
		except Exception as e:
			await self._log_error(f"Failed to get SageMaker asset schema: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get sample data from SageMaker model or training job"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to SageMaker")
				return []
			
			# Parse asset name to determine type
			if asset_name.startswith("model:"):
				model_name = asset_name[6:]  # Remove "model:" prefix
				return await self._sample_model_data(model_name, limit)
			elif asset_name.startswith("training_job:"):
				job_name = asset_name[13:]  # Remove "training_job:" prefix
				return await self._sample_training_job_data(job_name, limit)
			else:
				await self._log_error(f"Unknown SageMaker asset type: {asset_name}")
				return []
				
		except Exception as e:
			await self._log_error(f"Failed to sample SageMaker data: {str(e)}")
			return []

	async def _sample_model_data(self, model_name: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from a SageMaker model"""
		try:
			# Simulate model endpoint data
			model_versions = []
			for i in range(min(limit, 3)):
				version_data = {
					"model_name": model_name,
					"endpoint_name": f"{model_name}-endpoint-{i:02d}",
					"model_version": f"v{i+1}.0",
					"creation_time": f"2025-01-{9-i:02d}T10:00:00Z",
					"endpoint_status": "InService" if i == 0 else "OutOfService",
					"instance_type": "ml.t2.medium" if i == 0 else "ml.m5.large",
					"instance_count": 1 + i,
					"model_artifacts": {
						"s3_path": f"s3://sagemaker-models/{model_name}/v{i+1}.0/model.tar.gz",
						"size_mb": 50 + (i * 25),
						"compression": "gzip"
					},
					"performance_metrics": {
						"latency_p50_ms": 45 + (i * 10),
						"latency_p95_ms": 85 + (i * 15),
						"throughput_rps": 100 - (i * 10),
						"error_rate": 0.01 * i
					},
					"monitoring": {
						"cpu_utilization": 35.5 + (i * 5),
						"memory_utilization": 60.2 + (i * 10),
						"invocations_per_hour": 500 - (i * 50)
					}
				}
				model_versions.append(version_data)
			
			return model_versions
			
		except Exception as e:
			await self._log_error(f"Failed to sample model data: {str(e)}")
			return []

	async def _sample_training_job_data(self, job_name: str, limit: int) -> List[Dict[str, Any]]:
		"""Sample data from a SageMaker training job"""
		try:
			# Simulate training job metrics and logs
			training_data = []
			for i in range(min(limit, 10)):
				epoch_data = {
					"training_job_name": job_name,
					"epoch": i + 1,
					"timestamp": f"2025-01-09T{10+i:02d}:00:00Z",
					"training_metrics": {
						"loss": 0.8 - (i * 0.05),
						"accuracy": 0.65 + (i * 0.03),
						"precision": 0.70 + (i * 0.025),
						"recall": 0.68 + (i * 0.027),
						"f1_score": 0.69 + (i * 0.026)
					},
					"validation_metrics": {
						"val_loss": 0.85 - (i * 0.04),
						"val_accuracy": 0.62 + (i * 0.032),
						"val_precision": 0.67 + (i * 0.028),
						"val_recall": 0.65 + (i * 0.029),
						"val_f1_score": 0.66 + (i * 0.027)
					},
					"system_metrics": {
						"gpu_utilization": 85.5 + (i * 2),
						"memory_utilization": 78.3 + (i * 1.5),
						"network_rx_mb": 12.5 + (i * 0.8),
						"network_tx_mb": 8.3 + (i * 0.5)
					},
					"hyperparameters": {
						"learning_rate": 0.001,
						"batch_size": 64,
						"optimizer": "adam",
						"weight_decay": 0.0001
					}
				}
				training_data.append(epoch_data)
			
			return training_data
			
		except Exception as e:
			await self._log_error(f"Failed to sample training job data: {str(e)}")
			return []


class JupyterConnector(BaseConnector):
	"""Jupyter notebook metadata discovery connector - placeholder"""
	
	def __init__(self, config: ConnectorConfig):
		super().__init__(config)
		self.connector_type = ConnectorType.ML_PLATFORM
		self.source_system = "jupyter"
	
	async def connect(self) -> bool:
		"""Connect to Jupyter notebooks directory"""
		try:
			# Connect to local Jupyter notebooks directory
			notebook_path = Path(self.config.connection_string or "./notebooks")
			self.is_connected = notebook_path.exists() and notebook_path.is_dir()
			
			if not self.is_connected:
				# Try to create the directory if it doesn't exist
				try:
					notebook_path.mkdir(parents=True, exist_ok=True)
					self.is_connected = True
				except:
					pass
			
			return self.is_connected
		except Exception as e:
			await self._log_error(f"Jupyter connection failed: {str(e)}")
			return False
	
	async def disconnect(self):
		"""Disconnect from Jupyter"""
		self.is_connected = False
	
	async def test_connection(self) -> Dict[str, Any]:
		"""Test Jupyter connection"""
		if await self.connect():
			return {"status": "success", "message": "Jupyter connection successful"}
		else:
			return {"status": "error", "message": "Failed to connect to Jupyter notebooks directory"}
	
	async def discover_assets(self) -> DiscoveryResult:
		"""Discover Jupyter notebooks"""
		result = DiscoveryResult(self.connector_type, self.source_system)
		
		try:
			if not await self.connect():
				result.add_error("Failed to connect to Jupyter notebooks directory")
				return result
			
			notebook_path = Path(self.config.connection_string or "./notebooks")
			
			# Find all notebook files
			for notebook_file in notebook_path.rglob("*.ipynb"):
				try:
					# Read notebook metadata
					with open(notebook_file, 'r', encoding='utf-8') as f:
						notebook_data = json.load(f)
					
					# Extract metadata
					metadata = notebook_data.get("metadata", {})
					cells = notebook_data.get("cells", [])
					
					# Count cell types
					code_cells = sum(1 for cell in cells if cell.get("cell_type") == "code")
					markdown_cells = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
					
					# Get file stats
					stats = notebook_file.stat()
					
					notebook_asset = AssetMetadata(
						name=f"notebook_{notebook_file.stem}",
						display_name=notebook_file.name,
						asset_type="notebook",
						source_system=self.source_system,
						description=f"Jupyter notebook: {notebook_file.name}",
						custom_attributes={
							"file_path": str(notebook_file),
							"file_size": stats.st_size,
							"last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
							"total_cells": len(cells),
							"code_cells": code_cells,
							"markdown_cells": markdown_cells,
							"kernel_spec": metadata.get("kernelspec", {}),
							"language_info": metadata.get("language_info", {})
						}
					)
					result.add_asset(notebook_asset)
					
				except Exception as e:
					await self._log_error(f"Failed to process notebook {notebook_file}: {str(e)}")
					continue
			
			result.complete_discovery()
			return result
			
		except Exception as e:
			result.add_error(f"Jupyter discovery failed: {str(e)}")
			result.complete_discovery()
			return result
	
	async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
		"""Get detailed Jupyter notebook schema"""
		try:
			if not self.is_connected:
				if not await self.connect():
					return None
			
			notebook_name = asset_name.replace("notebook_", "")
			notebook_path = Path(self.config.connection_string or "./notebooks")
			
			# Find the notebook file
			for notebook_file in notebook_path.rglob(f"{notebook_name}.ipynb"):
				try:
					with open(notebook_file, 'r', encoding='utf-8') as f:
						notebook_data = json.load(f)
					
					metadata = notebook_data.get("metadata", {})
					cells = notebook_data.get("cells", [])
					stats = notebook_file.stat()
					
					# Extract libraries used (from imports in code cells)
					libraries = set()
					for cell in cells:
						if cell.get("cell_type") == "code":
							source = ''.join(cell.get("source", []))
							# Simple import detection
							for line in source.split('\n'):
								if line.strip().startswith(('import ', 'from ')):
									parts = line.strip().replace('from ', '').replace('import ', '').split()
									if parts:
										libraries.add(parts[0].split('.')[0])
					
					return AssetMetadata(
						name=asset_name,
						display_name=notebook_file.name,
						asset_type="notebook",
						source_system=self.source_system,
						description=f"Jupyter notebook: {notebook_file.name}",
						custom_attributes={
							"file_path": str(notebook_file),
							"file_size": stats.st_size,
							"last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
							"total_cells": len(cells),
							"code_cells": sum(1 for cell in cells if cell.get("cell_type") == "code"),
							"markdown_cells": sum(1 for cell in cells if cell.get("cell_type") == "markdown"),
							"libraries_used": list(libraries),
							"kernel_spec": metadata.get("kernelspec", {}),
							"language_info": metadata.get("language_info", {})
						}
					)
					
				except Exception as e:
					await self._log_error(f"Failed to read notebook {notebook_file}: {str(e)}")
					return None
			
			return None
			
		except Exception as e:
			await self._log_error(f"Failed to get Jupyter asset schema: {str(e)}")
			return None
	
	async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get sample data from Jupyter notebook"""
		try:
			if not self.is_connected:
				await self._log_error("Not connected to Jupyter notebooks directory")
				return []
			
			notebook_name = asset_name.replace("notebook_", "")
			notebook_path = Path(self.config.connection_string or "./notebooks")
			
			# Find the notebook file
			for notebook_file in notebook_path.rglob(f"{notebook_name}.ipynb"):
				try:
					with open(notebook_file, 'r', encoding='utf-8') as f:
						notebook_data = json.load(f)
					
					cells = notebook_data.get("cells", [])
					sample_data = []
					
					# Extract sample data from notebook cells
					for i, cell in enumerate(cells[:limit]):
						cell_data = {
							"cell_number": i + 1,
							"cell_type": cell.get("cell_type", "unknown"),
							"execution_count": cell.get("execution_count"),
							"source_preview": self._get_cell_preview(cell.get("source", [])),
							"source_length": len(''.join(cell.get("source", []))),
							"has_outputs": bool(cell.get("outputs", [])),
							"output_types": [output.get("output_type") for output in cell.get("outputs", [])],
							"output_count": len(cell.get("outputs", []))
						}
						
						# Extract metadata if available
						if cell.get("metadata"):
							cell_data["metadata"] = cell["metadata"]
						
						# For code cells, extract more details
						if cell.get("cell_type") == "code":
							cell_data["language"] = notebook_data.get("metadata", {}).get("language_info", {}).get("name", "python")
							
							# Extract imports and function definitions
							source_text = ''.join(cell.get("source", []))
							cell_data["imports"] = self._extract_imports(source_text)
							cell_data["functions"] = self._extract_functions(source_text)
							
							# Extract output data if available
							outputs = cell.get("outputs", [])
							if outputs:
								cell_data["output_preview"] = self._get_output_preview(outputs)
						
						sample_data.append(cell_data)
					
					return sample_data
					
				except Exception as e:
					await self._log_error(f"Failed to read notebook {notebook_file}: {str(e)}")
					return []
			
			# If notebook not found, return empty
			await self._log_error(f"Notebook {notebook_name} not found")
			return []
			
		except Exception as e:
			await self._log_error(f"Failed to sample Jupyter data: {str(e)}")
			return []

	def _get_cell_preview(self, source: List[str], max_length: int = 200) -> str:
		"""Get preview of cell source code"""
		if not source:
			return ""
		
		full_source = ''.join(source)
		if len(full_source) <= max_length:
			return full_source
		
		return full_source[:max_length] + "..."

	def _extract_imports(self, source_text: str) -> List[str]:
		"""Extract import statements from source code"""
		imports = []
		for line in source_text.split('\n'):
			line = line.strip()
			if line.startswith(('import ', 'from ')):
				imports.append(line)
		return imports

	def _extract_functions(self, source_text: str) -> List[str]:
		"""Extract function definitions from source code"""
		functions = []
		for line in source_text.split('\n'):
			line = line.strip()
			if line.startswith('def '):
				# Extract function name
				func_name = line[4:].split('(')[0].strip()
				functions.append(func_name)
		return functions

	def _get_output_preview(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Get preview of cell outputs"""
		output_preview = {}
		
		for output in outputs:
			output_type = output.get("output_type", "unknown")
			
			if output_type == "stream":
				text = ''.join(output.get("text", []))
				output_preview["stream"] = text[:200] + "..." if len(text) > 200 else text
			
			elif output_type == "execute_result" or output_type == "display_data":
				data = output.get("data", {})
				if "text/plain" in data:
					text_data = ''.join(data["text/plain"])
					output_preview["text"] = text_data[:200] + "..." if len(text_data) > 200 else text_data
				
				# Check for other data types
				if "image/png" in data:
					output_preview["has_image"] = True
				if "text/html" in data:
					output_preview["has_html"] = True
				if "application/json" in data:
					output_preview["has_json"] = True
			
			elif output_type == "error":
				output_preview["error"] = {
					"ename": output.get("ename", "Unknown"),
					"evalue": output.get("evalue", ""),
					"traceback_lines": len(output.get("traceback", []))
				}
		
		return output_preview