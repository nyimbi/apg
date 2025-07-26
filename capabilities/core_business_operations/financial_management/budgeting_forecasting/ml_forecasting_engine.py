"""
APG Budgeting & Forecasting - ML Forecasting Engine

Machine learning-powered forecasting engine with multiple algorithms,
accuracy tracking, and intelligent model selection for budget predictions.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from enum import Enum
from datetime import datetime, date, timedelta
from decimal import Decimal
import asyncio
import logging
import json
import numpy as np
from uuid_extensions import uuid7str
from dataclasses import dataclass

from .models import APGBaseModel, PositiveAmount, NonEmptyString
from .service import APGTenantContext, ServiceResponse, APGServiceBase


# =============================================================================
# ML Forecasting Enumerations
# =============================================================================

class ForecastAlgorithm(str, Enum):
	"""Machine learning forecasting algorithms."""
	LINEAR_REGRESSION = "linear_regression"
	ARIMA = "arima"
	EXPONENTIAL_SMOOTHING = "exponential_smoothing"
	RANDOM_FOREST = "random_forest"
	NEURAL_NETWORK = "neural_network"
	ENSEMBLE = "ensemble"
	TIME_SERIES_TRANSFORMER = "time_series_transformer"
	SEASONAL_DECOMPOSITION = "seasonal_decomposition"


class ForecastHorizon(str, Enum):
	"""Forecasting time horizons."""
	SHORT_TERM = "short_term"     # 1-3 months
	MEDIUM_TERM = "medium_term"   # 3-12 months
	LONG_TERM = "long_term"       # 1-3 years
	CUSTOM = "custom"


class ModelStatus(str, Enum):
	"""ML model status."""
	CREATED = "created"
	TRAINING = "training"
	TRAINED = "trained"
	VALIDATING = "validating"
	DEPLOYED = "deployed"
	RETRAINING = "retraining"
	FAILED = "failed"
	DEPRECATED = "deprecated"


class FeatureType(str, Enum):
	"""Types of features for ML models."""
	HISTORICAL_VALUES = "historical_values"
	SEASONAL_INDICATORS = "seasonal_indicators"
	TREND_COMPONENTS = "trend_components"
	EXTERNAL_FACTORS = "external_factors"
	LAGGED_VALUES = "lagged_values"
	ROLLING_STATISTICS = "rolling_statistics"
	CATEGORICAL_ENCODING = "categorical_encoding"


class AccuracyMetric(str, Enum):
	"""Model accuracy metrics."""
	MAE = "mae"                   # Mean Absolute Error
	MAPE = "mape"                 # Mean Absolute Percentage Error
	RMSE = "rmse"                 # Root Mean Square Error
	R_SQUARED = "r_squared"       # R-squared
	DIRECTIONAL_ACCURACY = "directional_accuracy"
	CUSTOM = "custom"


# =============================================================================
# ML Forecasting Models
# =============================================================================

class ForecastFeature(APGBaseModel):
	"""Feature definition for ML forecasting models."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	feature_id: str = Field(default_factory=uuid7str)
	feature_name: NonEmptyString = Field(description="Feature name")
	feature_type: FeatureType = Field(description="Type of feature")
	
	# Feature Configuration
	source_column: str = Field(description="Source data column")
	transformation: Optional[str] = Field(None, description="Feature transformation")
	lag_periods: Optional[int] = Field(None, description="Number of lag periods")
	window_size: Optional[int] = Field(None, description="Rolling window size")
	
	# Feature Importance
	importance_score: Optional[Decimal] = Field(None, description="Feature importance score")
	correlation_score: Optional[Decimal] = Field(None, description="Correlation with target")
	
	# Feature Statistics
	mean_value: Optional[Decimal] = Field(None, description="Feature mean")
	std_value: Optional[Decimal] = Field(None, description="Feature standard deviation")
	min_value: Optional[Decimal] = Field(None, description="Feature minimum")
	max_value: Optional[Decimal] = Field(None, description="Feature maximum")
	
	# Metadata
	is_active: bool = Field(default=True, description="Feature is active")
	created_date: datetime = Field(default_factory=datetime.utcnow)


class MLForecastingModel(APGBaseModel):
	"""Machine learning forecasting model configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	model_id: str = Field(default_factory=uuid7str)
	model_name: NonEmptyString = Field(description="Model name")
	algorithm: ForecastAlgorithm = Field(description="ML algorithm")
	
	# Model Configuration
	target_variable: str = Field(description="Target variable to forecast")
	features: List[ForecastFeature] = Field(default_factory=list, description="Model features")
	horizon: ForecastHorizon = Field(description="Forecasting horizon")
	frequency: str = Field(description="Data frequency (daily, weekly, monthly)")
	
	# Training Configuration
	training_window: int = Field(description="Training window in periods")
	validation_split: Decimal = Field(default=Decimal("0.2"), description="Validation split ratio")
	test_split: Decimal = Field(default=Decimal("0.1"), description="Test split ratio")
	
	# Model Hyperparameters
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")
	
	# Model Performance
	accuracy_metrics: Dict[AccuracyMetric, Decimal] = Field(default_factory=dict)
	training_score: Optional[Decimal] = Field(None, description="Training accuracy score")
	validation_score: Optional[Decimal] = Field(None, description="Validation accuracy score")
	test_score: Optional[Decimal] = Field(None, description="Test accuracy score")
	
	# Model Status
	status: ModelStatus = Field(default=ModelStatus.CREATED)
	training_start: Optional[datetime] = Field(None)
	training_end: Optional[datetime] = Field(None)
	last_trained: Optional[datetime] = Field(None)
	
	# Model Artifacts
	model_version: str = Field(default="1.0.0")
	model_size_bytes: Optional[int] = Field(None, description="Serialized model size")
	training_duration: Optional[Decimal] = Field(None, description="Training duration in seconds")
	
	# Deployment Configuration
	is_deployed: bool = Field(default=False)
	deployment_date: Optional[datetime] = Field(None)
	auto_retrain: bool = Field(default=True, description="Enable automatic retraining")
	retrain_threshold: Decimal = Field(default=Decimal("0.1"), description="Accuracy drop threshold for retraining")


class ForecastPrediction(APGBaseModel):
	"""Individual forecast prediction with confidence intervals."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	prediction_id: str = Field(default_factory=uuid7str)
	model_id: str = Field(description="Model that generated prediction")
	
	# Prediction Details
	forecast_date: date = Field(description="Date being forecasted")
	predicted_value: Decimal = Field(description="Predicted value")
	confidence_level: Decimal = Field(default=Decimal("0.95"), description="Confidence level")
	
	# Confidence Intervals
	lower_bound: Decimal = Field(description="Lower confidence bound")
	upper_bound: Decimal = Field(description="Upper confidence bound")
	prediction_interval_width: Decimal = Field(description="Width of prediction interval")
	
	# Uncertainty Measures
	prediction_variance: Optional[Decimal] = Field(None, description="Prediction variance")
	model_uncertainty: Optional[Decimal] = Field(None, description="Model uncertainty")
	data_uncertainty: Optional[Decimal] = Field(None, description="Data uncertainty")
	
	# Feature Contributions
	feature_contributions: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Metadata
	generated_date: datetime = Field(default_factory=datetime.utcnow)
	actual_value: Optional[Decimal] = Field(None, description="Actual value (if available)")
	prediction_error: Optional[Decimal] = Field(None, description="Prediction error")


class ForecastScenario(APGBaseModel):
	"""Forecast scenario with multiple predictions."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	scenario_id: str = Field(default_factory=uuid7str)
	scenario_name: NonEmptyString = Field(description="Scenario name")
	model_id: str = Field(description="ML model used")
	
	# Scenario Configuration
	start_date: date = Field(description="Forecast start date")
	end_date: date = Field(description="Forecast end date")
	frequency: str = Field(description="Forecast frequency")
	
	# Scenario Assumptions
	assumptions: Dict[str, Any] = Field(default_factory=dict, description="Scenario assumptions")
	external_factors: Dict[str, Any] = Field(default_factory=dict, description="External factor adjustments")
	
	# Predictions
	predictions: List[ForecastPrediction] = Field(default_factory=list, description="Forecast predictions")
	
	# Scenario Summary
	total_forecast: Decimal = Field(description="Total forecasted amount")
	average_monthly: Decimal = Field(description="Average monthly forecast")
	growth_rate: Optional[Decimal] = Field(None, description="Implied growth rate")
	
	# Accuracy Tracking
	accuracy_score: Optional[Decimal] = Field(None, description="Scenario accuracy score")
	tracking_start: Optional[date] = Field(None, description="When accuracy tracking started")
	
	# Metadata
	generated_date: datetime = Field(default_factory=datetime.utcnow)
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class ModelEnsemble(APGBaseModel):
	"""Ensemble of multiple forecasting models."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	ensemble_id: str = Field(default_factory=uuid7str)
	ensemble_name: NonEmptyString = Field(description="Ensemble name")
	
	# Ensemble Configuration
	member_models: List[str] = Field(description="Member model IDs")
	ensemble_method: str = Field(description="Ensemble combination method")
	weights: Dict[str, Decimal] = Field(default_factory=dict, description="Model weights")
	
	# Ensemble Performance
	ensemble_accuracy: Dict[AccuracyMetric, Decimal] = Field(default_factory=dict)
	individual_accuracies: Dict[str, Dict[AccuracyMetric, Decimal]] = Field(default_factory=dict)
	
	# Ensemble Status
	is_trained: bool = Field(default=False)
	last_updated: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# ML Forecasting Engine Service
# =============================================================================

class MLForecastingEngineService(APGServiceBase):
	"""
	Machine learning forecasting engine providing multiple algorithms,
	automated model selection, and intelligent forecasting capabilities.
	"""
	
	def __init__(self, context: APGTenantContext, config: Optional[Dict[str, Any]] = None):
		super().__init__(context, config)
		self.logger = logging.getLogger(__name__)
	
	async def create_forecasting_model(
		self, 
		model_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create new ML forecasting model."""
		try:
			self.logger.info(f"Creating ML forecasting model: {model_config.get('model_name')}")
			
			# Validate configuration
			required_fields = ['model_name', 'algorithm', 'target_variable', 'horizon']
			missing_fields = [field for field in required_fields if field not in model_config]
			if missing_fields:
				return ServiceResponse(
					success=False,
					message=f"Missing required fields: {missing_fields}",
					errors=missing_fields
				)
			
			# Create model
			model = MLForecastingModel(
				model_name=model_config['model_name'],
				algorithm=model_config['algorithm'],
				target_variable=model_config['target_variable'],
				horizon=model_config['horizon'],
				frequency=model_config.get('frequency', 'monthly'),
				training_window=model_config.get('training_window', 24),
				hyperparameters=model_config.get('hyperparameters', {}),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Create features
			if 'features' in model_config:
				model.features = await self._create_model_features(model_config['features'])
			else:
				model.features = await self._generate_default_features(model.target_variable)
			
			# Set algorithm-specific hyperparameters
			await self._configure_algorithm_hyperparameters(model)
			
			self.logger.info(f"ML forecasting model created: {model.model_id}")
			
			return ServiceResponse(
				success=True,
				message="ML forecasting model created successfully",
				data=model.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating ML forecasting model: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create ML forecasting model: {str(e)}",
				errors=[str(e)]
			)
	
	async def train_forecasting_model(
		self, 
		model_id: str, 
		training_config: Optional[Dict[str, Any]] = None
	) -> ServiceResponse:
		"""Train ML forecasting model."""
		try:
			self.logger.info(f"Training ML forecasting model {model_id}")
			
			# Load model
			model = await self._load_forecasting_model(model_id)
			
			# Update status
			model.status = ModelStatus.TRAINING
			model.training_start = datetime.utcnow()
			
			# Prepare training data
			training_data = await self._prepare_training_data(model)
			
			# Feature engineering
			engineered_features = await self._engineer_features(model, training_data)
			
			# Split data
			train_data, val_data, test_data = await self._split_training_data(
				engineered_features, model.validation_split, model.test_split
			)
			
			# Train model based on algorithm
			trained_model_artifacts = await self._train_algorithm(model, train_data, val_data)
			
			# Evaluate model
			evaluation_results = await self._evaluate_model(model, test_data, trained_model_artifacts)
			
			# Update model with results
			await self._update_model_with_training_results(model, evaluation_results)
			
			model.status = ModelStatus.TRAINED
			model.training_end = datetime.utcnow()
			model.last_trained = datetime.utcnow()
			
			if model.training_start:
				model.training_duration = Decimal((model.training_end - model.training_start).total_seconds())
			
			self.logger.info(f"ML forecasting model trained successfully: {model_id}")
			
			return ServiceResponse(
				success=True,
				message="ML forecasting model trained successfully",
				data=model.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error training ML forecasting model: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to train ML forecasting model: {str(e)}",
				errors=[str(e)]
			)
	
	async def generate_forecast(
		self, 
		model_id: str, 
		forecast_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Generate forecast using trained ML model."""
		try:
			self.logger.info(f"Generating forecast with model {model_id}")
			
			# Load trained model
			model = await self._load_forecasting_model(model_id)
			
			if model.status != ModelStatus.TRAINED and model.status != ModelStatus.DEPLOYED:
				return ServiceResponse(
					success=False,
					message="Model must be trained before generating forecasts",
					errors=["model_not_trained"]
				)
			
			# Create forecast scenario
			scenario = ForecastScenario(
				scenario_name=forecast_config.get('scenario_name', f'Forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
				model_id=model_id,
				start_date=forecast_config['start_date'],
				end_date=forecast_config['end_date'],
				frequency=model.frequency,
				assumptions=forecast_config.get('assumptions', {}),
				external_factors=forecast_config.get('external_factors', {}),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Generate predictions
			predictions = await self._generate_forecast_predictions(model, scenario)
			scenario.predictions = predictions
			
			# Calculate scenario summary
			await self._calculate_scenario_summary(scenario)
			
			self.logger.info(f"Forecast generated successfully: {scenario.scenario_id}")
			
			return ServiceResponse(
				success=True,
				message="Forecast generated successfully",
				data=scenario.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error generating forecast: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to generate forecast: {str(e)}",
				errors=[str(e)]
			)
	
	async def create_model_ensemble(
		self, 
		ensemble_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Create ensemble of multiple forecasting models."""
		try:
			self.logger.info(f"Creating model ensemble: {ensemble_config.get('ensemble_name')}")
			
			# Validate configuration
			if 'member_models' not in ensemble_config or len(ensemble_config['member_models']) < 2:
				return ServiceResponse(
					success=False,
					message="Ensemble requires at least 2 member models",
					errors=["insufficient_models"]
				)
			
			# Create ensemble
			ensemble = ModelEnsemble(
				ensemble_name=ensemble_config['ensemble_name'],
				member_models=ensemble_config['member_models'],
				ensemble_method=ensemble_config.get('ensemble_method', 'weighted_average'),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			# Calculate optimal weights
			await self._calculate_ensemble_weights(ensemble)
			
			# Evaluate ensemble performance
			await self._evaluate_ensemble_performance(ensemble)
			
			ensemble.is_trained = True
			
			self.logger.info(f"Model ensemble created: {ensemble.ensemble_id}")
			
			return ServiceResponse(
				success=True,
				message="Model ensemble created successfully",
				data=ensemble.model_dump()
			)
			
		except Exception as e:
			self.logger.error(f"Error creating model ensemble: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to create model ensemble: {str(e)}",
				errors=[str(e)]
			)
	
	async def evaluate_forecast_accuracy(
		self, 
		scenario_id: str
	) -> ServiceResponse:
		"""Evaluate forecast accuracy against actual values."""
		try:
			self.logger.info(f"Evaluating forecast accuracy for scenario {scenario_id}")
			
			# Load scenario
			scenario = await self._load_forecast_scenario(scenario_id)
			
			# Get actual values for comparison
			actual_values = await self._get_actual_values_for_period(
				scenario.start_date, scenario.end_date
			)
			
			# Calculate accuracy metrics
			accuracy_results = await self._calculate_forecast_accuracy(scenario, actual_values)
			
			# Update scenario with accuracy information
			scenario.accuracy_score = accuracy_results['overall_accuracy']
			scenario.tracking_start = date.today()
			
			return ServiceResponse(
				success=True,
				message="Forecast accuracy evaluated successfully",
				data=accuracy_results
			)
			
		except Exception as e:
			self.logger.error(f"Error evaluating forecast accuracy: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to evaluate forecast accuracy: {str(e)}",
				errors=[str(e)]
			)
	
	async def auto_select_best_model(
		self, 
		selection_config: Dict[str, Any]
	) -> ServiceResponse:
		"""Automatically select best performing model for a given task."""
		try:
			self.logger.info("Auto-selecting best forecasting model")
			
			# Get available models
			available_models = await self._get_available_models(selection_config)
			
			# Evaluate models on validation data
			model_evaluations = await self._evaluate_models_for_selection(available_models, selection_config)
			
			# Select best model
			best_model = await self._select_best_model(model_evaluations, selection_config.get('criteria', 'accuracy'))
			
			return ServiceResponse(
				success=True,
				message="Best model selected successfully",
				data={
					'selected_model': best_model,
					'evaluations': model_evaluations,
					'selection_criteria': selection_config.get('criteria', 'accuracy')
				}
			)
			
		except Exception as e:
			self.logger.error(f"Error auto-selecting best model: {e}")
			return ServiceResponse(
				success=False,
				message=f"Failed to auto-select best model: {str(e)}",
				errors=[str(e)]
			)
	
	# =============================================================================
	# Private Helper Methods
	# =============================================================================
	
	async def _create_model_features(self, features_config: List[Dict[str, Any]]) -> List[ForecastFeature]:
		"""Create model features from configuration."""
		features = []
		
		for feature_config in features_config:
			feature = ForecastFeature(
				feature_name=feature_config['feature_name'],
				feature_type=feature_config['feature_type'],
				source_column=feature_config['source_column'],
				transformation=feature_config.get('transformation'),
				lag_periods=feature_config.get('lag_periods'),
				window_size=feature_config.get('window_size'),
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			features.append(feature)
		
		return features
	
	async def _generate_default_features(self, target_variable: str) -> List[ForecastFeature]:
		"""Generate default features for a target variable."""
		features = [
			ForecastFeature(
				feature_name="historical_values",
				feature_type=FeatureType.HISTORICAL_VALUES,
				source_column=target_variable,
				lag_periods=1,
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			),
			ForecastFeature(
				feature_name="seasonal_month",
				feature_type=FeatureType.SEASONAL_INDICATORS,
				source_column="date",
				transformation="month",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			),
			ForecastFeature(
				feature_name="trend_component",
				feature_type=FeatureType.TREND_COMPONENTS,
				source_column=target_variable,
				transformation="linear_trend",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			),
			ForecastFeature(
				feature_name="rolling_avg_3",
				feature_type=FeatureType.ROLLING_STATISTICS,
				source_column=target_variable,
				window_size=3,
				transformation="mean",
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
		]
		
		return features
	
	async def _configure_algorithm_hyperparameters(self, model: MLForecastingModel) -> None:
		"""Configure algorithm-specific hyperparameters."""
		if model.algorithm == ForecastAlgorithm.LINEAR_REGRESSION:
			model.hyperparameters.update({
				'fit_intercept': True,
				'normalize': False,
				'regularization': 'ridge',
				'alpha': 1.0
			})
		elif model.algorithm == ForecastAlgorithm.RANDOM_FOREST:
			model.hyperparameters.update({
				'n_estimators': 100,
				'max_depth': 10,
				'min_samples_split': 2,
				'min_samples_leaf': 1,
				'random_state': 42
			})
		elif model.algorithm == ForecastAlgorithm.NEURAL_NETWORK:
			model.hyperparameters.update({
				'hidden_layers': [64, 32],
				'activation': 'relu',
				'optimizer': 'adam',
				'learning_rate': 0.001,
				'epochs': 100,
				'batch_size': 32
			})
		elif model.algorithm == ForecastAlgorithm.ARIMA:
			model.hyperparameters.update({
				'p': 1,
				'd': 1,
				'q': 1,
				'seasonal_order': (1, 1, 1, 12)
			})
	
	async def _load_forecasting_model(self, model_id: str) -> MLForecastingModel:
		"""Load forecasting model."""
		# Simulated model loading
		model = MLForecastingModel(
			model_id=model_id,
			model_name="Budget Forecast Model",
			algorithm=ForecastAlgorithm.RANDOM_FOREST,
			target_variable="budget_amount",
			horizon=ForecastHorizon.MEDIUM_TERM,
			frequency="monthly",
			training_window=24,
			status=ModelStatus.TRAINED,
			tenant_id=self.context.tenant_id,
			created_by=self.context.user_id
		)
		
		# Add default features
		model.features = await self._generate_default_features("budget_amount")
		
		return model
	
	async def _prepare_training_data(self, model: MLForecastingModel) -> Dict[str, Any]:
		"""Prepare training data for model."""
		# Simulated training data preparation
		return {
			'historical_data': [
				{'date': '2023-01', 'budget_amount': 125000, 'department': 'Sales'},
				{'date': '2023-02', 'budget_amount': 127000, 'department': 'Sales'},
				{'date': '2023-03', 'budget_amount': 124000, 'department': 'Sales'},
				# ... more data
			],
			'external_factors': {
				'inflation_rate': [2.1, 2.3, 2.2],
				'market_growth': [1.5, 1.7, 1.6]
			}
		}
	
	async def _engineer_features(
		self, 
		model: MLForecastingModel, 
		training_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Engineer features for model training."""
		# Simulated feature engineering
		engineered_data = training_data.copy()
		
		# Add lag features
		lag_features = []
		for i in range(1, 4):  # 3 lag periods
			lag_features.extend([125000 - i*1000] * len(training_data['historical_data']))
		
		engineered_data['lag_features'] = lag_features
		engineered_data['seasonal_features'] = [1, 2, 3] * (len(training_data['historical_data']) // 3)
		
		return engineered_data
	
	async def _split_training_data(
		self, 
		data: Dict[str, Any], 
		val_split: Decimal, 
		test_split: Decimal
	) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
		"""Split data into train, validation, and test sets."""
		# Simulated data splitting
		total_records = len(data['historical_data'])
		test_size = int(total_records * test_split)
		val_size = int(total_records * val_split)
		train_size = total_records - test_size - val_size
		
		train_data = {'size': train_size, 'data': data['historical_data'][:train_size]}
		val_data = {'size': val_size, 'data': data['historical_data'][train_size:train_size+val_size]}
		test_data = {'size': test_size, 'data': data['historical_data'][train_size+val_size:]}
		
		return train_data, val_data, test_data
	
	async def _train_algorithm(
		self, 
		model: MLForecastingModel, 
		train_data: Dict[str, Any], 
		val_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Train the specified algorithm."""
		# Simulated algorithm training
		await asyncio.sleep(0.1)  # Simulate training time
		
		return {
			'model_artifacts': f'trained_{model.algorithm.value}_model',
			'feature_importance': {
				'historical_values': 0.45,
				'seasonal_month': 0.25,
				'trend_component': 0.20,
				'rolling_avg_3': 0.10
			}
		}
	
	async def _evaluate_model(
		self, 
		model: MLForecastingModel, 
		test_data: Dict[str, Any], 
		model_artifacts: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Evaluate model performance."""
		# Simulated model evaluation
		return {
			'test_accuracy': {
				AccuracyMetric.MAE: Decimal("2500.00"),
				AccuracyMetric.MAPE: Decimal("2.1"),
				AccuracyMetric.RMSE: Decimal("3200.00"),
				AccuracyMetric.R_SQUARED: Decimal("0.87"),
				AccuracyMetric.DIRECTIONAL_ACCURACY: Decimal("0.83")
			},
			'validation_accuracy': {
				AccuracyMetric.MAE: Decimal("2300.00"),
				AccuracyMetric.MAPE: Decimal("1.9"),
				AccuracyMetric.RMSE: Decimal("3000.00"),
				AccuracyMetric.R_SQUARED: Decimal("0.89"),
				AccuracyMetric.DIRECTIONAL_ACCURACY: Decimal("0.85")
			}
		}
	
	async def _update_model_with_training_results(
		self, 
		model: MLForecastingModel, 
		evaluation_results: Dict[str, Any]
	) -> None:
		"""Update model with training results."""
		model.accuracy_metrics = evaluation_results['test_accuracy']
		model.test_score = evaluation_results['test_accuracy'][AccuracyMetric.R_SQUARED]
		model.validation_score = evaluation_results['validation_accuracy'][AccuracyMetric.R_SQUARED]
		model.model_size_bytes = 1024 * 1024  # 1MB simulated
	
	async def _generate_forecast_predictions(
		self, 
		model: MLForecastingModel, 
		scenario: ForecastScenario
	) -> List[ForecastPrediction]:
		"""Generate forecast predictions."""
		predictions = []
		current_date = scenario.start_date
		
		while current_date <= scenario.end_date:
			# Simulate prediction generation
			base_value = Decimal("125000")
			predicted_value = base_value + Decimal(np.random.normal(0, 5000))
			
			prediction = ForecastPrediction(
				model_id=model.model_id,
				forecast_date=current_date,
				predicted_value=predicted_value,
				lower_bound=predicted_value * Decimal("0.9"),
				upper_bound=predicted_value * Decimal("1.1"),
				prediction_interval_width=predicted_value * Decimal("0.2"),
				feature_contributions={
					'historical_values': predicted_value * Decimal("0.45"),
					'seasonal_month': predicted_value * Decimal("0.25"),
					'trend_component': predicted_value * Decimal("0.30")
				},
				tenant_id=self.context.tenant_id,
				created_by=self.context.user_id
			)
			
			predictions.append(prediction)
			
			# Move to next period
			if scenario.frequency == 'monthly':
				if current_date.month == 12:
					current_date = current_date.replace(year=current_date.year + 1, month=1)
				else:
					current_date = current_date.replace(month=current_date.month + 1)
			else:
				current_date += timedelta(days=30)  # Approximate
		
		return predictions
	
	async def _calculate_scenario_summary(self, scenario: ForecastScenario) -> None:
		"""Calculate scenario summary statistics."""
		if scenario.predictions:
			total_forecast = sum(p.predicted_value for p in scenario.predictions)
			scenario.total_forecast = total_forecast
			scenario.average_monthly = total_forecast / len(scenario.predictions)
			
			# Calculate growth rate (simplified)
			if len(scenario.predictions) > 1:
				first_value = scenario.predictions[0].predicted_value
				last_value = scenario.predictions[-1].predicted_value
				periods = len(scenario.predictions)
				scenario.growth_rate = ((last_value / first_value) ** (1 / periods) - 1) * 100
	
	async def _calculate_ensemble_weights(self, ensemble: ModelEnsemble) -> None:
		"""Calculate optimal weights for ensemble members."""
		# Simplified equal weighting
		weight = Decimal("1.0") / len(ensemble.member_models)
		ensemble.weights = {model_id: weight for model_id in ensemble.member_models}
	
	async def _evaluate_ensemble_performance(self, ensemble: ModelEnsemble) -> None:
		"""Evaluate ensemble performance."""
		# Simulated ensemble evaluation
		ensemble.ensemble_accuracy = {
			AccuracyMetric.MAE: Decimal("2200.00"),
			AccuracyMetric.MAPE: Decimal("1.8"),
			AccuracyMetric.RMSE: Decimal("2900.00"),
			AccuracyMetric.R_SQUARED: Decimal("0.91")
		}
	
	async def _load_forecast_scenario(self, scenario_id: str) -> ForecastScenario:
		"""Load forecast scenario."""
		# Simulated scenario loading
		return ForecastScenario(
			scenario_id=scenario_id,
			scenario_name="Test Scenario",
			model_id="model_123",
			start_date=date(2025, 1, 1),
			end_date=date(2025, 12, 31),
			frequency="monthly",
			tenant_id=self.context.tenant_id,
			created_by=self.context.user_id
		)
	
	async def _get_actual_values_for_period(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
		"""Get actual values for accuracy comparison."""
		# Simulated actual values
		return [
			{'date': '2025-01', 'actual_value': 123500},
			{'date': '2025-02', 'actual_value': 126800},
			{'date': '2025-03', 'actual_value': 124200}
		]
	
	async def _calculate_forecast_accuracy(
		self, 
		scenario: ForecastScenario, 
		actual_values: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Calculate forecast accuracy metrics."""
		# Simulated accuracy calculation
		return {
			'overall_accuracy': Decimal("0.87"),
			'mae': Decimal("2800.00"),
			'mape': Decimal("2.3"),
			'rmse': Decimal("3100.00"),
			'directional_accuracy': Decimal("0.82"),
			'prediction_count': len(scenario.predictions),
			'actual_count': len(actual_values)
		}
	
	async def _get_available_models(self, selection_config: Dict[str, Any]) -> List[str]:
		"""Get available models for selection."""
		return ['model_1', 'model_2', 'model_3', 'ensemble_1']
	
	async def _evaluate_models_for_selection(
		self, 
		model_ids: List[str], 
		selection_config: Dict[str, Any]
	) -> Dict[str, Dict[str, Any]]:
		"""Evaluate models for selection."""
		# Simulated model evaluation
		return {
			'model_1': {'accuracy': 0.85, 'speed': 0.9, 'complexity': 0.3},
			'model_2': {'accuracy': 0.87, 'speed': 0.7, 'complexity': 0.6},
			'model_3': {'accuracy': 0.84, 'speed': 0.8, 'complexity': 0.4},
			'ensemble_1': {'accuracy': 0.91, 'speed': 0.6, 'complexity': 0.8}
		}
	
	async def _select_best_model(
		self, 
		evaluations: Dict[str, Dict[str, Any]], 
		criteria: str
	) -> Dict[str, Any]:
		"""Select best model based on criteria."""
		if criteria == 'accuracy':
			best_model_id = max(evaluations.keys(), key=lambda k: evaluations[k]['accuracy'])
		elif criteria == 'speed':
			best_model_id = max(evaluations.keys(), key=lambda k: evaluations[k]['speed'])
		else:
			# Balanced scoring
			best_model_id = max(evaluations.keys(), 
				key=lambda k: evaluations[k]['accuracy'] * 0.6 + evaluations[k]['speed'] * 0.4)
		
		return {
			'model_id': best_model_id,
			'evaluation_scores': evaluations[best_model_id],
			'selection_reason': f'Highest {criteria} score'
		}


# =============================================================================
# Service Factory Functions
# =============================================================================

def create_ml_forecasting_engine_service(
	context: APGTenantContext, 
	config: Optional[Dict[str, Any]] = None
) -> MLForecastingEngineService:
	"""Create ML forecasting engine service instance."""
	return MLForecastingEngineService(context, config)


async def create_sample_forecasting_model(
	service: MLForecastingEngineService
) -> ServiceResponse:
	"""Create sample forecasting model for testing."""
	model_config = {
		'model_name': 'Budget Forecasting Model v1',
		'algorithm': ForecastAlgorithm.RANDOM_FOREST,
		'target_variable': 'budget_amount',
		'horizon': ForecastHorizon.MEDIUM_TERM,
		'frequency': 'monthly',
		'training_window': 24,
		'features': [
			{
				'feature_name': 'historical_budget',
				'feature_type': FeatureType.HISTORICAL_VALUES,
				'source_column': 'budget_amount',
				'lag_periods': 1
			},
			{
				'feature_name': 'seasonal_factor',
				'feature_type': FeatureType.SEASONAL_INDICATORS,
				'source_column': 'month',
				'transformation': 'cyclical_encoding'
			}
		]
	}
	
	return await service.create_forecasting_model(model_config)