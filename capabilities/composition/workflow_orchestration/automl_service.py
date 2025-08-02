"""
APG Workflow AutoML Service

Automated machine learning service for workflow optimization and intelligent automation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path
import pandas as pd

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from uuid_extensions import uuid7str

# AutoML Libraries
try:
	from autogluon.tabular import TabularPredictor
	from autogluon.multimodal import MultiModalPredictor
	from autogluon.timeseries import TimeSeriesPredictor
	AUTOGLUON_AVAILABLE = True
except ImportError:
	AUTOGLUON_AVAILABLE = False

try:
	import optuna
	from optuna.integration import SklearnOptimizer
	OPTUNA_AVAILABLE = True
except ImportError:
	OPTUNA_AVAILABLE = False

try:
	from h2o.automl import H2OAutoML
	import h2o
	H2O_AVAILABLE = True
except ImportError:
	H2O_AVAILABLE = False

try:
	from auto_ml import Predictor as AutoMLPredictor
	AUTO_ML_AVAILABLE = True
except ImportError:
	AUTO_ML_AVAILABLE = False

from .models import WorkflowTemplate, WorkflowExecution, WorkflowNode
from .ml_engine import MLModelMetrics, MLTask

logger = logging.getLogger(__name__)

@dataclass
class AutoMLExperiment:
	"""AutoML experiment tracking."""
	experiment_id: str = field(default_factory=uuid7str)
	name: str = ""
	description: str = ""
	
	# Dataset information
	dataset_name: str = ""
	dataset_size: int = 0
	feature_count: int = 0
	target_type: str = ""  # classification, regression, timeseries
	
	# Experiment configuration
	time_budget_minutes: int = 60
	quality_preset: str = "medium_quality"
	ensemble_size: int = 10
	
	# Results
	best_model_name: str = ""
	best_score: float = 0.0
	total_models_trained: int = 0
	experiment_duration: float = 0.0
	
	# Status
	status: str = "pending"  # pending, running, completed, failed
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	
	# Model artifacts
	model_path: str = ""
	leaderboard: List[Dict[str, Any]] = field(default_factory=list)
	feature_importance: Dict[str, float] = field(default_factory=dict)

class AutoMLTaskType(str):
	"""AutoML task types."""
	TABULAR_CLASSIFICATION = "tabular_classification"
	TABULAR_REGRESSION = "tabular_regression"
	TIME_SERIES_FORECASTING = "time_series_forecasting"
	MULTIMODAL_PREDICTION = "multimodal_prediction"
	HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
	NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"

class AutoMLConfig(BaseModel):
	"""AutoML configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	task_type: AutoMLTaskType = Field(..., description="AutoML task type")
	time_budget_minutes: int = Field(default=60, ge=1, le=1440, description="Time budget in minutes")
	quality_preset: str = Field(default="medium_quality", description="Quality preset")
	
	# Data configuration
	train_data_path: str = Field(..., description="Training data path")
	test_data_path: Optional[str] = Field(default=None, description="Test data path")
	target_column: str = Field(..., description="Target column name")
	
	# Model configuration
	eval_metric: str = Field(default="auto", description="Evaluation metric")
	optimization_metric: str = Field(default="accuracy", description="Optimization metric")
	ensemble_size: int = Field(default=10, ge=1, le=50, description="Ensemble size")
	
	# Advanced options
	include_models: List[str] = Field(default_factory=list, description="Models to include")
	exclude_models: List[str] = Field(default_factory=list, description="Models to exclude")
	hyperparameter_tune_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameter tuning options")
	
	# Cross-validation
	num_bag_folds: int = Field(default=8, ge=2, le=20, description="Number of bagging folds")
	num_stack_levels: int = Field(default=1, ge=0, le=3, description="Number of stacking levels")
	auto_stack: bool = Field(default=True, description="Enable auto stacking")
	
	# Resource limits
	memory_limit: Optional[str] = Field(default=None, description="Memory limit (e.g., '4GB')")
	cpu_count: Optional[int] = Field(default=None, description="CPU count limit")
	gpu_count: Optional[int] = Field(default=None, description="GPU count")

class HyperparameterSpace(BaseModel):
	"""Hyperparameter optimization space."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	space_id: str = Field(default_factory=uuid7str, description="Space ID")
	algorithm: str = Field(..., description="Algorithm name")
	
	# Parameter definitions
	int_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Integer parameters")
	float_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Float parameters")
	categorical_params: Dict[str, List[str]] = Field(default_factory=dict, description="Categorical parameters")
	
	# Optimization settings
	n_trials: int = Field(default=100, description="Number of optimization trials")
	timeout: Optional[int] = Field(default=None, description="Timeout in seconds")
	pruner: str = Field(default="median", description="Pruning algorithm")
	sampler: str = Field(default="tpe", description="Sampling algorithm")

class AutoMLService:
	"""
	Automated Machine Learning Service for workflow optimization.
	
	Features:
	- Automated model selection and training
	- Hyperparameter optimization
	- Neural architecture search
	- Time series forecasting
	- Multimodal learning
	- Model interpretation and explanation
	"""
	
	def __init__(self, config: Dict[str, Any], db_session: Session):
		self.config = config
		self.db_session = db_session
		self.experiments: Dict[str, AutoMLExperiment] = {}
		self.active_predictors: Dict[str, Any] = {}
		
		# Initialize H2O if available
		self.h2o_initialized = False
		if H2O_AVAILABLE:
			try:
				h2o.init(nthreads=-1, max_mem_size="4G")
				self.h2o_initialized = True
				logger.info("H2O AutoML initialized")
			except Exception as e:
				logger.warning(f"Failed to initialize H2O: {e}")
		
		logger.info("AutoML Service initialized")
	
	async def run_automl_experiment(self, automl_config: AutoMLConfig, experiment_name: str = "") -> str:
		"""Run AutoML experiment."""
		experiment = AutoMLExperiment(
			name=experiment_name or f"AutoML_{automl_config.task_type}",
			description=f"AutoML experiment for {automl_config.task_type}",
			time_budget_minutes=automl_config.time_budget_minutes,
			quality_preset=automl_config.quality_preset,
			ensemble_size=automl_config.ensemble_size
		)
		
		self.experiments[experiment.experiment_id] = experiment
		
		try:
			experiment.status = "running"
			experiment.started_at = datetime.now(timezone.utc)
			
			# Load and validate data
			train_data = await self._load_training_data(automl_config)
			experiment.dataset_name = Path(automl_config.train_data_path).name
			experiment.dataset_size = len(train_data)
			experiment.feature_count = len(train_data.columns) - 1  # Excluding target
			experiment.target_type = automl_config.task_type
			
			# Run AutoML based on task type
			if automl_config.task_type == AutoMLTaskType.TABULAR_CLASSIFICATION:
				results = await self._run_tabular_classification(automl_config, train_data, experiment)
			elif automl_config.task_type == AutoMLTaskType.TABULAR_REGRESSION:
				results = await self._run_tabular_regression(automl_config, train_data, experiment)
			elif automl_config.task_type == AutoMLTaskType.TIME_SERIES_FORECASTING:
				results = await self._run_time_series_forecasting(automl_config, train_data, experiment)
			elif automl_config.task_type == AutoMLTaskType.MULTIMODAL_PREDICTION:
				results = await self._run_multimodal_prediction(automl_config, train_data, experiment)
			elif automl_config.task_type == AutoMLTaskType.HYPERPARAMETER_OPTIMIZATION:
				results = await self._run_hyperparameter_optimization(automl_config, train_data, experiment)
			else:
				raise ValueError(f"Unsupported AutoML task type: {automl_config.task_type}")
			
			# Update experiment with results
			experiment.best_model_name = results.get('best_model', 'Unknown')
			experiment.best_score = results.get('best_score', 0.0)
			experiment.total_models_trained = results.get('models_trained', 0)
			experiment.leaderboard = results.get('leaderboard', [])
			experiment.feature_importance = results.get('feature_importance', {})
			experiment.model_path = results.get('model_path', '')
			
			experiment.status = "completed"
			experiment.completed_at = datetime.now(timezone.utc)
			experiment.experiment_duration = (experiment.completed_at - experiment.started_at).total_seconds()
			
			logger.info(f"AutoML experiment completed: {experiment.experiment_id}")
			return experiment.experiment_id
			
		except Exception as e:
			experiment.status = "failed"
			experiment.completed_at = datetime.now(timezone.utc)
			logger.error(f"AutoML experiment failed: {e}")
			raise
	
	async def _run_tabular_classification(self, config: AutoMLConfig, data: pd.DataFrame, experiment: AutoMLExperiment) -> Dict[str, Any]:
		"""Run tabular classification AutoML."""
		if not AUTOGLUON_AVAILABLE:
			return await self._run_sklearn_automl(config, data, "classification")
		
		try:
			# Split data if test data not provided
			if config.test_data_path:
				test_data = pd.read_csv(config.test_data_path)
			else:
				train_data = data.sample(frac=0.8, random_state=42)
				test_data = data.drop(train_data.index)
				data = train_data
			
			# Configure AutoGluon predictor
			predictor = TabularPredictor(
				label=config.target_column,
				eval_metric=config.eval_metric,
				path=f'./automl_models/{experiment.experiment_id}',
				verbosity=2
			)
			
			# Fit predictor
			fit_kwargs = {
				'time_limit': config.time_budget_minutes * 60,
				'presets': [config.quality_preset],
				'num_bag_folds': config.num_bag_folds,
				'num_stack_levels': config.num_stack_levels,
				'auto_stack': config.auto_stack,
				'hyperparameter_tune_kwargs': config.hyperparameter_tune_kwargs
			}
			
			if config.include_models:
				fit_kwargs['included_model_types'] = config.include_models
			if config.exclude_models:
				fit_kwargs['excluded_model_types'] = config.exclude_models
			
			predictor.fit(data, **fit_kwargs)
			
			# Evaluate predictor
			test_performance = predictor.evaluate(test_data)
			leaderboard = predictor.leaderboard(test_data).to_dict('records')
			
			# Get feature importance
			feature_importance = {}
			try:
				importance = predictor.feature_importance(test_data)
				feature_importance = importance.to_dict()
			except Exception as e:
				logger.warning(f"Failed to get feature importance: {e}")
			
			# Store predictor
			self.active_predictors[experiment.experiment_id] = predictor
			
			return {
				'best_model': leaderboard[0]['model'] if leaderboard else 'Unknown',
				'best_score': test_performance.get(config.eval_metric, 0.0),
				'models_trained': len(leaderboard),
				'leaderboard': leaderboard,
				'feature_importance': feature_importance,
				'model_path': predictor.path,
				'test_performance': test_performance
			}
			
		except Exception as e:
			logger.error(f"AutoGluon classification failed: {e}")
			return await self._run_sklearn_automl(config, data, "classification")
	
	async def _run_tabular_regression(self, config: AutoMLConfig, data: pd.DataFrame, experiment: AutoMLExperiment) -> Dict[str, Any]:
		"""Run tabular regression AutoML."""
		if not AUTOGLUON_AVAILABLE:
			return await self._run_sklearn_automl(config, data, "regression")
		
		try:
			# Split data if test data not provided
			if config.test_data_path:
				test_data = pd.read_csv(config.test_data_path)
			else:
				train_data = data.sample(frac=0.8, random_state=42)
				test_data = data.drop(train_data.index)
				data = train_data
			
			# Configure AutoGluon predictor
			predictor = TabularPredictor(
				label=config.target_column,
				problem_type='regression',
				eval_metric=config.eval_metric,
				path=f'./automl_models/{experiment.experiment_id}',
				verbosity=2
			)
			
			# Fit predictor
			fit_kwargs = {
				'time_limit': config.time_budget_minutes * 60,
				'presets': [config.quality_preset],
				'num_bag_folds': config.num_bag_folds,
				'num_stack_levels': config.num_stack_levels,
				'auto_stack': config.auto_stack
			}
			
			predictor.fit(data, **fit_kwargs)
			
			# Evaluate predictor
			test_performance = predictor.evaluate(test_data)
			leaderboard = predictor.leaderboard(test_data).to_dict('records')
			
			# Get feature importance
			feature_importance = {}
			try:
				importance = predictor.feature_importance(test_data)
				feature_importance = importance.to_dict()
			except Exception as e:
				logger.warning(f"Failed to get feature importance: {e}")
			
			# Store predictor
			self.active_predictors[experiment.experiment_id] = predictor
			
			return {
				'best_model': leaderboard[0]['model'] if leaderboard else 'Unknown',
				'best_score': test_performance.get('root_mean_squared_error', 0.0),
				'models_trained': len(leaderboard),
				'leaderboard': leaderboard,
				'feature_importance': feature_importance,
				'model_path': predictor.path,
				'test_performance': test_performance
			}
			
		except Exception as e:
			logger.error(f"AutoGluon regression failed: {e}")
			return await self._run_sklearn_automl(config, data, "regression")
	
	async def _run_time_series_forecasting(self, config: AutoMLConfig, data: pd.DataFrame, experiment: AutoMLExperiment) -> Dict[str, Any]:
		"""Run time series forecasting AutoML."""
		if not AUTOGLUON_AVAILABLE:
			return await self._run_basic_time_series(config, data)
		
		try:
			# Prepare time series data
			ts_data = data.copy()
			
			# AutoGluon TimeSeriesPredictor
			predictor = TimeSeriesPredictor(
				path=f'./automl_models/{experiment.experiment_id}',
				target=config.target_column,
				prediction_length=24,  # Default forecast horizon
				verbosity=2
			)
			
			# Fit predictor
			predictor.fit(
				ts_data,
				time_limit=config.time_budget_minutes * 60,
				presets=config.quality_preset
			)
			
			# Generate forecasts
			forecasts = predictor.predict(ts_data)
			
			# Evaluate predictor
			leaderboard = predictor.leaderboard().to_dict('records')
			
			# Store predictor
			self.active_predictors[experiment.experiment_id] = predictor
			
			return {
				'best_model': leaderboard[0]['model'] if leaderboard else 'Unknown',
				'best_score': leaderboard[0]['score_val'] if leaderboard else 0.0,
				'models_trained': len(leaderboard),
				'leaderboard': leaderboard,
				'feature_importance': {},
				'model_path': predictor.path,
				'forecasts': forecasts.to_dict('records') if hasattr(forecasts, 'to_dict') else str(forecasts)
			}
			
		except Exception as e:
			logger.error(f"Time series forecasting failed: {e}")
			return await self._run_basic_time_series(config, data)
	
	async def _run_multimodal_prediction(self, config: AutoMLConfig, data: pd.DataFrame, experiment: AutoMLExperiment) -> Dict[str, Any]:
		"""Run multimodal prediction AutoML."""
		if not AUTOGLUON_AVAILABLE:
			raise ValueError("AutoGluon is required for multimodal prediction")
		
		try:
			# Split data
			train_data = data.sample(frac=0.8, random_state=42)
			test_data = data.drop(train_data.index)
			
			# Configure multimodal predictor
			predictor = MultiModalPredictor(
				label=config.target_column,
				path=f'./automl_models/{experiment.experiment_id}',
				verbosity=2
			)
			
			# Fit predictor
			predictor.fit(
				train_data,
				time_limit=config.time_budget_minutes * 60,
				presets=config.quality_preset
			)
			
			# Evaluate predictor
			test_score = predictor.evaluate(test_data)
			
			# Store predictor
			self.active_predictors[experiment.experiment_id] = predictor
			
			return {
				'best_model': 'MultiModalPredictor',
				'best_score': test_score.get('accuracy', 0.0),
				'models_trained': 1,
				'leaderboard': [{'model': 'MultiModalPredictor', 'score': test_score}],
				'feature_importance': {},
				'model_path': predictor.path,
				'test_performance': test_score
			}
			
		except Exception as e:
			logger.error(f"Multimodal prediction failed: {e}")
			raise
	
	async def _run_hyperparameter_optimization(self, config: AutoMLConfig, data: pd.DataFrame, experiment: AutoMLExperiment) -> Dict[str, Any]:
		"""Run hyperparameter optimization."""
		if not OPTUNA_AVAILABLE:
			return await self._run_basic_hyperopt(config, data)
		
		try:
			from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
			from sklearn.model_selection import cross_val_score
			from sklearn.preprocessing import LabelEncoder
			
			# Prepare data
			X = data.drop(columns=[config.target_column])
			y = data[config.target_column]
			
			# Encode categorical variables
			categorical_columns = X.select_dtypes(include=['object']).columns
			le_dict = {}
			for col in categorical_columns:
				le = LabelEncoder()
				X[col] = le.fit_transform(X[col].astype(str))
				le_dict[col] = le
			
			# Determine problem type
			is_classification = len(y.unique()) < 50  # Heuristic
			
			def objective(trial):
				# Define hyperparameter space
				n_estimators = trial.suggest_int('n_estimators', 10, 300)
				max_depth = trial.suggest_int('max_depth', 3, 20)
				min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
				min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
				
				# Create model
				if is_classification:
					model = RandomForestClassifier(
						n_estimators=n_estimators,
						max_depth=max_depth,
						min_samples_split=min_samples_split,
						min_samples_leaf=min_samples_leaf,
						random_state=42
					)
					scoring = 'accuracy'
				else:
					model = RandomForestRegressor(
						n_estimators=n_estimators,
						max_depth=max_depth,
						min_samples_split=min_samples_split,
						min_samples_leaf=min_samples_leaf,
						random_state=42
					)
					scoring = 'neg_mean_squared_error'
				
				# Cross-validation
				scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
				return scores.mean()
			
			# Run optimization
			study = optuna.create_study(direction='maximize' if is_classification else 'maximize')
			study.optimize(objective, n_trials=min(100, config.time_budget_minutes * 2))
			
			# Get best parameters
			best_params = study.best_params
			best_score = study.best_value
			
			# Train final model with best parameters
			if is_classification:
				final_model = RandomForestClassifier(**best_params, random_state=42)
			else:
				final_model = RandomForestRegressor(**best_params, random_state=42)
			
			final_model.fit(X, y)
			
			# Feature importance
			feature_importance = dict(zip(X.columns, final_model.feature_importances_))
			
			return {
				'best_model': 'RandomForest_Optimized',
				'best_score': best_score,
				'models_trained': len(study.trials),
				'leaderboard': [{'model': 'RandomForest_Optimized', 'score': best_score}],
				'feature_importance': feature_importance,
				'model_path': f'./automl_models/{experiment.experiment_id}',
				'best_params': best_params,
				'optimization_history': [trial.value for trial in study.trials]
			}
			
		except Exception as e:
			logger.error(f"Hyperparameter optimization failed: {e}")
			return await self._run_basic_hyperopt(config, data)
	
	async def _run_sklearn_automl(self, config: AutoMLConfig, data: pd.DataFrame, problem_type: str) -> Dict[str, Any]:
		"""Run basic AutoML using scikit-learn."""
		try:
			from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
			from sklearn.linear_model import LogisticRegression, LinearRegression
			from sklearn.svm import SVC, SVR
			from sklearn.model_selection import cross_val_score
			from sklearn.preprocessing import StandardScaler, LabelEncoder
			from sklearn.metrics import accuracy_score, mean_squared_error
			
			# Prepare data
			X = data.drop(columns=[config.target_column])
			y = data[config.target_column]
			
			# Encode categorical variables
			categorical_columns = X.select_dtypes(include=['object']).columns
			for col in categorical_columns:
				le = LabelEncoder()
				X[col] = le.fit_transform(X[col].astype(str))
			
			# Scale features
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)
			
			# Define models
			if problem_type == "classification":
				models = {
					'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
					'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
					'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
					'SVM': SVC(random_state=42)
				}
				scoring = 'accuracy'
			else:
				models = {
					'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
					'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
					'LinearRegression': LinearRegression(),
					'SVR': SVR()
				}
				scoring = 'neg_mean_squared_error'
			
			# Train and evaluate models
			results = []
			for name, model in models.items():
				try:
					scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
					results.append({
						'model': name,
						'score': scores.mean(),
						'std': scores.std()
					})
				except Exception as e:
					logger.warning(f"Failed to train {name}: {e}")
			
			# Sort by score
			results.sort(key=lambda x: x['score'], reverse=(problem_type == "classification"))
			
			# Train best model on full data
			best_model_name = results[0]['model']
			best_model = models[best_model_name]
			best_model.fit(X_scaled, y)
			
			# Feature importance (if available)
			feature_importance = {}
			if hasattr(best_model, 'feature_importances_'):
				feature_importance = dict(zip(X.columns, best_model.feature_importances_))
			
			return {
				'best_model': best_model_name,
				'best_score': results[0]['score'],
				'models_trained': len(results),
				'leaderboard': results,
				'feature_importance': feature_importance,
				'model_path': f'./automl_models/{uuid7str()}'
			}
			
		except Exception as e:
			logger.error(f"sklearn AutoML failed: {e}")
			raise
	
	async def _run_basic_time_series(self, config: AutoMLConfig, data: pd.DataFrame) -> Dict[str, Any]:
		"""Run basic time series forecasting."""
		try:
			from sklearn.ensemble import RandomForestRegressor
			from sklearn.metrics import mean_squared_error, mean_absolute_error
			
			# Simple time series approach: use lagged features
			target_col = config.target_column
			
			# Create lagged features
			for lag in [1, 2, 3, 7, 14]:
				data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
			
			# Remove rows with NaN values
			data = data.dropna()
			
			# Split data
			train_size = int(0.8 * len(data))
			train_data = data[:train_size]
			test_data = data[train_size:]
			
			# Prepare features and targets
			feature_cols = [col for col in data.columns if col != target_col]
			X_train = train_data[feature_cols]
			y_train = train_data[target_col]
			X_test = test_data[feature_cols]
			y_test = test_data[target_col]
			
			# Train model
			model = RandomForestRegressor(n_estimators=100, random_state=42)
			model.fit(X_train, y_train)
			
			# Make predictions
			predictions = model.predict(X_test)
			
			# Calculate metrics
			mse = mean_squared_error(y_test, predictions)
			mae = mean_absolute_error(y_test, predictions)
			
			return {
				'best_model': 'RandomForest_TimeSeries',
				'best_score': -mse,  # Negative MSE for consistency
				'models_trained': 1,
				'leaderboard': [{'model': 'RandomForest_TimeSeries', 'score': -mse}],
				'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
				'model_path': f'./automl_models/{uuid7str()}',
				'mse': mse,
				'mae': mae
			}
			
		except Exception as e:
			logger.error(f"Basic time series forecasting failed: {e}")
			raise
	
	async def _run_basic_hyperopt(self, config: AutoMLConfig, data: pd.DataFrame) -> Dict[str, Any]:
		"""Run basic hyperparameter optimization without Optuna."""
		try:
			from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
			from sklearn.model_selection import GridSearchCV
			from sklearn.preprocessing import LabelEncoder
			
			# Prepare data
			X = data.drop(columns=[config.target_column])
			y = data[config.target_column]
			
			# Encode categorical variables
			categorical_columns = X.select_dtypes(include=['object']).columns
			for col in categorical_columns:
				le = LabelEncoder()
				X[col] = le.fit_transform(X[col].astype(str))
			
			# Determine problem type
			is_classification = len(y.unique()) < 50
			
			# Define parameter grid
			param_grid = {
				'n_estimators': [50, 100, 200],
				'max_depth': [5, 10, 15, None],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [1, 2, 4]
			}
			
			# Create model
			if is_classification:
				model = RandomForestClassifier(random_state=42)
				scoring = 'accuracy'
			else:
				model = RandomForestRegressor(random_state=42)
				scoring = 'neg_mean_squared_error'
			
			# Grid search
			grid_search = GridSearchCV(
				model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0
			)
			grid_search.fit(X, y)
			
			# Get results
			best_score = grid_search.best_score_
			best_params = grid_search.best_params_
			
			return {
				'best_model': 'RandomForest_GridSearch',
				'best_score': best_score,
				'models_trained': len(grid_search.cv_results_['params']),
				'leaderboard': [{'model': 'RandomForest_GridSearch', 'score': best_score}],
				'feature_importance': dict(zip(X.columns, grid_search.best_estimator_.feature_importances_)),
				'model_path': f'./automl_models/{uuid7str()}',
				'best_params': best_params
			}
			
		except Exception as e:
			logger.error(f"Basic hyperparameter optimization failed: {e}")
			raise
	
	async def _load_training_data(self, config: AutoMLConfig) -> pd.DataFrame:
		"""Load and validate training data."""
		try:
			# Load data
			if config.train_data_path.endswith('.csv'):
				data = pd.read_csv(config.train_data_path)
			elif config.train_data_path.endswith('.json'):
				data = pd.read_json(config.train_data_path)
			elif config.train_data_path.endswith('.parquet'):
				data = pd.read_parquet(config.train_data_path)
			else:
				raise ValueError(f"Unsupported file format: {config.train_data_path}")
			
			# Validate target column
			if config.target_column not in data.columns:
				raise ValueError(f"Target column '{config.target_column}' not found in data")
			
			# Basic data quality checks
			if len(data) == 0:
				raise ValueError("Training data is empty")
			
			if data[config.target_column].isnull().all():
				raise ValueError("Target column contains only null values")
			
			logger.info(f"Loaded training data: {len(data)} rows, {len(data.columns)} columns")
			return data
			
		except Exception as e:
			logger.error(f"Failed to load training data: {e}")
			raise
	
	async def predict_with_automl(self, experiment_id: str, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
		"""Make predictions using trained AutoML model."""
		try:
			if experiment_id not in self.active_predictors:
				raise ValueError(f"No active predictor found for experiment {experiment_id}")
			
			predictor = self.active_predictors[experiment_id]
			
			# Convert input data to DataFrame if needed
			if isinstance(input_data, dict):
				input_data = pd.DataFrame([input_data])
			
			# Make predictions
			predictions = predictor.predict(input_data)
			
			# Get prediction probabilities if available
			probabilities = None
			try:
				if hasattr(predictor, 'predict_proba'):
					probabilities = predictor.predict_proba(input_data)
				elif hasattr(predictor, 'predict_proba') and hasattr(predictions, 'predict_proba'):
					probabilities = predictions.predict_proba(input_data)
			except Exception as e:
				logger.debug(f"Could not get prediction probabilities: {e}")
			
			return {
				'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else str(predictions),
				'probabilities': probabilities.tolist() if probabilities is not None and hasattr(probabilities, 'tolist') else None,
				'experiment_id': experiment_id
			}
			
		except Exception as e:
			logger.error(f"Prediction failed for experiment {experiment_id}: {e}")
			raise
	
	def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
		"""Get AutoML experiment status."""
		if experiment_id not in self.experiments:
			return {'error': 'Experiment not found'}
		
		experiment = self.experiments[experiment_id]
		
		return {
			'experiment_id': experiment_id,
			'name': experiment.name,
			'status': experiment.status,
			'progress': self._calculate_progress(experiment),
			'best_score': experiment.best_score,
			'best_model': experiment.best_model_name,
			'models_trained': experiment.total_models_trained,
			'duration_minutes': experiment.experiment_duration / 60 if experiment.experiment_duration > 0 else 0,
			'created_at': experiment.created_at.isoformat(),
			'started_at': experiment.started_at.isoformat() if experiment.started_at else None,
			'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None
		}
	
	def _calculate_progress(self, experiment: AutoMLExperiment) -> float:
		"""Calculate experiment progress percentage."""
		if experiment.status == "pending":
			return 0.0
		elif experiment.status == "completed" or experiment.status == "failed":
			return 100.0
		elif experiment.status == "running":
			if experiment.started_at:
				elapsed = (datetime.now(timezone.utc) - experiment.started_at).total_seconds()
				expected_duration = experiment.time_budget_minutes * 60
				return min(95.0, (elapsed / expected_duration) * 100)  # Cap at 95% for running
		
		return 0.0
	
	def list_experiments(self) -> List[Dict[str, Any]]:
		"""List all AutoML experiments."""
		experiments = []
		for exp_id, experiment in self.experiments.items():
			experiments.append({
				'experiment_id': exp_id,
				'name': experiment.name,
				'status': experiment.status,
				'task_type': experiment.target_type,
				'best_score': experiment.best_score,
				'created_at': experiment.created_at.isoformat(),
				'duration_minutes': experiment.experiment_duration / 60 if experiment.experiment_duration > 0 else 0
			})
		
		return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
	
	def get_model_explanation(self, experiment_id: str) -> Dict[str, Any]:
		"""Get model explanation and interpretation."""
		if experiment_id not in self.experiments:
			return {'error': 'Experiment not found'}
		
		experiment = self.experiments[experiment_id]
		
		explanation = {
			'experiment_id': experiment_id,
			'model_name': experiment.best_model_name,
			'feature_importance': experiment.feature_importance,
			'model_performance': {
				'best_score': experiment.best_score,
				'models_evaluated': experiment.total_models_trained
			},
			'data_summary': {
				'dataset_size': experiment.dataset_size,
				'feature_count': experiment.feature_count,
				'target_type': experiment.target_type
			}
		}
		
		# Add leaderboard information
		if experiment.leaderboard:
			explanation['model_comparison'] = experiment.leaderboard[:5]  # Top 5 models
		
		return explanation
	
	async def cleanup_experiments(self, older_than_days: int = 30) -> Dict[str, int]:
		"""Cleanup old experiments and models."""
		cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
		
		cleaned_experiments = 0
		cleaned_models = 0
		
		# Cleanup experiments
		experiments_to_remove = []
		for exp_id, experiment in self.experiments.items():
			if experiment.created_at < cutoff_date:
				experiments_to_remove.append(exp_id)
				
				# Remove from active predictors
				if exp_id in self.active_predictors:
					del self.active_predictors[exp_id]
					cleaned_models += 1
		
		for exp_id in experiments_to_remove:
			del self.experiments[exp_id]
			cleaned_experiments += 1
		
		# Cleanup model files
		models_dir = Path('./automl_models')
		if models_dir.exists():
			for model_dir in models_dir.iterdir():
				if model_dir.is_dir():
					try:
						# Check if directory is old
						if datetime.fromtimestamp(model_dir.stat().st_mtime) < cutoff_date:
							import shutil
							shutil.rmtree(model_dir)
							cleaned_models += 1
					except Exception as e:
						logger.warning(f"Failed to cleanup model directory {model_dir}: {e}")
		
		return {
			'cleaned_experiments': cleaned_experiments,
			'cleaned_models': cleaned_models
		}
	
	async def shutdown(self) -> None:
		"""Shutdown the AutoML service."""
		try:
			# Clear active predictors
			self.active_predictors.clear()
			
			# Shutdown H2O if initialized
			if self.h2o_initialized and H2O_AVAILABLE:
				h2o.cluster().shutdown()
			
			logger.info("AutoML Service shutdown completed")
		except Exception as e:
			logger.error(f"Error during AutoML service shutdown: {e}")

# Global AutoML service instance
_automl_service_instance: Optional[AutoMLService] = None

def get_automl_service(config: Optional[Dict[str, Any]] = None, db_session: Optional[Session] = None) -> AutoMLService:
	"""Get the global AutoML service instance."""
	global _automl_service_instance
	if _automl_service_instance is None:
		if config is None:
			config = {}
		_automl_service_instance = AutoMLService(config, db_session)
	return _automl_service_instance