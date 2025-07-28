"""APG Cash Management - Advanced Machine Learning Models

World-class AI enhancements with sophisticated ML algorithms, ensemble methods,
and advanced risk analytics for revolutionary cash flow prediction accuracy.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Advanced ML Imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
	RandomForestRegressor, GradientBoostingRegressor, 
	ExtraTreesRegressor, VotingRegressor
)
from sklearn.linear_model import (
	ElasticNet, Ridge, Lasso, BayesianRidge,
	HuberRegressor, TheilSenRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, 
	mean_absolute_percentage_error, r2_score
)

# Time Series Specific
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from scipy.optimize import minimize

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .cache import CashCacheManager
from .events import CashEventManager

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

def _log_model_performance(model_name: str, metric: str, value: float) -> str:
	"""Log model performance metrics with APG formatting"""
	return f"ML_PERFORMANCE | model={model_name} | metric={metric} | value={value:.6f}"

def _log_ensemble_composition(ensemble_type: str, models: List[str]) -> str:
	"""Log ensemble model composition"""
	return f"ENSEMBLE_COMPOSITION | type={ensemble_type} | models={','.join(models)}"

# ============================================================================
# Advanced ML Model Enums and Data Classes
# ============================================================================

class ModelType(str, Enum):
	"""Advanced ML model types for cash flow prediction"""
	
	# Tree-Based Models
	XGBOOST = "xgboost"
	LIGHTGBM = "lightgbm" 
	RANDOM_FOREST = "random_forest"
	EXTRA_TREES = "extra_trees"
	GRADIENT_BOOSTING = "gradient_boosting"
	
	# Linear Models
	ELASTIC_NET = "elastic_net"
	RIDGE = "ridge"
	LASSO = "lasso"
	BAYESIAN_RIDGE = "bayesian_ridge"
	HUBER = "huber"
	THEIL_SEN = "theil_sen"
	
	# Neural Networks
	MLP = "mlp"
	LSTM = "lstm"
	TRANSFORMER = "transformer"
	
	# Support Vector Machines
	SVR_RBF = "svr_rbf"
	SVR_LINEAR = "svr_linear"
	
	# Gaussian Processes
	GAUSSIAN_PROCESS = "gaussian_process"
	
	# Time Series Specific
	ARIMA = "arima"
	EXPONENTIAL_SMOOTHING = "exponential_smoothing"
	
	# Ensemble Methods
	VOTING_REGRESSOR = "voting_regressor"
	STACKING_REGRESSOR = "stacking_regressor"
	DYNAMIC_ENSEMBLE = "dynamic_ensemble"

class EnsembleStrategy(str, Enum):
	"""Ensemble combination strategies"""
	VOTING = "voting"
	STACKING = "stacking"
	DYNAMIC_WEIGHTING = "dynamic_weighting"
	BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"
	ADAPTIVE_BOOSTING = "adaptive_boosting"

@dataclass
class ModelPerformance:
	"""Model performance metrics tracking"""
	model_name: str
	model_type: ModelType
	train_metrics: Dict[str, float]
	validation_metrics: Dict[str, float]
	test_metrics: Dict[str, float]
	training_time: float
	prediction_time: float
	feature_importance: Dict[str, float]
	hyperparameters: Dict[str, Any]
	cv_scores: List[float]
	stability_score: float
	complexity_score: float

@dataclass
class ForecastResult:
	"""Enhanced forecast result with uncertainty quantification"""
	predictions: np.ndarray
	confidence_intervals: Tuple[np.ndarray, np.ndarray]
	prediction_intervals: Tuple[np.ndarray, np.ndarray]
	model_uncertainty: np.ndarray
	aleatoric_uncertainty: np.ndarray
	epistemic_uncertainty: np.ndarray
	feature_contributions: Dict[str, np.ndarray]
	model_weights: Dict[str, float]
	risk_metrics: Dict[str, float]

# ============================================================================
# Advanced Feature Engineering
# ============================================================================

class AdvancedFeatureEngineer:
	"""World-class feature engineering for cash flow prediction"""
	
	def __init__(self, cache_manager: CashCacheManager):
		self.cache = cache_manager
		self.scalers: Dict[str, Any] = {}
		self.feature_selectors: Dict[str, Any] = {}
		self.dimensionality_reducers: Dict[str, Any] = {}
		
	async def engineer_features(
		self, 
		data: pd.DataFrame,
		target_column: str = "amount",
		mode: str = "train"
	) -> Tuple[pd.DataFrame, pd.Series]:
		"""Generate sophisticated features for ML models"""
		
		logger.info("Starting advanced feature engineering...")
		
		# Time-based features
		data = self._create_temporal_features(data)
		
		# Statistical features
		data = self._create_statistical_features(data, target_column)
		
		# Lag and rolling features
		data = self._create_lag_rolling_features(data, target_column)
		
		# Fourier and seasonal features
		data = self._create_frequency_features(data)
		
		# Business calendar features
		data = self._create_calendar_features(data)
		
		# Technical indicators
		data = self._create_technical_indicators(data, target_column)
		
		# Interaction features
		data = self._create_interaction_features(data)
		
		# Polynomial features (selective)
		data = self._create_polynomial_features(data)
		
		# Categorical encoding
		data = self._encode_categorical_features(data)
		
		# Handle missing values
		data = self._handle_missing_values(data)
		
		# Feature scaling
		if mode == "train":
			data = self._fit_transform_scaling(data)
		else:
			data = self._transform_scaling(data)
		
		# Feature selection
		if mode == "train":
			data, target = self._fit_feature_selection(data, target_column)
		else:
			data, target = self._transform_feature_selection(data, target_column)
		
		# Dimensionality reduction
		if mode == "train":
			data = self._fit_dimensionality_reduction(data)
		else:
			data = self._transform_dimensionality_reduction(data)
		
		logger.info(f"Feature engineering completed: {data.shape[1]} features generated")
		
		return data, target
	
	def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Create sophisticated time-based features"""
		
		if 'date' not in data.columns:
			return data
		
		data['date'] = pd.to_datetime(data['date'])
		
		# Basic temporal features
		data['year'] = data['date'].dt.year
		data['month'] = data['date'].dt.month
		data['day'] = data['date'].dt.day
		data['day_of_week'] = data['date'].dt.dayofweek
		data['day_of_year'] = data['date'].dt.dayofyear
		data['week_of_year'] = data['date'].dt.isocalendar().week
		data['quarter'] = data['date'].dt.quarter
		
		# Cyclical encoding
		data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
		data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
		data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
		data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
		data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
		data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
		
		# Business calendar features
		data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
		data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
		data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
		data['is_quarter_start'] = data['date'].dt.is_quarter_start.astype(int)
		data['is_quarter_end'] = data['date'].dt.is_quarter_end.astype(int)
		data['is_year_start'] = data['date'].dt.is_year_start.astype(int)
		data['is_year_end'] = data['date'].dt.is_year_end.astype(int)
		
		# Days from reference points
		reference_date = data['date'].min()
		data['days_from_start'] = (data['date'] - reference_date).dt.days
		
		return data
	
	def _create_statistical_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
		"""Create statistical features based on target variable"""
		
		if target_col not in data.columns:
			return data
		
		# Rolling statistics (multiple windows)
		windows = [7, 14, 30, 60, 90]
		
		for window in windows:
			# Central tendency
			data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window).mean()
			data[f'{target_col}_rolling_median_{window}'] = data[target_col].rolling(window).median()
			
			# Variability
			data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window).std()
			data[f'{target_col}_rolling_var_{window}'] = data[target_col].rolling(window).var()
			data[f'{target_col}_rolling_cv_{window}'] = (
				data[f'{target_col}_rolling_std_{window}'] / 
				data[f'{target_col}_rolling_mean_{window}']
			)
			
			# Distribution shape
			data[f'{target_col}_rolling_skew_{window}'] = data[target_col].rolling(window).skew()
			data[f'{target_col}_rolling_kurt_{window}'] = data[target_col].rolling(window).kurt()
			
			# Extremes
			data[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window).min()
			data[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window).max()
			data[f'{target_col}_rolling_range_{window}'] = (
				data[f'{target_col}_rolling_max_{window}'] - 
				data[f'{target_col}_rolling_min_{window}']
			)
			
			# Percentiles
			data[f'{target_col}_rolling_q25_{window}'] = data[target_col].rolling(window).quantile(0.25)
			data[f'{target_col}_rolling_q75_{window}'] = data[target_col].rolling(window).quantile(0.75)
			data[f'{target_col}_rolling_iqr_{window}'] = (
				data[f'{target_col}_rolling_q75_{window}'] - 
				data[f'{target_col}_rolling_q25_{window}']
			)
		
		return data
	
	def _create_lag_rolling_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
		"""Create lag and rolling window features"""
		
		if target_col not in data.columns:
			return data
		
		# Lag features
		lags = [1, 2, 3, 5, 7, 14, 21, 30, 60, 90]
		
		for lag in lags:
			data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
		
		# Rolling correlation with lags
		for window in [30, 60]:
			for lag in [1, 7, 30]:
				data[f'{target_col}_rolling_corr_lag{lag}_{window}'] = (
					data[target_col].rolling(window).corr(data[target_col].shift(lag))
				)
		
		# Exponentially weighted features
		alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
		
		for alpha in alphas:
			data[f'{target_col}_ewm_mean_{alpha}'] = data[target_col].ewm(alpha=alpha).mean()
			data[f'{target_col}_ewm_std_{alpha}'] = data[target_col].ewm(alpha=alpha).std()
		
		# Differences and rates of change
		data[f'{target_col}_diff_1'] = data[target_col].diff(1)
		data[f'{target_col}_diff_7'] = data[target_col].diff(7)
		data[f'{target_col}_pct_change_1'] = data[target_col].pct_change(1)
		data[f'{target_col}_pct_change_7'] = data[target_col].pct_change(7)
		
		return data
	
	def _create_frequency_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Create Fourier and frequency-based features"""
		
		if 'date' not in data.columns:
			return data
		
		data['date'] = pd.to_datetime(data['date'])
		
		# Time index for Fourier features
		time_idx = np.arange(len(data))
		
		# Multiple frequency components
		frequencies = [1, 2, 3, 4, 6, 12]  # Annual patterns
		
		for freq in frequencies:
			data[f'fourier_sin_{freq}'] = np.sin(2 * np.pi * freq * time_idx / 365.25)
			data[f'fourier_cos_{freq}'] = np.cos(2 * np.pi * freq * time_idx / 365.25)
		
		# Weekly patterns
		for freq in [1, 2, 4]:
			data[f'weekly_sin_{freq}'] = np.sin(2 * np.pi * freq * time_idx / 7)
			data[f'weekly_cos_{freq}'] = np.cos(2 * np.pi * freq * time_idx / 7)
		
		# Monthly patterns
		for freq in [1, 2, 3]:
			data[f'monthly_sin_{freq}'] = np.sin(2 * np.pi * freq * time_idx / 30.44)
			data[f'monthly_cos_{freq}'] = np.cos(2 * np.pi * freq * time_idx / 30.44)
		
		return data
	
	def _create_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Create business calendar and holiday features"""
		
		if 'date' not in data.columns:
			return data
		
		data['date'] = pd.to_datetime(data['date'])
		
		# Business day features
		data['is_business_day'] = data['date'].dt.dayofweek < 5
		data['business_days_in_month'] = data['date'].dt.day.where(data['is_business_day']).groupby(
			[data['date'].dt.year, data['date'].dt.month]
		).transform('count')
		
		# Distance to month/quarter boundaries
		data['days_to_month_end'] = (data['date'] + pd.offsets.MonthEnd(0) - data['date']).dt.days
		data['days_from_month_start'] = (data['date'] - pd.offsets.MonthBegin(0)).dt.days
		data['days_to_quarter_end'] = (data['date'] + pd.offsets.QuarterEnd(0) - data['date']).dt.days
		
		# Payroll patterns (bi-weekly, monthly)
		data['is_biweekly_payroll'] = ((data['date'].dt.dayofweek == 4) & 
									   ((data['date'].dt.day <= 15) | (data['date'].dt.day > 15))).astype(int)
		data['is_monthly_payroll'] = ((data['date'].dt.dayofweek == 4) & 
									  (data['date'].dt.day <= 5)).astype(int)
		
		return data
	
	def _create_technical_indicators(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
		"""Create technical analysis indicators"""
		
		if target_col not in data.columns:
			return data
		
		# Simple moving averages
		sma_periods = [5, 10, 20, 50]
		for period in sma_periods:
			data[f'{target_col}_sma_{period}'] = data[target_col].rolling(period).mean()
			data[f'{target_col}_sma_ratio_{period}'] = data[target_col] / data[f'{target_col}_sma_{period}']
		
		# Bollinger Bands
		for period in [20, 50]:
			sma = data[target_col].rolling(period).mean()
			std = data[target_col].rolling(period).std()
			data[f'{target_col}_bb_upper_{period}'] = sma + (2 * std)
			data[f'{target_col}_bb_lower_{period}'] = sma - (2 * std)
			data[f'{target_col}_bb_width_{period}'] = data[f'{target_col}_bb_upper_{period}'] - data[f'{target_col}_bb_lower_{period}']
			data[f'{target_col}_bb_position_{period}'] = (
				(data[target_col] - data[f'{target_col}_bb_lower_{period}']) / 
				data[f'{target_col}_bb_width_{period}']
			)
		
		# RSI (Relative Strength Index)
		for period in [14, 30]:
			delta = data[target_col].diff()
			gain = (delta.where(delta > 0, 0)).rolling(period).mean()
			loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
			rs = gain / loss
			data[f'{target_col}_rsi_{period}'] = 100 - (100 / (1 + rs))
		
		# MACD (Moving Average Convergence Divergence)
		ema_12 = data[target_col].ewm(span=12).mean()
		ema_26 = data[target_col].ewm(span=26).mean()
		data[f'{target_col}_macd'] = ema_12 - ema_26
		data[f'{target_col}_macd_signal'] = data[f'{target_col}_macd'].ewm(span=9).mean()
		data[f'{target_col}_macd_histogram'] = data[f'{target_col}_macd'] - data[f'{target_col}_macd_signal']
		
		return data
	
	def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Create interaction features between important variables"""
		
		# Get numeric columns
		numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
		
		# Remove target and id columns
		excluded_cols = ['amount', 'id', 'tenant_id']
		feature_cols = [col for col in numeric_cols if col not in excluded_cols]
		
		# Create interactions for top features (to avoid feature explosion)
		if len(feature_cols) >= 2:
			# Take top 10 features by variance
			top_features = data[feature_cols].var().nlargest(10).index.tolist()
			
			# Create pairwise interactions
			for i, col1 in enumerate(top_features):
				for col2 in top_features[i+1:]:
					if col1 != col2:
						data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
		
		return data
	
	def _create_polynomial_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Create selective polynomial features"""
		
		# Get numeric columns with high correlation to target
		numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
		
		# Create squared features for selected columns
		polynomial_candidates = [col for col in numeric_cols if 'lag_' in col or 'rolling_' in col][:5]
		
		for col in polynomial_candidates:
			if col in data.columns:
				data[f'{col}_squared'] = data[col] ** 2
				data[f'{col}_sqrt'] = np.sqrt(np.abs(data[col]))
		
		return data
	
	def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Encode categorical features using advanced techniques"""
		
		categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
		
		for col in categorical_cols:
			if col != 'date':
				# Target encoding for high cardinality
				if data[col].nunique() > 10:
					# Simple frequency encoding
					freq_encoding = data[col].value_counts(normalize=True)
					data[f'{col}_frequency'] = data[col].map(freq_encoding)
				else:
					# One-hot encoding for low cardinality
					dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
					data = pd.concat([data, dummies], axis=1)
				
				# Drop original categorical column
				data = data.drop(columns=[col])
		
		return data
	
	def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Handle missing values with sophisticated imputation"""
		
		# Forward fill for time series data
		time_series_cols = [col for col in data.columns if any(x in col for x in ['lag_', 'rolling_', 'ewm_'])]
		for col in time_series_cols:
			data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
		
		# Median fill for other numeric columns
		numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
		for col in numeric_cols:
			if data[col].isnull().sum() > 0:
				data[col] = data[col].fillna(data[col].median())
		
		return data
	
	def _fit_transform_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Fit and transform feature scaling"""
		
		numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
		
		# Use RobustScaler for outlier resistance
		self.scalers['robust'] = RobustScaler()
		data[numeric_cols] = self.scalers['robust'].fit_transform(data[numeric_cols])
		
		# Store column names for later use
		self.scalers['columns'] = numeric_cols
		
		return data
	
	def _transform_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Transform features using fitted scalers"""
		
		if 'robust' in self.scalers and 'columns' in self.scalers:
			columns_to_scale = [col for col in self.scalers['columns'] if col in data.columns]
			data[columns_to_scale] = self.scalers['robust'].transform(data[columns_to_scale])
		
		return data
	
	def _fit_feature_selection(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
		"""Fit feature selection and return selected features"""
		
		if target_col not in data.columns:
			return data, pd.Series()
		
		target = data[target_col]
		features = data.drop(columns=[target_col])
		
		# Remove features with low variance
		from sklearn.feature_selection import VarianceThreshold
		self.feature_selectors['variance'] = VarianceThreshold(threshold=0.01)
		features_variance = self.feature_selectors['variance'].fit_transform(features)
		selected_features = features.columns[self.feature_selectors['variance'].get_support()]
		features = features[selected_features]
		
		# Select top K features based on mutual information
		k = min(100, len(features.columns))  # Cap at 100 features
		self.feature_selectors['mutual_info'] = SelectKBest(mutual_info_regression, k=k)
		features_selected = self.feature_selectors['mutual_info'].fit_transform(features, target)
		selected_features = features.columns[self.feature_selectors['mutual_info'].get_support()]
		
		# Store selected feature names
		self.feature_selectors['selected_features'] = selected_features
		
		result_data = pd.DataFrame(features_selected, columns=selected_features, index=features.index)
		
		return result_data, target
	
	def _transform_feature_selection(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
		"""Transform features using fitted feature selectors"""
		
		if target_col not in data.columns:
			return data, pd.Series()
		
		target = data[target_col]
		features = data.drop(columns=[target_col])
		
		if 'selected_features' in self.feature_selectors:
			# Use only previously selected features
			available_features = [col for col in self.feature_selectors['selected_features'] if col in features.columns]
			features = features[available_features]
		
		return features, target
	
	def _fit_dimensionality_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Fit dimensionality reduction if needed"""
		
		if data.shape[1] > 50:  # Only apply if too many features
			# PCA for linear relationships
			n_components = min(30, data.shape[1] - 1)
			self.dimensionality_reducers['pca'] = PCA(n_components=n_components)
			data_pca = self.dimensionality_reducers['pca'].fit_transform(data)
			
			# Create new column names
			pca_columns = [f'pca_{i}' for i in range(n_components)]
			return pd.DataFrame(data_pca, columns=pca_columns, index=data.index)
		
		return data
	
	def _transform_dimensionality_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Transform data using fitted dimensionality reduction"""
		
		if 'pca' in self.dimensionality_reducers:
			data_pca = self.dimensionality_reducers['pca'].transform(data)
			n_components = data_pca.shape[1]
			pca_columns = [f'pca_{i}' for i in range(n_components)]
			return pd.DataFrame(data_pca, columns=pca_columns, index=data.index)
		
		return data

# ============================================================================
# Deep Learning Models
# ============================================================================

class LSTMNet(nn.Module):
	"""LSTM network for time series forecasting"""
	
	def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
		super(LSTMNet, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.lstm = nn.LSTM(
			input_size, hidden_size, num_layers, 
			batch_first=True, dropout=dropout
		)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_size, 1)
		
	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		
		out, _ = self.lstm(x, (h0, c0))
		out = self.dropout(out[:, -1, :])
		out = self.fc(out)
		return out

class TransformerNet(nn.Module):
	"""Transformer network for sequence modeling"""
	
	def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
		super(TransformerNet, self).__init__()
		self.d_model = d_model
		self.embedding = nn.Linear(input_size, d_model)
		self.pos_encoding = self._create_positional_encoding(1000, d_model)
		
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(d_model, 1)
		
	def _create_positional_encoding(self, max_len: int, d_model: int):
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe.unsqueeze(0)
		
	def forward(self, x):
		seq_len = x.size(1)
		x = self.embedding(x) * np.sqrt(self.d_model)
		x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
		x = self.transformer(x)
		x = self.fc(x[:, -1, :])
		return x

# ============================================================================
# Advanced ML Model Manager
# ============================================================================

class AdvancedMLModelManager:
	"""World-class ML model management with sophisticated algorithms"""
	
	def __init__(
		self, 
		tenant_id: str, 
		cache_manager: CashCacheManager, 
		event_manager: CashEventManager
	):
		self.tenant_id = tenant_id
		self.cache = cache_manager
		self.events = event_manager
		self.models: Dict[str, Any] = {}
		self.model_performances: Dict[str, ModelPerformance] = {}
		self.feature_engineer = AdvancedFeatureEngineer(cache_manager)
		self.ensemble_weights: Dict[str, float] = {}
		
	async def train_all_models(
		self, 
		training_data: pd.DataFrame,
		target_column: str = "amount",
		validation_split: float = 0.2
	) -> Dict[str, ModelPerformance]:
		"""Train all available ML models with cross-validation"""
		
		logger.info(f"Training all ML models for tenant {self.tenant_id}...")
		
		# Feature engineering
		features, target = await self.feature_engineer.engineer_features(
			training_data, target_column, mode="train"
		)
		
		# Split data
		split_idx = int(len(features) * (1 - validation_split))
		X_train, X_val = features[:split_idx], features[split_idx:]
		y_train, y_val = target[:split_idx], target[split_idx:]
		
		# Train individual models
		model_tasks = []
		
		# Tree-based models
		model_tasks.extend([
			self._train_xgboost(X_train, y_train, X_val, y_val),
			self._train_lightgbm(X_train, y_train, X_val, y_val),
			self._train_random_forest(X_train, y_train, X_val, y_val),
			self._train_extra_trees(X_train, y_train, X_val, y_val),
			self._train_gradient_boosting(X_train, y_train, X_val, y_val)
		])
		
		# Linear models
		model_tasks.extend([
			self._train_elastic_net(X_train, y_train, X_val, y_val),
			self._train_ridge(X_train, y_train, X_val, y_val),
			self._train_bayesian_ridge(X_train, y_train, X_val, y_val),
			self._train_huber(X_train, y_train, X_val, y_val)
		])
		
		# Neural networks
		model_tasks.extend([
			self._train_mlp(X_train, y_train, X_val, y_val),
			self._train_lstm(X_train, y_train, X_val, y_val),
			self._train_transformer(X_train, y_train, X_val, y_val)
		])
		
		# Other models
		model_tasks.extend([
			self._train_svr(X_train, y_train, X_val, y_val),
			self._train_gaussian_process(X_train, y_train, X_val, y_val)
		])
		
		# Execute training in parallel
		performances = await asyncio.gather(*model_tasks, return_exceptions=True)
		
		# Filter successful training results
		valid_performances = [p for p in performances if isinstance(p, ModelPerformance)]
		
		# Train ensemble models
		ensemble_performances = await self._train_ensemble_models(
			X_train, y_train, X_val, y_val, valid_performances
		)
		
		# Combine all performances
		all_performances = valid_performances + ensemble_performances
		
		# Update model registry
		for perf in all_performances:
			self.model_performances[perf.model_name] = perf
		
		# Cache model performances
		await self._cache_model_performances(all_performances)
		
		# Log results
		best_model = max(all_performances, key=lambda x: x.validation_metrics.get('r2', 0))
		logger.info(f"Best model: {best_model.model_name} (R² = {best_model.validation_metrics['r2']:.4f})")
		
		return {perf.model_name: perf for perf in all_performances}
	
	async def _train_xgboost(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train XGBoost model with hyperparameter optimization"""
		
		start_time = datetime.now()
		
		# Hyperparameter grid
		param_grid = {
			'max_depth': [3, 6, 10],
			'learning_rate': [0.01, 0.1, 0.2],
			'n_estimators': [100, 500, 1000],
			'subsample': [0.8, 0.9, 1.0],
			'colsample_bytree': [0.8, 0.9, 1.0]
		}
		
		# Time series cross-validation
		tscv = TimeSeriesSplit(n_splits=5)
		
		# Grid search
		xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
		grid_search = GridSearchCV(
			xgb_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		# Train final model
		best_model.fit(X_train, y_train)
		
		# Predictions
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		# Calculate metrics
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		# Feature importance
		feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
		
		# CV scores
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		# Store model
		self.models['xgboost'] = best_model
		
		return ModelPerformance(
			model_name='xgboost',
			model_type=ModelType.XGBOOST,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_lightgbm(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train LightGBM model with hyperparameter optimization"""
		
		start_time = datetime.now()
		
		# Hyperparameter grid
		param_grid = {
			'max_depth': [3, 6, 10, -1],
			'learning_rate': [0.01, 0.1, 0.2],
			'n_estimators': [100, 500, 1000],
			'subsample': [0.8, 0.9, 1.0],
			'colsample_bytree': [0.8, 0.9, 1.0],
			'num_leaves': [31, 50, 100]
		}
		
		# Time series cross-validation
		tscv = TimeSeriesSplit(n_splits=5)
		
		# Grid search
		lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
		grid_search = GridSearchCV(
			lgb_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		# Predictions
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		# Calculate metrics
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		# Feature importance
		feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
		
		# CV scores
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		# Store model
		self.models['lightgbm'] = best_model
		
		return ModelPerformance(
			model_name='lightgbm',
			model_type=ModelType.LIGHTGBM,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_random_forest(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Random Forest model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'n_estimators': [100, 300, 500],
			'max_depth': [10, 20, 30, None],
			'min_samples_split': [2, 5, 10],
			'min_samples_leaf': [1, 2, 4],
			'max_features': ['auto', 'sqrt', 'log2']
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
		grid_search = GridSearchCV(
			rf_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['random_forest'] = best_model
		
		return ModelPerformance(
			model_name='random_forest',
			model_type=ModelType.RANDOM_FOREST,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_extra_trees(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Extra Trees model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'n_estimators': [100, 300, 500],
			'max_depth': [10, 20, 30, None],
			'min_samples_split': [2, 5, 10],
			'min_samples_leaf': [1, 2, 4]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		et_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
		grid_search = GridSearchCV(
			et_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['extra_trees'] = best_model
		
		return ModelPerformance(
			model_name='extra_trees',
			model_type=ModelType.EXTRA_TREES,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_gradient_boosting(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Gradient Boosting model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'n_estimators': [100, 200, 300],
			'learning_rate': [0.01, 0.1, 0.2],
			'max_depth': [3, 5, 7],
			'subsample': [0.8, 0.9, 1.0]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		gb_model = GradientBoostingRegressor(random_state=42)
		grid_search = GridSearchCV(
			gb_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['gradient_boosting'] = best_model
		
		return ModelPerformance(
			model_name='gradient_boosting',
			model_type=ModelType.GRADIENT_BOOSTING,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_elastic_net(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Elastic Net model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
			'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		en_model = ElasticNet(random_state=42, max_iter=10000)
		grid_search = GridSearchCV(
			en_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['elastic_net'] = best_model
		
		return ModelPerformance(
			model_name='elastic_net',
			model_type=ModelType.ELASTIC_NET,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_ridge(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Ridge regression model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		ridge_model = Ridge(random_state=42, max_iter=10000)
		grid_search = GridSearchCV(
			ridge_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['ridge'] = best_model
		
		return ModelPerformance(
			model_name='ridge',
			model_type=ModelType.RIDGE,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_bayesian_ridge(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Bayesian Ridge regression model"""
		
		start_time = datetime.now()
		
		param_grid = {
			'alpha_1': [1e-6, 1e-5, 1e-4],
			'alpha_2': [1e-6, 1e-5, 1e-4],
			'lambda_1': [1e-6, 1e-5, 1e-4],
			'lambda_2': [1e-6, 1e-5, 1e-4]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		br_model = BayesianRidge()
		grid_search = GridSearchCV(
			br_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['bayesian_ridge'] = best_model
		
		return ModelPerformance(
			model_name='bayesian_ridge',
			model_type=ModelType.BAYESIAN_RIDGE,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_huber(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Huber regressor for robust regression"""
		
		start_time = datetime.now()
		
		param_grid = {
			'epsilon': [1.1, 1.35, 1.5, 2.0],
			'alpha': [0.0001, 0.001, 0.01, 0.1]
		}
		
		tscv = TimeSeriesSplit(n_splits=5)
		
		huber_model = HuberRegressor(max_iter=10000)
		grid_search = GridSearchCV(
			huber_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['huber'] = best_model
		
		return ModelPerformance(
			model_name='huber',
			model_type=ModelType.HUBER,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_mlp(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Multi-layer Perceptron neural network"""
		
		start_time = datetime.now()
		
		param_grid = {
			'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
			'activation': ['relu', 'tanh'],
			'alpha': [0.0001, 0.001, 0.01],
			'learning_rate': ['constant', 'adaptive']
		}
		
		tscv = TimeSeriesSplit(n_splits=3)  # Reduced for neural networks
		
		mlp_model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
		grid_search = GridSearchCV(
			mlp_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=1  # Neural nets don't parallelize well
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		# Feature importance approximation for MLP
		feature_importance = {}
		if hasattr(best_model, 'coefs_'):
			input_weights = np.abs(best_model.coefs_[0]).sum(axis=1)
			feature_importance = dict(zip(X_train.columns, input_weights))
		
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['mlp'] = best_model
		
		return ModelPerformance(
			model_name='mlp',
			model_type=ModelType.MLP,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance=feature_importance,
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_lstm(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train LSTM neural network for time series"""
		
		start_time = datetime.now()
		
		# Convert to sequences for LSTM
		sequence_length = min(30, len(X_train) // 10)
		X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, sequence_length)
		X_val_seq, y_val_seq = self._create_sequences(X_val.values, y_val.values, sequence_length)
		
		# Convert to PyTorch tensors
		X_train_tensor = torch.FloatTensor(X_train_seq)
		y_train_tensor = torch.FloatTensor(y_train_seq)
		X_val_tensor = torch.FloatTensor(X_val_seq)
		y_val_tensor = torch.FloatTensor(y_val_seq)
		
		# Create data loaders
		train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
		train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
		
		# Initialize model
		input_size = X_train.shape[1]
		model = LSTMNet(input_size, hidden_size=64, num_layers=2, dropout=0.2)
		
		# Training setup
		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
		
		# Training loop
		model.train()
		for epoch in range(100):
			epoch_loss = 0
			for X_batch, y_batch in train_loader:
				optimizer.zero_grad()
				y_pred = model(X_batch)
				loss = criterion(y_pred.squeeze(), y_batch)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			
			# Validation
			model.eval()
			with torch.no_grad():
				val_pred = model(X_val_tensor)
				val_loss = criterion(val_pred.squeeze(), y_val_tensor)
			
			scheduler.step(val_loss)
			
			if epoch % 20 == 0:
				logger.info(f"LSTM Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
			
			model.train()
		
		# Final predictions
		model.eval()
		with torch.no_grad():
			train_pred = model(X_train_tensor).squeeze().numpy()
			val_pred = model(X_val_tensor).squeeze().numpy()
		
		train_metrics = self._calculate_metrics(y_train_seq, train_pred)
		val_metrics = self._calculate_metrics(y_val_seq, val_pred)
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['lstm'] = model
		
		return ModelPerformance(
			model_name='lstm',
			model_type=ModelType.LSTM,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters={'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
			cv_scores=[val_metrics['rmse']],
			stability_score=0.0,
			complexity_score=3
		)
	
	async def _train_transformer(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Transformer neural network"""
		
		start_time = datetime.now()
		
		# Convert to sequences
		sequence_length = min(20, len(X_train) // 10)
		X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, sequence_length)
		X_val_seq, y_val_seq = self._create_sequences(X_val.values, y_val.values, sequence_length)
		
		# Convert to PyTorch tensors
		X_train_tensor = torch.FloatTensor(X_train_seq)
		y_train_tensor = torch.FloatTensor(y_train_seq)
		X_val_tensor = torch.FloatTensor(X_val_seq)
		y_val_tensor = torch.FloatTensor(y_val_seq)
		
		# Create data loaders
		train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
		train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
		
		# Initialize model
		input_size = X_train.shape[1]
		model = TransformerNet(input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1)
		
		# Training setup
		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=0.0001)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
		
		# Training loop
		model.train()
		for epoch in range(50):  # Fewer epochs for Transformer
			epoch_loss = 0
			for X_batch, y_batch in train_loader:
				optimizer.zero_grad()
				y_pred = model(X_batch)
				loss = criterion(y_pred.squeeze(), y_batch)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			
			# Validation
			model.eval()
			with torch.no_grad():
				val_pred = model(X_val_tensor)
				val_loss = criterion(val_pred.squeeze(), y_val_tensor)
			
			scheduler.step(val_loss)
			
			if epoch % 10 == 0:
				logger.info(f"Transformer Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}")
			
			model.train()
		
		# Final predictions
		model.eval()
		with torch.no_grad():
			train_pred = model(X_train_tensor).squeeze().numpy()
			val_pred = model(X_val_tensor).squeeze().numpy()
		
		train_metrics = self._calculate_metrics(y_train_seq, train_pred)
		val_metrics = self._calculate_metrics(y_val_seq, val_pred)
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['transformer'] = model
		
		return ModelPerformance(
			model_name='transformer',
			model_type=ModelType.TRANSFORMER,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters={'d_model': 64, 'nhead': 4, 'num_layers': 2},
			cv_scores=[val_metrics['rmse']],
			stability_score=0.0,
			complexity_score=4
		)
	
	async def _train_svr(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Support Vector Regressor"""
		
		start_time = datetime.now()
		
		param_grid = {
			'kernel': ['rbf', 'linear'],
			'C': [0.1, 1, 10],
			'gamma': ['scale', 'auto', 0.001, 0.01],
			'epsilon': [0.01, 0.1, 1.0]
		}
		
		tscv = TimeSeriesSplit(n_splits=3)  # Reduced for SVR
		
		svr_model = SVR()
		grid_search = GridSearchCV(
			svr_model, param_grid, cv=tscv, 
			scoring='neg_mean_squared_error', n_jobs=-1
		)
		
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
		
		train_pred = best_model.predict(X_train)
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		cv_scores = -grid_search.cv_results_['mean_test_score']
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['svr'] = best_model
		
		return ModelPerformance(
			model_name='svr',
			model_type=ModelType.SVR_RBF,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters=grid_search.best_params_,
			cv_scores=cv_scores.tolist(),
			stability_score=np.std(cv_scores),
			complexity_score=len(grid_search.best_params_)
		)
	
	async def _train_gaussian_process(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series
	) -> ModelPerformance:
		"""Train Gaussian Process Regressor"""
		
		start_time = datetime.now()
		
		# Define kernels
		kernels = [
			RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
			Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
			RBF(length_scale=1.0) * WhiteKernel(noise_level=0.1)
		]
		
		best_score = float('-inf')
		best_model = None
		best_kernel = None
		
		# Try different kernels
		for kernel in kernels:
			gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
			
			# Use subset for GP due to computational complexity
			n_samples = min(1000, len(X_train))
			X_subset = X_train.iloc[:n_samples]
			y_subset = y_train.iloc[:n_samples]
			
			gp_model.fit(X_subset, y_subset)
			val_pred = gp_model.predict(X_val)
			
			score = r2_score(y_val, val_pred)
			if score > best_score:
				best_score = score
				best_model = gp_model
				best_kernel = kernel
		
		# Final predictions
		train_pred = best_model.predict(X_train.iloc[:n_samples])
		val_pred = best_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train.iloc[:n_samples], train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['gaussian_process'] = best_model
		
		return ModelPerformance(
			model_name='gaussian_process',
			model_type=ModelType.GAUSSIAN_PROCESS,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters={'kernel': str(best_kernel)},
			cv_scores=[val_metrics['rmse']],
			stability_score=0.0,
			complexity_score=2
		)
	
	async def _train_ensemble_models(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series,
		base_performances: List[ModelPerformance]
	) -> List[ModelPerformance]:
		"""Train ensemble models using best individual models"""
		
		logger.info("Training ensemble models...")
		
		# Select top 5 models for ensemble
		top_models = sorted(base_performances, key=lambda x: x.validation_metrics.get('r2', 0), reverse=True)[:5]
		
		ensemble_performances = []
		
		# Voting Regressor
		voting_perf = await self._train_voting_ensemble(X_train, y_train, X_val, y_val, top_models)
		if voting_perf:
			ensemble_performances.append(voting_perf)
		
		# Dynamic Weighted Ensemble
		dynamic_perf = await self._train_dynamic_ensemble(X_train, y_train, X_val, y_val, top_models)
		if dynamic_perf:
			ensemble_performances.append(dynamic_perf)
		
		return ensemble_performances
	
	async def _train_voting_ensemble(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series,
		top_models: List[ModelPerformance]
	) -> Optional[ModelPerformance]:
		"""Train voting ensemble"""
		
		start_time = datetime.now()
		
		# Get models that can be used in ensemble
		estimators = []
		for perf in top_models:
			model_name = perf.model_name
			if model_name in self.models and model_name not in ['lstm', 'transformer']:  # Exclude deep learning models
				estimators.append((model_name, self.models[model_name]))
		
		if len(estimators) < 2:
			return None
		
		# Create voting regressor
		voting_model = VotingRegressor(estimators=estimators)
		voting_model.fit(X_train, y_train)
		
		# Predictions
		train_pred = voting_model.predict(X_train)
		val_pred = voting_model.predict(X_val)
		
		train_metrics = self._calculate_metrics(y_train, train_pred)
		val_metrics = self._calculate_metrics(y_val, val_pred)
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		self.models['voting_ensemble'] = voting_model
		
		return ModelPerformance(
			model_name='voting_ensemble',
			model_type=ModelType.VOTING_REGRESSOR,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters={'estimators': [name for name, _ in estimators]},
			cv_scores=[val_metrics['rmse']],
			stability_score=0.0,
			complexity_score=len(estimators)
		)
	
	async def _train_dynamic_ensemble(
		self, 
		X_train: pd.DataFrame, 
		y_train: pd.Series, 
		X_val: pd.DataFrame, 
		y_val: pd.Series,
		top_models: List[ModelPerformance]
	) -> Optional[ModelPerformance]:
		"""Train dynamic weighted ensemble"""
		
		start_time = datetime.now()
		
		# Get predictions from all models
		model_predictions = {}
		for perf in top_models:
			model_name = perf.model_name
			if model_name in self.models:
				try:
					if model_name in ['lstm', 'transformer']:
						# Handle deep learning models differently
						continue
					else:
						val_pred = self.models[model_name].predict(X_val)
					model_predictions[model_name] = val_pred
				except Exception as e:
					logger.warning(f"Failed to get predictions from {model_name}: {e}")
		
		if len(model_predictions) < 2:
			return None
		
		# Calculate dynamic weights based on validation performance
		weights = {}
		total_weight = 0
		for model_name, predictions in model_predictions.items():
			# Use inverse of validation error as weight
			error = mean_squared_error(y_val, predictions)
			weight = 1 / (error + 1e-8)
			weights[model_name] = weight
			total_weight += weight
		
		# Normalize weights
		for model_name in weights:
			weights[model_name] /= total_weight
		
		# Create ensemble prediction
		ensemble_pred = np.zeros(len(y_val))
		for model_name, predictions in model_predictions.items():
			ensemble_pred += weights[model_name] * predictions
		
		# Calculate metrics
		val_metrics = self._calculate_metrics(y_val, ensemble_pred)
		
		# Calculate train metrics (approximate)
		train_predictions = {}
		for model_name in model_predictions.keys():
			if model_name not in ['lstm', 'transformer']:
				train_predictions[model_name] = self.models[model_name].predict(X_train)
		
		train_ensemble_pred = np.zeros(len(y_train))
		for model_name, predictions in train_predictions.items():
			train_ensemble_pred += weights[model_name] * predictions
		
		train_metrics = self._calculate_metrics(y_train, train_ensemble_pred)
		
		training_time = (datetime.now() - start_time).total_seconds()
		
		# Store weights for future use
		self.ensemble_weights = weights
		
		return ModelPerformance(
			model_name='dynamic_ensemble',
			model_type=ModelType.DYNAMIC_ENSEMBLE,
			train_metrics=train_metrics,
			validation_metrics=val_metrics,
			test_metrics={},
			training_time=training_time,
			prediction_time=0.0,
			feature_importance={},
			hyperparameters={'weights': weights},
			cv_scores=[val_metrics['rmse']],
			stability_score=0.0,
			complexity_score=len(weights)
		)
	
	def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Create sequences for time series models"""
		
		X_sequences = []
		y_sequences = []
		
		for i in range(sequence_length, len(X)):
			X_sequences.append(X[i-sequence_length:i])
			y_sequences.append(y[i])
		
		return np.array(X_sequences), np.array(y_sequences)
	
	def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
		"""Calculate comprehensive metrics"""
		
		return {
			'mae': mean_absolute_error(y_true, y_pred),
			'mse': mean_squared_error(y_true, y_pred),
			'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
			'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
			'r2': r2_score(y_true, y_pred),
			'max_error': np.max(np.abs(y_true - y_pred))
		}
	
	async def _cache_model_performances(self, performances: List[ModelPerformance]) -> None:
		"""Cache model performance results"""
		
		cache_key = f"ml_model_performances:{self.tenant_id}"
		
		performance_data = []
		for perf in performances:
			performance_data.append({
				'model_name': perf.model_name,
				'model_type': perf.model_type.value,
				'train_metrics': perf.train_metrics,
				'validation_metrics': perf.validation_metrics,
				'training_time': perf.training_time,
				'hyperparameters': perf.hyperparameters,
				'cv_scores': perf.cv_scores,
				'stability_score': perf.stability_score,
				'complexity_score': perf.complexity_score
			})
		
		await self.cache.set(cache_key, performance_data, ttl=86400)  # 24 hours
		
		logger.info(f"Cached performance data for {len(performances)} models")
	
	async def predict_with_uncertainty(
		self, 
		data: pd.DataFrame,
		model_names: Optional[List[str]] = None,
		return_intervals: bool = True
	) -> ForecastResult:
		"""Generate predictions with uncertainty quantification"""
		
		# Feature engineering for prediction data
		features, _ = await self.feature_engineer.engineer_features(
			data, target_column="amount", mode="predict"
		)
		
		# Select models to use
		if model_names is None:
			# Use top 3 performing models
			top_performances = sorted(
				self.model_performances.values(), 
				key=lambda x: x.validation_metrics.get('r2', 0), 
				reverse=True
			)[:3]
			model_names = [perf.model_name for perf in top_performances]
		
		# Get predictions from selected models
		predictions = {}
		for model_name in model_names:
			if model_name in self.models:
				try:
					if model_name in ['lstm', 'transformer']:
						# Handle deep learning models
						continue
					else:
						pred = self.models[model_name].predict(features)
					predictions[model_name] = pred
				except Exception as e:
					logger.warning(f"Failed to predict with {model_name}: {e}")
		
		if not predictions:
			raise ValueError("No valid predictions could be generated")
		
		# Calculate ensemble prediction
		if len(predictions) == 1:
			# Single model
			model_name = list(predictions.keys())[0]
			final_predictions = predictions[model_name]
			model_weights = {model_name: 1.0}
		else:
			# Ensemble using dynamic weights or equal weights
			if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
				weights = self.ensemble_weights
			else:
				# Equal weights
				weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
			
			final_predictions = np.zeros(len(features))
			for model_name, pred in predictions.items():
				weight = weights.get(model_name, 0.0)
				final_predictions += weight * pred
			
			model_weights = weights
		
		# Calculate prediction intervals and uncertainty
		if return_intervals and len(predictions) > 1:
			# Model uncertainty (epistemic) - disagreement between models
			pred_matrix = np.column_stack(list(predictions.values()))
			epistemic_uncertainty = np.std(pred_matrix, axis=1)
			
			# Total uncertainty estimation
			total_uncertainty = epistemic_uncertainty
			
			# Confidence intervals (assume normal distribution)
			z_score_95 = 1.96
			z_score_80 = 1.28
			
			confidence_lower = final_predictions - z_score_95 * total_uncertainty
			confidence_upper = final_predictions + z_score_95 * total_uncertainty
			
			prediction_lower = final_predictions - z_score_80 * total_uncertainty
			prediction_upper = final_predictions + z_score_80 * total_uncertainty
		else:
			# Single model - use historical error for uncertainty
			if model_names[0] in self.model_performances:
				historical_error = self.model_performances[model_names[0]].validation_metrics.get('rmse', 0)
				total_uncertainty = np.full(len(final_predictions), historical_error)
				epistemic_uncertainty = total_uncertainty
				
				confidence_lower = final_predictions - 1.96 * total_uncertainty
				confidence_upper = final_predictions + 1.96 * total_uncertainty
				prediction_lower = final_predictions - 1.28 * total_uncertainty
				prediction_upper = final_predictions + 1.28 * total_uncertainty
			else:
				total_uncertainty = np.zeros(len(final_predictions))
				epistemic_uncertainty = total_uncertainty
				confidence_lower = final_predictions
				confidence_upper = final_predictions
				prediction_lower = final_predictions
				prediction_upper = final_predictions
		
		# Aleatoric uncertainty (data noise) - simplified estimation
		aleatoric_uncertainty = total_uncertainty * 0.3  # Assume 30% of uncertainty is aleatoric
		
		# Feature contributions (simplified - use first model's feature importance if available)
		feature_contributions = {}
		if model_names and model_names[0] in self.model_performances:
			feature_importance = self.model_performances[model_names[0]].feature_importance
			if feature_importance:
				# Approximate feature contributions
				for feature, importance in feature_importance.items():
					if feature in features.columns:
						feature_values = features[feature].values
						contribution = feature_values * importance
						feature_contributions[feature] = contribution
		
		# Risk metrics
		risk_metrics = {
			'prediction_volatility': np.std(final_predictions),
			'max_uncertainty': np.max(total_uncertainty),
			'mean_uncertainty': np.mean(total_uncertainty),
			'uncertainty_ratio': np.mean(total_uncertainty) / (np.std(final_predictions) + 1e-8)
		}
		
		return ForecastResult(
			predictions=final_predictions,
			confidence_intervals=(confidence_lower, confidence_upper),
			prediction_intervals=(prediction_lower, prediction_upper),
			model_uncertainty=total_uncertainty,
			aleatoric_uncertainty=aleatoric_uncertainty,
			epistemic_uncertainty=epistemic_uncertainty,
			feature_contributions=feature_contributions,
			model_weights=model_weights,
			risk_metrics=risk_metrics
		)
	
	async def get_model_insights(self) -> Dict[str, Any]:
		"""Get comprehensive insights about model performance and behavior"""
		
		if not self.model_performances:
			return {"error": "No trained models available"}
		
		# Performance summary
		performance_summary = {}
		for name, perf in self.model_performances.items():
			performance_summary[name] = {
				'validation_r2': perf.validation_metrics.get('r2', 0),
				'validation_rmse': perf.validation_metrics.get('rmse', 0),
				'training_time': perf.training_time,
				'stability': perf.stability_score,
				'complexity': perf.complexity_score
			}
		
		# Best models by different criteria
		best_accuracy = max(self.model_performances.values(), key=lambda x: x.validation_metrics.get('r2', 0))
		best_speed = min(self.model_performances.values(), key=lambda x: x.training_time)
		best_stability = min(self.model_performances.values(), key=lambda x: x.stability_score)
		
		# Feature importance analysis
		feature_importance_combined = {}
		for perf in self.model_performances.values():
			for feature, importance in perf.feature_importance.items():
				if feature not in feature_importance_combined:
					feature_importance_combined[feature] = []
				feature_importance_combined[feature].append(importance)
		
		# Average feature importance
		avg_feature_importance = {}
		for feature, importances in feature_importance_combined.items():
			avg_feature_importance[feature] = np.mean(importances)
		
		# Sort by importance
		top_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
		
		insights = {
			'performance_summary': performance_summary,
			'best_models': {
				'accuracy': best_accuracy.model_name,
				'speed': best_speed.model_name,
				'stability': best_stability.model_name
			},
			'top_features': dict(top_features),
			'model_count': len(self.model_performances),
			'ensemble_available': 'dynamic_ensemble' in self.model_performances or 'voting_ensemble' in self.model_performances,
			'deep_learning_models': [name for name in self.model_performances.keys() if name in ['lstm', 'transformer']],
			'traditional_ml_models': [name for name in self.model_performances.keys() if name not in ['lstm', 'transformer', 'dynamic_ensemble', 'voting_ensemble']]
		}
		
		return insights

# ============================================================================
# Export
# ============================================================================

__all__ = [
	'AdvancedMLModelManager',
	'AdvancedFeatureEngineer',
	'ModelType',
	'EnsembleStrategy',
	'ModelPerformance',
	'ForecastResult',
	'LSTMNet',
	'TransformerNet'
]