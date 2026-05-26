"""
APG Connection Management ML Insights and Analytics
Advanced machine learning capabilities for data analysis, pattern recognition, and predictive insights

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import statistics
from collections import defaultdict, Counter
import pickle
import warnings
import sys

# Machine Learning imports
try:
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.ensemble import RandomForestClassifier, IsolationForest
	from sklearn.cluster import KMeans, DBSCAN
	from sklearn.decomposition import PCA
	from sklearn.metrics import classification_report, silhouette_score
	from sklearn.linear_model import LinearRegression
	import joblib
	SKLEARN_AVAILABLE = True
except ImportError:
	SKLEARN_AVAILABLE = False
	logging.warning("Scikit-learn not available. ML features will be limited.")

# Time series analysis
try:
	from statsmodels.tsa.arima.model import ARIMA
	from statsmodels.tsa.seasonal import seasonal_decompose
	import statsmodels.api as sm
	STATSMODELS_AVAILABLE = True
except ImportError:
	STATSMODELS_AVAILABLE = False
	logging.warning("Statsmodels not available. Time series analysis will be limited.")

# Natural Language Processing
try:
	from textblob import TextBlob
	import nltk
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize
	NLP_AVAILABLE = True
except ImportError:
	NLP_AVAILABLE = False
	logging.warning("NLP libraries not available. Text analysis will be limited.")

# Deep learning (optional)
try:
	import torch
	import torch.nn as nn
	from transformers import pipeline
	TORCH_AVAILABLE = True
except ImportError:
	TORCH_AVAILABLE = False
	logging.warning("PyTorch/Transformers not available. Deep learning features will be limited.")

from .error_handling import APGError, ErrorContext
from .monitoring import global_metrics_collector, monitor_performance
from .performance import cached
from .data_quality import DataQualityMetrics

logger = logging.getLogger(__name__)
sys.modules.setdefault("ml_insights", sys.modules[__name__])

# Suppress ML warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AnalysisType(str, Enum):
	"""Types of ML analysis"""
	ANOMALY_DETECTION = "anomaly_detection"
	CLUSTERING = "clustering"
	CLASSIFICATION = "classification"
	REGRESSION = "regression"
	TIME_SERIES_FORECASTING = "time_series_forecasting"
	PATTERN_RECOGNITION = "pattern_recognition"
	SENTIMENT_ANALYSIS = "sentiment_analysis"
	DATA_PROFILING = "data_profiling"
	RECOMMENDATION = "recommendation"


class ModelType(str, Enum):
	"""ML model types"""
	RANDOM_FOREST = "random_forest"
	ISOLATION_FOREST = "isolation_forest"
	KMEANS = "kmeans"
	DBSCAN = "dbscan"
	LINEAR_REGRESSION = "linear_regression"
	ARIMA = "arima"
	NEURAL_NETWORK = "neural_network"
	TRANSFORMER = "transformer"


class InsightSeverity(str, Enum):
	"""Insight severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


@dataclass
class MLInsight:
	"""Machine learning insight"""
	insight_id: str
	analysis_type: AnalysisType
	title: str
	description: str
	severity: InsightSeverity
	confidence: float
	evidence: Dict[str, Any] = field(default_factory=dict)
	recommendations: List[str] = field(default_factory=list)
	affected_fields: List[str] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
	generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelMetrics:
	"""ML model performance metrics"""
	model_type: ModelType
	accuracy: Optional[float] = None
	precision: Optional[float] = None
	recall: Optional[float] = None
	f1_score: Optional[float] = None
	mse: Optional[float] = None
	rmse: Optional[float] = None
	r2_score: Optional[float] = None
	silhouette_score: Optional[float] = None
	training_time: Optional[float] = None
	prediction_time: Optional[float] = None
	feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataPattern:
	"""Identified data pattern"""
	pattern_id: str
	pattern_type: str
	description: str
	frequency: int
	confidence: float
	examples: List[Any] = field(default_factory=list)
	fields_involved: List[str] = field(default_factory=list)
	temporal_info: Optional[Dict[str, Any]] = None


@dataclass
class AnomalyResult:
	"""Anomaly detection result"""
	total_records: int
	anomaly_count: int
	anomaly_rate: float
	anomalies: List[Dict[str, Any]] = field(default_factory=list)
	model_metrics: Optional[ModelMetrics] = None
	feature_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClusteringResult:
	"""Clustering analysis result"""
	num_clusters: int
	cluster_labels: List[int]
	cluster_centers: Optional[List[List[float]]] = None
	silhouette_score: Optional[float] = None
	cluster_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
	model_metrics: Optional[ModelMetrics] = None


@dataclass
class ForecastResult:
	"""Time series forecasting result"""
	forecast_values: List[float]
	confidence_intervals: Optional[List[Tuple[float, float]]] = None
	forecast_dates: Optional[List[datetime]] = None
	model_metrics: Optional[ModelMetrics] = None
	seasonal_decomposition: Optional[Dict[str, List[float]]] = None


class AnomalyDetector:
	"""Advanced anomaly detection using multiple algorithms"""

	def __init__(self):
		self.models = {}
		self.scalers = {}
		self.is_trained = False

	def train(self, data: pd.DataFrame, contamination: float = 0.1) -> ModelMetrics:
		"""Train anomaly detection models"""
		if not SKLEARN_AVAILABLE:
			raise APGError(
				message="Scikit-learn not available for anomaly detection",
				context=ErrorContext(tenant_id="system", operation="train_anomaly_detector")
			)

		try:
			# Prepare data
			numeric_cols = data.select_dtypes(include=[np.number]).columns
			if len(numeric_cols) == 0:
				raise APGError(
					message="No numeric columns found for anomaly detection",
					context=ErrorContext(tenant_id="system", operation="train_anomaly_detector")
				)

			X = data[numeric_cols].fillna(data[numeric_cols].mean())

			# Scale features
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)

			# Train Isolation Forest
			start_time = datetime.now()
			isolation_forest = IsolationForest(
				contamination=contamination,
				random_state=42,
				n_estimators=100
			)
			isolation_forest.fit(X_scaled)
			training_time = (datetime.now() - start_time).total_seconds()

			# Store models and scalers
			self.models['isolation_forest'] = isolation_forest
			self.scalers['main'] = scaler
			self.feature_names = list(numeric_cols)
			self.is_trained = True

			# Calculate metrics
			predictions = isolation_forest.predict(X_scaled)
			anomaly_count = np.sum(predictions == -1)
			anomaly_rate = anomaly_count / len(data)

			metrics = ModelMetrics(
				model_type=ModelType.ISOLATION_FOREST,
				training_time=training_time,
				feature_importance={col: 1.0/len(numeric_cols) for col in numeric_cols}
			)

			logger.info(f"Anomaly detection model trained: {anomaly_count} anomalies detected ({anomaly_rate:.2%})")
			return metrics

		except Exception as e:
			logger.error(f"Error training anomaly detection model: {e}")
			raise APGError(
				message=f"Failed to train anomaly detection model: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="train_anomaly_detector"),
				cause=e
			)

	def detect_anomalies(self, data: pd.DataFrame) -> AnomalyResult:
		"""Detect anomalies in data"""
		if not self.is_trained:
			raise APGError(
				message="Anomaly detection model not trained",
				context=ErrorContext(tenant_id="system", operation="detect_anomalies")
			)

		try:
			# Prepare data
			X = data[self.feature_names].fillna(data[self.feature_names].mean())
			X_scaled = self.scalers['main'].transform(X)

			# Predict anomalies
			start_time = datetime.now()
			predictions = self.models['isolation_forest'].predict(X_scaled)
			anomaly_scores = self.models['isolation_forest'].score_samples(X_scaled)
			prediction_time = (datetime.now() - start_time).total_seconds()

			# Extract anomalies
			anomaly_indices = np.where(predictions == -1)[0]
			anomalies = []

			for idx in anomaly_indices:
				anomaly_record = {
					'index': int(idx),
					'anomaly_score': float(anomaly_scores[idx]),
					'values': data.iloc[idx].to_dict(),
					'deviations': self._calculate_deviations(data.iloc[idx], data)
				}
				anomalies.append(anomaly_record)

			# Sort by anomaly score
			anomalies = sorted(anomalies, key=lambda x: x['anomaly_score'])

			# Calculate feature contributions
			feature_contributions = self._calculate_feature_contributions(X_scaled, anomaly_indices)

			result = AnomalyResult(
				total_records=len(data),
				anomaly_count=len(anomalies),
				anomaly_rate=len(anomalies) / len(data),
				anomalies=anomalies,
				feature_contributions=feature_contributions,
				model_metrics=ModelMetrics(
					model_type=ModelType.ISOLATION_FOREST,
					prediction_time=prediction_time
				)
			)

			return result

		except Exception as e:
			logger.error(f"Error detecting anomalies: {e}")
			raise APGError(
				message=f"Anomaly detection failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="detect_anomalies"),
				cause=e
			)

	def _calculate_deviations(self, record: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
		"""Calculate how much each field deviates from normal"""
		deviations = {}

		for col in self.feature_names:
			if col in record:
				value = record[col]
				col_data = data[col].dropna()

				if len(col_data) > 0:
					mean_val = col_data.mean()
					std_val = col_data.std()

					if std_val > 0:
						z_score = abs((value - mean_val) / std_val)
						deviations[col] = float(z_score)

		return deviations

	def _calculate_feature_contributions(self, X_scaled: np.ndarray, anomaly_indices: np.ndarray) -> Dict[str, float]:
		"""Calculate which features contribute most to anomalies"""
		if len(anomaly_indices) == 0:
			return {}

		anomaly_data = X_scaled[anomaly_indices]
		normal_data = X_scaled[np.setdiff1d(np.arange(len(X_scaled)), anomaly_indices)]

		contributions = {}

		for i, feature_name in enumerate(self.feature_names):
			if len(normal_data) > 0:
				anomaly_mean = np.mean(np.abs(anomaly_data[:, i]))
				normal_mean = np.mean(np.abs(normal_data[:, i]))
				contribution = anomaly_mean - normal_mean
				contributions[feature_name] = float(contribution)

		return contributions


class ClusterAnalyzer:
	"""Advanced clustering analysis"""

	def __init__(self):
		self.models = {}
		self.scalers = {}
		self.feature_names = []

	def analyze_clusters(self, data: pd.DataFrame,
						method: str = "kmeans",
						n_clusters: Optional[int] = None) -> ClusteringResult:
		"""Perform clustering analysis"""
		if not SKLEARN_AVAILABLE:
			raise APGError(
				message="Scikit-learn not available for clustering",
				context=ErrorContext(tenant_id="system", operation="analyze_clusters")
			)

		try:
			# Prepare data
			numeric_cols = data.select_dtypes(include=[np.number]).columns
			if len(numeric_cols) == 0:
				raise APGError(
					message="No numeric columns found for clustering",
					context=ErrorContext(tenant_id="system", operation="analyze_clusters")
				)

			X = data[numeric_cols].fillna(data[numeric_cols].mean())

			# Scale features
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)
			self.feature_names = list(numeric_cols)

			# Determine optimal number of clusters
			if n_clusters is None:
				n_clusters = self._find_optimal_clusters(X_scaled)

			# Perform clustering
			if method == "kmeans":
				model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
				cluster_labels = model.fit_predict(X_scaled)
				cluster_centers = model.cluster_centers_
				model_type = ModelType.KMEANS

			elif method == "dbscan":
				model = DBSCAN(eps=0.5, min_samples=5)
				cluster_labels = model.fit_predict(X_scaled)
				n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
				cluster_centers = None
				model_type = ModelType.DBSCAN

			else:
				raise APGError(
					message=f"Unknown clustering method: {method}",
					context=ErrorContext(tenant_id="system", operation="analyze_clusters")
				)

			# Calculate metrics
			if n_clusters > 1:
				silhouette_avg = silhouette_score(X_scaled, cluster_labels)
			else:
				silhouette_avg = 0

			# Calculate cluster statistics
			cluster_stats = self._calculate_cluster_stats(data, cluster_labels, numeric_cols)

			# Store model
			self.models[method] = model
			self.scalers[method] = scaler

			result = ClusteringResult(
				num_clusters=n_clusters,
				cluster_labels=cluster_labels.tolist(),
				cluster_centers=cluster_centers.tolist() if cluster_centers is not None else None,
				silhouette_score=float(silhouette_avg),
				cluster_stats=cluster_stats,
				model_metrics=ModelMetrics(
					model_type=model_type,
					silhouette_score=float(silhouette_avg)
				)
			)

			logger.info(f"Clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
			return result

		except Exception as e:
			logger.error(f"Error in clustering analysis: {e}")
			raise APGError(
				message=f"Clustering analysis failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="analyze_clusters"),
				cause=e
			)

	def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
		"""Find optimal number of clusters using elbow method"""
		inertias = []
		k_range = range(2, min(max_clusters + 1, len(X) // 2))

		for k in k_range:
			kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
			kmeans.fit(X)
			inertias.append(kmeans.inertia_)

		# Simple elbow detection
		if len(inertias) >= 2:
			diffs = np.diff(inertias)
			second_diffs = np.diff(diffs)

			if len(second_diffs) > 0:
				elbow_idx = np.argmax(second_diffs) + 2
				return int(min(elbow_idx, max_clusters))

		return 3  # Default

	def _calculate_cluster_stats(self, data: pd.DataFrame,
								 cluster_labels: np.ndarray,
								 numeric_cols: List[str]) -> Dict[int, Dict[str, Any]]:
		"""Calculate statistics for each cluster"""
		cluster_stats = {}
		unique_labels = set(cluster_labels)

		for label in unique_labels:
			if label == -1:  # Noise points in DBSCAN
				continue

			cluster_mask = cluster_labels == label
			cluster_data = data[cluster_mask]

			stats = {
				'size': int(np.sum(cluster_mask)),
				'percentage': float(np.sum(cluster_mask) / len(data) * 100),
				'feature_means': {},
				'feature_stds': {}
			}

			for col in numeric_cols:
				if col in cluster_data.columns:
					col_data = cluster_data[col].dropna()
					if len(col_data) > 0:
						stats['feature_means'][col] = float(col_data.mean())
						stats['feature_stds'][col] = float(col_data.std())

			cluster_stats[int(label)] = stats

		return cluster_stats


class TimeSeriesAnalyzer:
	"""Advanced time series analysis and forecasting"""

	def __init__(self):
		self.models = {}
		self.is_fitted = {}

	def forecast(self, data: pd.Series, periods: int = 10,
				method: str = "arima") -> ForecastResult:
		"""Generate time series forecast"""

		if method == "arima" and not STATSMODELS_AVAILABLE:
			raise APGError(
				message="Statsmodels not available for ARIMA forecasting",
				context=ErrorContext(tenant_id="system", operation="time_series_forecast")
			)

		try:
			# Prepare data
			if data.isnull().sum() > len(data) * 0.5:
				raise APGError(
					message="Too many missing values in time series data",
					context=ErrorContext(tenant_id="system", operation="time_series_forecast")
				)

			# Fill missing values
			data_filled = data.fillna(data.median())

			if method == "arima":
				return self._arima_forecast(data_filled, periods)
			elif method == "linear":
				return self._linear_forecast(data_filled, periods)
			else:
				raise APGError(
					message=f"Unknown forecasting method: {method}",
					context=ErrorContext(tenant_id="system", operation="time_series_forecast")
				)

		except Exception as e:
			logger.error(f"Error in time series forecasting: {e}")
			raise APGError(
				message=f"Time series forecasting failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="time_series_forecast"),
				cause=e
			)

	def _arima_forecast(self, data: pd.Series, periods: int) -> ForecastResult:
		"""ARIMA forecasting"""
		try:
			# Fit ARIMA model (auto-detect parameters)
			model = ARIMA(data, order=(1, 1, 1))
			fitted_model = model.fit()

			# Generate forecast
			forecast = fitted_model.forecast(steps=periods)
			conf_int = fitted_model.get_forecast(steps=periods).conf_int()

			# Seasonal decomposition if enough data
			seasonal_decomp = None
			if len(data) >= 24:  # Need at least 2 seasons
				try:
					decomposition = seasonal_decompose(data, period=min(12, len(data)//2))
					seasonal_decomp = {
						'trend': decomposition.trend.dropna().tolist(),
						'seasonal': decomposition.seasonal.dropna().tolist(),
						'residual': decomposition.resid.dropna().tolist()
					}
				except Exception as e:
					logger.warning(f"Could not perform seasonal decomposition: {e}")

			# Generate forecast dates
			last_date = data.index[-1] if hasattr(data, 'index') else None
			forecast_dates = None
			if last_date and hasattr(last_date, 'to_pydatetime'):
				forecast_dates = [
					(last_date + pd.Timedelta(days=i+1)).to_pydatetime()
					for i in range(periods)
				]

			return ForecastResult(
				forecast_values=forecast.tolist(),
				confidence_intervals=[(float(row[0]), float(row[1])) for row in conf_int.values],
				forecast_dates=forecast_dates,
				seasonal_decomposition=seasonal_decomp,
				model_metrics=ModelMetrics(
					model_type=ModelType.ARIMA,
					mse=float(fitted_model.mse) if hasattr(fitted_model, 'mse') else None
				)
			)

		except Exception as e:
			logger.error(f"ARIMA forecasting error: {e}")
			# Fallback to linear forecast
			return self._linear_forecast(data, periods)

	def _linear_forecast(self, data: pd.Series, periods: int) -> ForecastResult:
		"""Simple linear trend forecasting"""
		if not SKLEARN_AVAILABLE:
			# Very simple trend calculation
			if len(data) < 2:
				return ForecastResult(forecast_values=[float(data.iloc[-1])] * periods)

			trend = (data.iloc[-1] - data.iloc[0]) / (len(data) - 1)
			last_value = data.iloc[-1]
			forecast = [float(last_value + trend * (i + 1)) for i in range(periods)]

			return ForecastResult(forecast_values=forecast)

		# Use scikit-learn linear regression
		X = np.arange(len(data)).reshape(-1, 1)
		y = data.values

		model = LinearRegression()
		model.fit(X, y)

		# Forecast
		future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
		forecast = model.predict(future_X)

		return ForecastResult(
			forecast_values=forecast.tolist(),
			model_metrics=ModelMetrics(
				model_type=ModelType.LINEAR_REGRESSION,
				r2_score=float(model.score(X, y))
			)
		)


class PatternRecognizer:
	"""Advanced pattern recognition in data"""

	def __init__(self):
		self.patterns = []

	def identify_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Identify various patterns in data"""
		patterns = []

		try:
			# Numeric patterns
			patterns.extend(self._find_numeric_patterns(data))

			# Categorical patterns
			patterns.extend(self._find_categorical_patterns(data))

			# Temporal patterns
			patterns.extend(self._find_temporal_patterns(data))

			# Correlation patterns
			patterns.extend(self._find_correlation_patterns(data))

			# Sequential patterns
			patterns.extend(self._find_sequential_patterns(data))

			logger.info(f"Identified {len(patterns)} data patterns")
			return patterns

		except Exception as e:
			logger.error(f"Error in pattern recognition: {e}")
			return []

	def _find_numeric_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Find patterns in numeric data"""
		patterns = []
		numeric_cols = data.select_dtypes(include=[np.number]).columns

		for col in numeric_cols:
			col_data = data[col].dropna()
			if len(col_data) == 0:
				continue

			# Detect constant values
			if col_data.nunique() == 1:
				patterns.append(DataPattern(
					pattern_id=f"constant_{col}",
					pattern_type="constant_value",
					description=f"Column '{col}' has constant value: {col_data.iloc[0]}",
					frequency=len(col_data),
					confidence=1.0,
					fields_involved=[col]
				))

			# Detect arithmetic sequences
			if len(col_data) >= 3:
				diffs = col_data.diff().dropna()
				if diffs.nunique() == 1 and not pd.isna(diffs.iloc[0]):
					patterns.append(DataPattern(
						pattern_id=f"arithmetic_{col}",
						pattern_type="arithmetic_sequence",
						description=f"Column '{col}' follows arithmetic sequence with difference: {diffs.iloc[0]}",
						frequency=len(col_data),
						confidence=0.95,
						fields_involved=[col]
					))

			# Detect outliers using IQR
			Q1 = col_data.quantile(0.25)
			Q3 = col_data.quantile(0.75)
			IQR = Q3 - Q1
			lower_bound = Q1 - 1.5 * IQR
			upper_bound = Q3 + 1.5 * IQR

			outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
			if len(outliers) > 0:
				patterns.append(DataPattern(
					pattern_id=f"outliers_{col}",
					pattern_type="statistical_outliers",
					description=f"Column '{col}' has {len(outliers)} outliers",
					frequency=len(outliers),
					confidence=0.8,
					examples=outliers.tolist()[:5],
					fields_involved=[col]
				))

		return patterns

	def _find_categorical_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Find patterns in categorical data"""
		patterns = []
		categorical_cols = data.select_dtypes(include=['object', 'category']).columns

		for col in categorical_cols:
			col_data = data[col].dropna()
			if len(col_data) == 0:
				continue

			# Value frequency patterns
			value_counts = col_data.value_counts()

			# Dominant value pattern
			if len(value_counts) > 1:
				dominant_ratio = value_counts.iloc[0] / len(col_data)
				if dominant_ratio > 0.8:
					patterns.append(DataPattern(
						pattern_id=f"dominant_{col}",
						pattern_type="dominant_value",
						description=f"Column '{col}' is dominated by value '{value_counts.index[0]}' ({dominant_ratio:.1%})",
						frequency=int(value_counts.iloc[0]),
						confidence=dominant_ratio,
						examples=[value_counts.index[0]],
						fields_involved=[col]
					))

			# High cardinality pattern
			cardinality_ratio = len(value_counts) / len(col_data)
			if cardinality_ratio > 0.8:
				patterns.append(DataPattern(
					pattern_id=f"high_cardinality_{col}",
					pattern_type="high_cardinality",
					description=f"Column '{col}' has high cardinality ({len(value_counts)} unique values)",
					frequency=len(value_counts),
					confidence=cardinality_ratio,
					fields_involved=[col]
				))

		return patterns

	def _find_temporal_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Find temporal patterns in data"""
		patterns = []

		# Look for datetime columns
		datetime_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns

		for col in datetime_cols:
			col_data = data[col].dropna()
			if len(col_data) == 0:
				continue

			# Regular intervals
			if len(col_data) >= 3:
				sorted_data = col_data.sort_values()
				intervals = sorted_data.diff().dropna()

				# Check if intervals are consistent
				if len(intervals.unique()) == 1:
					interval = intervals.iloc[0]
					patterns.append(DataPattern(
						pattern_id=f"regular_interval_{col}",
						pattern_type="regular_time_interval",
						description=f"Column '{col}' has regular intervals of {interval}",
						frequency=len(col_data),
						confidence=0.9,
						fields_involved=[col],
						temporal_info={"interval": str(interval)}
					))

		return patterns

	def _find_correlation_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Find correlation patterns between columns"""
		patterns = []
		numeric_cols = data.select_dtypes(include=[np.number]).columns

		if len(numeric_cols) >= 2:
			corr_matrix = data[numeric_cols].corr()

			# Find high correlations
			for i, col1 in enumerate(numeric_cols):
				for j, col2 in enumerate(numeric_cols[i+1:], i+1):
					corr_value = corr_matrix.loc[col1, col2]

					if abs(corr_value) > 0.8 and not pd.isna(corr_value):
						pattern_type = "positive_correlation" if corr_value > 0 else "negative_correlation"
						patterns.append(DataPattern(
							pattern_id=f"corr_{col1}_{col2}",
							pattern_type=pattern_type,
							description=f"Strong {pattern_type.replace('_', ' ')} between '{col1}' and '{col2}' (r={corr_value:.3f})",
							frequency=len(data),
							confidence=abs(corr_value),
							fields_involved=[col1, col2]
						))

		return patterns

	def _find_sequential_patterns(self, data: pd.DataFrame) -> List[DataPattern]:
		"""Find sequential patterns in data"""
		patterns = []

		# Look for patterns in row sequences
		for col in data.columns:
			col_data = data[col].dropna()
			if len(col_data) < 3:
				continue

			# Look for repeating subsequences
			if col_data.dtype == 'object':
				# Check for repeating string patterns
				values = col_data.tolist()
				for window_size in [2, 3, 4]:
					if len(values) >= window_size * 2:
						for i in range(len(values) - window_size + 1):
							pattern = values[i:i+window_size]
							count = 0

							for j in range(i + window_size, len(values) - window_size + 1):
								if values[j:j+window_size] == pattern:
									count += 1

							if count >= 2:  # Pattern repeats at least twice
								patterns.append(DataPattern(
									pattern_id=f"sequential_{col}_{i}",
									pattern_type="repeating_sequence",
									description=f"Repeating sequence in '{col}': {pattern}",
									frequency=count + 1,
									confidence=0.7,
									examples=[pattern],
									fields_involved=[col]
								))
								break  # Avoid duplicate patterns

		return patterns


class SentimentAnalyzer:
	"""Text sentiment analysis"""

	def __init__(self):
		self.is_initialized = False

	def _initialize_nltk(self):
		"""Initialize NLTK data"""
		if not self.is_initialized and NLP_AVAILABLE:
			try:
				nltk.download('punkt', quiet=True)
				nltk.download('stopwords', quiet=True)
				nltk.download('vader_lexicon', quiet=True)
				self.is_initialized = True
			except Exception as e:
				logger.warning(f"Could not initialize NLTK: {e}")

	def analyze_sentiment(self, data: pd.DataFrame, text_columns: List[str] = None) -> Dict[str, Any]:
		"""Analyze sentiment in text columns"""
		if not NLP_AVAILABLE:
			logger.warning("NLP libraries not available for sentiment analysis")
			return {'error': 'NLP libraries not available'}

		self._initialize_nltk()

		try:
			results = {}

			# Auto-detect text columns if not specified
			if text_columns is None:
				text_columns = data.select_dtypes(include=['object']).columns.tolist()

			for col in text_columns:
				if col not in data.columns:
					continue

				text_data = data[col].dropna().astype(str)
				if len(text_data) == 0:
					continue

				sentiments = []
				polarities = []
				subjectivities = []

				for text in text_data:
					try:
						blob = TextBlob(text)
						sentiment = blob.sentiment

						polarities.append(sentiment.polarity)
						subjectivities.append(sentiment.subjectivity)

						# Classify sentiment
						if sentiment.polarity > 0.1:
							sentiments.append('positive')
						elif sentiment.polarity < -0.1:
							sentiments.append('negative')
						else:
							sentiments.append('neutral')
					except Exception:
						sentiments.append('neutral')
						polarities.append(0.0)
						subjectivities.append(0.0)

				# Calculate statistics
				sentiment_counts = Counter(sentiments)
				avg_polarity = statistics.mean(polarities) if polarities else 0
				avg_subjectivity = statistics.mean(subjectivities) if subjectivities else 0

				results[col] = {
					'sentiment_distribution': dict(sentiment_counts),
					'average_polarity': avg_polarity,
					'average_subjectivity': avg_subjectivity,
					'total_texts': len(sentiments),
					'sentiment_trend': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral'
				}

			return results

		except Exception as e:
			logger.error(f"Error in sentiment analysis: {e}")
			return {'error': str(e)}


class MLInsightsEngine:
	"""Main ML insights and analytics engine"""

	def __init__(self):
		self.anomaly_detector = AnomalyDetector()
		self.cluster_analyzer = ClusterAnalyzer()
		self.time_series_analyzer = TimeSeriesAnalyzer()
		self.pattern_recognizer = PatternRecognizer()
		self.sentiment_analyzer = SentimentAnalyzer()
		self.insights_cache = {}

	@monitor_performance("ml_insights_analysis")
	async def analyze_data(self, data: Union[pd.DataFrame, List[Dict[str, Any]]],
						   analysis_types: List[AnalysisType] = None,
						   connection_id: str = None) -> List[MLInsight]:
		"""Perform comprehensive ML analysis and generate insights"""

		try:
			# Convert data to DataFrame if needed
			if isinstance(data, list):
				df = pd.DataFrame(data)
			elif isinstance(data, pd.DataFrame):
				df = data.copy()
			else:
				raise APGError(
					message=f"Unsupported data format: {type(data)}",
					context=ErrorContext(tenant_id="system", operation="ml_analysis")
				)

			if df.empty:
				return []

			# Default analysis types
			if analysis_types is None:
				analysis_types = [
					AnalysisType.ANOMALY_DETECTION,
					AnalysisType.PATTERN_RECOGNITION,
					AnalysisType.DATA_PROFILING
				]

			insights = []

			# Anomaly detection
			if AnalysisType.ANOMALY_DETECTION in analysis_types:
				insights.extend(await self._perform_anomaly_analysis(df))

			# Clustering analysis
			if AnalysisType.CLUSTERING in analysis_types:
				insights.extend(await self._perform_clustering_analysis(df))

			# Pattern recognition
			if AnalysisType.PATTERN_RECOGNITION in analysis_types:
				insights.extend(await self._perform_pattern_analysis(df))

			# Time series forecasting
			if AnalysisType.TIME_SERIES_FORECASTING in analysis_types:
				insights.extend(await self._perform_time_series_analysis(df))

			# Sentiment analysis
			if AnalysisType.SENTIMENT_ANALYSIS in analysis_types:
				insights.extend(await self._perform_sentiment_analysis(df))

			# Data profiling insights
			if AnalysisType.DATA_PROFILING in analysis_types:
				insights.extend(await self._perform_data_profiling_analysis(df))

			# Update metrics
			global_metrics_collector.record_counter(
				"ml_insights_analyses_total",
				1,
				{
					"connection_id": connection_id or "unknown",
					"analysis_count": str(len(analysis_types))
				}
			)

			global_metrics_collector.record_gauge(
				"ml_insights_generated",
				len(insights),
				{"connection_id": connection_id or "unknown"}
			)

			logger.info(f"ML analysis completed: {len(insights)} insights generated")
			return insights

		except Exception as e:
			logger.error(f"Error in ML analysis: {e}")
			raise APGError(
				message=f"ML analysis failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="ml_analysis"),
				cause=e
			)

	async def _perform_anomaly_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform anomaly detection analysis"""
		insights = []

		try:
			# Train and detect anomalies
			self.anomaly_detector.train(df)
			result = self.anomaly_detector.detect_anomalies(df)

			if result.anomaly_rate > 0.05:  # More than 5% anomalies
				severity = InsightSeverity.HIGH if result.anomaly_rate > 0.2 else InsightSeverity.MEDIUM

				insight = MLInsight(
					insight_id=f"anomalies_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
					analysis_type=AnalysisType.ANOMALY_DETECTION,
					title="Anomalous Data Points Detected",
					description=f"Detected {result.anomaly_count} anomalous records ({result.anomaly_rate:.1%} of total data)",
					severity=severity,
					confidence=0.8,
					evidence={
						'anomaly_count': result.anomaly_count,
						'anomaly_rate': result.anomaly_rate,
						'top_anomalies': result.anomalies[:5],
						'feature_contributions': result.feature_contributions
					},
					recommendations=[
						"Investigate the identified anomalous records for data quality issues",
						"Consider implementing automated anomaly detection in your data pipeline",
						"Review data collection processes for the fields contributing most to anomalies"
					],
					affected_fields=list(result.feature_contributions.keys())
				)
				insights.append(insight)

		except Exception as e:
			logger.warning(f"Anomaly detection failed: {e}")

		return insights

	async def _perform_clustering_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform clustering analysis"""
		insights = []

		try:
			result = self.cluster_analyzer.analyze_clusters(df)

			if result.num_clusters > 1:
				insight = MLInsight(
					insight_id=f"clusters_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
					analysis_type=AnalysisType.CLUSTERING,
					title="Natural Data Groupings Discovered",
					description=f"Data naturally groups into {result.num_clusters} clusters with silhouette score {result.silhouette_score:.3f}",
					severity=InsightSeverity.MEDIUM,
					confidence=result.silhouette_score,
					evidence={
						'num_clusters': result.num_clusters,
						'silhouette_score': result.silhouette_score,
						'cluster_stats': result.cluster_stats
					},
					recommendations=[
						"Consider segmenting your data processing based on these natural groupings",
						"Use cluster information for targeted data quality improvements",
						"Explore cluster-specific patterns and behaviors"
					]
				)
				insights.append(insight)

		except Exception as e:
			logger.warning(f"Clustering analysis failed: {e}")

		return insights

	async def _perform_pattern_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform pattern recognition analysis"""
		insights = []

		try:
			patterns = self.pattern_recognizer.identify_patterns(df)

			# Group patterns by type
			pattern_types = defaultdict(list)
			for pattern in patterns:
				pattern_types[pattern.pattern_type].append(pattern)

			# Create insights for significant patterns
			for pattern_type, pattern_list in pattern_types.items():
				if len(pattern_list) >= 1:
					high_conf_patterns = [p for p in pattern_list if p.confidence > 0.7]

					if high_conf_patterns:
						insight = MLInsight(
							insight_id=f"patterns_{pattern_type}_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
							analysis_type=AnalysisType.PATTERN_RECOGNITION,
							title=f"{pattern_type.replace('_', ' ').title()} Pattern Detected",
							description=f"Found {len(high_conf_patterns)} instances of {pattern_type.replace('_', ' ')} patterns",
							severity=InsightSeverity.LOW,
							confidence=statistics.mean([p.confidence for p in high_conf_patterns]),
							evidence={
								'pattern_type': pattern_type,
								'pattern_count': len(high_conf_patterns),
								'patterns': [
									{
										'description': p.description,
										'frequency': p.frequency,
										'confidence': p.confidence,
										'fields': p.fields_involved
									}
									for p in high_conf_patterns[:3]  # Top 3 patterns
								]
							},
							recommendations=[
								"Leverage identified patterns for data validation rules",
								"Use patterns to optimize data storage and indexing",
								"Consider pattern-based data compression techniques"
							],
							affected_fields=list(set().union(*[p.fields_involved for p in high_conf_patterns]))
						)
						insights.append(insight)

		except Exception as e:
			logger.warning(f"Pattern analysis failed: {e}")

		return insights

	async def _perform_time_series_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform time series analysis"""
		insights = []

		try:
			# Look for numeric columns that could be time series
			numeric_cols = df.select_dtypes(include=[np.number]).columns
			datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns

			if len(numeric_cols) > 0 and len(df) >= 10:
				for col in numeric_cols[:3]:  # Analyze up to 3 numeric columns
					series = df[col].dropna()
					if len(series) >= 10:
						try:
							forecast_result = self.time_series_analyzer.forecast(series, periods=5)

							if forecast_result.forecast_values:
								trend_direction = "increasing" if forecast_result.forecast_values[-1] > series.iloc[-1] else "decreasing"

								insight = MLInsight(
									insight_id=f"forecast_{col}_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
									analysis_type=AnalysisType.TIME_SERIES_FORECASTING,
									title=f"Forecast for {col}",
									description=f"Time series forecast shows {trend_direction} trend for column '{col}'",
									severity=InsightSeverity.LOW,
									confidence=0.6,
									evidence={
										'column': col,
										'forecast_values': forecast_result.forecast_values,
										'trend_direction': trend_direction,
										'forecast_periods': 5
									},
									recommendations=[
										f"Monitor {col} values for expected {trend_direction} trend",
										"Use forecasts for capacity planning and resource allocation",
										"Set up alerts if actual values deviate significantly from forecast"
									],
									affected_fields=[col]
								)
								insights.append(insight)
						except Exception as e:
							logger.debug(f"Time series analysis failed for {col}: {e}")

		except Exception as e:
			logger.warning(f"Time series analysis failed: {e}")

		return insights

	async def _perform_sentiment_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform sentiment analysis"""
		insights = []

		try:
			text_cols = df.select_dtypes(include=['object']).columns
			if len(text_cols) > 0:
				sentiment_results = self.sentiment_analyzer.analyze_sentiment(df, text_cols.tolist())

				if 'error' not in sentiment_results:
					for col, results in sentiment_results.items():
						if results['total_texts'] > 0:
							dominant_sentiment = max(results['sentiment_distribution'].items(), key=lambda x: x[1])

							if dominant_sentiment[1] / results['total_texts'] > 0.7:  # Strong sentiment bias
								severity = InsightSeverity.MEDIUM if abs(results['average_polarity']) > 0.5 else InsightSeverity.LOW

								insight = MLInsight(
									insight_id=f"sentiment_{col}_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
									analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
									title=f"Sentiment Bias in {col}",
									description=f"Column '{col}' shows strong {dominant_sentiment[0]} sentiment bias ({dominant_sentiment[1]/results['total_texts']:.1%})",
									severity=severity,
									confidence=0.7,
									evidence={
										'column': col,
										'sentiment_distribution': results['sentiment_distribution'],
										'average_polarity': results['average_polarity'],
										'dominant_sentiment': dominant_sentiment[0],
										'bias_percentage': dominant_sentiment[1] / results['total_texts']
									},
									recommendations=[
										"Consider the impact of sentiment bias on your analysis",
										"Use sentiment information for customer feedback analysis",
										"Monitor sentiment trends over time"
									],
									affected_fields=[col]
								)
								insights.append(insight)

		except Exception as e:
			logger.warning(f"Sentiment analysis failed: {e}")

		return insights

	async def _perform_data_profiling_analysis(self, df: pd.DataFrame) -> List[MLInsight]:
		"""Perform data profiling analysis"""
		insights = []

		try:
			# Data quality insights
			missing_data = df.isnull().sum()
			high_missing_cols = missing_data[missing_data > len(df) * 0.3].index.tolist()

			if high_missing_cols:
				insight = MLInsight(
					insight_id=f"missing_data_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
					analysis_type=AnalysisType.DATA_PROFILING,
					title="High Missing Data Detected",
					description=f"{len(high_missing_cols)} columns have >30% missing values",
					severity=InsightSeverity.HIGH,
					confidence=0.9,
					evidence={
						'high_missing_columns': high_missing_cols,
						'missing_percentages': {col: float(missing_data[col] / len(df)) for col in high_missing_cols}
					},
					recommendations=[
						"Investigate data collection processes for high missing value columns",
						"Consider imputation strategies or column removal",
						"Implement data quality monitoring"
					],
					affected_fields=high_missing_cols
				)
				insights.append(insight)

			# Data type insights
			potential_categorical = []
			for col in df.select_dtypes(include=['object']).columns:
				unique_ratio = df[col].nunique() / len(df)
				if unique_ratio < 0.1 and df[col].nunique() > 1:  # Low cardinality
					potential_categorical.append(col)

			if potential_categorical:
				insight = MLInsight(
					insight_id=f"categorical_{hashlib.md5(str(df.shape).encode()).hexdigest()[:8]}",
					analysis_type=AnalysisType.DATA_PROFILING,
					title="Categorical Data Optimization Opportunity",
					description=f"{len(potential_categorical)} columns could benefit from categorical encoding",
					severity=InsightSeverity.LOW,
					confidence=0.8,
					evidence={
						'categorical_candidates': potential_categorical,
						'potential_memory_savings': f"{len(potential_categorical) * len(df) * 0.1:.0f} bytes"
					},
					recommendations=[
						"Convert low-cardinality string columns to categorical data type",
						"Use categorical encoding for better memory efficiency",
						"Consider one-hot encoding for machine learning models"
					],
					affected_fields=potential_categorical
				)
				insights.append(insight)

		except Exception as e:
			logger.warning(f"Data profiling analysis failed: {e}")

		return insights


# Global ML insights engine
global_ml_insights_engine = MLInsightsEngine()


# Convenience functions
@cached(ttl=600)  # Cache for 10 minutes
async def generate_ml_insights(data: Union[pd.DataFrame, List[Dict[str, Any]]],
							   connection_id: str = None,
							   analysis_types: List[str] = None) -> List[MLInsight]:
	"""Generate ML insights for connection data"""
	if analysis_types:
		analysis_types = [AnalysisType(t) for t in analysis_types]

	return await global_ml_insights_engine.analyze_data(data, analysis_types, connection_id)


async def get_anomaly_insights(data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> AnomalyResult:
	"""Get anomaly detection insights"""
	engine = MLInsightsEngine()

	if isinstance(data, list):
		df = pd.DataFrame(data)
	else:
		df = data

	engine.anomaly_detector.train(df)
	return engine.anomaly_detector.detect_anomalies(df)


async def get_clustering_insights(data: Union[pd.DataFrame, List[Dict[str, Any]]],
								  n_clusters: int = None) -> ClusteringResult:
	"""Get clustering analysis insights"""
	engine = MLInsightsEngine()

	if isinstance(data, list):
		df = pd.DataFrame(data)
	else:
		df = data

	return engine.cluster_analyzer.analyze_clusters(df, n_clusters=n_clusters)


async def forecast_time_series(data: pd.Series, periods: int = 10) -> ForecastResult:
	"""Generate time series forecast"""
	engine = MLInsightsEngine()
	return engine.time_series_analyzer.forecast(data, periods)
