#!/usr/bin/env python3
"""
Advanced Time-Series Analytics and Forecasting Engine
=====================================================

Sophisticated time-series analysis system for digital twins with multi-variate forecasting,
anomaly detection, seasonality analysis, and uncertainty quantification.
Supports streaming data, real-time analytics, and business intelligence integration.
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics imports
try:
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	from sklearn.ensemble import IsolationForest
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
	from sklearn.model_selection import TimeSeriesSplit
	import joblib
except ImportError:
	print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
	from scipy import stats
	from scipy.signal import find_peaks, savgol_filter, periodogram
	from scipy.optimize import minimize
except ImportError:
	print("Warning: scipy not available. Install with: pip install scipy")

try:
	from statsmodels.tsa.arima.model import ARIMA
	from statsmodels.tsa.holtwinters import ExponentialSmoothing
	from statsmodels.tsa.seasonal import seasonal_decompose
	from statsmodels.tsa.stattools import adfuller, acf, pacf
	from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
	print("Warning: statsmodels not available. Install with: pip install statsmodels")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("time_series_analytics")

class ForecastModel(Enum):
	"""Types of forecasting models"""
	ARIMA = "arima"
	EXPONENTIAL_SMOOTHING = "exponential_smoothing"
	LINEAR_REGRESSION = "linear_regression"
	POLYNOMIAL = "polynomial"
	LSTM = "lstm"
	TRANSFORMER = "transformer"
	PROPHET = "prophet"
	ENSEMBLE = "ensemble"

class SeasonalityType(Enum):
	"""Types of seasonality patterns"""
	NONE = "none"
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	YEARLY = "yearly"
	CUSTOM = "custom"

class TrendType(Enum):
	"""Types of trends"""
	NONE = "none"
	LINEAR = "linear"
	EXPONENTIAL = "exponential"
	LOGARITHMIC = "logarithmic"
	POLYNOMIAL = "polynomial"
	CYCLIC = "cyclic"

class AnomalyType(Enum):
	"""Types of anomalies in time series"""
	POINT = "point"				# Single point anomaly
	CONTEXTUAL = "contextual"	# Anomaly in specific context
	COLLECTIVE = "collective"	# Group of anomalous points
	SEASONAL = "seasonal"		# Seasonal pattern anomaly
	TREND = "trend"				# Trend change anomaly

@dataclass
class TimeSeriesMetrics:
	"""Time series analysis metrics"""
	stationarity_pvalue: float
	trend_strength: float
	seasonal_strength: float
	noise_level: float
	autocorrelation_1: float
	partial_autocorr_1: float
	mean: float
	std: float
	skewness: float
	kurtosis: float
	missing_ratio: float
	
@dataclass
class ForecastResult:
	"""Forecast result with confidence intervals"""
	timestamp: datetime
	predicted_value: float
	confidence_lower: float
	confidence_upper: float
	prediction_interval_lower: float
	prediction_interval_upper: float
	model_used: str
	confidence_score: float

@dataclass
class SeasonalityPattern:
	"""Detected seasonality pattern"""
	pattern_type: SeasonalityType
	period: int
	strength: float
	phase: float
	amplitude: float
	confidence: float

@dataclass
class TrendAnalysis:
	"""Trend analysis results"""
	trend_type: TrendType
	slope: float
	strength: float
	change_points: List[datetime]
	trend_equation: str
	r_squared: float

@dataclass
class AnomalyDetection:
	"""Anomaly detection result"""
	timestamp: datetime
	value: float
	anomaly_type: AnomalyType
	severity_score: float
	confidence: float
	expected_value: float
	deviation: float
	context: Dict[str, Any]

class TimeSeriesPreprocessor:
	"""Preprocess time series data for analysis"""
	
	@staticmethod
	def clean_data(df: pd.DataFrame, timestamp_col: str, value_col: str) -> pd.DataFrame:
		"""Clean and prepare time series data"""
		
		# Ensure datetime index
		df[timestamp_col] = pd.to_datetime(df[timestamp_col])
		df = df.set_index(timestamp_col).sort_index()
		
		# Remove duplicates
		df = df[~df.index.duplicated(keep='first')]
		
		# Handle missing values
		df[value_col] = df[value_col].interpolate(method='time')
		
		# Remove extreme outliers (>5 std deviations)
		mean_val = df[value_col].mean()
		std_val = df[value_col].std()
		df = df[np.abs(df[value_col] - mean_val) <= 5 * std_val]
		
		return df
		
	@staticmethod
	def resample_data(df: pd.DataFrame, frequency: str, 
					  aggregation: str = 'mean') -> pd.DataFrame:
		"""Resample time series to different frequency"""
		
		agg_methods = {
			'mean': 'mean',
			'sum': 'sum',
			'max': 'max',
			'min': 'min',
			'median': 'median',
			'std': 'std'
		}
		
		method = agg_methods.get(aggregation, 'mean')
		return df.resample(frequency).agg(method)
		
	@staticmethod
	def detect_frequency(df: pd.DataFrame) -> str:
		"""Auto-detect time series frequency"""
		
		time_diffs = df.index.to_series().diff().dropna()
		mode_diff = time_diffs.mode()[0]
		
		if mode_diff <= pd.Timedelta(minutes=1):
			return 'T'  # Minute
		elif mode_diff <= pd.Timedelta(hours=1):
			return 'H'  # Hour
		elif mode_diff <= pd.Timedelta(days=1):
			return 'D'  # Day
		elif mode_diff <= pd.Timedelta(weeks=1):
			return 'W'  # Week
		else:
			return 'M'  # Month

class SeasonalityDetector:
	"""Detect seasonality patterns in time series"""
	
	@staticmethod
	def detect_seasonality(data: pd.Series, max_period: int = None) -> List[SeasonalityPattern]:
		"""Detect all seasonality patterns"""
		
		patterns = []
		
		if max_period is None:
			max_period = min(len(data) // 3, 365)
			
		# Common periods to check
		periods_to_check = [
			(24, SeasonalityType.DAILY),		# Daily (hourly data)
			(7, SeasonalityType.WEEKLY),		# Weekly (daily data)
			(30, SeasonalityType.MONTHLY),		# Monthly (daily data)
			(365, SeasonalityType.YEARLY)		# Yearly (daily data)
		]
		
		for period, pattern_type in periods_to_check:
			if period <= max_period and len(data) >= 2 * period:
				strength = SeasonalityDetector._calculate_seasonal_strength(data, period)
				
				if strength > 0.1:  # Threshold for significance
					phase, amplitude = SeasonalityDetector._calculate_phase_amplitude(data, period)
					confidence = min(1.0, strength * 2)  # Convert to confidence score
					
					patterns.append(SeasonalityPattern(
						pattern_type=pattern_type,
						period=period,
						strength=strength,
						phase=phase,
						amplitude=amplitude,
						confidence=confidence
					))
		
		# Sort by strength
		patterns.sort(key=lambda x: x.strength, reverse=True)
		return patterns
		
	@staticmethod
	def _calculate_seasonal_strength(data: pd.Series, period: int) -> float:
		"""Calculate strength of seasonal pattern"""
		
		if len(data) < 2 * period:
			return 0.0
			
		try:
			# Decompose time series
			decomposition = seasonal_decompose(data, model='additive', period=period)
			
			# Calculate seasonal strength
			seasonal_var = np.var(decomposition.seasonal.dropna())
			residual_var = np.var(decomposition.resid.dropna())
			
			if residual_var == 0:
				return 1.0
				
			strength = seasonal_var / (seasonal_var + residual_var)
			return min(1.0, max(0.0, strength))
			
		except Exception:
			return 0.0
			
	@staticmethod
	def _calculate_phase_amplitude(data: pd.Series, period: int) -> Tuple[float, float]:
		"""Calculate phase and amplitude of seasonal pattern"""
		
		try:
			# Group by seasonal position
			seasonal_positions = np.arange(len(data)) % period
			seasonal_means = []
			
			for pos in range(period):
				values_at_pos = data[seasonal_positions == pos]
				if len(values_at_pos) > 0:
					seasonal_means.append(values_at_pos.mean())
				else:
					seasonal_means.append(np.nan)
			
			seasonal_means = np.array(seasonal_means)
			seasonal_means = seasonal_means[~np.isnan(seasonal_means)]
			
			if len(seasonal_means) == 0:
				return 0.0, 0.0
			
			# Calculate amplitude as range
			amplitude = np.max(seasonal_means) - np.min(seasonal_means)
			
			# Calculate phase as position of maximum
			phase = np.argmax(seasonal_means) / period
			
			return phase, amplitude
			
		except Exception:
			return 0.0, 0.0

class TrendAnalyzer:
	"""Analyze trends in time series data"""
	
	@staticmethod
	def analyze_trend(data: pd.Series) -> TrendAnalysis:
		"""Comprehensive trend analysis"""
		
		# Convert index to numeric for analysis
		x = np.arange(len(data))
		y = data.values
		
		# Remove NaN values
		mask = ~np.isnan(y)
		x_clean = x[mask]
		y_clean = y[mask]
		
		if len(x_clean) < 2:
			return TrendAnalysis(
				trend_type=TrendType.NONE,
				slope=0.0,
				strength=0.0,
				change_points=[],
				trend_equation="No trend",
				r_squared=0.0
			)
		
		# Test different trend models
		models = {
			TrendType.LINEAR: TrendAnalyzer._fit_linear_trend,
			TrendType.EXPONENTIAL: TrendAnalyzer._fit_exponential_trend,
			TrendType.LOGARITHMIC: TrendAnalyzer._fit_logarithmic_trend,
			TrendType.POLYNOMIAL: TrendAnalyzer._fit_polynomial_trend
		}
		
		best_model = TrendType.NONE
		best_r2 = 0.0
		best_params = {}
		
		for trend_type, fit_func in models.items():
			try:
				r2, params = fit_func(x_clean, y_clean)
				if r2 > best_r2 and r2 > 0.1:  # Minimum threshold
					best_r2 = r2
					best_model = trend_type
					best_params = params
			except Exception:
				continue
		
		# Calculate slope and strength
		if best_model != TrendType.NONE:
			slope = best_params.get('slope', 0.0)
			strength = min(1.0, best_r2)
			equation = TrendAnalyzer._get_trend_equation(best_model, best_params)
		else:
			slope = 0.0
			strength = 0.0
			equation = "No significant trend"
		
		# Detect change points
		change_points = TrendAnalyzer._detect_change_points(data)
		
		return TrendAnalysis(
			trend_type=best_model,
			slope=slope,
			strength=strength,
			change_points=change_points,
			trend_equation=equation,
			r_squared=best_r2
		)
		
	@staticmethod
	def _fit_linear_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
		"""Fit linear trend"""
		coeffs = np.polyfit(x, y, 1)
		y_pred = np.polyval(coeffs, x)
		r2 = r2_score(y, y_pred)
		
		return r2, {'slope': coeffs[0], 'intercept': coeffs[1]}
		
	@staticmethod
	def _fit_exponential_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
		"""Fit exponential trend"""
		# Ensure positive values for log
		y_positive = np.maximum(y, 1e-10)
		
		# Fit log-linear model
		log_y = np.log(y_positive)
		coeffs = np.polyfit(x, log_y, 1)
		
		# Calculate R²
		y_pred = np.exp(np.polyval(coeffs, x))
		r2 = r2_score(y, y_pred)
		
		return r2, {'growth_rate': coeffs[0], 'initial_value': np.exp(coeffs[1])}
		
	@staticmethod
	def _fit_logarithmic_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
		"""Fit logarithmic trend"""
		# Ensure positive x values
		x_positive = np.maximum(x, 1)
		log_x = np.log(x_positive)
		
		coeffs = np.polyfit(log_x, y, 1)
		y_pred = np.polyval(coeffs, log_x)
		r2 = r2_score(y, y_pred)
		
		return r2, {'slope': coeffs[0], 'intercept': coeffs[1]}
		
	@staticmethod
	def _fit_polynomial_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
		"""Fit polynomial trend (degree 2)"""
		coeffs = np.polyfit(x, y, 2)
		y_pred = np.polyval(coeffs, x)
		r2 = r2_score(y, y_pred)
		
		return r2, {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}
		
	@staticmethod
	def _get_trend_equation(trend_type: TrendType, params: Dict) -> str:
		"""Generate trend equation string"""
		
		if trend_type == TrendType.LINEAR:
			return f"y = {params['slope']:.3f}x + {params['intercept']:.3f}"
		elif trend_type == TrendType.EXPONENTIAL:
			return f"y = {params['initial_value']:.3f} * exp({params['growth_rate']:.3f}x)"
		elif trend_type == TrendType.LOGARITHMIC:
			return f"y = {params['slope']:.3f} * log(x) + {params['intercept']:.3f}"
		elif trend_type == TrendType.POLYNOMIAL:
			return f"y = {params['a']:.3f}x² + {params['b']:.3f}x + {params['c']:.3f}"
		else:
			return "No equation"
			
	@staticmethod
	def _detect_change_points(data: pd.Series, min_segment_length: int = 10) -> List[datetime]:
		"""Detect trend change points"""
		
		change_points = []
		
		if len(data) < 2 * min_segment_length:
			return change_points
		
		# Simple change point detection using rolling regression slopes
		window_size = max(min_segment_length, len(data) // 10)
		slopes = []
		
		for i in range(window_size, len(data) - window_size):
			# Calculate slope for window before and after point i
			before_x = np.arange(window_size)
			before_y = data.iloc[i-window_size:i].values
			after_x = np.arange(window_size)
			after_y = data.iloc[i:i+window_size].values
			
			try:
				slope_before = np.polyfit(before_x, before_y, 1)[0]
				slope_after = np.polyfit(after_x, after_y, 1)[0]
				slope_change = abs(slope_after - slope_before)
				slopes.append((i, slope_change))
			except:
				slopes.append((i, 0))
		
		# Find significant slope changes
		if slopes:
			slope_values = [s[1] for s in slopes]
			threshold = np.mean(slope_values) + 2 * np.std(slope_values)
			
			for i, slope_change in slopes:
				if slope_change > threshold:
					change_points.append(data.index[i])
		
		return change_points

class AdvancedForecaster:
	"""Advanced forecasting with multiple models"""
	
	def __init__(self):
		self.fitted_models = {}
		self.model_performance = {}
		
	def fit_models(self, data: pd.Series, forecast_horizon: int = 30) -> Dict[str, Any]:
		"""Fit multiple forecasting models"""
		
		results = {}
		
		# Split data for validation
		split_point = int(len(data) * 0.8)
		train_data = data[:split_point]
		test_data = data[split_point:]
		
		# ARIMA model
		try:
			arima_result = self._fit_arima(train_data, test_data, forecast_horizon)
			results['arima'] = arima_result
		except Exception as e:
			logger.warning(f"ARIMA fitting failed: {e}")
			
		# Exponential Smoothing
		try:
			ets_result = self._fit_exponential_smoothing(train_data, test_data, forecast_horizon)
			results['exponential_smoothing'] = ets_result
		except Exception as e:
			logger.warning(f"Exponential Smoothing fitting failed: {e}")
		
		# Simple trend models
		try:
			trend_result = self._fit_trend_model(train_data, test_data, forecast_horizon)
			results['trend'] = trend_result
		except Exception as e:
			logger.warning(f"Trend model fitting failed: {e}")
		
		return results
		
	def _fit_arima(self, train_data: pd.Series, test_data: pd.Series, 
				   forecast_horizon: int) -> Dict[str, Any]:
		"""Fit ARIMA model with automatic parameter selection"""
		
		# Simple parameter selection (can be enhanced with auto_arima)
		best_aic = float('inf')
		best_params = (1, 1, 1)
		best_model = None
		
		# Try different parameter combinations
		for p in range(3):
			for d in range(2):
				for q in range(3):
					try:
						model = ARIMA(train_data, order=(p, d, q))
						fitted_model = model.fit()
						
						if fitted_model.aic < best_aic:
							best_aic = fitted_model.aic
							best_params = (p, d, q)
							best_model = fitted_model
					except:
						continue
		
		if best_model is None:
			raise ValueError("Could not fit ARIMA model")
		
		# Generate forecasts
		forecast = best_model.forecast(steps=forecast_horizon)
		forecast_ci = best_model.get_forecast(steps=forecast_horizon).conf_int()
		
		# Calculate performance on test data
		if len(test_data) > 0:
			test_forecast = best_model.forecast(steps=len(test_data))
			mae = mean_absolute_error(test_data, test_forecast)
			rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
		else:
			mae = np.nan
			rmse = np.nan
		
		return {
			'model': best_model,
			'parameters': best_params,
			'forecast': forecast.tolist(),
			'confidence_intervals': forecast_ci.values.tolist(),
			'aic': best_aic,
			'mae': mae,
			'rmse': rmse
		}
		
	def _fit_exponential_smoothing(self, train_data: pd.Series, test_data: pd.Series,
								   forecast_horizon: int) -> Dict[str, Any]:
		"""Fit Exponential Smoothing model"""
		
		# Determine seasonality
		seasonal_period = None
		if len(train_data) >= 24:
			seasonal_period = min(12, len(train_data) // 2)
		
		# Fit model
		if seasonal_period:
			model = ExponentialSmoothing(
				train_data,
				trend='add',
				seasonal='add',
				seasonal_periods=seasonal_period
			)
		else:
			model = ExponentialSmoothing(train_data, trend='add')
		
		fitted_model = model.fit()
		
		# Generate forecasts
		forecast = fitted_model.forecast(steps=forecast_horizon)
		
		# Calculate confidence intervals (simplified)
		residuals = fitted_model.resid
		residual_std = np.std(residuals)
		confidence_lower = forecast - 1.96 * residual_std
		confidence_upper = forecast + 1.96 * residual_std
		
		# Calculate performance on test data
		if len(test_data) > 0:
			test_forecast = fitted_model.forecast(steps=len(test_data))
			mae = mean_absolute_error(test_data, test_forecast)
			rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
		else:
			mae = np.nan
			rmse = np.nan
		
		return {
			'model': fitted_model,
			'forecast': forecast.tolist(),
			'confidence_lower': confidence_lower.tolist(),
			'confidence_upper': confidence_upper.tolist(),
			'mae': mae,
			'rmse': rmse
		}
		
	def _fit_trend_model(self, train_data: pd.Series, test_data: pd.Series,
						 forecast_horizon: int) -> Dict[str, Any]:
		"""Fit simple trend extrapolation model"""
		
		# Fit linear trend
		x = np.arange(len(train_data))
		y = train_data.values
		
		coeffs = np.polyfit(x, y, 1)
		
		# Generate future predictions
		future_x = np.arange(len(train_data), len(train_data) + forecast_horizon)
		forecast = np.polyval(coeffs, future_x)
		
		# Calculate prediction interval based on residuals
		train_pred = np.polyval(coeffs, x)
		residuals = y - train_pred
		residual_std = np.std(residuals)
		
		confidence_lower = forecast - 1.96 * residual_std
		confidence_upper = forecast + 1.96 * residual_std
		
		# Calculate performance on test data
		if len(test_data) > 0:
			test_x = np.arange(len(train_data), len(train_data) + len(test_data))
			test_forecast = np.polyval(coeffs, test_x)
			mae = mean_absolute_error(test_data, test_forecast)
			rmse = np.sqrt(mean_squared_error(test_data, test_forecast))
		else:
			mae = np.nan
			rmse = np.nan
		
		return {
			'slope': coeffs[0],
			'intercept': coeffs[1],
			'forecast': forecast.tolist(),
			'confidence_lower': confidence_lower.tolist(),
			'confidence_upper': confidence_upper.tolist(),
			'mae': mae,
			'rmse': rmse
		}

class RealTimeAnomalyDetector:
	"""Real-time anomaly detection for streaming data"""
	
	def __init__(self, window_size: int = 100, contamination: float = 0.1):
		self.window_size = window_size
		self.contamination = contamination
		self.data_buffer = []
		self.model = IsolationForest(contamination=contamination, random_state=42)
		self.is_trained = False
		self.baseline_stats = {}
		
	def add_data_point(self, timestamp: datetime, value: float) -> Optional[AnomalyDetection]:
		"""Add new data point and detect anomalies"""
		
		# Add to buffer
		self.data_buffer.append({'timestamp': timestamp, 'value': value})
		
		# Maintain buffer size
		if len(self.data_buffer) > self.window_size * 2:
			self.data_buffer = self.data_buffer[-self.window_size:]
		
		# Train model if we have enough data
		if len(self.data_buffer) >= self.window_size and not self.is_trained:
			self._train_model()
			self.is_trained = True
		
		# Detect anomaly if model is trained
		if self.is_trained:
			return self._detect_anomaly(timestamp, value)
		
		return None
		
	def _train_model(self):
		"""Train anomaly detection model"""
		
		values = [point['value'] for point in self.data_buffer]
		
		# Calculate baseline statistics
		self.baseline_stats = {
			'mean': np.mean(values),
			'std': np.std(values),
			'median': np.median(values),
			'q25': np.percentile(values, 25),
			'q75': np.percentile(values, 75)
		}
		
		# Prepare features for isolation forest
		features = []
		for i, point in enumerate(self.data_buffer):
			feature_vector = [
				point['value'],
				point['value'] - self.baseline_stats['mean'],  # Deviation from mean
				abs(point['value'] - self.baseline_stats['median']),  # Deviation from median
			]
			
			# Add temporal features if we have enough history
			if i >= 5:
				recent_values = [self.data_buffer[j]['value'] for j in range(i-4, i+1)]
				feature_vector.extend([
					np.mean(recent_values),
					np.std(recent_values),
					recent_values[-1] - recent_values[0]  # Recent trend
				])
			else:
				feature_vector.extend([0, 0, 0])
			
			features.append(feature_vector)
		
		# Train isolation forest
		self.model.fit(features)
		
	def _detect_anomaly(self, timestamp: datetime, value: float) -> Optional[AnomalyDetection]:
		"""Detect if current point is anomalous"""
		
		# Prepare feature vector for current point
		feature_vector = [
			value,
			value - self.baseline_stats['mean'],
			abs(value - self.baseline_stats['median']),
		]
		
		# Add temporal features
		if len(self.data_buffer) >= 5:
			recent_values = [self.data_buffer[i]['value'] for i in range(-5, 0)]
			feature_vector.extend([
				np.mean(recent_values),
				np.std(recent_values),
				recent_values[-1] - recent_values[0]
			])
		else:
			feature_vector.extend([0, 0, 0])
		
		# Predict anomaly
		prediction = self.model.predict([feature_vector])[0]
		anomaly_score = self.model.decision_function([feature_vector])[0]
		
		if prediction == -1:  # Anomaly detected
			# Determine anomaly type
			anomaly_type = self._classify_anomaly_type(value)
			
			# Calculate severity
			severity = min(1.0, abs(anomaly_score) / 0.5)
			
			# Calculate expected value
			expected_value = self.baseline_stats['mean']
			
			return AnomalyDetection(
				timestamp=timestamp,
				value=value,
				anomaly_type=anomaly_type,
				severity_score=severity,
				confidence=0.8,  # Based on isolation forest performance
				expected_value=expected_value,
				deviation=abs(value - expected_value),
				context={
					'anomaly_score': anomaly_score,
					'baseline_mean': self.baseline_stats['mean'],
					'baseline_std': self.baseline_stats['std']
				}
			)
		
		return None
		
	def _classify_anomaly_type(self, value: float) -> AnomalyType:
		"""Classify type of anomaly"""
		
		# Simple classification based on deviation magnitude
		mean_val = self.baseline_stats['mean']
		std_val = self.baseline_stats['std']
		
		deviation = abs(value - mean_val)
		
		if deviation > 3 * std_val:
			return AnomalyType.POINT
		elif deviation > 2 * std_val:
			return AnomalyType.CONTEXTUAL
		else:
			return AnomalyType.COLLECTIVE

class TimeSeriesAnalyticsEngine:
	"""Main time series analytics engine"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.preprocessor = TimeSeriesPreprocessor()
		self.seasonality_detector = SeasonalityDetector()
		self.trend_analyzer = TrendAnalyzer()
		self.forecaster = AdvancedForecaster()
		self.anomaly_detector = RealTimeAnomalyDetector()
		
		# Active analyses
		self.active_analyses: Dict[str, Dict[str, Any]] = {}
		
		logger.info("Time Series Analytics Engine initialized")
		
	async def analyze_time_series(self, twin_id: str, data: pd.DataFrame,
								 timestamp_col: str = 'timestamp',
								 value_col: str = 'value') -> Dict[str, Any]:
		"""Comprehensive time series analysis"""
		
		try:
			# Preprocess data
			clean_data = self.preprocessor.clean_data(data, timestamp_col, value_col)
			
			if len(clean_data) < 10:
				return {
					'twin_id': twin_id,
					'error': 'Insufficient data for analysis',
					'data_points': len(clean_data)
				}
			
			series = clean_data[value_col]
			
			# Basic metrics
			metrics = self._calculate_metrics(series)
			
			# Seasonality analysis
			seasonality_patterns = self.seasonality_detector.detect_seasonality(series)
			
			# Trend analysis
			trend_analysis = self.trend_analyzer.analyze_trend(series)
			
			# Forecast
			forecast_results = self.forecaster.fit_models(series)
			
			# Generate forecast points
			forecast_points = self._generate_forecast_points(
				series, forecast_results, horizon=30
			)
			
			analysis_result = {
				'twin_id': twin_id,
				'analysis_timestamp': datetime.utcnow().isoformat(),
				'data_points': len(series),
				'time_range': {
					'start': series.index.min().isoformat(),
					'end': series.index.max().isoformat(),
					'duration_days': (series.index.max() - series.index.min()).days
				},
				'metrics': asdict(metrics),
				'seasonality': [asdict(pattern) for pattern in seasonality_patterns],
				'trend': asdict(trend_analysis),
				'forecast': {
					'models': list(forecast_results.keys()),
					'best_model': self._select_best_model(forecast_results),
					'predictions': forecast_points
				},
				'recommendations': self._generate_recommendations(
					metrics, seasonality_patterns, trend_analysis
				)
			}
			
			# Store analysis
			self.active_analyses[twin_id] = analysis_result
			
			return analysis_result
			
		except Exception as e:
			logger.error(f"Error analyzing time series for twin {twin_id}: {e}")
			return {
				'twin_id': twin_id,
				'error': str(e),
				'analysis_timestamp': datetime.utcnow().isoformat()
			}
			
	def _calculate_metrics(self, series: pd.Series) -> TimeSeriesMetrics:
		"""Calculate comprehensive time series metrics"""
		
		# Stationarity test
		try:
			adf_stat, adf_pvalue, _, _, _, _ = adfuller(series.dropna())
		except:
			adf_pvalue = 1.0
		
		# Autocorrelation
		try:
			acf_values = acf(series.dropna(), nlags=1, fft=True)
			pacf_values = pacf(series.dropna(), nlags=1)
			autocorr_1 = acf_values[1] if len(acf_values) > 1 else 0
			partial_autocorr_1 = pacf_values[1] if len(pacf_values) > 1 else 0
		except:
			autocorr_1 = 0
			partial_autocorr_1 = 0
		
		# Basic statistics
		values = series.dropna()
		
		return TimeSeriesMetrics(
			stationarity_pvalue=adf_pvalue,
			trend_strength=0.0,  # Will be updated by trend analysis
			seasonal_strength=0.0,  # Will be updated by seasonality analysis
			noise_level=np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
			autocorrelation_1=autocorr_1,
			partial_autocorr_1=partial_autocorr_1,
			mean=np.mean(values),
			std=np.std(values),
			skewness=stats.skew(values),
			kurtosis=stats.kurtosis(values),
			missing_ratio=series.isna().sum() / len(series)
		)
		
	def _generate_forecast_points(self, series: pd.Series, 
								 forecast_results: Dict[str, Any],
								 horizon: int = 30) -> List[Dict[str, Any]]:
		"""Generate forecast points from best model"""
		
		best_model_name = self._select_best_model(forecast_results)
		
		if best_model_name not in forecast_results:
			return []
		
		best_result = forecast_results[best_model_name]
		forecast_values = best_result.get('forecast', [])
		
		# Generate future timestamps
		last_timestamp = series.index[-1]
		freq = pd.infer_freq(series.index) or 'D'
		future_dates = pd.date_range(
			start=last_timestamp + pd.Timedelta(freq),
			periods=len(forecast_values),
			freq=freq
		)
		
		forecast_points = []
		
		for i, (timestamp, value) in enumerate(zip(future_dates, forecast_values)):
			confidence_lower = best_result.get('confidence_lower', [None] * len(forecast_values))[i]
			confidence_upper = best_result.get('confidence_upper', [None] * len(forecast_values))[i]
			
			forecast_points.append({
				'timestamp': timestamp.isoformat(),
				'predicted_value': float(value),
				'confidence_lower': float(confidence_lower) if confidence_lower is not None else None,
				'confidence_upper': float(confidence_upper) if confidence_upper is not None else None,
				'model_used': best_model_name,
				'horizon_days': i + 1
			})
		
		return forecast_points
		
	def _select_best_model(self, forecast_results: Dict[str, Any]) -> str:
		"""Select best forecasting model based on performance"""
		
		if not forecast_results:
			return 'none'
		
		# Score models based on available metrics
		model_scores = {}
		
		for model_name, result in forecast_results.items():
			score = 0
			
			# Lower MAE is better
			mae = result.get('mae')
			if mae is not None and not np.isnan(mae):
				score += 1 / (1 + mae)
			
			# Lower RMSE is better
			rmse = result.get('rmse')
			if rmse is not None and not np.isnan(rmse):
				score += 1 / (1 + rmse)
			
			# Lower AIC is better (for ARIMA)
			aic = result.get('aic')
			if aic is not None and not np.isnan(aic):
				score += 1 / (1 + abs(aic))
			
			model_scores[model_name] = score
		
		# Return model with highest score
		if model_scores:
			return max(model_scores.items(), key=lambda x: x[1])[0]
		else:
			return list(forecast_results.keys())[0]
			
	def _generate_recommendations(self, metrics: TimeSeriesMetrics,
								 seasonality: List[SeasonalityPattern],
								 trend: TrendAnalysis) -> List[str]:
		"""Generate actionable recommendations"""
		
		recommendations = []
		
		# Data quality recommendations
		if metrics.missing_ratio > 0.05:
			recommendations.append(
				f"High missing data ratio ({metrics.missing_ratio:.1%}). "
				"Consider improving data collection reliability."
			)
		
		if metrics.noise_level > 0.5:
			recommendations.append(
				"High noise level detected. Consider data smoothing or "
				"investigating sensor calibration."
			)
		
		# Seasonality recommendations
		if seasonality:
			strongest_pattern = seasonality[0]
			recommendations.append(
				f"Strong {strongest_pattern.pattern_type.value} seasonality detected "
				f"(strength: {strongest_pattern.strength:.2f}). "
				"Consider seasonal adjustment in planning."
			)
		
		# Trend recommendations
		if trend.trend_type != TrendType.NONE and trend.strength > 0.3:
			if trend.slope > 0:
				recommendations.append(
					f"Positive {trend.trend_type.value} trend detected. "
					"Plan for continued growth in capacity/resources."
				)
			else:
				recommendations.append(
					f"Negative {trend.trend_type.value} trend detected. "
					"Investigate potential issues and corrective actions."
				)
		
		# Stationarity recommendations
		if metrics.stationarity_pvalue > 0.05:
			recommendations.append(
				"Non-stationary data detected. Consider differencing "
				"or detrending for improved forecasting accuracy."
			)
		
		return recommendations

# Test and example usage
async def test_time_series_analytics():
	"""Test the time series analytics system"""
	
	# Generate sample time series data
	np.random.seed(42)
	
	# Create dates
	dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
	
	# Generate synthetic data with trend, seasonality, and noise
	t = np.arange(len(dates))
	trend = 0.01 * t  # Linear trend
	seasonal = 5 * np.sin(2 * np.pi * t / 365.25)  # Annual seasonality
	weekly = 2 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
	noise = np.random.normal(0, 1, len(dates))
	
	values = 100 + trend + seasonal + weekly + noise
	
	# Create DataFrame
	data = pd.DataFrame({
		'timestamp': dates,
		'value': values
	})
	
	# Initialize analytics engine
	analytics_engine = TimeSeriesAnalyticsEngine()
	
	# Run analysis
	print("Running comprehensive time series analysis...")
	result = await analytics_engine.analyze_time_series('test_twin', data)
	
	print(f"\nAnalysis Results:")
	print(f"Twin ID: {result['twin_id']}")
	print(f"Data Points: {result['data_points']}")
	print(f"Time Range: {result['time_range']['duration_days']} days")
	
	print(f"\nMetrics:")
	metrics = result['metrics']
	print(f"  Mean: {metrics['mean']:.2f}")
	print(f"  Std: {metrics['std']:.2f}")
	print(f"  Stationarity p-value: {metrics['stationarity_pvalue']:.3f}")
	print(f"  Autocorrelation: {metrics['autocorrelation_1']:.3f}")
	
	print(f"\nSeasonality:")
	for pattern in result['seasonality']:
		print(f"  {pattern['pattern_type']}: strength={pattern['strength']:.2f}, "
			  f"period={pattern['period']}")
	
	print(f"\nTrend:")
	trend = result['trend']
	print(f"  Type: {trend['trend_type']}")
	print(f"  Strength: {trend['strength']:.2f}")
	print(f"  Equation: {trend['trend_equation']}")
	
	print(f"\nForecast:")
	forecast = result['forecast']
	print(f"  Best Model: {forecast['best_model']}")
	print(f"  Forecast Points: {len(forecast['predictions'])}")
	
	print(f"\nRecommendations:")
	for rec in result['recommendations']:
		print(f"  - {rec}")

if __name__ == "__main__":
	asyncio.run(test_time_series_analytics())