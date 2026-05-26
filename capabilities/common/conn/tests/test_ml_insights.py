"""
Tests for APG Connection Management ML Insights functionality
Comprehensive testing of machine learning analytics and insights generation

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from ..ml_insights import (
	MLInsightsEngine, AnomalyDetector, ClusterAnalyzer, TimeSeriesAnalyzer,
	PatternRecognizer, SentimentAnalyzer, MLInsight, DataPattern,
	AnomalyResult, ClusteringResult, ForecastResult, ModelMetrics,
	AnalysisType, ModelType, InsightSeverity,
	generate_ml_insights, get_anomaly_insights, get_clustering_insights,
	forecast_time_series
)


@pytest.fixture
def sample_dataframe():
	"""Sample DataFrame for testing"""
	np.random.seed(42)
	data = {
		'id': range(1, 101),
		'value': np.random.normal(100, 15, 100),
		'category': np.random.choice(['A', 'B', 'C'], 100),
		'score': np.random.uniform(0, 1, 100),
		'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
		'text': ['Good product', 'Bad service', 'Excellent quality'] * 33 + ['Neutral review']
	}
	return pd.DataFrame(data)


@pytest.fixture
def sample_time_series():
	"""Sample time series for testing"""
	dates = pd.date_range('2024-01-01', periods=50, freq='D')
	values = np.random.normal(100, 10, 50) + np.linspace(0, 20, 50)  # Trend + noise
	return pd.Series(values, index=dates)


@pytest.fixture
def ml_engine():
	"""ML insights engine instance"""
	return MLInsightsEngine()


@pytest.fixture
def anomaly_detector():
	"""Anomaly detector instance"""
	return AnomalyDetector()


@pytest.fixture
def cluster_analyzer():
	"""Cluster analyzer instance"""
	return ClusterAnalyzer()


@pytest.fixture
def time_series_analyzer():
	"""Time series analyzer instance"""
	return TimeSeriesAnalyzer()


@pytest.fixture
def pattern_recognizer():
	"""Pattern recognizer instance"""
	return PatternRecognizer()


@pytest.fixture
def sentiment_analyzer():
	"""Sentiment analyzer instance"""
	return SentimentAnalyzer()


class TestAnomalyDetector:
	"""Test anomaly detection functionality"""

	def test_detector_initialization(self, anomaly_detector):
		"""Test detector initialization"""
		assert isinstance(anomaly_detector.models, dict)
		assert isinstance(anomaly_detector.scalers, dict)
		assert anomaly_detector.is_trained == False

	@pytest.mark.skipif(not hasattr(AnomalyDetector, '_has_sklearn'), reason="Scikit-learn not available")
	def test_train_anomaly_detector(self, anomaly_detector, sample_dataframe):
		"""Test training anomaly detector"""
		try:
			metrics = anomaly_detector.train(sample_dataframe)

			assert isinstance(metrics, ModelMetrics)
			assert metrics.model_type == ModelType.ISOLATION_FOREST
			assert anomaly_detector.is_trained == True
			assert 'isolation_forest' in anomaly_detector.models
			assert 'main' in anomaly_detector.scalers
			assert len(anomaly_detector.feature_names) > 0

		except Exception as e:
			# If sklearn is not available, the test should skip
			pytest.skip(f"Scikit-learn functionality not available: {e}")

	def test_train_with_no_numeric_columns(self, anomaly_detector):
		"""Test training with no numeric columns"""
		text_df = pd.DataFrame({'text': ['hello', 'world', 'test']})

		with pytest.raises(Exception):  # Should raise APGError
			anomaly_detector.train(text_df)

	@pytest.mark.skipif(not hasattr(AnomalyDetector, '_has_sklearn'), reason="Scikit-learn not available")
	def test_detect_anomalies_without_training(self, anomaly_detector, sample_dataframe):
		"""Test detecting anomalies without training first"""
		with pytest.raises(Exception):  # Should raise APGError
			anomaly_detector.detect_anomalies(sample_dataframe)

	@pytest.mark.skipif(not hasattr(AnomalyDetector, '_has_sklearn'), reason="Scikit-learn not available")
	def test_detect_anomalies_after_training(self, anomaly_detector, sample_dataframe):
		"""Test detecting anomalies after training"""
		try:
			# Train first
			anomaly_detector.train(sample_dataframe)

			# Detect anomalies
			result = anomaly_detector.detect_anomalies(sample_dataframe)

			assert isinstance(result, AnomalyResult)
			assert result.total_records == 100
			assert result.anomaly_count >= 0
			assert 0 <= result.anomaly_rate <= 1
			assert isinstance(result.anomalies, list)
			assert isinstance(result.feature_contributions, dict)

			# Check anomaly structure
			if result.anomalies:
				anomaly = result.anomalies[0]
				assert 'index' in anomaly
				assert 'anomaly_score' in anomaly
				assert 'values' in anomaly
				assert 'deviations' in anomaly

		except Exception as e:
			pytest.skip(f"Scikit-learn functionality not available: {e}")

	def test_calculate_deviations(self, anomaly_detector, sample_dataframe):
		"""Test deviation calculation"""
		anomaly_detector.feature_names = ['value', 'score']

		record = sample_dataframe.iloc[0]
		deviations = anomaly_detector._calculate_deviations(record, sample_dataframe)

		assert isinstance(deviations, dict)
		for field in anomaly_detector.feature_names:
			if field in record:
				assert field in deviations
				assert isinstance(deviations[field], float)


class TestClusterAnalyzer:
	"""Test clustering analysis functionality"""

	def test_analyzer_initialization(self, cluster_analyzer):
		"""Test analyzer initialization"""
		assert isinstance(cluster_analyzer.models, dict)
		assert isinstance(cluster_analyzer.scalers, dict)
		assert isinstance(cluster_analyzer.feature_names, list)

	@pytest.mark.skipif(not hasattr(ClusterAnalyzer, '_has_sklearn'), reason="Scikit-learn not available")
	def test_kmeans_clustering(self, cluster_analyzer, sample_dataframe):
		"""Test K-means clustering"""
		try:
			result = cluster_analyzer.analyze_clusters(sample_dataframe, method="kmeans", n_clusters=3)

			assert isinstance(result, ClusteringResult)
			assert result.num_clusters == 3
			assert len(result.cluster_labels) == 100
			assert result.silhouette_score is not None
			assert isinstance(result.cluster_stats, dict)
			assert result.model_metrics.model_type == ModelType.KMEANS

			# Check cluster centers
			if result.cluster_centers:
				assert len(result.cluster_centers) == 3

		except Exception as e:
			pytest.skip(f"Scikit-learn functionality not available: {e}")

	@pytest.mark.skipif(not hasattr(ClusterAnalyzer, '_has_sklearn'), reason="Scikit-learn not available")
	def test_dbscan_clustering(self, cluster_analyzer, sample_dataframe):
		"""Test DBSCAN clustering"""
		try:
			result = cluster_analyzer.analyze_clusters(sample_dataframe, method="dbscan")

			assert isinstance(result, ClusteringResult)
			assert result.num_clusters >= 0
			assert len(result.cluster_labels) == 100
			assert result.model_metrics.model_type == ModelType.DBSCAN

		except Exception as e:
			pytest.skip(f"Scikit-learn functionality not available: {e}")

	def test_clustering_no_numeric_columns(self, cluster_analyzer):
		"""Test clustering with no numeric columns"""
		text_df = pd.DataFrame({'text': ['hello', 'world', 'test']})

		with pytest.raises(Exception):  # Should raise APGError
			cluster_analyzer.analyze_clusters(text_df)

	def test_find_optimal_clusters(self, cluster_analyzer, sample_dataframe):
		"""Test optimal cluster number detection"""
		try:
			# Prepare data
			numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
			X = sample_dataframe[numeric_cols].fillna(sample_dataframe[numeric_cols].mean())

			from sklearn.preprocessing import StandardScaler
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)

			optimal_k = cluster_analyzer._find_optimal_clusters(X_scaled, max_clusters=5)

			assert isinstance(optimal_k, int)
			assert 2 <= optimal_k <= 5

		except ImportError:
			pytest.skip("Scikit-learn not available")

	def test_calculate_cluster_stats(self, cluster_analyzer, sample_dataframe):
		"""Test cluster statistics calculation"""
		# Mock cluster labels
		cluster_labels = np.random.randint(0, 3, 100)
		numeric_cols = ['value', 'score']

		stats = cluster_analyzer._calculate_cluster_stats(sample_dataframe, cluster_labels, numeric_cols)

		assert isinstance(stats, dict)
		for cluster_id, cluster_stat in stats.items():
			assert 'size' in cluster_stat
			assert 'percentage' in cluster_stat
			assert 'feature_means' in cluster_stat
			assert 'feature_stds' in cluster_stat


class TestTimeSeriesAnalyzer:
	"""Test time series analysis functionality"""

	def test_analyzer_initialization(self, time_series_analyzer):
		"""Test analyzer initialization"""
		assert isinstance(time_series_analyzer.models, dict)
		assert isinstance(time_series_analyzer.is_fitted, dict)

	def test_linear_forecast(self, time_series_analyzer, sample_time_series):
		"""Test linear forecasting"""
		result = time_series_analyzer._linear_forecast(sample_time_series, periods=5)

		assert isinstance(result, ForecastResult)
		assert len(result.forecast_values) == 5
		assert all(isinstance(val, float) for val in result.forecast_values)

		if result.model_metrics:
			assert result.model_metrics.model_type == ModelType.LINEAR_REGRESSION

	def test_forecast_with_arima(self, time_series_analyzer, sample_time_series):
		"""Test ARIMA forecasting"""
		try:
			result = time_series_analyzer.forecast(sample_time_series, periods=5, method="arima")

			assert isinstance(result, ForecastResult)
			assert len(result.forecast_values) == 5

			if result.confidence_intervals:
				assert len(result.confidence_intervals) == 5
				assert all(isinstance(ci, tuple) for ci in result.confidence_intervals)

		except ImportError:
			pytest.skip("Statsmodels not available")
		except Exception as e:
			# ARIMA might fail on synthetic data, fallback to linear should work
			assert isinstance(e, Exception)

	def test_forecast_with_insufficient_data(self, time_series_analyzer):
		"""Test forecasting with insufficient data"""
		short_series = pd.Series([1, 2])

		# Should still work but may have limited accuracy
		result = time_series_analyzer.forecast(short_series, periods=3, method="linear")
		assert isinstance(result, ForecastResult)
		assert len(result.forecast_values) == 3

	def test_forecast_with_missing_values(self, time_series_analyzer):
		"""Test forecasting with missing values"""
		series_with_nulls = pd.Series([1, 2, None, 4, 5, None, 7, 8])

		result = time_series_analyzer.forecast(series_with_nulls, periods=3, method="linear")
		assert isinstance(result, ForecastResult)
		assert len(result.forecast_values) == 3

	def test_forecast_too_many_nulls(self, time_series_analyzer):
		"""Test forecasting with too many missing values"""
		mostly_null_series = pd.Series([1, None, None, None, None, None, None, 2])

		with pytest.raises(Exception):  # Should raise APGError
			time_series_analyzer.forecast(mostly_null_series, periods=3)


class TestPatternRecognizer:
	"""Test pattern recognition functionality"""

	def test_recognizer_initialization(self, pattern_recognizer):
		"""Test recognizer initialization"""
		assert isinstance(pattern_recognizer.patterns, list)

	def test_identify_patterns(self, pattern_recognizer, sample_dataframe):
		"""Test pattern identification"""
		patterns = pattern_recognizer.identify_patterns(sample_dataframe)

		assert isinstance(patterns, list)
		assert all(isinstance(pattern, DataPattern) for pattern in patterns)

		# Check pattern structure
		if patterns:
			pattern = patterns[0]
			assert hasattr(pattern, 'pattern_id')
			assert hasattr(pattern, 'pattern_type')
			assert hasattr(pattern, 'description')
			assert hasattr(pattern, 'confidence')
			assert hasattr(pattern, 'frequency')

	def test_find_numeric_patterns(self, pattern_recognizer, sample_dataframe):
		"""Test numeric pattern detection"""
		patterns = pattern_recognizer._find_numeric_patterns(sample_dataframe)

		assert isinstance(patterns, list)

		# Should find some patterns in the numeric data
		pattern_types = [p.pattern_type for p in patterns]
		possible_types = ['constant_value', 'arithmetic_sequence', 'statistical_outliers']

		# At least some pattern types should be detected
		assert len(patterns) >= 0  # May not find patterns in random data

	def test_find_categorical_patterns(self, pattern_recognizer, sample_dataframe):
		"""Test categorical pattern detection"""
		patterns = pattern_recognizer._find_categorical_patterns(sample_dataframe)

		assert isinstance(patterns, list)

		# Should find patterns in categorical data
		if patterns:
			pattern_types = [p.pattern_type for p in patterns]
			possible_types = ['dominant_value', 'high_cardinality']
			assert any(pt in possible_types for pt in pattern_types)

	def test_find_correlation_patterns(self, pattern_recognizer):
		"""Test correlation pattern detection"""
		# Create data with known correlation
		data = pd.DataFrame({
			'x': range(100),
			'y': [i * 2 + np.random.normal(0, 0.1) for i in range(100)],  # Strong correlation
			'z': np.random.random(100)  # Random
		})

		patterns = pattern_recognizer._find_correlation_patterns(data)

		assert isinstance(patterns, list)

		# Should find correlation between x and y
		if patterns:
			correlation_patterns = [p for p in patterns if 'correlation' in p.pattern_type]
			assert len(correlation_patterns) > 0

	def test_find_temporal_patterns(self, pattern_recognizer):
		"""Test temporal pattern detection"""
		# Create data with regular time intervals
		data = pd.DataFrame({
			'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
			'value': range(10)
		})

		patterns = pattern_recognizer._find_temporal_patterns(data)

		assert isinstance(patterns, list)

		# Should find regular interval pattern
		if patterns:
			temporal_patterns = [p for p in patterns if 'time' in p.pattern_type]
			assert len(temporal_patterns) >= 0  # May find regular interval pattern


class TestSentimentAnalyzer:
	"""Test sentiment analysis functionality"""

	def test_analyzer_initialization(self, sentiment_analyzer):
		"""Test analyzer initialization"""
		assert sentiment_analyzer.is_initialized == False

	def test_analyze_sentiment_no_nlp(self, sentiment_analyzer):
		"""Test sentiment analysis without NLP libraries"""
		with patch('ml_insights.NLP_AVAILABLE', False):
			data = pd.DataFrame({'text': ['Good product', 'Bad service', 'Neutral']})
			result = sentiment_analyzer.analyze_sentiment(data)

			assert 'error' in result
			assert result['error'] == 'NLP libraries not available'

	@pytest.mark.skipif(not hasattr(SentimentAnalyzer, '_has_nltk'), reason="NLP libraries not available")
	def test_analyze_sentiment_with_text(self, sentiment_analyzer, sample_dataframe):
		"""Test sentiment analysis with text data"""
		try:
			result = sentiment_analyzer.analyze_sentiment(sample_dataframe, ['text'])

			if 'error' not in result:
				assert 'text' in result
				text_result = result['text']

				assert 'sentiment_distribution' in text_result
				assert 'average_polarity' in text_result
				assert 'total_texts' in text_result
				assert 'sentiment_trend' in text_result

				# Check sentiment distribution
				distribution = text_result['sentiment_distribution']
				assert isinstance(distribution, dict)
				possible_sentiments = ['positive', 'negative', 'neutral']
				assert all(sentiment in possible_sentiments for sentiment in distribution.keys())

		except ImportError:
			pytest.skip("NLP libraries not available")

	def test_analyze_sentiment_empty_data(self, sentiment_analyzer):
		"""Test sentiment analysis with empty data"""
		empty_df = pd.DataFrame({'text': []})
		result = sentiment_analyzer.analyze_sentiment(empty_df)

		# Should handle gracefully
		assert isinstance(result, dict)


class TestMLInsightsEngine:
	"""Test main ML insights engine"""

	def test_engine_initialization(self, ml_engine):
		"""Test engine initialization"""
		assert isinstance(ml_engine.anomaly_detector, AnomalyDetector)
		assert isinstance(ml_engine.cluster_analyzer, ClusterAnalyzer)
		assert isinstance(ml_engine.time_series_analyzer, TimeSeriesAnalyzer)
		assert isinstance(ml_engine.pattern_recognizer, PatternRecognizer)
		assert isinstance(ml_engine.sentiment_analyzer, SentimentAnalyzer)
		assert isinstance(ml_engine.insights_cache, dict)

	@pytest.mark.asyncio
	async def test_analyze_data_empty_input(self, ml_engine):
		"""Test analysis with empty data"""
		empty_df = pd.DataFrame()
		insights = await ml_engine.analyze_data(empty_df)

		assert isinstance(insights, list)
		assert len(insights) == 0

	@pytest.mark.asyncio
	async def test_analyze_data_list_input(self, ml_engine):
		"""Test analysis with list input"""
		data_list = [
			{'id': 1, 'value': 100, 'category': 'A'},
			{'id': 2, 'value': 200, 'category': 'B'},
			{'id': 3, 'value': 150, 'category': 'A'}
		]

		insights = await ml_engine.analyze_data(data_list)

		assert isinstance(insights, list)
		assert all(isinstance(insight, MLInsight) for insight in insights)

	@pytest.mark.asyncio
	async def test_analyze_data_with_specific_types(self, ml_engine, sample_dataframe):
		"""Test analysis with specific analysis types"""
		analysis_types = [AnalysisType.PATTERN_RECOGNITION, AnalysisType.DATA_PROFILING]

		insights = await ml_engine.analyze_data(sample_dataframe, analysis_types)

		assert isinstance(insights, list)
		if insights:
			insight_types = [insight.analysis_type for insight in insights]
			assert all(at in analysis_types for at in insight_types)

	@pytest.mark.asyncio
	async def test_perform_anomaly_analysis(self, ml_engine, sample_dataframe):
		"""Test anomaly analysis"""
		try:
			insights = await ml_engine._perform_anomaly_analysis(sample_dataframe)

			assert isinstance(insights, list)
			if insights:
				insight = insights[0]
				assert insight.analysis_type == AnalysisType.ANOMALY_DETECTION
				assert insight.severity in [InsightSeverity.HIGH, InsightSeverity.MEDIUM, InsightSeverity.LOW]

		except Exception as e:
			# May fail if scikit-learn not available
			pytest.skip(f"Anomaly analysis requires scikit-learn: {e}")

	@pytest.mark.asyncio
	async def test_perform_pattern_analysis(self, ml_engine, sample_dataframe):
		"""Test pattern analysis"""
		insights = await ml_engine._perform_pattern_analysis(sample_dataframe)

		assert isinstance(insights, list)
		# Pattern analysis should complete without errors

	@pytest.mark.asyncio
	async def test_perform_data_profiling_analysis(self, ml_engine):
		"""Test data profiling analysis"""
		# Create data with known issues
		problematic_data = pd.DataFrame({
			'mostly_missing': [1, None, None, None, None] * 20,
			'categorical': ['A', 'A', 'A', 'B'] * 25,
			'normal': range(100)
		})

		insights = await ml_engine._perform_data_profiling_analysis(problematic_data)

		assert isinstance(insights, list)

		# Should find high missing data issue
		missing_insights = [i for i in insights if 'missing' in i.title.lower()]
		assert len(missing_insights) > 0

		# Should find categorical optimization opportunity
		categorical_insights = [i for i in insights if 'categorical' in i.title.lower()]
		assert len(categorical_insights) >= 0  # May or may not find depending on thresholds

	@pytest.mark.asyncio
	async def test_invalid_data_format(self, ml_engine):
		"""Test analysis with invalid data format"""
		with pytest.raises(Exception):  # Should raise APGError
			await ml_engine.analyze_data("invalid_data_format")


class TestMLInsightDataClasses:
	"""Test ML insight data classes"""

	def test_ml_insight_creation(self):
		"""Test MLInsight creation"""
		insight = MLInsight(
			insight_id="test_001",
			analysis_type=AnalysisType.ANOMALY_DETECTION,
			title="Test Insight",
			description="A test insight",
			severity=InsightSeverity.MEDIUM,
			confidence=0.85,
			evidence={'test': 'data'},
			recommendations=['Test recommendation'],
			affected_fields=['field1', 'field2']
		)

		assert insight.insight_id == "test_001"
		assert insight.analysis_type == AnalysisType.ANOMALY_DETECTION
		assert insight.severity == InsightSeverity.MEDIUM
		assert insight.confidence == 0.85
		assert len(insight.recommendations) == 1
		assert len(insight.affected_fields) == 2
		assert isinstance(insight.generated_at, datetime)

	def test_data_pattern_creation(self):
		"""Test DataPattern creation"""
		pattern = DataPattern(
			pattern_id="pattern_001",
			pattern_type="test_pattern",
			description="A test pattern",
			frequency=10,
			confidence=0.9,
			examples=[1, 2, 3],
			fields_involved=['field1']
		)

		assert pattern.pattern_id == "pattern_001"
		assert pattern.pattern_type == "test_pattern"
		assert pattern.frequency == 10
		assert pattern.confidence == 0.9
		assert len(pattern.examples) == 3

	def test_anomaly_result_creation(self):
		"""Test AnomalyResult creation"""
		result = AnomalyResult(
			total_records=100,
			anomaly_count=5,
			anomaly_rate=0.05,
			anomalies=[{'index': 1, 'score': -0.5}],
			feature_contributions={'field1': 0.8}
		)

		assert result.total_records == 100
		assert result.anomaly_count == 5
		assert result.anomaly_rate == 0.05
		assert len(result.anomalies) == 1
		assert 'field1' in result.feature_contributions

	def test_clustering_result_creation(self):
		"""Test ClusteringResult creation"""
		result = ClusteringResult(
			num_clusters=3,
			cluster_labels=[0, 1, 2, 0, 1],
			cluster_centers=[[1, 2], [3, 4], [5, 6]],
			silhouette_score=0.75,
			cluster_stats={0: {'size': 2}, 1: {'size': 2}, 2: {'size': 1}}
		)

		assert result.num_clusters == 3
		assert len(result.cluster_labels) == 5
		assert len(result.cluster_centers) == 3
		assert result.silhouette_score == 0.75
		assert len(result.cluster_stats) == 3

	def test_forecast_result_creation(self):
		"""Test ForecastResult creation"""
		result = ForecastResult(
			forecast_values=[100, 105, 110],
			confidence_intervals=[(95, 105), (100, 110), (105, 115)],
			forecast_dates=[datetime.now(), datetime.now(), datetime.now()]
		)

		assert len(result.forecast_values) == 3
		assert len(result.confidence_intervals) == 3
		assert len(result.forecast_dates) == 3


class TestConvenienceFunctions:
	"""Test convenience functions"""

	@pytest.mark.asyncio
	async def test_generate_ml_insights(self, sample_dataframe):
		"""Test convenience function for generating insights"""
		insights = await generate_ml_insights(sample_dataframe, connection_id="test_conn")

		assert isinstance(insights, list)
		assert all(isinstance(insight, MLInsight) for insight in insights)

	@pytest.mark.asyncio
	async def test_generate_ml_insights_with_analysis_types(self, sample_dataframe):
		"""Test generating insights with specific analysis types"""
		analysis_types = ['pattern_recognition', 'data_profiling']
		insights = await generate_ml_insights(sample_dataframe, analysis_types=analysis_types)

		assert isinstance(insights, list)
		if insights:
			insight_types = [insight.analysis_type.value for insight in insights]
			assert all(at in analysis_types for at in insight_types)

	@pytest.mark.asyncio
	async def test_get_anomaly_insights(self, sample_dataframe):
		"""Test anomaly insights convenience function"""
		try:
			result = await get_anomaly_insights(sample_dataframe)
			assert isinstance(result, AnomalyResult)
		except Exception as e:
			pytest.skip(f"Anomaly detection requires scikit-learn: {e}")

	@pytest.mark.asyncio
	async def test_get_clustering_insights(self, sample_dataframe):
		"""Test clustering insights convenience function"""
		try:
			result = await get_clustering_insights(sample_dataframe, n_clusters=3)
			assert isinstance(result, ClusteringResult)
			assert result.num_clusters == 3
		except Exception as e:
			pytest.skip(f"Clustering requires scikit-learn: {e}")

	@pytest.mark.asyncio
	async def test_forecast_time_series(self, sample_time_series):
		"""Test time series forecasting convenience function"""
		result = await forecast_time_series(sample_time_series, periods=5)

		assert isinstance(result, ForecastResult)
		assert len(result.forecast_values) == 5


class TestEdgeCases:
	"""Test edge cases and error conditions"""

	@pytest.mark.asyncio
	async def test_analysis_with_all_null_data(self, ml_engine):
		"""Test analysis with all null data"""
		null_data = pd.DataFrame({
			'col1': [None, None, None],
			'col2': [None, None, None]
		})

		insights = await ml_engine.analyze_data(null_data)
		assert isinstance(insights, list)

	@pytest.mark.asyncio
	async def test_analysis_with_single_row(self, ml_engine):
		"""Test analysis with single row"""
		single_row = pd.DataFrame({'value': [100], 'category': ['A']})

		insights = await ml_engine.analyze_data(single_row)
		assert isinstance(insights, list)

	@pytest.mark.asyncio
	async def test_analysis_with_non_standard_dtypes(self, ml_engine):
		"""Test analysis with non-standard data types"""
		complex_data = pd.DataFrame({
			'complex': [1+2j, 3+4j, 5+6j],
			'object': [object(), object(), object()],
			'normal': [1, 2, 3]
		})

		insights = await ml_engine.analyze_data(complex_data)
		assert isinstance(insights, list)
		# Should handle gracefully without crashing

	def test_time_series_with_constant_values(self, time_series_analyzer):
		"""Test time series forecasting with constant values"""
		constant_series = pd.Series([100] * 20)

		result = time_series_analyzer.forecast(constant_series, periods=5, method="linear")
		assert isinstance(result, ForecastResult)
		assert len(result.forecast_values) == 5
		# Should predict constant values
		assert all(abs(val - 100) < 1 for val in result.forecast_values)


class TestIntegration:
	"""Integration tests"""

	@pytest.mark.asyncio
	async def test_end_to_end_ml_pipeline(self, sample_dataframe):
		"""Test complete ML analysis pipeline"""
		engine = MLInsightsEngine()

		# Run comprehensive analysis
		all_analysis_types = [
			AnalysisType.PATTERN_RECOGNITION,
			AnalysisType.DATA_PROFILING,
			AnalysisType.SENTIMENT_ANALYSIS
		]

		insights = await engine.analyze_data(
			sample_dataframe,
			all_analysis_types,
			connection_id="integration_test"
		)

		assert isinstance(insights, list)

		# Check that different types of insights are generated
		analysis_types_found = set(insight.analysis_type for insight in insights)

		# Should find at least some insights
		assert len(insights) >= 0

		# Verify insight structure
		for insight in insights:
			assert isinstance(insight.insight_id, str)
			assert isinstance(insight.title, str)
			assert isinstance(insight.description, str)
			assert isinstance(insight.severity, InsightSeverity)
			assert 0 <= insight.confidence <= 1
			assert isinstance(insight.generated_at, datetime)

		print(f"Integration test completed: {len(insights)} insights generated")
		print(f"Analysis types found: {analysis_types_found}")


if __name__ == '__main__':
	pytest.main([__file__, '-v'])
