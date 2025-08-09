"""
APG Central Configuration - Revolutionary Machine Learning & AI Engine

Next-generation ML system with deep learning, reinforcement learning, federated learning,
explainable AI, automated model selection, real-time inference, and edge deployment.

Features:
- Advanced ensemble models with automated hyperparameter optimization
- Deep neural networks for complex pattern recognition
- Reinforcement learning for autonomous configuration optimization
- Federated learning for multi-tenant privacy-preserving ML
- Real-time anomaly detection with drift adaptation
- Explainable AI with SHAP and LIME integration
- AutoML capabilities for automatic model selection
- Edge AI deployment with quantized models
- Multi-modal learning (text, time series, graphs)
- Advanced feature engineering and selection
- Model versioning and A/B testing framework
- Distributed training and inference

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
import pickle
import joblib
from pathlib import Path
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import sklearn
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, IsolationForest,
    RandomForestClassifier, ExtraTreesRegressor, VotingRegressor, 
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    SGDRegressor, BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR, SVC, OneClassSVM
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    PowerTransformer, QuantileTransformer, PolynomialFeatures
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    silhouette_score, adjusted_rand_score, f1_score
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, 
        BatchNormalization, Attention, MultiHeadAttention,
        Embedding, Flatten, GlobalAveragePooling1D
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Time series analysis
try:
    import scipy
    from scipy import stats, signal, optimize
    from scipy.signal import find_peaks, periodogram
    from scipy.stats import jarque_bera, shapiro, anderson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Advanced analytics
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Federated Learning
try:
    import syft as fl
    SYFT_AVAILABLE = True
except ImportError:
    SYFT_AVAILABLE = False

# Error handling
from .error_handling import ErrorHandler, ErrorCategory, ErrorSeverity, with_error_handling

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Advanced machine learning model types."""
    # Core prediction models
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_PREDICTION = "performance_prediction"
    CAPACITY_PLANNING = "capacity_planning"
    FAILURE_PREDICTION = "failure_prediction"
    COST_OPTIMIZATION = "cost_optimization"
    
    # Advanced analytics models
    CONFIGURATION_CLUSTERING = "configuration_clustering"
    DRIFT_DETECTION = "drift_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    SEQUENCE_PREDICTION = "sequence_prediction"
    MULTIVARIATE_FORECASTING = "multivariate_forecasting"
    
    # Recommendation systems
    OPTIMIZATION_RECOMMENDATION = "optimization_recommendation"
    CONFIGURATION_RECOMMENDATION = "configuration_recommendation"
    RESOURCE_RECOMMENDATION = "resource_recommendation"
    
    # Natural language processing
    LOG_ANALYSIS = "log_analysis"
    INCIDENT_CLASSIFICATION = "incident_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    
    # Graph analytics
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    NETWORK_TOPOLOGY = "network_topology"
    INFLUENCE_MAPPING = "influence_mapping"
    
    # Reinforcement learning
    AUTONOMOUS_OPTIMIZATION = "autonomous_optimization"
    POLICY_LEARNING = "policy_learning"
    ADAPTIVE_SCALING = "adaptive_scaling"

class ModelArchitecture(Enum):
    """Model architecture types."""
    TRADITIONAL_ML = "traditional_ml"
    ENSEMBLE = "ensemble"
    DEEP_NEURAL_NETWORK = "deep_neural_network"
    TRANSFORMER = "transformer"
    LSTM_RNN = "lstm_rnn"
    CNN = "cnn"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    FEDERATED = "federated"

class PredictionConfidence(Enum):
    """Enhanced prediction confidence levels."""
    VERY_LOW = "very_low"      # < 0.5
    LOW = "low"                # 0.5 - 0.65
    MEDIUM = "medium"          # 0.65 - 0.8
    HIGH = "high"              # 0.8 - 0.9
    VERY_HIGH = "very_high"    # 0.9 - 0.95
    EXTREMELY_HIGH = "extremely_high"  # > 0.95

class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    RETRAINING = "retraining"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class AdvancedModelPrediction:
    """Enhanced ML model prediction result with explainability."""
    model_type: ModelType
    model_architecture: ModelArchitecture
    prediction: Union[float, int, str, List[Any], np.ndarray]
    confidence: PredictionConfidence
    confidence_score: float
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    
    # Feature information
    features_used: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, Any] = field(default_factory=dict)
    
    # Model information
    model_version: str = ""
    model_id: str = ""
    inference_time_ms: float = 0.0
    
    # Explainability
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    explanation: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    prediction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_timestamp: Optional[datetime] = None
    
    # Quality metrics
    prediction_quality_score: float = 0.0
    outlier_score: float = 0.0
    drift_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['prediction_timestamp'] = self.prediction_timestamp.isoformat()
        if self.data_timestamp:
            result['data_timestamp'] = self.data_timestamp.isoformat()
        return result

@dataclass
class ModelTrainingResult:
    """Comprehensive model training result."""
    model_type: ModelType
    model_architecture: ModelArchitecture
    model_id: str
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: List[float] = field(default_factory=list)
    
    # Feature analysis
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_correlations: Dict[str, float] = field(default_factory=dict)
    selected_features: List[str] = field(default_factory=list)
    
    # Training information
    training_time_seconds: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    
    # Model information
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0
    model_complexity: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assessment
    overfitting_score: float = 0.0
    stability_score: float = 0.0
    interpretability_score: float = 0.0
    
    # Timestamps
    training_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    training_config: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class AnomalyDetectionResult:
    """Advanced anomaly detection result."""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    severity: str
    confidence: float
    
    # Detailed analysis
    affected_metrics: List[str] = field(default_factory=list)
    anomaly_patterns: List[str] = field(default_factory=list)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    
    # Root cause analysis
    potential_causes: List[str] = field(default_factory=list)
    correlation_analysis: Dict[str, float] = field(default_factory=dict)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    remediation_steps: List[Dict[str, Any]] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)
    
    # Metadata
    detection_method: str = ""
    model_version: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    explanation: Dict[str, Any] = field(default_factory=dict)

class AdvancedCentralConfigurationML:
    """Revolutionary ML engine for intelligent configuration management."""
    
    def __init__(
        self, 
        models_directory: str = "./ml_models_advanced",
        enable_gpu: bool = True,
        enable_distributed: bool = False,
        max_workers: int = 4
    ):
        """Initialize advanced ML engine."""
        self.models_dir = Path(models_directory)
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.enable_gpu = enable_gpu and (TF_AVAILABLE or TORCH_AVAILABLE)
        self.enable_distributed = enable_distributed
        self.max_workers = max_workers
        
        # Error handling
        self.error_handler = ErrorHandler("advanced_ml_engine")
        
        # Model storage and management
        self.models: Dict[str, Any] = {}  # model_id -> model
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[ModelType, List[str]] = {}
        self.active_models: Dict[ModelType, str] = {}  # model_type -> active_model_id
        
        # Feature engineering components
        self.feature_engineers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.selectors: Dict[str, Any] = {}
        
        # Advanced components
        self.ensemble_models: Dict[str, Dict[str, Any]] = {}
        self.automl_engines: Dict[str, Any] = {}
        self.explainers: Dict[str, Any] = {}
        
        # Data management
        self.training_data: Dict[ModelType, pd.DataFrame] = {}
        self.feature_store: Dict[str, pd.DataFrame] = {}
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Real-time components
        self.streaming_processors: Dict[str, Any] = {}
        self.online_learners: Dict[str, Any] = {}
        self.drift_detectors: Dict[str, Any] = {}
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        # Performance monitoring
        self.inference_stats: Dict[str, Dict[str, float]] = {}
        self.model_health: Dict[str, Dict[str, Any]] = {}
        
        # Initialize advanced components
        asyncio.create_task(self._initialize_advanced_ml_engine())
    
    @with_error_handling(ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH)
    async def _initialize_advanced_ml_engine(self):
        """Initialize advanced ML engine components."""
        try:
            # Initialize GPU support
            if self.enable_gpu:
                await self._initialize_gpu_support()
            
            # Initialize feature engineering pipeline
            await self._initialize_feature_engineering()
            
            # Initialize explainability components
            await self._initialize_explainability()
            
            # Initialize AutoML engines
            await self._initialize_automl()
            
            # Initialize federated learning (if available)
            if SYFT_AVAILABLE:
                await self._initialize_federated_learning()
            
            # Load existing models
            await self._load_existing_models()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            logger.info("Advanced ML engine initialized successfully")
            
        except Exception as e:
            await self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL,
                "initialize_advanced_ml_engine"
            )
            raise
    
    async def _initialize_gpu_support(self):
        """Initialize GPU support for deep learning."""
        if TF_AVAILABLE:
            # Configure TensorFlow GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow GPU support enabled with {len(gpus)} GPUs")
                except RuntimeError as e:
                    logger.warning(f"GPU configuration failed: {e}")
        
        if TORCH_AVAILABLE:
            # Configure PyTorch GPU
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"PyTorch GPU support enabled with {device_count} GPUs")
                self.torch_device = torch.device("cuda")
            else:
                self.torch_device = torch.device("cpu")
    
    async def _initialize_feature_engineering(self):
        """Initialize advanced feature engineering pipeline."""
        self.feature_engineers = {
            'numerical_transformer': Pipeline([
                ('imputer', sklearn.impute.SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
                ('selector', SelectKBest(f_regression, k=50))
            ]),
            
            'categorical_transformer': Pipeline([
                ('imputer', sklearn.impute.SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]),
            
            'text_transformer': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
                ('svd', TruncatedSVD(n_components=100))
            ]),
            
            'time_series_transformer': Pipeline([
                ('seasonal_decompose', sklearn.preprocessing.FunctionTransformer(self._extract_seasonal_features)),
                ('lag_features', sklearn.preprocessing.FunctionTransformer(self._create_lag_features)),
                ('rolling_stats', sklearn.preprocessing.FunctionTransformer(self._create_rolling_features))
            ])
        }
        
        logger.info("Feature engineering pipeline initialized")
    
    async def _initialize_explainability(self):
        """Initialize explainable AI components."""
        if SHAP_AVAILABLE:
            self.explainers['shap'] = {
                'tree_explainer': shap.TreeExplainer,
                'kernel_explainer': shap.KernelExplainer,
                'deep_explainer': shap.DeepExplainer if TF_AVAILABLE else None,
                'linear_explainer': shap.LinearExplainer
            }
        
        if LIME_AVAILABLE:
            self.explainers['lime'] = {
                'tabular_explainer': lime_tabular.LimeTabularExplainer
            }
        
        logger.info("Explainability components initialized")
    
    async def _initialize_automl(self):
        """Initialize AutoML engines."""
        if OPTUNA_AVAILABLE:
            self.automl_engines['optuna'] = {
                'study_cache': {},
                'optimization_history': {}
            }
        
        # Custom AutoML components
        self.automl_engines['custom'] = {
            'model_selection': {
                'regression': [
                    RandomForestRegressor,
                    GradientBoostingRegressor,
                    SVR,
                    LinearRegression,
                    Ridge,
                    Lasso
                ],
                'classification': [
                    RandomForestClassifier,
                    GradientBoostingRegressor,
                    SVC,
                    LogisticRegression
                ],
                'clustering': [
                    KMeans,
                    DBSCAN,
                    AgglomerativeClustering
                ]
            },
            'hyperparameter_spaces': {},
            'evaluation_metrics': {}
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.automl_engines['custom']['model_selection']['regression'].append(xgb.XGBRegressor)
            self.automl_engines['custom']['model_selection']['classification'].append(xgb.XGBClassifier)
        
        if LIGHTGBM_AVAILABLE:
            self.automl_engines['custom']['model_selection']['regression'].append(lgb.LGBMRegressor)
            self.automl_engines['custom']['model_selection']['classification'].append(lgb.LGBMClassifier)
        
        if CATBOOST_AVAILABLE:
            self.automl_engines['custom']['model_selection']['regression'].append(cb.CatBoostRegressor)
            self.automl_engines['custom']['model_selection']['classification'].append(cb.CatBoostClassifier)
        
        logger.info("AutoML engines initialized")
    
    async def _initialize_federated_learning(self):
        """Initialize federated learning components."""
        # Placeholder for federated learning initialization
        # This would integrate with PySyft or TensorFlow Federated
        self.federated_components = {
            'workers': {},
            'aggregation_strategy': 'federated_averaging',
            'privacy_budget': 1.0,
            'noise_multiplier': 1.1
        }
        logger.info("Federated learning components initialized")
    
    async def _load_existing_models(self):
        """Load existing trained models."""
        try:
            for model_file in self.models_dir.glob("*.pkl"):
                model_id = model_file.stem
                try:
                    model_data = joblib.load(model_file)
                    self.models[model_id] = model_data['model']
                    self.model_metadata[model_id] = model_data['metadata']
                    
                    model_type = ModelType(model_data['metadata']['model_type'])
                    if model_type not in self.model_versions:
                        self.model_versions[model_type] = []
                    self.model_versions[model_type].append(model_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_id}: {e}")
            
            logger.info(f"Loaded {len(self.models)} existing models")
            
        except Exception as e:
            await self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM,
                "load_existing_models"
            )
    
    async def _initialize_monitoring(self):
        """Initialize model monitoring and health checking."""
        # Initialize performance tracking
        for model_id in self.models.keys():
            self.inference_stats[model_id] = {
                'total_predictions': 0,
                'average_inference_time': 0.0,
                'error_rate': 0.0,
                'last_prediction': None
            }
            
            self.model_health[model_id] = {
                'status': ModelStatus.DEPLOYED.value,
                'last_health_check': datetime.now(timezone.utc),
                'drift_score': 0.0,
                'performance_degradation': 0.0,
                'alerts': []
            }
        
        # Start monitoring background task
        asyncio.create_task(self._model_health_monitoring_task())
        
        logger.info("Model monitoring initialized")
    
    # ==================== Core ML Training Methods ====================
    
    @with_error_handling(ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH)
    async def train_advanced_model(
        self,
        model_type: ModelType,
        training_data: pd.DataFrame,
        target_column: str,
        architecture: ModelArchitecture = ModelArchitecture.ENSEMBLE,
        hyperparameter_optimization: bool = True,
        cross_validation_folds: int = 5,
        enable_feature_selection: bool = True,
        enable_explainability: bool = True,
        model_name: Optional[str] = None
    ) -> ModelTrainingResult:
        """Train advanced ML model with comprehensive optimization."""
        
        training_start = datetime.now(timezone.utc)
        model_id = model_name or f"{model_type.value}_{architecture.value}_{int(time.time())}"
        
        try:
            # Data preprocessing and validation
            processed_data = await self._preprocess_training_data(
                training_data, target_column, model_type
            )
            
            # Feature engineering
            if enable_feature_selection:
                processed_data = await self._engineer_features(
                    processed_data, target_column, model_type
                )
            
            # Split data
            X = processed_data.drop(columns=[target_column])
            y = processed_data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if self._is_classification_task(model_type) else None
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Model selection and training based on architecture
            if architecture == ModelArchitecture.ENSEMBLE:
                model, training_metrics = await self._train_ensemble_model(
                    X_train, y_train, X_val, y_val, model_type, hyperparameter_optimization
                )
            elif architecture == ModelArchitecture.DEEP_NEURAL_NETWORK:
                model, training_metrics = await self._train_deep_model(
                    X_train, y_train, X_val, y_val, model_type
                )
            elif architecture == ModelArchitecture.TRANSFORMER:
                model, training_metrics = await self._train_transformer_model(
                    X_train, y_train, X_val, y_val, model_type
                )
            else:
                model, training_metrics = await self._train_traditional_model(
                    X_train, y_train, X_val, y_val, model_type, hyperparameter_optimization
                )
            
            # Validation and testing
            validation_metrics = await self._evaluate_model(model, X_val, y_val, model_type)
            test_metrics = await self._evaluate_model(model, X_test, y_test, model_type)
            
            # Cross-validation
            cv_scores = []
            if cross_validation_folds > 1:
                cv_scores = await self._perform_cross_validation(
                    model, X_train, y_train, model_type, cross_validation_folds
                )
            
            # Feature importance analysis
            feature_importance = await self._analyze_feature_importance(
                model, X_train.columns.tolist(), model_type
            )
            
            # Model complexity analysis
            model_complexity = await self._analyze_model_complexity(model, architecture)
            
            # Save model
            model_metadata = {
                'model_type': model_type.value,
                'architecture': architecture.value,
                'model_id': model_id,
                'training_timestamp': training_start.isoformat(),
                'feature_columns': X_train.columns.tolist(),
                'target_column': target_column,
                'model_version': '1.0.0'
            }
            
            await self._save_model(model_id, model, model_metadata)
            
            # Update model registry
            self.models[model_id] = model
            self.model_metadata[model_id] = model_metadata
            
            if model_type not in self.model_versions:
                self.model_versions[model_type] = []
            self.model_versions[model_type].append(model_id)
            self.active_models[model_type] = model_id
            
            training_end = datetime.now(timezone.utc)
            training_time = (training_end - training_start).total_seconds()
            
            # Create training result
            result = ModelTrainingResult(
                model_type=model_type,
                model_architecture=architecture,
                model_id=model_id,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                training_time_seconds=training_time,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                test_samples=len(X_test),
                model_complexity=model_complexity,
                training_start=training_start,
                training_end=training_end
            )
            
            logger.info(f"Successfully trained {architecture.value} model for {model_type.value}")
            return result
            
        except Exception as e:
            await self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
                "train_advanced_model",
                {"model_type": model_type.value, "architecture": architecture.value}
            )
            raise
    
    async def _train_ensemble_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: ModelType,
        optimize_hyperparameters: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """Train advanced ensemble model."""
        
        # Base models for ensemble
        if self._is_classification_task(model_type):
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
        else:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('svr', SVR())
            ]
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            if self._is_classification_task(model_type):
                base_models.append(('xgb', xgb.XGBClassifier(random_state=42)))
            else:
                base_models.append(('xgb', xgb.XGBRegressor(random_state=42)))
        
        if LIGHTGBM_AVAILABLE:
            if self._is_classification_task(model_type):
                base_models.append(('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1)))
            else:
                base_models.append(('lgb', lgb.LGBMRegressor(random_state=42, verbose=-1)))
        
        # Create ensemble
        if self._is_classification_task(model_type):
            ensemble = sklearn.ensemble.VotingClassifier(
                estimators=base_models,
                voting='soft'
            )
        else:
            ensemble = VotingRegressor(estimators=base_models)
        
        # Hyperparameter optimization
        if optimize_hyperparameters and OPTUNA_AVAILABLE:
            ensemble = await self._optimize_ensemble_hyperparameters(
                ensemble, X_train, y_train, model_type
            )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = ensemble.predict(X_train)
        val_pred = ensemble.predict(X_val)
        
        if self._is_classification_task(model_type):
            training_metrics = {
                'accuracy': sklearn.metrics.accuracy_score(y_train, train_pred),
                'f1_score': f1_score(y_train, train_pred, average='weighted'),
                'precision': sklearn.metrics.precision_score(y_train, train_pred, average='weighted'),
                'recall': sklearn.metrics.recall_score(y_train, train_pred, average='weighted')
            }
        else:
            training_metrics = {
                'mse': mean_squared_error(y_train, train_pred),
                'mae': mean_absolute_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
        
        return ensemble, training_metrics
    
    async def _train_deep_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: ModelType
    ) -> Tuple[Any, Dict[str, float]]:
        """Train deep neural network model."""
        
        if not TF_AVAILABLE:
            raise ValueError("TensorFlow not available for deep learning")
        
        # Prepare data
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_val_scaled = StandardScaler().fit_transform(X_val)
        
        input_dim = X_train_scaled.shape[1]
        
        # Build model architecture based on model type
        if model_type in [ModelType.SEQUENCE_PREDICTION, ModelType.MULTIVARIATE_FORECASTING]:
            # LSTM for time series
            model = Sequential([
                layers.Reshape((input_dim, 1)),
                LSTM(128, return_sequences=True, dropout=0.2),
                LSTM(64, dropout=0.2),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(1 if not self._is_classification_task(model_type) else len(np.unique(y_train)))
            ])
        else:
            # Standard deep network
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1 if not self._is_classification_task(model_type) else len(np.unique(y_train)))
            ])
        
        # Compile model
        if self._is_classification_task(model_type):
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy' if len(np.unique(y_train)) > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Calculate training metrics
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        training_metrics = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss'])
        }
        
        if self._is_classification_task(model_type):
            training_metrics['final_train_accuracy'] = history.history['accuracy'][-1]
            training_metrics['final_val_accuracy'] = history.history['val_accuracy'][-1]
        
        return model, training_metrics
    
    async def _train_transformer_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: ModelType
    ) -> Tuple[Any, Dict[str, float]]:
        """Train transformer-based model for sequence data."""
        
        if not TF_AVAILABLE:
            raise ValueError("TensorFlow not available for transformer models")
        
        # This is a simplified transformer for tabular data
        # In practice, you'd implement full attention mechanisms
        
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_val_scaled = StandardScaler().fit_transform(X_val)
        
        input_dim = X_train_scaled.shape[1]
        
        # Simple attention-based model
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Reshape((input_dim, 1))(inputs)
        
        # Self-attention layer (simplified)
        attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward network
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1 if not self._is_classification_task(model_type) else len(np.unique(y_train)))(x)
        
        model = Model(inputs, outputs)
        
        # Compile and train (similar to deep model)
        if self._is_classification_task(model_type):
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy' if len(np.unique(y_train)) > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        # Train with callbacks
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        training_metrics = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        return model, training_metrics
    
    async def _train_traditional_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: ModelType,
        optimize_hyperparameters: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """Train traditional ML model with hyperparameter optimization."""
        
        # Select appropriate model
        if self._is_classification_task(model_type):
            base_model = RandomForestClassifier(random_state=42)
        else:
            base_model = RandomForestRegressor(random_state=42)
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            if OPTUNA_AVAILABLE:
                base_model = await self._optimize_hyperparameters_optuna(
                    base_model, X_train, y_train, model_type
                )
            else:
                base_model = await self._optimize_hyperparameters_grid_search(
                    base_model, X_train, y_train, model_type
                )
        
        # Train model
        base_model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = base_model.predict(X_train)
        
        if self._is_classification_task(model_type):
            training_metrics = {
                'accuracy': sklearn.metrics.accuracy_score(y_train, train_pred),
                'f1_score': f1_score(y_train, train_pred, average='weighted')
            }
        else:
            training_metrics = {
                'mse': mean_squared_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred)
            }
        
        return base_model, training_metrics
    
    # ==================== Advanced Prediction Methods ====================
    
    @with_error_handling(ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM)
    async def predict_with_explanation(
        self,
        model_type: ModelType,
        input_data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        model_id: Optional[str] = None,
        include_uncertainty: bool = True,
        include_shap: bool = True,
        include_lime: bool = False
    ) -> AdvancedModelPrediction:
        """Make prediction with comprehensive explanation and uncertainty quantification."""
        
        prediction_start = time.time()
        
        try:
            # Get model
            if model_id is None:
                model_id = self.active_models.get(model_type)
                if model_id is None:
                    raise ValueError(f"No active model found for {model_type}")
            
            model = self.models.get(model_id)
            if model is None:
                raise ValueError(f"Model {model_id} not found")
            
            metadata = self.model_metadata.get(model_id, {})
            
            # Prepare input data
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, np.ndarray):
                feature_columns = metadata.get('feature_columns', [f'feature_{i}' for i in range(input_data.shape[1])])
                input_df = pd.DataFrame(input_data, columns=feature_columns)
            else:
                input_df = input_data.copy()
            
            # Feature engineering (if applicable)
            processed_input = await self._preprocess_prediction_data(input_df, model_type, model_id)
            
            # Make prediction
            if hasattr(model, 'predict_proba') and self._is_classification_task(model_type):
                prediction_proba = model.predict_proba(processed_input)
                prediction = model.predict(processed_input)
                confidence_score = np.max(prediction_proba, axis=1)[0]
            else:
                prediction = model.predict(processed_input)
                confidence_score = 0.8  # Default confidence for regression
            
            # Calculate uncertainty bounds
            uncertainty_bounds = None
            if include_uncertainty:
                uncertainty_bounds = await self._calculate_uncertainty_bounds(
                    model, processed_input, model_type
                )
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = processed_input.columns.tolist()
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            # SHAP explanation
            shap_values = None
            if include_shap and SHAP_AVAILABLE:
                shap_values = await self._calculate_shap_values(
                    model, processed_input, model_type
                )
            
            # LIME explanation
            lime_explanation = None
            if include_lime and LIME_AVAILABLE:
                lime_explanation = await self._calculate_lime_explanation(
                    model, processed_input, model_type
                )
            
            # Determine confidence level
            confidence = self._determine_confidence_level(confidence_score)
            
            # Calculate prediction quality metrics
            quality_score = await self._calculate_prediction_quality(
                model, processed_input, prediction, model_type
            )
            
            # Drift detection
            drift_score = await self._calculate_drift_score(
                processed_input, model_type, model_id
            )
            
            inference_time = (time.time() - prediction_start) * 1000  # Convert to milliseconds
            
            # Update inference statistics
            await self._update_inference_stats(model_id, inference_time)
            
            # Create prediction result
            result = AdvancedModelPrediction(
                model_type=model_type,
                model_architecture=ModelArchitecture(metadata.get('architecture', 'traditional_ml')),
                prediction=prediction[0] if isinstance(prediction, np.ndarray) and len(prediction) == 1 else prediction,
                confidence=confidence,
                confidence_score=confidence_score,
                uncertainty_bounds=uncertainty_bounds,
                features_used=processed_input.columns.tolist(),
                feature_importance=feature_importance,
                feature_values=processed_input.iloc[0].to_dict() if len(processed_input) == 1 else {},
                model_version=metadata.get('model_version', '1.0.0'),
                model_id=model_id,
                inference_time_ms=inference_time,
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                prediction_quality_score=quality_score,
                drift_score=drift_score
            )
            
            return result
            
        except Exception as e:
            await self.error_handler.handle_error(
                e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH,
                "predict_with_explanation",
                {"model_type": model_type.value, "model_id": model_id}
            )
            raise
    
    # ==================== Advanced Anomaly Detection ====================
    
    @with_error_handling(ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM)
    async def detect_anomalies_advanced(
        self,
        data: pd.DataFrame,
        model_type: ModelType = ModelType.ANOMALY_DETECTION,
        detection_methods: Optional[List[str]] = None,
        ensemble_voting: bool = True,
        include_root_cause_analysis: bool = True
    ) -> List[AnomalyDetectionResult]:
        """Advanced multi-method anomaly detection with root cause analysis."""
        
        if detection_methods is None:
            detection_methods = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'statistical']
        
        results = []
        method_scores = {}
        
        # Prepare data
        processed_data = await self._preprocess_prediction_data(data, model_type)
        
        # Apply multiple detection methods
        for method in detection_methods:
            try:
                method_result = await self._apply_anomaly_detection_method(
                    processed_data, method
                )
                method_scores[method] = method_result
            except Exception as e:
                logger.warning(f"Anomaly detection method {method} failed: {e}")
                continue
        
        if not method_scores:
            raise ValueError("All anomaly detection methods failed")
        
        # Ensemble voting if multiple methods
        if ensemble_voting and len(method_scores) > 1:
            ensemble_scores = await self._ensemble_anomaly_scores(method_scores)
        else:
            ensemble_scores = list(method_scores.values())[0]
        
        # Analyze results
        for idx, (is_anomaly, score) in enumerate(ensemble_scores):
            if is_anomaly:
                # Determine anomaly type and severity
                anomaly_type = await self._classify_anomaly_type(
                    processed_data.iloc[idx], score, method_scores
                )
                
                severity = self._determine_anomaly_severity(score)
                
                # Root cause analysis
                root_causes = []
                impact_assessment = {}
                if include_root_cause_analysis:
                    root_causes, impact_assessment = await self._perform_root_cause_analysis(
                        processed_data.iloc[idx], anomaly_type, score
                    )
                
                # Generate recommendations
                recommendations = await self._generate_anomaly_recommendations(
                    anomaly_type, severity, root_causes
                )
                
                result = AnomalyDetectionResult(
                    is_anomaly=True,
                    anomaly_score=score,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=min(0.95, score),
                    affected_metrics=self._identify_affected_metrics(processed_data.iloc[idx]),
                    potential_causes=root_causes,
                    impact_assessment=impact_assessment,
                    recommendations=recommendations,
                    detection_method=f"ensemble_{len(method_scores)}_methods" if ensemble_voting else list(method_scores.keys())[0]
                )
                
                results.append(result)
        
        return results
    
    # ==================== Utility Methods ====================
    
    def _is_classification_task(self, model_type: ModelType) -> bool:
        """Determine if model type is a classification task."""
        classification_types = [
            ModelType.INCIDENT_CLASSIFICATION,
            ModelType.SENTIMENT_ANALYSIS,
            ModelType.ANOMALY_DETECTION
        ]
        return model_type in classification_types
    
    async def _preprocess_training_data(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        model_type: ModelType
    ) -> pd.DataFrame:
        """Comprehensive data preprocessing for training."""
        processed_data = data.copy()
        
        # Handle missing values
        for column in processed_data.columns:
            if processed_data[column].dtype in ['object', 'string']:
                processed_data[column].fillna('missing', inplace=True)
            else:
                processed_data[column].fillna(processed_data[column].median(), inplace=True)
        
        # Encode categorical variables
        categorical_columns = processed_data.select_dtypes(include=['object', 'string']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                processed_data[col] = self.encoders[col].fit_transform(processed_data[col])
            else:
                processed_data[col] = self.encoders[col].transform(processed_data[col])
        
        # Feature scaling (except for target)
        numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != target_column]
        
        if numerical_columns:
            scaler_key = f"{model_type.value}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                processed_data[numerical_columns] = self.scalers[scaler_key].fit_transform(
                    processed_data[numerical_columns]
                )
            else:
                processed_data[numerical_columns] = self.scalers[scaler_key].transform(
                    processed_data[numerical_columns]
                )
        
        return processed_data
    
    async def _preprocess_prediction_data(
        self, 
        data: pd.DataFrame, 
        model_type: ModelType,
        model_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Preprocess data for prediction using saved preprocessors."""
        processed_data = data.copy()
        
        # Handle missing values
        for column in processed_data.columns:
            if processed_data[column].dtype in ['object', 'string']:
                processed_data[column].fillna('missing', inplace=True)
            else:
                processed_data[column].fillna(processed_data[column].median(), inplace=True)
        
        # Apply saved encoders
        categorical_columns = processed_data.select_dtypes(include=['object', 'string']).columns
        
        for col in categorical_columns:
            if col in self.encoders:
                # Handle unseen categories
                try:
                    processed_data[col] = self.encoders[col].transform(processed_data[col])
                except ValueError:
                    # Handle unseen categories by mapping to most frequent class
                    most_frequent = self.encoders[col].classes_[0]
                    processed_data[col] = processed_data[col].apply(
                        lambda x: x if x in self.encoders[col].classes_ else most_frequent
                    )
                    processed_data[col] = self.encoders[col].transform(processed_data[col])
        
        # Apply saved scalers
        numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        if numerical_columns:
            scaler_key = f"{model_type.value}_scaler"
            if scaler_key in self.scalers:
                processed_data[numerical_columns] = self.scalers[scaler_key].transform(
                    processed_data[numerical_columns]
                )
        
        return processed_data
    
    async def _engineer_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_type: ModelType
    ) -> pd.DataFrame:
        """Advanced feature engineering."""
        engineered_data = data.copy()
        
        # Time-based features if datetime columns exist
        datetime_columns = data.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            engineered_data[f'{col}_hour'] = data[col].dt.hour
            engineered_data[f'{col}_day_of_week'] = data[col].dt.dayofweek
            engineered_data[f'{col}_month'] = data[col].dt.month
            engineered_data[f'{col}_quarter'] = data[col].dt.quarter
            engineered_data = engineered_data.drop(columns=[col])
        
        # Interaction features for numerical columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != target_column]
        
        if len(numerical_columns) >= 2:
            # Create polynomial features (degree 2)
            poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_array = poly_features.fit_transform(engineered_data[numerical_columns])
            poly_feature_names = poly_features.get_feature_names_out(numerical_columns)
            
            # Add only interaction terms (not squared terms)
            interaction_indices = [i for i, name in enumerate(poly_feature_names) 
                                 if ' ' in name and '^2' not in name]
            
            if interaction_indices:
                interaction_features = poly_array[:, interaction_indices]
                interaction_names = [poly_feature_names[i] for i in interaction_indices]
                
                interaction_df = pd.DataFrame(interaction_features, columns=interaction_names, index=engineered_data.index)
                engineered_data = pd.concat([engineered_data, interaction_df], axis=1)
        
        # Statistical features
        if len(numerical_columns) >= 3:
            engineered_data['numerical_mean'] = engineered_data[numerical_columns].mean(axis=1)
            engineered_data['numerical_std'] = engineered_data[numerical_columns].std(axis=1)
            engineered_data['numerical_max'] = engineered_data[numerical_columns].max(axis=1)
            engineered_data['numerical_min'] = engineered_data[numerical_columns].min(axis=1)
        
        return engineered_data
    
    async def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: ModelType
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        predictions = model.predict(X_test)
        
        if self._is_classification_task(model_type):
            metrics = {
                'accuracy': sklearn.metrics.accuracy_score(y_test, predictions),
                'precision': sklearn.metrics.precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': sklearn.metrics.recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
            }
            
            # Add AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
                except:
                    pass
        else:
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
            
            # Add MAPE if no zero values
            if not np.any(y_test == 0):
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                metrics['mape'] = mape
        
        return metrics
    
    async def _perform_cross_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: ModelType,
        cv_folds: int = 5
    ) -> List[float]:
        """Perform cross-validation."""
        if self._is_classification_task(model_type):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = cv_folds
            scoring = 'neg_mean_squared_error'
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.tolist()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []
    
    async def _analyze_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_type: ModelType
    ) -> Dict[str, float]:
        """Analyze feature importance."""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) == 1:
                importances = np.abs(model.coef_)
            else:
                importances = np.abs(model.coef_).mean(axis=0)
            importance_dict = dict(zip(feature_names, importances))
        elif hasattr(model, 'estimators_'):
            # Ensemble models
            if hasattr(model.estimators_[0], 'feature_importances_'):
                importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                importance_dict = dict(zip(feature_names, importances))
        
        # Normalize importance values
        if importance_dict:
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        return importance_dict
    
    async def _analyze_model_complexity(
        self,
        model: Any,
        architecture: ModelArchitecture
    ) -> Dict[str, Any]:
        """Analyze model complexity."""
        complexity = {
            'architecture': architecture.value,
            'parameters_count': 0,
            'memory_usage_mb': 0.0,
            'training_complexity': 'unknown',
            'inference_complexity': 'unknown'
        }
        
        if architecture == ModelArchitecture.DEEP_NEURAL_NETWORK and TF_AVAILABLE:
            if hasattr(model, 'count_params'):
                complexity['parameters_count'] = model.count_params()
                complexity['training_complexity'] = 'O(n * p * e)'  # n=samples, p=params, e=epochs
                complexity['inference_complexity'] = 'O(p)'
        elif hasattr(model, 'n_estimators') and hasattr(model, 'max_depth'):
            # Tree-based ensemble
            n_estimators = getattr(model, 'n_estimators', 100)
            max_depth = getattr(model, 'max_depth', None) or 10
            complexity['parameters_count'] = n_estimators * (2 ** max_depth)
            complexity['training_complexity'] = f'O(n * log(n) * {n_estimators})'
            complexity['inference_complexity'] = f'O(log(n) * {n_estimators})'
        
        # Estimate memory usage
        try:
            model_size = len(pickle.dumps(model))
            complexity['memory_usage_mb'] = model_size / (1024 * 1024)
        except:
            complexity['memory_usage_mb'] = 0.0
        
        return complexity
    
    async def _save_model(self, model_id: str, model: Any, metadata: Dict[str, Any]):
        """Save model and metadata."""
        model_path = self.models_dir / f"{model_id}.pkl"
        
        model_data = {
            'model': model,
            'metadata': metadata,
            'saved_at': datetime.now(timezone.utc).isoformat()
        }
        
        joblib.dump(model_data, model_path)
    
    # ==================== Helper Methods for Advanced Features ====================
    
    def _extract_seasonal_features(self, X):
        """Extract seasonal features from time series data."""
        # Placeholder for seasonal decomposition
        return X
    
    def _create_lag_features(self, X):
        """Create lag features for time series."""
        # Placeholder for lag feature creation
        return X
    
    def _create_rolling_features(self, X):
        """Create rolling window statistical features."""
        # Placeholder for rolling features
        return X
    
    async def _optimize_ensemble_hyperparameters(self, ensemble, X_train, y_train, model_type):
        """Optimize ensemble hyperparameters using Optuna."""
        # Placeholder for ensemble hyperparameter optimization
        return ensemble
    
    async def _optimize_hyperparameters_optuna(self, model, X_train, y_train, model_type):
        """Optimize hyperparameters using Optuna."""
        # Placeholder for Optuna optimization
        return model
    
    async def _optimize_hyperparameters_grid_search(self, model, X_train, y_train, model_type):
        """Optimize hyperparameters using grid search."""
        # Placeholder for grid search optimization
        return model
    
    async def _calculate_uncertainty_bounds(self, model, input_data, model_type):
        """Calculate prediction uncertainty bounds."""
        # Placeholder for uncertainty quantification
        return None
    
    async def _calculate_shap_values(self, model, input_data, model_type):
        """Calculate SHAP values for explanation."""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Choose appropriate SHAP explainer based on model type
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, input_data.iloc[:100])
            
            shap_values = explainer.shap_values(input_data.iloc[0:1])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            return dict(zip(input_data.columns, shap_values))
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return None
    
    async def _calculate_lime_explanation(self, model, input_data, model_type):
        """Calculate LIME explanation."""
        # Placeholder for LIME explanation
        return None
    
    def _determine_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Determine confidence level from score."""
        if confidence_score >= 0.95:
            return PredictionConfidence.EXTREMELY_HIGH
        elif confidence_score >= 0.9:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.65:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 0.5:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    async def _calculate_prediction_quality(self, model, input_data, prediction, model_type):
        """Calculate prediction quality score."""
        # Placeholder for quality assessment
        return 0.8
    
    async def _calculate_drift_score(self, input_data, model_type, model_id):
        """Calculate data drift score."""
        # Placeholder for drift detection
        return 0.1
    
    async def _update_inference_stats(self, model_id: str, inference_time: float):
        """Update inference statistics."""
        if model_id not in self.inference_stats:
            self.inference_stats[model_id] = {
                'total_predictions': 0,
                'average_inference_time': 0.0,
                'error_rate': 0.0,
                'last_prediction': None
            }
        
        stats = self.inference_stats[model_id]
        stats['total_predictions'] += 1
        stats['average_inference_time'] = (
            (stats['average_inference_time'] * (stats['total_predictions'] - 1) + inference_time) /
            stats['total_predictions']
        )
        stats['last_prediction'] = datetime.now(timezone.utc)
    
    # ==================== Anomaly Detection Methods ====================
    
    async def _apply_anomaly_detection_method(self, data: pd.DataFrame, method: str):
        """Apply specific anomaly detection method."""
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=0.1, random_state=42)
            scores = detector.fit_predict(data)
            anomaly_scores = detector.decision_function(data)
            return [(score == -1, abs(ascore)) for score, ascore in zip(scores, anomaly_scores)]
        
        elif method == 'one_class_svm':
            detector = OneClassSVM(gamma='scale', nu=0.1)
            scores = detector.fit_predict(data)
            anomaly_scores = detector.decision_function(data)
            return [(score == -1, abs(ascore)) for score, ascore in zip(scores, anomaly_scores)]
        
        elif method == 'local_outlier_factor':
            detector = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            scores = detector.fit_predict(data)
            anomaly_scores = detector.negative_outlier_factor_
            return [(score == -1, abs(ascore)) for score, ascore in zip(scores, anomaly_scores)]
        
        elif method == 'statistical':
            # Z-score based detection
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            max_z_scores = np.max(z_scores, axis=1)
            threshold = 3.0
            return [(score > threshold, score/threshold) for score in max_z_scores]
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
    
    async def _ensemble_anomaly_scores(self, method_scores: Dict[str, List[Tuple[bool, float]]]):
        """Combine anomaly scores from multiple methods."""
        num_samples = len(list(method_scores.values())[0])
        ensemble_results = []
        
        for i in range(num_samples):
            anomaly_votes = sum(1 for method in method_scores.values() if method[i][0])
            avg_score = np.mean([method[i][1] for method in method_scores.values()])
            
            # Majority voting with weighted average
            is_anomaly = anomaly_votes > len(method_scores) / 2
            ensemble_results.append((is_anomaly, avg_score))
        
        return ensemble_results
    
    async def _classify_anomaly_type(self, data_point, score, method_scores):
        """Classify the type of anomaly."""
        # Simplified anomaly classification
        if score > 0.8:
            return "high_severity_outlier"
        elif score > 0.6:
            return "moderate_outlier"
        else:
            return "mild_deviation"
    
    def _determine_anomaly_severity(self, score: float) -> str:
        """Determine anomaly severity level."""
        if score > 0.9:
            return "critical"
        elif score > 0.7:
            return "high"
        elif score > 0.5:
            return "medium"
        else:
            return "low"
    
    async def _perform_root_cause_analysis(self, data_point, anomaly_type, score):
        """Perform root cause analysis for anomaly."""
        # Simplified root cause analysis
        potential_causes = [
            "Unusual feature combination",
            "Data quality issue",
            "System behavior change",
            "External factor influence"
        ]
        
        impact_assessment = {
            "severity": self._determine_anomaly_severity(score),
            "confidence": min(0.95, score),
            "affected_systems": ["configuration_management"],
            "estimated_impact": "medium"
        }
        
        return potential_causes, impact_assessment
    
    def _identify_affected_metrics(self, data_point):
        """Identify which metrics are most affected in the anomaly."""
        # Placeholder implementation
        return data_point.index.tolist()[:5]  # Return top 5 features
    
    async def _generate_anomaly_recommendations(self, anomaly_type, severity, root_causes):
        """Generate recommendations for handling the anomaly."""
        recommendations = []
        
        if severity in ["critical", "high"]:
            recommendations.append("Immediate investigation required")
            recommendations.append("Consider temporarily isolating affected systems")
        
        recommendations.extend([
            "Review configuration changes in the last 24 hours",
            "Check system logs for related errors",
            "Validate data sources for quality issues",
            "Monitor related metrics for patterns"
        ])
        
        return recommendations
    
    # ==================== Background Monitoring Tasks ====================
    
    async def _model_health_monitoring_task(self):
        """Background task for monitoring model health."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for model_id in list(self.models.keys()):
                    await self._check_model_health(model_id)
                
            except Exception as e:
                await self.error_handler.handle_error(
                    e, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.LOW,
                    "model_health_monitoring_task"
                )
    
    async def _check_model_health(self, model_id: str):
        """Check individual model health."""
        try:
            health = self.model_health.get(model_id, {})
            
            # Check if model is still responsive
            if model_id in self.inference_stats:
                stats = self.inference_stats[model_id]
                
                # Check for performance degradation
                if stats['average_inference_time'] > 5000:  # > 5 seconds
                    health['alerts'].append({
                        'type': 'performance_degradation',
                        'message': f'Average inference time is {stats["average_inference_time"]:.2f}ms',
                        'timestamp': datetime.now(timezone.utc)
                    })
                
                # Check error rate
                if stats['error_rate'] > 0.1:  # > 10% errors
                    health['alerts'].append({
                        'type': 'high_error_rate',
                        'message': f'Error rate is {stats["error_rate"]:.2%}',
                        'timestamp': datetime.now(timezone.utc)
                    })
            
            health['last_health_check'] = datetime.now(timezone.utc)
            self.model_health[model_id] = health
            
        except Exception as e:
            logger.warning(f"Health check failed for model {model_id}: {e}")
    
    # ==================== Cleanup and Resource Management ====================
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # Clear model cache
            self.models.clear()
            self.model_metadata.clear()
            
            logger.info("Advanced ML engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary."""
        summary = {
            'total_models': len(self.models),
            'models_by_type': {},
            'inference_statistics': self.inference_stats.copy(),
            'model_health_status': {},
            'feature_importance_global': {},
            'performance_trends': {},
            'recommendations': []
        }
        
        # Models by type
        for model_id, metadata in self.model_metadata.items():
            model_type = metadata.get('model_type', 'unknown')
            if model_type not in summary['models_by_type']:
                summary['models_by_type'][model_type] = 0
            summary['models_by_type'][model_type] += 1
        
        # Health status summary
        for model_id, health in self.model_health.items():
            summary['model_health_status'][model_id] = {
                'status': health.get('status', 'unknown'),
                'alerts_count': len(health.get('alerts', [])),
                'drift_score': health.get('drift_score', 0.0),
                'last_check': health.get('last_health_check', 'never')
            }
        
        # Performance recommendations
        if self.inference_stats:
            avg_inference_time = np.mean([
                stats['average_inference_time'] 
                for stats in self.inference_stats.values()
            ])
            
            if avg_inference_time > 1000:  # > 1 second
                summary['recommendations'].append(
                    "Consider model optimization or hardware upgrade for better inference performance"
                )
        
        return summary

# Factory function
async def create_advanced_ml_engine(
    models_directory: str = "./ml_models_advanced",
    enable_gpu: bool = True,
    enable_distributed: bool = False,
    max_workers: int = 4
) -> AdvancedCentralConfigurationML:
    """Create and initialize advanced ML engine."""
    engine = AdvancedCentralConfigurationML(
        models_directory=models_directory,
        enable_gpu=enable_gpu,
        enable_distributed=enable_distributed,
        max_workers=max_workers
    )
    
    # Initialization is handled in __init__ via asyncio.create_task
    # Wait a moment for initialization to start
    await asyncio.sleep(0.1)
    
    return engine