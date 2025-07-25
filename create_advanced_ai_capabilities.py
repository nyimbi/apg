#!/usr/bin/env python3
"""
Create Advanced AI Capabilities
===============================

Create comprehensive AI/ML capabilities for the APG composable template system.
"""

import json
from pathlib import Path
from templates.composable.capability import Capability, CapabilityCategory, CapabilityDependency, CapabilityIntegration

def create_computer_vision_capability():
    """Create computer vision capability"""
    return Capability(
        name="Computer Vision",
        category=CapabilityCategory.AI,
        description="Computer vision and image processing with OpenCV and deep learning",
        version="1.0.0",
        python_requirements=[
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.0",
            "face-recognition>=1.3.0"
        ],
        features=[
            "Image Classification",
            "Object Detection",
            "Face Recognition",
            "Image Segmentation",
            "OCR Text Extraction",
            "Video Processing",
            "Real-time Analysis",
            "Custom Model Training"
        ],
        compatible_bases=["flask_webapp", "microservice", "api_only"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store image metadata and results"),
            CapabilityDependency("data/vector_database", reason="Store image embeddings for similarity search", optional=True)
        ],
        integration=CapabilityIntegration(
            models=["Image", "DetectionResult", "CVModel", "ProcessingJob"],
            views=["ImageView", "DetectionView", "ModelView"],
            apis=["cv/upload", "cv/detect", "cv/classify", "cv/ocr"],
            templates=["cv_dashboard.html", "image_gallery.html"],
            config_additions={
                "CV_UPLOAD_FOLDER": "/tmp/cv_uploads",
                "CV_MAX_FILE_SIZE": 16777216,  # 16MB
                "CV_ALLOWED_EXTENSIONS": ["jpg", "jpeg", "png", "bmp", "tiff"]
            }
        )
    )

def create_nlp_capability():
    """Create natural language processing capability"""
    return Capability(
        name="Natural Language Processing",
        category=CapabilityCategory.AI,
        description="Advanced NLP with sentiment analysis, entity extraction, and text processing",
        version="1.0.0",
        python_requirements=[
            "spacy>=3.6.0",
            "nltk>=3.8.0",
            "transformers>=4.30.0",
            "textblob>=0.17.0",
            "wordcloud>=1.9.0",
            "scikit-learn>=1.3.0"
        ],
        features=[
            "Sentiment Analysis",
            "Named Entity Recognition",
            "Text Classification",
            "Language Detection",
            "Text Summarization",
            "Keyword Extraction",
            "Topic Modeling",
            "Text Similarity"
        ],
        compatible_bases=["flask_webapp", "microservice", "api_only"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store text analysis results"),
            CapabilityDependency("ai/llm_integration", reason="Enhanced text generation capabilities", optional=True)
        ],
        integration=CapabilityIntegration(
            models=["TextDocument", "NLPResult", "Entity", "Sentiment"],
            views=["TextAnalysisView", "EntityView", "SentimentView"],
            apis=["nlp/analyze", "nlp/sentiment", "nlp/entities", "nlp/summarize"],
            templates=["nlp_dashboard.html", "text_analysis.html"]
        )
    )

def create_ml_training_capability():
    """Create machine learning training capability"""
    return Capability(
        name="ML Model Training",
        category=CapabilityCategory.AI,
        description="Machine learning model training, evaluation, and deployment pipeline",
        version="1.0.0",
        python_requirements=[
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "joblib>=1.3.0",
            "mlflow>=2.5.0",
            "optuna>=3.2.0",
            "xgboost>=1.7.0"
        ],
        features=[
            "Automated ML Pipeline",
            "Model Training",
            "Hyperparameter Optimization",
            "Cross-validation",
            "Model Evaluation",
            "Model Versioning",
            "A/B Testing",
            "Performance Monitoring"
        ],
        compatible_bases=["flask_webapp", "microservice"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store training data and model metadata"),
            CapabilityDependency("analytics/basic_analytics", reason="Model performance visualization")
        ],
        integration=CapabilityIntegration(
            models=["MLModel", "TrainingJob", "Dataset", "Experiment", "ModelVersion"],
            views=["ModelView", "TrainingView", "ExperimentView"],
            apis=["ml/train", "ml/predict", "ml/evaluate", "ml/deploy"],
            templates=["ml_dashboard.html", "training_monitor.html"],
            config_additions={
                "ML_MODEL_STORAGE": "/var/models",
                "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"
            }
        )
    )

def create_recommendation_engine_capability():
    """Create recommendation engine capability"""
    return Capability(
        name="Recommendation Engine",
        category=CapabilityCategory.AI,
        description="Collaborative and content-based recommendation system",
        version="1.0.0",
        python_requirements=[
            "scikit-learn>=1.3.0",
            "surprise>=1.1.3",
            "implicit>=0.7.0",
            "pandas>=2.0.0",
            "scipy>=1.11.0"
        ],
        features=[
            "Collaborative Filtering",
            "Content-based Filtering",
            "Hybrid Recommendations",
            "Real-time Scoring",
            "A/B Testing",
            "Cold Start Handling",
            "Popularity-based Fallback",
            "Recommendation Explanation"
        ],
        compatible_bases=["flask_webapp", "microservice"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store user interactions and recommendations"),
            CapabilityDependency("analytics/basic_analytics", reason="Recommendation performance analysis")
        ],
        integration=CapabilityIntegration(
            models=["User", "Item", "Interaction", "Recommendation", "RecommendationModel"],
            views=["RecommendationView", "InteractionView"],
            apis=["recommendations/get", "recommendations/feedback", "recommendations/similar"],
            templates=["recommendations_dashboard.html"]
        )
    )

def create_time_series_capability():
    """Create time series forecasting capability"""
    return Capability(
        name="Time Series Forecasting",
        category=CapabilityCategory.AI,
        description="Time series analysis and forecasting with multiple algorithms",
        version="1.0.0",
        python_requirements=[
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "statsmodels>=0.14.0",
            "prophet>=1.1.4",
            "plotly>=5.15.0",
            "scikit-learn>=1.3.0"
        ],
        features=[
            "ARIMA Modeling",
            "Prophet Forecasting",
            "Seasonal Decomposition",
            "Anomaly Detection",
            "Multiple Forecasts",
            "Confidence Intervals",
            "Model Comparison",
            "Interactive Visualizations"
        ],
        compatible_bases=["flask_webapp", "dashboard"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store time series data"),
            CapabilityDependency("analytics/basic_analytics", reason="Time series visualization")
        ],
        integration=CapabilityIntegration(
            models=["TimeSeries", "Forecast", "ForecastModel", "Anomaly"],
            views=["TimeSeriesView", "ForecastView", "AnomalyView"],
            apis=["timeseries/forecast", "timeseries/detect_anomalies", "timeseries/analyze"],
            templates=["timeseries_dashboard.html", "forecast_charts.html"]
        )
    )

def create_speech_processing_capability():
    """Create speech processing capability"""
    return Capability(
        name="Speech Processing",
        category=CapabilityCategory.AI,
        description="Speech-to-text, text-to-speech, and audio processing",
        version="1.0.0",
        python_requirements=[
            "speech-recognition>=3.10.0",
            "pydub>=0.25.0",
            "gTTS>=2.3.0",
            "whisper-openai>=0.4.1",
            "librosa>=0.10.0",
            "pyaudio>=0.2.11"
        ],
        features=[
            "Speech-to-Text",
            "Text-to-Speech",
            "Audio Transcription",
            "Voice Activity Detection",
            "Speaker Identification",
            "Audio Enhancement",
            "Real-time Processing",
            "Multiple Language Support"
        ],
        compatible_bases=["flask_webapp", "microservice", "real_time"],
        dependencies=[
            CapabilityDependency("data/postgresql_database", reason="Store audio metadata and transcriptions")
        ],
        integration=CapabilityIntegration(
            models=["AudioFile", "Transcription", "Speaker", "VoiceCommand"],
            views=["AudioView", "TranscriptionView", "SpeakerView"],
            apis=["speech/transcribe", "speech/synthesize", "speech/identify"],
            templates=["speech_dashboard.html", "audio_player.html"],
            config_additions={
                "AUDIO_UPLOAD_FOLDER": "/tmp/audio_uploads",
                "SUPPORTED_AUDIO_FORMATS": ["wav", "mp3", "m4a", "flac"]
            }
        )
    )

def save_ai_capabilities():
    """Save all AI capabilities to the filesystem"""
    print("🧠 Creating Advanced AI Capabilities")
    print("=" * 60)
    
    # Create capabilities
    capabilities = [
        create_computer_vision_capability(),
        create_nlp_capability(),
        create_ml_training_capability(),
        create_recommendation_engine_capability(),
        create_time_series_capability(),
        create_speech_processing_capability()
    ]
    
    # Save each capability
    capabilities_dir = Path(__file__).parent / 'templates' / 'composable' / 'capabilities' / 'ai'
    capabilities_dir.mkdir(parents=True, exist_ok=True)
    
    for capability in capabilities:
        # Create capability directory
        cap_name = capability.name.lower().replace(' ', '_').replace('/', '_')
        cap_dir = capabilities_dir / cap_name
        cap_dir.mkdir(exist_ok=True)
        
        # Create standard directories
        for subdir in ['models', 'views', 'templates', 'static', 'tests', 'config', 'scripts']:
            (cap_dir / subdir).mkdir(exist_ok=True)
        
        # Save capability.json
        with open(cap_dir / 'capability.json', 'w') as f:
            json.dump(capability.to_dict(), f, indent=2)
        
        # Create integration template
        create_ai_integration_template(cap_dir, capability)
        
        print(f"  ✅ Created {capability.name}")
    
    print(f"\n📁 AI capabilities saved to: {capabilities_dir}")
    return capabilities

def create_ai_integration_template(cap_dir: Path, capability: Capability):
    """Create integration template for AI capability"""
    cap_name_snake = capability.name.lower().replace(' ', '_').replace('/', '_')
    cap_name_class = capability.name.replace(' ', '').replace('/', '')
    
    integration_content = f'''"""
{capability.name} Integration
{'=' * (len(capability.name) + 12)}

Integration logic for the {capability.name} capability.
Handles AI/ML-specific setup and model loading.
"""

import logging
from flask import Blueprint
from flask_appbuilder import BaseView

# Configure logging
log = logging.getLogger(__name__)

# Create capability blueprint
{cap_name_snake}_bp = Blueprint(
    '{cap_name_snake}',
    __name__,
    url_prefix='/ai/{cap_name_snake}',
    template_folder='templates',
    static_folder='static'
)


def integrate_{cap_name_snake}(app, appbuilder, db):
    """
    Integrate {capability.name} capability into the application.
    
    Args:
        app: Flask application instance
        appbuilder: Flask-AppBuilder instance
        db: SQLAlchemy database instance
    """
    try:
        # Register blueprint
        app.register_blueprint({cap_name_snake}_bp)
        
        # Import and register models
        from .models import *  # noqa
        
        # Import and register views
        from .views import *  # noqa
        
        # Apply AI-specific configuration
        config_additions = {repr(capability.integration.config_additions)}
        for key, value in config_additions.items():
            app.config[key] = value
        
        # Initialize AI service
        ai_service = {cap_name_class}Service(app, appbuilder, db)
        app.extensions['{cap_name_snake}_service'] = ai_service
        
        log.info(f"Successfully integrated {capability.name} capability")
        
    except Exception as e:
        log.error(f"Failed to integrate {capability.name} capability: {{e}}")
        raise


class {cap_name_class}Service:
    """
    Main service class for {capability.name}.
    
    Handles AI/ML model loading and inference.
    """
    
    def __init__(self, app, appbuilder, db):
        self.app = app
        self.appbuilder = appbuilder
        self.db = db
        self.models = {{}}
        self.initialize_service()
    
    def initialize_service(self):
        """Initialize AI service and load models"""
        log.info(f"Initializing {capability.name} service")
        
        try:
            # Load pre-trained models
            self.load_models()
            
            # Setup processing pipeline
            self.setup_pipeline()
            
        except Exception as e:
            log.error(f"Error initializing AI service: {{e}}")
    
    def load_models(self):
        """Load AI/ML models"""
        # Model loading logic specific to capability
        pass
    
    def setup_pipeline(self):
        """Setup AI processing pipeline"""
        # Pipeline setup logic
        pass
    
    def process(self, input_data):
        """Process input data through AI pipeline"""
        # Main processing logic
        return {{"status": "processed", "result": None}}
'''
    
    # Save integration template
    with open(cap_dir / 'integration.py.template', 'w') as f:
        f.write(integration_content)
    
    # Create models template for AI
    models_content = f'''"""
{capability.name} Models
{'=' * (len(capability.name) + 7)}

Database models for {capability.name} capability.
"""

from flask_appbuilder import Model
from flask_appbuilder.models.mixins import AuditMixin
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime


class AIBaseModel(AuditMixin, Model):
    """Base model for AI entities"""
    __abstract__ = True
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Add AI-specific models based on capability
{generate_ai_models(capability)}
'''
    
    with open(cap_dir / 'models' / '__init__.py.template', 'w') as f:
        f.write(models_content)

def generate_ai_models(capability: Capability) -> str:
    """Generate AI-specific models based on capability type"""
    if "Computer Vision" in capability.name:
        return '''
class Image(AIBaseModel):
    """Image processing model"""
    __tablename__ = 'cv_images'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(256), nullable=False)
    original_name = Column(String(256))
    file_path = Column(String(512))
    file_size = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String(32))
    
    detections = relationship("DetectionResult", back_populates="image")


class DetectionResult(AIBaseModel):
    """Object detection result"""
    __tablename__ = 'cv_detections'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('cv_images.id'))
    object_class = Column(String(128))
    confidence = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)
    
    image = relationship("Image", back_populates="detections")
'''
    elif "NLP" in capability.name or "Natural Language" in capability.name:
        return '''
class TextDocument(AIBaseModel):
    """Text document model"""
    __tablename__ = 'nlp_documents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(256))
    content = Column(Text, nullable=False)
    language = Column(String(32))
    word_count = Column(Integer)
    
    results = relationship("NLPResult", back_populates="document")


class NLPResult(AIBaseModel):
    """NLP analysis result"""
    __tablename__ = 'nlp_results'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('nlp_documents.id'))
    analysis_type = Column(String(64))
    result_data = Column(JSON)
    confidence = Column(Float)
    
    document = relationship("TextDocument", back_populates="results")
'''
    elif "ML Model Training" in capability.name:
        return '''
class MLModel(AIBaseModel):
    """Machine learning model"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    model_type = Column(String(64))
    algorithm = Column(String(128))
    version = Column(String(32))
    file_path = Column(String(512))
    status = Column(String(32), default='training')
    
    training_jobs = relationship("TrainingJob", back_populates="model")


class TrainingJob(AIBaseModel):
    """ML model training job"""
    __tablename__ = 'ml_training_jobs'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    dataset_path = Column(String(512))
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    status = Column(String(32), default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    model = relationship("MLModel", back_populates="training_jobs")
'''
    elif "Time Series" in capability.name:
        return '''
class TimeSeries(AIBaseModel):
    """Time series data model"""
    __tablename__ = 'timeseries_data'
    
    id = Column(Integer, primary_key=True)
    series_name = Column(String(256), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    metadata = Column(JSON)
    
    forecasts = relationship("Forecast", back_populates="series")


class Forecast(AIBaseModel):
    """Time series forecast"""
    __tablename__ = 'timeseries_forecasts'
    
    id = Column(Integer, primary_key=True)
    series_id = Column(Integer, ForeignKey('timeseries_data.id'))
    forecast_date = Column(DateTime, nullable=False)
    predicted_value = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_used = Column(String(128))
    
    series = relationship("TimeSeries", back_populates="forecasts")
'''
    else:
        return '''
# Generic AI model
class AIProcessingJob(AIBaseModel):
    """Generic AI processing job"""
    __tablename__ = 'ai_jobs'
    
    id = Column(Integer, primary_key=True)
    job_type = Column(String(64), nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    status = Column(String(32), default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
'''

def main():
    """Create all advanced AI capabilities"""
    try:
        capabilities = save_ai_capabilities()
        
        print(f"\n🎉 Successfully created {len(capabilities)} advanced AI capabilities!")
        print(f"\n📋 AI Capabilities Created:")
        for cap in capabilities:
            print(f"   • {cap.name} - {cap.description}")
        
        print(f"\n🚀 These capabilities enable:")
        print(f"   • Computer vision and image processing")
        print(f"   • Natural language processing and analysis")
        print(f"   • Machine learning model training and deployment")
        print(f"   • Recommendation systems")
        print(f"   • Time series forecasting and anomaly detection")
        print(f"   • Speech processing and voice interfaces")
        
        return True
        
    except Exception as e:
        print(f"💥 Error creating AI capabilities: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)