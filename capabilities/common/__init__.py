"""
APG Common ERP Capabilities

Enterprise-grade common capabilities following canonical ERP architecture
with standardized 4-character codes for maximum interoperability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "2.0.0"

# === CORE INFRASTRUCTURE ===
from .agnt import *  # Agents - Autonomous software agents
from .aicr import *  # AI Core Framework - Base AI/ML infrastructure
# Temporarily disable complex auth imports during development
# from .auth import *  # Authentication & RBAC - User auth and permissions
from .audl import *  # Audit Logging - Immutable compliance logging
from .conf import *  # Configuration Management - System-wide config store
from .mten import *  # Multi-Tenancy - Logical tenant isolation
from .usrm import *  # User Management - User lifecycle management

# === SECURITY & COMPLIANCE ===
# Temporarily disable during testing
# from .secu import *  # Security Framework - Core security controls
# from .mfau import *  # Multi-Factor Authentication - 2FA/MFA services
# from .biop import *  # Biometric Processing - Biometric auth/verification
# from .encr import *  # Encryption Services - Data encryption at rest/transit
# from .keym import *  # Key Management - Secure key lifecycle
# from .comp import *  # Compliance Management - Regulatory frameworks
# from .idfd import *  # Identity Federation - SSO, OIDC, SAML
# from .dlpd import *  # Data Loss Prevention - Unauthorized data protection
# from .ztna import *  # Zero Trust Network Access - Context-aware access

# === DATA & INTEGRATION ===
from .conn import *  # Connectors - Third-party service adapters
from .apig import *  # API Gateway & Management - API routing/analytics
from .mqeb import *  # Message Queue & Event Bus - Async communication
from .etlp import *  # ETL/ELT Processing - Data pipelines
from .mdm import *   # Master Data Management - Centralized master data
from .meta import *  # Metadata Management - Semantic metadata registry
from .imex import *  # Data Import/Export - Bulk data operations
from .regy import *  # API/Service Registry - Endpoint catalog
from .dvrl import *  # Data Virtualization - Unified distributed data access

# === SEARCH & KNOWLEDGE ===
from .srch import *  # Search Engine - Indexed ERP data search
from .ragn import *  # Retrieval-Augmented Generation - RAG pipelines
from .grag import *  # Graph-based RAG - Graph-structured knowledge
from .grph import *  # Graph Data Management - Graph database services
from .kngr import *  # Knowledge Graph - Semantic knowledge representation
from .onto import *  # Ontology Management - Vocabulary governance

# === AI & MACHINE LEARNING ===
from .cvsn import *  # Computer Vision - Visual recognition/analysis
from .frec import *  # Facial Recognition - Face detection/identification
from .pose import *  # Pose Estimation - Human pose analysis
from .nlpc import *  # NLP Core - Text processing/language understanding
from .recs import *  # Recommender Systems - Personalized recommendations
from .anom import *  # Anomaly Detection - Pattern deviation detection
from .pred import *  # Predictive Analytics - Forecasting/trend prediction
from .mlcm import *  # AI Model Lifecycle Management - Model deployment/monitoring
from .fedl import *  # Federated Learning - Distributed model training

# === COLLABORATION & COMMUNICATION ===
from .colb import *  # Collaboration Tools - Team communication platform
from .chat import *  # Chat & Messaging - Real-time messaging
from .vidc import *  # Video Conferencing - Video/screen sharing
from .ntfy import *  # Notifications & Alerts - Multi-channel notifications
from .help import *  # Help & Knowledge Base - Contextual help/KB
from .esgn import *  # Digital Forms & eSign - Electronic document signing

# === WORKFLOW & AUTOMATION ===
from .wflo import *  # Workflow Orchestration - Business process management
from .schd import *  # Scheduling & Job Orchestration - Background jobs
from .scpt import *  # Custom Scripting Engine - Sandboxed custom scripts
from .ncod import *  # No-Code/Low-Code Builder - Visual app creation

# === INFRASTRUCTURE & OPERATIONS ===
from .cach import *  # Caching Layer - Distributed in-memory caching
from .moni import *  # Monitoring & Observability - Metrics/logs/traces
from .logt import *  # Logging & Tracing - Centralized structured logging
from .hlth import *  # Health Checks & Diagnostics - Service health reporting
from .depl import *  # Deployment Management - Release/deployment pipelines
from .dist import *  # Distributed Computing - Multi-node processing
from .edge import *  # Edge Computing - Network edge compute
from .bkup import *  # Backup & Restore - Data/config backup/recovery
from .cicd import *  # Continuous Integration/Delivery - Build/test/deploy
from .envm import *  # Environment Management - Dev/test/prod environments
from .shdn import *  # Shutdown & Lifecycle Control - Service lifecycle

# === SPECIALIZED SERVICES ===
from .geos import *  # Geo-Spatial Services - Mapping/geocoding/spatial analytics
from .i18n import *  # Internationalization - Multi-language/locale support
from .walt import *  # Wallet/Payment Core - Digital wallet/transfers
from .wsbl import *  # Website Builder - Low-code web page creation
from .scrp import *  # Scraper/Data Harvesting - Web data extraction
from .mchn import *  # Multi-Channel Output - Email/SMS/push notifications
from .them import *  # UI/UX Theming & Branding - Custom UI themes
from .accs import *  # Accessibility Services - WCAG compliance
from .cons import *  # Consent & Privacy Management - User data consent
from .plgn import *  # Plugin/Extension Framework - Modular add-ons
from .sbox import *  # Sandbox/Testing Environment - Isolated test environments
from .audp import *  # Audio Processing - Audio analysis/recognition

# === EMERGING TECHNOLOGIES ===
from .dtwn import *  # Digital Twin Framework - Virtual asset/process replicas
from .iotd import *  # IoT Device Integration - IoT device management/ingestion
from .bclg import *  # Blockchain Ledger Services - Distributed ledger
from .esgc import *  # ESG/Carbon Tracking - Environmental/governance metrics
from .quan import *  # Quantum Computing - Quantum processing capabilities

# === LEGACY/SPECIAL MODULES ===
from .seop import *  # Security Operations - Advanced security operations
from .plfd import *  # Platform Foundation - Core platform services
from .tens import *  # Tenants - Legacy tenant management

__all__ = [
    # Core Infrastructure
    "agnt", "aicr", "auth", "audl", "conf", "mten", "usrm",
    
    # Security & Compliance
    "secu", "mfau", "biop", "encr", "keym", "comp", "idfd", "dlpd", "ztna",
    
    # Data & Integration
    "conn", "apig", "mqeb", "etlp", "mdm", "meta", "imex", "regy", "dvrl",
    
    # Search & Knowledge
    "srch", "ragn", "grag", "grph", "kngr", "onto",
    
    # AI & Machine Learning
    "cvsn", "frec", "pose", "nlpc", "recs", "anom", "pred", "mlcm", "fedl",
    
    # Collaboration & Communication
    "colb", "chat", "vidc", "ntfy", "help", "esgn",
    
    # Workflow & Automation
    "wflo", "schd", "scpt", "ncod",
    
    # Infrastructure & Operations
    "cach", "moni", "logt", "hlth", "depl", "dist", "edge", "bkup", "cicd", 
    "envm", "shdn",
    
    # Specialized Services
    "geos", "i18n", "walt", "wsbl", "scrp", "mchn", "them", "accs", "cons", 
    "plgn", "sbox", "audp",
    
    # Emerging Technologies
    "dtwn", "iotd", "bclg", "esgc", "quan",
    
    # Legacy/Special
    "seop", "plfd", "tens",
]