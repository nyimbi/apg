# Common Capabilities Reorganization Plan

## Mapping Current Structure to ERP 4-Character Codes

### Direct Mappings (Exact Matches):
- `agents/` → `AGNT` (Agents)
- `ai/` → `AICR` (AI Core Framework)
- `auth_rbac/` → `AUTH` (Authentication & RBAC)
- `audit_logging/` → `AUDL` (Audit Logging)
- `biometric/` → `BIOP` (Biometric Processing)
- `caching/` → `CACH` (Caching Layer)
- `computer_vision/` → `CVSN` (Computer Vision)
- `connectors/` → `CONN` (Connectors / Integration Adapters)
- `deployment/` → `DEPL` (Deployment Management)
- `distributed_computing/` → `DIST` (Distributed Computing)
- `edge_computing/` → `EDGE` (Edge Computing)
- `facial/` → `FREC` (Facial Recognition)
- `geo/` → `GEOS` (Geo-Spatial Services)
- `help/` → `HELP` (Help & Knowledge Base)
- `i8n/` → `I18N` (Internationalization)
- `mfa/` → `MFAU` (Multi-Factor Authentication)
- `monitoring/` → `MONI` (Monitoring & Observability)
- `nlp/` → `NLPC` (NLP Core)
- `pose_estimation/` → `POSE` (Pose Estimation)
- `quantum/` → Special (Quantum Computing - not in standard ERP)
- `rag/` → `RAGN` (Retrieval-Augmented Generation)
- `scraper/` → `SCRP` (Scraper / Data Harvesting)
- `search/` → `SRCH` (Search Engine)
- `security/` → `SECU` (Security Framework)
- `shutdown/` → `SHDN` (Shutdown & Lifecycle Control)
- `ten/` → `MTEN` (Multi-Tenancy)
- `user_management/` → `USRM` (User Management)
- `wallet/` → `WALT` (Wallet / Payment Core)
- `website_builder/` → `WSBL` (Website Builder)
- `workflow/` → `WFLO` (Workflow Orchestration)

### Complex Mappings (Need Restructuring):
- `graphrag/` → `GRAG` (Graph-based RAG)
- `security_operations/` → Multiple (`SECU`, `MONI`, `AUDL`)
- `collaboration/` → `COLB` (Collaboration Tools)
- `platform_foundation/` → Multiple modules

### Missing ERP Capabilities (Need Creation):
- `APIG` - API Gateway & Management
- `BKUP` - Backup & Restore  
- `CHAT` - Chat & Messaging
- `CICD` - Continuous Integration/Delivery
- `COMP` - Compliance Management
- `CONF` - Configuration Management
- `ENCR` - Encryption Services
- `ETLP` - ETL / ELT Processing
- `KEYM` - Key Management
- `LOGT` - Logging & Tracing
- `MDM` - Master Data Management
- `META` - Metadata Management
- `MQEB` - Message Queue & Event Bus
- `NTFY` - Notifications & Alerts
- `SCHD` - Scheduling & Job Orchestration
- And others...

## Implementation Strategy:
1. Rename existing directories to 4-character codes
2. Create missing capability directories
3. Restructure complex mappings
4. Update __init__.py files with proper imports
5. Create comprehensive ERP capability registry

© 2025 Datacraft. All rights reserved.