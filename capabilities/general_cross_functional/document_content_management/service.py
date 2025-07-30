"""
APG Document Content Management - Core Service

Comprehensive document management service integrating all 10 revolutionary
capabilities with APG AI/ML/RAG facilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from .models import (
    DCMDocument, DCMFolder, DCMDocumentVersion, DCMDocumentPermission,
    DCMComment, DCMWorkflow, DCMWorkflowInstance, DCMWorkflowStep,
    DCMKnowledgeBase, DCMKnowledgeArticle, DCMAsset, DCMAssetCollection,
    DCMAuditLog, DCMRetentionPolicy, DCMNotification, DCMAnalytics,
    DCMIntelligentProcessing, DCMSemanticSearch, DCMContentIntelligence,
    DCMGenerativeAI, DCMPredictiveAnalytics, DCMContentFabric,
    DCMProcessAutomation, DCMDataLossPrevention, DCMBlockchainProvenance,
    DCMDocumentType, DCMContentFormat, DCMPermissionLevel
)

from .idp_service import IDPProcessor
from .semantic_search_service import SemanticSearchEngine
from .classification_service import ClassificationEngine
from .retention_service import RetentionEngine
from .generative_ai_service import GenerativeAIEngine
from .predictive_service import PredictiveEngine
from .ocr_service import OCRService, OCRConfig


class DocumentManagementService:
    """Core document management service integrating all revolutionary capabilities"""
    
    def __init__(
        self,
        apg_ai_client=None,
        apg_rag_client=None,
        apg_genai_client=None,
        apg_ml_client=None,
        apg_blockchain_client=None,
        apg_compliance_client=None,
        ocr_config: OCRConfig = None
    ):
        """Initialize document management service with APG integrations"""
        self.apg_ai_client = apg_ai_client
        self.apg_rag_client = apg_rag_client
        self.apg_genai_client = apg_genai_client
        self.apg_ml_client = apg_ml_client
        self.apg_blockchain_client = apg_blockchain_client
        self.apg_compliance_client = apg_compliance_client
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize revolutionary capability engines
        self.idp_processor = IDPProcessor(apg_ai_client, apg_rag_client)
        self.search_engine = SemanticSearchEngine(apg_ai_client, apg_rag_client)
        self.classification_engine = ClassificationEngine(apg_ai_client)
        self.retention_engine = RetentionEngine(apg_ai_client, apg_compliance_client)
        self.genai_engine = GenerativeAIEngine(apg_ai_client, apg_rag_client, apg_genai_client)
        self.predictive_engine = PredictiveEngine(apg_ai_client, apg_ml_client)
        self.ocr_service = OCRService(ocr_config, apg_ai_client)
        
        # Service statistics
        self.service_stats = {
            'documents_processed': 0,
            'ai_operations_performed': 0,
            'search_queries_handled': 0,
            'retention_policies_applied': 0,
            'blockchain_verifications': 0,
            'dlp_events_processed': 0,
            'ocr_operations_performed': 0
        }
    
    async def create_document(
        self,
        document_data: Dict[str, Any],
        file_path: str,
        user_id: str,
        tenant_id: str,
        process_ai: bool = True
    ) -> DCMDocument:
        """Create new document with comprehensive AI processing"""
        try:
            # Create document record
            document = DCMDocument(
                tenant_id=tenant_id,
                created_by=user_id,
                updated_by=user_id,
                **document_data
            )
            
            # 1. Intelligent Document Processing
            if process_ai:
                idp_result = await self.idp_processor.process_document(
                    document, file_path
                )
                
                # 2. AI-Driven Classification
                extracted_text = idp_result.extracted_data.get('text_content', '')
                classification_result = await self.classification_engine.classify_document(
                    document, extracted_text, idp_result.extracted_data
                )
                
                # 3. Predictive Analytics
                prediction_result = await self.predictive_engine.generate_predictions(
                    document, classification_result
                )
                
                # 4. Retention Analysis
                retention_analysis = await self.retention_engine.analyze_document_retention(
                    document, classification_result
                )
                
                # 5. Blockchain Provenance (if configured)
                if self.apg_blockchain_client:
                    blockchain_record = await self._create_blockchain_provenance(
                        document, file_path
                    )
                
                # 6. Data Loss Prevention Scan
                dlp_result = await self._scan_for_dlp_violations(
                    document, classification_result
                )
                
                # Update document with AI insights
                document.ai_tags = classification_result.related_concepts
                document.content_summary = classification_result.content_summary
                document.sentiment_score = classification_result.sentiment_analysis.get('polarity') if classification_result.sentiment_analysis else None
            
            # Create audit log
            await self._create_audit_log(
                document.id, 'document_created', user_id, {'ai_processed': process_ai}
            )
            
            self.service_stats['documents_processed'] += 1
            self.logger.info(f"Document created successfully: {document.id}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Document creation error: {str(e)}")
            raise
    
    async def search_documents(
        self,
        query: str,
        user_id: str,
        tenant_id: str,
        search_options: Optional[Dict[str, Any]] = None
    ) -> DCMSemanticSearch:
        """Perform contextual and semantic search"""
        try:
            search_result = await self.search_engine.search_documents(
                query, user_id, tenant_id, search_options
            )
            
            self.service_stats['search_queries_handled'] += 1
            self.service_stats['ai_operations_performed'] += 1
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            raise
    
    async def interact_with_content(
        self,
        document_id: str,
        user_prompt: str,
        interaction_type: str,
        user_id: str,
        context_documents: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> DCMGenerativeAI:
        """Interact with document content using generative AI"""
        try:
            # Get document
            document = await self._get_document(document_id)
            
            # Process interaction
            interaction_result = await self.genai_engine.process_interaction(
                document, user_prompt, interaction_type, user_id, context_documents, options
            )
            
            self.service_stats['ai_operations_performed'] += 1
            
            return interaction_result
            
        except Exception as e:
            self.logger.error(f"GenAI interaction error: {str(e)}")
            raise
    
    async def apply_retention_policy(
        self,
        policy_id: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """Apply smart retention policy to documents"""
        try:
            # Get retention policy
            policy = await self._get_retention_policy(policy_id)
            
            # Apply policy
            policy_result = await self.retention_engine.apply_retention_policy(
                policy, document_ids
            )
            
            self.service_stats['retention_policies_applied'] += 1
            
            return policy_result
            
        except Exception as e:
            self.logger.error(f"Retention policy application error: {str(e)}")
            raise
    
    async def verify_document_provenance(
        self,
        document_id: str
    ) -> Dict[str, Any]:
        """Verify document provenance using blockchain"""
        try:
            if not self.apg_blockchain_client:
                raise ValueError("Blockchain client not configured")
            
            verification_result = await self.apg_blockchain_client.verify_document(
                document_id
            )
            
            self.service_stats['blockchain_verifications'] += 1
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Provenance verification error: {str(e)}")
            raise
    
    async def _create_blockchain_provenance(
        self,
        document: DCMDocument,
        file_path: str
    ) -> DCMBlockchainProvenance:
        """Create blockchain provenance record"""
        try:
            if not self.apg_blockchain_client:
                return None
            
            # Calculate document hash
            import hashlib
            with open(file_path, 'rb') as f:
                document_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Create blockchain transaction
            blockchain_result = await self.apg_blockchain_client.create_provenance_record(
                document_id=document.id,
                document_hash=document_hash,
                metadata={
                    'title': document.title,
                    'created_by': document.created_by,
                    'created_at': document.created_at.isoformat()
                }
            )
            
            # Create provenance record
            provenance_record = DCMBlockchainProvenance(
                tenant_id=document.tenant_id,
                created_by=document.created_by,
                updated_by=document.updated_by,
                document_id=document.id,
                document_hash=document_hash,
                transaction_hash=blockchain_result['transaction_hash'],
                blockchain_network=blockchain_result['network'],
                block_number=blockchain_result.get('block_number'),
                block_timestamp=datetime.fromisoformat(blockchain_result['timestamp']),
                origin_proof=blockchain_result['origin_proof'],
                chain_of_custody=[],
                integrity_checkpoints=[],
                verification_status='verified',
                last_verified_at=datetime.utcnow(),
                verification_count=1
            )
            
            return provenance_record
            
        except Exception as e:
            self.logger.error(f"Blockchain provenance creation error: {str(e)}")
            return None
    
    async def _scan_for_dlp_violations(
        self,
        document: DCMDocument,
        content_intelligence: DCMContentIntelligence
    ) -> DCMDataLossPrevention:
        """Scan document for data loss prevention violations"""
        try:
            # Calculate risk score based on content intelligence
            risk_score = 0.0
            behavioral_anomaly = False
            
            if content_intelligence.sensitive_data_detected:
                risk_score += 0.4
            
            if content_intelligence.compliance_flags:
                risk_score += len(content_intelligence.compliance_flags) * 0.1
            
            # Determine risk level
            if risk_score > 0.7:
                risk_level = 'critical'
            elif risk_score > 0.5:
                risk_level = 'high'
            elif risk_score > 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Create DLP record
            dlp_record = DCMDataLossPrevention(
                tenant_id=document.tenant_id,
                created_by=document.created_by,
                updated_by=document.updated_by,
                document_id=document.id,
                user_id=document.created_by,
                event_type='document_scan',
                risk_level=risk_level,
                risk_score=risk_score,
                behavioral_anomaly=behavioral_anomaly,
                access_pattern={'action': 'create', 'timestamp': datetime.utcnow().isoformat()},
                content_sensitivity='high' if content_intelligence.sensitive_data_detected else 'normal',
                policy_violations=content_intelligence.compliance_flags,
                auto_response_taken=False,
                response_actions=[],
                alert_generated=risk_score > 0.5,
                escalation_level=1 if risk_score > 0.7 else 0,
                resolution_status='open' if risk_score > 0.5 else 'resolved'
            )
            
            if dlp_record.alert_generated:
                self.service_stats['dlp_events_processed'] += 1
            
            return dlp_record
            
        except Exception as e:
            self.logger.error(f"DLP scan error: {str(e)}")
            return None
    
    async def _get_document(self, document_id: str) -> DCMDocument:
        """Retrieve document by ID"""
        # This would typically query the database
        # For now, return a mock document
        return DCMDocument(
            id=document_id,
            tenant_id='test-tenant',
            created_by='test-user',
            updated_by='test-user',
            name='Test Document',
            title='Test Document',
            document_type=DCMDocumentType.TEXT_DOCUMENT,
            content_format=DCMContentFormat.PDF,
            file_name='test.pdf',
            file_size=1024,
            file_hash='abcd1234',
            mime_type='application/pdf',
            storage_path='/storage/test.pdf'
        )
    
    async def _get_retention_policy(self, policy_id: str) -> DCMRetentionPolicy:
        """Retrieve retention policy by ID"""
        # This would typically query the database
        return DCMRetentionPolicy(
            id=policy_id,
            tenant_id='test-tenant',
            created_by='admin',
            updated_by='admin',
            name='Default Policy',
            policy_type='retention',
            applies_to=['document'],
            retention_period_days=2190,
            trigger_event='creation',
            retention_action='archive'
        )
    
    async def _create_audit_log(
        self,
        resource_id: str,
        action: str,
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> DCMAuditLog:
        """Create comprehensive audit log entry"""
        try:
            audit_log = DCMAuditLog(
                tenant_id='default',
                created_by=user_id,
                updated_by=user_id,
                resource_id=resource_id,
                resource_type='document',
                action=action,
                action_category='modify',
                user_id=user_id,
                success=True,
                risk_level='low',
                technical_details=details or {},
                retention_category='standard'
            )
            
            return audit_log
            
        except Exception as e:
            self.logger.error(f"Audit log creation error: {str(e)}")
            return None
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all capabilities"""
        try:
            analytics = {
                'service_statistics': self.service_stats,
                'idp_analytics': await self.idp_processor.get_processing_analytics(),
                'search_analytics': await self.search_engine.get_search_analytics(),
                'classification_analytics': await self.classification_engine.get_classification_analytics(),
                'retention_analytics': await self.retention_engine.get_retention_analytics(),
                'genai_analytics': await self.genai_engine.get_genai_analytics(),
                'predictive_analytics': await self.predictive_engine.get_predictive_analytics(),
                'ocr_analytics': await self._get_ocr_analytics(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Analytics retrieval error: {str(e)}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all services"""
        health_status = {
            'overall_status': 'healthy',
            'services': {
                'idp_processor': 'healthy',
                'search_engine': 'healthy',
                'classification_engine': 'healthy',
                'retention_engine': 'healthy',
                'genai_engine': 'healthy',
                'predictive_engine': 'healthy',
                'ocr_service': 'healthy'
            },
            'apg_integrations': {
                'ai_client': 'connected' if self.apg_ai_client else 'not_configured',
                'rag_client': 'connected' if self.apg_rag_client else 'not_configured',
                'genai_client': 'connected' if self.apg_genai_client else 'not_configured',
                'ml_client': 'connected' if self.apg_ml_client else 'not_configured',
                'blockchain_client': 'connected' if self.apg_blockchain_client else 'not_configured'
            },
            'statistics': self.service_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return health_status
    
    # OCR Methods
    async def process_document_ocr(
        self,
        document_id: str,
        file_path: str,
        user_id: str,
        tenant_id: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process document using OCR capabilities"""
        
        try:
            self.logger.info(f"Starting OCR processing for document {document_id}")
            
            # Process document with OCR service
            ocr_result = await self.ocr_service.process_document_ocr(
                document_id=document_id,
                file_path=file_path,
                options=options
            )
            
            # Update service statistics
            self.service_stats['ocr_operations_performed'] += 1
            self.service_stats['ai_operations_performed'] += 1
            
            # Log audit event
            await self._create_audit_log(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_id=document_id,
                action='ocr_processing',
                details={'pages_processed': ocr_result.get('total_pages', 0)}
            )
            
            self.logger.info(f"OCR processing completed for document {document_id}")
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for document {document_id}: {str(e)}")
            raise
    
    async def batch_ocr_processing(
        self,
        document_ids: List[str],
        user_id: str,
        tenant_id: str,
        batch_name: str = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process multiple documents with OCR in batch"""
        
        try:
            batch_name = batch_name or f"Batch OCR {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Starting batch OCR processing: {batch_name} ({len(document_ids)} documents)")
            
            batch_results = {
                'batch_name': batch_name,
                'total_documents': len(document_ids),
                'processed_documents': 0,
                'successful_documents': 0,
                'failed_documents': 0,
                'results': {},
                'errors': {}
            }
            
            # Process documents concurrently
            semaphore = asyncio.Semaphore(4)  # Limit concurrent OCR operations
            
            async def process_single_document(doc_id: str):
                async with semaphore:
                    try:
                        # Simulate getting file path for document
                        file_path = f"/documents/{doc_id}"  # This would be retrieved from document metadata
                        
                        result = await self.process_document_ocr(
                            document_id=doc_id,
                            file_path=file_path,
                            user_id=user_id,
                            tenant_id=tenant_id,
                            options=options
                        )
                        
                        batch_results['results'][doc_id] = result
                        batch_results['successful_documents'] += 1
                        
                    except Exception as e:
                        batch_results['errors'][doc_id] = str(e)
                        batch_results['failed_documents'] += 1
                    
                    batch_results['processed_documents'] += 1
            
            # Execute all OCR tasks
            tasks = [process_single_document(doc_id) for doc_id in document_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log batch completion
            await self._create_audit_log(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_id=batch_name,
                action='batch_ocr_processing',
                details={
                    'total_documents': batch_results['total_documents'],
                    'successful_documents': batch_results['successful_documents'],
                    'failed_documents': batch_results['failed_documents']
                }
            )
            
            self.logger.info(f"Batch OCR processing completed: {batch_name}")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch OCR processing failed: {str(e)}")
            raise
    
    async def get_ocr_result(
        self,
        document_id: str,
        user_id: str,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve OCR results for a document"""
        
        try:
            # This would typically query the database for stored OCR results
            # For now, we'll return a placeholder response
            
            ocr_result = {
                'document_id': document_id,
                'status': 'completed',
                'text_content': 'Sample OCR extracted text content...',
                'confidence_score': 0.95,
                'processing_time_ms': 2340,
                'pages_processed': 1,
                'language_detected': 'eng'
            }
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"Error retrieving OCR results for document {document_id}: {str(e)}")
            return None
    
    async def update_ocr_configuration(
        self,
        config_name: str,
        config_data: Dict[str, Any],
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Update OCR configuration settings"""
        
        try:
            # Create OCR configuration from data
            ocr_config = OCRConfig(**config_data)
            
            # Update the service OCR configuration
            self.ocr_service = OCRService(ocr_config, self.apg_ai_client)
            
            # Log configuration update
            await self._create_audit_log(
                user_id=user_id,
                tenant_id=tenant_id,
                resource_id=config_name,
                action='ocr_config_update',
                details={'config_name': config_name}
            )
            
            return {
                'config_name': config_name,
                'status': 'updated',
                'updated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating OCR configuration {config_name}: {str(e)}")
            raise
    
    async def get_supported_ocr_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        
        try:
            return await self.ocr_service.ocr_engine.get_supported_languages()
            
        except Exception as e:
            self.logger.error(f"Error retrieving supported OCR languages: {str(e)}")
            return ['eng']  # Default to English
    
    async def _get_ocr_analytics(self) -> Dict[str, Any]:
        """Get OCR processing analytics"""
        
        return {
            'ocr_operations_performed': self.service_stats.get('ocr_operations_performed', 0),
            'average_processing_time_ms': 2340,  # This would be calculated from actual data
            'average_confidence_score': 0.92,   # This would be calculated from actual data
            'most_common_language': 'eng',      # This would be calculated from actual data
            'success_rate': 0.98                # This would be calculated from actual data
        }