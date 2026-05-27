#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Flask Blueprint
Flask-AppBuilder blueprint for web UI integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import wraps

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app, g, has_request_context, session
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, SimpleFormView
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.security.decorators import protect
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, TextAreaField, FloatField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Length, NumberRange, Optional as OptionalValidator

from .service import MDMService, MDMOperationType, MDMOperationContext
from .models import MdEntity, MdEntityVersion, MdGoldenRecord, MdDataQualityAssessment
from .models import EntityType, EntityStatus, DataQualityStatus


# Flask Blueprint
mdm_bp = Blueprint('mdm', __name__, template_folder='templates', static_folder='static')


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _object_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _first_text(candidates: List[Any], fallback: str) -> str:
    for candidate in candidates:
        text = _clean_text(candidate)
        if text:
            return text
    return fallback


def _normalise_permissions(value: Any) -> List[str]:
    default_permissions = ["mdm.read", "mdm.write"]
    if value is None:
        return default_permissions
    if isinstance(value, str):
        permissions = [permission.strip() for permission in value.replace(",", " ").split()]
        return permissions or default_permissions
    permissions: List[str] = []
    for permission in value:
        name = (
            _object_value(permission, "name")
            or _object_value(permission, "permission")
            or _object_value(permission, "role_name")
            or permission
        )
        text = _clean_text(name)
        if text:
            permissions.append(text)
    return permissions or default_permissions


def _appbuilder_user(view: Any = None) -> Any:
    appbuilder = getattr(view, "appbuilder", None)
    security_manager = getattr(appbuilder, "sm", None)
    get_user = getattr(security_manager, "get_user", None)
    if callable(get_user):
        try:
            return get_user()
        except Exception:
            return None
    return getattr(security_manager, "user", None)


def _resolve_mdm_user_context(view: Any = None) -> Dict[str, Any]:
    """Resolve MDM tenant and actor context from APG runtime request sources."""
    default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
    default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
    default_permissions = os.getenv("APG_DEFAULT_PERMISSIONS", os.getenv("APG_PERMISSIONS", "mdm.read mdm.write"))

    if not has_request_context():
        return {
            "user_id": default_user,
            "tenant_id": default_tenant,
            "permissions": _normalise_permissions(default_permissions),
            "client_ip": None,
            "user_agent": None,
        }

    request_user = getattr(request, "current_user", None)
    g_user = (
        getattr(g, "current_user", None)
        or getattr(g, "user", None)
        or getattr(g, "auth_user", None)
    )
    app_user = _appbuilder_user(view)

    tenant_id = _first_text([
        getattr(g, "tenant_id", None),
        _object_value(request_user, "tenant_id"),
        _object_value(g_user, "tenant_id"),
        _object_value(app_user, "tenant_id"),
        session.get("tenant_id"),
        request.headers.get("X-Tenant-ID"),
        request.headers.get("X-APG-Tenant-ID"),
        request.headers.get("X-Organization-ID"),
        request.args.get("tenant_id"),
        request.args.get("tenant"),
        os.getenv("APG_TENANT_ID"),
    ], default_tenant)

    user_id = _first_text([
        getattr(g, "user_id", None),
        getattr(request, "current_user_id", None),
        _object_value(request_user, "user_id"),
        _object_value(request_user, "id"),
        _object_value(request_user, "username"),
        _object_value(g_user, "user_id"),
        _object_value(g_user, "id"),
        _object_value(g_user, "username"),
        _object_value(app_user, "user_id"),
        _object_value(app_user, "id"),
        _object_value(app_user, "username"),
        session.get("user_id"),
        session.get("username"),
        request.headers.get("X-User-ID"),
        request.headers.get("X-APG-User-ID"),
        request.args.get("user_id"),
        os.getenv("APG_USER_ID"),
    ], default_user)

    permissions = _normalise_permissions(
        _object_value(request_user, "permissions")
        or _object_value(request_user, "roles")
        or _object_value(g_user, "permissions")
        or _object_value(g_user, "roles")
        or _object_value(app_user, "permissions")
        or _object_value(app_user, "roles")
        or session.get("permissions")
        or session.get("roles")
        or request.headers.get("X-APG-Permissions")
        or request.headers.get("X-APG-Roles")
        or os.getenv("APG_PERMISSIONS")
        or default_permissions
    )

    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "permissions": permissions,
        "client_ip": request.remote_addr,
        "user_agent": request.user_agent.string if request.user_agent else None,
    }


# Forms for MDM Operations

class EntityCreateForm(FlaskForm):
    """Form for creating new entities"""
    entity_type = SelectField(
        'Entity Type',
        choices=[(e.value, e.value.title()) for e in EntityType],
        validators=[DataRequired()]
    )
    entity_name = StringField(
        'Entity Name',
        validators=[DataRequired(), Length(min=1, max=255)]
    )
    entity_description = TextAreaField(
        'Description',
        validators=[OptionalValidator(), Length(max=1000)]
    )
    business_key = StringField(
        'Business Key',
        validators=[DataRequired(), Length(min=1, max=100)]
    )
    source_system = StringField(
        'Source System',
        validators=[DataRequired(), Length(min=1, max=100)]
    )
    data_classification = SelectField(
        'Data Classification',
        choices=[
            ('public', 'Public'),
            ('internal', 'Internal'),
            ('confidential', 'Confidential'),
            ('restricted', 'Restricted')
        ],
        default='internal'
    )
    attributes_json = TextAreaField(
        'Attributes (JSON)',
        validators=[OptionalValidator()],
        description='Entity attributes as JSON object'
    )
    tags = StringField(
        'Tags',
        validators=[OptionalValidator()],
        description='Comma-separated tags'
    )


class EntitySearchForm(FlaskForm):
    """Form for searching entities"""
    search_term = StringField(
        'Search Term',
        validators=[OptionalValidator()],
        description='Search in entity name and description'
    )
    entity_type = SelectField(
        'Entity Type',
        choices=[('', 'All Types')] + [(e.value, e.value.title()) for e in EntityType],
        default=''
    )
    source_system = StringField(
        'Source System',
        validators=[OptionalValidator()]
    )
    status = SelectField(
        'Status',
        choices=[('', 'All Statuses')] + [(e.value, e.value.title()) for e in EntityStatus],
        default=''
    )
    min_quality_score = FloatField(
        'Minimum Quality Score',
        validators=[OptionalValidator(), NumberRange(min=0, max=100)]
    )
    is_golden_record = SelectField(
        'Golden Record',
        choices=[('', 'All'), ('true', 'Golden Records Only'), ('false', 'Non-Golden Records')],
        default=''
    )


class QualityAssessmentForm(FlaskForm):
    """Form for quality assessment configuration"""
    entity_ids = TextAreaField(
        'Entity IDs',
        validators=[DataRequired()],
        description='One entity ID per line'
    )
    include_recommendations = BooleanField(
        'Include Recommendations',
        default=True
    )
    include_issues = BooleanField(
        'Include Issues',
        default=True
    )


# Base MDM View Class

class MDMBaseView(BaseView):
    """Base view class for MDM operations"""
    
    def __init__(self):
        super().__init__()
        self.mdm_service = None
    
    def get_mdm_service(self) -> MDMService:
        """Get MDM service instance"""
        if not self.mdm_service:
            # In production, get from application context
            self.mdm_service = current_app.config.get('MDM_SERVICE')
            if not self.mdm_service:
                raise RuntimeError("MDM service not configured")
        return self.mdm_service
    
    def get_current_user_context(self) -> Dict[str, Any]:
        """Get current user context for operations"""
        return _resolve_mdm_user_context(self)
    
    def create_operation_context(self, operation_type: MDMOperationType, 
                                entity_id: str = None, entity_type: str = None) -> MDMOperationContext:
        """Create operation context for MDM operations"""
        user_ctx = self.get_current_user_context()
        mdm_service = self.get_mdm_service()
        
        return mdm_service.create_operation_context(
            tenant_id=user_ctx['tenant_id'],
            user_id=user_ctx['user_id'],
            operation_type=operation_type,
            entity_id=entity_id,
            entity_type=entity_type,
            source_system="web_ui",
            client_ip=user_ctx.get('client_ip'),
            user_agent=user_ctx.get('user_agent')
        )


# Main Dashboard View

class MDMDashboardView(MDMBaseView):
    """Main MDM dashboard with KPIs and overview"""
    
    route_base = '/mdm'
    default_view = 'dashboard'
    
    @expose('/')
    @expose('/dashboard')
    @has_access
    def dashboard(self):
        """Main dashboard with KPIs and statistics"""
        try:
            mdm_service = self.get_mdm_service()
            user_ctx = self.get_current_user_context()
            tenant_id = user_ctx['tenant_id']
            
            # Get dashboard statistics
            stats = asyncio.run(mdm_service.db_manager.get_database_stats(tenant_id))
            health = asyncio.run(mdm_service.health_check())
            
            # Prepare dashboard data
            dashboard_data = {
                'entity_statistics': stats.get('entity_statistics', []),
                'quality_statistics': stats.get('quality_statistics', []),
                'recent_activity': stats.get('recent_activity', []),
                'health_status': health.get('status', 'unknown'),
                'generated_at': stats.get('timestamp', datetime.utcnow().isoformat())
            }
            
            # Calculate summary metrics
            total_entities = sum(stat['total_entities'] for stat in dashboard_data['entity_statistics'])
            avg_quality = sum(stat['avg_quality_score'] * stat['total_entities'] for stat in dashboard_data['entity_statistics'])
            avg_quality = avg_quality / total_entities if total_entities > 0 else 0
            golden_records = sum(stat['golden_records'] for stat in dashboard_data['entity_statistics'])
            
            summary_metrics = {
                'total_entities': total_entities,
                'average_quality_score': round(avg_quality, 2),
                'golden_records_count': golden_records,
                'health_status': dashboard_data['health_status']
            }
            
            return self.render_template(
                'mdm/dashboard.html',
                dashboard_data=dashboard_data,
                summary_metrics=summary_metrics
            )
            
        except Exception as e:
            flash(f"Error loading dashboard: {str(e)}", 'error')
            return self.render_template('mdm/error.html', error_message=str(e))


# Entity Management Views

class EntityManagementView(MDMBaseView):
    """Entity management interface"""
    
    route_base = '/mdm/entities'
    
    @expose('/')
    @has_access
    def list_entities(self):
        """List entities with search and filtering"""
        form = EntitySearchForm()
        entities = []
        pagination_info = {}
        
        try:
            mdm_service = self.get_mdm_service()
            user_ctx = self.get_current_user_context()
            
            # Build search criteria from form
            search_criteria = {}
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            
            if form.validate_on_submit() or request.args:
                if form.search_term.data or request.args.get('search_term'):
                    search_term = form.search_term.data or request.args.get('search_term')
                    search_criteria['entity_name'] = search_term
                
                if form.entity_type.data or request.args.get('entity_type'):
                    entity_type = form.entity_type.data or request.args.get('entity_type')
                    if entity_type:
                        search_criteria['entity_type'] = entity_type
                
                if form.source_system.data or request.args.get('source_system'):
                    search_criteria['source_system'] = form.source_system.data or request.args.get('source_system')
                
                if form.status.data or request.args.get('status'):
                    status = form.status.data or request.args.get('status')
                    if status:
                        search_criteria['status'] = status
                
                if form.min_quality_score.data is not None or request.args.get('min_quality_score'):
                    min_score = form.min_quality_score.data or float(request.args.get('min_quality_score', 0))
                    search_criteria['min_quality_score'] = min_score
                
                is_golden = form.is_golden_record.data or request.args.get('is_golden_record')
                if is_golden == 'true':
                    search_criteria['is_golden_record'] = True
                elif is_golden == 'false':
                    search_criteria['is_golden_record'] = False
            
            # Set pagination
            search_criteria['limit'] = per_page
            search_criteria['offset'] = (page - 1) * per_page
            search_criteria['sort_by'] = request.args.get('sort_by', 'updated_at')
            search_criteria['sort_order'] = request.args.get('sort_order', 'desc')
            
            # Execute search
            import asyncio
            result = asyncio.run(mdm_service.entity_service.search_entities(
                user_ctx['tenant_id'], search_criteria
            ))
            
            if result['status'] == 'success':
                entities = result['entities']
                pagination_info = result['pagination']
            else:
                flash(f"Search error: {result['message']}", 'error')
        
        except Exception as e:
            flash(f"Error searching entities: {str(e)}", 'error')
        
        return self.render_template(
            'mdm/entity_list.html',
            form=form,
            entities=entities,
            pagination=pagination_info,
            current_page=page,
            per_page=per_page
        )
    
    @expose('/create', methods=['GET', 'POST'])
    @has_access
    def create_entity(self):
        """Create new entity"""
        form = EntityCreateForm()
        
        if form.validate_on_submit():
            try:
                mdm_service = self.get_mdm_service()
                
                # Parse attributes JSON
                attributes = {}
                if form.attributes_json.data:
                    try:
                        attributes = json.loads(form.attributes_json.data)
                    except json.JSONDecodeError:
                        flash("Invalid JSON in attributes field", 'error')
                        return self.render_template('mdm/entity_create.html', form=form)
                
                # Parse tags
                tags = []
                if form.tags.data:
                    tags = [tag.strip() for tag in form.tags.data.split(',') if tag.strip()]
                
                # Create entity data
                from .models import MdEntityCreate, EntityType, EntityStatus
                entity_data = MdEntityCreate(
                    entity_type=EntityType(form.entity_type.data),
                    entity_name=form.entity_name.data,
                    entity_description=form.entity_description.data,
                    business_key=form.business_key.data,
                    source_system=form.source_system.data,
                    status=EntityStatus.ACTIVE,
                    attributes=attributes,
                    tags=tags,
                    data_classification=form.data_classification.data,
                    tenant_id=self.get_current_user_context()['tenant_id']
                )
                
                # Create operation context
                context = self.create_operation_context(
                    MDMOperationType.CREATE_ENTITY,
                    entity_type=form.entity_type.data
                )
                
                # Create entity
                import asyncio
                result = asyncio.run(mdm_service.entity_service.create_entity(entity_data, context))
                
                if result['status'] == 'success':
                    flash(f"Entity created successfully (ID: {result['entity_id']})", 'success')
                    return redirect(url_for('EntityManagementView.view_entity', entity_id=result['entity_id']))
                else:
                    flash(f"Error creating entity: {result['message']}", 'error')
                    
            except Exception as e:
                flash(f"Error creating entity: {str(e)}", 'error')
        
        return self.render_template('mdm/entity_create.html', form=form)
    
    @expose('/view/<entity_id>')
    @has_access
    def view_entity(self, entity_id):
        """View entity details"""
        try:
            mdm_service = self.get_mdm_service()
            user_ctx = self.get_current_user_context()
            
            import asyncio
            result = asyncio.run(mdm_service.entity_service.get_entity(
                entity_id, user_ctx['tenant_id'],
                include_versions=True,
                include_quality=True,
                include_cross_refs=True
            ))
            
            if result['status'] == 'success':
                entity = result['entity']
                return self.render_template('mdm/entity_detail.html', entity=entity)
            else:
                flash(f"Entity not found: {result['message']}", 'error')
                return redirect(url_for('EntityManagementView.list_entities'))
                
        except Exception as e:
            flash(f"Error loading entity: {str(e)}", 'error')
            return redirect(url_for('EntityManagementView.list_entities'))


# Quality Management Views

class QualityManagementView(MDMBaseView):
    """Data quality management interface"""
    
    route_base = '/mdm/quality'
    
    @expose('/')
    @has_access
    def quality_dashboard(self):
        """Quality assessment dashboard"""
        try:
            mdm_service = self.get_mdm_service()
            user_ctx = self.get_current_user_context()
            
            # Get quality statistics
            import asyncio
            stats = asyncio.run(mdm_service.db_manager.get_database_stats(user_ctx['tenant_id']))
            
            quality_stats = stats.get('quality_statistics', [])
            entity_stats = stats.get('entity_statistics', [])
            
            # Calculate quality metrics
            total_assessments = sum(stat['assessment_count'] for stat in quality_stats)
            quality_distribution = {stat['quality_status']: stat['assessment_count'] for stat in quality_stats}
            avg_quality_by_type = {stat['entity_type']: stat['avg_quality_score'] for stat in entity_stats}
            
            quality_data = {
                'total_assessments': total_assessments,
                'quality_distribution': quality_distribution,
                'avg_quality_by_type': avg_quality_by_type,
                'recent_assessments': []  # Would fetch recent assessments
            }
            
            return self.render_template('mdm/quality_dashboard.html', quality_data=quality_data)
            
        except Exception as e:
            flash(f"Error loading quality dashboard: {str(e)}", 'error')
            return self.render_template('mdm/error.html', error_message=str(e))
    
    @expose('/assess', methods=['GET', 'POST'])
    @has_access
    def assess_quality(self):
        """Run quality assessment"""
        form = QualityAssessmentForm()
        results = []
        
        if form.validate_on_submit():
            try:
                mdm_service = self.get_mdm_service()
                user_ctx = self.get_current_user_context()
                
                # Parse entity IDs
                entity_ids = [eid.strip() for eid in form.entity_ids.data.split('\n') if eid.strip()]
                
                if not entity_ids:
                    flash("No entity IDs provided", 'error')
                    return self.render_template('mdm/quality_assess.html', form=form)
                
                # Run quality assessment for each entity
                import asyncio
                for entity_id in entity_ids:
                    try:
                        # Get entity data
                        entity_result = asyncio.run(mdm_service.entity_service.get_entity(
                            entity_id, user_ctx['tenant_id']
                        ))
                        
                        if entity_result['status'] == 'success':
                            entity_data = entity_result['entity']
                            
                            # Run quality assessment
                            quality_result = asyncio.run(mdm_service.quality_service.assess_quality(
                                entity_id, user_ctx['tenant_id'],
                                entity_data['attributes'], entity_data['entity_type']
                            ))
                            
                            results.append({
                                'entity_id': entity_id,
                                'entity_name': entity_data['entity_name'],
                                'quality_result': quality_result
                            })
                        else:
                            results.append({
                                'entity_id': entity_id,
                                'error': entity_result['message']
                            })
                    except Exception as e:
                        results.append({
                            'entity_id': entity_id,
                            'error': str(e)
                        })
                
                flash(f"Quality assessment completed for {len(results)} entities", 'success')
                
            except Exception as e:
                flash(f"Error running quality assessment: {str(e)}", 'error')
        
        return self.render_template('mdm/quality_assess.html', form=form, results=results)


# Registration function for Flask-AppBuilder

def register_mdm_views(appbuilder, mdm_service: MDMService):
    """Register MDM views with Flask-AppBuilder"""
    
    # Store MDM service in app config
    appbuilder.app.config['MDM_SERVICE'] = mdm_service
    
    # Register blueprint
    appbuilder.app.register_blueprint(mdm_bp)
    
    # Register views
    appbuilder.add_view(
        MDMDashboardView,
        "Dashboard",
        icon="fa-dashboard",
        category="Master Data Management"
    )
    
    appbuilder.add_view(
        EntityManagementView,
        "Entities",
        icon="fa-database",
        category="Master Data Management"
    )
    
    appbuilder.add_view(
        QualityManagementView,
        "Data Quality",
        icon="fa-check-circle",
        category="Master Data Management"
    )
    
    # Add menu separator
    appbuilder.add_separator("Master Data Management")


# Template helpers
@mdm_bp.app_template_filter('quality_status_badge')
def quality_status_badge(status):
    """Generate Bootstrap badge for quality status"""
    badge_map = {
        'excellent': 'success',
        'good': 'success',
        'fair': 'warning',
        'poor': 'danger',
        'critical': 'danger'
    }
    badge_class = badge_map.get(status.lower(), 'secondary')
    return f'<span class="badge badge-{badge_class}">{status.title()}</span>'


@mdm_bp.app_template_filter('entity_status_badge')
def entity_status_badge(status):
    """Generate Bootstrap badge for entity status"""
    badge_map = {
        'active': 'success',
        'inactive': 'secondary',
        'pending': 'warning',
        'merged': 'info',
        'deleted': 'danger',
        'archived': 'secondary'
    }
    badge_class = badge_map.get(status.lower(), 'secondary')
    return f'<span class="badge badge-{badge_class}">{status.title()}</span>'


@mdm_bp.app_template_filter('quality_score_progress')
def quality_score_progress(score):
    """Generate progress bar for quality score"""
    if score >= 95:
        bar_class = 'success'
    elif score >= 80:
        bar_class = 'success'
    elif score >= 60:
        bar_class = 'warning'
    elif score >= 40:
        bar_class = 'danger'
    else:
        bar_class = 'danger'
    
    return f'''
    <div class="progress" style="height: 20px;">
        <div class="progress-bar bg-{bar_class}" role="progressbar" 
             style="width: {score}%" aria-valuenow="{score}" 
             aria-valuemin="0" aria-valuemax="100">
            {score:.1f}%
        </div>
    </div>
    '''


# Export main components
__all__ = [
    'mdm_bp', 'register_mdm_views',
    'MDMBaseView', 'MDMDashboardView', 'EntityManagementView', 'QualityManagementView',
    'EntityCreateForm', 'EntitySearchForm', 'QualityAssessmentForm'
]
