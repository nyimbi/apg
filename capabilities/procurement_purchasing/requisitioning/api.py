"""
Requisitioning API

REST API endpoints for requisition management including CRUD operations,
approval workflows, and integration endpoints.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, date
from decimal import Decimal
from flask import Blueprint, request, jsonify, g
from flask_restful import Api, Resource, reqparse
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access_api

from .models import PPRRequisition, PPRRequisitionLine, PPRApprovalWorkflow, PPRRequisitionComment
from .service import RequisitioningService


class RequisitionApi(BaseApi):
	"""REST API for requisition management"""
	
	resource_name = 'requisition'
	datamodel = SQLAInterface(PPRRequisition)
	
	@expose('/list')
	@has_access_api
	def list(self):
		"""List requisitions with optional filtering"""
		
		# Get query parameters
		args = request.args
		page = int(args.get('page', 1))
		page_size = int(args.get('page_size', 20))
		
		# Build search parameters
		search_params = {}
		if args.get('status'):
			search_params['status'] = args.get('status')
		if args.get('requestor_id'):
			search_params['requestor_id'] = args.get('requestor_id')
		if args.get('department'):
			search_params['department'] = args.get('department')
		if args.get('date_from'):
			search_params['date_from'] = datetime.strptime(args.get('date_from'), '%Y-%m-%d').date()
		if args.get('date_to'):
			search_params['date_to'] = datetime.strptime(args.get('date_to'), '%Y-%m-%d').date()
		if args.get('min_amount'):
			search_params['min_amount'] = Decimal(args.get('min_amount'))
		if args.get('max_amount'):
			search_params['max_amount'] = Decimal(args.get('max_amount'))
		if args.get('search_text'):
			search_params['search_text'] = args.get('search_text')
		
		try:
			service = RequisitioningService(self.get_tenant_id())
			
			# Get requisitions
			requisitions = service.search_requisitions(search_params, limit=page_size * page)
			
			# Apply pagination
			total_count = len(requisitions)
			start_idx = (page - 1) * page_size
			end_idx = start_idx + page_size
			page_requisitions = requisitions[start_idx:end_idx]
			
			# Convert to dict
			result = []
			for req in page_requisitions:
				result.append(self._requisition_to_dict(req))
			
			return jsonify({
				'success': True,
				'data': result,
				'pagination': {
					'page': page,
					'page_size': page_size,
					'total_count': total_count,
					'total_pages': (total_count + page_size - 1) // page_size
				}
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>')
	@has_access_api
	def get(self, requisition_id: str):
		"""Get requisition by ID"""
		
		try:
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.get_requisition_by_id(requisition_id)
			
			if not requisition:
				return jsonify({
					'success': False,
					'error': 'Requisition not found'
				}), 404
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition, include_lines=True, include_workflow=True)
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/', methods=['POST'])
	@has_access_api
	def post(self):
		"""Create new requisition"""
		
		try:
			data = request.get_json()
			if not data:
				return jsonify({
					'success': False,
					'error': 'Request data is required'
				}), 400
			
			# Validate required fields
			required_fields = ['title', 'requestor_name', 'required_date']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Field {field} is required'
					}), 400
			
			# Convert date strings
			if 'required_date' in data:
				data['required_date'] = datetime.strptime(data['required_date'], '%Y-%m-%d').date()
			
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.create_requisition(data, self.get_current_user_id())
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition, include_lines=True),
				'message': f'Requisition {requisition.requisition_number} created successfully'
			}), 201
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>', methods=['PUT'])
	@has_access_api
	def put(self, requisition_id: str):
		"""Update requisition"""
		
		try:
			data = request.get_json()
			if not data:
				return jsonify({
					'success': False,
					'error': 'Request data is required'
				}), 400
			
			# Convert date strings if present
			if 'required_date' in data:
				data['required_date'] = datetime.strptime(data['required_date'], '%Y-%m-%d').date()
			
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.update_requisition(requisition_id, data, self.get_current_user_id())
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition, include_lines=True),
				'message': f'Requisition {requisition.requisition_number} updated successfully'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/submit', methods=['POST'])
	@has_access_api
	def submit(self, requisition_id: str):
		"""Submit requisition for approval"""
		
		try:
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.submit_requisition(requisition_id, self.get_current_user_id())
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition),
				'message': f'Requisition {requisition.requisition_number} submitted for approval'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/approve', methods=['POST'])
	@has_access_api
	def approve(self, requisition_id: str):
		"""Approve requisition"""
		
		try:
			data = request.get_json() or {}
			comments = data.get('comments')
			
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.approve_requisition(requisition_id, self.get_current_user_id(), comments)
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition),
				'message': f'Requisition {requisition.requisition_number} approved'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/reject', methods=['POST'])
	@has_access_api
	def reject(self, requisition_id: str):
		"""Reject requisition"""
		
		try:
			data = request.get_json()
			if not data or not data.get('reason'):
				return jsonify({
					'success': False,
					'error': 'Rejection reason is required'
				}), 400
			
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.reject_requisition(requisition_id, self.get_current_user_id(), data['reason'])
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition),
				'message': f'Requisition {requisition.requisition_number} rejected'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/cancel', methods=['POST'])
	@has_access_api
	def cancel(self, requisition_id: str):
		"""Cancel requisition"""
		
		try:
			data = request.get_json() or {}
			reason = data.get('reason', 'Cancelled via API')
			
			service = RequisitioningService(self.get_tenant_id())
			requisition = service.cancel_requisition(requisition_id, self.get_current_user_id(), reason)
			
			return jsonify({
				'success': True,
				'data': self._requisition_to_dict(requisition),
				'message': f'Requisition {requisition.requisition_number} cancelled'
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/comments')
	@has_access_api
	def get_comments(self, requisition_id: str):
		"""Get requisition comments"""
		
		try:
			include_internal = request.args.get('include_internal', 'true').lower() == 'true'
			
			service = RequisitioningService(self.get_tenant_id())
			comments = service.get_requisition_comments(requisition_id, include_internal)
			
			result = []
			for comment in comments:
				result.append({
					'comment_id': comment.comment_id,
					'comment_text': comment.comment_text,
					'comment_type': comment.comment_type,
					'is_internal': comment.is_internal,
					'author_name': comment.author_name,
					'author_role': comment.author_role,
					'comment_date': comment.comment_date.isoformat(),
					'requires_response': comment.requires_response,
					'response_count': comment.response_count
				})
			
			return jsonify({
				'success': True,
				'data': result
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/<string:requisition_id>/comments', methods=['POST'])
	@has_access_api
	def add_comment(self, requisition_id: str):
		"""Add comment to requisition"""
		
		try:
			data = request.get_json()
			if not data or not data.get('comment_text'):
				return jsonify({
					'success': False,
					'error': 'Comment text is required'
				}), 400
			
			service = RequisitioningService(self.get_tenant_id())
			comment = service.add_requisition_comment(
				requisition_id,
				data['comment_text'],
				self.get_current_user_id(),
				data.get('comment_type', 'General'),
				data.get('is_internal', True)
			)
			
			return jsonify({
				'success': True,
				'data': {
					'comment_id': comment.comment_id,
					'comment_text': comment.comment_text,
					'comment_type': comment.comment_type,
					'author_name': comment.author_name,
					'comment_date': comment.comment_date.isoformat()
				},
				'message': 'Comment added successfully'
			}), 201
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/my_approvals')
	@has_access_api
	def my_approvals(self):
		"""Get requisitions pending approval by current user"""
		
		try:
			service = RequisitioningService(self.get_tenant_id())
			requisitions = service.get_requisitions_for_approval(self.get_current_user_id())
			
			result = []
			for req in requisitions:
				result.append(self._requisition_to_dict(req))
			
			return jsonify({
				'success': True,
				'data': result,
				'count': len(result)
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/my_requisitions')
	@has_access_api
	def my_requisitions(self):
		"""Get current user's requisitions"""
		
		try:
			limit = int(request.args.get('limit', 50))
			
			service = RequisitioningService(self.get_tenant_id())
			requisitions = service.get_requisitions_by_requestor(self.get_current_user_id(), limit)
			
			result = []
			for req in requisitions:
				result.append(self._requisition_to_dict(req))
			
			return jsonify({
				'success': True,
				'data': result,
				'count': len(result)
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	@expose('/metrics')
	@has_access_api
	def metrics(self):
		"""Get requisition metrics"""
		
		try:
			# Get date range from query parameters
			date_from = request.args.get('date_from')
			date_to = request.args.get('date_to')
			
			# Convert to date objects
			if date_from:
				date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
			if date_to:
				date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
			
			service = RequisitioningService(self.get_tenant_id())
			metrics = service.get_requisition_metrics(date_from, date_to)
			
			return jsonify({
				'success': True,
				'data': metrics
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	def _requisition_to_dict(self, requisition: PPRRequisition, include_lines: bool = False, 
							include_workflow: bool = False) -> Dict[str, Any]:
		"""Convert requisition model to dictionary"""
		
		result = {
			'requisition_id': requisition.requisition_id,
			'requisition_number': requisition.requisition_number,
			'title': requisition.title,
			'description': requisition.description,
			'business_justification': requisition.business_justification,
			'requestor_id': requisition.requestor_id,
			'requestor_name': requisition.requestor_name,
			'requestor_email': requisition.requestor_email,
			'department': requisition.department,
			'cost_center': requisition.cost_center,
			'request_date': requisition.request_date.isoformat(),
			'required_date': requisition.required_date.isoformat(),
			'delivery_location': requisition.delivery_location,
			'status': requisition.status,
			'workflow_status': requisition.workflow_status,
			'priority': requisition.priority,
			'currency_code': requisition.currency_code,
			'subtotal_amount': float(requisition.subtotal_amount),
			'tax_amount': float(requisition.tax_amount),
			'total_amount': float(requisition.total_amount),
			'budget_account_id': requisition.budget_account_id,
			'budget_checked': requisition.budget_checked,
			'budget_available': float(requisition.budget_available) if requisition.budget_available else 0,
			'approval_level': requisition.approval_level,
			'current_approver_id': requisition.current_approver_id,
			'current_approver_name': requisition.current_approver_name,
			'approved': requisition.approved,
			'approved_by': requisition.approved_by,
			'approved_date': requisition.approved_date.isoformat() if requisition.approved_date else None,
			'approved_amount': float(requisition.approved_amount),
			'rejected': requisition.rejected,
			'rejected_by': requisition.rejected_by,
			'rejected_date': requisition.rejected_date.isoformat() if requisition.rejected_date else None,
			'rejection_reason': requisition.rejection_reason,
			'submitted': requisition.submitted,
			'submitted_date': requisition.submitted_date.isoformat() if requisition.submitted_date else None,
			'converted_to_po': requisition.converted_to_po,
			'purchase_order_id': requisition.purchase_order_id,
			'conversion_date': requisition.conversion_date.isoformat() if requisition.conversion_date else None,
			'project_id': requisition.project_id,
			'project_name': requisition.project_name,
			'activity_code': requisition.activity_code,
			'rush_order': requisition.rush_order,
			'drop_ship': requisition.drop_ship,
			'special_instructions': requisition.special_instructions,
			'attachment_count': requisition.attachment_count,
			'notes': requisition.notes,
			'created_date': requisition.created_date.isoformat(),
			'modified_date': requisition.modified_date.isoformat()
		}
		
		# Include lines if requested
		if include_lines and requisition.lines:
			result['lines'] = []
			for line in requisition.lines:
				result['lines'].append({
					'line_id': line.line_id,
					'line_number': line.line_number,
					'description': line.description,
					'detailed_specification': line.detailed_specification,
					'item_code': line.item_code,
					'item_description': line.item_description,
					'quantity_requested': float(line.quantity_requested),
					'unit_of_measure': line.unit_of_measure,
					'unit_price': float(line.unit_price),
					'line_amount': float(line.line_amount),
					'tax_code': line.tax_code,
					'tax_rate': float(line.tax_rate),
					'tax_amount': float(line.tax_amount),
					'is_tax_inclusive': line.is_tax_inclusive,
					'gl_account_id': line.gl_account_id,
					'cost_center': line.cost_center,
					'department': line.department,
					'project_id': line.project_id,
					'required_date': line.required_date.isoformat() if line.required_date else None,
					'delivery_location': line.delivery_location,
					'preferred_vendor_id': line.preferred_vendor_id,
					'preferred_vendor_name': line.preferred_vendor_name,
					'is_asset': line.is_asset,
					'is_service': line.is_service,
					'warranty_required': line.warranty_required,
					'notes': line.notes
				})
		
		# Include workflow if requested
		if include_workflow and requisition.workflow_steps:
			result['workflow_steps'] = []
			for step in sorted(requisition.workflow_steps, key=lambda x: x.step_order):
				result['workflow_steps'].append({
					'workflow_id': step.workflow_id,
					'step_order': step.step_order,
					'step_name': step.step_name,
					'approver_id': step.approver_id,
					'approver_name': step.approver_name,
					'approver_role': step.approver_role,
					'status': step.status,
					'required': step.required,
					'approval_limit': float(step.approval_limit) if step.approval_limit else None,
					'assigned_date': step.assigned_date.isoformat(),
					'due_date': step.due_date.isoformat() if step.due_date else None,
					'approved': step.approved,
					'approved_date': step.approved_date.isoformat() if step.approved_date else None,
					'rejected': step.rejected,
					'rejected_date': step.rejected_date.isoformat() if step.rejected_date else None,
					'comments': step.comments,
					'rejection_reason': step.rejection_reason
				})
		
		return result
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement tenant resolution
		return "default_tenant"
	
	def get_current_user_id(self) -> str:
		"""Get current user ID"""
		# TODO: Get from Flask-Login or similar
		return "current_user"


def register_api_views(appbuilder: AppBuilder):
	"""Register API views with Flask-AppBuilder"""
	
	# Register the main API
	appbuilder.add_api(RequisitionApi)
	
	# Create API blueprint
	api_bp = Blueprint('requisitioning_api', __name__, url_prefix='/api/v1/requisitioning')
	api = Api(api_bp)
	
	# Additional REST resources can be added here
	
	# Register the blueprint
	appbuilder.get_app.register_blueprint(api_bp)