"""
Requisitioning Sub-capability

Employee requisition management and approval workflows for procurement requests.
Handles the creation, approval, and tracking of purchase requisitions with
configurable approval workflows based on amount thresholds and organizational hierarchy.

Key Features:
- Purchase requisition creation and management
- Multi-level approval workflows with escalation
- Budget checking and encumbrance accounting
- Requisition-to-PO conversion
- Comprehensive audit trail and reporting
- Mobile-friendly requisition submission
"""

from typing import Dict, Any

# Sub-capability metadata
SUBCAPABILITY_INFO = {
	'code': 'PPR',
	'name': 'Requisitioning',
	'description': 'Employee requisition management and approval workflows',
	'version': '1.0.0',
	'models': [
		'PPRRequisition',
		'PPRRequisitionLine', 
		'PPRApprovalWorkflow',
		'PPRRequisitionComment'
	],
	'views': [
		'RequisitionView',
		'RequisitionLineView',
		'ApprovalWorkflowView',
		'RequisitionDashboardView'
	],
	'permissions': [
		'can_create_requisition',
		'can_edit_requisition',
		'can_approve_requisition',
		'can_reject_requisition',
		'can_view_all_requisitions',
		'can_override_approvals'
	]
}


def get_subcapability_info() -> Dict[str, Any]:
	"""Get Requisitioning sub-capability information"""
	return SUBCAPABILITY_INFO.copy()