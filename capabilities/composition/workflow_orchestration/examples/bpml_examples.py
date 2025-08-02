"""
APG Workflow Orchestration BPML Examples

Example BPML workflows in various formats (XML, JSON, simplified) demonstrating
the comprehensive capabilities of the BPML parser and execution engine.

© 2025 Datacraft. All rights reserved.
Author: APG Development Team
"""

# Example 1: Full BPML 1.0 XML Format - Purchase Order Approval Process
PURCHASE_ORDER_APPROVAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  xmlns:apg="http://apg.datacraft.co.ke/workflow"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  id="PurchaseOrderDefinitions"
                  targetNamespace="http://apg.datacraft.co.ke/workflow/purchase-order">
  
  <bpmn:process id="PurchaseOrderApproval" name="Purchase Order Approval Process" isExecutable="true">
    <bpmn:documentation>
      Enterprise purchase order approval workflow with multi-level approvals,
      vendor validation, and automated compliance checks.
    </bpmn:documentation>
    
    <!-- Start Event -->
    <bpmn:startEvent id="StartEvent_PO" name="PO Submitted">
      <bpmn:outgoing>Flow_ToValidation</bpmn:outgoing>
    </bpmn:startEvent>
    
    <!-- Automated Validation -->
    <bpmn:serviceTask id="Task_ValidatePO" name="Validate Purchase Order" apg:capability="ai_orchestration">
      <bpmn:incoming>Flow_ToValidation</bpmn:incoming>
      <bpmn:outgoing>Flow_ToAmountGateway</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:operation>validate_purchase_order</apg:operation>
          <apg:requirements>
            <apg:field name="amount" required="true"/>
            <apg:field name="vendor" required="true"/>
            <apg:field name="department" required="true"/>
          </apg:requirements>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:serviceTask>
    
    <!-- Amount-based Gateway -->
    <bpmn:exclusiveGateway id="Gateway_Amount" name="Check Amount">
      <bpmn:incoming>Flow_ToAmountGateway</bpmn:incoming>
      <bpmn:outgoing>Flow_ToManagerApproval</bpmn:outgoing>
      <bpmn:outgoing>Flow_ToDirectorApproval</bpmn:outgoing>
      <bpmn:outgoing>Flow_ToCFOApproval</bpmn:outgoing>
    </bpmn:exclusiveGateway>
    
    <!-- Manager Approval (< $5,000) -->
    <bpmn:userTask id="Task_ManagerApproval" name="Manager Approval" apg:assignee="manager">
      <bpmn:incoming>Flow_ToManagerApproval</bpmn:incoming>
      <bpmn:outgoing>Flow_ToVendorCheck</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:form>
            <apg:field name="approval_decision" type="select" options="approve,reject,request_changes"/>
            <apg:field name="comments" type="text" multiline="true"/>
          </apg:form>
          <apg:sla hours="24"/>
          <apg:escalation>
            <apg:level1 hours="12" assignee="senior_manager"/>
            <apg:level2 hours="20" assignee="director"/>
          </apg:escalation>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:userTask>
    
    <!-- Director Approval ($5,000 - $25,000) -->
    <bpmn:userTask id="Task_DirectorApproval" name="Director Approval" apg:assignee="director">
      <bpmn:incoming>Flow_ToDirectorApproval</bpmn:incoming>
      <bpmn:outgoing>Flow_ToVendorCheck</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:form>
            <apg:field name="approval_decision" type="select" options="approve,reject,escalate"/>
            <apg:field name="budget_impact" type="text"/>
            <apg:field name="justification" type="text" multiline="true"/>
          </apg:form>
          <apg:sla hours="48"/>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:userTask>
    
    <!-- CFO Approval (> $25,000) -->
    <bpmn:userTask id="Task_CFOApproval" name="CFO Approval" apg:assignee="cfo">
      <bpmn:incoming>Flow_ToCFOApproval</bpmn:incoming>
      <bpmn:outgoing>Flow_ToVendorCheck</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:form>
            <apg:field name="approval_decision" type="select" options="approve,reject,board_review"/>
            <apg:field name="financial_impact" type="number"/>
            <apg:field name="strategic_alignment" type="text" multiline="true"/>
          </apg:form>
          <apg:sla hours="72"/>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:userTask>
    
    <!-- Parallel Gateway for Vendor Validation -->
    <bpmn:parallelGateway id="Gateway_ParallelStart" name="Start Parallel Tasks">
      <bpmn:incoming>Flow_ToVendorCheck</bpmn:incoming>
      <bpmn:outgoing>Flow_ToVendorValidation</bpmn:outgoing>
      <bpmn:outgoing>Flow_ToComplianceCheck</bpmn:outgoing>
    </bpmn:parallelGateway>
    
    <!-- Vendor Validation -->
    <bpmn:serviceTask id="Task_VendorValidation" name="Validate Vendor" apg:capability="auth_rbac">
      <bpmn:incoming>Flow_ToVendorValidation</bpmn:incoming>
      <bpmn:outgoing>Flow_ToParallelJoin</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:operation>validate_vendor_credentials</apg:operation>
          <apg:timeout>300</apg:timeout>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:serviceTask>
    
    <!-- Compliance Check -->
    <bpmn:serviceTask id="Task_ComplianceCheck" name="Compliance Check" apg:capability="audit_compliance">
      <bpmn:incoming>Flow_ToComplianceCheck</bpmn:incoming>
      <bpmn:outgoing>Flow_ToParallelJoin</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:operation>compliance_validation</apg:operation>
          <apg:requirements>["SOX", "GDPR", "PCI-DSS"]</apg:requirements>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:serviceTask>
    
    <!-- Parallel Join -->
    <bpmn:parallelGateway id="Gateway_ParallelJoin" name="Join Parallel Tasks">
      <bpmn:incoming>Flow_ToParallelJoin</bpmn:incoming>
      <bpmn:incoming>Flow_ToParallelJoin</bpmn:incoming>
      <bpmn:outgoing>Flow_ToCreatePO</bpmn:outgoing>
    </bpmn:parallelGateway>
    
    <!-- Create Purchase Order -->
    <bpmn:serviceTask id="Task_CreatePO" name="Create Purchase Order" apg:capability="document_management">
      <bpmn:incoming>Flow_ToCreatePO</bpmn:incoming>
      <bpmn:outgoing>Flow_ToEnd</bpmn:outgoing>
      <bpmn:extensionElements>
        <apg:taskConfiguration>
          <apg:operation>create_purchase_order_document</apg:operation>
          <apg:template>po_template_v2</apg:template>
        </apg:taskConfiguration>
      </bpmn:extensionElements>
    </bpmn:serviceTask>
    
    <!-- End Event -->
    <bpmn:endEvent id="EndEvent_POComplete" name="PO Approved">
      <bpmn:incoming>Flow_ToEnd</bpmn:incoming>
    </bpmn:endEvent>
    
    <!-- Sequence Flows -->
    <bpmn:sequenceFlow id="Flow_ToValidation" sourceRef="StartEvent_PO" targetRef="Task_ValidatePO"/>
    <bpmn:sequenceFlow id="Flow_ToAmountGateway" sourceRef="Task_ValidatePO" targetRef="Gateway_Amount"/>
    
    <bpmn:sequenceFlow id="Flow_ToManagerApproval" sourceRef="Gateway_Amount" targetRef="Task_ManagerApproval">
      <bpmn:conditionExpression>amount &lt; 5000</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    
    <bpmn:sequenceFlow id="Flow_ToDirectorApproval" sourceRef="Gateway_Amount" targetRef="Task_DirectorApproval">
      <bpmn:conditionExpression>amount >= 5000 and amount &lt; 25000</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    
    <bpmn:sequenceFlow id="Flow_ToCFOApproval" sourceRef="Gateway_Amount" targetRef="Task_CFOApproval">
      <bpmn:conditionExpression>amount >= 25000</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    
    <bpmn:sequenceFlow id="Flow_ToVendorCheck" sourceRef="Task_ManagerApproval" targetRef="Gateway_ParallelStart"/>
    <bpmn:sequenceFlow id="Flow_ToVendorCheck" sourceRef="Task_DirectorApproval" targetRef="Gateway_ParallelStart"/>
    <brmn:sequenceFlow id="Flow_ToVendorCheck" sourceRef="Task_CFOApproval" targetRef="Gateway_ParallelStart"/>
    
    <bpmn:sequenceFlow id="Flow_ToVendorValidation" sourceRef="Gateway_ParallelStart" targetRef="Task_VendorValidation"/>
    <bpmn:sequenceFlow id="Flow_ToComplianceCheck" sourceRef="Gateway_ParallelStart" targetRef="Task_ComplianceCheck"/>
    
    <bpmn:sequenceFlow id="Flow_ToParallelJoin" sourceRef="Task_VendorValidation" targetRef="Gateway_ParallelJoin"/>
    <bpmn:sequenceFlow id="Flow_ToParallelJoin" sourceRef="Task_ComplianceCheck" targetRef="Gateway_ParallelJoin"/>
    
    <bpmn:sequenceFlow id="Flow_ToCreatePO" sourceRef="Gateway_ParallelJoin" targetRef="Task_CreatePO"/>
    <bpmn:sequenceFlow id="Flow_ToEnd" sourceRef="Task_CreatePO" targetRef="EndEvent_POComplete"/>
  </bpmn:process>
</bpmn:definitions>"""

# Example 2: Simplified JSON BPML Format - Employee Onboarding
EMPLOYEE_ONBOARDING_JSON = """{
  "id": "EmployeeOnboarding",
  "name": "Employee Onboarding Process",
  "version": "2.1",
  "documentation": "Comprehensive employee onboarding workflow with IT setup, HR orientation, and manager introduction.",
  "is_executable": true,
  "variables": {
    "employee_type": "full_time",
    "department": "",
    "manager_id": "",
    "start_date": "",
    "security_clearance": "standard"
  },
  "elements": [
    {
      "id": "start_onboarding",
      "name": "New Employee Hired",
      "type": "start",
      "attributes": {
        "trigger_type": "api",
        "data_schema": "employee_hire_event"
      }
    },
    {
      "id": "create_employee_profile",
      "name": "Create Employee Profile",
      "type": "serviceTask",
      "attributes": {
        "capability": "document_management",
        "operation": "create_employee_record",
        "timeout": 300
      },
      "metadata": {
        "estimated_duration": 15,
        "automation_level": "full"
      }
    },
    {
      "id": "setup_it_accounts",
      "name": "Setup IT Accounts",
      "type": "serviceTask",
      "attributes": {
        "capability": "auth_rbac",
        "operation": "provision_user_accounts",
        "async": true
      },
      "metadata": {
        "estimated_duration": 30,
        "critical_path": true
      }
    },
    {
      "id": "security_background_check",
      "name": "Security Background Check",
      "type": "userTask",
      "attributes": {
        "assignee": "security_team",
        "candidate_groups": ["security", "hr"],
        "form_definition": {
          "fields": [
            {"name": "background_check_status", "type": "select", "options": ["approved", "pending", "rejected"]},
            {"name": "clearance_level", "type": "select", "options": ["standard", "confidential", "secret"]},
            {"name": "notes", "type": "textarea", "required": false}
          ]
        }
      },
      "metadata": {
        "sla_hours": 72,
        "priority": "high",
        "escalation": {
          "level1": {"hours": 24, "assignee": "security_manager"},
          "level2": {"hours": 48, "assignee": "ciso"}
        }
      }
    },
    {
      "id": "check_clearance_level",
      "name": "Check Security Clearance",
      "type": "exclusiveGateway",
      "attributes": {
        "gateway_direction": "diverging"
      }
    },
    {
      "id": "standard_orientation",
      "name": "Standard Orientation",
      "type": "userTask",
      "attributes": {
        "assignee": "hr_coordinator",
        "form_definition": {
          "fields": [
            {"name": "orientation_completed", "type": "checkbox"},
            {"name": "handbook_acknowledged", "type": "checkbox"},
            {"name": "benefits_explained", "type": "checkbox"},
            {"name": "feedback", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 8,
        "location": "main_office"
      }
    },
    {
      "id": "enhanced_security_briefing",
      "name": "Enhanced Security Briefing",
      "type": "userTask",
      "attributes": {
        "assignee": "security_officer",
        "form_definition": {
          "fields": [
            {"name": "security_training_completed", "type": "checkbox"},
            {"name": "nda_signed", "type": "checkbox"},
            {"name": "access_agreements_signed", "type": "checkbox"},
            {"name": "badge_issued", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 4,
        "location": "secure_facility",
        "prerequisites": ["security_clearance_approved"]
      }
    },
    {
      "id": "join_orientation_paths",
      "name": "Join Orientation Paths",
      "type": "exclusiveGateway",
      "attributes": {
        "gateway_direction": "converging"
      }
    },
    {
      "id": "manager_introduction",
      "name": "Manager Introduction",
      "type": "userTask",
      "attributes": {
        "assignee": "${manager_id}",
        "form_definition": {
          "fields": [
            {"name": "introduction_completed", "type": "checkbox"},
            {"name": "team_introduced", "type": "checkbox"},
            {"name": "goals_discussed", "type": "checkbox"},
            {"name": "first_week_plan", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 24,
        "priority": "high"
      }
    },
    {
      "id": "parallel_setup_start",
      "name": "Start Parallel Setup",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "diverging"
      }
    },
    {
      "id": "workspace_setup",
      "name": "Workspace Setup",
      "type": "userTask",
      "attributes": {
        "assignee": "facilities_team",
        "form_definition": {
          "fields": [
            {"name": "desk_assigned", "type": "checkbox"},
            {"name": "equipment_issued", "type": "checkbox"},
            {"name": "parking_assigned", "type": "checkbox"},
            {"name": "access_cards_issued", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 48,
        "automation_candidate": true
      }
    },
    {
      "id": "payroll_setup",
      "name": "Payroll Setup",
      "type": "serviceTask",
      "attributes": {
        "capability": "financial_services",
        "operation": "setup_payroll_account",
        "timeout": 600
      },
      "metadata": {
        "estimated_duration": 20,
        "compliance_required": ["tax_forms", "banking_details"]
      }
    },
    {
      "id": "benefits_enrollment",
      "name": "Benefits Enrollment",
      "type": "userTask",
      "attributes": {
        "assignee": "benefits_coordinator",
        "form_definition": {
          "fields": [
            {"name": "health_insurance_selected", "type": "select", "options": ["basic", "premium", "family"]},
            {"name": "dental_coverage", "type": "checkbox"},
            {"name": "vision_coverage", "type": "checkbox"},
            {"name": "retirement_plan_enrollment", "type": "checkbox"},
            {"name": "life_insurance_amount", "type": "number"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 168,
        "deadline_type": "regulatory"
      }
    },
    {
      "id": "parallel_setup_join",
      "name": "Join Parallel Setup",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "converging"
      }
    },
    {
      "id": "first_week_checkin",
      "name": "First Week Check-in",
      "type": "userTask",
      "attributes": {
        "assignee": "${manager_id}",
        "form_definition": {
          "fields": [
            {"name": "employee_satisfaction", "type": "scale", "min": 1, "max": 10},
            {"name": "onboarding_feedback", "type": "textarea"},
            {"name": "additional_support_needed", "type": "textarea"},
            {"name": "ready_for_full_duties", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 40,
        "scheduled_delay": 168
      }
    },
    {
      "id": "generate_onboarding_report",
      "name": "Generate Onboarding Report",
      "type": "serviceTask",
      "attributes": {
        "capability": "time_series_analytics",
        "operation": "create_onboarding_analytics",
        "template": "onboarding_completion_report"
      },
      "metadata": {
        "estimated_duration": 5,
        "output_format": "pdf"
      }
    },
    {
      "id": "onboarding_complete",
      "name": "Onboarding Complete",
      "type": "end",
      "attributes": {
        "completion_criteria": "all_tasks_completed",
        "notification_targets": ["manager", "hr", "employee"]
      }
    }
  ],
  "flows": [
    {"id": "flow_1", "from": "start_onboarding", "to": "create_employee_profile"},
    {"id": "flow_2", "from": "create_employee_profile", "to": "setup_it_accounts"},
    {"id": "flow_3", "from": "setup_it_accounts", "to": "security_background_check"},
    {"id": "flow_4", "from": "security_background_check", "to": "check_clearance_level"},
    {
      "id": "flow_5", 
      "from": "check_clearance_level", 
      "to": "standard_orientation",
      "condition": "security_clearance == 'standard'",
      "default": true
    },
    {
      "id": "flow_6", 
      "from": "check_clearance_level", 
      "to": "enhanced_security_briefing",
      "condition": "security_clearance != 'standard'"
    },
    {"id": "flow_7", "from": "standard_orientation", "to": "join_orientation_paths"},
    {"id": "flow_8", "from": "enhanced_security_briefing", "to": "join_orientation_paths"},
    {"id": "flow_9", "from": "join_orientation_paths", "to": "manager_introduction"},
    {"id": "flow_10", "from": "manager_introduction", "to": "parallel_setup_start"},
    {"id": "flow_11", "from": "parallel_setup_start", "to": "workspace_setup"},
    {"id": "flow_12", "from": "parallel_setup_start", "to": "payroll_setup"},
    {"id": "flow_13", "from": "parallel_setup_start", "to": "benefits_enrollment"},
    {"id": "flow_14", "from": "workspace_setup", "to": "parallel_setup_join"},
    {"id": "flow_15", "from": "payroll_setup", "to": "parallel_setup_join"},
    {"id": "flow_16", "from": "benefits_enrollment", "to": "parallel_setup_join"},
    {"id": "flow_17", "from": "parallel_setup_join", "to": "first_week_checkin"},
    {"id": "flow_18", "from": "first_week_checkin", "to": "generate_onboarding_report"},
    {"id": "flow_19", "from": "generate_onboarding_report", "to": "onboarding_complete"}
  ]
}"""

# Example 3: Incident Response Workflow - APG Extended Format
INCIDENT_RESPONSE_APG_EXTENDED = """{
  "id": "IncidentResponse",
  "name": "Security Incident Response",
  "version": "3.0",
  "documentation": "Comprehensive security incident response workflow with automated analysis, escalation, and remediation.",
  "is_executable": true,
  "extensions": {
    "apg_features": {
      "ai_enabled": true,
      "cross_capability_workflows": true,
      "real_time_collaboration": true,
      "performance_tracking": true
    },
    "compliance": ["SOC2", "ISO27001", "NIST"],
    "criticality": "high",
    "automation_level": "hybrid"
  },
  "variables": {
    "incident_severity": "unknown",
    "incident_type": "",
    "affected_systems": [],
    "response_team": [],
    "estimated_impact": "",
    "containment_status": "open"
  },
  "elements": [
    {
      "id": "incident_detected",
      "name": "Incident Detected",
      "type": "start",
      "attributes": {
        "trigger_types": ["monitoring_alert", "user_report", "automated_scan"],
        "event_sources": ["siem", "monitoring", "helpdesk"]
      },
      "metadata": {
        "criticality": "immediate",
        "sla_start": true
      }
    },
    {
      "id": "automated_triage",
      "name": "Automated Incident Triage",
      "type": "serviceTask",
      "attributes": {
        "capability": "ai_orchestration",
        "operation": "analyze_security_incident",
        "timeout": 30,
        "configuration": {
          "ai_model": "incident_classifier_v3",
          "confidence_threshold": 0.85,
          "analysis_depth": "comprehensive"
        }
      },
      "metadata": {
        "estimated_duration": 15,
        "automation_level": "full",
        "ml_powered": true
      }
    },
    {
      "id": "severity_classification",
      "name": "Classify Incident Severity",
      "type": "exclusiveGateway",
      "attributes": {
        "gateway_direction": "diverging",
        "decision_criteria": "ai_confidence_and_impact"
      }
    },
    {
      "id": "low_priority_handling",
      "name": "Low Priority Incident Handling",
      "type": "serviceTask",
      "attributes": {
        "capability": "notification_engine",
        "operation": "create_incident_ticket",
        "configuration": {
          "priority": "low",
          "assignment_queue": "l1_support",
          "sla_hours": 72
        }
      },
      "metadata": {
        "automation_level": "full",
        "escalation_required": false
      }
    },
    {
      "id": "medium_priority_investigation",
      "name": "Medium Priority Investigation",
      "type": "userTask",
      "attributes": {
        "assignee": "security_analyst",
        "candidate_groups": ["security_team", "l2_support"],
        "form_definition": {
          "fields": [
            {"name": "investigation_findings", "type": "textarea", "required": true},
            {"name": "affected_systems_confirmed", "type": "multiselect", "required": true},
            {"name": "threat_indicators", "type": "textarea"},
            {"name": "recommended_actions", "type": "textarea", "required": true},
            {"name": "escalation_needed", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 4,
        "priority": "medium",
        "collaboration_enabled": true,
        "escalation": {
          "level1": {"hours": 2, "assignee": "senior_security_analyst"},
          "level2": {"hours": 3, "assignee": "security_manager"}
        }
      }
    },
    {
      "id": "high_critical_response",
      "name": "High/Critical Incident Response",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "diverging",
        "parallel_execution": "immediate"
      }
    },
    {
      "id": "immediate_containment",
      "name": "Immediate Containment Actions",
      "type": "userTask",
      "attributes": {
        "assignee": "incident_commander",
        "candidate_groups": ["security_team", "incident_response"],
        "form_definition": {
          "fields": [
            {"name": "containment_actions_taken", "type": "checklist", "options": [
              "Isolated affected systems",
              "Blocked malicious IPs",
              "Disabled compromised accounts",
              "Preserved forensic evidence",
              "Notified stakeholders"
            ]},
            {"name": "containment_status", "type": "select", "options": ["contained", "partially_contained", "not_contained"]},
            {"name": "additional_measures_needed", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_minutes": 30,
        "priority": "critical",
        "real_time_updates": true
      }
    },
    {
      "id": "stakeholder_notification",
      "name": "Notify Key Stakeholders",
      "type": "serviceTask",
      "attributes": {
        "capability": "real_time_collaboration",
        "operation": "broadcast_incident_alert",
        "configuration": {
          "notification_channels": ["email", "sms", "slack", "teams"],
          "stakeholder_groups": ["c_suite", "security_team", "compliance", "pr_team"],
          "message_template": "critical_incident_alert",
          "priority": "urgent"
        }
      },
      "metadata": {
        "estimated_duration": 2,
        "automation_level": "full"
      }
    },
    {
      "id": "forensic_analysis",
      "name": "Digital Forensic Analysis",
      "type": "userTask",
      "attributes": {
        "assignee": "forensic_analyst",
        "candidate_groups": ["forensics_team", "external_consultants"],
        "form_definition": {
          "fields": [
            {"name": "forensic_artifacts_collected", "type": "checklist", "options": [
              "Memory dumps",
              "Disk images",
              "Network logs",
              "System logs",
              "Application logs"
            ]},
            {"name": "attack_vector_identified", "type": "textarea"},
            {"name": "timeline_of_events", "type": "textarea"},
            {"name": "indicators_of_compromise", "type": "textarea"},
            {"name": "attribution_assessment", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 12,
        "priority": "high",
        "specialized_skills_required": true
      }
    },
    {
      "id": "parallel_response_join",
      "name": "Consolidate Response Actions",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "converging"
      }
    },
    {
      "id": "impact_assessment",
      "name": "Comprehensive Impact Assessment",
      "type": "userTask",
      "attributes": {
        "assignee": "security_manager",
        "candidate_groups": ["security_leadership", "risk_management"],
        "form_definition": {
          "fields": [
            {"name": "business_impact", "type": "select", "options": ["minimal", "moderate", "significant", "severe"]},
            {"name": "data_compromise_assessment", "type": "textarea"},
            {"name": "financial_impact_estimate", "type": "number"},
            {"name": "regulatory_notification_required", "type": "checkbox"},
            {"name": "customer_notification_required", "type": "checkbox"},
            {"name": "media_response_needed", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 6,
        "priority": "high",
        "compliance_critical": true
      }
    },
    {
      "id": "regulatory_compliance_check",
      "name": "Check Regulatory Compliance",
      "type": "exclusiveGateway",
      "attributes": {
        "gateway_direction": "diverging"
      }
    },
    {
      "id": "regulatory_notification",
      "name": "Regulatory Notification Process",
      "type": "userTask",
      "attributes": {
        "assignee": "compliance_officer",
        "candidate_groups": ["compliance_team", "legal_team"],
        "form_definition": {
          "fields": [
            {"name": "regulatory_bodies_notified", "type": "multiselect", "options": [
              "SEC", "GDPR Authority", "Industry Regulator", "Law Enforcement"
            ]},
            {"name": "notification_timeline_met", "type": "checkbox"},
            {"name": "required_documentation_submitted", "type": "checkbox"},
            {"name": "follow_up_actions_required", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 24,
        "priority": "critical",
        "regulatory_deadline": true
      }
    },
    {
      "id": "remediation_planning",
      "name": "Develop Remediation Plan",
      "type": "userTask",
      "attributes": {
        "assignee": "incident_commander",
        "candidate_groups": ["security_team", "engineering_team"],
        "form_definition": {
          "fields": [
            {"name": "short_term_remediation", "type": "textarea", "required": true},
            {"name": "long_term_remediation", "type": "textarea", "required": true},
            {"name": "system_hardening_measures", "type": "textarea"},
            {"name": "security_control_improvements", "type": "textarea"},
            {"name": "estimated_remediation_timeline", "type": "text"},
            {"name": "resource_requirements", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 8,
        "priority": "high",
        "collaboration_enabled": true
      }
    },
    {
      "id": "implement_remediation",
      "name": "Implement Remediation Measures",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "diverging"
      }
    },
    {
      "id": "technical_remediation",
      "name": "Technical Remediation",
      "type": "userTask",
      "attributes": {
        "assignee": "engineering_team",
        "candidate_groups": ["devops", "system_administrators"],
        "form_definition": {
          "fields": [
            {"name": "patches_applied", "type": "checklist"},
            {"name": "configurations_updated", "type": "checklist"},
            {"name": "monitoring_enhanced", "type": "checkbox"},
            {"name": "testing_completed", "type": "checkbox"},
            {"name": "rollback_plan_prepared", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 48,
        "priority": "high",
        "technical_complexity": "high"
      }
    },
    {
      "id": "process_improvements",
      "name": "Process and Policy Updates",
      "type": "userTask",
      "attributes": {
        "assignee": "security_architect",
        "candidate_groups": ["security_team", "governance"],
        "form_definition": {
          "fields": [
            {"name": "policies_updated", "type": "checklist"},
            {"name": "procedures_revised", "type": "checklist"},
            {"name": "training_materials_updated", "type": "checkbox"},
            {"name": "awareness_campaign_planned", "type": "checkbox"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 72,
        "priority": "medium",
        "governance_impact": true
      }
    },
    {
      "id": "remediation_complete",
      "name": "Remediation Complete",
      "type": "parallelGateway",
      "attributes": {
        "gateway_direction": "converging"
      }
    },
    {
      "id": "post_incident_review",
      "name": "Post-Incident Review",
      "type": "userTask",
      "attributes": {
        "assignee": "security_manager",
        "candidate_groups": ["incident_response_team"],
        "form_definition": {
          "fields": [
            {"name": "lessons_learned", "type": "textarea", "required": true},
            {"name": "response_effectiveness", "type": "scale", "min": 1, "max": 10},
            {"name": "areas_for_improvement", "type": "textarea"},
            {"name": "recommendations", "type": "textarea"},
            {"name": "training_needs_identified", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 168,
        "priority": "medium",
        "learning_opportunity": true
      }
    },
    {
      "id": "generate_final_report",
      "name": "Generate Incident Report",
      "type": "serviceTask",
      "attributes": {
        "capability": "document_management",
        "operation": "create_incident_report",
        "configuration": {
          "template": "security_incident_report_v2",
          "include_sections": ["executive_summary", "timeline", "impact", "remediation", "lessons_learned"],
          "distribution_list": ["c_suite", "board", "compliance", "audit"]
        }
      },
      "metadata": {
        "estimated_duration": 10,
        "automation_level": "full"
      }
    },
    {
      "id": "incident_closed",
      "name": "Incident Closed",
      "type": "end",
      "attributes": {
        "closure_criteria": "all_remediation_complete",
        "final_notifications": ["all_stakeholders", "affected_users"],
        "archive_evidence": true
      }
    }
  ],
  "flows": [
    {"id": "flow_1", "from": "incident_detected", "to": "automated_triage"},
    {"id": "flow_2", "from": "automated_triage", "to": "severity_classification"},
    {
      "id": "flow_3", 
      "from": "severity_classification", 
      "to": "low_priority_handling",
      "condition": "incident_severity == 'low'"
    },
    {
      "id": "flow_4", 
      "from": "severity_classification", 
      "to": "medium_priority_investigation",
      "condition": "incident_severity == 'medium'"
    },
    {
      "id": "flow_5", 
      "from": "severity_classification", 
      "to": "high_critical_response",
      "condition": "incident_severity in ['high', 'critical']"
    },
    {"id": "flow_6", "from": "high_critical_response", "to": "immediate_containment"},
    {"id": "flow_7", "from": "high_critical_response", "to": "stakeholder_notification"},
    {"id": "flow_8", "from": "high_critical_response", "to": "forensic_analysis"},
    {"id": "flow_9", "from": "immediate_containment", "to": "parallel_response_join"},
    {"id": "flow_10", "from": "stakeholder_notification", "to": "parallel_response_join"},
    {"id": "flow_11", "from": "forensic_analysis", "to": "parallel_response_join"},
    {"id": "flow_12", "from": "parallel_response_join", "to": "impact_assessment"},
    {"id": "flow_13", "from": "impact_assessment", "to": "regulatory_compliance_check"},
    {
      "id": "flow_14", 
      "from": "regulatory_compliance_check", 
      "to": "regulatory_notification",
      "condition": "regulatory_notification_required == true"
    },
    {
      "id": "flow_15", 
      "from": "regulatory_compliance_check", 
      "to": "remediation_planning",
      "condition": "regulatory_notification_required == false",
      "default": true
    },
    {"id": "flow_16", "from": "regulatory_notification", "to": "remediation_planning"},
    {"id": "flow_17", "from": "remediation_planning", "to": "implement_remediation"},
    {"id": "flow_18", "from": "implement_remediation", "to": "technical_remediation"},
    {"id": "flow_19", "from": "implement_remediation", "to": "process_improvements"},
    {"id": "flow_20", "from": "technical_remediation", "to": "remediation_complete"},
    {"id": "flow_21", "from": "process_improvements", "to": "remediation_complete"},
    {"id": "flow_22", "from": "remediation_complete", "to": "post_incident_review"},
    {"id": "flow_23", "from": "post_incident_review", "to": "generate_final_report"},
    {"id": "flow_24", "from": "generate_final_report", "to": "incident_closed"},
    {"id": "flow_25", "from": "low_priority_handling", "to": "incident_closed"},
    {"id": "flow_26", "from": "medium_priority_investigation", "to": "incident_closed"}
  ]
}"""

# Example 4: Simple Linear Workflow - Document Approval
SIMPLE_DOCUMENT_APPROVAL = """{
  "id": "DocumentApproval",
  "name": "Simple Document Approval",
  "version": "1.0",
  "documentation": "Basic document approval workflow for testing simplified BPML format.",
  "is_executable": true,
  "variables": {
    "document_type": "",
    "approver": "",
    "urgency": "normal"
  },
  "elements": [
    {
      "id": "doc_submitted",
      "name": "Document Submitted",
      "type": "start"
    },
    {
      "id": "review_document",
      "name": "Review Document",
      "type": "userTask",
      "attributes": {
        "assignee": "${approver}",
        "form_definition": {
          "fields": [
            {"name": "approval_decision", "type": "select", "options": ["approve", "reject", "revise"]},
            {"name": "comments", "type": "textarea"}
          ]
        }
      },
      "metadata": {
        "sla_hours": 24
      }
    },
    {
      "id": "doc_approved",
      "name": "Document Approved",
      "type": "end"
    }
  ],
  "flows": [
    {"id": "flow_1", "from": "doc_submitted", "to": "review_document"},
    {"id": "flow_2", "from": "review_document", "to": "doc_approved"}
  ]
}"""

# Example usage demonstration
def demonstrate_bpml_examples():
    """Demonstrate parsing different BPML formats."""
    from .bpml_engine import BPMLParser, BPMLVersion
    
    examples = [
        ("Purchase Order Approval (XML)", PURCHASE_ORDER_APPROVAL_XML, "xml"),
        ("Employee Onboarding (JSON)", EMPLOYEE_ONBOARDING_JSON, "json"),
        ("Incident Response (APG Extended)", INCIDENT_RESPONSE_APG_EXTENDED, "json"),
        ("Simple Document Approval", SIMPLE_DOCUMENT_APPROVAL, "json")
    ]
    
    for name, content, format_type in examples:
        print(f"\\n=== {name} ===")
        try:
            parser = BPMLParser(BPMLVersion.FULL_1_0)
            
            if format_type == "xml":
                process = parser.parse_xml(content)
            else:
                process = parser.parse_json(content)
            
            print(f"Process ID: {process.id}")
            print(f"Process Name: {process.name}")
            print(f"Elements: {len(process.elements)}")
            print(f"Flows: {len(process.flows)}")
            print(f"Start Events: {len(process.start_events)}")
            print(f"End Events: {len(process.end_events)}")
            
            # Show element types
            element_types = {}
            for element in process.elements.values():
                element_type = element.element_type.value
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            print("Element Types:")
            for elem_type, count in element_types.items():
                print(f"  {elem_type}: {count}")
            
        except Exception as e:
            print(f"Error parsing {name}: {e}")

if __name__ == "__main__":
    demonstrate_bpml_examples()

__all__ = [
    "PURCHASE_ORDER_APPROVAL_XML",
    "EMPLOYEE_ONBOARDING_JSON", 
    "INCIDENT_RESPONSE_APG_EXTENDED",
    "SIMPLE_DOCUMENT_APPROVAL",
    "demonstrate_bpml_examples"
]