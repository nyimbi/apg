# APG Workflow & Business Process Management - Quick Start Guide

**Get Up and Running in 15 Minutes**

© 2025 Datacraft | Author: Nyimbi Odero | Version 1.0

---

## 🚀 **Quick Start Overview**

This guide will get you up and running with APG Workflow & Business Process Management (WBPM) in just 15 minutes. You'll create your first workflow, execute it, and see the results.

### **What You'll Accomplish**

- ✅ Set up WBPM development environment
- ✅ Create your first workflow using APG DSL
- ✅ Execute the workflow and manage tasks
- ✅ Monitor process performance and timing
- ✅ Set up automated scheduling

---

## 🛠️ **Prerequisites (2 minutes)**

Before starting, ensure you have:

- **Python 3.11+** installed
- **Docker & Docker Compose** (recommended) OR PostgreSQL and Redis locally
- **Git** for cloning the repository
- **Basic knowledge** of workflows and business processes

---

## 📦 **Installation (5 minutes)**

### **Option 1: Docker Compose (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform/capabilities/general_cross_functional/workflow_business_process_mgmt

# 2. Start services with Docker Compose
docker-compose up -d

# 3. Wait for services to be ready (about 30 seconds)
docker-compose logs -f api
# Wait until you see "Application startup complete"
```

### **Option 2: Local Development**

```bash
# 1. Clone and setup
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform/capabilities/general_cross_functional/workflow_business_process_mgmt

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export DATABASE_URL="postgresql://localhost:5432/wbpm_dev"
export REDIS_URL="redis://localhost:6379/0"

# 5. Initialize database
alembic upgrade head

# 6. Start the server
uvicorn main:app --reload --port 8000
```

### **Verify Installation**

```bash
# Check API is running
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "2025-01-XX...", "checks": {...}}
```

---

## 🎯 **Create Your First Workflow (3 minutes)**

We'll create a simple "Employee Onboarding" workflow using APG's Domain Specific Language (DSL).

### **Step 1: Create the Workflow Definition**

Create a file called `employee_onboarding.apg`:

```apg
WORKFLOW "Employee Onboarding Process"
DESCRIPTION "Complete onboarding process for new employees"
VERSION "1.0"
CATEGORY "Human Resources"

START "New Employee Hired"

USER_TASK "Complete HR Forms"
  assignee: "hr@company.com"
  duration: "2h"
  sla: "4h"
  description: "Fill out all required HR documentation"

SERVICE_TASK "Create Email Account"
  url: "https://api.company.com/accounts"
  method: "POST"
  duration: "5m"
  description: "Automatically create company email account"

DECISION "Manager Approval Required?"
  description: "Check if manager approval is needed"

USER_TASK "Manager Approval"
  groups: "managers"
  duration: "1h"
  sla: "2h"
  description: "Manager reviews and approves new employee setup"

SERVICE_TASK "Send Welcome Email"
  url: "https://api.company.com/notifications/email"
  method: "POST"
  duration: "2m"
  description: "Send welcome email with login details"

END "Onboarding Complete"

// Define the workflow connections
"New Employee Hired" → "Complete HR Forms"
"Complete HR Forms" → "Create Email Account"
"Create Email Account" → "Manager Approval Required?"
"Manager Approval Required?" → "Manager Approval" [condition: approval_needed == true]
"Manager Approval Required?" → "Send Welcome Email" [condition: approval_needed == false]
"Manager Approval" → "Send Welcome Email"
"Send Welcome Email" → "Onboarding Complete"
```

### **Step 2: Parse and Create the Workflow**

```python
# create_workflow.py
import asyncio
from apg_workflow_parser import APGWorkflowService, APGWorkflowFormat
from enhanced_visual_designer import EnhancedVisualDesignerService
from workflow_scheduler import SchedulerFactory
from models import APGTenantContext

async def create_onboarding_workflow():
    """Create the employee onboarding workflow."""
    
    # Setup context
    context = APGTenantContext(
        tenant_id="quickstart_tenant",
        user_id="admin@company.com",
        user_roles=["admin", "designer"],
        permissions=["workflow_design", "workflow_execute", "workflow_manage"]
    )
    
    # Read the workflow definition
    with open('employee_onboarding.apg', 'r') as f:
        workflow_definition = f.read()
    
    # Create services
    scheduler = await SchedulerFactory.get_scheduler(context)
    visual_designer = EnhancedVisualDesignerService(scheduler)
    workflow_service = APGWorkflowService(visual_designer, scheduler)
    
    # Parse and create the workflow
    result = await workflow_service.create_workflow_from_definition(
        workflow_definition,
        APGWorkflowFormat.APG_DSL,
        context
    )
    
    if result.success:
        print(f"✅ Workflow created successfully!")
        print(f"Canvas ID: {result.data['implementation']['canvas_id']}")
        print(f"Elements: {result.data['implementation']['element_count']}")
        return result.data['implementation']['canvas_id']
    else:
        print(f"❌ Failed to create workflow: {result.message}")
        return None

# Run the creation
if __name__ == "__main__":
    canvas_id = asyncio.run(create_onboarding_workflow())
```

Run the script:
```bash
python create_workflow.py
```

---

## ▶️ **Execute Your First Workflow (2 minutes)**

### **Step 1: Start a Process Instance**

```python
# execute_workflow.py
import asyncio
from workflow_engine import WorkflowExecutionEngine
from models import APGTenantContext, WBPMProcessDefinition

async def start_onboarding_process():
    """Start a new employee onboarding process."""
    
    context = APGTenantContext(
        tenant_id="quickstart_tenant",
        user_id="manager@company.com",
        user_roles=["manager"],
        permissions=["workflow_execute"]
    )
    
    # Initialize workflow engine
    engine = WorkflowExecutionEngine()
    
    # Load process definition (in production, this would come from database)
    process_definition = WBPMProcessDefinition(
        process_id="employee_onboarding_v1",
        name="Employee Onboarding Process",
        version="1.0"
        # Definition would include activities and flows from parsed workflow
    )
    
    # Start process with initial data
    initial_variables = {
        "employee_name": "John Doe",
        "employee_email": "john.doe@company.com",
        "department": "Engineering",
        "start_date": "2025-02-01",
        "approval_needed": True
    }
    
    execution = await engine.start_process_instance(
        process_definition,
        initial_variables,
        context
    )
    
    print(f"✅ Process started!")
    print(f"Instance ID: {execution.instance_id}")
    print(f"Status: {execution.status}")
    print(f"Variables: {execution.variables}")
    
    return execution.instance_id

# Run the execution
if __name__ == "__main__":
    instance_id = asyncio.run(start_onboarding_process())
```

### **Step 2: Get and Complete Tasks**

```python
# manage_tasks.py
import asyncio
from task_management import TaskManagementService
from models import APGTenantContext

async def manage_workflow_tasks():
    """Get and complete workflow tasks."""
    
    # HR User context
    hr_context = APGTenantContext(
        tenant_id="quickstart_tenant",
        user_id="hr@company.com",
        user_roles=["hr"],
        permissions=["task_execute"]
    )
    
    # Initialize task service
    task_service = TaskManagementService()
    
    # Get tasks for HR user
    tasks_result = await task_service.get_user_tasks(hr_context.user_id, hr_context)
    
    if tasks_result.success and tasks_result.data['tasks']:
        task = tasks_result.data['tasks'][0]
        print(f"📋 Found task: {task['name']}")
        print(f"Description: {task['description']}")
        print(f"Due: {task['due_date']}")
        
        # Complete the task
        completion_data = {
            "form_completed": True,
            "emergency_contact": "Jane Doe - 555-0123",
            "tax_withholding": "Standard",
            "benefits_selected": ["Health", "Dental", "401k"]
        }
        
        completion_result = await task_service.complete_task(
            task['task_id'],
            completion_data,
            hr_context
        )
        
        if completion_result.success:
            print("✅ Task completed successfully!")
        else:
            print(f"❌ Task completion failed: {completion_result.message}")
    else:
        print("📭 No tasks found for HR user")

# Run task management
if __name__ == "__main__":
    asyncio.run(manage_workflow_tasks())
```

---

## 📊 **Monitor Your Workflow (2 minutes)**

### **View Process Performance**

```python
# monitor_workflow.py
import asyncio
from analytics_engine import ProcessAnalyticsEngine
from workflow_scheduler import SchedulerFactory
from models import APGTenantContext

async def monitor_process_performance():
    """Monitor workflow performance and timing."""
    
    context = APGTenantContext(
        tenant_id="quickstart_tenant",
        user_id="admin@company.com",
        user_roles=["admin"],
        permissions=["analytics_view"]
    )
    
    # Get scheduler for timing information
    scheduler = await SchedulerFactory.get_scheduler(context)
    
    # Check timer status
    timer_result = await scheduler.get_timer_status()
    if timer_result.success:
        timers = timer_result.data['timers']
        print(f"⏱️  Active Timers: {len(timers)}")
        
        for timer in timers[:3]:  # Show first 3 timers
            print(f"  • Process: {timer['process_instance_id']}")
            print(f"    Duration: {timer['elapsed_minutes']:.1f} minutes")
            print(f"    Warning sent: {timer['warning_sent']}")
    
    # Get execution statistics
    stats_result = await scheduler.get_execution_stats()
    if stats_result.success:
        stats = stats_result.data['execution_stats']
        print(f"\n📈 Execution Statistics:")
        print(f"  • Total executions: {stats['total_scheduled_executions']}")
        print(f"  • Successful: {stats['successful_executions']}")
        print(f"  • Failed: {stats['failed_executions']}")
        print(f"  • Active schedules: {stats['active_schedules']}")
        print(f"  • Active timers: {stats['active_timers']}")

# Run monitoring
if __name__ == "__main__":
    asyncio.run(monitor_process_performance())
```

### **Check Real-time Status via API**

```bash
# Get process instances
curl http://localhost:8000/api/v1/processes/instances

# Get user tasks
curl http://localhost:8000/api/v1/tasks

# Get timing metrics
curl http://localhost:8000/api/v1/analytics/timing

# Get schedule status
curl http://localhost:8000/api/v1/schedules/status
```

---

## 🕐 **Set Up Automated Scheduling (1 minute)**

### **Create a Recurring Schedule**

```python
# schedule_workflow.py
import asyncio
from workflow_scheduler import SchedulerFactory, create_recurring_schedule
from models import APGTenantContext

async def setup_automated_scheduling():
    """Set up automated workflow scheduling."""
    
    context = APGTenantContext(
        tenant_id="quickstart_tenant",
        user_id="admin@company.com",
        user_roles=["admin"],
        permissions=["schedule_manage"]
    )
    
    # Get scheduler
    scheduler = await SchedulerFactory.get_scheduler(context)
    
    # Create a daily schedule for new employee checks
    daily_schedule = create_recurring_schedule(
        name="Daily New Employee Check",
        process_definition_id="employee_onboarding_v1",
        interval_minutes=1440,  # 24 hours
        tenant_context=context,
        input_variables={
            "check_type": "daily_review",
            "auto_process": True
        }
    )
    
    # Create the schedule
    result = await scheduler.create_schedule(daily_schedule)
    
    if result.success:
        print("✅ Automated schedule created!")
        print(f"Schedule ID: {result.data['schedule_id']}")
        print(f"Next execution: {result.data['next_execution']}")
    else:
        print(f"❌ Schedule creation failed: {result.message}")

# Run scheduling setup
if __name__ == "__main__":
    asyncio.run(setup_automated_scheduling())
```

---

## 🎨 **Visual Workflow Designer (Bonus)**

### **Access the Visual Designer**

1. **Open your browser** and navigate to: `http://localhost:8000/designer`

2. **Create a new canvas** or load your existing workflow

3. **Visual Features Available:**
   - Drag-and-drop BPMN 2.0 elements
   - Visual connections between process blocks
   - Element permissions configuration
   - Timing and SLA settings
   - Real-time collaboration
   - Export to BPMN XML

### **Quick Designer Actions**

```javascript
// Connect to the designer via WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/admin@company.com');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    if (update.type === 'process_update') {
        console.log('Process updated:', update.data);
    }
};

// Subscribe to process updates
ws.send(JSON.stringify({
    type: 'subscribe_process',
    process_id: 'employee_onboarding_v1'
}));
```

---

## 🔍 **What's Next?**

Now that you have WBPM running, explore these advanced features:

### **Immediate Next Steps**

1. **📚 Read the User Guide** - `docs/user_guide.md` for comprehensive feature documentation
2. **🔧 Explore the API** - Visit `http://localhost:8000/docs` for interactive API documentation  
3. **📊 Set up Analytics** - Configure process performance monitoring
4. **🔗 Add Integrations** - Connect to external systems and APIs

### **Advanced Features to Explore**

- **Natural Language Workflow Creation** - Create workflows from plain English descriptions
- **AI-Powered Optimization** - Get intelligent process improvement recommendations
- **Advanced Permissions** - Set up role-based access control
- **Integration Hub** - Connect to ERP, CRM, and other business systems
- **Mobile Task Management** - Use mobile apps for task completion

### **Sample Workflows to Try**

```apg
// Invoice Processing Workflow
WORKFLOW "Invoice Processing"
START "Invoice Received"
SERVICE_TASK "Validate Invoice"
  url: "https://api.validation.service/validate"
  duration: "5m"
DECISION "Approval Required?"
USER_TASK "Finance Approval"
  groups: "finance"
  sla: "4h"
SERVICE_TASK "Payment Processing"
END "Invoice Paid"

// Customer Support Ticket
WORKFLOW "Support Ticket Resolution"
START "Ticket Created"
USER_TASK "Initial Assessment"
  assignee: "support@company.com"
  duration: "30m"
DECISION "Escalation Needed?"
USER_TASK "L2 Support Review"
  groups: "l2_support"
  sla: "2h"
END "Ticket Resolved"
```

### **Getting Help**

- **📖 Documentation** - Complete guides in `docs/` folder
- **💬 Community** - Join our developer community
- **🐛 Issues** - Report bugs or request features on GitHub
- **📧 Support** - Contact support@datacraft.co.ke for enterprise support

---

## 🎉 **Congratulations!**

You've successfully:

✅ **Set up** APG Workflow & Business Process Management  
✅ **Created** your first workflow using APG DSL  
✅ **Executed** the workflow and managed tasks  
✅ **Monitored** process performance and timing  
✅ **Configured** automated scheduling  

You now have a powerful, enterprise-grade workflow management system ready for your business processes!

### **Performance Achieved**

- **Setup Time**: ~15 minutes total
- **Workflow Creation**: 2-3 minutes per workflow
- **Process Execution**: Sub-second response times
- **Task Management**: Intelligent routing and assignment
- **Real-time Monitoring**: Live process tracking
- **Automated Scheduling**: Hands-off process execution

---

## 📞 **Quick Reference**

### **Essential Commands**

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Run tests
pytest tests/

# Check API health
curl http://localhost:8000/health
```

### **Key Endpoints**

- **API Documentation**: `http://localhost:8000/docs`
- **Visual Designer**: `http://localhost:8000/designer`
- **Health Check**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`

### **Configuration Files**

- **Docker Compose**: `docker-compose.yml`
- **Environment**: `.env`
- **Database Migrations**: `migrations/`
- **Examples**: `examples/`

---

**© 2025 Datacraft. All rights reserved.**  
**Contact: www.datacraft.co.ke | nyimbi@gmail.com**

*Get started with enterprise workflow automation in minutes, not months.*