"""
Task Manager - Flask-AppBuilder APG Application
==============================================

This Flask-AppBuilder application was automatically generated from APG source code.
It provides a complete, functional web application with agents, workflows, and database management.
"""

import logging
from flask import Flask, request, jsonify
from flask_appbuilder import AppBuilder, BaseView, expose
from flask_appbuilder.security.decorators import has_access
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
appbuilder = AppBuilder(app, db.session)


# ========================================
# APG Agent Runtime Implementation
# ========================================

class TaskManagerAgentRuntime:
    """Runtime implementation for TaskManagerAgent agent"""
    
    def __init__(self):
        self.name = "Task Manager"
        self.total_tasks = 0
        self.completed_tasks = 0
        self.active = False
        self.tasks = []
        self._running = False
        self._logger = logging.getLogger(f'{self.__class__.__name__}')
    
    def add_task(self, title, priority="medium"):
        """Runtime implementation of add_task"""
        task = {
            "id": self.total_tasks + 1,
            "title": title,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        self.tasks.append(task)
        self.total_tasks = self.total_tasks + 1
        
        return task
    
    def complete_task(self, task_id):
        """Runtime implementation of complete_task"""
        for task in self.tasks:
            if task["id"] == task_id:
                task["status"] = "completed"
                self.completed_tasks = self.completed_tasks + 1
                return True
        return False
    
    def get_stats(self):
        """Runtime implementation of get_stats"""
        completion_rate = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        return {
            "total": self.total_tasks,
            "completed": self.completed_tasks,
            "pending": self.total_tasks - self.completed_tasks,
            "completion_rate": completion_rate
        }
    
    def process(self):
        """Runtime implementation of process"""
        if self.active:
            return f"Processing {len(self.tasks)} tasks"
        return "Agent is inactive"
    
    def start(self):
        """Start the agent"""
        if not self._running:
            self._running = True
            self.active = True
            self._logger.info(f'Agent {self.__class__.__name__} started')
            return True
        return False
    
    def stop(self):
        """Stop the agent"""
        if self._running:
            self._running = False
            self.active = False
            self._logger.info(f'Agent {self.__class__.__name__} stopped')
            return True
        return False
    
    def is_running(self):
        """Check if agent is running"""
        return self._running
    
    def get_status(self):
        """Get agent status information"""
        return {
            'name': self.__class__.__name__,
            'running': self._running,
            'active': self.active,
            'name': self.name,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'timestamp': datetime.now().isoformat()
        }


# Create global agent instance
taskmanageragent_instance = TaskManagerAgentRuntime()


# ========================================
# Flask-AppBuilder Views
# ========================================

class TaskManagerAgentView(BaseView):
    """Flask-AppBuilder view for TaskManagerAgent agent"""
    
    default_view = 'agent_dashboard'
    
    @expose('/dashboard/')
    @has_access
    def agent_dashboard(self):
        """Agent dashboard view with live data"""
        agent = taskmanageragent_instance
        status = agent.get_status()
        return self.render_template('agent_dashboard.html',
                                    agent_name='TaskManagerAgent',
                                    agent_status=status,
                                    agent_running=agent.is_running())
    
    @expose('/start/', methods=['POST'])
    @has_access
    def start_agent(self):
        """Start the agent"""
        try:
            agent = taskmanageragent_instance
            success = agent.start()
            if success:
                return jsonify({'status': 'success', 'message': 'Agent started successfully'})
            else:
                return jsonify({'status': 'warning', 'message': 'Agent was already running'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/stop/', methods=['POST'])
    @has_access
    def stop_agent(self):
        """Stop the agent"""
        try:
            agent = taskmanageragent_instance
            success = agent.stop()
            if success:
                return jsonify({'status': 'success', 'message': 'Agent stopped successfully'})
            else:
                return jsonify({'status': 'warning', 'message': 'Agent was already stopped'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/status/', methods=['GET'])
    @has_access
    def get_agent_status(self):
        """Get agent status"""
        try:
            agent = taskmanageragent_instance
            status = agent.get_status()
            return jsonify({'status': 'success', 'data': status})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/add-task/', methods=['POST'])
    @has_access
    def add_task_api(self):
        """API endpoint for add_task method"""
        try:
            # Get request parameters
            data = request.get_json() or {}
            agent = taskmanageragent_instance
            
            # Extract parameters from request
            title = data.get('title', 'New Task')
            priority = data.get('priority', 'medium')
            
            # Call agent method
            result = agent.add_task(title, priority)
            
            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/complete-task/', methods=['POST'])
    @has_access
    def complete_task_api(self):
        """API endpoint for complete_task method"""
        try:
            # Get request parameters
            data = request.get_json() or {}
            agent = taskmanageragent_instance
            
            # Extract parameters from request
            task_id = data.get('task_id', 1)
            
            # Call agent method
            result = agent.complete_task(task_id)
            
            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/get-stats/', methods=['POST'])
    @has_access
    def get_stats_api(self):
        """API endpoint for get_stats method"""
        try:
            # Get request parameters
            data = request.get_json() or {}
            agent = taskmanageragent_instance
            
            # Call agent method
            result = agent.get_stats()
            
            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    
    @expose('/process/', methods=['POST'])
    @has_access
    def process_api(self):
        """API endpoint for process method"""
        try:
            # Get request parameters
            data = request.get_json() or {}
            agent = taskmanageragent_instance
            
            # Call agent method
            result = agent.process()
            
            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})


# ========================================
# Register Views and Initialize Database
# ========================================

# Register APG entity views
appbuilder.add_view(TaskManagerAgentView, 'TaskManagerAgent', icon='fa-cog', category='Agents')

# Create database tables
with app.app_context():
    try:
        db.create_all()
        logging.info('Database tables created successfully')
    except Exception as e:
        logging.error(f'Error creating database tables: {e}')

if __name__ == "__main__":
    import os
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    
    print(f'Starting APG Flask-AppBuilder application...')
    print(f'Host: {host}')
    print(f'Port: {port}')
    print(f'Debug: {debug}')
    print(f'Access at: http://{host}:{port}')
    
    app.run(host=host, port=port, debug=debug)
