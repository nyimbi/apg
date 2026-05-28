"""
Database Model Views
===================

Flask-AppBuilder ModelViews for database tables.
"""

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface
from .models import Tasks, Users, TaskAssignments

class TasksView(ModelView):
    """ModelView for tasks table"""
    
    datamodel = SQLAInterface(Tasks)
    
    list_columns = ['id', 'title', 'priority', 'status', 'created_at']
    show_columns = ['id', 'title', 'description', 'priority', 'status', 'created_at', 'updated_at']
    edit_columns = ['title', 'description', 'priority', 'status']
    add_columns = ['title', 'description', 'priority', 'status']
    
    search_columns = ['title', 'description']

class UsersView(ModelView):
    """ModelView for users table"""
    
    datamodel = SQLAInterface(Users)
    
    list_columns = ['id', 'username', 'email', 'created_at']
    show_columns = ['id', 'username', 'email', 'created_at']
    edit_columns = ['username', 'email']
    add_columns = ['username', 'email']
    
    search_columns = ['username', 'email']

class TaskAssignmentsView(ModelView):
    """ModelView for task_assignments table"""
    
    datamodel = SQLAInterface(TaskAssignments)
    
    list_columns = ['id', 'task_id', 'user_id', 'assigned_at']
    show_columns = ['id', 'task_id', 'user_id', 'assigned_at']
    edit_columns = ['task_id', 'user_id']
    add_columns = ['task_id', 'user_id']
