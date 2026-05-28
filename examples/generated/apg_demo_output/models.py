"""
Database Models
===============

SQLAlchemy models generated from APG database definitions.
"""

from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class Tasks(Model):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    priority = Column(String(20), default='medium')
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Tasks({self.id})>'

class Users(Model):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Users({self.id})>'

class TaskAssignments(Model):
    __tablename__ = 'task_assignments'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    assigned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("Tasks")
    user = relationship("Users")
    
    def __repr__(self):
        return f'<TaskAssignments({self.id})>'
