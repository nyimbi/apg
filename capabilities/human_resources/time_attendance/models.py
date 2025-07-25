"""
Time & Attendance Models

Database models for time tracking, attendance, scheduling, and leave management.
"""

from datetime import datetime, date, time
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, Time, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class HRWorkCalendar(Model, AuditMixin, BaseMixin):
	"""
	Work calendar defining business days and holidays.
	"""
	__tablename__ = 'hr_ta_work_calendar'
	
	# Identity
	calendar_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Calendar Information
	calendar_name = Column(String(100), nullable=False, index=True)
	description = Column(Text, nullable=True)
	year = Column(Integer, nullable=False, index=True)
	
	# Date Information
	calendar_date = Column(Date, nullable=False, index=True)
	is_working_day = Column(Boolean, default=True)
	is_holiday = Column(Boolean, default=False)
	holiday_name = Column(String(200), nullable=True)
	
	# Working Hours
	standard_hours = Column(DECIMAL(4, 2), default=8.00)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'calendar_date', name='uq_calendar_date_tenant'),
	)
	
	def __repr__(self):
		return f"<HRWorkCalendar {self.calendar_date}>"


class HRTimeEntry(Model, AuditMixin, BaseMixin):
	"""
	Individual time entries for employees.
	"""
	__tablename__ = 'hr_ta_time_entry'
	
	# Identity
	time_entry_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Time Information
	entry_date = Column(Date, nullable=False, index=True)
	clock_in = Column(DateTime, nullable=True)
	clock_out = Column(DateTime, nullable=True)
	
	# Calculated Hours
	total_hours = Column(DECIMAL(6, 2), nullable=True)
	regular_hours = Column(DECIMAL(6, 2), nullable=True)
	overtime_hours = Column(DECIMAL(6, 2), nullable=True)
	
	# Entry Type
	entry_type = Column(String(20), default='Regular', index=True)  # Regular, Overtime, Holiday, Sick, Vacation
	
	# Status and Approval
	status = Column(String(20), default='Draft', index=True)  # Draft, Submitted, Approved, Rejected
	approved_by = Column(String(36), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	notes = Column(Text, nullable=True)
	
	# Relationships
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	
	def __repr__(self):
		return f"<HRTimeEntry {self.employee.full_name} - {self.entry_date}>"


class HRLeaveRequest(Model, AuditMixin, BaseMixin):
	"""
	Employee leave/time-off requests.
	"""
	__tablename__ = 'hr_ta_leave_request'
	
	# Identity
	leave_request_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Request Information
	request_date = Column(Date, nullable=False, index=True)
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	leave_type = Column(String(50), nullable=False, index=True)  # Vacation, Sick, Personal, etc.
	
	# Duration
	total_days = Column(DECIMAL(4, 1), nullable=False)
	total_hours = Column(DECIMAL(6, 2), nullable=False)
	
	# Request Details
	reason = Column(Text, nullable=True)
	
	# Status and Approval
	status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Rejected, Cancelled
	approved_by = Column(String(36), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	
	def __repr__(self):
		return f"<HRLeaveRequest {self.employee.full_name} - {self.leave_type}: {self.start_date} to {self.end_date}>"