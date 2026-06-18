"""SQLAlchemy models stub for customer_master (test fixture)."""
from __future__ import annotations
from sqlalchemy import Column, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
	pass


class CustomerMaster(Base):
	__tablename__ = "customer_master"
	id = Column(String, primary_key=True)
	tenant_id = Column(String, nullable=False)
	name = Column(String, nullable=False)
