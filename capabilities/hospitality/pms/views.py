"""Flask-AppBuilder compatible views and re-exported Pydantic schemas for PMS."""

from __future__ import annotations

# Re-export all models for external consumers
from .models import (
	AuditEvent,
	FolioCreate,
	FolioResponse,
	GuestCreate,
	GuestResponse,
	GuestUpdate,
	HousekeepingTaskCreate,
	HousekeepingTaskResponse,
	NightAuditReport,
	PMSListFilter,
	ReservationCreate,
	ReservationResponse,
	ReservationUpdate,
	RoomCreate,
	RoomResponse,
	RoomUpdate,
)

__all__ = [
	"RoomCreate", "RoomUpdate", "RoomResponse",
	"GuestCreate", "GuestUpdate", "GuestResponse",
	"ReservationCreate", "ReservationUpdate", "ReservationResponse",
	"FolioCreate", "FolioResponse",
	"HousekeepingTaskCreate", "HousekeepingTaskResponse",
	"NightAuditReport", "PMSListFilter", "AuditEvent",
]
