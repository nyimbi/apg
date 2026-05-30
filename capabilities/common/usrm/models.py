"""User Management data models."""

from __future__ import annotations

from .user_runtime import (
	AccessReviewRecord,
	BulkUserActionRecord,
	DeprovisionRecord,
	RoleAssignmentRecord,
	UsrmAgentRecord,
	UserAuditEventRecord,
	UserInvitationRecord,
	UserProfileRecord,
	UserRecord,
)


UsrmRecord = UserRecord


__all__ = [
	"AccessReviewRecord",
	"BulkUserActionRecord",
	"DeprovisionRecord",
	"RoleAssignmentRecord",
	"UsrmAgentRecord",
	"UserAuditEventRecord",
	"UserInvitationRecord",
	"UserProfileRecord",
	"UserRecord",
	"UsrmRecord",
]
