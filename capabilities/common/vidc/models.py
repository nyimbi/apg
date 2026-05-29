"""Video Conferencing data models."""

from __future__ import annotations

from .video_runtime import (
	CaptionRecord,
	MeetingAuditEventRecord,
	MeetingRecord,
	MeetingRoomRecord,
	ParticipantRecord,
	RecordingRecord,
)


VidcRecord = MeetingRoomRecord


__all__ = [
	"CaptionRecord",
	"MeetingAuditEventRecord",
	"MeetingRecord",
	"MeetingRoomRecord",
	"ParticipantRecord",
	"RecordingRecord",
	"VidcRecord",
]
