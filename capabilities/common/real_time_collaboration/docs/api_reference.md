# APG Real-Time Collaboration - API Reference

Complete API reference for the Real-Time Collaboration capability with Teams/Zoom/Meet feature parity and Flask-AppBuilder integration.

## 📋 Table of Contents

1. [Authentication](#authentication)
2. [Session Management](#session-management)
3. [Page Collaboration](#page-collaboration)
4. [Video Calls](#video-calls)
5. [Chat & Messaging](#chat--messaging)
6. [Third-Party Integration](#third-party-integration)
7. [Analytics](#analytics)
8. [WebSocket Protocol](#websocket-protocol)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)

## 🔐 Authentication

All API endpoints require authentication through APG's auth_rbac system.

### Headers Required
```http
Authorization: Bearer <token>
Content-Type: application/json
X-Tenant-ID: <tenant_id>
```

### Permissions
- `rtc:session:create` - Create collaboration sessions
- `rtc:session:join` - Join collaboration sessions
- `rtc:video:create` - Start video calls
- `rtc:recording:create` - Start recordings
- `rtc:admin` - Administrative functions

## 🎯 Session Management

### Create Session

Create a new collaboration session.

```http
POST /api/v1/rtc/sessions
```

**Request Body:**
```json
{
  "session_name": "Budget Planning Session",
  "session_type": "page_collaboration",
  "page_url": "/finance/budget/planning"
}
```

**Response:**
```json
{
  "session_id": "01HKQK2V3N4M5P6Q7R8S9T0U1V",
  "session_name": "Budget Planning Session",
  "session_type": "page_collaboration",
  "owner_user_id": "user123",
  "is_active": true,
  "created_at": "2024-01-30T10:30:00Z",
  "participant_count": 1,
  "meeting_url": "/rtc/join/01HKQK2V3N4M5P6Q7R8S9T0U1V"
}
```

### Join Session

Join an existing collaboration session.

```http
POST /api/v1/rtc/sessions/{session_id}/join?role=viewer
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1W",
  "session_id": "01HKQK2V3N4M5P6Q7R8S9T0U1V",
  "role": "viewer",
  "joined_at": "2024-01-30T10:35:00Z",
  "permissions": {
    "can_edit": false,
    "can_annotate": true,
    "can_chat": true,
    "can_share_screen": false
  }
}
```

### Get Session Details

Retrieve information about a specific session.

```http
GET /api/v1/rtc/sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "01HKQK2V3N4M5P6Q7R8S9T0U1V",
  "session_name": "Budget Planning Session",
  "session_type": "page_collaboration",
  "owner_user_id": "user123",
  "is_active": true,
  "created_at": "2024-01-30T10:30:00Z",
  "participant_count": 3,
  "meeting_url": "/rtc/join/01HKQK2V3N4M5P6Q7R8S9T0U1V"
}
```

### End Session

End a collaboration session.

```http
DELETE /api/v1/rtc/sessions/{session_id}
```

**Response:**
```json
{
  "message": "Session ended successfully",
  "session_id": "01HKQK2V3N4M5P6Q7R8S9T0U1V"
}
```

## 📄 Page Collaboration

### Enable Page Collaboration

Enable collaboration features for a specific Flask-AppBuilder page.

```http
POST /api/v1/rtc/page-collaboration
```

**Request Body:**
```json
{
  "page_url": "/admin/users/list",
  "page_title": "User Management",
  "page_type": "list_view"
}
```

**Response:**
```json
{
  "page_collab_id": "01HKQK2V3N4M5P6Q7R8S9T0U1X",
  "page_url": "/admin/users/list",
  "page_title": "User Management",
  "current_users": ["user123"],
  "is_active": true,
  "total_delegations": 0,
  "total_assistance_requests": 0
}
```

### Delegate Form Field

Delegate a specific form field to another user.

```http
POST /api/v1/rtc/page-collaboration/delegate-field?page_url=/admin/users/add
```

**Request Body:**
```json
{
  "field_name": "email",
  "delegatee_id": "user456",
  "instructions": "Please fill in the user's email address"
}
```

**Response:**
```json
{
  "message": "Field delegated successfully",
  "field_name": "email"
}
```

### Request Assistance

Request assistance for a page or specific field.

```http
POST /api/v1/rtc/page-collaboration/request-assistance?page_url=/admin/users/edit/123
```

**Request Body:**
```json
{
  "field_name": "permissions",
  "description": "I need help setting up the correct permissions for this user role"
}
```

**Response:**
```json
{
  "message": "Assistance requested successfully"
}
```

### Get Page Presence

Get real-time presence information for a page.

```http
GET /api/v1/rtc/page-collaboration/presence?page_url=/admin/users/list
```

**Response:**
```json
[
  {
    "user_id": "user123",
    "display_name": "John Doe",
    "status": "active",
    "page_url": "/admin/users/list",
    "last_activity": "2024-01-30T10:45:00Z",
    "is_typing": false,
    "video_enabled": false,
    "audio_enabled": false
  },
  {
    "user_id": "user456",
    "display_name": "Jane Smith",
    "status": "active",
    "page_url": "/admin/users/list",
    "last_activity": "2024-01-30T10:44:30Z",
    "is_typing": true,
    "video_enabled": false,
    "audio_enabled": false
  }
]
```

## 🎥 Video Calls

### Start Video Call

Start a video call with Teams/Zoom/Meet features.

```http
POST /api/v1/rtc/video-calls?page_url=/admin/dashboard
```

**Request Body:**
```json
{
  "call_name": "Weekly Team Sync",
  "call_type": "video",
  "enable_recording": true
}
```

**Response:**
```json
{
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "call_name": "Weekly Team Sync",
  "call_type": "video",
  "status": "active",
  "meeting_id": "1234567890",
  "teams_meeting_url": null,
  "zoom_meeting_id": null,
  "meet_url": null,
  "host_user_id": "user123",
  "current_participants": 1,
  "max_participants": 100,
  "recording_enabled": true
}
```

### Join Video Call

Join a video call as a participant.

```http
POST /api/v1/rtc/video-calls/{call_id}/participants
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Z",
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "user_id": "user456",
  "role": "attendee",
  "joined_at": "2024-01-30T10:50:00Z",
  "permissions": {
    "can_share_screen": true,
    "can_unmute_self": true,
    "can_start_video": true,
    "can_chat": true
  }
}
```

### Start Screen Share

Start screen sharing in a video call.

```http
POST /api/v1/rtc/video-calls/{call_id}/screen-share?share_type=desktop
```

**Response:**
```json
{
  "share_id": "01HKQK2V3N4M5P6Q7R8S9T0U20",
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "share_type": "desktop",
  "presenter_id": "user123",
  "status": "active",
  "started_at": "2024-01-30T10:55:00Z"
}
```

### Start Recording

Start recording a video call.

```http
POST /api/v1/rtc/video-calls/{call_id}/recording?recording_name=Team Sync Recording&recording_type=full_meeting
```

**Response:**
```json
{
  "recording_id": "01HKQK2V3N4M5P6Q7R8S9T0U21",
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "recording_name": "Team Sync Recording",
  "recording_type": "full_meeting",
  "status": "recording",
  "started_at": "2024-01-30T10:56:00Z",
  "auto_transcription": true
}
```

### Control Participant Audio/Video

Toggle participant audio or video.

```http
PUT /api/v1/rtc/video-calls/{call_id}/participants/{participant_id}/audio?enabled=false
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Z",
  "audio_enabled": false,
  "updated_at": "2024-01-30T11:00:00Z"
}
```

```http
PUT /api/v1/rtc/video-calls/{call_id}/participants/{participant_id}/video?enabled=true
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Z",
  "video_enabled": true,
  "updated_at": "2024-01-30T11:01:00Z"
}
```

### Raise Hand

Raise or lower hand in a video call.

```http
POST /api/v1/rtc/video-calls/{call_id}/participants/{participant_id}/hand
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Z",
  "hand_raised": true,
  "raised_at": "2024-01-30T11:02:00Z"
}
```

### Send Reaction

Send an emoji reaction in a video call.

```http
POST /api/v1/rtc/video-calls/{call_id}/participants/{participant_id}/reaction?reaction=👍
```

**Response:**
```json
{
  "participant_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Z",
  "reaction": "👍",
  "sent_at": "2024-01-30T11:03:00Z"
}
```

### End Video Call

End a video call for all participants.

```http
DELETE /api/v1/rtc/video-calls/{call_id}
```

**Response:**
```json
{
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "ended_at": "2024-01-30T11:30:00Z",
  "message": "Video call ended successfully"
}
```

## 💬 Chat & Messaging

### Send Chat Message

Send a chat message to a page.

```http
POST /api/v1/rtc/chat/messages?message=Hello everyone!&page_url=/admin/users/list&message_type=text
```

**Response:**
```json
{
  "message_id": "01HKQK2V3N4M5P6Q7R8S9T0U22",
  "message": "Hello everyone!",
  "sent_at": "2024-01-30T11:05:00Z"
}
```

### Get Chat Messages

Retrieve chat messages for a page.

```http
GET /api/v1/rtc/chat/messages?page_url=/admin/users/list&limit=50
```

**Response:**
```json
{
  "messages": [
    {
      "message_id": "01HKQK2V3N4M5P6Q7R8S9T0U22",
      "user_id": "user123",
      "username": "John Doe",
      "message": "Hello everyone!",
      "message_type": "text",
      "timestamp": "2024-01-30T11:05:00Z"
    }
  ],
  "page_url": "/admin/users/list",
  "total_count": 1
}
```

## 🔗 Third-Party Integration

### Setup Microsoft Teams Integration

Configure Microsoft Teams integration for the tenant.

```http
POST /api/v1/rtc/integrations/teams?teams_tenant_id=12345678-1234-1234-1234-123456789012&application_id=87654321-4321-4321-4321-210987654321
```

**Response:**
```json
{
  "integration_id": "01HKQK2V3N4M5P6Q7R8S9T0U23",
  "platform": "teams",
  "status": "active",
  "teams_tenant_id": "12345678-1234-1234-1234-123456789012",
  "created_at": "2024-01-30T11:10:00Z"
}
```

### Setup Zoom Integration

Configure Zoom integration for the tenant.

```http
POST /api/v1/rtc/integrations/zoom?zoom_account_id=ABCDEfghij1234567890&api_key=your_zoom_api_key&api_secret=your_zoom_api_secret
```

**Response:**
```json
{
  "integration_id": "01HKQK2V3N4M5P6Q7R8S9T0U24",
  "platform": "zoom",
  "status": "active",
  "zoom_account_id": "ABCDEfghij1234567890",
  "created_at": "2024-01-30T11:15:00Z"
}
```

### Setup Google Meet Integration

Configure Google Meet integration for the tenant.

```http
POST /api/v1/rtc/integrations/google-meet?workspace_domain=company.com&client_id=google_client_id&client_secret=google_client_secret
```

**Response:**
```json
{
  "integration_id": "01HKQK2V3N4M5P6Q7R8S9T0U25",
  "platform": "google_meet",
  "status": "active",
  "workspace_domain": "company.com",
  "created_at": "2024-01-30T11:20:00Z"
}
```

## 📊 Analytics

### Get Collaboration Analytics

Retrieve collaboration analytics and insights.

```http
GET /api/v1/rtc/analytics?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z
```

**Response:**
```json
{
  "date_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "page_collaboration": {
    "total_pages": 25,
    "total_delegations": 157,
    "total_assistance_requests": 43,
    "average_users_per_session": 3.2
  },
  "sessions": {
    "total_sessions": 89,
    "active_sessions": 5,
    "average_duration": 42.5
  },
  "video_calls": {
    "total_calls": 34,
    "total_participants": 178,
    "average_duration": 28.3,
    "recording_usage": 67.6
  },
  "websocket_stats": {
    "total_connections": 127,
    "unique_users": 45,
    "unique_pages": 18
  }
}
```

### Get Presence Analytics

Get real-time presence analytics.

```http
GET /api/v1/rtc/analytics/presence
```

**Response:**
```json
{
  "realtime_stats": {
    "total_connections": 89,
    "unique_users": 34,
    "unique_pages": 12,
    "connections_by_page": {
      "/admin/users/list": 8,
      "/finance/budget/planning": 5,
      "/crm/opportunities/list": 3
    }
  },
  "timestamp": "2024-01-30T11:25:00Z",
  "tenant_id": "tenant123"
}
```

## 🔌 WebSocket Protocol

### Connection

Connect to the WebSocket for real-time communication.

```
wss://your-domain.com/api/v1/rtc/ws/{tenant_id}/{user_id}?page_url=/admin/users/list
```

### Message Types

#### Presence Update
```json
{
  "type": "presence_update",
  "user_id": "user123",
  "page_url": "/admin/users/list",
  "status": "active",
  "timestamp": "2024-01-30T11:30:00Z"
}
```

#### Chat Message
```json
{
  "type": "chat_message",
  "user_id": "user123",
  "username": "John Doe",
  "message": "Hello team!",
  "page_url": "/admin/users/list",
  "timestamp": "2024-01-30T11:31:00Z"
}
```

#### Form Delegation
```json
{
  "type": "form_delegation",
  "delegator_id": "user123",
  "delegatee_id": "user456",
  "field_name": "email",
  "instructions": "Please fill this field",
  "page_url": "/admin/users/add",
  "timestamp": "2024-01-30T11:32:00Z"
}
```

#### Assistance Request
```json
{
  "type": "assistance_request",
  "requester_id": "user123",
  "field_name": "permissions",
  "description": "Need help with user permissions",
  "page_url": "/admin/users/edit/456",
  "timestamp": "2024-01-30T11:33:00Z"
}
```

#### Video Call Events
```json
{
  "type": "video_call_start",
  "call_id": "01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "host_id": "user123",
  "meeting_url": "/rtc-video/meeting/01HKQK2V3N4M5P6Q7R8S9T0U1Y",
  "timestamp": "2024-01-30T11:34:00Z"
}
```

#### Heartbeat
```json
{
  "type": "heartbeat",
  "timestamp": "2024-01-30T11:35:00Z"
}
```

## ❌ Error Handling

### Standard Error Response

All API endpoints return errors in a consistent format:

```json
{
  "error": "Error description",
  "error_code": "RTC_001",
  "timestamp": "2024-01-30T11:40:00Z",
  "status_code": 400,
  "request_id": "req_01HKQK2V3N4M5P6Q7R8S9T0U26"
}
```

### Common Error Codes

| Code | Description | Status |
|------|-------------|--------|
| RTC_001 | Invalid session ID | 400 |
| RTC_002 | Session not found | 404 |
| RTC_003 | Permission denied | 403 |
| RTC_004 | User not in session | 400 |
| RTC_005 | Session full | 409 |
| RTC_006 | Invalid video call ID | 400 |
| RTC_007 | Video call not active | 400 |
| RTC_008 | Recording not supported | 400 |
| RTC_009 | Integration not configured | 400 |
| RTC_010 | WebSocket connection failed | 503 |

### Error Handling Best Practices

1. **Always check status codes** before processing responses
2. **Log error_code and request_id** for troubleshooting
3. **Implement retry logic** for transient errors (5xx codes)
4. **Show user-friendly messages** instead of raw error text
5. **Handle WebSocket disconnections** gracefully with reconnection

## 🚦 Rate Limiting

### Rate Limits

Different endpoints have different rate limits:

| Endpoint Category | Rate Limit | Window |
|------------------|------------|--------|
| Session Management | 100 req/min | Per user |
| Chat Messages | 1000 msg/min | Per user |
| Video Calls | 50 req/min | Per user |
| Analytics | 100 req/min | Per tenant |
| WebSocket Messages | 1000 msg/min | Per connection |

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706616000
X-RateLimit-Window: 60
```

### Rate Limit Exceeded

When rate limit is exceeded:

```json
{
  "error": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "status_code": 429,
  "retry_after": 60,
  "timestamp": "2024-01-30T11:45:00Z"
}
```

## 🏥 Health & Status

### Health Check

Check the health of the collaboration service.

```http
GET /api/v1/rtc/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-30T11:50:00Z",
  "websocket_stats": {
    "total_connections": 234,
    "unique_users": 89
  },
  "version": "1.0.0"
}
```

### Service Status

Get detailed service status and configuration.

```http
GET /api/v1/rtc/status
```

**Response:**
```json
{
  "service": "real-time-collaboration",
  "version": "1.0.0",
  "features": {
    "page_collaboration": true,
    "video_calls": true,
    "screen_sharing": true,
    "recording": true,
    "teams_integration": true,
    "zoom_integration": true,
    "google_meet_integration": true,
    "ai_features": true,
    "analytics": true
  },
  "limits": {
    "max_participants_per_call": 100,
    "max_concurrent_sessions": 1000,
    "recording_duration_hours": 8,
    "file_share_size_mb": 100
  }
}
```

## 📚 SDK Examples

### JavaScript/TypeScript

```typescript
import { RTCClient } from '@apg/rtc-client';

const client = new RTCClient({
  baseUrl: 'https://your-api.com',
  token: 'your-auth-token',
  tenantId: 'your-tenant-id'
});

// Create session
const session = await client.sessions.create({
  sessionName: 'Budget Review',
  sessionType: 'page_collaboration',
  pageUrl: '/finance/budget/review'
});

// Join session
const participant = await client.sessions.join(session.sessionId, 'viewer');

// Start video call
const videoCall = await client.videoCalls.start({
  callName: 'Team Sync',
  pageUrl: '/admin/dashboard'
});
```

### Python

```python
from apg_rtc import RTCClient

client = RTCClient(
    base_url='https://your-api.com',
    token='your-auth-token',
    tenant_id='your-tenant-id'
)

# Create session
session = await client.sessions.create(
    session_name='Budget Review',
    session_type='page_collaboration',
    page_url='/finance/budget/review'
)

# Join session
participant = await client.sessions.join(session.session_id, role='viewer')

# Start video call
video_call = await client.video_calls.start(
    call_name='Team Sync',
    page_url='/admin/dashboard'
)
```

---

## 📞 Support

- **API Questions**: [Developer forum](https://dev.apg.com/forum)
- **Bug Reports**: [GitHub Issues](https://github.com/apg/rtc/issues)
- **Feature Requests**: [Feature portal](https://features.apg.com)
- **Enterprise Support**: [Support portal](https://support.apg.com)

---

**© 2025 Datacraft | Contact: nyimbi@gmail.com | Website: www.datacraft.co.ke**

*Complete API reference for APG Real-Time Collaboration*