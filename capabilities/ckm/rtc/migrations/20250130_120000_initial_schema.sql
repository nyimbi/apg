-- Migration: initial_schema
-- Created: 2025-01-30T12:00:00

-- UP
-- Create rtc_sessions table
CREATE TABLE IF NOT EXISTS rtc_sessions (
	session_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	session_name VARCHAR(200) NOT NULL,
	session_type VARCHAR(50) NOT NULL DEFAULT 'page_collaboration',
	digital_twin_id VARCHAR(200),
	owner_user_id VARCHAR(100) NOT NULL,
	is_active BOOLEAN DEFAULT true,
	max_participants INTEGER DEFAULT 10,
	current_participant_count INTEGER DEFAULT 0,
	participant_user_ids TEXT DEFAULT '[]',
	collaboration_mode VARCHAR(50) DEFAULT 'open',
	require_approval BOOLEAN DEFAULT false,
	scheduled_start TIMESTAMP,
	scheduled_end TIMESTAMP,
	actual_start TIMESTAMP,
	actual_end TIMESTAMP,
	duration_minutes REAL,
	recording_enabled BOOLEAN DEFAULT true,
	voice_chat_enabled BOOLEAN DEFAULT false,
	video_chat_enabled BOOLEAN DEFAULT false,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rtc_participants table
CREATE TABLE IF NOT EXISTS rtc_participants (
	participant_id VARCHAR(36) PRIMARY KEY,
	session_id VARCHAR(36) NOT NULL,
	user_id VARCHAR(100) NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	display_name VARCHAR(200),
	role VARCHAR(50) DEFAULT 'viewer',
	joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	left_at TIMESTAMP,
	is_active BOOLEAN DEFAULT true,
	can_edit BOOLEAN DEFAULT false,
	can_annotate BOOLEAN DEFAULT true,
	can_chat BOOLEAN DEFAULT true,
	can_share_screen BOOLEAN DEFAULT false,
	last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	presence_status VARCHAR(50) DEFAULT 'active',
	cursor_position TEXT,
	FOREIGN KEY (session_id) REFERENCES rtc_sessions(session_id) ON DELETE CASCADE
);

-- Create rtc_video_calls table
CREATE TABLE IF NOT EXISTS rtc_video_calls (
	call_id VARCHAR(36) PRIMARY KEY,
	session_id VARCHAR(36) NOT NULL,
	tenant_id VARCHAR(100) NOT NULL,
	call_name VARCHAR(200) NOT NULL,
	call_type VARCHAR(50) DEFAULT 'video',
	status VARCHAR(50) DEFAULT 'scheduled',
	meeting_id VARCHAR(100),
	host_user_id VARCHAR(100) NOT NULL,
	current_participants INTEGER DEFAULT 0,
	max_participants INTEGER DEFAULT 100,
	scheduled_start TIMESTAMP,
	started_at TIMESTAMP,
	ended_at TIMESTAMP,
	duration_minutes REAL,
	video_quality VARCHAR(20) DEFAULT 'hd',
	audio_quality VARCHAR(20) DEFAULT 'high',
	enable_recording BOOLEAN DEFAULT false,
	waiting_room_enabled BOOLEAN DEFAULT true,
	end_to_end_encryption BOOLEAN DEFAULT true,
	breakout_rooms_enabled BOOLEAN DEFAULT false,
	polls_enabled BOOLEAN DEFAULT true,
	whiteboard_enabled BOOLEAN DEFAULT true,
	screen_sharing_enabled BOOLEAN DEFAULT true,
	chat_enabled BOOLEAN DEFAULT true,
	teams_meeting_url TEXT,
	teams_meeting_id VARCHAR(100),
	zoom_meeting_id VARCHAR(100),
	meet_url TEXT,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY (session_id) REFERENCES rtc_sessions(session_id) ON DELETE CASCADE
);

-- Create rtc_page_collaboration table
CREATE TABLE IF NOT EXISTS rtc_page_collaboration (
	page_collab_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	page_url VARCHAR(500) NOT NULL,
	page_title VARCHAR(200),
	page_type VARCHAR(100),
	blueprint_name VARCHAR(100),
	view_name VARCHAR(100),
	is_active BOOLEAN DEFAULT true,
	current_users TEXT DEFAULT '[]',
	total_collaboration_sessions INTEGER DEFAULT 0,
	total_form_delegations INTEGER DEFAULT 0,
	total_assistance_requests INTEGER DEFAULT 0,
	average_users_per_session REAL DEFAULT 0.0,
	delegated_fields TEXT DEFAULT '{}',
	assistance_requests TEXT DEFAULT '[]',
	first_collaboration TIMESTAMP,
	last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rtc_third_party_integrations table
CREATE TABLE IF NOT EXISTS rtc_third_party_integrations (
	integration_id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(100) NOT NULL,
	platform VARCHAR(50) NOT NULL,
	platform_name VARCHAR(100) NOT NULL,
	integration_type VARCHAR(50) DEFAULT 'api',
	status VARCHAR(50) DEFAULT 'active',
	api_key VARCHAR(500),
	api_secret VARCHAR(500),
	webhook_url VARCHAR(500),
	last_sync TIMESTAMP,
	sync_frequency_minutes INTEGER DEFAULT 60,
	total_meetings_synced INTEGER DEFAULT 0,
	total_api_calls INTEGER DEFAULT 0,
	monthly_api_limit INTEGER,
	current_month_usage INTEGER DEFAULT 0,
	sync_meetings BOOLEAN DEFAULT true,
	sync_participants BOOLEAN DEFAULT true,
	sync_recordings BOOLEAN DEFAULT true,
	auto_create_meetings BOOLEAN DEFAULT false,
	teams_tenant_id VARCHAR(100),
	teams_application_id VARCHAR(100),
	zoom_account_id VARCHAR(100),
	google_workspace_domain VARCHAR(200),
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_rtc_sessions_tenant_active ON rtc_sessions(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_rtc_sessions_owner ON rtc_sessions(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_participants_session ON rtc_participants(session_id);
CREATE INDEX IF NOT EXISTS idx_rtc_participants_user ON rtc_participants(user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_video_calls_session ON rtc_video_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_rtc_video_calls_host ON rtc_video_calls(host_user_id);
CREATE INDEX IF NOT EXISTS idx_rtc_page_collaboration_url ON rtc_page_collaboration(page_url);
CREATE INDEX IF NOT EXISTS idx_rtc_page_collaboration_tenant ON rtc_page_collaboration(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rtc_integrations_tenant ON rtc_third_party_integrations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rtc_integrations_platform ON rtc_third_party_integrations(platform);

-- DOWN (for rollback)
-- Drop tables in reverse order
DROP TABLE IF EXISTS rtc_third_party_integrations;
DROP TABLE IF EXISTS rtc_page_collaboration;
DROP TABLE IF EXISTS rtc_video_calls;
DROP TABLE IF EXISTS rtc_participants;
DROP TABLE IF EXISTS rtc_sessions;