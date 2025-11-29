CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Core users and auth tables
CREATE TABLE users (
  user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE NOT NULL,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT,
  last_name TEXT,
  account_status TEXT NOT NULL DEFAULT 'active',
  email_verified_at TIMESTAMP,
  last_login_at TIMESTAMP,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE roles (
  role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  role_name TEXT UNIQUE NOT NULL,
  role_description TEXT
);

CREATE TABLE permissions (
  permission_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  permission_name TEXT UNIQUE NOT NULL,
  permission_description TEXT
);

CREATE TABLE role_permissions (
  role_id UUID NOT NULL REFERENCES roles(role_id) ON DELETE CASCADE,
  permission_id UUID NOT NULL REFERENCES permissions(permission_id) ON DELETE CASCADE,
  PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE user_roles (
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  role_id UUID NOT NULL REFERENCES roles(role_id) ON DELETE CASCADE,
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, role_id)
);

-- Meetings and participants
CREATE TABLE meetings (
  meeting_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_id UUID NOT NULL REFERENCES users(user_id),
  title TEXT NOT NULL,
  description TEXT,
  meeting_status TEXT NOT NULL DEFAULT 'scheduled',
  scheduled_start TIMESTAMPTZ,
  scheduled_end TIMESTAMPTZ,
  actual_start TIMESTAMPTZ,
  actual_end TIMESTAMPTZ,
  meeting_duration_seconds INT,
  location_type TEXT,
  recording_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE participants (
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  participant_role TEXT NOT NULL DEFAULT 'attendee',
  join_time TIMESTAMPTZ,
  leave_time TIMESTAMPTZ,
  recording_consent BOOLEAN DEFAULT FALSE,
  PRIMARY KEY (meeting_id, user_id)
);

-- Transcripts and segments
CREATE TABLE transcripts (
  transcript_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  full_transcript_text TEXT,
  transcript_source TEXT,
  transcription_confidence REAL,
  raw_transcript_data JSONB,
  total_speakers INT,
  language_code TEXT DEFAULT 'en',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE transcript_segments (
  segment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  transcript_id UUID NOT NULL REFERENCES transcripts(transcript_id) ON DELETE CASCADE,
  segment_number INT NOT NULL,
  start_ts_seconds NUMERIC(10,3),
  end_ts_seconds NUMERIC(10,3),
  speaker_name TEXT,
  speaker_user_id UUID REFERENCES users(user_id),
  segment_text TEXT NOT NULL,
  confidence REAL,
  UNIQUE (transcript_id, segment_number)
);

-- Notes and summaries
CREATE TABLE meeting_notes (
  note_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  created_by UUID NOT NULL REFERENCES users(user_id),
  note_content TEXT,
  note_format TEXT DEFAULT 'markdown',
  is_ai_generated BOOLEAN DEFAULT FALSE,
  note_type TEXT DEFAULT 'summary',
  is_shared BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE meeting_summaries (
  summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  summary_content TEXT NOT NULL,
  key_topics TEXT[],
  sentiment_score NUMERIC(4,3),
  model_version TEXT,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  confidence_score NUMERIC(4,3)
);

-- Action items
CREATE TABLE action_items (
  action_item_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  originating_segment_id UUID REFERENCES transcript_segments(segment_id),
  title TEXT NOT NULL,
  description TEXT,
  assigned_to UUID NOT NULL REFERENCES users(user_id),
  created_by UUID NOT NULL REFERENCES users(user_id),
  priority_level TEXT DEFAULT 'medium',
  due_date DATE,
  due_timezone TEXT,
  status TEXT DEFAULT 'open',
  completion_date TIMESTAMPTZ,
  estimated_effort_hours NUMERIC(6,2),
  actual_effort_hours NUMERIC(6,2),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tagging
CREATE TABLE tags (
  tag_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_by UUID NOT NULL REFERENCES users(user_id),
  tag_name TEXT NOT NULL,
  tag_description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (created_by, tag_name)
);

CREATE TABLE meeting_tags (
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  tag_id UUID NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
  PRIMARY KEY (meeting_id, tag_id)
);

-- Additional metadata/customization
CREATE TABLE meeting_metadata (
  metadata_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  metadata_key TEXT NOT NULL,
  metadata_value TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (meeting_id, metadata_key)
);

CREATE TABLE meeting_custom_fields (
  custom_field_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  meeting_id UUID NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  field_name TEXT NOT NULL,
  field_type TEXT NOT NULL,
  field_value JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (meeting_id, field_name)
);


