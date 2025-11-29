-- User-level settings and privacy
CREATE TABLE user_settings (
  settings_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  setting_key TEXT NOT NULL,
  setting_value JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id, setting_key)
);

CREATE TABLE user_preferences (
  preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  calendar_integration_enabled BOOLEAN DEFAULT FALSE,
  automatic_recording_consent BOOLEAN DEFAULT FALSE,
  ai_note_generation_enabled BOOLEAN DEFAULT TRUE,
  action_item_auto_assignment BOOLEAN DEFAULT FALSE,
  default_meeting_template_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id)
);

CREATE TABLE user_privacy_settings (
  privacy_setting_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  allow_meeting_recording BOOLEAN DEFAULT TRUE,
  allow_transcription BOOLEAN DEFAULT TRUE,
  allow_ai_analysis BOOLEAN DEFAULT TRUE,
  data_retention_days INT,
  allow_tagging BOOLEAN DEFAULT TRUE,
  visibility_level TEXT DEFAULT 'team_only',
  gdpr_consent BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id)
);

CREATE TABLE notification_preferences (
  notification_preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  email_notifications_enabled BOOLEAN DEFAULT TRUE,
  action_item_reminders_enabled BOOLEAN DEFAULT TRUE,
  meeting_start_reminders_enabled BOOLEAN DEFAULT TRUE,
  summary_delivery TEXT DEFAULT 'immediate',
  quiet_hours_start TIME,
  quiet_hours_end TIME,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (user_id)
);

-- Fine-grained access control
CREATE TABLE user_access_control (
  access_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  resource_type TEXT NOT NULL,
  resource_id UUID NOT NULL,
  access_level TEXT NOT NULL,
  granted_by UUID NOT NULL REFERENCES users(user_id),
  granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ
);

CREATE INDEX idx_uac_resource ON user_access_control(resource_type, resource_id);
CREATE INDEX idx_uac_user_resource ON user_access_control(user_id, resource_type, resource_id);

-- Audit log
CREATE TABLE audit_log (
  audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(user_id),
  action_type TEXT NOT NULL,
  resource_type TEXT,
  resource_id UUID,
  old_value JSONB,
  new_value JSONB,
  ip_address INET,
  user_agent TEXT,
  performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Consent & retention
CREATE TABLE consent_records (
  consent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  consent_type TEXT NOT NULL,
  consent_given BOOLEAN NOT NULL,
  consent_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ip_address INET,
  user_agent TEXT,
  expires_at TIMESTAMPTZ
);

CREATE TABLE data_retention_schedule (
  retention_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  resource_type TEXT NOT NULL,
  resource_id UUID NOT NULL,
  scheduled_deletion TIMESTAMPTZ NOT NULL,
  deleted BOOLEAN DEFAULT FALSE,
  deleted_at TIMESTAMPTZ,
  UNIQUE (resource_type, resource_id)
);

-- Envelope-encrypted blobs
CREATE TABLE encrypted_user_data (
  encrypted_data_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  data_type TEXT NOT NULL,
  encrypted_value BYTEA NOT NULL,
  key_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_accessed TIMESTAMPTZ,
  UNIQUE (user_id, data_type)
);


