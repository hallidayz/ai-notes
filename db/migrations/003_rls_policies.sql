-- Enable Row Level Security on sensitive tables
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_privacy_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE meetings ENABLE ROW LEVEL SECURITY;
ALTER TABLE meeting_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE meeting_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcripts ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_items ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies using app.current_user_id

CREATE POLICY user_owns_settings ON user_settings
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY user_owns_preferences ON user_preferences
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY user_owns_privacy ON user_privacy_settings
  USING (user_id = current_setting('app.current_user_id')::uuid);

CREATE POLICY user_owns_notifications ON notification_preferences
  USING (user_id = current_setting('app.current_user_id')::uuid);

-- Meetings: owner or explicitly granted access
CREATE POLICY meeting_owner_or_shared ON meetings
  USING (
    owner_id = current_setting('app.current_user_id')::uuid
    OR EXISTS (
      SELECT 1 FROM user_access_control uac
      WHERE uac.user_id = current_setting('app.current_user_id')::uuid
        AND uac.resource_type = 'meeting'
        AND uac.resource_id = meetings.meeting_id
        AND (uac.expires_at IS NULL OR uac.expires_at > NOW())
    )
  );

-- Meeting notes inherit meeting access
CREATE POLICY notes_inherit_meeting ON meeting_notes
  USING (
    EXISTS (
      SELECT 1 FROM meetings
      WHERE meetings.meeting_id = meeting_notes.meeting_id
      AND (
        meetings.owner_id = current_setting('app.current_user_id')::uuid
        OR EXISTS (
          SELECT 1 FROM user_access_control uac
          WHERE uac.user_id = current_setting('app.current_user_id')::uuid
            AND uac.resource_type = 'meeting'
            AND uac.resource_id = meeting_notes.meeting_id
            AND (uac.expires_at IS NULL OR uac.expires_at > NOW())
        )
      )
    )
  );

-- Meeting summaries inherit meeting access
CREATE POLICY summaries_inherit_meeting ON meeting_summaries
  USING (
    EXISTS (
      SELECT 1 FROM meetings
      WHERE meetings.meeting_id = meeting_summaries.meeting_id
      AND (
        meetings.owner_id = current_setting('app.current_user_id')::uuid
        OR EXISTS (
          SELECT 1 FROM user_access_control uac
          WHERE uac.user_id = current_setting('app.current_user_id')::uuid
            AND uac.resource_type = 'meeting'
            AND uac.resource_id = meeting_summaries.meeting_id
            AND (uac.expires_at IS NULL OR uac.expires_at > NOW())
        )
      )
    )
  );

-- Transcripts inherit meeting access
CREATE POLICY transcripts_inherit_meeting ON transcripts
  USING (
    EXISTS (
      SELECT 1 FROM meetings
      WHERE meetings.meeting_id = transcripts.meeting_id
      AND (
        meetings.owner_id = current_setting('app.current_user_id')::uuid
        OR EXISTS (
          SELECT 1 FROM user_access_control uac
          WHERE uac.user_id = current_setting('app.current_user_id')::uuid
            AND uac.resource_type = 'meeting'
            AND uac.resource_id = transcripts.meeting_id
            AND (uac.expires_at IS NULL OR uac.expires_at > NOW())
        )
      )
    )
  );

-- Action items visibility: assignee, creator, meeting owner, or explicit access
CREATE POLICY action_items_visibility ON action_items
  USING (
    assigned_to = current_setting('app.current_user_id')::uuid
    OR created_by = current_setting('app.current_user_id')::uuid
    OR EXISTS (
      SELECT 1 FROM meetings
      WHERE meetings.meeting_id = action_items.meeting_id
      AND meetings.owner_id = current_setting('app.current_user_id')::uuid
    )
    OR EXISTS (
      SELECT 1 FROM user_access_control uac
      WHERE uac.user_id = current_setting('app.current_user_id')::uuid
        AND (
          (uac.resource_type = 'meeting' AND uac.resource_id = action_items.meeting_id) OR
          (uac.resource_type = 'action_item' AND uac.resource_id = action_items.action_item_id)
        )
        AND (uac.expires_at IS NULL OR uac.expires_at > NOW())
    )
  );


