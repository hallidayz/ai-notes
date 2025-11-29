import express from 'express';
import { Pool } from 'pg';

// Basic Express + Postgres skeleton for MeetingMinds API
// Note: this is intentionally minimal; real deployment should
// configure SSL, connection pooling, secrets, etc.

const app = express();
app.use(express.json());

// In a real deployment, these come from server-side env vars, never the PWA.
const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

// Middleware to set app.current_user_id for RLS policies.
// For now this assumes an upstream auth layer populates req.userId.
app.use(async (req, _res, next) => {
  // TODO: integrate with real auth and JWT validation.
  const userId = (req as any).userId as string | undefined;

  if (userId) {
    try {
      await pool.query('SELECT set_config($1, $2, true)', [
        'app.current_user_id',
        userId
      ]);
    } catch (e) {
      console.error('Failed to set app.current_user_id', e);
    }
  }

  next();
});

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'meetingminds-api' });
});

// Skeleton sessions endpoint matching ai-notes semantics.
app.get('/api/sessions', async (_req, res) => {
  try {
    const result = await pool.query(
      `SELECT meeting_id, title, description, meeting_status, created_at
       FROM meetings
       ORDER BY created_at DESC
       LIMIT 100`
    );
    res.json(result.rows);
  } catch (e) {
    console.error('Error listing sessions', e);
    res.status(500).json({ error: 'Failed to list sessions' });
  }
});

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`meetingminds-api listening on port ${port}`);
});


