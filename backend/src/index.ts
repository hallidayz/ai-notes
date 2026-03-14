import express from 'express';
import { Pool } from 'pg';
import cors from 'cors';
import jwt from 'jsonwebtoken';

// Basic Express + Postgres skeleton for MeetingMinds API
// Note: this is intentionally minimal; real deployment should
// configure SSL, connection pooling, secrets, etc.

const app = express();
app.use(express.json());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));

// In a real deployment, these come from server-side env vars, never the PWA.
const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

// Middleware to set app.current_user_id for RLS policies.
// Uses JWT authentication.
app.use(async (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return next();
  }

  const token = authHeader.split(' ')[1];
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    console.error('JWT_SECRET environment variable is not set');
    return res.status(500).json({ error: 'Internal server error' });
  }

  try {
    const decoded = jwt.verify(token, secret) as jwt.JwtPayload;
    const userId = decoded.sub || decoded.userId;

    if (!userId) {
      return res.status(401).json({ error: 'Invalid token: missing subject/userId' });
    }

    (req as any).userId = userId as string;

    try {
      await pool.query('SELECT set_config($1, $2, true)', [
        'app.current_user_id',
        userId
      ]);
    } catch (e) {
      console.error('Failed to set app.current_user_id', e);
      // Even if setting config fails, we let the request proceed since auth succeeded
    }

    next();
  } catch (error) {
    console.error('JWT verification failed:', error);
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
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

// OAuth endpoints for calendar integrations

/**
 * Google Calendar OAuth Token Exchange
 * Exchanges authorization code for access/refresh tokens
 */
app.post('/api/oauth/google/token', async (req, res) => {
  try {
    const { code, redirect_uri } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Authorization code required' });
    }

    const clientId = process.env.GOOGLE_CLIENT_ID;
    const clientSecret = process.env.GOOGLE_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      console.error('Google OAuth credentials not configured');
      return res.status(500).json({ error: 'OAuth not configured' });
    }

    // Exchange code for tokens
    const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        code,
        client_id: clientId,
        client_secret: clientSecret,
        redirect_uri: redirect_uri || `${process.env.FRONTEND_URL || 'http://localhost:5173'}/oauth/google/callback`,
        grant_type: 'authorization_code',
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Google token exchange error:', errorText);
      return res.status(tokenResponse.status).json({ error: 'Token exchange failed' });
    }

    const tokens = await tokenResponse.json();
    res.json(tokens);
  } catch (error: any) {
    console.error('Google OAuth error:', error);
    res.status(500).json({ error: error.message || 'OAuth token exchange failed' });
  }
});

/**
 * Google Calendar OAuth Token Refresh
 */
app.post('/api/oauth/google/refresh', async (req, res) => {
  try {
    const { refresh_token } = req.body;
    
    if (!refresh_token) {
      return res.status(400).json({ error: 'Refresh token required' });
    }

    const clientId = process.env.GOOGLE_CLIENT_ID;
    const clientSecret = process.env.GOOGLE_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      return res.status(500).json({ error: 'OAuth not configured' });
    }

    const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        refresh_token,
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'refresh_token',
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Google token refresh error:', errorText);
      return res.status(tokenResponse.status).json({ error: 'Token refresh failed' });
    }

    const tokens = await tokenResponse.json();
    res.json(tokens);
  } catch (error: any) {
    console.error('Google OAuth refresh error:', error);
    res.status(500).json({ error: error.message || 'Token refresh failed' });
  }
});

/**
 * Outlook/Microsoft OAuth Token Exchange
 */
app.post('/api/oauth/outlook/token', async (req, res) => {
  try {
    const { code, redirect_uri } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Authorization code required' });
    }

    const clientId = process.env.OUTLOOK_CLIENT_ID;
    const clientSecret = process.env.OUTLOOK_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      console.error('Outlook OAuth credentials not configured');
      return res.status(500).json({ error: 'OAuth not configured' });
    }

    // Exchange code for tokens
    const tokenResponse = await fetch('https://login.microsoftonline.com/common/oauth2/v2.0/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        code,
        client_id: clientId,
        client_secret: clientSecret,
        redirect_uri: redirect_uri || `${process.env.FRONTEND_URL || 'http://localhost:5173'}/oauth/outlook/callback`,
        grant_type: 'authorization_code',
        scope: 'Calendars.Read offline_access',
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Outlook token exchange error:', errorText);
      return res.status(tokenResponse.status).json({ error: 'Token exchange failed' });
    }

    const tokens = await tokenResponse.json();
    res.json(tokens);
  } catch (error: any) {
    console.error('Outlook OAuth error:', error);
    res.status(500).json({ error: error.message || 'OAuth token exchange failed' });
  }
});

/**
 * Outlook/Microsoft OAuth Token Refresh
 */
app.post('/api/oauth/outlook/refresh', async (req, res) => {
  try {
    const { refresh_token } = req.body;
    
    if (!refresh_token) {
      return res.status(400).json({ error: 'Refresh token required' });
    }

    const clientId = process.env.OUTLOOK_CLIENT_ID;
    const clientSecret = process.env.OUTLOOK_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      return res.status(500).json({ error: 'OAuth not configured' });
    }

    const tokenResponse = await fetch('https://login.microsoftonline.com/common/oauth2/v2.0/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        refresh_token,
        client_id: clientId,
        client_secret: clientSecret,
        grant_type: 'refresh_token',
        scope: 'Calendars.Read offline_access',
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Outlook token refresh error:', errorText);
      return res.status(tokenResponse.status).json({ error: 'Token refresh failed' });
    }

    const tokens = await tokenResponse.json();
    res.json(tokens);
  } catch (error: any) {
    console.error('Outlook OAuth refresh error:', error);
    res.status(500).json({ error: error.message || 'Token refresh failed' });
  }
});

const port = process.env.PORT || 4000;
app.listen(port, () => {
  console.log(`meetingminds-api listening on port ${port}`);
});


