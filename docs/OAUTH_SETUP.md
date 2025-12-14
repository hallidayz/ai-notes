# OAuth Calendar Integration Setup Guide

This guide explains how to set up OAuth credentials for Google Calendar and Outlook/Microsoft 365 calendar integrations.

## Overview

The calendar integration uses OAuth 2.0 to securely connect to your calendar providers. You'll need to:

1. Create OAuth applications in Google Cloud Console (for Google Calendar) or Azure Portal (for Outlook)
2. Configure the client IDs in the app
3. Set up the backend API to handle token exchange (requires client secrets)

## Google Calendar Setup

### Step 1: Create OAuth Credentials in Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Calendar API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Calendar API"
   - Click "Enable"

4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - If prompted, configure the OAuth consent screen first
   - Choose "Web application" as the application type
   - Add authorized redirect URIs:
     - `http://localhost:5173/oauth/google/callback` (for development)
     - `https://yourdomain.com/oauth/google/callback` (for production)
   - Click "Create"
   - Copy the **Client ID** (you'll need this)

### Step 2: Configure in the App

1. Open the Calendar Settings in the app
2. Click "Configure" next to "OAuth Credentials"
3. Paste your Google Calendar Client ID
4. Click "Save Credentials"

### Step 3: Backend Configuration (Required for Token Exchange)

The backend needs the **Client Secret** to exchange authorization codes for tokens.

1. In Google Cloud Console, download the credentials JSON or copy the Client Secret
2. Set environment variables in your backend:
   ```bash
   export GOOGLE_CLIENT_ID="your-client-id"
   export GOOGLE_CLIENT_SECRET="your-client-secret"
   export FRONTEND_URL="http://localhost:5173"  # or your production URL
   ```

## Outlook/Microsoft 365 Setup

### Step 1: Register App in Azure Portal

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to "Azure Active Directory" > "App registrations"
3. Click "New registration"
4. Fill in:
   - **Name**: MiNDS Talk Calendar Integration (or your choice)
   - **Supported account types**: Accounts in any organizational directory and personal Microsoft accounts
   - **Redirect URI**: 
     - Type: Web
     - URI: `http://localhost:5173/oauth/outlook/callback` (for development)
     - Add another: `https://yourdomain.com/oauth/outlook/callback` (for production)
5. Click "Register"
6. Copy the **Application (client) ID** from the Overview page

### Step 2: Configure API Permissions

1. In your app registration, go to "API permissions"
2. Click "Add a permission"
3. Select "Microsoft Graph"
4. Choose "Delegated permissions"
5. Add the following permissions:
   - `Calendars.Read` (to read calendar events)
   - `offline_access` (to get refresh tokens)
6. Click "Add permissions"
7. Click "Grant admin consent" (if you're an admin)

### Step 3: Create Client Secret

1. Go to "Certificates & secrets"
2. Click "New client secret"
3. Add a description and choose expiration
4. Click "Add"
5. **Important**: Copy the secret value immediately (you won't see it again!)

### Step 4: Configure in the App

1. Open the Calendar Settings in the app
2. Click "Configure" next to "OAuth Credentials"
3. Paste your Microsoft Client ID
4. Click "Save Credentials"

### Step 5: Backend Configuration

Set environment variables in your backend:
```bash
export OUTLOOK_CLIENT_ID="your-client-id"
export OUTLOOK_CLIENT_SECRET="your-client-secret"
export FRONTEND_URL="http://localhost:5173"  # or your production URL
```

## Backend Setup

The backend API handles the secure token exchange (which requires client secrets).

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Outlook OAuth
OUTLOOK_CLIENT_ID=your-outlook-client-id
OUTLOOK_CLIENT_SECRET=your-outlook-client-secret

# Frontend URL (for CORS and redirect URIs)
FRONTEND_URL=http://localhost:5173

# Database (if using)
DATABASE_URL=your-database-url

# Port
PORT=4000
```

### Running the Backend

```bash
cd backend
npm install
npm run dev  # Development mode
# or
npm run build
npm start    # Production mode
```

## Frontend Configuration (Alternative)

You can also set client IDs as environment variables in the frontend (for development):

Create a `.env` file in the root directory:

```env
VITE_GOOGLE_CLIENT_ID=your-google-client-id
VITE_OUTLOOK_CLIENT_ID=your-outlook-client-id
VITE_API_URL=http://localhost:4000
```

**Note**: Client secrets should NEVER be in the frontend. They must be kept in the backend.

## Security Notes

1. **Client Secrets**: Never expose client secrets in frontend code or public repositories
2. **Redirect URIs**: Always use HTTPS in production
3. **Token Storage**: Tokens are encrypted and stored in browser localStorage
4. **Refresh Tokens**: The app automatically refreshes expired access tokens

## Troubleshooting

### "Client ID not configured" Error

- Make sure you've entered the client ID in Calendar Settings
- Check that the client ID is saved in localStorage
- Verify environment variables if using them

### "Token exchange failed" Error

- Ensure the backend is running
- Check that client secrets are set in backend environment variables
- Verify redirect URIs match exactly in both the OAuth provider and your app

### "Invalid redirect URI" Error

- Check that the redirect URI in your OAuth provider matches exactly:
  - Development: `http://localhost:5173/oauth/google/callback` or `/oauth/outlook/callback`
  - Production: `https://yourdomain.com/oauth/google/callback` or `/oauth/outlook/callback`

### CORS Errors

- Ensure `FRONTEND_URL` is set correctly in backend
- Check that the backend CORS configuration allows your frontend origin

## Testing

1. Start the backend: `cd backend && npm run dev`
2. Start the frontend: `npm run dev`
3. Open Calendar Settings
4. Enter your OAuth client IDs
5. Click "Connect" - you should be redirected to the OAuth provider
6. Authorize the app
7. You should be redirected back and see "Successfully connected"

## Production Deployment

1. Use environment variables for all secrets
2. Ensure backend is accessible from your frontend domain
3. Update redirect URIs in OAuth providers to production URLs
4. Use HTTPS for all OAuth redirects
5. Consider using a secrets management service (AWS Secrets Manager, Azure Key Vault, etc.)
