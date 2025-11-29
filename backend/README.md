## MeetingMinds API (backend skeleton)

This folder contains a minimal Express + Postgres backend that corresponds to the
execution roadmap in `secure.plan.md`. It is intentionally light-weight so it
does not disturb the existing `ai-notes` PWA.

### What exists now

- `db/migrations/001_initial_schema.sql` – core users/meetings/transcripts/action items schema.
- `db/migrations/002_security_layers.sql` – user privacy/settings, access control, audit, consent.
- `db/migrations/003_rls_policies.sql` – RLS policies based on `app.current_user_id`.
- `backend/src/index.ts` – Express server with:
  - Postgres connection via `pg`.
  - Middleware that sets `app.current_user_id` (placeholder; hook to real auth later).
  - `/health` and a basic `/api/sessions` listing endpoint.

### Next steps (per roadmap)

- Add real auth (JWT) middleware that populates `req.userId`.
- Flesh out REST endpoints for creating/updating/deleting sessions and action items.
- Add integration tests to verify RLS enforcement.


