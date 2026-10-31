# Integration and sync failures
Errors such as INVALID_SESSION_ID mean the integration's OAuth token expired or was revoked. Re-authorize the connected app.
UNABLE_TO_LOCK_ROW appears when two jobs update the same records. Retry with backoff or serialize the jobs.
