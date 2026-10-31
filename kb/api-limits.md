# API request limits
Error REQUEST_LIMIT_EXCEEDED means the org used its 24-hour API allocation.
Check which integration user consumed the calls in the API usage log, then add backoff or move to the Bulk API for large loads.
