# Security review (Phase 7)

Scope: what the code does today, what was checked, and what is still open. "Checked" means a test exists or I read
the code path; it is not an external audit.

## Controls in place
| Area | Control | Where |
|---|---|---|
| Passwords | Argon2id (argon2-cffi defaults), never logged; login says the same thing for unknown email and wrong password | `auth/security.py`, `api/auth.py` |
| Sessions | 15-minute JWT access tokens; 7-day refresh tokens, single use, rotated; replaying a used refresh token revokes all of that user's refresh tokens; token `type` is checked so an access token cannot refresh | `auth/security.py`, `tests/test_auth.py` |
| Production guard | API refuses to start in `MLREADY_ENV=production` with the default JWT secret | `config.py` (Phase 1) |
| Authorisation | Every route resolves the caller's workspace and every query includes `workspace_id`; foreign ids return 404 (not 403) so existence is not leaked; viewer role is read-only | `api/deps.py`, `tests/test_auth.py::test_workspaces_are_isolated` |
| Rate limiting | Mongo-backed fixed window (correct across several API processes) on register/login/guest/refresh and uploads | `auth/ratelimit.py` |
| Uploads | Extension allow-list, magic-byte check, size cap enforced while streaming, zip-bomb guard for xlsx, filename sanitising | `ingestion/validate.py` (Phase 1) |
| CSV injection | Opt-in `?spreadsheet_safe=true` download prefixes formula-like text cells; HTML reports escape every dynamic string and ship a restrictive CSP | `ingestion/csvsafe.py`, `reports/html.py`, `api/reports.py` |
| Share links | 24-byte random token, per-link expiry (1-90 days), revocable, unknown/expired/revoked all return the same 404, shared page is read-only HTML served with `default-src 'none'` | `api/reports.py` |
| Privacy | No row values are logged (logs carry ids, counts, timings); evidence in findings holds a handful of example values by design and is only visible to the owning workspace; one-click "delete everything" removes versions, files, analyses, findings, plans, model runs, reports and jobs; demo data auto-expires | `storage.py`, `jobs/worker.py` |
| Container | API/worker run as a non-root user | `backend/Dockerfile` |

## Known gaps (not done)
* **Encryption at rest** is delegated to the database/disk (enable MongoDB encrypted storage engine or volume encryption). The app does not encrypt GridFS content itself.
* **Tokens in the browser** live in `sessionStorage` (cleared with the tab) and are readable by any script on the page; there is no HttpOnly-cookie mode and no CSRF layer because auth is bearer-header based.
* **No email verification or password reset**, no account lockout beyond the rate limit, no MFA.
* **Model/engine code runs in-process** on the worker. Uploaded data is parsed with pandas/openpyxl/pyarrow; a parser CVE would execute in the worker. Run workers with no network egress and a read-only filesystem for defence in depth.
* **Exported scripts** execute on the user's machine: they contain only generated, parameterised calls (column names and parameters are rendered with `pprint`/`repr`), but users should read a script before running it.
* **Prometheus metrics** are per-process and unauthenticated; do not expose `/metrics` publicly.
* **Audit log** records actions but has no UI and no retention policy.
* The SSE endpoint polls MongoDB every 0.5 s per open stream; many concurrent streams will load the database.
