"""
Google OAuth2 + API connection.

Installed-app OAuth flow via `google-auth-oauthlib`. Caches the token to a
local JSON file (gitignored). Refreshes on expiry. Returns an authorized
`google.oauth2.credentials.Credentials` object that the Google API clients
in this module's `build_*_service` helpers consume.

**First-time auth is interactive** — `flow.run_local_server()` opens a
browser. Run `python -m auracast.scripts.auth_setup` once; the resulting
`token.json` is reused silently afterwards (and refreshed when it expires).

Refresh tokens are long-lived (~6 months for Google) but not forever — if
the app ever errors with `invalid_grant`, re-run the setup script.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Scopes. We need full `drive` (not just `drive.readonly`) because the
# Streamlit "Finalize" action moves rejected images to Drive Trash, which
# requires write permission. Photos scope is kept for compatibility with
# pre-2024 GCP projects that still have Library API access.
DEFAULT_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/drive",
)


def _default_secrets_dir() -> Path:
    """Default secrets directory, overridable via AURACAST_SECRETS_DIR."""
    if env := os.environ.get("AURACAST_SECRETS_DIR"):
        return Path(env)
    return Path.home() / ".config" / "auracast"


def default_client_secrets_path() -> Path:
    if env := os.environ.get("AURACAST_CLIENT_SECRETS_PATH"):
        return Path(env)
    return _default_secrets_dir() / "client_secrets.json"


def default_token_path() -> Path:
    if env := os.environ.get("AURACAST_TOKEN_PATH"):
        return Path(env)
    return _default_secrets_dir() / "token.json"


@dataclass(frozen=True)
class OAuthConfig:
    client_secrets_path: Path = field(default_factory=default_client_secrets_path)
    token_cache_path: Path = field(default_factory=default_token_path)
    scopes: tuple[str, ...] = DEFAULT_SCOPES


def load_credentials(
    cfg: OAuthConfig | None = None,
    *,
    interactive: bool | None = None,
):
    """Return an authorized google-auth Credentials object.

    Args:
        cfg: OAuthConfig (paths + scopes). Defaults pick up env vars.
        interactive: If None (default), allow the browser flow only when a
            TTY is present. Set True/False to force either way.

    Raises:
        FileNotFoundError: token cache absent AND not allowed to run the
            interactive flow.
        RuntimeError: client_secrets.json missing when needed.
    """
    # Imports kept local so importing this module never costs us the
    # google-auth init time when callers only need the path helpers.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    cfg = cfg or OAuthConfig()
    if interactive is None:
        interactive = bool(os.isatty(0))  # heuristic: TTY -> probably a dev box

    creds = None
    if cfg.token_cache_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(cfg.token_cache_path), list(cfg.scopes)
            )
        except Exception as e:  # noqa: BLE001 — corrupt cache, treat as missing
            logger.warning("token cache %s unreadable (%s); ignoring", cfg.token_cache_path, e)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("refreshing expired Google OAuth token")
        creds.refresh(Request())
        _write_token(cfg.token_cache_path, creds)
        return creds

    # No usable cached creds — need the interactive flow.
    if not interactive:
        raise FileNotFoundError(
            f"No valid Google OAuth token at {cfg.token_cache_path}. "
            f"Run `python -m auracast.scripts.auth_setup` to generate one."
        )

    if not cfg.client_secrets_path.exists():
        raise RuntimeError(
            f"client_secrets.json not found at {cfg.client_secrets_path}. "
            f"Download from Google Cloud Console > APIs & Services > Credentials > "
            f"OAuth 2.0 Client IDs (Desktop app) and place it there, "
            f"or set AURACAST_CLIENT_SECRETS_PATH."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    logger.info("starting browser OAuth flow (scopes=%s)", cfg.scopes)
    flow = InstalledAppFlow.from_client_secrets_file(
        str(cfg.client_secrets_path), list(cfg.scopes)
    )
    creds = flow.run_local_server(port=0)
    _write_token(cfg.token_cache_path, creds)
    return creds


def _write_token(path: Path, creds) -> None:
    """Persist a Credentials object as JSON. Parent dir auto-created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json())
    try:
        path.chmod(0o600)
    except OSError:  # noqa: BLE001 — filesystem may not support chmod
        pass


# -------- Service builders ----------------------------------------------


def build_photos_service(credentials):
    """Returns a Google Photos Library API client."""
    from googleapiclient.discovery import build

    # static_discovery=False is required for the Photos Library API as of 2024.
    return build("photoslibrary", "v1", credentials=credentials, static_discovery=False)


def build_drive_service(credentials):
    """Returns a Google Drive API v3 client."""
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=credentials)
