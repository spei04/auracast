"""
Google OAuth2 + API connection (Phase 2 — skeleton for now).

The real flow uses google-auth + google-auth-oauthlib to do the installed-app
OAuth dance, caches the token to a local JSON file (gitignored), and returns
an authorized `Credentials` object that the Google API clients consume.

Phase 0 ships only the interface so downstream `ingest/google_pipeline.py`
(not yet written) can be designed against it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Scopes we'll need for Phase 2. Keep narrow — Photos read-only is enough for
# curation; Drive is a fallback ingest path.
DEFAULT_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
)


@dataclass(frozen=True)
class OAuthConfig:
    client_secrets_path: Path
    token_cache_path: Path
    scopes: tuple[str, ...] = DEFAULT_SCOPES


def load_credentials(cfg: OAuthConfig):
    """Returns an authorized google-auth Credentials object.

    Phase 2 will implement the full flow:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials

        if cfg.token_cache_path.exists():
            creds = Credentials.from_authorized_user_file(str(cfg.token_cache_path), cfg.scopes)
            if creds.valid:
                return creds
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                cfg.token_cache_path.write_text(creds.to_json())
                return creds

        flow = InstalledAppFlow.from_client_secrets_file(str(cfg.client_secrets_path), cfg.scopes)
        creds = flow.run_local_server(port=0)
        cfg.token_cache_path.write_text(creds.to_json())
        return creds
    """
    raise NotImplementedError(
        "Google OAuth lands in Phase 2. Use LocalDirectoryIngest until then."
    )
