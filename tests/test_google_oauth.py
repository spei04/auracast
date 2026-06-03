"""Tests for auracast.auth.google_oauth.

The browser flow is impossible to exercise in CI. We mock the
google-auth Credentials object + the InstalledAppFlow class and verify the
control flow: token cache load, refresh path, interactive-only fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from auracast.auth.google_oauth import (
    DEFAULT_SCOPES,
    OAuthConfig,
    default_client_secrets_path,
    default_token_path,
    load_credentials,
)


def test_default_paths_respect_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AURACAST_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("AURACAST_CLIENT_SECRETS_PATH", raising=False)
    monkeypatch.delenv("AURACAST_TOKEN_PATH", raising=False)
    assert default_client_secrets_path() == tmp_path / "client_secrets.json"
    assert default_token_path() == tmp_path / "token.json"


def test_default_paths_env_var_overrides_directly(monkeypatch, tmp_path):
    monkeypatch.setenv("AURACAST_TOKEN_PATH", str(tmp_path / "alt.json"))
    assert default_token_path() == tmp_path / "alt.json"


def test_oauth_config_uses_defaults():
    cfg = OAuthConfig()
    assert cfg.scopes == DEFAULT_SCOPES


def test_load_credentials_uses_valid_cached_token(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")  # contents are mocked

    fake_creds = MagicMock()
    fake_creds.valid = True

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=fake_creds):
        result = load_credentials(
            OAuthConfig(client_secrets_path=tmp_path / "cs.json", token_cache_path=token_path),
            interactive=False,
        )
    assert result is fake_creds


def test_load_credentials_refreshes_expired_token(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")

    fake_creds = MagicMock()
    fake_creds.valid = False
    fake_creds.expired = True
    fake_creds.refresh_token = "rt"
    fake_creds.to_json.return_value = '{"refreshed": true}'

    def _refresh(_request):
        fake_creds.valid = True

    fake_creds.refresh.side_effect = _refresh

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=fake_creds):
        load_credentials(
            OAuthConfig(client_secrets_path=tmp_path / "cs.json", token_cache_path=token_path),
            interactive=False,
        )
    fake_creds.refresh.assert_called_once()
    # Token was re-written
    assert token_path.read_text() == '{"refreshed": true}'


def test_load_credentials_non_interactive_no_token_raises(tmp_path):
    cfg = OAuthConfig(
        client_secrets_path=tmp_path / "missing_cs.json",
        token_cache_path=tmp_path / "missing_token.json",
    )
    with pytest.raises(FileNotFoundError, match="No valid Google OAuth token"):
        load_credentials(cfg, interactive=False)


def test_load_credentials_interactive_missing_client_secrets_raises(tmp_path):
    cfg = OAuthConfig(
        client_secrets_path=tmp_path / "missing_cs.json",
        token_cache_path=tmp_path / "missing_token.json",
    )
    with pytest.raises(RuntimeError, match="client_secrets.json not found"):
        load_credentials(cfg, interactive=True)


def test_load_credentials_corrupt_cache_treated_as_missing(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text("totally not json")

    cfg = OAuthConfig(
        client_secrets_path=tmp_path / "cs.json",
        token_cache_path=token_path,
    )
    # We fail before the InstalledAppFlow, because client_secrets is missing,
    # but the token-cache read should be tolerantly skipped.
    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               side_effect=ValueError("bad json")):
        with pytest.raises(FileNotFoundError):
            load_credentials(cfg, interactive=False)
