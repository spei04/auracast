"""
First-time Google OAuth setup.

Runs on your *Mac* (or any machine with a browser). Opens the consent screen,
asks you to grant the requested scopes, writes the resulting token to
`AURACAST_TOKEN_PATH` (or `~/.config/auracast/token.json`). After this you
copy that token file to the cluster.

Usage:
    # 1. Put your OAuth client_secrets.json in ~/.config/auracast/
    #    (downloaded from Google Cloud Console -> APIs & Services ->
    #     Credentials -> OAuth 2.0 Client IDs -> Desktop app)
    # 2. Run this:
    python -m auracast.scripts.auth_setup
    # 3. Browser opens, you grant scopes, token.json is written.
    # 4. Copy the token to the cluster:
    scp ~/.config/auracast/token.json \\
        beery:/data/vision/beery/scratch/serena/.auracast/secrets/token.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from auracast.auth.google_oauth import (
    DEFAULT_SCOPES,
    OAuthConfig,
    default_client_secrets_path,
    default_token_path,
    load_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("auth_setup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-secrets", type=Path, default=default_client_secrets_path(),
        help="Path to the OAuth client_secrets.json downloaded from GCP.",
    )
    parser.add_argument(
        "--token-path", type=Path, default=default_token_path(),
        help="Where to write the OAuth token JSON after consent.",
    )
    parser.add_argument(
        "--scopes", nargs="+", default=list(DEFAULT_SCOPES),
        help="Override the OAuth scopes (space-separated).",
    )
    args = parser.parse_args()

    cfg = OAuthConfig(
        client_secrets_path=args.client_secrets,
        token_cache_path=args.token_path,
        scopes=tuple(args.scopes),
    )
    logger.info("client_secrets: %s", cfg.client_secrets_path)
    logger.info("token will be written to: %s", cfg.token_cache_path)
    logger.info("scopes: %s", cfg.scopes)

    if not cfg.client_secrets_path.exists():
        raise SystemExit(
            f"\nNo client_secrets.json at {cfg.client_secrets_path}.\n"
            f"Download from https://console.cloud.google.com/apis/credentials\n"
            f"(OAuth 2.0 Client IDs -> 'Desktop app') and place it there.\n"
            f"Or pass --client-secrets PATH."
        )

    # interactive=True forces the browser flow even when stdin isn't a TTY
    # (e.g. when running under VSCode terminal which sometimes doesn't isatty).
    load_credentials(cfg, interactive=True)
    logger.info("done. token written to %s", cfg.token_cache_path)
    print(f"\nTo use on the cluster:\n"
          f"  scp {cfg.token_cache_path} <user>@cluster:<scratch>/.auracast/secrets/token.json\n"
          f"  ssh into cluster; set\n"
          f"    export AURACAST_TOKEN_PATH=<scratch>/.auracast/secrets/token.json\n"
          f"  in your .bashrc.\n")


if __name__ == "__main__":
    main()
