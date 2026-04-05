"""Configuration and settings."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".social-graph"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

KEYCHAIN_ACCOUNT = "social-graph"


# ---------------------------------------------------------------------------
# Git remote parsing
# ---------------------------------------------------------------------------

def resolve_github_remote(repo_path: str) -> tuple[str, str]:
    """Infer GitHub org and repo name from a local git clone's remote.

    Parses the 'origin' remote URL (HTTPS or SSH) and returns (org, repo).
    Raises ValueError if the remote can't be parsed.
    """
    import re

    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise ValueError(f"No 'origin' remote found in {repo_path}")
    except FileNotFoundError:
        raise ValueError("git is not installed or not on PATH")

    url = result.stdout.strip()
    # Match SSH (git@github.com:org/repo.git) or HTTPS (https://[user@]github.com/org/repo.git)
    m = re.match(r"(?:https?://(?:[^@]+@)?github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?$", url)
    if not m:
        raise ValueError(f"Could not parse GitHub org/repo from remote URL: {url}")
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Cross-platform secrets (keyring → platform CLI → env var)
# ---------------------------------------------------------------------------

def _macos_keychain_get(service: str) -> str:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _macos_keychain_set(service: str, value: str) -> bool:
    subprocess.run(
        ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service],
        capture_output=True,
        timeout=5,
    )
    try:
        result = subprocess.run(
            ["security", "add-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service, "-w", value],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _windows_credential_get(service: str) -> str:
    target = f"{KEYCHAIN_ACCOUNT}/{service}"
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"(Get-StoredCredential -Target '{target}').GetNetworkCredential().Password"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _windows_credential_set(service: str, value: str) -> bool:
    target = f"{KEYCHAIN_ACCOUNT}/{service}"
    try:
        result = subprocess.run(
            ["cmdkey", f"/generic:{target}", f"/user:{KEYCHAIN_ACCOUNT}", f"/pass:{value}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def secrets_get(service: str) -> str:
    """Read a secret. Tries: platform CLI → keyring lib → empty string."""
    if sys.platform == "darwin":
        val = _macos_keychain_get(service)
        if val:
            return val
    elif sys.platform == "win32":
        val = _windows_credential_get(service)
        if val:
            return val
    try:
        import keyring
        val = keyring.get_password(KEYCHAIN_ACCOUNT, service)
        if val:
            return val
    except Exception:
        pass
    return ""


def secrets_set(service: str, value: str) -> bool:
    """Store a secret. Tries: platform CLI → keyring lib → False."""
    if sys.platform == "darwin":
        return _macos_keychain_set(service, value)
    if sys.platform == "win32":
        return _windows_credential_set(service, value)
    try:
        import keyring
        keyring.set_password(KEYCHAIN_ACCOUNT, service, value)
        return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Token / key resolution
# ---------------------------------------------------------------------------

def _resolve_github_token() -> str:
    """Resolve GitHub token: env var → gh CLI → system keychain."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return secrets_get("GITHUB_TOKEN")


def _resolve_anthropic_key() -> str:
    """Resolve Anthropic API key: env var → system keychain."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    return secrets_get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    github_token: str = field(default_factory=_resolve_github_token)
    anthropic_api_key: str = ""
    cache_dir: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        if not self.anthropic_api_key:
            self.anthropic_api_key = _resolve_anthropic_key()

    @property
    def cache_db_path(self) -> Path:
        return self.cache_dir / "cache.db"

    def ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
