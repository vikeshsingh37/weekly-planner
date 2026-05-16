"""
User authentication: password hashing with bcrypt, JWT tokens,
and a simple JSON user store (data/users.json).

User IDs are derived deterministically from email addresses so that
each user's session files live in a stable, filesystem-safe directory.
"""
import json
import os
import re
import time
from pathlib import Path

import bcrypt
import jwt

_SECRET = os.getenv("SECRET_KEY", "weekly-planner-dev-secret-change-me")
_ALGO = "HS256"
_EXPIRE_SECONDS = 7 * 24 * 3600  # 7 days
_USERS_FILE = Path("data/users.json")


def _load() -> dict:
    if not _USERS_FILE.exists():
        return {}
    return json.loads(_USERS_FILE.read_text())


def _save(users: dict) -> None:
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(json.dumps(users, indent=2))


def email_to_user_id(email: str) -> str:
    """Derive a filesystem-safe user ID from an email address.

    Example: vikeshsingh37@gmail.com → vikeshsingh37_gmail.com
    """
    return re.sub(r"[^a-zA-Z0-9._-]", "_", email.lower().strip())


def register(email: str, password: str) -> bool:
    """Hash password and persist user. Returns False if email already taken."""
    key = email.lower().strip()
    users = _load()
    if key in users:
        return False
    users[key] = {
        "hashed": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        "user_id": email_to_user_id(key),
    }
    _save(users)
    return True


def authenticate(email: str, password: str) -> str | None:
    """Returns user_id if credentials are valid, None otherwise."""
    key = email.lower().strip()
    u = _load().get(key)
    if u and bcrypt.checkpw(password.encode(), u["hashed"].encode()):
        return u["user_id"]
    return None


def create_token(email: str) -> str:
    return jwt.encode(
        {"sub": email.lower().strip(), "exp": int(time.time()) + _EXPIRE_SECONDS},
        _SECRET,
        algorithm=_ALGO,
    )


def verify_token(token: str) -> str | None:
    """Returns user_id if the token is valid and the user exists, None otherwise."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGO])
        u = _load().get(payload.get("sub", ""))
        return u["user_id"] if u else None
    except Exception:
        return None