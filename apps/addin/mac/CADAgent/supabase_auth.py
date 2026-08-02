"""
Supabase authentication client for CADAgent.

This module provides a lightweight HTTP-based wrapper for Supabase auth that
doesn't require compiled dependencies (pydantic, pydantic_core).

Replaced the supabase-py SDK with direct HTTP calls via httpx to support
Fusion 360's embedded Python 3.14 environment.
"""

import json
import logging
import os
import sys
import time
import secrets
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure bundled dependencies are available
_LIB_PATH = Path(__file__).parent / "lib"
if str(_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(_LIB_PATH))

import httpx

logger = logging.getLogger(__name__)


def _fusion_probe_auth(message: str) -> None:
    """Best-effort log to Fusion Text Commands for auth debugging."""
    try:
        import adsk.core  # type: ignore

        app = adsk.core.Application.get()
        if app:
            app.log(message)
    except Exception:
        pass


def _fmt_minutes_until(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    delta = ts - time.time()
    return f"{delta/60:.1f}m" if delta >= 0 else f"{delta/60:.1f}m ago"


def _is_invalid_refresh_token(err: Exception) -> bool:
    """Detect irrecoverable refresh-token errors."""
    msg = str(err).lower()
    return "invalid refresh token" in msg or "already used" in msg or "expired refresh token" in msg


class SimpleSession:
    """Simple session object to hold auth tokens."""
    def __init__(self, access_token: str, refresh_token: str, expires_at: Optional[float] = None, user: Optional[Dict] = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.user = user


class SupabaseAuthClient:
    """
    Manages Supabase authentication and session for CADAgent.
    
    Uses direct HTTP calls instead of supabase-py SDK to avoid
    compiled dependency issues with Fusion 360's Python 3.14.
    """

    def __init__(self, supabase_url: str, supabase_key: str, supabase_client: Optional[Any] = None):
        """
        Initialize the Supabase auth client.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase publishable (anon) key
            supabase_client: Ignored (kept for API compatibility)
        """
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self._session: Optional[SimpleSession] = None

        # Session storage path (in user's home directory)
        self.session_file = Path.home() / ".cadagent" / "session.json"
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("SupabaseAuthClient initialized (HTTP-based)")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get headers for Supabase Auth API calls."""
        return {
            "apikey": self.supabase_key,
            "Content-Type": "application/json",
        }

    def _get_authenticated_headers(self, access_token: str) -> Dict[str, str]:
        """Get headers for authenticated API calls."""
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_email(email: str) -> str:
        """Normalize user-provided email for consistent Supabase lookups."""
        return (email or "").strip().lower()

    @staticmethod
    def _generate_random_password() -> str:
        """Generate a password users never need to know."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def _is_existing_user_error(error_msg: str) -> bool:
        """Heuristics for Supabase errors that mean the user already exists."""
        msg = (error_msg or "").lower()
        return any(keyword in msg for keyword in [
            "already registered",
            "already exists",
            "user already",
            "duplicate key value",
            "user exists",
            "email address is already registered",
        ])

    def _parse_session_response(self, data: Dict) -> Optional[SimpleSession]:
        """Parse session from Supabase auth response."""
        if not data:
            return None
        
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        
        if not access_token or not refresh_token:
            return None
        
        expires_at = data.get("expires_at")
        if not expires_at and data.get("expires_in"):
            expires_at = time.time() + data.get("expires_in")
        
        user = data.get("user")
        return SimpleSession(access_token, refresh_token, expires_at, user)

    def send_magic_link(self, email: str) -> Dict[str, Any]:
        """Send a magic link to the user's email for passwordless login."""
        try:
            logger.info(f"[auth] Sending magic link to {email}")

            redirect_url = os.environ.get(
                "SUPABASE_EMAIL_REDIRECT_TO",
                "https://cadagentpro.com/cadagent-auth/"
            )
            logger.info(f"[auth] Using redirect URL: {redirect_url}")

            url = f"{self.supabase_url}/auth/v1/otp"
            payload = {
                "email": email,
                "create_user": True,
                "data": {},
                "gotrue_meta_security": {},
            }
            # Add redirect for magic link
            if redirect_url:
                payload["options"] = {"email_redirect_to": redirect_url}

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code in (200, 201):
                logger.info(f"[auth] Magic link request sent")
                return {"success": True, "message": "Magic link sent! Check your email."}
            else:
                error_msg = response.json().get("error_description", response.json().get("msg", response.text))
                raise Exception(f"Failed to send magic link: {error_msg}")

        except Exception as e:
            logger.error(f"[auth] Failed to send magic link: {e}")
            raise Exception(f"Failed to send magic link: {str(e)}")

    def send_otp_code(self, email: str, *, allow_signup: bool = False) -> Dict[str, Any]:
        """Send a one-time code (OTP) to the user's email."""
        try:
            email = self._normalize_email(email)
            if not email:
                raise Exception("Email is required")

            logger.info(f"[auth] Sending OTP code to {email} (allow_signup={allow_signup})")

            url = f"{self.supabase_url}/auth/v1/otp"
            payload = {
                "email": email,
                "create_user": bool(allow_signup),
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code in (200, 201):
                logger.info(f"[auth] OTP code request sent")
                return {"success": True, "message": "Code sent! Check your email."}
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error_description", error_data.get("msg", response.text))
                raise Exception(f"Failed to send OTP code: {error_msg}")

        except Exception as e:
            logger.error(f"[auth] Failed to send OTP code: {e}")
            raise Exception(f"Failed to send OTP code: {str(e)}")

    def instant_signup(self, email: str) -> Dict[str, Any]:
        """Create a new account and immediately log in without email verification."""
        try:
            email = self._normalize_email(email)
            if not email:
                raise Exception("Email is required")

            logger.info(f"[auth] Creating instant signup for {email}")
            random_password = self._generate_random_password()

            url = f"{self.supabase_url}/auth/v1/signup"
            payload = {
                "email": email,
                "password": random_password,
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code in (200, 201):
                data = response.json()
                session = self._parse_session_response(data)
                
                if session:
                    self._session = session
                    self.save_session(session)
                    logger.info(f"[auth] Instant signup successful for {email}")
                    return {
                        "success": True,
                        "user": session.user,
                        "is_new_user": True
                    }
                else:
                    raise Exception("Signup returned no session")
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error_description", error_data.get("msg", response.text))
                raise Exception(error_msg)

        except Exception as e:
            detail = str(e)
            logger.error(f"[auth] Instant signup failed for {email}: {detail}")
            raise Exception(f"Instant signup failed: {detail}")

    def check_and_handle_signup(self, email: str) -> Dict[str, Any]:
        """Ensure account exists, then require OTP verification to log in."""
        try:
            email = self._normalize_email(email)
            if not email:
                raise Exception("Email is required")

            logger.info(f"[auth] Checking user status for {email}")
            created_new_user = False

            try:
                logger.info(f"[auth] Attempting instant signup for {email}")
                # This creates the account on first run (email captured in Supabase auth.users).
                # We intentionally do not keep the returned session; OTP is always required.
                self.instant_signup(email)
                created_new_user = True
                self.clear_session(delete_disk=True)
                logger.info(f"[auth] New account created for {email}; requiring OTP for login")
            except Exception as signup_error:
                error_msg = str(signup_error).lower()

                if self._is_existing_user_error(error_msg):
                    logger.info(f"[auth] User {email} already exists - sending OTP")
                elif "signups not allowed" in error_msg or "signup disabled" in error_msg:
                    raise Exception("Signups are disabled in Supabase")
                else:
                    raise

            # Always require OTP verification before establishing a persisted session.
            self.send_otp_code(email, allow_signup=False)
            return {
                "success": True,
                "needs_otp": True,
                "is_new_user": created_new_user,
                "message": "Account created. Enter the code sent to your email." if created_new_user else "Code sent! Check your email.",
            }

        except Exception as e:
            detail = str(e)
            logger.error(f"[auth] Check and handle signup failed for {email}: {detail}")
            raise Exception(f"Authentication failed: {detail}")

    def login_with_password(self, email: str, password: str) -> Dict[str, Any]:
        """Log in using Supabase password auth."""
        try:
            logger.info(f"[auth] Password login attempt for {email}")
            
            url = f"{self.supabase_url}/auth/v1/token?grant_type=password"
            payload = {
                "email": email,
                "password": password,
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code == 200:
                data = response.json()
                session = self._parse_session_response(data)
                
                if session:
                    self._session = session
                    self.save_session(session)
                    logger.info(f"[auth] Password login succeeded for {email}")
                    return {
                        "success": True,
                        "user": session.user
                    }
                else:
                    raise Exception("Login returned no session")
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error_description", error_data.get("msg", "Login failed"))
                raise Exception(error_msg)

        except Exception as e:
            detail = str(e)
            logger.error(f"[auth] Password login failed for {email}: {detail}")
            raise Exception(f"Password login failed: {detail}")

    def verify_otp_code(self, email: str, code: str) -> Dict[str, Any]:
        """Verify the OTP code and establish a Supabase session."""
        try:
            email = self._normalize_email(email)
            code = (code or "").strip()
            if not email:
                raise Exception("Email is required")

            logger.info(f"[auth] Verifying OTP code for {email}")

            url = f"{self.supabase_url}/auth/v1/verify"
            payload = {
                "email": email,
                "token": code,
                "type": "email",
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code == 200:
                data = response.json()
                session = self._parse_session_response(data)
                
                if session:
                    self._session = session
                    self.save_session(session)
                    logger.info(f"[auth] OTP verified and session saved")
                    return {
                        "success": True,
                        "user": session.user
                    }
                else:
                    raise Exception("OTP verification returned no session")
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error_description", error_data.get("msg", "Verification failed"))
                raise Exception(error_msg)

        except Exception as e:
            detail = str(e)
            logger.error(f"[auth] OTP verification failed for {email}: {detail}")
            raise Exception(f"OTP verification failed: {detail}")

    def set_session_from_callback(self, access_token: str, refresh_token: str) -> Dict[str, Any]:
        """Set the session from auth callback tokens (magic link clicked)."""
        try:
            logger.info("[auth] Setting session from callback tokens")

            # Validate by getting user info
            url = f"{self.supabase_url}/auth/v1/user"
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=self._get_authenticated_headers(access_token))

            if response.status_code == 200:
                user = response.json()
                session = SimpleSession(access_token, refresh_token, user=user)
                self._session = session
                self.save_session(session)
                logger.info("[auth] Session set and saved")
                return {
                    "success": True,
                    "user": user
                }
            else:
                raise Exception("Failed to validate session tokens")

        except Exception as e:
            logger.error(f"Failed to set session: {e}")
            raise Exception(f"Failed to set session: {str(e)}")

    def get_session(self) -> Optional[Dict[str, Any]]:
        """Get the current session."""
        if self._session:
            return {
                "access_token": self._session.access_token,
                "refresh_token": self._session.refresh_token,
                "expires_at": self._session.expires_at,
                "user": self._session.user
            }
        return None

    def _refresh_session(self, refresh_token: str) -> Optional[SimpleSession]:
        """Refresh the session using a refresh token."""
        try:
            url = f"{self.supabase_url}/auth/v1/token?grant_type=refresh_token"
            payload = {"refresh_token": refresh_token}

            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload, headers=self._get_auth_headers())

            if response.status_code == 200:
                data = response.json()
                return self._parse_session_response(data)
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error_description", error_data.get("msg", "Refresh failed"))
                logger.error(f"[auth] Token refresh failed: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"[auth] Token refresh error: {e}")
            return None

    def get_valid_access_token(self, min_buffer_seconds: int = 120) -> Optional[str]:
        """Return a non-expired access token, proactively refreshing if needed."""
        _fusion_probe_auth("[AUTH_TOKEN] get_valid_access_token start")

        if not self._session:
            if self.session_file.exists():
                logger.info("[auth] No active session; attempting restore from disk")
                if not self.restore_session():
                    _fusion_probe_auth("[AUTH_TOKEN] no session; restore failed")
                    return None
            else:
                _fusion_probe_auth("[AUTH_TOKEN] no session file; user not signed in")
                return None

        session = self._session
        exp_ts = session.expires_at
        _fusion_probe_auth(f"[AUTH_TOKEN] session exp={exp_ts} ({_fmt_minutes_until(exp_ts)})")

        now = time.time()
        needs_refresh = exp_ts is not None and exp_ts <= now + min_buffer_seconds

        if needs_refresh:
            logger.info("[auth] Access token expires soon; refreshing")
            _fusion_probe_auth("[AUTH_TOKEN] refresh start")
            
            new_session = self._refresh_session(session.refresh_token)
            if new_session:
                self._session = new_session
                self.save_session(new_session)
                _fusion_probe_auth(f"[AUTH_TOKEN] refresh ok exp={new_session.expires_at}")
                return new_session.access_token
            else:
                logger.error("[auth] Token refresh failed")
                self.clear_session(delete_disk=False)
                _fusion_probe_auth("[AUTH_TOKEN] refresh failed; session cleared")
                return None
        else:
            _fusion_probe_auth("[AUTH_TOKEN] token valid; no refresh needed")

        return session.access_token

    def restore_session(self) -> bool:
        """Restore session from disk if it exists."""
        try:
            if not self.session_file.exists():
                logger.info("No saved session found")
                return False

            logger.info("Restoring session from disk")
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

            refresh_token = session_data.get("refresh_token")
            if not refresh_token:
                logger.warning("No refresh token in session file")
                return False

            # Try to refresh the session
            new_session = self._refresh_session(refresh_token)
            if new_session:
                self._session = new_session
                self.save_session(new_session)
                logger.info("[auth] Session restored via token refresh")
                return True

            # Fallback: try using stored access token directly
            access_token = session_data.get("access_token")
            if access_token:
                self._session = SimpleSession(
                    access_token,
                    refresh_token,
                    session_data.get("expires_at")
                )
                logger.info("[auth] Session restored from stored tokens")
                return True

            return False

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse session.json: {e}")
            self.clear_session(delete_disk=True)
            return False
        except Exception as e:
            logger.error(f"Failed to restore session: {e}")
            return False

    def save_session(self, session: SimpleSession) -> None:
        """Save session to disk for persistence across restarts."""
        try:
            session_data = {
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "expires_at": session.expires_at,
            }

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)

            if os.name != 'nt':
                os.chmod(self.session_file, 0o600)

            logger.info("Session saved to disk")
            _fusion_probe_auth(f"[AUTH_TOKEN] save_session: exp={session.expires_at}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def clear_session(self, delete_disk: bool = True) -> None:
        """Clear the current session."""
        try:
            logger.info(f"Clearing session (delete_disk={delete_disk})")
            self._session = None

            if delete_disk and self.session_file.exists():
                self.session_file.unlink()
                logger.info("Session file deleted")

            logger.info("Session cleared")

        except Exception as e:
            logger.error(f"Failed to clear session: {e}")

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Fetch the user's profile from the /me endpoint."""
        try:
            _fusion_probe_auth("[AUTH_PROFILE] get_profile start")
            
            if not self._session:
                if self.session_file.exists():
                    if not self.restore_session():
                        return None
                else:
                    return None

            access_token = self._session.access_token

            url = f"{self.supabase_url}/functions/v1/me"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                logger.info("[auth] Profile fetched successfully")
                return data.get("user")
            elif response.status_code == 401:
                # Try refreshing token
                logger.warning("[auth] Profile fetch 401 - attempting refresh")
                new_session = self._refresh_session(self._session.refresh_token)
                if new_session:
                    self._session = new_session
                    self.save_session(new_session)
                    headers["Authorization"] = f"Bearer {new_session.access_token}"
                    retry_response = client.get(url, headers=headers)
                    if retry_response.status_code == 200:
                        return retry_response.json().get("user")
                return None
            else:
                logger.warning(f"[auth] Profile fetch failed: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"[auth] Profile fetch error: {e}")
            return None

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        if self._session:
            return True
        
        if self.session_file.exists():
            if self.restore_session():
                return self._session is not None
        
        return False

    def _serialize_user(self, user: Any) -> Optional[Dict[str, Any]]:
        """Return a JSON-serializable user dict."""
        if user is None:
            return None
        if isinstance(user, dict):
            return user
        try:
            return dict(user)
        except Exception:
            return None

    def get_user_email(self) -> Optional[str]:
        """Get the current user's email."""
        if not self._session and self.session_file.exists():
            self.restore_session()
        
        if self._session and self._session.user:
            return self._session.user.get("email")
        return None
