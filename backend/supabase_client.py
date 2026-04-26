"""
Supabase API Gateway Client for Usage Tracking

This module provides a client for routing LLM API calls through the Supabase
api-generate edge function, which handles:
- JWT authentication
- Quota reservation and enforcement
- Cost calculation and usage tracking
- Stripe metadata updates

Usage:
    client = SupabaseAPIGateway(user_token="jwt_token")
    response = await client.call_llm(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        messages=[...],
        max_tokens=4096
    )
"""

import os
import uuid
import logging
import httpx
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Supabase configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# The anon/publishable key is required as the `apikey` header when calling
# Supabase Edge Functions; missing it causes 401s even when the bearer token is
# valid. Keep it optional to avoid breaking existing environments.
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

class SupabaseAPIGateway:
    """
    Client for routing LLM API calls through Supabase edge functions.

    This ensures all API calls are tracked, quota-enforced, and billed correctly.
    """

    def __init__(self, user_token: Optional[str] = None, timeout: float = 120.0):
        """
        Initialize the Supabase API Gateway client.

        Args:
            user_token: JWT access token for the authenticated user (optional for service role)
            timeout: HTTP timeout in seconds (default: 120s for LLM calls)
        """
        if not SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required")

        self.supabase_url = SUPABASE_URL
        self.user_token = user_token
        self.timeout = timeout

    async def call_llm(
        self,
        provider: str,
        model: str,
        messages: List[Dict],
        max_tokens: int = 4096,
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Call LLM API through Supabase edge function with usage tracking.

        Args:
            provider: "anthropic" or "openai"
            model: Model identifier (e.g., "claude-sonnet-4-5-20250929")
            messages: List of messages in Anthropic format
            max_tokens: Maximum tokens for response
            system: System prompt (optional, for Anthropic)
            tools: Tool definitions (optional)

        Returns:
            Dict with:
                - result: LLM response (Anthropic format)
                - usage: Usage statistics including quota information

        Raises:
            httpx.HTTPStatusError: If API call fails
            ValueError: If quota exceeded or other business logic errors
        """
        # Generate unique request ID for idempotency
        request_id = str(uuid.uuid4())

        # Build request payload for edge function
        payload = {
            "provider": provider,
            "model": model,
            "request_id": request_id,
            "input": {
                "messages": messages,
                "max_tokens": max_tokens,
            }
        }

        # Add optional parameters
        if system:
            payload["input"]["system"] = system
        if tools:
            payload["input"]["tools"] = tools

        # Build headers with authentication
        headers = {
            "Content-Type": "application/json",
        }

        # Gate token: what the Edge Function expects in Authorization to admit the call.
        # Use the user's JWT when available; fall back to service role only when no
        # user context is present (admin/system calls).
        gate_token = self.user_token or SUPABASE_SERVICE_ROLE_KEY
        if not gate_token:
            raise ValueError("Either user_token or SUPABASE_SERVICE_ROLE_KEY must be provided")
        headers["Authorization"] = f"Bearer {gate_token}"

        # User token: used inside the function for per-user quota/usage attribution.
        if self.user_token:
            headers["x-user-token"] = self.user_token
        elif not SUPABASE_SERVICE_ROLE_KEY:
            # Should never happen because gate_token check above, but keep a breadcrumb.
            logger.warning("No user_token provided; usage will not be attributed to a user")

        # Edge Functions expect an `apikey` header; without it, Supabase returns 401
        # even if the bearer token is valid. Prefer the publishable/anon key; fall
        # back to the service role if that's all we have.
        api_key = SUPABASE_PUBLISHABLE_KEY or SUPABASE_SERVICE_ROLE_KEY
        if api_key:
            headers["apikey"] = api_key
        else:
            logger.warning("No SUPABASE_PUBLISHABLE_KEY provided; requests may be rejected with 401")

        # Call api-generate edge function
        url = f"{self.supabase_url}/functions/v1/api-generate"

        logger.info(f"Calling Supabase api-generate: provider={provider}, model={model}, request_id={request_id}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()
                logger.info(f"api-generate success: cost_cents={data.get('usage', {}).get('cost_cents')}")

                return data

            except httpx.HTTPStatusError as e:
                # Handle specific error codes
                if e.response.status_code == 402:
                    # Quota exceeded
                    error_data = e.response.json()
                    logger.error(f"Quota exceeded: {error_data}")
                    raise ValueError(f"Quota exceeded: {error_data.get('message', 'No quota remaining')}")
                elif e.response.status_code == 401:
                    # Authentication failed
                    logger.error("Authentication failed - invalid or expired token")
                    raise ValueError("Authentication failed - please sign in again")
                else:
                    # Other HTTP errors
                    logger.error(f"api-generate HTTP error {e.response.status_code}: {e.response.text}")
                    raise

            except Exception as e:
                logger.error(f"api-generate request failed: {e}")
                raise


# Convenience function for backward compatibility
async def call_llm_with_usage_tracking(
    provider: str,
    model: str,
    messages: List[Dict],
    max_tokens: int = 4096,
    system: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    user_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to call LLM with usage tracking.

    This is a simple wrapper around SupabaseAPIGateway.call_llm() for
    backward compatibility with existing code.
    """
    gateway = SupabaseAPIGateway(user_token=user_token)
    return await gateway.call_llm(
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
    )
