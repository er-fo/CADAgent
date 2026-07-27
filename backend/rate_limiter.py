"""
Rate Limiter for Fusion 360 LLM CAD Agent

Implements per-user rate limiting using token bucket algorithm.
Supports both request frequency and token usage tracking.

Generous defaults designed to allow 6-7 concurrent sessions per user.
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Rate limit configuration (environment variables with generous defaults)
REQUESTS_PER_MINUTE = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "200"))
BURST_CAPACITY = int(os.environ.get("RATE_LIMIT_BURST_CAPACITY", "50"))
TOKENS_PER_DAY = int(os.environ.get("RATE_LIMIT_TOKENS_PER_DAY", "500000"))
ENABLE_RATE_LIMITING = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    Allows bursty traffic up to burst_capacity while enforcing average rate.
    """

    def __init__(self, rate: float, burst_capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            burst_capacity: Maximum tokens that can accumulate
        """
        self.rate = rate  # tokens/second
        self.burst_capacity = burst_capacity
        self.tokens = float(burst_capacity)
        self.last_update = time.time()

    def add_tokens(self) -> None:
        """Replenish tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if limit exceeded
        """
        self.add_tokens()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def available_tokens(self) -> float:
        """Get current available tokens without consuming."""
        self.add_tokens()
        return self.tokens


class DailyQuota:
    """Track daily token usage for a user."""

    def __init__(self, limit: int):
        """
        Initialize daily quota.

        Args:
            limit: Maximum tokens per day
        """
        self.limit = limit
        self.used = 0
        self.reset_date = datetime.now().date()

    def try_consume(self, tokens: int) -> Tuple[bool, int]:
        """
        Try to consume tokens from daily quota.

        Args:
            tokens: Number of tokens to consume

        Returns:
            Tuple of (success, remaining_quota)
        """
        # Reset quota if new day
        now = datetime.now().date()
        if now > self.reset_date:
            self.used = 0
            self.reset_date = now

        if self.used + tokens <= self.limit:
            self.used += tokens
            return True, self.limit - self.used

        return False, 0

    def get_remaining(self) -> int:
        """Get remaining tokens for today."""
        # Reset quota if new day
        now = datetime.now().date()
        if now > self.reset_date:
            self.used = 0
            self.reset_date = now

        return self.limit - self.used


class RateLimiter:
    """
    Per-user rate limiter using token bucket + daily quota.

    Tracks both request frequency and token usage per user.
    Designed to support 6-7 concurrent sessions per user.
    """

    def __init__(self):
        """Initialize the rate limiter."""
        self.user_buckets: Dict[str, TokenBucket] = {}
        self.user_quotas: Dict[str, DailyQuota] = {}
        self.lock = asyncio.Lock()

        # Calculate rate from requests per minute
        # 200 req/min = 3.33 req/sec
        rate_per_second = REQUESTS_PER_MINUTE / 60.0

        logger.info(
            f"Rate limiter initialized: "
            f"{REQUESTS_PER_MINUTE} req/min ({rate_per_second:.2f} req/sec), "
            f"burst capacity: {BURST_CAPACITY}, "
            f"daily quota: {TOKENS_PER_DAY} tokens, "
            f"enabled: {ENABLE_RATE_LIMITING}"
        )

    def get_or_create_bucket(self, user_id: str) -> TokenBucket:
        """Get or create token bucket for user."""
        if user_id not in self.user_buckets:
            rate_per_second = REQUESTS_PER_MINUTE / 60.0
            self.user_buckets[user_id] = TokenBucket(rate_per_second, BURST_CAPACITY)
            logger.debug(f"Created token bucket for user {user_id}")
        return self.user_buckets[user_id]

    def get_or_create_quota(self, user_id: str) -> DailyQuota:
        """Get or create daily quota for user."""
        if user_id not in self.user_quotas:
            self.user_quotas[user_id] = DailyQuota(TOKENS_PER_DAY)
            logger.debug(f"Created daily quota for user {user_id}")
        return self.user_quotas[user_id]

    async def check_request_limit(
        self,
        user_id: str,
        tokens: int = 1
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if user is within request rate limit.

        Args:
            user_id: User identifier from JWT
            tokens: Number of tokens to consume (default 1 for request)

        Returns:
            Tuple of (allowed, info_dict) where info_dict contains:
            - allowed: bool - Whether request is allowed
            - available: float - Available tokens in bucket
            - refill_rate: float - Tokens per second
            - burst_capacity: int - Max burst tokens
        """
        if not ENABLE_RATE_LIMITING:
            return True, {
                "allowed": True,
                "rate_limiting_enabled": False,
            }

        async with self.lock:
            bucket = self.get_or_create_bucket(user_id)
            allowed = bucket.try_consume(tokens)
            available = bucket.available_tokens()

            status = {
                "allowed": allowed,
                "available": round(available, 2),
                "refill_rate": round(REQUESTS_PER_MINUTE / 60.0, 2),
                "burst_capacity": BURST_CAPACITY,
            }

            if not allowed:
                logger.warning(
                    f"Request rate limit exceeded for user {user_id}: "
                    f"available={available:.2f}/{BURST_CAPACITY}"
                )

            return allowed, status

    async def check_token_quota(
        self,
        user_id: str,
        tokens_used: int
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if user is within daily token quota.

        Args:
            user_id: User identifier from JWT
            tokens_used: Number of tokens used in this request

        Returns:
            Tuple of (allowed, info_dict) where info_dict contains:
            - allowed: bool - Whether request is allowed
            - tokens_used: int - Tokens used in this request
            - remaining: int - Remaining tokens for today
            - limit: int - Daily token limit
            - reset_time: str - When quota resets (ISO format)
        """
        if not ENABLE_RATE_LIMITING:
            return True, {
                "allowed": True,
                "rate_limiting_enabled": False,
            }

        async with self.lock:
            quota = self.get_or_create_quota(user_id)
            allowed, remaining = quota.try_consume(tokens_used)

            reset_time = (datetime.now() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()

            status = {
                "allowed": allowed,
                "tokens_used": tokens_used,
                "remaining": remaining,
                "limit": TOKENS_PER_DAY,
                "reset_time": reset_time,
            }

            if not allowed:
                logger.warning(
                    f"Daily token quota exceeded for user {user_id}: "
                    f"used={quota.used}, limit={TOKENS_PER_DAY}"
                )

            return allowed, status

    async def get_user_limits(self, user_id: str) -> Dict[str, any]:
        """
        Get current limit status for a user.

        Returns dict with:
        - request_rate: Current available request tokens
        - daily_tokens: Current remaining daily tokens
        """
        if not ENABLE_RATE_LIMITING:
            return {
                "request_rate_enabled": False,
                "daily_tokens_enabled": False,
            }

        async with self.lock:
            bucket = self.get_or_create_bucket(user_id)
            quota = self.get_or_create_quota(user_id)

            return {
                "request_rate": {
                    "available": round(bucket.available_tokens(), 2),
                    "burst_capacity": BURST_CAPACITY,
                    "refill_rate": round(REQUESTS_PER_MINUTE / 60.0, 2),
                },
                "daily_tokens": {
                    "remaining": quota.get_remaining(),
                    "limit": TOKENS_PER_DAY,
                    "reset_date": quota.reset_date.isoformat(),
                }
            }

    def cleanup_user(self, user_id: str) -> None:
        """Clean up rate limiter state for a user (optional, for memory efficiency)."""
        self.user_buckets.pop(user_id, None)
        self.user_quotas.pop(user_id, None)
        logger.debug(f"Cleaned up rate limiter for user {user_id}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
