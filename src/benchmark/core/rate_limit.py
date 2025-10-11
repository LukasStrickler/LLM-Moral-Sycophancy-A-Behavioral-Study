from __future__ import annotations

import asyncio
import random
import time


class TokenBucket:
    """Simple async token bucket limiter.

    - capacity: max tokens in the bucket (burst)
    - rate: tokens refilled per second (RPS)
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = float(rate)
        if self.rate <= 0:
            raise ValueError("rate must be > 0")
        cap_int = int(capacity)
        if cap_int <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = cap_int
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                # refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # need to wait for next token
                deficit = 1.0 - self._tokens
                wait_s = deficit / self.rate
                await asyncio.sleep(wait_s)


def compute_backoff(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    """Exponential backoff with jitter."""
    if attempt < 0:
        raise ValueError("attempt must be non-negative")

    exp_raw = base * (2**attempt)
    jitter = random.uniform(0, base)
    return min(cap, exp_raw + jitter)
