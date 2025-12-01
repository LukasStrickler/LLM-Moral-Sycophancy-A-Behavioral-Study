"""Budget tracking and enforcement."""

from __future__ import annotations


class BudgetTracker:
    """Track spending and enforce budget limits."""

    def __init__(self, limit: float, current: float = 0.0) -> None:
        """Initialize budget tracker.

        Args:
            limit: Maximum spending in USD
            current: Current spending (default: 0.0)
        """
        self._limit = limit
        self._current = current

    @property
    def limit(self) -> float:
        """Maximum spending limit."""
        return self._limit

    @property
    def current(self) -> float:
        """Current spending."""
        return self._current

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return max(0.0, self._limit - self._current)

    @property
    def exceeded(self) -> bool:
        """Check if budget has been exceeded."""
        return self._current >= self._limit

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford a request.

        Args:
            estimated_cost: Estimated cost in USD

        Returns:
            True if we can afford the request, False otherwise
        """
        return (self._current + estimated_cost) <= self._limit

    def add_cost(self, cost: float) -> None:
        """Add cost to current spending with strict budget enforcement.

        Args:
            cost: Cost in USD to add (must be non-negative)

        Raises:
            ValueError: If cost is negative or would exceed budget limit

        Note:
            This method enforces strict budget limits - it will raise an error
            if adding the cost would exceed the limit, preventing any overspending.
        """
        if cost < 0:
            raise ValueError(
                f"Cost must be non-negative, got ${cost:.4f}. "
                f"Negative costs are not allowed."
            )
        new_total = self._current + cost
        if new_total > self._limit:
            raise ValueError(
                f"Budget limit exceeded: Adding ${cost:.4f} would exceed limit "
                f"of ${self._limit:.2f}. Current spending: ${self._current:.4f}, "
                f"would be: ${new_total:.4f}. Remaining budget: ${self.remaining:.2f}"
            )
        self._current = new_total

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Budget: ${self._current:.4f} / ${self._limit:.2f} "
            f"(remaining: ${self.remaining:.2f})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"BudgetTracker(limit=${self._limit:.2f}, "
            f"current=${self._current:.4f}, "
            f"remaining=${self.remaining:.2f})"
        )

