"""CSV output writer for paid runner results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TypedDict


class CSVRow(TypedDict, total=False):
    """Structure for a CSV result row."""

    prompt_identifier: str
    prompt_body: str
    model_id: str
    response_text: str
    input_tokens: int | str
    output_tokens: int | str
    total_tokens: int | str
    cost_usd: float | str
    latency_ms: int | str
    timestamp: str
    error: str
    retry_count: int


class CSVWriter:
    """Write paid runner results to CSV format."""

    COLUMNS = [
        "prompt_identifier",
        "prompt_body",
        "model_id",
        "response_text",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "latency_ms",
        "timestamp",
        "error",
        "retry_count",
    ]

    def __init__(self, output_path: Path, append: bool = False) -> None:
        """Initialize CSV writer.

        Args:
            output_path: Path to CSV output file
            append: If True, append to existing file; if False, create new file
        """
        self.output_path = output_path
        self.append = append
        self._file_handle: Any = None
        self._writer: csv.DictWriter | None = None
        self._header_written = False
        # Track what we've written in this session to avoid duplicates
        self._written_keys: set[tuple[str, str]] = set()  # (prompt_id, model_id)

    def __enter__(self) -> CSVWriter:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open CSV file for writing."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and has content
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0

        if self.append and file_exists:
            # Append mode: file exists, don't write header
            self._file_handle = self.output_path.open("a", encoding="utf-8", newline="")
            self._header_written = True
        else:
            # New file or overwrite: write header
            self._file_handle = self.output_path.open("w", encoding="utf-8", newline="")
            self._header_written = False

        self._writer = csv.DictWriter(self._file_handle, fieldnames=self.COLUMNS)
        if not self._header_written:
            self._writer.writeheader()
            self._header_written = True

    def close(self) -> None:
        """Close CSV file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._writer = None

    def has_result(self, prompt_identifier: str, model_id: str) -> bool:
        """Check if a successful result already exists for this prompt/model combination.
        
        Checks both the in-memory cache and the CSV file.

        Args:
            prompt_identifier: Prompt identifier
            model_id: Model ID

        Returns:
            True if a successful result exists, False otherwise
        """
        key = (prompt_identifier, model_id)
        
        # Check in-memory cache first (fast)
        if key in self._written_keys:
            return True
        
        # Check CSV file for existing successful result
        if self.output_path.exists():
            try:
                with self.output_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if (row.get("prompt_identifier", "").strip() == prompt_identifier and
                            row.get("model_id", "").strip() == model_id):
                            # Only successful results are written, so if we find it, it exists
                            return True
            except Exception:
                # If CSV read fails, assume no result exists
                pass
        
        return False

    def write_result(
        self,
        prompt_identifier: str,
        prompt_body: str,
        model_id: str,
        response_text: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        timestamp: str | None = None,
        error: str | None = None,
        retry_count: int = 0,
    ) -> None:
        """Write a single successful result row.
        
        Only writes successful results (no errors). Skips if a result already exists
        for this (prompt_id, model_id) combination.

        Args:
            prompt_identifier: Prompt identifier from seed file
            prompt_body: Prompt body text
            model_id: Model ID used
            response_text: Model response text
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            total_tokens: Total tokens used
            cost_usd: Cost in USD
            latency_ms: Request latency in milliseconds
            timestamp: ISO timestamp
            error: Error message (should be None for successful results)
            retry_count: Number of retries needed
        """
        if self._writer is None:
            raise RuntimeError(
                "CSV writer not opened. Call open() first or use context manager. "
                f"Output path: {self.output_path}"
            )

        # Validate required fields
        if not prompt_identifier:
            raise ValueError("prompt_identifier is required for CSV row")
        if not prompt_body:
            raise ValueError("prompt_body is required for CSV row")
        if not model_id:
            raise ValueError("model_id is required for CSV row")

        key = (prompt_identifier, model_id)
        
        # Skip if result already exists (prevent duplicates from retries)
        if self.has_result(prompt_identifier, model_id):
            return
        
        row: CSVRow = {
            "prompt_identifier": prompt_identifier,
            "prompt_body": prompt_body,
            "model_id": model_id,
            "response_text": response_text or "",
            "input_tokens": input_tokens or "",
            "output_tokens": output_tokens or "",
            "total_tokens": total_tokens or "",
            "cost_usd": cost_usd or "",
            "latency_ms": latency_ms or "",
            "timestamp": timestamp or "",
            "error": error or "",  # Should be empty for successful results
            "retry_count": retry_count,
        }

        try:
            self._writer.writerow(row)
            self._file_handle.flush()  # Ensure data is written immediately
            # Track what we've written
            self._written_keys.add(key)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write CSV row for prompt '{prompt_identifier}', "
                f"model '{model_id}': {e}"
            ) from e

