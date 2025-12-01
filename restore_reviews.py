#!/usr/bin/env python3
"""Restore reviews from exported JSONL files by matching on identifier instead of ID.

Optimized version using batch operations for much faster performance.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset  # noqa: E402
from src.labeling_app.db import client_scope, queries as db_queries  # noqa: E402
from src.labeling_app.settings import get_settings  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn  # noqa: E402

console = Console()

REVIEWS_DIR = PROJECT_ROOT / "data" / "humanLabel" / "reviews"

# Batch size for inserts (libsql may have limits, so we batch in chunks)
# Reduced from 100 to 50 for better reliability with libsql
BATCH_SIZE = 50


def load_reviews_from_export(file_path: Path) -> list[dict]:
    """Load review records from exported JSONL file.
    
    Handles JSON parsing errors and file encoding issues gracefully.
    """
    reviews = []
    line_num = 0
    errors = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    reviews.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    errors += 1
                    console.print(f"[yellow]Warning: Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise
    except UnicodeDecodeError as e:
        console.print(f"[red]Error: File encoding issue (not UTF-8) on line {line_num}: {e}[/red]")
        raise
    
    if errors > 0:
        console.print(f"[yellow]Skipped {errors} lines with JSON parsing errors[/yellow]")
    
    return reviews


def build_response_id_map(client, dataset: Dataset, identifier_run_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Build a mapping from (identifier, run_id) to response_id using batch queries.
    
    Chunks the lookups to avoid query size limits.
    Handles database errors gracefully.
    
    Returns: dict mapping (identifier, run_id) -> response_id
    """
    if not identifier_run_pairs:
        return {}
    
    mapping: dict[tuple[str, str], int] = {}
    
    # Chunk lookups to avoid query size limits (libsql has stricter limits, use smaller chunks)
    # Reduced from 100 to 20 to avoid query size limits with OR conditions
    CHUNK_SIZE = 20
    total_chunks = (len(identifier_run_pairs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(0, len(identifier_run_pairs), CHUNK_SIZE):
        chunk = identifier_run_pairs[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1
        if not chunk:  # Handle empty chunks
            continue
        
        # Use OR conditions for maximum compatibility (works on all SQLite versions)
        conditions = []
        params: list[object] = [dataset.value]
        for identifier, run_id in chunk:
            # Validate inputs are strings (not None)
            if not isinstance(identifier, str) or not isinstance(run_id, str):
                continue
            conditions.append("(identifier = ? AND run_id = ?)")
            params.extend([identifier, run_id])
        
        if not conditions:  # All items in chunk were invalid
            continue
        
        try:
            query = f"""
                SELECT id, identifier, run_id
                FROM llm_responses
                WHERE dataset = ? AND ({' OR '.join(conditions)})
            """
            
            result = client.execute(query, params)
            for row in result.rows:
                if len(row) >= 3:
                    response_id, identifier, run_id = row[0], row[1], row[2]
                    mapping[(identifier, run_id)] = int(response_id)
            
            # Show progress for large batches
            if total_chunks > 5 and chunk_num % 10 == 0:
                console.print(f"[dim]Processed {chunk_num}/{total_chunks} chunks...[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Error building response ID map for chunk {chunk_num}/{total_chunks}: {e}[/yellow]")
            console.print(f"[dim]Trying individual lookups for this chunk...[/dim]")
            # Fallback: try individual lookups for this chunk
            for identifier, run_id in chunk:
                try:
                    result = client.execute(
                        """
                        SELECT id FROM llm_responses
                        WHERE dataset = ? AND identifier = ? AND run_id = ?
                        """,
                        [dataset.value, identifier, run_id],
                    )
                    if result.rows:
                        mapping[(identifier, run_id)] = int(result.rows[0][0])
                except Exception:
                    continue
    
    return mapping


def build_existing_reviews_set(client, response_ids: set[int]) -> set[tuple[int, str]]:
    """Build a set of (response_id, reviewer_code) pairs that already exist.
    
    Chunks the lookups to avoid query size limits.
    Handles database errors gracefully.
    
    Returns: set of (response_id, reviewer_code) tuples
    """
    if not response_ids:
        return set()
    
    existing: set[tuple[int, str]] = set()
    response_ids_list = [rid for rid in response_ids if isinstance(rid, int) and rid > 0]
    
    if not response_ids_list:
        return set()
    
    # Chunk lookups to avoid query size limits (500 IDs per query)
    CHUNK_SIZE = 500
    for i in range(0, len(response_ids_list), CHUNK_SIZE):
        chunk = response_ids_list[i:i + CHUNK_SIZE]
        if not chunk:  # Handle empty chunks
            continue
        
        try:
            placeholders = ",".join("?" for _ in chunk)
            params = chunk
            
            query = f"""
                SELECT llm_response_id, reviewer_code
                FROM reviews
                WHERE llm_response_id IN ({placeholders})
            """
            
            result = client.execute(query, params)
            for row in result.rows:
                if len(row) >= 2:
                    existing.add((int(row[0]), str(row[1])))
        except Exception as e:
            console.print(f"[red]Error checking existing reviews for chunk {i//CHUNK_SIZE + 1}: {e}[/red]")
            # Continue with next chunk rather than failing completely
            continue
    
    return existing


def restore_reviews_for_dataset(dataset: Dataset) -> tuple[int, int, int]:
    """Restore reviews for a dataset from exported file using optimized batch operations.
    
    Returns: (restored_count, skipped_count, not_found_count)
    """
    file_path = REVIEWS_DIR / f"{dataset.value}_reviews.jsonl"
    
    if not file_path.exists():
        console.print(f"[yellow]No export file found for {dataset.value}: {file_path}[/yellow]")
        return (0, 0, 0)
    
    console.print(f"[cyan]Loading reviews from:[/cyan] {file_path}")
    try:
        reviews = load_reviews_from_export(file_path)
    except Exception as e:
        console.print(f"[red]Error loading reviews from file: {e}[/red]")
        return (0, 0, 0)
    
    if not reviews:
        console.print("[yellow]No reviews found in file[/yellow]")
        return (0, 0, 0)
    
    console.print(f"[cyan]Found {len(reviews)} review records[/cyan]\n")
    
    # Step 1: Filter and validate reviews
    console.print("[cyan]Validating reviews...[/cyan]")
    valid_reviews = []
    skipped_invalid = 0
    
    for review_idx, review in enumerate(reviews, start=1):
        if not isinstance(review, dict):
            skipped_invalid += 1
            console.print(f"[yellow]Warning: Review {review_idx} is not a dictionary, skipping[/yellow]")
            continue
        
        identifier = review.get("identifier")
        run_id = review.get("run_id")
        reviewer_code = review.get("reviewer_code")
        score = review.get("score")
        notes = review.get("notes")
        
        # Validate required fields
        if not identifier or not isinstance(identifier, str):
            skipped_invalid += 1
            continue
        if not run_id or not isinstance(run_id, str):
            skipped_invalid += 1
            continue
        if not reviewer_code or not isinstance(reviewer_code, str):
            skipped_invalid += 1
            continue
        if score is None:
            skipped_invalid += 1
            continue
        
        # Validate and convert score
        try:
            score_float = float(score)
        except (ValueError, TypeError):
            skipped_invalid += 1
            console.print(f"[yellow]Warning: Invalid score '{score}' in review {review_idx}, skipping[/yellow]")
            continue
        
        # Validate score range: -1.0 to 1.0 (per schema CHECK constraint)
        if not (-1.0 <= score_float <= 1.0):
            skipped_invalid += 1
            console.print(f"[yellow]Warning: Score {score_float} out of range [-1.0, 1.0] in review {review_idx}, skipping[/yellow]")
            continue
        
        # Validate notes is string or None
        if notes is not None and not isinstance(notes, str):
            # Try to convert to string if it's another type
            notes = str(notes)
        
        valid_reviews.append({
            "identifier": identifier,
            "run_id": run_id,
            "reviewer_code": reviewer_code,
            "score": score_float,
            "notes": notes,
        })
    
    if skipped_invalid > 0:
        console.print(f"[yellow]Skipped {skipped_invalid} invalid reviews[/yellow]")
    
    if not valid_reviews:
        console.print("[yellow]No valid reviews to restore[/yellow]")
        return (0, skipped_invalid, 0)
    
    console.print(f"[green]Processing {len(valid_reviews)} valid reviews[/green]\n")
    
    settings = get_settings()
    restored = 0
    skipped = skipped_invalid
    not_found = 0
    
    try:
        with client_scope(settings) as client:
            # Step 2: Build response_id mapping in one batch query
            console.print("[cyan]Building response ID mapping...[/cyan]")
            identifier_run_pairs = [(r["identifier"], r["run_id"]) for r in valid_reviews]
            # Remove duplicates for the query
            unique_pairs = list(set(identifier_run_pairs))
            response_id_map = build_response_id_map(client, dataset, unique_pairs)
            console.print(f"[green]Found {len(response_id_map)} matching responses[/green]\n")
            
            # Step 3: Map reviews to response_ids and filter out not found
            reviews_with_ids = []
            for review in valid_reviews:
                key = (review["identifier"], review["run_id"])
                response_id = response_id_map.get(key)
                if response_id is None:
                    not_found += 1
                    continue
                review["response_id"] = response_id
                reviews_with_ids.append(review)
            
            if not reviews_with_ids:
                console.print("[yellow]No reviews with matching responses found[/yellow]")
                return (0, skipped, not_found)
            
            # Step 4: Check which reviews already exist in one batch query
            console.print("[cyan]Checking for existing reviews...[/cyan]")
            response_ids = {r["response_id"] for r in reviews_with_ids}
            existing_reviews = build_existing_reviews_set(client, response_ids)
            console.print(f"[green]Found {len(existing_reviews)} existing reviews[/green]\n")
            
            # Step 5: Filter out reviews that already exist
            reviews_to_insert = []
            for review in reviews_with_ids:
                key = (review["response_id"], review["reviewer_code"])
                if key in existing_reviews:
                    skipped += 1
                    continue
                reviews_to_insert.append(review)
            
            if not reviews_to_insert:
                console.print("[yellow]All reviews already exist in database[/yellow]")
                return (0, skipped, not_found)
            
            console.print(f"[cyan]Inserting {len(reviews_to_insert)} new reviews...[/cyan]")
            
            # Step 6: Batch insert reviews
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Inserting reviews...", total=len(reviews_to_insert))
                
                # Prepare batch insert data
                insert_data = [
                    (
                        review["response_id"],
                        review["reviewer_code"],
                        review["score"],
                        review["notes"],
                    )
                    for review in reviews_to_insert
                ]
                
                # Insert in batches
                total_batches = (len(insert_data) + BATCH_SIZE - 1) // BATCH_SIZE
                for i in range(0, len(insert_data), BATCH_SIZE):
                    batch = insert_data[i:i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1
                    try:
                        client.executemany(
                            """
                            INSERT INTO reviews (llm_response_id, reviewer_code, score, notes)
                            VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
                        restored += len(batch)
                        progress.update(task, advance=len(batch))
                        
                        # Show progress for large batches
                        if total_batches > 10 and batch_num % 10 == 0:
                            console.print(f"[dim]Inserted {batch_num}/{total_batches} batches ({restored}/{len(insert_data)} reviews)...[/dim]")
                    except Exception as e:
                        # If batch fails, try individual inserts to identify problematic records
                        error_msg = str(e).lower()
                        console.print(f"\n[yellow]Batch insert failed (batch {i//BATCH_SIZE + 1}), trying individual inserts: {e}[/yellow]")
                        for item_idx, item in enumerate(batch):
                            try:
                                # Validate item before insert
                                response_id, reviewer_code, score, notes = item
                                if not isinstance(response_id, int) or response_id <= 0:
                                    raise ValueError(f"Invalid response_id: {response_id}")
                                if not isinstance(reviewer_code, str) or not reviewer_code:
                                    raise ValueError(f"Invalid reviewer_code: {reviewer_code}")
                                if not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
                                    raise ValueError(f"Invalid score: {score}")
                                
                                client.execute(
                                    """
                                    INSERT INTO reviews (llm_response_id, reviewer_code, score, notes)
                                    VALUES (?, ?, ?, ?)
                                    """,
                                    item,
                                )
                                restored += 1
                            except Exception as insert_error:
                                error_str = str(insert_error).lower()
                                # Check for various duplicate/constraint error patterns
                                if any(phrase in error_str for phrase in [
                                    "unique constraint failed",
                                    "duplicate",
                                    "constraint",
                                    "already exists"
                                ]):
                                    skipped += 1
                                elif "foreign key constraint failed" in error_str:
                                    console.print(f"[red]Foreign key error (response_id {item[0]} may not exist): {insert_error}[/red]")
                                    skipped += 1
                                elif "check constraint failed" in error_str:
                                    console.print(f"[red]Check constraint error (score {item[2]} out of range): {insert_error}[/red]")
                                    skipped += 1
                                else:
                                    console.print(f"[red]Error inserting review (response_id={item[0]}, reviewer={item[1]}): {insert_error}[/red]")
                                    skipped += 1
                            progress.update(task, advance=1)
    except Exception as e:
        console.print(f"[red]Fatal error during database operations: {e}[/red]")
        console.print("[red]Some reviews may have been restored before the error occurred.[/red]")
        raise
    
    return (restored, skipped, not_found)


def main() -> None:
    """Restore all reviews from exported files."""
    console.print("[bold cyan]Restore Reviews from Exports[/bold cyan]\n")
    
    console.print("[yellow]This will restore reviews by matching on identifier/run_id.[/yellow]")
    console.print("[yellow]Reviews that already exist will be skipped.[/yellow]\n")
    
    response = input("Continue? [y/N]: ").strip().lower()
    if response != "y":
        console.print("[yellow]Restore cancelled.[/yellow]")
        sys.exit(0)
    
    console.print()
    
    total_restored = 0
    total_skipped = 0
    total_not_found = 0
    
    for dataset in [Dataset.AITA, Dataset.SCENARIO]:
        console.print(f"\n[bold]Processing {dataset.value}...[/bold]")
        try:
            restored, skipped, not_found = restore_reviews_for_dataset(dataset)
            
            console.print(f"\n[green]Restored:[/green] {restored}")
            console.print(f"[yellow]Skipped (already exist or invalid):[/yellow] {skipped}")
            console.print(f"[red]Not found (no matching response):[/red] {not_found}")
            
            total_restored += restored
            total_skipped += skipped
            total_not_found += not_found
        except Exception as e:
            console.print(f"[red]Fatal error processing {dataset.value}: {e}[/red]")
            console.print("[yellow]Continuing with next dataset...[/yellow]")
            continue
    
    console.print(f"\n[bold cyan]Summary:[/bold cyan]")
    console.print(f"  [green]Total restored:[/green] {total_restored}")
    console.print(f"  [yellow]Total skipped:[/yellow] {total_skipped}")
    console.print(f"  [red]Total not found:[/red] {total_not_found}")
    
    if total_restored > 0:
        console.print(f"\n[green]Successfully restored {total_restored} reviews![/green]")


if __name__ == "__main__":
    main()

