#!/usr/bin/env python3
"""Compare model responses from final_consensus_export.csv with aita_seed_v2.jsonl.

This script:
1. Loads model_response_text from final_consensus_export.csv
2. Loads model_response_text from aita_seed_v2.jsonl
3. Compares them to find duplicates based on actual response content
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Set

from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSENSUS_CSV = PROJECT_ROOT / "data" / "humanLabel" / "reviews" / "final_consensus_export.csv"
V2_SEED = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_seed_v2.jsonl"


def load_responses_from_csv(csv_file: Path) -> Dict[str, str]:
    """Load model_response_text from CSV, keyed by a normalized version for comparison."""
    if not csv_file.exists():
        console.print(f"[red]Error: CSV file not found: {csv_file}[/red]")
        return {}
    
    responses = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                response_text = row.get("model_response_text", "").strip()
                if response_text:
                    # Normalize for comparison (remove extra whitespace)
                    normalized = " ".join(response_text.split())
                    responses[normalized] = response_text
            except Exception as e:
                continue
    
    return responses


def load_responses_from_seed(seed_file: Path) -> Dict[str, Dict]:
    """Load model_response_text from seed file, return dict mapping normalized text to full record."""
    if not seed_file.exists():
        console.print(f"[red]Error: Seed file not found: {seed_file}[/red]")
        return {}
    
    responses = {}
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    response_text = record.get("model_response_text", "").strip()
                    if response_text:
                        # Normalize for comparison
                        normalized = " ".join(response_text.split())
                        responses[normalized] = {
                            "identifier": record.get("identifier", ""),
                            "model_id": record.get("model_id", ""),
                            "prompt_title": record.get("prompt_title", ""),
                            "response_text": response_text,
                        }
                except json.JSONDecodeError:
                    continue
    
    return responses


def main() -> None:
    """Compare responses from CSV and seed file."""
    console.print("[bold cyan]Compare Model Responses: Consensus CSV vs V2 Seed[/bold cyan]\n")
    
    # Step 1: Load responses from CSV
    console.print("[cyan]Step 1: Loading model responses from consensus CSV...[/cyan]")
    csv_responses = load_responses_from_csv(CONSENSUS_CSV)
    console.print(f"[green]Found {len(csv_responses)} unique model responses in CSV[/green]\n")
    
    if not csv_responses:
        console.print("[yellow]No responses found in CSV[/yellow]")
        sys.exit(0)
    
    # Step 2: Load responses from v2 seed
    console.print("[cyan]Step 2: Loading model responses from v2 seed...[/cyan]")
    seed_responses = load_responses_from_seed(V2_SEED)
    console.print(f"[green]Found {len(seed_responses)} unique model responses in v2 seed[/green]\n")
    
    if not seed_responses:
        console.print("[yellow]No responses found in v2 seed[/yellow]")
        sys.exit(0)
    
    # Step 3: Compare
    console.print("[cyan]Step 3: Comparing responses...[/cyan]\n")
    
    # Find overlaps by comparing normalized response text
    csv_normalized = set(csv_responses.keys())
    seed_normalized = set(seed_responses.keys())
    overlap = csv_normalized & seed_normalized
    
    if overlap:
        console.print(f"[red]⚠️  DUPLICATES FOUND: {len(overlap)} model responses in v2 seed already exist in consensus CSV![/red]\n")
        console.print("[red]These responses in v2 seed were already reviewed:[/red]\n")
        
        for i, normalized_text in enumerate(sorted(list(overlap))[:20], 1):
            seed_record = seed_responses[normalized_text]
            console.print(f"[red]{i}. Identifier: {seed_record['identifier']}[/red]")
            console.print(f"   Model: {seed_record['model_id']}")
            console.print(f"   Prompt: {seed_record['prompt_title'][:60]}...")
            console.print(f"   Response preview: {seed_record['response_text'][:100]}...")
            console.print()
        
        if len(overlap) > 20:
            console.print(f"[red]  ... and {len(overlap) - 20} more duplicates[/red]\n")
        
        console.print("[red]❌ VERIFICATION FAILED: V2 seed contains duplicate responses that were already reviewed[/red]")
    else:
        console.print("[green]✅ VERIFICATION PASSED: No duplicate responses found![/green]")
        console.print(f"[green]All {len(seed_responses)} responses in v2 seed are new and different from reviewed responses[/green]")
    
    console.print()
    console.print("=" * 60)
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Responses in consensus CSV: {len(csv_responses)}")
    console.print(f"  Responses in v2 seed: {len(seed_responses)}")
    console.print(f"  Overlap (duplicate responses): {len(overlap)}")
    if overlap:
        console.print(f"\n[red]Percentage of v2 seed that are duplicates: {len(overlap) / len(seed_responses) * 100:.1f}%[/red]")
    console.print("=" * 60)


if __name__ == "__main__":
    main()


