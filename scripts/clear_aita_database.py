#!/usr/bin/env python3
"""Clear all AITA responses and reviews from the database.

This script deletes all responses for the AITA dataset, which will
automatically cascade delete all associated reviews via foreign key constraint.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset
from src.labeling_app.db import client_scope, queries
from src.labeling_app.settings import get_settings


def count_aita_reviews(client) -> int:
    """Count reviews for AITA dataset."""
    result = client.execute(
        """
        SELECT COUNT(*) 
        FROM reviews r
        JOIN llm_responses resp ON r.llm_response_id = resp.id
        WHERE resp.dataset = ?
        """,
        [Dataset.AITA.value],
    )
    return int(result.first_value() or 0)


def main() -> None:
    """Main execution."""
    parser = argparse.ArgumentParser(description="Clear all AITA responses and reviews from database")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()
    
    settings = get_settings()
    
    with client_scope(settings) as client:
        # Count responses and reviews
        response_count = queries.count_responses(client, Dataset.AITA)
        review_count = count_aita_reviews(client)
        
        if response_count == 0:
            print("No AITA responses found in database. Nothing to delete.")
            return
        
        print("=" * 60)
        print("Clear AITA Database")
        print("=" * 60)
        print()
        print(f"Found in database:")
        print(f"  - Responses: {response_count}")
        print(f"  - Reviews: {review_count}")
        print()
        
        if not args.yes:
            response = input("Are you sure you want to delete all AITA responses and reviews? [y/N]: ").strip().lower()
            if response != "y":
                print("Cancelled.")
                return
        
        print()
        print("Deleting all AITA responses (reviews will cascade delete)...")
        
        # Delete all AITA responses (reviews will cascade delete automatically)
        client.execute("DELETE FROM llm_responses WHERE dataset = ?", [Dataset.AITA.value])
        
        # Verify deletion
        remaining_responses = queries.count_responses(client, Dataset.AITA)
        remaining_reviews = count_aita_reviews(client)
        
        print()
        print("=" * 60)
        print("Deletion complete!")
        print(f"  - Deleted {response_count} responses")
        print(f"  - Deleted {review_count} reviews (cascade)")
        print(f"  - Remaining responses: {remaining_responses}")
        print(f"  - Remaining reviews: {remaining_reviews}")
        print("=" * 60)


if __name__ == "__main__":
    main()





