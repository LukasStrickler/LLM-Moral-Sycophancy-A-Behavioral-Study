#!/usr/bin/env python3
"""Export LLM scored reviews to JSONL file.

This script exports all scored responses from the database to aita_1k_scored.jsonl
with their LLM review scores.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LABEL_DATA_ROOT", str(PROJECT_ROOT / "data" / "humanLabel"))
os.environ.setdefault("STREAMLIT_RUNS_ROOT", str(PROJECT_ROOT / "outputs" / "runs"))

from src.labeling_app.core.models import Dataset
from src.labeling_app.db import client_scope
from src.labeling_app.db.queries import select_reviews_with_responses
from src.labeling_app.settings import get_settings


OUTPUT_FILE = PROJECT_ROOT / "data" / "humanLabel" / "seeds" / "aita_1k_scored.jsonl"


def main() -> None:
    """Main execution."""
    print("=" * 60)
    print("Export Scored Reviews")
    print("=" * 60)
    print()
    
    settings = get_settings()
    
    # Reviewer codes for the 3 models
    REVIEWER_CODES = {
        'cohere': 'llm:cohere/command-r-08-2024',
        'gemini': 'llm:openrouter/google/gemini-2.0-flash-lite-001',
        'deepseek': 'llm:openrouter/tngtech/deepseek-r1t-chimera:free',
    }
    
    with client_scope(settings) as client:
        # Get all reviews with responses
        print("Fetching reviews from database...")
        reviews_data = select_reviews_with_responses(client, Dataset.AITA)
        print(f"Found {len(reviews_data)} review records")
        
        # Group by identifier to collect all scores for each response
        responses_dict = {}
        for review in reviews_data:
            identifier = review['identifier']
            
            if identifier not in responses_dict:
                # Parse metadata if it's a string
                metadata = review.get('metadata_json', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                responses_dict[identifier] = {
                    'identifier': identifier,
                    'prompt_title': review.get('prompt_title', ''),
                    'prompt_body': review.get('prompt_body', ''),
                    'model_response_text': review.get('model_response_text', ''),
                    'model_id': review.get('model_id', ''),
                    'run_id': review.get('run_id', ''),
                    'metadata': metadata,
                    'scores': {}
                }
            
            # Add score based on reviewer code
            reviewer_code = review.get('reviewer_code', '')
            score = review.get('score')
            
            if reviewer_code == REVIEWER_CODES['cohere']:
                responses_dict[identifier]['scores']['cohere'] = score
            elif reviewer_code == REVIEWER_CODES['gemini']:
                responses_dict[identifier]['scores']['gemini'] = score
            elif reviewer_code == REVIEWER_CODES['deepseek']:
                responses_dict[identifier]['scores']['deepseek'] = score
        
        print(f"Found {len(responses_dict)} unique responses")
        
        # Convert to list and sort by identifier
        responses_list = list(responses_dict.values())
        responses_list.sort(key=lambda x: x['identifier'])
        
        # Write to JSONL file
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting to: {OUTPUT_FILE}")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for response in responses_list:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        
        print(f"✓ Exported {len(responses_list)} scored responses to {OUTPUT_FILE}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Total responses: {len(responses_list)}")
        cohere_count = sum(1 for r in responses_list if 'cohere' in r['scores'])
        gemini_count = sum(1 for r in responses_list if 'gemini' in r['scores'])
        deepseek_count = sum(1 for r in responses_list if 'deepseek' in r['scores'])
        print(f"  With Cohere scores: {cohere_count}")
        print(f"  With Gemini scores: {gemini_count}")
        print(f"  With DeepSeek scores: {deepseek_count}")
        all_three = sum(1 for r in responses_list if all(k in r['scores'] for k in ['cohere', 'gemini', 'deepseek']))
        print(f"  With all 3 scores: {all_three}")


if __name__ == "__main__":
    main()




