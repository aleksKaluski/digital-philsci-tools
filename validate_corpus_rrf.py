""""
Validate corpus IDs by fetching titles from Semantic Scholar API.
Samples based on RRF score distribution (high, medium, low tertiles).

This script:
1. Loads corpus IDs and RRF scores from a JSON file
2. Sorts papers by RRF score (descending - highest first)
3. Divides into three tertiles (high, medium, low RRF scores)
4. Samples equally from each tertile
5. Fetches titles, authors, venue, year from Semantic Scholar API for the sampled IDs
6. Includes RRF score and percentile information in results
7. Saves validation results as JSON
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import requests
import numpy as np

def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_headers() -> Dict:
    """Load API headers from .env file."""
    load_dotenv()
    headers_str = os.getenv('HEADERS')

    try:
        headers = json.loads(headers_str) if headers_str else {}
    except json.JSONDecodeError:
        print("Error: HEADERS in .env file is not valid JSON")
        headers = {}

    if not headers:
        print("Warning: HEADERS not found in .env file")
        print("The script will use the public API with rate limits")

    return headers

def load_rrf_data(filepath: str) -> List[Dict]:
    """
    Load corpus IDs with RRF scores from a JSON file.

    Expected format:
    [
      {"corpus_id": 255971812, "rrf_score": 0.18798602052282487},
      {"corpus_id": 9261973, "rrf_score": 0.18598112248215903},
      ...
    ]
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Error: Expected a list in JSON file, got {type(data)}")
        return []

    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Item {i} is not a dict, skipping")
            continue
        if 'corpus_id' not in item:
            print(f"Warning: Item {i} missing 'corpus_id', skipping")
            continue
        if 'rrf_score' not in item:
            print(f"Warning: Item {i} missing 'rrf_score', using 0.0")
            item['rrf_score'] = 0.0
        valid_items.append(item)

    print(f"Loaded {len(valid_items)} papers with RRF scores from {filepath}")
    return valid_items

def split_into_tertiles(data: List[Dict], sample_size: int) -> List[List[Dict]]:
    """
    Split sorted data into three tertiles and return samples from each.
    """
    if not data:
        return [[], [], []]

    n = len(data)
    tertile_size = n // 3

    high_tertile = data[:tertile_size]
    medium_tertile = data[tertile_size:2*tertile_size]
    low_tertile = data[2*tertile_size:]

    samples_per_tertile = max(1, sample_size // 3)

    high_sample = random.sample(high_tertile, min(samples_per_tertile, len(high_tertile)))
    medium_sample = random.sample(medium_tertile, min(samples_per_tertile, len(medium_tertile)))
    low_sample = random.sample(low_tertile, min(samples_per_tertile, len(low_tertile)))

    print(f"  High tertile (top {tertile_size} papers): {len(high_sample)} samples")
    print(f"  Medium tertile (middle {tertile_size} papers): {len(medium_sample)} samples")
    print(f"  Low tertile (bottom {len(low_tertile)} papers): {len(low_sample)} samples")

    return [high_sample, medium_sample, low_sample]

def sample_by_rrf_distribution(data: List[Dict], sample_percent: float = 5.0,
                               sample_count: Optional[int] = None) -> Tuple[List[Dict], List[str], Dict]:
    """
    Sample papers based on RRF score distribution (tertiles).
    Returns (sampled_items, sampled_corpus_ids, percentile_info)
    """
    if not data:
        return [], [], {}

    total = len(data)

    if sample_count is not None:
        n = min(sample_count, total)
    else:
        n = max(1, int(total * sample_percent / 100))

    sorted_data = sorted(data, key=lambda x: x.get('rrf_score', 0), reverse=True)
    all_scores = [item.get('rrf_score', 0) for item in sorted_data]

    print(f"Sampling {n}/{total} papers ({n/total*100:.1f}%)")
    print("Sorting by RRF score (descending) and splitting into tertiles...")

    tertile_samples = split_into_tertiles(sorted_data, n)

    sampled_items = []
    for sample in tertile_samples:
        sampled_items.extend(sample)

    sampled_ids = []
    percentile_info = {}

    for item in sampled_items:
        corpus_id = str(item['corpus_id'])
        rrf_score = item.get('rrf_score', 0)
        sampled_ids.append(corpus_id)

        # Calculate percentile (0-100, where 100 is highest RRF score)
        count_higher_or_equal = sum(1 for s in all_scores if s >= rrf_score)
        percentile = 100 * count_higher_or_equal / len(all_scores)
        percentile_info[corpus_id] = round(percentile, 2)

    random.shuffle(sampled_items)
    return sampled_items, sampled_ids, percentile_info

def fetch_paper_metadata(paper_id: str, headers: Dict, sleep_time: float = 1.0,
                         max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Fetch metadata for a single paper with retry logic.
    """
    formatted_id = f"CorpusId:{paper_id}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{formatted_id}"

    params = {
        'fields': 'title,corpusId,authors,venue,year'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                metadata = {
                    'title': data.get('title', ''),
                    'corpus_id': data.get('corpusId', paper_id),
                    'authors': [author.get('name', '') for author in data.get('authors', [])] if data.get('authors') else [],
                    'venue': data.get('venue', ''),
                    'year': data.get('year', None)
                }
                return metadata, None
            elif response.status_code == 404:
                return None, '404_not_found'
            elif response.status_code == 429:
                wait_time = sleep_time * (2 ** attempt)
                print(f"    Rate limit hit for {paper_id}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:
                wait_time = sleep_time * (2 ** attempt)
                print(f"    Server error {response.status_code} for {paper_id}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                return None, f"error_{response.status_code}"

        except requests.exceptions.Timeout:
            wait_time = sleep_time * (2 ** attempt)
            print(f"    Timeout for {paper_id}, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            return None, f"exception_{str(e)}"

    return None, f"max_retries_exceeded_{max_retries}"

def validate_corpus_sample(corpus_ids: List[str], headers: Dict,
                          rate_limit_delay: float = 1.0,
                          rrf_scores: Dict[str, float] = None,
                          percentile_info: Dict[str, float] = None) -> Dict[str, Dict]:
    """
    Validate a sample of corpus IDs by fetching their metadata.
    """
    results = {}
    print(f"\nValidating {len(corpus_ids)} corpus IDs...")
    print("-" * 60)

    rrf_scores = rrf_scores or {}
    percentile_info = percentile_info or {}

    for i, corpus_id in enumerate(corpus_ids, 1):
        metadata, error = fetch_paper_metadata(corpus_id, headers,
                                               sleep_time=rate_limit_delay)

        result = {
            'corpus_id': corpus_id,
            'valid': metadata is not None,
            'title': metadata.get('title', '') if metadata else None,
            'authors': metadata.get('authors', []) if metadata else [],
            'venue': metadata.get('venue', '') if metadata else None,
            'year': metadata.get('year') if metadata else None,
            'rrf_score': rrf_scores.get(corpus_id, None),
            'percentile': percentile_info.get(corpus_id, None),
            'error': error
        }
        results[corpus_id] = result

        status = "✓" if metadata else "✗"
        title = metadata.get('title', '')[:40] if metadata else str(error)
        print(f"  [{i}/{len(corpus_ids)}] {status} {corpus_id}: {title}")

        time.sleep(rate_limit_delay)

    return results

def calculate_statistics(results: Dict[str, Dict]) -> Dict:
    """Calculate validation statistics."""
    total = len(results)
    valid = sum(1 for r in results.values() if r['valid'])
    invalid = total - valid

    errors = {}
    for r in results.values():
        if not r['valid'] and r.get('error'):
            error_type = r['error'].split('_')[0] if '_' in r['error'] else r['error']
            errors[error_type] = errors.get(error_type, 0) + 1

    return {
        'total_checked': total,
        'valid': valid,
        'invalid': invalid,
        'valid_percentage': round(100 * valid / total, 2) if total > 0 else 0,
        'error_counts': errors
    }

def calculate_rrf_statistics(sampled_items: List[Dict]) -> Dict:
    """Calculate RRF score statistics for the sample."""
    if not sampled_items:
        return {}

    scores = [item.get('rrf_score', 0) for item in sampled_items]

    return {
        'mean_rrf': round(float(np.mean(scores)), 6),
        'median_rrf': round(float(np.median(scores)), 6),
        'min_rrf': round(float(np.min(scores)), 6),
        'max_rrf': round(float(np.max(scores)), 6),
        'std_rrf': round(float(np.std(scores)), 6)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Validate corpus IDs by fetching metadata from Semantic Scholar API. '
                    'Samples from high, medium, and low RRF score tertiles.'
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to JSON file containing corpus IDs with RRF scores')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save validation results as JSON')
    parser.add_argument('--sample-percent', type=float, default=5.0,
                        help='Percentage of corpus IDs to sample (default: 5.0)')
    parser.add_argument('--sample-count', type=int, default=None,
                        help='Exact number of corpus IDs to sample (overrides --sample-percent)')
    parser.add_argument('--rate-limit-delay', type=float, default=1.0,
                        help='Delay between API requests in seconds (default: 1.0)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for failed requests (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("CORPUS ID VALIDATION (RRF-BASED SAMPLING)")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Sample: {args.sample_count or f'{args.sample_percent}%'} of corpus IDs")
    print(f"Rate limit delay: {args.rate_limit_delay}s")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    headers = load_headers()
    rrf_data = load_rrf_data(args.input)
    if not rrf_data:
        print("Error: No RRF data loaded. Exiting.")
        return

    sampled_items, sampled_ids, percentile_info = sample_by_rrf_distribution(
        rrf_data,
        sample_percent=args.sample_percent,
        sample_count=args.sample_count
    )

    if not sampled_ids:
        print("Error: No corpus IDs sampled. Exiting.")
        return

    rrf_scores = {str(item['corpus_id']): item.get('rrf_score', 0) for item in sampled_items}
    rrf_stats = calculate_rrf_statistics(sampled_items)

    results = validate_corpus_sample(
        sampled_ids,
        headers,
        rate_limit_delay=args.rate_limit_delay,
        rrf_scores=rrf_scores,
        percentile_info=percentile_info
    )

    stats = calculate_statistics(results)

    output_data = {
        'metadata': {
            'timestamp': get_timestamp(),
            'input_file': args.input,
            'total_corpus_ids': len(rrf_data),
            'sampled_count': len(sampled_ids),
            'sample_percentage': round(100 * len(sampled_ids) / len(rrf_data), 2),
            'rrf_statistics': rrf_stats,
            'sampling_method': 'RRF tertile-based',
            'parameters': {
                'sample_percent': args.sample_percent,
                'sample_count': args.sample_count,
                'rate_limit_delay': args.rate_limit_delay,
                'random_seed': args.seed
            }
        },
        'statistics': stats,
        'results': results
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total corpus IDs: {stats['total_checked']}")
    print(f"Valid (metadata found): {stats['valid']} ({stats['valid_percentage']}%)")
    print(f"Invalid: {stats['invalid']}")
    if stats['error_counts']:
        print(f"Error types: {stats['error_counts']}")
    print(f"\nRRF Statistics for sample:")
    for key, value in rrf_stats.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {args.output}")
    print("=" * 70)

if __name__ == "__main__":
    main()
