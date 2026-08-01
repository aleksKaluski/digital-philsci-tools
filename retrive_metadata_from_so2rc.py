"""
Modified s2_api_requests.py created in order to fetch the additional metadata from the results
of enrich_tm_with_metadata.py
"""

import requests
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# provide path to your folder  with seed papers

# on Windows
# DATA_PATH = '/mnt/c/Python_files/digital-philsci-tools/files/operational_files'

# on Linux
DATA_PATH = '/home/akaluski/PycharmProjects/digital-philsci-tools/files/operational_files'


# ============================================================================
# CORE INDIVIDUAL QUERY FUNCTION
# ============================================================================

def get_paper(paper_id: str, headers: Dict, fields: List[str], sleep_time=1, max_retries=3):
    """
    Generic function to fetch a single paper with retry logic.

    Args:
        paper_id: Paper identifier (already formatted, e.g., 'CorpusId:123', 'DOI:10.1234/...')
        headers: API headers including API key
        fields: List of fields to retrieve (e.g., ['corpusId', 'title', 'embedding.specter_v2'])
        sleep_time: Initial sleep time for exponential backoff
        max_retries: Maximum number of retries for transient errors

    Returns:
        Tuple of (paper data dictionary or None, error_code or None)
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

    params = {
        'fields': ','.join(fields)
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
                return response.json(), None
            elif response.status_code == 404:
                # Paper not found - not a retry case
                return None, 404
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = sleep_time * (2 ** attempt)
                print(
                    f"    Rate limit hit for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry
                wait_time = sleep_time * (2 ** attempt)
                print(
                    f"    Server error {response.status_code} for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                # Other errors - don't retry
                print(f"    Error fetching {paper_id}: Status {response.status_code}")
                return None, response.status_code

        except requests.exceptions.Timeout:
            wait_time = sleep_time * (2 ** attempt)
            print(f"    Timeout for {paper_id}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"    Error fetching {paper_id}: {e}")
            return None, 'exception'

    # All retries exhausted
    print(f"    ✗ Failed after {max_retries} attempts: {paper_id}")
    return None, 'max_retries_exceeded'


# ============================================================================
# HELPER FUNCTIONS FOR LOADING DATA AND CONFIGURATION
# ============================================================================

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


def load_corpus_ids(filepath: str):
    """
    Load corpus IDs from a JSON file.
    """
    list_of_ids = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        paragraphs = data.get("paragraphs", [])

        # iterate through paragraphs
        for paragraph in paragraphs:
            metadata = paragraph.get("metadata", {})
            if isinstance(metadata, dict):
                corpus_id = metadata.get("corpusid")
                if corpus_id is not None:
                    list_of_ids.append(str(corpus_id))

    return list_of_ids, data



def load_dois(filepath: str) -> List[str]:
    """Load DOIs from a text file (one per line), removing duplicates."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    with open(filepath, 'r') as f:
        dois = [line.strip() for line in f if line.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_dois = []
    for doi in dois:
        if doi not in seen:
            seen.add(doi)
            unique_dois.append(doi)

    print(f"Loaded {len(unique_dois)} unique DOIs from {filepath}")
    return unique_dois


def format_corpus_id(corpus_id: str) -> str:
    """Format corpus ID with CorpusId: prefix."""
    return f"CorpusId:{corpus_id}"


def format_doi(doi: str) -> str:
    """Format DOI with DOI: prefix."""
    return f"DOI:{doi}"


def save_failed_items(failed_items: List[Tuple[str, str]], prefix: str, timestamp: str):
    """
    Save failed items to a timestamped JSON file.

    Args:
        failed_items: List of (identifier, error_code) tuples
        prefix: Prefix for filename (e.g., 'corpus_ids', 'dois')
        timestamp: Timestamp string for filename
    """
    if not failed_items:
        return None

    failed_file = f'{DATA_PATH}/failed_{prefix}_{timestamp}.json'

    # Convert to list of dicts for better JSON structure
    failed_data = [
        {
            'identifier': identifier,
            'error_code': str(error_code)
        }
        for identifier, error_code in failed_items
    ]

    with open(failed_file, 'w') as f:
        json.dump(failed_data, f, indent=2)

    return failed_file


def retry_failed_requests(failed_items: List[Tuple[str, str]], headers: Dict, fields: List[str],
                          format_func, rate_limit_delay=1, max_retry_rounds=3):
    """
    Retry failed requests with progressive backoff.

    Args:
        failed_items: List of (identifier, error_code) tuples
        headers: API headers
        fields: Fields to retrieve
        format_func: Function to format the identifier (format_corpus_id or format_doi)
        rate_limit_delay: Base delay between requests
        max_retry_rounds: Maximum number of retry rounds

    Returns:
        Tuple of (successful_results, still_failed_items)
    """
    successful = []
    still_failed = failed_items.copy()

    for round_num in range(1, max_retry_rounds + 1):
        if not still_failed:
            break

        print(f"\n{'=' * 60}")
        print(f"Retry Round {round_num}/{max_retry_rounds}")
        print(f"Attempting to retry {len(still_failed)} failed requests...")
        print(f"{'=' * 60}\n")

        round_failed = []
        wait_time = rate_limit_delay * (2 ** (round_num - 1))  # Exponential backoff between rounds

        for identifier, error_code in still_failed:
            paper_id = format_func(identifier)
            paper, new_error = get_paper(paper_id, headers, fields, sleep_time=wait_time)

            if paper:
                successful.append((identifier, paper))
                print(f"  ✓ Retry successful: {identifier}")
            else:
                round_failed.append((identifier, new_error))

            time.sleep(wait_time)

        print(f"\nRound {round_num} results: {len(successful)} recovered, {len(round_failed)} still failed")
        still_failed = round_failed

    return successful, still_failed


def fetch_papers_metadata(input_file_path:str|Path,
                          output_folder_path: str|Path,
                          rate_limit_delay:int=1):
    """
    Fetch paper metadata for corpus IDs individually with automatic retry.
    Saves results as a JSON file with full paper metadata.
    """
    timestamp = get_timestamp()
    headers = load_headers()

    input_file_path = Path(input_file_path)
    assert input_file_path.is_file(), "The file that you passed is not a file!"

    corpus_ids, file_content = load_corpus_ids(input_file_path)

    if not corpus_ids:
        return

    print(f"Processing {len(corpus_ids)} papers individually...\n")

    fields = [
        'corpusId', 'paperId', 'title', 'year', 'authors',
        'citationCount', 'abstract', 'venue', 'publicationDate',
        'referenceCount', 'influentialCitationCount', 'fieldsOfStudy',
        'publicationTypes', 'journal', 'isOpenAccess'
    ]

    paper_metadata = []
    failed_items = []
    not_found_count = 0

    # First pass
    for i, corpus_id in enumerate(corpus_ids, 1):
        paper_id = format_corpus_id(corpus_id)
        paper, error_code = get_paper(paper_id, headers, fields)

        if paper:
            paper_metadata.append(paper)
            if i % 100 == 0:
                print(f"  [{i}/{len(corpus_ids)}] Retrieved metadata")
        elif error_code == 404:
            not_found_count += 1
        else:
            failed_items.append((corpus_id, error_code))
            print(f"  [{i}/{len(corpus_ids)}] ⚠ Failed (will retry): {corpus_id}")

        time.sleep(rate_limit_delay)

    # Retry failed requests
    if failed_items:
        successful_retries, still_failed = retry_failed_requests(
            failed_items, headers, fields, format_corpus_id, rate_limit_delay
        )

        for corpus_id, paper in successful_retries:
            paper_metadata.append(paper)

        if still_failed:
            failed_file = save_failed_items(still_failed, 'corpus_ids_metadata', timestamp)
            print(f"\n⚠ {len(still_failed)} corpus IDs still failed after retries")
            print(f"  Saved to: {failed_file}")

    # map metadata on the papers
    paper_metadata_map = {paper['corpusId']: paper for paper in paper_metadata}

    # Enrich each paragraph
    for paragraph in file_content['paragraphs']:
        corpus_id = paragraph['metadata']['corpusid']

        if corpus_id in paper_metadata_map:
            paper_data = paper_metadata_map[corpus_id]

            # Add paper metadata to paragraph metadata
            paragraph['metadata'].update({
                'paper_title': paper_data.get('title'),
                'paper_year': paper_data.get('year'),
                'paper_authors': paper_data.get('authors'),
                'paper_abstract': paper_data.get('abstract'),
                'paper_venue': paper_data.get('venue'),
                'paper_citation_count': paper_data.get('citationCount'),
                'paper_fields_of_study': paper_data.get('fieldsOfStudy'),
                'paper_journal': paper_data.get('journal'),
                'paper_is_open_access': paper_data.get('isOpenAccess')
            })



    # create output path
    output_folder_path = Path(output_folder_path)
    assert output_folder_path.is_dir(), "The path to a folder you passed is not dir!"

    output_file_path = output_folder_path.joinpath(input_file_path.name.replace("_processed.json", "_final.json"))

    with open(output_file_path, 'w') as f:
        json.dump(file_content, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Metadata retrieved: {len(paper_metadata)}/{len(corpus_ids)} papers")
    print(f"  Not found (404): {not_found_count}")


    if failed_items:
        print(f"  Failed after retries: {len(still_failed)}")
    print(f"\nOutput:")
    print(f"  {output_file_path}")
    print(f"{'=' * 60}")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("=" * 60)
    print("Semantic Scholar API - Individual Query Interface")
    print("=" * 60)
    print("\n" + "=" * 60)

    folder_path = Path(r'BERTopic_results/processed/d7b_processed')

    new_folder_path = r"BERTopic_results/final/" + folder_path.name.split("_")[0] + "_final"
    new_folder_path = Path(new_folder_path)
    new_folder_path.mkdir(parents=True, exist_ok=True)

    for file_path in folder_path.iterdir():
        fetch_papers_metadata(input_file_path=file_path,
                              output_folder_path=new_folder_path)




if __name__ == "__main__":
    from datetime import datetime

    main()
    print(f"\n{'=' * 60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")