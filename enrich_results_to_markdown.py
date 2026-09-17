#!/usr/bin/env python3
"""
Script to enrich results from a JSON file with Semantic Scholar metadata
and output the top N ranked results as a readable Markdown file.

Works with any results JSON file that has the structure:
  [
    {
      "rank": 1,
      "rrf_score": 0.123,
      "corpusid": 12345,
      "text": "...",
      ...
    },
    ...
  ]

Usage:
    python enrich_results_to_markdown.py path/to/results_file.json [--top N] [--output-dir DIR]

Arguments:
    results_file.json   Path to the input JSON file (required)
    --top N           Number of top results to process (default: 100)
    --output-dir DIR  Directory for output files (default: same as input file)

Requirements:
    - requests
    - python-dotenv
    - tqdm

Environment:
    Create a .env file with your Semantic Scholar API key:
    SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
"""

import argparse
import json
import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of top results to process (can be overridden by --top)
TOP_N = 100

# Rate limiting
RATE_LIMIT_DELAY = 2.0  # seconds between API requests

# Rate limiting
RATE_LIMIT_DELAY = 2.0  # seconds between API requests

# API fields to retrieve from Semantic Scholar
S2_FIELDS = [
    'corpusId',
    'paperId', 
    'title',
    'year',
    'authors',
    'abstract',
    'venue',
    'publicationDate',
    'citationCount',
    'referenceCount',
    'influentialCitationCount',
    'fieldsOfStudy',
    'publicationTypes',
    'journal',
    'isOpenAccess',
    'url',
    'externalIds',
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_api_key() -> Optional[str]:
    """Load Semantic Scholar API key from environment.

    Accepts either SEMANTIC_SCHOLAR_API_KEY=<key> or the repo's
    HEADERS={"x-api-key": "<key>"} convention.
    """
    load_dotenv()

    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    if api_key:
        return api_key.strip()

    raw = os.getenv('HEADERS')
    if raw:
        try:
            parsed = json.loads(raw)
            api_key = parsed.get('x-api-key')
            if api_key:
                return api_key.strip()
        except json.JSONDecodeError:
            print("Warning: HEADERS in .env is not valid JSON")

    print("Warning: no Semantic Scholar API key found in .env")
    print("Falling back to the public API (heavily rate limited)")
    return None


def get_headers() -> Dict[str, str]:
    """Get headers for API requests."""
    api_key = load_api_key()
    headers = {
        'User-Agent': 'digital-philsci-tools/1.0'
    }
    if api_key:
        headers['x-api-key'] = api_key
    return headers


def get_paper_metadata(corpus_id: int, headers: Dict[str, str], max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch metadata for a single paper from Semantic Scholar API.
    
    Args:
        corpus_id: The Semantic Scholar corpus ID
        headers: API headers
        max_retries: Maximum number of retry attempts
        
    Returns:
        Paper metadata dict or None if failed
    """
    paper_id = f"CorpusId:{corpus_id}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    
    params = {
        'fields': ','.join(S2_FIELDS)
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
                return response.json()
            elif response.status_code == 429:
                # Rate limited
                wait_time = RATE_LIMIT_DELAY * (2 ** attempt)
                print(f"  Rate limited for {corpus_id}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 404:
                print(f"  Paper {corpus_id} not found (404)")
                return None
            elif response.status_code in [500, 502, 503, 504]:
                # Server error, retry
                wait_time = RATE_LIMIT_DELAY * (2 ** attempt)
                print(f"  Server error for {corpus_id}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Error {response.status_code} for {corpus_id}")
                return None
                
        except requests.exceptions.Timeout:
            wait_time = RATE_LIMIT_DELAY * (2 ** attempt)
            print(f"  Timeout for {corpus_id}, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"  Exception for {corpus_id}: {e}")
            return None
    
    return None


def format_authors(authors: List[Dict]) -> str:
    """Format authors list into a readable string."""
    if not authors:
        return "No authors listed"
    
    # Extract names
    names = []
    for author in authors:
        name = author.get('name', '')
        if name:
            names.append(name)
    
    if not names:
        return "No authors listed"
    
    # Format as "First A. Last, Second B. Last, et al." for many authors
    if len(names) <= 3:
        return ", ".join(names)
    else:
        return ", ".join(names[:3]) + f" et al. ({len(names)} total)"


def format_fields_of_study(fields) -> str:
    """Format fields of study into a readable string.
    
    fields can be either a list of strings or a list of dicts.
    """
    if not fields:
        return ""
    
    # Handle both string list and dict list
    if isinstance(fields, list):
        if fields and isinstance(fields[0], dict):
            # List of dicts with 'category' key
            return ", ".join([f.get('category', '') for f in fields if f.get('category')])
        else:
            # List of strings
            return ", ".join([f for f in fields if f])
    return ""


def format_date(date_str: Optional[str]) -> str:
    """Format publication date."""
    if not date_str:
        return ""
    return date_str


def get_paper_url(paper_data: Dict) -> str:
    """Get the best available URL for the paper."""
    url = paper_data.get('url', '')
    if url:
        return url
    
    # Try to construct URL from paperId
    paper_id = paper_data.get('paperId', '')
    if paper_id:
        return f"https://www.semanticscholar.org/paper/{paper_id}"
    
    return ""



# ============================================================================
# MAIN PROCESSING
# ============================================================================

def load_results(filepath: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def enrich_results(results: List[Dict], headers: Dict[str, str]) -> List[Dict]:
    """
    Enrich results with metadata from Semantic Scholar.
    
    Args:
        results: List of result dicts from JSON file
        headers: API headers
        
    Returns:
        List of enriched result dicts
    """
    enriched = []
    unique_corpus_ids = set()
    
    # Get unique corpus IDs from results, preserving order
    for result in results:
        corpus_id = result.get('corpusid')
        if corpus_id and corpus_id not in unique_corpus_ids:
            unique_corpus_ids.add(corpus_id)
    
    # Create a mapping of corpus_id to metadata
    metadata_cache = {}
    
    print(f"Fetching metadata for {len(unique_corpus_ids)} unique papers...")
    
    for corpus_id in tqdm(unique_corpus_ids, desc="Fetching metadata"):
        metadata = get_paper_metadata(corpus_id, headers)
        if metadata:
            metadata_cache[corpus_id] = metadata
        else:
            metadata_cache[corpus_id] = None
        time.sleep(RATE_LIMIT_DELAY)
    
    # Enrich each result
    for result in results:
        corpus_id = result.get('corpusid')
        metadata = metadata_cache.get(corpus_id)
        
        enriched_result = result.copy()
        enriched_result['metadata'] = metadata
        enriched.append(enriched_result)
    
    return enriched


def generate_markdown(enriched_results: List[Dict]) -> str:
    """
    Generate Markdown content from enriched results.
    
    Args:
        enriched_results: List of enriched result dicts
        
    Returns:
        Markdown content as string
    """
    lines = []
    
    # Header
    lines.append("# Top 100 Results with Semantic Scholar Metadata")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("--- ")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total results processed:** {len(enriched_results)}")
    lines.append(f"- **Results with metadata:** {sum(1 for r in enriched_results if r.get('metadata'))}")
    lines.append(f"- **Results without metadata:** {sum(1 for r in enriched_results if not r.get('metadata'))}")
    lines.append("")
    lines.append("--- ")
    lines.append("")
    
    # Main content
    lines.append("## Results")
    lines.append("")
    
    for i, result in enumerate(enriched_results, 1):
        rank = result.get('rank', 'N/A')
        rrf_score = result.get('rrf_score', 0)
        corpus_id = result.get('corpusid', 'N/A')
        paragraph_number = result.get('paragraph_number', 'N/A')
        sentence_start = result.get('sentence_start', 'N/A')
        sentence_end = result.get('sentence_end', 'N/A')
        text = result.get('text', '')
        metadata = result.get('metadata')
        
        lines.append(f"### Result {i} (Rank: {rank}, RRF Score: {rrf_score:.6f})")
        lines.append("")
        
        # Metadata section
        if metadata:
            lines.append("**Paper Information:**")
            lines.append("")
            
            # Title
            title = metadata.get('title', 'No title')
            lines.append(f"- **Title:** {title}")
            
            # Authors
            authors = metadata.get('authors', [])
            authors_str = format_authors(authors)
            lines.append(f"- **Authors:** {authors_str}")
            
            # Year and Venue
            year = metadata.get('year', 'N/A')
            venue = metadata.get('venue', '')
            journal = metadata.get('journal', '')
            pub_date = format_date(metadata.get('publicationDate'))
            
            # Format venue - venue is a string, journal can be a string or dict
            # venue contains the publication venue name (e.g., "Synthese")
            # journal can be a dict with name, volume, pages or a string
            
            # Start with venue if it exists
            if venue and venue.strip():
                venue_info = venue.strip()
            else:
                venue_info = ''
            
            # Format journal - can be a string or a dict
            if journal:
                if isinstance(journal, dict):
                    journal_name = journal.get('name', '')
                    journal_volume = journal.get('volume', '')
                    journal_pages = journal.get('pages', '')
                    
                    # If journal has volume/pages but no name, or name matches venue
                    # Append the volume/pages info to venue
                    if journal_name:
                        if not venue or str(venue).lower().strip() != journal_name.lower().strip():
                            # Different from venue, add as separate
                            venue_info = f"{venue_info} | {journal_name}" if venue_info else journal_name
                        # Append volume and pages
                        if journal_volume:
                            venue_info += f" (Vol. {journal_volume})"
                        if journal_pages:
                            venue_info += f" pp. {journal_pages}"
                    else:
                        # No name, just append volume/pages to venue
                        if journal_volume:
                            venue_info += f" (Vol. {journal_volume})"
                        if journal_pages:
                            venue_info += f" pp. {journal_pages}"
                else:
                    # journal is a string
                    if not venue or str(venue).lower().strip() != journal.lower().strip():
                        venue_info = f"{venue_info} | {journal}" if venue_info else journal
            
            if not venue_info:
                venue_info = 'N/A'
            
            lines.append(f"- **Year:** {year}")
            lines.append(f"- **Venue:** {venue_info}")
            if pub_date:
                lines.append(f"- **Publication Date:** {pub_date}")
            
            # Citation counts
            citation_count = metadata.get('citationCount', 0)
            influential_count = metadata.get('influentialCitationCount', 0)
            lines.append(f"- **Citations:** {citation_count} (Influential: {influential_count})")
            
            # Fields of Study
            fields = metadata.get('fieldsOfStudy', [])
            fields_str = format_fields_of_study(fields)
            if fields_str:
                lines.append(f"- **Fields of Study:** {fields_str}")
            
            # URLs
            url = get_paper_url(metadata)
            # DOI is in externalIds
            external_ids = metadata.get('externalIds', {})
            doi = external_ids.get('DOI', '') if external_ids else ''
            paper_id = metadata.get('paperId', '')
            
            urls = []
            if url:
                urls.append(f"[Semantic Scholar]({url})")
            if doi:
                urls.append(f"[DOI](https://doi.org/{doi})")
            if paper_id and not url:
                urls.append(f"[Semantic Scholar](https://www.semanticscholar.org/paper/{paper_id})")
            
            if urls:
                lines.append(f"- **Links:** {' | '.join(urls)}")
            
            # Abstract
            abstract = metadata.get('abstract', '')
            if abstract:
                lines.append("")
                lines.append("**Abstract:**")
                lines.append("")
                lines.append(f"> {abstract}")
            
            lines.append("")
            lines.append("**Paragraph Information:**")
            lines.append("")
        else:
            lines.append("**Warning:** No metadata available for this paper")
            lines.append("")
            lines.append("**Paragraph Information:**")
            lines.append("")
        
        # Paragraph metadata
        lines.append(f"- **Corpus ID:** {corpus_id}")
        lines.append(f"- **Paragraph Number:** {paragraph_number}")
        lines.append(f"- **Sentence Range:** {sentence_start} - {sentence_end}")
        
        # Text
        lines.append("")
        lines.append("**Extracted Text:**")
        lines.append("")
        lines.append(f"> {text}")
        lines.append("")
        
        # Separator
        lines.append("--- ")
        lines.append("")
    
    return "\n".join(lines)


def get_output_paths(input_path, output_dir=None, top_n=100):
    """
    Generate output file paths based on input file path.
    
    Args:
        input_path: Path to the input JSON file
        output_dir: Optional output directory (default: same as input file)
        top_n: Number of top results
        
    Returns:
        Tuple of (json_output_path, md_output_path)
    """
    # Get the base name of the input file (without extension)
    input_dir = os.path.dirname(input_path)
    input_base = os.path.basename(input_path)
    base_name = os.path.splitext(input_base)[0]
    
    # Use specified output dir or input dir
    output_dir = output_dir if output_dir else input_dir
    
    # Generate output file names
    json_filename = f"{base_name}_top_{top_n}_enriched.json"
    md_filename = f"{base_name}_top_{top_n}_enriched.md"
    
    json_output = os.path.join(output_dir, json_filename)
    md_output = os.path.join(output_dir, md_filename)
    
    return json_output, md_output


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enrich results from a JSON file with Semantic Scholar metadata'
    )
    parser.add_argument(
        'input_file',
        help='Path to the input JSON results file'
    )
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=TOP_N,
        help=f'Number of top results to process (default: {TOP_N})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for result files (default: same as input file)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths (handle relative paths)
    input_path = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    
    print("=" * 60)
    print("Enriching Results with Semantic Scholar Metadata")
    print("=" * 60)
    print()
    
    # Load configuration
    headers = get_headers()
    
    # Check input file
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Loading results from: {input_path}")
    
    # Load results
    try:
        results = load_results(input_path)
        print(f"Loaded {len(results)} results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Sort by rank (ascending - lower rank = higher priority)
    sorted_results = sorted(results, key=lambda x: x.get('rank', float('inf')))
    
    # Take top N
    top_results = sorted_results[:args.top]
    print(f"Processing top {len(top_results)} results by rank...")
    
    # Generate output paths based on input file
    json_output_path, md_output_path = get_output_paths(input_path, output_dir, args.top)
    
    # Check if we already have enriched results saved
    use_cached = False
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, 'r', encoding='utf-8') as f:
                cached_results = json.load(f)
            # Verify cached results have metadata
            if cached_results and all(r.get('metadata') for r in cached_results):
                enriched_results = cached_results
                use_cached = True
                print(f"Using cached enriched results from: {json_output_path}")
        except Exception as e:
            print(f"Could not load cached results: {e}. Refetching...")
    
    if not use_cached:
        # Enrich with metadata
        enriched_results = enrich_results(top_results, headers)
        # Save enriched results as JSON (backup)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_results, f, indent=2, ensure_ascii=False)
        print(f"Saved enriched results to: {json_output_path}")
    else:
        print(f"Skipped API calls - using cached data")
    
    # Generate markdown
    print("Generating Markdown file...")
    markdown_content = generate_markdown(enriched_results)
    
    # Save to file
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\nDone!")
    print(f"JSON backup saved to: {json_output_path}")
    print(f"Markdown file saved to: {md_output_path}")
    print(f"File size: {len(markdown_content):,} characters")
    print("=" * 60)


if __name__ == "__main__":
    main()
