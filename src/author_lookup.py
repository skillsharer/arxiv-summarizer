"""
Author lookup module
Handles finding Twitter handles for paper authors using ORCID and Semantic Scholar APIs
"""

import re
import time
import os
import requests

from prompts import MessageTemplates


def get_orcid_from_name(author_name, max_results=5):
    """
    Search for ORCID profiles using author name.
    Returns a list of potential ORCID IDs.
    """
    try:
        # ORCID search API
        search_url = "https://pub.orcid.org/v3.0/search"
        headers = {"Accept": "application/json", "User-Agent": "ArxivSummarizer/1.0"}

        # Clean and format the name for search
        clean_name = re.sub(r"[^\w\s]", "", author_name).strip()
        name_parts = clean_name.split()

        if not name_parts:  # Handle empty name case
            return []

        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            params = {
                "q": f'given-names:"{first_name}" AND family-name:"{last_name}"',
                "rows": max_results,
            }
        else:
            # Fallback for single name
            params = {"q": f'"{clean_name}"', "rows": max_results}

        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            orcids = []
            results = data.get("result", [])
            if results:  # Check if results exist and is not None
                for result in results:
                    orcid_path = result.get("orcid-identifier", {}).get("path")
                    if orcid_path:
                        orcids.append(orcid_path)
            return orcids
        else:
            print(f"ORCID search failed with status {response.status_code}")
            return []
    except Exception as e:
        print(f"Error searching ORCID for {author_name}: {e}")
        return []


def get_twitter_from_orcid(orcid_id):
    """
    Extract Twitter handle from ORCID profile.
    Returns Twitter handle without @ symbol, or None if not found.
    """
    try:
        # ORCID profile API
        profile_url = f"https://pub.orcid.org/v3.0/{orcid_id}/record"
        headers = {"Accept": "application/json", "User-Agent": "ArxivSummarizer/1.0"}

        response = requests.get(profile_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Look for Twitter in researcher-urls
            researcher_urls = (
                data.get("person", {}).get("researcher-urls", {}).get("researcher-url", [])
            )
            if researcher_urls:
                for url_info in researcher_urls:
                    url = url_info.get("url", {}).get("value", "")
                    if "twitter.com" in url or "x.com" in url:
                        handle = extract_twitter_handle_from_url(url)
                        if handle:
                            return handle

            # Also check external-identifiers for Twitter
            external_ids = (
                data.get("person", {})
                .get("external-identifiers", {})
                .get("external-identifier", [])
            )
            if external_ids:
                for ext_id in external_ids:
                    id_type = ext_id.get("external-id-type", "")
                    if "twitter" in id_type.lower():
                        return ext_id.get("external-id-value", "")

        return None
    except Exception as e:
        print(f"Error getting Twitter from ORCID {orcid_id}: {e}")
        return None


def extract_twitter_handle_from_url(url):
    """
    Extract Twitter handle from a Twitter URL.
    Returns handle without @ symbol, or None if invalid.
    """
    try:
        # Handle various Twitter URL formats
        patterns = [r"twitter\.com/([a-zA-Z0-9_]+)", r"x\.com/([a-zA-Z0-9_]+)", r"@([a-zA-Z0-9_]+)"]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                handle = match.group(1)
                # Validate handle (basic Twitter handle rules)
                if re.match(r"^[a-zA-Z0-9_]{1,15}$", handle):
                    return handle

        return None
    except Exception as e:
        print(f"Error extracting handle from URL {url}: {e}")
        return None


def get_author_info_from_semantic_scholar(arxiv_id):
    """
    Get author information from Semantic Scholar API using arXiv ID.
    Returns a list of author dictionaries with social links if available.
    """
    try:
        # Semantic Scholar API for paper details
        paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        headers = {"User-Agent": "ArxivSummarizer/1.0 (mailto:your-email@example.com)"}

        # Request paper info with author details
        params = {"fields": "authors,authors.name,authors.url,authors.externalIds,authors.homepage"}

        # Add retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.get(paper_url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                authors_info = []

                for author in data.get("authors", []):
                    author_data = {
                        "name": author.get("name", ""),
                        "semantic_scholar_id": author.get("authorId"),
                        "twitter_handle": None,
                    }

                    # Check homepage for Twitter
                    homepage = author.get("homepage")
                    if homepage and ("twitter.com" in homepage or "x.com" in homepage):
                        handle = extract_twitter_handle_from_url(homepage)
                        if handle:
                            author_data["twitter_handle"] = handle

                    # Check external IDs
                    external_ids = author.get("externalIds", {})
                    if external_ids and not author_data["twitter_handle"]:
                        for platform, identifier in external_ids.items():
                            if "twitter" in platform.lower() and identifier:
                                author_data["twitter_handle"] = identifier.replace("@", "")
                                break

                    authors_info.append(author_data)

                return authors_info

            elif response.status_code == 429:  # Rate limited
                wait_time = (2**attempt) + 1  # Exponential backoff: 3, 5, 9 seconds
                print(
                    MessageTemplates.format_message(
                        MessageTemplates.RATE_LIMITED_SEMANTIC_SCHOLAR, seconds=wait_time
                    )
                )
                time.sleep(wait_time)
                continue
            else:
                print(
                    MessageTemplates.format_message(
                        MessageTemplates.SEMANTIC_SCHOLAR_FAILED, status_code=response.status_code
                    )
                )
                break

        return []
    except Exception as e:
        print(f"Error fetching from Semantic Scholar for {arxiv_id}: {e}")
        return []


def get_detailed_author_info_from_semantic_scholar(semantic_scholar_id):
    """
    Get detailed author information from Semantic Scholar using author ID.
    Returns Twitter handle if found in author's profile.
    """
    try:
        twitter_email = os.getenv('TWITTER_EMAIL')
        if not twitter_email:
            twitter_email = "noreply@example.com"
        headers = {"User-Agent": f"ArxivSummarizer/1.0 (mailto:{twitter_email})"}

        params = {"fields": "name,url,homepage,externalIds"}

        # Add retry logic for rate limiting
        max_retries = 2
        for attempt in range(max_retries):
            response = requests.get(author_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Check homepage for Twitter
                homepage = data.get("homepage")
                if homepage:
                    handle = extract_twitter_handle_from_url(homepage)
                    if handle:
                        return handle

                # Check external IDs for social media
                external_ids = data.get("externalIds", {})
                for platform, identifier in external_ids.items():
                    if "twitter" in platform.lower() and identifier:
                        return identifier.replace("@", "")

                return None

            elif response.status_code == 429:  # Rate limited
                wait_time = 1 + attempt
                print(
                    MessageTemplates.format_message(
                        MessageTemplates.RATE_LIMITED_AUTHOR_LOOKUP, seconds=wait_time
                    )
                )
                time.sleep(wait_time)
                continue
            else:
                break

        return None
    except Exception as e:
        print(
            f"Error getting detailed author info from Semantic Scholar {semantic_scholar_id}: {e}"
        )
        return None


def get_twitter_handles_enhanced(authors, arxiv_id):
    """
    Enhanced Twitter handle lookup using both ORCID and Semantic Scholar.
    Returns a dictionary mapping author names to Twitter handles.
    """
    author_handles = {}

    # First, try Semantic Scholar for all authors at once
    print(MessageTemplates.SEMANTIC_SCHOLAR_SEARCH)
    time.sleep(1)  # Initial delay to be respectful
    semantic_authors = get_author_info_from_semantic_scholar(arxiv_id)

    # Create a mapping of names to Semantic Scholar data
    semantic_mapping = {}
    for s_author in semantic_authors:
        name = s_author["name"].strip()
        semantic_mapping[name] = s_author

    # Track progress
    total_authors = len(authors)
    found_count = 0

    for i, author in enumerate(authors, 1):
        author_name = str(author.name).strip()
        print(
            MessageTemplates.format_message(
                MessageTemplates.AUTHOR_PROGRESS,
                current=i,
                total=total_authors,
                author_name=author_name,
            )
        )

        found_handle = None

        # Method 1: Check Semantic Scholar data first
        if author_name in semantic_mapping:
            s_author = semantic_mapping[author_name]

            # Check if we already found a Twitter handle in the basic search
            if s_author["twitter_handle"]:
                found_handle = s_author["twitter_handle"]
                print(
                    MessageTemplates.format_message(
                        MessageTemplates.FOUND_VIA_SEMANTIC_SCHOLAR, handle=found_handle
                    )
                )

            # If not found, try detailed author lookup (with rate limiting)
            elif s_author["semantic_scholar_id"]:
                print(MessageTemplates.CHECKING_DETAILED_PROFILE)
                time.sleep(1)  # Delay before detailed lookup
                handle = get_detailed_author_info_from_semantic_scholar(
                    s_author["semantic_scholar_id"]
                )
                if handle:
                    found_handle = handle
                    print(
                        MessageTemplates.format_message(
                            MessageTemplates.FOUND_VIA_DETAILED_LOOKUP, handle=found_handle
                        )
                    )

        # Method 2: Fallback to ORCID if Semantic Scholar didn't find anything
        if not found_handle:
            print(MessageTemplates.TRYING_ORCID_FALLBACK)
            orcid_ids = get_orcid_from_name(author_name)

            if orcid_ids:
                for orcid_id in orcid_ids[:2]:  # Limit to first 2 to avoid too many requests
                    twitter_handle = get_twitter_from_orcid(orcid_id)
                    if twitter_handle:
                        found_handle = twitter_handle
                        print(
                            MessageTemplates.format_message(
                                MessageTemplates.FOUND_VIA_ORCID, handle=found_handle
                            )
                        )
                        break
                    time.sleep(0.5)  # Small delay between ORCID requests

        # Store the result
        if found_handle:
            author_handles[author_name] = found_handle
            found_count += 1
        else:
            print(
                MessageTemplates.format_message(
                    MessageTemplates.NO_HANDLE_FOUND, author_name=author_name
                )
            )

        # Progress update
        if i < total_authors:
            print(
                MessageTemplates.format_message(
                    MessageTemplates.PROGRESS_UPDATE, found=found_count, processed=i
                )
            )
            time.sleep(1.2)  # Delay between authors to respect APIs

    print(
        MessageTemplates.format_message(
            MessageTemplates.AUTHOR_LOOKUP_COMPLETE, found=found_count, total=total_authors
        )
    )
    return author_handles


def get_twitter_handles_for_authors(authors):
    """
    Get Twitter handles for a list of authors using arXiv → ORCID → Twitter pipeline.
    Returns a dictionary mapping author names to Twitter handles.
    """
    author_handles = {}

    for author in authors:
        author_name = str(author.name).strip()
        print(f"Looking up Twitter handle for: {author_name}")

        # Get potential ORCID IDs
        orcid_ids = get_orcid_from_name(author_name)

        if orcid_ids:
            # Try each ORCID ID until we find a Twitter handle
            for orcid_id in orcid_ids:
                twitter_handle = get_twitter_from_orcid(orcid_id)
                if twitter_handle:
                    author_handles[author_name] = twitter_handle
                    print(f"Found Twitter handle for {author_name}: @{twitter_handle}")
                    break

                # Add small delay to avoid rate limiting
                time.sleep(0.5)

        if author_name not in author_handles:
            print(f"No Twitter handle found for {author_name}")

        # Add delay between author lookups to be respectful to APIs
        time.sleep(1)

    return author_handles
