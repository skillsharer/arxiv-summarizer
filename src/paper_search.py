"""
Paper search and scoring module
Handles ArXiv paper search, AI-powered scoring, and intelligent selection
"""

import re
import time
from datetime import datetime, timedelta

import arxiv

from openai_client import get_text
from prompts import PaperScoringPrompts


def score_papers_for_engagement(papers, max_papers=100):
    """
    Score papers for non-technical audience engagement using AI.
    Returns a list of scored papers with metadata.
    """
    if len(papers) > max_papers:
        papers = papers[:max_papers]  # Limit to avoid token limits

    # Prepare batch scoring prompt
    paper_texts = []
    for i, paper in enumerate(papers, 1):
        paper_text = f"Paper {i}: {paper.title}\nAbstract: {paper.summary[:300]}...\n"  # Limit abstract length
        paper_texts.append(paper_text)

    full_prompt = PaperScoringPrompts.BATCH_SCORING_PROMPT + "\n\n".join(paper_texts)

    # Get AI scoring
    try:
        print(f"🤖 Sending {len(papers)} papers to AI for scoring...")
        scoring_response = get_text(full_prompt, model="gpt-4o-mini")
        print(f"📝 AI Response Preview: {scoring_response[:200]}...")
        return parse_paper_scores(scoring_response, papers)
    except Exception as e:
        print(f"Error scoring papers: {e}")
        # Fallback: return papers with default scores
        return [(paper, 5, "Default score", "Unknown") for paper in papers]


def parse_paper_scores(scoring_response, papers):
    """
    Parse the AI scoring response into structured data.
    Returns list of tuples: (paper, score, reason, area)
    """
    scored_papers = []

    # Handle both line-separated and space-separated formats

    # Split by "Paper N:" pattern to get individual paper scores
    paper_pattern = r"Paper (\d+):"
    paper_matches = list(re.finditer(paper_pattern, scoring_response))

    for i, match in enumerate(paper_matches):
        try:
            paper_num = int(match.group(1))
            if paper_num > len(papers):
                continue

            paper = papers[paper_num - 1]

            # Extract the content for this paper (from current match to next match)
            start_pos = match.end()
            if i + 1 < len(paper_matches):
                end_pos = paper_matches[i + 1].start()
                content = scoring_response[start_pos:end_pos].strip()
            else:
                content = scoring_response[start_pos:].strip()

            # Extract score, reason, area using regex
            score = 5  # default
            reason = "No specific reason"
            area = "Unknown"

            # Find Score=X pattern
            score_match = re.search(r"Score=(\d+)", content)
            if score_match:
                score = int(score_match.group(1))

            # Find Reason="..." pattern (handle various quote styles)
            reason_match = re.search(r'Reason="([^"]*)"', content)
            if not reason_match:
                reason_match = re.search(r"Reason='([^']*)'", content)
            if not reason_match:
                reason_match = re.search(r"Reason=([^,]*)", content)

            if reason_match:
                reason = reason_match.group(1).strip().strip("\"'")

            # Find Area="..." pattern (handle various quote styles)
            area_match = re.search(r'Area="([^"]*)"', content)
            if not area_match:
                area_match = re.search(r"Area='([^']*)'", content)
            if not area_match:
                area_match = re.search(r"Area=([^,\s]*)", content)

            if area_match:
                area = area_match.group(1).strip().strip("\"'")

            scored_papers.append((paper, score, reason, area))

        except Exception as e:
            print(f"Error parsing paper {paper_num}: {e}")
            continue

    # If parsing failed completely, return papers with default scores
    if not scored_papers and papers:
        print("Parsing failed, using default scores for all papers")
        return [(paper, 5, "Default score", "Unknown") for paper in papers[:10]]  # Limit fallback

    return scored_papers


def select_diverse_papers(scored_papers, target_count=6):
    """
    Select a diverse set of high-quality papers from scored results.
    Prioritizes high scores while ensuring diversity across research areas.
    """
    # Sort by score (descending)
    sorted_papers = sorted(scored_papers, key=lambda x: x[1], reverse=True)

    # Track selected areas to ensure diversity
    selected_areas = {}
    selected_papers = []

    # First pass: select highest scoring papers with area diversity
    for paper, score, reason, area in sorted_papers:
        if len(selected_papers) >= target_count:
            break

        # Prioritize high-scoring papers (7+) and ensure area diversity
        if score >= 7:
            area_count = selected_areas.get(area, 0)
            # Limit to max 2 papers per area for diversity
            if area_count < 2:
                selected_papers.append((paper, score, reason, area))
                selected_areas[area] = area_count + 1

    # Second pass: fill remaining slots with best available papers
    if len(selected_papers) < target_count:
        for paper, score, reason, area in sorted_papers:
            if len(selected_papers) >= target_count:
                break

            # Skip if already selected
            if any(p[0] == paper for p in selected_papers):
                continue

            # Accept papers with score 5+ if we need more
            if score >= 5:
                selected_papers.append((paper, score, reason, area))

    return selected_papers


def smart_paper_search(days=1, max_fetch=100, target_papers=10):
    """
    Intelligent paper search that fetches more papers, scores them for engagement,
    and selects the best diverse set for non-technical audiences.
    """
    # Define different research topics/areas to search
    research_topics = [
        "cs.AI OR cs.LG OR cs.CV OR cs.CL",  # AI/ML/Computer Vision/NLP
        "cs.RO OR cs.HC OR cs.CY",  # Robotics/Human-Computer Interaction/Cybersecurity
        "q-bio OR physics.bio-ph",  # Biology/Biophysics
        "econ OR q-fin",  # Economics/Finance
        "stat.ML OR math.OC",  # Statistics/Optimization
        "cs.DC OR cs.NI OR cs.CR",  # Distributed Computing/Networking/Cryptography
        "astro-ph OR gr-qc OR hep-th",  # Astronomy/General Relativity/High Energy Physics
        "cond-mat OR quant-ph",  # Condensed Matter/Quantum Physics
        "math.PR OR math.ST",  # Probability/Statistics
        "cs.SI OR cs.SE OR cs.CE",  # Social and Information Networks/Software Engineering/Computational Engineering
        "cs.CY OR cs.CR OR cs.DS",  # Cybersecurity/Cryptography/Data Science
        "cs.NE OR cs.PF OR cs.PL",  # Neural and Evolutionary Computing/Performance/Programming Languages
        "cs.DB OR cs.DL OR cs.IR",  # Databases/Deep Learning/Information Retrieval
    ]

    # Start with the specified time range
    search_attempts = 0
    all_papers = []

    for current_topic in research_topics:  # Ensure we have enough papers to choose from

        start_date = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")

        # Fetch papers with topic-specific query
        search_query = arxiv.Search(
            query=f"submittedDate:[{start_date}0000 TO {end_date}2359] AND cat:{current_topic}",
            max_results=max_fetch // len(research_topics),  # Distribute fetch quota across attempts
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        arxiv_client = arxiv.Client()
        results = list(arxiv_client.results(search_query))

        # Deduplicate papers by entry_id to avoid duplicates
        existing_ids = {paper.entry_id for paper in all_papers}
        new_papers = [paper for paper in results if paper.entry_id not in existing_ids]

        all_papers.extend(new_papers)

        print(
            f"Found {len(results)} papers ({len(new_papers)} new) in {current_topic} (total: {len(all_papers)})"
        )
        search_attempts += 1

    if not all_papers:
        print("No papers found in the expanded search period.")
        return []

    print(f"Scoring {len(all_papers)} papers for engagement potential...")

    # Score papers for engagement
    scored_papers = score_papers_for_engagement(all_papers)

    # Select diverse high-quality papers
    selected_papers = select_diverse_papers(scored_papers, target_papers)

    print(f"Selected {len(selected_papers)} high-quality, diverse papers:")
    for paper, score, reason, area in selected_papers:
        print(f"  • {paper.title[:60]}... (Score: {score}/10, Area: {area})")
        print(f"    Reason: {reason}")

    return [paper for paper, score, reason, area in selected_papers]


def traditional_paper_search(days=1, max_results=6):
    """
    Traditional date-based paper search without AI scoring.
    """
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")
    end_date = datetime.today().strftime("%Y%m%d")

    search_query = arxiv.Search(
        query=f"submittedDate:[{start_date}0000 TO {end_date}2359]",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    arxiv_client = arxiv.Client()
    results = list(arxiv_client.results(search_query))

    return results
