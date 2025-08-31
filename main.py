import arxiv
import fitz  # PyMuPDF
import os
import requests
import tweepy
import numpy as np
import time
import re
from PIL import Image, ImageOps
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pymupdf import FileDataError
from prompts import PromptTemplates, MessageTemplates, PaperScoringPrompts

load_dotenv()

OPENAI_BEARER_TOKEN = os.getenv('OPENAI_BEARER_TOKEN')
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def get_text(prompt, model="gpt-4o-mini"):
    try:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "Bearer " + OPENAI_BEARER_TOKEN
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response = response.json()["choices"][0]["message"]["content"].replace('\n', '')
        time.sleep(3) # Wait for 3 seconds to avoid rate limiting
    except Exception as e:
        response = ""
        print(f"An error occurred: {e}")

    return response

def get_orcid_from_name(author_name, max_results=5):
    """
    Search for ORCID profiles using author name.
    Returns a list of potential ORCID IDs.
    """
    try:
        # ORCID search API
        search_url = "https://pub.orcid.org/v3.0/search"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'ArxivSummarizer/1.0'
        }
        
        # Clean and format the name for search
        clean_name = re.sub(r'[^\w\s]', '', author_name).strip()
        name_parts = clean_name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            params = {
                'q': f'given-names:"{first_name}" AND family-name:"{last_name}"',
                'rows': max_results
            }
        else:
            # Fallback for single name
            params = {
                'q': f'"{clean_name}"',
                'rows': max_results
            }
        
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            orcids = []
            for result in data.get('result', []):
                orcid_id = result.get('orcid-identifier', {}).get('path')
                if orcid_id:
                    orcids.append(orcid_id)
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
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'ArxivSummarizer/1.0'
        }
        
        response = requests.get(profile_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Look for Twitter in researcher-urls
            researcher_urls = data.get('person', {}).get('researcher-urls', {}).get('researcher-url', [])
            if researcher_urls:
                for url_info in researcher_urls:
                    url = url_info.get('url', {}).get('value', '')
                    url_name = url_info.get('url-name', '')
                    
                    # Check if it's a Twitter URL
                    if any(domain in url.lower() for domain in ['twitter.com', 'x.com']):
                        # Extract handle from URL
                        handle = extract_twitter_handle_from_url(url)
                        if handle:
                            return handle
                    
                    # Check if the URL name suggests it's Twitter
                    if 'twitter' in url_name.lower() or 'x.com' in url_name.lower():
                        handle = extract_twitter_handle_from_url(url)
                        if handle:
                            return handle
            
            # Also check external-identifiers for Twitter
            external_ids = data.get('person', {}).get('external-identifiers', {}).get('external-identifier', [])
            if external_ids:
                for ext_id in external_ids:
                    id_type = ext_id.get('external-id-type', '')
                    id_value = ext_id.get('external-id-value', '')
                    if 'twitter' in id_type.lower():
                        return id_value.replace('@', '').strip()
        
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
        patterns = [
            r'twitter\.com/([a-zA-Z0-9_]+)',
            r'x\.com/([a-zA-Z0-9_]+)',
            r'@([a-zA-Z0-9_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                handle = match.group(1)
                # Validate handle (basic Twitter handle rules)
                if re.match(r'^[a-zA-Z0-9_]{1,15}$', handle):
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
        headers = {
            'User-Agent': 'ArxivSummarizer/1.0 (mailto:your-email@example.com)'
        }
        
        # Request paper info with author details
        params = {
            'fields': 'authors,authors.name,authors.url,authors.externalIds,authors.homepage'
        }
        
        # Add retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.get(paper_url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                authors_info = []
                
                for author in data.get('authors', []):
                    author_data = {
                        'name': author.get('name', ''),
                        'semantic_scholar_id': author.get('authorId'),
                        'url': author.get('url'),
                        'homepage': author.get('homepage'),
                        'external_ids': author.get('externalIds', {}),
                        'twitter_handle': None
                    }
                    
                    # Check homepage for Twitter links
                    if author_data['homepage']:
                        handle = extract_twitter_handle_from_url(author_data['homepage'])
                        if handle:
                            author_data['twitter_handle'] = handle
                    
                    authors_info.append(author_data)
                
                return authors_info
                
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt  # Exponential backoff
                print(MessageTemplates.format_message(MessageTemplates.RATE_LIMITED_SEMANTIC_SCHOLAR, seconds=wait_time))
                time.sleep(wait_time)
                continue
            else:
                print(MessageTemplates.format_message(MessageTemplates.SEMANTIC_SCHOLAR_FAILED, status_code=response.status_code))
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
        author_url = f"https://api.semanticscholar.org/graph/v1/author/{semantic_scholar_id}"
        headers = {
            'User-Agent': 'ArxivSummarizer/1.0 (mailto:your-email@example.com)'
        }
        
        params = {
            'fields': 'name,url,homepage,externalIds'
        }
        
        # Add retry logic for rate limiting
        max_retries = 2
        for attempt in range(max_retries):
            response = requests.get(author_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check homepage for Twitter
                homepage = data.get('homepage')
                if homepage:
                    handle = extract_twitter_handle_from_url(homepage)
                    if handle:
                        return handle
                
                # Check external IDs for social media
                external_ids = data.get('externalIds', {})
                for platform, identifier in external_ids.items():
                    if 'twitter' in platform.lower():
                        # Clean and validate the identifier
                        handle = identifier.replace('@', '').strip()
                        if re.match(r'^[a-zA-Z0-9_]{1,15}$', handle):
                            return handle
                
                return None
                
            elif response.status_code == 429:  # Rate limited
                wait_time = 1 + attempt
                print(MessageTemplates.format_message(MessageTemplates.RATE_LIMITED_AUTHOR_LOOKUP, seconds=wait_time))
                time.sleep(wait_time)
                continue
            else:
                break
        
        return None
    except Exception as e:
        print(f"Error getting detailed author info from Semantic Scholar {semantic_scholar_id}: {e}")
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
        name = s_author['name'].strip()
        semantic_mapping[name] = s_author
    
    # Track progress
    total_authors = len(authors)
    found_count = 0
    
    for i, author in enumerate(authors, 1):
        author_name = str(author.name).strip()
        print(MessageTemplates.format_message(MessageTemplates.AUTHOR_PROGRESS, 
                                             current=i, total=total_authors, author_name=author_name))
        
        found_handle = None
        
        # Method 1: Check Semantic Scholar data first
        if author_name in semantic_mapping:
            s_author = semantic_mapping[author_name]
            
            # Check if we already found a Twitter handle in the basic search
            if s_author['twitter_handle']:
                found_handle = s_author['twitter_handle']
                print(MessageTemplates.format_message(MessageTemplates.FOUND_VIA_SEMANTIC_SCHOLAR, handle=found_handle))
            
            # If not found, try detailed author lookup (with rate limiting)
            elif s_author['semantic_scholar_id']:
                print(MessageTemplates.CHECKING_DETAILED_PROFILE)
                time.sleep(1)  # Delay before detailed lookup
                handle = get_detailed_author_info_from_semantic_scholar(s_author['semantic_scholar_id'])
                if handle:
                    found_handle = handle
                    print(MessageTemplates.format_message(MessageTemplates.FOUND_VIA_SEMANTIC_SCHOLAR_DETAILED, handle=found_handle))
        
        # Method 2: Fallback to ORCID if Semantic Scholar didn't find anything
        if not found_handle:
            print(MessageTemplates.TRYING_ORCID_FALLBACK)
            orcid_ids = get_orcid_from_name(author_name)
            
            if orcid_ids:
                for orcid_id in orcid_ids[:2]:  # Limit to first 2 ORCID results
                    twitter_handle = get_twitter_from_orcid(orcid_id)
                    if twitter_handle:
                        found_handle = twitter_handle
                        print(MessageTemplates.format_message(MessageTemplates.FOUND_VIA_ORCID, handle=found_handle))
                        break
                    time.sleep(0.5)  # Delay between ORCID requests
        
        # Store the result
        if found_handle:
            author_handles[author_name] = found_handle
            found_count += 1
        else:
            print(MessageTemplates.format_message(MessageTemplates.NO_HANDLE_FOUND, author_name=author_name))
        
        # Progress update
        if i < total_authors:
            print(MessageTemplates.format_message(MessageTemplates.PROGRESS_UPDATE, found=found_count, processed=i))
            time.sleep(1.2)  # Delay between authors to respect APIs
    
    print(MessageTemplates.format_message(MessageTemplates.AUTHOR_LOOKUP_COMPLETE, found=found_count, total=total_authors))
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

def format_tweet_with_author_tags(base_text, authors, author_handles, max_length=280):
    """
    Format tweet text with author tags, ensuring we stay within Twitter's character limit.
    """
    # Start with the base text
    tweet_text = base_text
    
    # Collect handles to tag
    handles_to_tag = []
    for author in authors:
        author_name = str(author.name).strip()
        if author_name in author_handles:
            handles_to_tag.append(f"@{author_handles[author_name]}")
    
    # If we have handles to tag, add them
    if handles_to_tag:
        # Create author tag string
        author_tag_string = " Authors: " + " ".join(handles_to_tag)
        
        # Check if adding tags would exceed character limit
        if len(tweet_text) + len(author_tag_string) <= max_length:
            tweet_text += author_tag_string
        else:
            # Try to fit as many as possible
            available_space = max_length - len(tweet_text) - len(" Authors: ")
            
            # Add handles one by one until we run out of space
            added_handles = []
            current_length = 0
            
            for handle in handles_to_tag:
                handle_with_space = handle + " " if added_handles else handle
                if current_length + len(handle_with_space) <= available_space:
                    added_handles.append(handle)
                    current_length += len(handle_with_space)
                else:
                    break
            
            if added_handles:
                tweet_text += " Authors: " + " ".join(added_handles)
    
    return tweet_text

def extract_relevant_image_from_pdf(pdf_path, image_dir, entry_id):
    """Extracts a relevant image from a PDF by analyzing surrounding text, checking for keywords in captions, 
       and filtering based on color and size criteria."""
    
    # Keywords indicating potentially relevant images
    keywords = ["figure", "graph", "diagram", "illustration", "results", "method", "conclusion"]
    image_path = None
    
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text("text").lower()  # Get text content in lowercase for keyword matching
            
            # Loop through each image on the page
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                
                # Convert to RGB if needed
                if pix.colorspace.n != 3:
                    try:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    except:
                        continue
                
                # Get image data
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # 1. Filter out black/blank images
                img_array = np.array(img_data)
                black_pixels_ratio = np.mean(np.all(img_array == [0, 0, 0], axis=2))
                
                if black_pixels_ratio > 0.8:  # Skip black/blank images
                    continue
                
                # 2. Apply size threshold to skip small or irrelevant images
                if img_data.width < 200 or img_data.height < 200:
                    continue
                
                # Resize the image while maintaining aspect ratio
                max_size = (800, 800)  # Define the maximum size for the image
                img_data.thumbnail(max_size, Image.Resampling.NEAREST)
                
                # 3. Check for relevant keywords around image in the page text
                surrounding_text = page_text[max(0, page_text.find(keywords[0])-100): page_text.find(keywords[0])+100]
                if any(keyword in surrounding_text for keyword in keywords):
                    
                    # Save the image if it passes all filters
                    image_path = os.path.join(image_dir, f"{entry_id}_image.png")
                    img_data.save(image_path)
                    pix = None  # Clean up Pixmap object
                    print(f"Relevant image extracted and saved to: {image_path}")
                    return image_path

                pix = None  # Clean up Pixmap object if not saved

    if image_path is None:
        print("No relevant images found in PDF.")
    return image_path

def extract_first_image_from_pdf(pdf_path, image_dir, entry_id):
    """Extracts the first image from a PDF and saves it as a PNG file."""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                if pix.colorspace.n != 3:  # Check if the colorspace is not RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_path = os.path.join(image_dir, f"{entry_id}_image.png")
                pix.save(image_path)
                pix = None  # Clean up Pixmap object
                print(f"Image extracted and saved to: {image_path}")
                return image_path
    print("No images found in PDF.")
    return None

def extract_images_and_create_gif(pdf_path, image_dir, entry_id, gif_path, duration=2000, size=(300, 300), transition_frames=20):
    """Extracts all images from a PDF and creates a GIF file with a smooth transition between each image."""
    images = []
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)
                if pix.colorspace.n != 3:  # Check if the colorspace is not RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_path = os.path.join(image_dir, f"{entry_id}_image_{page_num}_{img_index}.png")
                pix.save(image_path)
                images.append(image_path)
                pix = None  # Clean up Pixmap object
                print(f"Image extracted and saved to: {image_path}")
                if len(images) >= 10:
                    break

    if images:
        # Create a GIF from the extracted images
        gif_images = []
        for i in range(len(images) - 1):
            img1 = Image.open(images[i])
            img1 = ImageOps.pad(img1, size, color="black")  # Resize while keeping aspect ratio and adding black padding
            img2 = Image.open(images[i + 1])
            img2 = ImageOps.pad(img2, size, color="black")  # Resize while keeping aspect ratio and adding black padding
            
            # Ensure both images are in the same mode (RGB)
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            gif_images.append(img1)
            for j in range(1, transition_frames + 1):
                blend = Image.blend(img1, img2, j / (transition_frames + 1))
                gif_images.append(blend)
        img_last = Image.open(images[-1])
        img_last = ImageOps.pad(img_last, size, color="black")  # Resize while keeping aspect ratio and adding black padding
        if img_last.mode != 'RGB':
            img_last = img_last.convert('RGB')
        gif_images.append(img_last)

        gif_images[0].save(gif_path, save_all=True, append_images=gif_images[1:], duration=duration // (transition_frames + 1), loop=0)
        print(f"GIF created and saved to: {gif_path}")
        return gif_path
    else:
        print("No images found in PDF.")
        return None

def score_papers_for_engagement(papers, max_papers=50):
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
    import re
    
    # Split by "Paper N:" pattern to get individual paper scores
    paper_pattern = r'Paper (\d+):'
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
            score_match = re.search(r'Score=(\d+)', content)
            if score_match:
                score = int(score_match.group(1))
            
            # Find Reason="..." pattern (handle various quote styles)
            reason_match = re.search(r'Reason="([^"]*)"', content)
            if not reason_match:
                reason_match = re.search(r"Reason='([^']*)'", content)
            if not reason_match:
                reason_match = re.search(r'Reason=([^,]*)', content)
            
            if reason_match:
                reason = reason_match.group(1).strip().strip('"\'')
            
            # Find Area="..." pattern (handle various quote styles)
            area_match = re.search(r'Area="([^"]*)"', content)
            if not area_match:
                area_match = re.search(r"Area='([^']*)'", content)
            if not area_match:
                area_match = re.search(r'Area=([^,\s]*)', content)
            
            if area_match:
                area = area_match.group(1).strip().strip('"\'')
            
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

def smart_paper_search(days=1, max_fetch=80, target_papers=6):
    """
    Intelligent paper search that fetches more papers, scores them for engagement,
    and selects the best diverse set for non-technical audiences.
    """
    # Start with the specified time range
    search_attempts = 0
    max_attempts = 3
    all_papers = []
    
    while len(all_papers) < 20 and search_attempts < max_attempts:  # Ensure we have enough papers to choose from
        current_days = days + (search_attempts * 1)  # Expand search window if needed
        
        start_date = (datetime.today() - timedelta(days=current_days)).strftime('%Y%m%d')
        end_date = datetime.today().strftime('%Y%m%d')
        
        print(f"Search attempt {search_attempts + 1}: Looking for papers from {current_days} days back...")
        
        # Fetch papers
        search_query = arxiv.Search(
            query=f"submittedDate:[{start_date}0000 TO {end_date}2359]",
            max_results=max_fetch,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        arxiv_client = arxiv.Client()
        results = list(arxiv_client.results(search_query))
        all_papers.extend(results)
        
        print(f"Found {len(results)} papers in this search (total: {len(all_papers)})")
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

def tweet_arxiv_papers(debug=False, days=1, max_results=6, enable_author_tagging=True, use_smart_selection=True):
    # Authenticate to Twitter
    if not debug:
        try:
            client = tweepy.Client(
                BEARER_TOKEN,
                CONSUMER_KEY,
                CONSUMER_SECRET,
                ACCESS_TOKEN,
                ACCESS_TOKEN_SECRET,
                wait_on_rate_limit=True
            )
            auth = tweepy.OAuth1UserHandler(
                CONSUMER_KEY,
                CONSUMER_SECRET,
                ACCESS_TOKEN,
                ACCESS_TOKEN_SECRET
            )
            api = tweepy.API(auth)
            print("Successfully authenticated to Twitter.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return

    # Create folders for PDFs and images
    pdf_dir = "arxiv_papers"
    image_dir = "arxiv_images"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Use smart paper selection or fallback to traditional method
    if use_smart_selection:
        print("🧠 Using smart paper selection for maximum engagement...")
        results = smart_paper_search(days=days, target_papers=max_results)
    else:
        print("📚 Using traditional date-based paper search...")
        # Traditional search logic (fallback)
        start_date = (datetime.today() - timedelta(days=days)).strftime('%Y%m%d')
        end_date = datetime.today().strftime('%Y%m%d')
        
        search_query = arxiv.Search(
            query=f"submittedDate:[{start_date}0000 TO {end_date}2359]",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        arxiv_client = arxiv.Client()
        results = list(arxiv_client.results(search_query))

    print(f"Number of selected papers: {len(results)}")

    if not results:
        print("No results found. Please check your query and try again.")
        return

    if not debug:
        # Post the main tweet to start the thread
        main_tweet = client.create_tweet(
            text=PromptTemplates.get_thread_opener(personality="viral")
        )
        main_tweet_id = main_tweet.data['id']  # Store the main tweet ID

    for result in results:
        print(MessageTemplates.format_message(MessageTemplates.PROCESSING_PAPER, title=result.title))
        print(MessageTemplates.format_message(MessageTemplates.EXTRACTING_AUTHORS, authors=[author.name for author in result.authors]))

        # Get Twitter handles for authors using enhanced method
        author_handles = {}
        if enable_author_tagging:
            # Extract arXiv ID from the entry_id (format: http://arxiv.org/abs/2024.12345v1)
            arxiv_id = result.entry_id.split('/')[-1]
            # Remove version number (v1, v2, etc.) to get clean arXiv ID
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            print(MessageTemplates.format_message(MessageTemplates.ARXIV_ID_EXTRACTED, arxiv_id=arxiv_id))
            author_handles = get_twitter_handles_enhanced(result.authors, arxiv_id)

        # Download the PDF
        pdf_path = os.path.join(pdf_dir, f"{result.entry_id.split('/')[-1]}.pdf")
        result.download_pdf(filename=pdf_path)

        # Extract text from the PDF
        text = ""
        try:
            with fitz.open(pdf_path) as pdf_document:
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()
        except FileDataError:
            print(f"Failed to open file '{pdf_path}'. Skipping this document.")
            continue


        # Extract the first image from the PDF
        #image_path = extract_relevant_image_from_pdf(pdf_path, image_dir, result.entry_id.split('/')[-1])
        image_path = extract_images_and_create_gif(pdf_path, image_dir, result.entry_id.split('/')[-1], f"{image_dir}/{result.entry_id.split('/')[-1]}.gif")
        
        # Generate tweet text using ChatGPT
        prompt = PromptTemplates.get_summary_prompt(text, style="viral")
        explanation = get_text(prompt)
        
        # Format tweet with author tags
        base_tweet = f"{explanation} Source: {result.entry_id}"
        final_tweet = format_tweet_with_author_tags(base_tweet, result.authors, author_handles)
        
        print(f"{final_tweet}\n")

        # Post the explanation with an image if available
        if not debug:
            media_ids = []
            if image_path:
                media = api.media_upload(image_path)
                media_ids.append(media.media_id)

            client.create_tweet(
                text=final_tweet,
                in_reply_to_tweet_id=main_tweet_id,
                media_ids=media_ids if media_ids else None
            )

    if not debug:
        # Final tweet to close the thread
        client.create_tweet(
            text=MessageTemplates.get_thread_closer(personality="viral"),
            in_reply_to_tweet_id=main_tweet_id
        )

# Run the function
if __name__ == "__main__":
    tweet_arxiv_papers(debug=True, enable_author_tagging=True, days=10, max_results=1)
